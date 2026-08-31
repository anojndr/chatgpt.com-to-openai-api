"""Async client for chatgpt.com's internal backend-api (no browser).

Protocol per account session (TLS-impersonated via curl_cffi):
  1. GET  /api/auth/session            -> fresh accessToken from session cookies
  2. GET  /?oai-dm=1                   -> build id + script paths (proof config)
  3. POST /sentinel/chat-requirements  -> {token, proofofwork} cached REQUIREMENTS_TTL
  4. POST /conversation                -> SSE; requires sentinel headers
  5. files: POST /files, PUT blob, POST /files/{id}/uploaded; download /files/{id}/download
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator

from curl_cffi.requests import AsyncSession

from . import config, pow_solver

log = logging.getLogger("chatgpt")

CHATGPT_ORIGIN = "https://chatgpt.com"


@dataclass
class Requirements:
    token: str
    proof: str
    expires_at: float


class ChatGPTError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


BLOCK_RE = re.compile(r"^<<<\s*$(.*?)^>>>\s*$", re.S | re.M)


def parse_accounts_text(text: str) -> list[dict[str, Any]]:
    """Parse accounts.txt: '<<<' ... '>>>' blocks with a netscape cookie jar and
    the JSON body of https://chatgpt.com/api/auth/session."""
    accounts: list[dict[str, Any]] = []
    for b in BLOCK_RE.findall(text):
        m = re.search(r"from https://chatgpt\.com/api/auth/session:\s*(\{.*)", b, re.S)
        if not m:
            continue
        try:
            sess = json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            continue
        cookies: dict[str, str] = {}
        raw_cookie_lines: list[str] = []
        for line in b.splitlines():
            line = line.strip()
            marker = "#HttpOnly_" if line.startswith("#HttpOnly_") else ""
            bare = line[len(marker):]  # standard netscape-jar HttpOnly marker
            if not bare or bare.startswith("#") or "\t" not in bare:
                continue
            parts = bare.split("\t")
            if len(parts) >= 7:
                cookies[parts[5]] = parts[6]
                raw_cookie_lines.append(line)  # marker kept for round-trip
        if not sess.get("accessToken"):
            continue
        accounts.append({
            "session": sess,
            "cookies": cookies,
            "raw_cookie_lines": raw_cookie_lines,
            "identity": sess.get("user", {}).get("id") or sess.get("account", {}).get("id"),
        })
    return accounts


def block_session(block_text: str) -> dict[str, Any] | None:
    """Session JSON of one raw '<<<' ... '>>>' block body, or None."""
    m = re.search(r"from https://chatgpt\.com/api/auth/session:\s*(\{.*)", block_text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1).strip())
    except json.JSONDecodeError:
        return None


def block_identity(block_text: str) -> str | None:
    """Identity of one raw '<<<' ... '>>>' block body (for block replacement)."""
    sess = block_session(block_text)
    if not sess:
        return None
    return sess.get("user", {}).get("id") or sess.get("account", {}).get("id")


class AccountSession:
    """One ChatGPT account: auth state + HTTP session + sentinel cache."""

    def __init__(self, parsed: dict[str, Any]):
        self.identity: str = parsed["identity"]
        self.session_json: dict[str, Any] = parsed["session"]
        self.cookies: dict[str, str] = dict(parsed["cookies"])
        self.raw_cookie_lines: list[str] = list(parsed.get("raw_cookie_lines") or [])
        s = parsed["session"]
        self.access_token: str = s["accessToken"]
        self._jwt_exp: float = _jwt_exp(self.access_token)
        self.expires_at: float = _parse_expires(s.get("expires"))
        self.account_id: str = s.get("account", {}).get("id", "")
        self.plan: str = s.get("account", {}).get("planType", "free")
        self.email: str = s.get("user", {}).get("email", "")
        # runtime LB state
        self.inflight: int = 0
        self.last_used: float = 0.0
        self.total_requests: int = 0
        self.cooldown_until: float = 0.0
        self.dead: bool = False
        # keepalive state
        self.refresh_strikes: int = 0
        self._last_strike_at: float = 0.0
        self.revive_after: float = 0.0
        self.state_dirty: bool = False
        self._refreshing: bool = False   # network refresh in flight
        self._refresh_ok: bool = False   # last completed refresh succeeded
        # http/sentinel state
        self._http: AsyncSession | None = None
        self._req: Requirements | None = None
        self._build_id: str = ""
        self._scripts: list[str] = []
        self._models_cache: tuple[float, list[dict]] | None = None
        self._last_refresh_attempt: float = 0.0
        self._closed: bool = False

    # ---------- http ----------
    async def http(self) -> AsyncSession:
        if self._closed:
            raise ChatGPTError(503, f"session for {self.email} was removed from the pool")
        if self._http is None:
            self._http = AsyncSession(impersonate=config.IMPERSONATE)
            self._http.headers.update({"User-Agent": config.USER_AGENT})
            self._http.cookies.update(self.cookies)
        return self._http

    def base_headers(self) -> dict[str, str]:
        h = {
            "Oai-Device-Id": self.cookies.get("oai-did", str(uuid.uuid4())),
            "Oai-Language": "en-US",
            "Accept": "*/*",
        }
        # Session cookies alone authenticate backend-api; include Bearer only when
        # the access-token JWT is actually valid (expired tokens cause 401s).
        if self.access_token and time.time() < self._jwt_exp - 60:
            h["Authorization"] = f"Bearer {self.access_token}"
        if self.account_id:
            h["Chatgpt-Account-Id"] = self.account_id
        return h

    async def close(self) -> None:
        self._closed = True
        if self._http is not None:
            await self._http.close()
            self._http = None

    # ---------- auth ----------
    async def refresh_access_token(self) -> bool:
        """Refresh via /api/auth/session. On any successful refresh the
        rotated session cookies are absorbed into self.cookies and marked
        dirty; the pool persists dirty state into accounts.txt.

        Failure policy: transport errors / non-200 / non-JSON responses are
        transient (return False, no strike). A 200 that cannot yield a
        fresh-enough token counts one strike; KEEPALIVE_MAX_STRIKES spaced
        strikes mean the session cookie itself is stale -> dead (needs
        re-login)."""
        try:
            # One refresh per account at a time; request paths and the
            # keepalive sweep all funnel through here. While one is in
            # flight (or concluded <5s ago) report the cached outcome
            # instead of firing a second request.
            if self._refreshing or time.time() - self._last_refresh_attempt < 5:
                return bool(self._refresh_ok and self.access_token
                            and time.time() < self._jwt_exp - 60)
            self._last_refresh_attempt = time.time()
            self._refreshing = True
            try:
                return await self._do_refresh()
            finally:
                self._refreshing = False
        except Exception as e:
            self._refresh_ok = False
            log.warning("[%s] refresh failed: %s", self.email, e)
            return False

    async def _do_refresh(self) -> bool:
        self._refresh_ok = False
        s = await self.http()
        r = await s.get(f"{CHATGPT_ORIGIN}/api/auth/session",
                        headers={"Accept": "*/*"}, timeout=30)
        if r.status_code != 200:
            log.warning("[%s] session refresh HTTP %s", self.email, r.status_code)
            return False
        try:
            j = r.json()
        except Exception:
            log.warning("[%s] session refresh returned non-JSON body", self.email)
            return False
        self._absorb_cookies(r)
        new_at = j.get("accessToken")
        if not new_at:
            return self._record_refresh_failure("session returned no accessToken")
        new_exp = _jwt_exp(new_at)
        if not new_exp or new_exp <= time.time() + 60:
            # /api/auth/session can keep serving a cached token past its
            # exp when the session cookie itself is stale. Never adopt it.
            return self._record_refresh_failure(
                "session returned an %s access token (%ss left)" % (
                    "unparseable" if not new_exp else "expired",
                    max(0.0, (new_exp or 0.0) - time.time())))
        self.access_token = new_at
        self._jwt_exp = new_exp
        self.session_json = j
        exp = _parse_expires(j.get("expires"))
        if exp:
            self.expires_at = exp
        plan = (j.get("account") or {}).get("planType")
        if plan:
            self.plan = plan
        self._req = None
        self.refresh_strikes = 0
        self.dead = False
        self.state_dirty = True
        self._refresh_ok = True
        log.info("[%s] access token refreshed", self.email)
        return True

    def _record_refresh_failure(self, detail: str) -> bool:
        """A confirmed-but-unusable session response. Counts one strike;
        strikes older than two sweep intervals are forgiven first, so only
        KEEPALIVE_MAX_STRIKES consecutive failures mark the account dead."""
        now = time.time()
        if now - self._last_strike_at > 2 * config.KEEPALIVE_SECONDS:
            self.refresh_strikes = 0  # stale evidence: not a consecutive run
        self._last_strike_at = now
        self.refresh_strikes += 1
        if self.refresh_strikes >= config.KEEPALIVE_MAX_STRIKES:
            self.dead = True
            self.revive_after = now + config.KEEPALIVE_REVIVE_SECONDS
            log.warning("[%s] needs re-login (%s; %d strikes)", self.email, detail,
                        self.refresh_strikes)
        else:
            log.warning("[%s] refresh unusable: %s (strike %d/%d)", self.email, detail,
                        self.refresh_strikes, config.KEEPALIVE_MAX_STRIKES)
        return False

    def _absorb_cookies(self, r: Any) -> None:
        """Merge session cookies rotated by the server into account state and
        regenerate raw_cookie_lines so accounts.txt stays re-exportable."""
        try:
            get_list = getattr(r.headers, "get_list", None)
            if get_list is not None:
                headers = get_list("set-cookie") or []
            else:
                # Generic fallback: multi_items keeps one entry per header,
                # unlike get() which comma-joins and garbles Expires dates.
                headers = [v for k, v in r.headers.multi_items()
                           if k.lower() == "set-cookie"]
        except Exception:
            return
        import http.cookies
        sc = http.cookies.SimpleCookie()
        for header in headers:
            if not header:
                continue
            try:
                sc.load(header)
            except Exception:
                continue  # malformed/obfuscated cookie (e.g. Cloudflare challenge)
        if not sc:
            return
        merged = 0
        for name, morsel in sc.items():
            value = morsel.value
            # Empty value == deletion cookie (tok=; Expires=<past>). Adopting
            # it would wipe the stored value for a challenge/glitch response,
            # which is unrecoverable — keeping the stale cookie is not.
            if not value:
                continue
            if self.cookies.get(name) == value:
                continue
            self.cookies[name] = value
            merged += 1
        if not merged:
            return
        if self._http is not None:
            for name in sc.keys():
                if name in self.cookies:
                    self._http.cookies.set(name, self.cookies[name],
                                           domain=".chatgpt.com", path="/")
        self._rebuild_raw_cookie_lines()
        self.state_dirty = True
        log.info("[%s] absorbed %d rotated session cookie(s)", self.email, merged)

    def _rebuild_raw_cookie_lines(self) -> None:
        """Regenerate netscape rows from raw_cookie_lines + self.cookies,
        replacing values of cookies the server rotated."""
        lines: list[str] = []
        seen: set[str] = set()
        for line in self.raw_cookie_lines:
            parts = line.split("\t")
            if len(parts) >= 7 and parts[5] in self.cookies:
                parts[6] = self.cookies[parts[5]]
                line = "\t".join(parts)
                seen.add(parts[5])
            lines.append(line)
        for name, value in self.cookies.items():
            if name in seen:
                continue
            lines.append(".chatgpt.com\tTRUE\t/\tTRUE\t0\t%s\t%s" % (name, value))
        self.raw_cookie_lines = lines

    async def ensure_token(self) -> None:
        # JWT expiry is authoritative; session-cookie auth covers us regardless.
        # refresh_access_token self-throttles; this gate reads the CURRENT
        # attempt stamp (never pre-set it here — that turned this into a
        # silent no-op) and spaces nudges to once a minute so a busy
        # final-600s window can't hammer /api/auth/session.
        if self.access_token and time.time() > self._jwt_exp - 600:
            if time.time() - self._last_refresh_attempt > 60:
                await self.refresh_access_token()

    # ---------- sentinel ----------
    async def requirements(self, force: bool = False) -> Requirements:
        now = time.time()
        if not force and self._req and self._req.expires_at > now + 30:
            return self._req
        s = await self.http()
        pre = pow_solver.pre_proof(user_agent=config.USER_AGENT)
        r = await s.post(
            f"{CHATGPT_ORIGIN}/backend-api/sentinel/chat-requirements",
            headers={**self.base_headers(), "Content-Type": "application/json"},
            json={"p": pre},
            timeout=30,
        )
        if r.status_code == 403:
            raise ChatGPTError(403, "sentinel blocked: " + r.text[:200])
        if r.status_code != 200:
            raise ChatGPTError(r.status_code, f"chat-requirements failed: {r.text[:200]}")
        j = r.json()
        pw = j.get("proofofwork") or {}
        proof = ""
        if pw.get("required"):
            script = self._scripts[0] if self._scripts else None
            try:
                # CPU-bound: keep the event loop free for other accounts' streams.
                proof = await asyncio.to_thread(
                    pow_solver.solve, pw.get("seed", ""), pw.get("difficulty", "0"),
                    script=script, build_id=self._build_id or None,
                    user_agent=config.USER_AGENT)
            except RuntimeError as e:
                raise ChatGPTError(503, str(e))
        ttl = int(j.get("expire_after") or config.REQUIREMENTS_TTL)
        self._req = Requirements(token=j.get("token", ""), proof=proof, expires_at=now + min(ttl, config.REQUIREMENTS_TTL))
        return self._req

    async def _ensure_build_info(self) -> None:
        if self._build_id:
            return
        s = await self.http()
        try:
            r = await s.get(f"{CHATGPT_ORIGIN}/?oai-dm=1", headers={"Accept": "*/*"},
                            timeout=30)
            if r.status_code == 200:
                m = re.search(r'data-build="([^"]+)"', r.text)
                self._build_id = m.group(1) if m else ""
                self._scripts = re.findall(r'<script[^>]+src="([^"]+)"', r.text)[:3]
        except Exception as e:
            log.debug("build info fetch failed: %s", e)

    # ---------- conversation ----------
    async def stream_conversation(
        self,
        *,
        prompt_text: str,
        image_pointers: list[dict[str, Any]] | None = None,
        attachments: list[dict[str, Any]] | None = None,
        parent_message_id: str,
        conversation_id: str | None = None,
        model: str = "auto",
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield raw SSE events of one conversation turn.

        prompt_text: user text (may be empty when only images are attached)
        image_pointers: ready image_asset_pointer dicts
        attachments: document descriptors -> metadata.attachments
        """
        await self.ensure_token()
        await self._ensure_build_info()  # warm build id/scripts for proof config
        # Image generation and editing are gated on a logged-in session server-side.
        # Sending the request cookie-only (expired JWT) silently degrades ChatGPT to
        # an anonymous conversation that answers image asks with "requires login"
        # refusals. Fail fast with 401 so the pool rotates to an account that can
        # actually serve images.
        if self.dead:
            raise ChatGPTError(
                401,
                f"[{self.email}] session is stale (dead); re-login this account "
                f"to generate images",
            )
        if not (self.access_token and time.time() < self._jwt_exp - 60):
            if not await self.refresh_access_token():
                raise ChatGPTError(
                    401,
                    f"[{self.email}] no valid access token "
                    f"({'session removed' if self._closed else 'stale session cookie'}); "
                    f"re-login this account to generate images",
                )
        image_pointers = [p for p in (image_pointers or []) if isinstance(p, dict)]
        parts: list[Any] = ([prompt_text] if prompt_text else []) + image_pointers
        has_media = bool(image_pointers)
        metadata: dict[str, Any] = {"attachments": attachments} if attachments else {}
        body: dict[str, Any] = {
            "action": "next",
            "messages": [{
                "id": str(uuid.uuid4()),
                "author": {"role": "user"},
                "content": {"content_type": "multimodal_text" if has_media else "text",
                            "parts": parts},
                "metadata": metadata,
            }],
            "parent_message_id": parent_message_id,
            "model": model,
            "timezone_offset_min": 0,
            "history_and_training_disabled": False,
            "conversation_mode": {"kind": "primary_assistant"},
            "websocket_request_id": str(uuid.uuid4()),
        }
        if conversation_id:
            body["conversation_id"] = conversation_id
        s = await self.http()
        r = None
        try:
            for attempt in (0, 1, 2):  # 403: fresh sentinel token; 401: refresh token
                req = await self.requirements(force=True)
                r = await s.post(
                    f"{CHATGPT_ORIGIN}/backend-api/conversation",
                    headers={
                        **self.base_headers(),
                        "Content-Type": "application/json",
                        "Origin": CHATGPT_ORIGIN,
                        "Referer": f"{CHATGPT_ORIGIN}/",
                        "Openai-Sentinel-Chat-Requirements-Token": req.token,
                        "Openai-Sentinel-Proof-Token": req.proof,
                    },
                    json=body,
                    stream=True,
                    timeout=300,
                )
                if r.status_code == 403 and attempt < 2:
                    await r.aclose()
                    log.warning("[%s] conversation 403, retrying with fresh sentinel token", self.email)
                    continue
                if r.status_code == 401 and attempt < 2 and await self.refresh_access_token():
                    await r.aclose()
                    log.warning("[%s] conversation 401, retried after token refresh", self.email)
                    continue
                break
            if r.status_code == 401:
                raise ChatGPTError(401, "unauthorized")
            if r.status_code != 200:
                # 429/403 bodies (quota/sentinel pages) get a tighter excerpt.
                limit = 300 if r.status_code in (429, 403) else 500
                try:
                    err_text = await r.atext()
                except Exception:
                    # Upstream aborted the error body mid-read. The HTTP status
                    # is the actionable part -- pool cooldown keys off e.status
                    # -- so never let a raw transport exception mask it.
                    err_text = ""
                raise ChatGPTError(r.status_code, err_text[:limit])
            async for raw in r.aiter_lines():
                if not isinstance(raw, str):
                    raw = raw.decode()
                if not raw.startswith("data: "):
                    continue
                payload = raw[6:].strip()
                if payload == "[DONE]":
                    yield {"type": "done"}
                    break
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    continue
        finally:
            # covers retry abandonment, error raises, [DONE] breaks, and client disconnects
            if r is not None:
                with contextlib.suppress(Exception):
                    await r.aclose()

    # ---------- models ----------
    async def models(self) -> list[dict[str, Any]]:
        now = time.time()
        if self._models_cache and now - self._models_cache[0] < 3600:
            return self._models_cache[1]
        await self.ensure_token()
        s = await self.http()
        r = await s.get(f"{CHATGPT_ORIGIN}/backend-api/models",
                        headers=self.base_headers(), timeout=30)
        if r.status_code != 200:
            raise ChatGPTError(r.status_code, "models failed")
        j = r.json()
        mlist = j.get("models") if isinstance(j, dict) else j
        if isinstance(mlist, list) and len(mlist) == 2 and isinstance(mlist[1], list):
            mlist = mlist[1]  # some accounts get [timestamp, models[]]
        self._models_cache = (now, [m for m in (mlist or []) if isinstance(m, dict)])
        return self._models_cache[1]

    # ---------- files ----------
    async def upload_file(self, name: str, data: bytes, mime: str, *, is_image: bool) -> str:
        use_case = "multimodal" if is_image else "ace_upload"
        await self.ensure_token()
        headers = {**self.base_headers(), "Content-Type": "application/json"}
        # /files/{id}/uploaded rejects cookie-only auth: create + blob PUT succeed
        # without a Bearer but finalize always 401s, so fail fast instead.
        if "Authorization" not in headers:
            raise ChatGPTError(
                401,
                f"[{self.email}] access token expired and refresh cannot renew it "
                f"(stale session cookie); re-login this account to upload files",
            )
        s = await self.http()
        r = await s.post(
            f"{CHATGPT_ORIGIN}/backend-api/files",
            headers=headers,
            json={"file_name": name, "file_size": len(data), "use_case": use_case},
            timeout=60,
        )
        if r.status_code != 200:
            raise ChatGPTError(r.status_code, f"file create failed: {r.text[:200]}")
        j = r.json()
        file_id, upload_url = j.get("file_id"), j.get("upload_url")
        if not file_id or not upload_url:
            raise ChatGPTError(500, f"unexpected file create response: {str(j)[:200]}")
        put = await s.put(upload_url, content=data,
                          headers={"x-ms-blob-type": "BlockBlob", "x-ms-version": "2020-04-08"},
                          timeout=120)
        if put.status_code not in (200, 201):
            raise ChatGPTError(put.status_code, f"blob upload failed: {put.status_code}")
        done = await s.post(
            f"{CHATGPT_ORIGIN}/backend-api/files/{file_id}/uploaded",
            headers=headers,
            json={}, timeout=60,
        )
        if done.status_code != 200:
            raise ChatGPTError(done.status_code, f"upload finalize failed: {done.text[:200]}")
        return file_id

    async def wait_file_ready(self, file_id: str, timeout_s: float = 20) -> bool:
        s = await self.http()
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            r = await s.get(f"{CHATGPT_ORIGIN}/backend-api/files/{file_id}",
                            headers=self.base_headers(), timeout=30)
            if r.status_code == 200:
                # The endpoint moved the processing flag from "status" to
                # "state"; read either so the poll can actually observe it.
                j = r.json()
                st = j.get("status") or j.get("state")
                if st in ("success", "ready"):
                    return True
                if st == "error":
                    return False
            await asyncio.sleep(0.5)
        return False

    async def download_file_url(self, pointer: str) -> tuple[str, bytes]:
        """Resolve sediment:// or file-service:// pointers to (filename, bytes)."""
        file_id = pointer.split("//", 1)[1]
        await self.ensure_token()
        s = await self.http()
        r = await s.get(f"{CHATGPT_ORIGIN}/backend-api/files/{file_id}/download",
                        headers=self.base_headers(), timeout=60)
        if r.status_code != 200:
            raise ChatGPTError(r.status_code, f"download resolve failed: {r.text[:200]}")
        j = r.json()
        url = j.get("download_url")
        if not url:
            raise ChatGPTError(500, "no download_url")
        img = await s.get(url, timeout=120)
        if img.status_code != 200:
            raise ChatGPTError(img.status_code, "image bytes fetch failed")
        name = j.get("file_name") or f"{file_id}.png"
        return name, img.content


def _jwt_exp(token: str) -> float:
    """Best-effort exp claim from an RS256 access-token JWT."""
    try:
        payload = token.split(".")[1]
        import base64
        padded = payload + "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(padded))
        return float(claims.get("exp") or 0)
    except Exception:
        return 0.0


def _parse_expires(value: Any) -> float:
    if not value:
        return 0.0
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.timestamp() if dt.tzinfo else dt.replace(tzinfo=timezone.utc).timestamp()
    except Exception:
        return 0.0
