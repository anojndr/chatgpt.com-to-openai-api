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


def parse_accounts_text(text: str) -> list[dict[str, Any]]:
    """Parse accounts.txt: '<<<' ... '>>>' blocks with a netscape cookie jar and
    the JSON body of https://chatgpt.com/api/auth/session."""
    blocks = re.findall(r"^<<<\s*$(.*?)^>>>\s*$", text, re.S | re.M)
    accounts: list[dict[str, Any]] = []
    for b in blocks:
        m = re.search(r"from https://chatgpt\.com/api/auth/session:\s*(\{.*)", b, re.S)
        if not m:
            continue
        try:
            sess = json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            continue
        cookies: dict[str, str] = {}
        for line in b.splitlines():
            line = line.strip()
            if line.startswith("#HttpOnly_"):
                line = line[len("#HttpOnly_"):]  # standard netscape-jar HttpOnly marker
            if not line or line.startswith("#") or "\t" not in line:
                continue
            parts = line.split("\t")
            if len(parts) >= 7:
                cookies[parts[5]] = parts[6]
        if not sess.get("accessToken"):
            continue
        accounts.append({
            "session": sess,
            "cookies": cookies,
            "identity": sess.get("user", {}).get("id") or sess.get("account", {}).get("id"),
        })
    return accounts


class AccountSession:
    """One ChatGPT account: auth state + HTTP session + sentinel cache."""

    def __init__(self, parsed: dict[str, Any]):
        self.identity: str = parsed["identity"]
        self.session_json: dict[str, Any] = parsed["session"]
        self.cookies: dict[str, str] = dict(parsed["cookies"])
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
        try:
            s = await self.http()
            r = await s.get(f"{CHATGPT_ORIGIN}/api/auth/session",
                            headers={"Accept": "*/*"}, timeout=30)
            if r.status_code != 200:
                log.warning("[%s] session refresh HTTP %s", self.email, r.status_code)
                return False
            j = r.json()
            new_at = j.get("accessToken")
            if not new_at:
                self.dead = True
                log.warning("[%s] session cookie expired (no accessToken)", self.email)
                return False
            self.access_token = new_at
            self._jwt_exp = _jwt_exp(new_at)
            self.session_json = j
            exp = _parse_expires(j.get("expires"))
            if exp:
                self.expires_at = exp
            plan = (j.get("account") or {}).get("planType")
            if plan:
                self.plan = plan
            self._req = None
            log.info("[%s] access token refreshed", self.email)
            return True
        except Exception as e:
            log.warning("[%s] refresh failed: %s", self.email, e)
            return False

    async def ensure_token(self) -> None:
        # JWT expiry is authoritative; session-cookie auth covers us regardless.
        if self.access_token and time.time() > self._jwt_exp - 600:
            if time.time() - self._last_refresh_attempt > 600:
                self._last_refresh_attempt = time.time()
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
            if r.status_code == 429:
                raise ChatGPTError(429, r.text[:300])
            if r.status_code == 403:
                raise ChatGPTError(403, r.text[:300])
            if r.status_code == 401:
                raise ChatGPTError(401, "unauthorized")
            if r.status_code != 200:
                raise ChatGPTError(r.status_code, r.text[:500])
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
        s = await self.http()
        r = await s.post(
            f"{CHATGPT_ORIGIN}/backend-api/files",
            headers={**self.base_headers(), "Content-Type": "application/json"},
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
            headers={**self.base_headers(), "Content-Type": "application/json"},
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
                st = r.json().get("status")
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
