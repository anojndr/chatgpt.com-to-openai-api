# Copyright 2026 chatgpt-to-openai-api contributors.
"""Async client for chatgpt.com's internal backend-api (no browser).

Protocol per account session (TLS-impersonated via curl_cffi):
  1. GET  /api/auth/session            -> fresh accessToken from session cookies
  2. GET  /?oai-dm=1                   -> build id + script paths (proof config)
  3. POST /sentinel/chat-requirements  -> {token, proofofwork} cached briefly
  4. POST /conversation                -> SSE; requires sentinel headers
  5. files: POST /files, PUT blob, POST /files/{id}/uploaded; file downloads
     via /files/{id}/download
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from http.cookies import CookieError, SimpleCookie
from typing import TYPE_CHECKING, Any

from curl_cffi.requests import AsyncSession
from curl_cffi.requests.exceptions import RequestException

from . import config, pow_solver

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from curl_cffi.requests import Response

log = logging.getLogger("chatgpt")

CHATGPT_ORIGIN = "https://chatgpt.com"

HTTP_OK = 200
HTTP_UNAUTHORIZED = 401
HTTP_FORBIDDEN = 403
HTTP_TOO_MANY_REQUESTS = 429

TOKEN_EXPIRY_SKEW_S = 60.0
TOKEN_EAGER_REFRESH_WINDOW_S = 600.0
REFRESH_DEDUP_WINDOW_S = 5.0
REFRESH_NUDGE_INTERVAL_S = 60.0
SENTINEL_REUSE_LEEWAY_S = 30.0
MODELS_CACHE_TTL_S = 3600.0
MAX_CONVERSATION_RETRIES = 2
TIMESTAMPED_MODELS_LEN = 2
NETSCAPE_FIELD_COUNT = 7


@dataclass
class Requirements:
    """Cached sentinel proof-of-work requirements for one account."""

    token: str
    proof: str
    expires_at: float


@dataclass
class StreamMedia:
    """Media inputs for one conversation turn."""

    image_pointers: list[dict[str, Any]] | None = None
    attachments: list[dict[str, Any]] | None = None


class ChatGPTError(Exception):
    """Backend-api failure carrying its HTTP status."""

    def __init__(self, status: int, message: str) -> None:
        """Store the HTTP status alongside the message."""
        super().__init__(message)
        self.status = status
        self.message = message


BLOCK_RE = re.compile(r"^<<<\s*$(.*?)^>>>\s*$", re.DOTALL | re.MULTILINE)


def parse_accounts_text(text: str) -> list[dict[str, Any]]:
    """Parse accounts.txt blocks into session dicts.

    Each '<<<' ... '>>>' block holds a netscape cookie jar and the JSON
    body of https://chatgpt.com/api/auth/session.
    """
    accounts: list[dict[str, Any]] = []
    for b in BLOCK_RE.findall(text):
        m = re.search(
            r"from https://chatgpt\.com/api/auth/session:\s*(\{.*)", b, re.DOTALL
        )
        if not m:
            continue
        try:
            sess = json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            continue
        cookies: dict[str, str] = {}
        raw_cookie_lines: list[str] = []
        for raw_line in b.splitlines():
            stripped = raw_line.strip()
            marker = "#HttpOnly_" if stripped.startswith("#HttpOnly_") else ""
            bare = stripped[len(marker) :]  # standard netscape-jar HttpOnly marker
            if not bare or bare.startswith("#") or "\t" not in bare:
                continue
            parts = bare.split("\t")
            if len(parts) >= NETSCAPE_FIELD_COUNT:
                cookies[parts[5]] = parts[6]
                raw_cookie_lines.append(stripped)  # marker kept for round-trip
        if not sess.get("accessToken"):
            continue
        accounts.append(
            {
                "session": sess,
                "cookies": cookies,
                "raw_cookie_lines": raw_cookie_lines,
                "identity": sess.get("user", {}).get("id")
                or sess.get("account", {}).get("id"),
            },
        )
    return accounts


def block_session(block_text: str) -> dict[str, Any] | None:
    """Session JSON of one raw '<<<' ... '>>>' block body, or None."""
    m = re.search(
        r"from https://chatgpt\.com/api/auth/session:\s*(\{.*)",
        block_text,
        re.DOTALL,
    )
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
    user = sess.get("user")
    user_id = user.get("id") if isinstance(user, dict) else None
    if isinstance(user_id, str) and user_id:
        return user_id
    account = sess.get("account")
    account_id = account.get("id") if isinstance(account, dict) else None
    return account_id if isinstance(account_id, str) and account_id else None


def _set_cookie_headers(response: object) -> list[str]:
    """Collect raw Set-Cookie header values from a response.

    Understands curl responses (get_list) and generic mappings
    (multi_items), so one entry per header survives Expires commas.
    """
    try:
        headers = getattr(response, "headers", None)
        get_list = getattr(headers, "get_list", None)
        if get_list is not None:
            values = get_list("set-cookie") or []
            return [v for v in values if isinstance(v, str)]
        multi_items = getattr(headers, "multi_items", None)
        if multi_items is not None:
            return [
                v
                for k, v in multi_items()
                if isinstance(k, str)
                and k.lower() == "set-cookie"
                and isinstance(v, str)
            ]
    except (AttributeError, TypeError, ValueError) as e:
        log.debug("set-cookie header read failed: %s", e)
    return []


def _parse_cookie_jar(headers: list[str]) -> SimpleCookie:
    """Parse raw Set-Cookie values, skipping malformed entries."""
    jar = SimpleCookie()
    for header in headers:
        if not header:
            continue
        try:
            jar.load(header)
        except (CookieError, TypeError, ValueError) as e:
            log.debug("skipping malformed set-cookie: %s", e)
            continue
    return jar


def _conversation_body(
    prompt_text: str,
    media: StreamMedia | None,
    parent_message_id: str,
    model: str,
) -> dict[str, Any]:
    """Build the POST /conversation JSON body for one turn."""
    raw_pointers = media.image_pointers if media else None
    image_pointers = [p for p in (raw_pointers or []) if isinstance(p, dict)]
    parts: list[str | dict[str, Any]] = ([prompt_text] if prompt_text else []) + (
        image_pointers
    )
    has_media = bool(image_pointers)
    attachments = media.attachments if media else None
    metadata: dict[str, Any] = {"attachments": attachments} if attachments else {}
    return {
        "action": "next",
        "messages": [
            {
                "id": str(uuid.uuid4()),
                "author": {"role": "user"},
                "content": {
                    "content_type": "multimodal_text" if has_media else "text",
                    "parts": parts,
                },
                "metadata": metadata,
            },
        ],
        "parent_message_id": parent_message_id,
        "model": model,
        "timezone_offset_min": 0,
        "history_and_training_disabled": False,
        "conversation_mode": {"kind": "primary_assistant"},
        "websocket_request_id": str(uuid.uuid4()),
    }


class AccountSession:
    """One ChatGPT account: auth state + HTTP session + sentinel cache."""

    def __init__(self, parsed: dict[str, Any]) -> None:
        """Capture one account's session export and reset runtime state."""
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
        self._refreshing: bool = False  # network refresh in flight
        self._refresh_ok: bool = False  # last completed refresh succeeded
        # http/sentinel state
        self._http: AsyncSession[Response] | None = None
        self._req: Requirements | None = None
        self._build_id: str = ""
        self._scripts: list[str] = []
        self._models_cache: tuple[float, list[dict[str, Any]]] | None = None
        self._last_refresh_attempt: float = 0.0
        self._closed: bool = False

    @property
    def jwt_exp(self) -> float:
        """Expiry of the current access token as a Unix timestamp."""
        return self._jwt_exp

    @jwt_exp.setter
    def jwt_exp(self, value: float) -> None:
        """Record a new access-token expiry timestamp."""
        self._jwt_exp = value

    @property
    def is_closed(self) -> bool:
        """Whether this session was removed from the pool."""
        return self._closed

    @property
    def is_refreshing(self) -> bool:
        """Whether a token refresh is currently in flight."""
        return self._refreshing

    @property
    def last_refresh_attempt(self) -> float:
        """Unix timestamp of the last refresh attempt."""
        return self._last_refresh_attempt

    @last_refresh_attempt.setter
    def last_refresh_attempt(self, value: float) -> None:
        """Record the timestamp of a refresh attempt."""
        self._last_refresh_attempt = value

    @property
    def last_strike_at(self) -> float:
        """Unix timestamp of the last refresh strike."""
        return self._last_strike_at

    @last_strike_at.setter
    def last_strike_at(self, value: float) -> None:
        """Record the timestamp of a refresh strike."""
        self._last_strike_at = value

    @property
    def refresh_ok(self) -> bool:
        """Whether the last completed refresh succeeded."""
        return self._refresh_ok

    @refresh_ok.setter
    def refresh_ok(self, value: bool) -> None:
        """Record the outcome of the last completed refresh."""
        self._refresh_ok = value

    # ---------- http ----------
    async def http(self) -> AsyncSession[Response]:
        """Return the live TLS-impersonated session, creating it on first use."""
        if self._closed:
            raise ChatGPTError(
                503,
                f"session for {self.email} was removed from the pool",
            )
        if self._http is None:
            self._http = AsyncSession(impersonate=config.IMPERSONATE)
            self._http.headers.update({"User-Agent": config.USER_AGENT})
            self._http.cookies.update(self.cookies)
        return self._http

    def base_headers(self) -> dict[str, str]:
        """Return the per-request headers shared by every backend call."""
        h = {
            "Oai-Device-Id": self.cookies.get("oai-did", str(uuid.uuid4())),
            "Oai-Language": "en-US",
            "Accept": "*/*",
        }
        # Session cookies alone authenticate backend-api; include Bearer only when
        # the access-token JWT is actually valid (expired tokens cause 401s).
        if self.access_token and time.time() < self._jwt_exp - TOKEN_EXPIRY_SKEW_S:
            h["Authorization"] = f"Bearer {self.access_token}"
        if self.account_id:
            h["Chatgpt-Account-Id"] = self.account_id
        return h

    async def close(self) -> None:
        """Close the HTTP session and mark this account removed."""
        self._closed = True
        if self._http is not None:
            await self._http.close()
            self._http = None

    def detach_transport(self) -> AsyncSession[Response] | None:
        """Detach the HTTP session and drop the sentinel cache.

        Return the detached session, if any, for the caller to close.
        """
        old = self._http
        self._http = None
        self._req = None
        return old

    # ---------- auth ----------
    async def refresh_access_token(self) -> bool:
        """Refresh via /api/auth/session.

        On any successful refresh the rotated session cookies are absorbed
        into self.cookies and marked dirty; the pool persists dirty state
        into accounts.txt.

        Failure policy: transport errors / non-200 / non-JSON responses are
        transient (return False, no strike). A 200 that cannot yield a
        fresh-enough token counts one strike; KEEPALIVE_MAX_STRIKES spaced
        strikes mean the session cookie itself is stale -> dead (needs
        re-login).
        """
        try:
            # One refresh per account at a time; request paths and the
            # keepalive sweep all funnel through here. While one is in
            # flight (or concluded recently) report the cached outcome
            # instead of firing a second request.
            if (
                self._refreshing
                or time.time() - self._last_refresh_attempt < REFRESH_DEDUP_WINDOW_S
            ):
                return bool(
                    self._refresh_ok
                    and self.access_token
                    and time.time() < self._jwt_exp - TOKEN_EXPIRY_SKEW_S,
                )
            self._last_refresh_attempt = time.time()
            self._refreshing = True
            try:
                return await self._do_refresh()
            finally:
                self._refreshing = False
        except (ChatGPTError, RequestException, OSError, ValueError) as e:
            self._refresh_ok = False
            log.warning("[%s] refresh failed: %s", self.email, e)
            return False

    async def _do_refresh(self) -> bool:
        self._refresh_ok = False
        s = await self.http()
        r = await s.get(
            f"{CHATGPT_ORIGIN}/api/auth/session",
            headers={"Accept": "*/*"},
            timeout=30,
        )
        if r.status_code != HTTP_OK:
            log.warning("[%s] session refresh HTTP %s", self.email, r.status_code)
            return False
        try:
            j = r.json()
        except (RequestException, ValueError):
            log.warning("[%s] session refresh returned non-JSON body", self.email)
            return False
        self._absorb_cookies(r)
        new_at = j.get("accessToken")
        if not new_at:
            return self._record_refresh_failure("session returned no accessToken")
        new_exp = _jwt_exp(new_at)
        if not new_exp or new_exp <= time.time() + TOKEN_EXPIRY_SKEW_S:
            # /api/auth/session can keep serving a cached token past its
            # exp when the session cookie itself is stale. Never adopt it.
            kind = "unparseable" if not new_exp else "expired"
            remaining = max(0.0, (new_exp or 0.0) - time.time())
            return self._record_refresh_failure(
                f"session returned an {kind} access token ({remaining}s left)"
            )
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
        """Count one confirmed-but-unusable session response.

        Strikes older than two sweep intervals are forgiven first, so only
        KEEPALIVE_MAX_STRIKES consecutive failures mark the account dead.
        """
        now = time.time()
        if now - self._last_strike_at > 2 * config.KEEPALIVE_SECONDS:
            self.refresh_strikes = 0  # stale evidence: not a consecutive run
        self._last_strike_at = now
        self.refresh_strikes += 1
        if self.refresh_strikes >= config.KEEPALIVE_MAX_STRIKES:
            self.dead = True
            self.revive_after = now + config.KEEPALIVE_REVIVE_SECONDS
            log.warning(
                "[%s] needs re-login (%s; %d strikes)",
                self.email,
                detail,
                self.refresh_strikes,
            )
        else:
            log.warning(
                "[%s] refresh unusable: %s (strike %d/%d)",
                self.email,
                detail,
                self.refresh_strikes,
                config.KEEPALIVE_MAX_STRIKES,
            )
        return False

    def record_refresh_failure(self, detail: str) -> bool:
        """Count one confirmed-but-unusable session response, returning False."""
        return self._record_refresh_failure(detail)

    def _absorb_cookies(self, response: object) -> None:
        """Merge session cookies rotated by the server into account state.

        Regenerate raw_cookie_lines so accounts.txt stays re-exportable.
        """
        headers = _set_cookie_headers(response)
        if not headers:
            return
        jar = _parse_cookie_jar(headers)
        if not jar:
            return
        merged = 0
        for name, morsel in jar.items():
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
        self._sync_cookies_to_http(jar)
        self._rebuild_raw_cookie_lines()
        self.state_dirty = True
        log.info("[%s] absorbed %d rotated session cookie(s)", self.email, merged)

    def absorb_cookies(self, response: object) -> None:
        """Merge server-rotated session cookies into this account."""
        self._absorb_cookies(response)

    def _sync_cookies_to_http(self, jar: SimpleCookie) -> None:
        """Mirror rotated cookies into the live HTTP session, if any."""
        if self._http is None:
            return
        for name in jar:
            if name in self.cookies:
                self._http.cookies.set(
                    name,
                    self.cookies[name],
                    domain=".chatgpt.com",
                    path="/",
                )

    def _rebuild_raw_cookie_lines(self) -> None:
        """Regenerate netscape rows from raw_cookie_lines plus self.cookies.

        Values of cookies the server rotated are replaced in place.
        """
        lines: list[str] = []
        seen: set[str] = set()
        for raw_line in self.raw_cookie_lines:
            parts = raw_line.split("\t")
            if len(parts) >= NETSCAPE_FIELD_COUNT and parts[5] in self.cookies:
                parts[6] = self.cookies[parts[5]]
                seen.add(parts[5])
                lines.append("\t".join(parts))
            else:
                lines.append(raw_line)
        for name, value in self.cookies.items():
            if name in seen:
                continue
            lines.append(f".chatgpt.com\tTRUE\t/\tTRUE\t0\t{name}\t{value}")
        self.raw_cookie_lines = lines

    def rebuild_raw_cookie_lines(self) -> None:
        """Regenerate netscape cookie rows from the current cookie state."""
        self._rebuild_raw_cookie_lines()

    async def ensure_token(self) -> None:
        """Nudge a token refresh when the JWT nears expiry.

        JWT expiry is authoritative; session-cookie auth covers the rest.
        refresh_access_token self-throttles; this gate reads the CURRENT
        attempt stamp (never pre-set it here — that turned this into a
        silent no-op) and spaces nudges so a busy final window cannot
        hammer /api/auth/session.
        """
        if (
            self.access_token
            and time.time() > self._jwt_exp - TOKEN_EAGER_REFRESH_WINDOW_S
            and time.time() - self._last_refresh_attempt > REFRESH_NUDGE_INTERVAL_S
        ):
            await self.refresh_access_token()

    # ---------- sentinel ----------
    async def requirements(self, *, force: bool = False) -> Requirements:
        """Fetch and cache sentinel chat-requirements, solving proof-of-work."""
        now = time.time()
        if (
            not force
            and self._req
            and self._req.expires_at > now + SENTINEL_REUSE_LEEWAY_S
        ):
            return self._req
        s = await self.http()
        pre = pow_solver.pre_proof(user_agent=config.USER_AGENT)
        r = await s.post(
            f"{CHATGPT_ORIGIN}/backend-api/sentinel/chat-requirements",
            headers={**self.base_headers(), "Content-Type": "application/json"},
            json={"p": pre},
            timeout=30,
        )
        if r.status_code == HTTP_FORBIDDEN:
            raise ChatGPTError(HTTP_FORBIDDEN, "sentinel blocked: " + r.text[:200])
        if r.status_code != HTTP_OK:
            raise ChatGPTError(
                r.status_code,
                f"chat-requirements failed: {r.text[:200]}",
            )
        j = r.json()
        pw = j.get("proofofwork") or {}
        proof = ""
        if pw.get("required"):
            script = self._scripts[0] if self._scripts else None
            try:
                # CPU-bound: keep the event loop free for other accounts' streams.
                options = pow_solver.PowOptions(
                    script=script,
                    build_id=self._build_id or None,
                    user_agent=config.USER_AGENT,
                )
                proof = await asyncio.to_thread(
                    pow_solver.solve,
                    pw.get("seed", ""),
                    pw.get("difficulty", "0"),
                    options,
                )
            except RuntimeError as e:
                raise ChatGPTError(503, str(e)) from e
        ttl = int(j.get("expire_after") or config.REQUIREMENTS_TTL)
        self._req = Requirements(
            token=j.get("token", ""),
            proof=proof,
            expires_at=now + min(ttl, config.REQUIREMENTS_TTL),
        )
        return self._req

    async def _ensure_build_info(self) -> None:
        if self._build_id:
            return
        s = await self.http()
        try:
            r = await s.get(
                f"{CHATGPT_ORIGIN}/?oai-dm=1",
                headers={"Accept": "*/*"},
                timeout=30,
            )
            if r.status_code == HTTP_OK:
                m = re.search(r'data-build="([^"]+)"', r.text)
                self._build_id = m.group(1) if m else ""
                self._scripts = re.findall(r'<script[^>]+src="([^"]+)"', r.text)[:3]
        except (RequestException, OSError, ValueError) as e:
            log.debug("build info fetch failed: %s", e)

    # ---------- conversation ----------
    async def stream_conversation(
        self,
        *,
        prompt_text: str,
        media: StreamMedia | None = None,
        parent_message_id: str,
        conversation_id: str | None = None,
        model: str = "auto",
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield raw SSE events of one conversation turn.

        prompt_text: user text (may be empty when only images are attached)
        media: ready image pointers and document attachments, if any
        parent_message_id: id to continue from, or a fresh client UUID
        conversation_id: server conversation to continue, if any
        model: backend model slug
        """
        await self.ensure_token()
        await self._ensure_build_info()  # warm build id/scripts for proof config
        await self._ensure_conversation_auth()
        body = _conversation_body(prompt_text, media, parent_message_id, model)
        if conversation_id:
            body["conversation_id"] = conversation_id
        response = await self._post_conversation(body)
        try:
            await self._raise_for_conversation_error(response)
            async for raw_line in response.aiter_lines():
                raw = raw_line if isinstance(raw_line, str) else raw_line.decode()
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
            # Covers retry abandonment, error raises, [DONE] breaks, and
            # client disconnects.
            with contextlib.suppress(RequestException, OSError, ValueError):
                await response.aclose()

    async def _ensure_conversation_auth(self) -> None:
        """Fail fast when this account cannot serve an authenticated turn.

        Image generation and editing are gated on a logged-in session
        server-side: a cookie-only request silently degrades ChatGPT to an
        anonymous conversation that refuses image asks. Raising 401 lets the
        pool rotate to an account that can actually serve them.
        """
        if self.dead:
            raise ChatGPTError(
                HTTP_UNAUTHORIZED,
                f"[{self.email}] session is stale (dead); re-login this "
                "account to generate images",
            )
        if self.access_token and time.time() < self._jwt_exp - TOKEN_EXPIRY_SKEW_S:
            return
        if not await self.refresh_access_token():
            reason = "session removed" if self._closed else "stale session cookie"
            message = (
                f"[{self.email}] no valid access token ({reason}); "
                "re-login this account to generate images"
            )
            raise ChatGPTError(HTTP_UNAUTHORIZED, message)

    async def _post_conversation(self, body: dict[str, Any]) -> Response:
        """POST one conversation turn, retrying sentinel/token failures.

        A 403 retries with a fresh sentinel token; a 401 retries after a
        token refresh. Return the final response for the caller to stream.
        """
        session = await self.http()
        response = None
        for attempt in range(MAX_CONVERSATION_RETRIES + 1):
            # 403: fresh sentinel token; 401: refresh token
            req = await self.requirements(force=True)
            response = await session.post(
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
            if (
                response.status_code == HTTP_FORBIDDEN
                and attempt < MAX_CONVERSATION_RETRIES
            ):
                await response.aclose()
                log.warning(
                    "[%s] conversation 403, retrying with fresh sentinel token",
                    self.email,
                )
                continue
            if (
                response.status_code == HTTP_UNAUTHORIZED
                and attempt < MAX_CONVERSATION_RETRIES
                and await self.refresh_access_token()
            ):
                await response.aclose()
                log.warning(
                    "[%s] conversation 401, retried after token refresh",
                    self.email,
                )
                continue
            break
        return response

    async def _raise_for_conversation_error(self, response: Response) -> None:
        """Raise ChatGPTError for a non-200 conversation response.

        Reads a short excerpt of the error body; a transport failure while
        reading falls back to an empty body since the HTTP status (which the
        pool cooldown keys off) is the actionable part.
        """
        if response.status_code == HTTP_UNAUTHORIZED:
            raise ChatGPTError(HTTP_UNAUTHORIZED, "unauthorized")
        if response.status_code == HTTP_OK:
            return
        # 429/403 bodies (quota/sentinel pages) get a tighter excerpt.
        limit = (
            300
            if response.status_code in (HTTP_TOO_MANY_REQUESTS, HTTP_FORBIDDEN)
            else 500
        )
        try:
            err_text = await response.atext()
        except (RequestException, OSError, ValueError) as e:
            log.debug("[%s] error-body read failed: %s", self.email, e)
            err_text = ""
        raise ChatGPTError(response.status_code, err_text[:limit])

    # ---------- models ----------
    async def models(self) -> list[dict[str, Any]]:
        """Return the cached backend model list, refreshing it hourly."""
        now = time.time()
        if self._models_cache and now - self._models_cache[0] < MODELS_CACHE_TTL_S:
            return self._models_cache[1]
        await self.ensure_token()
        s = await self.http()
        r = await s.get(
            f"{CHATGPT_ORIGIN}/backend-api/models",
            headers=self.base_headers(),
            timeout=30,
        )
        if r.status_code != HTTP_OK:
            raise ChatGPTError(r.status_code, "models failed")
        j = r.json()
        mlist = j.get("models") if isinstance(j, dict) else j
        if (
            isinstance(mlist, list)
            and len(mlist) == TIMESTAMPED_MODELS_LEN
            and isinstance(mlist[1], list)
        ):
            mlist = mlist[1]  # some accounts get [timestamp, models[]]
        self._models_cache = (now, [m for m in (mlist or []) if isinstance(m, dict)])
        return self._models_cache[1]

    # ---------- files ----------
    async def upload_file(
        self,
        name: str,
        data: bytes,
        mime: str,
        *,
        is_image: bool,
    ) -> str:
        """Upload bytes and return the file id once the blob finalizes."""
        use_case = "multimodal" if is_image else "ace_upload"
        log.debug("[%s] uploading %s (%s, %d bytes)", self.email, name, mime, len(data))
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
        if r.status_code != HTTP_OK:
            raise ChatGPTError(r.status_code, f"file create failed: {r.text[:200]}")
        j = r.json()
        file_id = j.get("file_id")
        upload_url = j.get("upload_url")
        if (
            not isinstance(file_id, str)
            or not file_id
            or not isinstance(upload_url, str)
            or not upload_url
        ):
            raise ChatGPTError(500, f"unexpected file create response: {str(j)[:200]}")
        put = await s.put(
            upload_url,
            content=data,
            headers={"x-ms-blob-type": "BlockBlob", "x-ms-version": "2020-04-08"},
            timeout=120,
        )
        if put.status_code not in (HTTP_OK, 201):
            raise ChatGPTError(
                put.status_code,
                f"blob upload failed: {put.status_code}",
            )
        done = await s.post(
            f"{CHATGPT_ORIGIN}/backend-api/files/{file_id}/uploaded",
            headers=headers,
            json={},
            timeout=60,
        )
        if done.status_code != HTTP_OK:
            raise ChatGPTError(
                done.status_code,
                f"upload finalize failed: {done.text[:200]}",
            )
        return file_id

    async def wait_file_ready(self, file_id: str, timeout_s: float = 20) -> bool:
        """Poll the file endpoint until processing succeeds or times out."""
        s = await self.http()
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            r = await s.get(
                f"{CHATGPT_ORIGIN}/backend-api/files/{file_id}",
                headers=self.base_headers(),
                timeout=30,
            )
            if r.status_code == HTTP_OK:
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
        r = await s.get(
            f"{CHATGPT_ORIGIN}/backend-api/files/{file_id}/download",
            headers=self.base_headers(),
            timeout=60,
        )
        if r.status_code != HTTP_OK:
            raise ChatGPTError(
                r.status_code,
                f"download resolve failed: {r.text[:200]}",
            )
        j = r.json()
        url = j.get("download_url")
        if not isinstance(url, str) or not url:
            raise ChatGPTError(500, "no download_url")
        img = await s.get(url, timeout=120)
        if img.status_code != HTTP_OK:
            raise ChatGPTError(img.status_code, "image bytes fetch failed")
        raw_name = j.get("file_name")
        name = raw_name if isinstance(raw_name, str) and raw_name else f"{file_id}.png"
        content = img.content
        if not isinstance(content, bytes):
            raise ChatGPTError(img.status_code, "image bytes fetch failed")
        return name, content


def _jwt_exp(token: str) -> float:
    """Best-effort exp claim from an RS256 access-token JWT."""
    try:
        payload = token.split(".")[1]
        padded = payload + "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(padded))
        return float(claims.get("exp") or 0)
    except (ValueError, TypeError, AttributeError, IndexError) as e:
        log.debug("unparseable access token: %s", e)
        return 0.0


def _parse_expires(value: object) -> float:
    """Parse an expiry timestamp into Unix seconds, forgiving garbage."""
    if not value:
        return 0.0
    try:
        dt = datetime.fromisoformat(str(value))
        return dt.timestamp() if dt.tzinfo else dt.replace(tzinfo=UTC).timestamp()
    except (ValueError, TypeError, OverflowError, OSError) as e:
        log.debug("unparseable expires value %r: %s", value, e)
        return 0.0
