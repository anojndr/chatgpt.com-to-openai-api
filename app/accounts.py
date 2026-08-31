"""Account pool: dynamic loading, load balancing, health/cooldown management.

- Parses accounts.txt ('<<<' ... '>>>' blocks). Any number of accounts; file is
  watched and new accounts are picked up within ~2 seconds without restart.
  A block whose stored session JWT is meaningfully fresher than the in-memory
  one is hot-swapped, so pasting a fresh cookie export instantly revives that
  account without restarting.
- Self-persisting sessions: after every successful token refresh, the fresh
  session JSON and rotated session cookies are written back into that
  account's block (atomic replace under an flock). accounts.txt therefore
  tracks the live session state and never rots back to a stale export.
- Keepalive: a background sweep refreshes access tokens before they expire
  (which re-issues the rolling session cookie server-side) and re-probes dead
  accounts every KEEPALIVE_REVIVE_SECONDS. As long as ChatGPT keeps accepting
  a stored cookie jar, that jar keeps working indefinitely — accounts no
  longer expire just because the proxy idles or restarts.
- Selection: least in-flight, then least-recently-used (fair round-robin under
  uniform load). Plus accounts get shorter 429 cooldowns, so they re-enter the
  pool sooner.
"""
from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import logging
import os
import time

from . import config
from .chatgpt import (
    AccountSession,
    BLOCK_RE,
    _jwt_exp,
    _parse_expires,
    block_identity,
    block_session,
    parse_accounts_text,
)

log = logging.getLogger("pool")


class NoAccountAvailable(Exception):
    def __init__(self, msg: str = "no ChatGPT account available"):
        super().__init__(msg)


@contextlib.contextmanager
def _exclusive_lock(path):
    """Cross-process writer lock for accounts.txt (stable lockfile inode).

    Single non-blocking attempt: on contention raise TimeoutError immediately
    so the async event loop never sleeps on a retry — the 2s watch cadence
    retries with state_dirty preserved.
    """
    lock_path = path.with_name(path.name + ".lock")
    with open(lock_path, "w") as lf:
        try:
            fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            raise TimeoutError(f"lock busy: {lock_path}") from e
        try:
            yield
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


class AccountPool:
    def __init__(self):
        self._accounts: dict[str, AccountSession] = {}
        self._mtime: float = 0.0

    # ---------- lifecycle ----------
    def load(self) -> None:
        try:
            text = config.ACCOUNTS_FILE.read_text()
            self._mtime = config.ACCOUNTS_FILE.stat().st_mtime
        except FileNotFoundError:
            log.error("accounts file not found: %s", config.ACCOUNTS_FILE)
            return
        parsed = parse_accounts_text(text)
        seen: set[str] = set()
        for p in parsed:
            ident = p["identity"]
            if not ident:
                continue
            seen.add(ident)
            existing = self._accounts.get(ident)
            if existing is None:
                self._accounts[ident] = AccountSession(p)
                log.info("account added: %s (%s)", p["session"].get("user", {}).get("email", "?"), p["session"].get("account", {}).get("planType"))
            else:
                self._hot_swap(existing, p)
        for ident in list(self._accounts):
            if ident not in seen:
                removed = self._accounts.pop(ident)
                log.info("account removed: %s", removed.email)
                self._schedule_close(removed)

    @staticmethod
    def _hot_swap(existing: AccountSession, parsed: dict) -> None:
        """Adopt a block from accounts.txt when it carries a fresher export
        for this identity (the user pasted a new cookie jar while running)."""
        if existing.inflight > 0 or existing._closed or existing._refreshing:
            return  # swap after the in-flight request/refresh drains (next tick)
        s = parsed["session"]
        tok = s.get("accessToken")
        if not tok:
            return
        fresh = _jwt_exp(tok)
        # One rule for alive and dead accounts alike: adopt only a meaningfully
        # newer export. This also keeps a dead account dead when the paste is
        # no fresher than whatever died.
        if not fresh or fresh <= existing._jwt_exp + config.KEEPALIVE_MIN_IMPROVEMENT:
            return  # nothing meaningfully newer on disk
        email = (s.get("user") or {}).get("email", existing.email)
        log.info("[%s] adopting fresher session from %s (jwt exp %s -> %s)",
                 email, config.ACCOUNTS_FILE.name, existing._jwt_exp, fresh)
        existing.session_json = s
        existing.access_token = tok
        existing._jwt_exp = fresh or 0.0
        exp = _parse_expires(s.get("expires"))
        if exp:
            existing.expires_at = exp
        plan = (s.get("account") or {}).get("planType")
        if plan:
            existing.plan = plan
        if email:
            existing.email = email
        existing.cookies = dict(parsed["cookies"])
        existing.raw_cookie_lines = list(parsed.get("raw_cookie_lines") or [])
        existing.refresh_strikes = 0
        existing.state_dirty = False
        existing.dead = fresh <= time.time()  # an already-expired paste stays dead
        existing.revive_after = 0.0
        old, existing._http = existing._http, None
        existing._req = None
        if old is not None:
            async def _close():
                with contextlib.suppress(Exception):
                    await old.close()
            try:
                asyncio.get_running_loop().create_task(_close())
            except RuntimeError:
                pass  # no loop (sync load at import time); GC will clean up

    @staticmethod
    def _schedule_close(acct: AccountSession) -> None:
        """Close a removed account once in-flight work drains (or immediately)."""

        async def _close_when_idle():
            for _ in range(300):  # up to 5 min, then force-close
                if acct.inflight == 0:
                    break
                await asyncio.sleep(1)
            await acct.close()

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_close_when_idle())
        except RuntimeError:
            pass  # no running loop (sync load at import time); GC will clean up

    async def watch(self) -> None:
        while True:
            await asyncio.sleep(2)
            try:
                mtime = config.ACCOUNTS_FILE.stat().st_mtime
                if mtime != self._mtime:
                    self.load()
                    log.info("reloaded accounts (%d total)", len(self._accounts))
                # AFTER load: never persist before adopting a concurrent user
                # edit, or the edit would be clobbered and the reload masked.
                self._persist_state()
            except FileNotFoundError:
                continue
            except Exception as e:  # noqa: BLE001
                log.warning("watch error: %s", e)

    async def start_watcher(self) -> None:
        self.load()
        asyncio.create_task(self.watch())
        asyncio.create_task(self.keepalive())

    # ---------- keepalive ----------
    async def keepalive(self) -> None:
        """Keep every stored session alive indefinitely: refresh access tokens
        before they lapse (which re-issues the rolling session cookie and gets
        persisted back to accounts.txt), and re-probe dead accounts so a
        pasted/fixed cookie jar recovers without a restart."""
        while True:
            await asyncio.sleep(config.KEEPALIVE_SECONDS)
            try:
                await self._keepalive_tick()
            except Exception as e:  # noqa: BLE001
                log.warning("keepalive error: %s", e)

    async def _keepalive_tick(self) -> None:
        now = time.time()
        for acct in list(self._accounts.values()):
            if acct._closed:
                continue
            if acct.dead:
                if now >= acct.revive_after:
                    if acct._refreshing or time.time() - acct._last_refresh_attempt < 60:
                        continue  # a real attempt is running/concluded; re-check next tick
                    log.info("[%s] probing dead account for revival", acct.email)
                    await acct.refresh_access_token()  # success clears dead
                    acct.revive_after = now + config.KEEPALIVE_REVIVE_SECONDS
                continue
            if acct.access_token and now < acct._jwt_exp - config.KEEPALIVE_REFRESH_WITHIN:
                continue  # token comfortably valid
            if time.time() - acct._last_refresh_attempt < 60:
                continue  # a request path refreshed moments ago
            await acct.refresh_access_token()
        self._persist_state()

    # ---------- persistence ----------
    def _persist_state(self) -> None:
        """Write refreshed session JSON + rotated cookies back into the
        accounts.txt blocks of accounts whose state changed.

        A block whose on-disk JWT is already newer than the in-memory one is
        never overwritten (an external edit or sibling instance raced us);
        instead load() runs afterwards to adopt it.
        """
        dirty = {a.identity: a for a in self._accounts.values() if a.state_dirty}
        if not dirty:
            return
        path = config.ACCOUNTS_FILE
        try:
            with _exclusive_lock(path):
                text = path.read_text()
                new_text, written, fresher = self._replace_blocks(text, dirty)
                if not written:
                    for a in dirty.values():
                        a.state_dirty = a.identity in fresher
                    if fresher:
                        self.load()  # adopt the raced edit now
                    return
                tmp = path.with_name(path.name + ".tmp")
                tmp.write_text(new_text)
                os.replace(tmp, path)
        except (OSError, TimeoutError) as e:
            log.warning("accounts.txt persist failed: %s", e)
            return
        try:
            self._mtime = path.stat().st_mtime
        except FileNotFoundError:
            pass
        for a in dirty.values():
            a.state_dirty = a.identity in fresher
        if fresher:
            self.load()  # adopt the raced edit now
        written_accts = [a for a in dirty.values() if a.identity not in fresher]
        if written_accts:
            log.info("persisted refreshed auth state for %s",
                     ", ".join(a.email or a.identity for a in written_accts))

    @staticmethod
    def _replace_blocks(text: str, dirty: dict[str, AccountSession]) -> tuple[str, list[str], set[str]]:
        """Replace '<<<' ... '>>>' bodies of dirty accounts. Returns the new
        text, the replaced identities, and identities left untouched because
        the on-disk block carries a strictly newer JWT than memory."""
        written: list[str] = []
        fresher: set[str] = set()
        out: list[str] = []
        pos = 0
        for m in BLOCK_RE.finditer(text):
            ident = block_identity(m.group(1))
            acct = dirty.get(ident) if ident else None
            if acct is None:
                continue
            disk = block_session(m.group(1))
            if disk:
                disk_exp = _jwt_exp(disk.get("accessToken") or "")
                if disk_exp and disk_exp > acct._jwt_exp:
                    fresher.add(ident)  # raced edit wins; load() adopts below
                    continue
            out.append(text[pos:m.start()])
            out.append(_render_block_body(acct))
            pos = m.end()
            written.append(ident)
        if not written:
            return text, written, fresher
        out.append(text[pos:])
        return "".join(out), written, fresher

    # ---------- selection ----------
    def available(self) -> list[AccountSession]:
        now = time.time()
        return [a for a in self._accounts.values() if not a.dead and a.cooldown_until <= now]

    def _find(self, key: str) -> AccountSession | None:
        a = self._accounts.get(key)
        if a is None:
            a = next((x for x in self._accounts.values() if x.email.lower() == key.lower()), None)
        return a

    def acquire(self, preferred_identity: str | None = None,
                exclude: frozenset[str] | set[str] = frozenset()) -> AccountSession:
        """Acquire an account, never returning one whose identity is excluded.

        Used by the engine's failover loop to walk every available account
        exactly once before giving up.
        """
        if preferred_identity and preferred_identity not in exclude:
            a = self._find(preferred_identity)
            if a and not a.dead and a.cooldown_until <= time.time():
                a.inflight += 1
                a.last_used = time.time()
                return a
        now = time.time()
        cands = [a for a in self._accounts.values()
                 if not a.dead and a.cooldown_until <= now and a.identity not in exclude]
        if not cands:
            raise NoAccountAvailable()
        cands.sort(key=lambda a: (a.inflight, a.last_used))
        chosen = cands[0]
        chosen.inflight += 1
        chosen.last_used = time.time()
        return chosen

    def release(self, acct: AccountSession) -> None:
        acct.inflight = max(0, acct.inflight - 1)

    def report_status(self, acct: AccountSession, status: int) -> None:
        """Update account health after a conversation attempt."""
        now = time.time()
        if status == 429:
            cd = config.COOLDOWN_PLUS_SECONDS if acct.plan == "plus" else config.COOLDOWN_FREE_SECONDS
            acct.cooldown_until = now + cd
            log.warning("[%s] rate limited, cooling down %ss", acct.email, cd)
        elif status == 401:
            # A 401 from a confirmed-stale session (dead=True set by
            # refresh_access_token) must stay out of rotation entirely;
            # transient 401s just get a short cooldown.
            if not acct.dead:
                acct.cooldown_until = now + 5
        elif status == 403:
            acct.cooldown_until = now + 10

    def snapshot(self) -> list[dict]:
        now = time.time()
        return [{
            "email": a.email,
            "plan": a.plan,
            "inflight": a.inflight,
            "total_requests": a.total_requests,
            "cooldown_s": max(0, round(a.cooldown_until - now)),
            "dead": a.dead,
            "refresh_strikes": a.refresh_strikes,
            "token_expires_in_s": max(0, round(a._jwt_exp - now)) if a._jwt_exp else None,
        } for a in self._accounts.values()]


def _render_block_body(acct: AccountSession) -> str:
    """Render the '<<<' ... '>>>' body for one account, mirroring the format
    parse_accounts_text expects (netscape cookie rows + session JSON)."""
    live = dict(acct.session_json)
    live["accessToken"] = acct.access_token
    lines = [
        "<<<",
        "netscape cookies:",
        "",
    ]
    lines.extend(acct.raw_cookie_lines)
    lines += [
        "",
        "from https://chatgpt.com/api/auth/session:",
        "",
        json.dumps(live, ensure_ascii=False),
        ">>>",
        "",
    ]
    return "\n".join(lines)


POOL = AccountPool()
