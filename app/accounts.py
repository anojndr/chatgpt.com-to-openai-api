"""Account pool: dynamic loading, load balancing, health/cooldown management.

- Parses accounts.txt ('<<<' ... '>>>' blocks). Any number of accounts; file is
  watched and new accounts are picked up within ~2 seconds without restart.
- Selection: least in-flight, then least-recently-used (fair round-robin under
  uniform load). Plus accounts get shorter 429 cooldowns, so they re-enter the
  pool sooner.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

from . import config
from .chatgpt import AccountSession, parse_accounts_text

log = logging.getLogger("pool")


class NoAccountAvailable(Exception):
    def __init__(self, msg: str = "no ChatGPT account available"):
        super().__init__(msg)


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
        for ident in list(self._accounts):
            if ident not in seen:
                removed = self._accounts.pop(ident)
                log.info("account removed: %s", removed.email)
                self._schedule_close(removed)

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
            except FileNotFoundError:
                continue
            except Exception as e:  # noqa: BLE001
                log.warning("watch error: %s", e)

    async def start_watcher(self) -> None:
        self.load()
        asyncio.create_task(self.watch())

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
        } for a in self._accounts.values()]


POOL = AccountPool()
