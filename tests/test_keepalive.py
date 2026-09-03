# Copyright 2026 chatgpt-to-openai-api contributors.
"""Keepalive, hot-swap, and persistence tests for account sessions."""

from __future__ import annotations

import asyncio
import base64
import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from typing import NoReturn
from unittest.mock import patch

from typing_extensions import override

from app import config
from app.accounts import AccountPool, _render_block_body
from app.chatgpt import AccountSession, parse_accounts_text

SEEDED_STRIKE_COUNT = 2
EXPECTED_BLOCK_TOTAL = 2


class _UnexpectedBodyError(AssertionError):
    """Signal unexpected JSON parsing on a non-200 response."""

    def __init__(self) -> None:
        super().__init__("must not parse body on non-200")


def _mk_jwt(exp: float) -> str:
    """Minimal unsigned JWT with just an exp claim (only exp is ever read)."""
    head = base64.urlsafe_b64encode(b'{"alg":"RS256","typ":"JWT"}').decode().rstrip("=")
    payload = (
        base64.urlsafe_b64encode(
            json.dumps({"exp": exp, "iat": int(exp) - 900}).encode(),
        )
        .decode()
        .rstrip("=")
    )
    return f"{head}.{payload}.sig"


def _mk_session(user_id: str, email: str, exp: float) -> dict[str, object]:
    """Build a minimal session payload for the given user."""
    return {
        "user": {
            "id": user_id,
            "email": email,
            "name": email.split("@", maxsplit=1)[0],
        },
        "account": {"id": "acct-" + user_id, "planType": "free"},
        "accessToken": _mk_jwt(exp),
        "expires": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(exp + 864000)),
    }


def _mk_block(user_id: str, email: str, exp: float, cookie_value: str = "st-1") -> str:
    """Render an accounts.txt block for the given user."""
    body = _render_block_body(
        AccountSession(
            {
                "session": _mk_session(user_id, email, exp),
                "cookies": {
                    "oai-did": "did-" + user_id,
                    "__Secure-next-auth.session-token.0": cookie_value,
                },
                "raw_cookie_lines": [
                    f".chatgpt.com\tTRUE\t/\tTRUE\t0\toai-did\tdid-{user_id}",
                    (
                        ".chatgpt.com\tTRUE\t/\tTRUE\t2147483647\t"
                        f"__Secure-next-auth.session-token.0\t{cookie_value}"
                    ),
                ],
                "identity": user_id,
            },
        ),
    )
    return f"account {user_id}:\n{body}"


class KeepaliveTestCase(unittest.TestCase):
    """Share an isolated accounts.txt file for keepalive tests."""

    @override
    def setUp(self) -> None:
        """Prepare an isolated accounts file for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.path = Path(self.temp_dir) / "accounts.txt"
        self._orig_accounts_file = config.ACCOUNTS_FILE
        config.ACCOUNTS_FILE = self.path
        self.now = time.time()

    @override
    def tearDown(self) -> None:
        """Restore config and remove the isolated accounts file."""
        config.ACCOUNTS_FILE = self._orig_accounts_file
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write(self, *blocks: str) -> None:
        self.path.write_text("\n".join(blocks))

    def _pool(self) -> AccountPool:
        pool = AccountPool()
        pool.load()
        return pool


class TestCookieAbsorption(KeepaliveTestCase):
    """Verify rotated cookies are absorbed into account state."""

    def test_absorb_updates_state_and_raw_lines(self) -> None:
        """Verify absorption updates cookies and raw lines."""
        acct = AccountSession(
            {
                "session": _mk_session("u1", "a@x.com", self.now + 3600),
                "cookies": {"oai-did": "d1", "tok": "old"},
                "raw_cookie_lines": [".chatgpt.com\tTRUE\t/\tTRUE\t0\ttok\told"],
                "identity": "u1",
            },
        )

        class R:
            """Stub refresh response carrying Set-Cookie headers."""

            class Headers:
                """Stub headers exposing canned Set-Cookie lines."""

                @staticmethod
                def get_list(_name: str) -> list[str]:
                    """Return canned Set-Cookie header values."""
                    return [
                        "tok=new; Path=/; Domain=.chatgpt.com; Secure; HttpOnly",
                    ]

            headers = Headers()

        acct.absorb_cookies(R())
        assert acct.cookies["tok"] == "new"
        assert acct.raw_cookie_lines == [
            ".chatgpt.com\tTRUE\t/\tTRUE\t0\ttok\tnew",
            ".chatgpt.com\tTRUE\t/\tTRUE\t0\toai-did\td1",
        ]

    def test_absorb_ignores_unchanged_cookies(self) -> None:
        """Verify unchanged cookies leave state untouched."""
        raw = [".chatgpt.com\tTRUE\t/\tTRUE\t0\ttok\told"]
        acct = AccountSession(
            {
                "session": _mk_session("u1", "a@x.com", self.now + 3600),
                "cookies": {"tok": "old"},
                "raw_cookie_lines": list(raw),
                "identity": "u1",
            },
        )

        class R:
            """Stub refresh response carrying Set-Cookie headers."""

            class Headers:
                """Stub headers exposing canned Set-Cookie lines."""

                @staticmethod
                def get_list(_name: str) -> list[str]:
                    """Return canned Set-Cookie header values."""
                    return ["tok=old; Path=/"]

            headers = Headers()

        acct.absorb_cookies(R())
        assert not acct.state_dirty
        assert acct.raw_cookie_lines == raw


class TestRefreshStrikes(KeepaliveTestCase):
    """Verify refresh strike counting and dead marking."""

    def test_strikes_mark_dead_then_success_revives(self) -> None:
        """Verify strikes mark dead and success clears them."""
        acct = AccountSession(
            {
                "session": _mk_session("u1", "a@x.com", self.now + 3600),
                "cookies": {},
                "raw_cookie_lines": [],
                "identity": "u1",
            },
        )
        for i in range(config.KEEPALIVE_MAX_STRIKES):
            assert not acct.record_refresh_failure("no accessToken")
            if i < config.KEEPALIVE_MAX_STRIKES - 1:
                assert not acct.dead
        assert acct.dead
        assert acct.revive_after > self.now

        # A stale strike (> two sweep intervals old) is forgiven first.
        acct.last_strike_at = self.now - 2 * config.KEEPALIVE_SECONDS - 1
        acct.record_refresh_failure("no accessToken")
        assert acct.refresh_strikes == 1

        # Refresh outcome tracking: failure flips refresh_ok off, a simulated
        # success flips it on (what the 5s short-circuit reports).
        assert not acct.refresh_ok
        acct.access_token = _mk_jwt(self.now + 7200)
        acct.jwt_exp = self.now + 7200
        acct.refresh_strikes = 0
        acct.dead = False
        acct.refresh_ok = True
        assert acct.refresh_ok

    def test_transient_failures_do_not_strike(self) -> None:
        """Verify transient HTTP failures never count strikes."""
        acct = AccountSession(
            {
                "session": _mk_session("u1", "a@x.com", self.now + 3600),
                "cookies": {},
                "raw_cookie_lines": [],
                "identity": "u1",
            },
        )
        acct.refresh_strikes = SEEDED_STRIKE_COUNT

        class FakeResponse:
            """Stub 503 session response that must never parse JSON."""

            status_code = 503

            def json(self) -> NoReturn:
                """Reject JSON parsing on non-200 responses."""
                raise _UnexpectedBodyError

        class FakeSession:
            """Stub HTTP session returning a canned 503 response."""

            async def get(self, *_args: object, **_kwargs: object) -> FakeResponse:
                """Return a canned transient-failure response."""
                return FakeResponse()

        async def run() -> bool:
            """Run one refresh against the stub session."""

            async def fake_http() -> FakeSession:
                """Return the stub session."""
                return FakeSession()

            with patch.object(acct, "http", new=fake_http):
                return await acct.refresh_access_token()

        assert not asyncio.run(run())
        assert acct.refresh_strikes == SEEDED_STRIKE_COUNT
        assert not acct.dead

    def test_absorb_skips_deletion_cookies(self) -> None:
        """Verify empty deletion cookies never wipe stored values."""
        acct = AccountSession(
            {
                "session": _mk_session("u1", "a@x.com", self.now + 3600),
                "cookies": {"tok": "good"},
                "raw_cookie_lines": [".chatgpt.com\tTRUE\t/\tTRUE\t0\ttok\tgood"],
                "identity": "u1",
            },
        )

        class R:
            """Stub refresh response carrying Set-Cookie headers."""

            class Headers:
                """Stub headers exposing canned Set-Cookie lines."""

                @staticmethod
                def get_list(_name: str) -> list[str]:
                    """Return canned Set-Cookie header values."""
                    return ["tok=; Path=/; Expires=Thu, 01 Jan 1970 00:00:00 GMT"]

            headers = Headers()

        acct.absorb_cookies(R())
        assert acct.cookies["tok"] == "good"
        assert not acct.state_dirty

    def test_throttled_refresh_reports_last_outcome(self) -> None:
        """Verify throttled refreshes report the cached outcome.

        Within the 5s window the cached outcome is returned, and a failed
        attempt must not masquerade as success (stream 401-retry relies on it).
        """
        acct = AccountSession(
            {
                "session": _mk_session("u1", "a@x.com", self.now + 3600),
                "cookies": {},
                "raw_cookie_lines": [],
                "identity": "u1",
            },
        )
        acct.last_refresh_attempt = self.now  # just attempted
        assert not asyncio.run(acct.refresh_access_token())
        acct.refresh_ok = True
        assert asyncio.run(acct.refresh_access_token())

    def test_ensure_token_attempts_real_refresh(self) -> None:
        """Verify ensure_token performs a real refresh when due.

        Regression: ensure_token must not pre-stamp the attempt time (the
        5s throttle would swallow every request-path refresh).
        """
        acct = AccountSession(
            {
                "session": _mk_session(
                    "u1",
                    "a@x.com",
                    self.now + 300,
                ),  # inside 600s window
                "cookies": {},
                "raw_cookie_lines": [],
                "identity": "u1",
            },
        )

        class FakeResponse:
            """Stub 503 session response for refresh probing."""

            status_code = 503

        class FakeSession:
            """Stub HTTP session returning a canned 503 response."""

            async def get(self, *_args: object, **_kwargs: object) -> FakeResponse:
                """Return a canned transient-failure response."""
                return FakeResponse()

        async def fake_http() -> FakeSession:
            """Return the stub session."""
            return FakeSession()

        with patch.object(acct, "http", new=fake_http):
            asyncio.run(acct.ensure_token())
        assert acct.last_refresh_attempt > self.now  # a real attempt ran

    def test_ensure_token_waits_between_attempts(self) -> None:
        """Verify ensure_token spaces refresh attempts by a minute."""
        acct = AccountSession(
            {
                "session": _mk_session("u1", "a@x.com", self.now + 300),
                "cookies": {},
                "raw_cookie_lines": [],
                "identity": "u1",
            },
        )
        acct.last_refresh_attempt = self.now - 10  # recent, but > 5s window

        calls: list[int] = []

        async def fake_refresh() -> bool:
            """Record the refresh call and report success."""
            calls.append(1)
            return True

        with patch.object(acct, "refresh_access_token", new=fake_refresh):
            asyncio.run(acct.ensure_token())
        assert calls == []  # 60s ensure_token spacing still applies


class TestPersistState(KeepaliveTestCase):
    """Verify dirty account state persists back to accounts.txt."""

    def test_persist_rewrites_only_dirty_blocks(self) -> None:
        """Verify persist rewrites only dirty blocks."""
        self._write(
            "account 1:\n" + _mk_block("u1", "a@x.com", self.now + 3600),
            "account 2:\n"
            + _mk_block("u2", "b@x.com", self.now + 3600, cookie_value="st-2"),
        )
        pool = self._pool()
        assert len(pool) == EXPECTED_BLOCK_TOTAL
        before = self.path.read_text()

        acct = pool.get_account("u1")
        assert acct is not None
        acct.session_json = _mk_session("u1", "a@x.com", self.now + 7200)
        acct.cookies["__Secure-next-auth.session-token.0"] = "st-rotated"
        acct.rebuild_raw_cookie_lines()
        acct.state_dirty = True
        pool.persist_state()

        assert not acct.state_dirty
        text = self.path.read_text()
        assert text != before
        parsed = {p["identity"]: p for p in parse_accounts_text(text)}
        assert len(parsed) == EXPECTED_BLOCK_TOTAL
        # u1 got the rotated cookie + fresh session JSON
        assert (
            parsed["u1"]["cookies"]["__Secure-next-auth.session-token.0"]
            == "st-rotated"
        )
        assert parsed["u1"]["session"]["accessToken"] == acct.access_token
        # u2 untouched
        assert parsed["u2"]["cookies"]["__Secure-next-auth.session-token.0"] == "st-2"
        assert "account 2:" in text
        # write is atomic-replace style: lock file remains, no tmp leftovers
        assert self.path.with_name(self.path.name + ".lock").exists()
        assert not self.path.with_name(self.path.name + ".tmp").exists()

    def test_persist_is_noop_when_clean(self) -> None:
        """Verify persist leaves the file untouched when clean."""
        self._write(_mk_block("u1", "a@x.com", self.now + 3600))
        pool = self._pool()
        before = self.path.read_text()
        pool.persist_state()
        assert self.path.read_text() == before

    def test_persist_round_trip_reparse_hot_swaps_nothing(self) -> None:
        """Verify a persisted block reparses to identical live state."""
        self._write(_mk_block("u1", "a@x.com", self.now + 3600))
        pool = self._pool()
        acct = pool.get_account("u1")
        assert acct is not None
        fresh_exp = self.now + 99999
        acct.session_json = _mk_session("u1", "a@x.com", fresh_exp)
        acct.access_token = _mk_jwt(fresh_exp)
        acct.jwt_exp = fresh_exp
        acct.cookies["oai-did"] = "did-new"
        acct.rebuild_raw_cookie_lines()
        acct.state_dirty = True
        pool.persist_state()

        pool2 = self._pool()
        again = pool2.get_account("u1")
        assert again is not None
        assert again.access_token == acct.access_token
        assert again.cookies == acct.cookies
        assert again.raw_cookie_lines == acct.raw_cookie_lines

    def test_persist_never_overwrites_fresher_disk_block(self) -> None:
        """Verify a raced fresher disk block wins over stale memory.

        A user edit that raced a dirty refresh must win on disk and be
        adopted into memory, not clobbered by the stale in-memory state.
        """
        self._write(_mk_block("u1", "a@x.com", self.now + 3600))
        pool = self._pool()
        acct = pool.get_account("u1")
        assert acct is not None
        stale_token = acct.access_token
        acct.state_dirty = True  # a refresh marked it dirty moments ago

        pasted_exp = self.now + 3600 * 3
        self._write(_mk_block("u1", "a@x.com", pasted_exp, cookie_value="st-pasted"))
        pool.persist_state()

        text = self.path.read_text()
        assert "st-pasted" in text  # paste survived on disk
        assert stale_token not in text  # stale memory not written back
        # persist() detected the fresher paste and adopted it via load()
        adopted = pool.get_account("u1")
        assert adopted is not None
        assert adopted.jwt_exp == pasted_exp
        assert adopted.cookies["__Secure-next-auth.session-token.0"] == "st-pasted"
        assert not adopted.state_dirty


class TestHotSwap(KeepaliveTestCase):
    """Verify hot-swap adoption and keepalive tick behavior."""

    def test_load_adopts_fresher_block(self) -> None:
        """Verify load adopts a fresher pasted block."""
        self._write(_mk_block("u1", "a@x.com", self.now + 3600))
        pool = self._pool()
        seeded = pool.get_account("u1")
        assert seeded is not None
        old_token = seeded.access_token

        self._write(
            _mk_block("u1", "a@x.com", self.now + 3600 * 3, cookie_value="st-fresh"),
        )
        pool.load()
        acct = pool.get_account("u1")
        assert acct is not None
        assert acct.access_token != old_token
        assert acct.cookies["__Secure-next-auth.session-token.0"] == "st-fresh"
        assert not acct.dead
        assert acct.refresh_strikes == 0

    def test_load_ignores_stale_block(self) -> None:
        """Verify load ignores a staler pasted block."""
        self._write(_mk_block("u1", "a@x.com", self.now + 3600 * 3))
        pool = self._pool()
        seeded = pool.get_account("u1")
        assert seeded is not None
        old_token = seeded.access_token
        old_cookies = dict(seeded.cookies)

        self._write(
            _mk_block("u1", "a@x.com", self.now + 3600, cookie_value="st-stale"),
        )
        pool.load()
        acct = pool.get_account("u1")
        assert acct is not None
        assert acct.access_token == old_token
        assert acct.cookies == old_cookies

    def test_load_revives_dead_account_from_fresher_paste(self) -> None:
        """Verify a fresher paste revives a dead account."""
        self._write(_mk_block("u1", "a@x.com", self.now + 3600))
        pool = self._pool()
        doomed = pool.get_account("u1")
        assert doomed is not None
        doomed.dead = True

        self._write(
            _mk_block("u1", "a@x.com", self.now + 3600 * 3, cookie_value="st-new"),
        )
        pool.load()

    def test_load_keeps_dead_for_stale_paste(self) -> None:
        """Verify a stale paste keeps a dead account dead."""
        self._write(_mk_block("u1", "a@x.com", self.now + 3600))
        pool = self._pool()
        acct = pool.get_account("u1")
        assert acct is not None
        acct.dead = True

        # Same-freshness paste: nothing newer on disk, stays dead.
        self._write(_mk_block("u1", "a@x.com", self.now + 3600))
        pool.load()
        kept = pool.get_account("u1")
        assert kept is not None
        assert kept.dead

    def test_tick_refreshes_stale_token_and_persists(self) -> None:
        """Verify the tick refreshes stale tokens and persists."""
        self._write(_mk_block("u1", "a@x.com", self.now - 60))  # token already old
        pool = self._pool()
        acct = pool.get_account("u1")
        assert acct is not None
        fresh = _mk_session("u1", "a@x.com", self.now + 7200)
        fresh["accessToken"] = _mk_jwt(self.now + 7200)
        rotated = fresh["accessToken"]
        assert isinstance(rotated, str)

        calls: list[int] = []

        async def fake_refresh() -> bool:
            """Apply a fresh session and report success."""
            calls.append(1)
            acct.session_json = fresh
            acct.access_token = rotated
            acct.jwt_exp = self.now + 7200
            acct.state_dirty = True
            return True

        with patch.object(acct, "refresh_access_token", new=fake_refresh):
            asyncio.run(pool.keepalive_tick())

        assert len(calls) == 1
        assert not acct.state_dirty  # consumed by persist
        text = self.path.read_text()
        assert rotated in text

    def test_tick_skips_healthy_token(self) -> None:
        """Verify the tick skips healthy tokens."""
        self._write(_mk_block("u1", "a@x.com", self.now + 7 * 86400))
        pool = self._pool()
        calls: list[int] = []

        async def fake_refresh() -> bool:
            """Record the refresh call and report success."""
            calls.append(1)
            return True

        idle = pool.get_account("u1")
        assert idle is not None
        with patch.object(idle, "refresh_access_token", new=fake_refresh):
            asyncio.run(pool.keepalive_tick())
        assert calls == []

    def test_tick_probes_dead_account_and_revives(self) -> None:
        """Verify the tick probes dead accounts past backoff."""
        self._write(_mk_block("u1", "a@x.com", self.now - 60))
        pool = self._pool()
        acct = pool.get_account("u1")
        assert acct is not None
        acct.dead = True
        acct.revive_after = self.now - 1

        async def fake_refresh() -> bool:
            """Clear the dead flag and report success."""
            acct.dead = False
            return True

        with patch.object(acct, "refresh_access_token", new=fake_refresh):
            asyncio.run(pool.keepalive_tick())
        assert not acct.dead
        assert acct.revive_after >= time.time() + config.KEEPALIVE_REVIVE_SECONDS - 5

    def test_tick_backs_off_recently_probed_dead_account(self) -> None:
        """Verify the tick backs off recently probed dead accounts."""
        self._write(_mk_block("u1", "a@x.com", self.now - 60))
        pool = self._pool()
        acct = pool.get_account("u1")
        assert acct is not None
        acct.dead = True
        acct.revive_after = self.now + 999

        calls: list[int] = []

        async def fake_refresh() -> bool:
            """Record the refresh call and report success."""
            calls.append(1)
            return True

        with patch.object(acct, "refresh_access_token", new=fake_refresh):
            asyncio.run(pool.keepalive_tick())
        assert calls == []
        assert acct.dead


class TestRenderRoundTrip(KeepaliveTestCase):
    """Verify block rendering round-trips through the parser."""

    def test_render_block_body_reparses(self) -> None:
        """Verify a rendered block reparses identically."""
        acct = AccountSession(
            {
                "session": _mk_session("u1", "a@x.com", self.now + 3600),
                "cookies": {"oai-did": "d1", "tok": "v1"},
                "raw_cookie_lines": [
                    ".chatgpt.com\tTRUE\t/\tTRUE\t0\toai-did\td1",
                    ".chatgpt.com\tTRUE\t/\tTRUE\t0\ttok\tv1",
                ],
                "identity": "u1",
            },
        )
        parsed = parse_accounts_text(_render_block_body(acct))
        assert len(parsed) == 1
        p = parsed[0]
        assert p["identity"] == "u1"
        assert p["cookies"] == {"oai-did": "d1", "tok": "v1"}
        assert p["session"]["accessToken"] == acct.access_token
        assert p["raw_cookie_lines"] == acct.raw_cookie_lines

    def test_httponly_marker_survives_round_trip(self) -> None:
        """Verify the HttpOnly marker survives a round trip."""
        raw = (
            "#HttpOnly_.chatgpt.com\tTRUE\t/\tTRUE\t2147483647\t"
            "__Secure-next-auth.session-token.0\tSECRET"
        )
        text = "account 1:\n" + _render_block_body(
            AccountSession(
                {
                    "session": _mk_session("u1", "a@x.com", self.now + 3600),
                    "cookies": {"__Secure-next-auth.session-token.0": "SECRET"},
                    "raw_cookie_lines": [raw],
                    "identity": "u1",
                },
            ),
        )
        parsed = parse_accounts_text(text)
        assert parsed[0]["cookies"]["__Secure-next-auth.session-token.0"] == "SECRET"
        assert parsed[0]["raw_cookie_lines"] == [raw]  # marker preserved


if __name__ == "__main__":
    unittest.main()
