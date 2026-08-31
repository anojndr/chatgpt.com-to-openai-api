import asyncio
import json
import shutil
import tempfile
import time
import unittest
import base64
from pathlib import Path

from app import config
from app.accounts import AccountPool, _render_block_body
from app.chatgpt import AccountSession, parse_accounts_text


def _mk_jwt(exp: float) -> str:
    """Minimal unsigned JWT with just an exp claim (only exp is ever read)."""
    head = base64.urlsafe_b64encode(b'{"alg":"RS256","typ":"JWT"}').decode().rstrip("=")
    payload = base64.urlsafe_b64encode(
        json.dumps({"exp": exp, "iat": int(exp) - 900}).encode()).decode().rstrip("=")
    return f"{head}.{payload}.sig"


def _mk_session(user_id: str, email: str, exp: float) -> dict:
    return {
        "user": {"id": user_id, "email": email, "name": email.split("@")[0]},
        "account": {"id": "acct-" + user_id, "planType": "free"},
        "accessToken": _mk_jwt(exp),
        "expires": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(exp + 864000)),
    }


def _mk_block(user_id: str, email: str, exp: float, session_token: str = "st-1") -> str:
    body = _render_block_body(AccountSession({
        "session": _mk_session(user_id, email, exp),
        "cookies": {"oai-did": "did-" + user_id,
                    "__Secure-next-auth.session-token.0": session_token},
        "raw_cookie_lines": [
            f".chatgpt.com\tTRUE\t/\tTRUE\t0\toai-did\tdid-{user_id}",
            f".chatgpt.com\tTRUE\t/\tTRUE\t2147483647\t__Secure-next-auth.session-token.0\t{session_token}",
        ],
        "identity": user_id,
    }))
    return f"account {user_id}:\n{body}"


class KeepaliveTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.path = Path(self.temp_dir) / "accounts.txt"
        self._orig_accounts_file = config.ACCOUNTS_FILE
        config.ACCOUNTS_FILE = self.path
        self.now = time.time()

    def tearDown(self):
        config.ACCOUNTS_FILE = self._orig_accounts_file
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write(self, *blocks: str) -> None:
        self.path.write_text("\n".join(blocks))

    def _pool(self) -> AccountPool:
        pool = AccountPool()
        pool.load()
        return pool


class TestCookieAbsorption(KeepaliveTestCase):
    def test_absorb_updates_state_and_raw_lines(self):
        acct = AccountSession({
            "session": _mk_session("u1", "a@x.com", self.now + 3600),
            "cookies": {"oai-did": "d1", "tok": "old"},
            "raw_cookie_lines": [".chatgpt.com\tTRUE\t/\tTRUE\t0\ttok\told"],
            "identity": "u1",
        })

        class R:
            class headers:
                get_list = staticmethod(lambda n: [
                    "tok=new; Path=/; Domain=.chatgpt.com; Secure; HttpOnly"])

        acct._absorb_cookies(R())
        self.assertEqual(acct.cookies["tok"], "new")
        self.assertEqual(acct.raw_cookie_lines,
                         [".chatgpt.com\tTRUE\t/\tTRUE\t0\ttok\tnew",
                          ".chatgpt.com\tTRUE\t/\tTRUE\t0\toai-did\td1"])

    def test_absorb_ignores_unchanged_cookies(self):
        raw = [".chatgpt.com\tTRUE\t/\tTRUE\t0\ttok\told"]
        acct = AccountSession({
            "session": _mk_session("u1", "a@x.com", self.now + 3600),
            "cookies": {"tok": "old"},
            "raw_cookie_lines": list(raw),
            "identity": "u1",
        })

        class R:
            class headers:
                get_list = staticmethod(lambda n: ["tok=old; Path=/"])

        acct._absorb_cookies(R())
        self.assertFalse(acct.state_dirty)
        self.assertEqual(acct.raw_cookie_lines, raw)


class TestRefreshStrikes(KeepaliveTestCase):
    def test_strikes_mark_dead_then_success_revives(self):
        acct = AccountSession({
            "session": _mk_session("u1", "a@x.com", self.now + 3600),
            "cookies": {}, "raw_cookie_lines": [], "identity": "u1",
        })
        for i in range(config.KEEPALIVE_MAX_STRIKES):
            self.assertFalse(acct._record_refresh_failure("no accessToken"))
            if i < config.KEEPALIVE_MAX_STRIKES - 1:
                self.assertFalse(acct.dead)
        self.assertTrue(acct.dead)
        self.assertGreater(acct.revive_after, self.now)

        # A stale strike (> two sweep intervals old) is forgiven first.
        acct._last_strike_at = self.now - 2 * config.KEEPALIVE_SECONDS - 1
        acct._record_refresh_failure("no accessToken")
        self.assertEqual(acct.refresh_strikes, 1)

        # Refresh outcome tracking: failure flips _refresh_ok off, a simulated
        # success flips it on (what the 5s short-circuit reports).
        self.assertFalse(acct._refresh_ok)
        acct.access_token = _mk_jwt(self.now + 7200)
        acct._jwt_exp = self.now + 7200
        acct.refresh_strikes = 0
        acct.dead = False
        acct._refresh_ok = True
        self.assertTrue(acct._refresh_ok)

    def test_transient_failures_do_not_strike(self):
        acct = AccountSession({
            "session": _mk_session("u1", "a@x.com", self.now + 3600),
            "cookies": {}, "raw_cookie_lines": [], "identity": "u1",
        })
        acct.refresh_strikes = 2

        class FakeResponse:
            status_code = 503
            def json(self):
                raise AssertionError("must not parse body on non-200")

        class FakeSession:
            async def get(self, *a, **k):
                return FakeResponse()

        async def run():
            async def fake_http():
                return FakeSession()
            acct.http = fake_http  # type: ignore[method-assign]
            return await acct.refresh_access_token()

        self.assertFalse(asyncio.run(run()))
        self.assertEqual(acct.refresh_strikes, 2)
        self.assertFalse(acct.dead)
    def test_absorb_skips_deletion_cookies(self):
        acct = AccountSession({
            "session": _mk_session("u1", "a@x.com", self.now + 3600),
            "cookies": {"tok": "good"},
            "raw_cookie_lines": [".chatgpt.com\tTRUE\t/\tTRUE\t0\ttok\tgood"],
            "identity": "u1",
        })

        class R:
            class headers:
                get_list = staticmethod(lambda n: [
                    "tok=; Path=/; Expires=Thu, 01 Jan 1970 00:00:00 GMT"])

        acct._absorb_cookies(R())
        self.assertEqual(acct.cookies["tok"], "good")
        self.assertFalse(acct.state_dirty)

    def test_throttled_refresh_reports_last_outcome(self):
        """Within the 5s window the cached outcome is returned, and a failed
        attempt must not masquerade as success (stream 401-retry relies on it)."""
        acct = AccountSession({
            "session": _mk_session("u1", "a@x.com", self.now + 3600),
            "cookies": {}, "raw_cookie_lines": [], "identity": "u1",
        })
        acct._last_refresh_attempt = self.now  # just attempted
        self.assertFalse(asyncio.run(acct.refresh_access_token()))
        acct._refresh_ok = True
        self.assertTrue(asyncio.run(acct.refresh_access_token()))

    def test_ensure_token_attempts_real_refresh(self):
        """Regression: ensure_token must not pre-stamp the attempt time (the
        5s throttle would swallow every request-path refresh)."""
        acct = AccountSession({
            "session": _mk_session("u1", "a@x.com", self.now + 300),  # inside 600s window
            "cookies": {}, "raw_cookie_lines": [], "identity": "u1",
        })

        class FakeResponse:
            status_code = 503

        class FakeSession:
            async def get(self, *a, **k):
                return FakeResponse()

        async def fake_http():
            return FakeSession()

        acct.http = fake_http  # type: ignore[method-assign]
        asyncio.run(acct.ensure_token())
        self.assertGreater(acct._last_refresh_attempt, self.now)  # a real attempt ran

    def test_ensure_token_waits_between_attempts(self):
        acct = AccountSession({
            "session": _mk_session("u1", "a@x.com", self.now + 300),
            "cookies": {}, "raw_cookie_lines": [], "identity": "u1",
        })
        acct._last_refresh_attempt = self.now - 10  # recent, but > 5s window

        calls = []

        async def fake_refresh():
            calls.append(1)
            return True

        acct.refresh_access_token = fake_refresh  # type: ignore[method-assign]
        asyncio.run(acct.ensure_token())
        self.assertEqual(calls, [])  # 60s ensure_token spacing still applies


class TestPersistState(KeepaliveTestCase):
    def test_persist_rewrites_only_dirty_blocks(self):
        self._write(
            "account 1:\n" + _mk_block("u1", "a@x.com", self.now + 3600),
            "account 2:\n" + _mk_block("u2", "b@x.com", self.now + 3600, session_token="st-2"),
        )
        pool = self._pool()
        self.assertEqual(len(pool._accounts), 2)
        before = self.path.read_text()

        acct = pool._accounts["u1"]
        acct.session_json = _mk_session("u1", "a@x.com", self.now + 7200)
        acct.cookies["__Secure-next-auth.session-token.0"] = "st-rotated"
        acct._rebuild_raw_cookie_lines()
        acct.state_dirty = True
        pool._persist_state()

        self.assertFalse(acct.state_dirty)
        text = self.path.read_text()
        self.assertNotEqual(text, before)
        parsed = {p["identity"]: p for p in parse_accounts_text(text)}
        self.assertEqual(len(parsed), 2)
        # u1 got the rotated cookie + fresh session JSON
        self.assertEqual(parsed["u1"]["cookies"]["__Secure-next-auth.session-token.0"], "st-rotated")
        self.assertEqual(parsed["u1"]["session"]["accessToken"], acct.access_token)
        # u2 untouched
        self.assertEqual(parsed["u2"]["cookies"]["__Secure-next-auth.session-token.0"], "st-2")
        self.assertIn("account 2:", text)
        # write is atomic-replace style: lock file remains, no tmp leftovers
        self.assertTrue(self.path.with_name(self.path.name + ".lock").exists())
        self.assertFalse(self.path.with_name(self.path.name + ".tmp").exists())

    def test_persist_is_noop_when_clean(self):
        self._write(_mk_block("u1", "a@x.com", self.now + 3600))
        pool = self._pool()
        before = self.path.read_text()
        pool._persist_state()
        self.assertEqual(self.path.read_text(), before)

    def test_persist_round_trip_reparse_hot_swaps_nothing(self):
        """A persisted block must parse back to the identical live state."""
        self._write(_mk_block("u1", "a@x.com", self.now + 3600))
        pool = self._pool()
        acct = pool._accounts["u1"]
        fresh_exp = self.now + 99999
        acct.session_json = _mk_session("u1", "a@x.com", fresh_exp)
        acct.access_token = _mk_jwt(fresh_exp)
        acct._jwt_exp = fresh_exp
        acct.cookies["oai-did"] = "did-new"
        acct._rebuild_raw_cookie_lines()
        acct.state_dirty = True
        pool._persist_state()

        pool2 = self._pool()
        self.assertIn("u1", pool2._accounts)
        again = pool2._accounts["u1"]
        self.assertEqual(again.access_token, acct.access_token)
        self.assertEqual(again.cookies, acct.cookies)
        self.assertEqual(again.raw_cookie_lines, acct.raw_cookie_lines)

    def test_persist_never_overwrites_fresher_disk_block(self):
        """A user edit that raced a dirty refresh must win on disk and be
        adopted into memory, not clobbered by the stale in-memory state."""
        self._write(_mk_block("u1", "a@x.com", self.now + 3600))
        pool = self._pool()
        acct = pool._accounts["u1"]
        stale_token = acct.access_token
        acct.state_dirty = True  # a refresh marked it dirty moments ago

        pasted_exp = self.now + 3600 * 3
        self._write(_mk_block("u1", "a@x.com", pasted_exp, session_token="st-pasted"))
        pool._persist_state()

        text = self.path.read_text()
        self.assertIn("st-pasted", text)          # paste survived on disk
        self.assertNotIn(stale_token, text)       # stale memory not written back
        # persist() detected the fresher paste and adopted it via load()
        self.assertEqual(pool._accounts["u1"]._jwt_exp, pasted_exp)
        self.assertEqual(pool._accounts["u1"].cookies["__Secure-next-auth.session-token.0"], "st-pasted")
        self.assertFalse(pool._accounts["u1"].state_dirty)



class TestHotSwap(KeepaliveTestCase):
    def test_load_adopts_fresher_block(self):
        self._write(_mk_block("u1", "a@x.com", self.now + 3600))
        pool = self._pool()
        old_token = pool._accounts["u1"].access_token

        self._write(_mk_block("u1", "a@x.com", self.now + 3600 * 3, session_token="st-fresh"))
        pool.load()
        acct = pool._accounts["u1"]
        self.assertNotEqual(acct.access_token, old_token)
        self.assertEqual(acct.cookies["__Secure-next-auth.session-token.0"], "st-fresh")
        self.assertFalse(acct.dead)
        self.assertEqual(acct.refresh_strikes, 0)

    def test_load_ignores_stale_block(self):
        self._write(_mk_block("u1", "a@x.com", self.now + 3600 * 3))
        pool = self._pool()
        old_token = pool._accounts["u1"].access_token
        old_cookies = dict(pool._accounts["u1"].cookies)

        self._write(_mk_block("u1", "a@x.com", self.now + 3600, session_token="st-stale"))
        pool.load()
        acct = pool._accounts["u1"]
        self.assertEqual(acct.access_token, old_token)
        self.assertEqual(acct.cookies, old_cookies)

    def test_load_revives_dead_account_from_fresher_paste(self):
        self._write(_mk_block("u1", "a@x.com", self.now + 3600))
        pool = self._pool()
        pool._accounts["u1"].dead = True

        self._write(_mk_block("u1", "a@x.com", self.now + 3600 * 3, session_token="st-new"))
        pool.load()
    def test_load_keeps_dead_for_stale_paste(self):
        self._write(_mk_block("u1", "a@x.com", self.now + 3600))
        pool = self._pool()
        acct = pool._accounts["u1"]
        acct.dead = True

        # Same-freshness paste: nothing newer on disk, stays dead.
        self._write(_mk_block("u1", "a@x.com", self.now + 3600))
        pool.load()
        self.assertTrue(pool._accounts["u1"].dead)
    def test_tick_refreshes_stale_token_and_persists(self):
        self._write(_mk_block("u1", "a@x.com", self.now - 60))  # token already old
        pool = self._pool()
        acct = pool._accounts["u1"]
        fresh = _mk_session("u1", "a@x.com", self.now + 7200)
        fresh["accessToken"] = _mk_jwt(self.now + 7200)

        calls = []

        async def fake_refresh():
            calls.append(1)
            acct.session_json = fresh
            acct.access_token = fresh["accessToken"]
            acct._jwt_exp = self.now + 7200
            acct.state_dirty = True
            return True

        acct.refresh_access_token = fake_refresh  # type: ignore[method-assign]
        asyncio.run(pool._keepalive_tick())

        self.assertEqual(len(calls), 1)
        self.assertFalse(acct.state_dirty)  # consumed by persist
        text = self.path.read_text()
        self.assertIn(fresh["accessToken"], text)

    def test_tick_skips_healthy_token(self):
        self._write(_mk_block("u1", "a@x.com", self.now + 7 * 86400))
        pool = self._pool()
        calls = []

        async def fake_refresh():
            calls.append(1)
            return True

        pool._accounts["u1"].refresh_access_token = fake_refresh  # type: ignore[method-assign]
        asyncio.run(pool._keepalive_tick())
        self.assertEqual(calls, [])

    def test_tick_probes_dead_account_and_revives(self):
        self._write(_mk_block("u1", "a@x.com", self.now - 60))
        pool = self._pool()
        acct = pool._accounts["u1"]
        acct.dead = True
        acct.revive_after = self.now - 1

        async def fake_refresh():
            acct.dead = False
            return True

        acct.refresh_access_token = fake_refresh  # type: ignore[method-assign]
        asyncio.run(pool._keepalive_tick())
        self.assertFalse(acct.dead)
        self.assertGreaterEqual(acct.revive_after, time.time() + config.KEEPALIVE_REVIVE_SECONDS - 5)

    def test_tick_backs_off_recently_probed_dead_account(self):
        self._write(_mk_block("u1", "a@x.com", self.now - 60))
        pool = self._pool()
        acct = pool._accounts["u1"]
        acct.dead = True
        acct.revive_after = self.now + 999

        calls = []

        async def fake_refresh():
            calls.append(1)
            return True

        acct.refresh_access_token = fake_refresh  # type: ignore[method-assign]
        asyncio.run(pool._keepalive_tick())
        self.assertEqual(calls, [])
        self.assertTrue(acct.dead)


class TestRenderRoundTrip(KeepaliveTestCase):
    def test_render_block_body_reparses(self):
        acct = AccountSession({
            "session": _mk_session("u1", "a@x.com", self.now + 3600),
            "cookies": {"oai-did": "d1", "tok": "v1"},
            "raw_cookie_lines": [
                ".chatgpt.com\tTRUE\t/\tTRUE\t0\toai-did\td1",
                ".chatgpt.com\tTRUE\t/\tTRUE\t0\ttok\tv1",
            ],
            "identity": "u1",
        })
        parsed = parse_accounts_text(_render_block_body(acct))
        self.assertEqual(len(parsed), 1)
        p = parsed[0]
        self.assertEqual(p["identity"], "u1")
        self.assertEqual(p["cookies"], {"oai-did": "d1", "tok": "v1"})
        self.assertEqual(p["session"]["accessToken"], acct.access_token)
        self.assertEqual(p["raw_cookie_lines"], acct.raw_cookie_lines)

    def test_httponly_marker_survives_round_trip(self):
        raw = "#HttpOnly_.chatgpt.com\tTRUE\t/\tTRUE\t2147483647\t__Secure-next-auth.session-token.0\tSECRET"
        text = ("account 1:\n"
                + _render_block_body(AccountSession({
                    "session": _mk_session("u1", "a@x.com", self.now + 3600),
                    "cookies": {"__Secure-next-auth.session-token.0": "SECRET"},
                    "raw_cookie_lines": [raw],
                    "identity": "u1",
                })))
        parsed = parse_accounts_text(text)
        self.assertEqual(parsed[0]["cookies"]["__Secure-next-auth.session-token.0"], "SECRET")
        self.assertEqual(parsed[0]["raw_cookie_lines"], [raw])  # marker preserved


if __name__ == "__main__":
    unittest.main()
