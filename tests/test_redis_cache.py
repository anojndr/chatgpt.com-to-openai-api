# Copyright 2026 chatgpt-to-openai-api contributors.
"""Redis read-through cache regression tests.

SQLite stays authoritative: Redis only accelerates hot lookups, and every
miss or failure degrades to SQLite without failing the request. These tests
use an in-memory fake client so no live server is required; live round-trip
verification happens against 127.0.0.1:6379 outside the suite.
"""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from typing_extensions import override

from app import config
from app.adapters import HistoryItem
from app.redis_cache import RedisCache, RedisOptions, get_cache, reset_cache
from app.store import (
    ConversationStore,
    ConvRef,
    ResponseRecord,
    TurnSnapshot,
    item_hash,
)

FAKE_PREFIX_HASH = "abc123"
FAKE_RESPONSE_ID = "resp_redis_1"
FAKE_IDENTITY = "user_redis"
FAKE_CONVERSATION = "conv_redis"
FAKE_PARENT = "msg_redis"
FAKE_MODEL = "auto"
FAKE_SLUG_AUTO = "auto"
FAKE_SLUG_NEW = "gpt-5-6"
PREFIX_TURNS = 3
EXPECTED_SINGLE_HIT = 1
FORK_MATCHED_LEN = 3
EXPECTED_HIT_RATIO = 0.8
CONNECTED_CLIENTS = 2
OVERSIZE_BYTES = 64
SMALL_SNAPSHOT_CAP = 8
FULL_CACHE_CAP = 1 << 20
OWNER_TURNS = 1
OWNER_UPDATED = 1.0
FORK_TURNS = 2
FORK_UPDATED = 2.0
META_CREATED = 1.0


class FakeRedisError(OSError):
    """Injected fake Redis transport failure."""


class FakePipeline:
    """Minimal pipeline double recording queued SETEX writes."""

    def __init__(self, client: FakeRedis) -> None:
        """Retain the backing fake client for flushed writes."""
        self._client = client
        self._writes: list[tuple[str, int, str]] = []

    def setex(self, name: str, ttl: int, value: str, /) -> bool:
        """Queue one write.

        Returns:
            True always (queued, not yet flushed).
        """
        self._writes.append((name, ttl, value))
        return True

    def execute(self) -> list[bool]:
        """Flush queued writes into the fake store.

        Returns:
            One True per flushed write.
        """
        for name, _ttl, value in self._writes:
            self._client.store[name] = value
        return [True] * len(self._writes)


class FakeRedis:
    """In-memory Redis double with failure injection."""

    def __init__(self, store: dict[str, str] | None = None) -> None:
        """Create the double over the given backing dict."""
        self.store: dict[str, str] = store if store is not None else {}
        self.fail_next = False

    def _maybe_fail(self, op: str) -> None:
        if self.fail_next:
            self.fail_next = False
            message = "fake " + op + " failure"
            raise FakeRedisError(message)

    def get(self, name: str) -> str | None:
        """Fetch one key from the fake store.

        Returns:
            The stored value, or None when missing.
        """
        self._maybe_fail("get")
        return self.store.get(name)

    def setex(self, name: str, ttl: int, value: str, /) -> bool:
        """Set one key in the fake store (ttl ignored).

        Returns:
            True always on success.
        """
        self._maybe_fail("setex")
        _ = ttl
        self.store[name] = value
        return True

    def mget(self, keys: list[str]) -> list[str | None]:
        """Fetch keys in input order from the fake store.

        Returns:
            The values aligned with the input keys.
        """
        self._maybe_fail("mget")
        return [self.store.get(key) for key in keys]

    def ping(self) -> bool:
        """Probe the fake (honours failure injection).

        Returns:
            True always on success.
        """
        self._maybe_fail("ping")
        return True

    def pipeline(self, *, transaction: bool = True) -> FakePipeline:
        """Open a fake write pipeline.

        Returns:
            The fake pipeline handle.
        """
        _ = transaction
        return FakePipeline(self)

    @staticmethod
    def info(section: str | None = None) -> dict[str, Any]:
        """Return a fixed observability payload.

        Returns:
            The fixed fake info mapping.
        """
        _ = section
        return {
            "used_memory_human": "1.00M",
            "connected_clients": CONNECTED_CLIENTS,
            "blocked_clients": 0,
            "instantaneous_ops_per_sec": 10,
            "rejected_connections": 0,
            "keyspace_hits": 8,
            "keyspace_misses": 2,
        }

    def close(self) -> None:
        """No-op close for the fake client."""


def _cache(
    fake: FakeRedis,
    *,
    prefix: str = "test:",
    snapshot_max: int = FULL_CACHE_CAP,
) -> RedisCache:
    """Build a cache over the fake client.

    Returns:
        A RedisCache wired to the fake client.
    """
    options = RedisOptions(
        prefix=prefix, ttl_seconds=600, snapshot_max_bytes=snapshot_max
    )
    return RedisCache("fake://redis", options=options, client=fake)


def _ref(identity: str = FAKE_IDENTITY) -> ConvRef:
    """Build a conversation ref for tests.

    Returns:
        A conversation ref pointing at the fake conversation.
    """
    return ConvRef(
        account_identity=identity,
        conversation_id=FAKE_CONVERSATION,
        parent_id=FAKE_PARENT,
        turns=PREFIX_TURNS,
        updated=time.time(),
    )


def _record() -> ResponseRecord:
    """Build a response record for tests.

    Returns:
        A response record with the fake ids.
    """
    return ResponseRecord(
        response_id=FAKE_RESPONSE_ID,
        account_identity=FAKE_IDENTITY,
        conversation_id=FAKE_CONVERSATION,
        parent_id=FAKE_PARENT,
        model=FAKE_MODEL,
        created=time.time(),
    )


def _snapshot() -> TurnSnapshot:
    """Build a small text-only snapshot for tests.

    Returns:
        A text-only turn snapshot.
    """
    return TurnSnapshot(
        system_text="redis system",
        items=[HistoryItem(role="user", text="hello")],
    )


def _prefix_payload(identity: str, turns: int, updated: float) -> dict[str, object]:
    """Build a prefix payload for tests.

    Returns:
        A prefix payload dict for the identity.
    """
    return {
        "account_identity": identity,
        "conversation_id": "c",
        "parent_id": "p",
        "turns": turns,
        "updated": updated,
    }


def _meta_payload() -> dict[str, object]:
    """Build a response metadata payload for tests.

    Returns:
        A response metadata dict.
    """
    return {
        "account_identity": "a",
        "conversation_id": "c",
        "parent_id": "p",
        "model": FAKE_MODEL,
        "created": META_CREATED,
    }


class TestRedisCacheBasics(unittest.TestCase):
    """Cache read/write behavior over the fake client."""

    @staticmethod
    def test_prefix_round_trip_and_sticky_ownership() -> None:
        """Verify prefix caching plus foreign-owner write protection."""
        cache = _cache(FakeRedis())
        cache.cache_prefixes([
            (FAKE_PREFIX_HASH, _prefix_payload("owner", OWNER_TURNS, OWNER_UPDATED))
        ])
        hit = cache.find_prefix([FAKE_PREFIX_HASH])
        assert hit is not None
        assert hit[0] == EXPECTED_SINGLE_HIT
        assert hit[1]["account_identity"] == "owner"
        # A foreign failover replay must not overwrite the owner's pointer.
        cache.cache_prefixes([
            (FAKE_PREFIX_HASH, _prefix_payload("other", FORK_TURNS, FORK_UPDATED))
        ])
        hit_after = cache.find_prefix([FAKE_PREFIX_HASH])
        assert hit_after is not None
        assert hit_after[1]["account_identity"] == "owner"

    @staticmethod
    def test_response_and_snapshot_round_trip() -> None:
        """Verify response metadata plus small snapshots cache together."""
        cache = _cache(FakeRedis())
        expected = b'{"system_text": "s", "items": []}'
        cache.put_response(FAKE_RESPONSE_ID, _meta_payload(), expected)
        hit = cache.get_response(FAKE_RESPONSE_ID)
        assert hit is not None
        assert hit[0]["model"] == FAKE_MODEL
        assert hit[1] == expected
        assert cache.get_snapshot(FAKE_RESPONSE_ID) == expected

    @staticmethod
    def test_oversize_snapshot_skipped_but_metadata_cached() -> None:
        """Verify large snapshots stay SQLite-only while metadata caches."""
        cache = _cache(FakeRedis(), snapshot_max=SMALL_SNAPSHOT_CAP)
        cache.put_response(FAKE_RESPONSE_ID, _meta_payload(), b"x" * OVERSIZE_BYTES)
        hit = cache.get_response(FAKE_RESPONSE_ID)
        assert hit is not None
        assert hit[1] is None

    @staticmethod
    def test_models_round_trip_and_health_signals() -> None:
        """Verify model list caching plus ping/info observability."""
        cache = _cache(FakeRedis())
        cache.put_models("acct-1", [{"slug": FAKE_SLUG_AUTO}, {"slug": FAKE_SLUG_NEW}])
        models = cache.get_models("acct-1")
        assert models == [{"slug": FAKE_SLUG_AUTO}, {"slug": FAKE_SLUG_NEW}]
        assert cache.ping_ms() is not None
        summary = cache.info_summary()
        assert summary["connected_clients"] == CONNECTED_CLIENTS
        assert summary["hit_ratio"] == pytest.approx(EXPECTED_HIT_RATIO)

    @staticmethod
    def test_redis_failure_returns_none_without_raising() -> None:
        """Verify a down Redis degrades to misses, never exceptions."""
        fake = FakeRedis()
        fake.fail_next = True
        cache = _cache(fake)
        assert cache.find_prefix([FAKE_PREFIX_HASH]) is None
        assert cache.get_response(FAKE_RESPONSE_ID) is None
        assert cache.get_snapshot(FAKE_RESPONSE_ID) is None
        assert cache.get_models("acct-1") is None
        stats = cache.stats()
        assert stats["errors"] >= 1


class TestStoreRedisReadThrough(unittest.TestCase):
    """ConversationStore falls back to SQLite and warms Redis."""

    @override
    def setUp(self) -> None:
        """Create an isolated store wired to a fresh fake Redis."""
        self.temp_dir = tempfile.mkdtemp()
        self.fake = FakeRedis()
        self.store = ConversationStore(
            db_path=Path(self.temp_dir) / "redis_store.db", cache=_cache(self.fake)
        )

    @override
    def tearDown(self) -> None:
        """Remove the temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_prefix_hit_served_and_warmed(self) -> None:
        """Verify SQLite writes populate Redis and Redis serves reads."""
        h1 = item_hash("", "user", "redis hello")
        ref = _ref()
        self.store.record_turn([h1], ref)
        # Written through to Redis on record.
        assert self.fake.store, "record_turn must populate Redis"
        # A fresh store over an empty SQLite DB still resolves from Redis.
        fresh = ConversationStore(
            db_path=Path(self.temp_dir) / "empty.db", cache=_cache(self.fake)
        )
        matched = fresh.find([h1])
        assert matched is not None
        assert matched[1].conversation_id == FAKE_CONVERSATION

    def test_response_and_snapshot_read_through(self) -> None:
        """Verify responses/snapshots cache through and read back."""
        self.store.put_response(_record(), _snapshot())
        fresh = ConversationStore(
            db_path=Path(self.temp_dir) / "empty.db", cache=_cache(self.fake)
        )
        rec = fresh.get_response(FAKE_RESPONSE_ID)
        assert rec is not None
        assert rec.model == FAKE_MODEL
        assert rec.snapshot is not None
        assert rec.snapshot.system_text == "redis system"
        snap = fresh.get_snapshot(FAKE_RESPONSE_ID)
        assert snap is not None
        assert snap.items[0].text == "hello"

    def test_redis_down_falls_back_to_sqlite(self) -> None:
        """Verify Redis failures still serve from SQLite."""
        h1 = item_hash("", "user", "fallback hello")
        self.store.record_turn([h1], _ref())
        self.fake.fail_next = True  # next Redis op fails once
        matched = self.store.find([h1])
        assert matched is not None
        assert matched[1].conversation_id == FAKE_CONVERSATION
        rec_id = "resp_fallback"
        rec = ResponseRecord(
            response_id=rec_id,
            account_identity=FAKE_IDENTITY,
            conversation_id=FAKE_CONVERSATION,
            parent_id=FAKE_PARENT,
            model=FAKE_MODEL,
            created=time.time(),
        )
        self.fake.fail_next = False
        self.store.put_response(rec, _snapshot())
        self.fake.fail_next = True
        got = self.store.get_response(rec_id)
        assert got is not None
        assert got.model == FAKE_MODEL

    def test_no_redis_configured_behaves_like_before(self) -> None:
        """Verify an unconfigured store keeps pure-SQLite behavior."""
        plain = ConversationStore(db_path=Path(self.temp_dir) / "plain.db")
        with patch("app.store.get_cache", return_value=None):
            h1 = item_hash("", "user", "plain hello")
            plain.record_turn([h1], _ref())
            matched = plain.find([h1])
            assert matched is not None
            assert matched[1].conversation_id == FAKE_CONVERSATION
            plain.put_response(_record(), _snapshot())
            assert plain.get_snapshot(FAKE_RESPONSE_ID) is not None

    def test_failover_replay_does_not_pollute_redis(self) -> None:
        """Verify foreign replay hashes never overwrite the owner in Redis."""
        base = item_hash("", "user", "sticky base")
        owner_reply = item_hash(base, "assistant", "owner answer")
        fork_user = item_hash(owner_reply, "user", "failover follow-up")
        self.store.record_turn(
            [base, owner_reply],
            ConvRef(
                account_identity="sticky_owner",
                conversation_id="conv_owner",
                parent_id="msg_owner",
                turns=2,
                updated=time.time(),
            ),
        )
        self.fake.store.clear()  # simulate cold Redis: owner pointer expired
        self.store.record_turn(
            [base, owner_reply, fork_user],
            ConvRef(
                account_identity="sticky_failover",
                conversation_id="conv_fork",
                parent_id="msg_fork",
                turns=3,
                updated=time.time(),
            ),
        )
        # Shared hashes must not point at the fork in Redis; only the
        # fork-owned tail may be cached. SQLite stays authoritative.
        assert self.store.find([base, owner_reply]) is not None
        owner_hit = self.store.find([base, owner_reply])
        assert owner_hit is not None
        assert owner_hit[1].account_identity == "sticky_owner"
        fresh = ConversationStore(
            db_path=Path(self.temp_dir) / "cold.db", cache=_cache(self.fake)
        )
        shared = fresh.find([base, owner_reply])
        assert shared is None or shared[1].account_identity != "sticky_failover"
        full = fresh.find([base, owner_reply, fork_user])
        assert full is not None
        assert full[0] == FORK_MATCHED_LEN
        assert full[1].account_identity == "sticky_failover"

    def test_oversize_snapshot_still_reads_from_sqlite(self) -> None:
        """Verify metadata-only cache hits fall back to the SQLite snapshot."""
        tiny = _cache(self.fake, snapshot_max=SMALL_SNAPSHOT_CAP)
        tiny_store = ConversationStore(
            db_path=Path(self.temp_dir) / "tiny.db", cache=tiny
        )
        tiny_store.put_response(_record(), _snapshot())
        # Same SQLite DB, metadata-only Redis hit: must still return snapshot.
        same = ConversationStore(db_path=Path(self.temp_dir) / "tiny.db", cache=tiny)
        rec = same.get_response(FAKE_RESPONSE_ID)
        assert rec is not None
        assert rec.snapshot is not None
        assert rec.snapshot.items[0].text == "hello"

    def test_text_snapshot_rows_still_decode(self) -> None:
        """Verify TEXT-stored snapshots decode instead of reading as missing."""
        self.store.put_response(_record(), _snapshot())
        with sqlite3.connect(str(self.store.db_path)) as conn:
            row = conn.execute(
                "SELECT snapshot_json FROM responses WHERE response_id = ?",
                (FAKE_RESPONSE_ID,),
            ).fetchone()
            assert row is not None
            assert isinstance(row[0], bytes)
            conn.execute(
                "UPDATE responses SET snapshot_json = ? WHERE response_id = ?",
                (row[0].decode("utf-8"), FAKE_RESPONSE_ID),
            )
            conn.commit()
        plain = ConversationStore(db_path=self.store.db_path)
        with patch("app.store.get_cache", return_value=None):
            snap = plain.get_snapshot(FAKE_RESPONSE_ID)
            assert snap is not None
            assert snap.items[0].text == "hello"

    @staticmethod
    def test_bool_payloads_rejected_as_misses() -> None:
        """Verify bool integer fields fail validation instead of caching."""
        fake = FakeRedis({
            "test:prefix:bad": (
                '{"account_identity": "a", "conversation_id": "c",'
                ' "parent_id": "p", "turns": true, "updated": 1.0}'
            ),
            "test:resp:bad": (
                '{"account_identity": "a", "conversation_id": "c",'
                ' "parent_id": "p", "model": "m", "created": true}'
            ),
        })
        cache = _cache(fake)
        assert cache.find_prefix(["bad"]) is None
        assert cache.get_response("bad") is None


class TestSharedCacheGating(unittest.TestCase):
    """Shared cache singleton honors the REDIS_URL gate."""

    @override
    def tearDown(self) -> None:
        """Reset config and the shared cache after each test."""
        config.REDIS_URL = ""
        reset_cache()

    @staticmethod
    def test_empty_url_disables_cache() -> None:
        """Verify an empty REDIS_URL returns no shared cache."""
        config.REDIS_URL = ""
        assert get_cache() is None

    @staticmethod
    def test_corrupt_payloads_fall_back_without_raising() -> None:
        """Verify corrupt Redis payloads degrade to misses, never exceptions."""
        fake = FakeRedis({
            "test:prefix:bad": "not json",
            "test:resp:bad": "[]",
            "test:models:bad": "{oops",
        })
        cache = _cache(fake)
        assert cache.find_prefix(["bad"]) is None
        assert cache.get_response("bad") is None
        assert cache.get_snapshot("bad") is None
        assert cache.get_models("bad") is None


if __name__ == "__main__":
    unittest.main()
