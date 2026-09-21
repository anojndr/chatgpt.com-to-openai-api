# Copyright 2026 chatgpt-to-openai-api contributors.
"""Redis read-through cache for conversation continuity and model lists.

SQLite stays authoritative: every read checks Redis first and falls back to
SQLite on a miss or any Redis failure, and every write persists to SQLite
before populating Redis best-effort. An empty ``REDIS_URL`` disables Redis
entirely; a down Redis only costs one fast timeout and never fails a request.

Key layout (lowercase, colon-separated per redis-core)::

    {prefix}prefix:{hash}        prefix hash -> ConvRef JSON
    {prefix}resp:{response_id}    response metadata JSON (no snapshot)
    {prefix}snap:{response_id}    snapshot JSON (only when small)
    {prefix}models:{identity}     backend model list JSON

Connection policy (per redis-connections): one shared client on a single
``ConnectionPool`` sized by ``REDIS_MAX_CONNECTIONS``, pipelined multi-key
writes/reads, fail-fast connect timeout, no ``KEYS``/``SCAN`` anywhere.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Protocol, cast

import redis

from . import config

log = logging.getLogger("redis_cache")

MODELS_CACHE_TTL_SECONDS = 3600
MIN_TTL_SECONDS = 60

PrefixHit = tuple[int, dict[str, object]]
ResponseHit = tuple[dict[str, object], bytes | None]


@dataclass
class RedisOptions:
    """Tunable knobs for the shared Redis cache."""

    prefix: str = "c2o:"
    ttl_seconds: int = 604800
    snapshot_max_bytes: int = 262144
    connect_timeout: float = 2.0
    socket_timeout: float = 5.0
    max_connections: int = 50


class _RedisPipeline(Protocol):
    """Narrow pipeline surface used for batched cache writes."""

    def setex(self, name: str, ttl: int, value: str, /) -> object:
        """Queue one key write with an expiry.

        Returns:
            The queued command handle.
        """
        ...

    def execute(self) -> object:
        """Flush queued writes.

        Returns:
            The per-command results.
        """
        ...


class _RedisClient(Protocol):
    """Narrow Redis surface used by the cache (real or test double)."""

    def get(self, name: str) -> object:
        """Fetch one key.

        Returns:
            The stored value, or None when missing.
        """
        ...

    def setex(self, name: str, ttl: int, value: str, /) -> object:
        """Set one key with an expiry in seconds.

        Returns:
            The set outcome.
        """
        ...

    def mget(self, keys: list[str]) -> object:
        """Fetch keys in one round trip.

        Returns:
            The values aligned with the input keys.
        """
        ...

    def ping(self) -> object:
        """Probe liveness.

        Returns:
            The probe outcome.
        """
        ...

    def pipeline(self, *, transaction: bool = True) -> _RedisPipeline:
        """Open a write pipeline.

        Returns:
            The pipeline handle.
        """
        ...

    def info(self, section: str | None = None) -> object:
        """Return a server info section.

        Returns:
            The info mapping.
        """
        ...

    def close(self) -> object:
        """Release pooled connections.

        Returns:
            The close outcome.
        """
        ...


def _as_text(value: object) -> str | None:
    """Coerce a Redis reply to text.

    Returns:
        The reply as text, or None for missing/non-text replies.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    return None


def _parse_dict(raw: str) -> dict[str, object] | None:
    """Parse a JSON object payload.

    Returns:
        The parsed dict, or None when malformed or not an object.
    """
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    return {str(key): val for key, val in data.items()}


def _is_json_int(value: object) -> bool:
    """Whether a value is a JSON integer (bool excluded).

    Returns:
        True for int values that are not bool.
    """
    return type(value) is int


def _is_json_number(value: object) -> bool:
    """Whether a value is a JSON number (bool excluded).

    Returns:
        True for int/float values that are not bool.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _valid_prefix(payload: dict[str, object]) -> bool:
    """Whether a cached prefix payload carries a usable conversation ref.

    Returns:
        True when all ConvRef fields are present with usable types.
    """
    return (
        isinstance(payload.get("account_identity"), str)
        and isinstance(payload.get("conversation_id"), str)
        and isinstance(payload.get("parent_id"), str)
        and _is_json_int(payload.get("turns"))
        and _is_json_number(payload.get("updated"))
    )


def _valid_meta(payload: dict[str, object]) -> bool:
    """Whether a cached payload carries usable response metadata.

    Returns:
        True when all response metadata fields are present.
    """
    return (
        isinstance(payload.get("account_identity"), str)
        and isinstance(payload.get("conversation_id"), str)
        and isinstance(payload.get("parent_id"), str)
        and isinstance(payload.get("model"), str)
        and _is_json_number(payload.get("created"))
    )


def _dump(payload: dict[str, object] | list[dict[str, object]]) -> str:
    """Encode a cache payload as JSON.

    Returns:
        The payload encoded as a JSON string.
    """
    return json.dumps(payload, ensure_ascii=False)


class RedisCache:
    """Fail-open Redis read-through cache over one pooled client."""

    def __init__(
        self,
        url: str = "",
        *,
        options: RedisOptions | None = None,
        client: _RedisClient | None = None,
    ) -> None:
        """Build the cache, opening one pooled client unless injected."""
        opts = options or RedisOptions()
        self.url = url
        self._prefix = opts.prefix if opts.prefix.endswith(":") else opts.prefix + ":"
        self._ttl = max(int(opts.ttl_seconds), MIN_TTL_SECONDS)
        self._snapshot_max = max(int(opts.snapshot_max_bytes), 0)
        self._connect_timeout = float(opts.connect_timeout)
        self._socket_timeout = float(opts.socket_timeout)
        self._max_connections = int(opts.max_connections)
        self._client: _RedisClient | None = client
        self._pool: redis.ConnectionPool | None = None
        if self._client is None and url:
            opened = self._open_pool(url)
            if opened is not None:
                self._client = opened
        self._stats_lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._errors = 0

    def _open_pool(self, url: str) -> _RedisClient | None:
        """Create one pooled client.

        Returns:
            The pooled Redis client, or None when creation failed.
        """
        try:
            pool = redis.ConnectionPool.from_url(
                url,
                max_connections=self._max_connections,
                socket_connect_timeout=self._connect_timeout,
                socket_timeout=self._socket_timeout,
                retry_on_timeout=True,
                health_check_interval=30,
                decode_responses=True,
            )
        except (redis.RedisError, OSError, ValueError) as e:
            log.debug("redis pool creation failed, disabled: %s", e)
            return None
        self._pool = pool
        return cast("_RedisClient", redis.Redis(connection_pool=pool))

    @property
    def enabled(self) -> bool:
        """Whether a Redis client backs this cache.

        Returns:
            True when a client is configured, else False.
        """
        return self._client is not None

    def _note_hit(self) -> None:
        with self._stats_lock:
            self._hits += 1

    def _note_miss(self) -> None:
        with self._stats_lock:
            self._misses += 1

    def _note_error(self, op: str, exc: BaseException) -> None:
        with self._stats_lock:
            self._errors += 1
        log.debug("redis %s failed, falling back: %s", op, exc)

    def stats(self) -> dict[str, int]:
        """Return cache hit/miss/error counters.

        Returns:
            The hit, miss, and error counters.
        """
        with self._stats_lock:
            return {"hits": self._hits, "misses": self._misses, "errors": self._errors}

    def _prefix_key(self, entry_hash: str) -> str:
        return f"{self._prefix}prefix:{entry_hash}"

    def _response_key(self, response_id: str) -> str:
        return f"{self._prefix}resp:{response_id}"

    def _snapshot_key(self, response_id: str) -> str:
        return f"{self._prefix}snap:{response_id}"

    def _models_key(self, identity: str) -> str:
        safe = re.sub(r"\s+", "_", identity.strip()) or "unknown"
        return f"{self._prefix}models:{safe}"

    def _mget(self, op: str, keys: list[str]) -> list[object] | None:
        """Fetch keys in one round trip.

        Returns:
            The values aligned with the input keys, or None on failure.
        """
        client = self._client
        if client is None:
            return None
        try:
            values = client.mget(keys)
        except (redis.RedisError, OSError, ValueError, TypeError) as e:
            self._note_error(op, e)
            return None
        if not isinstance(values, list):
            return None
        return list(values)

    @staticmethod
    def _owned_by_other(old: object, owner: object) -> bool:
        """Whether a cached entry belongs to another account's thread.

        Returns:
            True when the cached pointer is valid and foreign-owned.
        """
        old_text = _as_text(old)
        old_payload = _parse_dict(old_text) if old_text is not None else None
        return (
            old_payload is not None
            and _valid_prefix(old_payload)
            and old_payload.get("account_identity") != owner
        )

    def _write_many(self, op: str, pairs: list[tuple[str, str]], ttl: int) -> None:
        """Write pairs through one non-transactional pipeline."""
        client = self._client
        if client is None or not pairs:
            return
        try:
            pipe = client.pipeline(transaction=False)
            for key, value in pairs:
                pipe.setex(key, ttl, value)
            pipe.execute()
        except (redis.RedisError, OSError, ValueError, TypeError) as e:
            self._note_error(op, e)

    def cache_prefixes(self, entries: list[tuple[str, dict[str, object]]]) -> None:
        """Cache prefix hashes, never overwriting a foreign owner's pointer."""
        if self._client is None or not entries:
            return
        keys = [self._prefix_key(h) for h, _ in entries]
        existing = self._mget("cache_prefixes", keys)
        if existing is None:
            return
        pending: list[tuple[str, str]] = []
        for (entry_hash, payload), old in zip(entries, existing, strict=True):
            if self._owned_by_other(old, payload.get("account_identity")):
                continue  # foreign-owned live thread: keep owner's pointer
            pending.append((self._prefix_key(entry_hash), _dump(payload)))
        self._write_many("cache_prefixes", pending, self._ttl)

    def find_prefix(self, hashes: list[str]) -> PrefixHit | None:
        """Return the longest cached prefix as (matched_len, payload).

        Returns:
            The matched length and payload, or None on miss/failure.
        """
        if self._client is None or not hashes:
            return None
        keys = [self._prefix_key(h) for h in hashes]
        values = self._mget("find_prefix", keys)
        if values is None:
            return None
        for idx in range(len(hashes) - 1, -1, -1):
            raw = _as_text(values[idx]) if idx < len(values) else None
            if raw is None:
                continue
            payload = _parse_dict(raw)
            if payload is not None and _valid_prefix(payload):
                self._note_hit()
                return (idx + 1, payload)
        self._note_miss()
        return None

    def get_response(self, response_id: str) -> ResponseHit | None:
        """Return cached (metadata, snapshot bytes) for a response id.

        Returns:
            The metadata and optional snapshot bytes, or None on miss.
        """
        if self._client is None or not response_id:
            return None
        values = self._mget(
            "get_response",
            [self._response_key(response_id), self._snapshot_key(response_id)],
        )
        if values is None:
            return None
        meta_raw = _as_text(values[0]) if len(values) > 0 else None
        if meta_raw is None:
            self._note_miss()
            return None
        meta = _parse_dict(meta_raw)
        if meta is None or not _valid_meta(meta):
            self._note_miss()
            return None
        snap_raw = _as_text(values[1]) if len(values) > 1 else None
        self._note_hit()
        snap_bytes = snap_raw.encode("utf-8") if snap_raw is not None else None
        return (meta, snap_bytes)

    def put_response(
        self,
        response_id: str,
        meta: dict[str, object],
        snapshot_raw: bytes | None,
    ) -> None:
        """Cache response metadata, plus the snapshot when small enough."""
        if self._client is None or not response_id:
            return
        pairs = [(self._response_key(response_id), _dump(meta))]
        if snapshot_raw is not None and 0 < len(snapshot_raw) <= self._snapshot_max:
            try:
                text = snapshot_raw.decode("utf-8")
            except UnicodeDecodeError as e:
                self._note_error("put_response", e)
                return
            pairs.append((self._snapshot_key(response_id), text))
        self._write_many("put_response", pairs, self._ttl)

    def get_snapshot(self, response_id: str) -> bytes | None:
        """Return cached snapshot bytes for a response id.

        Returns:
            The snapshot bytes, or None on miss/failure.
        """
        client = self._client
        if client is None or not response_id:
            return None
        try:
            raw = _as_text(client.get(self._snapshot_key(response_id)))
        except (redis.RedisError, OSError, ValueError, TypeError) as e:
            self._note_error("get_snapshot", e)
            return None
        if raw is None:
            self._note_miss()
            return None
        self._note_hit()
        return raw.encode("utf-8")

    def put_snapshot(self, response_id: str, snapshot_raw: bytes) -> None:
        """Cache snapshot bytes, skipping payloads over the size cap."""
        client = self._client
        if client is None or not response_id:
            return
        if not 0 < len(snapshot_raw) <= self._snapshot_max:
            return
        try:
            text = snapshot_raw.decode("utf-8")
        except UnicodeDecodeError as e:
            self._note_error("put_snapshot", e)
            return
        try:
            client.setex(self._snapshot_key(response_id), self._ttl, text)
        except (redis.RedisError, OSError, ValueError, TypeError) as e:
            self._note_error("put_snapshot", e)

    def get_models(self, identity: str) -> list[dict[str, object]] | None:
        """Return the cached backend model list for one account.

        Returns:
            The cached model dicts, or None on miss/failure.
        """
        client = self._client
        if client is None or not identity.strip():
            return None
        try:
            raw = _as_text(client.get(self._models_key(identity)))
        except (redis.RedisError, OSError, ValueError, TypeError) as e:
            self._note_error("get_models", e)
            return None
        if raw is None:
            self._note_miss()
            return None
        try:
            data = json.loads(raw)
        except ValueError:
            self._note_miss()
            return None
        if not isinstance(data, list):
            self._note_miss()
            return None
        self._note_hit()
        return [
            {str(key): val for key, val in item.items()}
            for item in data
            if isinstance(item, dict)
        ]

    def put_models(self, identity: str, models: list[dict[str, object]]) -> None:
        """Cache the backend model list for one account (hourly TTL)."""
        client = self._client
        if client is None or not identity.strip():
            return
        try:
            client.setex(
                self._models_key(identity),
                MODELS_CACHE_TTL_SECONDS,
                _dump(models),
            )
        except (redis.RedisError, OSError, ValueError, TypeError) as e:
            self._note_error("put_models", e)

    def ping_ms(self) -> float | None:
        """Return the Redis round-trip latency in ms, or None when down.

        Returns:
            The round-trip latency in milliseconds, or None when down.
        """
        client = self._client
        if client is None:
            return None
        start = time.perf_counter()
        try:
            client.ping()
        except (redis.RedisError, OSError, ValueError) as e:
            self._note_error("ping", e)
            return None
        return (time.perf_counter() - start) * 1000.0

    def info_summary(self) -> dict[str, object]:
        """Return memory/client/throughput/hit-ratio signals for observability.

        Returns:
            The info subset, or an empty dict when unavailable.
        """
        client = self._client
        if client is None:
            return {}
        try:
            raw_info = client.info()
        except (redis.RedisError, OSError, ValueError, TypeError) as e:
            self._note_error("info", e)
            return {}
        if not isinstance(raw_info, dict):
            return {}
        info = cast("dict[str, object]", raw_info)
        out: dict[str, object] = {}
        for key in (
            "used_memory_human",
            "connected_clients",
            "blocked_clients",
            "instantaneous_ops_per_sec",
            "rejected_connections",
        ):
            if key in info:
                out[key] = info[key]
        hits = info.get("keyspace_hits")
        misses = info.get("keyspace_misses")
        if isinstance(hits, int) and isinstance(misses, int) and hits + misses > 0:
            out["hit_ratio"] = hits / (hits + misses)
        return out

    def close(self) -> None:
        """Release pooled connections."""
        client, self._client = self._client, None
        if self._pool is not None:
            pool, self._pool = self._pool, None
            try:
                pool.disconnect()
            except (redis.RedisError, OSError) as e:
                log.debug("redis pool disconnect failed: %s", e)
        if client is not None:
            try:
                client.close()
            except (redis.RedisError, OSError, AttributeError) as e:
                log.debug("redis close failed: %s", e)


@dataclass
class _SharedCache:
    """Process-wide shared cache holder (avoids rebinding a module global)."""

    cache: RedisCache | None = None


_SHARED = _SharedCache()
_SHARED_LOCK = threading.Lock()


def _options_from_config() -> RedisOptions:
    """Build cache options from the current process configuration.

    Returns:
        The cache options reflecting the current config module.
    """
    return RedisOptions(
        prefix=config.REDIS_KEY_PREFIX,
        ttl_seconds=config.REDIS_TTL_SECONDS,
        snapshot_max_bytes=config.REDIS_SNAPSHOT_MAX_BYTES,
        connect_timeout=config.REDIS_CONNECT_TIMEOUT,
        socket_timeout=config.REDIS_SOCKET_TIMEOUT,
        max_connections=config.REDIS_MAX_CONNECTIONS,
    )


def get_cache() -> RedisCache | None:
    """Return the shared cache, or None when REDIS_URL is empty.

    Returns:
        The shared cache instance, or None when Redis is disabled.
    """
    url = config.REDIS_URL.strip()
    if not url:
        return None
    with _SHARED_LOCK:
        if _SHARED.cache is None or _SHARED.cache.url != url:
            if _SHARED.cache is not None:
                _SHARED.cache.close()
            _SHARED.cache = RedisCache(url, options=_options_from_config())
        return _SHARED.cache


def reset_cache() -> None:
    """Drop the shared cache (test seam for isolation between tests)."""
    with _SHARED_LOCK:
        cache, _SHARED.cache = _SHARED.cache, None
    if cache is not None:
        cache.close()
