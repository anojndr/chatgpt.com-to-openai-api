# Copyright 2026 chatgpt-to-openai-api contributors.
"""Multi-turn registry: maps client-side histories to real ChatGPT conversations.

A conversation is identified by a rolling hash chain over normalized history
items [(role, canonical_text)]. Given a client request with the full history
(the usual Chat Completions pattern), the longest previously-recorded prefix
match lets us continue the real ChatGPT conversation by sending ONLY the new
trailing messages instead of re-sending everything. Persisted with SQLite so
all mappings, references, and snapshots survive restarts.

Hot lookups are served from Redis when configured (``REDIS_URL``): a
read-through cache over the same prefix/response payloads. SQLite stays
authoritative -- every Redis miss or failure falls back to SQLite, and every
write persists to SQLite before populating Redis best-effort.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from . import config
from .adapters import FileInput, HistoryItem, ImageInput
from .redis_cache import RedisCache, get_cache

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


log = logging.getLogger("store")

MAX_PREFIX_ROWS = 20000
MAX_RESPONSE_ROWS = 5000


def item_hash(prev: str, role: str, canon: str) -> str:
    """Hash one history item into the rolling conversation chain.

    Returns:
        The 32-character hex hash for the history item.

    """
    h = hashlib.sha256()
    h.update(prev.encode())
    h.update(f"|{role}|".encode())
    h.update(canon.encode())
    return h.hexdigest()[:32]


def canon_content(text: str, extra: Iterable[object] | None = None) -> str:
    """Build canonical text for one message plus attachment descriptors.

    Returns:
        Canonical text with attachment descriptors appended when present.

    """
    if extra:
        return text + "\x00" + repr(sorted(map(str, extra)))
    return text


@dataclass
class ConvRef:
    """Pointer to a live ChatGPT conversation for prefix continuation."""

    account_identity: str
    conversation_id: str
    parent_id: str  # last assistant message id
    turns: int
    updated: float


@dataclass
class TurnSnapshot:
    """Full client-visible context of one completed turn.

    Retained so a later previous_response_id call can rebuild the entire
    conversation -- every turn's text plus its images and file attachments --
    on ANY account if the owning account fails. Items share parse-time binary
    buffers and are treated as read-only; trimming never mutates them.
    """

    system_text: str
    items: list[HistoryItem]

    def payload_bytes(self) -> int:
        """Sum the binary payload held by this snapshot.

        Returns:
            Total bytes of image and file payloads in the snapshot.

        """
        return sum(len(im.data) for it in self.items for im in it.images) + sum(
            len(f.data) for it in self.items for f in it.files
        )


def _trim_snapshot(snap: TurnSnapshot, cap_bytes: int) -> TurnSnapshot:
    """Drop largest binaries until under cap. Text is always kept whole.

    Returns:
        The original snapshot when under cap, else a trimmed copy.

    """
    if snap.payload_bytes() <= cap_bytes:
        return snap
    sized: list[tuple[int, int, str, int]] = []  # (size, item_idx, kind, part_idx)
    for ii, it in enumerate(snap.items):
        for ki, im in enumerate(it.images):
            sized.append((len(im.data), ii, "img", ki))
        for kf, f in enumerate(it.files):
            sized.append((len(f.data), ii, "file", kf))
    sized.sort(reverse=True)
    drop: set[tuple[int, str, int]] = set()
    total = snap.payload_bytes()
    for size, ii, kind, idx in sized:
        if total <= cap_bytes:
            break
        total -= size
        drop.add((ii, kind, idx))
    out: list[HistoryItem] = []
    for ii, it in enumerate(snap.items):
        ni = copy(it)
        ni.images = [im for k, im in enumerate(it.images) if (ii, "img", k) not in drop]
        ni.files = [f for k, f in enumerate(it.files) if (ii, "file", k) not in drop]
        out.append(ni)
    return TurnSnapshot(system_text=snap.system_text, items=out)


@dataclass
class ResponseRecord:
    """Stored Responses API response with its replayable snapshot."""

    response_id: str
    account_identity: str
    conversation_id: str
    parent_id: str  # parent for the NEXT turn (assistant msg id of this response)
    model: str
    created: float
    snapshot: TurnSnapshot | None = None


def _serialize_snapshot(snap: TurnSnapshot) -> bytes:
    """Encode a turn snapshot as UTF-8 JSON bytes.

    Returns:
        The snapshot encoded as UTF-8 JSON bytes.

    """
    data = {
        "system_text": snap.system_text,
        "items": [
            {
                "role": it.role,
                "text": it.text,
                "images": [
                    {
                        "filename": im.filename,
                        "mime": im.mime,
                        "data_b64": base64.b64encode(im.data).decode("ascii"),
                    }
                    for im in it.images
                ],
                "files": [
                    {
                        "filename": f.filename,
                        "mime": f.mime,
                        "data_b64": base64.b64encode(f.data).decode("ascii"),
                    }
                    for f in it.files
                ],
            }
            for it in snap.items
        ],
    }
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


def _decode_binary(entry: dict[str, object]) -> bytes:
    """Decode one snapshot binary entry (base64, falling back to hex).

    Returns:
        The decoded binary bytes, or empty bytes when absent.

    """
    if "data_b64" in entry:
        raw_b64 = entry["data_b64"]
        text_b64 = raw_b64 if isinstance(raw_b64, str) else ""
        return base64.b64decode(text_b64)
    if "data" in entry:
        raw = entry["data"]
        text = raw if isinstance(raw, str) else ""
        try:
            return bytes.fromhex(text)
        except ValueError:
            return base64.b64decode(text)
    return b""


def _deserialize_snapshot(raw: bytes | str) -> TurnSnapshot | None:
    """Rebuild a turn snapshot from stored JSON bytes, or None if corrupt.

    Returns:
        The rebuilt turn snapshot, or None when the payload is corrupt.

    """
    items: list[HistoryItem] = []
    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        data = json.loads(text)
        for it in data.get("items", []):
            images = [
                ImageInput(im["filename"], im["mime"], _decode_binary(im))
                for im in it.get("images", [])
                if "data_b64" in im or "data" in im
            ]
            files = [
                FileInput(f["filename"], f["mime"], _decode_binary(f))
                for f in it.get("files", [])
                if "data_b64" in f or "data" in f
            ]
            items.append(
                HistoryItem(
                    role=it.get("role", "user"),
                    text=it.get("text", ""),
                    images=images,
                    files=files,
                ),
            )
        return TurnSnapshot(system_text=data.get("system_text", ""), items=items)
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        log.warning("failed to deserialize snapshot: %s", e)
        return None


def _coerce_snapshot_bytes(value: object) -> bytes | None:
    """Coerce a SQLite snapshot cell to bytes.

    Returns:
        The cell as UTF-8 bytes, or None when absent or non-text.
    """
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, memoryview):
        return value.tobytes()
    return None


def _is_plain_int(value: object) -> bool:
    """Whether a value is a JSON integer (bool excluded).

    Returns:
        True for int values that are not bool.
    """
    return type(value) is int


def _is_number(value: object) -> bool:
    """Whether a value is a JSON number (bool excluded).

    Returns:
        True for int/float values that are not bool.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _prefix_payload(ref: ConvRef) -> dict[str, object]:
    """Encode a conversation ref as a Redis payload.

    Returns:
        The conversation ref as a JSON-able dict.
    """
    return {
        "account_identity": ref.account_identity,
        "conversation_id": ref.conversation_id,
        "parent_id": ref.parent_id,
        "turns": ref.turns,
        "updated": ref.updated,
    }


def _ref_from_payload(payload: dict[str, object]) -> ConvRef | None:
    """Rebuild a conversation ref from a Redis payload.

    Returns:
        The conversation ref, or None when the payload is unusable.
    """
    account = payload.get("account_identity")
    conversation = payload.get("conversation_id")
    parent = payload.get("parent_id")
    turns = payload.get("turns")
    updated = payload.get("updated")
    if (
        not isinstance(account, str)
        or not isinstance(conversation, str)
        or not isinstance(parent, str)
        or not _is_plain_int(turns)
        or not _is_number(updated)
    ):
        return None
    turns_int = turns if type(turns) is int else 0
    updated_num = updated if isinstance(updated, (int, float)) else 0.0
    return ConvRef(
        account_identity=account,
        conversation_id=conversation,
        parent_id=parent,
        turns=turns_int,
        updated=float(updated_num),
    )


def _response_meta(rec: ResponseRecord) -> dict[str, object]:
    """Encode response metadata as a Redis payload.

    Returns:
        The response metadata as a JSON-able dict.
    """
    return {
        "account_identity": rec.account_identity,
        "conversation_id": rec.conversation_id,
        "parent_id": rec.parent_id,
        "model": rec.model,
        "created": rec.created,
    }


def _record_from_meta(
    response_id: str, meta: dict[str, object], snapshot_raw: bytes | None
) -> ResponseRecord | None:
    """Rebuild a response record from cached metadata.

    Returns:
        The response record, or None when the metadata is unusable.
    """
    account = meta.get("account_identity")
    conversation = meta.get("conversation_id")
    parent = meta.get("parent_id")
    model = meta.get("model")
    created = meta.get("created")
    if (
        not isinstance(account, str)
        or not isinstance(conversation, str)
        or not isinstance(parent, str)
        or not isinstance(model, str)
        or not _is_number(created)
    ):
        return None
    created_num = created if isinstance(created, (int, float)) else 0.0
    snapshot = _deserialize_snapshot(snapshot_raw) if snapshot_raw is not None else None
    return ResponseRecord(
        response_id=response_id,
        account_identity=account,
        conversation_id=conversation,
        parent_id=parent,
        model=model,
        created=float(created_num),
        snapshot=snapshot,
    )


class ConversationStore:
    """SQLite-backed conversation prefix match and response snapshot store."""

    def __init__(
        self, db_path: Path | str | None = None, cache: RedisCache | None = None
    ) -> None:
        """Open (creating parent directories) the backing SQLite database."""
        self.db_path = Path(db_path) if db_path is not None else config.DB_PATH
        self._cache_override = cache
        self._lock = threading.RLock()
        self._local = threading.local()
        self._init_db()

    def _cache(self) -> RedisCache | None:
        """Return the Redis cache override, else the shared configured cache."""
        return self._cache_override if self._cache_override is not None else get_cache()

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if not isinstance(conn, sqlite3.Connection):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                check_same_thread=False,
                isolation_level=None,  # autocommit mode
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=30000")
            self._local.conn = conn
        return conn

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._get_conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Yield a transactional connection with rollback on error.

        Public seam for tests and maintenance scripts that need raw
        SQL access; delegates identically to the internal transaction.

        Yields:
            The transactional SQLite connection.

        """
        with self._transaction() as conn:
            yield conn

    def _init_db(self) -> None:
        with self._lock, self._transaction() as conn:
            conn.execute("""
                    CREATE TABLE IF NOT EXISTS prefixes (
                        hash TEXT PRIMARY KEY,
                        account_identity TEXT NOT NULL,
                        conversation_id TEXT NOT NULL,
                        parent_id TEXT NOT NULL,
                        turns INTEGER NOT NULL,
                        updated REAL NOT NULL
                    )
                """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_prefixes_updated ON prefixes(updated)",
            )

            conn.execute("""
                    CREATE TABLE IF NOT EXISTS responses (
                        response_id TEXT PRIMARY KEY,
                        account_identity TEXT NOT NULL,
                        conversation_id TEXT NOT NULL,
                        parent_id TEXT NOT NULL,
                        model TEXT NOT NULL,
                        created REAL NOT NULL,
                        snapshot_bytes INTEGER NOT NULL DEFAULT 0,
                        snapshot_json BLOB
                    )
                """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_responses_created "
                "ON responses(created)",
            )

    # ---------- chat-completions style ----------
    def find(self, hashes: list[str]) -> tuple[int, ConvRef] | None:
        """Return the longest prefix match as (matched_len, ref).

        Returns:
            The matched length and conversation ref, or None when no match.

        """
        if not hashes:
            return None
        cached = self._find_cached(hashes)
        if cached is not None:
            return cached
        return self._find_sqlite(hashes)

    def _find_cached(self, hashes: list[str]) -> tuple[int, ConvRef] | None:
        """Return the longest Redis-cached prefix, or None on miss.

        Returns:
            The cached matched length and ref, or None when uncached.
        """
        cache = self._cache()
        if cache is None:
            return None
        hit = cache.find_prefix(hashes)
        if hit is None:
            return None
        matched, payload = hit
        ref = _ref_from_payload(payload)
        return (matched, ref) if ref is not None else None

    def _find_sqlite(self, hashes: list[str]) -> tuple[int, ConvRef] | None:
        """Return the longest SQLite prefix match, warming Redis on hit.

        Returns:
            The matched length and conversation ref, or None when no match.
        """
        found: tuple[int, ConvRef] | None = None
        with self._lock:
            conn = self._get_conn()
            for k in range(len(hashes), 0, -1):
                cur = conn.execute(
                    "SELECT account_identity, conversation_id, parent_id,"
                    " turns, updated FROM prefixes WHERE hash = ?",
                    (hashes[k - 1],),
                )
                row = cur.fetchone()
                if row is not None:
                    found = (
                        k,
                        ConvRef(
                            account_identity=row[0],
                            conversation_id=row[1],
                            parent_id=row[2],
                            turns=row[3],
                            updated=row[4],
                        ),
                    )
                    break
        if found is not None:
            self._warm_prefix_cache(hashes[: found[0]], found[1])
        return found

    def _warm_prefix_cache(self, hashes: list[str], ref: ConvRef) -> None:
        """Best-effort cache of one SQLite prefix hit (never raises)."""
        cache = self._cache()
        if cache is None:
            return
        cache.cache_prefixes([(h, _prefix_payload(ref)) for h in hashes])

    def record_turn(self, hashes: list[str], ref: ConvRef) -> None:
        """Store hash chain entries for turn prefix hashes.

        Ownership is sticky: the account that first recorded a prefix keeps
        it. A later turn served by another account (failover replay) shares
        those early hashes but lives in a different server-side thread --
        overwriting them would repoint the owner's live conversation at the
        fork, so the next genuine continuation would resume the wrong
        thread with phantom turns. Same-owner re-records still advance
        (new parent/turns); other-owner writes are skipped entirely.
        """
        if not hashes:
            return
        now = time.time()
        ref.updated = now
        with self._lock, self._transaction() as conn:
            owned: list[str] = []
            for h in hashes:
                cur = conn.execute(
                    "SELECT account_identity FROM prefixes WHERE hash = ?",
                    (h,),
                )
                row = cur.fetchone()
                if row is None:
                    conn.execute(
                        """
                        INSERT INTO prefixes (
                            hash, account_identity, conversation_id,
                            parent_id, turns, updated
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            h,
                            ref.account_identity,
                            ref.conversation_id,
                            ref.parent_id,
                            ref.turns,
                            ref.updated,
                        ),
                    )
                    owned.append(h)
                elif row[0] == ref.account_identity:
                    conn.execute(
                        """
                        UPDATE prefixes SET
                            conversation_id = ?,
                            parent_id = ?,
                            turns = ?,
                            updated = ?
                        WHERE hash = ?
                        """,
                        (
                            ref.conversation_id,
                            ref.parent_id,
                            ref.turns,
                            ref.updated,
                            h,
                        ),
                    )
                    owned.append(h)
                # Else: foreign-owned prefix shared with another account's
                # live thread (a failover replay forked it). Leave the
                # owner's pointer alone so its continuations keep resuming
                # the right server-side conversation.

            # Prune old prefixes if table is large
            cur = conn.execute("SELECT COUNT(*) FROM prefixes")
            count = cur.fetchone()[0]
            if count > MAX_PREFIX_ROWS:
                cutoff = now - config.CONVERSATION_TTL_HOURS * 3600
                conn.execute("DELETE FROM prefixes WHERE updated < ?", (cutoff,))
        cache = self._cache()
        if cache is not None and owned:
            cache.cache_prefixes([(h, _prefix_payload(ref)) for h in owned])

    # ---------- responses API ----------
    def put_response(
        self,
        rec: ResponseRecord,
        snapshot: TurnSnapshot | None = None,
    ) -> None:
        """Persist a response record with its optional turn snapshot."""
        snap_blob: bytes | None = None
        snap_size: int = 0
        if snapshot is not None:
            trimmed = _trim_snapshot(snapshot, config.SNAPSHOT_FILE_CAP_MB << 20)
            rec.snapshot = trimmed
            snap_size = trimmed.payload_bytes()
            snap_blob = _serialize_snapshot(trimmed)

        with self._lock, self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO responses (
                    response_id, account_identity, conversation_id,
                    parent_id, model, created, snapshot_bytes, snapshot_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(response_id) DO UPDATE SET
                    account_identity=excluded.account_identity,
                    conversation_id=excluded.conversation_id,
                    parent_id=excluded.parent_id,
                    model=excluded.model,
                    created=excluded.created,
                    snapshot_bytes=excluded.snapshot_bytes,
                    snapshot_json=excluded.snapshot_json
                """,
                (
                    rec.response_id,
                    rec.account_identity,
                    rec.conversation_id,
                    rec.parent_id,
                    rec.model,
                    rec.created,
                    snap_size,
                    snap_blob,
                ),
            )

            # Enforce store-wide snapshot bytes budget
            budget = config.SNAPSHOT_STORE_CAP_MB << 20
            cur = conn.execute(
                "SELECT COALESCE(SUM(snapshot_bytes), 0) FROM responses",
            )
            total_snap_bytes = cur.fetchone()[0]
            if total_snap_bytes > budget:
                # Drop snapshots from oldest records first
                cur = conn.execute(
                    "SELECT response_id, snapshot_bytes FROM responses "
                    "WHERE snapshot_bytes > 0 ORDER BY created ASC",
                )
                for row in cur.fetchall():
                    r_id, r_size = row[0], row[1]
                    conn.execute(
                        "UPDATE responses SET snapshot_bytes = 0,"
                        " snapshot_json = NULL WHERE response_id = ?",
                        (r_id,),
                    )
                    total_snap_bytes -= r_size
                    if total_snap_bytes <= budget:
                        break

            # Prune old responses count / TTL
            cur = conn.execute("SELECT COUNT(*) FROM responses")
            count = cur.fetchone()[0]
            if count > MAX_RESPONSE_ROWS:
                cutoff = time.time() - config.CONVERSATION_TTL_HOURS * 3600
                conn.execute("DELETE FROM responses WHERE created < ?", (cutoff,))
        cache = self._cache()
        if cache is not None:
            cache.put_response(rec.response_id, _response_meta(rec), snap_blob)

    def get_response(self, response_id: str) -> ResponseRecord | None:
        """Fetch one response record by id, or None when unknown.

        Returns:
            The stored response record, or None when unknown.

        """
        cached = self._get_response_cached(response_id)
        if cached is not None:
            return cached
        return self._get_response_sqlite(response_id)

    def _get_response_cached(self, response_id: str) -> ResponseRecord | None:
        """Return the Redis-cached response, or None on miss.

        A metadata-only hit (oversize snapshot stayed SQLite-only) falls back
        to SQLite so the full replayable snapshot is never hidden.

        Returns:
            The cached response record, or None when uncached.
        """
        cache = self._cache()
        if cache is None or not response_id:
            return None
        hit = cache.get_response(response_id)
        if hit is None:
            return None
        meta, snapshot_raw = hit
        if snapshot_raw is None:
            return self._get_response_sqlite(response_id)
        return _record_from_meta(response_id, meta, snapshot_raw)

    def _get_response_sqlite(self, response_id: str) -> ResponseRecord | None:
        """Return the SQLite response, warming Redis on hit.

        Returns:
            The stored response record, or None when unknown.
        """
        with self._lock:
            conn = self._get_conn()
            cur = conn.execute(
                "SELECT response_id, account_identity, conversation_id,"
                " parent_id, model, created, snapshot_json"
                " FROM responses WHERE response_id = ?",
                (response_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            raw = _coerce_snapshot_bytes(row[6])
            snapshot = _deserialize_snapshot(raw) if raw is not None else None
            rec = ResponseRecord(
                response_id=row[0],
                account_identity=row[1],
                conversation_id=row[2],
                parent_id=row[3],
                model=row[4],
                created=row[5],
                snapshot=snapshot,
            )
        cache = self._cache()
        if cache is not None:
            cache.put_response(response_id, _response_meta(rec), raw)
        return rec

    def get_snapshot(self, response_id: str) -> TurnSnapshot | None:
        """Fetch one response snapshot by id, or None when missing.

        Returns:
            The stored turn snapshot, or None when missing.

        """
        cache = self._cache()
        if cache is not None and response_id:
            raw_cached = cache.get_snapshot(response_id)
            if raw_cached is not None:
                snap = _deserialize_snapshot(raw_cached)
                if snap is not None:
                    return snap
        with self._lock:
            conn = self._get_conn()
            cur = conn.execute(
                "SELECT snapshot_json FROM responses WHERE response_id = ?",
                (response_id,),
            )
            row = cur.fetchone()
            if row is None or row[0] is None:
                return None
            raw = _coerce_snapshot_bytes(row[0])
            if raw is None:
                return None
        if cache is not None:
            cache.put_snapshot(response_id, raw)
        return _deserialize_snapshot(raw)


STORE = ConversationStore()
