"""Multi-turn registry: maps client-side histories to real ChatGPT conversations.

A conversation is identified by a rolling hash chain over normalized history
items [(role, canonical_text)]. Given a client request with the full history
(the usual Chat Completions pattern), the longest previously-recorded prefix
match lets us continue the real ChatGPT conversation by sending ONLY the new
trailing messages instead of re-sending everything. Persisted with SQLite so
all mappings, references, and snapshots survive restarts.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
import time
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from . import config
from .adapters import FileInput, HistoryItem, ImageInput

log = logging.getLogger("store")


def item_hash(prev: str, role: str, canon: str) -> str:
    h = hashlib.sha256()
    h.update(prev.encode())
    h.update(f"|{role}|".encode())
    h.update(canon.encode())
    return h.hexdigest()[:32]


def canon_content(text: str, extra: Any = None) -> str:
    """Canonical text of one message including binary attachment descriptors."""
    if extra:
        return text + "\x00" + repr(sorted(map(str, extra)))
    return text


@dataclass
class ConvRef:
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
        return sum(len(im.data) for it in self.items for im in it.images) \
            + sum(len(f.data) for it in self.items for f in it.files)


def _trim_snapshot(snap: TurnSnapshot, cap_bytes: int) -> TurnSnapshot:
    """Drop largest binaries until under cap. Text is always kept whole."""
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
    response_id: str
    account_identity: str
    conversation_id: str
    parent_id: str  # parent for the NEXT turn (assistant msg id of this response)
    model: str
    created: float
    snapshot: TurnSnapshot | None = None


def _serialize_snapshot(snap: TurnSnapshot) -> bytes:
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


def _decode_binary(entry: dict) -> bytes:
    if "data_b64" in entry:
        return base64.b64decode(entry["data_b64"])
    if "data" in entry:
        try:
            return bytes.fromhex(entry["data"])
        except ValueError:
            return base64.b64decode(entry["data"])
    return b""


def _deserialize_snapshot(raw: bytes | str) -> TurnSnapshot | None:
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        data = json.loads(raw)
        items = []
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
            items.append(HistoryItem(role=it.get("role", "user"), text=it.get("text", ""), images=images, files=files))
        return TurnSnapshot(system_text=data.get("system_text", ""), items=items)
    except Exception as e:
        log.warning("failed to deserialize snapshot: %s", e)
        return None


class ConversationStore:
    """SQLite-backed conversation prefix match and response snapshot store."""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path is not None else config.DB_PATH
        self._lock = threading.RLock()
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
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

    def _init_db(self) -> None:
        with self._lock:
            with self._transaction() as conn:
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
                conn.execute("CREATE INDEX IF NOT EXISTS idx_prefixes_updated ON prefixes(updated)")

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
                conn.execute("CREATE INDEX IF NOT EXISTS idx_responses_created ON responses(created)")

    # ---------- chat-completions style ----------
    def find(self, hashes: list[str]) -> tuple[int, ConvRef] | None:
        """Longest prefix match. Returns (matched_len, ref)."""
        if not hashes:
            return None
        with self._lock:
            conn = self._get_conn()
            for k in range(len(hashes), 0, -1):
                cur = conn.execute(
                    "SELECT account_identity, conversation_id, parent_id, turns, updated FROM prefixes WHERE hash = ?",
                    (hashes[k - 1],),
                )
                row = cur.fetchone()
                if row is not None:
                    ref = ConvRef(
                        account_identity=row[0],
                        conversation_id=row[1],
                        parent_id=row[2],
                        turns=row[3],
                        updated=row[4],
                    )
                    return (k, ref)
        return None

    def record_turn(self, hashes: list[str], ref: ConvRef) -> None:
        """Store hash chain entries covering the whole updated history."""
        if not hashes:
            return
        now = time.time()
        ref.updated = now
        with self._lock:
            with self._transaction() as conn:
                for h in hashes:
                    conn.execute(
                        """
                        INSERT INTO prefixes (hash, account_identity, conversation_id, parent_id, turns, updated)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(hash) DO UPDATE SET
                            account_identity=excluded.account_identity,
                            conversation_id=excluded.conversation_id,
                            parent_id=excluded.parent_id,
                            turns=excluded.turns,
                            updated=excluded.updated
                        """,
                        (h, ref.account_identity, ref.conversation_id, ref.parent_id, ref.turns, ref.updated),
                    )

                # Prune old prefixes if table is large
                cur = conn.execute("SELECT COUNT(*) FROM prefixes")
                count = cur.fetchone()[0]
                if count > 20000:
                    cutoff = now - config.CONVERSATION_TTL_HOURS * 3600
                    conn.execute("DELETE FROM prefixes WHERE updated < ?", (cutoff,))

    # ---------- responses API ----------
    def put_response(self, rec: ResponseRecord, snapshot: TurnSnapshot | None = None) -> None:
        snap_blob: bytes | None = None
        snap_size: int = 0
        if snapshot is not None:
            trimmed = _trim_snapshot(snapshot, config.SNAPSHOT_FILE_CAP_MB << 20)
            rec.snapshot = trimmed
            snap_size = trimmed.payload_bytes()
            snap_blob = _serialize_snapshot(trimmed)

        with self._lock:
            with self._transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO responses (
                        response_id, account_identity, conversation_id, parent_id, model, created, snapshot_bytes, snapshot_json
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
                cur = conn.execute("SELECT COALESCE(SUM(snapshot_bytes), 0) FROM responses")
                total_snap_bytes = cur.fetchone()[0]
                if total_snap_bytes > budget:
                    # Drop snapshots from oldest records first
                    cur = conn.execute("SELECT response_id, snapshot_bytes FROM responses WHERE snapshot_bytes > 0 ORDER BY created ASC")
                    for row in cur.fetchall():
                        r_id, r_size = row[0], row[1]
                        conn.execute("UPDATE responses SET snapshot_bytes = 0, snapshot_json = NULL WHERE response_id = ?", (r_id,))
                        total_snap_bytes -= r_size
                        if total_snap_bytes <= budget:
                            break

                # Prune old responses count / TTL
                cur = conn.execute("SELECT COUNT(*) FROM responses")
                count = cur.fetchone()[0]
                if count > 5000:
                    cutoff = time.time() - config.CONVERSATION_TTL_HOURS * 3600
                    conn.execute("DELETE FROM responses WHERE created < ?", (cutoff,))

    def get_response(self, response_id: str) -> ResponseRecord | None:
        with self._lock:
            conn = self._get_conn()
            cur = conn.execute(
                "SELECT response_id, account_identity, conversation_id, parent_id, model, created, snapshot_json FROM responses WHERE response_id = ?",
                (response_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            snapshot = _deserialize_snapshot(row[6]) if row[6] is not None else None
            return ResponseRecord(
                response_id=row[0],
                account_identity=row[1],
                conversation_id=row[2],
                parent_id=row[3],
                model=row[4],
                created=row[5],
                snapshot=snapshot,
            )

    def get_snapshot(self, response_id: str) -> TurnSnapshot | None:
        with self._lock:
            conn = self._get_conn()
            cur = conn.execute(
                "SELECT snapshot_json FROM responses WHERE response_id = ?",
                (response_id,),
            )
            row = cur.fetchone()
            if row is None or row[0] is None:
                return None
            return _deserialize_snapshot(row[0])


STORE = ConversationStore()
