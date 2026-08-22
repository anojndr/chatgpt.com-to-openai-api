"""Multi-turn registry: maps client-side histories to real ChatGPT conversations.

A conversation is identified by a rolling hash chain over normalized history
items [(role, canonical_text)]. Given a client request with the full history
(the usual Chat Completions pattern), the longest previously-recorded prefix
match lets us continue the real ChatGPT conversation by sending ONLY the new
trailing messages instead of re-sending everything.
"""
from __future__ import annotations

import hashlib
import time
from copy import copy
from dataclasses import dataclass
from typing import Any

from . import config
from .adapters import HistoryItem


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


class ConversationStore:
    def __init__(self):
        self._prefixes: dict[str, ConvRef] = {}
        self._responses: dict[str, ResponseRecord] = {}
        self._snap_bytes: int = 0

    # ---------- chat-completions style ----------
    def find(self, hashes: list[str]) -> tuple[int, ConvRef] | None:
        """Longest prefix match. Returns (matched_len, ref)."""
        for k in range(len(hashes), 0, -1):
            ref = self._prefixes.get(hashes[k - 1])
            if ref is not None:
                return (k, ref)  # first hit scanning from longest = best match
        return None

    def record_turn(self, hashes: list[str], ref: ConvRef) -> None:
        """Store hash chain entries covering the whole updated history."""
        now = time.time()
        ref.updated = now
        for h in hashes:
            self._prefixes[h] = ref
        if len(self._prefixes) > 20000:
            cutoff = now - config.CONVERSATION_TTL_HOURS * 3600
            self._prefixes = {h: r for h, r in self._prefixes.items() if r.updated >= cutoff}

    # ---------- responses API ----------
    def put_response(self, rec: ResponseRecord, snapshot: TurnSnapshot | None = None) -> None:
        if snapshot is not None:
            rec.snapshot = _trim_snapshot(snapshot, config.SNAPSHOT_FILE_CAP_MB << 20)
            self._snap_bytes += rec.snapshot.payload_bytes()
        self._responses[rec.response_id] = rec
        budget = config.SNAPSHOT_STORE_CAP_MB << 20
        if self._snap_bytes > budget:
            for old in self._responses.values():  # insertion order: oldest first
                if old.snapshot is None:
                    continue
                self._snap_bytes -= old.snapshot.payload_bytes()
                old.snapshot = None
                if self._snap_bytes <= budget:
                    break
        if len(self._responses) > 5000:
            cutoff = time.time() - config.CONVERSATION_TTL_HOURS * 3600
            stale = {k: v for k, v in self._responses.items() if v.created < cutoff}
            for old in stale.values():
                if old.snapshot is not None:
                    self._snap_bytes -= old.snapshot.payload_bytes()
            self._responses = {k: v for k, v in self._responses.items() if v.created >= cutoff}

    def get_response(self, response_id: str) -> ResponseRecord | None:
        return self._responses.get(response_id)

    def get_snapshot(self, response_id: str) -> TurnSnapshot | None:
        rec = self._responses.get(response_id)
        return rec.snapshot if rec else None


STORE = ConversationStore()
