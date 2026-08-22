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
from dataclasses import dataclass
from typing import Any

from . import config


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
class ResponseRecord:
    response_id: str
    account_identity: str
    conversation_id: str
    parent_id: str  # parent for the NEXT turn (assistant msg id of this response)
    model: str
    created: float


class ConversationStore:
    def __init__(self):
        self._prefixes: dict[str, ConvRef] = {}
        self._responses: dict[str, ResponseRecord] = {}

    # ---------- chat-completions style ----------
    def find(self, hashes: list[str]) -> tuple[str, ConvRef] | None:
        """Longest prefix match. Returns (matched_len, ref)."""
        best: tuple[int, ConvRef] | None = None
        for k in range(len(hashes), 0, -1):
            ref = self._prefixes.get(hashes[k - 1])
            if ref is None:
                # guard against hash collisions across chains
                continue
            if best is None or k > best[0]:
                best = (k, ref)
            if k == len(hashes):
                break
        return best

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
    def put_response(self, rec: ResponseRecord) -> None:
        self._responses[rec.response_id] = rec
        if len(self._responses) > 5000:
            cutoff = time.time() - config.CONVERSATION_TTL_HOURS * 3600
            self._responses = {k: v for k, v in self._responses.items() if v.created >= cutoff}

    def get_response(self, response_id: str) -> ResponseRecord | None:
        return self._responses.get(response_id)


STORE = ConversationStore()
