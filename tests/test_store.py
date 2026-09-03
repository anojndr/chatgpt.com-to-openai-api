# Copyright 2026 chatgpt-to-openai-api contributors.
"""SQLite conversation-store persistence tests."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import pytest
from typing_extensions import override

from app.adapters import FileInput, HistoryItem, ImageInput
from app.store import (
    ConversationStore,
    ConvRef,
    ResponseRecord,
    TurnSnapshot,
    item_hash,
)

FULL_PREFIX_MATCH = 3
PARTIAL_PREFIX_MATCH = 2
FIRST_TURN_MATCH = 1
SNAPSHOT_ITEM_COUNT = 2
LEGACY_ITEM_COUNT = 1

ACCOUNT_IDENTITY = "user_123"
CONVERSATION_ID = "conv_abc"
PARENT_ID = "msg_xyz"
BRANCH_ACCOUNT = "user_1"
BRANCH_CONVERSATION = "conv_1"
SECOND_TURN_PARENT = "msg_asst_2"
FIRST_TURN_PARENT = "msg_asst_1"

MODEL_NAME = "gpt-5-4"
SYSTEM_TEXT = "You are a helpful assistant"
LEGACY_SYSTEM_TEXT = "Legacy system text"
PNG_DATA = b"\x89PNG\r\n\x1a\n\x00test"
DOC_DATA = b"Document content here"
LEGACY_DATA = b"legacy_hex_bytes"
PNG_FILENAME = "test.png"
DOC_FILENAME = "doc.txt"

INSERT_RESPONSE_SQL = (
    "INSERT INTO responses (response_id, account_identity,"
    " conversation_id, parent_id, model, created, snapshot_bytes,"
    " snapshot_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
)
INSERT_PREFIX_SQL = (
    "INSERT INTO prefixes (hash, account_identity, conversation_id,"
    " parent_id, turns, updated) VALUES (?, ?, ?, ?, ?, ?)"
)


class _SimulatedTransactionError(RuntimeError):
    """Simulated mid-transaction failure for rollback tests."""

    def __init__(self) -> None:
        """Initialize with the fixed failure message."""
        super().__init__("Simulated transaction failure")


class TestConversationStoreSqlite(unittest.TestCase):
    """SQLite-backed conversation store tests."""

    @override
    def setUp(self) -> None:
        """Create an isolated store backed by a temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_store.db"
        self.store = ConversationStore(db_path=self.db_path)

    @override
    def tearDown(self) -> None:
        """Remove the temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_prefix_matching_and_persistence(self) -> None:
        """Verify prefix matching survives a store restart."""
        h1 = item_hash("", "user", "Hello")
        h2 = item_hash(h1, "assistant", "Hi there")
        h3 = item_hash(h2, "user", "How are you?")

        ref = ConvRef(
            account_identity=ACCOUNT_IDENTITY,
            conversation_id=CONVERSATION_ID,
            parent_id=PARENT_ID,
            turns=FULL_PREFIX_MATCH,
            updated=time.time(),
        )

        self.store.record_turn([h1, h2, h3], ref)

        # Query in original store
        matched = self.store.find([h1, h2, h3])
        assert matched is not None
        assert matched[0] == FULL_PREFIX_MATCH
        assert matched[1].conversation_id == CONVERSATION_ID
        assert matched[1].parent_id == PARENT_ID

        # Simulate restart with a new store on the same SQLite DB.
        store_restarted = ConversationStore(db_path=self.db_path)
        matched_after_restart = store_restarted.find([h1, h2, h3])
        assert matched_after_restart is not None
        assert matched_after_restart[0] == FULL_PREFIX_MATCH
        assert matched_after_restart[1].conversation_id == CONVERSATION_ID
        assert matched_after_restart[1].account_identity == ACCOUNT_IDENTITY
        assert matched_after_restart[1].parent_id == PARENT_ID

        # Prefix subset matching
        matched_prefix = store_restarted.find([h1, h2, "other_hash"])
        assert matched_prefix is not None
        assert matched_prefix[0] == PARTIAL_PREFIX_MATCH
        assert matched_prefix[1].conversation_id == CONVERSATION_ID

    def test_branching_conversation_parent_ids(self) -> None:
        """Verify forked histories resolve to the longest recorded prefix."""
        # Turn 1: user says "my name is tyler" (request items: [user_item])
        # engine computes hashes = [h1], records [h1] -> ref1
        h1 = item_hash("", "user", "my name is tyler")
        ref1 = ConvRef(
            account_identity=BRANCH_ACCOUNT,
            conversation_id=BRANCH_CONVERSATION,
            parent_id=FIRST_TURN_PARENT,
            turns=FIRST_TURN_MATCH,
            updated=time.time(),
        )
        self.store.record_turn([h1], ref1)

        # Turn 2: user says "what is my name again?"
        # engine computes hashes = [h1, h2, h3], records [h3] -> ref2
        h2 = item_hash(h1, "assistant", "Nice to meet you, Tyler.")
        h3 = item_hash(h2, "user", "what is my name again?")
        ref2 = ConvRef(
            account_identity=BRANCH_ACCOUNT,
            conversation_id=BRANCH_CONVERSATION,
            parent_id=SECOND_TURN_PARENT,
            turns=FULL_PREFIX_MATCH,
            updated=time.time(),
        )
        self.store.record_turn([h3], ref2)

        # Branch A (Turn 3): user says "remember code X"
        # engine computes hashes = [h1, h2, h3, h4, h5_a],
        # records [h5_a] -> ref3_a
        h4 = item_hash(h3, "assistant", "Your name is Tyler.")
        h5_a = item_hash(h4, "user", "remember code X")
        ref3_a = ConvRef(
            account_identity=BRANCH_ACCOUNT,
            conversation_id=BRANCH_CONVERSATION,
            parent_id="msg_asst_3_code",
            turns=5,
            updated=time.time(),
        )
        self.store.record_turn([h5_a], ref3_a)

        # Now, Branch B forks from Turn 2
        # where user3_b is "what code did I ask you to remember?"
        h5_b = item_hash(h4, "user", "what code did I ask you to remember?")
        matched = self.store.find([h1, h2, h3, h4, h5_b])
        assert matched is not None
        # Longest prefix match matches up to h3 and returns ref2.
        assert matched[0] == FULL_PREFIX_MATCH
        assert matched[1].parent_id == SECOND_TURN_PARENT
        assert matched[1].conversation_id == BRANCH_CONVERSATION

        # Also verify branching directly from Turn 1
        h3_c = item_hash(h2, "user", "different question branching from turn 1")
        matched_from_turn1 = self.store.find([h1, h2, h3_c])
        assert matched_from_turn1 is not None
        assert matched_from_turn1[0] == FIRST_TURN_MATCH
        assert matched_from_turn1[1].parent_id == FIRST_TURN_PARENT
        assert matched_from_turn1[1].conversation_id == BRANCH_CONVERSATION

    def test_response_and_snapshot_persistence(self) -> None:
        """Verify responses and snapshots persist across restarts."""
        img = ImageInput(
            filename=PNG_FILENAME,
            mime="image/png",
            data=PNG_DATA,
        )
        file_att = FileInput(
            filename=DOC_FILENAME,
            mime="text/plain",
            data=DOC_DATA,
        )
        item1 = HistoryItem(role="user", text="Look at this image", images=[img])
        item2 = HistoryItem(role="assistant", text="I see the image", files=[file_att])

        snap = TurnSnapshot(
            system_text=SYSTEM_TEXT,
            items=[item1, item2],
        )

        rec = ResponseRecord(
            response_id="resp_12345",
            account_identity=ACCOUNT_IDENTITY,
            conversation_id=CONVERSATION_ID,
            parent_id=PARENT_ID,
            model=MODEL_NAME,
            created=time.time(),
        )

        self.store.put_response(rec, snap)

        # Verify before restart
        got_rec = self.store.get_response("resp_12345")
        assert got_rec is not None
        assert got_rec.model == MODEL_NAME
        assert got_rec.conversation_id == CONVERSATION_ID
        got_snap = self.store.get_snapshot("resp_12345")
        assert got_snap is not None
        assert got_snap.system_text == SYSTEM_TEXT
        assert len(got_snap.items) == SNAPSHOT_ITEM_COUNT
        assert got_snap.items[0].images[0].data == PNG_DATA
        assert got_snap.items[1].files[0].data == DOC_DATA

        # Verify after restart
        store_restarted = ConversationStore(db_path=self.db_path)
        got_rec2 = store_restarted.get_response("resp_12345")
        assert got_rec2 is not None
        assert got_rec2.model == MODEL_NAME
        assert got_rec2.account_identity == ACCOUNT_IDENTITY
        assert got_rec2.parent_id == PARENT_ID

        got_snap2 = store_restarted.get_snapshot("resp_12345")
        assert got_snap2 is not None
        assert got_snap2.system_text == SYSTEM_TEXT
        assert len(got_snap2.items) == SNAPSHOT_ITEM_COUNT
        assert got_snap2.items[0].images[0].filename == PNG_FILENAME
        assert got_snap2.items[0].images[0].data == PNG_DATA
        assert got_snap2.items[1].files[0].filename == DOC_FILENAME
        assert got_snap2.items[1].files[0].data == DOC_DATA

    def test_legacy_hex_snapshot_compatibility(self) -> None:
        """Verify snapshots stored in legacy hex format still decode."""
        legacy_data = {
            "system_text": LEGACY_SYSTEM_TEXT,
            "items": [
                {
                    "role": "user",
                    "text": "Legacy text",
                    "images": [
                        {
                            "filename": "hex.png",
                            "mime": "image/png",
                            "data": LEGACY_DATA.hex(),
                        },
                    ],
                    "files": [],
                },
            ],
        }
        raw_json = json.dumps(legacy_data).encode("utf-8")
        with self.store.transaction() as conn:
            conn.execute(
                INSERT_RESPONSE_SQL,
                (
                    "resp_legacy",
                    "acc_1",
                    "conv_1",
                    "msg_1",
                    "auto",
                    time.time(),
                    len(LEGACY_DATA),
                    raw_json,
                ),
            )

        snap = self.store.get_snapshot("resp_legacy")
        assert snap is not None
        assert snap.system_text == LEGACY_SYSTEM_TEXT
        assert len(snap.items) == LEGACY_ITEM_COUNT
        assert snap.items[0].images[0].data == LEGACY_DATA

    def _insert_prefix_then_fail(self, entry_hash: str, ref: ConvRef) -> None:
        """Insert one prefix row then raise to trigger a rollback."""
        with self.store.transaction() as conn:
            conn.execute(
                INSERT_PREFIX_SQL,
                (
                    entry_hash,
                    ref.account_identity,
                    ref.conversation_id,
                    ref.parent_id,
                    ref.turns,
                    ref.updated,
                ),
            )
            raise _SimulatedTransactionError

    def test_transaction_rollback_on_failure(self) -> None:
        """Verify a mid-transaction failure rolls back partial writes."""
        h1 = item_hash("", "user", "Rollback test")
        ref = ConvRef(
            account_identity="user_rb",
            conversation_id="conv_rb",
            parent_id="msg_rb",
            turns=FIRST_TURN_MATCH,
            updated=time.time(),
        )

        # Force a failure inside transaction
        with pytest.raises(_SimulatedTransactionError):
            self._insert_prefix_then_fail(h1, ref)

        # Verify that h1 was rolled back and does not exist in prefixes
        assert self.store.find([h1]) is None


if __name__ == "__main__":
    unittest.main()
