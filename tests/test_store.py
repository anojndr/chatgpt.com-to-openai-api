import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from app.adapters import FileInput, HistoryItem, ImageInput
from app.store import (
    ConvRef,
    ConversationStore,
    ResponseRecord,
    TurnSnapshot,
    canon_content,
    item_hash,
)


class TestConversationStoreSqlite(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_store.db"
        self.store = ConversationStore(db_path=self.db_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_prefix_matching_and_persistence(self):
        h1 = item_hash("", "user", "Hello")
        h2 = item_hash(h1, "assistant", "Hi there")
        h3 = item_hash(h2, "user", "How are you?")

        ref = ConvRef(
            account_identity="user_123",
            conversation_id="conv_abc",
            parent_id="msg_xyz",
            turns=3,
            updated=time.time(),
        )

        self.store.record_turn([h1, h2, h3], ref)

        # Query in original store
        matched = self.store.find([h1, h2, h3])
        self.assertIsNotNone(matched)
        self.assertEqual(matched[0], 3)
        self.assertEqual(matched[1].conversation_id, "conv_abc")
        self.assertEqual(matched[1].parent_id, "msg_xyz")

        # Simulate restart by instantiating a new ConversationStore pointing to the same SQLite DB
        store_restarted = ConversationStore(db_path=self.db_path)
        matched_after_restart = store_restarted.find([h1, h2, h3])
        self.assertIsNotNone(matched_after_restart)
        self.assertEqual(matched_after_restart[0], 3)
        self.assertEqual(matched_after_restart[1].conversation_id, "conv_abc")
        self.assertEqual(matched_after_restart[1].account_identity, "user_123")
        self.assertEqual(matched_after_restart[1].parent_id, "msg_xyz")

        # Prefix subset matching
        matched_prefix = store_restarted.find([h1, h2, "other_hash"])
        self.assertIsNotNone(matched_prefix)
        self.assertEqual(matched_prefix[0], 2)
        self.assertEqual(matched_prefix[1].conversation_id, "conv_abc")

    def test_response_and_snapshot_persistence(self):
        img = ImageInput(filename="test.png", mime="image/png", data=b"\x89PNG\r\n\x1a\n\x00test")
        file_att = FileInput(filename="doc.txt", mime="text/plain", data=b"Document content here")
        item1 = HistoryItem(role="user", text="Look at this image", images=[img])
        item2 = HistoryItem(role="assistant", text="I see the image", files=[file_att])

        snap = TurnSnapshot(
            system_text="You are a helpful assistant",
            items=[item1, item2],
        )

        rec = ResponseRecord(
            response_id="resp_12345",
            account_identity="user_123",
            conversation_id="conv_abc",
            parent_id="msg_xyz",
            model="gpt-5-4",
            created=time.time(),
        )

        self.store.put_response(rec, snap)

        # Verify before restart
        got_rec = self.store.get_response("resp_12345")
        self.assertIsNotNone(got_rec)
        self.assertEqual(got_rec.model, "gpt-5-4")
        self.assertEqual(got_rec.conversation_id, "conv_abc")
        got_snap = self.store.get_snapshot("resp_12345")
        self.assertIsNotNone(got_snap)
        self.assertEqual(got_snap.system_text, "You are a helpful assistant")
        self.assertEqual(len(got_snap.items), 2)
        self.assertEqual(got_snap.items[0].images[0].data, b"\x89PNG\r\n\x1a\n\x00test")
        self.assertEqual(got_snap.items[1].files[0].data, b"Document content here")

        # Verify after restart
        store_restarted = ConversationStore(db_path=self.db_path)
        got_rec2 = store_restarted.get_response("resp_12345")
        self.assertIsNotNone(got_rec2)
        self.assertEqual(got_rec2.model, "gpt-5-4")
        self.assertEqual(got_rec2.account_identity, "user_123")
        self.assertEqual(got_rec2.parent_id, "msg_xyz")

        got_snap2 = store_restarted.get_snapshot("resp_12345")
        self.assertIsNotNone(got_snap2)
        self.assertEqual(got_snap2.system_text, "You are a helpful assistant")
        self.assertEqual(len(got_snap2.items), 2)
        self.assertEqual(got_snap2.items[0].images[0].filename, "test.png")
        self.assertEqual(got_snap2.items[0].images[0].data, b"\x89PNG\r\n\x1a\n\x00test")
        self.assertEqual(got_snap2.items[1].files[0].filename, "doc.txt")
        self.assertEqual(got_snap2.items[1].files[0].data, b"Document content here")

    def test_legacy_hex_snapshot_compatibility(self):
        # Insert a snapshot using legacy hex representation
        import json
        legacy_data = {
            "system_text": "Legacy system text",
            "items": [
                {
                    "role": "user",
                    "text": "Legacy text",
                    "images": [{"filename": "hex.png", "mime": "image/png", "data": b"legacy_hex_bytes".hex()}],
                    "files": [],
                }
            ],
        }
        raw_json = json.dumps(legacy_data).encode("utf-8")
        conn = self.store._get_conn()
        with self.store._transaction() as c:
            c.execute(
                """
                INSERT INTO responses (response_id, account_identity, conversation_id, parent_id, model, created, snapshot_bytes, snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("resp_legacy", "acc_1", "conv_1", "msg_1", "auto", time.time(), len(b"legacy_hex_bytes"), raw_json),
            )

        snap = self.store.get_snapshot("resp_legacy")
        self.assertIsNotNone(snap)
        self.assertEqual(snap.system_text, "Legacy system text")
        self.assertEqual(len(snap.items), 1)
        self.assertEqual(snap.items[0].images[0].data, b"legacy_hex_bytes")

    def test_transaction_rollback_on_failure(self):
        h1 = item_hash("", "user", "Rollback test")
        ref = ConvRef(
            account_identity="user_rb",
            conversation_id="conv_rb",
            parent_id="msg_rb",
            turns=1,
            updated=time.time(),
        )

        # Force a failure inside transaction
        try:
            with self.store._lock:
                with self.store._transaction() as conn:
                    conn.execute(
                        "INSERT INTO prefixes (hash, account_identity, conversation_id, parent_id, turns, updated) VALUES (?, ?, ?, ?, ?, ?)",
                        (h1, ref.account_identity, ref.conversation_id, ref.parent_id, ref.turns, ref.updated),
                    )
                    # Trigger an error
                    raise RuntimeError("Simulated transaction failure")
        except RuntimeError:
            pass

        # Verify that h1 was rolled back and does not exist in prefixes
        matched = self.store.find([h1])
        self.assertIsNone(matched)


if __name__ == "__main__":
    unittest.main()
