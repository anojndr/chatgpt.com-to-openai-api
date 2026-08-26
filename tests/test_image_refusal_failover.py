"""Regression tests for the "image creation temporarily unavailable" refusal.

ChatGPT serves some accounts a plain-text refusal ("It looks like image
creation is temporarily unavailable. Do you want to try something else?")
as a normal HTTP-200 stream -- no metadata.is_error flag. The engine must
classify it as an image-limit refusal BEFORE any byte streams, so the
failover loop rotates through every remaining account; only when ALL of
them refuse may the failure surface to the client.
"""
import asyncio
import unittest

from app.adapters import HistoryItem, ParsedRequest
from app.engine import EngineError, run_turn
from app.accounts import AccountPool


class FakeAccount:
    """Minimal AccountSession stand-in replaying canned SSE events."""

    def __init__(self, identity, email, events, plan="free"):
        self.identity = identity
        self.email = email
        self.plan = plan
        self.inflight = 0
        self.total_requests = 0
        self.dead = False
        self.cooldown_until = 0.0
        self.last_used = 0.0
        self._events = events

    async def models(self):
        return [{"slug": "auto"}]

    async def stream_conversation(self, **kwargs):
        for ev in self._events:
            yield ev


def _user_then(events_tail, conv_id="conv-x"):
    head = [{
        "message": {"id": "u1", "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["q"]},
                    "metadata": {}},
        "conversation_id": conv_id,
    }]
    return head + events_tail


TEMP_REFUSAL = ("It looks like image creation is temporarily unavailable. "
                "Do you want to try something else?")

REFUSAL_EVENTS = _user_then([
    # Plain text node: no is_error flag, no "next" marker.
    {"message": {"id": "a1", "author": {"role": "assistant"},
                 "content": {"content_type": "text", "parts": [TEMP_REFUSAL]},
                 "metadata": {}}},
    {"type": "done"},
])

OK_EVENTS = _user_then([
    {"message": {"id": "n1", "author": {"role": "assistant"},
                 "content": {"content_type": "text",
                             "parts": ["Here is your image."]},
                 "metadata": {"message_type": "next"}}},
    {"type": "done"},
], conv_id="conv-ok")

# The phrase quoted MID-sentence in an otherwise normal reply must stream.
QUOTE_EVENTS = _user_then([
    {"message": {"id": "n2", "author": {"role": "assistant"},
                 "content": {"content_type": "text",
                             "parts": ["Support said: 'it looks like image "
                                       "creation is temporarily unavailable' "
                                       "-- but it works again now."]},
                 "metadata": {"message_type": "next"}}},
    {"type": "done"},
], conv_id="conv-quote")

# Variant wording (currently/temporarily, request(s), other nouns) arrives
# CHUNKED: early chunks diverge from any exact string before IMAGE_LIMIT_RE
# can confirm, so mid-stream withholding must rely on the generic starter.
VARIANT_REFUSAL = "It looks like image creation is currently unavailable."

CHUNKED_VARIANT_EVENTS = _user_then([
    {"message": {"id": "a1", "author": {"role": "assistant"},
                 "content": {"content_type": "text",
                             "parts": ["It looks like image creation is cu"]},
                 "metadata": {}}},
    {"message": {"id": "a1", "author": {"role": "assistant"},
                 "content": {"content_type": "text",
                             "parts": [VARIANT_REFUSAL]},
                 "metadata": {}}},
    {"type": "done"},
])

# A normal reply that genuinely BEGINS with "It looks like image ..." must
# still reach the client (noun gate proves divergence immediately).
NORMAL_IMAGE_PHRASE_EVENTS = _user_then([
    {"message": {"id": "n3", "author": {"role": "assistant"},
                 "content": {"content_type": "text",
                             "parts": ["It looks like image quality improved "
                                       "after the last update."]},
                 "metadata": {"message_type": "next"}}},
    {"type": "done"},
], conv_id="conv-norm")


class TestTemporaryUnavailableFailover(unittest.TestCase):
    @staticmethod
    def _parsed():
        return ParsedRequest(system_text="", items=[HistoryItem(role="user", text="q")],
                             model_requested="auto", stream=False)

    def test_refusal_fails_over_to_next_account(self):
        pool = AccountPool()
        free = FakeAccount("free", "free@example.com", REFUSAL_EVENTS)
        plus = FakeAccount("plus", "plus@example.com", OK_EVENTS, plan="plus")
        pool._accounts = {"free": free, "plus": plus}

        deltas: list[str] = []
        result = None

        async def run():
            nonlocal result
            async for ev in run_turn(self._parsed(), pool):
                if ev["type"] == "delta":
                    deltas.append(ev["text"])
                elif ev["type"] == "done":
                    result = ev["result"]

        asyncio.run(run())
        self.assertFalse(any("temporarily unavailable" in d.lower() for d in deltas),
                         "refusal bytes must never reach the client")
        self.assertIsNotNone(result)
        self.assertEqual(result.account_email, "plus@example.com")

    def test_all_accounts_tried_before_error_surfaces(self):
        pool = AccountPool()
        a = FakeAccount("a", "a@example.com", REFUSAL_EVENTS)
        b = FakeAccount("b", "b@example.com", REFUSAL_EVENTS)
        pool._accounts = {"a": a, "b": b}

        saw_delta = False

        async def run():
            nonlocal saw_delta
            async for ev in run_turn(self._parsed(), pool):
                if ev["type"] == "delta":
                    saw_delta = True

        with self.assertRaises(EngineError) as ctx:
            asyncio.run(run())
        # Both accounts attempted exactly once; nothing leaked.
        self.assertEqual((a.total_requests, b.total_requests), (1, 1))
        self.assertFalse(saw_delta)
        self.assertIn("all 2 available account(s) failed", str(ctx.exception))
        self.assertIn("temporarily unavailable", str(ctx.exception))

    def test_mid_text_quote_is_not_a_refusal(self):
        pool = AccountPool()
        pool._accounts = {"q": FakeAccount("q", "q@example.com", QUOTE_EVENTS)}
        deltas: list[str] = []

        async def run():
            async for ev in run_turn(self._parsed(), pool):
                if ev["type"] == "delta":
                    deltas.append(ev["text"])

        asyncio.run(run())
        self.assertIn("works again now", "".join(deltas))

    def test_variant_refusal_never_leaks_mid_stream(self):
        pool = AccountPool()
        free = FakeAccount("free", "free@example.com", CHUNKED_VARIANT_EVENTS)
        plus = FakeAccount("plus", "plus@example.com", OK_EVENTS, plan="plus")
        pool._accounts = {"free": free, "plus": plus}

        deltas: list[str] = []
        result = None

        async def run():
            nonlocal result
            async for ev in run_turn(self._parsed(), pool):
                if ev["type"] == "delta":
                    deltas.append(ev["text"])
                elif ev["type"] == "done":
                    result = ev["result"]

        asyncio.run(run())
        self.assertFalse(
            any("unavailable" in d.lower() or "it looks like" in d.lower()
                for d in deltas),
            f"variant refusal bytes leaked mid-stream: {deltas!r}")
        self.assertIsNotNone(result)
        self.assertEqual(result.account_email, "plus@example.com")

    def test_normal_reply_starting_with_phrase_streams_fully(self):
        pool = AccountPool()
        pool._accounts = {"n": FakeAccount("n", "n@example.com",
                                           NORMAL_IMAGE_PHRASE_EVENTS)}
        deltas: list[str] = []

        async def run():
            async for ev in run_turn(self._parsed(), pool):
                if ev["type"] == "delta":
                    deltas.append(ev["text"])

        asyncio.run(run())
        self.assertIn("quality improved", "".join(deltas))


if __name__ == "__main__":
    unittest.main()
