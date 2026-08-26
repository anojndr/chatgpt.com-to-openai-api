"""Regression tests for upstream moderation-error nodes (metadata.is_error).

ChatGPT delivers prompt-policy refusals as HTTP-200 SSE streams whose final
assistant text node carries metadata.is_error=true and no "next" marker. The
engine must raise BEFORE streaming any of that text, so the failover loop can
retry on another account (moderation verdicts are model/plan-tier dependent).
"""
import asyncio
import unittest

from app.adapters import HistoryItem, ParsedRequest
from app.chatgpt import ChatGPTError
from app.engine import run_turn
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


REFUSAL = ("We\u2019re so sorry, but the prompt may violate our content policies. "
           "If you think we got it wrong, please retry or edit your prompt.")

INCIDENT_EVENTS = _user_then([
    # code node: assistant role, non-text content type, parts=None
    {"message": {"id": "a1", "author": {"role": "assistant"},
                 "content": {"content_type": "code", "parts": None},
                 "metadata": {}}},
    {"message": {"id": "a2", "author": {"role": "assistant"},
                 "content": {"content_type": "text", "parts": [REFUSAL]},
                 "metadata": {"is_error": True}}},
    {"type": "done"},
])

OK_EVENTS = _user_then([
    {"message": {"id": "n1", "author": {"role": "assistant"},
                 "content": {"content_type": "text",
                             "parts": ["Room-clearing diagram."]},
                 "metadata": {"message_type": "next"}}},
    {"type": "done"},
], conv_id="conv-ok")


class TestUpstreamErrorNodes(unittest.TestCase):
    @staticmethod
    def _parsed():
        return ParsedRequest(system_text="", items=[HistoryItem(role="user", text="q")],
                             model_requested="auto", stream=False)

    def test_refusal_never_streams_and_fails_over(self):
        pool = AccountPool()
        free = FakeAccount("free", "free@example.com", INCIDENT_EVENTS)
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
        self.assertFalse(any("sorry" in d for d in deltas),
                         "refusal bytes must never reach the client")
        self.assertIsNotNone(result)
        self.assertEqual(result.account_email, "plus@example.com")

    def test_single_account_raises_before_streaming(self):
        pool = AccountPool()
        pool._accounts = {"free": FakeAccount("free", "f@example.com", INCIDENT_EVENTS)}

        saw_delta = False

        async def run():
            nonlocal saw_delta
            async for ev in run_turn(self._parsed(), pool):
                if ev["type"] == "delta":
                    saw_delta = True

        with self.assertRaises(Exception) as ctx:
            asyncio.run(run())
        self.assertFalse(saw_delta)
        self.assertIn("rejected this prompt", str(ctx.exception))

    def test_normal_turn_unaffected(self):
        pool = AccountPool()
        pool._accounts = {"a": FakeAccount("a", "a@example.com", OK_EVENTS)}
        deltas: list[str] = []

        async def run():
            async for ev in run_turn(self._parsed(), pool):
                if ev["type"] == "delta":
                    deltas.append(ev["text"])

        asyncio.run(run())
        self.assertEqual("".join(deltas), "Room-clearing diagram.")

    def test_incomplete_stream_without_is_error_still_flags_502(self):
        events = _user_then([{"type": "done"}])
        pool = AccountPool()
        pool._accounts = {"b": FakeAccount("b", "b@example.com", events)}

        async def run():
            async for _ in run_turn(self._parsed(), pool):
                pass

        with self.assertRaises(Exception) as ctx:
            asyncio.run(run())
        self.assertIn("incomplete response", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
