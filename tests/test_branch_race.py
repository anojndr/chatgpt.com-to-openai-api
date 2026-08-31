"""Regression tests for concurrent-generation branch races and JSX citations.

Root cause (2026-08-31, account rjandrongian@gmail.com / gpt-5-6, conversation
6a94e940-9e0c-83ec-9856-de304ad81252): upstream served one turn as TWO
parallel generation branches -- interleaved SSE nodes, each re-sending its own
cumulative snapshot. run_turn restarted delta accounting on every new assistant
node id, so each branch switch re-emitted the rival's snapshot from byte 0;
the client received ~23 overlapping copies of two answer variants. The same
stream variant serializes citations as JSX (<Cite refs={["turn0news9"]}/>)
instead of \\ue200cite blocks, and those tags leaked raw to the client.
"""
import asyncio
import unittest

from app.adapters import HistoryItem, ParsedRequest
from app.engine import _jsx_cite_cut, _render_citations, run_turn
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


def _msg(mid, role, ctype, parts, mtype=None):
    md = {"message_type": mtype} if mtype else {}
    return {"message": {"id": mid, "author": {"role": role},
                        "content": {"content_type": ctype, "parts": parts},
                        "metadata": md},
            "conversation_id": "conv-race"}


def _user_then(events_tail):
    return [{
        "message": {"id": "u1", "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["q"]},
                    "metadata": {}},
        "conversation_id": "conv-race",
    }] + events_tail


A1 = ("Alpha text one. The answer continues past the refusal window so the "
      "emitter actually streams bytes mid-turn rather than only flushing at "
      "the end of the stream. More prose follows here to make the snapshot "
      "long enough to clear every classifier gate.")
B1 = "Beta text one. The rival branch opens with different wording entirely."
A2 = A1 + " Alpha grows. <Cite ref={[\"turn0search1"  # half-received JSX tag
B2 = B1 + " Beta grows further with its own second snapshot."
A3 = (A1 + " Alpha grows. <Cite refs={[\"turn0news9\",\"turn0search10\"]}/>"
      " Alpha tail after the citation tag.")

RACE_EVENTS = _user_then([
    _msg("ca", "assistant", "code", None),      # branch A chain
    _msg("ta", "assistant", "text", [A1]),
    _msg("cb", "assistant", "code", None),      # branch B chain
    _msg("tb", "assistant", "text", [B1]),      # rival snapshot arrives
    _msg("ta", "assistant", "text", [A2]),      # adopted branch, mid-tag
    _msg("tb", "assistant", "text", [B2]),      # rival grows
    _msg("ta", "assistant", "text", [A3], mtype="next"),
    _msg("tb", "assistant", "text", [B2], mtype="next"),  # rival completes after
    {"type": "done"},
])


class TestConcurrentGenerationBranchRace(unittest.TestCase):
    @staticmethod
    def _parsed():
        return ParsedRequest(system_text="", items=[HistoryItem(role="user", text="q")],
                             model_requested="auto", stream=False)

    def test_interleaved_generations_stream_one_answer(self):
        pool = AccountPool()
        pool._accounts = {"a": FakeAccount("a", "a@example.com", RACE_EVENTS)}

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
        joined = "".join(deltas)
        self.assertEqual(joined.count("Alpha text one."), 1,
                         "each branch switch must not re-emit the streamed "
                         "branch's snapshot from byte 0")
        self.assertNotIn("Beta", joined,
                         "rival-branch text must never reach the client")
        self.assertNotIn("<Cite", joined,
                         "JSX citation tags must be rendered, not leaked")
        self.assertIsNotNone(result)
        self.assertEqual(result.text, joined,
                         "stored turn text and streamed bytes must agree")
        self.assertEqual(result.parent_id, "ta",
                         "a rival branch's completion marker must not repoint "
                         "the conversation parent away from the streamed branch")

    def test_idless_streamed_branch_without_completion_fails_over(self):
        # A text node without an id streams (emission gate's not-m["id"] arm,
        # current_msg_id stays ""); if its stream never completes and the
        # only marker belongs to a rival branch, the turn is incomplete: the
        # client keeps what it received and the pool may fail over, instead
        # of silently continuing the rival branch server-side.
        events = _user_then([
            _msg("ca", "assistant", "code", None),
            {"message": {"author": {"role": "assistant"},
                         "content": {"content_type": "text", "parts": [A1]}},
             "conversation_id": "conv-race"},
            _msg("tb", "assistant", "text", [B2], mtype="next"),
            {"type": "done"},
        ])
        pool = AccountPool()
        pool._accounts = {"a": FakeAccount("a", "a@example.com", events)}

        deltas: list[str] = []

        async def run():
            async for ev in run_turn(self._parsed(), pool):
                if ev["type"] == "delta":
                    deltas.append(ev["text"])

        with self.assertRaises(Exception) as ctx:
            asyncio.run(run())
        self.assertIn("incomplete response", str(ctx.exception))
        self.assertIn("Alpha text one.", "".join(deltas),
                      "streamed bytes remain delivered; the rival branch "
                      "must not silently take over the answer")


class TestJsxCitations(unittest.TestCase):
    def test_unresolved_tag_holds_then_drops_at_final(self):
        raw = 'Answer. <Cite refs={["turn0news9","turn0search10"]}/> Tail.'
        text, safe = _render_citations(raw, {})
        self.assertEqual(safe, len("Answer. "),
                         "unresolved tokens must hold the stream at the tag")
        self.assertFalse(text[:safe].endswith("<Cite"))
        final, _ = _render_citations(raw, {}, final=True)
        self.assertNotIn("<Cite", final)
        self.assertIn("Answer.", final)
        self.assertIn("Tail.", final)

    def test_half_received_tag_withheld_mid_stream(self):
        raw = 'Answer. <Cite ref={["turn0search1'
        text, safe = _render_citations(raw, {})
        self.assertEqual(text[:safe], "Answer. ")
        self.assertEqual(_jsx_cite_cut(raw), len("Answer. "))
        final, _ = _render_citations(raw, {}, final=True)
        self.assertEqual(final, "Answer. ")

    def test_resolved_tokens_render_links(self):
        cmap = {(0, "search", 1): {"title": "NIDCD", "url": "https://x.gov/a"}}
        raw = 'A. <Cite refs={["turn0search1"]}/> B.'
        text, safe = _render_citations(raw, cmap)
        self.assertIn("[NIDCD](https://x.gov/a)", text)
        self.assertEqual(safe, len(text))

    def test_token_free_tag_is_code_not_citation(self):
        raw = "return <Cite Foo bar/> from the component"
        text, _ = _render_citations(raw, {}, final=True)
        self.assertEqual(text, raw)

    def test_long_token_free_fragment_withheld_then_kept(self):
        raw = "x = <Cite" + " a" * 80  # token-free, >120 chars, unclosed
        text, safe = _render_citations(raw, {})
        self.assertEqual(text[:safe], "x = ",
                         "any unclosed single-line fragment must be withheld "
                         "mid-stream: it could still complete into a tag")
        final, _ = _render_citations(raw, {}, final=True)
        self.assertEqual(final, raw,
                         "a token-free >120-char fragment is code content and "
                         "must survive the final flush")

    def test_multiline_cite_mention_is_not_treated_as_tag(self):
        raw = "see <Cite\n  for details"
        self.assertEqual(_jsx_cite_cut(raw), -1)
        text, _ = _render_citations(raw, {}, final=True)
        self.assertIn("<Cite", text)


if __name__ == "__main__":
    unittest.main()
