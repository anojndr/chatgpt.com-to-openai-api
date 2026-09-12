# Copyright 2026 chatgpt-to-openai-api contributors.
"""Regression tests for concurrent-generation branch races and JSX citations.

Root cause (2026-08-31, account rjandrongian@gmail.com / gpt-5-6,
conversation 6a94e940-9e0c-83ec-9856-de304ad81252): upstream served one turn
as TWO parallel generation branches -- interleaved SSE nodes, each
re-sending its own cumulative snapshot. run_turn restarted delta accounting
on every new assistant node id, so each branch switch re-emitted the rival's
snapshot from byte 0; the client received ~23 overlapping copies of two
answer variants. The same stream variant serializes citations as JSX
(<Cite refs={["turn0news9"]}/>) instead of private-use cite blocks, and those
tags leaked raw to the client.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias
from unittest.mock import patch

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

import pytest
from typing_extensions import override

from app.accounts import AccountPool
from app.adapters import HistoryItem, ParsedRequest
from app.chatgpt import AccountSession
from app.engine import (
    EngineError,
    TurnResult,
    jsx_cite_cut,
    render_citations,
    run_turn,
)
from app.store import ConversationStore

SSEEvent: TypeAlias = dict[str, object]

CONVERSATION_ID = "conv-race"
EXPECTED_SINGLE_DELIVERY = 1
EXPECTED_TWO_PROMPTS = 2
STREAMED_BRANCH_ID = "ta"
NO_WITHHOLD = -1


class FakeAccount(AccountSession):
    """AccountSession stand-in replaying canned SSE events."""

    def __init__(
        self,
        identity: str,
        email: str,
        events: list[SSEEvent],
        plan: str = "free",
    ) -> None:
        """Create a replay account serving the given canned events."""
        super().__init__({
            "identity": identity,
            "session": {
                "accessToken": "",
                "account": {"planType": plan},
                "user": {"email": email},
            },
            "cookies": {},
        })
        self._events = events

    @override
    async def models(self) -> list[dict[str, str]]:
        """Return the canned single-model listing.

        Returns:
            The single-entry model list with slug ``auto``.

        """
        return [{"slug": "auto"}]

    @override
    async def stream_conversation(self, **_kwargs: object) -> AsyncIterator[SSEEvent]:
        """Replay the canned SSE events.

        Yields:
            Each canned SSE event in order.

        """
        for event in self._events:
            yield event


def _make_request() -> ParsedRequest:
    """Build the single-turn user request shared by every test here.

    Returns:
        A single-turn user request for ``auto`` with streaming disabled.

    """
    return ParsedRequest(
        system_text="",
        items=[HistoryItem(role="user", text="q")],
        model_requested="auto",
        stream=False,
    )


async def _stream_turn(pool: AccountPool) -> tuple[list[str], TurnResult | None]:
    """Run one turn, returning streamed bytes and the completion result.

    Returns:
        A tuple of streamed delta texts and the completion result, if any.

    """
    deltas: list[str] = []
    result: TurnResult | None = None
    async for event in run_turn(_make_request(), pool):
        if event["type"] == "delta":
            text = event["text"]
            if isinstance(text, str):
                deltas.append(text)
        elif event["type"] == "done":
            outcome = event["result"]
            if isinstance(outcome, TurnResult):
                result = outcome
    return deltas, result


async def _drain_partial(pool: AccountPool) -> tuple[list[str], EngineError]:
    """Drain a failing turn, returning partial bytes and the failure.

    Returns:
        A tuple of partial delta texts and the engine failure.

    """
    deltas: list[str] = []
    try:
        async for event in run_turn(_make_request(), pool):
            if event["type"] == "delta":
                text = event["text"]
                if isinstance(text, str):
                    deltas.append(text)
    except EngineError as exc:
        return deltas, exc
    failure = "expected the turn to fail with an incomplete-response error"
    pytest.fail(failure)


def _msg(
    mid: str, role: str, ctype: str, parts: list[str] | None, mtype: str | None = None
) -> SSEEvent:
    """Build one SSE node for the canned race stream.

    Returns:
        One SSE event dict carrying the given message node.

    """
    md: SSEEvent = {"message_type": mtype} if mtype else {}
    return {
        "message": {
            "id": mid,
            "author": {"role": role},
            "content": {"content_type": ctype, "parts": parts},
            "metadata": md,
        },
        "conversation_id": CONVERSATION_ID,
    }


def _user_then(events_tail: list[SSEEvent]) -> list[SSEEvent]:
    """Prepend the shared user turn to a canned assistant event tail.

    Returns:
        The user turn followed by the given assistant events.

    """
    return [
        {
            "message": {
                "id": "u1",
                "author": {"role": "user"},
                "content": {"content_type": "text", "parts": ["q"]},
                "metadata": {},
            },
            "conversation_id": CONVERSATION_ID,
        },
        *events_tail,
    ]


A1 = (
    "Alpha text one. The answer continues past the refusal window so the "
    "emitter actually streams bytes mid-turn rather than only flushing at "
    "the end of the stream. More prose follows here to make the snapshot "
    "long enough to clear every classifier gate."
)
B1 = "Beta text one. The rival branch opens with different wording entirely."
A2 = A1 + ' Alpha grows. <Cite ref={["turn0search1'  # half-received JSX tag
B2 = B1 + " Beta grows further with its own second snapshot."
A3 = (
    A1 + ' Alpha grows. <Cite refs={["turn0news9","turn0search10"]}/>'
    " Alpha tail after the citation tag."
)

RACE_EVENTS = _user_then(
    [
        _msg("ca", "assistant", "code", None),  # branch A chain
        _msg("ta", "assistant", "text", [A1]),
        _msg("cb", "assistant", "code", None),  # branch B chain
        _msg("tb", "assistant", "text", [B1]),  # rival snapshot arrives
        _msg("ta", "assistant", "text", [A2]),  # adopted branch, mid-tag
        _msg("tb", "assistant", "text", [B2]),  # rival grows
        _msg("ta", "assistant", "text", [A3], mtype="next"),
        _msg("tb", "assistant", "text", [B2], mtype="next"),  # rival completes after
        {"type": "done"},
    ],
)


class TestConcurrentGenerationBranchRace(unittest.TestCase):
    """Regression tests for interleaved generation branches in one turn."""

    @staticmethod
    def test_interleaved_generations_stream_one_answer() -> None:
        """Stream one answer when upstream interleaves rival branches."""
        pool = AccountPool()
        pool.register(FakeAccount("a", "a@example.com", RACE_EVENTS))

        deltas, result = asyncio.run(_stream_turn(pool))
        joined = "".join(deltas)
        assert joined.count("Alpha text one.") == EXPECTED_SINGLE_DELIVERY, (
            "each branch switch must not re-emit the streamed "
            "branch's snapshot from byte 0"
        )
        assert "Beta" not in joined, "rival-branch text must never reach the client"
        assert "<Cite" not in joined, "JSX citation tags must be rendered, not leaked"
        assert result is not None
        assert result.text == joined, "stored turn text and streamed bytes must agree"
        assert result.parent_id == STREAMED_BRANCH_ID, (
            "a rival branch's completion marker must not repoint "
            "the conversation parent away from the streamed branch"
        )

    @staticmethod
    def test_idless_streamed_branch_without_completion_fails_over() -> None:
        """Fail over when the streamed branch never completes.

        A text node without an id streams (emission gate's not-m["id"] arm,
        current_msg_id stays ""); if its stream never completes and the
        only marker belongs to a rival branch, the turn is incomplete: the
        client keeps what it received and the pool may fail over, instead
        of silently continuing the rival branch server-side.
        """
        events = _user_then(
            [
                _msg("ca", "assistant", "code", None),
                {
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"content_type": "text", "parts": [A1]},
                    },
                    "conversation_id": CONVERSATION_ID,
                },
                _msg("tb", "assistant", "text", [B2], mtype="next"),
                {"type": "done"},
            ],
        )
        pool = AccountPool()
        pool.register(FakeAccount("a", "a@example.com", events))

        deltas, exc = asyncio.run(_drain_partial(pool))
        assert "incomplete response" in exc.message
        assert "Alpha text one." in "".join(deltas), (
            "streamed bytes remain delivered; the rival branch "
            "must not silently take over the answer"
        )


class TestJsxCitations(unittest.TestCase):
    """Unit tests for JSX citation rendering and stream withholding."""

    @staticmethod
    def test_unresolved_tag_holds_then_drops_at_final() -> None:
        """Hold unresolved tags mid-stream, then drop them at final."""
        raw = 'Answer. <Cite refs={["turn0news9","turn0search10"]}/> Tail.'
        text, safe = render_citations(raw, {})
        assert safe == len("Answer. "), "unresolved tokens must hold the stream"
        assert not text[:safe].endswith("<Cite")
        final, _ = render_citations(raw, {}, final=True)
        assert "<Cite" not in final
        assert "Answer." in final
        assert "Tail." in final

    @staticmethod
    def test_half_received_tag_withheld_mid_stream() -> None:
        """Withhold a half-received tag mid-stream, then drop it."""
        raw = 'Answer. <Cite ref={["turn0search1'
        text, safe = render_citations(raw, {})
        assert text[:safe] == "Answer. "
        assert jsx_cite_cut(raw) == len("Answer. ")
        final, _ = render_citations(raw, {}, final=True)
        assert final == "Answer. "

    @staticmethod
    def test_resolved_tokens_render_links() -> None:
        """Render resolved citation tokens as markdown links."""
        cmap = {(0, "search", 1): {"title": "NIDCD", "url": "https://x.gov/a"}}
        raw = 'A. <Cite refs={["turn0search1"]}/> B.'
        text, safe = render_citations(raw, cmap)
        assert "[NIDCD](https://x.gov/a)" in text
        assert safe == len(text)

    @staticmethod
    def test_token_free_tag_is_code_not_citation() -> None:
        """Keep token-free tags verbatim as code, not citations."""
        raw = "return <Cite Foo bar/> from the component"
        text, _ = render_citations(raw, {}, final=True)
        assert text == raw

    @staticmethod
    def test_long_token_free_fragment_withheld_then_kept() -> None:
        """Withhold a long unclosed fragment mid-stream, keep it at final."""
        raw = "x = <Cite" + " a" * 80  # token-free, >120 chars, unclosed
        text, safe = render_citations(raw, {})
        assert text[:safe] == "x = ", (
            "any unclosed single-line fragment must be withheld "
            "mid-stream: it could still complete into a tag"
        )
        final, _ = render_citations(raw, {}, final=True)
        assert final == raw, (
            "a token-free >120-char fragment is code content and "
            "must survive the final flush"
        )

    @staticmethod
    def test_multiline_cite_mention_is_not_treated_as_tag() -> None:
        """Treat a multiline cite mention as prose, not a tag."""
        raw = "see <Cite\n  for details"
        assert jsx_cite_cut(raw) == NO_WITHHOLD
        text, _ = render_citations(raw, {}, final=True)
        assert "<Cite" in text


class CapturingAccount(AccountSession):
    """Account stand-in recording prompts and replaying canned events."""

    def __init__(self, identity: str, email: str, events: list[SSEEvent]) -> None:
        """Create the account with canned events and an empty prompt log."""
        super().__init__({
            "identity": identity,
            "session": {
                "accessToken": "",
                "account": {"planType": "plus"},
                "user": {"email": email},
            },
            "cookies": {},
        })
        self._events = events
        self.prompts: list[str] = []

    @override
    async def models(self) -> list[dict[str, str]]:
        """Return the canned single-model listing.

        Returns:
            The single-entry model listing with slug ``auto``.
        """
        return [{"slug": "auto"}]

    @override
    async def stream_conversation(
        self, *, prompt_text: str, **_kwargs: object
    ) -> AsyncIterator[SSEEvent]:
        """Record the prompt and replay the canned SSE events.

        Yields:
            Each canned server-sent event in order.
        """
        self.prompts.append(prompt_text)
        for event in self._events:
            yield event


class TestBranchContinuation(unittest.TestCase):
    """Branched retries must continue the shared prefix, not replay all."""

    @staticmethod
    def test_edited_retry_continues_shared_prefix() -> None:
        """Verify an edited retry sends only the new message."""
        with tempfile.TemporaryDirectory() as tmp:
            store = ConversationStore(db_path=Path(tmp) / "branch.db")
            with patch("app.engine.STORE", store):
                pool = AccountPool()
                acct = CapturingAccount("a", "a@example.com", RACE_EVENTS)
                pool.register(acct)

                async def run(items: list[HistoryItem]) -> None:
                    """Drain one turn for the given history items."""
                    parsed = ParsedRequest(
                        system_text="",
                        items=items,
                        model_requested="auto",
                        stream=False,
                    )
                    async for _ in run_turn(parsed, pool):
                        pass

                first = [
                    HistoryItem(role="user", text="u1"),
                    HistoryItem(role="assistant", text="a1"),
                    HistoryItem(role="user", text="u2 suffix"),
                ]
                asyncio.run(run(first))
                branched = [
                    HistoryItem(role="user", text="u1"),
                    HistoryItem(role="assistant", text="a1"),
                    HistoryItem(role="user", text="u2"),
                    HistoryItem(role="assistant", text="a2"),
                    HistoryItem(role="user", text="u3"),
                ]
                asyncio.run(run(branched))
                assert len(acct.prompts) == EXPECTED_TWO_PROMPTS
                assert acct.prompts[1] == "u3"


if __name__ == "__main__":
    unittest.main()
