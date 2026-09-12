# Copyright 2026 chatgpt-to-openai-api contributors.
"""Regression tests for upstream moderation-error nodes (metadata.is_error).

ChatGPT delivers prompt-policy refusals as HTTP-200 SSE streams whose final
assistant text node carries metadata.is_error=true and no "next" marker. The
engine must raise BEFORE streaming any of that text, so the failover loop can
retry on another account (moderation verdicts are model/plan-tier dependent).
"""

from __future__ import annotations

import asyncio
import unittest
from typing import TYPE_CHECKING

import pytest
from typing_extensions import override

from app.accounts import AccountPool
from app.adapters import HistoryItem, ParsedRequest
from app.chatgpt import AccountSession, ChatGPTError
from app.engine import EngineError, TurnResult, run_turn

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

DELTA_EVENT_TYPE = "delta"
DONE_EVENT_TYPE = "done"
REFUSAL_MARKER = "sorry"
REJECTION_MESSAGE = "rejected this prompt"
INCOMPLETE_MESSAGE = "incomplete response"
EXPECTED_DIAGRAM_TEXT = "Room-clearing diagram."
PLUS_EMAIL = "plus@example.com"
HTTP_BAD_REQUEST_STATUS = 400
EXPECTED_SINGLE_REQUEST = 1
EXPECTED_NO_REQUESTS = 0


class FakeAccount(AccountSession):
    """Minimal AccountSession stand-in replaying canned SSE events."""

    def __init__(
        self,
        identity: str,
        email: str,
        events: list[dict[str, object]],
        plan: str = "free",
    ) -> None:
        """Initialize the fake with canned SSE events."""
        self.identity = identity
        self.email = email
        self.plan = plan
        self.inflight = 0
        self.total_requests = 0
        self.dead = False
        self.cooldown_until = 0.0
        self.last_used = 0.0
        self._events = events

    @override
    async def models(self) -> list[dict[str, str]]:
        """Return the canned model list.

        Returns:
            The single-entry model list with slug ``auto``.

        """
        return [{"slug": "auto"}]

    @override
    async def stream_conversation(
        self, **_kwargs: object
    ) -> AsyncIterator[dict[str, object]]:
        """Replay canned SSE events, ignoring request details.

        Yields:
            Each canned SSE event in order.

        """
        for event in self._events:
            yield event


def _user_then(
    events_tail: list[dict[str, object]], conv_id: str = "conv-x"
) -> list[dict[str, object]]:
    """Prepend a user message event to canned assistant events.

    Returns:
        The user message event followed by the given assistant events.

    """
    head: list[dict[str, object]] = [
        {
            "message": {
                "id": "u1",
                "author": {"role": "user"},
                "content": {"content_type": "text", "parts": ["q"]},
                "metadata": {},
            },
            "conversation_id": conv_id,
        },
    ]
    return head + events_tail


REFUSAL = (
    "We\u2019re so sorry, but the prompt may violate our content policies. "
    "If you think we got it wrong, please retry or edit your prompt."
)

INCIDENT_EVENTS = _user_then(
    [
        # code node: assistant role, non-text content type, parts=None
        {
            "message": {
                "id": "a1",
                "author": {"role": "assistant"},
                "content": {"content_type": "code", "parts": None},
                "metadata": {},
            },
        },
        {
            "message": {
                "id": "a2",
                "author": {"role": "assistant"},
                "content": {"content_type": "text", "parts": [REFUSAL]},
                "metadata": {"is_error": True},
            },
        },
        {"type": "done"},
    ],
)

OK_EVENTS = _user_then(
    [
        {
            "message": {
                "id": "n1",
                "author": {"role": "assistant"},
                "content": {
                    "content_type": "text",
                    "parts": ["Room-clearing diagram."],
                },
                "metadata": {"message_type": "next"},
            },
        },
        {"type": "done"},
    ],
    conv_id="conv-ok",
)


class TestUpstreamErrorNodes(unittest.TestCase):
    """Upstream error-node failover tests."""

    @staticmethod
    def _parsed() -> ParsedRequest:
        """Build the minimal user request shared by all tests.

        Returns:
            A single-turn user request for ``auto`` with streaming disabled.

        """
        return ParsedRequest(
            system_text="",
            items=[HistoryItem(role="user", text="q")],
            model_requested="auto",
            stream=False,
        )

    def test_refusal_never_streams_and_fails_over(self) -> None:
        """Verify refusal bytes never stream and failover serves the turn."""
        pool = AccountPool()
        free = FakeAccount("free", "free@example.com", INCIDENT_EVENTS)
        plus = FakeAccount("plus", PLUS_EMAIL, OK_EVENTS, plan="plus")
        pool.register(free)
        pool.register(plus)

        deltas: list[str] = []
        result: TurnResult | None = None

        async def run() -> None:
            """Collect streamed deltas and the final turn result."""
            nonlocal result
            async for event in run_turn(self._parsed(), pool):
                if event["type"] == DELTA_EVENT_TYPE:
                    text = event["text"]
                    assert isinstance(text, str)
                    deltas.append(text)
                elif event["type"] == DONE_EVENT_TYPE:
                    done = event["result"]
                    assert isinstance(done, TurnResult)
                    result = done

        asyncio.run(run())
        leaked = any(REFUSAL_MARKER in delta for delta in deltas)
        assert not leaked, "refusal bytes must never reach the client"
        assert result is not None
        assert result.account_email == PLUS_EMAIL

    def test_single_account_raises_before_streaming(self) -> None:
        """Verify a lone refusal raises before streaming anything."""
        pool = AccountPool()
        pool.register(FakeAccount("free", "f@example.com", INCIDENT_EVENTS))

        saw_delta = False

        async def run() -> None:
            """Record whether any delta reaches the client."""
            nonlocal saw_delta
            async for event in run_turn(self._parsed(), pool):
                if event["type"] == DELTA_EVENT_TYPE:
                    saw_delta = True

        with pytest.raises(Exception, match=REJECTION_MESSAGE):
            asyncio.run(run())
        assert not saw_delta

    def test_normal_turn_unaffected(self) -> None:
        """Verify a healthy turn streams its text unchanged."""
        pool = AccountPool()
        pool.register(FakeAccount("a", "a@example.com", OK_EVENTS))
        deltas: list[str] = []

        async def run() -> None:
            """Collect streamed delta texts."""
            deltas.extend([
                text
                async for event in run_turn(self._parsed(), pool)
                if event["type"] == DELTA_EVENT_TYPE
                for text in [event["text"]]
                if isinstance(text, str)
            ])

        asyncio.run(run())
        assert "".join(deltas) == EXPECTED_DIAGRAM_TEXT

    def test_incomplete_stream_without_is_error_still_flags_502(self) -> None:
        """Verify a stream ending without content raises a 502 error."""
        events = _user_then([{"type": "done"}])
        pool = AccountPool()
        pool.register(FakeAccount("b", "b@example.com", events))

        async def run() -> None:
            """Drain the turn stream."""
            async for _ in run_turn(self._parsed(), pool):
                pass

        with pytest.raises(Exception, match=INCOMPLETE_MESSAGE):
            asyncio.run(run())


class TooLargeAccount(AccountSession):
    """Account stand-in failing fast with backend input_too_large."""

    def __init__(self, identity: str, email: str) -> None:
        """Create the failing account with the given identity."""
        super().__init__({
            "identity": identity,
            "session": {
                "accessToken": "",
                "account": {"planType": "plus"},
                "user": {"email": email},
            },
            "cookies": {},
        })

    @override
    async def models(self) -> list[dict[str, str]]:
        """Return the canned single-model listing.

        Returns:
            The single-entry model listing with slug ``auto``.
        """
        return [{"slug": "auto"}]

    @override
    async def stream_conversation(
        self, **_kwargs: object
    ) -> AsyncIterator[dict[str, object]]:
        """Raise the backend oversized-message error without streaming.

        Yields:
            Never yields: raises before the first event.

        Raises:
            ChatGPTError: Always raised carrying the input_too_large body.
        """
        raise ChatGPTError(
            HTTP_BAD_REQUEST_STATUS,
            '{"detail":{"message":"The message you submitted was too long, '
            'please edit it and resubmit.","code":"input_too_large",'
            '"can_retry":false,"type":"last_user_message"}}',
        )
        yield {"type": "done"}


class TestInputTooLarge(unittest.TestCase):
    """Backend input_too_large must surface as 400 without failover."""

    @staticmethod
    def test_oversized_history_fails_fast_as_400() -> None:
        """Verify oversized history raises 400 and tries one account only."""
        pool = AccountPool()
        first = TooLargeAccount("a", "a@example.com")
        second = TooLargeAccount("b", "b@example.com")
        pool.register(first)
        pool.register(second)
        parsed = ParsedRequest(
            system_text="",
            items=[HistoryItem(role="user", text="q")],
            model_requested="auto",
            stream=False,
        )

        async def run() -> None:
            """Drain the turn stream."""
            async for _ in run_turn(parsed, pool):
                pass

        with pytest.raises(EngineError) as exc_info:
            asyncio.run(run())
        assert exc_info.value.status == HTTP_BAD_REQUEST_STATUS
        assert exc_info.value.error_type == "invalid_request_error"
        assert first.total_requests == EXPECTED_SINGLE_REQUEST
        assert second.total_requests == EXPECTED_NO_REQUESTS


if __name__ == "__main__":
    unittest.main()
