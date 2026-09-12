# Copyright 2026 chatgpt-to-openai-api contributors.
"""Regression tests for the "image creation temporarily unavailable" refusal.

ChatGPT serves some accounts a plain-text refusal ("It looks like image
creation is temporarily unavailable. Do you want to try something else?")
as a normal HTTP-200 stream -- no metadata.is_error flag. The engine must
classify it as an image-limit refusal BEFORE any byte streams, so the
failover loop rotates through every remaining account; only when ALL of
them refuse may the failure surface to the client.
"""

from __future__ import annotations

import asyncio
import unittest
from typing import TYPE_CHECKING, TypeAlias
from unittest.mock import patch

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

import pytest
from typing_extensions import override

from app.accounts import AccountPool
from app.adapters import HistoryItem, ParsedRequest
from app.chatgpt import AccountSession
from app.engine import EngineError, TurnResult, run_turn
from app.freeimage import FreeimageError

SSEEvent: TypeAlias = dict[str, object]

EXPECTED_ATTEMPTS_EACH = 1


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
            The single-entry model listing with slug ``auto``.

        """
        return [{"slug": "auto"}]

    @override
    async def stream_conversation(self, **_kwargs: object) -> AsyncIterator[SSEEvent]:
        """Replay the canned SSE events.

        Yields:
            The canned server-sent events in order.

        """
        for event in self._events:
            yield event


def _make_request() -> ParsedRequest:
    """Build the single-turn user request shared by every test here.

    Returns:
        The parsed single-turn request used by every test.

    """
    return ParsedRequest(
        system_text="",
        items=[HistoryItem(role="user", text="q")],
        model_requested="auto",
        stream=False,
    )


async def _stream_text(pool: AccountPool) -> list[str]:
    """Run one turn, returning only the streamed delta bytes.

    Returns:
        The streamed delta bytes in order.

    """
    deltas, _ = await _stream_turn(pool)
    return deltas


async def _stream_turn(pool: AccountPool) -> tuple[list[str], TurnResult | None]:
    """Run one turn, returning streamed bytes and the completion result.

    Returns:
        A tuple of streamed delta bytes and the completion result, if any.

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


def _user_then(events_tail: list[SSEEvent], conv_id: str = "conv-x") -> list[SSEEvent]:
    """Prepend the shared user turn to a canned assistant event tail.

    Returns:
        The user-turn head followed by the given assistant event tail.

    """
    head: list[SSEEvent] = [
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
    return [*head, *events_tail]


TEMP_REFUSAL = (
    "It looks like image creation is temporarily unavailable. "
    "Do you want to try something else?"
)

REFUSAL_EVENTS = _user_then(
    [
        # Plain text node: no is_error flag, no "next" marker.
        {
            "message": {
                "id": "a1",
                "author": {"role": "assistant"},
                "content": {"content_type": "text", "parts": [TEMP_REFUSAL]},
                "metadata": {},
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
                "content": {"content_type": "text", "parts": ["Here is your image."]},
                "metadata": {"message_type": "next"},
            },
        },
        {"type": "done"},
    ],
    conv_id="conv-ok",
)

QUOTED_REFUSAL_TEXT = (
    "Support said: 'it looks like image creation is temporarily "
    "unavailable' -- but it works again now."
)

# The phrase quoted MID-sentence in an otherwise normal reply must stream.
QUOTE_EVENTS = _user_then(
    [
        {
            "message": {
                "id": "n2",
                "author": {"role": "assistant"},
                "content": {
                    "content_type": "text",
                    "parts": [QUOTED_REFUSAL_TEXT],
                },
                "metadata": {"message_type": "next"},
            },
        },
        {"type": "done"},
    ],
    conv_id="conv-quote",
)

# Variant wording (currently/temporarily, request(s), other nouns) arrives
# CHUNKED: early chunks diverge from any exact string before IMAGE_LIMIT_RE
# can confirm, so mid-stream withholding must rely on the generic starter.
VARIANT_REFUSAL = "It looks like image creation is currently unavailable."

CHUNKED_VARIANT_EVENTS = _user_then(
    [
        {
            "message": {
                "id": "a1",
                "author": {"role": "assistant"},
                "content": {
                    "content_type": "text",
                    "parts": ["It looks like image creation is cu"],
                },
                "metadata": {},
            },
        },
        {
            "message": {
                "id": "a1",
                "author": {"role": "assistant"},
                "content": {"content_type": "text", "parts": [VARIANT_REFUSAL]},
                "metadata": {},
            },
        },
        {"type": "done"},
    ],
)

# A normal reply that genuinely BEGINS with "It looks like image ..." must
# still reach the client (noun gate proves divergence immediately).
NORMAL_IMAGE_PHRASE_EVENTS = _user_then(
    [
        {
            "message": {
                "id": "n3",
                "author": {"role": "assistant"},
                "content": {
                    "content_type": "text",
                    "parts": [
                        "It looks like image quality improved after the last update.",
                    ],
                },
                "metadata": {"message_type": "next"},
            },
        },
        {"type": "done"},
    ],
    conv_id="conv-norm",
)


class TestTemporaryUnavailableFailover(unittest.TestCase):
    """Regression tests for image-limit refusal failover."""

    @staticmethod
    def test_refusal_fails_over_to_next_account() -> None:
        """Fail over to the next account on an image-limit refusal."""
        pool = AccountPool()
        pool.register(FakeAccount("free", "free@example.com", REFUSAL_EVENTS))
        pool.register(FakeAccount("plus", "plus@example.com", OK_EVENTS, plan="plus"))

        deltas, result = asyncio.run(_stream_turn(pool))
        assert not any(
            "temporarily unavailable" in chunk.lower() for chunk in deltas
        ), "refusal bytes must never reach the client"
        assert result is not None
        assert result.account_email == "plus@example.com"

    @staticmethod
    def test_all_accounts_tried_before_error_surfaces() -> None:
        """Try every account before surfacing the refusal error."""
        pool = AccountPool()
        first = FakeAccount("a", "a@example.com", REFUSAL_EVENTS)
        second = FakeAccount("b", "b@example.com", REFUSAL_EVENTS)
        pool.register(first)
        pool.register(second)
        streamed: list[str] = []

        async def _run() -> None:
            async for event in run_turn(_make_request(), pool):
                if event["type"] == "delta":
                    text = event["text"]
                    if isinstance(text, str):
                        streamed.append(text)

        with pytest.raises(
            EngineError, match=r"all 2 available account\(s\) failed"
        ) as exc_info:
            asyncio.run(_run())
        # Both accounts attempted exactly once; nothing leaked.
        assert (first.total_requests, second.total_requests) == (
            EXPECTED_ATTEMPTS_EACH,
            EXPECTED_ATTEMPTS_EACH,
        )
        assert streamed == []
        assert "temporarily unavailable" in str(exc_info.value)

    @staticmethod
    def test_mid_text_quote_is_not_a_refusal() -> None:
        """Stream a mid-text refusal quote instead of failing over."""
        pool = AccountPool()
        pool.register(FakeAccount("q", "q@example.com", QUOTE_EVENTS))

        deltas = asyncio.run(_stream_text(pool))
        assert "works again now" in "".join(deltas)

    @staticmethod
    def test_variant_refusal_never_leaks_mid_stream() -> None:
        """Withhold chunked variant-refusal bytes and fail over."""
        pool = AccountPool()
        pool.register(FakeAccount("free", "free@example.com", CHUNKED_VARIANT_EVENTS))
        pool.register(FakeAccount("plus", "plus@example.com", OK_EVENTS, plan="plus"))

        deltas, result = asyncio.run(_stream_turn(pool))
        assert not any(
            "unavailable" in chunk.lower() or "it looks like" in chunk.lower()
            for chunk in deltas
        ), f"variant refusal bytes leaked mid-stream: {deltas!r}"
        assert result is not None
        assert result.account_email == "plus@example.com"

    @staticmethod
    def test_normal_reply_starting_with_phrase_streams_fully() -> None:
        """Stream a normal reply sharing the refusal's opening words."""
        pool = AccountPool()
        pool.register(FakeAccount("n", "n@example.com", NORMAL_IMAGE_PHRASE_EVENTS))

        deltas = asyncio.run(_stream_text(pool))
        assert "quality improved" in "".join(deltas)


IMAGE_ONLY_EVENTS = _user_then(
    [
        {
            "message": {
                "id": "c1",
                "author": {"role": "assistant"},
                "content": {"content_type": "code", "parts": ["// generating"]},
                "metadata": {},
            },
        },
        {
            "message": {
                "id": "t1",
                "author": {"role": "tool"},
                "content": {
                    "content_type": "multimodal_text",
                    "parts": [
                        {
                            "content_type": "image_asset_pointer",
                            "asset_pointer": "sediment://file_test123",
                        }
                    ],
                },
                "metadata": {},
            },
        },
        {
            "message": {
                "id": "a2",
                "author": {"role": "assistant"},
                "content": {"content_type": "text", "parts": [""]},
                "metadata": {"message_type": "next"},
            },
        },
        {"type": "done"},
    ],
    conv_id="conv-img",
)


TEXT_AND_IMAGE_EVENTS = _user_then(
    [
        {
            "message": {
                "id": "a1",
                "author": {"role": "assistant"},
                "content": {
                    "content_type": "text",
                    "parts": ["Here is the analysis."],
                },
                "metadata": {"message_type": "next"},
            },
        },
        {
            "message": {
                "id": "t1",
                "author": {"role": "tool"},
                "content": {
                    "content_type": "multimodal_text",
                    "parts": [
                        {
                            "content_type": "image_asset_pointer",
                            "asset_pointer": "sediment://file_test123",
                        }
                    ],
                },
                "metadata": {},
            },
        },
        {"type": "done"},
    ],
    conv_id="conv-mixed",
)


class TestImageDeliveryFailure(unittest.TestCase):
    """Regression tests for undeliverable generated images."""

    @staticmethod
    def test_image_only_turn_with_failed_delivery_errors() -> None:
        """Surface an explicit error instead of an empty success."""
        pool = AccountPool()
        pool.register(FakeAccount("img", "img@example.com", IMAGE_ONLY_EVENTS))
        streamed: list[str] = []

        async def _run() -> None:
            async for event in run_turn(_make_request(), pool):
                if event["type"] == "delta":
                    text = event["text"]
                    if isinstance(text, str):
                        streamed.append(text)

        async def _fake_download(_self: AccountSession, _ptr: str) -> tuple[str, bytes]:
            # Interface requires async because callers await this fake.
            await asyncio.sleep(0)
            return ("night.png", b"\x89PNG fake-bytes")

        async def _fake_upload(_name: str, _data: bytes, _mime: str) -> str:
            # Interface requires async because callers await this fake.
            await asyncio.sleep(0)
            msg = "upload failed HTTP 500: internal server error"
            raise FreeimageError(msg)

        with (
            patch.object(AccountSession, "download_file_url", _fake_download),
            patch("app.engine.upload_image", _fake_upload),
            pytest.raises(EngineError, match=r"image delivery failed"),
        ):
            asyncio.run(_run())
        assert streamed == []

    @staticmethod
    def test_text_turn_with_failed_delivery_streams_text() -> None:
        """Deliver produced text alone when images cannot be delivered."""
        pool = AccountPool()
        pool.register(FakeAccount("mix", "mix@example.com", TEXT_AND_IMAGE_EVENTS))

        async def _fake_download(_self: AccountSession, _ptr: str) -> tuple[str, bytes]:
            # Interface requires async because callers await this fake.
            await asyncio.sleep(0)
            return ("night.png", b"\x89PNG fake-bytes")

        async def _fake_upload(_name: str, _data: bytes, _mime: str) -> str:
            # Interface requires async because callers await this fake.
            await asyncio.sleep(0)
            msg = "upload failed HTTP 500: internal server error"
            raise FreeimageError(msg)

        with (
            patch.object(AccountSession, "download_file_url", _fake_download),
            patch("app.engine.upload_image", _fake_upload),
        ):
            deltas, result = asyncio.run(_stream_turn(pool))
        assert result is not None
        assert "Here is the analysis." in "".join(deltas)
        assert "Here is the analysis." in result.text


if __name__ == "__main__":
    unittest.main()
