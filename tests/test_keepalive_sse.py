# Copyright 2026 chatgpt-to-openai-api contributors.
"""SSE keepalive regression tests for slow image-generation turns.

Image-generation turns can produce zero SSE bytes for minutes while ChatGPT
renders the image. Both streaming paths in app/main.py must emit SSE comment
frames (``: ping``) at the idle interval so downstream clients with idle
timeouts do not abort the request before the proxy answers.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import unittest
from contextlib import contextmanager
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from app import main
from app.adapters import HistoryItem, ParsedRequest
from app.engine import EngineError, TurnResult

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator

UPSTREAM_STALL_MESSAGE = "upstream stall"

SLOW_DELTA_DELAY = 0.3
DONE_DELAY = 0.1
TEST_IDLE_SECONDS = 0.05
DELTA_TEXT = "hello"
KEEPALIVE_FRAME = ": ping\n\n"
DATA_DONE = "data: [DONE]\n\n"


def _make_request() -> ParsedRequest:
    """Build the single-turn user request shared by every test here.

    Returns:
        The parsed single-turn request used by every test.

    """
    return ParsedRequest(
        system_text="",
        items=[HistoryItem(role="user", text="draw a cat")],
        model_requested="auto",
        stream=True,
    )


def _make_result() -> TurnResult:
    """Build a minimal completed-turn result.

    Returns:
        A minimal completed-turn result.

    """
    return TurnResult(
        text=DELTA_TEXT,
        conversation_id="conv-k",
        parent_id="p-k",
        model="auto",
        created=1700000000,
        response_id="resp_k",
    )


def _slow_events() -> AsyncIterator[dict[str, object]]:
    """Return a turn-event stream that stalls before its first delta.

    Returns:
        A single-delta event stream with an image-render-like stall first.

    """

    async def _gen() -> AsyncIterator[dict[str, object]]:
        await asyncio.sleep(SLOW_DELTA_DELAY)
        yield {"type": "delta", "text": DELTA_TEXT}
        await asyncio.sleep(DONE_DELAY)
        yield {"type": "done", "result": _make_result()}

    return _gen()


def _fast_events() -> AsyncIterator[dict[str, object]]:
    """Return a turn-event stream with no idle gap.

    Returns:
        A single-delta event stream that resolves immediately.

    """

    async def _gen() -> AsyncIterator[dict[str, object]]:
        yield {"type": "delta", "text": DELTA_TEXT}
        await asyncio.sleep(0)
        yield {"type": "done", "result": _make_result()}

    return _gen()


def _run_chunks(stream: AsyncIterator[str]) -> list[str]:
    """Drain a stream of SSE chunks into a list.

    Returns:
        The drained SSE chunks in order.

    """

    async def _drain() -> list[str]:
        return [chunk async for chunk in stream]

    return asyncio.run(_drain())


@contextmanager
def _patch_events(
    factory: Callable[[], AsyncIterator[dict[str, object]]],
) -> Iterator[None]:
    """Patch the turn-event source with a canned event factory."""
    with patch.object(main, "_turn_events", side_effect=lambda *_a, **_k: factory()):
        yield


class TestSseKeepalive(unittest.TestCase):
    """Keepalive frames must precede slow deltas and never pollute fast ones."""

    @staticmethod
    def test_chat_stream_emits_keepalive_before_delayed_delta() -> None:
        """Emit a comment frame before a delayed chat delta."""
        with (
            patch.object(main, "KEEPALIVE_IDLE_SECONDS", TEST_IDLE_SECONDS),
            _patch_events(_slow_events),
        ):
            chunks = _run_chunks(main.chat_stream(_make_request(), include_usage=False))
        assert KEEPALIVE_FRAME in chunks
        assert chunks.index(KEEPALIVE_FRAME) < len(chunks) - 1
        assert any(DELTA_TEXT in chunk for chunk in chunks)
        assert chunks[-1] == DATA_DONE

    @staticmethod
    def test_chat_stream_fast_emits_no_keepalive() -> None:
        """Emit zero keepalives when deltas arrive promptly."""
        with (
            patch.object(main, "KEEPALIVE_IDLE_SECONDS", TEST_IDLE_SECONDS),
            _patch_events(_fast_events),
        ):
            chunks = _run_chunks(main.chat_stream(_make_request(), include_usage=False))
        assert KEEPALIVE_FRAME not in chunks
        assert chunks[-1] == DATA_DONE

    @staticmethod
    def test_responses_stream_emits_keepalive_before_delayed_delta() -> None:
        """Emit a comment frame before a delayed Responses delta."""
        with (
            patch.object(main, "KEEPALIVE_IDLE_SECONDS", TEST_IDLE_SECONDS),
            _patch_events(_slow_events),
        ):
            chunks = _run_chunks(main.responses_stream(_make_request(), None))
        assert KEEPALIVE_FRAME in chunks
        first_data = next(chunk for chunk in chunks if chunk.startswith("data:"))
        assert chunks.index(KEEPALIVE_FRAME) > chunks.index(first_data)
        assert any("output_text.delta" in chunk for chunk in chunks)
        assert chunks[-1] == DATA_DONE

    @staticmethod
    def test_responses_stream_fast_emits_no_keepalive() -> None:
        """Emit zero keepalives on the Responses path when deltas are prompt."""
        with (
            patch.object(main, "KEEPALIVE_IDLE_SECONDS", TEST_IDLE_SECONDS),
            _patch_events(_fast_events),
        ):
            chunks = _run_chunks(main.responses_stream(_make_request(), None))
        assert KEEPALIVE_FRAME not in chunks
        assert chunks[-1] == DATA_DONE

    @staticmethod
    def test_keepalive_survives_slow_upstream_and_closes_it() -> None:
        """Keep waiting across ticks and close the upstream generator on exit."""
        closed = False

        async def _tracked() -> AsyncIterator[dict[str, object]]:
            nonlocal closed
            try:
                await asyncio.sleep(SLOW_DELTA_DELAY)
                yield {"type": "delta", "text": DELTA_TEXT}
            finally:
                closed = True

        async def _run() -> list[dict[str, object] | None]:
            return [item async for item in main.keepalive_events(_tracked())]

        with patch.object(main, "KEEPALIVE_IDLE_SECONDS", TEST_IDLE_SECONDS):
            seen = asyncio.run(_run())
        assert any(item is None for item in seen)
        assert any(
            isinstance(item, dict) and item.get("type") == "delta" for item in seen
        )
        assert closed

    @staticmethod
    def test_chat_stream_error_after_idle_still_single_done() -> None:
        """Deliver ping-then-error in-band with exactly one done sentinel."""

        async def _idle_then_error() -> AsyncIterator[dict[str, object]]:
            await asyncio.sleep(SLOW_DELTA_DELAY)
            yield {"type": "error", "error": EngineError(502, "boom")}
            yield {"type": "unreachable", "text": "x"}  # pragma: no cover

        with (
            patch.object(main, "KEEPALIVE_IDLE_SECONDS", TEST_IDLE_SECONDS),
            _patch_events(_idle_then_error),
        ):
            chunks = _run_chunks(main.chat_stream(_make_request(), include_usage=False))
        assert KEEPALIVE_FRAME in chunks
        assert chunks.index(KEEPALIVE_FRAME) < chunks.index(DATA_DONE)
        assert chunks.count(DATA_DONE) == 1
        assert any('"boom"' in chunk for chunk in chunks)

    @staticmethod
    def test_responses_stream_error_after_idle_skips_completed() -> None:
        """Deliver ping-then-failure with no completed envelope after it."""

        async def _idle_then_error() -> AsyncIterator[dict[str, object]]:
            await asyncio.sleep(SLOW_DELTA_DELAY)
            yield {"type": "error", "error": EngineError(502, "boom")}
            yield {"type": "unreachable", "text": "x"}  # pragma: no cover

        with (
            patch.object(main, "KEEPALIVE_IDLE_SECONDS", TEST_IDLE_SECONDS),
            _patch_events(_idle_then_error),
        ):
            chunks = _run_chunks(main.responses_stream(_make_request(), None))
        assert KEEPALIVE_FRAME in chunks
        assert any("response.failed" in chunk for chunk in chunks)
        assert not any("response.completed" in chunk for chunk in chunks)
        assert chunks.count(DATA_DONE) == 1

    @staticmethod
    def test_responses_stream_keepalive_preserves_sequence_order() -> None:
        """Keep sequence numbers strictly increasing across keepalive ticks."""
        with (
            patch.object(main, "KEEPALIVE_IDLE_SECONDS", TEST_IDLE_SECONDS),
            _patch_events(_slow_events),
        ):
            chunks = _run_chunks(main.responses_stream(_make_request(), None))
        seqs = [
            int(json.loads(chunk[len("data: ") :])["sequence_number"])
            for chunk in chunks
            if chunk.startswith("data: {")
        ]
        assert seqs == sorted(seqs)
        assert len(seqs) == len(set(seqs))

    @staticmethod
    def test_upstream_timeout_error_propagates_not_keepalive() -> None:
        """Never misclassify a real upstream TimeoutError as an idle tick."""

        async def _raise_timeout() -> AsyncIterator[dict[str, object]]:
            await asyncio.sleep(SLOW_DELTA_DELAY)
            raise TimeoutError(UPSTREAM_STALL_MESSAGE)
            yield {"type": "unreachable", "text": "x"}  # pragma: no cover

        async def _run() -> None:
            async with contextlib.aclosing(
                main.keepalive_events(_raise_timeout()),
            ) as wrapped:
                async for _ in wrapped:
                    pass

        with (
            patch.object(main, "KEEPALIVE_IDLE_SECONDS", TEST_IDLE_SECONDS),
            pytest.raises(TimeoutError, match="upstream stall"),
        ):
            asyncio.run(_run())

    @staticmethod
    def test_break_closes_wrapper_and_upstream() -> None:
        """Breaking out of the consumer must close upstream deterministically."""
        closed = False

        async def _tracked() -> AsyncIterator[dict[str, object]]:
            nonlocal closed
            try:
                await asyncio.sleep(SLOW_DELTA_DELAY)
                yield {"type": "delta", "text": DELTA_TEXT}
            finally:
                closed = True

        async def _run() -> None:
            async with contextlib.aclosing(
                main.keepalive_events(_tracked()),
            ) as wrapped:
                async for _ in wrapped:
                    break

        with patch.object(main, "KEEPALIVE_IDLE_SECONDS", TEST_IDLE_SECONDS):
            asyncio.run(_run())
        assert closed

    @staticmethod
    def test_double_aclose_is_safe() -> None:
        """Closing the exhausted wrapper twice must not raise."""

        async def _run() -> None:
            wrapped = main.keepalive_events(_fast_events())
            async for _ in wrapped:
                pass
            await wrapped.aclose()
            await wrapped.aclose()

        asyncio.run(_run())

    @staticmethod
    def test_keepalive_frame_is_sse_comment() -> None:
        """Keepalive frames must parse as SSE comments, not data lines."""
        assert KEEPALIVE_FRAME.startswith(":")
        assert KEEPALIVE_FRAME.endswith("\n\n")
        assert not KEEPALIVE_FRAME.startswith("data:")


if __name__ == "__main__":
    unittest.main()
