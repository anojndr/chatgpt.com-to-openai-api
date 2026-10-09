# Copyright 2026 chatgpt-to-openai-api contributors.
"""FastAPI server exposing ChatGPT web accounts as an OpenAI-compatible API.

Endpoints:
  GET  /v1/models
  POST /v1/chat/completions   (stream + non-stream)
  POST /v1/responses          (stream + non-stream, previous_response_id)
  GET  /v1/accounts           (pool debug snapshot)
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import config
from .accounts import POOL, NoAccountAvailableError
from .adapters import (
    ParsedRequest,
    parse_chat_request,
    parse_responses_request,
    public_models,
)
from .chatgpt import ChatGPTError
from .engine import EngineError, TurnResult, collect, run_turn
from .redis_cache import get_cache

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Iterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("main")

# Idle gap between SSE comment frames (``: ping``) while a turn produces no
# bytes. Image-generation turns can stall for minutes; without traffic,
# downstream clients with idle timeouts abort the request before the proxy
# answers. Comments are valid SSE that OpenAI clients ignore.
KEEPALIVE_IDLE_SECONDS = 15.0
KEEPALIVE_PING = ": ping\n\n"

_HTTP_BAD_REQUEST = 400
_HTTP_BAD_GATEWAY = 502

# Any failure to reach the live models endpoint — or to interpret its
# payload — degrades to the static fallback so the endpoint stays available.
_MODELS_FALLBACK_ERRORS = (
    ChatGPTError,
    OSError,
    ValueError,
    KeyError,
    AttributeError,
    TypeError,
)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Start the account pool watcher for the application lifetime."""
    await POOL.start_watcher()
    yield


app = FastAPI(title="chatgpt-to-openai-api", version="1.0.0", lifespan=lifespan)


def auth_ok(request: Request) -> bool:
    """Check whether the request carries a valid API key.

    Returns:
        True when the request is authorized, else False.

    """
    if not config.API_KEY:
        return True
    auth = request.headers.get("authorization", "")
    expected = f"Bearer {config.API_KEY}"
    return hmac.compare_digest(auth.encode(), expected.encode())


def oai_error(
    status: int,
    message: str,
    err_type: str = "invalid_request_error",
    code: str | None = None,
) -> JSONResponse:
    """Build an OpenAI-style JSON error response.

    Returns:
        The JSON error response with the given status and message.

    """
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": err_type,
                "param": None,
                "code": code,
            },
        },
    )


@app.exception_handler(EngineError)
def engine_error_handler(_request: Request, exc: EngineError) -> JSONResponse:
    """Translate an EngineError into an OpenAI-style error response.

    Returns:
        The OpenAI-style JSON error response for the failure.

    """
    status = exc.status if exc.status >= _HTTP_BAD_REQUEST else _HTTP_BAD_GATEWAY
    return oai_error(status, exc.message, exc.error_type)


async def _body(request: Request) -> dict[str, object]:
    """Parse the request body as a JSON object.

    Returns:
        The request body as a string-keyed dict.

    Raises:
        EngineError: If the body is not valid JSON or not a JSON object.

    """
    try:
        raw = await request.body()
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        message = "request body must be valid JSON"
        raise EngineError(_HTTP_BAD_REQUEST, message) from exc
    if not isinstance(parsed, dict):
        raise EngineError(_HTTP_BAD_REQUEST, "request body must be a JSON object")
    return {str(key): value for key, value in parsed.items()}


# ------------------------------------------------------------------ models


@app.get("/v1/models", response_model=None)
@app.get("/models", response_model=None)
async def list_models(request: Request) -> JSONResponse | dict[str, object]:
    """Return the live model list, falling back to a static entry.

    Returns:
        The live model list payload, or a static fallback entry.

    """
    if not auth_ok(request):
        return oai_error(401, "invalid api key", "authentication_error")
    fallback: dict[str, object] = {
        "object": "list",
        "data": [
            {
                "id": "auto",
                "object": "model",
                "created": 1785000000,
                "owned_by": "chatgpt-proxy",
            },
        ],
    }
    try:
        acct = POOL.acquire(None)
    except NoAccountAvailableError:
        return fallback
    try:
        live_models = public_models(await acct.models())
    except _MODELS_FALLBACK_ERRORS:
        return fallback
    else:
        payload: dict[str, object] = {"object": "list"}
        payload["data"] = live_models
        return payload
    finally:
        POOL.release(acct)


# ------------------------------------------------------------------ chat completions


def _sse(obj: dict[str, object]) -> str:
    """Encode a payload as a server-sent event line.

    Returns:
        The payload encoded as a server-sent event line.

    """
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


async def _turn_events(
    parsed: ParsedRequest,
    *,
    previous_response_id: str | None = None,
    preferred: str | None = None,
) -> AsyncIterator[dict[str, object]]:
    """Yield turn events, converting engine failures into error events.

    Yields:
        Turn event dicts, with an error event carrying the EngineError.

    """
    try:
        async for ev in run_turn(
            parsed,
            POOL,
            previous_response_id=previous_response_id,
            response_id_prefix="resp_",
            preferred_email=preferred,
        ):
            yield ev
    except EngineError as exc:
        yield {"type": "error", "error": exc}


async def keepalive_events(
    events: AsyncIterator[dict[str, object]],
) -> AsyncGenerator[dict[str, object] | None]:
    """Yield turn events, substituting ``None`` for each idle keepalive tick.

    The upstream iterator is pushed one event ahead behind a shielded task,
    so the idle timeout never cancels into the engine: a slow image
    generation keeps running while the stream layer emits ``None`` ticks.
    The caller must consume this wrapper to exhaustion or ``aclose`` it;
    either path closes ``events``.

    Yields:
        Turn event dicts, or ``None`` once per idle interval with no event.

    """
    task: asyncio.Task[dict[str, object] | None] = asyncio.ensure_future(
        anext(events, None)
    )
    try:
        while True:
            try:
                event = await asyncio.wait_for(
                    asyncio.shield(task), KEEPALIVE_IDLE_SECONDS
                )
            except TimeoutError:
                if task.done():
                    # Settled in a race with the timeout: deliver the event
                    # (or raise the upstream error) instead of ticking.
                    event = task.result()
                else:
                    yield None
                    continue
            if event is None:
                return
            yield event
            task = asyncio.ensure_future(anext(events, None))
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        aclose = getattr(events, "aclose", None)
        if aclose is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await aclose()


def _chat_done_chunks(
    *,
    cid: str,
    created: int,
    model: str,
    res: object,
    include_usage: bool,
) -> Iterator[str]:
    """Yield closing chunks for a completed turn.

    Yields:
        SSE-encoded stop and usage chunks ending with a done sentinel.

    """
    if not isinstance(res, TurnResult):
        return
    usage_block = {
        "prompt_tokens": res.prompt_tokens,
        "completion_tokens": res.completion_tokens,
        "total_tokens": res.prompt_tokens + res.completion_tokens,
    }
    if include_usage:
        yield _sse(
            {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "system_fingerprint": None,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "logprobs": None,
                        "finish_reason": "stop",
                    },
                ],
            },
        )
        yield _sse(
            {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "system_fingerprint": None,
                "choices": [],
                "usage": usage_block,
            },
        )
    else:
        yield _sse(
            {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "system_fingerprint": None,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "logprobs": None,
                        "finish_reason": "stop",
                    },
                ],
            },
        )
    yield "data: [DONE]\n\n"


@dataclass
class _ChatStreamState:
    """Mutable cursor for chat chunk framing."""

    cid: str
    created: int
    model: str
    include_usage: bool
    first: bool = True


def _chat_event_chunks(state: _ChatStreamState, ev: dict[str, object]) -> Iterator[str]:
    """Yield SSE chunks for one turn event, advancing the stream cursor.

    Yields:
        SSE-encoded chunks for the event (possibly none).

    """
    etype = ev["type"]
    if etype == "model":
        model_name = ev["model"]  # resolved live slug, stamped on chunks
        if isinstance(model_name, str):
            state.model = model_name
    elif etype == "delta":
        text = ev["text"]
        if not isinstance(text, str):
            return
        # First chunk carries role + any initial content (never drop text).
        delta = (
            {"role": "assistant", "content": text} if state.first else {"content": text}
        )
        chunk: dict[str, object] = {
            "id": state.cid,
            "object": "chat.completion.chunk",
            "created": state.created,
            "model": state.model,
            "system_fingerprint": None,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "logprobs": None,
                    "finish_reason": None,
                },
            ],
        }
        state.first = False
        yield _sse(chunk)
    elif etype == "done":
        yield from _chat_done_chunks(
            cid=state.cid,
            created=state.created,
            model=state.model,
            res=ev["result"],
            include_usage=state.include_usage,
        )
    elif etype == "error":
        # Headers are already sent; deliver the failure in-band so clients
        # see a well-formed error instead of a truncated stream.
        err = ev["error"]
        if isinstance(err, EngineError):
            yield _sse({
                "error": {
                    "message": err.message,
                    "type": err.error_type,
                    "param": None,
                    "code": str(err.status),
                },
            })
        yield "data: [DONE]\n\n"


async def chat_stream(
    parsed: ParsedRequest, *, include_usage: bool, preferred: str | None = None
) -> AsyncIterator[str]:
    """Stream chat completion chunks for a parsed request.

    Yields:
        SSE-encoded chat completion chunks ending with a done sentinel.

    """
    state = _ChatStreamState(
        cid="chatcmpl-" + uuid.uuid4().hex,
        created=int(time.time()),
        model=parsed.model_requested or "auto",
        include_usage=include_usage,
    )
    events = _turn_events(parsed, preferred=preferred)
    ticked_events = keepalive_events(events)
    try:
        async for ticked in ticked_events:
            if ticked is None:
                yield KEEPALIVE_PING
                continue
            for chunk in _chat_event_chunks(state, ticked):
                yield chunk
    finally:
        await ticked_events.aclose()


@app.post("/v1/chat/completions", response_model=None)
@app.post("/chat/completions", response_model=None)
async def chat_completions(
    request: Request,
) -> JSONResponse | StreamingResponse | dict[str, object]:
    """Handle a chat completion request, streaming or buffered.

    Returns:
        The streaming response when requested, else the buffered completion.

    """
    if not auth_ok(request):
        return oai_error(401, "invalid api key", "authentication_error")
    body = await _body(request)
    try:
        parsed = await parse_chat_request(body)
    except (ValueError, KeyError) as e:
        return oai_error(400, str(e))
    preferred = request.headers.get("x-chatgpt-account") or None
    stream = bool(body.get("stream"))
    if stream:
        stream_options = body.get("stream_options")
        include_usage = (
            bool(stream_options.get("include_usage"))
            if isinstance(stream_options, dict)
            else False
        )
        return StreamingResponse(
            chat_stream(parsed, include_usage=include_usage, preferred=preferred),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    result = await collect(parsed, POOL, preferred_email=preferred)
    return {
        "id": "chatcmpl-" + result.response_id[5:],
        "object": "chat.completion",
        "created": result.created,
        "model": result.model,
        "system_fingerprint": None,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.text,
                    "refusal": None,
                },
                "logprobs": None,
                "finish_reason": "stop",
            },
        ],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        },
    }


# ------------------------------------------------------------------ responses


@dataclass
class _ResponseContent:
    """Variable portion of a Responses API resource envelope."""

    status: str
    output_items: list[dict[str, object]]
    usage: dict[str, object] | None


def _response_resource(
    rid: str,
    model: str,
    created: int,
    prev: str | None,
    content: _ResponseContent,
) -> dict[str, object]:
    """Build a Responses API resource envelope.

    Returns:
        The Responses API resource envelope dict.

    """
    return {
        "id": rid,
        "object": "response",
        "created_at": created,
        "status": content.status,
        "background": False,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "metadata": {},
        "model": model,
        "output": content.output_items,
        "parallel_tool_calls": True,
        "previous_response_id": prev,
        "reasoning": {"effort": None, "summary": None},
        "store": True,
        "temperature": None,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "truncation": "disabled",
        "usage": content.usage,
        "user": None,
    }


def _failed_response_resource(
    rid: str,
    model: str,
    prev: str | None,
    message: str,
    error_type: str,
) -> dict[str, object]:
    """Build a failed Responses API resource envelope.

    Returns:
        The failed resource envelope with error details attached.

    """
    failed = _ResponseContent(status="failed", output_items=[], usage=None)
    resource = _response_resource(rid, model, int(time.time()), prev, failed)
    return resource | {"error": {"message": message, "type": error_type}}


def _msg_item(
    msg_id: str,
    text: str,
    status: str = "in_progress",
    *,
    with_content: bool = True,
) -> dict[str, object]:
    """Build a Responses message output item.

    Returns:
        The Responses message output item dict.

    """
    item: dict[str, object] = {
        "id": msg_id,
        "type": "message",
        "status": status,
        "role": "assistant",
    }
    if with_content:
        item["content"] = [{"type": "output_text", "text": text, "annotations": []}]
    else:
        item["content"] = []
    return item


@dataclass
class _ResponsesStreamState:
    """Mutable cursor for Responses event framing."""

    rid: str
    msg_id: str
    default_model: str
    prev: str | None
    seq: int = 0
    text_acc: str = ""
    result: TurnResult | None = None

    def frame(self, etype: str, **kw: object) -> str:
        """Encode one Responses event with the next sequence number.

        Returns:
            The SSE-encoded Responses event.

        """
        self.seq += 1
        payload: dict[str, object] = {"type": etype, "sequence_number": self.seq}
        payload.update(kw)
        return _sse(payload)

    def opening_chunks(self) -> Iterator[str]:
        """Yield the opening Responses envelope events.

        Yields:
            Opening envelope events (created, in-progress, item/part added).

        """
        in_prog = _response_resource(
            self.rid,
            self.default_model,
            int(time.time()),
            self.prev,
            _ResponseContent(status="in_progress", output_items=[], usage=None),
        )
        yield self.frame("response.created", response=in_prog)
        yield self.frame("response.in_progress", response=in_prog)
        yield self.frame(
            "response.output_item.added",
            output_index=0,
            item=_msg_item(self.msg_id, "", with_content=False),
        )
        yield self.frame(
            "response.content_part.added",
            item_id=self.msg_id,
            output_index=0,
            content_index=0,
            part={"type": "output_text", "text": "", "annotations": []},
        )

    def event_chunks(self, ev: dict[str, object]) -> Iterator[str]:
        """Yield SSE chunks for one turn event, advancing the cursor.

        Yields:
            SSE-encoded Responses events for the turn event (possibly none).

        """
        etype = ev["type"]
        if etype == "delta":
            text = ev["text"]
            if not isinstance(text, str):
                return
            self.text_acc += text
            yield self.frame(
                "response.output_text.delta",
                item_id=self.msg_id,
                output_index=0,
                content_index=0,
                delta=text,
                logprobs=[],
                obfuscation=None,
            )
        elif etype == "done":
            done_result = ev["result"]
            if isinstance(done_result, TurnResult):
                self.result = done_result
        elif etype == "error":
            err = ev["error"]
            if isinstance(err, EngineError):
                yield self.frame(
                    "response.failed",
                    response=_failed_response_resource(
                        self.rid,
                        self.default_model,
                        self.prev,
                        err.message,
                        err.error_type,
                    ),
                )
            yield "data: [DONE]\n\n"

    def closing_chunks(self) -> Iterator[str]:
        """Yield the completed-response envelope events for the stored result.

        Yields:
            Completed envelope events, or a failed envelope when no result.

        """
        if self.result is None:
            yield self.frame(
                "response.failed",
                response=_failed_response_resource(
                    self.rid,
                    self.default_model,
                    self.prev,
                    "engine returned no result",
                    "server_error",
                ),
            )
            yield "data: [DONE]\n\n"
            return
        full_part = {"type": "output_text", "text": self.text_acc, "annotations": []}
        yield self.frame(
            "response.output_text.done",
            item_id=self.msg_id,
            output_index=0,
            content_index=0,
            text=self.text_acc,
        )
        yield self.frame(
            "response.content_part.done",
            item_id=self.msg_id,
            output_index=0,
            content_index=0,
            part=full_part,
        )
        done_item = _msg_item(self.msg_id, self.text_acc, status="completed")
        yield self.frame("response.output_item.done", output_index=0, item=done_item)
        final = _response_resource(
            self.result.response_id,
            self.result.model,
            self.result.created,
            self.prev,
            _ResponseContent(
                status="completed",
                output_items=[done_item],
                usage={
                    "input_tokens": self.result.prompt_tokens,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": self.result.completion_tokens,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": self.result.prompt_tokens
                    + self.result.completion_tokens,
                },
            ),
        )
        yield self.frame("response.completed", response=final)
        yield "data: [DONE]\n\n"


async def responses_stream(
    parsed: ParsedRequest, prev: str | None, preferred: str | None = None
) -> AsyncIterator[str]:
    """Stream Responses API events for a parsed request.

    Yields:
        SSE-encoded Responses API events ending with a done sentinel.

    """
    state = _ResponsesStreamState(
        rid="resp_" + uuid.uuid4().hex,
        msg_id="msg_" + uuid.uuid4().hex,
        default_model=parsed.model_requested or "auto",
        prev=prev,
    )
    for chunk in state.opening_chunks():
        yield chunk
    events = _turn_events(parsed, previous_response_id=prev, preferred=preferred)
    ticked_events = keepalive_events(events)
    try:
        async for ticked in ticked_events:
            if ticked is None:
                yield KEEPALIVE_PING
                continue
            failed = ticked["type"] == "error"
            for chunk in state.event_chunks(ticked):
                yield chunk
            if failed:
                return
    finally:
        await ticked_events.aclose()
    for chunk in state.closing_chunks():
        yield chunk


@app.post("/v1/responses", response_model=None)
@app.post("/responses", response_model=None)
async def responses_api(
    request: Request,
) -> JSONResponse | StreamingResponse | dict[str, object]:
    """Handle a Responses API request, streaming or buffered.

    Returns:
        The streaming response when requested, else the buffered resource.

    """
    if not auth_ok(request):
        return oai_error(401, "invalid api key", "authentication_error")
    body = await _body(request)
    try:
        parsed, prev, _store = await parse_responses_request(body)
    except (ValueError, KeyError) as e:
        return oai_error(400, str(e))
    preferred = request.headers.get("x-chatgpt-account") or None
    if bool(body.get("stream")):
        return StreamingResponse(
            responses_stream(parsed, prev, preferred),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    result = await collect(
        parsed,
        POOL,
        previous_response_id=prev,
        preferred_email=request.headers.get("x-chatgpt-account") or None,
    )
    msg_id = "msg_" + uuid.uuid4().hex
    item = _msg_item(msg_id, result.text, status="completed")
    usage: dict[str, object] = {
        "input_tokens": result.prompt_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": result.completion_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": result.prompt_tokens + result.completion_tokens,
    }
    return _response_resource(
        result.response_id,
        result.model,
        result.created,
        prev,
        _ResponseContent(status="completed", output_items=[item], usage=usage),
    )


# ------------------------------------------------------------------ misc


@app.get("/healthz", response_model=None)
async def healthz() -> dict[str, object]:
    """Report pool and Redis cache health.

    Returns:
        The pool health, account count, and Redis status/summary.

    """
    avail = len(POOL.available())
    payload: dict[str, object] = {
        "status": "ok" if avail else "degraded",
        "accounts_available": avail,
    }
    payload["redis"] = await asyncio.to_thread(_redis_health)
    return payload


def _redis_health() -> dict[str, object]:
    """Probe the shared Redis cache with observability signals.

    Runs in a worker thread (called via ``asyncio.to_thread``) so the sync
    ping/info round trips never block the event loop. Info is skipped when
    ping fails so an unreachable Redis costs one timeout, not two.

    Returns:
        The Redis status, latency, cache counters, and server summary.
    """
    cache = get_cache()
    if cache is None:
        return {"enabled": False, "status": "disabled"}
    latency = cache.ping_ms()
    if latency is None:
        return {
            "enabled": True,
            "status": "unreachable",
            "cache": cache.stats(),
        }
    health: dict[str, object] = {
        "enabled": True,
        "status": "ok",
        "latency_ms": round(latency, 2),
        "cache": cache.stats(),
    }
    summary = cache.info_summary()
    if summary:
        health["server"] = summary
    return health


@app.get("/v1/accounts", response_model=None)
async def accounts_snapshot(request: Request) -> JSONResponse | dict[str, object]:
    """Return a snapshot of pooled accounts, masking emails when open.

    Returns:
        The account snapshot payload.

    """
    if not auth_ok(request):
        return oai_error(401, "invalid api key", "authentication_error")
    snap = POOL.snapshot()
    if not config.API_KEY:
        # only mask identities when there is no gate in front of the endpoint
        for account in snap:
            email = account.get("email")
            if isinstance(email, str):
                account["email"] = email[:2] + "***"
    return {"accounts": snap}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")
