# Copyright 2026 chatgpt-to-openai-api contributors.
"""FastAPI server exposing ChatGPT web accounts as an OpenAI-compatible API.

Endpoints:
  GET  /v1/models
  POST /v1/chat/completions   (stream + non-stream)
  POST /v1/responses          (stream + non-stream, previous_response_id)
  GET  /v1/accounts           (pool debug snapshot)
"""

from __future__ import annotations

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

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("main")

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
    """Check whether the request carries a valid API key."""
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
    """Build an OpenAI-style JSON error response."""
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
async def engine_error_handler(_request: Request, exc: EngineError) -> JSONResponse:
    """Translate an EngineError into an OpenAI-style error response."""
    status = exc.status if exc.status >= _HTTP_BAD_REQUEST else _HTTP_BAD_GATEWAY
    return oai_error(status, exc.message, exc.error_type)


async def _body(request: Request) -> dict[str, object]:
    """Parse the request body as a JSON object."""
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
    """Return the live model list, falling back to a static entry."""
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
    """Encode a payload as a server-sent event line."""
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


async def chat_stream(
    parsed: ParsedRequest, *, include_usage: bool, preferred: str | None = None
) -> AsyncIterator[str]:
    """Stream chat completion chunks for a parsed request."""
    cid = "chatcmpl-" + uuid.uuid4().hex
    created = int(time.time())
    model = parsed.model_requested or "auto"
    first = True
    try:
        async for ev in run_turn(parsed, POOL, preferred_email=preferred):
            if ev["type"] == "model":
                model_name = ev["model"]  # resolved live slug, stamped on chunks
                if isinstance(model_name, str):
                    model = model_name
            elif ev["type"] == "delta":
                t = ev["text"]
                if not isinstance(t, str):
                    continue
                # first chunk carries role + any initial content (never drop text)
                delta = {"role": "assistant", "content": t} if first else {"content": t}
                chunk: dict[str, object] = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
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
                first = False
                yield _sse(chunk)
            elif ev["type"] == "done":
                res = ev["result"]
                if not isinstance(res, TurnResult):
                    continue
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
    except EngineError as e:
        # headers are already sent; deliver the failure in-band so clients see a
        # well-formed error instead of a truncated stream
        yield _sse(
            {
                "error": {
                    "message": e.message,
                    "type": e.error_type,
                    "param": None,
                    "code": str(e.status),
                },
            },
        )
        yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions", response_model=None)
@app.post("/chat/completions", response_model=None)
async def chat_completions(
    request: Request,
) -> JSONResponse | StreamingResponse | dict[str, object]:
    """Handle a chat completion request, streaming or buffered."""
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
    """Build a Responses API resource envelope."""
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
    """Build a failed Responses API resource envelope."""
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
    """Build a Responses message output item."""
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


async def responses_stream(
    parsed: ParsedRequest, prev: str | None, preferred: str | None = None
) -> AsyncIterator[str]:
    """Stream Responses API events for a parsed request."""
    seq = 0
    rid = "resp_" + uuid.uuid4().hex
    msg_id = "msg_" + uuid.uuid4().hex

    def ev(etype: str, **kw: object) -> str:
        nonlocal seq
        seq += 1
        payload: dict[str, object] = {"type": etype, "sequence_number": seq}
        payload.update(kw)
        return _sse(payload)

    default_model = parsed.model_requested or "auto"
    in_prog = _response_resource(
        rid,
        default_model,
        int(time.time()),
        prev,
        _ResponseContent(status="in_progress", output_items=[], usage=None),
    )
    yield ev("response.created", response=in_prog)
    yield ev("response.in_progress", response=in_prog)
    yield ev(
        "response.output_item.added",
        output_index=0,
        item=_msg_item(msg_id, "", with_content=False),
    )
    yield ev(
        "response.content_part.added",
        item_id=msg_id,
        output_index=0,
        content_index=0,
        part={"type": "output_text", "text": "", "annotations": []},
    )

    text_acc = ""
    result: TurnResult | None = None
    try:
        async for e in run_turn(
            parsed,
            POOL,
            previous_response_id=prev,
            response_id_prefix="resp_",
            preferred_email=preferred,
        ):
            if e["type"] == "delta":
                text = e["text"]
                if not isinstance(text, str):
                    continue
                text_acc += text
                yield ev(
                    "response.output_text.delta",
                    item_id=msg_id,
                    output_index=0,
                    content_index=0,
                    delta=text,
                    logprobs=[],
                    obfuscation=None,
                )
            elif e["type"] == "done":
                done_result = e["result"]
                if isinstance(done_result, TurnResult):
                    result = done_result
    except EngineError as e:
        yield ev(
            "response.failed",
            response=_failed_response_resource(
                rid, default_model, prev, e.message, e.error_type
            ),
        )
        yield "data: [DONE]\n\n"
        return

    if result is None:
        yield ev(
            "response.failed",
            response=_failed_response_resource(
                rid, default_model, prev, "engine returned no result", "server_error"
            ),
        )
        yield "data: [DONE]\n\n"
        return

    full_part = {"type": "output_text", "text": text_acc, "annotations": []}
    yield ev(
        "response.output_text.done",
        item_id=msg_id,
        output_index=0,
        content_index=0,
        text=text_acc,
    )
    yield ev(
        "response.content_part.done",
        item_id=msg_id,
        output_index=0,
        content_index=0,
        part=full_part,
    )
    done_item = _msg_item(msg_id, text_acc, status="completed")
    yield ev("response.output_item.done", output_index=0, item=done_item)

    final = _response_resource(
        result.response_id,
        result.model,
        result.created,
        prev,
        _ResponseContent(
            status="completed",
            output_items=[done_item],
            usage={
                "input_tokens": result.prompt_tokens,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": result.completion_tokens,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": result.prompt_tokens + result.completion_tokens,
            },
        ),
    )
    yield ev("response.completed", response=final)
    yield "data: [DONE]\n\n"


@app.post("/v1/responses", response_model=None)
@app.post("/responses", response_model=None)
async def responses_api(
    request: Request,
) -> JSONResponse | StreamingResponse | dict[str, object]:
    """Handle a Responses API request, streaming or buffered."""
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
    """Report pool health."""
    avail = len(POOL.available())
    return {"status": "ok" if avail else "degraded", "accounts_available": avail}


@app.get("/v1/accounts", response_model=None)
async def accounts_snapshot(request: Request) -> JSONResponse | dict[str, object]:
    """Return a snapshot of pooled accounts, masking emails when open."""
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
