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

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import config
from .accounts import POOL
from .adapters import parse_chat_request, parse_responses_request, public_models

from .engine import EngineError, collect, run_turn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await POOL.start_watcher()
    yield


app = FastAPI(title="chatgpt-to-openai-api", version="1.0.0", lifespan=lifespan)


def auth_ok(request: Request) -> bool:
    if not config.API_KEY:
        return True
    auth = request.headers.get("authorization", "")
    expected = f"Bearer {config.API_KEY}"
    return hmac.compare_digest(auth.encode(), expected.encode())


def oai_error(status: int, message: str, err_type: str = "invalid_request_error", code: str | None = None):
    return JSONResponse(status_code=status, content={
        "error": {"message": message, "type": err_type, "param": None, "code": code}
    })


@app.exception_handler(EngineError)
async def engine_error_handler(request: Request, exc: EngineError):
    return oai_error(exc.status if exc.status >= 400 else 502, exc.message,
                     exc.error_type)


async def _body(request: Request) -> dict:
    try:
        raw = await request.body()
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise EngineError(400, "request body must be valid JSON")
    if not isinstance(parsed, dict):
        raise EngineError(400, "request body must be a JSON object")
    return parsed


# ------------------------------------------------------------------ models

@app.get("/v1/models")
@app.get("/models")
async def list_models(request: Request):
    if not auth_ok(request):
        return oai_error(401, "invalid api key", "authentication_error")
    fallback = {"object": "list", "data": [{"id": "auto", "object": "model", "created": 1785000000, "owned_by": "chatgpt-proxy"}]}
    try:
        acct = POOL.acquire(None)
    except Exception:
        return fallback
    try:
        return {"object": "list", "data": public_models(await acct.models())}
    except Exception:
        return fallback
    finally:
        POOL.release(acct)


# ------------------------------------------------------------------ chat completions

def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


async def chat_stream(parsed, include_usage: bool, preferred: str | None = None):
    cid = "chatcmpl-" + uuid.uuid4().hex
    created = int(time.time())
    model = parsed.model_requested or "auto"
    first = True
    try:
        async for ev in run_turn(parsed, POOL, preferred_email=preferred):
            if ev["type"] == "delta":
                t = ev["text"]
                chunk = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "system_fingerprint": None,
                    "choices": [{"index": 0, "delta": ({"role": "assistant", "content": ""} if first else {"content": t}),
                                 "logprobs": None, "finish_reason": None}],
                }
                first = False
                yield _sse(chunk)
            elif ev["type"] == "done":
                res = ev["result"]
                usage_block = {
                    "prompt_tokens": res.prompt_tokens,
                    "completion_tokens": res.completion_tokens,
                    "total_tokens": res.prompt_tokens + res.completion_tokens,
                }
                if include_usage:
                    yield _sse({
                        "id": cid, "object": "chat.completion.chunk",
                        "created": created, "model": model, "system_fingerprint": None,
                        "choices": [{"index": 0, "delta": {}, "logprobs": None, "finish_reason": "stop"}],
                    })
                    yield _sse({
                        "id": cid, "object": "chat.completion.chunk",
                        "created": created, "model": model, "system_fingerprint": None,
                        "choices": [], "usage": usage_block,
                    })
                else:
                    yield _sse({
                        "id": cid, "object": "chat.completion.chunk",
                        "created": created, "model": model, "system_fingerprint": None,
                        "choices": [{"index": 0, "delta": {}, "logprobs": None, "finish_reason": "stop"}],
                    })
                yield "data: [DONE]\n\n"
    except EngineError as e:
        # headers are already sent; deliver the failure in-band so clients see a
        # well-formed error instead of a truncated stream
        yield _sse({"error": {"message": e.message, "type": e.error_type, "param": None,
                              "code": str(e.status)}})
        yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
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
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        return StreamingResponse(chat_stream(parsed, include_usage, preferred),
                                 media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    try:
        result = await collect(parsed, POOL, preferred_email=preferred)
    except EngineError:
        raise
    return {
        "id": "chatcmpl-" + result.response_id[5:],
        "object": "chat.completion",
        "created": result.created,
        "model": result.model,
        "system_fingerprint": None,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result.text, "refusal": None},
            "logprobs": None,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        },
    }


# ------------------------------------------------------------------ responses

def _response_resource(rid: str, model: str, created: int, status: str, output_items: list,
                       prev: str | None, usage: dict | None) -> dict:
    r = {
        "id": rid,
        "object": "response",
        "created_at": created,
        "status": status,
        "background": False,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "metadata": {},
        "model": model,
        "output": output_items,
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
        "usage": usage,
        "user": None,
    }
    return r


def _msg_item(msg_id: str, text: str, status: str = "in_progress", with_content=True):
    item = {"id": msg_id, "type": "message", "status": status, "role": "assistant"}
    if with_content:
        item["content"] = [{"type": "output_text", "text": text, "annotations": []}]
    else:
        item["content"] = []
    return item


async def responses_stream(parsed, prev, preferred: str | None = None):
    seq = 0
    rid = "resp_" + uuid.uuid4().hex
    msg_id = "msg_" + uuid.uuid4().hex

    def ev(etype: str, **kw) -> str:
        nonlocal seq
        seq += 1
        payload = {"type": etype, "sequence_number": seq}
        payload.update(kw)
        return _sse(payload)

    in_prog = _response_resource(rid, parsed.model_requested or "auto", int(time.time()),
                                 "in_progress", [], prev, None)
    yield ev("response.created", response=in_prog)
    yield ev("response.in_progress", response=in_prog)
    yield ev("response.output_item.added", output_index=0, item=_msg_item(msg_id, "", with_content=False))
    yield ev("response.content_part.added", item_id=msg_id, output_index=0, content_index=0,
             part={"type": "output_text", "text": "", "annotations": []})

    text_acc = ""
    result = None
    try:
        async for e in run_turn(parsed, POOL, previous_response_id=prev, response_id_prefix="resp_", preferred_email=preferred):
            if e["type"] == "delta":
                text_acc += e["text"]
                yield ev("response.output_text.delta", item_id=msg_id, output_index=0, content_index=0,
                         delta=e["text"], logprobs=[], obfuscation=None)
            elif e["type"] == "done":
                result = e["result"]
    except EngineError as e:
        yield ev("response.failed", response=_response_resource(
            rid, parsed.model_requested or "auto", int(time.time()), "failed", [], prev,
            None) | {"error": {"message": e.message, "type": e.error_type}})
        yield "data: [DONE]\n\n"
        return

    full_part = {"type": "output_text", "text": text_acc, "annotations": []}
    yield ev("response.output_text.done", item_id=msg_id, output_index=0, content_index=0, text=text_acc)
    yield ev("response.content_part.done", item_id=msg_id, output_index=0, content_index=0, part=full_part)
    done_item = _msg_item(msg_id, text_acc, status="completed")
    yield ev("response.output_item.done", output_index=0, item=done_item)

    final = _response_resource(
        result.response_id, result.model, result.created, "completed", [done_item], prev,
        {"input_tokens": result.prompt_tokens, "input_tokens_details": {"cached_tokens": 0},
         "output_tokens": result.completion_tokens,
         "output_tokens_details": {"reasoning_tokens": 0},
         "total_tokens": result.prompt_tokens + result.completion_tokens})
    yield ev("response.completed", response=final)
    yield "data: [DONE]\n\n"


@app.post("/v1/responses")
@app.post("/responses")
async def responses_api(request: Request):
    if not auth_ok(request):
        return oai_error(401, "invalid api key", "authentication_error")
    body = await _body(request)
    try:
        parsed, prev, _store = await parse_responses_request(body)
    except (ValueError, KeyError) as e:
        return oai_error(400, str(e))
    preferred = request.headers.get("x-chatgpt-account") or None
    if bool(body.get("stream")):
        return StreamingResponse(responses_stream(parsed, prev, preferred),
                                 media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    result = await collect(parsed, POOL, previous_response_id=prev,
                           preferred_email=request.headers.get("x-chatgpt-account") or None)
    msg_id = "msg_" + uuid.uuid4().hex
    item = _msg_item(msg_id, result.text, status="completed")
    usage = {
        "input_tokens": result.prompt_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": result.completion_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": result.prompt_tokens + result.completion_tokens,
    }
    return _response_resource(result.response_id, result.model, result.created, "completed",
                              [item], prev, usage)


# ------------------------------------------------------------------ misc

@app.get("/healthz")
async def healthz():
    avail = len(POOL.available())
    return {"status": "ok" if avail else "degraded", "accounts_available": avail}


@app.get("/v1/accounts")
async def accounts_snapshot(request: Request):
    if not auth_ok(request):
        return oai_error(401, "invalid api key", "authentication_error")
    snap = POOL.snapshot()
    if not config.API_KEY:
        # only mask identities when there is no gate in front of the endpoint
        for a in snap:
            a["email"] = a["email"][:2] + "***"
    return {"accounts": snap}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")
