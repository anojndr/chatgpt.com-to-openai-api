"""Shared execution path used by both the Chat Completions and Responses APIs.

Streaming-first: emits text deltas as ChatGPT produces them, resolves generated
images to PixelVault URLs at the end, and records conversation state for
proper multi-turn continuation.
"""
from __future__ import annotations


import io
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from PIL import Image

from .adapters import HistoryItem, ParsedRequest, estimate_tokens, map_model
from .accounts import AccountPool, NoAccountAvailable
from .chatgpt import AccountSession, ChatGPTError
from .pixelvault import PixelVaultError, upload_image
from .store import ConvRef, ResponseRecord, STORE, item_hash

log = logging.getLogger("engine")


@dataclass
class TurnResult:
    text: str
    conversation_id: str
    parent_id: str
    model: str
    created: int
    response_id: str
    image_urls: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    account_email: str = ""


class EngineError(Exception):
    def __init__(self, status: int, message: str, error_type: str = "server_error"):
        super().__init__(message)
        self.status = status
        self.message = message
        self.error_type = error_type


async def _upload_inputs(acct: AccountSession, item: HistoryItem) -> tuple[list[dict], list[dict]]:
    """Upload the current turn's images and non-text files."""
    pointers: list[dict] = []
    attachments: list[dict] = []
    for im in item.images:
        try:
            with Image.open(io.BytesIO(im.data)) as p:
                width, height = p.size
        except Exception:
            raise EngineError(400, f"unsupported or corrupt image: {im.filename}", "invalid_request_error")
        fid = await acct.upload_file(im.filename, im.data, im.mime, is_image=True)
        pointer: dict[str, Any] = {
            "content_type": "image_asset_pointer",
            "asset_pointer": f"file-service://{fid}",
            "size_bytes": len(im.data),
            "width": width,
            "height": height,
        }
        pointers.append(pointer)
    for f in item.files:
        fid = await acct.upload_file(f.filename, f.data, f.mime, is_image=False)
        ready = await acct.wait_file_ready(fid)
        if not ready:
            # ChatGPT often processes lazily; attachments still work pre-ready.
            log.info("file %s not 'ready' yet; sending anyway", f.filename)
        attachments.append({"id": fid, "name": f.filename, "mimeType": f.mime, "size": len(f.data)})
    return pointers, attachments


def _history_hashes(items: list[HistoryItem], system_text: str) -> list[str]:
    prev = item_hash("", "system", system_text) if system_text else ""
    out = []
    for item in items:
        prev = item_hash(prev, item.role, item.canon())
        out.append(prev)
    return out


async def _resolve_images(acct: AccountSession, pointers: list[str]) -> list[str]:
    urls: list[str] = []
    for ptr in pointers:
        if not ptr.startswith("sediment://"):
            continue
        try:
            name, data = await acct.download_file_url(ptr)
            mime = "image/jpeg" if name.lower().endswith((".jpg", ".jpeg")) else "image/png"
            urls.append(await upload_image(name, data, mime))
        except (ChatGPTError, PixelVaultError) as e:
            log.warning("image resolution failed (%s): %s", ptr, e)
    return urls


def _plan(parsed: ParsedRequest, previous_response_id: str | None) -> tuple[list[str], ConvRef | None, int]:
    """Decide which stored server conversation to continue and how much of the
    request history is new."""
    hashes = _history_hashes(parsed.items, parsed.system_text)
    if previous_response_id:
        rec = STORE.get_response(previous_response_id)
        if rec is None:
            raise EngineError(400, f"previous_response_id '{previous_response_id}' not found or expired",
                              "invalid_request_error")
        ref = ConvRef(rec.account_identity, rec.conversation_id, rec.parent_id, 0, time.time())
        # Client sends only NEW input items for stateful calls.
        return hashes, ref, max(len(parsed.items) - 1, 0)

    match = STORE.find(hashes)
    matched = match[0] if match else 0
    ref = match[1] if match else None
    if ref is not None and matched >= len(parsed.items):
        # Exact duplicate request: re-ask the final user turn in the same thread.
        matched = len(parsed.items) - 1
        rematch = STORE.find(hashes[:matched]) if matched else None
        ref = rematch[1] if rematch else None
    return hashes, ref, matched


async def run_turn(
    parsed: ParsedRequest,
    pool: AccountPool,
    *,
    previous_response_id: str | None = None,
    response_id_prefix: str = "resp_",
    preferred_email: str | None = None,
) -> AsyncIterator[dict]:
    """Yield {'type': 'delta', 'text': ...} events, then {'type': 'done', 'result': TurnResult}."""
    if not parsed.items:
        raise EngineError(400, "no messages", "invalid_request_error")
    if parsed.items[-1].role != "user":
        raise EngineError(400, "last message must have role=user", "invalid_request_error")

    hashes, ref, matched = _plan(parsed, previous_response_id)
    current = parsed.items[-1]
    log.info("turn plan: history=%d matched=%d %s", len(parsed.items), matched,
             f"continue conv={ref.conversation_id}" if ref else "new conversation")

    preferred = ref.account_identity if ref else (preferred_email or None)
    max_attempts = 1 if ref else 3  # continuations are account-bound; fresh convs can move
    rid = response_id_prefix + uuid.uuid4().hex

    for attempt in range(1, max_attempts + 1):
        try:
            acct = pool.acquire(preferred)
        except NoAccountAvailable as e:
            raise EngineError(503, str(e), "rate_limit_error") from e

        produced = False  # never switch accounts after content started flowing
        try:
            acct.total_requests += 1
            live_slugs = {m.get("slug") for m in await acct.models() if m.get("slug")}
            model = map_model(parsed.model_requested, live_slugs)
            pointers, attachments = await _upload_inputs(acct, current)

            prompt = current.text
            if parsed.system_text and ref is None:
                # Instructions ride once, on the first turn of a fresh conversation.
                prompt = f"{parsed.system_text}\n\n{prompt}" if prompt else parsed.system_text

            text_acc = ""
            emitted = 0
            current_msg_id = ""
            sediment: list[str] = []
            cid = ""
            parent = ""

            async for ev in acct.stream_conversation(
                prompt_text=prompt,
                image_pointers=pointers,
                attachments=attachments,
                parent_message_id=ref.parent_id if ref else str(uuid.uuid4()),
                conversation_id=ref.conversation_id if ref else None,
                model=model,
            ):
                if ev.get("conversation_id") and not cid:
                    cid = ev["conversation_id"]
                m = ev.get("message") or {}
                author = m.get("author") or {}
                if author.get("role") not in ("assistant", "tool"):
                    continue
                if author.get("role") == "assistant" and m.get("id") and m["id"] != current_msg_id:
                    # new assistant node: restart delta accounting so its text streams
                    current_msg_id = m["id"]
                    emitted = 0
                    text_acc = ""
                content = m.get("content") or {}
                parts = content.get("parts") or []
                for p in parts:
                    if isinstance(p, dict):
                        ptr = p.get("asset_pointer")
                        if isinstance(ptr, str) and "sediment://" in ptr and ptr not in sediment:
                            sediment.append(ptr)
                    elif isinstance(p, str) and author.get("role") == "assistant":
                        text_acc = p
                        if len(text_acc) > emitted:
                            produced = True
                            yield {"type": "delta", "text": text_acc[emitted:]}
                            emitted = len(text_acc)
                meta = m.get("metadata") or {}
                if meta.get("message_type") in ("next", "continue"):
                    parent = m.get("id") or parent

            if not parent or not cid:
                raise EngineError(502, "ChatGPT returned an incomplete response")

            image_urls = await _resolve_images(acct, sediment)
            if image_urls:
                links = "\n\n" + "\n\n".join(f"![generated image]({u})" for u in image_urls)
                produced = True
                yield {"type": "delta", "text": links}
                text_acc += links

            created = int(time.time())
            new_ref = ConvRef(acct.identity, cid, parent, len(parsed.items), time.time())
            STORE.record_turn(hashes, new_ref)
            STORE.put_response(ResponseRecord(rid, acct.identity, cid, parent, model, time.time()))
            yield {"type": "done", "result": TurnResult(
                text=text_acc, conversation_id=cid, parent_id=parent, model=model,
                created=created, response_id=rid, image_urls=image_urls,
                prompt_tokens=estimate_tokens(parsed.system_text, *(x.text for x in parsed.items)),
                completion_tokens=estimate_tokens(text_acc),
                account_email=acct.email,
            )}
            return  # success

        except ChatGPTError as e:
            pool.report_status(acct, e.status)
            if produced or attempt >= max_attempts:
                raise EngineError(
                    429 if e.status == 429 else (502 if e.status >= 500 else 502),
                    e.message,
                    "rate_limit_error" if e.status == 429 else "server_error",
                ) from e
            log.warning("attempt %d/%d on %s failed (%s %s); rotating account",
                        attempt, max_attempts, acct.email, e.status, e.message[:120])
        finally:
            pool.release(acct)


async def collect(parsed: ParsedRequest, pool: AccountPool, *, previous_response_id: str | None = None,
                  preferred_email: str | None = None) -> TurnResult:
    """Non-streaming convenience wrapper."""
    result: TurnResult | None = None
    async for ev in run_turn(parsed, pool, previous_response_id=previous_response_id,
                             preferred_email=preferred_email):
        if ev["type"] == "done":
            result = ev["result"]
    if result is None:
        raise EngineError(502, "no result")
    return result
