"""Shared execution path used by both the Chat Completions and Responses APIs.

Streaming-first: emits text deltas as ChatGPT produces them, resolves generated
images to PixelVault URLs at the end, and records conversation state for
proper multi-turn continuation.
"""
from __future__ import annotations


import io
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from PIL import Image

from .adapters import HistoryItem, ParsedRequest, estimate_tokens, map_model
from .accounts import AccountPool, NoAccountAvailable
from .chatgpt import AccountSession, ChatGPTError
from .pixelvault import PixelVaultError, upload_image
from .store import ConvRef, ResponseRecord, TurnSnapshot, STORE, item_hash

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

class _ImageLimitError(Exception):
    """The model replied with a per-account image-generation quota refusal.

    Raised BEFORE any byte reaches the client, so the failover loop may serve
    the same request on another account.
    """


# Matches e.g. "You've hit the Free plan limit for image generations requests.
# You can create more images when the limit resets in 9 hours and 38 minutes."
# Plan name and reset time vary; the fixed head/tail phrases do not.


# Anchored to the reply head (<=64 lead-in chars) so a reply that merely QUOTES
# the template mid-text is never mistaken for a refusal.

# Anchored to the very head of the reply (only whitespace/bullet/markdown
# flourish may precede it), so a reply that merely QUOTES the template
# mid-text -- even after a short prose lead-in -- is never mistaken for one.
IMAGE_LIMIT_RE = re.compile(
    r"\A[\s>*#\-`]{0,16}you(?:'ve| have)\s+hit\s+(?:the|your)\s+"
    r".*?plan\s+limit\s+for\s+image\s+generations?\s+requests",
    re.IGNORECASE | re.DOTALL,
)
_LIMIT_STARTERS = ("you've hit the ", "you have hit the ", "you've hit your ")
_PLAN_WORDS = ("free", "plus", "pro", "team", "enterprise")
_LIMIT_MAX_LEN = 220  # refusal sentences stay well under this; longer text can't be one


def _classify_accumulated(text: str) -> tuple[int, bool]:
    """Decide how much of the buffered assistant text is safe to stream.

    Returns (chars_safe_to_emit_now, is_image_limit_refusal). While the text
    could still grow into the quota refusal, nothing is released; the moment
    it diverges from every refusal opening (or exceeds plausible length), the
    whole buffer is released so normal streaming is never visibly delayed.
    """
    if IMAGE_LIMIT_RE.search(text):
        return 0, True
    if len(text) > _LIMIT_MAX_LEN:
        return len(text), False
    low = text.lower()
    for starter in _LIMIT_STARTERS:
        if low.startswith(starter):
            # Fixed head matched; only a plan-name-looking next word can
            # still grow into the refusal ("Free"/"Plus"/...).
            rest = low[len(starter):]
            m = re.match(r"[a-z]+", rest)
            word = m.group(0) if m else ""
            if not word or any(w.startswith(word) or word.startswith(w) for w in _PLAN_WORDS):
                return 0, False
            return len(text), False  # "You've hit the nail..." -> normal reply
    # Still compatible with some refusal opening? Hold until it diverges.
    for starter in _LIMIT_STARTERS:
        n = min(len(low), len(starter))
        i = 0
        while i < n and low[i] == starter[i]:
            i += 1
        if i == n:  # consumed the whole text within this opening
            return 0, False
    return len(text), False  # diverged from every opening -> cannot be the refusal



async def _upload_inputs(acct: AccountSession, items: list[HistoryItem], *,
                         strict_from: int = 0) -> tuple[list[dict], list[dict]]:
    """Upload every image and non-text file of the given turns to THIS account.

    Uploads are account-scoped: an attempt served by another account must
    re-upload them all -- that is how attachments survive account failover.

    strict_from: index of the first item whose images MUST validate. Items
    before it are replayed history: an image that no longer opens (e.g. an
    expired remote URL re-fetched at parse time) is skipped with a warning
    instead of turning a servable conversation into a hard 400.
    """
    pointers: list[dict] = []
    attachments: list[dict] = []
    for idx, item in enumerate(items):
        for im in item.images:
            try:
                with Image.open(io.BytesIO(im.data)) as p:
                    width, height = p.size
            except Exception:
                if idx >= strict_from:
                    raise EngineError(400, f"unsupported or corrupt image: {im.filename}",
                                      "invalid_request_error")
                log.warning("skipping unopenable history image %s during replay", im.filename)
                continue
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


def _replay_prompt(system_text: str, items: list[HistoryItem]) -> str:
    """Render the whole transcript for a fresh conversation on a non-owner
    account, preserving every turn. Text-like files were already fenced into
    their turn's text at parse time; images/binary files attach separately."""
    blocks: list[str] = []
    if system_text:
        blocks.append(f"[system]\n{system_text}")
    for it in items:
        blocks.append(f"[{it.role}]\n{it.text}")
    return "\n\n".join(blocks)


def _chain_context(prev_snap: TurnSnapshot, parsed: ParsedRequest,
                   hashes: list[str]) -> tuple[str, list[HistoryItem]]:
    """Merge a referenced response's recorded context with this request's turns.

    Clients of previous_response_id flows usually send only NEW items per call;
    if instead they resent the full history (hash chain extends the snapshot's),
    the request alone is already complete.
    """
    base_hashes = _history_hashes(prev_snap.items, prev_snap.system_text)
    client_resent_all = len(hashes) >= len(base_hashes) and hashes[:len(base_hashes)] == base_hashes
    if client_resent_all:
        return parsed.system_text, list(parsed.items)
    # Cumulative: ancestor context + this round. Items are shared references,
    # so store-wide byte accounting may over-count shared buffers -- that only
    # makes budget eviction earlier, never memory use larger than reality.
    return prev_snap.system_text, list(prev_snap.items) + list(parsed.items)


def _fallback_context(parsed: ParsedRequest, hashes: list[str],
                      prev_response_id: str | None) -> tuple[str, list[HistoryItem]]:
    """Full context to replay when serving the turn on a non-owner account."""
    ctx_sys, ctx_items = parsed.system_text, parsed.items
    if prev_response_id:
        snap = STORE.get_snapshot(prev_response_id)
        if snap is not None:
            ctx_sys, ctx_items = _chain_context(snap, parsed, hashes)
    return ctx_sys, ctx_items


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
    """Yield {'type': 'delta', 'text': ...} events, then {'type': 'done', 'result': TurnResult}.

    Every available account is tried exactly once before an error is raised.
    Attempts that cannot continue the stored server-side conversation start a
    fresh one and replay the FULL client context: every turn's text plus that
    turn's images and file attachments, re-uploaded to whichever account
    actually serves the attempt -- so failover never loses content.
    """
    if not parsed.items:
        raise EngineError(400, "no messages", "invalid_request_error")
    if parsed.items[-1].role != "user":
        raise EngineError(400, "last message must have role=user", "invalid_request_error")

    hashes, ref, matched = _plan(parsed, previous_response_id)
    current = parsed.items[-1]
    log.info("turn plan: history=%d matched=%d %s", len(parsed.items), matched,
             f"continue conv={ref.conversation_id}" if ref else "new conversation")

    rid = response_id_prefix + uuid.uuid4().hex

    # Attempt #1 prefers the stored conversation's owner / requested account;
    # afterwards EVERY remaining available account is tried before failing.
    preferred = ref.account_identity if ref else (preferred_email or None)
    tried: set[str] = set()
    last_failure: EngineError | None = None

    while True:
        try:
            acct = pool.acquire(preferred, exclude=tried)
        except NoAccountAvailable as e:
            if last_failure is not None:
                raise EngineError(
                    last_failure.status,
                    f"all {len(tried)} available account(s) failed; last error: {last_failure.message}",
                    last_failure.error_type,
                ) from e
            raise EngineError(503, str(e), "rate_limit_error") from e
        tried.add(acct.identity)
        preferred = None  # only the first attempt honors a preference

        # Continue the stored server-side conversation only on its owning
        # account; anywhere else start fresh and replay ALL client context --
        # every turn's text plus its images/files re-uploaded to this account.
        continuing = ref is not None and acct.identity == ref.account_identity
        produced = False  # never switch accounts after content started flowing
        try:
            acct.total_requests += 1
            live_slugs = {m.get("slug") for m in await acct.models() if m.get("slug")}
            model = map_model(parsed.model_requested, live_slugs)
            yield {"type": "model", "model": model}


            # Incremental only when this account owns the stored conversation,
            # or the request itself is complete context in one turn.
            if continuing or (ref is None and len(parsed.items) == 1):
                pointers, attachments = await _upload_inputs(acct, [current])
                prompt = current.text
                if parsed.system_text and ref is None:
                    # Instructions ride once, on the first turn of a fresh conversation.
                    prompt = f"{parsed.system_text}\n\n{prompt}" if prompt else parsed.system_text

            else:
                ctx_sys, ctx_items = _fallback_context(parsed, hashes, previous_response_id)
                # Everything the client sent THIS request validates strictly;
                # replayed history items only skip unopenable images.
                hist_len = len(ctx_items) - len(parsed.items)
                pointers, attachments = await _upload_inputs(acct, ctx_items,
                                                             strict_from=hist_len)
                prompt = _replay_prompt(ctx_sys, ctx_items)

            text_acc = ""
            emitted = 0
            current_msg_id = ""
            sediment: list[str] = []
            cid = ""
            parent = ""
            limit_hit = False

            async for ev in acct.stream_conversation(
                prompt_text=prompt,
                image_pointers=pointers,
                attachments=attachments,
                # Server-side ids are valid only on the conversation's owner;
                # replay attempts start a brand-new conversation instead.
                parent_message_id=ref.parent_id if continuing else str(uuid.uuid4()),
                conversation_id=ref.conversation_id if continuing else None,
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
                        # Withhold bytes while they could still be the image-limit
                        # refusal: once visible, accounts can no longer be switched.
                        emit_upto, is_limit = _classify_accumulated(text_acc)
                        if is_limit:
                            limit_hit = True
                            break
                        if emit_upto > emitted:
                            produced = True
                            yield {"type": "delta", "text": text_acc[emitted:emit_upto]}
                            emitted = emit_upto
                if limit_hit:
                    break
                meta = m.get("metadata") or {}
                if meta.get("message_type") in ("next", "continue"):
                    parent = m.get("id") or parent

            # Refusals are valid HTTP-200 streams and may intentionally stop
            # before ChatGPT emits the normal assistant "next" marker.
            if limit_hit or (text_acc and IMAGE_LIMIT_RE.search(text_acc)):
                raise _ImageLimitError(text_acc.strip()[:300])

            if not parent or not cid:
                raise EngineError(502, "ChatGPT returned an incomplete response")

            # Flush anything still withheld first: short normal replies that
            # begin like the refusal must reach the client in full.
            if emitted < len(text_acc):
                produced = True
                yield {"type": "delta", "text": text_acc[emitted:]}
                emitted = len(text_acc)

            image_urls = await _resolve_images(acct, sediment)
            if image_urls:
                links = "\n\n" + "\n\n".join(f"![generated image]({u})" for u in image_urls)
                produced = True
                yield {"type": "delta", "text": links}
                text_acc += links

            created = int(time.time())

            new_ref = ConvRef(acct.identity, cid, parent, len(parsed.items), time.time())
            STORE.record_turn(hashes, new_ref)
            # Record CUMULATIVE context: stateful clients send only new items per
            # call, so a delta-only snapshot would lose earlier turns on a later
            # failover (chained via previous_response_id).
            snap_sys, snap_items = parsed.system_text, list(parsed.items)
            if previous_response_id:
                prev_snap = STORE.get_snapshot(previous_response_id)
                if prev_snap is not None:
                    snap_sys, snap_items = _chain_context(prev_snap, parsed, hashes)
            STORE.put_response(
                ResponseRecord(rid, acct.identity, cid, parent, model, time.time()),
                TurnSnapshot(system_text=snap_sys, items=snap_items),
            )
            yield {"type": "done", "result": TurnResult(
                text=text_acc, conversation_id=cid, parent_id=parent, model=model,
                created=created, response_id=rid, image_urls=image_urls,
                prompt_tokens=estimate_tokens(parsed.system_text, *(x.text for x in parsed.items)),
                completion_tokens=estimate_tokens(text_acc),
                account_email=acct.email,
            )}
            return  # success

        except Exception as e:
            if isinstance(e, EngineError) and e.error_type == "invalid_request_error":
                raise  # deterministic request problem; no account can serve it
            if isinstance(e, _ImageLimitError):
                # Per-account image quota refusal delivered as a normal 200
                # stream. Nothing was shown to the client, so another account
                # may still serve this request; cool this one down like a 429.
                pool.report_status(acct, 429)
                failure = EngineError(429, str(e), "rate_limit_error")

            elif isinstance(e, ChatGPTError):
                pool.report_status(acct, e.status)
                failure = EngineError(
                    429 if e.status == 429 else 502,
                    e.message,
                    "rate_limit_error" if e.status == 429 else "server_error",
                )
            elif isinstance(e, EngineError):
                failure = e
            else:
                failure = EngineError(502, f"{type(e).__name__}: {e}")
            if produced:
                # Content already streamed to the client: rotating would
                # duplicate or contradict visible output.
                raise failure from e
            last_failure = failure
            remaining = sum(1 for a in pool.available() if a.identity not in tried)
            log.warning("attempt on %s failed (%s): %s; %d account(s) still untried",
                        acct.email, failure.error_type, failure.message[:160], remaining)
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
