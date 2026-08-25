"""Shared execution path used by both the Chat Completions and Responses APIs.

Streaming-first: emits text deltas as ChatGPT produces them, resolves generated
images to PixelVault URLs at the end, and records conversation state for
proper multi-turn continuation.
"""
from __future__ import annotations


import asyncio
import io
import json
import logging
import re
import time
import uuid
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from PIL import Image

from .adapters import HistoryItem, ParsedRequest, estimate_tokens, map_model
from . import config
from .accounts import AccountPool, NoAccountAvailable
from .charts import ChartError, chart_from_payload, render_chart_png
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
    r"\A[\s>*#\-`]{0,16}(?:"
    r"you(?:'ve| have)\s+hit\s+(?:the|your)\s+.*?plan\s+limit\s+for\s+image\s+generations?\s+requests"
    r"|"
    r"(?:image\s+(?:generation|editing)(?:\s+and\s+(?:image\s+)?(?:generation|editing))?|image\s+creation)\s+require(?:s)?\s+(?:you\s+to\s+be\s+|being\s+|to\s+be\s+)?log(?:ged|ging)\s+in"
    r"|"
    r"(?:i\s+can\s+(?:help\s+)?(?:create|generate)\s+that\s+image,\s+but\s+)?"
    r"image\s+generation\s+isn['’]t\s+available\s+in\s+this\s+chat(?:\s+right\s+now)?"
    r"|"
    r"log\s+in\s+to\s+(?:generate|create|edit)\s+images?"
    r")",
    re.IGNORECASE | re.DOTALL,
)
_LIMIT_STARTERS = ("you've hit the ", "you have hit the ", "you've hit your ", "you have hit your ",
    "image generation and image editing require",
    "image generation and editing require",
    "image generation require",
    "image generation isn't available",
    "image generation isn’t available",
    "image editing and image generation require",
    "image editing and generation require",
    "image editing require",
    "image creation require",
    "i can help create that image, but ",
    "i can help generate that image, but ",
    "i can create that image, but ",
    "i can generate that image, but ",
    "log in to generate ",
    "log in to create ",
    "log in to edit ",)
_PLAN_WORDS = ("free", "plus", "pro", "team", "enterprise")
# After a full non-quota starter match ("image generation requires..."), only these
# next words can still grow into a refusal; anything else ("requires a lot of...")
# is provably a normal reply and releases immediately.
def _refusal_next_words(starter):
    """Next words after a full starter match that can still grow into a refusal;
    any other word proves a normal reply and releases the stream immediately."""
    if starter.startswith("log in to "):
        return ("images",)          # "log in to generate/create/edit images"
    if starter.startswith("i can "):
        return ("image",)           # "...but image generation isn't available"
    if "require" in starter:
        return ("log", "you", "being", "to")  # logging/logged in, you to be, being, to be
    return None  # no gate known: hold until the regex or max-len releases
_LIMIT_MAX_LEN = 260  # refusal sentences stay well under this; longer text can't be one
# Login/availability refusals ("...isn't available in this chat right now.
# Once you're logged in...") run longer than quota refusals, hence the cap.


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
    stripped = low.lstrip(" \t\n\r>*#-`")
    # Still compatible with some refusal opening? Hold until it diverges.
    for starter in _LIMIT_STARTERS:
        if not stripped.startswith(starter[:1]):
            continue  # cheap first-char gate before per-char comparison
        n = min(len(stripped), len(starter))
        i = 0
        while i < n and stripped[i] == starter[i]:
            i += 1
        if i < n:
            continue  # diverged from this starter; try the next one
        if len(stripped) < len(starter):
            return 0, False  # text is still a strict prefix of this starter: hold
        if len(stripped) == len(starter):
            return 0, False  # text ends exactly at the starter boundary: hold
        # Full starter matched with more text following. For quota starters,
        # only a plan-name-looking next word can still grow into the refusal
        # ("Free"/"Plus"/...); any other continuation ("You've hit the nail...")
        # is a normal reply. Refusal starters always stay held until the regex
        # confirms (or _LIMIT_MAX_LEN releases).
        rest = stripped[len(starter):]
        m = re.match(r"[a-z]+", rest)
        word = m.group(0) if m else ""
        if starter.startswith(("you've hit", "you have hit")):
            if not word or any(w.startswith(word) or word.startswith(w) for w in _PLAN_WORDS):
                return 0, False
            return len(text), False  # "You've hit the nail..." -> normal reply
        allowed = _refusal_next_words(starter)
        if allowed is None:
            return 0, False  # no gate for this starter: hold until regex/max-len
        # Starters ending at "require": a streamed plural "s"/space is not the
        # next word — skip it before the continuation check.
        after = rest[1:] if word == "s" else rest
        m2 = re.match(r"[a-z]+", after.lstrip(" "))
        nxt_word = m2.group(0) if m2 else ""
        if not nxt_word:
            return 0, False  # bare "requires"/whitespace so far: ambiguous, hold
        if any(n.startswith(nxt_word) or nxt_word.startswith(n) for n in allowed):
            return 0, False  # could still grow into a refusal ("logging in"...)
        return len(text), False  # provably diverged -> normal reply
    return len(text), False  # diverged from every opening -> cannot be the refusal


# ChatGPT web-search citations arrive as private-use-area delimited markers
# embedded in the assistant text itself:
#   "\ue200cite\ue202turn0search0\ue202turn0search5\ue201"
# The referenced sources ride alongside in message.metadata:
#   - search_result_groups[].entries[] carry an explicit ref_id
#     {turn_index, ref_type, ref_index} matching each marker token;
#   - grouped_webpages content_references pair their refs[] positionally with
#     [url] + supporting_websites[].url (the same index space).
# Markers are rewritten to titled markdown links before anything is streamed;
# a block whose sources have not arrived yet is withheld until it resolves,
# and dropped at end of turn if its sources never show up. URLs lose utm_*
# tracking parameters so clients see clean links.
_CITE_BLOCK_RE = re.compile("\ue200cite(?:\ue202turn\\d+[a-z]+\\d+)+\ue201")
_CITE_TOKEN_RE = re.compile(r"turn(\d+)([a-z]+)(\d+)")
# Non-citation rich-content widgets ride the same \ue200..\ue201 delimiters:
#   "\ue200navlist\ue202<title>\ue202turn0news1...\ue201"   -- UI nav chips
#   "\ue200image_group\ue202{\"layout\":...}\ue201"          -- image carousel query
#   "\ue200genui\ue202{\"chart\":...}\ue201"                 -- chart spec: rendered
#       locally to a PNG and delivered as a PixelVault image link (see run_turn)
#   "\ue200map\ue202{\"query\":...}\ue201"                   -- map card
# Entity/product cards carry visible content and must not be stripped.
#   "\ue200entity\ue202["product","Microsoft Wireless Optical Mouse 2000","Model 1067"]\ue201"
# Product references render as titled markdown links when possible.
_ENTITY_BLOCK_RE = re.compile("\\ue200entity\\ue202([^\\ue201]*?)\\ue201")
# Inline link widgets carry a VISIBLE title and either a ref token or the raw URL:
#   "\ue200url\ue202Oh My Pi SDK Docs\ue202turn0search0\ue201"      (ref token -> cite_map)
#   "\ue200video\ue202Eternity scene\ue202turn0youtube12\ue201"    (ref token -> cite_map)
#   "\ue200url\ue202Bitwarden\ue202https://bitwarden.com\ue201"    (raw URL inline)
# Stripping them leaves dangling "- " bullets / "label: " lines with no link.
_LINK_BLOCK_RE = re.compile("\\ue200(?:url|video)\\ue202([^\\ue201]*?)(?:\\ue202([^\\ue201]*?))?\\ue201")
_PRODUCTS_BLOCK_RE = re.compile("\\ue200products\\ue202([^\\ue201]*?)\\ue201")
# The rest duplicate what the surrounding prose/links/table already say, so
# they are stripped outright. The [a-z_]+ arm catches ANY other named widget
# generically (payload after an optional \ue202 separator), so new kinds
# degrade to stripped instead of leaking; well-formed cite/entity/link/product
# blocks are excluded so all five regexes can be merged positionally.
_WIDGET_BLOCK_RE = re.compile(
    "\\ue200(?!cite\\ue202turn)(?!entity\\ue202)(?!url\\ue202)(?!video\\ue202)(?!products\\ue202)"
    "(?:navlist|[a-z_]+)(?:\\ue202[^\\ue201]*?)?\\ue201")
_PUA_CHARS_RE = re.compile("[\\ue200-\\ue205]")
_GENUI_PREFIX = "\ue200genui"


def _cite_map_extend(cmap: dict[tuple[int, str, int], dict], meta: Any) -> None:
    """Merge one SSE event's citation sources into the per-turn map."""
    if not isinstance(meta, dict):
        return
    content_refs = meta.get("content_references") or []
    if not isinstance(content_refs, list):
        content_refs = []
    groups = meta.get("search_result_groups") or []
    if not isinstance(groups, list):
        groups = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        entries = group.get("entries") or []
        if not isinstance(entries, list):
            entries = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            ref = entry.get("ref_id") or {}
            url = entry.get("url")
            key = _cite_key(ref)
            if url and key is not None:
                cmap.setdefault(key, {"url": url, "title": entry.get("title"),
                                      "attr": entry.get("attribution")})
    for cr in content_refs:
        if not isinstance(cr, dict) or cr.get("type") != "grouped_webpages":
            continue
        items = cr.get("items") or []
        if not isinstance(items, list):
            items = []
        for item in items:
            if not isinstance(item, dict):
                continue
            sources = [{"url": item.get("url"), "title": item.get("title"),
                        "attr": item.get("attribution")}]
            supporting = item.get("supporting_websites") or []
            if not isinstance(supporting, list):
                supporting = []
            sources += [{"url": s.get("url"), "title": s.get("title"),
                         "attr": s.get("attribution")}
                        for s in supporting if isinstance(s, dict)]
            refs = item.get("refs") or []
            if not isinstance(refs, list):
                refs = []
            for ref, src in zip(refs, sources):
                key = _cite_key(ref)
                if src.get("url") and key is not None:
                    cmap.setdefault(key, src)

    for cr in content_refs:
        if not isinstance(cr, dict):
            continue
        ctype = cr.get("type")
        if ctype == "products":
            refs = cr.get("refs") or []
            products = cr.get("products") or []
            if not isinstance(refs, list):
                refs = []
            if not isinstance(products, list):
                products = []
            for index, product in enumerate(products):
                source = _product_source(product)
                if source is None:
                    continue
                keys = []
                cite = product.get("cite") if isinstance(product, dict) else None
                if cite is not None:
                    keys.append(_cite_key(cite))
                if index < len(refs):
                    keys.append(_cite_key(refs[index]))
                for key in keys:
                    if key is not None:
                        cmap.setdefault(key, source)
        elif ctype == "product_entity":
            source = _product_source(cr.get("product"))
            if source is None:
                continue
            product = cr.get("product") or {}
            cite = product.get("cite") if isinstance(product, dict) else None
            refs = cr.get("refs") or []
            if not isinstance(refs, list):
                refs = []
            keys = [_cite_key(cite)] if cite is not None else []
            keys.extend(_cite_key(ref) for ref in refs)
            for key in keys:
                if key is not None:
                    cmap.setdefault(key, source)


def _product_search_url(title: str) -> str:
    """Return an actionable shopping search when ChatGPT has no product URL."""
    query = " ".join(str(title or "").split()).strip()
    if not query:
        return ""
    return "https://www.google.com/search?tbm=shop&q=" + urllib.parse.quote_plus(query)


def _cite_key(token: Any) -> tuple[int, str, int] | None:
    if isinstance(token, dict):
        key = (token.get("turn_index"), token.get("ref_type"), token.get("ref_index"))
        return key if (isinstance(key[0], int) and isinstance(key[1], str)
                       and isinstance(key[2], int)) else None
    match = _CITE_TOKEN_RE.fullmatch(str(token or "").strip())
    if not match:
        return None
    return int(match.group(1)), match.group(2), int(match.group(3))


def _product_source(product: Any) -> dict | None:
    """Normalize a ChatGPT product reference into a citation source."""
    if not isinstance(product, dict):
        return None
    title = " ".join(str(product.get("title") or "").split())
    if not title:
        return None
    raw_url = product.get("url")
    url = _clean_url(raw_url) if isinstance(raw_url, str) else ""
    if not url:
        url = _product_search_url(title)
    return {"url": url, "title": title,
            "attr": " ".join(str(product.get("merchants") or "").split()),
            "kind": "product"}


def _product_link(label: str, cmap: dict[tuple[int, str, int], dict],
                  token: Any = None) -> str:
    """Render one product as a link, with a search fallback for empty URLs."""
    source = cmap.get(_cite_key(token)) if token is not None else None
    title = " ".join(str((source or {}).get("title") or label).split())
    title = title.replace("[", "(").replace("]", ")")
    url = _clean_url((source or {}).get("url") or "") or _product_search_url(title)
    return f"[{title}]({_clean_url(url)})" if url else title


def _format_products(payload: str, cmap: dict[tuple[int, str, int], dict]) -> str:
    """Render a products carousel as compact markdown links."""
    try:
        parsed = json.loads(payload)
    except Exception:
        return ""
    selections = parsed.get("selections") if isinstance(parsed, dict) else parsed
    if not isinstance(selections, list):
        return ""
    links: list[str] = []
    for selection in selections:
        if not isinstance(selection, list) or len(selection) < 2:
            continue
        token, label = selection[0], selection[1]
        if not isinstance(label, str) or not label.strip():
            continue
        links.append(_product_link(label, cmap, token))
    return " · ".join(links)


def _products_unresolved(payload: str, cmap: dict[tuple[int, str, int], dict]) -> bool:
    try:
        parsed = json.loads(payload)
    except Exception:
        return False
    selections = parsed.get("selections") if isinstance(parsed, dict) else parsed
    if not isinstance(selections, list):
        return False
    return any(
        isinstance(selection, list)
        and len(selection) >= 2
        and (key := _cite_key(selection[0])) is not None
        and key not in cmap
        for selection in selections
    )


def _clean_url(url: str) -> str:
    """Drop utm_* tracking parameters; keep everything else byte-faithful."""
    # Split the raw query instead of round-tripping through parse_qsl:
    # re-encoding survivors corrupts values (semicolon pairs collapse,
    # invalid-UTF-8 escapes get replaced, %20 becomes '+').
    if not isinstance(url, str) or any(ord(ch) < 0x20 or ord(ch) == 0x7f for ch in url):
        return ""
    try:
        parts = urllib.parse.urlsplit(url.strip())
    except ValueError:
        return ""
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        return ""
    kept = [pair for pair in parts.query.split("&") if not urllib.parse.unquote_plus(
        pair.partition("=")[0]).lower().startswith("utm_")] if parts.query else []
    out = urllib.parse.urlunsplit(parts._replace(query="&".join(kept)))
    # () would terminate a markdown link early; keep them percent-encoded
    return out.replace("(", "%28").replace(")", "%29").replace(" ", "%20")


def _format_source(src: dict) -> str:
    """One citation as [label](url); label falls back to attribution/domain."""
    if not isinstance(src, dict):
        return ""
    url = _clean_url(src.get("url", ""))
    label = " ".join(str(src.get("title") or "").split())
    if not label:
        label = " ".join(str(src.get("attr") or "").split())
    if not label and url:
        netloc = urllib.parse.urlsplit(url).netloc
        label = netloc[4:] if netloc.startswith("www.") else netloc
    label = label.replace("[", "(").replace("]", ")")
    if not url:
        return label
    return f"[{label}]({url})" if label else url

SOURCE_APPENDIX_MAX = 50


def _host_of(url: str) -> str:
    try:
        return (urllib.parse.urlsplit(url).netloc or "").lower()
    except Exception:
        return ""


def _source_appendix(sources: list[dict], query: str = "") -> str:
    """Bridge source appendix for llmcord-go's "Show Sources" button.

    Matches the appendix contract parsed by llmcord-go across bridge providers:
        \n\nSources
        1. [Title](url) (domain) via `query`

        Search Queries
        1. `query`
    """
    entries: list[str] = []
    seen_urls: set[str] = set()
    clean_query = query.replace("`", "'").strip() if query else ""
    for src in sources[:SOURCE_APPENDIX_MAX]:
        if not isinstance(src, dict):
            continue
        raw_url = src.get("url")
        if not isinstance(raw_url, str) or not raw_url.strip():
            continue
        url = _clean_url(raw_url).replace("\n", "").replace("\r", "")
        if not url or url.lower() in seen_urls:
            continue
        seen_urls.add(url.lower())

        title = " ".join((src.get("title") or "").split()).replace("[", "(").replace("]", ")")
        if not title:
            title = " ".join((src.get("attr") or "").split()).replace("[", "(").replace("]", ")")
        if not title:
            host = _host_of(url)
            title = host[4:] if host.startswith("www.") else host
        if not title:
            title = url

        entry = f"[{title}]({url})"
        host = _host_of(url)
        if title != url and host:
            entry += f" ({host})"
        if clean_query:
            entry += f" via `{clean_query}`"
        entries.append(entry)

    if not entries:
        return ""
    lines = ["Sources"]
    lines.extend(f"{i}. {entry}" for i, entry in enumerate(entries, start=1))
    if clean_query:
        lines.append("")
        lines.append("Search Queries")
        lines.append(f"1. `{clean_query}`")
    return "\n\n" + "\n".join(lines)


def _parse_entity(payload: str) -> tuple[str, Any, tuple[int, str, int] | None, bool] | None:
    """Return (label, token, citation key, is_product) for one entity widget."""
    try:
        parsed = json.loads(payload.split("\ue202")[0])
    except Exception:
        return None
    if isinstance(parsed, list):
        token = parsed[0] if parsed and isinstance(parsed[0], str) else None
        label = next((item for item in parsed[1:] if isinstance(item, str) and item.strip()), "")
        key = _cite_key(token)
        is_product = parsed[0] == "product" if parsed else False
        is_product = is_product or (key is not None and key[1] == "product")
    elif isinstance(parsed, dict):
        token = parsed.get("cite") or parsed.get("ref_id")
        key = _cite_key(token)
        kind = str(parsed.get("type") or "").lower()
        is_product = kind in {"product", "product_entity"}
        is_product = is_product or (key is not None and key[1] == "product")
        label = parsed.get("title") or parsed.get("name") or ""
    else:
        return None
    if not isinstance(label, str) or not label.strip():
        return None
    return " ".join(label.split()).replace("[", "(").replace("]", ")"), token, key, is_product


def _format_entity(payload: str, cmap: dict[tuple[int, str, int], dict]) -> str:
    """Render an entity; product entities retain an actionable markdown link."""
    parsed = _parse_entity(payload)
    if parsed is None:
        return ""
    label, token, _key, is_product = parsed
    return _product_link(label, cmap, token) if is_product else label


def _render_citations(raw: str, cmap: dict[tuple[int, str, int], dict], *,
                      final: bool = False,
                      charts_out: list[dict] | None = None) -> tuple[str, int]:
    """Rewrite citation-marker blocks in one cumulative text snapshot to links.

    Returns (rendered_text, safe_len). safe_len bounds the prefix that is
    provably stable: complete blocks with still-unresolved tokens hold the
    stream at their start (their bytes would change once sources arrive);
    a trailing half-received block holds likewise. With final=True the stream
    is over -- unresolved tokens are dropped instead of held. When charts_out
    is a list (final pass), genui chart specs are collected into it for the
    caller to render and upload.
    """
    out: list[str] = []
    pos = 0
    safe = 0
    hold = -1  # display index from which output may still change
    matches = sorted((m for pat in (_CITE_BLOCK_RE, _ENTITY_BLOCK_RE, _LINK_BLOCK_RE,
                                    _PRODUCTS_BLOCK_RE, _WIDGET_BLOCK_RE)
                      for m in pat.finditer(raw)), key=lambda m: m.start())
    for m in matches:
        gap = raw[pos:m.start()]
        out.append(gap)
        if hold < 0:
            safe += len(gap)
        pos = m.end()
        piece = ""
        if m.group(0).startswith("\ue200cite"):
            tokens = _CITE_TOKEN_RE.findall(m.group(0))
            sources = [cmap.get((int(t), rt, int(n))) for t, rt, n in tokens]
            resolved = [s for s in sources if s]
            if len(resolved) == len(tokens) or final:
                piece = " ".join(_format_source(s) for s in resolved)
            elif hold < 0:
                hold = safe  # withhold this block (and everything after it)
        elif m.group(0).startswith("\ue200products\ue202"):
            if not final and _products_unresolved(m.group(1), cmap):
                if hold < 0:
                    hold = safe
            else:
                piece = _format_products(m.group(1), cmap)
        elif m.group(0).startswith("\ue200entity\ue202"):
            parsed = _parse_entity(m.group(1))
            if not final and parsed is not None and parsed[2] is not None and parsed[2] not in cmap:
                if hold < 0:
                    hold = safe  # sources may still arrive; hold the block
            else:
                piece = _format_entity(m.group(1), cmap)
        elif m.group(0).startswith(("\ue200url\ue202", "\ue200video\ue202")):
            # predicate must mirror _LINK_BLOCK_RE exactly: title is always
            # visible; target (group 2) is either a ref token resolved via
            # cite_map or a raw URL. Unresolved ref tokens hold mid-stream;
            # final flush preserves the visible title if metadata never arrives.
            title = " ".join((m.group(1) or "").split()).replace("[", "(").replace("]", ")")
            target = (m.group(2) or "").strip()
            src: dict | None = None
            rt = _CITE_TOKEN_RE.fullmatch(target)
            if rt:
                src = cmap.get((int(rt.group(1)), rt.group(2), int(rt.group(3))))
                if src is None:
                    if not final and hold < 0:
                        hold = safe  # sources may still arrive; hold the block
                    elif final:
                        piece = title
                else:
                    url = _clean_url(src.get("url", ""))
                    piece = f"[{title}]({url})" if title and url else (title or _format_source(src))
            elif target.startswith(("http://", "https://")):
                url = _clean_url(target)
                piece = f"[{title}]({url})" if title and url else (title or url)
            elif title.startswith(("http://", "https://")):
                piece = _clean_url(title) or title  # two-part form: the URL IS the payload
            else:
                piece = title  # no/unknown target: the title itself is the content
        elif charts_out is not None and m.group(0).startswith(_GENUI_PREFIX):
            payload = m.group(0)[len(_GENUI_PREFIX):-1].lstrip("\ue202")
            try:
                # model-supplied JSON can be anything (non-object scalars,
                # Infinity/NaN literals) -- a bad widget must never raise into
                # the post-flush failure path once content already streamed
                spec = chart_from_payload(json.loads(payload))
            except Exception:
                spec = None
            if spec:
                charts_out.append(spec)
            # widget itself is still stripped from the text; the caller appends
            # a rendered image link after the flush
        out.append(piece)
        if hold < 0:
            safe += len(piece)
    tail = raw[pos:]
    if tail:
        # A trailing \ue200 may be a half-received marker (withhold mid-stream)
        # or content we simply cannot parse -- at end of stream never discard
        # past it, or real text after it would silently vanish.
        cut = -1 if hold >= 0 or final else tail.find("\ue200")
        if cut >= 0:  # half-received trailing block: withhold it
            out.append(tail[:cut])
            if hold < 0:
                safe += cut
                hold = safe
        else:
            out.append(tail)
            if hold < 0:
                safe += len(tail)
    text = "".join(out)
    if final:
        # end of stream: drop leftover invisible marker scaffolding (stray
        # delimiters/separators); visible content is never made of these
        text = _PUA_CHARS_RE.sub("", text)
        return text, len(text)
    return text, min(safe, len(text))



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
            # Set once the success-path final flush has released everything
            # renderable; from then on the failure-path salvage pass must be a
            # no-op (it would only re-send bytes the client already has,
            # because success-path yields advance text_acc but not emitted).
            fully_flushed = False
            cite_map: dict[tuple[int, str, int], dict] = {}
            chart_specs: list[dict] = []
            current_msg_id = ""
            sediment: list[str] = []
            cid = ""
            parent = ""
            limit_hit = False
            last_event: dict[str, Any] = {}
            last_node_id = ""

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
                last_event = ev
                if ev.get("conversation_id") and not cid:
                    cid = ev["conversation_id"]
                m = ev.get("message") or {}
                _cite_map_extend(cite_map, m.get("metadata"))
                author = m.get("author") or {}
                if m.get("id"):
                    last_node_id = m["id"]
                if author.get("role") not in ("assistant", "tool"):
                    continue
                if author.get("role") == "assistant" and m.get("id") and m["id"] != current_msg_id:
                    # new assistant node: restart delta accounting so its text streams
                    current_msg_id = m["id"]
                    emitted = 0
                    text_acc = ""
                content = m.get("content") or {}
                parts = content.get("parts") or []
                # Only 'text' nodes carry client-visible prose. thoughts/code/
                # reasoning_recap/model_editable_context nodes share the
                # assistant role; their string parts (when present) are
                # internal channel data and must never stream as the answer.
                is_text_node = (content.get("content_type") or "text") == "text"
                for p in parts:
                    if isinstance(p, dict):
                        ptr = p.get("asset_pointer")
                        if isinstance(ptr, str) and "sediment://" in ptr and ptr not in sediment:
                            sediment.append(ptr)
                    elif isinstance(p, str) and author.get("role") == "assistant" and is_text_node:
                        text_acc = p
                        # Withhold bytes while they could still be the image-limit
                        # refusal: once visible, accounts can no longer be switched.
                        emit_upto, is_limit = _classify_accumulated(text_acc)
                        if is_limit:
                            limit_hit = True
                            break
                        display, safe_upto = _render_citations(text_acc, cite_map)
                        if emit_upto > 0 and safe_upto > emitted:
                            produced = True
                            yield {"type": "delta", "text": display[emitted:safe_upto]}
                            emitted = safe_upto
                if limit_hit:
                    break
                meta = m.get("metadata") or {}
                if meta.get("message_type") in ("next", "continue"):
                    parent = m.get("id") or parent

            # Refusals are valid HTTP-200 streams and may intentionally stop
            # before ChatGPT emits the normal assistant "next" marker.
            if limit_hit or (text_acc and IMAGE_LIMIT_RE.search(text_acc)):
                raise _ImageLimitError(text_acc.strip()[:300])

            
            # Image-generation turns can end with ONLY a tool node carrying the
            # sediment pointer -- no assistant message, no "next" marker. Such a
            # turn is complete: its product is the generated image.
            if not cid or (not parent and not sediment):
                # Full frame goes to the server log only; the client-facing
                # error keeps just the event type so session-scoped tokens and
                # ids never leave the process.
                if last_event:
                    log.warning("incomplete stream on %s: last SSE event %s",
                                acct.email, str(last_event)[:500])
                tail = (f"; last SSE event type: {last_event.get('type', '?')}"
                        if last_event else "; stream ended without any events")
                raise EngineError(502, "ChatGPT returned an incomplete response" + tail)
            if not parent:
                # Keep the stored conversation continuable: parent the next turn
                # to the last node we saw (the tool node) instead of "".
                parent = last_node_id
            # Flush anything still withheld first: short normal replies that
            # begin like the refusal must reach the client in full. final=True
            # also releases citation blocks whose sources never arrived.
            display, _ = _render_citations(text_acc, cite_map, final=True,
                                           charts_out=chart_specs)
            if emitted < len(display):
                produced = True
                yield {"type": "delta", "text": display[emitted:]}
            text_acc = display
            fully_flushed = True

            # genui chart specs -> locally rendered PNG -> PixelVault link.
            # Any failure degrades to the widget simply being stripped.
            chart_links: list[str] = []
            for spec in chart_specs:
                title = " ".join(str((spec.get("meta") or {}).get("title") or "chart").split())
                alt = f"chart: {title}".replace("[", "(").replace("]", ")")
                try:
                    # PIL draw time scales with row count -- keep it off the
                    # event loop so concurrent requests never stall behind a
                    # pathological widget payload
                    png = await asyncio.to_thread(render_chart_png, spec)
                    url = await upload_image("chart.png", png, "image/png")
                    chart_links.append(f"![{alt}]({url})")
                except Exception as e:
                    log.warning("chart render/upload failed (%s): %s", title[:60], e)
            if chart_links:
                links = "\n\n" + "\n\n".join(chart_links)
                produced = True
                yield {"type": "delta", "text": links}
                text_acc += links

            image_urls = await _resolve_images(acct, sediment)
            if image_urls:
                links = "\n\n" + "\n\n".join(f"![generated image]({u})" for u in image_urls)
                produced = True
                yield {"type": "delta", "text": links}
                text_acc += links
            inc_sources = config.INCLUDE_SOURCES if parsed.include_sources is None else parsed.include_sources
            if inc_sources and cite_map:
                # Extract latest user message text as query fallback if available
                user_query = ""
                for item in reversed(parsed.items):
                    if item.role == "user" and item.text:
                        user_query = item.text
                        break
                appendix = _source_appendix(list(cite_map.values()), query=user_query)
                if appendix:
                    produced = True
                    yield {"type": "delta", "text": appendix}
                    text_acc += appendix

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
                # Salvage pass: release whatever upstream already delivered
                # but was still withheld (unresolved cite blocks hold the
                # emitter, half-received markers trail). An aborted turn then
                # degrades to a shorter answer instead of one that stops
                # mid-sentence right where the first held block sat.
                # Refusal-shaped text stays withheld, and chart specs are not
                # collected here (widgets strip): the failure path must never
                # start new render/upload work.
                _, still_limit = _classify_accumulated(text_acc)
                if not still_limit and not fully_flushed:
                    try:
                        salvaged, _ = _render_citations(text_acc, cite_map, final=True)
                        if emitted < len(salvaged):
                            yield {"type": "delta", "text": salvaged[emitted:]}
                            text_acc = salvaged
                    except Exception as salvage_error:
                        log.warning("salvage flush failed: %s", salvage_error)
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
