# Copyright 2026 chatgpt-to-openai-api contributors.
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
import urllib.parse
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator
    from typing import Any

from curl_cffi.requests.exceptions import RequestException
from PIL import Image

from . import config
from .accounts import AccountPool, NoAccountAvailableError
from .adapters import HistoryItem, ParsedRequest, estimate_tokens, map_model
from .charts import chart_from_payload, render_chart_png
from .chatgpt import AccountSession, ChatGPTError, StreamMedia
from .pixelvault import PixelVaultError, upload_image
from .store import STORE, ConvRef, ResponseRecord, TurnSnapshot, item_hash

log = logging.getLogger("engine")

CiteKey = tuple[int, str, int]
CiteSource = dict[str, str]
CiteMap = dict[CiteKey, CiteSource]
JsonObject = dict[str, object]
ChartSpec = dict[str, object]

MIN_SELECTION_FIELDS = 2
ASCII_CONTROL_LIMIT = 0x20
ASCII_DELETE_CODE = 0x7F
JSX_CITE_SHORT_MAX = 120
HTTP_RATE_LIMITED = 429


@dataclass
class TurnResult:
    """Outcome of one executed turn."""

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
    """Deterministic engine failure with a client-facing status."""

    def __init__(
        self, status: int, message: str, error_type: str = "server_error"
    ) -> None:
        """Store the client-facing status, message, and error type."""
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
    r"you(?:'ve| have)\s+hit\s+(?:the|your)\s+.*?plan\s+limit\s+for\s+"
    r"image\s+generations?\s+requests"
    r"|"
    r"(?:image\s+(?:generation|editing)(?:\s+and\s+(?:image\s+)?(?:generation|editing))?"
    r"|image\s+creation)\s+require(?:s)?\s+"
    r"(?:you\s+to\s+be\s+|being\s+|to\s+be\s+)?log(?:ged|ging)\s+in"
    r"|"
    r"(?:i\s+can\s+(?:help\s+)?(?:create|generate)\s+that\s+image,\s+but\s+)?"
    r"image\s+generation\s+isn['\u2019]t\s+available\s+in\s+this\s+chat"
    r"(?:\s+right\s+now)?"
    r"|"
    r"log\s+in\s+to\s+(?:generate|create|edit)\s+images?"
    r"|"
    r"it\s+looks\s+like\s+(?:image\s+(?:creation|generation|editing)"
    r"|image\s+requests?)\s+(?:is|are)\s+(?:currently\s+|temporarily\s+)?unavailable"
    r")",
    re.IGNORECASE | re.DOTALL,
)
_LIMIT_STARTERS = (
    "you've hit the ",
    "you have hit the ",
    "you've hit your ",
    "you have hit your ",
    "image generation and image editing require",
    "image generation and editing require",
    "image generation require",
    "image generation isn't available",
    "image generation isn\u2019t available",
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
    "log in to edit ",
    # Temporary-unavailability refusal ("It looks like image creation is
    # temporarily unavailable. Do you want to try something else?"): same
    # account-failover treatment as quota/login refusals. One generic starter,
    # not exact strings: every IMAGE_LIMIT_RE variant of this template
    # (currently/temporarily, is/are, request(s), creation/generation/
    # editing) must stay HELD past the noun until the regex confirms --
    # exact-string starters release other variants' bytes mid-stream, which
    # sets produced=True and aborts account failover.
    "it looks like image ",
)
_PLAN_WORDS = ("free", "plus", "pro", "team", "enterprise")


# After a full non-quota starter match ("image generation requires..."), only these
# next words can still grow into a refusal; anything else ("requires a lot of...")
# is provably a normal reply and releases immediately.
def _refusal_next_words(starter: str) -> tuple[str, ...] | None:
    """Next words that can still grow into a refusal.

    Any other word proves a normal reply and releases the stream immediately.
    """
    if starter.startswith("log in to "):
        return ("images",)  # "log in to generate/create/edit images"
    if starter.startswith("i can "):
        return ("image",)  # "...but image generation isn't available"
    if starter.startswith("it looks like image "):
        # Noun gate: only these can still grow into "image <noun> is/are
        # ... unavailable"; anything else ("image quality...") proves a
        # normal reply and releases immediately.
        return ("creation", "generation", "editing", "request")
    if "require" in starter:
        return (
            "log",
            "you",
            "being",
            "to",
        )  # logging/logged in, you to be, being, to be
    return None  # no gate known: hold until the regex or max-len releases


def _starter_remainder(stripped: str, starter: str) -> str | None:
    """Remainder after a starter, or None when diverged, or "" when held."""
    if not stripped or not stripped.startswith(starter[:1]):
        return None
    common = min(len(stripped), len(starter))
    for pos in range(common):
        if stripped[pos] != starter[pos]:
            return None
    if len(stripped) <= len(starter):
        return ""
    return stripped[len(starter) :]


def _classify_quota_rest(rest: str, text_len: int) -> tuple[int, bool]:
    """Classify text following a quota starter."""
    match = re.match(r"[a-z]+", rest)
    word = match.group(0) if match else ""
    if not word:
        return 0, False
    for plan in _PLAN_WORDS:
        if plan.startswith(word) or word.startswith(plan):
            return 0, False
    return text_len, False


def _classify_gated_rest(starter: str, rest: str, text_len: int) -> tuple[int, bool]:
    """Classify text following a gated non-quota starter."""
    allowed = _refusal_next_words(starter)
    if allowed is None:
        return 0, False
    match = re.match(r"[a-z]+", rest)
    word = match.group(0) if match else ""
    after = rest[1:] if word == "s" else rest
    follow = re.match(r"[a-z]+", after.lstrip(" "))
    nxt_word = follow.group(0) if follow else ""
    if not nxt_word:
        return 0, False
    for candidate in allowed:
        if candidate.startswith(nxt_word) or nxt_word.startswith(candidate):
            return 0, False
    return text_len, False


def _classify_starter_match(starter: str, rest: str, text_len: int) -> tuple[int, bool]:
    """Classify text that fully matched a starter plus trailing text."""
    if starter.startswith(("you've hit", "you have hit")):
        return _classify_quota_rest(rest, text_len)
    return _classify_gated_rest(starter, rest, text_len)


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
    stripped = text.lower().lstrip(" \t\n\r>*#-`")
    # Still compatible with some refusal opening? Hold until it diverges.
    for starter in _LIMIT_STARTERS:
        rest = _starter_remainder(stripped, starter)
        if rest is None:
            continue
        if not rest:
            return 0, False
        return _classify_starter_match(starter, rest, len(text))
    return len(text), False


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
#   "\ue200entity\ue202["product","Microsoft...", "Model 1067"]\ue201"
# Product references render as titled markdown links when possible.
_ENTITY_BLOCK_RE = re.compile("\\ue200entity\\ue202([^\\ue201]*?)\\ue201")
# Inline link widgets carry a VISIBLE title plus a ref token or raw URL:
#   "\ue200url\ue202Oh My Pi SDK Docs\ue202turn0search0\ue201" (via cite_map)
#   "\ue200video\ue202Eternity scene\ue202turn0youtube12\ue201" (via cite_map)
#   "\ue200url\ue202Bitwarden\ue202https://bitwarden.com\ue201" (raw URL)
# Stripping them leaves dangling "- " bullets / "label: " lines with no link.
_LINK_BLOCK_RE = re.compile(
    "\\ue200(?:url|video)\\ue202([^\\ue201]*?)(?:\\ue202([^\\ue201]*?))?\\ue201",
)
_PRODUCTS_BLOCK_RE = re.compile("\\ue200products\\ue202([^\\ue201]*?)\\ue201")
# The rest duplicate what the surrounding prose/links/table already say, so
# they are stripped outright. The [a-z_]+ arm catches ANY other named widget
# generically (payload after an optional \ue202 separator), so new kinds
# degrade to stripped instead of leaking; well-formed cite/entity/link/product
# blocks are excluded so all five regexes can be merged positionally.
_WIDGET_BLOCK_RE = re.compile(
    "\\ue200(?!cite\\ue202turn)(?!entity\\ue202)(?!url\\ue202)(?!video\\ue202)(?!products\\ue202)"
    "(?:navlist|[a-z_]+)(?:\\ue202[^\\ue201]*?)?\\ue201",
)
# The concurrent-generation stream variant serializes citations as JSX
# instead of \ue200 blocks:  <Cite refs={["turn0news9","turn0search10"]}/>
# (seen on gpt-5-6 branch races; both ref= and refs= spellings occur).
# Well-formed tags carry at least one turn-ref token and resolve through the
# same cite_map as PUA cite blocks; a token-free match is legit code content
# and must stay untouched.
_CITE_TAG_RE = re.compile(r"<Cite\b[^>\n]*?/>")
_PUA_CHARS_RE = re.compile("[\\ue200-\\ue205]")
_GENUI_PREFIX = "\ue200genui"


def _opt_str(value: object) -> str:
    """Coerce a JSON field to text."""
    if isinstance(value, str):
        return value
    if not value:
        return ""
    return str(value)


def _as_object_list(value: object) -> list[object]:
    """Return value items when it is a list, otherwise an empty list."""
    if not isinstance(value, list):
        return []
    return list(value)


def _make_source(url: object, title: object, attr: object) -> CiteSource:
    """Build a citation source with text-only fields."""
    return {"url": _opt_str(url), "title": _opt_str(title), "attr": _opt_str(attr)}


def _extend_search_groups(cmap: CiteMap, groups: list[object]) -> None:
    """Merge search_result_groups entries into the per-turn map."""
    for group in groups:
        if not isinstance(group, dict):
            continue
        entries = _as_object_list(group.get("entries"))
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            key = _cite_key(entry.get("ref_id"))
            url_raw = entry.get("url")
            url = url_raw if isinstance(url_raw, str) else ""
            if url and key is not None:
                cmap.setdefault(
                    key,
                    _make_source(url, entry.get("title"), entry.get("attribution")),
                )


def _extend_grouped_webpages(cmap: CiteMap, content_refs: list[object]) -> None:
    """Merge grouped_webpages positional refs into the per-turn map."""
    for content in content_refs:
        if not isinstance(content, dict):
            continue
        if content.get("type") != "grouped_webpages":
            continue
        for item in _as_object_list(content.get("items")):
            if not isinstance(item, dict):
                continue
            _extend_webpage_item(cmap, item)


def _extend_webpage_item(cmap: CiteMap, item: object) -> None:
    """Merge one grouped_webpages item's positional refs."""
    if not isinstance(item, dict):
        return
    sources: list[CiteSource] = [
        _make_source(item.get("url"), item.get("title"), item.get("attribution"))
    ]
    for support in _as_object_list(item.get("supporting_websites")):
        if not isinstance(support, dict):
            continue
        sources.append(
            _make_source(
                support.get("url"), support.get("title"), support.get("attribution")
            )
        )
    refs = _as_object_list(item.get("refs"))
    for ref, src in zip(refs, sources, strict=False):
        key = _cite_key(ref)
        if src.get("url") and key is not None:
            cmap.setdefault(key, src)


def _product_keys(product: object, refs: list[object], index: int) -> list[CiteKey]:
    """Citation keys addressed by one products-carousel entry."""
    keys: list[CiteKey] = []
    cite: object = None
    if isinstance(product, dict):
        cite = product.get("cite")
    if cite is not None:
        key = _cite_key(cite)
        if key is not None:
            keys.append(key)
    if index < len(refs):
        key = _cite_key(refs[index])
        if key is not None:
            keys.append(key)
    return keys


def _extend_products_block(cmap: CiteMap, content: object) -> None:
    """Merge one products content_reference into the per-turn map."""
    if not isinstance(content, dict):
        return
    refs = _as_object_list(content.get("refs"))
    products = _as_object_list(content.get("products"))
    for index, product in enumerate(products):
        source = _product_source(product)
        if source is None:
            continue
        for key in _product_keys(product, refs, index):
            cmap.setdefault(key, source)


def _extend_product_entity(cmap: CiteMap, content: object) -> None:
    """Merge one product_entity content_reference into the per-turn map."""
    if not isinstance(content, dict):
        return
    source = _product_source(content.get("product"))
    if source is None:
        return
    product = content.get("product")
    cite: object = None
    if isinstance(product, dict):
        cite = product.get("cite")
    keys: list[CiteKey] = []
    if cite is not None:
        key = _cite_key(cite)
        if key is not None:
            keys.append(key)
    for ref in _as_object_list(content.get("refs")):
        key = _cite_key(ref)
        if key is not None:
            keys.append(key)
    for key in keys:
        cmap.setdefault(key, source)


def _extend_product_refs(cmap: CiteMap, content_refs: list[object]) -> None:
    """Merge product carousel/entity refs into the per-turn map."""
    for content in content_refs:
        if not isinstance(content, dict):
            continue
        ctype = content.get("type")
        if ctype == "products":
            _extend_products_block(cmap, content)
        elif ctype == "product_entity":
            _extend_product_entity(cmap, content)


def _cite_map_extend(cmap: CiteMap, meta: object) -> None:
    """Merge one SSE event's citation sources into the per-turn map."""
    if not isinstance(meta, dict):
        return
    content_refs = _as_object_list(meta.get("content_references") or [])
    groups = _as_object_list(meta.get("search_result_groups") or [])
    _extend_search_groups(cmap, groups)
    _extend_grouped_webpages(cmap, content_refs)
    _extend_product_refs(cmap, content_refs)


def _product_search_url(title: str) -> str:
    """Return an actionable shopping search when ChatGPT has no product URL."""
    query = " ".join(str(title or "").split()).strip()
    if not query:
        return ""
    base = "https://www.google.com/search?tbm=shop&q="
    return base + urllib.parse.quote_plus(query)


def _cite_key(token: object) -> CiteKey | None:
    """Citation key for a ref token, or None when the token is not a ref."""
    if isinstance(token, dict):
        turn = token.get("turn_index")
        ref_type = token.get("ref_type")
        ref_index = token.get("ref_index")
        if (
            isinstance(turn, int)
            and isinstance(ref_type, str)
            and isinstance(ref_index, int)
        ):
            return (turn, ref_type, ref_index)
        return None
    if not token:
        return None
    text = token.strip() if isinstance(token, str) else str(token).strip()
    match = _CITE_TOKEN_RE.fullmatch(text)
    if not match:
        return None
    ref_name = match.group(2)
    if not isinstance(ref_name, str):
        return None
    return int(match.group(1)), ref_name, int(match.group(3))


def _product_source(product: object) -> CiteSource | None:
    """Normalize a ChatGPT product reference into a citation source."""
    if not isinstance(product, dict):
        return None
    title = " ".join(_opt_str(product.get("title")).split())
    if not title:
        return None
    raw_url = product.get("url")
    url = _clean_url(raw_url) if isinstance(raw_url, str) else ""
    if not url:
        url = _product_search_url(title)
    return {
        "url": url,
        "title": title,
        "attr": " ".join(_opt_str(product.get("merchants")).split()),
        "kind": "product",
    }


def _product_link(label: str, cmap: CiteMap, token: object = None) -> str:
    """Render one product as a link, with a search fallback for empty URLs."""
    source: CiteSource | None = None
    if token is not None:
        key = _cite_key(token)
        if key is not None:
            source = cmap.get(key)
    title = " ".join(_opt_str((source or {}).get("title") or label).split())
    title = title.replace("[", "(").replace("]", ")")
    raw = (source or {}).get("url") or ""
    url = _clean_url(raw) if isinstance(raw, str) else ""
    url = url or _product_search_url(title)
    return f"[{title}]({_clean_url(url)})" if url else title


def _selections_from_payload(payload: str) -> list[object]:
    """Product selections list from a carousel payload, or empty."""
    try:
        parsed: object = json.loads(payload)
    except ValueError:
        return []
    if isinstance(parsed, dict):
        selections: object = parsed.get("selections")
    else:
        selections = parsed
    return _as_object_list(selections)


def _format_products(payload: str, cmap: CiteMap) -> str:
    """Render a products carousel as compact markdown links."""
    selections = _selections_from_payload(payload)
    links: list[str] = []
    for selection in selections:
        if not isinstance(selection, list) or len(selection) < MIN_SELECTION_FIELDS:
            continue
        token, label = selection[0], selection[1]
        if not isinstance(label, str) or not label.strip():
            continue
        links.append(_product_link(label, cmap, token))
    return " · ".join(links)


def _products_unresolved(payload: str, cmap: CiteMap) -> bool:
    """Whether a products carousel references sources not yet in the map."""
    selections = _selections_from_payload(payload)
    for selection in selections:
        if not isinstance(selection, list):
            continue
        if len(selection) < MIN_SELECTION_FIELDS:
            continue
        key = _cite_key(selection[0])
        if key is not None and key not in cmap:
            return True
    return False


def _clean_url(url: str) -> str:
    """Drop utm_* tracking parameters; keep everything else byte-faithful."""
    # Split the raw query instead of round-tripping through parse_qsl:
    # re-encoding survivors corrupts values (semicolon pairs collapse,
    # invalid-UTF-8 escapes get replaced, %20 becomes '+').
    if not isinstance(url, str):
        return ""
    if any(ord(ch) < ASCII_CONTROL_LIMIT or ord(ch) == ASCII_DELETE_CODE for ch in url):
        return ""
    try:
        parts = urllib.parse.urlsplit(url.strip())
    except ValueError:
        return ""
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        return ""
    kept = (
        [
            pair
            for pair in parts.query.split("&")
            if not urllib.parse.unquote_plus(pair.partition("=")[0])
            .lower()
            .startswith("utm_")
        ]
        if parts.query
        else []
    )
    out = urllib.parse.urlunsplit(parts._replace(query="&".join(kept)))
    # () would terminate a markdown link early; keep them percent-encoded
    return out.replace("(", "%28").replace(")", "%29").replace(" ", "%20")


def _format_source(src: CiteSource) -> str:
    """One citation as [label](url); label falls back to attribution/domain."""
    if not isinstance(src, dict):
        return ""
    url = _clean_url(src.get("url", ""))
    label = " ".join(_opt_str(src.get("title")).split())
    if not label:
        label = " ".join(_opt_str(src.get("attr")).split())
    if not label and url:
        netloc = urllib.parse.urlsplit(url).netloc
        label = netloc.removeprefix("www.")
    label = label.replace("[", "(").replace("]", ")")
    if not url:
        return label
    return f"[{label}]({url})" if label else url


SOURCE_APPENDIX_MAX = 50


def _host_of(url: str) -> str:
    """Host of a URL, lowercased, or empty when unparsable."""
    try:
        return (urllib.parse.urlsplit(url).netloc or "").lower()
    except ValueError:
        return ""


def _appendix_title(src: CiteSource, url: str) -> str:
    """Display title for one appendix entry."""
    title = " ".join(_opt_str(src.get("title")).split())
    title = title.replace("[", "(").replace("]", ")")
    if not title:
        title = " ".join(_opt_str(src.get("attr")).split())
        title = title.replace("[", "(").replace("]", ")")
    if not title:
        title = _host_of(url).removeprefix("www.")
    return title or url


def _appendix_entry(url: str, title: str, clean_query: str) -> str:
    """One appendix line for an already-cleaned URL and title."""
    entry = f"[{title}]({url})"
    host = _host_of(url)
    if title != url and host:
        entry += f" ({host})"
    if clean_query:
        entry += f" via `{clean_query}`"
    return entry


def _source_appendix(sources: list[CiteSource], query: str = "") -> str:
    r"""Bridge source appendix for llmcord-go's "Show Sources" button.

    Matches the appendix contract parsed by llmcord-go across bridge providers:
        \n\nSources
        1. [Title](url) (domain) via `query`

        Search Queries
        1. `query`
    """
    clean_query = query.replace("`", "'").strip() if query else ""
    entries: list[str] = []
    seen_urls: set[str] = set()
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
        title = _appendix_title(src, url)
        entries.append(_appendix_entry(url, title, clean_query))
    if not entries:
        return ""
    lines = ["Sources"]
    lines.extend(f"{num}. {entry}" for num, entry in enumerate(entries, start=1))
    if clean_query:
        lines.append("")
        lines.append("Search Queries")
        lines.append(f"1. `{clean_query}`")
    return "\n\n" + "\n".join(lines)


def _entity_from_list(
    parsed: object,
) -> tuple[str, object, CiteKey | None, bool] | None:
    """Parse a list-form entity widget."""
    if not isinstance(parsed, list):
        return None
    token: object = parsed[0] if parsed and isinstance(parsed[0], str) else None
    label = next(
        (item for item in parsed[1:] if isinstance(item, str) and item.strip()),
        "",
    )
    key = _cite_key(token)
    is_product = parsed[0] == "product" if parsed else False
    is_product = is_product or (key is not None and key[1] == "product")
    if not label.strip():
        return None
    clean = " ".join(label.split()).replace("[", "(").replace("]", ")")
    return (clean, token, key, is_product)


def _entity_from_dict(
    entity: object,
) -> tuple[str, object, CiteKey | None, bool] | None:
    """Parse a dict-form entity widget."""
    if not isinstance(entity, dict):
        return None
    token: object = entity.get("cite") or entity.get("ref_id")
    key = _cite_key(token)
    kind = _opt_str(entity.get("type")).lower()
    is_product = kind in {"product", "product_entity"}
    is_product = is_product or (key is not None and key[1] == "product")
    label_raw: object = entity.get("title") or entity.get("name") or ""
    if not isinstance(label_raw, str) or not label_raw.strip():
        return None
    clean = " ".join(label_raw.split()).replace("[", "(").replace("]", ")")
    return (clean, token, key, is_product)


def _parse_entity(
    payload: str,
) -> tuple[str, object, CiteKey | None, bool] | None:
    """Return (label, token, citation key, is_product) for one entity widget."""
    try:
        parsed: object = json.loads(payload.split("\ue202", maxsplit=1)[0])
    except ValueError:
        return None
    if isinstance(parsed, list):
        return _entity_from_list(parsed)
    if isinstance(parsed, dict):
        return _entity_from_dict(parsed)
    return None


def _format_entity(payload: str, cmap: CiteMap) -> str:
    """Render an entity; product entities retain an actionable markdown link."""
    parsed = _parse_entity(payload)
    if parsed is None:
        return ""
    label, token, _key, is_product = parsed
    return _product_link(label, cmap, token) if is_product else label


def _jsx_cite_cut(raw: str, *, final: bool = False) -> int:
    """Start index of a trailing unclosed fragment worth withholding.

    A half-streamed citation tag (<Cite ref={["turn0search1) must
    never reach the client: it completes in a later cumulative snapshot.
    Mid-stream (final=False) ANY single-line unclosed fragment is withheld:
    even a token-free stub can complete into a resolvable tag later, and
    bytes emitted now could never match the re-rendered prefix. At final the
    token/120 gates decide keep-vs-drop: multi-line or token-free <Cite
    occurrences are code content and stay untouched.
    """
    p = raw.rfind("<Cite")
    if p < 0:
        return -1
    frag = raw[p:]
    if "/>" in frag or "\n" in frag:
        return -1
    if final and not (_CITE_TOKEN_RE.search(frag) or len(frag) <= JSX_CITE_SHORT_MAX):
        return -1
    return p


def _cite_tag_piece(block: str, cmap: CiteMap, *, final: bool) -> tuple[str, bool]:
    """Render one JSX cite tag; bool flags a mid-stream hold."""
    tokens = _CITE_TOKEN_RE.findall(block)
    sources = [cmap.get((int(t), rt, int(n))) for t, rt, n in tokens]
    resolved = [s for s in sources if s is not None]
    if not tokens:
        return block, False
    if len(resolved) == len(tokens):
        return " ".join(_format_source(s) for s in resolved), False
    return "", not final


def _cite_block_piece(block: str, cmap: CiteMap, *, final: bool) -> tuple[str, bool]:
    """Render one PUA cite block; bool flags a mid-stream hold."""
    tokens = _CITE_TOKEN_RE.findall(block)
    sources = [cmap.get((int(t), rt, int(n))) for t, rt, n in tokens]
    resolved = [s for s in sources if s is not None]
    if len(resolved) == len(tokens) or final:
        return " ".join(_format_source(s) for s in resolved), False
    return "", True


def _products_piece(payload: str, cmap: CiteMap, *, final: bool) -> tuple[str, bool]:
    """Render one products block; bool flags a mid-stream hold."""
    if not final and _products_unresolved(payload, cmap):
        return "", True
    return _format_products(payload, cmap), False


def _entity_piece(payload: str, cmap: CiteMap, *, final: bool) -> tuple[str, bool]:
    """Render one entity block; bool flags a mid-stream hold."""
    parsed = _parse_entity(payload)
    if parsed is None or final:
        return _format_entity(payload, cmap), False
    key = parsed[2]
    if key is not None and key not in cmap:
        return "", True
    return _format_entity(payload, cmap), False


def _link_ref_piece(
    title: str, target: str, cmap: CiteMap, *, final: bool
) -> tuple[str, bool]:
    """Render a ref-target link widget."""
    match = _CITE_TOKEN_RE.fullmatch(target)
    if match is None:
        return title, False
    first, second, third = match.group(1), match.group(2), match.group(3)
    if not all(isinstance(part, str) for part in (first, second, third)):
        return title, False
    key: CiteKey = (int(str(first)), str(second), int(str(third)))
    src = cmap.get(key)
    if src is None:
        if not final:
            return "", True
        return title, False
    url = _clean_url(src.get("url", ""))
    if title and url:
        return f"[{title}]({url})", False
    return (title or _format_source(src)), False


def _link_piece(
    raw_title: object, raw_target: object, cmap: CiteMap, *, final: bool
) -> tuple[str, bool]:
    """Render one url/video widget; bool flags a mid-stream hold."""
    title = " ".join(_opt_str(raw_title).split()).replace("[", "(").replace("]", ")")
    target = _opt_str(raw_target).strip()
    if _CITE_TOKEN_RE.fullmatch(target) is not None:
        return _link_ref_piece(title, target, cmap, final=final)
    if target.startswith(("http://", "https://")):
        url = _clean_url(target)
        return (f"[{title}]({url})" if title and url else (title or url)), False
    if title.startswith(("http://", "https://")):
        return (_clean_url(title) or title), False
    return title, False


def _collect_chart(block: str, charts_out: list[ChartSpec] | None) -> None:
    """Collect a genui chart spec without ever raising into the stream."""
    payload = block[len(_GENUI_PREFIX) : -1].lstrip("\ue202")
    try:
        loaded: object = json.loads(payload)
    except ValueError:
        return
    if not isinstance(loaded, dict):
        return
    clean: dict[str, object] = {
        key: value for key, value in loaded.items() if isinstance(key, str)
    }
    try:
        spec = chart_from_payload(clean)
    except (ValueError, TypeError, AttributeError):
        return
    if spec is not None and charts_out is not None:
        charts_out.append(spec)


def _collect_matches(raw: str) -> list[re.Match[str]]:
    """All widget/citation matches in positional order."""
    patterns = (
        _CITE_BLOCK_RE,
        _CITE_TAG_RE,
        _ENTITY_BLOCK_RE,
        _LINK_BLOCK_RE,
        _PRODUCTS_BLOCK_RE,
        _WIDGET_BLOCK_RE,
    )
    found: list[re.Match[str]] = []
    for pattern in patterns:
        found.extend(pattern.finditer(raw))
    found.sort(key=lambda item: item.start())
    return found


def _render_citations(
    raw: str,
    cmap: CiteMap,
    *,
    final: bool = False,
    charts_out: list[ChartSpec] | None = None,
) -> tuple[str, int]:
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
    hold = -1
    if not final:
        cut = _jsx_cite_cut(raw)
        if cut >= 0:
            raw = raw[:cut]
    matches = _collect_matches(raw)
    for match in matches:
        gap = raw[pos : match.start()]
        out.append(gap)
        if hold < 0:
            safe += len(gap)
        pos = match.end()
        piece, needs_hold = _render_match(match, cmap, charts_out, final=final)
        if needs_hold and hold < 0:
            hold = safe
        out.append(piece)
        if hold < 0:
            safe += len(piece)
    tail = raw[pos:]
    if tail:
        safe, hold = _append_tail(out, tail, safe, hold, final=final)
    text = "".join(out)
    if final:
        return _finalize_text(text)
    return text, min(safe, len(text))


def _render_match(
    match: re.Match[str],
    cmap: CiteMap,
    charts_out: list[ChartSpec] | None,
    *,
    final: bool,
) -> tuple[str, bool]:
    """Render one citation/widget match; bool flags a mid-stream hold."""
    text = match.group(0)
    if match.re is _CITE_TAG_RE:
        return _cite_tag_piece(text, cmap, final=final)
    if text.startswith("\ue200cite"):
        return _cite_block_piece(text, cmap, final=final)
    if text.startswith("\ue200products\ue202"):
        return _products_piece(match.group(1), cmap, final=final)
    if text.startswith("\ue200entity\ue202"):
        return _entity_piece(match.group(1), cmap, final=final)
    if text.startswith(("\ue200url\ue202", "\ue200video\ue202")):
        return _link_piece(match.group(1), match.group(2), cmap, final=final)
    if text.startswith(_GENUI_PREFIX):
        _collect_chart(text, charts_out)
    return "", False


def _append_tail(
    out: list[str], tail: str, safe: int, hold: int, *, final: bool
) -> tuple[int, int]:
    """Append trailing text, withholding a half-received marker mid-stream."""
    cut = -1 if hold >= 0 or final else tail.find("\ue200")
    if cut >= 0:
        out.append(tail[:cut])
        if hold < 0:
            safe += cut
            hold = safe
    else:
        out.append(tail)
        if hold < 0:
            safe += len(tail)
    return safe, hold


def _finalize_text(text: str) -> tuple[str, int]:
    """Strip leftover marker scaffolding at end of stream."""
    cut = _jsx_cite_cut(text, final=True)
    if cut >= 0:
        text = text[:cut]
    text = _PUA_CHARS_RE.sub("", text)
    return text, len(text)


async def _upload_inputs(
    acct: AccountSession,
    items: list[HistoryItem],
    *,
    strict_from: int = 0,
) -> tuple[list[JsonObject], list[JsonObject]]:
    """Upload every image and non-text file of the given turns to THIS account.

    Uploads are account-scoped: an attempt served by another account must
    re-upload them all -- that is how attachments survive account failover.

    strict_from: index of the first item whose images MUST validate. Items
    before it are replayed history: an image that no longer opens (e.g. an
    expired remote URL re-fetched at parse time) is skipped with a warning
    instead of turning a servable conversation into a hard 400.
    """
    pointers: list[JsonObject] = []
    attachments: list[JsonObject] = []
    for idx, item in enumerate(items):
        for im in item.images:
            try:
                with Image.open(io.BytesIO(im.data)) as pic:
                    width, height = pic.size
            except (OSError, ValueError) as exc:
                if idx >= strict_from:
                    message = f"unsupported or corrupt image: {im.filename}"
                    raise EngineError(400, message, "invalid_request_error") from exc
                log.warning(
                    "skipping unopenable history image %s during replay",
                    im.filename,
                )
                continue
            fid = await acct.upload_file(im.filename, im.data, im.mime, is_image=True)
            pointer: JsonObject = {
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
            attachments.append(
                {
                    "id": fid,
                    "name": f.filename,
                    "mimeType": f.mime,
                    "size": len(f.data),
                },
            )
    return pointers, attachments


def _history_hashes(items: list[HistoryItem], system_text: str) -> list[str]:
    """Hash chain for prefix matching stored conversations."""
    prev = item_hash("", "system", system_text) if system_text else ""
    out: list[str] = []
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
            mime = (
                "image/jpeg"
                if name.lower().endswith((".jpg", ".jpeg"))
                else "image/png"
            )
            urls.append(await upload_image(name, data, mime))
        except (ChatGPTError, PixelVaultError) as e:
            log.warning("image resolution failed (%s): %s", ptr, e)
    return urls


def _replay_prompt(system_text: str, items: list[HistoryItem]) -> str:
    """Render the whole transcript for a fresh conversation.

    Preserves every turn on a non-owner account. Text-like files were already
    fenced into their turn's text at parse time; images/binary files attach
    separately.
    """
    blocks: list[str] = []
    if system_text:
        blocks.append(f"[system]\n{system_text}")
    blocks.extend(f"[{item.role}]\n{item.text}" for item in items)
    return "\n\n".join(blocks)


def _chain_context(
    prev_snap: TurnSnapshot,
    parsed: ParsedRequest,
    hashes: list[str],
) -> tuple[str, list[HistoryItem]]:
    """Merge a referenced response's recorded context with this request's turns.

    Clients of previous_response_id flows usually send only NEW items per call;
    if instead they resent the full history (hash chain extends the snapshot's),
    the request alone is already complete.
    """
    base_hashes = _history_hashes(prev_snap.items, prev_snap.system_text)
    client_resent_all = (
        len(hashes) >= len(base_hashes) and hashes[: len(base_hashes)] == base_hashes
    )
    if client_resent_all:
        return parsed.system_text, list(parsed.items)
    # Cumulative: ancestor context + this round. Items are shared references,
    # so store-wide byte accounting may over-count shared buffers -- that only
    # makes budget eviction earlier, never memory use larger than reality.
    return prev_snap.system_text, list(prev_snap.items) + list(parsed.items)


def _fallback_context(
    parsed: ParsedRequest,
    hashes: list[str],
    prev_response_id: str | None,
) -> tuple[str, list[HistoryItem]]:
    """Full context to replay when serving the turn on a non-owner account."""
    ctx_sys, ctx_items = parsed.system_text, parsed.items
    if prev_response_id:
        snap = STORE.get_snapshot(prev_response_id)
        if snap is not None:
            ctx_sys, ctx_items = _chain_context(snap, parsed, hashes)
    return ctx_sys, ctx_items


def _plan(
    parsed: ParsedRequest,
    previous_response_id: str | None,
) -> tuple[list[str], ConvRef | None, int]:
    """Decide which stored server conversation to continue.

    Returns how much of the request history is new.
    """
    hashes = _history_hashes(parsed.items, parsed.system_text)
    if previous_response_id:
        rec = STORE.get_response(previous_response_id)
        if rec is None:
            message = (
                f"previous_response_id '{previous_response_id}' not found or expired"
            )
            raise EngineError(400, message, "invalid_request_error")
        ref = ConvRef(
            rec.account_identity,
            rec.conversation_id,
            rec.parent_id,
            0,
            time.time(),
        )
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


def _raise_policy_error(acct_email: str, err_text: str) -> None:
    """Raise a moderation rejection before any byte streams."""
    message = f"[{acct_email}] ChatGPT rejected this prompt: {err_text[:200]}"
    raise ChatGPTError(502, message)


def _raise_image_limit(text: str) -> None:
    """Raise a per-account image quota refusal."""
    message = text.strip()[:300]
    raise _ImageLimitError(message)


def _raise_incomplete(last_event: object) -> None:
    """Raise when the stream ended without a usable conversation."""
    tail = "; stream ended without any events"
    if isinstance(last_event, dict) and last_event:
        event_type = _opt_str(last_event.get("type", "?"))
        tail = f"; last SSE event type: {event_type}"
    message = "ChatGPT returned an incomplete response" + tail
    raise EngineError(502, message)


def _live_slugs(models: object) -> set[str]:
    """Live model slugs as a string set."""
    if not isinstance(models, list):
        return set()
    slugs: set[str] = set()
    for entry in models:
        if not isinstance(entry, dict):
            continue
        slug = entry.get("slug")
        if isinstance(slug, str) and slug:
            slugs.add(slug)
    return slugs


def _failure_for(
    exc: BaseException, acct: AccountSession, pool: AccountPool
) -> EngineError:
    """Map an attempt exception to the failover error."""
    if isinstance(exc, _ImageLimitError):
        pool.report_status(acct, HTTP_RATE_LIMITED)
        message = str(exc)
        return EngineError(HTTP_RATE_LIMITED, message, "rate_limit_error")
    if isinstance(exc, ChatGPTError):
        pool.report_status(acct, exc.status)
        if exc.status == HTTP_RATE_LIMITED:
            return EngineError(HTTP_RATE_LIMITED, exc.message, "rate_limit_error")
        return EngineError(502, exc.message, "server_error")
    if isinstance(exc, EngineError):
        return exc
    message = f"{type(exc).__name__}: {exc}"
    return EngineError(502, message)


async def _prepare_attempt(
    acct: AccountSession,
    parsed: ParsedRequest,
    ref: ConvRef | None,
    hashes: list[str],
    previous_response_id: str | None,
) -> tuple[str, list[JsonObject], list[JsonObject]]:
    """Upload inputs and build the prompt for one attempt."""
    current = parsed.items[-1]
    continuing = ref is not None and acct.identity == ref.account_identity
    if continuing or (ref is None and len(parsed.items) == 1):
        pointers, attachments = await _upload_inputs(acct, [current])
        prompt = current.text
        if parsed.system_text and ref is None:
            if prompt:
                prompt = f"{parsed.system_text}\n\n{prompt}"
            else:
                prompt = parsed.system_text
        return prompt, pointers, attachments
    ctx_sys, ctx_items = _fallback_context(parsed, hashes, previous_response_id)
    hist_len = len(ctx_items) - len(parsed.items)
    pointers, attachments = await _upload_inputs(acct, ctx_items, strict_from=hist_len)
    return _replay_prompt(ctx_sys, ctx_items), pointers, attachments


def _is_rival_branch(text_acc: str, parts: list[object]) -> bool:
    """Whether a new node belongs to a rival concurrent branch."""
    if not text_acc:
        return False
    for part in parts:
        if isinstance(part, str) and part.startswith(text_acc):
            return False
    return True


def _maybe_adopt_branch(
    author: JsonObject,
    message: JsonObject,
    current_msg_id: str,
    text_acc: str,
    parts: list[object],
) -> str:
    """Adopt a new assistant node when it continues the streamed text."""
    if author.get("role") != "assistant":
        return current_msg_id
    node_id = message.get("id")
    if not isinstance(node_id, str) or not node_id:
        return current_msg_id
    if node_id == current_msg_id:
        return current_msg_id
    if _is_rival_branch(text_acc, parts):
        return current_msg_id
    return node_id


def _policy_error_text(
    author: JsonObject,
    message: JsonObject,
    parts: list[object],
    *,
    is_text_node: bool,
) -> str | None:
    """Moderation error text when the node flags is_error, else None."""
    if author.get("role") != "assistant" or not is_text_node:
        return None
    metadata = message.get("metadata")
    if not isinstance(metadata, dict) or not metadata.get("is_error"):
        return None
    return " ".join(str(p) for p in parts if isinstance(p, str)).strip()


def _is_streamable_part(
    part: object,
    author: JsonObject,
    message: JsonObject,
    current_msg_id: str,
    *,
    is_text_node: bool,
) -> bool:
    """Whether a message part carries client-visible streamed prose."""
    if not isinstance(part, str):
        return False
    if author.get("role") != "assistant" or not is_text_node:
        return False
    node_id = message.get("id")
    return not node_id or node_id == current_msg_id


def _text_delta(
    text_acc: str, cite_map: CiteMap, emitted: int
) -> tuple[str | None, bool]:
    """Delta to emit for new accumulated text plus limit flag."""
    emit_upto, is_limit = _classify_accumulated(text_acc)
    if is_limit:
        return None, True
    display, safe_upto = _render_citations(text_acc, cite_map)
    if emit_upto > 0 and safe_upto > emitted:
        return display[emitted:safe_upto], False
    return None, False


async def _chart_link_deltas(specs: list[ChartSpec]) -> list[str]:
    """Rendered chart image links, degrading to stripped on failure."""
    links: list[str] = []
    for spec in specs:
        meta = spec.get("meta")
        raw_title: object = "chart"
        if isinstance(meta, dict):
            raw_title = meta.get("title") or "chart"
        title = " ".join(_opt_str(raw_title).split())
        alt = f"chart: {title}".replace("[", "(").replace("]", ")")
        try:
            png = await asyncio.to_thread(render_chart_png, spec)
            url = await upload_image("chart.png", png, "image/png")
            links.append(f"![{alt}]({url})")
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            RuntimeError,
            PixelVaultError,
        ) as exc:
            log.warning("chart render/upload failed (%s): %s", title[:60], exc)
    return links


def _latest_user_query(items: list[HistoryItem]) -> str:
    """Most recent user text for source appendix attribution."""
    for item in reversed(items):
        if item.role == "user" and item.text:
            return item.text
    return ""


def _salvage_text(
    text_acc: str, cite_map: CiteMap, emitted: int, *, fully_flushed: bool
) -> str | None:
    """Withheld text releasable on the failure path, or None."""
    _, still_limit = _classify_accumulated(text_acc)
    if still_limit or fully_flushed:
        return None
    try:
        salvaged, _ = _render_citations(text_acc, cite_map, final=True)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        log.warning("salvage flush failed: %s", exc)
        return None
    if emitted < len(salvaged):
        return salvaged[emitted:]
    return None


@dataclass
class _AttemptState:
    """Mutable per-attempt stream state for one account try."""

    text_acc: str = ""
    emitted: int = 0
    fully_flushed: bool = False
    cite_map: CiteMap = field(default_factory=dict)
    chart_specs: list[ChartSpec] = field(default_factory=list)
    current_msg_id: str = ""
    sediment: list[str] = field(default_factory=list)
    cid: str = ""
    parent: str = ""
    limit_hit: bool = False
    last_event: object = field(default_factory=dict)
    last_node_id: str = ""
    produced: bool = False
    model: str = ""
    created: int = 0
    image_urls: list[str] = field(default_factory=list)


@dataclass
class _TurnContext:
    """Immutable per-turn context shared across account attempts."""

    parsed: ParsedRequest
    ref: ConvRef | None
    hashes: list[str]
    previous_response_id: str | None
    rid: str


def _validate_request(parsed: ParsedRequest) -> None:
    """Reject empty or non-user-terminated histories."""
    if not parsed.items:
        message = "no messages"
        raise EngineError(400, message, "invalid_request_error")
    if parsed.items[-1].role != "user":
        message = "last message must have role=user"
        raise EngineError(400, message, "invalid_request_error")


def _acquire_attempt_account(
    pool: AccountPool,
    preferred: str | None,
    tried: set[str],
    last_failure: EngineError | None,
) -> AccountSession:
    """Acquire the next untried account or raise the terminal error."""
    try:
        return pool.acquire(preferred, exclude=tried)
    except NoAccountAvailableError as exc:
        if last_failure is not None:
            count = len(tried)
            message = (
                f"all {count} available account(s) failed; "
                f"last error: {last_failure.message}"
            )
            raise EngineError(
                last_failure.status, message, last_failure.error_type
            ) from exc
        message = str(exc)
        raise EngineError(503, message, "rate_limit_error") from exc


def _adopt_branch_state(
    state: _AttemptState,
    author: dict[str, Any],
    message: dict[str, Any],
    parts: list[object],
) -> None:
    """Adopt a newly continued assistant node into the attempt state."""
    if author.get("role") != "assistant" or not message.get("id"):
        return
    adopted = _maybe_adopt_branch(
        author, message, state.current_msg_id, state.text_acc, parts
    )
    if adopted != state.current_msg_id:
        state.current_msg_id = adopted
        if not state.text_acc:
            state.emitted = 0
            state.text_acc = ""


def _ingest_parts(
    state: _AttemptState,
    parts: list[object],
    author: dict[str, Any],
    message: dict[str, Any],
    *,
    is_text_node: bool,
) -> Iterator[dict[str, object]]:
    """Fold stream parts into state, yielding text deltas."""
    for part in parts:
        if isinstance(part, dict):
            ptr = part.get("asset_pointer")
            if (
                isinstance(ptr, str)
                and "sediment://" in ptr
                and ptr not in state.sediment
            ):
                state.sediment.append(ptr)
        elif _is_streamable_part(
            part, author, message, state.current_msg_id, is_text_node=is_text_node
        ):
            state.text_acc = str(part)
            delta, is_limit = _text_delta(state.text_acc, state.cite_map, state.emitted)
            if is_limit:
                state.limit_hit = True
                return
            if delta:
                state.produced = True
                state.emitted += len(delta)
                yield {"type": "delta", "text": delta}


def _ingest_event(
    state: _AttemptState, ev: dict[str, Any], acct_email: str
) -> Iterator[dict[str, object]]:
    """Fold one SSE event into state, yielding text deltas."""
    state.last_event = ev
    if ev.get("conversation_id") and not state.cid:
        state.cid = ev["conversation_id"]
    message = ev.get("message") or {}
    _cite_map_extend(state.cite_map, message.get("metadata"))
    author = message.get("author") or {}
    if message.get("id"):
        state.last_node_id = message["id"]
    if author.get("role") not in ("assistant", "tool"):
        return
    content = message.get("content") or {}
    parts = content.get("parts") or []
    # Only 'text' nodes carry client-visible prose. thoughts/code/
    # reasoning_recap/model_editable_context nodes share the
    # assistant role; their string parts (when present) are
    # internal channel data and must never stream as the answer.
    is_text_node = (content.get("content_type") or "text") == "text"
    _adopt_branch_state(state, author, message, parts)
    err_text = _policy_error_text(author, message, parts, is_text_node=is_text_node)
    if err_text is not None:
        _raise_policy_error(acct_email, err_text)
    yield from _ingest_parts(state, parts, author, message, is_text_node=is_text_node)
    if state.limit_hit:
        return
    meta = message.get("metadata") or {}
    is_completion = meta.get("message_type") in ("next", "continue")
    is_rival_marker = (
        author.get("role") == "assistant"
        and message.get("id")
        and message["id"] != state.current_msg_id
    )
    # A rival branch's completion marker must not repoint the
    # stored conversation parent away from the branch the
    # client actually received. An unadopted assistant marker
    # is always a rival's: a node that streamed text would
    # have been adopted within its own event.
    if is_completion and not is_rival_marker:
        state.parent = message.get("id") or state.parent


def _ensure_stream_complete(state: _AttemptState, acct_email: str) -> None:
    """Raise for quota refusals and incomplete streams after the pump."""
    # Refusals are valid HTTP-200 streams and may intentionally stop
    # before ChatGPT emits the normal assistant "next" marker.
    if state.limit_hit or (state.text_acc and IMAGE_LIMIT_RE.search(state.text_acc)):
        _raise_image_limit(state.text_acc)
    # Image-generation turns can end with ONLY a tool node carrying the
    # sediment pointer -- no assistant message, no "next" marker. Such a
    # turn is complete: its product is the generated image.
    if not state.cid or (not state.parent and not state.sediment):
        # Full frame goes to the server log only; the client-facing
        # error keeps just the event type so session-scoped tokens and
        # ids never leave the process.
        if state.last_event:
            log.warning(
                "incomplete stream on %s: last SSE event %s",
                acct_email,
                str(state.last_event)[:500],
            )
        _raise_incomplete(state.last_event)
    if not state.parent:
        # Keep the stored conversation continuable: parent the next
        # turn to the branch the client actually received (the
        # adopted assistant node), falling back to the last node we
        # saw (e.g. a tool node that carries the sediment pointer).
        state.parent = state.current_msg_id or state.last_node_id


def _flush_attempt_text(state: _AttemptState) -> str | None:
    """Release withheld reply text at the end of a successful stream."""
    # Flush anything still withheld first: short normal replies that
    # begin like the refusal must reach the client in full. final=True
    # also releases citation blocks whose sources never arrived.
    display, _ = _render_citations(
        state.text_acc,
        state.cite_map,
        final=True,
        charts_out=state.chart_specs,
    )
    state.text_acc = display
    state.fully_flushed = True
    if state.emitted < len(display):
        state.produced = True
        return display[state.emitted :]
    return None


async def _collect_media_deltas(
    state: _AttemptState, acct: AccountSession, parsed: ParsedRequest
) -> AsyncIterator[dict[str, object]]:
    """Yield chart, image, and source deltas for a flushed attempt."""
    # genui chart specs -> locally rendered PNG -> PixelVault link.
    # Any failure degrades to the widget simply being stripped.
    chart_links = await _chart_link_deltas(state.chart_specs)
    if chart_links:
        links = "\n\n" + "\n\n".join(chart_links)
        state.produced = True
        state.text_acc += links
        yield {"type": "delta", "text": links}
    image_urls = await _resolve_images(acct, state.sediment)
    if image_urls:
        links = "\n\n" + "\n\n".join(f"![generated image]({url})" for url in image_urls)
        state.produced = True
        state.text_acc += links
        state.image_urls = image_urls
        yield {"type": "delta", "text": links}
    inc_sources = (
        config.INCLUDE_SOURCES
        if parsed.include_sources is None
        else parsed.include_sources
    )
    if inc_sources and state.cite_map:
        user_query = _latest_user_query(parsed.items)
        appendix = _source_appendix(list(state.cite_map.values()), query=user_query)
        if appendix:
            state.produced = True
            state.text_acc += appendix
            yield {"type": "delta", "text": appendix}


async def _pump_stream_events(
    acct: AccountSession, state: _AttemptState, req: _StreamRequest
) -> AsyncIterator[dict[str, object]]:
    """Stream one attempt's SSE events into state, yielding deltas."""
    async for ev in acct.stream_conversation(
        prompt_text=req.prompt,
        media=req.media,
        parent_message_id=req.parent_id,
        conversation_id=req.conversation_id,
        model=state.model,
    ):
        for delta in _ingest_event(state, ev, acct.email):
            yield delta
        if state.limit_hit:
            break


async def _emit_tail_deltas(
    state: _AttemptState, acct: AccountSession, parsed: ParsedRequest
) -> AsyncIterator[dict[str, object]]:
    """Yield flush, media, and source deltas for a completed attempt."""
    flushed = _flush_attempt_text(state)
    if flushed is not None:
        yield {"type": "delta", "text": flushed}
    async for delta in _collect_media_deltas(state, acct, parsed):
        yield delta


def _record_done(
    acct: AccountSession, turn: _TurnContext, state: _AttemptState
) -> dict[str, object]:
    """Persist turn state and build the done event for a success."""
    parsed = turn.parsed
    new_ref = ConvRef(
        acct.identity,
        state.cid,
        state.parent,
        len(parsed.items),
        time.time(),
    )
    if turn.hashes:
        STORE.record_turn([turn.hashes[-1]], new_ref)
    # Record CUMULATIVE context: stateful clients send only new items per
    # call, so a delta-only snapshot would lose earlier turns on a later
    # failover (chained via previous_response_id).
    snap_sys, snap_items = parsed.system_text, list(parsed.items)
    if turn.previous_response_id:
        prev_snap = STORE.get_snapshot(turn.previous_response_id)
        if prev_snap is not None:
            snap_sys, snap_items = _chain_context(prev_snap, parsed, turn.hashes)
    STORE.put_response(
        ResponseRecord(
            turn.rid,
            acct.identity,
            state.cid,
            state.parent,
            state.model,
            time.time(),
        ),
        TurnSnapshot(system_text=snap_sys, items=snap_items),
    )
    return {
        "type": "done",
        "result": TurnResult(
            text=state.text_acc,
            conversation_id=state.cid,
            parent_id=state.parent,
            model=state.model,
            created=state.created,
            response_id=turn.rid,
            image_urls=state.image_urls,
            prompt_tokens=estimate_tokens(
                parsed.system_text,
                *(x.text for x in parsed.items),
            ),
            completion_tokens=estimate_tokens(state.text_acc),
            account_email=acct.email,
        ),
    }


@dataclass
class _StreamRequest:
    """Everything needed to open one attempt's SSE stream."""

    prompt: str
    media: StreamMedia
    parent_id: str
    conversation_id: str | None


def _initial_preference(ref: ConvRef | None, preferred_email: str | None) -> str | None:
    """Account preferred for the first attempt of a turn."""
    if ref is not None:
        return ref.account_identity
    return preferred_email or None


def _untried_count(pool: AccountPool, tried: set[str]) -> int:
    """Available accounts not yet tried this turn."""
    return sum(1 for acct in pool.available() if acct.identity not in tried)


async def run_turn(
    parsed: ParsedRequest,
    pool: AccountPool,
    *,
    previous_response_id: str | None = None,
    response_id_prefix: str = "resp_",
    preferred_email: str | None = None,
) -> AsyncIterator[dict[str, object]]:
    """Yield delta events, then a done event with the turn result.

    Every available account is tried exactly once before an error is raised.
    Attempts that cannot continue the stored server-side conversation start a
    fresh one and replay the FULL client context: every turn's text plus that
    turn's images and file attachments, re-uploaded to whichever account
    actually serves the attempt -- so failover never loses content.
    """
    _validate_request(parsed)

    hashes, ref, matched = _plan(parsed, previous_response_id)
    log.info(
        "turn plan: history=%d matched=%d %s",
        len(parsed.items),
        matched,
        f"continue conv={ref.conversation_id}" if ref else "new conversation",
    )

    turn = _TurnContext(
        parsed=parsed,
        ref=ref,
        hashes=hashes,
        previous_response_id=previous_response_id,
        rid=response_id_prefix + uuid.uuid4().hex,
    )

    # Attempt #1 prefers the stored conversation's owner / requested account;
    # afterwards EVERY remaining available account is tried before failing.
    preferred = _initial_preference(ref, preferred_email)
    tried: set[str] = set()
    last_failure: EngineError | None = None

    while True:
        acct = _acquire_attempt_account(pool, preferred, tried, last_failure)
        tried.add(acct.identity)
        preferred = None
        continuing = ref is not None and acct.identity == ref.account_identity
        state = _AttemptState()
        try:
            acct.total_requests += 1
            live = _live_slugs(await acct.models())
            state.model = map_model(parsed.model_requested, live)
            yield {"type": "model", "model": state.model}
            prompt, pointers, attachments = await _prepare_attempt(
                acct, parsed, ref, hashes, previous_response_id
            )

            parent_hint = ref.parent_id if continuing else str(uuid.uuid4())
            conv_hint = ref.conversation_id if continuing else None
            req = _StreamRequest(
                prompt=prompt,
                media=StreamMedia(image_pointers=pointers, attachments=attachments),
                parent_id=parent_hint,
                conversation_id=conv_hint,
            )
            async for delta in _pump_stream_events(acct, state, req):
                yield delta

            _ensure_stream_complete(state, acct.email)
            async for delta in _emit_tail_deltas(state, acct, parsed):
                yield delta
            state.created = int(time.time())
            yield _record_done(acct, turn, state)
        except (
            ChatGPTError,
            EngineError,
            RequestException,
            _ImageLimitError,
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            RuntimeError,
            PixelVaultError,
        ) as exc:
            if isinstance(exc, EngineError) and (
                exc.error_type == "invalid_request_error"
            ):
                raise
            failure = _failure_for(exc, acct, pool)
            if state.produced:
                salvaged = _salvage_text(
                    state.text_acc,
                    state.cite_map,
                    state.emitted,
                    fully_flushed=state.fully_flushed,
                )
                if salvaged is not None:
                    yield {"type": "delta", "text": salvaged}
                raise failure from exc
            last_failure = failure
            remaining = _untried_count(pool, tried)
            log.warning(
                "attempt on %s failed (%s): %s; %d account(s) still untried",
                acct.email,
                failure.error_type,
                failure.message[:160],
                remaining,
            )
        else:
            return
        finally:
            pool.release(acct)


async def collect(
    parsed: ParsedRequest,
    pool: AccountPool,
    *,
    previous_response_id: str | None = None,
    preferred_email: str | None = None,
) -> TurnResult:
    """Non-streaming convenience wrapper."""
    result: TurnResult | None = None
    async for ev in run_turn(
        parsed,
        pool,
        previous_response_id=previous_response_id,
        preferred_email=preferred_email,
    ):
        if ev.get("type") == "done":
            candidate = ev.get("result")
            if isinstance(candidate, TurnResult):
                result = candidate
    if result is None:
        message = "no result"
        raise EngineError(502, message)
    return result
