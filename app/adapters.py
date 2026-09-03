# Copyright 2026 chatgpt-to-openai-api contributors.
"""Pure translation between OpenAI wire formats and internal turn structures."""

from __future__ import annotations

import base64
import ipaddress
import mimetypes
import re
import socket
import time
import uuid
from dataclasses import dataclass, field
from typing import TypeGuard
from urllib.parse import unquote_to_bytes, urlparse

from curl_cffi.requests import AsyncSession

from . import config as cfg

# ---------------------------------------------------------------- models

LIVE_SLUG_ALIASES: dict[str, list[str]] = {
    # incoming name -> candidate chatgpt slugs (first match against live list wins)
    "gpt-4o": ["gpt-4o"],
    "gpt-4o-mini": ["gpt-4o-mini"],
    "gpt-4": ["gpt-4"],
    "gpt-4.1": ["gpt-4-1"],
    "gpt-5": ["gpt-5", "gpt-5-5", "auto"],
    "gpt-5.5": ["gpt-5-5"],
    "gpt-5.6": ["gpt-5-6", "auto"],
    "chatgpt-auto": ["auto"],
    "auto": ["auto"],
}

HTTP_OK = 200
MAX_REMOTE_BYTES = 25 * 1024 * 1024


def map_model(requested: str | None, live_slugs: set[str]) -> str:
    """Map a requested model name to a live ChatGPT slug."""
    if not requested:
        return config_default(live_slugs)
    r = requested.strip().lower()
    if r in live_slugs:
        return r
    normalized = r.replace(".", "-")
    if normalized in live_slugs:
        return normalized
    for cand in LIVE_SLUG_ALIASES.get(r, []):
        if cand in live_slugs:
            return cand
    return config_default(live_slugs)


def config_default(live_slugs: set[str]) -> str:
    """Return the configured default model when live, else auto."""
    d = cfg.DEFAULT_MODEL
    if d in live_slugs:
        return d
    return "auto"


def public_models(
    models_payload: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Build an OpenAI-style model list from live ChatGPT slugs plus auto."""
    seen: list[str] = []
    for model in models_payload:
        slug = model.get("slug")
        if isinstance(slug, str) and slug and slug not in seen:
            seen.append(slug)
    return [
        {
            "id": model_id,
            "object": "model",
            "created": 1785000000,
            "owned_by": "chatgpt-proxy",
        }
        for model_id in sorted(set(seen) | {"auto"})
    ]


# ---------------------------------------------------------------- history items


@dataclass
class ImageInput:
    """One image attachment with its raw bytes."""

    filename: str
    mime: str
    data: bytes


@dataclass
class FileInput:
    """One non-image file attachment with its raw bytes."""

    filename: str
    mime: str
    data: bytes


@dataclass
class HistoryItem:
    """One conversation turn with its text and binary attachments."""

    role: str  # user | assistant
    text: str = ""
    images: list[ImageInput] = field(default_factory=list)
    files: list[FileInput] = field(default_factory=list)

    def canon(self) -> str:
        """Return stable canonical text including attachment descriptors."""
        extras = [f"img:{im.filename}:{len(im.data)}" for im in self.images]
        extras.extend(f"file:{f.filename}:{len(f.data)}" for f in self.files)
        if extras:
            return self.text + "\x00" + "\x00".join(extras)
        return self.text


@dataclass
class ParsedRequest:
    """Normalized turns parsed from an OpenAI wire request."""

    system_text: str
    items: list[HistoryItem]
    model_requested: str | None
    stream: bool
    include_sources: bool | None = None


# ---------------------------------------------------------------- content helpers

_DATA_URL_RE = re.compile(r"^data:([^;,]+)?(;base64)?,(.*)$", re.DOTALL)

TEXTUAL_MIME_PREFIXES: tuple[str, ...] = ("text/",)
TEXTUAL_MIMES: set[str] = {
    "application/json",
    "application/xml",
    "application/javascript",
    "application/x-yaml",
    "application/yaml",
    "application/toml",
    "application/x-sh",
    "application/sql",
    "application/graphql",
    "application/x-python",
    "application/csv",
}
CODE_EXTS: set[str] = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".json",
    ".yml",
    ".yaml",
    ".toml",
    ".ini",
    ".cfg",
    ".sh",
    ".bash",
    ".zsh",
    ".sql",
    ".html",
    ".css",
    ".scss",
    ".rs",
    ".go",
    ".java",
    ".kt",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cs",
    ".rb",
    ".php",
    ".swift",
    ".m",
    ".mm",
    ".lua",
    ".pl",
    ".r",
    ".jl",
    ".md",
    ".txt",
    ".csv",
    ".tsv",
    ".xml",
    ".svg",
    ".diff",
    ".patch",
    ".log",
    ".dockerfile",
    ".makefile",
    ".env",
    ".gitignore",
    ".proto",
    ".tf",
    ".hcl",
    ".vue",
    ".svelte",
    ".astro",
    ".dart",
    ".ex",
    ".exs",
    ".erl",
    ".hs",
    ".clj",
    ".scala",
    ".groovy",
}
IMAGE_MIMES: set[str] = {"image/png", "image/jpeg", "image/webp", "image/gif"}
EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "jsx",
    ".json": "json",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".toml": "toml",
    ".sh": "bash",
    ".bash": "bash",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".c": "c",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".md": "markdown",
    ".xml": "xml",
    ".svg": "xml",
    ".csv": "csv",
    ".ini": "ini",
    ".lua": "lua",
    ".r": "r",
    ".dart": "dart",
}


def decode_data_url(value: str) -> tuple[str, bytes]:
    """Split a data: URL into its mime type and raw bytes."""
    match = _DATA_URL_RE.match(value.strip())
    if not match:
        msg = "expected a data: URL"
        raise ValueError(msg)
    mime_raw: object = match.group(1)
    mime = mime_raw.lower() if isinstance(mime_raw, str) else "application/octet-stream"
    payload_raw: object = match.group(3)
    payload = payload_raw if isinstance(payload_raw, str) else ""
    if match.group(2):
        return mime, base64.b64decode(payload)
    return mime, unquote_to_bytes(payload)


async def fetch_remote(url: str) -> tuple[str, bytes]:
    """Fetch a remote image URL, refusing SSRF targets."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        msg = "only http(s) image URLs are supported"
        raise ValueError(msg)
    host = parsed.hostname or ""
    # SSRF guard: refuse loopback/private/link-local/metadata targets.
    bad_ip = False
    try:
        infos = socket.getaddrinfo(host, None)
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if (
                ip.is_loopback
                or ip.is_private
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
            ):
                bad_ip = True
                break
    except socket.gaierror as e:
        msg = f"cannot resolve image host: {host}"
        raise ValueError(msg) from e
    if bad_ip or host in ("localhost", "metadata.google.internal"):
        msg = "refusing to fetch image from private address"
        raise ValueError(msg)

    session = AsyncSession()
    try:
        response = await session.get(url, timeout=60)
        if response.status_code != HTTP_OK:
            msg = f"failed to fetch {url}: HTTP {response.status_code}"
            raise ValueError(msg)
        content_raw: object = response.headers.get("content-type")
        media_type = "application/octet-stream"
        if isinstance(content_raw, str):
            media_type = content_raw.split(";")[0].strip().lower()
        raw_body = response.content
        body: bytes = raw_body if isinstance(raw_body, bytes) else bytes(raw_body)
        if len(body) > MAX_REMOTE_BYTES:
            msg = "image larger than 25 MB"
            raise ValueError(msg)
        return media_type, body
    finally:
        await session.close()


def is_textual(mime: str, filename: str) -> bool:
    """Check whether an attachment should be inlined as text."""
    if mime in TEXTUAL_MIMES or mime.startswith(TEXTUAL_MIME_PREFIXES):
        return True
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in CODE_EXTS


def guess_mime(filename: str, fallback: str = "application/octet-stream") -> str:
    """Guess a filename's mime type, returning the fallback when unknown."""
    mt, _ = mimetypes.guess_type(filename)
    return mt or fallback


def fence(filename: str, data: bytes) -> str:
    """Wrap file bytes as a labeled markdown code block."""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    lang = EXT_TO_LANG.get(ext, "")
    try:
        body = data.decode("utf-8")
    except UnicodeDecodeError:
        body = base64.b64encode(data).decode()
        lang = ""
    return f"{filename}\n```{lang}\n{body}\n```"


def extract_text(content: object) -> str:
    """Collect plain text from OpenAI message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for chunk in content:
            if isinstance(chunk, str):
                parts.append(chunk)
            elif isinstance(chunk, dict):
                text_raw = chunk.get("text")
                if isinstance(text_raw, str):
                    parts.append(text_raw)
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def _optional_str(value: object) -> str | None:
    """Return the value when it is a string, else None."""
    return value if isinstance(value, str) else None


def _optional_bool(value: object) -> bool | None:
    """Return the value when it is a bool, else None."""
    return value if isinstance(value, bool) else None


def _require_str(value: object, message: str) -> str:
    """Return the value when it is a string, raising ValueError otherwise."""
    if isinstance(value, str):
        return value
    raise ValueError(message)


def _is_dict(value: object) -> TypeGuard[dict[str, object]]:
    """Check whether the value is a string-keyed mapping."""
    return isinstance(value, dict)


def _require_dict(value: object, message: str) -> dict[str, object]:
    """Return the value when it is a dict, raising ValueError otherwise."""
    if _is_dict(value):
        return value
    raise ValueError(message)


async def _load_image_bytes(url: str) -> tuple[str, bytes] | None:
    """Decode image bytes from a data: or http(s) URL, else None."""
    if url.startswith("data:"):
        return decode_data_url(url)
    if url.startswith(("http://", "https://")):
        return await fetch_remote(url)
    return None


async def _chat_image_url(url: str) -> ImageInput:
    """Build an image input from a chat image_url URL."""
    loaded = await _load_image_bytes(url)
    if loaded is None:
        msg = "image_url must be a data or http(s) URL"
        raise ValueError(msg)
    mime, data = loaded
    if url.startswith("data:"):
        subtype = mime.split("/", maxsplit=1)[1] if "/" in mime else "bin"
        return ImageInput("image." + subtype, mime, data)
    name = url.rsplit("/", 1)[-1].split("?", maxsplit=1)[0] or "image"
    return ImageInput(name, mime, data)


def _attach_file_data(item: HistoryItem, texts: list[str], fname: str, fd: str) -> None:
    """Decode inline file bytes into an image, text block, or binary file."""
    if fd.startswith("data:"):
        mime, data = decode_data_url(fd)
    else:
        mime, data = guess_mime(fname), base64.b64decode(fd)
    if mime in IMAGE_MIMES:
        item.images.append(ImageInput(fname, mime, data))
    elif is_textual(mime, fname):
        texts.append("\n\n" + fence(fname, data))
    else:
        item.files.append(FileInput(fname, mime, data))


def _attach_chat_file(
    item: HistoryItem, texts: list[str], part: dict[str, object]
) -> None:
    """Decode a chat file part into the attachment buffers."""
    file_raw = part.get("file")
    file_map = file_raw if isinstance(file_raw, dict) else {}
    fname_raw = file_map.get("filename") or "file"
    fname = fname_raw if isinstance(fname_raw, str) else "file"
    fd = _require_str(
        file_map.get("file_data"), "only inline file_data (data URL) is supported"
    )
    _attach_file_data(item, texts, fname, fd)


async def _apply_chat_part(item: HistoryItem, texts: list[str], part: object) -> None:
    """Fold one chat content part into text and attachment buffers."""
    if isinstance(part, str):
        texts.append(part)
        return
    checked = _require_dict(part, "message content parts must be strings or objects")
    part_type = checked.get("type")
    if part_type == "text":
        text_raw = checked.get("text", "")
        texts.append(text_raw if isinstance(text_raw, str) else "")
    elif part_type == "image_url":
        url_raw = checked.get("image_url") or {}
        url_candidate = url_raw.get("url") if isinstance(url_raw, dict) else url_raw
        url = _require_str(url_candidate, "image_url.url must be a string")
        item.images.append(await _chat_image_url(url))
    elif part_type == "file":
        _attach_chat_file(item, texts, checked)
    elif part_type == "input_audio":
        msg = "audio input is not supported"
        raise ValueError(msg)
    else:
        texts.append(extract_text(checked))


async def _parse_chat_user(content: object) -> HistoryItem:
    """Build a user turn from chat message content."""
    item = HistoryItem(role="user")
    if isinstance(content, str):
        item.text = content
    elif isinstance(content, list):
        texts: list[str] = []
        for part in content:
            await _apply_chat_part(item, texts, part)
        item.text = "".join(texts)
    else:
        item.text = extract_text(content)
    return item


async def _parse_chat_message(
    msg: dict[str, object], system_parts: list[str]
) -> HistoryItem | None:
    """Parse one chat message, appending system text and returning the turn."""
    role_raw = msg.get("role")
    role = role_raw if isinstance(role_raw, str) else ""
    content = msg.get("content")
    if role in ("system", "developer"):
        system_parts.append(extract_text(content))
        return None
    if role == "tool":
        msg_error = "tool role is not supported by this proxy"
        raise ValueError(msg_error)
    if role == "assistant":
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            msg_error = "assistant tool_calls are not supported by this proxy"
            raise ValueError(msg_error)
        return HistoryItem(role="assistant", text=extract_text(content))
    return await _parse_chat_user(content)


# ------------------------------------------------ chat completions parsing


async def parse_chat_request(body: dict[str, object]) -> ParsedRequest:
    """Parse an OpenAI Chat Completions body into internal turns."""
    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        msg = "messages must not be empty"
        raise ValueError(msg)
    messages = [
        _require_dict(candidate, "messages entries must be objects")
        for candidate in raw_messages
    ]
    system_parts: list[str] = []
    items: list[HistoryItem] = []
    for msg_data in messages:
        item = await _parse_chat_message(msg_data, system_parts)
        if item is not None:
            items.append(item)
    if not items:
        msg = "request must contain at least one non-system message"
        raise ValueError(msg)
    return ParsedRequest(
        system_text="\n\n".join(p for p in system_parts if p),
        items=items,
        model_requested=_optional_str(body.get("model")),
        stream=bool(body.get("stream")),
        include_sources=_optional_bool(body.get("include_sources")),
    )


# ---------------------------------------------------------------- responses parsing


async def _response_image(part: dict[str, object]) -> ImageInput:
    """Build an image input from a Responses input_image part."""
    url_raw = part.get("image_url")
    file_id = part.get("file_id")
    if file_id and not url_raw:
        msg = "referencing server file_ids is not supported"
        raise ValueError(msg)
    url = _require_str(url_raw, "input_image requires image_url")
    loaded = await _load_image_bytes(url)
    if loaded is None:
        msg = "input_image.image_url must be a data or http(s) URL"
        raise ValueError(msg)
    mime, data = loaded
    if url.startswith("data:"):
        return ImageInput("image", mime, data)
    name = url.rsplit("/", 1)[-1].split("?", maxsplit=1)[0] or "image"
    return ImageInput(name, mime, data)


def _attach_response_file(
    item: HistoryItem, texts: list[str], part: dict[str, object]
) -> None:
    """Decode a Responses input_file part into the attachment buffers."""
    fname_raw = part.get("filename") or "file"
    fname = fname_raw if isinstance(fname_raw, str) else "file"
    fd = _require_str(part.get("file_data"), "input_file requires file_data (data URL)")
    _attach_file_data(item, texts, fname, fd)


async def _apply_response_part(
    item: HistoryItem, texts: list[str], part: object
) -> None:
    """Fold one Responses content part into text and attachment buffers."""
    if isinstance(part, str):
        texts.append(part)
        return
    checked = _require_dict(part, "message content parts must be strings or objects")
    part_type = checked.get("type")
    if part_type in ("input_text", "output_text", "text", "summary_text"):
        text_raw = checked.get("text", "")
        texts.append(text_raw if isinstance(text_raw, str) else "")
    elif part_type == "input_image":
        item.images.append(await _response_image(checked))
    elif part_type == "input_file":
        _attach_response_file(item, texts, checked)
    else:
        texts.append(extract_text(checked))


async def _append_response_message(
    items: list[HistoryItem],
    system_text: str,
    role: object,
    content: object,
) -> str:
    """Append one Responses message; system roles extend system text."""
    if role in ("system", "developer"):
        return (system_text + "\n\n" + extract_text(content)).strip()
    item = HistoryItem(role="user" if role != "assistant" else "assistant")
    if isinstance(content, str):
        item.text = content
    elif isinstance(content, list):
        texts: list[str] = []
        for part in content:
            await _apply_response_part(item, texts, part)
        item.text = "".join(texts)
    items.append(item)
    return system_text


async def _apply_response_entry(
    items: list[HistoryItem], system_text: str, entry: object
) -> str:
    """Fold one Responses input entry into items, returning system text."""
    if isinstance(entry, str):
        items.append(HistoryItem(role="user", text=entry))
        return system_text
    checked = _require_dict(entry, "Responses input entries must be strings or objects")
    entry_type = checked.get("type", "message")
    if entry_type == "message":
        role = checked.get("role", "user")
        return await _append_response_message(
            items, system_text, role, checked.get("content")
        )
    msg = f"Responses input item type '{entry_type}' is not supported by this proxy"
    raise ValueError(msg)


async def parse_responses_request(
    body: dict[str, object],
) -> tuple[ParsedRequest, str | None, bool]:
    """Parse a Responses body, returning (parsed, previous id, store flag)."""
    instructions = body.get("instructions")
    system_text = instructions if isinstance(instructions, str) else ""
    prev_raw = body.get("previous_response_id")
    prev_id = prev_raw if isinstance(prev_raw, str) else None
    items: list[HistoryItem] = []
    raw_input = body.get("input")
    if isinstance(raw_input, str):
        if raw_input:
            items.append(HistoryItem(role="user", text=raw_input))
    elif isinstance(raw_input, list):
        for entry in raw_input:
            system_text = await _apply_response_entry(items, system_text, entry)
    if not items:
        msg = "input must not be empty"
        raise ValueError(msg)
    parsed = ParsedRequest(
        system_text=system_text,
        items=items,
        model_requested=_optional_str(body.get("model")),
        stream=bool(body.get("stream")),
        include_sources=_optional_bool(body.get("include_sources")),
    )
    return parsed, prev_id, bool(body.get("store", True))


# ---------------------------------------------------------------- id helpers


def gen_id(prefix: str) -> str:
    """Generate a prefixed hex id."""
    return prefix + uuid.uuid4().hex


def now_ts() -> int:
    """Return the current Unix timestamp in seconds."""
    return int(time.time())


def estimate_tokens(*texts: str) -> int:
    """Estimate token count for the given texts at ~4 chars per token."""
    return sum(len(t) for t in texts) // 4
