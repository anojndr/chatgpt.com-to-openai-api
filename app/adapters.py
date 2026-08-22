"""Pure translation between OpenAI wire formats and internal turn structures."""
from __future__ import annotations

import base64
import json
import mimetypes
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------- models

LIVE_SLUG_ALIASES = {
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


def map_model(requested: str | None, live_slugs: set[str]) -> str:
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
    from . import config as cfg
    d = cfg.DEFAULT_MODEL
    if d in live_slugs:
        return d
    return "auto"


def public_models(models_payload: list[dict]) -> list[dict]:
    """OpenAI-style model list derived from live ChatGPT models + auto."""
    seen = []
    for m in models_payload:
        slug = m.get("slug")
        if slug and slug not in seen:
            seen.append(slug)
    ids = sorted(set(seen) | {"auto"})
    out = []
    for i in ids:
        out.append({
            "id": i,
            "object": "model",
            "created": 1785000000,
            "owned_by": "chatgpt-proxy",
        })
    return out


# ---------------------------------------------------------------- history items

@dataclass
class ImageInput:
    filename: str
    mime: str
    data: bytes


@dataclass
class FileInput:
    filename: str
    mime: str
    data: bytes


@dataclass
class HistoryItem:
    role: str  # user | assistant
    text: str = ""
    images: list[ImageInput] = field(default_factory=list)
    files: list[FileInput] = field(default_factory=list)

    def canon(self) -> str:
        extras = []
        for im in self.images:
            extras.append(f"img:{im.filename}:{len(im.data)}")
        for f in self.files:
            extras.append(f"file:{f.filename}:{len(f.data)}")
        if extras:
            return self.text + "\x00" + "\x00".join(extras)
        return self.text


@dataclass
class ParsedRequest:
    system_text: str
    items: list[HistoryItem]
    model_requested: str | None
    stream: bool


# ---------------------------------------------------------------- content helpers

_DATA_URL_RE = re.compile(r"^data:([^;,]+)?(;base64)?,(.*)$", re.S)

TEXTUAL_MIME_PREFIXES = ("text/",)
TEXTUAL_MIMES = {
    "application/json", "application/xml", "application/javascript", "application/x-yaml",
    "application/yaml", "application/toml", "application/x-sh", "application/sql",
    "application/graphql", "application/x-python", "application/csv",
}
CODE_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".yml", ".yaml", ".toml", ".ini", ".cfg",
    ".sh", ".bash", ".zsh", ".sql", ".html", ".css", ".scss", ".rs", ".go", ".java", ".kt",
    ".c", ".h", ".cpp", ".hpp", ".cs", ".rb", ".php", ".swift", ".m", ".mm", ".lua", ".pl",
    ".r", ".jl", ".md", ".txt", ".csv", ".tsv", ".xml", ".svg", ".diff", ".patch", ".log",
    ".dockerfile", ".makefile", ".env", ".gitignore", ".proto", ".tf", ".hcl", ".vue",
    ".svelte", ".astro", ".dart", ".ex", ".exs", ".erl", ".hs", ".clj", ".scala", ".groovy",
}
IMAGE_MIMES = {"image/png", "image/jpeg", "image/webp", "image/gif"}
EXT_TO_LANG = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "tsx", ".jsx": "jsx",
    ".json": "json", ".yml": "yaml", ".yaml": "yaml", ".toml": "toml", ".sh": "bash",
    ".bash": "bash", ".sql": "sql", ".html": "html", ".css": "css", ".rs": "rust",
    ".go": "go", ".java": "java", ".kt": "kotlin", ".c": "c", ".cpp": "cpp", ".cs": "csharp",
    ".rb": "ruby", ".php": "php", ".swift": "swift", ".md": "markdown", ".xml": "xml",
    ".svg": "xml", ".csv": "csv", ".ini": "ini", ".lua": "lua", ".r": "r", ".dart": "dart",
}


def decode_data_url(value: str) -> tuple[str, bytes]:
    m = _DATA_URL_RE.match(value.strip())
    if not m:
        raise ValueError("expected a data: URL")
    mime = (m.group(1) or "application/octet-stream").lower()
    payload = m.group(3)
    if m.group(2):
        data = base64.b64decode(payload)
    else:
        import urllib.parse
        data = urllib.parse.unquote_to_bytes(payload)
    return mime, data


async def fetch_remote(url: str) -> tuple[str, bytes]:
    from curl_cffi.requests import AsyncSession
    import ipaddress
    import socket as _socket
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("only http(s) image URLs are supported")
    host = parsed.hostname or ""
    # SSRF guard: refuse loopback/private/link-local/metadata targets.
    bad_ip = False
    try:
        infos = _socket.getaddrinfo(host, None)
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                bad_ip = True
                break
    except _socket.gaierror as e:
        raise ValueError(f"cannot resolve image host: {host}") from e
    if bad_ip or host in ("localhost", "metadata.google.internal"):
        raise ValueError("refusing to fetch image from private address")

    s = AsyncSession()
    try:
        r = await s.get(url, timeout=60)
        if r.status_code != 200:
            raise ValueError(f"failed to fetch {url}: HTTP {r.status_code}")
        ct = (r.headers.get("content-type") or "application/octet-stream").split(";")[0].strip().lower()
        data = r.content
        if len(data) > 25 * 1024 * 1024:
            raise ValueError("image larger than 25 MB")
        return ct, data
    finally:
        await s.close()


def is_textual(mime: str, filename: str) -> bool:
    if mime in TEXTUAL_MIMES or mime.startswith(TEXTUAL_MIME_PREFIXES):
        return True
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in CODE_EXTS


def guess_mime(filename: str, fallback: str = "application/octet-stream") -> str:
    mt, _ = mimetypes.guess_type(filename)
    return mt or fallback


def fence(filename: str, data: bytes) -> str:
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    lang = EXT_TO_LANG.get(ext, "")
    try:
        body = data.decode("utf-8")
    except UnicodeDecodeError:
        body = base64.b64encode(data).decode()
        lang = ""
    return f"{filename}\n```{lang}\n{body}\n```"


def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                t = p.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


# ---------------------------------------------------------------- chat completions parsing

async def parse_chat_request(body: dict) -> ParsedRequest:
    messages = body.get("messages") or []
    if not messages:
        raise ValueError("messages must not be empty")
    system_parts: list[str] = []
    items: list[HistoryItem] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role in ("system", "developer"):
            system_parts.append(extract_text(content))
            continue
        if role == "tool":
            raise ValueError("tool role is not supported by this proxy")
        if role == "assistant":
            tc = msg.get("tool_calls")
            if tc:
                raise ValueError("assistant tool_calls are not supported by this proxy")
            items.append(HistoryItem(role="assistant", text=extract_text(content)))
            continue
        # user
        item = HistoryItem(role="user")
        if isinstance(content, str):
            item.text = content
        elif isinstance(content, list):
            texts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    texts.append(part)
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    texts.append(part.get("text", ""))
                elif ptype == "image_url":
                    url = (part.get("image_url") or {})
                    url = url.get("url") if isinstance(url, dict) else url
                    if not isinstance(url, str):
                        raise ValueError("image_url.url must be a string")
                    if url.startswith("data:"):
                        mime, data = decode_data_url(url)
                        name = "image." + (mime.split("/")[1] if "/" in mime else "bin")
                        item.images.append(ImageInput(name, mime, data))
                    elif url.startswith("http://") or url.startswith("https://"):
                        mime, data = await fetch_remote(url)
                        name = url.rsplit("/", 1)[-1].split("?")[0] or "image"
                        item.images.append(ImageInput(name, mime, data))
                    else:
                        raise ValueError("image_url must be a data or http(s) URL")
                elif ptype == "file":
                    f = part.get("file") or {}
                    fd = f.get("file_data")
                    fname = f.get("filename") or "file"
                    if not isinstance(fd, str):
                        raise ValueError("only inline file_data (data URL) is supported")
                    mime, data = decode_data_url(fd) if fd.startswith("data:") else (guess_mime(fname), base64.b64decode(fd))
                    if mime in IMAGE_MIMES:
                        item.images.append(ImageInput(fname, mime, data))
                    elif is_textual(mime, fname):
                        texts.append("\n\n" + fence(fname, data))
                    else:
                        item.files.append(FileInput(fname, mime, data))
                elif ptype == "input_audio":
                    raise ValueError("audio input is not supported")
                else:
                    texts.append(extract_text(part))
            item.text = "".join(texts)
        else:
            item.text = extract_text(content)
        items.append(item)
    if not items:
        raise ValueError("request must contain at least one non-system message")
    return ParsedRequest(
        system_text="\n\n".join(p for p in system_parts if p),
        items=items,
        model_requested=body.get("model"),
        stream=bool(body.get("stream")),
    )


# ---------------------------------------------------------------- responses parsing

async def parse_responses_request(body: dict) -> tuple[ParsedRequest, str | None, bool]:
    """Returns (parsed, previous_response_id, store_flag)."""
    instructions = body.get("instructions")
    system_text = instructions if isinstance(instructions, str) else ""
    prev_id = body.get("previous_response_id")
    raw_input = body.get("input")
    items: list[HistoryItem] = []

    async def to_image(filename: str, url: str) -> ImageInput:
        if url.startswith("data:"):
            mime, data = decode_data_url(url)
            return ImageInput(filename or "image", mime, data)
        if url.startswith("http://") or url.startswith("https://"):
            mime, data = await fetch_remote(url)
            return ImageInput(url.rsplit("/", 1)[-1].split("?")[0] or "image", mime, data)
        raise ValueError("input_image.image_url must be a data or http(s) URL")

    async def add_message(role: str, content: Any) -> None:
        nonlocal system_text
        if role in ("system", "developer"):
            t = extract_text(content)
            system_text = (system_text + "\n\n" + t).strip()
            return
        item = HistoryItem(role="user" if role != "assistant" else "assistant")
        if isinstance(content, str):
            item.text = content
        elif isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, str):
                    texts.append(part)
                    continue
                pt = part.get("type")
                if pt in ("input_text", "output_text", "text", "summary_text"):
                    texts.append(part.get("text", ""))
                elif pt == "input_image":
                    url = part.get("image_url")
                    fid = part.get("file_id")
                    if fid and not url:
                        raise ValueError("referencing server file_ids is not supported")
                    if not isinstance(url, str):
                        raise ValueError("input_image requires image_url")
                    item.images.append(await to_image("image", url))
                elif pt == "input_file":
                    fd = part.get("file_data")
                    fname = part.get("filename") or "file"
                    if not isinstance(fd, str):
                        raise ValueError("input_file requires file_data (data URL)")
                    mime, data = decode_data_url(fd) if fd.startswith("data:") else (guess_mime(fname), base64.b64decode(fd))
                    if mime in IMAGE_MIMES:
                        item.images.append(ImageInput(fname, mime, data))
                    elif is_textual(mime, fname):
                        texts.append("\n\n" + fence(fname, data))
                    else:
                        item.files.append(FileInput(fname, mime, data))
                else:
                    texts.append(extract_text(part))
            item.text = "".join(texts)
        items.append(item)

    if isinstance(raw_input, str):
        if raw_input:
            items.append(HistoryItem(role="user", text=raw_input))
    elif isinstance(raw_input, list):
        for entry in raw_input:
            if isinstance(entry, str):
                items.append(HistoryItem(role="user", text=entry))
                continue
            etype = entry.get("type", "message")
            if etype == "message":
                await add_message(entry.get("role", "user"), entry.get("content"))
            else:
                raise ValueError(f"Responses input item type '{etype}' is not supported by this proxy")
    if not items:
        raise ValueError("input must not be empty")
    parsed = ParsedRequest(
        system_text=system_text,
        items=items,
        model_requested=body.get("model"),
        stream=bool(body.get("stream")),
    )
    return parsed, prev_id, body.get("store", True)


# ---------------------------------------------------------------- id helpers

def gen_id(prefix: str) -> str:
    return prefix + uuid.uuid4().hex


def now_ts() -> int:
    return int(time.time())


def estimate_tokens(*texts: str) -> int:
    return sum(len(t) for t in texts) // 4
