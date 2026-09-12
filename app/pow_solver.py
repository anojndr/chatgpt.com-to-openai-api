# Copyright 2026 chatgpt-to-openai-api contributors.
"""ChatGPT sentinel proof-of-work solver (FNV-1a hashcash)."""

from __future__ import annotations

import base64
import json
import secrets
import time
import uuid
from dataclasses import dataclass

_SCREENS = [3000, 4000, 6000]
_CORES = [8, 12, 16, 24]
_PROOF_PREFIX = "gAAAAAB"
_PROOF_SUFFIX = "~S"
_REQUIREMENTS_DIFFICULTY = "0"


@dataclass(frozen=True)
class PowOptions:
    """Tunable inputs for the sentinel proof-of-work solver."""

    script: str | None = None
    build_id: str | None = None
    user_agent: str = ""
    max_iters: int = 500_000


def _parse_time() -> str:
    return (
        time.strftime("%a %b %d %Y %H:%M:%S") + " GMT+0000 (Coordinated Universal Time)"
    )


def fnv_hash(value: str) -> str:
    """FNV-1a 32-bit hash matching the chatgpt.com frontend (pFt).

    Returns:
        The 8-character lowercase hex digest of the value.

    """
    h = 2166136261
    for ch in value:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 2246822507) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 3266489909) & 0xFFFFFFFF
    h ^= h >> 16
    return format(h & 0xFFFFFFFF, "08x")


def solve(
    seed: str,
    difficulty: str,
    options: PowOptions | None = None,
) -> str:
    """Solve the sentinel proof of work, returning the proof string.

    The proof is 'gAAAAAB' plus base64(config json) plus '~S' whose
    FNV-1a(seed + base) hex prefix does not exceed difficulty.

    Returns:
        The proof string to send with the sentinel request.

    Raises:
        RuntimeError: If no proof is found within max_iters iterations.

    """
    opts = options if options is not None else PowOptions()
    start_wall = time.time()
    cfg: list[object] = [
        secrets.choice(_SCREENS),
        _parse_time(),
        4294705152,
        0,
        opts.user_agent,
        opts.script,
        opts.build_id,
        "en-US",
        "en-US",
        0,
        "webkitGetUserMedia\u2212function webkitGetUserMedia() { [native code] }",
        "location",
        "ontransitionend",
        13.37,
        str(uuid.uuid4()),
        "",
        secrets.choice(_CORES),
        start_wall * 1000,
    ]
    width = len(difficulty)
    started = time.time()
    for i in range(opts.max_iters):
        cfg[3] = i
        cfg[9] = round((time.time() - started) * 1000)
        encoded = base64.b64encode(json.dumps(cfg, separators=(",", ":")).encode())
        proof_body = encoded.decode()
        digest = fnv_hash(seed + proof_body)
        if digest[:width] <= difficulty:
            return _PROOF_PREFIX + proof_body + _PROOF_SUFFIX
    msg = (
        f"proof-of-work unsolved after {opts.max_iters} "
        f"iterations (difficulty={difficulty})"
    )
    raise RuntimeError(msg)


def pre_proof(user_agent: str = "") -> str:
    """Build the proof for the chat-requirements call itself.

    The requirements endpoint always uses difficulty '0'.

    Returns:
        The proof string for the chat-requirements call.

    """
    seed = str(secrets.SystemRandom().random())
    options = PowOptions(user_agent=user_agent)
    return "gAAAAAC" + solve(seed, _REQUIREMENTS_DIFFICULTY, options)[7:]
