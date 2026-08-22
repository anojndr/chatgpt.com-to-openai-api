"""ChatGPT sentinel proof-of-work solver (sha3-512 hashcash)."""
import base64
import hashlib
import json
import random
import time
import uuid

_CORES = [8, 12, 16, 24]
_SCREENS = [3000, 4000, 6000]


def _parse_time() -> str:
    return time.strftime("%a %b %d %Y %H:%M:%S") + " GMT+0000 (Coordinated Universal Time)"


def solve(seed: str, difficulty: str, *, script: str | None = None, build_id: str | None = None,
          user_agent: str = "", max_iters: int = 500_000) -> str:
    """Return 'gAAAAAB' + base64(config json) whose sha3-512(seed+base) hex prefix <= difficulty."""
    ua = user_agent
    start_wall = time.time()
    cfg = [
        random.choice(_SCREENS),
        _parse_time(),
        4294705152,
        0,
        ua,
        script,
        build_id,
        "en-US",
        "en-US",
        0,
        "webkitGetUserMedia\u2212function webkitGetUserMedia() { [native code] }",
        "location",
        "ontransitionend",
        13.37,
        str(uuid.uuid4()),
        "",
        random.choice(_CORES),
        start_wall * 1000,
    ]
    d = len(difficulty)
    t0 = time.time()
    for i in range(max_iters):
        cfg[3] = i
        cfg[9] = round((time.time() - t0) * 1000)
        bs = base64.b64encode(json.dumps(cfg, separators=(",", ":")).encode()).decode()
        h = hashlib.sha3_512((seed + bs).encode()).hexdigest()
        if h[:d] <= difficulty:
            return "gAAAAAB" + bs
    raise RuntimeError(f"proof-of-work unsolved after {max_iters} iterations (difficulty={difficulty})")


def pre_proof(user_agent: str = "") -> str:
    """Proof used when calling the chat-requirements endpoint itself (difficulty '0')."""
    return "gAAAAAC" + solve(str(random.random()), "0", user_agent=user_agent)[7:]
