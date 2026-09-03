# Copyright 2026 chatgpt-to-openai-api contributors.
"""Application configuration loaded from the environment and `.env`.

Nothing sensitive is hardcoded here; every value below may be overridden
with an environment variable of the same name.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")

PIXELVAULT_API_KEY: str = os.environ.get("PIXELVAULT_API_KEY", "")
PIXELVAULT_BASE_URL: str = os.environ.get(
    "PIXELVAULT_BASE_URL",
    "https://api.pixelvault.dev",
)

PORT: int = int(os.environ.get("PORT", "4035"))
# Bind to loopback by default so a fresh checkout never exposes the proxy
# to the local network. Set HOST=0.0.0.0 explicitly to serve LAN clients;
# the environment always wins over this default.
HOST: str = os.environ.get("HOST", "127.0.0.1")

ACCOUNTS_FILE: Path = ROOT / os.environ.get("ACCOUNTS_FILE", "accounts.txt")
DB_PATH: Path = ROOT / os.environ.get("DB_PATH", "data/conversations.db")
API_KEY: str = os.environ.get("API_KEY", "")  # optional bearer gate for THIS proxy
INCLUDE_SOURCES: bool = os.environ.get(
    "CHATGPT_INCLUDE_SOURCES",
    os.environ.get("INCLUDE_SOURCES", "0"),
).strip().lower() in ("1", "true", "yes", "on")

DEFAULT_MODEL: str = os.environ.get("DEFAULT_MODEL", "auto")
COOLDOWN_FREE_SECONDS: int = int(os.environ.get("COOLDOWN_FREE_SECONDS", "900"))
COOLDOWN_PLUS_SECONDS: int = int(os.environ.get("COOLDOWN_PLUS_SECONDS", "120"))
REQUIREMENTS_TTL: int = int(
    os.environ.get("REQUIREMENTS_TTL", "480"),
)  # server says 540
CONVERSATION_TTL_HOURS: float = float(os.environ.get("CONVERSATION_TTL_HOURS", "168"))
# Context snapshots retained per response so account failover can replay the
# full conversation (text + binaries), capped by SNAPSHOT_FILE_CAP_MB and
# SNAPSHOT_STORE_CAP_MB.
SNAPSHOT_FILE_CAP_MB: int = int(os.environ.get("SNAPSHOT_FILE_CAP_MB", "32"))
SNAPSHOT_STORE_CAP_MB: int = int(os.environ.get("SNAPSHOT_STORE_CAP_MB", "256"))

# --- session keepalive: keep accounts.txt sessions alive indefinitely ---
# Sweep interval: every account is checked once per tick.
KEEPALIVE_SECONDS: int = int(os.environ.get("KEEPALIVE_SECONDS", "600"))
# Refresh the access token when less than this much validity remains, so the
# JWT never lapses and the rolling session cookie keeps being exercised.
KEEPALIVE_REFRESH_WITHIN: int = int(os.environ.get("KEEPALIVE_REFRESH_WITHIN", "86400"))
# A 200 from /api/auth/session without a usable token counts one strike;
# this many spaced strikes pull the account from rotation as needing re-login.
KEEPALIVE_MAX_STRIKES: int = int(os.environ.get("KEEPALIVE_MAX_STRIKES", "3"))
# Dead accounts are re-probed this often and revived automatically if the
# stored cookie jar produces a fresh token again.
KEEPALIVE_REVIVE_SECONDS: int = int(os.environ.get("KEEPALIVE_REVIVE_SECONDS", "900"))
# A pasted block only hot-swaps if its JWT outlives the in-memory one by this
# much (prevents churn from borderline exports).
KEEPALIVE_MIN_IMPROVEMENT: float = float(
    os.environ.get("KEEPALIVE_MIN_IMPROVEMENT", "300"),
)

USER_AGENT: str = os.environ.get(
    "UA",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
)
IMPERSONATE: str = os.environ.get("TLS_IMPERSONATE", "chrome")
