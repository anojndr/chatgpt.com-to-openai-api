"""Configuration loaded from .env next to the project root. Nothing sensitive is hardcoded."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")

PIXELVAULT_API_KEY: str = os.environ.get("PIXELVAULT_API_KEY", "")
PIXELVAULT_BASE_URL: str = os.environ.get("PIXELVAULT_BASE_URL", "https://api.pixelvault.dev")

PORT: int = int(os.environ.get("PORT", "4035"))
HOST: str = os.environ.get("HOST", "0.0.0.0")

ACCOUNTS_FILE: Path = ROOT / os.environ.get("ACCOUNTS_FILE", "accounts.txt")
DB_PATH: Path = ROOT / os.environ.get("DB_PATH", "data/conversations.db")
API_KEY: str = os.environ.get("API_KEY", "")  # optional bearer gate for THIS proxy
INCLUDE_SOURCES: bool = os.environ.get("CHATGPT_INCLUDE_SOURCES", os.environ.get("INCLUDE_SOURCES", "0")).strip().lower() in ("1", "true", "yes", "on")

DEFAULT_MODEL: str = os.environ.get("DEFAULT_MODEL", "auto")
COOLDOWN_FREE_SECONDS: int = int(os.environ.get("COOLDOWN_FREE_SECONDS", "900"))
COOLDOWN_PLUS_SECONDS: int = int(os.environ.get("COOLDOWN_PLUS_SECONDS", "120"))
REQUIREMENTS_TTL: int = int(os.environ.get("REQUIREMENTS_TTL", "480"))  # server says 540
CONVERSATION_TTL_HOURS: float = float(os.environ.get("CONVERSATION_TTL_HOURS", "168"))
# Context snapshots retained per response so account failover can replay the
# full conversation (text + binaries) on another account. Per-response cap
# bounds one snapshot's binaries; the store-wide budget evicts oldest first.
SNAPSHOT_FILE_CAP_MB: int = int(os.environ.get("SNAPSHOT_FILE_CAP_MB", "32"))
SNAPSHOT_STORE_CAP_MB: int = int(os.environ.get("SNAPSHOT_STORE_CAP_MB", "256"))

USER_AGENT: str = os.environ.get(
    "UA",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
)
IMPERSONATE: str = os.environ.get("TLS_IMPERSONATE", "chrome")
