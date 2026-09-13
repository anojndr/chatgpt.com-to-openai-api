#!/usr/bin/env bash
# Restart the chatgpt-to-openai-api server: kills any running instance, starts fresh.
set -euo pipefail
cd "$(dirname "$0")"

PORT=$(grep -E '^PORT=' .env 2>/dev/null | cut -d= -f2 | tr -d '"' || true)
PORT=${PORT:-4035}
LOG="$PWD/server.log"
BASE_URL="http://127.0.0.1:$PORT/v1"

if ! .venv/bin/python -c "import sys" >/dev/null 2>&1; then
  echo "creating venv..."
  rm -rf .venv
  if command -v uv >/dev/null 2>&1; then
    uv venv .venv
    uv pip install --python .venv/bin/python -r requirements.txt
  else
    python3 -m venv .venv
    .venv/bin/pip install -q -r requirements.txt
  fi
fi

echo "killing any running instance..."
pkill -f "python -m app.main" 2>/dev/null && sleep 1 || true

echo "starting server..."
# setsid detaches the server into its own process group so Ctrl-C on this
# script's tail never takes the API down.
setsid nohup .venv/bin/python -m app.main > "$LOG" 2>&1 < /dev/null &
disown 2>/dev/null || true

HEALTHY=0
for i in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then
    HEALTHY=1
    break
  fi
  sleep 0.5
done

if [ "$HEALTHY" -ne 1 ]; then
  echo "Error: Server failed to start. Last log lines:" >&2
  tail -n 20 "$LOG" >&2 || true
  exit 1
fi

echo
echo "Base URL (copy & paste):"
echo "  $BASE_URL"
echo
echo "Follow logs (copy & paste):"
echo "  tail -f $LOG"
