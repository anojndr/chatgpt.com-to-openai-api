#!/usr/bin/env bash
# Restart the chatgpt-to-openai-api server: kills any running instance, starts fresh.
set -euo pipefail
cd "$(dirname "$0")"

PORT=$(grep -E '^PORT=' .env 2>/dev/null | cut -d= -f2 | tr -d '"' || true)
PORT=${PORT:-4035}
LOG="$PWD/server.log"
BASE_URL="http://127.0.0.1:$PORT/v1"

if [ ! -x .venv/bin/python ]; then
  echo "creating venv..."
  uv venv .venv
  uv pip install --python .venv/bin/python -r requirements.txt
fi

echo "killing any running instance..."
pkill -f "python -m app.main" 2>/dev/null && sleep 1 || true

echo "starting server..."
# setsid detaches the server into its own process group so Ctrl-C on this
# script's tail never takes the API down.
setsid nohup .venv/bin/python -m app.main > "$LOG" 2>&1 < /dev/null &
disown 2>/dev/null || true

for i in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

echo
echo "Base URL (copy & paste):"
echo "  $BASE_URL"
echo
echo "tailing $LOG  (Ctrl-C to stop watching; server keeps running)"
tail -f "$LOG"
