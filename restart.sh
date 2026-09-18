#!/usr/bin/env bash
# Restart the chatgpt-to-openai-api server: kills any running instance, starts fresh.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

PORT=$(grep -E '^CHATGPT_PORT=' .env 2>/dev/null | cut -d= -f2 | tr -d '"' || true)
PORT=${PORT:-$(grep -E '^PORT=' .env 2>/dev/null | cut -d= -f2 | tr -d '"' || true)}
PORT=${PORT:-4035}
LOG="$DIR/server.log"
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
# Scoped to $DIR: match `python -m app.main` processes whose cwd is this repo
# (a bare `pkill -f "python -m app.main"` would also match any other checkout
# running the same module). Port kill below is port-exact so it stays safe.
for _pid in $(pgrep -f "python -m app\.main" || true); do
  if [ "$(readlink "/proc/$_pid/cwd" 2>/dev/null || true)" = "$DIR" ]; then
    echo "Stopping server (pid $_pid)..."
    kill "$_pid" 2>/dev/null || true
  fi
done
sleep 1
# Free our own port if a manually-started instance squats it.
if command -v fuser >/dev/null 2>&1; then
  fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
  sleep 1
fi
# Wait for the port to free up (max ~10s) before starting.
for _ in $(seq 1 20); do
  if ! ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
    break
  fi
  sleep 0.5
done

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
