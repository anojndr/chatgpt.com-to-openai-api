#!/usr/bin/env bash
# Launch the ChatGPT -> OpenAI-compatible API on port 4035.
set -euo pipefail
cd "$(dirname "$0")"

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

exec .venv/bin/python -m app.main
