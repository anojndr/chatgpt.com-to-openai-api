#!/usr/bin/env bash
# Launch the ChatGPT -> OpenAI-compatible API on port 4035.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "creating venv..."
  uv venv .venv
  uv pip install --python .venv/bin/python -r requirements.txt
fi

exec .venv/bin/python -m app.main
