#!/usr/bin/env bash
# Start the Odin-ML FastAPI service: load .env, activate the venv, run uvicorn.
# Usage: ./start.sh [extra uvicorn args]   (HOST/PORT can be overridden via env)
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ ! -f .venv/bin/activate ]]; then
  echo "No .venv found. Create it with: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

exec uvicorn app.main:app --reload --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" "$@"
