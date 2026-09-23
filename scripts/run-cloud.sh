#!/usr/bin/env bash
# run-cloud.sh — start the FastAPI runtime for cloud platforms.
#
# Reads PORT / MECHANISTIC_PORT (default 8010) and binds 0.0.0.0 by default.
# Railway, Render, Fly.io, and Cursor Cloud inject PORT automatically.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d .venv ]]; then
  echo "Missing .venv — run: bash scripts/setup-cloud.sh" >&2
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

HOST="${MECHANISTIC_HOST:-0.0.0.0}"
PORT="${PORT:-${MECHANISTIC_PORT:-8010}}"

echo "Starting Mechanistic runtime on ${HOST}:${PORT}"
echo "UI: http://${HOST}:${PORT}/"
echo "Data root: ${MECHANISTIC_DATA_DIR:-${WIGGUM_DATA_DIR:-$ROOT_DIR (in-repo)}}"

exec python main.py serve --host "$HOST" --port "$PORT"
