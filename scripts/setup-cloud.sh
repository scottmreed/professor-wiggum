#!/usr/bin/env bash
# setup-cloud.sh — bootstrap Mechanistic Agent for cloud VMs / containers.
#
# Usage (from repo root):
#   bash scripts/setup-cloud.sh
#   bash scripts/setup-cloud.sh --skip-tests
#   MECHANISTIC_DATA_DIR=/data/wiggum-data bash scripts/setup-cloud.sh
#
# After setup, configure env vars (see .env.cloud.example), then:
#   bash scripts/run-cloud.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SKIP_TESTS=0
SKIP_NODE=0
SKIP_RDKIT=0

for arg in "$@"; do
  case "$arg" in
    --skip-tests) SKIP_TESTS=1 ;;
    --skip-node) SKIP_NODE=1 ;;
    --skip-rdkit) SKIP_RDKIT=1 ;;
    -h|--help)
      cat <<'EOF'
Usage: bash scripts/setup-cloud.sh [--skip-tests] [--skip-node] [--skip-rdkit]

Installs Python deps, optional Node/rdkit-agent, chemistry extras, and
initializes the runtime SQLite DB under MECHANISTIC_DATA_DIR (or in-repo fallback).

Environment (optional before running):
  MECHANISTIC_DATA_DIR   Persistent bulk-data root (mount a volume here in prod)
  PYTHON                 Python executable (default: python3, then python)
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      exit 2
      ;;
  esac
done

pick_python() {
  if [[ -n "${PYTHON:-}" ]] && command -v "$PYTHON" >/dev/null 2>&1; then
    echo "$PYTHON"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo python3
    return
  fi
  if command -v python >/dev/null 2>&1; then
    echo python
    return
  fi
  echo "Python 3.10+ is required but python3/python was not found." >&2
  exit 1
}

PY="$(pick_python)"
PY_VERSION="$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PY_MAJOR="${PY_VERSION%%.*}"
PY_MINOR="${PY_VERSION#*.}"
if (( PY_MAJOR < 3 || (PY_MAJOR == 3 && PY_MINOR < 10) )); then
  echo "Python 3.10+ required, found $PY_VERSION ($PY)" >&2
  exit 1
fi

echo "==> Mechanistic cloud setup (Python $PY_VERSION via $PY)"
echo "    repo: $ROOT_DIR"

DATA_DIR="${MECHANISTIC_DATA_DIR:-${WIGGUM_DATA_DIR:-$ROOT_DIR}}"
export MECHANISTIC_DATA_DIR="$DATA_DIR"
mkdir -p "$DATA_DIR/data" "$DATA_DIR/traces/runs"

if [[ ! -d .venv ]]; then
  echo "==> Creating virtual environment"
  "$PY" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip"
python -m pip install --upgrade pip

echo "==> Installing Python dependencies"
python -m pip install -r requirements.txt
python -m pip install -e .

if [[ "$SKIP_RDKIT" -eq 0 ]]; then
  echo "==> Installing chemistry extras (RDKit + Dimorphite)"
  if ! python -m pip install -e ".[chemistry]"; then
    echo "WARNING: chemistry extras failed; validators may be unavailable." >&2
  fi
fi

if [[ "$SKIP_NODE" -eq 0 ]]; then
  if command -v npm >/dev/null 2>&1; then
    echo "==> Installing Node deps (rdkit-agent CLI backend)"
    npm ci 2>/dev/null || npm install
  else
    echo "WARNING: npm not found; set MECHANISTIC_CHEMISTRY_BACKEND=python or install Node.js." >&2
  fi
fi

echo "==> Initializing runtime SQLite (if missing)"
python - <<'PY'
from mechanistic_agent.core.db import RunStore
from mechanistic_agent.data_paths import db_path

path = db_path()
RunStore(path)
print(f"Runtime DB ready at {path}")
PY

echo "==> Verifying imports"
python -c "from mechanistic_agent.config import ReactionInputs; print('import ok')"

if [[ "$SKIP_TESTS" -eq 0 ]]; then
  echo "==> Running fast smoke tests (no API keys)"
  if command -v pytest >/dev/null 2>&1 || python -m pytest --version >/dev/null 2>&1; then
    python -m pytest tests/fast/ -q --tb=no -x 2>/dev/null || {
      echo "NOTE: some fast tests may fail without RDKit or bulk data; core import passed." >&2
    }
  else
    python -m pip install pytest
    python -m pytest tests/fast/test_prompt_cap_defaults.py -q
  fi
fi

cat <<EOF

Cloud setup complete.

Next steps:
  1. Copy env template:  cp .env.cloud.example .env   # or set vars in your platform UI
  2. Set at least one LLM API key (or configure agent-bridge — see docs/agent_bridge.md)
  3. Mount persistent storage at:  $DATA_DIR
  4. Start the server:   bash scripts/run-cloud.sh

Docs: docs/CLOUD_SETUP.md
EOF
