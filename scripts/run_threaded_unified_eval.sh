#!/usr/bin/env bash
# threaded_unified harness on the hard10 eval set (products hidden) through the keyless bridge,
# answered by the committed headless responder (scripts/bridge_headless_responder.py).
#
#   scripts/run_threaded_unified_eval.sh <trial_dir> [case_id ...]
#
# No case ids = all ten hard10 cases. Environment overrides:
#   EFFORT      thinking level for the run record and the responder (default: low)
#   RUN_GROUP   eval run group (default: threaded_unified_hard10_noprod_<effort>)
#   EVAL_SET_ID (default: 4d5405b940884e469c01a9fdef161b80, hard10)
#   PYTHON      interpreter (default: .venv/bin/python)
#
# The trial dir holds bridge/ (requests, responses) and the eval and responder logs; keep it
# outside the repo. Compare afterwards with scripts/compare_harness_runs.py.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <trial_dir> [case_id ...]" >&2
  exit 2
fi
TRIAL_DIR="$(mkdir -p "$1" && cd "$1" && pwd)"
shift

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
PYTHON="${PYTHON:-.venv/bin/python}"
EFFORT="${EFFORT:-low}"
RUN_GROUP="${RUN_GROUP:-threaded_unified_hard10_noprod_${EFFORT}}"
EVAL_SET_ID="${EVAL_SET_ID:-4d5405b940884e469c01a9fdef161b80}"

CASE_ARGS=()
for case_id in "$@"; do
  CASE_ARGS+=(--case-id "$case_id")
done

export MECHANISTIC_AGENT_BRIDGE_DIR="$TRIAL_DIR/bridge"
export MECHANISTIC_ACTIVE_MODEL=agent-bridge
export MECHANISTIC_AGENT_BRIDGE_TIMEOUT=3600
export MECHANISTIC_AGENT_BRIDGE_DECLARED_MODEL=claude-opus-5-5
export MECHANISTIC_AGENT_BRIDGE_RESPONDER_KIND=headless_claude_cli
export MECHANISTIC_AGENT_BRIDGE_SAW_GROUND_TRUTH=false
export MECHANISTIC_AGENT_BRIDGE_NOTES="threaded_unified hard10 noprod: headless claude -p --effort ${EFFORT} (scripts/bridge_headless_responder.py: --tools '' --restricted --strict-mcp-config --no-session-persistence, empty tmp cwd); sees only model_input"
export RESPONDER_MODEL=claude-opus-5-5
export RESPONDER_EFFORT="$EFFORT"
mkdir -p "$MECHANISTIC_AGENT_BRIDGE_DIR"

# claude -p must not see an API key, or it ignores the claude.ai login.
env -u ANTHROPIC_API_KEY -u OPENAI_API_KEY -u OPENROUTER_API_KEY \
  "$PYTHON" main.py bridge-serve --bridge-dir "$MECHANISTIC_AGENT_BRIDGE_DIR" \
  --command "$PYTHON scripts/bridge_headless_responder.py" \
  > "$TRIAL_DIR/responder.log" 2>&1 &
RESPONDER_PID=$!
trap 'kill "$RESPONDER_PID" 2>/dev/null || true' EXIT

OPENAI_API_KEY=blocked ANTHROPIC_API_KEY=blocked OPENROUTER_API_KEY=blocked \
  "$PYTHON" main.py eval --eval-set-id "$EVAL_SET_ID" "${CASE_ARGS[@]}" \
  --harness threaded_unified \
  --model-name anthropic/claude-opus-5.5 --thinking-level "$EFFORT" \
  --hide-products --max-steps 14 --max-runtime 10800 \
  --run-group "$RUN_GROUP" \
  2>&1 | tee "$TRIAL_DIR/eval.log"
