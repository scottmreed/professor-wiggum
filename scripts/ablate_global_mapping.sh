#!/usr/bin/env bash
# Global atom-mapping ablation: `default` vs `no_mapping` harness.
#
# PRD_jev_atom_identity_mechanistic.md section 8.2 asks whether the global
# atom_mapping LLM call matters before anyone invests in deterministic mapping
# candidates. This script runs both arms on the SAME case slice and prints a
# side-by-side readout. See docs/ablations/global_mapping.md for how to read it.
#
# Paid model calls: `run` spends real API budget (2 x MAX_CASES harness runs).
# Nothing is executed unless you pass the `run` subcommand explicitly.
#
# Usage:
#   scripts/ablate_global_mapping.sh plan     # print the exact commands (default)
#   scripts/ablate_global_mapping.sh run      # execute both eval arms (PAID)
#   scripts/ablate_global_mapping.sh report   # read both run groups from the DB
#
# Environment overrides:
#   MODEL=<model id>        default: the CLI default model (omit --model)
#   THINKING=<low|high|max> default: unset (omit --thinking-level)
#   TIER=easy               eval tier; medium/hard once populated
#   MAX_CASES=25            cases per arm (first N of the tier, same for both arms)
#   MAX_STEPS=10            mechanism steps per case
#   MAX_RUNTIME=1200        per-case timeout, seconds
#   DB=data/mechanistic.db  results DB read by `report`
#   LOG_DIR=local_contributions/runs/ablation_global_mapping   (gitignored)
set -euo pipefail

cd "$(dirname "$0")/.."

MODE="${1:-plan}"
TIER="${TIER:-easy}"
MAX_CASES="${MAX_CASES:-25}"
MAX_STEPS="${MAX_STEPS:-10}"
MAX_RUNTIME="${MAX_RUNTIME:-1200}"
DB="${DB:-data/mechanistic.db}"
LOG_DIR="${LOG_DIR:-local_contributions/runs/ablation_global_mapping}"

GROUP_ON="${TIER}_ablation_mapping_default"
GROUP_OFF="${TIER}_ablation_mapping_off"

MODEL_ARGS=()
if [[ -n "${MODEL:-}" ]]; then MODEL_ARGS+=(--model "$MODEL"); fi
if [[ -n "${THINKING:-}" ]]; then MODEL_ARGS+=(--thinking-level "$THINKING"); fi

# --leaderboard-route custom + --allow-repeats => both arms take the first
# MAX_CASES case ids of the tier, in tier order, regardless of what this model
# has already run. That makes the two arms a paired comparison.
COMMON_ARGS=(
  --eval-set-id ignored
  --tier "$TIER"
  --leaderboard-route custom
  --allow-repeats
  --max-cases "$MAX_CASES"
  --max-steps "$MAX_STEPS"
  --max-runtime "$MAX_RUNTIME"
)

arm_cmd() {
  local harness="$1" group="$2"
  printf '%q ' python main.py eval "${COMMON_ARGS[@]}" "${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"}" \
    --harness "$harness" --run-group "$group"
}

plan() {
  cat <<EOF
# Global atom-mapping ablation (tier=$TIER, max_cases=$MAX_CASES)
source .venv/bin/activate
mkdir -p $LOG_DIR

# 0. Preview the tier slice (no model calls)
python main.py eval --eval-set-id ignored --tier $TIER ${MODEL_ARGS[*]+"${MODEL_ARGS[*]}"} --leaderboard-status-only

# 1. Arm A: mapping ON (reference)
$(arm_cmd default "$GROUP_ON")| tee $LOG_DIR/$GROUP_ON.log

# 2. Arm B: mapping OFF (atom_mapping + step_atom_mapping disabled)
$(arm_cmd no_mapping "$GROUP_OFF")| tee $LOG_DIR/$GROUP_OFF.log

# 3. Read the result
scripts/ablate_global_mapping.sh report
#    eval_run ids per group:
sqlite3 $DB "select id, run_group_name, eval_set_id, status from eval_runs where run_group_name in ('$GROUP_ON','$GROUP_OFF') order by created_at;"
#    reproducibility parity (same case_ids_hash, model, prompt hashes):
python main.py compare-eval-runs --run-a <eval_run_id_A> --run-b <eval_run_id_B>
#    leaderboard rows for the tier eval set:
python main.py leaderboard --eval-set-id <eval_set_id> --limit 20
EOF
}

run() {
  mkdir -p "$LOG_DIR"
  echo ">>> PAID: running 2 x $MAX_CASES harness eval cases on tier '$TIER'." >&2
  eval "$(arm_cmd default "$GROUP_ON")" 2>&1 | tee "$LOG_DIR/$GROUP_ON.log"
  eval "$(arm_cmd no_mapping "$GROUP_OFF")" 2>&1 | tee "$LOG_DIR/$GROUP_OFF.log"
  report
}

report() {
  if [[ ! -f "$DB" ]]; then
    echo "DB not found: $DB" >&2
    exit 1
  fi
  echo "== Per-arm metrics (latest eval_run per group) =="
  sqlite3 -header -column "$DB" <<SQL
WITH latest AS (
  SELECT run_group_name, id, eval_set_id, harness_bundle_hash,
         ROW_NUMBER() OVER (PARTITION BY run_group_name ORDER BY created_at DESC) AS rn
  FROM eval_runs
  WHERE run_group_name IN ('$GROUP_ON', '$GROUP_OFF')
),
res AS (
  SELECT l.run_group_name AS grp, r.run_id, r.score, r.pass_bool, r.latency_ms,
         CAST(json_extract(r.cost_json, '$.total_cost') AS REAL) AS cost,
         json_extract(r.summary_json, '$.run_status') AS run_status,
         json_extract(r.summary_json, '$.mapping_agreement') AS mapping_agreement
  FROM latest l JOIN eval_run_results r ON r.eval_run_id = l.id
  WHERE l.rn = 1
),
ev AS (
  SELECT e.run_id,
         SUM(e.event_type = 'mechanism_retry_started') AS retries,
         SUM(e.event_type = 'backtrack') AS backtracks
  FROM run_events e WHERE e.run_id IN (SELECT run_id FROM res)
  GROUP BY e.run_id
),
so AS (
  SELECT s.run_id,
         SUM(s.step_name = 'atom_mapping') AS global_mapping_calls,
         SUM(s.step_name = 'step_atom_mapping') AS step_mapping_calls,
         SUM(s.step_name = 'mechanism_synthesis' AND s.accepted_bool = 1) AS accepted_steps
  FROM step_outputs s WHERE s.run_id IN (SELECT run_id FROM res)
  GROUP BY s.run_id
)
SELECT res.grp                                         AS run_group,
       COUNT(*)                                        AS cases,
       SUM(res.run_status = 'completed')               AS completed,
       ROUND(AVG(res.pass_bool), 3)                    AS pass_rate,
       ROUND(AVG(res.score), 4)                        AS mean_score,
       SUM(COALESCE(ev.retries, 0))                    AS retries,
       SUM(COALESCE(ev.backtracks, 0))                 AS backtracks,
       SUM(COALESCE(so.accepted_steps, 0))             AS accepted_steps,
       ROUND(SUM(COALESCE(res.cost, 0)), 4)            AS total_cost,
       ROUND(AVG(res.latency_ms) / 1000.0, 1)          AS mean_latency_s,
       SUM(COALESCE(so.global_mapping_calls, 0))       AS atom_mapping_calls,
       SUM(COALESCE(so.step_mapping_calls, 0))         AS step_mapping_calls,
       ROUND(AVG(res.mapping_agreement), 3)            AS mapping_agreement
FROM res
LEFT JOIN ev ON ev.run_id = res.run_id
LEFT JOIN so ON so.run_id = res.run_id
GROUP BY res.grp
ORDER BY res.grp;
SQL
  echo
  echo "== Paired per-case deltas (OFF minus ON; only cases present in both arms) =="
  sqlite3 -header -column "$DB" <<SQL
WITH latest AS (
  SELECT run_group_name, id,
         ROW_NUMBER() OVER (PARTITION BY run_group_name ORDER BY created_at DESC) AS rn
  FROM eval_runs WHERE run_group_name IN ('$GROUP_ON', '$GROUP_OFF')
),
a AS (SELECT r.case_id, r.score, r.pass_bool FROM latest l JOIN eval_run_results r ON r.eval_run_id = l.id
      WHERE l.rn = 1 AND l.run_group_name = '$GROUP_ON'),
b AS (SELECT r.case_id, r.score, r.pass_bool FROM latest l JOIN eval_run_results r ON r.eval_run_id = l.id
      WHERE l.rn = 1 AND l.run_group_name = '$GROUP_OFF')
SELECT COUNT(*)                                              AS paired_cases,
       ROUND(AVG(b.score - a.score), 4)                      AS mean_score_delta,
       SUM(b.pass_bool = 1 AND a.pass_bool = 0)              AS off_only_pass,
       SUM(a.pass_bool = 1 AND b.pass_bool = 0)              AS on_only_pass
FROM a JOIN b ON a.case_id = b.case_id;
SQL
  echo
  echo "Product-reached and pathway coverage: see the 'Score Summary' block at the end of"
  echo "  $LOG_DIR/$GROUP_ON.log and $LOG_DIR/$GROUP_OFF.log"
  echo "Caveat: per-step validity (and therefore score/pass) includes step-mapping confidence;"
  echo "  an absent mapping scores 0.5 (scoring.py). Weigh product-reached and pathway first."
}

case "$MODE" in
  plan) plan ;;
  run) run ;;
  report) report ;;
  *) echo "usage: $0 [plan|run|report]" >&2; exit 2 ;;
esac
