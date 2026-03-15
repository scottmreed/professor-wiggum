# Ralph Overnight Program

This file configures the **overnight Ralph** loop: automated hill-climbing over a small, frozen evaluation slice. The orchestrator mutates one “lane” per experiment (e.g. topology, harness, prompt, or few-shot), runs a micro-eval, and keeps or discards the change based on acceptance rules. Nothing is auto-merged; kept candidates are written to `traces/overnight_ralph/candidates/` for human review and PR.

For full context, see [docs/ralph_overnight.md](docs/ralph_overnight.md).

---

## Budget limits

# How many experiments to run per “night” and the maximum spend. The loop stops when either limit is reached.
mutation_budget_per_night_experiments: 20
mutation_budget_per_night_cost_usd: 15.0
max_experiments: 20
max_cost_usd: 15.0

## Eval slice

# Which fixed set of cases to use. Same eval_slice_id + eval_slice_size gives the same cases (deterministic seed). Larger slices are more stable but cost more.
eval_slice_id: default
eval_slice_size: 10

## Mutation lanes

# Exactly one of these lanes is mutated per experiment. topology = harness topology profile; harness = module on/off; prompt = SKILL.md variant; few_shot = few_shot.jsonl variant.
allowed_lanes:
- topology
- harness
- prompt
- few_shot

## Frozen surfaces

# Paths that overnight Ralph must never modify (validators, scoring, eval data, holdout, model pricing).
frozen_surfaces:
- skills/mechanistic/*/validator.py
- mechanistic_agent/scoring.py
- training_data/eval_set.json
- training_data/eval_tiers.json
- training_data/leaderboard_holdout/
- model_pricing.json

## Acceptance rule

# A candidate is “kept” only if it beats the current baseline by at least this much (e.g. 0.02 = 2%) on completion_pct, validator_pass_pct, and cost-adjusted score. Otherwise it is discarded and reverted.
acceptance_threshold_pct: 0.02

## Anti-overfitting

# Avoid getting stuck on one lane: after this many consecutive keeps in the same lane, the loop will try other lanes. Minimum slice size is also enforced.
anti_overfitting_rules_max_consecutive_keeps_same_lane: 2
anti_overfitting_rules_min_eval_slice_size: 10

## Rollback and stop conditions

# Hard rollback gate: do not keep if validator pass rate falls below this (0 = no hard floor). stop_conditions_require_human_promotion: true means the loop never auto-merges; humans promote from candidates.
rollback_rules_min_validator_pass_pct: 0.0
stop_conditions_require_human_promotion: true

## Trace artifacts

# Where to write ledger and kept candidates. Required for promotion review.
trace_artifact_requirements_write_jsonl_ledger: true
trace_artifact_requirements_write_sqlite_ledger: true
trace_artifact_requirements_candidates_dir: traces/overnight_ralph/candidates

---

# =============================================================================
# Sample command line and workflow
# =============================================================================
#
# 1. Activate the project and (optionally) set API keys:
#
#    source .venv/bin/activate
#    export OPENAI_API_KEY=sk-...
#
# 2. Run overnight Ralph with defaults from this program file:
#
#    python main.py overnight-ralph --program ralph_program.md
#
# 3. Or override budget, slice, lanes, and acceptance on the CLI:
#
#    python main.py overnight-ralph \
#      --program ralph_program.md \
#      --eval-slice-id default \
#      --lanes topology,harness \
#      --max-experiments 20 \
#      --max-cost-usd 15 \
#      --acceptance-threshold 0.02 \
#      --model anthropic/claude-sonnet-4 \
#      --harness default
#
# 4. Watch progress: experiments are logged to the terminal. Ledger rows are
#    appended to data/mechanistic.db (ralph_experiments) and to
#    traces/overnight_ralph/YYYY-MM-DD_ledger.jsonl.
#
# 5. When the run finishes (or you stop it), review kept candidates in
#    traces/overnight_ralph/candidates/. Open a PR with the desired changes
#    and run evidence-gate validation (e.g. validate_prompt_trace_evidence)
#    before merging.
#
# API (with server running via `python main.py serve`):
#
#    POST /api/overnight-ralph/start   -- start a run (body: program path, lanes, etc.)
#    GET  /api/overnight-ralph/status  -- running state and ledger tail
#    GET  /api/overnight-ralph/ledger  -- full ledger (query params: limit, eval_slice_id)
#    POST /api/overnight-ralph/stop    -- request stop
