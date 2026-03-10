# Ralph Overnight Program

mutation_budget_per_night_experiments: 20
mutation_budget_per_night_cost_usd: 15.0
max_experiments: 20
max_cost_usd: 15.0

# Optional explicit frozen eval slice selector (deterministic sample seed)
eval_slice_id: default
eval_slice_size: 10

allowed_lanes:
- topology
- harness
- prompt
- few_shot

frozen_surfaces:
- skills/mechanistic/*/validator.py
- mechanistic_agent/scoring.py
- training_data/eval_set.json
- training_data/eval_tiers.json
- training_data/leaderboard_holdout/
- model_pricing.json

acceptance_threshold_pct: 0.02

# Anti-overfitting rules are intentionally strict by default.
anti_overfitting_rules_max_consecutive_keeps_same_lane: 2
anti_overfitting_rules_min_eval_slice_size: 10

# Hard rollback gate.
rollback_rules_min_validator_pass_pct: 0.0

# Stop loop when any budget or manual stop condition is hit.
stop_conditions_require_human_promotion: true

# Required trace artifacts for promotion review.
trace_artifact_requirements_write_jsonl_ledger: true
trace_artifact_requirements_write_sqlite_ledger: true
trace_artifact_requirements_candidates_dir: traces/overnight_ralph/candidates
