# Ralph Overnight Loop

The overnight Ralph loop is a separate orchestration mode from the existing per-reaction Ralph retry loop.

## Purpose

`overnight-ralph` runs a frozen micro-eval slice repeatedly, mutating exactly one lane per experiment and applying hard keep/discard acceptance rules.

## Lanes

- `topology`: mutate one topology profile field in harness JSON
- `harness`: toggle one harness module enablement
- `prompt`: create a prompt variant artifact from `skills/mechanistic/.../SKILL.md`
- `few_shot`: create a few-shot variant artifact from `few_shot.jsonl`

## Mutation Proposer

Two proposers are available (program key `mutation_proposer`, CLI `--mutation-proposer`):

- `random` (default): the blind lane mutators — nudge one topology integer by ±1, toggle one
  module, append a fixed sentence to a prompt, drop the last few-shot line.
- `llm`: `mechanistic_agent/core/llm_mutator.py` builds a compact **failure digest** from the
  last micro-eval slice (failed validator checks with error text, reproposal reasons, rescue
  outcomes, soft-advances) and asks `mutation_model` (default: the run model; `agent-bridge`
  works keyless) for ONE targeted edit through the forced `harness_mutation_proposal` tool:
  a topology field, a module flag or `run_config_defaults` key, a prompt instruction (append /
  replace), or a few-shot example (remove / add). Proposals that name frozen surfaces, unknown
  targets, load-bearing modules, or disallowed lanes are rejected and the blind mutator for the
  scheduled lane runs instead (the ledger summary records the rejection). The keep/discard
  acceptance rule below is unchanged: the model proposes, the eval decides.

The same proposer is available to the island loop: `scripts/evolve_harness.py --island-mode
--mutation-proposer llm [--mutation-model <id>]`, where the digest comes from the previous
generation's case results for that island.

## Freeze Policy

Overnight Ralph must not mutate deterministic validators, scoring logic, holdout data, or model pricing.

Frozen surfaces:
- `skills/mechanistic/*/validator.py`
- `mechanistic_agent/scoring.py`
- `training_data/eval_set.json`
- `training_data/eval_tiers.json`
- `training_data/leaderboard_holdout/` (in wiggum-data by default)
- `model_pricing.json`

## Program File

Program defaults are defined in root `ralph_program.md`.

Key fields:
- `eval_slice_id`, `eval_slice_size`
- `allowed_lanes`
- `max_experiments`, `max_cost_usd`
- `acceptance_threshold_pct`

## Acceptance Rule

Each candidate is kept only when it beats the current parent baseline by `acceptance_threshold_pct` on:
- `completion_pct`
- `validator_pass_pct`
- composite cost-adjusted score

Otherwise the mutation is discarded and logged.

## Ledger Schema

Rows are written to:
- SQLite table: `ralph_experiments`
- JSONL: `<wiggum-data>/traces/overnight_ralph/YYYY-MM-DD_ledger.jsonl` (or in-repo fallback)

Fields:
- `parent_checkpoint_sha`
- `mutated_asset_path`
- `mutation_lane`
- `mutation_summary`
- `eval_slice_id`
- `completion_pct`
- `validator_pass_pct`
- `avg_retries`
- `avg_backtracks`
- `token_cost_usd`
- `keep`
- `revert_reason`
- `timestamp`

## CLI

```bash
python main.py overnight-ralph \
  --eval-slice-id default \
  --lanes topology,harness \
  --max-experiments 20 \
  --max-cost-usd 15 \
  --acceptance-threshold 0.02 \
  --program ralph_program.md
```

## API

- `POST /api/overnight-ralph/start`
- `GET /api/overnight-ralph/status`
- `GET /api/overnight-ralph/ledger`
- `POST /api/overnight-ralph/stop`

## Promotion Path

Overnight Ralph never auto-merges winners. Kept artifacts are copied to `traces/overnight_ralph/candidates/` and require a human PR with evidence-gate validation.
