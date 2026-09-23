# Change Evidence Policy

This is the internal merge policy for `main`. It applies to the maintainer and to automated agents preparing changes; external contributors are not expected to satisfy it themselves (see [CONTRIBUTING.md](../CONTRIBUTING.md)).

Read [SOUL.md](../SOUL.md) first. It explains why the project optimizes for auditable, evidence-backed improvement instead of anecdotal wins.

## The invariant

**A behavior-changing change does not enter `main` without evidence that it improves the system on its required eval tier.**

"No regression" is not enough for behavior-changing changes. Acceptance requires a measurable improvement against the current leaderboard reference for the same eval scope. Improvement on a single reaction is not sufficient; the eval tiers are the arbiter.

## Merge bar by change type

| Change type | Paths | Required gate |
| --- | --- | --- |
| Few-shot examples and prompt instructions | `skills/mechanistic/<call>/few_shot.jsonl`, `skills/mechanistic/<call>/SKILL.md`, model lanes under `skills/mechanistic/<call>/models/<slug>/` | Approved, linked evidence trace for each changed call **and** improvement on the `medium` tier |
| Subagents, validators, schemas, coordinator wiring | `mechanistic_agent/core/subagents.py`, `core/coordinator.py`, `core/validators.py`, `tool_schemas.py`, new `skills/mechanistic/<call>/` | Fast tests for the new logic **and** improvement on `medium`; `hard` improvement strongly preferred |
| Model catalog entries or adapters | `mechanistic_agent/model_pricing.json`, `mechanistic_agent/llm.py` | `tests/fast/test_model_registry.py` **and** improvement on the `easy` tier SOTA for the relevant cost class |
| Harness pipeline and topology profiles | `harness_versions/<name>/harness.json` | `tests/fast/test_harness_config.py`, `tests/fast/test_coordination_topology.py` **and** improvement on `medium` |
| Infrastructure, bug fixes, docs, eval/leaderboard plumbing | everything else | Fast tests only (see Infra Exception) |

You do not have to run all three tiers to merge. Each change type requires improvement on its **required** tier only. When the leaderboard is empty for that scope, the first completed run for the required tier establishes the baseline.

### Infra Exception

Eval and leaderboard workflow infrastructure changes are the one explicit exception to the improvement rule. Examples: `main.py eval` route-planner behavior, leaderboard documentation and policy files, eval-run metadata or reproducibility plumbing.

These merge without a leaderboard claim if they do not claim a harness-quality improvement, include targeted fast tests for the new behavior, and keep the gates above intact for actual prompt, subagent, model, or harness changes.

### Deterministic arbiters require human review

Changes to the validators, the evidence gate, eval tiers, holdout sets, and the model catalog require human core review via [`.github/CODEOWNERS`](../.github/CODEOWNERS), whether the author is a human or an agent. An automated contributor must not be able to change the checks that gate its own work.

## Required baseline checks

Every PR:

```bash
source .venv/bin/activate
python -m pytest tests/fast/ -q
```

Prompt or few-shot changes additionally run the evidence gate:

```bash
PYTHONPATH=. python scripts/validate_prompt_trace_evidence.py --call <call_name>
```

> **Status (September 2026).** The evidence gate's change detector, prompt-bundle hashing, and `traces/evidence/` gitignore rules are being repaired in a follow-up PR; until that lands the CI gate passes vacuously and the maintainer runs the tier evals as the effective gate.

## Evidence flow

1. Run reactions (UI, API, or `python main.py run ...`). Runs produce traces.
2. Approve good traces (`POST /api/traces/{trace_id}/approve`).
3. Export evidence (`POST /api/traces/export_evidence`). Evidence files land under `traces/evidence/<call_name>/<prompt_bundle_sha>/<trace_id>.json` and record the prompt bundle hash, the model version, and `responder_saw_ground_truth`.
4. Make the prompt or few-shot change.
5. Run the evidence gate and the required tier eval.
6. Open the PR with the before/after leaderboard delta.

Evidence must carry `responder_saw_ground_truth: false`. The gate rejects `true` and undeclared, and the leaderboard drops runs that declare `true`. Replaying FlowER's verified steps through a responder is not a capability measurement.

## Agent-bridge provenance

You do not need a provider API key to produce evidence. The keyless **agent bridge** (`--model-name agent-bridge`) lets an external agent answer each model call; see [agent_bridge.md](agent_bridge.md) and `python main.py bridge-serve --help`.

Agent-authored work meets the same gates as everyone else; there is no separate lane or leaderboard. What differs is origin labeling:

- Bridge runs are stamped with a declared `config.origin` block (`responder`, `declared_underlying_model`, `responder_kind`, `budget_observability`). Declare yours with `MECHANISTIC_AGENT_BRIDGE_DECLARED_MODEL` and `MECHANISTIC_AGENT_BRIDGE_RESPONDER_KIND` before running.
- Declare ground-truth exposure with `MECHANISTIC_AGENT_BRIDGE_SAW_GROUND_TRUTH=false|true`.
- Record `responder`, `declared_underlying_model`, `budget_observability`, `responder_saw_ground_truth`, and `official_holdout_exposed_to_agent` in the PR description.
- `agent-bridge` is a delegated *system*, not a raw model, and its cost is `opaque`. It is therefore **not eligible for cost-class SOTA claims**; use it where the artifact is chemistry or structure, not a model-cost claim.

## Running evals and reading the leaderboard

1. **Read [LEADERBOARD.md](../LEADERBOARD.md).** The rank 1 completed row is the current SOTA for that eval scope. Compare your results against that row and state the delta in the PR.
2. **Run the required tier.** Results are stored in `data/mechanistic.db`. Use an explicit `run_group` so comparisons stay readable:

   ```bash
   python main.py eval \
     --eval-set-id <eval_set_id> \
     --tier medium \
     --harness default \
     --run-group medium_<short_descriptive_slug>
   ```

   Find eval set IDs with `sqlite3 data/mechanistic.db "select id, name, version from eval_sets order by created_at desc;"`.

3. **Regenerate the leaderboard** if the result improves it:

   ```bash
   python main.py leaderboard --eval-set-id <eval_set_id> --limit 20 --markdown --output LEADERBOARD.md
   ```

   After official holdout runs, `python main.py update-leaderboard-artifacts` refreshes the Arena table and curriculum artifacts. Never edit leaderboard rows by hand.

Other useful commands:

- Policy-driven single-tier runs: [development_leaderboard_routes.md](development_leaderboard_routes.md); status with `python main.py eval --eval-set-id ignored --tier easy --model <model> --thinking-level high --leaderboard-status-only`.
- Harness-free baselines: `python main.py baseline --all-tiers --model <model> --thinking-level high`.
- Tier sweeps: `python main.py eval --eval-set-id eval_set --all-tiers --model <model> --thinking-level low`.
- Tier mode reads `training_data/baseline_tier_eval_set_map.json`; the active tier inventory comes from `training_data/development_leaderboard_policy.json`, which selects between `training_data/eval_tiers.json` and `training_data/baseline_tiers_clawdiator.json`.

### Practice eval set

A 20-reaction practice set with the same format as the official set but **completely disjoint reactions**:

```bash
python main.py eval --eval-set training_data/practice_eval/practice_set.json --tier easy
```

It is not the leaderboard set. Use it to verify a change end-to-end before the real tier run. See [../training_data/practice_eval/README.md](../training_data/practice_eval/README.md).

### Data provenance

Eval sets are FlowER-derived: *Electron flow matching for generative reaction mechanism prediction*, Nature 645, 115–123 (2025), DOI [10.1038/s41586-025-09426-9](https://doi.org/10.1038/s41586-025-09426-9). To recreate or extend them, see the [FlowER dataset on figshare](https://figshare.com/articles/dataset/FlowER_-_Mechanistic_datasets_and_model_checkpoint/28359407/3) and [../training_data/REGENERATE.md](../training_data/REGENERATE.md).

## Canonical paths

| Area | Path |
| --- | --- |
| Mechanistic prompts and few-shots | [skills/mechanistic](../skills/mechanistic) |
| Project-level skills | [skills/project](../skills/project) |
| Harness configs | [harness_versions](../harness_versions) |
| Model catalog | [mechanistic_agent/model_pricing.json](../mechanistic_agent/model_pricing.json) |
| LLM adapters | [mechanistic_agent/llm.py](../mechanistic_agent/llm.py) |
| Tool schemas | [mechanistic_agent/tool_schemas.py](../mechanistic_agent/tool_schemas.py) |
| Subagents | [mechanistic_agent/core/subagents.py](../mechanistic_agent/core/subagents.py) |
| Coordinator | [mechanistic_agent/core/coordinator.py](../mechanistic_agent/core/coordinator.py) |
| Validators | [mechanistic_agent/core/validators.py](../mechanistic_agent/core/validators.py) |
| Evidence traces | `traces/evidence/<call_name>/<prompt_bundle_sha>/` |
| Eval tiers | [training_data/eval_tiers.json](../training_data/eval_tiers.json) |
| Agent playbooks | [agent_playbooks.md](agent_playbooks.md) |

## Local scratch

Dry-run PR drafts and private notes belong under `local_contributions/pr_drafts/` or `local_contributions/runs/`, which are gitignored. Other files under `local_contributions/` are tracked maintainer evidence notes and scripts.
