# Contributing to Mechanistic Agent

Mechanistic Agent only accepts changes that move the leaderboard forward. A contribution that helps one reaction but does not improve the tracked eval tiers is not mergeable.

Read [SOUL.md](SOUL.md) first. It explains why the project optimizes for auditable, evidence-backed improvement instead of anecdotal wins.

## Core rule

Every mergeable PR must show leaderboard improvement on its required eval gate.

| Track | What changes | Mergeable | Required gate |
| --- | --- | --- | --- |
| Track 1 | Few-shot examples in [skills/mechanistic](skills/mechanistic) | Yes | Must improve `medium` tier |
| Track 2 | New or replaced subagents, validators, schemas, coordinator wiring | Yes | Must improve `medium`; `hard` improvement strongly preferred |
| Track 3 | New model catalog entries or adapters | Yes | Must improve `easy` tier SOTA for the relevant cost class |
| Track 4 | Harness pipeline changes in [harness_versions](harness_versions) | Yes | Must improve `medium` tier |
| Track 5 | Single reaction submissions, success or failure | No | No merge gate; reviewed as evidence for future changes |

“No regression” is not enough for Tracks 1 through 4. Acceptance requires a measurable improvement against the current leaderboard reference for the same eval scope.

## Check Current SOTA

Use the public Markdown leaderboard to see the current bar before opening a PR.

1. Find the eval set ID:
```bash
sqlite3 data/mechanistic.db "select id, name, version from eval_sets order by created_at desc;"
```
2. Regenerate the Markdown leaderboard:
```bash
source .venv/bin/activate
python main.py leaderboard --eval-set-id <eval_set_id> --limit 20 --markdown --output LEADERBOARD.md
```
3. Read [LEADERBOARD.md](LEADERBOARD.md):
The rank 1 completed row is the current SOTA for that eval scope.
4. Compare your PR results against that row and state the delta in the PR description.

Recommended eval naming:

```bash
python main.py eval \
  --eval-set-id <eval_set_id> \
  --tier medium \
  --harness default \
  --run-group pr_medium_<short_slug>
```

Use explicit `run_group` names such as `pr_medium_atom_mapping_fix` so leaderboard comparisons stay readable.

## Public Leaderboard Policy

- [LEADERBOARD.md](LEADERBOARD.md) is the human-readable snapshot for quick review.
- Regenerate it for any PR that claims a new SOTA or changes an eval gate.
- Do not hand-edit leaderboard scores.
- Single-reaction submissions do not update `LEADERBOARD.md`.

## Local-Only Drafts and Dry Runs

Dry runs, fake PRs, and single-reaction submissions belong under `local_contributions/`, which is gitignored.

Suggested layout:

```text
local_contributions/
  pr_drafts/<yyyy-mm-dd>-<slug>.md
  single_reactions/<yyyy-mm-dd>-<slug>.md
  leaderboard/<yyyy-mm-dd>-<slug>.md
```

Use this for:

- dry-run PR drafts you want reviewed locally before a real PR exists
- single-reaction success or failure submissions
- private leaderboard snapshots or reviewer notes

Nothing under `local_contributions/` should be committed.

## Required Baseline Checks

All mergeable tracks require:

```bash
source .venv/bin/activate
python -m pytest tests/fast/ -q
```

LLM-backed changes also require the appropriate eval tier run and, where relevant, evidence validation:

```bash
PYTHONPATH=. python scripts/validate_prompt_trace_evidence.py --call <call_name>
```

## Canonical Paths

| Area | Path |
| --- | --- |
| Mechanistic prompts | [skills/mechanistic](skills/mechanistic) |
| Project-level skills | [skills/project](skills/project) |
| Harness configs | [harness_versions](harness_versions) |
| Model catalog | [mechanistic_agent/model_pricing.json](mechanistic_agent/model_pricing.json) |
| LLM adapters | [mechanistic_agent/llm.py](mechanistic_agent/llm.py) |
| Tool schemas | [mechanistic_agent/tool_schemas.py](mechanistic_agent/tool_schemas.py) |
| Subagents | [mechanistic_agent/core/subagents.py](mechanistic_agent/core/subagents.py) |
| Coordinator | [mechanistic_agent/core/coordinator.py](mechanistic_agent/core/coordinator.py) |
| Validators | [mechanistic_agent/core/validators.py](mechanistic_agent/core/validators.py) |
| Evidence traces | `traces/evidence/<call_name>/<prompt_bundle_sha>/` |
| Eval tiers | [training_data/eval_tiers.json](training_data/eval_tiers.json) |
| Contribution templates | [templates/contributions/README.md](templates/contributions/README.md) |

## Track 1: Few-Shot Examples

Use this when you are adding new examples to [skills/mechanistic/<call_name>/few_shot.jsonl](skills/mechanistic).

Template:
[templates/contributions/track1_few_shot_pr.md](templates/contributions/track1_few_shot_pr.md)

Required tests:

- `PYTHONPATH=. python scripts/validate_prompt_trace_evidence.py --call <call_name>`
- `python -m pytest tests/fast/ -q`
- `PYTHONPATH=. pytest tests/llm/test_eval_tiers.py --tier medium -k medium`

Checklist:

- [ ] Added example lines only to the target [skills/mechanistic/<call_name>/few_shot.jsonl](skills/mechanistic)
- [ ] Linked approved evidence trace under `traces/evidence/<call_name>/...`
- [ ] Confirmed `input` and `output` are serialized strings, not nested JSON objects
- [ ] Included before/after medium-tier leaderboard delta
- [ ] Added contact info for attribution in future manuscript updates

## Track 2: New Subagents

Use this when you add or replace a deterministic subagent, validator, or LLM-backed subagent.

Template:
[templates/contributions/track2_subagent_pr.md](templates/contributions/track2_subagent_pr.md)

Required tests:

- `python -m pytest tests/fast/ -q`
- Relevant new `tests/fast/test_<subagent_name>.py`
- `PYTHONPATH=. pytest tests/llm/test_eval_tiers.py --tier medium -k medium`

Strongly recommended:

- `PYTHONPATH=. pytest tests/llm/test_eval_tiers.py --tier hard -k hard`

Checklist:

- [ ] Added or updated the relevant skill directory under [skills/mechanistic](skills/mechanistic) if the subagent is LLM-backed
- [ ] Added tool schema and text fallback if the subagent calls an LLM
- [ ] Wired the subagent into [mechanistic_agent/core/subagents.py](mechanistic_agent/core/subagents.py) and [mechanistic_agent/core/coordinator.py](mechanistic_agent/core/coordinator.py) where needed
- [ ] Added fast tests for core logic and failure handling
- [ ] Included before/after medium-tier leaderboard delta
- [ ] Included hard-tier delta or explained why it was not run
- [ ] Added contact info for attribution in future manuscript updates

## Track 3: New Models

Use this when you add a model catalog entry or a new provider adapter.

Template:
[templates/contributions/track3_model_pr.md](templates/contributions/track3_model_pr.md)

Required tests:

- `python -m pytest tests/fast/ -q`
- `PYTHONPATH=. pytest tests/fast/test_model_registry.py`
- `PYTHONPATH=. pytest tests/llm/test_eval_tiers.py --tier easy -k easy`

Checklist:

- [ ] Added the model entry to [mechanistic_agent/model_pricing.json](mechanistic_agent/model_pricing.json)
- [ ] Added adapter routing in [mechanistic_agent/llm.py](mechanistic_agent/llm.py) if needed
- [ ] Declared whether forced tools are supported
- [ ] Compared the new model against the current easy-tier SOTA for its cost class
- [ ] Included price, provider, and reasoning support details
- [ ] Added contact info for attribution in future manuscript updates

## Track 4: Harness Configuration Changes

Use this when you change module ordering, module enablement, validator patching, or topology profiles in [harness_versions](harness_versions).

This track also covers **coordination topology experiments** — comparing `sas`, `centralized_mas`, `independent_mas`, or `decentralized_mas` on the eval tiers. Topology is set at run time via `coordination_topology` in the request, so no harness file change is required for a basic experiment. However, if you want to tune per-harness `topology_profiles` defaults (e.g. agent count, peer rounds, consensus key), that change goes through Track 4 and requires a new harness subdirectory or a modified `harness.json`.

Template:
[templates/contributions/track4_harness_pr.md](templates/contributions/track4_harness_pr.md)

Required tests:

- `python -m pytest tests/fast/ -q`
- `PYTHONPATH=. pytest tests/fast/test_harness_config.py`
- `PYTHONPATH=. pytest tests/fast/test_coordination_topology.py`
- `PYTHONPATH=. pytest tests/llm/test_eval_tiers.py --tier medium -k medium`

Recommended local check:

- one or more dry runs with your candidate harness (or topology setting) before the eval tier run

Checklist:

- [ ] Saved the harness variant under [harness_versions/<name>/harness.json](harness_versions)
- [ ] Described exactly which modules moved, were added, or were removed
- [ ] If changing topology profiles: documented `agent_count`, `peer_rounds`, and `aggregation_mode` changes
- [ ] If comparing topologies: included per-topology leaderboard rows so the delta is visible
- [ ] Included before/after medium-tier leaderboard delta
- [ ] Explained any validator removal or relaxation
- [ ] Added contact info for attribution in future manuscript updates

## Track 5: Single Reaction Submission

Use this when you want to submit one success or failure case for review, not merge.

These submissions are triage inputs. They may lead to a later few-shot update, harness change, subagent change, or model test. They do not merge directly and they do not satisfy leaderboard gates on their own.

Template:
[templates/contributions/track5_single_reaction_submission.md](templates/contributions/track5_single_reaction_submission.md)

Storage rule:

- Save the filled template under `local_contributions/single_reactions/`
- If you want a fake PR draft for discussion, save it under `local_contributions/pr_drafts/`
- Do not commit either file

Suggested checks:

- run the reaction once with the exact model and harness you are reporting
- include the relevant run ID, trace IDs, and whether the example is a success or failure
- if possible, include why you think it matters: prompt gap, missing reagent pattern, mapping failure, validator issue, harness ordering issue, or model weakness

Checklist:

- [ ] Marked the example as `success` or `failure`
- [ ] Included exact starting materials, products, model, harness, and run ID
- [ ] Attached or linked local traces/screenshots if useful
- [ ] Explained why this case might justify a tracked change
- [ ] Added contact info for attribution in future manuscript updates

## Attribution Request

Every template asks for optional contact info for possible manuscript acknowledgements or update notes.

Preferred fields:

- name
- email
- ORCID, GitHub, or other preferred attribution handle
- whether you want to be contacted before attribution

Providing contact info is optional, but if you want attribution later, include it now so it does not have to be reconstructed from commit history.
