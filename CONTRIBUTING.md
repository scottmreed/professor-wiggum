# Contributing to Mechanistic Agent

Mechanistic Agent only accepts changes that move the leaderboard forward. A contribution that helps one reaction but does not improve the tracked eval tiers is not mergeable.

Read [SOUL.md](SOUL.md) first. It explains why the project optimizes for auditable, evidence-backed improvement instead of anecdotal wins.

If you are deciding whether to clone the repo first, start with [SETUP.md](SETUP.md). You can contribute without a clone for Track 5, while Tracks 1 through 4 will usually require a local checkout plus tests and eval runs.

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

## Infra Exception

Eval and leaderboard workflow infrastructure changes are the one explicit exception to the leaderboard-improvement rule.

Examples:

- `main.py eval` route-planner behavior
- leaderboard documentation and policy files
- eval-run metadata or reproducibility plumbing

These PRs are mergeable without claiming a new leaderboard improvement if they:

- do not claim a harness-quality improvement
- include targeted fast tests for the new workflow behavior
- keep the existing track gates intact for actual prompt, subagent, model, or harness changes

## Keyless and agent-authored contributions

You do not need a provider API key to contribute. The keyless **agent bridge**
(`--model-name agent-bridge`) lets an external agent or subagent answer each model
call, so an agent surface can drive real runs, produce traces and eval-tier
evidence, and open PRs. See [docs/agent_bridge.md](docs/agent_bridge.md) and
`python main.py bridge-serve --help`.

Agent-authored work uses the **same tracks and the same gates as everyone else** —
there is no separate "agent" lane and no separate leaderboard:

- A keyless run that improves the required eval tier is a normal Track 1, 2, or 4 PR.
- Single-reaction agent evidence goes through Track 5, exactly like a human's.
- Correctness is decided by the deterministic RDKit validators regardless of who
  produced the step, so no quarantine lane is needed (SOUL.md Guardrail 1).

What differs is **origin labeling**, so the data's provenance stays auditable:

- Bridge runs are stamped with a declared `config.origin` block (`responder`,
  `declared_underlying_model`, `responder_kind`, `budget_observability`). Declare
  yours with the `MECHANISTIC_AGENT_BRIDGE_DECLARED_MODEL` and
  `MECHANISTIC_AGENT_BRIDGE_RESPONDER_KIND` env vars before running.
- Fill the origin fields in the PR template (`responder`,
  `declared_underlying_model`, `budget_observability`,
  `official_holdout_exposed_to_agent`).
- `agent-bridge` is a delegated *system*, not a raw model, and its cost is
  `opaque`. It is therefore **not eligible for Track 3 cost-class SOTA claims**;
  use Tracks 1/2/4 where the artifact is chemistry/structure, not a model-cost claim.

Changes to the deterministic arbiters (validators, evidence gate, eval tiers,
holdout sets, model catalog) require human core review via
[`.github/CODEOWNERS`](.github/CODEOWNERS) — for every contributor, human or agent.

## professor-wiggum (maintainer fork)

This checkout may use **long-lived integration branches** (for example `curriculum-workflow`) in addition to `main`. **Pushing does not clear local edits:** `git status` can still list modified or untracked files until you commit, stash, or restore them—that is independent of merge **conflicts** (which only appear after a failed merge/rebase with conflict markers). When opening a PR, pick the **correct base branch** on GitHub and say which branch you integrated from.

Agent-oriented PR workflow notes (branch hygiene, draft paths, optional release workflows) live in the tracked skill **[`.claude/skills/pr-creation-helper/SKILL.md`](.claude/skills/pr-creation-helper/SKILL.md)**.

## Clone Expectations

- Track 5 usually does not require a git clone. You can submit a single reaction through the UI or API and provide traces or notes for review.
- Tracks 1 through 4 will usually require a git clone because they change repo files, run local tests, and need eval evidence suitable for a PR.
- If you are preparing for Tracks 1 through 4, use [SETUP.md](SETUP.md) for environment setup before making changes.

## Check Current SOTA

Use the public Markdown leaderboard to see the current bar and where there is room for improvement.

1. **Read [LEADERBOARD.md](LEADERBOARD.md)**  
   If completed rows exist, the rank 1 completed row is the current SOTA for that eval scope. If the file is still a placeholder or has no completed rows for your tier, the first completed row you add for that scope establishes the initial baseline. Compare your PR results against that row and state the delta in the PR description.

2. **Run evals** (before claiming a new SOTA). Run the eval tier required by your track (see Core rule table). Results are stored in `data/mechanistic.db`.

3. **If you have run evals and improved on the leaderboard**, regenerate the Markdown leaderboard so your commit includes an up-to-date snapshot:
   - Find the eval set ID:
   ```bash
   sqlite3 data/mechanistic.db "select id, name, version from eval_sets order by created_at desc;"
   ```
   - Regenerate the leaderboard:
   ```bash
   source .venv/bin/activate
   python main.py leaderboard --eval-set-id <eval_set_id> --limit 20 --markdown --output LEADERBOARD.md
   ```

**Recommended eval naming (when adding novel reactions or running evals)**  
You can use existing FlowER-derived eval sets or create new eval sets from deeper in the FlowER data. FlowER data is from: *Electron flow matching for generative reaction mechanism prediction.* Nature 645, 115–123 (2025). DOI: [10.1038/s41586-025-09426-9](https://doi.org/10.1038/s41586-025-09426-9). To recreate or extend the data, see the [FlowER dataset on figshare](https://figshare.com/articles/dataset/FlowER_-_Mechanistic_datasets_and_model_checkpoint/28359407/3) and [training_data/REGENERATE.md](training_data/REGENERATE.md). When running evals, use an explicit `run_group` so leaderboard comparisons stay readable:

```bash
python main.py eval \
  --eval-set-id <eval_set_id> \
  --tier medium \
  --harness default \
  --run-group medium_<short_descriptive_slug>
```

Example: `medium_few_shot_conditions_v2` or `medium_harness_default_mar2026` (a short slug describing the change, not a PR title).

For policy-driven single-tier development runs, use the route planner documented in [docs/development_leaderboard_routes.md](docs/development_leaderboard_routes.md). The planner status can be inspected with:

```bash
python main.py eval \
  --eval-set-id ignored \
  --tier easy \
  --model anthropic/claude-opus-4.6 \
  --thinking-level high \
  --leaderboard-status-only
```

For harness-free baseline references (no API server), run all baseline tiers in one command:

```bash
python main.py baseline --all-tiers --model anthropic/claude-opus-4.6 --thinking-level high
```

Tier mode reads `training_data/baseline_tier_eval_set_map.json` to resolve tier-specific `eval_set_id` values.
The active tier inventory for planner-managed `eval --tier` runs comes from the policy in [training_data/development_leaderboard_policy.json](training_data/development_leaderboard_policy.json), which selects between [training_data/eval_tiers.json](training_data/eval_tiers.json) and [training_data/baseline_tiers_clawdiator.json](training_data/baseline_tiers_clawdiator.json) per tier.

**Tier requirement for merge.** You do not have to run all three tiers (easy/medium/hard) to merge. Each track requires improvement on its **required** tier only (e.g. medium for Tracks 1, 2, 4; easy for Track 3). When the leaderboard is empty for that scope, the first completed run for the required tier establishes the baseline and a PR can merge if it meets the track’s gate.

**Tier requirement for merge.** You do not have to run all three tiers (easy/medium/hard) to merge. Each track requires improvement on its **required** tier only (e.g. medium for Tracks 1, 2, 4; easy for Track 3). When the leaderboard is empty for that scope, the first completed run for the required tier establishes the baseline and a PR can merge if it meets the track’s gate.

Harness eval tier sweeps can also run in one command:

```bash
python main.py eval --eval-set-id eval_set --all-tiers --model anthropic/claude-opus-4.6 --thinking-level low
```

## Public Leaderboard Policy

- [LEADERBOARD.md](LEADERBOARD.md) is the human-readable snapshot for quick review.
- Regenerate it for any PR that claims a new SOTA or changes an eval gate. Use the CLI only: `python main.py leaderboard --eval-set-id <eval_set_id> --limit 20 --markdown --output LEADERBOARD.md` (or, after official holdout runs, `python main.py update-leaderboard-artifacts` to refresh the Arena table and curriculum artifacts). Do not edit leaderboard scores or table rows by hand.
- Single-reaction submissions do not update `LEADERBOARD.md`.

## Local-Only Drafts and Dry Runs

Dry runs, fake PRs, and single-reaction submissions belong under `local_contributions/`, which is gitignored.

Suggested layout:

```text
local_contributions/
  pr_drafts/<slug>.md
  single_reactions/<slug>.md
  leaderboard/<slug>.md
```

You may add an optional `YYYY-MM-DD-` (or similar) prefix to filenames for your own ordering; it is not required.

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

## Practice Eval Set

A 20-reaction practice eval set is provided for local testing. It uses the same format and FlowER source (see [FlowER dataset on figshare](https://figshare.com/articles/dataset/FlowER_-_Mechanistic_datasets_and_model_checkpoint/28359407/3) and [training_data/REGENERATE.md](training_data/REGENERATE.md)) as the official eval set but contains **completely disjoint reactions**.

```bash
source .venv/bin/activate
python main.py eval --eval-set training_data/practice_eval/practice_set.json --tier easy
```

This set is **not** the leaderboard eval set. Use it to verify your changes work end-to-end before running the real eval tier for your PR. See [training_data/practice_eval/README.md](training_data/practice_eval/README.md) for details.

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

> **Not for `agent-bridge`.** The keyless agent bridge is a delegated *system* with
> `budget_observability: opaque`, not a raw model with a real cost. It cannot make a
> Track 3 cost-class SOTA claim. Contribute agent-bridge results through Tracks 1/2/4.

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

**Island-based evolution:** The `--island-mode` flag in `scripts/evolve_harness.py` enables archive-based parent selection inspired by [ShinkaEvolve](https://arxiv.org/abs/2509.19349). Islands partition the search space by mutation target (mapping, reagent/conditions, topology, hard multi-step) and migration between islands is gated on real eval improvement. See `mechanistic_agent/core/archive.py` for the archive and parent selection logic.

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

---

## Agent Quick-Start Prompts

These are example prompts you can give to an AI coding agent (Cursor, Claude Code, etc.) that has cloned this repo. Each covers a common contribution scenario end-to-end. Copy, adapt, and run.

### Benchmark a new model and submit a PR if it beats the leaderboard

```
I want to test a new model on this mechanistic reaction prediction system and open a PR if it beats the current leaderboard.

Steps:
1. Read CONTRIBUTING.md (Track 3 — New Models) and LEADERBOARD.md to understand the current easy-tier SOTA.
2. Check mechanistic_agent/model_pricing.json to see if the model is already registered. If not, add it following the schema used for existing models.
3. If the model needs a new LLM adapter, add it to mechanistic_agent/llm.py.
4. Run the fast test suite to confirm nothing is broken: `source .venv/bin/activate && python -m pytest tests/fast/ -q`
5. Run the easy-tier eval for the new model (Track 3 requires easy-tier improvement):
   `python main.py eval --tier easy --model <new_model_id> --thinking-level <level>`
6. Run the no-harness baseline for comparison:
   `python main.py baseline --tier easy --model <new_model_id> --thinking-level <level>`
7. Check the leaderboard: `python main.py leaderboard --eval-set-id <eval_set_id>`
8. If the new model beats the current easy-tier SOTA for its cost class, regenerate LEADERBOARD.md:
   `python main.py leaderboard --eval-set-id <eval_set_id> --limit 20 --markdown --output LEADERBOARD.md`
9. Open a PR using the Track 3 template at templates/contributions/track3_model_pr.md. Include the before/after leaderboard delta.

Model to test: <model_id, e.g. openai/gpt-5-mini or google/gemini-2.5-flash>
```

---

### Push forward on an existing model family (try harder thinking level or more cases)

```
I want to improve the leaderboard score for an existing model family by testing a higher thinking level or running more cases.

Steps:
1. Check the current leaderboard status for the model:
   `python main.py eval --tier easy --model <model_id> --thinking-level <current_level> --leaderboard-status-only`
2. Check what runs have already been attempted:
   `python main.py leaderboard --eval-set-id <eval_set_id>`
3. Run the route planner for the next recommended step (it will auto-select unrun cases):
   `python main.py eval --tier easy --model <model_id> --thinking-level high`
4. If easy-tier looks good, extend to medium:
   `python main.py eval --tier medium --model <model_id> --thinking-level high`
5. If you see improvement, run the official holdout eval (20 cases):
   `python main.py eval-runset-official --model <model_id> --thinking-level high`
6. Refresh the leaderboard snapshot:
   `python main.py update-leaderboard-artifacts`
7. If the score improves over the previous SOTA for that model family, open a PR with the leaderboard delta.

Model family: <e.g. anthropic/claude-opus-4.6 or openai/gpt-5.4>
```

---

### Test a novel harness component (new module, topology, or validator patch)

```
I want to experiment with a harness change — either a new module order, a different coordination topology, or a relaxed validator — and see if it improves the medium-tier leaderboard score.

Steps:
1. Read docs/harness_cookbook.md and AGENTS.md (Harness Configuration section) to understand the harness schema.
2. Check the existing harness configs: `ls harness_versions/`
3. Create a new harness variant by copying the default:
   `cp -r harness_versions/default harness_versions/<my_experiment_name>`
   Then edit `harness_versions/<my_experiment_name>/harness.json` with your change.
4. Run the fast harness tests: `python -m pytest tests/fast/test_harness_config.py -q`
5. Run a dry-run single case to verify the harness loads:
   `python main.py run --starting "<SMILES>" --products "<SMILES>" --harness <my_experiment_name>`
6. Run medium-tier eval with the new harness (Track 4 requires medium-tier improvement):
   `python main.py eval --tier medium --harness <my_experiment_name> --model anthropic/claude-opus-4.6 --thinking-level high`
7. Compare with the default harness result using:
   `python main.py leaderboard --eval-set-id <eval_set_id>`
8. If it improves on medium, open a PR using templates/contributions/track4_harness_pr.md. Include the before/after delta and a description of which modules changed.

My harness change: <describe the change — e.g. "add a reagent pre-check module before atom mapping" or "switch to decentralized_mas topology with 3 agents">
```

---

### Add few-shot examples to improve a specific failure mode

```
I've noticed the agent struggles with a particular reaction type and I want to add few-shot examples to improve it.

Steps:
1. Identify which skill/subagent handles the step that's failing. Read AGENTS.md (Subagent Architecture) and check skills/mechanistic/ for the relevant call_name.
2. Run a single failing case to capture a trace:
   `python main.py run --starting "<SMILES>" --products "<SMILES>" --model anthropic/claude-opus-4.6`
   Note the run ID from the output.
3. Review the trace in the UI (http://127.0.0.1:8010 if server is running) or via:
   `python main.py compare-eval-runs --run-a <id> --run-b <id>`
4. Draft a new few-shot example following the format in skills/mechanistic/<call_name>/few_shot.jsonl.
5. Validate the evidence: `PYTHONPATH=. python scripts/validate_prompt_trace_evidence.py --call <call_name>`
6. Run the medium-tier eval to confirm improvement (Track 1 requires medium-tier improvement):
   `python main.py eval --tier medium --model anthropic/claude-opus-4.6 --thinking-level high`
7. If it improves, regenerate LEADERBOARD.md and open a PR using templates/contributions/track1_few_shot_pr.md.

Failing reaction type: <e.g. "ester hydrolysis under acidic conditions" or "Mitsunobu reaction">
```

---

### Run a full Clawdiators-scale benchmark from scratch

```
I want to run a complete benchmark on this repo — no-harness baseline + full harness — at the 20-case official holdout scale, then view the leaderboard.

Steps:
1. Activate the venv: `source .venv/bin/activate`
2. Make sure the holdout eval set is imported (one-time setup):
   `python main.py import-holdout-eval-set`
3. Run the no-harness single-shot baseline on the official holdout (20 cases):
   `python main.py baseline-runset-official --model anthropic/claude-opus-4.6 --thinking-level high`
4. Run the full harness on the official holdout (20 cases):
   `python main.py eval-runset-official --model anthropic/claude-opus-4.6 --thinking-level high`
5. View the official leaderboard:
   `python main.py leaderboard-official`
6. Refresh LEADERBOARD.md and curriculum artifacts:
   `python main.py update-leaderboard-artifacts`
7. Compare harness vs baseline:
   `python local_contributions/compare_harness_vs_baseline_samples.py`

Model to benchmark: <model_id>
Thinking level: <high / low / none>
```
