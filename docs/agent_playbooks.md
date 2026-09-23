# Agent Playbooks

Example prompts for an AI coding agent (Cursor, Claude Code, etc.) that has cloned this repo, one per common maintainer scenario. Each ends in a PR that satisfies [change_evidence_policy.md](change_evidence_policy.md). Copy, adapt, and run.

---

## Benchmark a new model and open a PR if it beats the leaderboard

```
I want to test a new model on this mechanistic reaction prediction system and open a PR if it beats the current leaderboard.

Steps:
1. Read docs/change_evidence_policy.md (model catalog entries) and LEADERBOARD.md to understand the current easy-tier SOTA.
2. Check mechanistic_agent/model_pricing.json to see if the model is already registered. If not, add it following the schema used for existing models.
3. If the model needs a new LLM adapter, add it to mechanistic_agent/llm.py.
4. Run the fast test suite to confirm nothing is broken: `source .venv/bin/activate && python -m pytest tests/fast/ -q`
5. Run the easy-tier eval for the new model (model changes require easy-tier improvement):
   `python main.py eval --tier easy --model <new_model_id> --thinking-level <level>`
6. Run the no-harness baseline for comparison:
   `python main.py baseline --tier easy --model <new_model_id> --thinking-level <level>`
7. Check the leaderboard: `python main.py leaderboard --eval-set-id <eval_set_id>`
8. If the new model beats the current easy-tier SOTA for its cost class, regenerate LEADERBOARD.md:
   `python main.py leaderboard --eval-set-id <eval_set_id> --limit 20 --markdown --output LEADERBOARD.md`
9. Open a PR using .github/pull_request_template.md. Mark it behavior-changing and include the before/after leaderboard delta, price, provider, reasoning support, and whether forced tools are supported.

Model to test: <model_id, e.g. openai/gpt-5-mini or google/gemini-2.5-flash>
```

---

## Push forward on an existing model family (higher thinking level or more cases)

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

## Test a novel harness component (new module, topology, or validator patch)

```
I want to experiment with a harness change — either a new module order, a different coordination topology, or a relaxed validator — and see if it improves the medium-tier leaderboard score.

Steps:
1. Read docs/harness_cookbook.md and AGENTS.md (Harness Configuration section) to understand the harness schema.
2. Check the existing harness configs: `ls harness_versions/`
3. Create a new harness variant by copying the default:
   `cp -r harness_versions/default harness_versions/<my_experiment_name>`
   Then edit `harness_versions/<my_experiment_name>/harness.json` with your change.
4. Run the fast harness tests: `python -m pytest tests/fast/test_harness_config.py tests/fast/test_coordination_topology.py -q`
5. Run a dry-run single case to verify the harness loads:
   `python main.py run --starting "<SMILES>" --products "<SMILES>" --harness <my_experiment_name>`
6. Run medium-tier eval with the new harness (harness changes require medium-tier improvement):
   `python main.py eval --tier medium --harness <my_experiment_name> --model anthropic/claude-opus-4.6 --thinking-level high`
7. Compare with the default harness result using:
   `python main.py leaderboard --eval-set-id <eval_set_id>`
8. If it improves on medium, open a PR. Mark it behavior-changing, include the before/after delta, describe exactly which modules moved, were added, or were removed, and explain any validator removal or relaxation. If comparing topologies, include per-topology leaderboard rows.

My harness change: <describe the change — e.g. "add a reagent pre-check module before atom mapping" or "switch to decentralized_mas topology with 3 agents">
```

Notes: `--island-mode` in `scripts/evolve_harness.py` enables archive-based parent selection (see `mechanistic_agent/core/archive.py`). Topology is set at run time via `coordination_topology` in the request, so a basic topology experiment needs no harness file change; tuning per-harness `topology_profiles` defaults does, and goes through the harness gate.

---

## Add few-shot examples to improve a specific failure mode

```
I've noticed the agent struggles with a particular reaction type and I want to add few-shot examples to improve it.

Steps:
1. Identify which skill/subagent handles the step that's failing. Read AGENTS.md (Subagent Architecture) and check skills/mechanistic/ for the relevant call_name.
2. Run a single failing case to capture a trace:
   `python main.py run --starting "<SMILES>" --products "<SMILES>" --model anthropic/claude-opus-4.6`
   Note the run ID from the output.
3. Review the trace in the UI (http://127.0.0.1:8010 if server is running) or via:
   `python main.py compare-eval-runs --run-a <id> --run-b <id>`
4. Approve the good trace and export evidence (see docs/change_evidence_policy.md, Evidence flow).
5. Draft a new few-shot example following the format in skills/mechanistic/<call_name>/few_shot.jsonl. `input` and `output` are serialized strings, not nested JSON objects.
6. Validate the evidence: `PYTHONPATH=. python scripts/validate_prompt_trace_evidence.py --call <call_name>`
7. Run the medium-tier eval to confirm improvement (prompt changes require medium-tier improvement):
   `python main.py eval --tier medium --model anthropic/claude-opus-4.6 --thinking-level high`
8. If it improves, regenerate LEADERBOARD.md and open a PR. Mark it behavior-changing and link the evidence trace under traces/evidence/<call_name>/.

Failing reaction type: <e.g. "ester hydrolysis under acidic conditions" or "Mitsunobu reaction">
```

---

## Run a full Clawdiators-scale benchmark from scratch

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
