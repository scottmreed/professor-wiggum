# Harness Cookbook

The harness is a JSON file that defines exactly which steps run, in what order, and with what configuration. Every named harness lives in `harness_versions/<name>/harness.json`. The pipeline executor reads this file at run time; changing the harness requires no Python edits.

This document covers two tasks:
1. **Removing (skipping) a step** — straightforward, two-field change
2. **Adding a new step** — more involved, requires a prompt file, harness entry, and optionally few-shot examples

---

## Harness structure

A harness has three sections of modules:

| Section | Runs when |
|---|---|
| `pre_loop_modules` | Once, before the mechanism loop starts |
| `loop_module` | Each iteration of the mechanism loop |
| `post_step_modules` | After each accepted mechanism step |

Each module has a `"removable"` flag that indicates whether the pipeline can operate without it. Modules with `"removable": false` are load-bearing (e.g., `balance_analysis`, `ph_recommendation`, `initial_conditions`) and cannot be safely disabled.

---

## Part 1: Removing (skipping) a step

### Which steps are removable?

From `harness_versions/default/harness.json`, the removable pre-loop steps are:

| Step | ID | What it does |
|---|---|---|
| 2 | `functional_groups` | SMARTS fingerprinting of reactive groups |
| 5 | `missing_reagents` | LLM prediction of missing reactants/products |
| 6 | `atom_mapping` | LLM atom-to-atom mapping across reactants/products |
| 7 | `reaction_type_mapping` | LLM taxonomy label selection |

All `post_step_modules` are also removable (validators and reflection).

### How to skip a step

1. Copy an existing harness as a starting point:
   ```
   cp -r harness_versions/default harness_versions/my_experiment
   ```

2. Open `harness_versions/my_experiment/harness.json`.

3. Find the module you want to skip and set `"enabled": false`:
   ```json
   {
     "step": 5,
     "id": "missing_reagents",
     "enabled": false,
     ...
   }
   ```

4. Update `"name"` at the top to match your directory name, and add a changelog entry:
   ```json
   "name": "my_experiment",
   ...
   "metadata": {
     "changelog": [
       { "version": 1, "date": "2026-03-04", "description": "Disabled missing_reagents to test impact." }
     ]
   }
   ```

5. Run a dry-run to confirm the harness loads cleanly:
   ```
   PYTHONPATH=. pytest tests/fast/test_harness_config.py
   ```

6. Test on a real reaction via the UI: select your harness from the harness dropdown, run a reaction, and compare the flow diagram and step outputs to the default harness.

### Working example

`harness_versions/no_mapping_no_reagents/harness.json` disables both `missing_reagents` (step 5) and `atom_mapping` (step 6). Use it as a template or load it directly from the UI harness selector.

---

## Part 2: Adding a new step

Adding a step requires four things:

1. A **skill file** with the prompt (`skills/mechanistic/<call_name>/SKILL.md`)
2. A **harness entry** in your harness JSON
3. A **tool function** registered in the coordinator (Python)
4. Optionally, **few-shot examples** (`skills/mechanistic/<call_name>/few_shot.jsonl`)

### Step-by-step

#### 1. Create the skill file

```
mkdir -p skills/mechanistic/my_new_step
```

Create `skills/mechanistic/my_new_step/SKILL.md`:

```markdown
---
kind: llm_call
call_name: my_new_step
---

<!-- PROMPT_START -->
You are a chemistry reasoning assistant. Given the starting materials and products of a reaction, [describe what this step should do].

Return a JSON object with:
- "result": string — [describe the output]
- "confidence": float — 0.0 to 1.0
- "reasoning": string — brief explanation
<!-- PROMPT_END -->
```

The `<!-- PROMPT_START -->` / `<!-- PROMPT_END -->` markers are required. The text between them is the system prompt for this LLM call.

#### 2. Add few-shot examples (optional but recommended)

Create `skills/mechanistic/my_new_step/few_shot.jsonl` with one JSON object per line:

```jsonl
{"input": {"starting_materials": ["CC(=O)O"], "products": ["CC=O"]}, "output": {"result": "...", "confidence": 0.9, "reasoning": "..."}}
```

Each line must be a valid JSON object with `"input"` and `"output"` keys matching what the LLM call receives and returns.

#### 3. Register the tool function (Python)

In `mechanistic_agent/tools.py`, add a new function following the pattern of existing tool functions:

```python
def my_new_step(
    starting_materials: list[str],
    products: list[str],
    **kwargs,
) -> dict:
    """Run my new analysis step."""
    call_name = "my_new_step"
    system_prompt = compose_system_prompt(call_name=call_name)
    few_shots = load_few_shots(call_name=call_name)
    # ... build prompt, call LLM, parse and validate response
    return {"result": ..., "confidence": ..., "reasoning": ...}
```

See `assess_initial_conditions` or `attempt_atom_mapping` in `tools.py` for complete examples of how prompts are composed, how few-shot examples are injected, and how LLM responses are parsed and validated.

You will also need to add the function to `tool_schemas.py` (the OpenAI-format schema) so the harness executor can call it as a tool.

#### 4. Add the harness entry

In your harness JSON, add a module entry to `pre_loop_modules` (for a one-shot pre-loop step) or `post_step_modules` (for a per-step validator). Pick the next available `"step"` number:

```json
{
  "step": 8,
  "id": "my_new_step",
  "label": "My New Step",
  "kind": "llm",
  "phase": "pre_loop",
  "enabled": true,
  "agent_class": "MyNewAgent",
  "tool_function": "my_new_step",
  "step_name": "my_new_step",
  "tool_name": "my_new_step",
  "validator_skill": null,
  "inputs": ["balance_analysis", "initial_conditions"],
  "outputs": ["my_new_step"],
  "movable": true,
  "removable": true,
  "config_gate": null,
  "prompt_call_name": "my_new_step",
  "group_key": null,
  "description": "One sentence description of what this step does.",
  "io_schema": {
    "inputs": ["starting_materials (SMILES[])", "products (SMILES[])", "..."],
    "outputs": ["result (string)", "confidence (float)", "reasoning (string)"]
  }
}
```

Key fields:
- `"inputs"`: list of output IDs from upstream modules your step reads
- `"outputs"`: the output key this step writes, available to downstream modules
- `"removable": true` — always set this for new contributions; it signals the step is optional
- `"config_gate"`: set to `null` unless this step needs to be gated on a run-config flag

#### 5. Run tests

```bash
PYTHONPATH=. pytest tests/fast/test_harness_config.py
PYTHONPATH=. pytest tests/fast/
```

Then run a dry-run reaction from the UI with your harness selected.

---

## Evaluating your harness change

Run the eval set against your harness and the default harness, then compare leaderboard rows:

```bash
# Run eval with your harness
python -m mechanistic_agent.eval --harness my_experiment --eval-set flower_100_default

# Compare in the UI Leaderboard tab or via the API
```

The leaderboard aggregates per-subagent scores. A well-designed harness change should improve quality or pass rate on at least one subagent without degrading others.

### Development vs holdout evals

- Use train-derived eval/sample assets for development and tuning:
  - `training_data/eval_set.json`
  - `training_data/eval_tiers.json`
  - UI examples/menu
  - `scripts/evolve_harness.py`
- The isolated holdout suite under `training_data/leaderboard_holdout/` is for official ranking only.
- User-facing runset endpoints intentionally reject holdout eval sets:
  - `POST /api/evals/runset`
  - `POST /api/evals/baseline-runset`
- Official holdout scoring uses:
  - `POST /api/evals/official-runset`
  - `GET /api/evals/leaderboard/official`

---

## Contribution checklist

Before opening a Track 4 PR, verify:

- [ ] Harness saved as a named variant under `harness_versions/<name>/`
- [ ] `tests/fast/test_harness_config.py` passes
- [ ] Dry-run reactions produce sensible output
- [ ] Changelog entry added to `metadata.changelog`
- [ ] For added steps: skill file exists and prompt is between the required markers
- [ ] PR description references the harness name and links eval delta evidence

See [templates/contributions/track4_harness_pr.md](../templates/contributions/track4_harness_pr.md) for the PR template.
