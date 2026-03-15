# CLI Reference

This document describes the main command-line tools for the mechanistic agent. Each command can be run via `python main.py <command>` (or `mechanistic-agent <command>` if the package is installed).

---

## `run`

**Purpose:** Run a single mechanistic prediction through the full harness pipeline.

Use `run` when you want the full prediction workflow: the system will propose mechanism steps, validate each step with RDKit, optionally backtrack on failures, and continue until the target products are reached or a limit is hit. This is the primary way to execute one reaction from start to finish with all subagents (conditions, mapping, reaction type, step proposal, validators, reflection) and the configured harness. You can run in **verified** mode (you supply the steps; the runtime validates them) or **unverified** mode (the runtime proposes and validates steps automatically). Ralph orchestration is available for automated retries with different harness/topology/prompt variants.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--starting` | Comma-separated SMILES for starting materials | (defaults from config) |
| `--products` | Comma-separated SMILES for products | (defaults from config) |
| `--mode` | Run mode: `verified` or `unverified` | `unverified` |
| `--model-name` / `--model` | Exact model ID for all LLM-backed subagents | from config |
| `--thinking-level` | Thinking/reasoning depth: `low`, `high`, or `max` (model-dependent) | (none) |
| `--max-steps` | Maximum mechanism loop steps | `10` |
| `--max-runtime` | Maximum runtime in seconds | `600` |
| `--orchestration-mode` | `standard` or `ralph` | `standard` |
| `--harness` | Harness name from `harness_versions/` | `default` |
| `--harness-strategy` | Ralph harness strategy: `latest`, `portfolio`, or `mutate` | `latest` |
| `--harness-list` | Repeatable harness names for portfolio strategy | — |
| `--max-iterations` | Ralph outer-loop max iterations (0 = unlimited) | `0` |
| `--ralph-max-runtime` | Ralph outer-loop runtime cap (seconds) | `900` |
| `--max-cost-usd` | Ralph cumulative run budget cap in USD | `2.0` |
| `--repeat-failure-signature-limit` | Stop Ralph after same failure signature repeats N times | `2` |
| `--babysit` | Babysit mode: `off` or `advisory` | `off` |
| `--mutation-lane` | Optional Ralph mutation lane: `topology`, `harness`, `prompt`, `few_shot` | — |
| `--allow-validator-mutation` / `--no-allow-validator-mutation` | Allow Ralph to mutate validator modules | `true` |
| `--temperature` | Reaction temperature in Celsius | `25.0` |
| `--ph` | Observed reaction pH (optional) | — |
| `--functional-groups` / `--no-functional-groups` | Enable functional group analysis | `true` |
| `--intermediates` / `--no-intermediates` | Enable intermediate prediction | `true` |
| `--llm-tool` / `-T` | Repeatable optional LLM tools (allowed names in help) | — |
| `--show-events` | Print recorded run events | `false` |
| `--json` | Emit final summary as JSON | `false` |

**Examples:**

```bash
# Single reaction, unverified (default)
python main.py run --starting "CCO" --products "CC=O"

# Verified mode (steps submitted via API/UI)
python main.py run --mode verified --starting "CCO" --products "CC=O"

# With model and thinking level
python main.py run --model anthropic/claude-opus-4.6 --thinking-level high
```

---

## `serve`

**Purpose:** Start the FastAPI server and the web UI so you can run predictions in the browser.

Use `serve` when you want to work interactively: it launches the local API and serves the static UI. You can create runs, start them, submit mechanism steps in verified mode, and view events and flow from the browser. No API server is required for the `run` or `baseline` CLI commands; `serve` is for human-in-the-loop and API-based workflows.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--host` | Host for the FastAPI server | `127.0.0.1` |
| `--port` | Port for the FastAPI server | `8010` |
| `--reload` | Enable auto-reload when code changes (development) | `false` |

**Examples:**

```bash
# Default: http://127.0.0.1:8010
python main.py serve

# Custom host/port and reload for development
python main.py serve --host 0.0.0.0 --port 9000 --reload
```

---

## `eval`

**Purpose:** Run the full harness pipeline against a development eval set or tier and record results on the leaderboard.

Use `eval` when you want a leaderboard-visible harness run. In plain eval-set mode it behaves like the existing harness evaluator. In tier mode (`--tier`) it now uses the development leaderboard route planner, which checks the current leaderboard status for the exact `model + thinking` scope and proposes the next qualifying route.

Tier planner behavior:

- `--tier` loads the development policy from [training_data/development_leaderboard_policy.json](../training_data/development_leaderboard_policy.json)
- the policy decides which tier inventory view is active per tier
- interactive terminals prompt for route confirmation unless `--yes` is passed
- non-interactive runs auto-select the recommended route
- `--case-id` or `--leaderboard-route custom` bypass planner case selection
- `--all-tiers` is an explicit sweep and bypasses the planner

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--eval-set-id` | Eval set to run against. Ignored in tier / all-tier flows because tier mapping drives the eval set. | required |
| `--tier` | Tier name: `easy`, `medium`, or `hard` | — |
| `--all-tiers` | Run easy, medium, and hard as an explicit sweep | `false` |
| `--tier-map-path` | Tier → eval-set mapping JSON | `training_data/baseline_tier_eval_set_map.json` |
| `--tier-definitions-path` | Override tier inventory JSON for advanced/testing use | policy source paths |
| `--leaderboard-route` | Planner route: `auto`, `same`, `extend`, `next`, or `custom` | `auto` |
| `--leaderboard-status-only` | Print planner status and exit | `false` |
| `--yes` | Auto-confirm the recommended planner route in TTY mode | `false` |
| `--case-id` | Repeatable explicit case IDs; bypasses planner case selection | — |
| `--run-group-prefix` | Prefix for tier flows; single-tier defaults to `<prefix>_<selected-tier>` | `cli_eval_tier` |
| `--run-group` | Explicit run-group name | — |
| `--model-name` / `--model` | Model identifier | from config |
| `--thinking-level` | Thinking level: `low`, `high`, or `max` | — |
| `--harness` | Harness name from `harness_versions/` | `default` |
| `--max-cases` | Max cases per run | `25` |
| `--max-per-tier` | Max cases per tier in `--all-tiers` mode | — |
| `--max-steps` | Max mechanism steps per case | `10` |
| `--max-runtime` | Per-case timeout in seconds | `600` |
| `--allow-repeats` | Allow rerunning already-attempted cases in non-planner custom flows | `false` |
| `--json` | Emit JSON output | `false` |

**Examples:**

```bash
# Show the planner status for a development tier flow
python main.py eval \
  --eval-set-id ignored \
  --tier easy \
  --model anthropic/claude-opus-4.6 \
  --thinking-level high \
  --leaderboard-status-only

# Run the recommended route
python main.py eval \
  --eval-set-id ignored \
  --tier easy \
  --model anthropic/claude-opus-4.6 \
  --thinking-level high

# Force the extend route
python main.py eval \
  --eval-set-id ignored \
  --tier easy \
  --model anthropic/claude-opus-4.6 \
  --thinking-level high \
  --leaderboard-route extend

# Bypass planner selection with explicit cases
python main.py eval \
  --eval-set-id ignored \
  --tier easy \
  --model anthropic/claude-opus-4.6 \
  --thinking-level high \
  --leaderboard-route custom \
  --case-id flower_135501 \
  --case-id flower_024300
```

See [Development leaderboard routes](development_leaderboard_routes.md) for the route rules, active tier sources, and current policy semantics.

---

## `baseline`

**Purpose:** Run harness-free, single-shot baseline mechanism predictions for evaluation or quick checks.

Use `baseline` when you want to measure model performance without the full harness (no pre-loop subagents, no step-by-step validation loop). The model produces one shot at the full mechanism; results are scored and can be recorded for the leaderboard. Useful for comparing models, building leaderboard rows without starting the API server, or running a single reaction quickly. Three modes: **(1)** single case (`--starting` + `--products`), **(2)** full eval set (`--eval-set-id`), or **(3)** tiered eval (`--tier` or `--all-tiers` using a tier map).

| Parameter | Description | Default |
|-----------|-------------|---------|
| **Single case** | | |
| `--starting` | Comma-separated SMILES for starting materials | (defaults) |
| `--products` | Comma-separated SMILES for products | (defaults) |
| **Eval set** | | |
| `--eval-set-id` | Run baseline against all cases in this eval set | — |
| **Tiers** | | |
| `--tier` | Repeatable tier: `easy`, `medium`, or `hard` (uses tier map) | — |
| `--all-tiers` | Run easy, medium, and hard tiers in one command | `false` |
| `--tier-map-path` | Path to tier → eval-set mapping JSON | `training_data/baseline_tier_eval_set_map.json` |
| `--tier-definitions-path` | Path to tier case-id definitions JSON | `baseline_tiers_clawdiator.json` or `eval_tiers.json` |
| `--run-group-prefix` | Run-group prefix; tier mode uses `<prefix>_<tier>` | `harness_free_baseline` |
| **Model & sampling** | | |
| `--model-name` / `--model` | Model identifier (e.g. gpt-5.4, claude-opus-4.6) | from config |
| `--thinking-level` | Thinking level: `low`, `high`, or `max` | — |
| `--temperature` | Reaction temperature in Celsius | `25.0` |
| `--ph` | Observed reaction pH (optional) | — |
| `--max-cases` | Max cases when running an eval set or tier | `25` |
| `--timeout` | Per-case timeout in seconds | `180` |
| `--llm-seed` | Seed for LLM randomness (when supported) | `42` |
| `--llm-temperature` | Sampling temperature when using fixed policy | `0.0` |
| `--sampling-policy` | `fixed` or `provider_default` | `fixed` |
| `--allow-repeats` | Allow rerunning cases already attempted for this model/thinking | `false` |
| **Output** | | |
| `--json` | Emit results as JSON | `false` |

**Examples:**

```bash
# Single reaction
python main.py baseline --starting "CCO" --products "CC=O"

# All tiers (easy + medium + hard) for leaderboard
python main.py baseline --all-tiers --model anthropic/claude-opus-4.6 --thinking-level high

# One tier and a specific eval set mapping
python main.py baseline --tier medium --eval-set-id my_eval_set

# Eval set with JSON output
python main.py baseline --eval-set-id eval_set --model openai/gpt-5.4 --json
```

---

## See also

- [Custom eval sets](custom_eval_sets.md) — defining and importing eval sets.
- [Development leaderboard routes](development_leaderboard_routes.md) — policy-driven tier access for `main.py eval`.
- [AGENTS.md](../AGENTS.md) — runtime architecture, endpoints, and leaderboard/holdout behavior.
- [CONTRIBUTING.md](../CONTRIBUTING.md) — contribution tracks and eval tier requirements.
