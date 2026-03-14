# Creating Custom Eval Sets Without FlowER Data

This guide explains how to define and run your own evaluation sets using your own reaction data, without relying on FlowER-derived artifacts.

## Why use a custom eval set?

- **Lab-specific chemistry**: Benchmark on reactions that matter for your project.
- **New benchmarks**: Add cases that are not in the default FlowER-derived `eval_set.json`.
- **Reproducibility**: Version your own JSON file and import it wherever the app runs.

The default eval set is built from `training_data/flower_mechanisms_100.json` and related FlowER artifacts. Custom eval sets are independent of that pipeline.

## Minimal JSON format

Each eval case is one object in a JSON array. Required fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique case ID (e.g. `my_ester_01`) |
| `starting_materials` | array of strings | SMILES for reactants |
| `products` | array of strings | SMILES for expected products |

Optional fields: `name`, `description`, `temperature_celsius`, `ph`, `run_config`, `verified_mechanism`, `notes`, `tags`.

Example:

```json
[
  {
    "id": "ester_01",
    "name": "Esterification example",
    "starting_materials": ["CCO", "CC(=O)O"],
    "products": ["CCOC(=O)C", "O"]
  },
  {
    "id": "sn2_01",
    "starting_materials": ["CCO", "ClCCCl"],
    "products": ["CCOCCCl", "Cl"]
  }
]
```

## How to author a new eval set

1. **Start from the template** (optional): Copy `training_data/my_reactions_template.json` and replace the single example with your cases. Remove any `_comment` fields before importing.

2. **Use valid SMILES**: Each string in `starting_materials` and `products` should be a valid SMILES. With `auto_convert: true` at import time, the API can try to convert names/InChI to SMILES.

3. **Save your file**: Place the JSON anywhere you can read from (e.g. `training_data/my_project.json`). The app does not auto-discover files in `training_data/` for eval sets; you register the set by importing (see below).

## How to register and run the eval set

### Option A: Import via API

1. Start the server: `python main.py serve`
2. POST the template payload with your cases:

```bash
curl -X POST http://127.0.0.1:8010/api/eval_sets/import_template \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_custom_set",
    "version": "v1",
    "cases": [
      {"id": "case_1", "starting_materials": ["CCO"], "products": ["CC=O"]}
    ],
    "auto_convert": true
  }'
```

3. The response includes `eval_set_id` (a UUID). Use that ID for running evals.

### Option B: Import from a file (same API)

Read your JSON file and send the `cases` array in the payload:

```bash
# With jq: send the array as the "cases" field
curl -X POST http://127.0.0.1:8010/api/eval_sets/import_template \
  -H "Content-Type: application/json" \
  -d "{\"name\": \"my_custom_set\", \"version\": \"v1\", \"cases\": $(cat training_data/my_project.json), \"auto_convert\": true}"
```

### Run the eval set

**CLI (harness eval):**

```bash
source .venv/bin/activate
python main.py eval --eval-set-id <eval_set_id> --model openai/gpt-4o
```

Use the `eval_set_id` returned from the import (e.g. the UUID in the response).

**CLI (baseline, no harness):**

```bash
python main.py baseline --eval-set-id <eval_set_id> --model anthropic/claude-opus-4.6
```

**API:**

```bash
curl -X POST http://127.0.0.1:8010/api/evals/runset \
  -H "Content-Type: application/json" \
  -d '{"eval_set_id": "<eval_set_id>", "model_name": "openai/gpt-4o"}'
```

After import, the eval set appears in the UI’s eval-set list and in the leaderboard dropdown (for non-holdout sets).

## Per-case options

In each case object you can include:

- **`run_config`**: Overrides for this reaction (e.g. `max_steps`, `ph`, `temperature_celsius`). Keys must match run config; omit `_comment`-style keys.
- **`verified_mechanism`**: Optional ground-truth mechanism (steps) for comparison during eval. Same shape as the verified step submission payload.

## Tiered evals (easy / medium / hard)

The fixed tier definitions (10 easy, 10 medium, 10 hard) and tier-to-eval-set mapping are defined in FlowER-derived artifacts (`training_data/eval_tiers.json`, `training_data/baseline_tier_eval_set_map.json`). Custom eval sets are registered as general-purpose sets; they are not automatically mapped into easy/medium/hard. To use tiers with custom data, you would need to add or adjust the tier mapping for your eval set (outside this guide).

## Summary

1. Create a JSON array of cases with `id`, `starting_materials`, and `products`.
2. Import via `POST /api/eval_sets/import_template` with `name`, `version`, and `cases`.
3. Use the returned `eval_set_id` in `python main.py eval --eval-set-id <id>` or `POST /api/evals/runset`.

No FlowER data is required; the only requirement is valid reaction SMILES and a running server for import.
