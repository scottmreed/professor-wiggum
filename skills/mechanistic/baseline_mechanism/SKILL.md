---
skill_type: mechanistic
call_name: baseline_mechanism
steps: [baseline_mechanism]
phase: standalone
kind: llm
tool_schema: predict_full_mechanism
version: 1
---

# Baseline Mechanism

Single-shot, harness-free mechanism prediction. Generates a complete stepwise mechanism
in one LLM call without any pre-loop analysis, validators, or iterative refinement.

Used as a control/baseline to measure raw LLM performance without harness scaffolding.
Scored identically to harness runs for direct comparison on the leaderboard
(appears with `is_baseline=true`).

## Inputs

- `starting_materials` (SMILES[]) — reactant species
- `products` (SMILES[]) — target product species
- `model` (string) — model identifier

## Outputs

A single `predict_full_mechanism` tool call containing a `steps` array where each step has:
- `current_state` (SMILES[]) — species at step start
- `resulting_state` (SMILES[]) — species after step
- `predicted_intermediate` (SMILES) — focal species of the step
- `reaction_smirks` (SMIRKS string)
- `electron_pushes` ([{start_atom, end_atom, electrons}])
- `step_label` (string)

## Tool Schema

```json
{
  "type": "function",
  "function": {
    "name": "predict_full_mechanism",
    "description": "Return the complete stepwise mechanism as a single structured response.",
    "parameters": {
      "type": "object",
      "required": ["steps"],
      "properties": {
        "steps": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["current_state", "resulting_state", "predicted_intermediate", "reaction_smirks"],
            "properties": {
              "current_state": {"type": "array", "items": {"type": "string"}},
              "resulting_state": {"type": "array", "items": {"type": "string"}},
              "predicted_intermediate": {"type": "string"},
              "reaction_smirks": {"type": "string"},
              "electron_pushes": {
                "type": "array",
                "items": {"type": "object", "properties": {
                  "start_atom": {"type": "integer"},
                  "end_atom": {"type": "integer"},
                  "electrons": {"type": "integer"}
                }}
              },
              "step_label": {"type": "string"}
            }
          }
        }
      }
    }
  }
}
```

## Prompt

<!-- PROMPT_START -->
You are an expert organic chemist. Given a reaction (starting materials and target products), produce the complete stepwise mechanism in a single response using the `predict_full_mechanism` tool.

Each step must represent a single elementary mechanistic transformation (bond formation, bond breaking, proton transfer, electron pair movement, etc.). Do not skip steps or merge multiple elementary events.

For each step, provide:
- `current_state`: list of SMILES for all species present at the start of this step
- `resulting_state`: list of SMILES for all species present after this step (including byproducts and spectator ions where relevant)
- `predicted_intermediate`: SMILES for the key intermediate generated or consumed (the focal species of the step)
- `reaction_smirks`: SMIRKS/CXSMILES for the elementary transformation; use atom map numbers when meaningful
- `electron_pushes`: list of `{start_atom, end_atom, electrons}` arrow-push descriptors (electrons: 1 for radical, 2 for pair)
- `step_label`: brief human-readable label (e.g., "nucleophilic addition", "proton transfer", "ring closure")

The final step's `resulting_state` must contain the target product SMILES.

SMILES rules (same as harness):
- Use RDKit-parseable notation: water=`O`, not `[H2O]`; acetic acid=`CC(O)=O`
- Abbreviations invalid: EtOH→`CCO`, MeOH→`CO`
- Charges in brackets: `[OH-]`, `[NH4+]`, `[O-]`
- Never place natural-language descriptors in SMILES fields
- Omit steps where no valid SMILES can be determined

Call `predict_full_mechanism` exactly once with the complete steps array.
<!-- PROMPT_END -->

## Standalone Usage

```bash
python main.py baseline --starting "CC(=O)Cl" --products "CC(=O)OCC" --model gpt-4o
```

Or for a full eval set:
```bash
python main.py baseline --eval-set-id <id> --model gpt-4o
```

## Notes

- No few-shot examples for this skill (single-shot by design).
- Scored via `score_baseline_result()` using the same rubric as harness runs.
- Results recorded with `run_group_name = "harness_free_baseline"` in the leaderboard.
