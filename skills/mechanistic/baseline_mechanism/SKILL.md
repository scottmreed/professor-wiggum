---
skill_type: mechanistic
call_name: baseline_mechanism
steps: [baseline_mechanism]
phase: standalone
kind: llm
tool_schema: predict_full_mechanism
version: 2
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
- `reaction_smirks` (atom-mapped SMIRKS with a `|mech:v1;...|` arrow block)
- `reaction_smirks` and `electron_pushes` are required on every step (see the prompt); pushes use the shared
  `{kind, source_atom | source_bond, through_atom, target_atom, electrons}` move format
- `step_label` (string)

## Tool Schema

`PREDICT_FULL_MECHANISM_TOOL` in `mechanistic_agent/tool_schemas.py`: one `steps` array whose items require
`step_index`, `current_state`, `resulting_state`, `reaction_smirks` and `electron_pushes` (items of the shared
`_ELECTRON_PUSH_ITEM` move schema).

## Prompt

<!-- PROMPT_START -->
You are an expert organic chemist. Given a reaction (starting materials and target products), produce the complete stepwise mechanism in a single response using the `predict_full_mechanism` tool.

Each step must represent a single elementary mechanistic transformation (bond formation, bond breaking, proton transfer, electron pair movement, etc.). Do not skip steps or merge multiple elementary events.

For each step, provide ALL of the following. A step missing `reaction_smirks` or `electron_pushes` is scored as invalid.
- `current_state`: SMILES of every species present at the start of the step. It must equal the previous step's `resulting_state`: you may drop a spectator or add a reagent from the starting materials, but never add a species that was neither supplied nor made earlier, and never drop an intermediate.
- `resulting_state`: SMILES of every species after the step, including byproducts and counter-ions.
- `predicted_intermediate`: SMILES of the key intermediate generated or consumed (the focal species of the step).
- `reaction_smirks`: an atom-mapped reaction SMIRKS for this one elementary step, `reactants>>products`. Every heavy atom carries the same map number on both sides, and the two sides are exactly the whole species that react and form in this step. Append the arrow block `|mech:v1;lp:a>b;pi:a-b>c;sigma:a-b>c|` using those map numbers.
- `electron_pushes`: one object per curved arrow, using the SAME map numbers as the SMIRKS:
  - lone pair: `{"kind": "lone_pair", "source_atom": 4, "target_atom": 2, "electrons": 2}`
  - pi or sigma bond: `{"kind": "pi_bond", "source_bond": [1, 2], "through_atom": 2, "target_atom": 3, "electrons": 2}` (`through_atom` is required for bond sources and must equal `source_bond[1]`)
- `step_label`: a short label ("nucleophilic addition", "proton transfer", "ring closure").

Electrons must be conserved in every step: count bonds and lone pairs on both sides before you answer.

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
