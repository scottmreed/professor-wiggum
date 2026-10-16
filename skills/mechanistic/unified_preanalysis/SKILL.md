---
skill_type: mechanistic
call_name: unified_preanalysis
steps: [unified_preanalysis]
phase: pre_loop
kind: llm
tool_schema: unified_preanalysis_result
version: 1
---

# Unified Pre-Analysis

One call that replaces four pre-loop decisions in the `threaded_unified`
harness: reaction conditions (`assess_initial_conditions`), missing reagents
(`predict_missing_reagents`), atom mapping (`attempt_atom_mapping`) and the
reaction-type label (`select_reaction_type`; Jev is not used).

Each section of the answer has exactly the arguments of the old tool, and the
coordinator replays each section through that tool
(`llm.preset_tool_answers`), so parsing, normalisation, the conditions cache
and the species registry work as before and the old step names are still
recorded (`source: deterministic`, `derived_from: unified_preanalysis`).

## Inputs

- `starting_materials` (SMILES[]), `products` (SMILES[], may be hidden)
- Deterministic context: atom balance, functional groups, pH recommendation
- The reaction-type taxonomy (exact labels and type ids)

## Outputs (`unified_preanalysis_result`)

- `conditions` — `environment`, `representative_ph`, `ph_range`, `justification`, `acid_candidates` / `base_candidates`, `warnings`
- `missing_reagents` — `missing_reactants`, `missing_products`, `verification`, `notes` (omitted when products are hidden)
- `atom_mapping` — `mapped_atoms`, `unmapped_atoms`, `confidence`, `reasoning` (omitted when products are hidden)
- `reaction_type` — `selected_label_exact`, `selected_type_id`, `confidence`, `rationale`, `top_candidates`

## Checks

`mechanistic_agent/core/preanalysis_checks.py`: conditions parse and pH in
range; reagents leave no deficit (hard failure: the run pauses) and parse;
the rdkit-agent atom-map check passes; the label is in the taxonomy or
`no_match`. Any failure gets one repair turn in the same chat. When every
section passes, the mechanism loop continues this chat turn by turn.

## Prompt

<!-- PROMPT_START -->
You prepare a reaction for step-by-step mechanism prediction. In one answer, decide the reaction conditions, any reagents or byproducts needed to balance the reaction, an atom mapping from starting materials to products, and the best mechanism type from the taxonomy.
Make the sections consistent with each other: the reagents must fit the conditions, the mapping must account for the reagents you add, and the mechanism type must fit the conditions and the bonds that change.
Use valid, RDKit-parseable SMILES everywhere. Prefer conservative, well-precedented choices; say where you are unsure instead of guessing.
After this answer the same conversation continues with the mechanism steps, so keep the reasoning you will need there short and explicit.
<!-- PROMPT_END -->

## Notes

- Behaviour-changing: PRs that edit this skill need approved evidence traces.
