---
skill_type: mechanistic
call_name: predict_missing_reagents
steps: [missing_reagents]
phase: pre_loop
kind: llm
tool_schema: missing_reagents_result
version: 1
---

# Predict Missing Reagents

Proposes missing reactants or byproducts needed to achieve atom balance under the
reaction conditions. Receives the full `conditions_guidance` JSON from
`assess_initial_conditions` and uses it to suggest stoichiometrically consistent additions.

Can operate in **step-rescue mode** when called during retry to fix balance gaps with
minimal additions that preserve the candidate mechanism.

## Inputs

- `starting_materials` (SMILES[])
- `products` (SMILES[])
- `conditions_guidance` (object) — full JSON output from `assess_initial_conditions`
  (includes `representative_ph`, `acid_candidates`/`base_candidates`)

## Outputs

- `missing_reactants` (SMILES[]) — species to add to the left side
- `missing_products` (SMILES[]) — species to add to the right side
- `verification` (string) — concise balance justification

## Tool Schema

```json
{
  "type": "function",
  "function": {
    "name": "missing_reagents_result",
    "description": "Return missing reactants and products needed to balance the reaction stoichiometry.",
    "parameters": {
      "type": "object",
      "properties": {
        "text": {"type": "string", "description": "Free-form reasoning."},
        "missing_reactants": {
          "type": "array",
          "items": {"type": "string"},
          "description": "SMILES strings for missing reactants."
        },
        "missing_products": {
          "type": "array",
          "items": {"type": "string"},
          "description": "SMILES strings for missing products."
        },
        "verification": {"type": "string", "description": "Balance justification."}
      }
    }
  }
}
```

## Prompt

<!-- PROMPT_START -->
Propose missing reactants or byproducts needed to achieve atom balance.
When used in step-rescue mode, prefer minimal additions that preserve the candidate mechanism while fixing balance gaps.
Use strict conservation logic and return only structured additions that improve stoichiometric consistency.
Avoid speculative additions that are unsupported by atom counts or reaction context.
Return molecule list entries as SMILES strings only, never natural-language descriptors.
SMILES format requirements:
- Use implicit-hydrogen SMILES: water is `O` (not `[H2O]`), ammonia is `N` (not `[NH3]`).
- Molecular formulas in brackets are NOT valid SMILES: `[H2SO4]` must be `OS(=O)(=O)O`.
- Common mappings: HCl=`Cl`, NaOH=`[Na+].[OH-]`, CO2=`O=C=O`, H2=`[HH]`.
- Each entry must be parseable by RDKit's MolFromSmiles function.
<!-- PROMPT_END -->

## Standalone Usage

```python
from mechanistic_agent.core.subagents import MissingReagentsAgent
agent = MissingReagentsAgent()
```

Or via API: include `"predict_missing_reagents"` in `optional_llm_tools`.

## Notes

- Optional module: gated by `optional_llm_tools: ["predict_missing_reagents"]`.
- Few-shot examples in `few_shot.jsonl` cover acidic, basic, and balanced cases.
- PRs updating this skill require approved evidence traces at medium eval tier minimum.
