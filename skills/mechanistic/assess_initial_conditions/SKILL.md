---
skill_type: mechanistic
call_name: assess_initial_conditions
steps: [initial_conditions]
phase: pre_loop
kind: llm
tool_schema: assess_conditions_result
version: 1
---

# Assess Initial Conditions

Evaluates the reaction environment (acidic/basic/neutral), estimates pH, and suggests
compatible acid or base candidates from starting materials and products. Runs once in
the pre-loop phase; output is forwarded as `conditions_guidance` to downstream skills.

Can be used standalone to analyze reaction conditions for any chemical transformation
without running the full harness.

## Inputs

- `starting_materials` (SMILES[]) — reactant species
- `products` (SMILES[]) — target product species
- `ph` (float, optional) — user-supplied or heuristic pH from `ph_recommendation`
- `functional_groups` (object, optional) — output of `functional_groups` module if enabled

## Outputs

- `environment`: `"acidic"` | `"basic"` | `"neutral"`
- `representative_ph` (float, 0–14)
- `ph_range` ([float, float], optional)
- `justification` (string, ≤12 words)
- `acid_candidates` (object[], acidic only) — up to 3 entries with `name`, `smiles`, `role`, `justification`
- `base_candidates` (object[], basic only) — up to 3 entries with `name`, `smiles`, `role`, `justification`
- `warnings` (string[], optional)

## Tool Schema

```json
{
  "type": "function",
  "function": {
    "name": "assess_conditions_result",
    "description": "Return the assessed reaction conditions including environment, pH, and compatible acid or base candidates.",
    "parameters": {
      "type": "object",
      "required": ["environment", "representative_ph"],
      "properties": {
        "text": {"type": "string", "description": "Free-form reasoning or commentary."},
        "environment": {"type": "string", "enum": ["acidic", "basic", "neutral"]},
        "representative_ph": {"type": "number", "description": "Estimated pH, 0–14."},
        "ph_range": {"type": "array", "items": {"type": "number"}, "description": "[lower, upper] pH range."},
        "justification": {"type": "string", "description": "12 words or fewer summarising reasoning."},
        "acid_candidates": {
          "type": "array",
          "items": {"type": "object", "required": ["name"], "properties": {
            "name": {"type": "string"}, "smiles": {"type": "string"},
            "role": {"type": "string"}, "justification": {"type": "string"}
          }},
          "description": "Up to 3 acid reagent suggestions. Only populate when environment is acidic."
        },
        "base_candidates": {
          "type": "array",
          "items": {"type": "object", "required": ["name"], "properties": {
            "name": {"type": "string"}, "smiles": {"type": "string"},
            "role": {"type": "string"}, "justification": {"type": "string"}
          }},
          "description": "Up to 3 base reagent suggestions. Only populate when environment is basic."
        },
        "warnings": {"type": "array", "items": {"type": "string"}}
      }
    }
  }
}
```

## Prompt

<!-- PROMPT_START -->
Assess the reaction environment using the supplied reactant and product context.
Decide whether conditions should be acidic, basic, or neutral, and provide compact structured reasoning.
Prefer conservative suggestions that maintain compatibility with the dominant functional groups.
<!-- PROMPT_END -->

## Standalone Usage

```python
from mechanistic_agent.core.subagents import ConditionsAgent
from mechanistic_agent.core.types import RunState, RunInput

state = RunState(run_input=RunInput(starting_materials=["CCO"], products=["CC=O"]))
agent = ConditionsAgent()
results = agent.run(state)
```

Or via API: `POST /api/run` with `optional_llm_tools: ["assess_initial_conditions"]`

## Notes

- Output `conditions_guidance` (full JSON) is passed to `predict_missing_reagents` as context.
- Few-shot examples in `few_shot.jsonl` show acidic, basic, and neutral cases.
- PRs updating this skill require approved evidence traces at medium eval tier minimum.
