---
skill_type: mechanistic
call_name: attempt_atom_mapping
steps: [atom_mapping, step_atom_mapping]
phase: pre_loop|post_step
kind: llm
tool_schema: atom_mapping_result
version: 1
---

# Attempt Atom Mapping

Proposes atom-to-atom correspondence between reactants and products (or between a
mechanism step's current and resulting states). Returns a mapping with confidence score
and explicit uncertainty for unmapped atoms.

Used in two contexts:
- **Pre-loop** (`atom_mapping`): maps across the full reaction (starting materials → products)
- **Post-step** (`step_atom_mapping`): maps a single accepted mechanism step (current_state → resulting_state)

Can be used standalone to analyze atom correspondences for any SMILES pair.

## Inputs

### atom_mapping (pre-loop)
- `starting_materials` (SMILES[])
- `products` (SMILES[])
- `functional_groups` (object, optional)

### step_atom_mapping (post-step)
- `current_state` (SMILES[]) — species before the step
- `resulting_state` (SMILES[]) — species after the step

## Outputs

- `mapped_atoms` (object) — atom-level correspondence map
- `compact_mapped_atoms` (string, step mapping) — compact lineage representation
- `unmapped_atoms` (list) — atoms that could not be confidently mapped
- `confidence` (float 0–1)
- `reasoning` (string)

## Tool Schema

```json
{
  "type": "function",
  "function": {
    "name": "atom_mapping_result",
    "description": "Return a best-effort atom mapping from reactants to products.",
    "parameters": {
      "type": "object",
      "properties": {
        "text": {"type": "string", "description": "Free-form reasoning."},
        "mapped_atoms": {"type": "object", "description": "Atom correspondence map."},
        "unmapped_atoms": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "description": "0–1 confidence score."},
        "reasoning": {"type": "string"}
      }
    }
  }
}
```

## Prompt

<!-- PROMPT_START -->
Produce a best-effort atom mapping from reactants to products with explicit uncertainty when needed.
Prioritize mappings that preserve plausible skeletal continuity and reactive center behavior.
Keep mapping outputs machine-readable and aligned with downstream mechanistic validation.
When mapping step-level transitions, favor compact lineage clues that identify where byproduct atoms likely originated.
<!-- PROMPT_END -->

## Standalone Usage

```python
from mechanistic_agent.core.subagents import MappingAgent
agent = MappingAgent()
# Results returned as StepResult list
```

## Notes

- Both `atom_mapping` and `step_atom_mapping` steps resolve to this call name.
- Step mapping (`step_atom_mapping`) is optional and gated by `step_mapping_enabled` flag.
- PRs updating this skill require approved evidence traces at medium eval tier minimum.
