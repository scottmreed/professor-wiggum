---
skill_type: mechanistic
call_name: propose_mechanism_step
steps: [mechanism_step_proposal, mechanism_synthesis, intermediates]
phase: loop
kind: llm
tool_schema: propose_intermediates
version: 1
---

# Propose Mechanism Step

Core LLM skill of the mechanistic harness. Called each iteration of the mechanism loop
to propose ranked candidate intermediates for the next elementary step. Returns up to
`max_candidates` (default 3) ranked proposals that are then evaluated by deterministic
validators.

## Inputs

- `current_state` (SMILES[]) — species currently present
- `previous_intermediates` (object[]) — accepted steps so far
- `conditions_guidance` (object, optional) — from `assess_initial_conditions`
- `reaction_type` (string, optional) — from `reaction_type_mapping`
- `atom_mapping_context` (object, optional) — from `atom_mapping`
- `step_atom_mapping` (object, optional) — lineage from previous step

## Outputs

Ranked list of candidates, each with:
- `intermediate_smiles` (SMILES) — focal intermediate
- `resulting_state` (SMILES[]) — all species after applying the step
- `reaction_smirks` (SMIRKS with `|mech:v1;...|` block using explicit move sources)
- `electron_pushes` ([{kind, source_atom|source_bond, through_atom?, target_atom, electrons}])
- `step_label` (string)
- `rank` (int)
- `confidence` (float 0–1)

## Tool Schema

```json
{
  "type": "function",
  "function": {
    "name": "mechanism_step_proposal_result",
    "description": "Propose up to 3 ranked candidates for the next mechanism step.",
    "parameters": {
      "type": "object",
      "required": ["classification", "candidates", "analysis"],
      "properties": {
        "text": {"type": "string", "description": "Free-form reasoning."},
        "classification": {"type": "string", "enum": ["intermediate_step", "final_step"]},
        "analysis": {"type": "string"},
        "candidates": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["rank", "intermediate_smiles", "reaction_description", "reaction_smirks", "electron_pushes"],
            "properties": {
              "rank": {"type": "integer"},
              "intermediate_smiles": {"type": "string"},
              "reaction_description": {"type": "string"},
              "resulting_state": {"type": "array", "items": {"type": "string"}},
              "reaction_smirks": {
                "type": "string",
                "description": "SMIRKS with |mech:v1;lp:a>b;pi:a-b>c;sigma:a-b>c| block. REQUIRED."
              },
              "electron_pushes": {
                "type": "array",
                "description": "REQUIRED. At least one move. For lone_pair: {kind, source_atom, target_atom, electrons:2}. For pi_bond/sigma_bond: {kind, source_bond:[bondStart,bondEnd], through_atom:bondEnd, target_atom, electrons:2}. through_atom is REQUIRED for pi_bond and sigma_bond moves.",
                "items": {"type": "object", "properties": {
                  "kind": {"type": "string", "enum": ["lone_pair", "pi_bond", "sigma_bond"]},
                  "source_atom": {"type": "string"},
                  "source_bond": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
                  "through_atom": {"type": "string", "description": "REQUIRED for pi_bond/sigma_bond: must equal source_bond[1]."},
                  "target_atom": {"type": "string"},
                  "electrons": {"type": "integer", "enum": [2]}
                }}
              },
              "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
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
Propose the next forward mechanism candidate(s) from the current state.
Return ranked candidates with concise balanced transformation descriptions and realistic intermediates.
Do not repeat prior accepted intermediates unless explicitly justified by the current step state.
Every `intermediate_smiles` must be valid SMILES only (no prose labels, mechanism adjectives, or condition text).
If atom-lineage context from the previous step is provided, use it to reduce candidate breadth and avoid impossible atom transfers.
Each candidate must be executable by deterministic validators, so include:
- `reaction_smirks`: CXSMILES/SMIRKS with a `|mech:v1;lp:a>b;pi:a-b>c;sigma:a-b>c|` block.
- `electron_pushes`: explicit move objects — at least one per candidate:
  - `lone_pair` move: `{"kind": "lone_pair", "source_atom": "4", "target_atom": "2", "electrons": 2}`
  - `pi_bond`/`sigma_bond` move: `{"kind": "pi_bond", "source_bond": ["1", "2"], "through_atom": "2", "target_atom": "3", "electrons": 2}`
  - CRITICAL: `through_atom` is **required** for `pi_bond` and `sigma_bond` moves and must equal `source_bond[1]`.
  - All atom references must be numeric strings matching atom-map indices.
- `resulting_state`: optional but preferred explicit species list after applying the step, including minimal needed byproducts/reagents when balance would otherwise fail.

CRITICAL VALIDATION REQUIREMENTS:
- All SMILES strings in `intermediate_smiles`, `resulting_state`, and `reaction_smirks` must be parseable by RDKit
- Invalid SMILES will cause immediate rejection with no retries
- Common failure modes to avoid:
  * Excessive radical electrons (>3 per molecule)
  * Unclosed rings or invalid ring notation
  * Invalid atom symbols or molecular formulas in brackets
  * Malformed SMILES syntax

SMILES format requirements:
- Use RDKit-parseable SMILES notation: water=`O`, not `[H2O]`.
- Never place molecular formulas inside brackets: `[H2SO4]` is invalid, use `OS(=O)(=O)O`.
- Never return descriptors like `acid-catalyzed` or `protonated` in SMILES fields.
- If you cannot determine a valid SMILES for an intermediate, omit that candidate entirely.

When proposing candidates, validate each SMILES mentally:
- Can this string be parsed by a chemical structure parser?
- Does it represent a chemically reasonable intermediate?
- Are all atoms properly bonded and charged?
<!-- PROMPT_END -->

## Standalone Usage

```python
from mechanistic_agent.core.subagents import IntermediateAgent
agent = IntermediateAgent()
```

## Notes

- This is the only LLM call inside the mechanism loop; all other post-step modules are deterministic.
- `reaction_smirks` must include a `|mech:v1;...|` CXSMILES annotation for `bond_electron_validation`.
- Candidates are tried in rank order; failed candidates trigger retry or backtracking.
- Few-shot examples in `few_shot.jsonl` demonstrate correct `mech:v1` block format.
- PRs updating this skill require approved evidence traces at medium and hard eval tiers.
