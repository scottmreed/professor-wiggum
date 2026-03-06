---
skill_type: mechanistic
call_name: atom_balance_validation
phase: post_step
kind: deterministic
implementation: validator.py
version: 1
---

# Check Step Atom Balance

Uses RDKit to verify that every atom in `current_state` appears in `resulting_state`.
No atoms are created or destroyed across a valid elementary mechanism step.
No LLM call.

## Algorithm

1. Compute atom counts for each element in `current_state` using RDKit's `analyse_balance`.
2. Compute atom counts for each element in `resulting_state`.
3. Compare: if deficit or surplus atoms exist, fail.
4. If all elements balance exactly, pass.

## Inputs

- `current_state` (SMILES[]) — species before the mechanism step
- `resulting_state` (SMILES[]) — species after the mechanism step

## Outputs

- `balanced` (bool)
- `deficit` (object) — `{element: count}` for atoms missing in resulting_state
- `surplus` (object) — `{element: count}` for atoms excess in resulting_state

## Implementation

See [validator.py](validator.py) in this directory. This is the **ground truth** implementation.

Harness-specific modifications are stored as Python overrides in
`harness_versions/<harness>/patches/atom_balance_validation.py`.

## Standalone Usage

```python
from skills.mechanistic.atom_balance_validation.validator import validate_atom_balance

result = validate_atom_balance(
    current_state=["CCO"],
    resulting_state=["CC=O", "O"]
)
print(result.passed, result.details)
```

## Notes

- Re-runs on every retry attempt within the mechanism loop.
- Validator can be individually disabled via `removable: true` in harness config.
- Delegates to `analyse_balance()` from `mechanistic_agent/tools.py` for RDKit counting.
- PRs updating `validator.py` require unit tests demonstrating the changed behavior.
