---
skill_type: mechanistic
call_name: state_progress_validation
phase: post_step
kind: deterministic
implementation: validator.py
version: 1
---

# Check State Changed

Verifies that a proposed mechanism step actually transforms the state: the resulting
species differ from the current species, and the starting materials have not simply been
returned unchanged. Prevents circular or no-op steps. No LLM call.

## Algorithm

1. Canonicalize SMILES sets for `current_state` and `resulting_state`.
2. Check `resulting_state_changed`: sets must differ.
3. Check `unchanged_starting_materials_detected`: resulting state must not be identical to the original starting materials.
4. Pass only if both conditions are met.

## Inputs

- `current_state` (SMILES[]) — species before the step
- `resulting_state` (SMILES[]) — species after the step

## Outputs

- `resulting_state_changed` (bool) — true if states differ
- `unchanged_starting_materials_detected` (bool) — true if starting materials were returned as-is

## Implementation

See [validator.py](validator.py) in this directory. This is the **ground truth** implementation.

Harness-specific modifications are stored as Python overrides in
`harness_versions/<harness>/patches/state_progress_validation.py`.

## Standalone Usage

```python
from skills.mechanistic.state_progress_validation.validator import validate_state_progress

result = validate_state_progress(
    current_state=["CCO"],
    resulting_state=["CC=O", "O"]
)
print(result.passed, result.details)
```

## Notes

- Re-runs on every retry attempt within the mechanism loop.
- Validator can be individually disabled via `removable: true` in harness config.
- A step that returns `current_state` unchanged will always fail this check.
- PRs updating `validator.py` require unit tests demonstrating the changed behavior.
