---
skill_type: mechanistic
call_name: bond_electron_validation
phase: post_step
kind: deterministic
implementation: validator.py
version: 1
---

# Check Bond/Electron Conservation

Parses the explicit mechanism-move metadata annotation from a reaction SMIRKS
and verifies that the implied bond changes match the actual reaction step.
Ensures proposed mechanism steps obey arrow-pushing bookkeeping without any LLM call.

## Algorithm

1. Extract the `|mech:v1;...|` CXSMILES annotation from `reaction_smirks`.
2. Parse each explicit move token such as `lp:4>2` or `sigma:2-3>3`.
3. Convert the moves into implied bond-order deltas.
4. Compare those deltas against the actual bond changes in the reaction SMIRKS.
5. If they mismatch and `dbe_policy == "strict"`: fail. If `dbe_policy == "soft"`: warn and pass.

## Inputs

- `reaction_smirks` (string) — SMIRKS with `|mech:v1;...|` metadata block
- `dbe_policy`: `"strict"` (default) | `"soft"`

## Outputs

- `valid` (bool) — whether electron conservation holds
- `expected_bond_deltas` / `observed_bond_deltas` — implied vs observed bond-order changes
- `message` (string) — human-readable summary

## Implementation

See [validator.py](validator.py) in this directory. This is the **ground truth** implementation.

Harness-specific modifications (e.g., forcing soft-fail mode for a particular harness)
are stored as Python overrides in `harness_versions/<harness>/patches/bond_electron_validation.py`.

## Standalone Usage

```python
from skills.mechanistic.bond_electron_validation.validator import validate_bond_electron

result = validate_bond_electron(
    payload={"bond_electron_validation": {"valid": True}},
    dbe_policy="strict"
)
print(result.passed, result.details)
```

## Notes

- Re-runs on every retry attempt within the mechanism loop.
- Validator can be individually disabled via `removable: true` in harness config.
- `dbe_policy: "soft"` converts move-validation failures to warnings (useful for exploratory harnesses).
- PRs updating `validator.py` require unit tests demonstrating the changed behavior.
