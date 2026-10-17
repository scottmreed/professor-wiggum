---
skill_type: mechanistic
call_name: atom_balance_validation
phase: post_step
kind: deterministic
implementation: validator.py
version: 1
---

# Check Step Atom Balance

Uses RDKit to verify that every atom in `current_state` appears in `resulting_state`
and that the net formal charge is unchanged. No atoms are created or destroyed and no
charge appears or vanishes across a valid elementary mechanism step.
No LLM call.

## Algorithm

1. Repair/canonicalise every species (`balance.assess_balance_diagnostics`).
2. Count every element in `current_state` and in `resulting_state`, including ALL
   hydrogens — explicit atoms such as a bare `[H+]`, `[H-]` or `[H][H]` and the
   implicit hydrogens on every heavy atom — so a proton that becomes bonded still
   balances while one that vanishes does not.
3. Sum the formal charges on each side.
4. Fail on any element deficit or surplus, or a net charge change (`charge_delta`);
   otherwise pass.

## Inputs

- `current_state` (SMILES[]) — species before the mechanism step
- `resulting_state` (SMILES[]) — species after the mechanism step

## Outputs

- `balanced` (bool)
- `deficit` (object) — `{element: count}` for atoms missing in resulting_state
- `surplus` (object) — `{element: count}` for atoms excess in resulting_state
- `current_charge`, `resulting_charge`, `charge_delta` (int) — net formal charges
- `balance_policy` — `all_atoms_with_hydrogens_and_net_charge`

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
- Counts through `mechanistic_agent/balance.py` (`assess_balance_diagnostics` with
  `include_hydrogens=True, include_charge=True`); the candidate rescue and the
  spectator carry use the same policy.
- The charge is part of this check, so `balance_mode` deferral treats a proton
  residual (H and charge together) as one residual.
- PRs updating `validator.py` require unit tests demonstrating the changed behavior.
