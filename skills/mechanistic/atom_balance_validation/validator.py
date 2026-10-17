"""Atom balance validator.

Ground truth implementation for the atom_balance_validation mechanistic skill.
Uses RDKit (via assess_balance_diagnostics) to verify that every atom in
current_state appears in resulting_state — no atoms created or destroyed — and
that the net formal charge is conserved.

Counting policy (``BALANCE_POLICY``): every element including ALL hydrogens,
explicit and implicit, on both sides, plus the net formal charge. A bare
``[H+]`` is an explicit H atom and becomes implicit once bonded; counting only
one kind made a correct protonation (``ketone + [H+] -> oxocarbenium``) fail as
``H: 1->0``. Counting both keeps that step balanced while a proton, hydride or
H2 that appears or vanishes still fails, as does a step that changes the net
charge. The charge is part of this check (not a separate one) so the balance
modes (``deferred`` / ``proton_deferred``) treat a proton residual — H and
charge together — exactly as before. The candidate rescue and the spectator
carry use the same policy. The rdkit-agent mechanism-step check counts heavy
atoms only; this Python check replaces its verdict.

Harness-specific overrides go in:
  harness_versions/<harness>/patches/atom_balance_validation.py
"""
from __future__ import annotations

from typing import Any, Dict, List

from mechanistic_agent.balance import assess_balance_diagnostics
from mechanistic_agent.core.types import StepValidationCheck

#: Counting policy shared by the validator, the candidate rescue and the spectator carry.
BALANCE_POLICY = "all_atoms_with_hydrogens_and_net_charge"
HYDROGEN_POLICY = "all_hydrogens"
INCLUDE_HYDROGENS = True
INCLUDE_CHARGE = True


def validate_atom_balance(
    current_state: List[str],
    resulting_state: List[str],
) -> StepValidationCheck:
    """Verify atom conservation between current and resulting states.

    Parameters
    ----------
    current_state:
        SMILES list of all species before the mechanism step.
    resulting_state:
        SMILES list of all species after the mechanism step.
    """
    try:
        diagnostics = assess_balance_diagnostics(
            current_state,
            resulting_state,
            include_hydrogens=INCLUDE_HYDROGENS,
            include_charge=INCLUDE_CHARGE,
            left_label="current_state",
            right_label="resulting_state",
        )
    except Exception as exc:  # pragma: no cover - defensive (e.g. a species that only parses with CXSMILES)
        return StepValidationCheck(
            name="atom_balance",
            passed=False,
            details={"balanced": False, "error": f"balance_check_failed: {exc}", "balance_policy": BALANCE_POLICY},
        )
    details: Dict[str, Any] = {
        "classification": diagnostics.get("classification"),
        "balanced": bool(diagnostics.get("balanced")),
        "deficit": diagnostics.get("deficit", {}),
        "surplus": diagnostics.get("surplus", {}),
        "invalid_species": diagnostics.get("invalid_species", []),
        "repaired_species": diagnostics.get("repaired_species", []),
        "sanitized_current": diagnostics.get("sanitized_left", []),
        "sanitized_resulting": diagnostics.get("sanitized_right", []),
        "hydrogen_policy": HYDROGEN_POLICY,
        "balance_policy": BALANCE_POLICY,
    }
    for key in ("left_charge", "right_charge", "charge_delta"):
        if key in diagnostics:
            details[{"left_charge": "current_charge", "right_charge": "resulting_charge"}.get(key, key)] = diagnostics[key]
    if diagnostics.get("left_counts"):
        details["current_counts"] = diagnostics.get("left_counts", {})
    if diagnostics.get("right_counts"):
        details["resulting_counts"] = diagnostics.get("right_counts", {})
    if diagnostics.get("error"):
        details["error"] = diagnostics.get("error")

    if diagnostics.get("classification") == "invalid_species":
        return StepValidationCheck(name="atom_balance", passed=False, details=details)

    # The diagnostics already counted both sides the same way (BALANCE_POLICY);
    # no second count with a different hydrogen policy may overrule them.
    return StepValidationCheck(name="atom_balance", passed=bool(details["balanced"]), details=details)
