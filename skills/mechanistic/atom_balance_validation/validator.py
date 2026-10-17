"""Atom balance validator.

Ground truth implementation for the atom_balance_validation mechanistic skill.
Uses RDKit (via assess_balance_diagnostics) to verify that every heavy atom in
current_state appears in resulting_state — no atoms created or destroyed.

Hydrogen policy: heavy atoms only, on both sides. A bare ``[H+]`` is an explicit
H atom but becomes implicit once bonded, so counting explicit hydrogens made a
correct protonation (``ketone + [H+] -> oxocarbenium``) fail as ``H: 1->0``.
The rdkit-agent mechanism-step check counts heavy atoms only as well, and the
candidate rescue uses the same policy (``HYDROGEN_POLICY``).

Harness-specific overrides go in:
  harness_versions/<harness>/patches/atom_balance_validation.py
"""
from __future__ import annotations

from typing import Any, Dict, List

from mechanistic_agent.balance import assess_balance_diagnostics
from mechanistic_agent.core.types import StepValidationCheck

#: Hydrogen accounting shared by the validator and the candidate rescue.
HYDROGEN_POLICY = "heavy_atoms_only"
INCLUDE_HYDROGENS = False


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
    diagnostics = assess_balance_diagnostics(
        current_state,
        resulting_state,
        include_hydrogens=INCLUDE_HYDROGENS,
        left_label="current_state",
        right_label="resulting_state",
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
    }
    if diagnostics.get("left_counts"):
        details["current_counts"] = diagnostics.get("left_counts", {})
    if diagnostics.get("right_counts"):
        details["resulting_counts"] = diagnostics.get("right_counts", {})
    if diagnostics.get("error"):
        details["error"] = diagnostics.get("error")

    if diagnostics.get("classification") == "invalid_species":
        return StepValidationCheck(name="atom_balance", passed=False, details=details)

    # The diagnostics already counted both sides the same way (heavy atoms only);
    # no second count with a different hydrogen policy may overrule them.
    return StepValidationCheck(name="atom_balance", passed=bool(details["balanced"]), details=details)
