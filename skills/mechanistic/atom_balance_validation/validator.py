"""Atom balance validator.

Ground truth implementation for the atom_balance_validation mechanistic skill.
Uses RDKit (via analyse_balance) to verify that every atom in current_state
appears in resulting_state — no atoms created or destroyed.

Harness-specific overrides go in:
  harness_versions/<harness>/patches/atom_balance_validation.py
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from mechanistic_agent.balance import assess_balance_diagnostics
from mechanistic_agent.core.types import StepValidationCheck


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
        include_hydrogens=False,
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
    }
    if diagnostics.get("left_counts"):
        details["current_counts"] = diagnostics.get("left_counts", {})
    if diagnostics.get("right_counts"):
        details["resulting_counts"] = diagnostics.get("right_counts", {})
    if diagnostics.get("error"):
        details["error"] = diagnostics.get("error")

    if diagnostics.get("classification") == "invalid_species":
        return StepValidationCheck(name="atom_balance", passed=False, details=details)

    try:
        from mechanistic_agent.tools import analyse_balance

        raw = analyse_balance(
            list(diagnostics.get("sanitized_left", [])),
            list(diagnostics.get("sanitized_right", [])),
        )
        parsed = json.loads(raw)
        rdkit = parsed.get("rdkit", {}) if isinstance(parsed, dict) else {}
        balanced = bool(rdkit.get("balanced", diagnostics.get("balanced")))
        details["balanced"] = balanced
        details["deficit"] = rdkit.get("deficit", diagnostics.get("deficit", {}))
        details["surplus"] = rdkit.get("surplus", diagnostics.get("surplus", {}))
        return StepValidationCheck(name="atom_balance", passed=balanced, details=details)
    except Exception as exc:  # pragma: no cover - defensive
        return StepValidationCheck(
            name="atom_balance",
            passed=False,
            details={**details, "error": f"balance_check_failed: {exc}"},
        )
