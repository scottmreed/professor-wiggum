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
    from mechanistic_agent.tools import analyse_balance
    from mechanistic_agent.smiles_utils import sanitize_smiles_list

    # Attempt to sanitize SMILES before validation
    current_valid, current_invalid = sanitize_smiles_list(current_state)
    resulting_valid, resulting_invalid = sanitize_smiles_list(resulting_state)

    # If we have invalid SMILES that couldn't be recovered, fail with details
    invalid_smiles = current_invalid + resulting_invalid
    if invalid_smiles:
        return StepValidationCheck(
            name="atom_balance",
            passed=False,
            details={
                "error": f"Invalid SMILES strings: {invalid_smiles}",
                "sanitized_current": current_valid,
                "sanitized_resulting": resulting_valid,
            },
        )

    # If sanitization removed all SMILES, fail
    if not current_valid or not resulting_valid:
        return StepValidationCheck(
            name="atom_balance",
            passed=False,
            details={"error": "No valid SMILES remaining after sanitization"},
        )

    try:
        raw = analyse_balance(current_valid, resulting_valid)
        parsed = json.loads(raw)
        rdkit = parsed.get("rdkit", {}) if isinstance(parsed, dict) else {}
        balanced = bool(rdkit.get("balanced"))
        details: Dict[str, Any] = {
            "balanced": balanced,
            "deficit": rdkit.get("deficit", {}),
            "surplus": rdkit.get("surplus", {}),
        }
        # Add sanitization info if any changes were made
        if len(current_valid) != len(current_state) or len(resulting_valid) != len(resulting_state):
            details["sanitization_applied"] = True
            details["original_current_count"] = len(current_state)
            details["original_resulting_count"] = len(resulting_state)

        return StepValidationCheck(name="atom_balance", passed=balanced, details=details)
    except Exception as exc:  # pragma: no cover - defensive
        return StepValidationCheck(
            name="atom_balance",
            passed=False,
            details={"error": f"balance_check_failed: {exc}"},
        )
