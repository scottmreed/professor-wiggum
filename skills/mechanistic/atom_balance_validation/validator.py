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
    from mechanistic_agent.smiles_utils import canonicalize_capture_error

    # Build per-SMILES error map using error-capturing function
    invalid_details = {}  # smiles → rdkit_error_string
    current_valid = []
    resulting_valid = []

    for smi in current_state:
        canon, err = canonicalize_capture_error(smi)
        if canon is not None:
            current_valid.append(canon)
        elif err is not None:
            invalid_details[smi] = err

    for smi in resulting_state:
        canon, err = canonicalize_capture_error(smi)
        if canon is not None:
            resulting_valid.append(canon)
        elif err is not None:
            invalid_details[smi] = err

    if invalid_details:
        return StepValidationCheck(
            name="atom_balance",
            passed=False,
            details={
                "error": f"Invalid SMILES: {list(invalid_details.keys())}",
                "rdkit_errors": invalid_details,  # NEW: per-SMILES diagnostic
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
