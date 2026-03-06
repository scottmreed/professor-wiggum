"""Deterministic validation checks for mechanism runtime steps.

This module is a thin dispatcher. Each validator's ground truth implementation
lives in skills/mechanistic/<validator_name>/validator.py and can be patched
per-harness via harness_versions/<harness>/patches/<validator_name>.py.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from skills.mechanistic.atom_balance_validation.validator import validate_atom_balance
from skills.mechanistic.bond_electron_validation.validator import validate_bond_electron
from skills.mechanistic.state_progress_validation.validator import validate_state_progress

from .types import StepValidationCheck, StepValidationResult

# Canonical validator IDs matching post_step_modules in harness config.
VALIDATOR_ATOM_BALANCE = "atom_balance_validation"
VALIDATOR_BOND_ELECTRON = "bond_electron_validation"
VALIDATOR_STATE_PROGRESS = "state_progress_validation"

ALL_VALIDATOR_IDS: Set[str] = {
    VALIDATOR_ATOM_BALANCE,
    VALIDATOR_BOND_ELECTRON,
    VALIDATOR_STATE_PROGRESS,
}


def validate_mechanism_step_output(
    payload: Dict[str, Any],
    *,
    dbe_policy: str = "strict",
    enabled_validators: Optional[Set[str]] = None,
) -> StepValidationResult:
    """Validate a ``predict_mechanistic_step`` style payload.

    Parameters
    ----------
    enabled_validators:
        Set of validator module IDs that are active.  When *None* (the
        default) all validators run, preserving backward compatibility.
        Pass a subset of :data:`ALL_VALIDATOR_IDS` to skip specific checks.
    """
    active = enabled_validators if enabled_validators is not None else ALL_VALIDATOR_IDS

    current_state = [str(item) for item in payload.get("current_state", [])]
    resulting_state = [str(item) for item in payload.get("resulting_state", [])]

    checks: List[StepValidationCheck] = []

    if VALIDATOR_ATOM_BALANCE in active:
        checks.append(validate_atom_balance(current_state, resulting_state))

    if VALIDATOR_BOND_ELECTRON in active:
        checks.append(validate_bond_electron(payload, dbe_policy=dbe_policy))

    if VALIDATOR_STATE_PROGRESS in active:
        checks.append(validate_state_progress(payload))

    return StepValidationResult(checks=checks)
