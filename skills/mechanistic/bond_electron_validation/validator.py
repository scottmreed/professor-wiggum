"""Bond/electron conservation validator.

Ground truth implementation for the bond_electron_validation mechanistic skill.
Consumes the runtime's ``bond_electron_validation`` payload, which continues to
use `dbe` bookkeeping internally even when the original proposal used explicit
mechanism-move notation.

Harness-specific overrides (e.g., always-soft mode) go in:
  harness_versions/<harness>/patches/bond_electron_validation.py
"""
from __future__ import annotations

from typing import Any, Dict

from mechanistic_agent.core.types import StepValidationCheck


def validate_bond_electron(
    payload: Dict[str, Any],
    *,
    dbe_policy: str = "strict",
) -> StepValidationCheck:
    """Validate bond/electron conservation from the payload's pre-computed result.

    The MechanismAgent pre-computes ``bond_electron_validation`` before this
    validator runs (via arrow_push.py). This function reads that result and
    applies the dbe policy.

    Parameters
    ----------
    payload:
        Step output dict, expected to contain a ``bond_electron_validation`` key
        with ``{"valid": bool, "total_delta": int, "message": str}``.
    dbe_policy:
        ``"strict"`` (default) — fail if dbe validation fails.
        ``"soft"`` — warn but pass if dbe validation fails.
    """
    bond_validation = payload.get("bond_electron_validation")
    if isinstance(bond_validation, dict):
        dbe_valid = bool(bond_validation.get("valid"))
        dbe_pass = dbe_valid
        if not dbe_valid and str(dbe_policy).lower() == "soft":
            dbe_pass = True
        return StepValidationCheck(
            name="dbe_metadata",
            passed=dbe_pass,
            details={
                "valid": dbe_valid,
                "policy": dbe_policy,
                "warning_only": (not dbe_valid and dbe_pass),
                "message": bond_validation.get("message"),
                "error": bond_validation.get("error"),
                "total_delta": bond_validation.get("total_delta"),
                "dbe": bond_validation.get("dbe"),
                "dbe_source": bond_validation.get("dbe_source"),
                "mech": bond_validation.get("mech"),
                "mech_warning": bond_validation.get("mech_warning"),
            },
        )
    return StepValidationCheck(
        name="dbe_metadata",
        passed=False,
        details={"error": "bond_electron_validation_missing"},
    )
