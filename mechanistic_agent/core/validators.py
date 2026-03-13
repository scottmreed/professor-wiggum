"""Deterministic validation checks for mechanism runtime steps.

This module is a thin dispatcher. Each validator's ground truth implementation
lives in skills/mechanistic/<validator_name>/validator.py and can be patched
per-harness via harness_versions/<harness>/patches/<validator_name>.py.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from skills.mechanistic.atom_balance_validation.validator import validate_atom_balance
from skills.mechanistic.bond_electron_validation.validator import validate_bond_electron
from skills.mechanistic.state_progress_validation.validator import validate_state_progress

from .chemistry_backend import execute_chemistry_check
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


_CHECK_TO_VALIDATOR: Dict[str, str] = {
    "atom_balance": VALIDATOR_ATOM_BALANCE,
    "dbe_metadata": VALIDATOR_BOND_ELECTRON,
    "state_progress": VALIDATOR_STATE_PROGRESS,
}


def _run_python_validators(
    payload: Dict[str, Any],
    *,
    dbe_policy: str,
    active: Set[str],
) -> StepValidationResult:
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


def _build_cli_payload(payload: Dict[str, Any], *, dbe_policy: str) -> Dict[str, Any]:
    return {
        "mechanism-step": True,
        "current_state": [str(item) for item in payload.get("current_state", [])],
        "resulting_state": [str(item) for item in payload.get("resulting_state", [])],
        "unchanged_starting_materials_detected": bool(payload.get("unchanged_starting_materials_detected")),
        "resulting_state_changed": bool(payload.get("resulting_state_changed")),
        "reaction_smirks": str(payload.get("reaction_smirks") or ""),
        "dbe": str(payload.get("dbe") or ""),
        "bond_electron_validation": payload.get("bond_electron_validation"),
        "strict": str(dbe_policy).lower() == "strict",
    }


def _cli_output_to_validation(cli_output: Dict[str, Any]) -> StepValidationResult:
    checks: List[StepValidationCheck] = []
    fix_suggestions = [
        str(item).strip()
        for item in (cli_output.get("fix_suggestions") or [])
        if isinstance(item, str) and item.strip()
    ]
    for raw in cli_output.get("checks") or []:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        if not name:
            continue
        passed = bool(raw.get("pass"))
        details = dict(raw.get("details") or {})
        error_code = raw.get("error_code")
        if isinstance(error_code, str) and error_code.strip():
            details.setdefault("error_code", error_code.strip())
        message = raw.get("message")
        if isinstance(message, str) and message.strip():
            details.setdefault("message", message.strip())
        if fix_suggestions:
            details.setdefault("fix_suggestions", fix_suggestions[:5])
        checks.append(StepValidationCheck(name=name, passed=passed, details=details))
    return StepValidationResult(checks=checks)


def _signature_from_validation(
    result: StepValidationResult,
    active: Set[str],
) -> Tuple[Tuple[str, bool], ...]:
    pairs: List[Tuple[str, bool]] = []
    for check in result.checks:
        validator_id = _CHECK_TO_VALIDATOR.get(check.name)
        if validator_id not in active:
            continue
        pairs.append((check.name, bool(check.passed)))
    pairs.sort(key=lambda item: item[0])
    return tuple(pairs)


def _signature_from_cli_output(
    cli_output: Dict[str, Any],
    active: Set[str],
) -> Tuple[Tuple[str, bool], ...]:
    pairs: List[Tuple[str, bool]] = []
    for raw in cli_output.get("checks") or []:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        validator_id = _CHECK_TO_VALIDATOR.get(name)
        if validator_id not in active:
            continue
        pairs.append((name, bool(raw.get("pass"))))
    pairs.sort(key=lambda item: item[0])
    return tuple(pairs)


def _filter_active_checks(result: StepValidationResult, active: Set[str]) -> StepValidationResult:
    filtered: List[StepValidationCheck] = []
    for check in result.checks:
        validator_id = _CHECK_TO_VALIDATOR.get(check.name)
        if validator_id not in active:
            continue
        filtered.append(check)
    return StepValidationResult(checks=filtered)


def _attach_backend_metadata(result: StepValidationResult, metadata: Dict[str, Any]) -> None:
    backend_meta = {
        "backend_requested": metadata.get("backend_requested"),
        "backend_used": metadata.get("backend_used"),
        "fallback_used": bool(metadata.get("fallback_used")),
        "fallback_reason": metadata.get("fallback_reason"),
        "rdkit_cli_available": bool(metadata.get("rdkit_cli_available")),
        "rdkit_cli_command": metadata.get("rdkit_cli_command"),
        "rdkit_cli_error_code": metadata.get("rdkit_cli_error_code"),
        "rdkit_cli_error": metadata.get("rdkit_cli_error"),
        "rdkit_cli_failed_check_names": list(metadata.get("rdkit_cli_failed_check_names") or []),
        "parity": metadata.get("parity"),
    }
    for check in result.checks:
        details = dict(check.details or {})
        details["chemistry_backend"] = backend_meta
        check.details = details


def _annotate_soft_backend_warnings(result: StepValidationResult) -> None:
    for check in result.checks:
        if check.name != "atom_balance" or not check.passed:
            continue
        details = dict(check.details or {})
        backend_meta = details.get("chemistry_backend")
        if not isinstance(backend_meta, dict):
            continue
        error_code = str(backend_meta.get("rdkit_cli_error_code") or "").strip()
        if error_code != "atom_balance_invalid_species":
            continue
        warnings = details.get("warnings")
        warning_lines = [str(item).strip() for item in warnings if str(item).strip()] if isinstance(warnings, list) else []
        warning_text = (
            "rdkit_cli reported invalid species; treating as a known soft pass "
            "because Python atom-balance validation succeeded."
        )
        if warning_text not in warning_lines:
            warning_lines.append(warning_text)
        details["warnings"] = warning_lines
        details["warning_only"] = True
        details["known_soft_pass"] = True
        details["soft_pass_reason"] = "rdkit_cli_invalid_species"
        details["retry_recommended"] = True
        check.details = details


def _override_atom_balance_check(
    result: StepValidationResult,
    *,
    payload: Dict[str, Any],
    active: Set[str],
) -> StepValidationResult:
    if VALIDATOR_ATOM_BALANCE not in active:
        return result
    current_state = [str(item) for item in payload.get("current_state", [])]
    resulting_state = [str(item) for item in payload.get("resulting_state", [])]
    atom_check = validate_atom_balance(current_state, resulting_state)

    updated: List[StepValidationCheck] = []
    replaced = False
    for check in result.checks:
        if check.name == "atom_balance":
            updated.append(atom_check)
            replaced = True
        else:
            updated.append(check)
    if not replaced:
        updated.append(atom_check)
    return StepValidationResult(checks=updated)


def validate_mechanism_step_output(
    payload: Dict[str, Any],
    *,
    dbe_policy: str = "strict",
    enabled_validators: Optional[Set[str]] = None,
    run_config: Any = None,
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

    def _python_path() -> StepValidationResult:
        return _run_python_validators(payload, dbe_policy=dbe_policy, active=active)

    cli_payload = _build_cli_payload(payload, dbe_policy=dbe_policy)
    result, backend_meta = execute_chemistry_check(
        mode="mechanism_step",
        payload=cli_payload,
        config=run_config,
        python_callable=_python_path,
        cli_to_result=_cli_output_to_validation,
        python_signature=lambda py: _signature_from_validation(py, active),
        cli_signature=lambda cli: _signature_from_cli_output(cli, active),
    )

    if not isinstance(result, StepValidationResult):
        result = _python_path()
    result = _filter_active_checks(result, active)
    result = _override_atom_balance_check(result, payload=payload, active=active)
    _attach_backend_metadata(result, backend_meta)
    _annotate_soft_backend_warnings(result)
    return result
