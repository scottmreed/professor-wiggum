"""Deterministic checks on the replayed unified pre-analysis sections.

Each section is ``pass``, ``soft_fail`` (repairable; the run can continue)
or ``hard_fail`` (the run cannot continue: the reagents leave a product-side
deficit, the same condition that pauses the old missing-reagents step).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

PASS = "pass"
SOFT_FAIL = "soft_fail"
HARD_FAIL = "hard_fail"


def _result(status: str, errors: Optional[List[str]] = None) -> Dict[str, Any]:
    return {"status": status, "errors": list(errors or [])}


def _smiles_parses(smiles: str) -> bool:
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return True


def check_conditions(output: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(output, dict):
        return _result(SOFT_FAIL, ["conditions section missing"])
    errors: List[str] = []
    if output.get("status") != "success":
        errors.append(str(output.get("error") or output.get("parse_error") or "conditions could not be parsed"))
    if (output.get("schema_validation") or {}).get("status") not in (None, "ok"):
        errors.append(f"schema: {(output.get('schema_validation') or {}).get('error')}")
    if output.get("environment") in (None, "", "undetermined"):
        errors.append("environment must be acidic, basic or neutral")
    ph = output.get("representative_ph")
    if not isinstance(ph, (int, float)) or not 0.0 <= float(ph) <= 14.0:
        errors.append(f"representative_ph must be a number in 0-14 (got {ph!r})")
    for key in ("acid_candidates", "base_candidates"):
        for item in output.get(key) or []:
            smiles = str((item or {}).get("smiles") or "").strip() if isinstance(item, dict) else ""
            if smiles and not _smiles_parses(smiles):
                errors.append(f"{key} SMILES does not parse: {smiles}")
    return _result(SOFT_FAIL if errors else PASS, errors)


def check_missing_reagents(output: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(output, dict):
        return _result(SOFT_FAIL, ["missing_reagents section missing"])
    status = output.get("status")
    if status in ("success", "balanced"):
        return _result(PASS)
    errors = [str(output.get("error") or output.get("message") or f"status {status}")]
    for item in output.get("invalid_reagents") or []:
        errors.append(f"invalid reagent: {item}")
    issues = output.get("balance_issues") or {}
    if issues.get("remaining_deficit"):
        errors.append(f"remaining deficit (atoms the products need): {issues['remaining_deficit']}")
    if issues.get("remaining_surplus"):
        errors.append(f"remaining surplus: {issues['remaining_surplus']}")
    return _result(HARD_FAIL if output.get("should_abort_mechanism") else SOFT_FAIL, errors)


def check_atom_mapping(output: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(output, dict):
        return _result(SOFT_FAIL, ["atom_mapping section missing"])
    if output.get("error"):
        return _result(SOFT_FAIL, [str(output["error"])])
    response = output.get("llm_response")
    if not isinstance(response, dict) or not isinstance(response.get("mapped_atoms"), list) or not response["mapped_atoms"]:
        return _result(SOFT_FAIL, ["mapped_atoms is empty or missing"])
    validation = output.get("atom_map_validation")
    if not isinstance(validation, dict) or validation.get("skipped") or validation.get("passed") is True:
        return _result(PASS)
    errors: List[str] = []
    for check in validation.get("checks") or []:
        if isinstance(check, dict) and not check.get("passed"):
            details = check.get("details") or {}
            detail_text = details.get("errors") or details.get("error") or details.get("error_code") or ""
            errors.append(f"{check.get('name')}: {detail_text}"[:300])
    return _result(SOFT_FAIL, errors or ["atom-map check failed"])


def check_reaction_type(
    output: Optional[Dict[str, Any]], raw: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    if not isinstance(output, dict):
        return _result(SOFT_FAIL, ["reaction_type section missing"])
    schema = output.get("schema_validation") or {}
    errors: List[str] = []
    if output.get("status") not in ("success", None):
        errors.append(str(output.get("rationale") or output.get("error") or f"status {output.get('status')}"))
    if schema.get("status") not in (None, "ok"):
        errors.append(f"schema: {schema.get('error')}")
    selected = str(output.get("selected_label_exact") or "").strip()
    if not selected:
        errors.append("selected_label_exact is empty")
    requested = str((raw or {}).get("selected_label_exact") or "").strip() if isinstance(raw, dict) else ""
    if requested and requested.lower() != "no_match" and selected == "no_match":
        errors.append(f"label {requested!r} is not an exact taxonomy label")
    return _result(SOFT_FAIL if errors else PASS, errors)


CHECKS = {
    "conditions": check_conditions,
    "missing_reagents": check_missing_reagents,
    "atom_mapping": check_atom_mapping,
    "reaction_type": check_reaction_type,
}


def check_sections(
    outputs: Dict[str, Optional[Dict[str, Any]]],
    sections: List[str],
    answer: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for section in sections:
        if section == "reaction_type":
            raw = (answer or {}).get("reaction_type") if isinstance(answer, dict) else None
            results[section] = check_reaction_type(outputs.get(section), raw)
        else:
            results[section] = CHECKS[section](outputs.get(section))
    return results


def failed_sections(results: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    return {name: r["errors"] for name, r in results.items() if r["status"] != PASS}


def has_hard_failure(results: Dict[str, Dict[str, Any]]) -> bool:
    return any(r["status"] == HARD_FAIL for r in results.values())
