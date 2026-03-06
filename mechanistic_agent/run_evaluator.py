"""Deterministic helpers for comparing outputs against known benchmark mechanisms."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def compare_with_known_answers(
    mechanism_rows: List[Dict[str, Any]],
    verified_mechanism: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compare actual mechanism step outputs against a known answer payload.

    Supports both formats:
    - verified_mechanism: {"steps": [{"resulting_state": [...]}, ...]}
    - known_mechanism: {"steps": [{"target_smiles": "..."}, ...]}
    """
    if not verified_mechanism or not isinstance(verified_mechanism, dict):
        return {"available": False}

    expected_steps = verified_mechanism.get("steps") or []
    # HumanBenchmark known_mechanism format.
    if expected_steps and isinstance(expected_steps[0], dict) and "target_smiles" in expected_steps[0]:
        expected_steps = [
            {"step_index": row.get("step_index"), "resulting_state": [row.get("target_smiles")]}
            for row in expected_steps
        ]

    if not expected_steps:
        return {
            "available": True,
            "expected_step_count": 0,
            "actual_step_count": len(mechanism_rows),
        }

    expected_count = len(expected_steps)
    actual_count = len(mechanism_rows)

    expected_final: set[str] = set()
    last_expected = expected_steps[-1]
    for smi in last_expected.get("resulting_state") or []:
        expected_final.add(str(smi).strip())

    actual_final: set[str] = set()
    if mechanism_rows:
        last_actual = mechanism_rows[-1]
        output = last_actual.get("output") or {}
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except Exception:
                output = {}
        for smi in output.get("resulting_state") or []:
            actual_final.add(str(smi).strip())

    product_overlap = len(expected_final & actual_final)

    return {
        "available": True,
        "expected_step_count": expected_count,
        "actual_step_count": actual_count,
        "step_count_match": actual_count == expected_count,
        "product_match": product_overlap > 0 if expected_final else False,
        "product_overlap_count": product_overlap,
        "provisional": bool(verified_mechanism.get("provisional", True)),
        "source_refs": list(verified_mechanism.get("source_refs") or []),
    }


__all__ = ["compare_with_known_answers"]
