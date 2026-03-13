"""Shared atom-balance diagnostics used across validation, rescue, and scoring."""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional

from mechanistic_agent.smiles_utils import normalize_common_smiles_alias


def _normalise_counter(counter: Mapping[str, Any] | None) -> Dict[str, int]:
    output: Dict[str, int] = {}
    if not counter:
        return output
    for element, amount in counter.items():
        try:
            value = int(amount)
        except Exception:
            continue
        if value:
            output[str(element)] = value
    return dict(sorted(output.items()))


def _count_atoms(smiles_list: Iterable[str], *, include_hydrogens: bool) -> Counter[str]:
    from mechanistic_agent.tools import _atom_counter, _atom_counter_with_hydrogens

    values = list(smiles_list)
    if include_hydrogens:
        return _atom_counter_with_hydrogens(values)
    return _atom_counter(values)


def _repair_species(
    smiles_list: Iterable[str],
    *,
    side: str,
    backend_config: Any = None,
) -> Dict[str, Any]:
    from mechanistic_agent.tools import _canonicalise_candidate_smiles

    sanitized: List[str] = []
    invalid_species: List[Dict[str, Any]] = []
    repaired_species: List[Dict[str, Any]] = []

    for raw in smiles_list:
        original = str(raw or "").strip()
        normalized = normalize_common_smiles_alias(original)
        canonical, details = _canonicalise_candidate_smiles(normalized, backend_config=backend_config)
        detail_map = dict(details or {})
        if canonical:
            sanitized.append(str(canonical))
            if canonical != normalized or normalized != original or bool(detail_map.get("repair_smiles_applied")):
                repaired_species.append(
                    {
                        "side": side,
                        "before": original,
                        "normalized": normalized,
                        "after": str(canonical),
                        "validated_from": detail_map.get("validated_from"),
                    }
                )
            continue
        invalid_species.append(
            {
                "side": side,
                "species": original,
                "normalized": normalized,
                "error": str(detail_map.get("error") or "invalid_species"),
                "error_code": detail_map.get("error_code"),
            }
        )

    return {
        "sanitized": sanitized,
        "invalid_species": invalid_species,
        "repaired_species": repaired_species,
    }


def assess_balance_diagnostics(
    left: Iterable[str],
    right: Iterable[str],
    *,
    include_hydrogens: bool = False,
    backend_config: Any = None,
    left_label: str = "left",
    right_label: str = "right",
) -> Dict[str, Any]:
    """Return repaired species lists plus balance diagnostics."""

    repaired_left = _repair_species(left, side=left_label, backend_config=backend_config)
    repaired_right = _repair_species(right, side=right_label, backend_config=backend_config)

    sanitized_left = list(repaired_left["sanitized"])
    sanitized_right = list(repaired_right["sanitized"])
    invalid_species = list(repaired_left["invalid_species"]) + list(repaired_right["invalid_species"])
    repaired_species = list(repaired_left["repaired_species"]) + list(repaired_right["repaired_species"])

    payload: Dict[str, Any] = {
        "classification": "exact",
        "balanced": False,
        "deficit": {},
        "surplus": {},
        "invalid_species": invalid_species,
        "repaired_species": repaired_species,
        "sanitized_left": sanitized_left,
        "sanitized_right": sanitized_right,
        "left_label": left_label,
        "right_label": right_label,
        "include_hydrogens": include_hydrogens,
    }

    if invalid_species:
        payload["classification"] = "invalid_species"
        payload["error"] = (
            f"Invalid species for atom balance: "
            + ", ".join(str(item.get("species") or "") for item in invalid_species)
        )
        return payload

    if not sanitized_left or not sanitized_right:
        payload["classification"] = "invalid_species"
        payload["error"] = "No valid species remaining after sanitization"
        return payload

    left_counts = _count_atoms(sanitized_left, include_hydrogens=include_hydrogens)
    right_counts = _count_atoms(sanitized_right, include_hydrogens=include_hydrogens)

    deficit: Counter[str] = Counter()
    surplus: Counter[str] = Counter()
    for element in set(left_counts) | set(right_counts):
        left_total = int(left_counts.get(element, 0))
        right_total = int(right_counts.get(element, 0))
        if left_total > right_total:
            deficit[element] = left_total - right_total
        elif right_total > left_total:
            surplus[element] = right_total - left_total

    payload["deficit"] = _normalise_counter(deficit)
    payload["surplus"] = _normalise_counter(surplus)
    payload["left_counts"] = _normalise_counter(left_counts)
    payload["right_counts"] = _normalise_counter(right_counts)
    payload["balanced"] = not payload["deficit"] and not payload["surplus"]
    payload["classification"] = "exact" if payload["balanced"] else "imbalanced"

    if not payload["balanced"]:
        deltas: List[str] = []
        for element, amount in payload["deficit"].items():
            deltas.append(f"{element}: {left_counts.get(element, 0)}->{right_counts.get(element, 0)} (Δ-{amount})")
        for element, amount in payload["surplus"].items():
            deltas.append(f"{element}: {left_counts.get(element, 0)}->{right_counts.get(element, 0)} (Δ+{amount})")
        payload["error"] = "Atom imbalance detected: " + ", ".join(deltas)

    return payload
