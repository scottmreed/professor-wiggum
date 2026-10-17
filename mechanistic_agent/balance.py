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
    """Element counts for a species list.

    ``include_hydrogens=True`` counts every hydrogen (explicit and implicit);
    ``include_hydrogens=False`` counts heavy atoms only and excludes every
    hydrogen, including a bare ``[H+]`` written as an explicit atom. Either way
    both sides of a comparison are counted the same way.
    """
    from mechanistic_agent.tools import _atom_counter_with_hydrogens, _heavy_atom_counter

    values = list(smiles_list)
    if include_hydrogens:
        return _atom_counter_with_hydrogens(values)
    return _heavy_atom_counter(values)


def _net_charge(smiles_list: Iterable[str]) -> int:
    from mechanistic_agent.tools import _mol_from_smiles

    total = 0
    for smiles in smiles_list:
        mol = _mol_from_smiles(smiles)
        total += sum(int(atom.GetFormalCharge()) for atom in mol.GetAtoms())
    return total


def species_composition(
    smiles_list: Iterable[str],
    *,
    include_hydrogens: bool = True,
    include_charge: bool = True,
) -> Optional[Counter[str]]:
    """Element counts (plus net formal charge under the key ``"charge"``) of a species list.

    Uses the same counters as :func:`assess_balance_diagnostics`, without the
    SMILES repair pass. Returns None when any species does not parse.
    """
    values = [str(item) for item in smiles_list]
    try:
        counts = Counter(_count_atoms(values, include_hydrogens=include_hydrogens))
        if include_charge:
            counts["charge"] = _net_charge(values)
    except Exception:
        return None
    return counts


def merge_unique_species(base: List[str], additions: List[str]) -> List[str]:
    """Append additions not already present (exact string), as the coordinator applies rescue additions."""
    merged = list(base)
    seen = set(base)
    for item in additions:
        if item in seen:
            continue
        merged.append(item)
        seen.add(item)
    return merged


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
    include_charge: bool = False,
    backend_config: Any = None,
    left_label: str = "left",
    right_label: str = "right",
) -> Dict[str, Any]:
    """Return repaired species lists plus balance diagnostics.

    ``include_charge=True`` also requires the net formal charge to be conserved
    (``left_charge``/``right_charge``/``charge_delta``); a charge change makes the
    sides unbalanced even when every element balances.
    """

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
        "include_charge": include_charge,
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
    charge_ok = True
    if include_charge:
        left_charge = _net_charge(sanitized_left)
        right_charge = _net_charge(sanitized_right)
        payload["left_charge"] = left_charge
        payload["right_charge"] = right_charge
        payload["charge_delta"] = right_charge - left_charge
        charge_ok = left_charge == right_charge
    payload["balanced"] = not payload["deficit"] and not payload["surplus"] and charge_ok
    payload["classification"] = "exact" if payload["balanced"] else "imbalanced"

    if not payload["balanced"]:
        deltas: List[str] = []
        for element, amount in payload["deficit"].items():
            deltas.append(f"{element}: {left_counts.get(element, 0)}->{right_counts.get(element, 0)} (Δ-{amount})")
        for element, amount in payload["surplus"].items():
            deltas.append(f"{element}: {left_counts.get(element, 0)}->{right_counts.get(element, 0)} (Δ+{amount})")
        if not charge_ok:
            delta = int(payload["charge_delta"])
            deltas.append(
                f"net charge: {payload['left_charge']:+d}->{payload['right_charge']:+d} (Δ{delta:+d})"
            )
        payload["error"] = "Atom imbalance detected: " + ", ".join(deltas)

    return payload
