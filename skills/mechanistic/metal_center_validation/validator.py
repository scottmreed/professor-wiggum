"""Metal-centre bookkeeping for a mechanism step (PRD_catalysis_handling P2).

Runs only when a step changes a species that contains a d-block metal. Both
sides are put in house convention (``organometallic_chem.notation``) and counted
with the covalent MLX model; the oxidation-state and valence-electron changes
are compared with the expected deltas of the declared (or inferred) elementary
step kind.

Policy ``soft`` (default) fails only on impossible metal states; ``strict``
also fails a declared step kind whose deltas disagree. Ambiguous notation,
multinuclear species and unpaired metal species abstain: the check passes with
``abstained`` set and never guesses.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from organometallic_chem.notation import canonical_unsanitized, to_house_convention
from organometallic_chem.organometallic_electron_count import STEP_DELTAS, try_count_electrons
from organometallic_chem.organometallic_smiles import is_d_block

from mechanistic_agent.core.types import StepValidationCheck

CHECK_NAME = "metal_center"
SCHEMA_VERSION = "metal_center.v1"
POLICIES = ("off", "soft", "strict")

# Highest oxidation state accepted per element; anything else is capped at its
# group number (d count >= 0). Generous on purpose: only clearly impossible states fail.
MAX_OXIDATION_STATE = {
    "Ni": 4, "Pd": 4, "Pt": 6,
    "Cu": 3, "Ag": 3, "Au": 5,
    "Zn": 2, "Cd": 2, "Hg": 2,
}
MIN_OXIDATION_STATE = -2

_KIND_PHRASE = {
    "oxidative_addition": "oxidative addition (OS +2, electrons +2)",
    "reductive_elimination": "reductive elimination (OS -2, electrons -2)",
    "transmetalation": "transmetalation (no OS or electron change)",
    "migratory_insertion": "migratory insertion (OS 0, electrons -2)",
    "beta_hydride_elimination": "beta-hydride elimination (OS 0, electrons +2)",
    "ligand_association": "ligand association (OS 0, electrons +2)",
    "ligand_dissociation": "ligand dissociation (OS 0, electrons -2)",
}


def _roman(value: int) -> str:
    from organometallic_chem.organometallic_electron_count import roman_numeral

    return roman_numeral(value)


def _strip_maps(smiles: str) -> Optional[str]:
    from rdkit import Chem, rdBase

    blocker = rdBase.BlockLogs()  # noqa: F841
    mol = Chem.MolFromSmiles(str(smiles), sanitize=False)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return canonical_unsanitized(mol)


def _metal_symbols(smiles: str) -> List[str]:
    from rdkit import Chem, rdBase

    blocker = rdBase.BlockLogs()  # noqa: F841
    mol = Chem.MolFromSmiles(str(smiles), sanitize=False)
    if mol is None:
        return []
    return [a.GetSymbol() for a in mol.GetAtoms() if is_d_block(a.GetAtomicNum())]


def _metal_species(state: List[str]) -> Tuple[Counter, Dict[str, str]]:
    """Canonical (unmapped) metal species with multiplicity, plus one original drawing each."""
    counts: Counter = Counter()
    originals: Dict[str, str] = {}
    for item in state or []:
        for fragment in str(item or "").split("."):
            if not fragment.strip() or not _metal_symbols(fragment):
                continue
            key = _strip_maps(fragment)
            if key is None:
                continue
            counts[key] += 1
            originals.setdefault(key, fragment)
    return counts, originals


def _count(smiles: str) -> Dict[str, Any]:
    notation = to_house_convention(smiles)
    entry: Dict[str, Any] = {
        "smiles": smiles,
        "house_smiles": notation.smiles,
        "notation_changes": list(notation.changes),
    }
    if not notation.ok:
        entry["abstain"] = "; ".join(notation.ambiguous) or "notation could not be converted"
        return entry
    count, error = try_count_electrons(notation.smiles)
    if count is None:
        entry["abstain"] = error or "electron count failed"
        return entry
    entry.update(
        metal=count.metal,
        group=count.group,
        oxidation_state=count.oxidation_state,
        d_electrons=count.d_electrons,
        valence_electrons=count.valence_electrons,
        coordination_number=count.coordination_number,
        mlx=f"L{count.n_l}X{count.n_x}" + (f"Z{count.n_z}" if count.n_z else ""),
        summary=count.summary,
    )
    return entry


def _state_findings(entry: Dict[str, Any], side: str) -> List[Dict[str, str]]:
    findings: List[Dict[str, str]] = []
    metal, os_, ve, group = entry["metal"], entry["oxidation_state"], entry["valence_electrons"], entry["group"]
    max_os = MAX_OXIDATION_STATE.get(metal, group)
    if os_ > max_os or os_ < MIN_OXIDATION_STATE or entry["d_electrons"] < 0:
        findings.append({
            "code": "impossible_oxidation_state",
            "severity": "error",
            "message": (
                f"{side} {metal} is {metal}({_roman(os_) if os_ > 0 else os_}) "
                f"(d{entry['d_electrons']}), outside the range {metal} can reach "
                f"({MIN_OXIDATION_STATE} to {max_os}); recount the X ligands and charges on {metal}"
            ),
        })
    if group in (8, 9, 10) and ve > 18:
        findings.append({
            "code": "over_18_electrons",
            "severity": "error",
            "message": (
                f"{side} {metal} has {ve} valence electrons (more than 18); a group-{group} "
                f"metal cannot take another ligand here, so a ligand must leave first"
            ),
        })
    elif ve > 18:
        findings.append({
            "code": "over_18_electrons",
            "severity": "warning",
            "message": f"{side} {metal} has {ve} valence electrons (more than 18)",
        })
    if entry.get("coordination_number", 0) > 6:
        findings.append({
            "code": "high_coordination_number",
            "severity": "warning",
            "message": f"{side} {metal} is {entry['coordination_number']}-coordinate",
        })
    return findings


def infer_step_kinds(delta_os: int, delta_ve: int) -> List[str]:
    """Every elementary step kind whose expected (OS, electron) deltas match."""
    return [
        kind for kind, (dos, dve) in STEP_DELTAS.items()
        if dos is not None and dos == delta_os and dve == delta_ve
    ]


def validate_metal_center(
    current_state: List[str],
    resulting_state: List[str],
    *,
    declared_kind: Optional[str] = None,
    policy: str = "soft",
) -> Optional[StepValidationCheck]:
    """``None`` when the step changes no metal species (or policy is ``off``)."""
    policy = policy if policy in POLICIES else "soft"
    if policy == "off":
        return None
    current, cur_orig = _metal_species(current_state)
    resulting, res_orig = _metal_species(resulting_state)
    removed = current - resulting
    added = resulting - current
    if not removed and not added:
        return None

    details: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy": policy,
        "declared_kind": declared_kind or None,
        "pairs": [],
        "findings": [],
        "abstained": None,
    }

    def _done(passed: bool) -> StepValidationCheck:
        errors = [f["message"] for f in details["findings"] if f["severity"] == "error"]
        if errors:
            details["error"] = "; ".join(errors)
        warnings = [f["message"] for f in details["findings"] if f["severity"] == "warning"]
        if warnings:
            details["warnings"] = warnings
            details["warning_only"] = passed
        return StepValidationCheck(name=CHECK_NAME, passed=passed, details=details)

    by_metal: Dict[str, Tuple[List[str], List[str]]] = {}
    for key in removed.elements():
        for sym in set(_metal_symbols(key)):
            by_metal.setdefault(sym, ([], []))[0].append(key)
    for key in added.elements():
        for sym in set(_metal_symbols(key)):
            by_metal.setdefault(sym, ([], []))[1].append(key)

    for metal, (before_keys, after_keys) in sorted(by_metal.items()):
        if len(before_keys) != 1 or len(after_keys) != 1:
            details["abstained"] = (
                f"{metal}: {len(before_keys)} species consumed and {len(after_keys)} formed; "
                "cannot pair the metal centre before and after"
            )
            return _done(True)
        before = _count(cur_orig[before_keys[0]])
        after = _count(res_orig[after_keys[0]])
        pair: Dict[str, Any] = {"metal": metal, "before": before, "after": after}
        details["pairs"].append(pair)
        if before.get("abstain") or after.get("abstain"):
            details["abstained"] = before.get("abstain") or after.get("abstain")
            continue
        details["findings"].extend(_state_findings(before, "before the step,"))
        details["findings"].extend(_state_findings(after, "after the step,"))
        d_os = after["oxidation_state"] - before["oxidation_state"]
        d_ve = after["valence_electrons"] - before["valence_electrons"]
        pair.update(delta_oxidation_state=d_os, delta_valence_electrons=d_ve)
        matching = infer_step_kinds(d_os, d_ve)
        pair["consistent_kinds"] = matching
        transition = (
            f"{metal} goes from {before['summary']} to {after['summary']} "
            f"(OS {d_os:+d}, electrons {d_ve:+d})"
        )
        if declared_kind and declared_kind in _KIND_PHRASE:
            if declared_kind not in matching:
                details["findings"].append({
                    "code": "declared_kind_mismatch",
                    "severity": "error" if policy == "strict" else "warning",
                    "message": f"{transition}, but the step is labelled {_KIND_PHRASE[declared_kind]}",
                })
        elif not matching:
            details["findings"].append({
                "code": "no_matching_step_kind",
                "severity": "warning",
                "message": f"{transition}, which matches no single elementary step; "
                "check whether two events were merged or a ligand was dropped",
            })
    passed = not any(f["severity"] == "error" for f in details["findings"])
    return _done(passed)


__all__ = ["CHECK_NAME", "POLICIES", "SCHEMA_VERSION", "infer_step_kinds", "validate_metal_center"]
