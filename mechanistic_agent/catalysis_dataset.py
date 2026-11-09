"""Train-only catalysis development cases from FlowER (PRD_catalysis_handling R5.1/R5.2).

Two slices, both drawn from ``flower_new_dataset/train.txt`` through the curriculum
lookup cache, in curriculum order (smallest mechanisms first):

* **transition_metal**: mechanisms in which a d-block atom changes its bonded
  neighbours in at least one elementary step (the metal takes part, it is not a
  Lewis-acid or counter-ion spectator). Stratified by coupling family, read from the
  first step's reagents: Suzuki (B), Stille (Sn), Negishi (Zn), Sonogashira (Pd + Cu),
  other Pd, Ru, Cu, other metal.
* **organocatalysis**: metal-free mechanisms in which a starting species with at least
  three heavy atoms is regenerated unchanged at the end and has a heavy-atom bond made
  or broken on its atoms during the path (nucleophilic or covalent catalysis such as
  DMAP acyl transfer; a pure proton shuttle does not qualify).

FlowER groups whose rows do not chain (``state_discontinuity``: the next row starts
from a different state, a gap in the source grouping) are skipped and counted, never
repaired: repairing would mean inventing steps.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from rdkit import Chem, rdBase

from organometallic_chem.organometallic_smiles import is_d_block

from .flower_curriculum import (
    ConversionError,
    _convert_group,
    eval_case_from_case,
    load_mechanism_reactions,
)
from .radical_dataset import _check_train_only

SCHEMA = "catalysis_development.v1"
DEFAULT_MAX_STEPS = 22
_D_BLOCK_TOKEN = re.compile(
    r"\[(Sc|Ti|V|Cr|Mn|Fe|Co|Ni|Cu|Zn|Y|Zr|Nb|Mo|Tc|Ru|Rh|Pd|Ag|Cd|Hf|Ta|W|Re|Os|Ir|Pt|Au|Hg)(?![a-z])"
)
TM_FAMILIES = (
    "suzuki", "stille", "negishi", "sonogashira", "other_pd", "ruthenium", "copper", "other_metal",
)


def _mol(smiles: str) -> Optional[Chem.Mol]:
    blocker = rdBase.BlockLogs()  # noqa: F841
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is not None:
        mol.UpdatePropertyCache(strict=False)
    return mol


def _neighbours_by_map(side: str, wanted: Optional[Set[int]] = None) -> Dict[int, frozenset]:
    mol = _mol(side)
    if mol is None:
        return {}
    out = {}
    for atom in mol.GetAtoms():
        tag = atom.GetAtomMapNum()
        if tag and (wanted is None or tag in wanted):
            out[tag] = frozenset(n.GetAtomMapNum() for n in atom.GetNeighbors())
    return out


def _heavy_bonds(side: str) -> Set[Tuple[int, int]]:
    mol = _mol(side)
    if mol is None:
        return set()
    bonds = set()
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if a.GetAtomicNum() > 1 and b.GetAtomicNum() > 1 and a.GetAtomMapNum() and b.GetAtomMapNum():
            bonds.add(tuple(sorted((a.GetAtomMapNum(), b.GetAtomMapNum()))))
    return bonds


def metal_active(mapped_reactions: Sequence[str]) -> bool:
    """A d-block atom gains or loses a bonded neighbour in some step."""
    for reaction in mapped_reactions:
        left, _, right = reaction.partition(">>")
        mol = _mol(left)
        if mol is None:
            continue
        metal_maps = {a.GetAtomMapNum() for a in mol.GetAtoms() if is_d_block(a.GetAtomicNum()) and a.GetAtomMapNum()}
        if not metal_maps:
            continue
        before = _neighbours_by_map(left, metal_maps)
        after = _neighbours_by_map(right, metal_maps)
        if any(before.get(tag) != after.get(tag) for tag in metal_maps):
            return True
    return False


def tm_family(starting: Sequence[str]) -> Tuple[str, List[str]]:
    text = ".".join(starting)
    metals = sorted({m for m in _D_BLOCK_TOKEN.findall(text)})
    has = lambda pattern: re.search(pattern, text) is not None  # noqa: E731
    if "Pd" in metals:
        if has(r"B(?![a-z])"):  # bracketed or not; not Br/Ba
            return "suzuki", metals
        if has(r"Sn"):
            return "stille", metals
        if has(r"Zn"):
            return "negishi", metals
        if "Cu" in metals:
            return "sonogashira", metals
        return "other_pd", metals
    if "Ru" in metals:
        return "ruthenium", metals
    if metals == ["Cu"]:
        return "copper", metals
    return "other_metal", metals


def _unmapped(smiles: str) -> Optional[str]:
    mol = _mol(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(Chem.RemoveHs(mol))


def _mapped_key(smiles: str) -> Optional[str]:
    """Canonical SMILES that keeps atom maps: the same atoms, not just the same species."""
    mol = _mol(smiles)
    if mol is None:
        return None
    Chem.FastFindRings(mol)
    return Chem.MolToSmiles(mol)


def organocatalyst(case: Dict[str, Any]) -> Optional[str]:
    """Unmapped SMILES of a regenerated, bond-participating metal-free catalyst, or None.

    The catalyst must come back as the *same atoms* (identical atom-mapped species at the
    end). A reagent used in excess, where one copy reacts and another copy is carried
    unchanged, does not qualify: the copy left at the end never took part.
    """
    starting = list(case.get("starting_materials") or [])
    products = list(case.get("products") or [])
    if any(_D_BLOCK_TOKEN.search(s) for s in starting):
        return None
    final_mapped = {_mapped_key(p) for p in products}
    steps = (case.get("verified_mechanism") or {}).get("steps") or []
    for species in starting:
        key = _unmapped(species)
        mol = _mol(species)
        if key is None or mol is None or _mapped_key(species) not in final_mapped:
            continue
        if sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1) < 3:
            continue
        maps = {a.GetAtomMapNum() for a in mol.GetAtoms() if a.GetAtomicNum() > 1 and a.GetAtomMapNum()}
        for step in steps:
            before = _heavy_bonds(".".join(step.get("current_state") or []))
            after = _heavy_bonds(".".join(step.get("resulting_state") or []))
            if any(set(bond) & maps for bond in before ^ after):
                return key
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_catalysis_cases(
    train: Path,
    index_rows: Iterable[Dict[str, Any]],
    *,
    cache_path: Path,
    per_family: int = 6,
    organo_cases: int = 12,
    organo_scan: int = 20000,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select catalysis development cases from the curriculum index (curriculum order)."""
    given = Path(train)
    resolved = given.resolve()
    _check_train_only(given, resolved)
    rows = sorted(index_rows, key=lambda r: int(r.get("global_rank") or 0))
    picked: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    counts: Counter = Counter()
    organo: List[Dict[str, Any]] = []
    organo_seen: Set[str] = set()
    organo_per_catalyst: Counter = Counter()
    scanned = 0
    for row in rows:
        if all(len(picked[f]) >= per_family for f in TM_FAMILIES) and len(organo) >= organo_cases:
            break
        if int(row.get("step_count") or 0) > max_steps:
            continue
        mechanism_id = int(row["mechanism_id"])
        try:
            reactions = load_mechanism_reactions(mechanism_id, input_path=resolved, cache_path=cache_path)
        except Exception:
            counts["load_failed"] += 1
            continue
        if not reactions:
            continue
        has_metal = _D_BLOCK_TOKEN.search(reactions[0]) is not None
        if has_metal:
            counts["metal_mechanisms_seen"] += 1
            if not metal_active(reactions):
                counts["metal_spectator"] += 1
                continue
            counts["metal_active"] += 1
            family, metals = tm_family(reactions[0].partition(">>")[0].split("."))
            if len(picked[family]) >= per_family:
                continue
        else:
            scanned += 1
            if scanned > organo_scan or len(organo) >= organo_cases:
                continue
        try:
            case = _convert_group(mechanism_id, reactions)
        except ConversionError as exc:
            counts[f"conversion_failed:{exc}"] += 1
            continue
        if has_metal:
            out = eval_case_from_case(case)
            out["tags"] = ["flower", "train", "catalysis", "transition_metal", family]
            out["catalysis"] = {"slice": "transition_metal", "family": family, "metals": metals}
            picked[family].append(out)
        else:
            catalyst = organocatalyst(case)
            if catalyst is None or organo_per_catalyst[catalyst] >= 3:
                continue
            organo_per_catalyst[catalyst] += 1
            organo_seen.add(catalyst)
            out = eval_case_from_case(case)
            out["tags"] = ["flower", "train", "catalysis", "organocatalysis"]
            out["catalysis"] = {"slice": "organocatalysis", "catalyst": catalyst}
            organo.append(out)
    cases = [case for family in TM_FAMILIES for case in picked[family]] + organo
    for case in cases:
        case["notes"] = (
            "Train-derived catalysis development case; FlowER reference path, not an expert-reviewed "
            "mechanism and not a held-out quality claim."
        )
    report = {
        "schema": SCHEMA,
        "source_path": str(given),
        "source_sha256": _sha256(resolved),
        "per_family": per_family,
        "organo_cases": organo_cases,
        "organo_scan": organo_scan,
        "max_steps": max_steps,
        "selected": {family: len(picked[family]) for family in TM_FAMILIES} | {"organocatalysis": len(organo)},
        "counts": dict(sorted(counts.items())),
        "organocatalysts": sorted(organo_seen),
    }
    return cases, report


__all__ = ["SCHEMA", "TM_FAMILIES", "build_catalysis_cases", "metal_active", "organocatalyst", "tm_family"]
