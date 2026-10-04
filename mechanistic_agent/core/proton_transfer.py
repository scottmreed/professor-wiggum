"""Proton-transfer classification and the intermolecular tie-break.

A mechanism step that leaves the heavy-atom skeleton of every species unchanged
moves only protons (and the charges that go with them). Such a step is
**intramolecular** when one species changes its protonation pattern on its own,
and **intermolecular** when the proton moves between species, through a
*shuttle*: water, the acid or base of the conditions, a conjugate formed earlier.

Intramolecular shifts are not wrong. When two validated candidates for the same
step reach the same heavy-atom outcome, though, the displayed route should be
the one that uses a shuttle actually available in the reaction
(:func:`choose_proton_transfer_preference`); the intramolecular candidate stays
as the branch alternative.

Keys are pure RDKit and ignore formal charges, hydrogens, atom maps and species
with at most one heavy atom (H2O, H3O+, OH-, halide ions) at the state level, so
a step that only adds or removes those is still comparable.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence

try:  # pragma: no cover - import guard
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover
    Chem = None  # type: ignore[assignment]

INTRAMOLECULAR = "intramolecular"
INTERMOLECULAR = "intermolecular"


@dataclass(frozen=True)
class _Species:
    smiles: str  # canonical, atom maps removed
    key: str  # heavy-atom skeleton with bond orders; charges/H dropped
    heavy_atoms: int
    charge: int


def _fragments(smiles_list: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for item in smiles_list or []:
        text = str(item or "").strip()
        if not text:
            continue
        out.extend(part for part in text.split(".") if part.strip())
    return out


def _parse(smiles: str) -> Optional[_Species]:
    if Chem is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    canonical = Chem.MolToSmiles(mol)
    charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    skeleton = Chem.RWMol(Chem.RemoveHs(mol))
    for atom in skeleton.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
        atom.SetNumRadicalElectrons(0)
        atom.SetIsotope(0)
    return _Species(
        smiles=canonical,
        key=Chem.MolToSmiles(skeleton),
        heavy_atoms=skeleton.GetNumAtoms(),
        charge=charge,
    )


def _parse_all(smiles_list: Iterable[Any]) -> Optional[List[_Species]]:
    parsed: List[_Species] = []
    for fragment in _fragments(smiles_list):
        species = _parse(fragment)
        if species is None:
            return None
        parsed.append(species)
    return parsed


def species_skeleton_key(smiles: str) -> Optional[str]:
    """Heavy-atom skeleton of one species (conjugate acids/bases share it)."""
    species = _parse(str(smiles or "").strip())
    return species.key if species is not None else None


def skeleton_state_key(smiles_list: Iterable[Any]) -> FrozenSet[str]:
    """Set of heavy-atom skeletons in a state, ignoring species with <= 1 heavy atom.

    Unparseable species are kept verbatim so they never compare equal to a
    parseable state.
    """
    keys = set()
    for fragment in _fragments(smiles_list):
        species = _parse(fragment)
        if species is None:
            keys.add(f"?{fragment}")
        elif species.heavy_atoms > 1:
            keys.add(species.key)
    return frozenset(keys)


def available_shuttles(
    current_state: Iterable[Any],
    starting_materials: Iterable[Any] = (),
    reagent_pool: Iterable[Any] = (),
) -> Dict[str, str]:
    """Proton shuttles available in the reaction, keyed by skeleton.

    The species in the current state, the starting materials and the acid/base
    pool of the conditions. Keying by skeleton makes every conjugate acid/base of
    those species available too. Bare protons have no heavy atom and are never
    shuttles.
    """
    shuttles: Dict[str, str] = {}
    for source in (current_state, starting_materials, reagent_pool):
        for fragment in _fragments(source):
            species = _parse(fragment)
            if species is not None and species.heavy_atoms >= 1:
                shuttles.setdefault(species.key, fragment)
    return shuttles


def _states_from_smirks(reaction_smirks: Optional[str]) -> Optional[tuple]:
    text = str(reaction_smirks or "").strip()
    if ">>" not in text:
        return None
    left, _, right = text.partition(">>")
    if ">" in right:  # reactants>agents>products written with a single '>'
        right = right.split(">")[-1]
    return [left], [right]


def _result(is_pt: bool, mode: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "is_proton_transfer": is_pt,
        "mode": mode,
        "shuttle": None,
        "shuttle_available": False,
        "changed_skeletons": [],
    }
    payload.update(extra)
    return payload


def classify_proton_transfer(
    current_state: Sequence[Any],
    resulting_state: Sequence[Any],
    reaction_smirks: Optional[str] = None,
    shuttles: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Classify a step as a proton transfer and, if so, intra- or intermolecular.

    Returns ``{is_proton_transfer, mode, shuttle, shuttle_available,
    changed_skeletons}``. ``mode`` is ``intramolecular`` / ``intermolecular`` /
    ``None``. ``shuttle`` is the species that carried the proton for an
    intermolecular transfer (from the current state, else the matching
    ``shuttles`` entry, else the resulting-state species). States fall back to
    the two sides of ``reaction_smirks`` when either is empty. Skeletons that
    appear on one side only are tolerated when they are available shuttles (a
    reagent drawn from the pool, such as solvent acid).
    """
    current = list(current_state or [])
    resulting = list(resulting_state or [])
    if (not current or not resulting) and reaction_smirks:
        sides = _states_from_smirks(reaction_smirks)
        if sides is not None:
            current = current or sides[0]
            resulting = resulting or sides[1]
    cur = _parse_all(current)
    res = _parse_all(resulting)
    if not cur or not res:
        return _result(False, reason="unparseable_or_empty")

    pool_keys = set((shuttles or {}).keys())
    cur_keys = {s.key for s in cur if s.heavy_atoms > 1}
    res_keys = {s.key for s in res if s.heavy_atoms > 1}
    if not (cur_keys ^ res_keys) <= pool_keys:
        return _result(False, reason="skeleton_changed")

    lost = Counter(s.smiles for s in cur) - Counter(s.smiles for s in res)
    gained = Counter(s.smiles for s in res) - Counter(s.smiles for s in cur)
    if not lost and not gained:
        return _result(False, reason="no_change")

    by_smiles = {s.smiles: s for s in cur + res}
    families: Dict[str, Dict[str, List[_Species]]] = {}
    for side, counter in (("lost", lost), ("gained", gained)):
        for smiles, count in counter.items():
            species = by_smiles[smiles]
            families.setdefault(species.key, {"lost": [], "gained": []})[side].extend([species] * count)
    changed = sorted(families)

    if len(families) == 1:
        key = changed[0]
        fam = families[key]
        if len(fam["lost"]) == 1 and len(fam["gained"]) == 1:
            if fam["lost"][0].charge != fam["gained"][0].charge:
                return _result(False, reason="unbalanced_charge", changed_skeletons=changed)
            return _result(True, INTRAMOLECULAR, changed_skeletons=changed)
        # Several copies of one species exchange a proton (self-shuttle, autoprotolysis).
        species = (fam["lost"] or fam["gained"])[0]
        return _result(
            True,
            INTERMOLECULAR,
            shuttle=species.smiles if species.heavy_atoms >= 1 else None,
            shuttle_available=species.heavy_atoms >= 1
            and (key in pool_keys or key in {s.key for s in cur}),
            changed_skeletons=changed,
        )

    charge_delta = {
        key: sum(s.charge for s in fam["gained"]) - sum(s.charge for s in fam["lost"])
        for key, fam in families.items()
    }
    if not any(charge_delta.values()):
        return _result(True, None, reason="independent_changes", changed_skeletons=changed)

    heavy = {key: (fam["lost"] or fam["gained"])[0].heavy_atoms for key, fam in families.items()}
    substrate = max(changed, key=lambda k: (heavy[k], k))
    current_keys_all = {s.key for s in cur}
    available = pool_keys | current_keys_all
    shuttle_keys = [k for k in changed if k != substrate and heavy[k] >= 1]
    if not shuttle_keys:
        # Only a bare proton moved: no shuttle species.
        return _result(True, INTERMOLECULAR, changed_skeletons=changed, substrate=substrate)
    shuttle_keys.sort(key=lambda k: (k not in available, heavy[k], k))
    shuttle_key = shuttle_keys[0]
    fam = families[shuttle_key]
    if fam["lost"]:
        shuttle = fam["lost"][0].smiles
    elif shuttles and shuttle_key in shuttles:
        shuttle = str(shuttles[shuttle_key])
    else:
        shuttle = fam["gained"][0].smiles
    return _result(
        True,
        INTERMOLECULAR,
        shuttle=shuttle,
        shuttle_available=shuttle_key in available,
        changed_skeletons=changed,
        substrate=substrate,
    )


def outcome_key(
    current_state: Sequence[Any],
    resulting_state: Sequence[Any],
    shuttles: Optional[Mapping[str, str]] = None,
) -> FrozenSet[str]:
    """Heavy-atom outcome of a step: the resulting skeleton set, minus shuttle
    skeletons that only appear because a pool reagent was drawn in."""
    current_keys = skeleton_state_key(current_state)
    extra_pool = set((shuttles or {}).keys()) - set(current_keys)
    return frozenset(skeleton_state_key(resulting_state) - extra_pool)


def choose_proton_transfer_preference(
    current_state: Sequence[Any],
    resulting_states: Sequence[Sequence[Any]],
    shuttles: Optional[Mapping[str, str]] = None,
    reaction_smirks: Optional[Sequence[Optional[str]]] = None,
) -> Optional[Dict[str, Any]]:
    """Pick an intermolecular candidate over an equivalent top-ranked intramolecular one.

    ``resulting_states`` are validated candidates in rank order (index 0 would be
    chosen). Returns ``None`` (keep the top) unless the top is an intramolecular
    proton transfer and a later candidate is an intermolecular proton transfer
    through an available shuttle, acting on the same species, with the same
    heavy-atom outcome. Otherwise returns ``{chosen_index, displaced_index,
    chosen, displaced, shuttle, classifications}``.
    """
    if len(resulting_states) < 2:
        return None
    smirks = list(reaction_smirks or [])
    classifications = [
        classify_proton_transfer(
            current_state,
            resulting,
            reaction_smirks=smirks[idx] if idx < len(smirks) else None,
            shuttles=shuttles,
        )
        for idx, resulting in enumerate(resulting_states)
    ]
    top = classifications[0]
    if not (top["is_proton_transfer"] and top["mode"] == INTRAMOLECULAR):
        return None
    top_outcome = outcome_key(current_state, resulting_states[0], shuttles)
    top_skeletons = set(top.get("changed_skeletons") or [])
    for idx in range(1, len(resulting_states)):
        info = classifications[idx]
        if not (
            info["is_proton_transfer"]
            and info["mode"] == INTERMOLECULAR
            and info["shuttle_available"]
            and info["shuttle"]
        ):
            continue
        if outcome_key(current_state, resulting_states[idx], shuttles) != top_outcome:
            continue
        if not top_skeletons & set(info.get("changed_skeletons") or []):
            continue  # the proton moves on a different species: a different route
        return {
            "chosen_index": idx,
            "displaced_index": 0,
            "chosen": info,
            "displaced": top,
            "shuttle": info["shuttle"],
            "classifications": classifications,
        }
    return None
