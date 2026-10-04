"""Proton-transfer / shuttle agnostic comparison of mechanism paths.

FlowER reference mechanisms often move a proton intramolecularly or through a
group of the substrate (an imidazole N relaying a proton), while an equally
valid route moves it through a shuttle (water / H3O+, TFA / trifluoroacetate,
a second amine equivalent). Exact state matching scores those routes as
unrelated. This module compares paths on a *heavy-atom skeleton* instead:

* Species key: heavy-atom connectivity (element per atom, which heavy atoms are
  bonded; no bond orders, charges or hydrogens) plus ``charge - n_hydrogens``.
  Adding or removing a proton changes charge and H count together, so the key is
  invariant under proton transfer and tautomerism; resonance forms share it too
  (same connectivity, charge and H count). A hydride or H-atom transfer changes
  it (ketone vs alkoxide), as does making or breaking any heavy-atom bond.
* Species with at most one heavy atom (H2O, H3O+, OH-, H+, halides, HCl, metal
  counter-ions) are dropped: they are the shuttles and leaving ions.
* State key: the *set* of species keys, so an extra equivalent of a shuttle or
  base (TFA vs trifluoroacetate, a second amine) does not change it.
* Path: states collapsed over consecutive duplicates, so proton-transfer-only
  steps vanish. Keys present in every state of either path (catalysts and
  spectators carried throughout) are removed from both paths first.

The alignment is ``LCS / max(len)`` over the collapsed paths after their
(shared) starting state; identical skeleton paths score 1.0.
"""
from __future__ import annotations

from contextlib import redirect_stderr
from io import StringIO
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence

try:  # pragma: no cover - RDKit is a hard dependency in practice
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover
    Chem = None  # type: ignore[assignment]

SkeletonState = FrozenSet[str]

SKELETON_KEY_VERSION = "heavy_atom_connectivity_charge_minus_h.v1"

_KEY_CACHE: Dict[str, tuple] = {}


def _parse(smiles: str) -> Any:
    with redirect_stderr(StringIO()):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return mol
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        try:
            mol.UpdatePropertyCache(strict=False)
        except Exception:
            return None
        return mol


def _fragment_key(mol: Any, atom_indices: Sequence[int]) -> Optional[str]:
    atoms = [mol.GetAtomWithIdx(idx) for idx in atom_indices]
    heavy = [atom for atom in atoms if atom.GetAtomicNum() > 1]
    if len(heavy) <= 1:
        return None
    charge = sum(atom.GetFormalCharge() for atom in atoms)
    n_h = sum(1 for atom in atoms if atom.GetAtomicNum() == 1)
    n_h += sum(atom.GetTotalNumHs() for atom in heavy)
    skeleton = Chem.RWMol()
    index: Dict[int, int] = {}
    for atom in heavy:
        new_atom = Chem.Atom(atom.GetAtomicNum())
        new_atom.SetNoImplicit(True)
        index[atom.GetIdx()] = skeleton.AddAtom(new_atom)
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if begin in index and end in index:
            skeleton.AddBond(index[begin], index[end], Chem.BondType.SINGLE)
    graph = skeleton.GetMol()
    graph.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(graph)
    return f"{Chem.MolToSmiles(graph, canonical=True)}|{charge - n_h}"


def skeleton_species_keys(smiles: str) -> List[str]:
    """Skeleton keys of one species string (one per fragment with >= 2 heavy atoms)."""
    text = str(smiles or "").strip()
    if not text:
        return []
    cached = _KEY_CACHE.get(text)
    if cached is not None:
        return list(cached)
    keys: List[str] = []
    mol = _parse(text) if Chem is not None else None
    if mol is None:
        # Unparseable placeholder: compare as text (never matches a real species).
        keys = [f"unparsed:{text}"]
    else:
        for atom_indices in Chem.GetMolFrags(mol):
            key = _fragment_key(mol, atom_indices)
            if key is not None:
                keys.append(key)
        keys = sorted(set(keys))
    _KEY_CACHE[text] = tuple(keys)
    return list(keys)


def skeleton_state_key(species: Iterable[str]) -> SkeletonState:
    """Set of skeleton keys for a state (list of species SMILES)."""
    out: set[str] = set()
    for item in species or []:
        out.update(skeleton_species_keys(str(item)))
    return frozenset(out)


def collapse_states(states: Sequence[SkeletonState]) -> List[SkeletonState]:
    out: List[SkeletonState] = []
    for state in states:
        if not out or out[-1] != state:
            out.append(state)
    return out


def collapse_skeleton_path(states: Sequence[Sequence[str]]) -> List[SkeletonState]:
    """Skeleton states of a path with consecutive duplicates (proton moves) collapsed."""
    return collapse_states([skeleton_state_key(state) for state in states])


def _persistent(states: Sequence[SkeletonState]) -> SkeletonState:
    if not states:
        return frozenset()
    common = set(states[0])
    for state in states[1:]:
        common &= state
    return frozenset(common)


def _lcs_length(a: Sequence[SkeletonState], b: Sequence[SkeletonState]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for item in a:
        cur = [0]
        for j, other in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if item == other else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def proton_agnostic_alignment(
    predicted_states: Sequence[Sequence[str]],
    reference_states: Sequence[Sequence[str]],
) -> Dict[str, Any]:
    """Align two paths on heavy-atom skeleton states.

    Both arguments are full paths whose first element is the starting state
    (starting materials), followed by each step's resulting state.

    Returns ``score`` (LCS / max length of the collapsed paths after the start;
    1.0 when both are empty), the collapsed step counts, the spectator keys
    removed, and ``step_labels`` for each predicted step after the start:
    ``proton_shuttle_step`` (skeleton unchanged from the previous state),
    ``skeleton_match`` (skeleton state on the reference path) or
    ``skeleton_unmatched``.
    """
    pred_raw = [skeleton_state_key(state) for state in predicted_states]
    ref_raw = [skeleton_state_key(state) for state in reference_states]
    spectators = _persistent(pred_raw) | _persistent(ref_raw)
    pred = [frozenset(state - spectators) for state in pred_raw]
    ref = [frozenset(state - spectators) for state in ref_raw]

    pred_collapsed = collapse_states(pred)
    ref_collapsed = collapse_states(ref)
    pred_tail = pred_collapsed[1:]
    ref_tail = ref_collapsed[1:]
    longest = max(len(pred_tail), len(ref_tail))
    matched = _lcs_length(pred_tail, ref_tail)
    score = 1.0 if longest == 0 else matched / longest

    ref_states = set(ref_tail)
    labels: List[str] = []
    for idx in range(1, len(pred)):
        if pred[idx] == pred[idx - 1]:
            labels.append("proton_shuttle_step")
        elif pred[idx] in ref_states:
            labels.append("skeleton_match")
        else:
            labels.append("skeleton_unmatched")

    return {
        "score": score,
        "matched_skeleton_steps": matched,
        "predicted_skeleton_steps": len(pred_tail),
        "reference_skeleton_steps": len(ref_tail),
        "spectator_keys": sorted(spectators),
        "step_labels": labels,
        "key_version": SKELETON_KEY_VERSION,
    }
