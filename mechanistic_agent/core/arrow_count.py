"""Arrow-count tie-break over validated candidates (deterministic, AI-free).

Ported from ChemIllusion's ``mechanism_resonance_service`` (``be_distance``,
``arrows_for`` and the ``(arrows, charged_atoms)`` sort key of ``assess_step``).

Every two-electron arrow changes the bond-electron (BE) matrix by exactly 4 in
L1 distance (lone pair -> new bond: -2 on the donor's diagonal, +2 on the new
bond's off-diagonal cell; heterolysis and bond -> new bond likewise), so
``ceil(BE_L1 / 4)`` is the arrow count of an ordinary step. The convention is
``core/bond_electron.py``'s ``ugi_flower_kekule_v1`` (off-diagonal = 2 x Kekulé
bond order, diagonal = non-bonding electrons with H folded in), computed sparsely
from the parsed atoms and bonds (O(atoms + bonds), no dense matrices): every
diagonal cell plus each bonded pair once, which equals ChemIllusion's
``Σ|Δfree| + 2·Σ|Δorder|``. Hydrogens that are not mapped atoms are folded into
their heavy atom, so their H-X bond is added back as 2 electrons per H gained or
lost: a proton transfer is 2 arrows, as with explicit hydrogen atoms.

Only a **fully mapped, whole-species** step is measured: both sides carry the
same atom maps and every heavy atom is mapped. An unmapped or one-sided atom, or
a reaction-centre (core-only) SMIRKS, would leave electrons out of the count and
let an under-mapped candidate look more parsimonious.

The count is measured from the candidate's mapped ``reaction_smirks``, never
from ``len(electron_pushes)``: the pushes are what the model claimed, and that
is the thing being judged.

:func:`choose_arrow_count_preference` only **reorders** candidates that every
validator already passed. It never rejects, filters or changes one.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from mechanistic_agent.resonance import resonance_key
from mechanistic_agent.skeleton_alignment import _parse


OUTCOME_KEY_VERSION = "resonance_multiset_heavy_ge2.v1"

OutcomeKey = Tuple[str, ...]


def _fragments(species: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for item in species or []:
        text = str(item or "").strip()
        if text:
            out.extend(part for part in text.split(".") if part.strip())
    return out


def _heavy_atoms(mol: Any) -> int:
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)


_SideAtoms = Dict[int, Tuple[float, int]]
_SideBonds = Dict[Tuple[int, int], float]


def _parse_side(side: str) -> Optional[Tuple[_SideAtoms, _SideBonds]]:
    """Mapped heavy atoms (non-bonding electrons, folded H) and bonds between mapped atoms.

    The same parse, Kekulé form and non-bonding formula as
    ``reaction_focus.parse_mapped_side`` (outer − charge − Σ bond orders − n_H).
    ``None`` when a fragment does not parse or a heavy atom is unmapped. Mapped
    ``[H:n]`` atoms are atoms in their own right; unmapped ``[H]`` atoms are folded.
    """
    from rdkit import Chem

    from .mechanism_moves import mapped_smiles_parser_params

    table = Chem.GetPeriodicTable()
    atoms: _SideAtoms = {}
    bonds: _SideBonds = {}
    for token in side.split("."):
        token = token.strip()
        if not token:
            continue
        mol = Chem.MolFromSmiles(token, mapped_smiles_parser_params(sanitize=True))
        if mol is None:
            return None
        # This tie-break estimates pair arrows only; fishhooks have a different
        # electron budget and must not receive an artificially cheaper score.
        if any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
            return None
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:  # noqa: BLE001 - keep aromatic orders, as parse_mapped_side does
            pass
        for atom in mol.GetAtoms():
            m = int(atom.GetAtomMapNum() or 0)
            if m <= 0:
                if atom.GetAtomicNum() > 1:
                    return None  # an unmapped heavy atom: its bond changes would vanish
                continue
            if m in atoms:
                return None  # a map number used twice
            order_sum = sum(float(b.GetBondTypeAsDouble()) for b in atom.GetBonds())
            n_h = int(atom.GetTotalNumHs())
            nonbonding = float(table.GetNOuterElecs(atom.GetAtomicNum())) - atom.GetFormalCharge() - order_sum - n_h
            unmapped_h = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 1 and not n.GetAtomMapNum())
            atoms[m] = (nonbonding, n_h + unmapped_h if atom.GetAtomicNum() > 1 else 0)
        for bond in mol.GetBonds():
            a = int(bond.GetBeginAtom().GetAtomMapNum() or 0)
            b = int(bond.GetEndAtom().GetAtomMapNum() or 0)
            if a > 0 and b > 0:
                bonds[(min(a, b), max(a, b))] = float(bond.GetBondTypeAsDouble())
    return atoms, bonds


def step_arrow_count(reaction_smirks: Any) -> Optional[int]:
    """``ceil(BE_L1 / 4)`` for a fully mapped step; ``None`` when it cannot be measured.

    Unmeasurable: no SMIRKS, a side that does not parse, an unmapped heavy atom,
    atom maps that differ between the sides, or a step that changes no electrons
    (a real step always takes at least one arrow).
    """
    text = str(reaction_smirks or "").strip()
    if ">>" not in text:
        return None
    left_text, right_text = text.split(">>", 1)
    try:
        left, right = _parse_side(left_text), _parse_side(right_text)
    except Exception:  # noqa: BLE001 - unmeasurable
        return None
    if left is None or right is None:
        return None
    (atoms_l, bonds_l), (atoms_r, bonds_r) = left, right
    if not atoms_l or set(atoms_l) != set(atoms_r):
        return None
    total = 0.0
    for m, (nonbonding_l, h_l) in atoms_l.items():
        nonbonding_r, h_r = atoms_r[m]
        total += abs(nonbonding_r - nonbonding_l) + 2.0 * abs(h_r - h_l)
    for pair in set(bonds_l) | set(bonds_r):
        total += 2.0 * abs(bonds_r.get(pair, 0.0) - bonds_l.get(pair, 0.0))
    arrows = math.ceil(total / 4.0 - 1e-9)
    return arrows if arrows >= 1 else None


def outcome_key(resulting_state: Iterable[Any]) -> Optional[OutcomeKey]:
    """Multiset of resonance keys of the resulting species, or ``None`` if one is unparseable.

    Species with at most one heavy atom (H2O, H3O+, OH-, H+, halides, metal
    counter-ions) are left out, as the proton-transfer and skeleton keys do:
    they are the shuttles and leaving ions. Spectators carried unchanged appear
    in every candidate's state, so they compare equal without special handling.
    """
    keys: List[str] = []
    for fragment in _fragments(resulting_state):
        mol = _parse(fragment)
        if mol is None:
            return None
        # This tie-break estimates pair arrows only; fishhooks have a different
        # electron budget and must not receive an artificially cheaper score.
        if any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
            return None
        if _heavy_atoms(mol) <= 1:
            continue
        key = resonance_key(fragment)
        if key is None:
            return None
        keys.append(key)
    return tuple(sorted(keys))


def _species_keys(species: Iterable[Any]) -> Optional[Counter]:
    """Resonance keys of every species (small ones included), or ``None`` if one is unparseable."""
    keys: Counter = Counter()
    for fragment in _fragments(species):
        key = resonance_key(fragment)
        if key is None:
            return None
        keys[key] += 1
    return keys


def candidate_arrow_count(
    reaction_smirks: Any,
    resulting_state: Iterable[Any],
    current_state: Iterable[Any],
) -> Optional[int]:
    """:func:`step_arrow_count` when the SMIRKS describes the candidate's own step.

    The arrows are measured on the SMIRKS while "same outcome" is judged on
    ``resulting_state``, so the count only stands for the candidate when

    * every SMIRKS product with at least two heavy atoms is, up to resonance, in
      ``resulting_state`` (as a sub-multiset), and
    * every SMIRKS reactant, small species included, is, up to resonance, in
      ``current_state`` (as a sub-multiset). Up to resonance, because pushing from
      another contributor of a current species is exactly what is being rewarded;
      a reactant that is not a current species (a leaving group left out of the
      SMIRKS, which turns CH3Br into a methyl radical) is not.

    Otherwise the candidate is unmeasurable and keeps its rank.
    """
    text = str(reaction_smirks or "").strip()
    if ">>" not in text:
        return None
    left_text, right_text = text.split(">>", 1)
    state_keys = outcome_key(list(resulting_state or []))
    product_keys = outcome_key([right_text])
    if state_keys is None or product_keys is None or Counter(product_keys) - Counter(state_keys):
        return None
    current_keys = _species_keys(list(current_state or []))
    reactant_keys = _species_keys([left_text])
    if current_keys is None or not reactant_keys or reactant_keys - current_keys:
        return None
    return step_arrow_count(text)


def charged_atom_count(resulting_state: Iterable[Any]) -> int:
    """Atoms with a non-zero formal charge in the species :func:`outcome_key` counts."""
    total = 0
    for fragment in _fragments(resulting_state):
        mol = _parse(fragment)
        if mol is None or _heavy_atoms(mol) <= 1:
            continue
        total += sum(1 for atom in mol.GetAtoms() if atom.GetFormalCharge() != 0)
    return total


def choose_arrow_count_preference(
    resulting_states: Sequence[Sequence[Any]],
    arrow_counts: Sequence[Optional[int]],
) -> Optional[Dict[str, Any]]:
    """Pick the fewest-arrow candidate with the same outcome as the top one.

    ``resulting_states`` are validated candidates in rank order (index 0 would be
    chosen) and ``arrow_counts`` their :func:`step_arrow_count` values. Only
    candidates whose outcome (:func:`outcome_key`) equals the top's compete; the
    sort key is ``(arrows, charged_atoms, index)``. A candidate with no arrow
    count takes no part: an unmeasurable top keeps its place and an unmeasurable
    lower rank is never chosen. Returns ``None`` when the top stays first,
    otherwise ``{chosen_index, displaced_index, chosen_arrows, displaced_arrows,
    chosen_charged_atoms, displaced_charged_atoms, outcome_key, reason}``.
    """
    if len(resulting_states) < 2 or len(arrow_counts) != len(resulting_states):
        return None
    top_arrows = arrow_counts[0]
    top_outcome = outcome_key(resulting_states[0])
    if top_arrows is None or top_outcome is None:
        return None
    charged = {0: charged_atom_count(resulting_states[0])}
    best_index = 0
    best_key = (top_arrows, charged[0], 0)
    for idx in range(1, len(resulting_states)):
        arrows = arrow_counts[idx]
        if arrows is None or outcome_key(resulting_states[idx]) != top_outcome:
            continue
        charged[idx] = charged_atom_count(resulting_states[idx])
        key = (arrows, charged[idx], idx)
        if key < best_key:
            best_index, best_key = idx, key
    if best_index == 0:
        return None
    chosen_arrows = int(arrow_counts[best_index])  # type: ignore[arg-type]
    return {
        "chosen_index": best_index,
        "displaced_index": 0,
        "chosen_arrows": chosen_arrows,
        "displaced_arrows": int(top_arrows),
        "chosen_charged_atoms": charged[best_index],
        "displaced_charged_atoms": charged[0],
        "outcome_key": ".".join(top_outcome),
        "reason": (
            "fewer_arrows_same_outcome" if chosen_arrows < top_arrows else "fewer_charged_atoms_same_arrows"
        ),
    }


__all__ = [
    "OUTCOME_KEY_VERSION",
    "charged_atom_count",
    "choose_arrow_count_preference",
    "outcome_key",
    "step_arrow_count",
]
