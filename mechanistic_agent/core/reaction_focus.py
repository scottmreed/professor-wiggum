"""ReactionFocus v1 — the deterministic active region of an elementary step (Observatory PRD §9).

Every candidate transition is a mapped reaction SMIRKS. Atom ids are the atom
map numbers rendered as ``a<map>`` (stable for the run, collision-free,
independent of SMILES position; persistent ids from ``mapped_state`` map onto
them 1:1 when the mapped loop state is on).

Core atoms are the union of atoms that:
  * gain, lose or change a bond;
  * change formal charge, hydrogen count or non-bonding electron count;
  * take part in an explicit electron push;
  * have a non-zero ΔBE entry;
  * appear on only one side of the SMIRKS (``unbalanced_atom_ids``).

Context is graph radius ``context_radius`` (default 1) around the core, taken
on both sides, plus the whole of any ring the core touches. Everything else is
``unchanged_atom_ids``. ``matrix_atom_ids`` (core ∪ context, sorted) is the
default row/column set for the bond-electron view (PRD §10.2).

The same mapped-side parser (:func:`parse_mapped_side`) feeds
``core/bond_electron.py`` so the two projections never disagree about an atom.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:  # pragma: no cover - import guard mirrors the rest of the package
    from rdkit import Chem, rdBase
except Exception:  # pragma: no cover
    Chem = None  # type: ignore[assignment]
    rdBase = None  # type: ignore[assignment]

from .mechanism_moves import mapped_smiles_parser_params, normalize_electron_pushes, split_cxsmiles_metadata

REACTION_FOCUS_SCHEMA = "reaction_focus.v1"


def atom_id(map_number: int) -> str:
    return f"a{int(map_number)}"


def _sorted_ids(maps: Iterable[int]) -> List[str]:
    return [atom_id(m) for m in sorted({int(m) for m in maps})]


@dataclass
class AtomProps:
    element: str
    formal_charge: int
    total_h: int
    heavy_bond_order_sum: float
    nonbonding_electrons: float
    radical_electrons: int


@dataclass
class SideGraph:
    """One side of a mapped SMIRKS, keyed by atom map number."""

    atoms: Dict[int, AtomProps] = field(default_factory=dict)
    bonds: Dict[Tuple[int, int], float] = field(default_factory=dict)
    neighbours: Dict[int, Set[int]] = field(default_factory=dict)
    rings: List[Set[int]] = field(default_factory=list)


def parse_mapped_side(side: str) -> SideGraph:
    """Parse one side of a mapped reaction. Raises ``ValueError`` on bad SMILES.

    Molecules are sanitized and Kekulized so bond orders are integral
    (``ugi_flower_kekule_v1``). Unmapped atoms are not ids; bonds to them still
    count toward the mapped atom's bonding electrons.
    """
    if Chem is None:  # pragma: no cover
        raise ValueError("RDKit is required for reaction focus")
    graph = SideGraph()
    table = Chem.GetPeriodicTable()
    blocker = rdBase.BlockLogs() if rdBase is not None else None
    try:
        for token in str(side or "").split("."):
            smiles = token.strip()
            if not smiles:
                continue
            mol = Chem.MolFromSmiles(smiles, mapped_smiles_parser_params(sanitize=True))
            if mol is None:
                raise ValueError(f"Invalid mapped SMILES: {smiles!r}")
            try:
                Chem.Kekulize(mol, clearAromaticFlags=True)
            except Exception:  # pragma: no cover - keep aromatic orders if Kekulé fails
                pass
            for atom in mol.GetAtoms():
                m = int(atom.GetAtomMapNum() or 0)
                if m <= 0:
                    continue
                order_sum = sum(float(b.GetBondTypeAsDouble()) for b in atom.GetBonds())
                outer = int(table.GetNOuterElecs(atom.GetAtomicNum()))
                charge = int(atom.GetFormalCharge())
                n_h = int(atom.GetTotalNumHs())
                nonbonding = float(outer - charge - order_sum - n_h)
                graph.atoms[m] = AtomProps(
                    element=atom.GetSymbol(),
                    formal_charge=charge,
                    total_h=n_h,
                    heavy_bond_order_sum=order_sum,
                    nonbonding_electrons=nonbonding,
                    radical_electrons=int(atom.GetNumRadicalElectrons()),
                )
                graph.neighbours.setdefault(m, set())
            for bond in mol.GetBonds():
                a = int(bond.GetBeginAtom().GetAtomMapNum() or 0)
                b = int(bond.GetEndAtom().GetAtomMapNum() or 0)
                if a <= 0 or b <= 0:
                    continue
                pair = (min(a, b), max(a, b))
                graph.bonds[pair] = float(bond.GetBondTypeAsDouble())
                graph.neighbours.setdefault(a, set()).add(b)
                graph.neighbours.setdefault(b, set()).add(a)
            for ring in mol.GetRingInfo().AtomRings():
                maps = {int(mol.GetAtomWithIdx(i).GetAtomMapNum() or 0) for i in ring}
                maps.discard(0)
                if maps:
                    graph.rings.append(maps)
    finally:
        del blocker
    return graph


def split_reaction(reaction_smirks: str) -> Tuple[str, str]:
    core, _meta = split_cxsmiles_metadata(str(reaction_smirks or ""))
    if ">>" not in core:
        raise ValueError("reaction SMIRKS must contain '>>'")
    left, right = core.split(">>", 1)
    return left.strip(), right.strip()


def _push_atoms(electron_pushes: Any) -> Set[int]:
    atoms: Set[int] = set()
    if not electron_pushes:
        return atoms
    try:
        moves = normalize_electron_pushes(electron_pushes)
    except Exception:
        return atoms
    for move in moves:
        for attr in ("source_atom", "target_atom", "bond_start", "bond_end", "through_atom"):
            value = getattr(move, attr, None)
            if value is not None:
                try:
                    atoms.add(int(value))
                except (TypeError, ValueError):
                    continue
    return atoms


def _delta_atoms(bond_electron_deltas: Any) -> Tuple[Set[int], Set[int]]:
    """(all atoms with a non-zero ΔBE entry, atoms with a lone-pair entry)."""
    atoms: Set[int] = set()
    lone: Set[int] = set()
    if not isinstance(bond_electron_deltas, list):
        return atoms, lone
    for entry in bond_electron_deltas:
        if not isinstance(entry, dict):
            continue
        try:
            i, j, d = int(entry.get("map_i")), int(entry.get("map_j")), float(entry.get("delta") or 0)
        except (TypeError, ValueError):
            continue
        if d == 0:
            continue
        atoms.update((i, j))
        if i == j:
            lone.add(i)
    return atoms, lone


def _empty_focus(error: str, **ids: Any) -> Dict[str, Any]:
    return {
        "schema_version": REACTION_FOCUS_SCHEMA,
        "source_state_id": ids.get("source_state_id"),
        "target_state_id": ids.get("target_state_id"),
        "core_atom_ids": [],
        "context_atom_ids": [],
        "unchanged_atom_ids": [],
        "all_atom_ids": [],
        "unbalanced_atom_ids": [],
        "changed_bonds": [],
        "changed_formal_charges": [],
        "changed_lone_pairs": [],
        "changed_hydrogens": [],
        "electron_flow_atom_ids": [],
        "matrix_atom_ids": [],
        "context_radius": ids.get("context_radius", 1),
        "error": error,
    }


def build_reaction_focus(
    reaction_smirks: str,
    *,
    electron_pushes: Any = None,
    bond_electron_deltas: Any = None,
    context_radius: int = 1,
    source_state_id: Optional[str] = None,
    target_state_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Deterministic ``reaction_focus.v1`` for one mapped elementary step."""
    try:
        left_text, right_text = split_reaction(reaction_smirks)
        left, right = parse_mapped_side(left_text), parse_mapped_side(right_text)
    except Exception as exc:
        return _empty_focus(f"{type(exc).__name__}: {exc}", source_state_id=source_state_id,
                            target_state_id=target_state_id, context_radius=context_radius)

    all_maps: Set[int] = set(left.atoms) | set(right.atoms)
    both = set(left.atoms) & set(right.atoms)
    unbalanced = all_maps - both

    changed_bonds: List[Dict[str, Any]] = []
    bond_atoms: Set[int] = set()
    for pair in sorted(set(left.bonds) | set(right.bonds)):
        before = left.bonds.get(pair, 0.0)
        after = right.bonds.get(pair, 0.0)
        if abs(before - after) < 1e-6:
            continue
        changed_bonds.append({"atom_ids": [atom_id(pair[0]), atom_id(pair[1])], "order_before": before, "order_after": after})
        bond_atoms.update(pair)

    charge_atoms = {m for m in both if left.atoms[m].formal_charge != right.atoms[m].formal_charge}
    h_atoms = {m for m in both if left.atoms[m].total_h != right.atoms[m].total_h}
    lone_atoms = {m for m in both if abs(left.atoms[m].nonbonding_electrons - right.atoms[m].nonbonding_electrons) > 1e-6}
    flow_atoms = _push_atoms(electron_pushes) & all_maps
    delta_atoms, delta_lone = _delta_atoms(bond_electron_deltas)
    lone_atoms |= delta_lone & all_maps

    core = bond_atoms | charge_atoms | h_atoms | lone_atoms | flow_atoms | (delta_atoms & all_maps) | unbalanced

    # Context: BFS to ``context_radius`` over the union adjacency, then ring completion.
    adjacency: Dict[int, Set[int]] = {}
    for graph in (left, right):
        for m, nbrs in graph.neighbours.items():
            adjacency.setdefault(m, set()).update(nbrs)
    frontier = set(core)
    context: Set[int] = set()
    for _ in range(max(0, int(context_radius))):
        nxt: Set[int] = set()
        for m in frontier:
            nxt |= adjacency.get(m, set())
        nxt -= core
        context |= nxt
        frontier = nxt
    for ring in left.rings + right.rings:
        if ring & core:
            context |= ring - core
    unchanged = all_maps - core - context

    return {
        "schema_version": REACTION_FOCUS_SCHEMA,
        "source_state_id": source_state_id,
        "target_state_id": target_state_id,
        "core_atom_ids": _sorted_ids(core),
        "context_atom_ids": _sorted_ids(context),
        "unchanged_atom_ids": _sorted_ids(unchanged),
        "all_atom_ids": _sorted_ids(all_maps),
        "unbalanced_atom_ids": _sorted_ids(unbalanced),
        "changed_bonds": changed_bonds,
        "changed_formal_charges": _sorted_ids(charge_atoms),
        "changed_lone_pairs": _sorted_ids(lone_atoms),
        "changed_hydrogens": _sorted_ids(h_atoms),
        "electron_flow_atom_ids": _sorted_ids(flow_atoms),
        "matrix_atom_ids": _sorted_ids(core | context),
        "context_radius": int(context_radius),
        "error": None,
    }


__all__ = [
    "REACTION_FOCUS_SCHEMA",
    "AtomProps",
    "SideGraph",
    "atom_id",
    "build_reaction_focus",
    "parse_mapped_side",
    "split_reaction",
]
