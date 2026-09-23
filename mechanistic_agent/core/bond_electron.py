"""Bond-electron matrices BE(t), ΔBE, BE(t+1) (Observatory PRD §10).

Convention ``ugi_flower_kekule_v1``:

* rows/columns are atom ids ``a<map>`` in the requested order (default: all
  mapped atoms sorted by map number);
* off-diagonal ``(i, j)``: shared bonding electrons between the two mapped
  atoms = 2 × Kekulé bond order (aromatic input is Kekulized first; 0 when
  the atoms are not bonded);
* diagonal ``(i, i)``: non-bonding valence electrons of the heavy atom with
  hydrogens folded in: ``outer_electrons − formal_charge − Σ bond orders − n_H``
  (radical electrons are therefore included);
* ``delta = after − before``; ``electron_delta_sum = Σ_diag Δ + Σ_{i<j} Δ``,
  which is exactly zero when the projected atoms exchange electrons only
  among themselves;
* ``conserved`` requires the sum to be zero **and** every atom with a changed
  row to be inside the projection (a partial projection can sum to zero by
  coincidence);
* unmapped atoms are not rows; a bond to one still counts in the diagonal.

The matrices are computed from the same mapped-side parse as
``core/reaction_focus.py`` so the two projections agree atom-for-atom.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .reaction_focus import SideGraph, atom_id, parse_mapped_side, split_reaction

BOND_ELECTRON_SCHEMA = "bond_electron_view.v1"
BE_CONVENTION = "ugi_flower_kekule_v1"


def _num(value: float) -> Any:
    return int(value) if abs(value - round(value)) < 1e-9 else round(value, 3)


def _matrix(graph: SideGraph, maps: Sequence[int]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for i in maps:
        row: List[Any] = []
        for j in maps:
            if i == j:
                props = graph.atoms.get(i)
                row.append(_num(props.nonbonding_electrons) if props is not None else 0)
            else:
                order = graph.bonds.get((min(i, j), max(i, j)), 0.0)
                row.append(_num(2.0 * order))
        rows.append(row)
    return rows


def _error_view(error: str, atom_ids: Optional[Sequence[str]]) -> Dict[str, Any]:
    return {
        "schema_version": BOND_ELECTRON_SCHEMA,
        "convention": BE_CONVENTION,
        "atom_ids": [],
        "before": [],
        "delta": [],
        "after": [],
        "changed_cells": [],
        "electron_delta_sum": 0,
        "conserved": False,
        "missing_changed_atom_ids": [],
        "full_atom_count": 0,
        "is_focus_projection": atom_ids is not None,
        "error": error,
    }


def build_bond_electron_view(
    reaction_smirks: str,
    *,
    atom_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """``bond_electron_view.v1`` for a mapped elementary step, optionally projected."""
    try:
        left_text, right_text = split_reaction(reaction_smirks)
        left, right = parse_mapped_side(left_text), parse_mapped_side(right_text)
    except Exception as exc:
        return _error_view(f"{type(exc).__name__}: {exc}", atom_ids)

    all_maps = sorted(set(left.atoms) | set(right.atoms))
    full_before = _matrix(left, all_maps)
    full_after = _matrix(right, all_maps)
    changed_maps = {
        all_maps[i]
        for i in range(len(all_maps))
        for j in range(len(all_maps))
        if full_before[i][j] != full_after[i][j]
    }

    if atom_ids is None:
        maps = list(all_maps)
        is_projection = False
    else:
        try:
            maps = [int(str(a).lstrip("a")) for a in atom_ids]
        except ValueError as exc:
            return _error_view(f"bad atom id: {exc}", atom_ids)
        unknown = [m for m in maps if m not in set(all_maps)]
        if unknown:
            return _error_view(f"atom ids not in reaction: {[atom_id(m) for m in unknown]}", atom_ids)
        # "Projection" means the rows were chosen by a ReactionFocus, even when
        # the focus happens to cover every mapped atom of a small step.
        is_projection = True

    before = _matrix(left, maps)
    after = _matrix(right, maps)
    delta = [[_num(after[i][j] - before[i][j]) for j in range(len(maps))] for i in range(len(maps))]

    changed_cells: List[Dict[str, Any]] = []
    total = 0.0
    for i in range(len(maps)):
        for j in range(i, len(maps)):
            d = delta[i][j]
            if d != 0:
                changed_cells.append({
                    "atom_ids": [atom_id(maps[i]), atom_id(maps[j])],
                    "before": before[i][j],
                    "after": after[i][j],
                    "delta": d,
                })
                total += float(d)
    missing = [atom_id(m) for m in sorted(changed_maps - set(maps))]
    return {
        "schema_version": BOND_ELECTRON_SCHEMA,
        "convention": BE_CONVENTION,
        "atom_ids": [atom_id(m) for m in maps],
        "before": before,
        "delta": delta,
        "after": after,
        "changed_cells": changed_cells,
        "electron_delta_sum": _num(total),
        "conserved": abs(total) < 1e-9 and not missing,
        "missing_changed_atom_ids": missing,
        "full_atom_count": len(all_maps),
        "is_focus_projection": is_projection,
        "error": None,
    }


__all__ = ["BE_CONVENTION", "BOND_ELECTRON_SCHEMA", "build_bond_electron_view"]
