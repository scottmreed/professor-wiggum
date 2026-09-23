"""Bond-electron matrices BE(t), ΔBE, BE(t+1) (Observatory PRD §10), convention ``ugi_flower_kekule_v1``.

Off-diagonal (i, j): shared bonding electrons = 2 × Kekulé bond order.
Diagonal (i, i): non-bonding valence electrons on the heavy atom, hydrogens
folded in (V_outer − charge − Σ heavy bond orders − n_H). Electron
conservation is Σ_diag ΔBE + Σ_{i<j} ΔBE == 0 over the projected atoms.
"""
from __future__ import annotations

import pytest

pytest.importorskip("rdkit")

from mechanistic_agent.core.bond_electron import BE_CONVENTION, build_bond_electron_view

SN2 = "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3]"


def _cell(view, matrix, i, j):
    ids = view["atom_ids"]
    return view[matrix][ids.index(i)][ids.index(j)]


def test_full_sn2_matrices_follow_the_convention() -> None:
    view = build_bond_electron_view(SN2)
    assert view["schema_version"] == "bond_electron_view.v1"
    assert view["convention"] == BE_CONVENTION == "ugi_flower_kekule_v1"
    assert view["atom_ids"] == ["a1", "a2", "a3", "a4"]
    assert view["is_focus_projection"] is False
    assert view["full_atom_count"] == 4
    # shared electrons: C2–Br3 single bond breaks, C2–Cl4 forms
    assert _cell(view, "before", "a2", "a3") == 2 and _cell(view, "after", "a2", "a3") == 0
    assert _cell(view, "delta", "a2", "a3") == -2
    assert _cell(view, "before", "a2", "a4") == 0 and _cell(view, "after", "a2", "a4") == 2
    assert _cell(view, "delta", "a2", "a4") == 2
    # non-bonding electrons: Br 6 → 8, Cl⁻ 8 → 6, methyl C stays 0, CH2 stays 0
    assert _cell(view, "before", "a3", "a3") == 6 and _cell(view, "after", "a3", "a3") == 8
    assert _cell(view, "before", "a4", "a4") == 8 and _cell(view, "after", "a4", "a4") == 6
    assert _cell(view, "before", "a1", "a1") == 0 and _cell(view, "delta", "a1", "a1") == 0
    assert _cell(view, "delta", "a2", "a2") == 0
    assert view["electron_delta_sum"] == 0
    assert view["conserved"] is True


def test_matrices_are_symmetric_and_delta_is_after_minus_before() -> None:
    view = build_bond_electron_view(SN2)
    n = len(view["atom_ids"])
    for m in ("before", "after", "delta"):
        for i in range(n):
            for j in range(n):
                assert view[m][i][j] == view[m][j][i]
    for i in range(n):
        for j in range(n):
            assert view["delta"][i][j] == view["after"][i][j] - view["before"][i][j]


def test_focus_projection_keeps_requested_order_and_reports_full_count() -> None:
    view = build_bond_electron_view(SN2, atom_ids=["a2", "a3", "a4"])
    assert view["atom_ids"] == ["a2", "a3", "a4"]
    assert view["is_focus_projection"] is True
    assert view["full_atom_count"] == 4
    assert len(view["before"]) == 3 and len(view["before"][0]) == 3
    assert view["electron_delta_sum"] == 0
    assert view["changed_cells"] == [
        {"atom_ids": ["a2", "a3"], "before": 2, "after": 0, "delta": -2},
        {"atom_ids": ["a2", "a4"], "before": 0, "after": 2, "delta": 2},
        {"atom_ids": ["a3", "a3"], "before": 6, "after": 8, "delta": 2},
        {"atom_ids": ["a4", "a4"], "before": 8, "after": 6, "delta": -2},
    ]


def test_aromatic_input_is_kekulized_to_integer_shared_electrons() -> None:
    smirks = "[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[Br:7].[OH-:8]>>[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[OH:8].[Br-:7]"
    view = build_bond_electron_view(smirks)
    ring_ids = [f"a{i}" for i in range(1, 7)]
    ring_cells = {_cell(view, "before", a, b) for a in ring_ids for b in ring_ids if a != b}
    assert ring_cells <= {0, 2, 4}, "Kekulé ring bonds are single (2) or double (4), never aromatic 1.5"
    assert view["electron_delta_sum"] == 0


def test_projection_missing_a_changed_atom_is_flagged_not_conserved() -> None:
    # C2 + Br3 alone do balance (Br keeps the C–Br pair), so the sum is zero by
    # coincidence; conservation must still be denied because Cl4 changed too.
    view = build_bond_electron_view(SN2, atom_ids=["a2", "a3"])
    assert view["electron_delta_sum"] == 0
    assert view["missing_changed_atom_ids"] == ["a4"]
    assert view["conserved"] is False
    lone = build_bond_electron_view(SN2, atom_ids=["a3"])
    assert lone["electron_delta_sum"] == 2 and lone["conserved"] is False


def test_invalid_smirks_returns_error_view() -> None:
    view = build_bond_electron_view("garbage")
    assert view["atom_ids"] == [] and view["error"]
