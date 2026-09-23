"""ReactionFocus v1 (Observatory PRD §9): one deterministic active-region mask per elementary step.

Atom ids are ``a<map>`` from the candidate's mapped reaction SMIRKS. Core =
atoms whose bonds, formal charge, hydrogen count or lone pairs change, or that
take part in an electron push. Context = graph radius 1 around the core on the
reactant side, with whole rings kept when the core touches a ring.
"""
from __future__ import annotations

import pytest

pytest.importorskip("rdkit")

from mechanistic_agent.core.reaction_focus import REACTION_FOCUS_SCHEMA, build_reaction_focus

SN2 = "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3]"
SN2_PUSHES = [
    {"kind": "lone_pair", "source_atom": "4", "target_atom": "2", "electrons": 2},
    {"kind": "sigma_bond", "source_bond": ["2", "3"], "through_atom": "3", "target_atom": "3", "electrons": 2},
]
SN2_DELTAS = [
    {"map_i": 2, "map_j": 3, "delta": -2, "type": "bond"},
    {"map_i": 3, "map_j": 3, "delta": 2, "type": "lone_pair"},
    {"map_i": 4, "map_j": 2, "delta": 2, "type": "bond"},
    {"map_i": 4, "map_j": 4, "delta": -2, "type": "lone_pair"},
]
HEXYL = (
    "[CH3:1][CH2:2][CH2:3][CH2:4][CH2:5][CH2:6][Br:7].[Cl-:8]"
    ">>[CH3:1][CH2:2][CH2:3][CH2:4][CH2:5][CH2:6][Cl:8].[Br-:7]"
)
RING = (
    "[CH2:1]1[CH2:2][CH2:3][CH2:4][CH2:5][CH:6]1[Br:7].[OH-:8]"
    ">>[CH2:1]1[CH2:2][CH2:3][CH2:4][CH2:5][CH:6]1[OH:8].[Br-:7]"
)


def test_sn2_focus_core_context_and_changes() -> None:
    focus = build_reaction_focus(SN2, electron_pushes=SN2_PUSHES, bond_electron_deltas=SN2_DELTAS,
                                 source_state_id="s0", target_state_id="s1")
    assert focus["schema_version"] == REACTION_FOCUS_SCHEMA == "reaction_focus.v1"
    assert focus["source_state_id"] == "s0" and focus["target_state_id"] == "s1"
    assert focus["core_atom_ids"] == ["a2", "a3", "a4"]
    assert focus["context_atom_ids"] == ["a1"]
    assert focus["unchanged_atom_ids"] == []
    assert focus["changed_bonds"] == [
        {"atom_ids": ["a2", "a3"], "order_before": 1.0, "order_after": 0.0},
        {"atom_ids": ["a2", "a4"], "order_before": 0.0, "order_after": 1.0},
    ]
    assert focus["changed_formal_charges"] == ["a3", "a4"]
    assert focus["changed_lone_pairs"] == ["a3", "a4"]
    assert focus["changed_hydrogens"] == []
    assert focus["electron_flow_atom_ids"] == ["a2", "a3", "a4"]
    assert focus["matrix_atom_ids"] == ["a1", "a2", "a3", "a4"]
    assert focus["all_atom_ids"] == ["a1", "a2", "a3", "a4"]


def test_remote_atoms_are_unchanged_and_context_is_radius_one() -> None:
    focus = build_reaction_focus(HEXYL)
    assert focus["core_atom_ids"] == ["a6", "a7", "a8"]
    assert focus["context_atom_ids"] == ["a5"]
    assert focus["unchanged_atom_ids"] == ["a1", "a2", "a3", "a4"]
    assert focus["matrix_atom_ids"] == ["a5", "a6", "a7", "a8"]


def test_ring_touched_by_core_is_kept_whole_in_context() -> None:
    focus = build_reaction_focus(RING)
    assert focus["core_atom_ids"] == ["a6", "a7", "a8"]
    assert set(focus["context_atom_ids"]) == {"a1", "a2", "a3", "a4", "a5"}
    assert focus["unchanged_atom_ids"] == []


def test_focus_is_deterministic_and_independent_of_species_order() -> None:
    a = build_reaction_focus(SN2, electron_pushes=SN2_PUSHES)
    b = build_reaction_focus("[Cl-:4].[CH3:1][CH2:2][Br:3]>>[Br-:3].[CH3:1][CH2:2][Cl:4]", electron_pushes=SN2_PUSHES)
    assert a == b


def test_focus_without_pushes_or_deltas_still_finds_core_from_smirks() -> None:
    focus = build_reaction_focus(SN2)
    assert focus["core_atom_ids"] == ["a2", "a3", "a4"]
    assert focus["electron_flow_atom_ids"] == []


def test_hydrogen_count_change_puts_atom_in_core() -> None:
    # protonation: water lone pair onto a proton-bearing oxonium; O gains an H
    smirks = "[OH2:1].[OH3+:2]>>[OH3+:1].[OH2:2]"
    focus = build_reaction_focus(smirks)
    assert focus["core_atom_ids"] == ["a1", "a2"]
    assert focus["changed_hydrogens"] == ["a1", "a2"]
    assert focus["changed_formal_charges"] == ["a1", "a2"]


def test_atom_missing_on_one_side_is_reported_not_crashed() -> None:
    focus = build_reaction_focus("[CH3:1][Br:2]>>[CH3:1]")
    assert "a2" in focus["core_atom_ids"]
    assert focus["unbalanced_atom_ids"] == ["a2"]


def test_invalid_smirks_returns_empty_focus_with_error() -> None:
    focus = build_reaction_focus("not a smirks")
    assert focus["core_atom_ids"] == []
    assert focus["error"]
