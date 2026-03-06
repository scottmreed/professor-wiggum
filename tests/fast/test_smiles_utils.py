from __future__ import annotations

import pytest

pytest.importorskip("rdkit")
from mechanistic_agent.smiles_utils import (
    remove_mapping_and_canonicalize,
    strip_atom_mapping_list,
    strip_atom_mapping_optional,
)


def test_remove_mapping_and_canonicalize_strips_atom_maps() -> None:
    assert remove_mapping_and_canonicalize("[CH3:1][OH:2]") == "CO"


def test_remove_mapping_and_canonicalize_keeps_unmapped_smiles_canonical() -> None:
    assert remove_mapping_and_canonicalize("OCC") == "CCO"


def test_remove_mapping_and_canonicalize_returns_original_for_invalid_input() -> None:
    raw = "not_a_smiles!!!"
    assert remove_mapping_and_canonicalize(raw) == raw


def test_strip_atom_mapping_helpers_preserve_none_and_order() -> None:
    assert strip_atom_mapping_optional(None) is None
    assert strip_atom_mapping_list(["[OH2:1]", "OCC"]) == ["O", "CCO"]
