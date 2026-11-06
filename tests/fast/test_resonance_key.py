"""Species identity up to resonance (ported from ChemIllusion's mechanism_resonance_service)."""
from __future__ import annotations

import pytest

pytest.importorskip("rdkit")

from mechanistic_agent.resonance import (
    RESONANCE_KEY_VERSION,
    match_species,
    resonance_key,
    same_up_to_resonance,
)
from mechanistic_agent.skeleton_alignment import skeleton_species_keys


def test_version_is_pinned() -> None:
    assert RESONANCE_KEY_VERSION == "connectivity_h_per_atom_net_charge_radicals.v1"


@pytest.mark.parametrize(
    "a, b",
    [
        ("CC(=O)[O-]", "CC([O-])=O"),  # acetate
        ("C1=CC=CC=C1", "C1C=CC=CC=1"),  # benzene Kekulé forms
        ("C=C[CH2+]", "[CH2+]C=C"),  # allyl cation
        ("C=C(C)[O-]", "[CH2-]C(C)=O"),  # enolate: O- form vs carbanion form
        ("C[N+](=O)[O-]", "C[N+]([O-])=O"),  # nitromethane
    ],
)
def test_resonance_forms_share_a_key(a: str, b: str) -> None:
    assert resonance_key(a) is not None
    assert resonance_key(a) == resonance_key(b)
    assert same_up_to_resonance(a, b)


@pytest.mark.parametrize(
    "a, b",
    [
        ("CC(C)=O", "C=C(C)O"),  # tautomers: an H moved
        ("CC(C)=O", "CCC=O"),  # constitutional isomers
        ("CC(=O)O", "CC(=O)[O-]"),  # acid vs conjugate base
        ("C[CH2]", "C[CH2+]"),  # radical vs cation
        # Same net charge, H moved C3 -> O (alkoxide ketone vs enolate alcohol): a
        # stored aldol step. A SMILES writer that drops H counts below default valence
        # joined these; the key must keep every atom's H count.
        ("C=C(C)C(C[O-])C(C)=O", "C=C(C)/C(CO)=C(\\C)[O-]"),
        ("CC[O-].CO", "CCO.C[O-]"),  # which oxygen carries the charge/H
    ],
)
def test_non_resonance_pairs_differ(a: str, b: str) -> None:
    assert resonance_key(a) != resonance_key(b)
    assert not same_up_to_resonance(a, b)


def test_key_format_records_net_charge_and_radicals() -> None:
    assert resonance_key("CC(=O)[O-]").endswith("|q=-1|r=0")
    assert resonance_key("C[CH2]").endswith("|q=0|r=1")


def test_atom_maps_and_stereo_are_ignored() -> None:
    assert resonance_key("[CH3:1][C:2](=[O:3])[O-:4]") == resonance_key("CC(=O)[O-]")
    assert resonance_key("C[C@H](N)C(=O)O") == resonance_key("CC(N)C(=O)O")
    assert resonance_key("C/C=C/C") == resonance_key("CC=CC")


def test_explicit_hydrogen_atoms_fold_onto_their_atom() -> None:
    assert resonance_key("[H]OC(C)=O") == resonance_key("CC(=O)O")


def test_unsanitizable_species_still_get_a_key() -> None:
    # Pentavalent carbon fails sanitization; the tolerant parse still names it.
    assert resonance_key("C(C)(C)(C)(C)C") is not None


@pytest.mark.parametrize("bad", ["", None, "not a smiles", "C1CC"])
def test_unparseable_input_returns_none(bad) -> None:
    assert resonance_key(bad) is None
    assert same_up_to_resonance(bad, "CC") is False
    assert same_up_to_resonance("CC", bad) is False
    assert same_up_to_resonance(bad, bad) is False


def test_match_species_reports_exact_resonance_and_missing() -> None:
    result = match_species(
        expected=["CC(=O)[O-]", "[CH2-]C(C)=O", "CCO"],
        produced=["CC([O-])=O", "C=C(C)[O-]"],
    )
    assert result == {"exact": ["CC(=O)[O-]"], "resonance": ["[CH2-]C(C)=O"], "missing": ["CCO"]}


def test_skeleton_alignment_key_stays_tautomer_agnostic() -> None:
    # The path-alignment key deliberately joins tautomers; the resonance key does not.
    assert skeleton_species_keys("CC(C)=O") == skeleton_species_keys("C=C(C)O")
    assert resonance_key("CC(C)=O") != resonance_key("C=C(C)O")
