import pytest

from mechanistic_agent import tools

pytest.importorskip("rdkit")


def test_phenol_not_labeled_as_alcohol():
    counts = tools.count_functional_groups("c1ccccc1O")
    assert counts.get("phenol", 0) >= 1
    assert counts.get("alcohol", 0) == 0


def test_carboxylic_acid_not_labeled_as_alcohol():
    counts = tools.count_functional_groups("CC(=O)O")
    assert counts.get("carboxylic_acid", 0) >= 1
    assert counts.get("alcohol", 0) == 0


def test_sulfoxide_and_sulfone_detected():
    sulfoxide = tools.count_functional_groups("CS(=O)C")
    sulfone = tools.count_functional_groups("CS(=O)(=O)C")

    assert sulfoxide.get("sulfoxide", 0) >= 1
    assert sulfone.get("sulfone", 0) >= 1


def test_niche_group_does_not_block_common_descriptors():
    groups = tools.find_functional_groups("O=[N+]([O-])c1ccccc1")
    assert "nitro" in groups
    assert "aromatic_ring" in groups


def test_multi_fg_assignment_keeps_distinct_sites():
    counts = tools.count_functional_groups("CC(=O)Oc1ccccc1C(=O)O")
    assert counts.get("ester", 0) >= 1
    assert counts.get("carboxylic_acid", 0) >= 1
    assert counts.get("aromatic_ring", 0) >= 1
