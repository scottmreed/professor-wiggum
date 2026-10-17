"""CXSMILES species (radicals, coordinate bonds) parse in the counting helpers.

repair-smiles writes radicals and dative bonds with a CXSMILES extension; the counting and
charge helpers used to reject the "|...|" text, so every radical step, metal redox step and
M-CSA dative-bond state failed atom balance with "Invalid SMILES string".
"""
import pytest

from mechanistic_agent import tools


@pytest.mark.parametrize("smiles", ["[CH3] |^1:0|", "[Mg] |^2:0|", "[OH-][Zn+2] |C:0.0|", "O->[Zn+2]", "[Cu+2]"])
def test_cxsmiles_species_parse(smiles):
    assert tools._mol_from_smiles(smiles) is not None


def test_cxsmiles_radical_keeps_its_atoms_in_counts():
    assert tools._atom_counter(["[CH3] |^1:0|"])["C"] == 1


def test_english_words_still_rejected():
    with pytest.raises(tools.ToolRuntimeError):
        tools._mol_from_smiles("classification |x|")
