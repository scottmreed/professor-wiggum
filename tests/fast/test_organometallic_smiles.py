"""Organometallic SMILES lint: dative-bond conventions for TM complexes."""

import pytest

from organometallic_chem.organometallic_smiles import (
    contains_transition_metal,
    group_number,
    lint_organometallic_smiles,
)

PPH3 = "P(c1ccccc1)(c1ccccc1)c1ccccc1"


@pytest.mark.parametrize(
    "smiles",
    [
        "CP(C)(C)->[Pd](Br)(c1ccccc1)<-P(C)(C)C",
        f"Cl[Rh](<-{PPH3})(<-{PPH3})<-{PPH3}",
        "[O+]#[C-]->[Cr](<-[C-]#[O+])(<-[C-]#[O+])(<-[C-]#[O+])(<-[C-]#[O+])<-[C-]#[O+]",
        "[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1",  # ionic Cp convention
        "Cl[Pt-2](Cl)(Cl)Cl",
        "[Cl-]->[Pd+2](<-[Cl-])(<-P(C)(C)C)<-P(C)(C)C",  # xyz2mol_tm style
        "Cl[Pd](Cl)<-n1ccccc1",
        "[Pd]N(C)C",  # amido: an X ligand, correctly a single bond
        "[Pd]O",  # hydroxo
        "Br[Fe](Br)Br",
        "[Pd]1CC1",  # eta2-alkene in its metallacyclopropane limit
    ],
)
def test_conventional_complexes_lint_clean(smiles):
    assert lint_organometallic_smiles(smiles) == []


@pytest.mark.parametrize(
    "smiles",
    [
        # Main-group organometallics are never ours to police (Mg, Li are not
        # transition metals), and a bare copper salt or cuprate has no donor.
        "C[Mg]Br",
        "CC[Mg]Br",
        "[Li]N(C(C)C)C(C)C",
        "C[Cu-]C.[Li+]",
        "[Cu]Br",
        "CCO",
        "",
        None,
    ],
)
def test_non_transition_metal_and_ligand_free_reagents_are_never_flagged(smiles):
    assert lint_organometallic_smiles(smiles) == []


def test_phosphine_on_an_ordinary_bond_is_flagged_with_the_phantom_h():
    problems = lint_organometallic_smiles(f"[Pd]({PPH3}){PPH3}")
    assert len(problems) == 2
    assert "phantom H" in problems[0]
    assert "[PH]" in problems[0]
    assert "->" in problems[0]


@pytest.mark.parametrize(
    ("smiles", "donor"),
    [
        ("[Pd]N(C)(C)C", "N"),  # RDKit would silently "clean up" this one
        ("[Pd]S(C)C", "S"),
        ("[Pd]n1ccccc1", "N"),
        ("[Pd][As](C)(C)C", "As"),
        ("[Pt]O(C)C", "O"),
    ],
)
def test_neutral_donors_on_ordinary_bonds_are_flagged(smiles, donor):
    problems = lint_organometallic_smiles(smiles)
    assert problems and f"neutral {donor} donor" in problems[0]


def test_haptic_cp_drawn_as_sigma_bonds_is_flagged():
    problems = lint_organometallic_smiles("[Fe]1234C5C1C2C3C45")
    assert any("eta5" in p and "[Fe+2].c1cc[cH-]c1" in p for p in problems)


def test_cxsmiles_coordination_extension_is_flagged():
    problems = lint_organometallic_smiles("[Fe]* |m:1:2.3.4.5.6|")
    assert any("CXSMILES" in p for p in problems)


def test_carbonyl_on_an_ordinary_bond_points_at_the_dative_form():
    problems = lint_organometallic_smiles("[Cr]C#O")
    assert any("[O+]#[C-]->[Cr]" in p for p in problems)


def test_group_numbers_cover_every_d_block_period():
    assert group_number(26) == 8  # Fe
    assert group_number(46) == 10  # Pd
    assert group_number(78) == 10  # Pt
    assert group_number(77) == 9  # Ir
    assert group_number(79) == 11  # Au
    assert group_number(74) == 6  # W
    assert group_number(57) is None  # La: f-block here
    assert group_number(12) is None  # Mg


def test_contains_transition_metal():
    assert contains_transition_metal("[Cu]Br")
    assert not contains_transition_metal("C[Mg]Br")
    assert not contains_transition_metal("not a smiles((")


def test_non_metal_radical_in_a_complex_is_flagged():
    from organometallic_chem.organometallic_smiles import lint_organometallic_smiles

    # A methyl written as [CH2]: the complex still counts, but the carbon is a radical.
    broken = "CP(C)(C)->[Pd](<-P(C)(C)C)[CH2]"
    assert any("unpaired electron" in p for p in lint_organometallic_smiles(broken))
    fixed = "CP(C)(C)->[Pd](<-P(C)(C)C)C"
    assert lint_organometallic_smiles(fixed) == []
    # The metal's own radical (RhCl(PR3)3) is a depiction artefact, not a lint error.
    assert lint_organometallic_smiles("Cl[Rh](<-P(C)(C)C)(<-P(C)(C)C)<-P(C)(C)C") == []


def test_dative_from_a_neutral_ring_carbon_is_flagged():
    from organometallic_chem.organometallic_smiles import lint_organometallic_smiles

    pph3 = "P(c1ccccc1)(c1ccccc1)c1ccccc1"
    wrong = f"{pph3}->[Pd]<-{pph3}"  # the first arrow starts at a phenyl carbon
    assert any("neutral carbon" in p for p in lint_organometallic_smiles(wrong))
    right = f"c1ccc(cc1)P(c1ccccc1)(c1ccccc1)->[Pd]<-{pph3}"
    assert lint_organometallic_smiles(right) == []
    # An alkene pi bond (one-arrow shorthand) and CO are legitimate carbon donors.
    assert lint_organometallic_smiles("Br[Pd](<-P(C)(C)C)(c1ccccc1)<-C=C") == []
    assert lint_organometallic_smiles("[O+]#[C-]->[Cr](<-[C-]#[O+])(<-[C-]#[O+])(<-[C-]#[O+])(<-[C-]#[O+])<-[C-]#[O+]") == []
