"""Covalent (neutral-ligand, Green MLX) electron counting for TM complexes."""

import pytest

from organometallic_chem.organometallic_electron_count import (
    ElectronCountError,
    count_electrons,
    mlx_class,
    roman_numeral,
)

PPH3 = "P(c1ccccc1)(c1ccccc1)c1ccccc1"
PME3 = "P(C)(C)C"
CO = "[C-]#[O+]"


def _six_co(metal):
    return f"[O+]#[C-]->[{metal}](<-{CO})(<-{CO})(<-{CO})(<-{CO})<-{CO}"


@pytest.mark.parametrize(
    ("name", "smiles", "metal", "ox", "d", "ve", "cn"),
    [
        ("Pd(PPh3)4", f"[Pd](<-{PPH3})(<-{PPH3})(<-{PPH3})<-{PPH3}", "Pd", 0, 10, 18, 4),
        ("Pd(PMe3)2", f"[Pd](<-{PME3})<-{PME3}", "Pd", 0, 10, 14, 2),
        ("Pd(PPh3)2", f"[Pd](<-{PPH3})<-{PPH3}", "Pd", 0, 10, 14, 2),
        (
            "trans-PdBr(Ph)(PMe3)2 (oxidative-addition product)",
            f"[Pd](<-{PME3})(c1ccccc1)(<-{PME3})Br",
            "Pd", 2, 8, 16, 4,
        ),
        ("Wilkinson's catalyst RhCl(PPh3)3", f"Cl[Rh](<-{PPH3})(<-{PPH3})<-{PPH3}", "Rh", 1, 8, 16, 4),
        ("ferrocene (ionic Cp convention)", "[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1", "Fe", 2, 6, 18, 6),
        ("Cr(CO)6", _six_co("Cr"), "Cr", 0, 6, 18, 6),
        ("W(CO)6 (Z > 54)", _six_co("W"), "W", 0, 6, 18, 6),
        ("[PtCl4]2-", "Cl[Pt-2](Cl)(Cl)Cl", "Pt", 2, 8, 16, 4),
        ("Vaska's complex", f"Cl[Ir](<-{CO})(<-{PPH3})<-{PPH3}", "Ir", 1, 8, 16, 4),
        ("Au(PPh3)Cl", f"Cl[Au]<-{PPH3}", "Au", 1, 10, 14, 2),
        ("trans-PdH(Br)(PMe3)2 hydride", f"[PdH](Br)(<-{PME3})<-{PME3}", "Pd", 2, 8, 16, 4),
    ],
)
def test_reference_complexes(name, smiles, metal, ox, d, ve, cn):
    count = count_electrons(smiles)
    assert (count.metal, count.oxidation_state, count.d_electrons, count.valence_electrons) == (
        metal, ox, d, ve,
    ), name
    assert count.coordination_number == cn, name


def test_ligand_classification_for_oxidative_addition_product():
    count = count_electrons(f"[Pd](<-{PME3})(c1ccccc1)(<-{PME3})Br")
    classes = sorted(ligand.ligand_class for ligand in count.ligands)
    assert classes == ["L", "L", "X", "X"]
    assert count.summary == "Pd(II), d8, 16 e"


def test_ionic_cp_is_eta5_l2x():
    count = count_electrons("[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1")
    assert [ligand.ligand_class for ligand in count.ligands] == ["L2X", "L2X"]
    assert all(ligand.hapticity == 5 and ligand.attachment == "ionic" for ligand in count.ligands)
    assert count.charge == 0


@pytest.mark.parametrize(
    "smiles",
    [
        "Cl[Pt-2](Cl)(Cl)Cl",  # covalent X ligands, charge on the metal
        "[Pt+2].[Cl-].[Cl-].[Cl-].[Cl-]",  # ionic fragments
        "[Cl-]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]",  # xyz2mol_tm: dative, OS on the metal
        "[K+].[K+].Cl[Pt-2](Cl)(Cl)Cl",  # counter-ions excluded from the complex charge
    ],
)
def test_every_charge_convention_gives_the_same_count(smiles):
    count = count_electrons(smiles)
    assert (count.oxidation_state, count.d_electrons, count.valence_electrons, count.charge) == (
        2, 8, 16, -2,
    )


def test_xyz2mol_tm_output_style_metal_charge_is_the_oxidation_state():
    """xyz2mol_tm writes every metal-donor bond dative and puts the proposed
    oxidation state on the metal as its formal charge, with anionic X ligands
    as charged atoms. The metal charge (+2) is not the complex charge (0)."""
    bonded = count_electrons("[Cl-]->[Pd+2](<-[Cl-])(<-P(C)(C)C)<-P(C)(C)C")
    separate = count_electrons("[Pd+2].[Cl-].[Cl-]")
    aryl = count_electrons("CP(C)(C)->[Pd+2](<-[Br-])(<-[c-]1ccccc1)<-P(C)(C)C")
    assert (bonded.oxidation_state, bonded.charge, bonded.valence_electrons) == (2, 0, 16)
    assert sorted(lig.ligand_class for lig in bonded.ligands) == ["L", "L", "X", "X"]
    assert (separate.oxidation_state, separate.valence_electrons) == (2, 12)
    assert aryl.summary == "Pd(II), d8, 16 e"


def test_xyz2mol_tm_haptic_cp_written_as_five_dative_bonds():
    # Ferrocene as xyz2mol_tm emits it: every ring carbon "->" Fe(+2).
    smiles = "c12->[Fe+2]3456789(<-c1c->3[cH-]->4c->52)<-c1c->6c->7[cH-]->8c->91"
    count = count_electrons(smiles)
    assert count.summary == "Fe(II), d6, 18 e"
    assert [(lig.ligand_class, lig.hapticity) for lig in count.ligands] == [("L2X", 5), ("L2X", 5)]
    assert count.coordination_number == 6


def test_eta2_alkene_on_two_dative_bonds_is_one_l_ligand():
    count = count_electrons("Cl[Pt-]1(Cl)(Cl)<-C=C->1")  # Zeise's anion
    assert count.summary == "Pt(II), d8, 16 e"
    assert count.coordination_number == 4


def test_alkylidene_is_x2_under_the_covalent_convention():
    count = count_electrons(f"Cl[Ru](Cl)(=Cc1ccccc1)(<-{PME3})<-{PME3}")
    assert count.summary == "Ru(IV), d4, 16 e"


def test_charge_separated_dative_counts_as_l():
    count = count_electrons("[NH3+][Pt-](Cl)(Cl)<-N")
    assert count.summary == "Pt(II), d8, 16 e"


@pytest.mark.parametrize(
    ("smiles", "message"),
    [
        (f"[Pd]({PPH3}){PPH3}", "phantom H"),
        ("CCO", "exactly one transition-metal"),
        ("[Pd]<-P(C)(C)C.[Pd]", "exactly one transition-metal"),
        ("[Pd].CCO", "not bonded to the metal"),
        ("[La+3].[Cl-].[Cl-].[Cl-]", "f-block"),
        ("", "required"),
    ],
)
def test_uncountable_inputs_raise(smiles, message):
    with pytest.raises(ElectronCountError, match=message):
        count_electrons(smiles)


def test_helpers():
    assert mlx_class(2, 1) == "L2X"
    assert mlx_class(0, 2) == "X2"
    assert roman_numeral(0) == "0"
    assert roman_numeral(4) == "IV"
    assert roman_numeral(-1) == "-I"
