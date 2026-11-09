"""Covalent <-> dative notation bridge (PRD_catalysis_handling R1b)."""
from pathlib import Path

import pytest

from organometallic_chem.notation import canonical_unsanitized, to_covalent, to_house_convention
from organometallic_chem.organometallic_electron_count import count_electrons, try_count_electrons
from organometallic_chem.organometallic_smiles import lint_organometallic_smiles

# 400 transition-metal species drawn as FlowER draws them (covalent M-L bonds),
# sampled from flower_new_dataset/train.txt only.
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "flower_train_tm_species.txt"


@pytest.mark.parametrize(
    "covalent, metal, oxidation_state, electrons",
    [
        ("[Pd]([P](C)(C)C)[P](C)(C)C", "Pd", 0, 14),
        ("Cl[Pd](Cl)([P](C)(C)C)[P](C)(C)C", "Pd", 2, 16),
        ("c1ccccc1[Pd](Br)([P](C)(C)C)[P](C)(C)C", "Pd", 2, 16),
        ("[Cr](C#O)(C#O)(C#O)(C#O)(C#O)C#O", "Cr", 0, 18),
        ("Cl[Pd]1(Cl)n2ccccc2-c2cccc[n]21", "Pd", 2, 16),
        ("CC(=O)O[Pd]OC(C)=O", "Pd", 2, 12),
    ],
)
def test_covalent_drawings_count_after_conversion(covalent, metal, oxidation_state, electrons):
    result = to_house_convention(covalent)
    assert result.ok, result.ambiguous
    assert lint_organometallic_smiles(result.smiles) == []
    count = count_electrons(result.smiles)
    assert (count.metal, count.oxidation_state, count.valence_electrons) == (metal, oxidation_state, electrons)


def test_atom_maps_survive_conversion():
    mapped = "[Cl:1][Pd:2]([Cl:3])([P:4]([CH3:5])([CH3:6])[CH3:7])[P:8]([CH3:9])([CH3:10])[CH3:11]"
    result = to_house_convention(mapped)
    assert result.ok
    assert "<-[P:4]" in result.smiles and "<-[P:8]" in result.smiles
    assert len(result.changes) == 2


@pytest.mark.parametrize(
    "smiles, reason",
    [
        ("C1C[Pd]1", "eta2-alkene vs metallacyclopropane"),
        ("Cl[Pd]1Cl[Pd](Cl)Cl1", "metal atoms"),
    ],
)
def test_ambiguous_forms_are_reported_not_guessed(smiles, reason):
    result = to_house_convention(smiles)
    assert not result.ok
    assert any(reason in item for item in result.ambiguous)


def test_species_without_metal_pass_through():
    result = to_house_convention("CCO")
    assert result.ok and result.smiles == "CCO" and not result.changes


def test_house_convention_input_is_left_alone():
    house = "CP(C)(C)->[Pd]<-P(C)(C)C"
    result = to_house_convention(house)
    assert result.ok and not result.changes
    assert count_electrons(result.smiles).valence_electrons == 14


def test_flower_train_species_round_trip_and_count():
    species = [line for line in FIXTURE.read_text().splitlines() if line.strip()]
    assert len(species) == 400
    round_trip = clean = counted = 0
    for smiles in species:
        result = to_house_convention(smiles)
        if result.smiles is not None and to_covalent(result.smiles).smiles == canonical_unsanitized(smiles):
            round_trip += 1
        if result.ok:
            clean += 1
            if try_count_electrons(result.smiles)[0] is not None:
                counted += 1
    # Measured on 10,000 train species: 99.6% round trip, 99.4% clean, all clean ones count.
    assert round_trip / len(species) >= 0.98
    assert clean / len(species) >= 0.98
    assert counted == clean
