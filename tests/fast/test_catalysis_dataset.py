"""Catalysis development set selection (PRD_catalysis_handling R5.1/R5.2)."""
from pathlib import Path

import pytest

from mechanistic_agent.catalysis_dataset import build_catalysis_cases, metal_active, organocatalyst, tm_family

PD_LOSS = (
    "[Cl:1][Pd:2]([Cl:3])([P:4]([CH3:5])([CH3:6])[CH3:7])[P:8]([CH3:9])([CH3:10])[CH3:11]"
    ">>[Cl:1][Pd:2]([Cl:3])[P:4]([CH3:5])([CH3:6])[CH3:7].[P:8]([CH3:9])([CH3:10])[CH3:11]"
)
FECL3_SPECTATOR = "[Fe:1]([Cl:2])([Cl:3])[Cl:4].[CH3:5][OH:6]>>[Fe:1]([Cl:2])([Cl:3])[Cl:4].[CH2:5]=[OH+:6]"


def test_metal_active_requires_a_bond_change_at_the_metal():
    assert metal_active([PD_LOSS])
    assert not metal_active([FECL3_SPECTATOR])


@pytest.mark.parametrize(
    "starting, family",
    [
        (["[Pd]", "Brc1ccccc1", "OB(O)c1ccccc1"], "suzuki"),
        (["[Pd]", "Brc1ccccc1", "C[Sn](C)(C)c1ccccc1"], "stille"),
        (["[Pd]", "Brc1ccccc1", "Cl[Zn]c1ccccc1"], "negishi"),
        (["[Pd]", "I[Cu]", "C#Cc1ccccc1", "Ic1ccccc1"], "sonogashira"),
        (["[Pd]", "Brc1ccccc1", "NC"], "other_pd"),
        (["[Ru]=C", "C=CCC=C"], "ruthenium"),
        (["I[Cu]", "Ic1ccccc1"], "copper"),
    ],
)
def test_tm_family(starting, family):
    assert tm_family(starting)[0] == family


def test_organocatalyst_needs_regeneration_and_heavy_atom_participation():
    dmap = "[CH3:1][N:2]([CH3:3])[c:4]1[cH:5][cH:6][n:7][cH:8][cH:9]1"
    acylium = "[CH3:1][N:2]([CH3:3])[c:4]1[cH:5][cH:6][n+:7]([C:10](=[O:11])[CH3:12])[cH:8][cH:9]1"
    case = {
        "starting_materials": [dmap, "[C:10](=[O:11])([CH3:12])[Cl:13]"],
        "products": [dmap, "[Cl-:13]"],
        "verified_mechanism": {"steps": [
            {"current_state": [dmap, "[C:10](=[O:11])([CH3:12])[Cl:13]"], "resulting_state": [acylium, "[Cl-:13]"]},
            {"current_state": [acylium, "[Cl-:13]"], "resulting_state": [dmap, "[Cl-:13]"]},
        ]},
    }
    assert organocatalyst(case) == "CN(C)c1ccncc1"
    spectator = dict(case, verified_mechanism={"steps": [
        {"current_state": [dmap, "CCO"], "resulting_state": [dmap, "CC=O"]},
    ]})
    assert organocatalyst(spectator) is None


@pytest.mark.parametrize("path", ["data/test/train.txt", "holdout/train.txt", "data/test.txt"])
def test_builder_refuses_non_train_sources(path):
    with pytest.raises(ValueError):
        build_catalysis_cases(Path(path), [], cache_path=Path("missing.sqlite"))


def test_excess_reagent_copy_is_not_an_organocatalyst():
    # Two butylamide equivalents: one (maps 1-5) attacks and is consumed, the other
    # (maps 11-15) is carried unchanged. The species "reappears" but no atom returns.
    reacting = "[CH3:1][CH2:2][CH2:3][CH2:4][NH-:5]"
    spare = "[CH3:11][CH2:12][CH2:13][CH2:14][NH-:15]"
    electrophile = "[CH3:20][Br:21]"
    product = "[CH3:1][CH2:2][CH2:3][CH2:4][NH:5][CH3:20]"
    case = {
        "starting_materials": [reacting, spare, electrophile],
        "products": [product, spare, "[Br-:21]"],
        "verified_mechanism": {"steps": [
            {"current_state": [reacting, spare, electrophile], "resulting_state": [product, spare, "[Br-:21]"]},
        ]},
    }
    assert organocatalyst(case) is None


def test_catalysis_holdout_builder_reads_only_the_test_split(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_catalysis_holdout", Path(__file__).resolve().parents[2] / "scripts" / "build_catalysis_holdout.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit):
        module.build(input_path=tmp_path / "train.txt", index_path=tmp_path / "i.jsonl",
                     cache_path=tmp_path / "c.sqlite", out_dir=tmp_path / "out", per_family_band=1, max_steps=22)
    assert module.SUITE == "catalysis_holdout"
    assert [b[0] for b in module.STEP_BANDS] == ["3-4", "5-8", "9-22"]
