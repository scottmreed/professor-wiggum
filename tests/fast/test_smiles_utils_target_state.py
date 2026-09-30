"""Completion tolerates leftover proton carriers and conjugate forms, not real extras."""

from mechanistic_agent.smiles_utils import assess_target_product_state

PRODUCT = "CC(=O)NNc1ccc([N+](=O)[O-])cc1"
STARTING = ["CC(=O)O", "NNc1ccc([N+](=O)[O-])cc1"]


def _check(resulting):
    return assess_target_product_state(
        current_state=["CC(O)(O)NNc1ccc([N+](=O)[O-])cc1"],
        resulting_state=resulting,
        target_products=[PRODUCT],
        starting_materials=STARTING,
    )


def test_stray_hydronium_does_not_block_completion() -> None:
    result = _check([PRODUCT, "[OH3+]"])
    assert result["contains_target_product"] is True
    assert result["tolerated_species"] == ["[OH3+]"]


def test_conjugate_base_of_starting_material_does_not_block_completion() -> None:
    assert _check([PRODUCT, "CC(=O)[O-]"])["contains_target_product"] is True


def test_unrelated_extra_species_still_blocks_completion() -> None:
    result = _check([PRODUCT, "CCCC"])
    assert result["contains_target_product"] is False
    assert result["tolerated_species"] == []


def test_water_target_present_as_hydronium_counts_as_reached() -> None:
    result = assess_target_product_state(
        current_state=["CC(=O)NNC(=[OH+])c1ccccc1", "O"],
        resulting_state=["CC(=O)NNC(=O)c1ccccc1", "[OH3+]"],
        target_products=["CC(=O)NNC(=O)c1ccccc1", "O"],
        starting_materials=["CC(=O)NN", "O=C(O)c1ccccc1"],
    )
    assert result["contains_target_product"] is True
    assert result["targets_as_conjugate"] == ["O"]
    assert result["missing_target_products"] == []


def test_protonated_main_product_is_not_the_product() -> None:
    # hard rerun of flower_002647 stopped one step early on the protonated amide.
    result = assess_target_product_state(
        current_state=["CC(=O)NNC(O)(O)c1ccccc1"],
        resulting_state=["CC(=O)NNC(=[OH+])c1ccccc1", "[OH-]"],
        target_products=["CC(=O)NNC(=O)c1ccccc1", "O"],
        starting_materials=["CC(=O)NN", "O=C(O)c1ccccc1"],
    )
    assert result["contains_target_product"] is False
    assert "CC(=O)NNC(=O)c1ccccc1" in result["missing_target_products"]
