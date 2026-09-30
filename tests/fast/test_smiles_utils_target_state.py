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
