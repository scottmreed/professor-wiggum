"""Run completion requires every target product; the scorer credits partial byproduct completion."""
from __future__ import annotations

from mechanistic_agent.scoring import score_snapshot_against_known
from mechanistic_agent.smiles_utils import assess_target_product_state


def test_assess_target_product_state_requires_all_products_and_no_extras() -> None:
    # Boc protection: carbamate formed, but tert-butyl hydrogen carbonate has
    # not yet decarboxylated to tBuOH + CO2 -> not complete.
    partial = assess_target_product_state(
        current_state=["CCOC(=O)C1CCC[NH+](C(=O)OC(C)(C)C)C1", "CC(C)(C)OC(=O)[O-]", "C1CCOC1"],
        resulting_state=["CCOC(=O)C1CCCN(C(=O)OC(C)(C)C)C1", "CC(C)(C)OC(=O)O", "C1CCOC1"],
        target_products=["CCOC(=O)C1CCCN(C(=O)OC(C)(C)C)C1", "C1CCOC1", "CC(C)(C)O", "O=C=O"],
        starting_materials=["CC(C)(C)OC(=O)OC(=O)OC(C)(C)C", "C1CCOC1", "CCOC(=O)C1CCCNC1"],
    )
    assert partial["contains_primary_target_product"] is True
    assert partial["contains_target_product"] is False
    assert partial["all_targets_reached"] is False
    assert set(partial["missing_target_products"]) == {"CC(C)(C)O", "O=C=O"}
    assert partial["unexpected_species"] == ["CC(C)(C)OC(=O)O"]

    complete = assess_target_product_state(
        current_state=["CCOC(=O)C1CCCN(C(=O)OC(C)(C)C)C1", "CC(C)(C)[O-]", "O=C=O", "C1CCOC1"],
        resulting_state=["CCOC(=O)C1CCCN(C(=O)OC(C)(C)C)C1", "CC(C)(C)O", "O=C=O", "C1CCOC1"],
        target_products=["CCOC(=O)C1CCCN(C(=O)OC(C)(C)C)C1", "C1CCOC1", "CC(C)(C)O", "O=C=O"],
        starting_materials=["CC(C)(C)OC(=O)OC(=O)OC(C)(C)C", "C1CCOC1", "CCOC(=O)C1CCCNC1"],
    )
    assert complete["contains_target_product"] is True
    assert complete["unexpected_species"] == []


def test_assess_target_product_state_allows_declared_spectators_and_leftover_reagent() -> None:
    result = assess_target_product_state(
        current_state=["O=[C+]C=Cc1cncc(Br)c1", "O=S=O", "[Cl-]", "Cl", "O=S(Cl)Cl", "CCN(CC)CC"],
        resulting_state=["O=C(Cl)C=Cc1cncc(Br)c1", "O=S=O", "Cl", "O=S(Cl)Cl", "CCN(CC)CC"],
        target_products=["O=C(Cl)C=Cc1cncc(Br)c1", "Cl", "O=S=O"],
        starting_materials=["O=C(O)C=Cc1cncc(Br)c1", "O=S(Cl)Cl"],
        allowed_extra_species=["CCN(CC)CC"],
    )
    # Excess starting material and a declared additive do not block completion.
    assert result["contains_target_product"] is True
    assert result["unexpected_species"] == []


def _accepted_event(seq: int, step_index: int, current, resulting, contains_target: bool):
    return {
        "seq": seq,
        "event_type": "mechanism_step_accepted",
        "payload": {
            "step_index": step_index,
            "candidate_rank": 1,
            "current_state": current,
            "resulting_state": resulting,
            "contains_target_product": contains_target,
            "validation_summary": {
                "passed": True,
                "checks": [
                    {"name": "dbe_metadata", "passed": True},
                    {"name": "atom_balance", "passed": True},
                    {"name": "state_progress", "passed": True},
                ],
            },
        },
    }


def test_scoring_credits_partial_byproduct_completion_but_does_not_pass() -> None:
    expected = {
        "products": ["P", "BY1", "BY2"],
        "known_mechanism": {
            "min_steps": 2,
            "steps": [{"step_index": 1, "target_smiles": "INT1"}, {"step_index": 2, "target_smiles": "P"}],
        },
    }
    snapshot = {
        "input": {"starting_materials": ["A", "B"]},
        "events": [
            _accepted_event(1, 1, ["A", "B"], ["INT1"], False),
            # Primary product reached, but only one of two byproducts and a leftover intermediate.
            _accepted_event(2, 2, ["INT1"], ["P", "BY1", "LEFTOVER"], True),
        ],
        "step_outputs": [],
    }
    scored = score_snapshot_against_known(snapshot, expected)
    assert scored["final_product_reached"] is True
    assert scored["all_target_products_reached"] is False
    assert abs(scored["final_product_component"] - (2 / 3)) < 1e-6
    assert scored["missing_target_products"] == ["BY2"]
    assert scored["unexpected_final_species"] == ["LEFTOVER"]
    assert any(p["type"] == "unexpected_final_species" for p in scored["penalties"])
    assert scored["passed"] is False

    complete = {
        "input": {"starting_materials": ["A", "B"]},
        "events": [
            _accepted_event(1, 1, ["A", "B"], ["INT1"], False),
            _accepted_event(2, 2, ["INT1"], ["P", "BY1", "BY2"], True),
        ],
        "step_outputs": [],
    }
    scored_complete = score_snapshot_against_known(complete, expected)
    assert scored_complete["all_target_products_reached"] is True
    assert scored_complete["final_product_component"] == 1.0
    assert scored_complete["passed"] is True
    assert scored_complete["score"] > scored["score"]
