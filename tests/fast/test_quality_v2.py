from __future__ import annotations

import pytest

pytest.importorskip("rdkit")

from mechanistic_agent.quality_scoring import QualityStep, score_mechanism  # noqa: E402
from mechanistic_agent.quality_v2 import (  # noqa: E402
    connectivity_links,
    polygon_area,
    score_case,
    summarize,
)

START = ["CCO", "CC(=O)Cl"]


def _step(i, cur, res):
    return QualityStep(i, cur, res)


def test_connected_path_with_bookkeeping_noise_is_fully_connected() -> None:
    steps = [
        _step(1, ["CCO", "CC(=O)Cl"], ["CC[OH+]C(C)([O-])Cl"]),
        # Chloride written as Cl then [Cl-], water and THF appear: small species and solvents are ignored.
        _step(2, ["CC[OH+]C(C)([O-])Cl", "C1CCOC1"], ["CC[OH+]C(C)=O", "[Cl-]"]),
        # Protonation change of the intermediate is the same neutral parent.
        _step(3, ["CCOC(C)=O", "O"], ["CCOC(C)=O", "[OH3+]"]),
    ]
    assert [link["connected"] for link in connectivity_links(steps, START)] == [True, True, True]


def test_vanishing_intermediate_and_product_from_nowhere_break_links() -> None:
    steps = [
        _step(1, ["CCO", "CC(=O)Cl"], ["CC[OH+]C(C)([O-])Cl"]),
        _step(2, ["CCO"], ["CCO"]),  # the intermediate vanished between steps
        _step(3, ["CCOC(C)=O"], ["CCOC(C)=O"]),  # the product appears from nowhere
    ]
    links = connectivity_links(steps, START)
    assert [link["connected"] for link in links] == [True, False, False]
    assert links[1]["vanished"] and links[2]["appeared"] == ["CCOC(C)=O"]


def test_first_step_must_start_from_the_starting_materials() -> None:
    links = connectivity_links([_step(1, ["CCOC(C)=O"], ["CCOC(C)=O"])], START)
    assert links[0]["connected"] is False


def test_a_solvent_starting_material_still_counts_as_a_start() -> None:
    links = connectivity_links([_step(1, ["C1CCOC1", "CC(=O)Cl"], ["C1CCOC1", "CC(=O)Cl"])], ["C1CCOC1", "CC(=O)Cl"])
    assert links[0]["connected"] is True


def test_dropping_a_spare_reagent_between_steps_is_not_a_break() -> None:
    steps = [
        _step(1, ["CCO", "CC(=O)Cl"], ["CC[OH+]C(C)([O-])Cl", "CCO"]),
        _step(2, ["CC[OH+]C(C)([O-])Cl"], ["CC[OH+]C(C)=O", "[Cl-]"]),
    ]
    assert all(link["connected"] for link in connectivity_links(steps, START))


def test_failed_start_scores_zero_everywhere() -> None:
    v1 = score_mechanism([], starting_materials=START, products=["CCOC(C)=O"])
    case = score_case([], v1, starting_materials=START)
    assert case["failed_start"] and case["score"] == 0.0 and set(case["axes"].values()) == {0.0}
    agg = summarize([case, {**case, "failed_start": False}])
    assert agg["failed_starts"] == 1 and agg["failed_start_fraction"] == 0.5


def test_connectivity_outweighs_the_product_when_most_links_break() -> None:
    step_ok = {"checks": {"atom_balance": True, "state_progress": True, "arrows": True,
                          "smirks_matches_states": True}, "electron_conserved": True}
    v1_jump = {"product_correct": True, "ratios": {"sequence": 0.5}, "passed": True, "steps": [step_ok] * 4}
    v1_stop = {**v1_jump, "product_correct": False, "passed": False}
    # Reaches the product, but every step after the first starts from nowhere.
    jump = [_step(1, ["CCO", "CC(=O)Cl"], ["CC[OH+]C(C)([O-])Cl"]),
            _step(2, ["CCC(=O)OCC"], ["CCC(=O)OCC"]),
            _step(3, ["CCCC(=O)OCC"], ["CCCC(=O)OCC"]),
            _step(4, ["CCOC(C)=O"], ["CCOC(C)=O"])]
    # Stops short of the product, every step connected.
    stop = [_step(1, ["CCO", "CC(=O)Cl"], ["CC[OH+]C(C)([O-])Cl"]),
            _step(2, ["CC[OH+]C(C)([O-])Cl"], ["CC[OH+]C(C)=O", "[Cl-]"]),
            _step(3, ["CC[OH+]C(C)=O"], ["CC[OH+]C(C)=O"]),
            _step(4, ["CC[OH+]C(C)=O"], ["CC[OH+]C(C)=O"])]
    s_jump = score_case(jump, v1_jump, starting_materials=START)
    s_stop = score_case(stop, v1_stop, starting_materials=START)
    assert s_jump["axes"]["connectivity"] == 0.25 and s_stop["axes"]["connectivity"] == 1.0
    assert s_jump["passed"] is False  # a pass needs every step connected
    # Each broken link costs 0.3/n and the product earns 0.2: breaking more than 2/3 of the links
    # (here 3 of 4) costs more than the product earns.
    assert s_jump["score"] == pytest.approx(0.2 + 0.075 + 0.5 * 0.875)
    assert s_stop["score"] == pytest.approx(0.3 + 0.5 * 0.875)
    assert s_jump["score"] < s_stop["score"]


def test_polygon_area_is_one_for_a_perfect_run() -> None:
    assert polygon_area({k: 1.0 for k in ("product", "connectivity", "balance", "notation", "electrons", "sequence")}) == 1.0
    assert polygon_area({}) == 0.0


def test_a_small_released_byproduct_leaving_is_not_a_break() -> None:
    start = ["CC(C)(C)OC(=O)NC1CCCCC1", "Cl"]
    steps = [
        _step(1, start, ["CC(C)(C)[OH+]C(=O)NC1CCCCC1", "[Cl-]"]),
        _step(2, ["CC(C)(C)[OH+]C(=O)NC1CCCCC1"], ["C=C(C)C", "O=C(O)NC1CCCCC1", "[H+]"]),
        _step(3, ["O=C(O)NC1CCCCC1"], ["O=C=O", "NC1CCCCC1"]),  # isobutylene released
    ]
    assert all(link["connected"] for link in connectivity_links(steps, start))
