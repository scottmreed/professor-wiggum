"""quality_v1: one rubric for harness and baseline mechanisms, scored from the accepted path alone."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

pytest.importorskip("rdkit")

from mechanistic_agent import quality_scoring as qs  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CASES = {c["id"]: c for c in json.loads((ROOT / "training_data" / "practice_eval" / "practice_set.json").read_text())}


def _case(case_id: str) -> Dict[str, Any]:
    return copy.deepcopy(CASES[case_id])


def _score(steps: List[Dict[str, Any]], case: Dict[str, Any], **kw: Any) -> Dict[str, Any]:
    kw.setdefault("sequence_score", 1.0)
    return qs.score_mechanism(steps, starting_materials=case["starting_materials"], products=case["products"], **kw)


def test_weights_sum_to_1000_and_have_no_speed_or_product_points() -> None:
    assert sum(qs.WEIGHTS.values()) == 1000
    assert not {"speed", "product", "methodology"} & set(qs.WEIGHTS)


def test_flower_reference_mechanism_scores_full_and_passes() -> None:
    case = _case("flower_254799")
    result = _score(case["verified_mechanism"]["steps"], case)
    assert result["passed"] is True
    assert result["valid_steps"] == result["step_count"] == 3
    assert result["points"] >= 950
    assert result["closure"]["grade"] == "exact"


def test_unreached_target_is_a_gate_not_points() -> None:
    case = _case("flower_254799")
    full = _score(case["verified_mechanism"]["steps"], case)
    short = _score(case["verified_mechanism"]["steps"][:-1], case)
    assert short["targets"]["all_reached"] is False and short["passed"] is False
    assert short["points"] == pytest.approx(short["raw_points"] * qs.UNREACHED_FACTOR, abs=0.2)
    assert short["points"] < full["points"] / 1.5


def test_step_without_arrows_or_smirks_is_invalid() -> None:
    case = _case("flower_254799")
    steps = case["verified_mechanism"]["steps"]
    steps[1]["reaction_smirks"] = ""
    result = _score(steps, case)
    bad = next(s for s in result["steps"] if s["step_index"] == steps[1]["step_index"])
    assert bad["valid"] is False and bad["checks"]["arrows"] is False
    assert result["ratios"]["electron_conservation"] < 1.0
    assert result["passed"] is False


def test_circular_sequence_costs_efficiency_and_blocks_pass() -> None:
    case = _case("flower_254799")
    steps = case["verified_mechanism"]["steps"]
    first = steps[0]
    undo = {
        "step_index": 2,
        "current_state": list(first["resulting_state"]),
        "resulting_state": list(first["current_state"]),
        "reaction_smirks": "",
        "electron_pushes": [],
    }
    redo = dict(first, step_index=3)
    rest = [dict(s, step_index=s["step_index"] + 2) for s in steps[1:]]
    result = _score([first, undo, redo, *rest], case)
    assert result["findings"]["circular"]
    assert result["ratios"]["efficiency"] < 1.0
    assert result["passed"] is False


def test_phantom_reagent_is_unexplained() -> None:
    case = _case("flower_254799")
    steps = case["verified_mechanism"]["steps"]
    steps[0]["current_state"] = [*steps[0]["current_state"], "CCCCCCO"]
    steps[0]["resulting_state"] = [*steps[0]["resulting_state"], "CCCCCCO"]
    result = _score(steps, case)
    assert result["findings"]["unexplained_species"]
    assert result["ratios"]["reagents_and_solvent"] < 1.0


def test_bare_proton_costs_proton_bookkeeping() -> None:
    case = _case("flower_254799")
    steps = case["verified_mechanism"]["steps"]
    steps[-1]["resulting_state"] = [*steps[-1]["resulting_state"], "[H+]"]
    result = _score(steps, case)
    assert result["findings"]["bare_proton_steps"] == [steps[-1]["step_index"]]
    assert result["ratios"]["proton_bookkeeping"] < 1.0


def test_intramolecular_shift_with_a_shuttle_available_is_not_credited() -> None:
    case = _case("flower_257551")
    result = _score(case["verified_mechanism"]["steps"], case)
    modes = {p["mode"]: p["credited"] for p in result["findings"]["proton_transfers"]}
    assert modes == {"intramolecular": False, "intermolecular": True}
    assert result["ratios"]["intermolecular"] == 0.5


@pytest.mark.parametrize(
    "starting, expected",
    [
        (["CC(C)(C)OC(=O)N", "O=C(O)C(F)(F)F"], "acidic"),
        (["CC(=O)Cl", "CCN(CC)CC"], "basic"),
        (["O=C(O)C(F)(F)F", "CCN(CC)CC"], "buffered"),
        (["CC(=O)OC", "O"], "neutral"),
    ],
)
def test_conditions_classifier(starting: List[str], expected: str) -> None:
    assert qs.classify_conditions(starting)["class"] == expected


def test_protonation_rules_flag_only_free_extremes() -> None:
    assert [f["rule"] for f in qs.implausible_species(["[OH-]", "CC(=O)[O-]", "CC[O-]"], "acidic")] == [
        "hydroxide", "alkoxide",
    ]
    assert qs.implausible_species(["CC([O-])(O)[NH2+]C"], "acidic") == []  # zwitterion: net neutral
    assert [f["rule"] for f in qs.implausible_species(["[OH3+]", "C[NH3+]", "CC(=[OH+])C"], "basic")] == [
        "hydronium", "protonated_carbonyl",
    ]
    assert qs.implausible_species(["[OH-]", "[OH3+]"], "neutral") == []


def test_baseline_snapshot_steps_keep_smirks_and_arrows() -> None:
    from mechanistic_agent.core.baseline_runner import _steps_to_synthetic_snapshot

    case = _case("flower_254799")
    snapshot = _steps_to_synthetic_snapshot(case["verified_mechanism"]["steps"], case["starting_materials"], case["products"])
    path = qs.steps_from_snapshot(snapshot)
    assert [s.step_index for s in path] == [1, 2, 3]
    assert all(s.reaction_smirks and s.electron_pushes for s in path)
    result = qs.score_snapshot_quality(snapshot, {"starting_materials": case["starting_materials"],
                                                  "products": case["products"],
                                                  "verified_mechanism": case["verified_mechanism"]})
    assert result["passed"] is True and result["sequence_basis"] in {"exact", "proton_agnostic"}


def test_summarize_means_components_and_counts() -> None:
    results = [
        {"points": 900.0, "passed": True, "components": {"step_validity": 250.0}, "targets": {"all_reached": True},
         "valid_steps": 3, "step_count": 3},
        {"points": 500.0, "passed": False, "components": {"step_validity": 150.0}, "targets": {"all_reached": False},
         "valid_steps": 1, "step_count": 4},
    ]
    summary = qs.summarize(results)
    assert summary["points"] == 700.0 and summary["passed"] == 1 and summary["targets_reached"] == 1
    assert summary["components"]["step_validity"] == 200.0
    assert summary["valid_step_fraction"] == round(4 / 7, 4)


def test_core_fragment_smirks_matches_states_but_unrelated_smirks_does_not() -> None:
    step = qs.QualityStep(
        1,
        ["O=C(O)C(F)(F)F", "CC(C)(C)OC(=O)NC1CCC(F)(F)CC1"],
        ["O=C([O-])C(F)(F)F", "CC(C)(C)OC(=[OH+])NC1CCC(F)(F)CC1"],
        "[C:1](=[O:2])[N:3].[O:4]([H:5])[C:6]>>[C:1](=[O+:2][H:5])[N:3].[O-:4][C:6]",
        [],
    )
    assert qs._smirks_matches_states(step) is True
    step.reaction_smirks = "[Cl:1][C:2]>>[Cl-:1].[C+:2]"
    assert qs._smirks_matches_states(step) is False


def test_mapped_explicit_hydrogens_canonicalize_like_implicit_ones() -> None:
    assert qs.canonical("[C:1]([H:2])([H:3])([H:4])[O:5][H:6]") == qs.canonical("CO") == "CO"
    assert qs.canonical("[H+]") == "[H+]"


def test_rescoring_a_baseline_recovers_starting_materials_from_the_reference() -> None:
    case = _case("flower_254799")
    expected = {"products": case["products"], "verified_mechanism": case["verified_mechanism"]}
    snapshot = qs._baseline_snapshot({"baseline_steps": case["verified_mechanism"]["steps"]}, expected)
    result = qs.score_snapshot_quality(snapshot, expected)
    assert result["findings"]["unexplained_species"] == []
    assert result["closure"]["grade"] == "exact"
