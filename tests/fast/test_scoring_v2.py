"""Scoring v2: measured mapping component (PRD v2 §11) and history regeneration."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mechanistic_agent.scoring import (
    DEFAULT_SCORING_VERSION,
    MAPPING_SOURCE_ATOM_MAP_CHECK,
    MAPPING_SOURCE_BENCHMARK,
    MAPPING_SOURCE_NEUTRAL,
    normalize_scoring_version,
    score_snapshot_against_known,
    score_subagents_from_step_outputs,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN = Path(__file__).parent / "fixtures" / "scoring_v1_golden.json"

_OK = {
    "checks": [
        {"name": "dbe_metadata", "passed": True},
        {"name": "atom_balance", "passed": True},
        {"name": "state_progress", "passed": True},
    ]
}
_EXPECTED_NO_BENCHMARK = {
    "known_mechanism": {
        "min_steps": 2,
        "steps": [{"step_index": 1, "target_smiles": "INT1"}, {"step_index": 2, "target_smiles": "P"}],
    }
}


def _synthetic_snapshot(step_mapping_rows: list) -> dict:
    return {
        "events": [
            {
                "seq": i,
                "event_type": "mechanism_step_accepted",
                "payload": {"step_index": i, "resulting_state": [target], "validation_summary": _OK},
            }
            for i, target in ((1, "INT1"), (2, "P"))
        ],
        "step_outputs": step_mapping_rows,
    }


def _check(passed: bool | None) -> dict:
    if passed is None:
        return {"checks": [{"name": "atom_map_check", "passed": True, "details": {"skipped": True}}]}
    return {"checks": [{"name": "atom_map_check", "passed": passed, "details": {"errors": []}}]}


# ---------------------------------------------------------------------------
# v1 regression guard
# ---------------------------------------------------------------------------


def _assert_subset_equal(expected, actual, path="") -> None:
    """Every key the pre-v2 scorer produced is reproduced exactly (new keys allowed)."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        for key, value in expected.items():
            assert key in actual, f"{path}.{key} missing"
            _assert_subset_equal(value, actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for idx, (a, b) in enumerate(zip(expected, actual)):
            _assert_subset_equal(a, b, f"{path}[{idx}]")
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def test_v1_reproduces_pre_versioning_scorer_exactly() -> None:
    # Fixture: synthetic snapshots plus trimmed real traces from flower_100_default
    # (committed eval set), each scored by the scorer on main before this change.
    golden = json.loads(GOLDEN.read_text())
    assert len(golden) >= 8
    assert any(item["v1_subagents"].get("step_atom_mapping") for item in golden)
    for item in golden:
        scored = score_snapshot_against_known(item["snapshot"], item["expected"], scoring_version="v1")
        _assert_subset_equal(item["v1_scored"], scored, item["name"])
        assert scored["scoring_version"] == "v1"
        subagents = score_subagents_from_step_outputs(item["snapshot"]["step_outputs"], scoring_version="v1")
        assert subagents == item["v1_subagents"], item["name"]


def test_default_scoring_version_is_v2() -> None:
    assert DEFAULT_SCORING_VERSION == "v2"
    assert normalize_scoring_version(None) == "v2"
    assert score_snapshot_against_known(_synthetic_snapshot([]), _EXPECTED_NO_BENCHMARK)["scoring_version"] == "v2"
    with pytest.raises(ValueError):
        normalize_scoring_version("v3")


# ---------------------------------------------------------------------------
# v2 mapping component: benchmark agreement / atom-map check / neutral
# ---------------------------------------------------------------------------


def _benchmark_case(perturb: bool, confidence: float):
    from mechanistic_agent.core.mapping_metrics import mapping_from_mapped_smiles, mapping_to_llm_mapped_atoms
    from mechanistic_agent.smiles_utils import strip_atom_mapping_list

    record = json.loads((REPO_ROOT / "training_data" / "eval_set.json").read_text())[0]
    ref = mapping_from_mapped_smiles(record["starting_materials"], record["products"])
    atoms = mapping_to_llm_mapped_atoms(ref)
    if perturb:
        # Send the nitrile N onto a ring carbon: a real, non-symmetric error.
        atoms[1] = {**atoms[1], "product_atom": atoms[5]["product_atom"]}
    sm = strip_atom_mapping_list(record["starting_materials"])
    pr = strip_atom_mapping_list(record["products"])
    snapshot = {
        "input_payload": {"starting_materials": sm, "products": pr},
        "events": [
            {
                "seq": 1,
                "event_type": "mechanism_step_accepted",
                "payload": {"step_index": 1, "current_state": sm, "resulting_state": pr, "validation_summary": _OK},
            }
        ],
        "step_outputs": [
            {
                "step_name": "step_atom_mapping",
                "attempt": 1,
                "output": {
                    "confidence": confidence,
                    "current_state": sm,
                    "resulting_state": pr,
                    "raw": {"llm_response": {"mapped_atoms": atoms, "confidence": confidence}},
                },
                # A stored check must not override benchmark agreement.
                "validation": _check(False),
            }
        ],
    }
    expected = {
        "products": record["products"],
        "known_mechanism": record["known_mechanism"],
        "verified_mechanism": record["verified_mechanism"],
    }
    return snapshot, expected


def test_v2_uses_benchmark_agreement_when_available() -> None:
    snapshot, expected = _benchmark_case(perturb=False, confidence=0.1)
    v2 = score_snapshot_against_known(snapshot, expected, scoring_version="v2")
    v1 = score_snapshot_against_known(snapshot, expected, scoring_version="v1")
    step = v2["step_breakdown"][0]
    assert step["mapping_component_source"] == MAPPING_SOURCE_BENCHMARK
    assert step["mapping_component"] == 1.0
    assert step["validity_score"] == pytest.approx(0.8 * step["validation_score"] + 0.2 * 1.0)
    assert v1["step_breakdown"][0]["mapping_component"] == pytest.approx(0.1)
    assert v2["score"] > v1["score"]

    bad_snapshot, _ = _benchmark_case(perturb=True, confidence=1.0)
    bad = score_snapshot_against_known(bad_snapshot, expected, scoring_version="v2")
    agreement = bad["mapping_agreement"]["steps"][0]["agreement"]
    assert 0.0 < agreement < 1.0
    assert bad["step_breakdown"][0]["mapping_component"] == pytest.approx(agreement, abs=1e-4)
    assert bad["step_breakdown"][0]["mapping_component_source"] == MAPPING_SOURCE_BENCHMARK

    sub = score_subagents_from_step_outputs(
        bad_snapshot["step_outputs"], scoring_version="v2", mapping_agreement=bad["mapping_agreement"]
    )["step_atom_mapping"]
    assert sub["quality_score"] == pytest.approx(round(agreement, 4))
    assert sub["pass_rate"] == (1.0 if agreement >= 0.5 else 0.0)
    assert sub["mapping_component_sources"] == {MAPPING_SOURCE_BENCHMARK: 1}


def test_v2_falls_back_to_atom_map_check_without_benchmark() -> None:
    rows = [
        {"step_name": "step_atom_mapping", "attempt": 1, "output": {"confidence": 1.0}, "validation": _check(True)},
        # Check stored on the output (MappingAgent.atom_map_validation) is honoured too.
        {"step_name": "step_atom_mapping", "attempt": 2, "output": {"confidence": 1.0, "atom_map_validation": _check(False)}},
    ]
    scored = score_snapshot_against_known(_synthetic_snapshot(rows), _EXPECTED_NO_BENCHMARK, scoring_version="v2")
    comps = [(s["mapping_component"], s["mapping_component_source"]) for s in scored["step_breakdown"]]
    assert comps == [(1.0, MAPPING_SOURCE_ATOM_MAP_CHECK), (0.0, MAPPING_SOURCE_ATOM_MAP_CHECK)]
    assert scored["mapping_component_sources"] == {MAPPING_SOURCE_ATOM_MAP_CHECK: 2}

    sub = score_subagents_from_step_outputs(rows, scoring_version="v2")["step_atom_mapping"]
    assert sub["quality_score"] == 0.5
    assert sub["pass_rate"] == 0.5
    assert sub["calls"] == 2


def test_v2_neutral_without_benchmark_or_check_and_ignores_self_report() -> None:
    def rows(conf):
        return [
            {"step_name": "step_atom_mapping", "attempt": 1, "output": {"confidence": conf}, "validation": _check(None)},
            {"step_name": "step_atom_mapping", "attempt": 2, "output": {"confidence": conf}},
        ]

    high = score_snapshot_against_known(_synthetic_snapshot(rows(1.0)), _EXPECTED_NO_BENCHMARK, scoring_version="v2")
    low = score_snapshot_against_known(_synthetic_snapshot(rows(0.0)), _EXPECTED_NO_BENCHMARK, scoring_version="v2")
    for scored in (high, low):
        assert [s["mapping_component"] for s in scored["step_breakdown"]] == [0.5, 0.5]
        assert {s["mapping_component_source"] for s in scored["step_breakdown"]} == {MAPPING_SOURCE_NEUTRAL}
    # A mapper reporting 1.0 no longer inflates the score (it did under v1).
    assert high["score"] == low["score"]
    v1_high = score_snapshot_against_known(_synthetic_snapshot(rows(1.0)), _EXPECTED_NO_BENCHMARK, scoring_version="v1")
    v1_low = score_snapshot_against_known(_synthetic_snapshot(rows(0.0)), _EXPECTED_NO_BENCHMARK, scoring_version="v1")
    assert v1_high["score"] > v1_low["score"]

    sub_high = score_subagents_from_step_outputs(rows(1.0), scoring_version="v2")["step_atom_mapping"]
    assert (sub_high["quality_score"], sub_high["pass_rate"]) == (0.5, 1.0)
    assert score_subagents_from_step_outputs(rows(1.0), scoring_version="v1")["step_atom_mapping"]["quality_score"] == 1.0


def test_v2_equals_v1_when_no_step_mapping_ran() -> None:
    snapshot = _synthetic_snapshot([])
    v1 = score_snapshot_against_known(snapshot, _EXPECTED_NO_BENCHMARK, scoring_version="v1")
    v2 = score_snapshot_against_known(snapshot, _EXPECTED_NO_BENCHMARK, scoring_version="v2")
    assert v1["score"] == v2["score"]


# ---------------------------------------------------------------------------
# Persistence: scoring_version on results and leaderboard rows; regeneration
# ---------------------------------------------------------------------------


def _seed_store(tmp_path: Path):
    from mechanistic_agent.core.db import RunStore

    store = RunStore(tmp_path / "mechanistic.db")
    snapshot, expected = _benchmark_case(perturb=True, confidence=0.95)
    eval_set_id = store.add_eval_set(
        name="unit", version="v", source_path=None, sha256=None,
        cases=[{"case_id": "case_a", "input": {}, "expected": expected}],
    )
    run_id = store.create_run(
        mode="unverified",
        input_payload=snapshot["input_payload"],
        config={"model": "m"},
        prompt_bundle_hash="p",
        skill_bundle_hash="s",
    )
    for event in snapshot["events"]:
        store.append_event(run_id, event["event_type"], event["payload"])
    for row in snapshot["step_outputs"]:
        store.record_step_output(
            run_id=run_id, step_name=row["step_name"], attempt=row["attempt"], model="m",
            reasoning_level=None, tool_name="t", output=row["output"], validation=row.get("validation"),
        )
    stored = store.get_run_snapshot(run_id)
    v1 = score_snapshot_against_known(stored, expected, scoring_version="v1")
    eval_run_id = store.create_eval_run(
        eval_set_id=eval_set_id, run_group_name="g", model="m", harness_bundle_hash="h", status="completed"
    )
    store.record_eval_run_result(
        eval_run_id=eval_run_id, case_id="case_a", run_id=run_id, score=v1["score"], passed=v1["passed"],
        cost={}, latency_ms=1.0, summary={"score": v1["score"], "eval_mode": "harness", "subagent_scores": {}},
    )
    # A crashed case with no trace: version-independent.
    store.record_eval_run_result(
        eval_run_id=eval_run_id, case_id="case_b", run_id=None, score=0.0, passed=False,
        cost={}, latency_ms=0.0, summary={"error": "boom", "eval_mode": "harness"},
    )
    return store, eval_set_id, eval_run_id, v1, stored, expected


def test_leaderboard_row_carries_scoring_version(tmp_path: Path) -> None:
    store, eval_set_id, eval_run_id, _v1, _snap, _exp = _seed_store(tmp_path)
    row = store.leaderboard(eval_set_id)[0]
    assert row["scoring_version"] == "v1"  # legacy results carry no version: v1
    results = store.list_eval_run_results(eval_run_id)
    store.update_eval_run_result(
        results[0]["id"], score=results[0]["score"], passed=results[0]["pass_bool"],
        summary={**results[0]["summary"], "scoring_version": "v2"},
    )
    assert store.leaderboard(eval_set_id)[0]["scoring_version"] == "mixed"
    store.update_eval_run_result(
        results[1]["id"], score=0.0, passed=False, summary={**results[1]["summary"], "scoring_version": "v2"},
    )
    assert store.leaderboard(eval_set_id)[0]["scoring_version"] == "v2"


def test_rescore_dry_run_leaves_db_untouched_and_reports_delta(tmp_path: Path) -> None:
    from mechanistic_agent.rescoring import format_delta_markdown, run_rescore

    store, eval_set_id, eval_run_id, v1, snap, expected = _seed_store(tmp_path)
    db = tmp_path / "mechanistic.db"
    before_bytes = db.read_bytes()
    result = run_rescore(db, scoring_version="v2", apply=False, use_curriculum=False, work_dir=tmp_path)
    assert db.read_bytes() == before_bytes
    assert result.report.counts() == {"rescored": 1, "version_independent": 1}
    [row] = result.delta_rows
    expected_v2 = score_snapshot_against_known(snap, expected, scoring_version="v2")["score"]
    assert row["v1_score"] == pytest.approx(v1["score"] / 2)  # mean over both cases
    assert row["v2_score"] == pytest.approx(expected_v2 / 2)
    assert row["delta"] == pytest.approx((expected_v2 - v1["score"]) / 2)
    # Unweighted over both cases; the trace-less case contributes 0 to both sides.
    assert row["v1_recomputed_mean"] == pytest.approx(v1["score"] / 2)
    assert row["version_delta"] == pytest.approx((expected_v2 - v1["score"]) / 2)
    assert row["scoring_version_after"] == "v2"
    assert "Delta" in format_delta_markdown(result.delta_rows, result.report)
    assert store.leaderboard(eval_set_id)[0]["scoring_version"] == "v1"


def test_rescore_apply_rewrites_results_and_keeps_history(tmp_path: Path) -> None:
    from mechanistic_agent.rescoring import run_rescore

    store, eval_set_id, eval_run_id, v1, snap, expected = _seed_store(tmp_path)
    db = tmp_path / "mechanistic.db"
    result = run_rescore(db, scoring_version="v2", apply=True, use_curriculum=False)
    assert result.backup_path is not None and result.backup_path.is_file()

    results = {r["case_id"]: r for r in store.list_eval_run_results(eval_run_id)}
    graded = score_snapshot_against_known(snap, expected, scoring_version="v2")
    a = results["case_a"]
    assert a["score"] == pytest.approx(graded["score"])
    assert a["summary"]["scoring_version"] == "v2"
    assert a["summary"]["scoring_history"]["v1"]["score"] == pytest.approx(v1["score"])
    assert a["summary"]["subagent_scores"]["step_atom_mapping"]["mapping_component_sources"] == {
        MAPPING_SOURCE_BENCHMARK: 1
    }
    assert a["summary"]["mapping_agreement"]["steps"][0]["status"] == "scored"
    assert results["case_b"]["summary"]["scoring_version"] == "v2"
    assert store.leaderboard(eval_set_id)[0]["scoring_version"] == "v2"

    # Idempotent: a second pass changes nothing and keeps the original v1 history.
    run_rescore(db, scoring_version="v2", apply=True, backup=False, use_curriculum=False)
    again = {r["case_id"]: r for r in store.list_eval_run_results(eval_run_id)}["case_a"]
    assert again["score"] == pytest.approx(a["score"])
    assert again["summary"]["scoring_history"] == a["summary"]["scoring_history"]


def test_rescore_cli_dry_run(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    import main as cli

    _seed_store(tmp_path)
    db = tmp_path / "mechanistic.db"
    before = db.read_bytes()
    out_md = tmp_path / "delta.md"
    runner = CliRunner()
    res = runner.invoke(
        cli.app,
        ["rescore-eval-results", "--db-path", str(db), "--no-curriculum", "--output", str(out_md)],
    )
    assert res.exit_code == 0, res.output
    assert "DRY RUN" in res.output
    assert "| unit@v |" in out_md.read_text()
    assert db.read_bytes() == before

    missing = runner.invoke(cli.app, ["rescore-eval-results", "--db-path", str(tmp_path / "nope.db")])
    assert missing.exit_code == 1
