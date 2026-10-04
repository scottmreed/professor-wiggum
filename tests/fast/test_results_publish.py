"""Publishing local eval runs as committed results + the generated public boards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from mechanistic_agent import results_publish as rp
from mechanistic_agent.scoring import graded_to_points


class _FakeStore:
    def __init__(self, *, purpose: str = "general", saw_ground_truth: bool = False) -> None:
        self.purpose = purpose
        self.saw = saw_ground_truth
        self.results = [
            {"case_id": "c_easy", "run_id": "r1", "score": 1.0, "pass_bool": True, "latency_ms": 60_000},
            {"case_id": "c_hard", "run_id": "r2", "score": 0.9, "pass_bool": True, "latency_ms": 120_000},
            {"case_id": "c_fail", "run_id": "r3", "score": 0.3, "pass_bool": False, "latency_ms": 90_000},
        ]

    def get_eval_run(self, eval_run_id: str) -> Dict[str, Any]:
        return {
            "id": eval_run_id,
            "eval_set_id": "set1",
            "run_group_name": "grp",
            "model": "agent-bridge",
            "model_name": "agent-bridge",
            "created_at": 1790000000.0,
            "metadata": {"tier_name": "hard", "selected_case_ids_hash": "abc"},
        }

    def list_eval_run_results(self, eval_run_id: str) -> List[Dict[str, Any]]:
        return list(self.results)

    def get_eval_set(self, eval_set_id: str) -> Dict[str, Any]:
        return {"id": eval_set_id, "purpose": self.purpose}

    def get_run_snapshot(self, run_id: str) -> Dict[str, Any]:
        return {
            "id": run_id,
            "status": "completed",
            "config": {
                "harness_name": "jev_reaction_type",
                "origin": {
                    "responder": "agent-bridge",
                    "declared_underlying_model": "claude-opus-5-5 (headless)",
                    "responder_saw_ground_truth": self.saw,
                },
            },
            "input_payload": {"starting_materials": ["[CH3:1][OH:2]"], "products": ["C=O"]},
            "events": [],
            "step_outputs": [],
        }


_GRADES = {
    "r1": {"score": 1.0, "passed": True, "final_product_reached": True, "known_alignment_component": 1.0,
           "step_validity_component": 1.0, "accepted_path_step_count": 1},
    "r2": {"score": 0.9, "passed": True, "final_product_reached": True, "known_alignment_component": 0.8,
           "step_validity_component": 0.9, "accepted_path_step_count": 5},
    "r3": {"score": 0.3, "passed": False, "final_product_reached": False, "known_alignment_component": 0.2,
           "step_validity_component": 0.5, "accepted_path_step_count": 2},
}
_KNOWN = {"c_easy": 1, "c_hard": 4, "c_fail": 6}


@pytest.fixture(autouse=True)
def _stub_grading(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rp, "score_snapshot_against_known", lambda snap, expected, **_: dict(_GRADES[snap["id"]]))
    monkeypatch.setattr(rp, "_accepted_path_record", lambda snap: [{"step_index": 1, "current_state": ["CO"],
                                                                     "resulting_state": ["C=O"]}])
    monkeypatch.setattr(rp, "_git_commit", lambda base: "deadbee")


def _resolver(result: Dict[str, Any], run: Dict[str, Any]) -> Dict[str, Any]:
    return {"n_mechanistic_steps": _KNOWN[result["case_id"]]}


def test_export_record_shape_scores_and_hardest_case() -> None:
    record = rp.export_eval_run(_FakeStore(), "ev1", expected_resolver=_resolver)

    assert record["schema"] == rp.RECORD_SCHEMA
    assert record["harness"] == "jev_reaction_type"
    assert record["tier"] == "hard"
    assert record["origin"]["declared_underlying_model"].startswith("claude-opus-5-5")
    summary = record["summary"]
    assert (summary["cases"], summary["targets_reached"], summary["passed"]) == (3, 2, 2)
    assert summary["points"] == graded_to_points(list(_GRADES.values()), [60_000, 120_000, 90_000])["total"]
    # Hardest = passed case with the longest reference mechanism (c_fail is longer but failed).
    hardest = record["hardest_solved"]
    assert hardest["case_id"] == "c_hard"
    assert hardest["known_steps"] == 4
    assert hardest["starting_materials"] == ["CO"]  # atom maps stripped


def test_ground_truth_replays_are_refused() -> None:
    with pytest.raises(rp.PublishError, match="ground-truth replay"):
        rp.export_eval_run(_FakeStore(saw_ground_truth=True), "ev1", expected_resolver=_resolver)


def test_holdout_runs_publish_aggregates_only() -> None:
    record = rp.export_eval_run(_FakeStore(purpose="leaderboard_holdout"), "ev1", expected_resolver=_resolver)
    assert record["holdout"] is True
    assert "cases" not in record and "hardest_solved" not in record
    assert record["summary"]["cases"] == 3


def test_graded_to_points_rubric() -> None:
    graded = [
        {"final_product_reached": True, "known_alignment_component": 1.0, "step_validity_component": 1.0},
        {"final_product_reached": False, "known_alignment_component": 0.5, "step_validity_component": 0.5},
    ]
    pts = graded_to_points(graded, [100_000, 100_000])
    assert (pts["product"], pts["pathway"], pts["push"], pts["speed"], pts["methodology"]) == (150, 225, 150, 75, 100)
    assert pts["total"] == 700 and pts["outcome"] == "WIN"
    assert graded_to_points([{"final_product_reached": False}], [1.0])["speed"] == 0


def _write_records(base: Path) -> None:
    for tier, points, model in (("easy", 940, "anthropic/claude-opus-5.5"), ("hard", 823, "agent-bridge")):
        record = rp.export_eval_run(_FakeStore(), f"ev_{tier}", expected_resolver=_resolver)
        record.update({"tier": tier, "run_group": f"grp_{tier}", "model": model})
        if model != "agent-bridge":
            record["origin"] = None  # hosted API run, no bridge provenance
        record["summary"]["points"] = points
        rp.write_record(record, base, render_image=False)


def test_regenerate_boards_is_idempotent_and_splices_readme(tmp_path: Path) -> None:
    _write_records(tmp_path)
    (tmp_path / "README.md").write_text(
        f"# Title\n\nIntro.\n\n{rp.README_LEADERBOARD_START}\nold\n{rp.README_LEADERBOARD_END}\n\nFooter.\n",
        encoding="utf-8",
    )

    rp.regenerate_boards(tmp_path)
    first = ((tmp_path / "LEADERBOARD.md").read_text(), (tmp_path / "README.md").read_text())
    rp.regenerate_boards(tmp_path)
    second = ((tmp_path / "LEADERBOARD.md").read_text(), (tmp_path / "README.md").read_text())

    assert first == second
    leaderboard, readme = first
    assert "| easy | **Claude Opus 5.5** | **940**/1000" in leaderboard
    assert "| hard | **Claude Opus 5.5** † | **823**/1000" in leaderboard
    assert "| medium | — |" in leaderboard
    assert readme.startswith("# Title\n\nIntro.") and readme.rstrip().endswith("Footer.")
    assert "old" not in readme and "LEADERBOARD.md#" in readme
    stored = json.loads(next((tmp_path / "results" / "runs").glob("*hard*.json")).read_text())
    assert stored["schema"] == rp.RECORD_SCHEMA


def test_open_results_pr_commits_only_results_paths(tmp_path: Path) -> None:
    calls: List[List[str]] = []

    def fake_run(cmd: List[str], cwd: Path) -> str:
        calls.append(cmd)
        return "https://github.com/x/y/pull/1" if cmd[:3] == ["gh", "pr", "create"] else ""

    def writer() -> List[Dict[str, Any]]:
        return [rp.export_eval_run(_FakeStore(), "ev1", expected_resolver=_resolver)]

    url = rp.open_results_pr(tmp_path, writer, branch="results/x", title="results: x", run=fake_run)

    assert url.endswith("/pull/1")
    assert ["git", "switch", "-c", "results/x", "origin/main"] in calls
    add = next(c for c in calls if c[:2] == ["git", "add"])
    assert add[3:] == rp.PUBLISH_PATHS
    assert not any("merge" in part for c in calls for part in c)


def test_open_results_pr_refuses_dirty_results(tmp_path: Path) -> None:
    def fake_run(cmd: List[str], cwd: Path) -> str:
        return " M LEADERBOARD.md" if cmd[:2] == ["git", "status"] else ""

    with pytest.raises(rp.PublishError, match="uncommitted changes"):
        rp.open_results_pr(tmp_path, lambda: [], branch="b", title="t", run=fake_run)


# --- Harness-free baseline eval runs ------------------------------------------------------------

_BRIDGE_ORIGIN = {
    "responder": "agent-bridge",
    "declared_underlying_model": "claude-opus-5-5",
    "responder_saw_ground_truth": False,
}
_KNOWN_PRODUCT = "N#CCCC1C=CC=C1"


def _baseline_summary(
    case_id: str, score: float, reached: bool, steps: int, *, thinking: Any = "high", origin: Any = None
) -> Dict[str, Any]:
    run_metadata: Dict[str, Any] = {"case_id": case_id, "model": "m", "thinking_level": thinking, "prompt_hash": "p"}
    if origin is not None:
        run_metadata["origin"] = origin
    return {
        "error": None,
        "eval_mode": "baseline",
        "mechanism_type": "conjugate addition",
        "passed": False,
        "score": score,
        "scoring_version": "v2",
        "step_count": steps,
        "run_metadata": run_metadata,
        "scoring_breakdown": {
            "accepted_path_step_count": steps,
            "final_known_product": _KNOWN_PRODUCT,
            "final_product_reached": reached,
            "known_alignment_component": 1.0 if reached else 0.25,
            "step_validity_component": 0.1,
            "known_step_count": 2,
            "step_breakdown": [{"resulting_state": [_KNOWN_PRODUCT], "step_index": 1}],
        },
    }


def _seed_baseline(
    tmp_path: Path,
    *,
    run_group: str,
    purpose: str = "general",
    model: str = "anthropic/claude-opus-4.6",
    thinking: Any = "high",
    origin: Any = None,
) -> tuple:
    from mechanistic_agent.core.db import RunStore

    store = RunStore(tmp_path / f"{run_group}_{model.replace('/', '_')}.db")
    cases = [
        {"case_id": cid, "input": {"starting_materials": ["C=CC#N", "C1=CCC=C1"], "products": [_KNOWN_PRODUCT]},
         "expected": {"n_mechanistic_steps": 2}}
        for cid in ("flower_1", "flower_2", "flower_3")
    ]
    set_id = store.add_eval_set(name="s", version="1", source_path=None, sha256=None, cases=cases, purpose=purpose)
    eval_run_id = store.create_eval_run(
        eval_set_id=set_id, run_group_name=run_group, model=model, model_name=model, harness_bundle_hash=None,
        thinking_level=thinking, metadata={"origin": origin} if origin else {}, status="completed",
    )
    rows = [
        ("flower_1", 0.8, 20_000.0, _baseline_summary("flower_1", 0.8, True, 2, thinking=thinking, origin=origin)),
        ("flower_2", 0.4, 40_000.0, _baseline_summary("flower_2", 0.4, False, 3, thinking=thinking, origin=origin)),
        ("flower_3", 0.0, 3_000.0, {"error": "Connection error.", "eval_mode": "baseline"}),
    ]
    for case_id, score, latency, summary in rows:
        store.record_eval_run_result(
            eval_run_id=eval_run_id, case_id=case_id, run_id=None, score=score, passed=False, cost={},
            latency_ms=latency, summary=summary,
        )
    return store, eval_run_id


def test_export_baseline_run_from_summaries(tmp_path: Path) -> None:
    store, eval_run_id = _seed_baseline(tmp_path, run_group="harness_free_baseline_easy")

    record = rp.export_eval_run(store, eval_run_id)

    assert record["kind"] == "baseline"
    assert record["tier"] == "easy" and record["holdout"] is False
    assert (record["model"], record["thinking_level"], record["origin"]) == ("anthropic/claude-opus-4.6", "high", None)
    s = record["summary"]
    assert (s["cases"], s["targets_reached"], s["passed"], s["errors"]) == (3, 1, 0, 1)
    assert s["mean_score"] == 0.4 and s["points"] == 400
    assert s["avg_latency_s"] == 21.0
    by_id = {c["case_id"]: c for c in record["cases"]}
    assert by_id["flower_1"]["target_reached"] is True and by_id["flower_1"]["predicted_steps"] == 2
    assert by_id["flower_1"]["known_steps"] == 2 and by_id["flower_1"]["mechanism_type"] == "conjugate addition"
    assert by_id["flower_3"]["error"] == "Connection error." and by_id["flower_3"]["score"] == 0.0
    assert "hardest_solved" not in record and rp.mechanism_image_path(record) is None
    assert _KNOWN_PRODUCT not in json.dumps(record)  # baselines never copy SMILES


def test_export_bridge_baseline_takes_origin_from_eval_run(tmp_path: Path) -> None:
    store, eval_run_id = _seed_baseline(
        tmp_path, run_group="harness_free_baseline_hard", model="agent-bridge", thinking=None, origin=_BRIDGE_ORIGIN,
    )
    record = rp.export_eval_run(store, eval_run_id)
    assert record["kind"] == "baseline" and record["tier"] == "hard"
    assert record["origin"]["declared_underlying_model"] == "claude-opus-5-5"
    assert rp._model_label(record) == "**Claude Opus 5.5** †"


def test_baseline_ground_truth_replays_are_refused(tmp_path: Path) -> None:
    store, eval_run_id = _seed_baseline(
        tmp_path, run_group="harness_free_baseline_easy", model="agent-bridge",
        origin={**_BRIDGE_ORIGIN, "responder_saw_ground_truth": True},
    )
    with pytest.raises(rp.PublishError, match="ground-truth replay"):
        rp.export_eval_run(store, eval_run_id)


def test_holdout_baseline_record_has_no_chemistry(tmp_path: Path) -> None:
    store, eval_run_id = _seed_baseline(tmp_path, run_group="harness_free_baseline", purpose="leaderboard_holdout")

    record = rp.export_eval_run(store, eval_run_id)

    assert record["kind"] == "baseline" and record["tier"] == "holdout" and record["holdout"] is True
    assert "cases" not in record and "hardest_solved" not in record
    assert record["summary"]["cases"] == 3
    dumped = json.dumps(record)
    for smiles in (_KNOWN_PRODUCT, "C=CC#N", "C1=CCC=C1"):
        assert smiles not in dumped
    assert "conjugate addition" not in dumped


def test_baselines_render_in_their_own_section_not_best_by_tier(tmp_path: Path) -> None:
    _write_records(tmp_path)
    seeds = [
        ("harness_free_baseline_easy", "general", "anthropic/claude-opus-4.6", "high", None),
        ("harness_free_baseline_easy", "general", "agent-bridge", None, _BRIDGE_ORIGIN),
        ("harness_free_baseline", "leaderboard_holdout", "anthropic/claude-opus-4.6", "high", None),
        ("harness_free_baseline_hard", "general", "anthropic/claude-opus-4.6", "high", None),
    ]
    for i, (group, purpose, model, thinking, origin) in enumerate(seeds):
        db_dir = tmp_path / f"db{i}"
        db_dir.mkdir()
        store, eval_run_id = _seed_baseline(
            db_dir, run_group=group, purpose=purpose, model=model, thinking=thinking, origin=origin,
        )
        record = rp.export_eval_run(store, eval_run_id)
        if origin:
            record["summary"]["mean_score"], record["summary"]["points"] = 0.95, 950
        rp.write_record(record, tmp_path, render_image=False)

    records = rp.load_records(tmp_path)
    assert len(records) == 6  # distinct files even when date + run group collide
    best = rp.best_by_tier(records)
    assert all(r.get("kind") != "baseline" for r in best.values())

    board = rp.render_leaderboard_markdown(records)
    assert '<a id="best-model-by-tier"></a>' in board
    assert '<a id="harness-free-baselines"></a>' in board
    assert board.index("## Best model by tier") < board.index("## Harness-free baselines") < board.index(
        "## Published runs"
    )
    section = board.split("## Harness-free baselines", 1)[1].split("## Published runs", 1)[0]
    rows = [line for line in section.splitlines() if line.startswith("| **")]
    assert len(rows) == 4
    # Sorted by tier (easy, medium, hard, holdout), then score descending: the bridge row leads easy.
    assert rows[0].startswith("| **Claude Opus 5.5** † | — | easy | 3 | 950 | 1/3 |")
    assert "| easy |" in rows[1] and "| hard |" in rows[2] and "| holdout |" in rows[3]
    assert "not validator-checked" in section
    assert board.count(rp.PROVENANCE_NOTE) == 1
    # Baselines get no per-run harness section.
    published = board.split("## Published runs", 1)[1]
    assert "harness_free_baseline" not in published


def test_combined_runs_later_run_wins_per_case() -> None:
    class _Resumed(_FakeStore):
        def list_eval_run_results(self, eval_run_id: str) -> List[Dict[str, Any]]:
            if eval_run_id == "first":
                return [dict(r) for r in self.results]  # c_fail came from a responder outage
            return [{"case_id": "c_fail", "run_id": "r2", "score": 0.9, "pass_bool": True, "latency_ms": 100_000}]

    record = rp.export_eval_run(_Resumed(), ["first", "resume"], expected_resolver=_resolver)
    assert record["eval_run_id"] == "resume"
    assert record["run_group"] == "grp+resumed"
    by_case = {c["case_id"]: c for c in record["cases"]}
    assert by_case["c_fail"]["passed"] is True  # replaced by the resumed result
    assert [s["cases"] for s in record["sources"]] == [["c_easy", "c_hard"], ["c_fail"]]
