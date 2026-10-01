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
