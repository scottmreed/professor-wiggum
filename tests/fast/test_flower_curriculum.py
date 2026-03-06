from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("rdkit")
from mechanistic_agent.core import RunStore
from mechanistic_agent.flower_curriculum import (
    build_curriculum_index,
    build_lookup_cache,
    current_curriculum_step_count,
    curriculum_history,
    load_mechanism_reactions,
    next_curriculum_candidates,
)


def _fixture_lines() -> list[str]:
    return [
        "[Cl-:1].[CH3:2][Br:3]>>[CH3:2][Cl:1].[Br-:3]|21\n",
        "[O-:1].[H+:2]>>[O:1][H:2]|30\n",
        "[O-:1].[CH2:2]=[O:3]>>[O:1][CH2:2][O-:3]|30\n",
        "[Cl-:1].[CH3:2][Br:3]>>[CH3:2][Cl:1].[Br-:3]|21\n",
        "[O-:1].[H+:2]>>[O:1][H:2]|11\n",
    ]


def _write_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "flower_fixture.txt"
    path.write_text("".join(_fixture_lines()), encoding="utf-8")
    return path


def test_lookup_cache_reconstructs_noncontiguous_mechanism(tmp_path: Path) -> None:
    input_path = _write_fixture(tmp_path)
    cache_path = tmp_path / "lookup.sqlite"
    build_lookup_cache(input_path=input_path, cache_path=cache_path)

    reactions = load_mechanism_reactions(21, input_path=input_path, cache_path=cache_path)
    assert len(reactions) == 2
    assert reactions[0].endswith(">>[CH3:2][Cl:1].[Br-:3]")
    assert reactions[1].endswith(">>[CH3:2][Cl:1].[Br-:3]")


def test_curriculum_index_ranks_by_step_then_complexity(tmp_path: Path) -> None:
    input_path = _write_fixture(tmp_path)
    entries, report = build_curriculum_index(input_path=input_path)

    assert report["total_grouped_mechanisms_discovered"] == 3
    assert [entry.case_id for entry in entries] == [
        "flower_000011",
        "flower_000030",
        "flower_000021",
    ]
    assert [entry.rank_within_step_count for entry in entries] == [1, 1, 2]


def test_curriculum_progress_is_scoped_by_model_harness_and_index(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "mechanistic.db")
    eval_set_id = store.add_eval_set(
        name="flower_100_default",
        version="test",
        source_path="training_data/eval_set.json",
        sha256=None,
        cases=[],
        active=True,
    )

    run_id = store.create_eval_run(
        eval_set_id=eval_set_id,
        run_group_name="curriculum_default_s1_r1_n4",
        model="gpt-5.2",
        model_name="gpt-5.2",
        model_family="openai",
        thinking_level=None,
        harness_bundle_hash="",
        status="completed",
    )
    for idx, passed in enumerate([True, True, True, False], start=1):
        store.record_eval_run_result(
            eval_run_id=run_id,
            case_id=f"flower_{idx:06d}",
            run_id=None,
            score=0.9 if passed else 0.1,
            passed=passed,
            cost={"total_cost": 0.0},
            latency_ms=0.0,
            summary={
                "curriculum_case_id": f"flower_{idx:06d}",
                "step_count": 1,
                "global_rank": idx,
                "rank_within_step_count": idx,
                "group_size": 4,
                "promotion_threshold": 3,
                "curriculum_index_path": str(tmp_path / "idx.jsonl"),
                "curriculum_harness": "default",
                "batch_start_rank": 1,
            },
        )

    other_run_id = store.create_eval_run(
        eval_set_id=eval_set_id,
        run_group_name="curriculum_other_s1_r1_n4",
        model="gpt-5.2",
        model_name="gpt-5.2",
        model_family="openai",
        thinking_level=None,
        harness_bundle_hash="",
        status="completed",
    )
    store.record_eval_run_result(
        eval_run_id=other_run_id,
        case_id="flower_999999",
        run_id=None,
        score=1.0,
        passed=True,
        cost={"total_cost": 0.0},
        latency_ms=0.0,
        summary={
            "curriculum_case_id": "flower_999999",
            "step_count": 2,
            "global_rank": 99,
            "rank_within_step_count": 1,
            "group_size": 4,
            "promotion_threshold": 3,
            "curriculum_index_path": str(tmp_path / "other.jsonl"),
            "curriculum_harness": "other",
            "batch_start_rank": 99,
        },
    )

    history = curriculum_history(
        store,
        model_name="gpt-5.2",
        harness="default",
        curriculum_index_path=tmp_path / "idx.jsonl",
    )
    assert history["highest_successful_step_count"] == 1
    assert history["pass_count_by_step"] == {1: 3}
    assert history["attempted_case_ids"] == {
        "flower_000001",
        "flower_000002",
        "flower_000003",
        "flower_000004",
    }


def test_current_curriculum_step_count_waits_for_50_passes_before_advancing() -> None:
    entries = [
        {"case_id": "flower_000001", "mechanism_id": 1, "step_count": 1, "global_rank": 1, "rank_within_step_count": 1},
        {"case_id": "flower_000002", "mechanism_id": 2, "step_count": 1, "global_rank": 2, "rank_within_step_count": 2},
        {"case_id": "flower_000003", "mechanism_id": 3, "step_count": 2, "global_rank": 3, "rank_within_step_count": 1},
        {"case_id": "flower_000004", "mechanism_id": 4, "step_count": 3, "global_rank": 4, "rank_within_step_count": 1},
    ]

    assert current_curriculum_step_count(entries, pass_count_by_step={}, required_passes_per_step=50) == 1
    assert current_curriculum_step_count(entries, pass_count_by_step={1: 49}, required_passes_per_step=50) == 1
    assert current_curriculum_step_count(entries, pass_count_by_step={1: 50}, required_passes_per_step=50) == 2
    assert current_curriculum_step_count(entries, pass_count_by_step={1: 50, 2: 50}, required_passes_per_step=50) == 3


def test_next_curriculum_candidates_stays_within_current_step_count() -> None:
    entries = [
        {"case_id": "flower_000001", "mechanism_id": 1, "step_count": 1, "global_rank": 1, "rank_within_step_count": 1},
        {"case_id": "flower_000002", "mechanism_id": 2, "step_count": 1, "global_rank": 2, "rank_within_step_count": 2},
        {"case_id": "flower_000003", "mechanism_id": 3, "step_count": 2, "global_rank": 3, "rank_within_step_count": 1},
        {"case_id": "flower_000004", "mechanism_id": 4, "step_count": 2, "global_rank": 4, "rank_within_step_count": 2},
    ]

    bootstrap = next_curriculum_candidates(
        entries,
        attempted_case_ids=set(),
        pass_count_by_step={},
        required_passes_per_step=50,
    )
    assert bootstrap["current_step_count"] == 1
    assert bootstrap["current_step_pass_count"] == 0
    assert [item["case_id"] for item in bootstrap["candidates"]] == [
        "flower_000001",
        "flower_000002",
    ]

    progressed = next_curriculum_candidates(
        entries,
        attempted_case_ids={"flower_000001"},
        pass_count_by_step={1: 50},
        required_passes_per_step=50,
    )
    assert progressed["current_step_count"] == 2
    assert progressed["current_step_pass_count"] == 0
    assert [item["case_id"] for item in progressed["candidates"]] == [
        "flower_000003",
        "flower_000004",
    ]
