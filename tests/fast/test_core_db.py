from __future__ import annotations

from pathlib import Path

from mechanistic_agent.core.db import RunStore


def test_run_store_persists_run_event_and_step(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "mechanistic.db")

    run_id = store.create_run(
        mode="unverified",
        input_payload={"starting_materials": ["C=O"], "products": ["CO"]},
        config={"model": "gpt-5", "max_steps": 1},
        prompt_bundle_hash="a",
        skill_bundle_hash="b",
        memory_bundle_hash="c",
    )

    seq = store.append_event(run_id, "test_event", {"ok": True})
    assert seq == 1

    store.record_step_output(
        run_id=run_id,
        step_name="mechanism_synthesis",
        attempt=1,
        model="gpt-5",
        reasoning_level="low",
        tool_name="predict_mechanistic_step",
        output={"contains_target_product": True},
        validation={"passed": True, "checks": []},
    )

    snapshot = store.get_run_snapshot(run_id)
    assert snapshot is not None
    assert snapshot["id"] == run_id
    assert snapshot["events"][0]["event_type"] == "test_event"
    assert snapshot["step_outputs"][0]["step_name"] == "mechanism_synthesis"


def test_eval_and_few_shot_storage(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "mechanistic.db")
    (tmp_path / "traces" / "runs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "traces" / "evidence").mkdir(parents=True, exist_ok=True)

    eval_set_id = store.add_eval_set(
        name="default_examples",
        version="v1",
        source_path="data/mechanism_examples.json",
        sha256=None,
        cases=[
            {
                "case_id": "ex-1",
                "input": {"starting_materials": ["C=O"], "products": ["CO"]},
                "expected": {"products": ["CO"]},
                "tags": ["default"],
            }
        ],
        active=True,
    )
    eval_run_id = store.create_eval_run(
        eval_set_id=eval_set_id,
        run_group_name="baseline",
        model="gpt-5-mini",
        model_name="gpt-5-mini",
        model_family="openai",
        thinking_level="low",
        harness_bundle_hash="abc",
        status="running",
    )
    store.record_eval_run_result(
        eval_run_id=eval_run_id,
        case_id="ex-1",
        run_id=None,
        score=0.9,
        passed=True,
        cost={"total_cost": 0.01},
        latency_ms=1200.0,
        summary={"ok": True},
    )
    store.set_eval_run_status(eval_run_id, "completed")

    leaderboard = store.leaderboard(eval_set_id)
    assert leaderboard
    assert leaderboard[0]["eval_run_id"] == eval_run_id
    assert leaderboard[0]["mean_quality_score"] == 0.9
    assert leaderboard[0]["model_name"] == "gpt-5-mini"
    assert leaderboard[0]["thinking_level"] == "low"

    prompt_ids = store.upsert_prompt_versions(
        [
            {
                "name": "assess_initial_conditions",
                "call_name": "assess_initial_conditions",
                "step": "initial_conditions",
                "version": "bundle-1",
                "path": "prompt_versions/calls/assess_initial_conditions/base.md",
                "sha256": "bundle-1",
                "prompt_bundle_sha256": "bundle-1",
                "shared_base_sha256": "shared",
                "call_base_sha256": "base",
                "few_shot_sha256": "few",
                "template": "template",
            }
        ]
    )
    model_version_id = store.upsert_model_version(model_name="gpt-5", reasoning_level="lowest")
    assert model_version_id

    trace_id = store.add_trace_record(
        step_name="mechanism_synthesis",
        trace={"tool_name": "predict_mechanistic_step", "output": {"contains_target_product": True}},
        source="baseline",
        model="gpt-5",
        reasoning_level="lowest",
        prompt_version_id=prompt_ids.get("initial_conditions"),
        model_version_id=model_version_id,
        score=1.0,
    )
    trace = store.get_trace_record(trace_id)
    assert trace is not None
    assert trace["step_name"] == "mechanism_synthesis"
    assert trace.get("model_version_id") == model_version_id
    trace_files = list((tmp_path / "traces" / "runs" / "unassigned").glob(f"*_{trace_id}.json"))
    assert trace_files

    fs_id = store.add_few_shot_example(
        step_name="mechanism_synthesis",
        example_key="baseline-1",
        input_text='{"step":"mechanism_synthesis"}',
        output_text='{"contains_target_product":true}',
        approved=True,
        source_trace_id=trace_id,
        score=1.0,
    )
    assert fs_id
    few_shot = store.list_few_shot_examples(step_name="mechanism_synthesis", approved_only=True)
    assert few_shot
    assert few_shot[0]["example_key"] == "baseline-1"


def test_arrow_push_annotation_storage_roundtrip(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "mechanistic.db")
    run_id = store.create_run(
        mode="unverified",
        input_payload={"starting_materials": ["CCBr", "[Cl-]"], "products": ["CCCl", "[Br-]"]},
        config={"model": "gpt-5", "max_steps": 1, "arrow_push_annotation_enabled": True},
        prompt_bundle_hash="a",
        skill_bundle_hash="b",
        memory_bundle_hash="c",
    )

    record_id = store.record_arrow_push_annotation(
        run_id=run_id,
        step_index=1,
        attempt=1,
        retry_index=0,
        candidate_rank=1,
        source="mechanism_loop",
        prediction={
            "annotation_suffix": "|aps:v1;src=2;snk=4;e=2;tpl=sn2_substitution;sc=0.900|",
            "selected_candidate": {"source_atom_ref": "2", "sink_atom_ref": "4", "template": "sn2_substitution"},
        },
    )
    assert record_id

    rows = store.list_arrow_push_annotations(run_id)
    assert len(rows) == 1
    assert rows[0]["run_id"] == run_id
    assert rows[0]["step_index"] == 1
    assert rows[0]["prediction"]["annotation_suffix"].startswith("|aps:v1;")


def test_eval_run_results_many_and_leaderboard_batching(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "mechanistic.db")
    eval_set_id = store.add_eval_set(
        name="batch_eval",
        version="v1",
        source_path="training_data/batch_eval.json",
        sha256=None,
        cases=[
            {
                "case_id": "case-1",
                "input": {"starting_materials": ["CCO"], "products": ["CC=O"]},
                "expected": {},
                "tags": [],
            }
        ],
        active=True,
    )

    run_a = store.create_eval_run(
        eval_set_id=eval_set_id,
        run_group_name="g1",
        model="gpt-5-mini",
        model_name="gpt-5-mini",
        model_family="openai",
        thinking_level="low",
        harness_bundle_hash="h1",
        status="completed",
    )
    run_b = store.create_eval_run(
        eval_set_id=eval_set_id,
        run_group_name="g1",
        model="gpt-5",
        model_name="gpt-5",
        model_family="openai",
        thinking_level="high",
        harness_bundle_hash="h1",
        status="completed",
    )
    store.record_eval_run_result(
        eval_run_id=run_a,
        case_id="case-1",
        run_id=None,
        score=0.4,
        passed=False,
        cost={"total_cost": 0.02},
        latency_ms=100.0,
        summary={},
    )
    store.record_eval_run_result(
        eval_run_id=run_b,
        case_id="case-1",
        run_id=None,
        score=0.8,
        passed=True,
        cost={"total_cost": 0.05},
        latency_ms=120.0,
        summary={},
    )

    grouped = store.list_eval_run_results_many([run_a, run_b])
    assert len(grouped[run_a]) == 1
    assert len(grouped[run_b]) == 1
    assert grouped[run_a][0]["cost"]["total_cost"] == 0.02
    assert grouped[run_b][0]["pass_bool"] is True

    leaderboard = store.leaderboard(eval_set_id)
    assert leaderboard[0]["eval_run_id"] == run_b
    assert leaderboard[0]["model_name"] == "gpt-5"
    assert leaderboard[0]["thinking_level"] == "high"


def test_eval_set_visibility_filters_and_holdout_weighted_leaderboard(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "mechanistic.db")

    general_eval_set_id = store.add_eval_set(
        name="general_eval",
        version="v1",
        source_path="training_data/eval_set.json",
        sha256=None,
        cases=[],
        active=True,
        purpose="general",
        exposed_in_ui=True,
    )
    holdout_eval_set_id = store.add_eval_set(
        name="official_holdout",
        version="v1",
        source_path="training_data/leaderboard_holdout/eval_set_holdout.json",
        sha256=None,
        cases=[
            {
                "case_id": "h_step1_a",
                "input": {"starting_materials": ["C=O"], "products": ["CO"]},
                "expected": {"n_mechanistic_steps": 1},
                "tags": ["leaderboard_holdout"],
            },
            {
                "case_id": "h_step5_a",
                "input": {"starting_materials": ["CCO"], "products": ["CC=O"]},
                "expected": {"n_mechanistic_steps": 5},
                "tags": ["leaderboard_holdout"],
            },
            {
                "case_id": "h_step5_b",
                "input": {"starting_materials": ["CCC"], "products": ["CC=C"]},
                "expected": {"n_mechanistic_steps": 5},
                "tags": ["leaderboard_holdout"],
            },
        ],
        active=True,
        purpose="leaderboard_holdout",
        exposed_in_ui=False,
    )
    assert general_eval_set_id
    assert holdout_eval_set_id

    visible = store.list_eval_sets(exposed_in_ui=True)
    hidden = store.list_eval_sets(exposed_in_ui=False)
    holdouts = store.list_eval_sets(purpose="leaderboard_holdout")
    assert any(item["id"] == general_eval_set_id for item in visible)
    assert all(item["id"] != holdout_eval_set_id for item in visible)
    assert any(item["id"] == holdout_eval_set_id for item in hidden)
    assert any(item["id"] == holdout_eval_set_id for item in holdouts)

    run_strong = store.create_eval_run(
        eval_set_id=holdout_eval_set_id,
        run_group_name="official_holdout_harness",
        model="gpt-5",
        model_name="gpt-5",
        model_family="openai",
        thinking_level="high",
        harness_bundle_hash="h",
        status="completed",
    )
    run_weak = store.create_eval_run(
        eval_set_id=holdout_eval_set_id,
        run_group_name="official_holdout_harness",
        model="gpt-5-mini",
        model_name="gpt-5-mini",
        model_family="openai",
        thinking_level="low",
        harness_bundle_hash="h",
        status="completed",
    )

    # Strong run: weighted quality = (1*1.0 + 5*0.5) / 6 = 0.583333...
    store.record_eval_run_result(
        eval_run_id=run_strong,
        case_id="h_step1_a",
        run_id=None,
        score=1.0,
        passed=True,
        cost={"total_cost": 0.02},
        latency_ms=10.0,
        summary={"n_mechanistic_steps": 1},
    )
    store.record_eval_run_result(
        eval_run_id=run_strong,
        case_id="h_step5_a",
        run_id=None,
        score=0.5,
        passed=True,
        cost={"total_cost": 0.02},
        latency_ms=10.0,
        summary={"n_mechanistic_steps": 5},
    )
    store.record_eval_run_result(
        eval_run_id=run_strong,
        case_id="h_step5_b",
        run_id=None,
        score=0.5,
        passed=False,
        cost={"total_cost": 0.02},
        latency_ms=10.0,
        summary={"n_mechanistic_steps": 5},
    )

    # Weak run: weighted quality = (1*0.8 + 5*0.2) / 6 = 0.3
    store.record_eval_run_result(
        eval_run_id=run_weak,
        case_id="h_step1_a",
        run_id=None,
        score=0.8,
        passed=True,
        cost={"total_cost": 0.01},
        latency_ms=10.0,
        summary={"n_mechanistic_steps": 1},
    )
    store.record_eval_run_result(
        eval_run_id=run_weak,
        case_id="h_step5_a",
        run_id=None,
        score=0.2,
        passed=False,
        cost={"total_cost": 0.01},
        latency_ms=10.0,
        summary={"n_mechanistic_steps": 5},
    )
    store.record_eval_run_result(
        eval_run_id=run_weak,
        case_id="h_step5_b",
        run_id=None,
        score=0.2,
        passed=False,
        cost={"total_cost": 0.01},
        latency_ms=10.0,
        summary={"n_mechanistic_steps": 5},
    )

    leaderboard = store.leaderboard(holdout_eval_set_id, limit=10)
    assert leaderboard
    assert leaderboard[0]["eval_run_id"] == run_strong
    assert abs(float(leaderboard[0]["weighted_quality_score"]) - (3.5 / 6.0)) < 1e-6
    assert leaderboard[0]["aggregate_weighting"] == "linear_step_count"
    assert leaderboard[0]["aggregate_gate_cases"] == 6
    assert "1" in leaderboard[0]["per_step_scores"]
    assert "5" in leaderboard[0]["per_step_scores"]
