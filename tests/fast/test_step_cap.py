"""Step cap 14 everywhere, and a run that hits it says so (event + stderr + eval summary)."""
from __future__ import annotations

import inspect
import threading

import main
from mechanistic_agent.api.schemas import CreateRunRequest
from mechanistic_agent.core.coordinator import RunCoordinator
from mechanistic_agent.core.db import RunStore
from mechanistic_agent.core.types import DEFAULT_MAX_STEPS, RunConfig


def test_default_step_cap_is_14_everywhere() -> None:
    assert DEFAULT_MAX_STEPS == 14
    assert RunConfig(model="gpt-5").max_steps == 14
    assert CreateRunRequest(starting_materials=["CCO"], products=["CC=O"]).max_steps == 14
    for command in (main.run, main.eval_cmd):
        default = inspect.signature(command).parameters["max_steps"].default
        assert getattr(default, "default", default) == 14, command.__name__


def test_stored_run_without_max_steps_resolves_to_14(tmp_path) -> None:
    store = RunStore(tmp_path / "data" / "mechanistic.db")
    run_id = store.create_run(
        mode="unverified",
        input_payload={"starting_materials": ["CCBr", "[Cl-]"], "products": ["CCCl", "[Br-]"]},
        config={"model": "gpt-5"},
        prompt_bundle_hash="p",
        skill_bundle_hash="s",
        memory_bundle_hash="m",
    )
    state = RunCoordinator(store)._build_state(store.get_run_row(run_id))
    assert state.run_config.max_steps == 14


def _run_to_cap(tmp_path, *, cap: int, reach: int):  # type: ignore[no-untyped-def]
    store = RunStore(tmp_path / "data" / "mechanistic.db")
    run_id = store.create_run(
        mode="unverified",
        input_payload={"starting_materials": ["CCBr", "[Cl-]"], "products": ["CCCl", "[Br-]"]},
        config={
            "model": "gpt-5",
            "max_steps": cap,
            "max_runtime_seconds": 600,
            "runtime_trace_label": "flower-test-002482",
        },
        prompt_bundle_hash="p",
        skill_bundle_hash="s",
        memory_bundle_hash="m",
    )
    coordinator = RunCoordinator(store)
    coordinator._run_initial_phase = lambda *_a, **_k: None  # type: ignore[method-assign]
    coordinator._run_post_loop_phase = lambda *_a, **_k: None  # type: ignore[method-assign]
    coordinator._run_overall_balance_reconciliation = lambda *_a, **_k: None  # type: ignore[method-assign]

    def _fake_loop(state, _stop_event, _harness):  # type: ignore[no-untyped-def]
        for step in range(1, reach + 1):
            store.record_step_output(
                run_id=state.run_id,
                step_name="mechanism_synthesis",
                attempt=step,
                source="deterministic",
                model=None,
                reasoning_level=None,
                tool_name="predict_mechanistic_step",
                output={"contains_target_product": False, "resulting_state": ["CCBr", "[Cl-]"]},
                validation={"passed": True, "checks": []},
            )
        state.step_index = reach

    coordinator._run_mechanism_loop = _fake_loop  # type: ignore[method-assign]
    coordinator.execute_run(run_id, threading.Event())
    return store, run_id


def test_run_that_uses_every_step_emits_step_cap_reached_and_warns(tmp_path, capsys) -> None:
    store, run_id = _run_to_cap(tmp_path, cap=14, reach=14)
    events = store.list_events(run_id)
    cap_events = [e["payload"] for e in events if e["event_type"] == "step_cap_reached"]
    assert cap_events == [{"cap": 14, "steps": 14, "run_id": run_id, "case_id": "flower-test-002482"}]
    failed = [e["payload"] for e in events if e["event_type"] == "run_failed"][-1]
    assert failed["reason"] == "completion_not_reached_within_limits"
    assert failed["step_cap_reached"] is True
    err = capsys.readouterr().err
    assert f"WARNING step cap reached: run={run_id} case=flower-test-002482 cap=14 steps=14" in err
    assert main._run_hit_step_cap(store, run_id)


def test_run_that_stops_early_does_not_claim_the_cap(tmp_path, capsys) -> None:
    store, run_id = _run_to_cap(tmp_path, cap=14, reach=3)
    assert not [e for e in store.list_events(run_id) if e["event_type"] == "step_cap_reached"]
    assert "step cap reached" not in capsys.readouterr().err
    assert not main._run_hit_step_cap(store, run_id)


def test_eval_summary_line_counts_cap_hits() -> None:
    assert main._format_step_cap_summary([], total=5, cap=14) == "Step cap (14) reached: 0/5 cases"
    line = main._format_step_cap_summary(["mcsa-613-1", "flower-test-002482"], total=92, cap=14)
    assert line == "Step cap (14) reached: 2/92 cases — mcsa-613-1, flower-test-002482"
