"""RunCoordinator provenance recording (Observatory PRD §12.3, §13.3, §17, M0).

Uses a real SQLite ``RunStore`` so the persisted ``step_outputs.model`` column
and the event log are both checked, and so a replay from events can be
compared against the rows.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

pytest.importorskip("rdkit")

from mechanistic_agent.core.coordinator import RunCoordinator
from mechanistic_agent.core.db import RunStore
from mechanistic_agent.core.provenance import step_provenance_from_events
from mechanistic_agent.core.types import (
    RunConfig,
    RunInput,
    RunState,
    StepResult,
    StepValidationCheck,
    StepValidationResult,
)

CONFIGURED = "anthropic/claude-opus-4.6"


def _setup(tmp_path: Path) -> tuple[RunCoordinator, RunStore, RunState]:
    store = RunStore(tmp_path / "mechanistic.db")
    run_input = RunInput(starting_materials=["CCBr", "[Cl-]"], products=["CCCl", "[Br-]"])
    run_config = RunConfig(
        model=CONFIGURED,
        model_family="claude",
        max_steps=1,
        intermediate_prediction_enabled=True,
        step_models={"mechanism_synthesis": CONFIGURED, "intermediates": CONFIGURED},
        step_reasoning={"intermediates": "high"},
    )
    run_id = store.create_run(
        mode="unverified",
        input_payload={"starting_materials": run_input.starting_materials, "products": run_input.products},
        config=run_config.as_dict() if hasattr(run_config, "as_dict") else {"model": CONFIGURED},
        prompt_bundle_hash="a",
        skill_bundle_hash="b",
        memory_bundle_hash="c",
    )
    state = RunState(run_id=run_id, mode="unverified", run_input=run_input, run_config=run_config)
    state.initialise()
    return RunCoordinator(store=store), store, state


def _events(store: RunStore, run_id: str, kind: str) -> List[Dict[str, Any]]:
    return [e for e in store.list_events(run_id) if e.get("event_type") == kind]


def test_deterministic_step_is_stored_without_the_run_model(tmp_path: Path) -> None:
    coordinator, store, state = _setup(tmp_path)
    result = StepResult(
        step_name="mechanism_synthesis",
        tool_name="predict_mechanistic_step",
        output={"status": "accepted"},
        source="deterministic",
        validation=StepValidationResult(checks=[StepValidationCheck(name="atom_balance", passed=True, details={})]),
    )

    coordinator._record_step(state, result)

    rows = store.list_step_outputs(state.run_id)
    assert [r["step_name"] for r in rows] == ["mechanism_synthesis"]
    assert rows[0]["model"] is None
    assert rows[0]["reasoning_level"] is None
    step_output = _events(store, state.run_id, "step_output")[0]["payload"]
    assert step_output["provenance"]["engine"] == "deterministic"
    assert step_output["provenance"]["resolved_model"] is None
    assert step_output["provenance"]["tool"] == "predict_mechanistic_step"
    assert _events(store, state.run_id, "inference_call_completed") == []


def test_llm_step_emits_inference_call_completed_and_summary(tmp_path: Path) -> None:
    coordinator, store, state = _setup(tmp_path)
    result = StepResult(
        step_name="mechanism_step_proposal",
        tool_name="propose_mechanism_step",
        output={"candidates": []},
        source="llm",
        token_usage={"total_tokens": 1200},
    )

    coordinator._record_step(state, result)

    rows = store.list_step_outputs(state.run_id)
    assert rows[0]["model"] == CONFIGURED
    calls = _events(store, state.run_id, "inference_call_completed")
    assert len(calls) == 1
    call = calls[0]["payload"]
    assert call["engine"] == "llm"
    assert call["role"] == "mechanism_proposal"
    assert call["resolved_model"] == CONFIGURED
    assert call["usage"] == {"total_tokens": 1200}
    assert call["event_schema_version"] == "mechanism_observatory_event.v1"
    step_output = _events(store, state.run_id, "step_output")[0]
    prov = step_output["payload"]["provenance"]
    assert prov["engine"] == "llm"
    assert prov["resolved_model"] == CONFIGURED
    assert prov["primary_call_id"] == call["call_id"]
    # call event is persisted before the step_output that points at it
    assert calls[0]["seq"] < step_output["seq"]


def test_llm_fallback_model_is_the_stored_model(tmp_path: Path) -> None:
    coordinator, store, state = _setup(tmp_path)
    result = StepResult(
        step_name="mechanism_step_proposal",
        tool_name="propose_mechanism_step",
        output={"candidates": [], "model_used": "openai/gpt-5.6-sol"},
        model=CONFIGURED,
        source="llm",
    )

    coordinator._record_step(state, result)

    assert store.list_step_outputs(state.run_id)[0]["model"] == "openai/gpt-5.6-sol"
    call = _events(store, state.run_id, "inference_call_completed")[0]["payload"]
    assert call["requested_model"] == CONFIGURED
    assert call["resolved_model"] == "openai/gpt-5.6-sol"
    assert call["model_fallback"] is True


def test_jev_failure_then_llm_fallback_emits_both_calls_in_order(tmp_path: Path) -> None:
    coordinator, store, state = _setup(tmp_path)
    result = StepResult(
        step_name="reaction_type_mapping",
        tool_name="select_reaction_type",
        output={
            "model_used": CONFIGURED,
            "decision_engine": "llm",
            "decision_trace": [
                {
                    "decision_engine": "jev",
                    "decision_type": "choice",
                    "model": "typesafe/jev-1.13",
                    "model_version": None,
                    "provider": "openrouter",
                    "request_id": "local-1",
                    "called": True,
                    "failure": "timeout",
                    "latency_ms": 30000.0,
                }
            ],
        },
        model=CONFIGURED,
        source="llm",
    )

    coordinator._record_step(state, result)

    failed = _events(store, state.run_id, "inference_call_failed")
    completed = _events(store, state.run_id, "inference_call_completed")
    assert len(failed) == 1 and len(completed) == 1
    assert failed[0]["payload"]["engine"] == "jev"
    assert failed[0]["payload"]["error"] == "timeout"
    assert completed[0]["payload"]["engine"] == "llm"
    assert completed[0]["payload"]["fallback_from_call_id"] == "local-1"
    assert failed[0]["seq"] < completed[0]["seq"]
    prov = _events(store, state.run_id, "step_output")[0]["payload"]["provenance"]
    assert prov["fallback_chain"] == ["jev", "llm"]


def test_jev_step_stores_reported_revision_as_model(tmp_path: Path) -> None:
    coordinator, store, state = _setup(tmp_path)
    result = StepResult(
        step_name="reaction_type_mapping",
        tool_name="select_reaction_type",
        output={
            "model_used": "typesafe/jev-1.13",
            "decision_engine": "jev",
            "decision_trace": [
                {
                    "decision_engine": "jev",
                    "decision_type": "choice",
                    "model": "typesafe/jev-1.13",
                    "model_version": "typesafe/jev-1.13-20260917",
                    "provider": "openrouter",
                    "request_id": "gen-1",
                    "called": True,
                    "failure": None,
                    "latency_ms": 110.0,
                    "usage": {"total_tokens": 800},
                }
            ],
        },
        model="typesafe/jev-1.13",
        source="jev",
    )

    coordinator._record_step(state, result)

    assert store.list_step_outputs(state.run_id)[0]["model"] == "typesafe/jev-1.13-20260917"
    call = _events(store, state.run_id, "inference_call_completed")[0]["payload"]
    assert call["engine"] == "jev"
    assert call["call_id"] == "gen-1"
    assert call["decision_type"] == "choice"


def test_step_started_carries_planned_engine_and_model(tmp_path: Path) -> None:
    coordinator, store, state = _setup(tmp_path)

    coordinator._mark_step_started(state, step_name="mechanism_step_proposal", tool_name="propose_mechanism_step", attempt=1)
    coordinator._mark_step_started(state, step_name="atom_balance_validation", tool_name="atom_balance_validation", attempt=1)

    started = _events(store, state.run_id, "step_started")
    proposal, validator = started[0]["payload"], started[1]["payload"]
    assert proposal["planned_engine"] == "llm"
    assert proposal["planned_model"] == CONFIGURED
    assert proposal["planned_reasoning"] == "high"
    assert validator["planned_engine"] == "deterministic"
    assert validator["planned_model"] is None


def test_replay_from_events_matches_persisted_rows(tmp_path: Path) -> None:
    coordinator, store, state = _setup(tmp_path)
    coordinator._record_step(
        state,
        StepResult(step_name="mechanism_step_proposal", tool_name="propose_mechanism_step", output={}, source="llm"),
    )
    coordinator._record_step(
        state,
        StepResult(step_name="mechanism_synthesis", tool_name="predict_mechanistic_step", output={}, source="deterministic"),
    )
    coordinator._record_step(
        state,
        StepResult(step_name="bond_electron_validation", tool_name="bond_electron_validation", output={}, source="deterministic"),
    )

    replay = step_provenance_from_events(store.list_events(state.run_id))
    rows = {r["step_name"]: r for r in store.list_step_outputs(state.run_id)}
    assert set(replay) == set(rows)
    for step_name, row in rows.items():
        assert replay[step_name]["resolved_model"] == row["model"], step_name
        assert replay[step_name]["engine"] == row["source"], step_name
