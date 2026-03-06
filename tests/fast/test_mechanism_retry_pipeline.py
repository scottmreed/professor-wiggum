from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mechanistic_agent.core.coordinator import RunCoordinator, _RunPaused
from mechanistic_agent.core.subagents import MechanismAgent
from mechanistic_agent.core.types import RunConfig, RunInput, RunState, StepResult, TemplateGuidanceState


class _EventStore:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.pauses: List[Dict[str, Any]] = []
        self.status_updates: List[Dict[str, str]] = []

    def append_event(
        self,
        run_id: str,
        event_type: str,
        payload: Dict[str, Any],
        *,
        step_name: Optional[str] = None,
    ) -> None:
        self.events.append(
            {
                "run_id": run_id,
                "event_type": event_type,
                "payload": payload,
                "step_name": step_name,
            }
        )

    def create_run_pause(self, *, run_id: str, reason: str, details: Dict[str, Any]) -> str:
        self.pauses.append({"run_id": run_id, "reason": reason, "details": details})
        return "pause-test-id"

    def set_run_status(self, run_id: str, status: str) -> None:
        self.status_updates.append({"run_id": run_id, "status": status})

    def list_step_outputs(self, run_id: str) -> List[Dict[str, Any]]:
        return []


def _state(max_steps: int = 1, max_runtime_seconds: float = 0.02) -> RunState:
    run_input = RunInput(starting_materials=["C=O", "OCCO"], products=["C1OCOC1"], ph=3.5, temperature_celsius=40.0)
    run_config = RunConfig(
        model="gpt-4",
        model_family="openai",
        max_steps=max_steps,
        max_runtime_seconds=max_runtime_seconds,
        intermediate_prediction_enabled=True,
    )
    state = RunState(run_id="run-test-id", mode="unverified", run_input=run_input, run_config=run_config)
    state.initialise()
    return state


def test_mechanism_agent_forwards_candidate_execution_fields() -> None:
    @dataclass
    class _CaptureExecutor:
        kwargs: Optional[Dict[str, Any]] = None

        def run_mechanism_step(self, **kwargs: Any) -> Dict[str, Any]:
            self.kwargs = kwargs
            return {"contains_target_product": False}

    executor = _CaptureExecutor()
    agent = MechanismAgent(executor=executor)  # type: ignore[arg-type]
    state = _state()
    output = {
        "selected_candidate": {
            "rank": 1,
            "intermediate_smiles": "C1OCOC1",
            "reaction_description": "Cyclization with proton transfer.",
            "reaction_smirks": "[CH2:1]=[O:2].[O:3][CH2:4][CH2:5][OH:6]>>[CH2:1]1[O:3][CH2:4][CH2:5][O:2]1.[OH2:6] |mech:v1;lp:3>1;pi:1-2>2|",
            "electron_pushes": [{"kind": "lone_pair", "source_atom": "3", "target_atom": "1", "electrons": 2}],
            "resulting_state": ["C1OCOC1", "O"],
        }
    }

    agent.run(state, output)
    assert executor.kwargs is not None
    assert executor.kwargs["reaction_smirks"].startswith("[CH2:1]=[O:2]")
    assert executor.kwargs["electron_pushes"] == [{"kind": "lone_pair", "source_atom": "3", "target_atom": "1", "electrons": 2}]
    assert executor.kwargs["resulting_state"] == ["C1OCOC1", "O"]


def test_loop_requests_reproposal_for_incomplete_candidate_payload() -> None:
    store = _EventStore()
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]
    state = _state(max_steps=1, max_runtime_seconds=0.02)
    state.template_guidance_state = TemplateGuidanceState(mode="active")

    coordinator._record_step = lambda *_args, **_kwargs: None  # type: ignore[method-assign]

    class _DummyIntermediateAgent:
        def run(self, _state: RunState) -> StepResult:
            return StepResult(
                step_name="mechanism_step_proposal",
                tool_name="propose_mechanism_step",
                output={
                    "classification": "intermediate_step",
                    "analysis": "Legacy incomplete candidate payload",
                    "candidates": [
                        {
                            "rank": 1,
                            "intermediate_smiles": "C1OCOC1",
                            "reaction_description": "Ring closure",
                        }
                    ],
                },
                source="llm",
            )

    coordinator.intermediate_agent = _DummyIntermediateAgent()  # type: ignore[assignment]
    coordinator._run_mechanism_loop(state, threading.Event())

    assert any(ev["event_type"] == "mechanism_candidate_incomplete" for ev in store.events)
    assert any(ev["event_type"] == "mechanism_reproposal_requested" for ev in store.events)
    assert any(ev["event_type"] == "proposal_quality_summary" for ev in store.events)
    assert any(ev["event_type"] == "template_guidance_preaccept_observation" for ev in store.events)
    assert state.step_index == 0


def test_candidate_ready_for_execution_accepts_modern_electron_push_schema() -> None:
    candidate = {
        "rank": 1,
        "intermediate_smiles": "C1OCOC1",
        "reaction_description": "Ring closure",
        "reaction_smirks": "[CH2:1]=[O:2].[O:3][CH2:4][CH2:5][OH:6]>>[CH2:1]1[O:3][CH2:4][CH2:5][O:2]1.[OH2:6] |mech:v1;lp:3>1;pi:1-2>2|",
        "electron_pushes": [{"kind": "lone_pair", "source_atom": "3", "target_atom": "1", "electrons": 2}],
    }

    ready, reason = RunCoordinator._candidate_ready_for_execution(candidate)

    assert ready is True
    assert reason == ""
    assert candidate["electron_pushes"][0]["notation"] == "lp:3>1"


def test_pause_for_retry_exhaustion_persists_last_validation_payload() -> None:
    store = _EventStore()
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]
    state = _state()

    validation = {
        "passed": False,
        "checks": [
            {"name": "dbe_metadata", "passed": False, "details": {"error": "reaction_smirks not provided"}}
        ],
    }
    try:
        coordinator._pause_for_retry_exhaustion(
            state,
            attempt=1,
            last_validation=validation,
        )
    except _RunPaused:
        pass

    exhausted = [ev for ev in store.events if ev["event_type"] == "mechanism_retry_exhausted"]
    assert exhausted
    assert exhausted[-1]["payload"]["validation"] == validation


def test_missing_reagents_surplus_only_emits_warning_without_pause() -> None:
    store = _EventStore()
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]
    state = _state()
    state.run_config.optional_llm_tools = ["predict_missing_reagents"]
    state.run_config.functional_groups_enabled = False

    coordinator._existing_steps = lambda _run_id: {"balance_analysis", "ph_recommendation", "initial_conditions"}  # type: ignore[method-assign]
    coordinator._record_step = lambda *_args, **_kwargs: None  # type: ignore[method-assign]

    class _DummyMissingReagentsAgent:
        def run(self, _state: RunState, _conditions_output: Optional[Dict[str, Any]]) -> StepResult:
            return StepResult(
                step_name="missing_reagents",
                tool_name="predict_missing_reagents",
                source="llm",
                output={
                    "status": "failed",
                    "should_abort_mechanism": False,
                    "abort_severity": "soft",
                    "message": "Surplus-only imbalance.",
                    "error": "partial_balance",
                    "balance_issues": {"remaining_deficit": {}, "remaining_surplus": {"B": 1}},
                },
            )

    coordinator.missing_reagents_agent = _DummyMissingReagentsAgent()  # type: ignore[assignment]
    coordinator._run_initial_phase(state)

    assert not state.paused
    assert any(ev["event_type"] == "missing_reagents_warning" for ev in store.events)
    assert not store.pauses


def test_missing_reagents_deficit_still_pauses_run() -> None:
    store = _EventStore()
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]
    state = _state()
    state.run_config.optional_llm_tools = ["predict_missing_reagents"]
    state.run_config.functional_groups_enabled = False

    coordinator._existing_steps = lambda _run_id: {"balance_analysis", "ph_recommendation", "initial_conditions"}  # type: ignore[method-assign]
    coordinator._record_step = lambda *_args, **_kwargs: None  # type: ignore[method-assign]

    class _DummyMissingReagentsAgent:
        def run(self, _state: RunState, _conditions_output: Optional[Dict[str, Any]]) -> StepResult:
            return StepResult(
                step_name="missing_reagents",
                tool_name="predict_missing_reagents",
                source="llm",
                output={
                    "status": "failed",
                    "should_abort_mechanism": True,
                    "abort_severity": "hard",
                    "message": "Deficit remains.",
                    "error": "partial_balance",
                    "balance_issues": {"remaining_deficit": {"O": 1}, "remaining_surplus": {}},
                },
            )

    coordinator.missing_reagents_agent = _DummyMissingReagentsAgent()  # type: ignore[assignment]
    try:
        coordinator._run_initial_phase(state)
    except _RunPaused:
        pass

    assert state.paused
    assert any(ev["event_type"] == "run_paused" for ev in store.events)
    assert store.pauses
