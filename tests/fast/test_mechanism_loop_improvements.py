from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mechanistic_agent.core.coordinator import RunCoordinator
from mechanistic_agent.core.types import (
    BranchCandidate,
    RunConfig,
    RunInput,
    RunState,
    StepResult,
    TemplateGuidanceState,
)
from mechanistic_agent.core.validators import validate_mechanism_step_output
from mechanistic_agent.tools import predict_mechanistic_step


class _EventStore:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.arrow_push_annotations: List[Dict[str, Any]] = []

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
        return "pause-test-id"

    def set_run_status(self, run_id: str, status: str) -> None:
        self.events.append(
            {
                "run_id": run_id,
                "event_type": "status_update",
                "payload": {"status": status},
                "step_name": None,
            }
        )

    def record_step_output(self, *args: Any, **kwargs: Any) -> None:
        return

    def add_trace_record(self, *args: Any, **kwargs: Any) -> None:
        return

    def resolve_run_step_prompt_id(self, *args: Any, **kwargs: Any) -> Optional[str]:
        return None

    def upsert_model_version(self, *args: Any, **kwargs: Any) -> str:
        return "model-version"

    def list_step_outputs(self, run_id: str) -> List[Dict[str, Any]]:
        return []

    def record_arrow_push_annotation(
        self,
        *,
        run_id: str,
        step_index: int,
        attempt: int,
        retry_index: int,
        candidate_rank: Optional[int],
        source: str,
        prediction: Dict[str, Any],
    ) -> str:
        self.arrow_push_annotations.append(
            {
                "run_id": run_id,
                "step_index": step_index,
                "attempt": attempt,
                "retry_index": retry_index,
                "candidate_rank": candidate_rank,
                "source": source,
                "prediction": prediction,
            }
        )
        return "arrow-annotation-id"



def _state(max_steps: int = 1, max_runtime_seconds: float = 0.02) -> RunState:
    run_input = RunInput(starting_materials=["CCBr", "[Cl-]"], products=["CCCl", "[Br-]"], ph=7.0, temperature_celsius=25.0)
    run_config = RunConfig(
        model="gpt-4",
        model_family="openai",
        max_steps=max_steps,
        max_runtime_seconds=max_runtime_seconds,
        intermediate_prediction_enabled=True,
        retry_same_candidate_max=1,
        reproposal_on_repeat_failure=True,
        step_mapping_enabled=True,
        dbe_policy="soft",
    )
    state = RunState(run_id="run-test-id", mode="unverified", run_input=run_input, run_config=run_config)
    state.initialise()
    return state


def test_predict_mechanistic_step_dedupes_resulting_state() -> None:
    raw = predict_mechanistic_step(
        step_index=1,
        current_state=["CCBr", "[Cl-]"],
        target_products=["CCCl", "[Br-]"],
        electron_pushes=[{"start_atom": "1", "end_atom": "2", "electrons": 2}],
        reaction_smirks="[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] |dbe:2-3:-2;2-4:+2;3-3:+2;4-4:-2|",
        predicted_intermediate="CCCl",
        resulting_state=["CCCl", "[Br-]", "CCCl"],
    )
    payload = json.loads(raw)
    assert payload["resulting_state"].count("CCCl") == 1


def test_soft_dbe_policy_allows_warning_only_dbe_failure() -> None:
    payload = {
        "current_state": ["CCBr", "[Cl-]"],
        "resulting_state": ["CCCl", "[Br-]"],
        "bond_electron_validation": {"valid": False, "error": "delta mismatch", "total_delta": 1},
        "unchanged_starting_materials_detected": False,
        "resulting_state_changed": True,
    }
    strict = validate_mechanism_step_output(payload, dbe_policy="strict")
    soft = validate_mechanism_step_output(payload, dbe_policy="soft")
    assert strict.passed is False
    assert soft.passed is True


def test_repeat_failure_requests_reproposal() -> None:
    store = _EventStore()
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]
    state = _state(max_steps=1, max_runtime_seconds=0.02)

    coordinator._record_step = lambda *_args, **_kwargs: None  # type: ignore[method-assign]

    class _DummyIntermediateAgent:
        def run(self, _state: RunState) -> StepResult:
            return StepResult(
                step_name="mechanism_step_proposal",
                tool_name="propose_mechanism_step",
                output={
                    "classification": "intermediate_step",
                    "analysis": "Single candidate",
                    "candidates": [
                        {
                            "rank": 1,
                            "intermediate_smiles": "CCCl",
                            "reaction_description": "SN2",
                            "reaction_smirks": "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] |dbe:2-3:-2;2-4:+2;3-3:+2;4-4:-2|",
                            "electron_pushes": [{"start_atom": "2", "end_atom": "4", "electrons": 2}],
                            "resulting_state": ["CCCl", "[Br-]"],
                        }
                    ],
                },
                source="llm",
            )

    coordinator.intermediate_agent = _DummyIntermediateAgent()  # type: ignore[assignment]

    def _fake_try(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        return {
            "status": "failed",
            "last_validation": {"passed": False, "checks": [{"name": "atom_balance", "passed": False, "details": {"error": "x"}}]},
            "force_reproposal": True,
            "failed_checks": ["atom_balance"],
            "validation_signature": "deadbeef",
            "candidate_rank": 1,
            "rescue_attempted": True,
            "rescue_outcome": "applied",
        }

    coordinator._try_candidate_with_retries = _fake_try  # type: ignore[method-assign]
    coordinator._run_mechanism_loop(state, threading.Event())

    assert any(ev["event_type"] == "mechanism_reproposal_requested" for ev in store.events)


def test_repeat_signature_not_forced_on_single_retry_budget() -> None:
    store = _EventStore()
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]
    state = _state(max_steps=1, max_runtime_seconds=0.02)
    state.run_config.retry_same_candidate_max = 1
    state.run_config.repeat_failure_signature_limit = 2
    state.run_config.candidate_rescue_enabled = False
    coordinator._record_step = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    coordinator._record_validation_checks = lambda *_args, **_kwargs: None  # type: ignore[method-assign]

    class _FailingMechanismAgent:
        def run(self, _state: RunState, _output: Dict[str, Any], *, retry_feedback: Optional[Dict[str, Any]] = None) -> StepResult:
            return StepResult(
                step_name="mechanism_synthesis",
                tool_name="predict_mechanistic_step",
                output={
                    "current_state": ["CCBr", "[Cl-]"],
                    "resulting_state": ["CCBr"],
                    "reaction_smirks": "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] |dbe:2-3:-2;2-4:+2;3-3:+2;4-4:-2|",
                    "electron_pushes": [{"start_atom": "2", "end_atom": "4", "electrons": 2}],
                },
                source="deterministic",
            )

    coordinator.mechanism_agent = _FailingMechanismAgent()  # type: ignore[assignment]
    candidate = {
        "rank": 1,
        "intermediate_smiles": "CCCl",
        "reaction_smirks": "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] |dbe:2-3:-2;2-4:+2;3-3:+2;4-4:-2|",
        "electron_pushes": [{"start_atom": "2", "end_atom": "4", "electrons": 2}],
        "resulting_state": ["CCBr"],
    }
    result = coordinator._try_candidate_with_retries(
        state,
        candidate,
        {"selected_candidate": candidate},
        enabled_validators={"atom_balance_validation"},
    )
    assert result["status"] == "failed"
    assert not bool(result.get("force_reproposal"))


def test_repeat_signature_forced_when_limit_reached() -> None:
    store = _EventStore()
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]
    state = _state(max_steps=1, max_runtime_seconds=0.02)
    state.run_config.retry_same_candidate_max = 2
    state.run_config.repeat_failure_signature_limit = 2
    state.run_config.candidate_rescue_enabled = False
    coordinator._record_step = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    coordinator._record_validation_checks = lambda *_args, **_kwargs: None  # type: ignore[method-assign]

    class _FailingMechanismAgent:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, _state: RunState, _output: Dict[str, Any], *, retry_feedback: Optional[Dict[str, Any]] = None) -> StepResult:
            self.calls += 1
            return StepResult(
                step_name="mechanism_synthesis",
                tool_name="predict_mechanistic_step",
                output={
                    "current_state": ["CCBr", "[Cl-]"],
                    "resulting_state": ["CCBr"],
                    "reaction_smirks": "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] |dbe:2-3:-2;2-4:+2;3-3:+2;4-4:-2|",
                    "electron_pushes": [{"start_atom": "2", "end_atom": "4", "electrons": 2}],
                },
                source="deterministic",
            )

    failing = _FailingMechanismAgent()
    coordinator.mechanism_agent = failing  # type: ignore[assignment]
    candidate = {
        "rank": 1,
        "intermediate_smiles": "CCCl",
        "reaction_smirks": "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] |dbe:2-3:-2;2-4:+2;3-3:+2;4-4:-2|",
        "electron_pushes": [{"start_atom": "2", "end_atom": "4", "electrons": 2}],
        "resulting_state": ["CCBr"],
    }
    result = coordinator._try_candidate_with_retries(
        state,
        candidate,
        {"selected_candidate": candidate},
        enabled_validators={"atom_balance_validation"},
    )
    assert failing.calls == 2
    assert result["status"] == "failed"
    assert bool(result.get("force_reproposal")) is True
    assert result.get("repeat_failure_signature_limit") == 2


def test_rescue_validation_records_superseding_validation_checks() -> None:
    store = _EventStore()
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]
    state = _state(max_steps=1, max_runtime_seconds=0.05)
    state.run_config.retry_same_candidate_max = 1
    state.run_config.candidate_rescue_enabled = True

    recorded_calls: List[Dict[str, Any]] = []

    def _capture_validation_calls(*_args: Any, **kwargs: Any) -> None:
        recorded_calls.append(dict(kwargs))

    coordinator._record_validation_checks = _capture_validation_calls  # type: ignore[method-assign]

    class _FailingMechanismAgent:
        def run(self, _state: RunState, _output: Dict[str, Any], *, retry_feedback: Optional[Dict[str, Any]] = None) -> StepResult:
            return StepResult(
                step_name="mechanism_synthesis",
                tool_name="predict_mechanistic_step",
                output={
                    "current_state": ["CCBr", "[Cl-]"],
                    "resulting_state": ["CCCl", "[Br-]", "O"],
                    "reaction_smirks": "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] |dbe:2-3:-2;2-4:+2;3-3:+2;4-4:-2|",
                    "electron_pushes": [{"start_atom": "2", "end_atom": "4", "electrons": 2}],
                },
                source="deterministic",
            )

    class _RescueAgent:
        def rescue_candidate(
            self,
            _state: RunState,
            *,
            current_state: List[str],
            resulting_state: List[str],
            failed_checks: Optional[List[str]] = None,
            validation_details: Optional[Dict[str, Any]] = None,
        ) -> StepResult:
            return StepResult(
                step_name="candidate_rescue",
                tool_name="predict_missing_reagents_for_candidate",
                output={"add_reactants": ["O"], "add_products": [], "status": "success"},
                source="llm",
            )

    coordinator.mechanism_agent = _FailingMechanismAgent()  # type: ignore[assignment]
    coordinator.missing_reagents_agent = _RescueAgent()  # type: ignore[assignment]

    candidate = {
        "rank": 1,
        "intermediate_smiles": "CCCl",
        "reaction_smirks": "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] |dbe:2-3:-2;2-4:+2;3-3:+2;4-4:-2|",
        "electron_pushes": [{"start_atom": "2", "end_atom": "4", "electrons": 2}],
        "resulting_state": ["CCCl", "[Br-]", "O"],
    }

    result = coordinator._try_candidate_with_retries(
        state,
        candidate,
        {"selected_candidate": candidate},
        enabled_validators={"atom_balance_validation"},
    )

    assert result["status"] == "validated"
    assert len(recorded_calls) == 2
    assert "mechanism_result" in recorded_calls[0]
    assert "validation_result" in recorded_calls[1]
    assert recorded_calls[1]["attempt"] == 1
    assert recorded_calls[1]["retry_index"] == 0


def test_rescue_validated_candidate_records_successful_mechanism_synthesis_row() -> None:
    store = _EventStore()
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]
    state = _state(max_steps=1, max_runtime_seconds=0.05)
    state.run_config.retry_same_candidate_max = 1
    state.run_config.candidate_rescue_enabled = True

    recorded_rows: List[StepResult] = []
    coordinator._record_step = lambda _state, result: recorded_rows.append(result)  # type: ignore[method-assign]
    coordinator._record_validation_checks = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    coordinator._record_arrow_push_annotation = lambda *_args, **_kwargs: None  # type: ignore[method-assign]

    class _FailingMechanismAgent:
        def run(self, _state: RunState, _output: Dict[str, Any], *, retry_feedback: Optional[Dict[str, Any]] = None) -> StepResult:
            return StepResult(
                step_name="mechanism_synthesis",
                tool_name="predict_mechanistic_step",
                output={
                    "current_state": ["CCBr", "[Cl-]"],
                    "resulting_state": ["CCCl", "[Br-]", "O"],
                    "reaction_smirks": "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] |dbe:2-3:-2;2-4:+2;3-3:+2;4-4:-2|",
                    "electron_pushes": [{"start_atom": "2", "end_atom": "4", "electrons": 2}],
                },
                source="llm",
            )

    class _RescueAgent:
        def rescue_candidate(
            self,
            _state: RunState,
            *,
            current_state: List[str],
            resulting_state: List[str],
            failed_checks: Optional[List[str]] = None,
            validation_details: Optional[Dict[str, Any]] = None,
        ) -> StepResult:
            return StepResult(
                step_name="candidate_rescue",
                tool_name="predict_missing_reagents_for_candidate",
                output={"add_reactants": ["O"], "add_products": [], "status": "success"},
                source="llm",
            )

    coordinator.mechanism_agent = _FailingMechanismAgent()  # type: ignore[assignment]
    coordinator.missing_reagents_agent = _RescueAgent()  # type: ignore[assignment]

    candidate = {
        "rank": 1,
        "intermediate_smiles": "CCCl",
        "reaction_smirks": "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] |dbe:2-3:-2;2-4:+2;3-3:+2;4-4:-2|",
        "electron_pushes": [{"start_atom": "2", "end_atom": "4", "electrons": 2}],
        "resulting_state": ["CCCl", "[Br-]", "O"],
    }
    result = coordinator._try_candidate_with_retries(
        state,
        candidate,
        {"selected_candidate": candidate},
        enabled_validators={"atom_balance_validation"},
    )

    assert result["status"] == "validated"
    validated_rows = [
        row
        for row in recorded_rows
        if row.step_name == "mechanism_synthesis"
        and row.validation is not None
        and row.validation.passed
    ]
    assert validated_rows


def test_step_mapping_event_emitted_after_nonfinal_step() -> None:
    store = _EventStore()
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]
    state = _state(max_steps=1, max_runtime_seconds=0.05)

    class _DummyIntermediateAgent:
        def run(self, _state: RunState) -> StepResult:
            return StepResult(
                step_name="mechanism_step_proposal",
                tool_name="propose_mechanism_step",
                output={
                    "classification": "intermediate_step",
                    "analysis": "Single candidate",
                    "candidates": [
                        {
                            "rank": 1,
                            "intermediate_smiles": "CCCl",
                            "reaction_description": "SN2",
                            "reaction_smirks": "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] |dbe:2-3:-2;2-4:+2;3-3:+2;4-4:-2|",
                            "electron_pushes": [{"start_atom": "2", "end_atom": "4", "electrons": 2}],
                            "resulting_state": ["CCCl", "[Br-]"],
                        }
                    ],
                },
                source="llm",
            )

    @dataclass
    class _DummyMappingAgent:
        def run_step_mapping(self, _state: RunState, *, current_state: List[str], resulting_state: List[str]) -> StepResult:
            return StepResult(
                step_name="step_atom_mapping",
                tool_name="attempt_atom_mapping_for_step",
                output={"compact_mapped_atoms": [{"product_atom": "C#0"}], "unmapped_atoms": [], "confidence": "medium"},
                source="llm",
            )

    class _DummyReflection:
        def run(self, _state: RunState, _latest_output: Dict[str, Any]) -> StepResult:
            return StepResult(step_name="reflection", tool_name="reflection_agent", output={"warnings": []}, source="deterministic")

    coordinator.intermediate_agent = _DummyIntermediateAgent()  # type: ignore[assignment]
    coordinator.mapping_agent = _DummyMappingAgent()  # type: ignore[assignment]
    coordinator.reflection_agent = _DummyReflection()  # type: ignore[assignment]

    def _ok_try(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        output = {
            "contains_target_product": False,
            "current_state": ["CCBr", "[Cl-]"],
            "resulting_state": ["CCCl", "[Br-]"],
            "predicted_intermediate": "CCCl",
        }
        return {
            "status": "validated",
            "branch_candidate": BranchCandidate(
                rank=1,
                intermediate_smiles="CCCl",
                intermediate_output={"rank": 1, "intermediate_smiles": "CCCl"},
                mechanism_output=output,
                resulting_state=output["resulting_state"],
            ),
            "last_validation": {"passed": True, "checks": []},
            "reason": "",
        }

    coordinator._try_candidate_with_retries = _ok_try  # type: ignore[method-assign]
    coordinator._run_mechanism_loop(state, threading.Event())

    assert any(ev["event_type"] == "step_mapping_generated" for ev in store.events)


def test_incomplete_reproposal_budget_exhaustion_fails_run() -> None:
    store = _EventStore()
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]
    state = _state(max_steps=1, max_runtime_seconds=5.0)
    state.run_config.max_reproposals_per_step = 2
    state.template_guidance_state = TemplateGuidanceState(mode="active")

    coordinator._record_step = lambda *_args, **_kwargs: None  # type: ignore[method-assign]

    class _DummyIntermediateAgent:
        def run(self, _state: RunState) -> StepResult:
            return StepResult(
                step_name="mechanism_step_proposal",
                tool_name="propose_mechanism_step",
                output={
                    "classification": "intermediate_step",
                    "analysis": "Always incomplete",
                    "candidates": [
                        {
                            "rank": 1,
                            "intermediate_smiles": "CCCl",
                            "reaction_description": "Missing execution fields",
                        }
                    ],
                },
                source="llm",
            )

    coordinator.intermediate_agent = _DummyIntermediateAgent()  # type: ignore[assignment]
    coordinator._run_mechanism_loop(state, threading.Event())

    assert any(ev["event_type"] == "proposal_quality_summary" for ev in store.events)
    assert any(ev["event_type"] == "template_guidance_preaccept_observation" for ev in store.events)
    assert any(ev["event_type"] == "mechanism_reproposal_limit_reached" for ev in store.events)
    failed_events = [ev for ev in store.events if ev["event_type"] == "run_failed"]
    assert failed_events
    assert failed_events[-1]["payload"]["reason"] == "proposal_incomplete_loop"


def test_arrow_push_annotation_recorded_for_validated_candidate() -> None:
    store = _EventStore()
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]
    state = _state(max_steps=1, max_runtime_seconds=0.2)
    state.run_config.arrow_push_annotation_enabled = True

    class _DummyMechanismAgent:
        def run(
            self,
            _state: RunState,
            _intermediate_output: Optional[Dict[str, Any]],
            *,
            retry_feedback: Optional[Dict[str, Any]] = None,
        ) -> StepResult:
            del retry_feedback
            raw = predict_mechanistic_step(
                step_index=1,
                current_state=["CCBr", "[Cl-]"],
                target_products=["CCCl", "[Br-]"],
                electron_pushes=[{"start_atom": "2", "end_atom": "4", "electrons": 2}],
                reaction_smirks="[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] |dbe:2-3:-2;2-4:+2;3-3:+2;4-4:-2|",
                predicted_intermediate="CCCl",
                resulting_state=["CCCl", "[Br-]"],
                previous_intermediates=[],
                starting_materials=["CCBr", "[Cl-]"],
            )
            return StepResult(
                step_name="mechanism_synthesis",
                tool_name="predict_mechanistic_step",
                output=json.loads(raw),
                source="llm",
            )

    coordinator.mechanism_agent = _DummyMechanismAgent()  # type: ignore[assignment]
    candidate = {
        "rank": 1,
        "intermediate_smiles": "CCCl",
        "reaction_smirks": "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] |dbe:2-3:-2;2-4:+2;3-3:+2;4-4:-2|",
        "electron_pushes": [{"start_atom": "2", "end_atom": "4", "electrons": 2}],
        "resulting_state": ["CCCl", "[Br-]"],
    }

    result = coordinator._try_candidate_with_retries(state, candidate, {})
    assert result["status"] == "validated"
    assert len(store.arrow_push_annotations) == 1
    prediction = store.arrow_push_annotations[0]["prediction"]
    assert prediction["annotation_suffix"].startswith("|aps:v1;")
    assert not any("aps:v1;" in json.dumps(ev.get("payload", {})) for ev in store.events)


def test_arrow_push_annotation_not_recorded_for_failed_candidate() -> None:
    store = _EventStore()
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]
    state = _state(max_steps=1, max_runtime_seconds=0.2)
    state.run_config.arrow_push_annotation_enabled = True

    class _FailingMechanismAgent:
        def run(
            self,
            _state: RunState,
            _intermediate_output: Optional[Dict[str, Any]],
            *,
            retry_feedback: Optional[Dict[str, Any]] = None,
        ) -> StepResult:
            del retry_feedback
            return StepResult(
                step_name="mechanism_synthesis",
                tool_name="predict_mechanistic_step",
                output={
                    "step_index": 1,
                    "current_state": ["CCBr", "[Cl-]"],
                    "resulting_state": ["CCBr", "[Cl-]"],
                    "electron_pushes": [{"start_atom": "2", "end_atom": "4", "electrons": 2}],
                    "reaction_smirks": "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3]",
                    "bond_electron_validation": {"valid": True},
                    "unchanged_starting_materials_detected": True,
                    "resulting_state_changed": False,
                    "contains_target_product": False,
                },
                source="llm",
            )

    coordinator.mechanism_agent = _FailingMechanismAgent()  # type: ignore[assignment]
    candidate = {
        "rank": 1,
        "intermediate_smiles": "CCCl",
        "reaction_smirks": "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] |dbe:2-3:-2;2-4:+2;3-3:+2;4-4:-2|",
        "electron_pushes": [{"start_atom": "2", "end_atom": "4", "electrons": 2}],
        "resulting_state": ["CCBr", "[Cl-]"],
    }

    result = coordinator._try_candidate_with_retries(state, candidate, {})
    assert result["status"] == "failed"
    assert store.arrow_push_annotations == []


def test_template_guidance_disables_after_consecutive_early_mismatches() -> None:
    store = _EventStore()
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]
    state = _state(max_steps=5, max_runtime_seconds=0.2)
    state.template_guidance_state = TemplateGuidanceState(
        mode="active",
        selected_type_id="mt_003",
        selected_label_exact="SN2 reaction",
        selection_confidence=0.9,
        suitable_step_count=2,
        current_template_step_index=1,
    )

    first = BranchCandidate(
        rank=1,
        intermediate_smiles="CCCl",
        intermediate_output={"template_alignment": "not_aligned"},
        mechanism_output={"contains_target_product": False},
        resulting_state=["CCCl", "[Br-]"],
    )
    second = BranchCandidate(
        rank=1,
        intermediate_smiles="CCCCl",
        intermediate_output={"template_alignment": "not_aligned"},
        mechanism_output={"contains_target_product": False},
        resulting_state=["CCCCl", "[Br-]"],
    )

    coordinator._apply_candidate(state, first)
    assert state.template_guidance_state is not None
    assert state.template_guidance_state.mode == "active"

    coordinator._apply_candidate(state, second)
    assert state.template_guidance_state is not None
    assert state.template_guidance_state.mode == "disabled"
    assert state.template_guidance_state.disable_reason == "early_consecutive_template_mismatch"


def test_loop_requests_reproposal_when_all_candidates_rejected() -> None:
    store = _EventStore()
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]
    state = _state(max_steps=1, max_runtime_seconds=0.02)

    coordinator._record_step = lambda *_args, **_kwargs: None  # type: ignore[method-assign]

    class _DummyIntermediateAgent:
        def run(self, _state: RunState) -> StepResult:
            return StepResult(
                step_name="mechanism_step_proposal",
                tool_name="propose_mechanism_step",
                output={
                    "classification": "intermediate_step",
                    "analysis": "All candidates rejected after payload checks.",
                    "candidates": [],
                    "rejected_candidates": [
                        {
                            "rank": 1,
                            "intermediate_smiles": "C1OCOC1",
                            "reason": "reaction_smirks_invalid_dbe_block",
                        }
                    ],
                },
                source="llm",
            )

    coordinator.intermediate_agent = _DummyIntermediateAgent()  # type: ignore[assignment]
    coordinator._run_mechanism_loop(state, threading.Event())

    requested = [ev for ev in store.events if ev["event_type"] == "mechanism_reproposal_requested"]
    assert requested
    assert requested[-1]["payload"]["reason"] == "all_candidates_rejected"


def test_reaction_type_small_confidence_gap_uses_weak_guidance() -> None:
    store = _EventStore()
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]
    state = _state(max_steps=1, max_runtime_seconds=0.02)
    state.run_config.reaction_template_confidence_threshold = 0.65
    state.run_config.reaction_template_margin_threshold = 0.10

    coordinator._apply_reaction_type_selection(
        state,
        {
            "selected_label_exact": "SN2 reaction",
            "selected_type_id": "mt_001",
            "confidence": 0.80,
            "rationale": "Top candidate is plausible.",
            "top_candidates": [
                {"label_exact": "SN2 reaction", "type_id": "mt_001", "confidence": 0.80},
                {"label_exact": "SN1 reaction", "type_id": "mt_003", "confidence": 0.75},
            ],
            "selected_template": {
                "type_id": "mt_001",
                "label_exact": "SN2 reaction",
                "suitable_step_count": 2,
                "generic_mechanism_steps": [{"step_index": 1, "reaction_generic": "R-Br>>R-Cl", "note": ""}],
            },
        },
        emit_event=False,
    )

    assert state.template_guidance_state is not None
    assert state.template_guidance_state.mode == "weak"
