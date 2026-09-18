"""Candidate rescue is cached per failure signature and skipped for alternates."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from mechanistic_agent.core.coordinator import RunCoordinator
from mechanistic_agent.core.types import RunConfig, RunInput, RunState, StepResult


class _EventStore:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.steps: List[StepResult] = []

    def append_event(self, run_id, event_type, payload, step_name=None):  # noqa: ANN001
        self.events.append({"run_id": run_id, "event_type": event_type, "payload": payload, "step_name": step_name})

    def record_step_output(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None

    def add_step_output(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None


class _CountingRescueAgent:
    def __init__(self) -> None:
        self.calls = 0

    def rescue_candidate(
        self,
        _state: RunState,
        *,
        current_state: List[str],
        resulting_state: List[str],
        failed_checks: Optional[List[str]] = None,
        validation_details: Optional[Dict[str, Any]] = None,
    ) -> StepResult:
        self.calls += 1
        return StepResult(
            step_name="candidate_rescue",
            tool_name="predict_missing_reagents_for_candidate",
            output={"add_reactants": [], "add_products": [], "status": "success"},
            source="llm",
        )


def _state() -> RunState:
    state = RunState(
        run_id="run-rescue",
        mode="unverified",
        run_input=RunInput(starting_materials=["O=C(O)C=Cc1cncc(Br)c1", "O=S(Cl)Cl"], products=["O=C(Cl)C=Cc1cncc(Br)c1", "Cl", "O=S=O"]),
        run_config=RunConfig(model="gpt-5"),
    )
    state.initialise()
    return state


def _mechanism_result(resulting: List[str]) -> StepResult:
    return StepResult(
        step_name="mechanism_synthesis",
        tool_name="predict_mechanistic_step",
        output={"current_state": ["O=C(O)C=Cc1cncc(Br)c1", "O=S(Cl)Cl"], "resulting_state": resulting},
        source="deterministic",
    )


def test_rescue_is_cached_by_failure_signature() -> None:
    store = _EventStore()
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]
    coordinator._record_step = lambda *_a, **_k: None  # type: ignore[method-assign]
    agent = _CountingRescueAgent()
    coordinator.missing_reagents_agent = agent  # type: ignore[assignment]
    state = _state()

    first = coordinator._attempt_candidate_rescue(
        state, mechanism_result=_mechanism_result(["X", "O=S(Cl)Cl"]), failed_checks=["atom_balance"], candidate_rank=1
    )
    second = coordinator._attempt_candidate_rescue(
        state, mechanism_result=_mechanism_result(["O=S(Cl)Cl", "X"]), failed_checks=["atom_balance"], candidate_rank=2
    )
    assert first is not None and second is first
    assert agent.calls == 1
    assert any(ev["event_type"] == "candidate_rescue_cache_hit" for ev in store.events)

    # A different species delta is a different signature -> new LLM call.
    coordinator._attempt_candidate_rescue(
        state, mechanism_result=_mechanism_result(["Y", "O=S(Cl)Cl"]), failed_checks=["atom_balance"], candidate_rank=3
    )
    assert agent.calls == 2

    # Cache is per run.
    other = _state()
    other.run_id = "run-other"
    coordinator._attempt_candidate_rescue(
        other, mechanism_result=_mechanism_result(["X", "O=S(Cl)Cl"]), failed_checks=["atom_balance"], candidate_rank=1
    )
    assert agent.calls == 3


def test_alternate_candidates_skip_rescue_once_a_candidate_validated() -> None:
    store = _EventStore()
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]
    coordinator._record_step = lambda *_a, **_k: None  # type: ignore[method-assign]
    coordinator._record_validation_checks = lambda *_a, **_k: None  # type: ignore[method-assign]
    coordinator._record_arrow_push_annotation = lambda *_a, **_k: None  # type: ignore[method-assign]
    coordinator._prevalidate_candidate_against_constraints = lambda _s, c: (dict(c), None)  # type: ignore[method-assign]
    agent = _CountingRescueAgent()
    coordinator.missing_reagents_agent = agent  # type: ignore[assignment]

    class _FailingMechanismAgent:
        def run(self, _state, _output, *, retry_feedback=None):  # noqa: ANN001
            return StepResult(
                step_name="mechanism_synthesis",
                tool_name="predict_mechanistic_step",
                output={
                    "current_state": ["CCBr", "[Cl-]"],
                    "resulting_state": ["CCCl", "[Br-]", "O"],
                    "reaction_smirks": "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] |mech:v1;lp:4>2;sigma:2-3>3|",
                    "electron_pushes": [{"kind": "lone_pair", "source_atom": "4", "target_atom": "2", "electrons": 2}],
                },
                source="deterministic",
            )

    coordinator.mechanism_agent = _FailingMechanismAgent()  # type: ignore[assignment]
    state = RunState(
        run_id="run-alt",
        mode="unverified",
        run_input=RunInput(starting_materials=["CCBr", "[Cl-]"], products=["CCCl", "[Br-]"]),
        run_config=RunConfig(model="gpt-5"),
    )
    state.initialise()
    candidate = {
        "rank": 2,
        "intermediate_smiles": "CCCl",
        "resulting_state": ["CCCl", "[Br-]", "O"],
        "reaction_smirks": "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] |mech:v1;lp:4>2;sigma:2-3>3|",
        "electron_pushes": [{"kind": "lone_pair", "source_atom": "4", "target_atom": "2", "electrons": 2}],
    }

    result = coordinator._try_candidate_with_retries(state, candidate, {"candidates": [candidate]}, allow_rescue=False)
    assert result["status"] == "failed"
    assert result["rescue_outcome"] == "skipped_alternate"
    assert agent.calls == 0
    assert any(ev["event_type"] == "candidate_rescue_skipped_alternate" for ev in store.events)

    # With rescue allowed the same failure does spend a rescue call.
    result2 = coordinator._try_candidate_with_retries(state, candidate, {"candidates": [candidate]}, allow_rescue=True)
    assert result2["status"] == "failed"
    assert agent.calls == 1
