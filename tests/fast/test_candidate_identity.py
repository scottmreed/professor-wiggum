"""Stable candidate identity and acceptance kinds (Observatory PRD §13.4, §16.1, §16.6, §3.7.4).

Ranks repeat across reproposal rounds, retries and topology rounds, so every
proposed candidate gets a ``candidate_id`` that follows it through the
proposal event, validation events, branch points, acceptance and backtracking.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

import pytest

pytest.importorskip("rdkit")

from mechanistic_agent.core.coordinator import RunCoordinator
from mechanistic_agent.core.provenance import assign_candidate_ids
from mechanistic_agent.core.types import BranchCandidate, RunConfig, RunInput, RunState, StepResult


class _EventStore:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def append_event(self, run_id: str, event_type: str, payload: Dict[str, Any], *, step_name: Optional[str] = None) -> None:
        self.events.append({"run_id": run_id, "event_type": event_type, "payload": payload, "step_name": step_name})

    def create_run_pause(self, *, run_id: str, reason: str, details: Dict[str, Any]) -> str:
        return "pause-test-id"

    def set_run_status(self, run_id: str, status: str) -> None:
        return

    def record_step_output(self, *args: Any, **kwargs: Any) -> None:
        return

    def list_step_outputs(self, run_id: str) -> List[Dict[str, Any]]:
        return []

    def of(self, kind: str) -> List[Dict[str, Any]]:
        return [e["payload"] for e in self.events if e["event_type"] == kind]


def _state(max_steps: int = 1, max_runtime_seconds: float = 0.05) -> RunState:
    run_input = RunInput(starting_materials=["CCBr", "[Cl-]"], products=["CCCl", "[Br-]"])
    run_config = RunConfig(model="gpt-4", model_family="openai", max_steps=max_steps,
                           max_runtime_seconds=max_runtime_seconds, intermediate_prediction_enabled=True)
    state = RunState(run_id="run-test-id", mode="unverified", run_input=run_input, run_config=run_config)
    state.initialise()
    return state


def _candidate(rank: int, smiles: str = "CCCl", **extra: Any) -> BranchCandidate:
    return BranchCandidate(
        rank=rank,
        intermediate_smiles=smiles,
        intermediate_output={"rank": rank, "intermediate_smiles": smiles, **extra},
        mechanism_output={"contains_target_product": False},
        resulting_state=[smiles, "[Br-]"],
        validation_summary={"passed": True, "checks": []},
    )


# ---------------------------------------------------------------------------
# assign_candidate_ids
# ---------------------------------------------------------------------------

def test_same_ranks_in_two_rounds_get_distinct_ids() -> None:
    round_1 = assign_candidate_ids([{"rank": 1, "intermediate_smiles": "A"}, {"rank": 2, "intermediate_smiles": "B"}], step_number=3)
    round_2 = assign_candidate_ids([{"rank": 1, "intermediate_smiles": "C"}, {"rank": 2, "intermediate_smiles": "D"}], step_number=3)
    ids = [c["candidate_id"] for c in round_1 + round_2]
    assert len(set(ids)) == 4
    assert all(cid.startswith("c3-r") for cid in ids)


def test_assign_is_idempotent_and_in_place() -> None:
    candidates = [{"rank": 1, "intermediate_smiles": "A", "candidate_id": "c1-r1-keepme"}, {"rank": 2}]
    out = assign_candidate_ids(candidates, step_number=1)
    assert out is candidates
    assert candidates[0]["candidate_id"] == "c1-r1-keepme"
    assert candidates[1]["candidate_id"].startswith("c1-r2-")


# ---------------------------------------------------------------------------
# BranchCandidate carries the id through persistence
# ---------------------------------------------------------------------------

def test_branch_candidate_exposes_candidate_id_from_proposal_dict() -> None:
    cand = _candidate(1, candidate_id="c1-r1-abcdef12")
    assert cand.candidate_id == "c1-r1-abcdef12"
    assert _candidate(1).candidate_id is None


def test_candidate_id_survives_persisted_round_trip() -> None:
    cand = _candidate(2, candidate_id="c4-r2-deadbeef")
    restored = BranchCandidate.from_persisted_dict(cand.to_persisted_dict())
    assert restored.candidate_id == "c4-r2-deadbeef"


# ---------------------------------------------------------------------------
# Loop: proposal event + candidate events share the id
# ---------------------------------------------------------------------------

def test_loop_emits_candidates_proposed_and_propagates_id_to_candidate_events() -> None:
    store = _EventStore()
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]
    state = _state()
    coordinator._record_step = lambda *_a, **_k: None  # type: ignore[method-assign]

    class _Proposer:
        def run(self, _state: RunState, **_kw: Any) -> StepResult:
            return StepResult(
                step_name="mechanism_step_proposal",
                tool_name="propose_mechanism_step",
                output={"candidates": [{"rank": 1, "intermediate_smiles": "CCCl", "reaction_description": "SN2"}]},
                source="llm",
            )

    coordinator.intermediate_agent = _Proposer()  # type: ignore[assignment]
    coordinator._run_mechanism_loop(state, threading.Event())

    proposed = store.of("mechanism_candidates_proposed")
    assert proposed, "mechanism_candidates_proposed not emitted"
    first = proposed[0]
    assert first["step_index"] == 1
    assert first["candidate_set_id"]
    ids = [c["candidate_id"] for c in first["candidates"]]
    assert ids and all(ids)
    assert first["candidates"][0]["rank"] == 1
    assert first["candidates"][0]["intermediate_smiles"] == "CCCl"

    incomplete = store.of("mechanism_candidate_incomplete")
    assert incomplete and incomplete[0]["candidate_id"] == ids[0]


# ---------------------------------------------------------------------------
# Acceptance kinds and branch/backtrack correlation
# ---------------------------------------------------------------------------

def test_apply_candidate_records_validated_acceptance_with_candidate_id() -> None:
    store = _EventStore()
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]
    state = _state()

    coordinator._apply_candidate(state, _candidate(1, candidate_id="c1-r1-aaaaaaaa"))

    accepted = store.of("mechanism_step_accepted")
    assert len(accepted) == 1
    assert accepted[0]["candidate_id"] == "c1-r1-aaaaaaaa"
    assert accepted[0]["acceptance_kind"] == "validated"


def test_soft_advanced_candidate_is_not_labelled_validated() -> None:
    store = _EventStore()
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]
    state = _state()
    soft = _candidate(99, candidate_id="c1-r99-bbbbbbbb")
    soft.validation_summary = {"passed": False, "soft_advance": True, "checks": []}
    soft.mechanism_output = {**soft.mechanism_output, "soft_advance": True}

    coordinator._apply_candidate(state, soft)

    accepted = store.of("mechanism_step_accepted")[0]
    assert accepted["acceptance_kind"] == "soft_advance"
    assert accepted["validation_summary"]["passed"] is False


def test_backtrack_alternative_acceptance_kind_and_branch_ids() -> None:
    store = _EventStore()
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]
    state = _state(max_steps=3)
    chosen = _candidate(1, "CCCl", candidate_id="c1-r1-chosen00")
    alt = _candidate(2, "CCI", candidate_id="c1-r2-altern00")

    coordinator._record_branch_point(state, chosen, [alt])
    coordinator._apply_candidate(state, chosen)
    state.step_index += 1
    assert coordinator._backtrack(state) is True

    bp = store.of("branch_point_created")[0]
    assert bp["chosen_candidate_id"] == "c1-r1-chosen00"
    assert bp["alternative_candidate_ids"] == ["c1-r2-altern00"]

    failed = store.of("failed_path_recorded")[0]
    assert failed["candidate_id"] == "c1-r1-chosen00"

    back = store.of("backtrack")[0]
    assert back["candidate_id"] == "c1-r2-altern00"

    kinds = [(a["candidate_id"], a["acceptance_kind"]) for a in store.of("mechanism_step_accepted")]
    assert kinds == [("c1-r1-chosen00", "validated"), ("c1-r2-altern00", "backtrack_alternative")]
