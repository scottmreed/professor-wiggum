"""Retry redesign: spectator carry, visible rescue re-validation, reproposal before pausing.

Fixtures come from the 2026-10 threaded_unified 92-case run (Opus 5.5, keyless
bridge), where 9 runs paused with ``mechanism_retry_exhausted`` after one
proposal: the rescue was the only second chance and a fresh proposal was never
requested.
"""
from __future__ import annotations

import json
import threading
from typing import Any, Dict, List, Optional

import pytest

from mechanistic_agent.core.coordinator import RunCoordinator, _RunPaused
from mechanistic_agent.core.spectators import analyse_step_species, carry_spectators, rejection_note
from mechanistic_agent.core.types import BranchCandidate, RunConfig, RunInput, RunState, StepResult
from mechanistic_agent.tools import propose_intermediates

# mcsa-112-1, step 1, proposal round 2: an iminium protonation by H3O+ on a
# 13-species state. The model wrote each spectator once, dropping two of three
# isopentanes, one of two alaninamides and the water from H3O+.
MCSA_112_CURRENT = [
    "CCC(=O)[O-]", "CCC(C)C", "CCc1ccccc1", "CCC(C)C", "CC(C)O", "CCC(C)C", "CCc1c[nH]c2ccccc12",
    "CC(N)C(N)=O", "CC(N)C(N)=O", "O", "[OH3+]",
    "NC(=O)C1=CN(C2OC(COP(=O)(O)OP(=O)(O)OCC3OC(n4cnc5c(N)ncnc54)C(OP(=O)(O)O)C3O)C(O)C2O)C=CC1",
    "Nc1nc2c(c(=O)[nH]1)N=C(CNc1ccc(C(=O)NC(CCC(=O)O)C(=O)O)cc1)CN2",
]
MCSA_112_SMIRKS = (
    "Nc1nc2c(c(=O)[nH]1)[N:1]=[C:2](CNc1ccc(C(=O)NC(CCC(=O)O)C(=O)O)cc1)CN2.[OH3+:3]>>"
    "Nc1nc2c(c(=O)[nH]1)[NH+:1]=[C:2](CNc1ccc(C(=O)NC(CCC(=O)O)C(=O)O)cc1)CN2.[OH2:3] |mech:v1;lp:1>3|"
)
MCSA_112_RESULTING = [
    "CCC(=O)[O-]", "CCC(C)C", "CCc1ccccc1", "CC(C)O", "CCc1c[nH]c2ccccc12", "CC(N)C(N)=O", "O",
    "NC(=O)C1=CN(C2OC(COP(=O)(O)OP(=O)(O)OCC3OC(n4cnc5c(N)ncnc54)C(OP(=O)(O)O)C3O)C(O)C2O)C=CC1",
    "Nc1nc2c(c(=O)[nH]1)[NH+]=C(CNc1ccc(C(=O)NC(CCC(=O)O)C(=O)O)cc1)CN2",
]

SN2_SMIRKS = "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] |mech:v1;lp:4>2;sigma:2-3>3|"
LP_PUSH = [{"kind": "lone_pair", "source_atom": "1", "target_atom": "3", "electrons": 2}]
ACYL_SMIRKS = (
    "[CH3:1][C:2](=[O:3])[Cl:4].[CH3:5][CH2:6][OH:7]>>[CH3:1][C:2](=[O:3])[O:7][CH2:6][CH3:5].[Cl:4] "
    "|mech:v1;lp:7>2;sigma:2-4>4|"
)


class _EventStore:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def append_event(self, run_id: str, event_type: str, payload: Dict[str, Any], *, step_name: Optional[str] = None) -> None:
        self.events.append({"run_id": run_id, "event_type": event_type, "payload": payload, "step_name": step_name})

    def create_run_pause(self, *, run_id: str, reason: str, details: Dict[str, Any]) -> str:
        return "pause-id"

    def set_run_status(self, run_id: str, status: str) -> None:
        self.events.append({"run_id": run_id, "event_type": "status_update", "payload": {"status": status}, "step_name": None})

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

    def types(self, event_type: str) -> List[Dict[str, Any]]:
        return [ev["payload"] for ev in self.events if ev["event_type"] == event_type]


def _state(*, mode: str = "unverified", max_reproposals: int = 2) -> RunState:
    state = RunState(
        run_id="run-test",
        mode=mode,
        run_input=RunInput(starting_materials=["CCBr", "[Cl-]"], products=["CCCl", "[Br-]"]),
        run_config=RunConfig(
            model="gpt-5",
            max_steps=3,
            max_runtime_seconds=600.0,
            retry_same_candidate_max=1,
            max_reproposals_per_step=max_reproposals,
            dbe_policy="soft",
            chemistry_backend="python",
        ),
    )
    state.initialise()
    return state


@pytest.fixture(autouse=True)
def _python_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MECHANISTIC_CHEMISTRY_BACKEND", "python")


# ── Spectator analysis and carry ──────────────────────────────────────────


def test_dropped_duplicate_spectators_are_identified_and_carried() -> None:
    analysis = analyse_step_species(MCSA_112_CURRENT, MCSA_112_RESULTING, MCSA_112_SMIRKS)
    assert analysis["reactants_in_current_state"] and analysis["products_in_resulting_state"]
    assert sorted(analysis["dropped_spectators"]) == ["CC(N)C(N)=O", "CCC(C)C", "CCC(C)C", "O"]
    repair = carry_spectators(MCSA_112_CURRENT, MCSA_112_RESULTING, MCSA_112_SMIRKS)
    assert repair is not None
    assert sorted(repair["carried"]) == ["CC(N)C(N)=O", "CCC(C)C", "CCC(C)C", "O"]
    assert len(repair["resulting_state"]) == len(MCSA_112_RESULTING) + 4


def test_carry_refuses_unclear_repairs() -> None:
    # Already balanced: nothing to do.
    assert carry_spectators(["CCBr", "[Cl-]"], ["CCCl", "[Br-]"], SN2_SMIRKS) is None
    # SMIRKS reactant not in the current state: reaction centre unknown.
    assert carry_spectators(["CCI", "[Cl-]", "O"], ["CCCl", "[Br-]"], SN2_SMIRKS) is None
    # SMIRKS product missing from the stated result: not a pure spectator drop.
    assert carry_spectators(["CCBr", "[Cl-]", "O"], ["CCCl"], SN2_SMIRKS) is None
    # Carrying spectators back would still leave an imbalance (an extra species).
    assert carry_spectators(["CCBr", "[Cl-]", "O"], ["CCCl", "[Br-]", "CC"], SN2_SMIRKS) is None


class _FixedMechanismAgent:
    def __init__(self, output: Dict[str, Any]) -> None:
        self.output = output

    def run(self, _state: RunState, _scoped: Dict[str, Any], *, retry_feedback: Optional[Dict[str, Any]] = None) -> StepResult:
        return StepResult(
            step_name="mechanism_synthesis",
            tool_name="predict_mechanistic_step",
            output=json.loads(json.dumps(self.output)),
            source="deterministic",
        )


def _quiet(coordinator: RunCoordinator) -> None:
    coordinator._record_step = lambda *_a, **_k: None  # type: ignore[method-assign]
    coordinator._record_validation_checks = lambda *_a, **_k: None  # type: ignore[method-assign]
    coordinator._emit_candidate_validation_result = lambda *_a, **_k: None  # type: ignore[method-assign]
    coordinator._record_arrow_push_annotation = lambda *_a, **_k: None  # type: ignore[method-assign]


def test_candidate_with_dropped_spectators_validates_after_carry() -> None:
    store = _EventStore()
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]
    _quiet(coordinator)
    state = _state()
    state.current_state = list(MCSA_112_CURRENT)
    coordinator.mechanism_agent = _FixedMechanismAgent(  # type: ignore[assignment]
        {
            "current_state": MCSA_112_CURRENT,
            "resulting_state": MCSA_112_RESULTING,
            "reaction_smirks": MCSA_112_SMIRKS,
            "unchanged_starting_materials_detected": False,
            "resulting_state_changed": True,
        }
    )
    candidate = {"rank": 1, "candidate_id": "c1-r1-x", "intermediate_smiles": MCSA_112_RESULTING[-1],
                 "reaction_smirks": MCSA_112_SMIRKS, "electron_pushes": LP_PUSH,
                 "resulting_state": MCSA_112_RESULTING}
    result = coordinator._try_candidate_with_retries(
        state, candidate, {"selected_candidate": candidate},
        enabled_validators={"atom_balance_validation", "state_progress_validation"},
    )
    assert result["status"] == "validated", result.get("last_validation")
    carried = store.types("spectators_carried")
    assert carried and sorted(carried[0]["carried"]) == ["CC(N)C(N)=O", "CCC(C)C", "CCC(C)C", "O"]
    assert result["branch_candidate"].mechanism_output["spectators_carried"]


def test_spectator_carry_can_be_switched_off() -> None:
    store = _EventStore()
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]
    _quiet(coordinator)
    state = _state()
    state.run_config.spectator_carry_enabled = False
    state.run_config.candidate_rescue_enabled = False
    coordinator.mechanism_agent = _FixedMechanismAgent(  # type: ignore[assignment]
        {"current_state": MCSA_112_CURRENT, "resulting_state": MCSA_112_RESULTING, "reaction_smirks": MCSA_112_SMIRKS,
         "unchanged_starting_materials_detected": False, "resulting_state_changed": True}
    )
    candidate = {"rank": 1, "intermediate_smiles": "X", "reaction_smirks": MCSA_112_SMIRKS,
                 "electron_pushes": LP_PUSH, "resulting_state": MCSA_112_RESULTING}
    result = coordinator._try_candidate_with_retries(
        state, candidate, {"selected_candidate": candidate}, enabled_validators={"atom_balance_validation"}
    )
    assert result["status"] == "failed"
    assert not store.types("spectators_carried")


# ── Rescue additions are applied, and a failing re-validation is visible ──


def test_rescue_additions_are_applied_and_failed_revalidation_is_recorded() -> None:
    """mcsa-200-1: the rescue reported success, but its kept additions left a deficit."""
    store = _EventStore()
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]
    _quiet(coordinator)
    state = _state()
    state.run_config.spectator_carry_enabled = False
    current = ["CC(=O)Cl", "CCO", "[Na+]", "[Na+]"]
    coordinator.mechanism_agent = _FixedMechanismAgent(  # type: ignore[assignment]
        {"current_state": current, "resulting_state": ["CCOC(C)=O"], "reaction_smirks": "",
         "unchanged_starting_materials_detected": False, "resulting_state_changed": True}
    )

    def _fake_rescue(_state: RunState, **_kwargs: Any) -> StepResult:
        return StepResult(
            step_name="candidate_rescue",
            tool_name="predict_missing_reagents_for_candidate",
            output={"status": "partial", "model_status": "success", "add_reactants": [], "add_products": ["[Na+]", "Cl"]},
            source="llm",
        )

    coordinator._attempt_candidate_rescue = lambda st, **kw: _fake_rescue(st, **kw)  # type: ignore[method-assign]
    candidate = {"rank": 1, "candidate_id": "c1-r1-y", "intermediate_smiles": "CCOC(C)=O",
                 "reaction_smirks": ACYL_SMIRKS, "electron_pushes": [{"kind": "lone_pair", "source_atom": "7", "target_atom": "2", "electrons": 2}],
                 "resulting_state": ["CCOC(C)=O"]}
    result = coordinator._try_candidate_with_retries(
        state, candidate, {"selected_candidate": candidate}, enabled_validators={"atom_balance_validation"}
    )
    assert result["status"] == "failed"
    assert result["rescue_outcome"] == "applied_still_failing"
    revalidation = store.types("candidate_rescue_revalidation_failed")
    assert revalidation, "the post-rescue verdict must be recorded"
    # The additions were in the state that was re-validated...
    assert {"[Na+]", "Cl"} <= set(revalidation[0]["resulting_state"])
    # ...and the residual is the one left AFTER them (one Na+), not the original deficit.
    assert revalidation[0]["deficit"] == {"Na": 1}
    retry_failed = store.types("mechanism_retry_failed")[-1]
    assert retry_failed["rescue_validation"]["deficit"] == {"Na": 1}
    assert result["rescue_validation"]["added"] == ["[Na+]", "Cl"]
    note = rejection_note(candidate, result)
    assert "reagent rescue added" in note and "Na1" in note


# ── Reproposal before pausing ────────────────────────────────────────────


class _ScriptedProposals:
    """Proposal agent that records the guidance each round received."""

    def __init__(self, rounds: List[List[Dict[str, Any]]]) -> None:
        self.rounds = rounds
        self.guidance: List[Optional[Dict[str, Any]]] = []

    def run(self, _state: RunState, template_guidance: Optional[Dict[str, Any]] = None) -> StepResult:
        index = min(len(self.guidance), len(self.rounds) - 1)
        self.guidance.append(dict(template_guidance or {}))
        return StepResult(
            step_name="mechanism_step_proposal",
            tool_name="propose_mechanism_step",
            output={"classification": "intermediate_step", "candidates": [dict(c) for c in self.rounds[index]]},
            source="llm",
        )


def _failing_attempt(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "failed",
        "branch_candidate": None,
        "last_validation": {
            "passed": False,
            "checks": [{"name": "atom_balance", "passed": False, "details": {"deficit": {"C": 2, "N": 1}, "surplus": {}}}],
        },
        "failed_checks": ["atom_balance"],
        "validation_signature": "sig",
        "candidate_rank": int(candidate.get("rank") or 0),
        "rescue_attempted": True,
        "rescue_outcome": "no_changes",
        "mechanism_output": {
            "current_state": ["CCBr", "[Cl-]", "CCN", "CCN"],
            "resulting_state": ["CCCl", "[Br-]", "CCN"],
            "reaction_smirks": SN2_SMIRKS,
        },
    }


def _candidate(rank: int, smiles: str) -> Dict[str, Any]:
    return {
        "rank": rank,
        "intermediate_smiles": smiles,
        "reaction_smirks": SN2_SMIRKS,
        "electron_pushes": [{"kind": "lone_pair", "source_atom": "4", "target_atom": "2", "electrons": 2}],
        "resulting_state": ["CCCl", "[Br-]"],
    }


def _loop_coordinator(store: _EventStore, proposals: _ScriptedProposals, try_fn: Any) -> RunCoordinator:
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]
    _quiet(coordinator)
    coordinator.intermediate_agent = proposals  # type: ignore[assignment]
    coordinator._try_candidate_with_retries = try_fn  # type: ignore[method-assign]
    coordinator._apply_candidate = lambda st, chosen, **_k: setattr(st, "step_index", st.step_index + 1)  # type: ignore[method-assign]
    coordinator._run_post_step_modules = lambda *_a, **_k: None  # type: ignore[method-assign]
    return coordinator


def test_all_candidates_failing_triggers_reproposal_then_pause_after_budget() -> None:
    store = _EventStore()
    proposals = _ScriptedProposals([[_candidate(1, "CCCl"), _candidate(2, "CCCl")]])
    coordinator = _loop_coordinator(store, proposals, lambda st, cand, *_a, **_k: _failing_attempt(cand))
    state = _state(max_reproposals=2)

    with pytest.raises(_RunPaused):
        coordinator._run_mechanism_loop(state, threading.Event())

    requested = [p for p in store.types("mechanism_reproposal_requested") if p["reason"] == "retry_exhausted"]
    assert [p["reproposal_count"] for p in requested] == [1, 2]
    assert len(proposals.guidance) == 3  # initial proposal + 2 reproposals
    notes = proposals.guidance[1]["rejection_notes"]
    assert len(notes) == 2 and notes[0].startswith("rank 1")
    assert "short C2 N1" in notes[0]
    assert "dropped spectators" in notes[0] and "CCN" in notes[0]
    limit = store.types("mechanism_reproposal_limit_reached")
    assert limit and limit[-1]["reason"] == "retry_exhausted"
    assert store.types("mechanism_retry_exhausted"), "pause only after the reproposals are spent"


def test_reproposal_that_validates_continues_the_run() -> None:
    store = _EventStore()
    proposals = _ScriptedProposals([[_candidate(1, "CCCl")], [_candidate(1, "CCCl")]])
    calls = {"n": 0}

    def _try(st: RunState, cand: Dict[str, Any], *_a: Any, **_k: Any) -> Dict[str, Any]:
        calls["n"] += 1
        if calls["n"] == 1:
            return _failing_attempt(cand)
        return {
            "status": "validated",
            "branch_candidate": BranchCandidate(
                rank=1,
                intermediate_smiles="CCCl",
                intermediate_output=cand,
                mechanism_output={"contains_target_product": True, "resulting_state": ["CCCl", "[Br-]"]},
                resulting_state=["CCCl", "[Br-]"],
                validation_summary={"passed": True},
            ),
            "last_validation": {"passed": True},
            "failed_checks": [],
        }

    coordinator = _loop_coordinator(store, proposals, _try)
    state = _state()
    coordinator._run_mechanism_loop(state, threading.Event())

    assert [p["reason"] for p in store.types("mechanism_reproposal_requested")] == ["retry_exhausted"]
    assert store.types("target_products_detected")
    assert not store.types("mechanism_retry_exhausted")


def test_reproposal_on_retry_exhausted_can_be_disabled() -> None:
    store = _EventStore()
    proposals = _ScriptedProposals([[_candidate(1, "CCCl")]])
    coordinator = _loop_coordinator(store, proposals, lambda st, cand, *_a, **_k: _failing_attempt(cand))
    state = _state()
    state.run_config.reproposal_on_retry_exhausted = False
    with pytest.raises(_RunPaused):
        coordinator._run_mechanism_loop(state, threading.Event())
    assert not store.types("mechanism_reproposal_requested")
    assert len(proposals.guidance) == 1


# ── The notes reach the model ────────────────────────────────────────────


def _capture_prompt(monkeypatch: pytest.MonkeyPatch) -> List[Any]:
    sent: List[Any] = []

    class _Response:
        usage = None

        def __init__(self) -> None:
            self.tool_calls = [{"arguments": json.dumps({"classification": "intermediate_step", "analysis": "a",
                                                          "candidates": [_candidate(1, "CCCl")]})}]

    class _LLM:
        def invoke(self, messages: Any, *_a: Any, **_k: Any) -> _Response:
            sent.append(messages)
            return _Response()

    monkeypatch.setattr("mechanistic_agent.tools.adapter_supports_forced_tools", lambda _m: True)
    monkeypatch.setattr("mechanistic_agent.tools.get_model_api_key", lambda *_a, **_k: "test-key")
    monkeypatch.setattr("mechanistic_agent.tools.get_chat_model", lambda *_a, **_k: _LLM())
    return sent


def _text(messages: Any) -> str:
    return json.dumps(messages, default=lambda o: getattr(o, "content", str(o)))


@pytest.mark.parametrize("turn_mode", [None, "delta"])
def test_rejection_notes_are_rendered_in_full_and_delta_turns(monkeypatch: pytest.MonkeyPatch, turn_mode: Optional[str]) -> None:
    sent = _capture_prompt(monkeypatch)
    note = "rank 1 (CCCl): failed atom_balance; your resulting_state dropped spectators [CCN]"
    conversation = None
    if turn_mode:
        conversation = {"prior_messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}],
                        "turn_mode": turn_mode, "preamble": "", "last_guidance_sha": ""}
    propose_intermediates(
        starting_materials=["CCBr", "[Cl-]"],
        products=["CCCl", "[Br-]"],
        current_state=["CCBr", "[Cl-]", "CCN"],
        template_guidance={"rejection_notes": [note], "avoid_failed_checks": ["atom_balance"]},
        conversation=conversation,
    )
    text = _text(sent[-1])
    assert "VALIDATION FEEDBACK" in text
    assert "dropped spectators [CCN]" in text
    assert '"rejection_notes"' not in text  # rendered as a section, not as guidance JSON


# ── Retry budget 3 (harness loop_module) through the real retry loop ─────

from pathlib import Path  # noqa: E402

from mechanistic_agent.core.registries import HarnessRegistry  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
ACID = "CC(=O)O"
# Acetic acid loses its proton with no acceptor drawn: a proton-only residual (H and charge).
PROTON_LOSS = {
    "current_state": [ACID, "CCN"],
    "resulting_state": ["CC(=O)[O-]", "CCN"],
    "reaction_smirks": "[CH3:1][C:2](=[O:3])[OH:4]>>[CH3:1][C:2](=[O:3])[O-:4] |mech:v1;sigma:4-5>4|",
}
# The ethylamine vanishes: a heavy-atom residual, never deferrable as a proton.
HEAVY_LOSS = {
    "current_state": [ACID, "CCN"],
    "resulting_state": ["CC(=O)[O-]", "[NH4+]"],
    "reaction_smirks": "[CH3:1][C:2](=[O:3])[OH:4]>>[CH3:1][C:2](=[O:3])[O-:4] |mech:v1;sigma:4-5>4|",
}


class _CountingMechanismAgent:
    def __init__(self, output: Dict[str, Any]) -> None:
        self.output = output
        self.calls = 0

    def run(self, _state: RunState, _scoped: Dict[str, Any], *, retry_feedback: Optional[Dict[str, Any]] = None) -> StepResult:
        self.calls += 1
        return StepResult(
            step_name="mechanism_synthesis",
            tool_name="predict_mechanistic_step",
            output={
                **json.loads(json.dumps(self.output)),
                "unchanged_starting_materials_detected": False,
                "resulting_state_changed": True,
                "contains_target_product": True,
                "bond_electron_validation": {"valid": True, "total_delta": 0},
            },
            source="deterministic",
        )


def _harness_loop(harness_name: str, output: Dict[str, Any], *, mode: str = "unverified"):  # type: ignore[no-untyped-def]
    store = _EventStore()
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]
    _quiet(coordinator)
    state = _state(mode=mode, max_reproposals=2)
    state.current_state = list(output["current_state"])
    harness = HarnessRegistry(_ROOT / "harness_versions").load(harness_name)
    coordinator._apply_harness_run_config_defaults(state, harness, raw_config={})
    assert state.run_config.retry_same_candidate_max == 3
    coordinator._attempt_candidate_rescue = lambda *_a, **_k: None  # type: ignore[method-assign]
    agent = _CountingMechanismAgent(output)
    coordinator.mechanism_agent = agent  # type: ignore[assignment]
    candidate = {
        "rank": 1,
        "intermediate_smiles": "CC(=O)[O-]",
        "reaction_smirks": output["reaction_smirks"],
        "electron_pushes": [{"kind": "lone_pair", "source_atom": "4", "target_atom": "1", "electrons": 2}],
        "resulting_state": list(output["resulting_state"]),
    }
    proposals = _ScriptedProposals([[candidate]])
    coordinator.intermediate_agent = proposals  # type: ignore[assignment]
    coordinator._apply_candidate = lambda st, chosen, **_k: setattr(st, "step_index", st.step_index + 1)  # type: ignore[method-assign]
    coordinator._run_post_step_modules = lambda *_a, **_k: None  # type: ignore[method-assign]
    return store, coordinator, state, agent, proposals


def _run(coordinator: RunCoordinator, state: RunState) -> bool:
    """Run the loop; True when it paused."""
    try:
        coordinator._run_mechanism_loop(state, threading.Event())
    except _RunPaused:
        return True
    return False


@pytest.mark.parametrize(
    "harness_name,balance_mode,proceed",
    [("default", "deferred", False), ("permissive_default", "strict", True), ("threaded_unified", "proton_deferred", False)],
)
def test_retry_budget_3_proton_residual_soft_advances(harness_name: str, balance_mode: str, proceed: bool) -> None:
    store, coordinator, state, agent, proposals = _harness_loop(harness_name, PROTON_LOSS)
    assert state.run_config.balance_mode == balance_mode
    assert state.run_config.proceed_on_validation_failure is proceed
    paused = _run(coordinator, state)
    assert not paused
    # The deterministic repeat stopped the retries after 2 of 3 without forcing a reproposal.
    assert agent.calls == 2
    assert store.types("mechanism_retry_stopped_deterministic")
    assert not store.types("mechanism_reproposal_requested")
    soft = store.types("mechanism_step_soft_advance")
    assert soft and soft[0]["reason"] == "balance_pending"
    assert not store.types("run_failed")


def test_retry_budget_3_heavy_residual_under_proton_deferred_reproposes_then_pauses() -> None:
    store, coordinator, state, agent, proposals = _harness_loop("threaded_unified", HEAVY_LOSS)
    paused = _run(coordinator, state)
    assert paused
    reasons = [p["reason"] for p in store.types("mechanism_reproposal_requested")]
    assert reasons == ["retry_exhausted", "retry_exhausted"]
    assert len(proposals.guidance) == 3 and proposals.guidance[1]["rejection_notes"]
    assert agent.calls == 6  # 2 per proposal: the third retry is skipped as a deterministic repeat
    assert store.types("mechanism_reproposal_limit_reached")[-1]["next"] == "pause"
    assert store.types("mechanism_retry_exhausted")
    assert not store.types("run_failed")


def test_retry_budget_3_default_harness_heavy_residual_soft_advances_under_deferred() -> None:
    # "deferred" accepts any atom-balance-only failure as a flagged step.
    store, coordinator, state, agent, _proposals = _harness_loop("default", HEAVY_LOSS)
    assert not _run(coordinator, state)
    assert store.types("mechanism_step_soft_advance")
    assert not store.types("mechanism_reproposal_requested")


def test_verified_mode_still_pauses_without_reproposal_or_soft_advance() -> None:
    store, coordinator, state, agent, proposals = _harness_loop("threaded_unified", PROTON_LOSS, mode="verified")
    assert _run(coordinator, state)
    assert not store.types("mechanism_reproposal_requested")
    assert not store.types("mechanism_step_soft_advance")
    assert not store.types("run_failed")
    assert store.types("mechanism_retry_exhausted")
