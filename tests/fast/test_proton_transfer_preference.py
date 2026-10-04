"""Intermolecular proton-transfer preference.

Among candidates that passed validation for the same step, an intermolecular
proton transfer through a shuttle actually available in the reaction is
preferred over an intramolecular proton shift with the same heavy-atom outcome.
Intramolecular shifts are never rejected: they stay as the branch alternative.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

import pytest

pytest.importorskip("rdkit")

from mechanistic_agent.core.coordinator import RunCoordinator, _RunPaused
from mechanistic_agent.core.proton_transfer import (
    available_shuttles,
    choose_proton_transfer_preference,
    classify_proton_transfer,
    skeleton_state_key,
    species_skeleton_key,
)
from mechanistic_agent.core.types import BranchCandidate, RunConfig, RunInput, RunState, StepResult

# Hemiketal-like intermediate with an oxonium on the hydrate oxygen.
CURRENT = ["COC(O)([OH2+])Cn1ccnc1"]
INTRA = ["C[OH+]C(O)(O)Cn1ccnc1"]  # 1,3-shift O->O within the intermediate
INTER = ["COC(O)(O)Cn1ccnc1", "[OH3+]"]  # water removes the proton
NOT_PT = ["CO", "OC(O)Cn1ccnc1"]  # heavy-atom skeleton changes (C-O cleavage)

CARBAMIC = "O=C(O)NC1CC2(CC2)C1"
CARBAMATE = "O=C([O-])NC1CC2(CC2)C1"
ZWITTERION = "O=C([O-])[NH2+]C1CC2(CC2)C1"
AMMONIUM_ACID = "O=C(O)[NH2+]C1CC2(CC2)C1"
TFA = "O=C(O)C(F)(F)F"
TFA_ANION = "O=C([O-])C(F)(F)F"


# ---------------------------------------------------------------------------
# skeleton_state_key
# ---------------------------------------------------------------------------

def test_skeleton_key_ignores_charge_hydrogens_and_small_species() -> None:
    assert skeleton_state_key(INTRA) == skeleton_state_key(INTER)
    assert skeleton_state_key(CURRENT) == skeleton_state_key(INTER)
    assert skeleton_state_key(["O", "[OH3+]", "[OH-]", "[Cl-]"]) == frozenset()
    assert skeleton_state_key([CARBAMIC, TFA_ANION]) == skeleton_state_key([ZWITTERION, TFA])
    assert skeleton_state_key(NOT_PT) != skeleton_state_key(CURRENT)


def test_skeleton_key_keeps_bond_orders() -> None:
    # Enolate C- vs O-protonation give different products.
    assert skeleton_state_key(["CC(C)=O"]) != skeleton_state_key(["C=C(C)O"])


# ---------------------------------------------------------------------------
# classify_proton_transfer
# ---------------------------------------------------------------------------

def test_intramolecular_shift_is_classified() -> None:
    result = classify_proton_transfer(CURRENT, INTRA)
    assert result["is_proton_transfer"] is True
    assert result["mode"] == "intramolecular"
    assert result["shuttle"] is None


def test_water_shuttle_from_pool_is_classified_intermolecular() -> None:
    shuttles = available_shuttles(CURRENT, starting_materials=["O"])
    result = classify_proton_transfer(CURRENT, INTER, shuttles=shuttles)
    assert result["is_proton_transfer"] is True
    assert result["mode"] == "intermolecular"
    assert result["shuttle"] == "O"
    assert result["shuttle_available"] is True


def test_water_shuttle_not_in_reaction_is_not_available() -> None:
    result = classify_proton_transfer(CURRENT, INTER, shuttles=available_shuttles(CURRENT))
    assert result["mode"] == "intermolecular"
    assert result["shuttle_available"] is False


def test_trifluoroacetate_shuttle_in_current_state() -> None:
    current = [CARBAMIC, TFA_ANION]
    result = classify_proton_transfer(current, [CARBAMATE, TFA], shuttles=available_shuttles(current))
    assert result["mode"] == "intermolecular"
    assert result["shuttle"] == TFA_ANION
    assert result["shuttle_available"] is True

    intra = classify_proton_transfer(current, [ZWITTERION, TFA_ANION])
    assert intra["mode"] == "intramolecular"


def test_conjugate_of_pool_species_counts_as_available() -> None:
    # TFA is a starting material; its conjugate base (not listed anywhere) is a shuttle too.
    shuttles = available_shuttles([AMMONIUM_ACID], starting_materials=[TFA])
    assert species_skeleton_key(TFA_ANION) in shuttles
    # Trifluoroacetate drawn from the reagent pool removes the N-H proton.
    result = classify_proton_transfer([AMMONIUM_ACID], [CARBAMIC, TFA], shuttles=shuttles)
    assert result["is_proton_transfer"] is True
    assert result["mode"] == "intermolecular"
    assert result["shuttle_available"] is True
    # Without the pool the appearing TFA skeleton makes it a non-proton-transfer step.
    assert classify_proton_transfer([AMMONIUM_ACID], [CARBAMIC, TFA])["is_proton_transfer"] is False


def test_non_proton_transfer_steps() -> None:
    assert classify_proton_transfer(CURRENT, NOT_PT)["is_proton_transfer"] is False
    assert classify_proton_transfer(CURRENT, CURRENT)["is_proton_transfer"] is False
    unparseable = classify_proton_transfer(CURRENT, ["not a smiles"])
    assert unparseable["is_proton_transfer"] is False
    assert unparseable["mode"] is None


def test_states_fall_back_to_reaction_smirks() -> None:
    smirks = "[CH3:1][OH:2].[OH2:3]>>[CH3:1][O-:2].[OH3+:3]"
    result = classify_proton_transfer([], [], reaction_smirks=smirks)
    assert result["mode"] == "intermolecular"
    assert result["shuttle"] == "O"


# ---------------------------------------------------------------------------
# choose_proton_transfer_preference (pure tie-break)
# ---------------------------------------------------------------------------

def test_tie_break_prefers_equivalent_intermolecular() -> None:
    shuttles = available_shuttles(CURRENT, starting_materials=["O"])
    decision = choose_proton_transfer_preference(CURRENT, [INTRA, INTER], shuttles)
    assert decision is not None
    assert decision["chosen_index"] == 1
    assert decision["displaced_index"] == 0
    assert decision["chosen"]["mode"] == "intermolecular"
    assert decision["displaced"]["mode"] == "intramolecular"
    assert decision["shuttle"] == "O"


def test_tie_break_prefers_tfa_shuttle_over_four_membered_carbamic_shift() -> None:
    current = [CARBAMIC, TFA]
    decision = choose_proton_transfer_preference(
        current, [[ZWITTERION, TFA], [AMMONIUM_ACID, TFA_ANION]], available_shuttles(current)
    )
    assert decision is not None and decision["chosen_index"] == 1
    assert decision["shuttle"] == TFA


def test_tie_break_keeps_top_when_shuttle_unavailable() -> None:
    assert choose_proton_transfer_preference(CURRENT, [INTRA, INTER], available_shuttles(CURRENT)) is None


def test_tie_break_keeps_top_when_outcomes_differ() -> None:
    shuttles = available_shuttles(CURRENT, starting_materials=["O"])
    assert choose_proton_transfer_preference(CURRENT, [INTRA, NOT_PT], shuttles) is None


def test_tie_break_keeps_top_when_top_is_already_intermolecular() -> None:
    shuttles = available_shuttles(CURRENT, starting_materials=["O"])
    assert choose_proton_transfer_preference(CURRENT, [INTER, INTRA], shuttles) is None


# ---------------------------------------------------------------------------
# Coordinator integration
# ---------------------------------------------------------------------------

class _Store:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.step_outputs: List[Dict[str, Any]] = []

    def append_event(self, run_id: str, event_type: str, payload: Dict[str, Any], *, step_name: Optional[str] = None) -> None:
        self.events.append({"run_id": run_id, "event_type": event_type, "payload": payload, "seq": len(self.events) + 1})

    def list_events(self, run_id: str, *, after_seq: int = 0, limit: int = 500) -> List[Dict[str, Any]]:
        return [ev for ev in self.events if int(ev.get("seq") or 0) > after_seq][:limit]

    def create_run_pause(self, *, run_id: str, reason: str, details: Dict[str, Any]) -> str:
        return "pause-id"

    def set_run_status(self, run_id: str, status: str) -> None:
        return

    def record_step_output(self, **kwargs: Any) -> None:
        self.step_outputs.append(dict(kwargs))

    def upsert_step_output(self, **kwargs: Any) -> None:
        self.step_outputs.append(dict(kwargs))

    def list_step_outputs(self, run_id: str) -> List[Dict[str, Any]]:
        return list(self.step_outputs)

    def of(self, kind: str) -> List[Dict[str, Any]]:
        return [e["payload"] for e in self.events if e["event_type"] == kind]


def _state(starting_materials: List[str]) -> RunState:
    run_input = RunInput(starting_materials=starting_materials, products=["COC(=O)Cn1ccnc1", "O"])
    run_config = RunConfig(
        model="gpt-4", model_family="openai", max_steps=1, max_runtime_seconds=60.0,
        intermediate_prediction_enabled=True, step_mapping_enabled=False,
    )
    state = RunState(run_id="run-pt", mode="unverified", run_input=run_input, run_config=run_config)
    state.initialise()
    state.current_state = list(CURRENT)
    return state


class _Proposer:
    def __init__(self, candidates: List[Dict[str, Any]]) -> None:
        self.candidates = candidates

    def run(self, _state: RunState, **_kw: Any) -> StepResult:
        return StepResult(
            step_name="mechanism_step_proposal",
            tool_name="propose_mechanism_step",
            output={"classification": "intermediate_step", "candidates": [dict(c) for c in self.candidates]},
            source="llm",
        )


def _validate_by(passing: Dict[int, bool]):
    def _try(_state: RunState, candidate: Dict[str, Any], *_a: Any, **_k: Any) -> Dict[str, Any]:
        rank = int(candidate.get("rank") or 0)
        if not passing.get(rank, True):
            return {"status": "failed", "last_validation": {"passed": False, "checks": []},
                    "failed_checks": ["bond_electron"], "validation_signature": "x", "candidate_rank": rank}
        return {
            "status": "validated",
            "branch_candidate": BranchCandidate(
                rank=rank,
                intermediate_smiles=candidate["intermediate_smiles"],
                intermediate_output=dict(candidate),
                mechanism_output={"contains_target_product": False,
                                  "resulting_state": list(candidate["resulting_state"])},
                resulting_state=list(candidate["resulting_state"]),
                validation_summary={"passed": True, "checks": []},
            ),
            "candidate_rank": rank,
        }
    return _try


def _run(starting_materials: List[str], candidates: List[Dict[str, Any]], passing: Optional[Dict[int, bool]] = None) -> _Store:
    store = _Store()
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]
    state = _state(starting_materials)
    coordinator.intermediate_agent = _Proposer(candidates)  # type: ignore[assignment]
    coordinator._try_candidate_with_retries = _validate_by(passing or {})  # type: ignore[method-assign]
    coordinator._record_step = lambda *_a, **_k: None  # type: ignore[method-assign]
    try:
        coordinator._run_mechanism_loop(state, threading.Event())
    except _RunPaused:
        pass
    return store


_CANDIDATES = [
    {"rank": 1, "intermediate_smiles": INTRA[0], "resulting_state": INTRA, "reaction_description": "1,3-H shift"},
    {"rank": 2, "intermediate_smiles": INTER[0], "resulting_state": INTER, "reaction_description": "water deprotonates"},
]


def test_loop_accepts_intermolecular_and_keeps_intramolecular_as_branch() -> None:
    store = _run(["COC(=O)Cn1ccnc1", "O"], _CANDIDATES)
    accepted = store.of("mechanism_step_accepted")
    assert accepted, [e["event_type"] for e in store.events]
    assert accepted[0]["candidate_rank"] == 2
    assert accepted[0]["proton_transfer_mode"] == "intermolecular"
    assert accepted[0]["resulting_state"] == INTER  # H3O+ is carried for the follow-up step

    applied = store.of("proton_transfer_preference_applied")
    assert len(applied) == 1
    assert applied[0]["chosen_mode"] == "intermolecular"
    assert applied[0]["displaced_mode"] == "intramolecular"
    assert applied[0]["shuttle"] == "O"
    assert applied[0]["chosen_candidate_id"] == accepted[0]["candidate_id"]

    branch = store.of("branch_point_created")
    assert branch and branch[0]["chosen_rank"] == 2
    assert branch[0]["alternative_ranks"] == [1]


def test_loop_keeps_top_when_no_shuttle_available() -> None:
    store = _run(["COC(=O)Cn1ccnc1"], _CANDIDATES)
    accepted = store.of("mechanism_step_accepted")
    assert accepted[0]["candidate_rank"] == 1
    assert accepted[0]["proton_transfer_mode"] == "intramolecular"
    assert not store.of("proton_transfer_preference_applied")


def test_loop_never_prefers_failed_intermolecular_candidate() -> None:
    store = _run(["COC(=O)Cn1ccnc1", "O"], _CANDIDATES, passing={2: False})
    accepted = store.of("mechanism_step_accepted")
    assert accepted[0]["candidate_rank"] == 1
    assert not store.of("proton_transfer_preference_applied")


def test_loop_does_not_override_non_equivalent_outcome() -> None:
    candidates = [
        dict(_CANDIDATES[0]),
        {"rank": 2, "intermediate_smiles": NOT_PT[1], "resulting_state": NOT_PT},
    ]
    store = _run(["COC(=O)Cn1ccnc1", "O"], candidates)
    accepted = store.of("mechanism_step_accepted")
    assert accepted[0]["candidate_rank"] == 1
    assert not store.of("proton_transfer_preference_applied")
