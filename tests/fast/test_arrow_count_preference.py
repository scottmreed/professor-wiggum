"""Arrow-count preference over validated candidates (ported from ChemIllusion).

Among candidates that passed every validator for the same step and reach the same
outcome up to resonance, the one that needs the fewest two-electron arrows
(``ceil(BE_L1 / 4)`` from its mapped SMIRKS) goes first; equal arrows → fewer
charged atoms; equal on both → rank order. It only reorders: it never rejects,
filters, adds or changes a candidate.
"""
from __future__ import annotations

import copy
import threading
from typing import Any, Dict, List, Optional

import pytest

pytest.importorskip("rdkit")

from mechanistic_agent.core import arrow_count as arrow_count_module
from mechanistic_agent.core.arrow_count import (
    candidate_arrow_count,
    charged_atom_count,
    choose_arrow_count_preference,
    outcome_key,
    step_arrow_count,
)
from mechanistic_agent.core.coordinator import RunCoordinator, _RunPaused
from mechanistic_agent.core.types import BranchCandidate, RunConfig, RunInput, RunState, StepResult

# Enolate alkylation. Same product, two drawings of the enolate.
ENOLATE_O = "C=C(C)[O-]"
ENOLATE_C = "[CH2-]C(C)=O"
SMIRKS_FROM_O = (
    "[CH2:1]=[C:2]([CH3:3])[O-:4].[CH3:5][I:6]>>[CH3:5][CH2:1][C:2]([CH3:3])=[O:4].[I-:6]"
)  # 3 arrows: O lone pair -> C=O, C=C pi -> C-CH3, C-I -> I-
SMIRKS_FROM_C = (
    "[CH2-:1][C:2]([CH3:3])=[O:4].[CH3:5][I:6]>>[CH3:5][CH2:1][C:2]([CH3:3])=[O:4].[I-:6]"
)  # 2 arrows: C lone pair -> C-CH3, C-I -> I-
PRODUCT = ["CCC(C)=O", "[I-]"]
ENOLATE_STATE = [ENOLATE_O, "CI"]
OTHER_PRODUCT = ["CC(=CC)C", "[I-]"]  # a different species


# ---------------------------------------------------------------------------
# Arrow count from the BE matrix
# ---------------------------------------------------------------------------

def test_lone_pair_to_bond_is_one_arrow() -> None:
    assert step_arrow_count("[NH3:1].[CH3+:2]>>[NH3+:1][CH3:2]") == 1


def test_sn2_is_two_arrows() -> None:
    assert step_arrow_count("[NH3:1].[CH3:2][Br:3]>>[NH3+:1][CH3:2].[Br-:3]") == 2


@pytest.mark.parametrize(
    "smirks",
    [
        "[OH2:1].[NH4+:2]>>[OH3+:1].[NH3:2]",  # implicit H
        "[O:1]([H])[H].[N+:2]([H])([H])([H])[H]>>[O+:1]([H])([H])[H].[N:2]([H])([H])[H]",  # unmapped explicit H
        "[OH2:1].[NH3+:2][H:3]>>[OH2+:1][H:3].[NH3:2]",  # mapped H
    ],
)
def test_proton_transfer_is_two_arrows(smirks: str) -> None:
    # Hydrogens folded into the diagonal still count their H-X bond: B: -> H, H-A -> A.
    assert step_arrow_count(smirks) == 2


def test_enolate_alkylation_from_each_contributor() -> None:
    assert step_arrow_count(SMIRKS_FROM_O) == 3
    assert step_arrow_count(SMIRKS_FROM_C) == 2


@pytest.mark.parametrize("smirks", [None, "", "CC>>CC", "not a reaction", "[CH3:1]>>[CH3:1"])
def test_unmeasurable_steps_have_no_count(smirks) -> None:
    # No SMIRKS, no atom maps, unparseable, or a BE projection error.
    assert step_arrow_count(smirks) is None


def test_candidate_arrow_count_requires_smirks_to_describe_the_stated_outcome() -> None:
    # The arrow count measures the SMIRKS; it only counts for the candidate when the
    # SMIRKS products (>= 2 heavy atoms) are, up to resonance, in its resulting state.
    assert candidate_arrow_count(SMIRKS_FROM_C, PRODUCT, ENOLATE_STATE) == 2
    assert candidate_arrow_count(SMIRKS_FROM_O, ["CCC(C)=O", "[I-]", "O"], ENOLATE_STATE) == 3
    assert candidate_arrow_count(SMIRKS_FROM_C, OTHER_PRODUCT, ENOLATE_STATE) is None
    assert candidate_arrow_count(None, PRODUCT, ENOLATE_STATE) is None
    assert candidate_arrow_count(SMIRKS_FROM_C, ["not a smiles"], ENOLATE_STATE) is None
    # Only a fully mapped, whole-species SMIRKS is a measurement: a reaction-centre
    # (core-only) SMIRKS leaves atoms out of the count, so it takes no part.
    core = "[N:1].[C:2](=[O:3])[Cl:4]>>[N+:1][C:2](=[O:3]).[Cl-:4]"
    assert candidate_arrow_count(core, ["CC(=O)[NH3+]", "[Cl-]"], ["N", "CC(=O)Cl"]) is None


SN2_STATE = ["CO", "[Br-]"]
SN2_START = ["[OH-]", "CBr"]


def test_full_mapping_is_required_so_under_mapping_cannot_win() -> None:
    full = "[OH-:1].[CH3:2][Br:3]>>[OH:1][CH3:2].[Br-:3]"
    assert candidate_arrow_count(full, SN2_STATE, SN2_START) == 2
    # Br left unmapped: its C-Br bond change would vanish from the BE count.
    assert candidate_arrow_count("[OH-:1].[CH3:2]Br>>[OH:1][CH3:2].[Br-]", SN2_STATE, SN2_START) is None
    # A map that exists on one side only.
    assert candidate_arrow_count("[OH-:1].[CH3:2][Br:3]>>[OH:1][CH3:2].[Br-]", SN2_STATE, SN2_START) is None
    # Leaving group left out: [CH3] is then a methyl radical, not the CH3Br in the state.
    assert candidate_arrow_count("[OH-:1].[CH3:2]>>[OH:1][CH3:2]", SN2_STATE, SN2_START) is None
    decision = choose_arrow_count_preference(
        [SN2_STATE, SN2_STATE],
        [candidate_arrow_count(full, SN2_STATE, SN2_START),
         candidate_arrow_count("[OH-:1].[CH3:2]Br>>[OH:1][CH3:2].[Br-]", SN2_STATE, SN2_START)],
    )
    assert decision is None


def test_large_fully_mapped_step_stays_fast() -> None:
    import time

    n = 400
    chain = "".join(f"[CH2:{i}]" for i in range(2, n))
    smirks = f"[CH3:1]{chain}[CH2:{n}][Br:{n + 1}].[OH-:{n + 2}]>>[CH3:1]{chain}[CH2:{n}][OH:{n + 2}].[Br-:{n + 1}]"
    assert step_arrow_count(smirks) == 2
    start = time.perf_counter()
    for _ in range(3):
        step_arrow_count(smirks)
    assert (time.perf_counter() - start) / 3 < 0.05  # dense n x n matrices took ~190 ms here


# ---------------------------------------------------------------------------
# Outcome and charge
# ---------------------------------------------------------------------------

def test_reactants_must_be_current_species_up_to_resonance() -> None:
    # The carbanion form is another contributor of the O- enolate in the state: allowed.
    assert candidate_arrow_count(SMIRKS_FROM_C, PRODUCT, [ENOLATE_O, "CI"]) == 2
    # A reactant that is not in the current state at all.
    assert candidate_arrow_count(SMIRKS_FROM_C, PRODUCT, ["CC(C)=O", "CI"]) is None


def test_outcome_joins_resonance_forms_and_ignores_small_species() -> None:
    assert outcome_key([ENOLATE_O, "O"]) == outcome_key([ENOLATE_C])
    assert outcome_key(["CC(C)=O"]) != outcome_key(["C=C(C)O"])  # tautomers differ
    assert outcome_key(["CC(=O)[O-]", "CC(=O)[O-]"]) != outcome_key(["CC(=O)[O-]"])  # multiset
    assert outcome_key(["not a smiles"]) is None


def test_charged_atom_count() -> None:
    assert charged_atom_count(["CC(=O)N"]) == 0
    assert charged_atom_count(["CC([O-])=[NH2+]"]) == 2
    assert charged_atom_count(["CC(=O)N", "[Cl-]"]) == 0  # small species do not count


# ---------------------------------------------------------------------------
# choose_arrow_count_preference (pure)
# ---------------------------------------------------------------------------

def test_fewer_arrows_same_outcome_moves_forward() -> None:
    decision = choose_arrow_count_preference([PRODUCT, PRODUCT], [3, 2])
    assert decision is not None
    assert decision["chosen_index"] == 1 and decision["displaced_index"] == 0
    assert decision["chosen_arrows"] == 2 and decision["displaced_arrows"] == 3
    assert decision["reason"] == "fewer_arrows_same_outcome"


def test_equal_arrows_fewer_charged_atoms_wins() -> None:
    decision = choose_arrow_count_preference([["CC([O-])=[NH2+]"], ["CC(=O)N"]], [2, 2])
    assert decision is not None and decision["chosen_index"] == 1
    assert decision["chosen_charged_atoms"] == 0 and decision["displaced_charged_atoms"] == 2
    assert decision["reason"] == "fewer_charged_atoms_same_arrows"


def test_equal_on_both_keeps_rank() -> None:
    assert choose_arrow_count_preference([PRODUCT, PRODUCT], [2, 2]) is None


def test_different_outcomes_never_reorder() -> None:
    assert choose_arrow_count_preference([PRODUCT, OTHER_PRODUCT], [3, 1]) is None


def test_unmeasurable_candidate_keeps_its_rank() -> None:
    # An unmeasurable top is never displaced; an unmeasurable lower rank is never chosen.
    assert choose_arrow_count_preference([PRODUCT, PRODUCT], [None, 1]) is None
    assert choose_arrow_count_preference([PRODUCT, PRODUCT], [2, None]) is None
    decision = choose_arrow_count_preference([PRODUCT, PRODUCT, PRODUCT], [3, None, 2])
    assert decision is not None and decision["chosen_index"] == 2


def test_best_of_three_picks_fewest_then_earliest() -> None:
    decision = choose_arrow_count_preference([PRODUCT, PRODUCT, PRODUCT, PRODUCT], [4, 2, 3, 2])
    assert decision is not None and decision["chosen_index"] == 1


# ---------------------------------------------------------------------------
# Coordinator method: never-reject invariants
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


def _state(mode: str = "unverified") -> RunState:
    run_input = RunInput(starting_materials=[ENOLATE_O, "CI"], products=["CCC(C)=O"])
    run_config = RunConfig(
        model="gpt-4", model_family="openai", max_steps=1, max_runtime_seconds=60.0,
        intermediate_prediction_enabled=True, step_mapping_enabled=False,
    )
    state = RunState(run_id="run-arrows", mode=mode, run_input=run_input, run_config=run_config)
    state.initialise()
    state.current_state = [ENOLATE_O, "CI"]
    return state


def _bc(rank: int, resulting: List[str], smirks: Optional[str], *, passed: bool = True) -> BranchCandidate:
    return BranchCandidate(
        rank=rank,
        intermediate_smiles=resulting[0],
        intermediate_output={"candidate_id": f"c1-r{rank}-abcd", "rank": rank},
        mechanism_output={"resulting_state": list(resulting), "reaction_smirks": smirks},
        resulting_state=list(resulting),
        validation_summary={"passed": passed, "checks": [{"name": "atom_balance", "passed": passed}]},
    )


FIXTURES = {
    "fewer_arrows": [_bc(1, PRODUCT, SMIRKS_FROM_O), _bc(2, PRODUCT, SMIRKS_FROM_C)],
    "already_best": [_bc(1, PRODUCT, SMIRKS_FROM_C), _bc(2, PRODUCT, SMIRKS_FROM_O)],
    "different_outcomes": [_bc(1, PRODUCT, SMIRKS_FROM_O), _bc(2, OTHER_PRODUCT, SMIRKS_FROM_C)],
    "no_smirks_top": [_bc(1, PRODUCT, None), _bc(2, PRODUCT, SMIRKS_FROM_C)],
    "projection_error_lower": [_bc(1, PRODUCT, SMIRKS_FROM_O), _bc(2, PRODUCT, "[CH3:1]>>[CH3:1")],
    # rank 2's SMIRKS makes a different product than its stated resulting state.
    "smirks_disagrees_lower": [
        _bc(1, PRODUCT, SMIRKS_FROM_O),
        _bc(2, PRODUCT, "[NH3:1].[CH3+:2]>>[NH3+:1][CH3:2]"),  # 1 arrow, but not this outcome
    ],
    "three": [_bc(1, PRODUCT, SMIRKS_FROM_O), _bc(2, OTHER_PRODUCT, SMIRKS_FROM_C), _bc(3, PRODUCT, SMIRKS_FROM_C)],
    "single": [_bc(1, PRODUCT, SMIRKS_FROM_O)],
}


def _apply(validated: List[BranchCandidate], *, mode: str = "unverified", proton_fired: bool = False):
    store = _Store()
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]
    result = coordinator._apply_arrow_count_preference(
        _state(mode), validated, proton_preference_applied=proton_fired
    )
    return result, store


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_never_reject_invariants(name: str) -> None:
    validated = FIXTURES[name]
    snapshot = [copy.deepcopy((c.validation_summary, c.mechanism_output, c.resulting_state)) for c in validated]
    before_ids = [c.candidate_id for c in validated]
    result, _ = _apply(list(validated))
    assert len(result) == len(validated)
    assert sorted(c.candidate_id for c in result) == sorted(before_ids)
    assert all(any(r is c for c in validated) for r in result)  # nothing new added
    for candidate, (summary, output, resulting) in zip(validated, snapshot):
        assert candidate.validation_summary == summary
        assert candidate.mechanism_output == output
        assert candidate.resulting_state == resulting


def test_method_moves_fewer_arrow_candidate_forward_and_keeps_the_other() -> None:
    validated = FIXTURES["fewer_arrows"]
    result, store = _apply(list(validated))
    assert [c.rank for c in result] == [2, 1]
    events = store.of("arrow_count_preference_applied")
    assert len(events) == 1
    event = events[0]
    assert event["chosen_candidate_id"] == "c1-r2-abcd" and event["chosen_rank"] == 2
    assert event["displaced_candidate_id"] == "c1-r1-abcd" and event["displaced_rank"] == 1
    assert event["chosen_arrows"] == 2 and event["displaced_arrows"] == 3
    assert event["chosen_charged_atoms"] == 0 and event["displaced_charged_atoms"] == 0
    assert event["reason"] == "fewer_arrows_same_outcome"
    assert event["outcome_key"]
    assert event["step_index"] == 1


def test_three_candidates_rest_keep_relative_order() -> None:
    result, _ = _apply(list(FIXTURES["three"]))
    assert [c.rank for c in result] == [3, 1, 2]


@pytest.mark.parametrize(
    "name",
    ["already_best", "different_outcomes", "no_smirks_top", "projection_error_lower", "smirks_disagrees_lower", "single"],
)
def test_method_no_op_cases(name: str) -> None:
    validated = list(FIXTURES[name])
    result, store = _apply(validated)
    assert [c.rank for c in result] == [c.rank for c in validated]
    assert not store.of("arrow_count_preference_applied")


def test_skipped_when_proton_transfer_preference_fired() -> None:
    validated = list(FIXTURES["fewer_arrows"])
    result, store = _apply(validated, proton_fired=True)
    assert result == validated
    assert not store.of("arrow_count_preference_applied")


def test_verified_mode_is_untouched() -> None:
    validated = list(FIXTURES["fewer_arrows"])
    result, store = _apply(validated, mode="verified")
    assert result == validated
    assert not store.of("arrow_count_preference_applied")


def test_soft_or_unpassed_candidates_are_never_reordered() -> None:
    validated = [_bc(1, PRODUCT, SMIRKS_FROM_O), _bc(2, PRODUCT, SMIRKS_FROM_C, passed=False)]
    result, store = _apply(list(validated))
    assert result == validated
    assert not store.of("arrow_count_preference_applied")


def test_exception_inside_helper_returns_input_order(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(arrow_count_module, "step_arrow_count", _boom)
    validated = list(FIXTURES["fewer_arrows"])
    result, store = _apply(validated)
    assert result == validated
    assert not store.of("arrow_count_preference_applied")


# ---------------------------------------------------------------------------
# Mechanism loop integration
# ---------------------------------------------------------------------------

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


def _validate_all(_state: RunState, candidate: Dict[str, Any], *_a: Any, **_k: Any) -> Dict[str, Any]:
    rank = int(candidate.get("rank") or 0)
    return {
        "status": "validated",
        "branch_candidate": BranchCandidate(
            rank=rank,
            intermediate_smiles=candidate["intermediate_smiles"],
            intermediate_output=dict(candidate),
            mechanism_output={
                "contains_target_product": False,
                "resulting_state": list(candidate["resulting_state"]),
                "reaction_smirks": candidate.get("reaction_smirks"),
            },
            resulting_state=list(candidate["resulting_state"]),
            validation_summary={"passed": True, "checks": []},
        ),
        "candidate_rank": rank,
    }


def _run_loop(candidates: List[Dict[str, Any]], proton_override=None) -> _Store:
    store = _Store()
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]
    state = _state()
    coordinator.intermediate_agent = _Proposer(candidates)  # type: ignore[assignment]
    coordinator._try_candidate_with_retries = _validate_all  # type: ignore[method-assign]
    coordinator._record_step = lambda *_a, **_k: None  # type: ignore[method-assign]
    if proton_override is not None:
        coordinator._apply_proton_transfer_preference = proton_override  # type: ignore[method-assign]
    try:
        coordinator._run_mechanism_loop(state, threading.Event())
    except _RunPaused:
        pass
    return store


_LOOP_CANDIDATES = [
    {"rank": 1, "intermediate_smiles": PRODUCT[0], "resulting_state": PRODUCT, "reaction_smirks": SMIRKS_FROM_O},
    {"rank": 2, "intermediate_smiles": PRODUCT[0], "resulting_state": PRODUCT, "reaction_smirks": SMIRKS_FROM_C},
]


def test_loop_accepts_fewer_arrow_candidate_and_keeps_other_as_branch() -> None:
    store = _run_loop(_LOOP_CANDIDATES)
    accepted = store.of("mechanism_step_accepted")
    assert accepted, [e["event_type"] for e in store.events]
    assert accepted[0]["candidate_rank"] == 2
    assert accepted[0]["acceptance_kind"] == "validated"
    applied = store.of("arrow_count_preference_applied")
    assert len(applied) == 1
    assert applied[0]["chosen_candidate_id"] == accepted[0]["candidate_id"]
    branch = store.of("branch_point_created")
    assert branch and branch[0]["chosen_rank"] == 2 and branch[0]["alternative_ranks"] == [1]


def test_loop_skips_arrow_preference_after_proton_preference_fired() -> None:
    def _proton_reorder(_state: RunState, validated: List[BranchCandidate]) -> List[BranchCandidate]:
        return [validated[-1]] + validated[:-1] if len(validated) > 1 else validated

    reversed_candidates = list(reversed(_LOOP_CANDIDATES))
    reversed_candidates[0] = dict(reversed_candidates[0], rank=1)
    reversed_candidates[1] = dict(reversed_candidates[1], rank=2)
    # rank 1 = carbanion form (2 arrows); the fake proton preference moves rank 2 (3 arrows) first.
    store = _run_loop(reversed_candidates, proton_override=_proton_reorder)
    accepted = store.of("mechanism_step_accepted")
    assert accepted[0]["candidate_rank"] == 2
    assert not store.of("arrow_count_preference_applied")
