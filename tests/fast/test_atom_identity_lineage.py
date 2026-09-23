"""Persistent atom identity on accepted steps and the atom-lineage projection (Observatory PRD §11).

``mechanism_step_accepted`` carries ``atom_identity`` from the mapped loop
state (persistent ids, map numbers, elements, identity counters), and
``build_observatory`` turns the accepted path into a per-atom lineage table.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List

import pytest

pytest.importorskip("rdkit")

from mechanistic_agent.core.coordinator import RunCoordinator
from mechanistic_agent.core.db import RunStore
from mechanistic_agent.core.observatory import build_observatory
from mechanistic_agent.core.types import RunConfig, RunInput, RunState, StepResult

SN2_SMIRKS = "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] |mech:v1;lp:4>2;sigma:2-3>3|"


class _Proposer:
    def run(self, _state: RunState, **_kw: Any) -> StepResult:
        return StepResult(step_name="mechanism_step_proposal", tool_name="propose_mechanism_step", source="llm", output={
            "candidates": [{"rank": 1, "intermediate_smiles": "CCCl", "reaction_description": "SN2", "reaction_smirks": SN2_SMIRKS,
                            "electron_pushes": [{"kind": "lone_pair", "source_atom": "4", "target_atom": "2", "electrons": 2},
                                                {"kind": "sigma_bond", "source_bond": ["2", "3"], "through_atom": "3", "target_atom": "3", "electrons": 2}],
                            "resulting_state": ["CCCl", "[Br-]"]}]})


def _run(tmp_path: Path) -> tuple[RunStore, str]:
    store = RunStore(tmp_path / "mechanistic.db")
    run_input = RunInput(starting_materials=["CCBr", "[Cl-]"], products=["CCCl", "[Br-]"])
    run_id = store.create_run(mode="unverified", input_payload={"starting_materials": run_input.starting_materials, "products": run_input.products},
                              config={"model": "gpt-4o"}, prompt_bundle_hash="a", skill_bundle_hash="b", memory_bundle_hash="c")
    state = RunState(run_id=run_id, mode="unverified", run_input=run_input,
                     run_config=RunConfig(model="gpt-4o", model_family="openai", max_steps=1, intermediate_prediction_enabled=True, max_runtime_seconds=30.0))
    state.initialise()
    coordinator = RunCoordinator(store=store)
    coordinator.intermediate_agent = _Proposer()  # type: ignore[assignment]
    coordinator._run_mechanism_loop(state, threading.Event())
    return store, run_id


def _accepted(store: RunStore, run_id: str) -> List[Dict[str, Any]]:
    return [e["payload"] for e in store.list_events(run_id) if e["event_type"] == "mechanism_step_accepted"]


def test_accepted_step_carries_atom_identity_from_mapped_loop_state(tmp_path: Path) -> None:
    store, run_id = _run(tmp_path)
    accepted = _accepted(store, run_id)
    assert len(accepted) == 1
    identity = accepted[0]["atom_identity"]
    assert identity["schema_version"] == "atom_identity.v1"
    assert identity["identity_source"] == "derived"  # loop_state_mapping is "stripped" by default
    assert identity["mapped_species"] == ["[CH3:1][CH2:2][Cl:4]", "[Br-:3]"]
    atoms = sorted(identity["atoms"], key=lambda a: a["map_number"])
    assert [a["element"] for a in atoms] == ["C", "C", "Br", "Cl"]
    assert [a["map_number"] for a in atoms] == [1, 2, 3, 4]
    assert all(isinstance(a["pid"], int) for a in atoms)
    assert len({a["pid"] for a in atoms}) == 4
    assert identity["preserved_id_count"] == 4
    assert identity["new_ids"] == [] and identity["lost_ids"] == []
    assert identity["identity_resynced"] is False
    assert identity["smirks_state_agreement"] is True


def test_observatory_builds_lineage_from_identity(tmp_path: Path) -> None:
    store, run_id = _run(tmp_path)
    obs = build_observatory(store.list_events(run_id), run_input={"starting_materials": ["CCBr", "[Cl-]"], "products": ["CCCl", "[Br-]"]})
    step = obs["accepted_path"][0]
    assert step["identity"]["identity_source"] == "derived"
    assert step["identity"]["preserved_id_count"] == 4
    state = obs["states"][step["to_state_id"]]
    assert state["mapped_species"] == ["[CH3:1][CH2:2][Cl:4]", "[Br-:3]"]
    lineage = obs["atom_lineage"]
    assert lineage["schema_version"] == "atom_lineage.v1"
    assert lineage["state_ids"] == [step["to_state_id"]]
    rows = {row["element"] + str(row["pid"]): row for row in lineage["atoms"]}
    assert len(rows) == 4
    br = next(r for r in lineage["atoms"] if r["element"] == "Br")
    assert br["path"] == [{"state_id": step["to_state_id"], "step_index": 1, "map_number": 3, "present": True}]
    assert lineage["changed_pids"] == []


# ---------------------------------------------------------------------------
# synthetic multi-step lineage (no chemistry needed)
# ---------------------------------------------------------------------------

def _accepted_event(seq: int, step: int, cid: str, species: List[str], atoms: List[tuple[int, int, str]], new: List[int] = (), lost: List[int] = ()) -> Dict[str, Any]:
    return {"seq": seq, "event_type": "mechanism_step_accepted", "step_name": "mechanism_synthesis", "payload": {
        "step_index": step, "candidate_id": cid, "candidate_rank": 1, "acceptance_kind": "validated",
        "current_state": [], "resulting_state": species, "predicted_intermediate": species[0], "contains_target_product": False,
        "validation_summary": {"passed": True},
        "atom_identity": {"schema_version": "atom_identity.v1", "identity_source": "persistent", "mapped_species": species,
                          "atoms": [{"pid": pid, "map_number": m, "element": el, "component": 0} for pid, m, el in atoms],
                          "preserved_id_count": len(atoms) - len(new), "new_ids": list(new), "lost_ids": list(lost),
                          "identity_resynced": False, "smirks_state_agreement": True, "executed": True},
    }}


def test_lineage_tracks_atoms_across_steps_and_flags_new_and_lost() -> None:
    events = [
        _accepted_event(1, 1, "c1", ["[CH3:1][OH2+:2]", "[Cl-:3]"], [(10, 1, "C"), (11, 2, "O"), (12, 3, "Cl")]),
        # step 2: water leaves (pid 11 lost), chloride bonds (pid 12 kept), a new species appears (pid 13)
        _accepted_event(2, 2, "c2", ["[CH3:1][Cl:3]", "[OH2:5]"], [(10, 1, "C"), (12, 3, "Cl"), (13, 5, "O")], new=[13], lost=[11]),
    ]
    obs = build_observatory(events)
    lineage = obs["atom_lineage"]
    assert lineage["state_ids"] == ["st:c1", "st:c2"]
    by_pid = {row["pid"]: row for row in lineage["atoms"]}
    assert by_pid[10]["path"] == [
        {"state_id": "st:c1", "step_index": 1, "map_number": 1, "present": True},
        {"state_id": "st:c2", "step_index": 2, "map_number": 1, "present": True},
    ]
    assert by_pid[11]["path"][1] == {"state_id": "st:c2", "step_index": 2, "map_number": None, "present": False}
    assert by_pid[13]["path"][0]["present"] is False and by_pid[13]["path"][1]["map_number"] == 5
    assert by_pid[11]["lost_at_step"] == 2 and by_pid[13]["new_at_step"] == 2
    assert lineage["changed_pids"] == [11, 13]
    assert obs["accepted_path"][1]["identity"]["new_ids"] == [13]


def test_legacy_accepted_events_without_identity_yield_empty_lineage() -> None:
    events = [{"seq": 1, "event_type": "mechanism_step_accepted", "payload": {"step_index": 1, "candidate_id": "c1", "acceptance_kind": "validated",
                                                                               "current_state": ["A"], "resulting_state": ["B"], "validation_summary": {"passed": True}}}]
    obs = build_observatory(events)
    assert obs["accepted_path"][0]["identity"] is None
    assert obs["atom_lineage"] == {"schema_version": "atom_lineage.v1", "state_ids": [], "atoms": [], "changed_pids": [], "identity_source": None}
