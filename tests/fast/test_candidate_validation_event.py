"""``candidate_validation_result`` (Observatory PRD §16.4): every validated or rejected
candidate carries its ReactionFocus and bond-electron view, keyed by candidate_id."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List

import pytest

pytest.importorskip("rdkit")

from mechanistic_agent.core.coordinator import RunCoordinator, _RunPaused
from mechanistic_agent.core.db import RunStore
from mechanistic_agent.core.types import RunConfig, RunInput, RunState, StepResult

SN2_SMIRKS = "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] |mech:v1;lp:4>2;sigma:2-3>3|"


class _Proposer:
    def __init__(self, smiles: str, resulting: List[str], smirks: str = SN2_SMIRKS) -> None:
        self.smiles, self.resulting, self.smirks = smiles, resulting, smirks

    def run(self, _state: RunState, **_kw: Any) -> StepResult:
        return StepResult(
            step_name="mechanism_step_proposal",
            tool_name="propose_mechanism_step",
            output={
                "classification": "intermediate_step",
                "candidates": [
                    {
                        "rank": 1,
                        "intermediate_smiles": self.smiles,
                        "reaction_description": "SN2",
                        "reaction_smirks": self.smirks,
                        "electron_pushes": [
                            {"kind": "lone_pair", "source_atom": "4", "target_atom": "2", "electrons": 2},
                            {"kind": "sigma_bond", "source_bond": ["2", "3"], "through_atom": "3", "target_atom": "3", "electrons": 2},
                        ],
                        "resulting_state": self.resulting,
                    }
                ],
            },
            source="llm",
        )


def _run(tmp_path: Path, proposer: _Proposer) -> tuple[RunStore, str]:
    store = RunStore(tmp_path / "mechanistic.db")
    run_input = RunInput(starting_materials=["CCBr", "[Cl-]"], products=["CCCl", "[Br-]"])
    run_config = RunConfig(model="gpt-4o", model_family="openai", max_steps=1, intermediate_prediction_enabled=True,
                           max_runtime_seconds=30.0)
    run_id = store.create_run(
        mode="unverified",
        input_payload={"starting_materials": run_input.starting_materials, "products": run_input.products},
        config={"model": "gpt-4o"}, prompt_bundle_hash="a", skill_bundle_hash="b", memory_bundle_hash="c",
    )
    state = RunState(run_id=run_id, mode="unverified", run_input=run_input, run_config=run_config)
    state.initialise()
    coordinator = RunCoordinator(store=store)
    coordinator.intermediate_agent = proposer  # type: ignore[assignment]
    try:
        coordinator._run_mechanism_loop(state, threading.Event())
    except _RunPaused:
        pass  # all candidates rejected → loop pauses for a user decision
    return store, run_id


def _events(store: RunStore, run_id: str, kind: str) -> List[Dict[str, Any]]:
    return [e["payload"] for e in store.list_events(run_id) if e.get("event_type") == kind]


def test_validated_candidate_emits_result_with_focus_and_be_view(tmp_path: Path) -> None:
    store, run_id = _run(tmp_path, _Proposer("CCCl", ["CCCl", "[Br-]"]))

    results = _events(store, run_id, "candidate_validation_result")
    assert len(results) == 1, [e["event_type"] for e in store.list_events(run_id)]
    result = results[0]
    proposed = _events(store, run_id, "mechanism_candidates_proposed")[0]
    assert result["candidate_id"] == proposed["candidates"][0]["candidate_id"]
    assert result["event_schema_version"] == "mechanism_observatory_event.v1"
    assert result["accepted"] is True
    assert result["validation"]["passed"] is True
    assert result["failed_checks"] == []
    focus = result["reaction_focus"]
    assert focus["schema_version"] == "reaction_focus.v1"
    assert focus["core_atom_ids"] == ["a2", "a3", "a4"]
    view = result["bond_electron_view"]
    assert view["convention"] == "ugi_flower_kekule_v1"
    assert view["is_focus_projection"] is True
    assert view["atom_ids"] == focus["matrix_atom_ids"]
    assert view["electron_delta_sum"] == 0
    assert result["current_state"] == ["CCBr", "[Cl-]"]
    assert result["resulting_state"] == ["CCCl", "[Br-]"]
    accepted = _events(store, run_id, "mechanism_step_accepted")
    assert accepted and accepted[0]["candidate_id"] == result["candidate_id"]


def test_rejected_candidate_emits_result_marked_not_accepted(tmp_path: Path) -> None:
    # resulting state claims a product that the SMIRKS does not produce → validators fail
    store, run_id = _run(tmp_path, _Proposer("CCI", ["CCI", "[Br-]"]))
    results = _events(store, run_id, "candidate_validation_result")
    assert results, [e["event_type"] for e in store.list_events(run_id)]
    assert all(r["accepted"] is False for r in results)
    assert results[0]["failed_checks"]
    assert results[0]["reaction_focus"]["schema_version"] == "reaction_focus.v1"
    assert all(r["candidate_id"] for r in results)
