"""``build_observatory`` — event-derived replay projection (Observatory PRD §17, §22).

The Observatory UI (Wiggum M2 and ChemIllusion) reconstructs the whole search
from persisted events only: accepted path, candidate sets with per-candidate
status, validation results with focus/BE payloads, branch points, backtracks,
failed paths and provenance. Building twice from the same events is identical.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List

import pytest

pytest.importorskip("rdkit")

from mechanistic_agent.core.observatory import OBSERVATORY_SCHEMA, build_observatory

# ---------------------------------------------------------------------------
# synthetic event helpers
# ---------------------------------------------------------------------------


class _Log:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def add(self, kind: str, payload: Dict[str, Any], step_name: str | None = None) -> None:
        self.events.append({"seq": len(self.events) + 1, "event_type": kind, "step_name": step_name, "ts": 0.0, "payload": payload})

    def proposed(self, step_index: int, cands: List[tuple[str, int, str]], current: List[str], round_: int = 1) -> None:
        self.add("mechanism_candidates_proposed", {
            "event_schema_version": "mechanism_observatory_event.v1",
            "step_index": step_index, "proposal_round": round_, "candidate_set_id": f"cs{step_index}-{round_}",
            "current_state": current, "coordination_topology": "centralized_mas", "rejected_candidate_count": 0,
            "candidates": [{"candidate_id": cid, "rank": rank, "intermediate_smiles": smi, "reaction_smirks": None,
                            "reaction_description": "", "resulting_state": [smi]} for cid, rank, smi in cands],
        }, "mechanism_step_proposal")

    def validated(self, cid: str, accepted: bool, failed: List[str], retry: int = 0, step_index: int = 1) -> None:
        self.add("candidate_validation_result", {
            "event_schema_version": "mechanism_observatory_event.v1", "step_index": step_index, "attempt": step_index,
            "retry_index": retry, "candidate_id": cid, "candidate_rank": 1, "accepted": accepted,
            "validation": {"passed": accepted, "checks": []}, "failed_checks": failed, "reaction_smirks": "x>>y",
            "current_state": ["A"], "resulting_state": ["B"], "predicted_intermediate": "B",
            "reaction_focus": {"schema_version": "reaction_focus.v1", "core_atom_ids": ["a1"]},
            "bond_electron_view": {"schema_version": "bond_electron_view.v1", "electron_delta_sum": 0},
            "smirks_state_agreement": None, "projection_error": None,
        }, "mechanism_synthesis")

    def accepted(self, step_index: int, cid: str | None, kind: str, current: List[str], resulting: List[str], smi: str, passed: bool = True, target: bool = False) -> None:
        self.add("mechanism_step_accepted", {
            "step_index": step_index, "candidate_rank": 1, "candidate_id": cid, "acceptance_kind": kind,
            "current_state": current, "resulting_state": resulting, "predicted_intermediate": smi,
            "contains_target_product": target, "validation_summary": {"passed": passed, "checks": []},
        }, "mechanism_synthesis")

    def step_output(self, step_name: str, attempt: int, engine: str, model: str | None) -> None:
        self.add("step_output", {"step_name": step_name, "attempt": attempt, "retry_index": 0, "source": engine,
                                 "output": {}, "validation": None,
                                 "provenance": {"engine": engine, "resolved_model": model, "resolved_reasoning": None,
                                                "primary_call_id": "call_1" if model else None, "call_ids": ["call_1"] if model else [],
                                                "model_fallback": False, "fallback_chain": []}}, step_name)


def _linear_run() -> _Log:
    log = _Log()
    log.add("run_started", {})
    log.step_output("mechanism_step_proposal", 1, "llm", "anthropic/claude-opus-4.6")
    log.proposed(1, [("c1-r1-aaaa", 1, "B"), ("c1-r2-bbbb", 2, "C")], current=["A"])
    log.validated("c1-r1-aaaa", False, ["atom_balance"], retry=0)
    log.validated("c1-r1-aaaa", False, ["atom_balance"], retry=1)
    log.validated("c1-r2-bbbb", True, [])
    log.accepted(1, "c1-r2-bbbb", "validated", current=["A"], resulting=["C"], smi="C", target=True)
    log.add("target_products_detected", {"step_index": 1})
    log.add("run_completed", {})
    return log


def test_linear_run_accepted_path_and_candidate_statuses() -> None:
    obs = build_observatory(_linear_run().events, run_id="r1", run_input={"starting_materials": ["A"], "products": ["C"]}, status="completed")
    assert obs["schema_version"] == OBSERVATORY_SCHEMA == "mechanism_observatory.v1"
    assert obs["run_id"] == "r1" and obs["status"] == "completed"
    assert obs["reaction"] == {"starting_materials": ["A"], "products": ["C"]}

    assert [s["step_index"] for s in obs["accepted_path"]] == [1]
    step = obs["accepted_path"][0]
    assert step["candidate_id"] == "c1-r2-bbbb"
    assert step["acceptance_kind"] == "validated"
    assert step["from_state_id"] == "s0"
    assert step["to_state_id"] == "st:c1-r2-bbbb"
    assert step["validation_passed"] is True
    assert step["contains_target_product"] is True
    assert step["proposal_provenance"]["engine"] == "llm"
    assert step["proposal_provenance"]["resolved_model"] == "anthropic/claude-opus-4.6"

    assert obs["states"]["s0"] == {"state_id": "s0", "species": ["A"], "kind": "initial", "step_index": 0, "candidate_id": None}
    assert obs["states"]["st:c1-r2-bbbb"]["species"] == ["C"]
    assert obs["states"]["st:c1-r2-bbbb"]["kind"] == "accepted"

    sets = obs["candidate_sets"]
    assert len(sets) == 1 and sets[0]["candidate_set_id"] == "cs1-1" and sets[0]["current_state_id"] == "s0"
    by_id = {c["candidate_id"]: c for c in sets[0]["candidates"]}
    assert by_id["c1-r1-aaaa"]["status"] == "rejected"
    assert by_id["c1-r1-aaaa"]["failed_checks"] == ["atom_balance"]
    assert by_id["c1-r1-aaaa"]["validation_attempts"] == 2
    assert by_id["c1-r1-aaaa"]["state_id"] == "st:c1-r1-aaaa"
    assert obs["states"]["st:c1-r1-aaaa"]["kind"] == "rejected"
    assert by_id["c1-r2-bbbb"]["status"] == "accepted"
    assert by_id["c1-r2-bbbb"]["reaction_focus"]["core_atom_ids"] == ["a1"]
    assert by_id["c1-r2-bbbb"]["bond_electron_view"]["electron_delta_sum"] == 0

    assert obs["completed"] is True
    assert obs["provenance"]["steps"]["mechanism_step_proposal"]["engine"] == "llm"
    assert obs["last_seq"] == len(_linear_run().events)


def test_backtrack_truncates_path_and_marks_abandoned() -> None:
    log = _Log()
    log.proposed(1, [("c1-r1", 1, "B"), ("c1-r2", 2, "D")], current=["A"])
    log.validated("c1-r1", True, [])
    log.validated("c1-r2", True, [])
    log.add("branch_point_created", {"step_index": 0, "chosen_rank": 1, "chosen_candidate_id": "c1-r1",
                                     "alternative_count": 1, "alternative_ranks": [2], "alternative_candidate_ids": ["c1-r2"]})
    log.accepted(1, "c1-r1", "validated", current=["A"], resulting=["B"], smi="B")
    log.proposed(2, [("c2-r1", 1, "X")], current=["B"])
    log.validated("c2-r1", False, ["state_progress"], step_index=2)
    log.add("mechanism_retry_exhausted", {"attempt": 2})
    log.add("failed_path_recorded", {"branch_step_index": 0, "candidate_rank": 1, "candidate_id": "c1-r1", "steps_in_path": 1})
    log.add("backtrack", {"reverted_to_step": 0, "alternative_rank": 2, "candidate_id": "c1-r2", "intermediate": "D", "remaining_alternatives": 0})
    log.accepted(1, "c1-r2", "backtrack_alternative", current=["A"], resulting=["D"], smi="D")

    obs = build_observatory(log.events)
    assert [(s["candidate_id"], s["acceptance_kind"]) for s in obs["accepted_path"]] == [("c1-r2", "backtrack_alternative")]
    assert obs["abandoned_candidate_ids"] == ["c1-r1"]
    statuses = {c["candidate_id"]: c["status"] for s in obs["candidate_sets"] for c in s["candidates"]}
    assert statuses == {"c1-r1": "abandoned", "c1-r2": "accepted", "c2-r1": "rejected"}
    assert obs["states"]["st:c1-r1"]["kind"] == "abandoned"
    assert obs["branch_points"] == [{"seq": 4, "step_index": 0, "chosen_candidate_id": "c1-r1", "alternative_candidate_ids": ["c1-r2"]}]
    assert obs["backtracks"][0]["candidate_id"] == "c1-r2" and obs["backtracks"][0]["reverted_to_step"] == 0
    assert obs["failed_paths"][0]["candidate_id"] == "c1-r1"
    # the second candidate set hangs off the abandoned state, not the final path
    assert obs["candidate_sets"][1]["current_state_id"] == "st:c1-r1"


def test_soft_advance_without_candidate_id_is_marked_unvalidated() -> None:
    log = _Log()
    log.proposed(1, [("c1-r1", 1, "B")], current=["A"])
    log.validated("c1-r1", False, ["atom_balance"])
    log.add("mechanism_step_soft_advance", {"step_index": 1, "reason": "proceed_on_validation_failure"})
    log.accepted(1, None, "soft_advance", current=["A"], resulting=["B"], smi="B", passed=False)
    obs = build_observatory(log.events)
    step = obs["accepted_path"][0]
    assert step["acceptance_kind"] == "soft_advance"
    assert step["validation_passed"] is False
    assert step["candidate_id"] == "soft-1"
    assert obs["states"][step["to_state_id"]]["kind"] == "soft_advance"
    assert obs["unvalidated_step_count"] == 1


def test_legacy_events_without_candidate_ids_do_not_crash() -> None:
    log = _Log()
    log.add("mechanism_step_accepted", {"step_index": 1, "candidate_rank": 1, "current_state": ["A"], "resulting_state": ["B"],
                                        "predicted_intermediate": "B", "contains_target_product": False, "validation_summary": {"passed": True}})
    obs = build_observatory(log.events)
    assert obs["accepted_path"][0]["candidate_id"] == "legacy-1"
    assert obs["accepted_path"][0]["acceptance_kind"] == "validated"


def test_projection_is_deterministic() -> None:
    events = _linear_run().events
    assert build_observatory(events) == build_observatory(list(events))


def test_active_step_is_the_last_started_without_output() -> None:
    log = _Log()
    log.add("step_started", {"step_name": "mechanism_step_proposal", "attempt": 1, "planned_engine": "llm", "planned_model": "m"}, "mechanism_step_proposal")
    obs = build_observatory(log.events)
    assert obs["active_step"] == {"step_name": "mechanism_step_proposal", "planned_engine": "llm", "planned_model": "m"}
    log.step_output("mechanism_step_proposal", 1, "llm", "m")
    assert build_observatory(log.events)["active_step"] is None


# ---------------------------------------------------------------------------
# integration: real loop + real validators, then API endpoint
# ---------------------------------------------------------------------------

from mechanistic_agent.core.coordinator import RunCoordinator  # noqa: E402
from mechanistic_agent.core.db import RunStore  # noqa: E402
from mechanistic_agent.core.types import RunConfig, RunInput, RunState, StepResult  # noqa: E402

SN2_SMIRKS = "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] |mech:v1;lp:4>2;sigma:2-3>3|"


class _Proposer:
    def run(self, _state: RunState, **_kw: Any) -> StepResult:
        return StepResult(step_name="mechanism_step_proposal", tool_name="propose_mechanism_step", source="llm", output={
            "candidates": [{"rank": 1, "intermediate_smiles": "CCCl", "reaction_description": "SN2", "reaction_smirks": SN2_SMIRKS,
                            "electron_pushes": [{"kind": "lone_pair", "source_atom": "4", "target_atom": "2", "electrons": 2},
                                                {"kind": "sigma_bond", "source_bond": ["2", "3"], "through_atom": "3", "target_atom": "3", "electrons": 2}],
                            "resulting_state": ["CCCl", "[Br-]"]}]})


def test_real_loop_projects_one_accepted_validated_step(tmp_path: Path) -> None:
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

    obs = build_observatory(store.list_events(run_id), run_id=run_id, run_input={"starting_materials": run_input.starting_materials, "products": run_input.products})
    assert len(obs["accepted_path"]) == 1
    step = obs["accepted_path"][0]
    assert step["acceptance_kind"] == "validated" and step["validation_passed"] is True
    cand = obs["candidate_sets"][0]["candidates"][0]
    assert cand["candidate_id"] == step["candidate_id"] and cand["status"] == "accepted"
    assert cand["reaction_focus"]["core_atom_ids"] == ["a2", "a3", "a4"]
    assert cand["bond_electron_view"]["conserved"] is True
    assert step["proposal_provenance"]["engine"] == "llm"
    assert obs["states"][step["to_state_id"]]["species"] == ["CCCl", "[Br-]"]


def test_api_observatory_endpoint(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from mechanistic_agent.api.app import create_app
    from tests.fast.test_api_runtime import _prepare_base

    client = TestClient(create_app(_prepare_base(tmp_path)))
    run_id = client.post("/api/runs", json={"mode": "unverified", "starting_materials": ["C=O"], "products": ["CO"], "model": "gpt-5", "max_steps": 1}).json()["run_id"]
    resp = client.get(f"/api/runs/{run_id}/observatory")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["schema_version"] == "mechanism_observatory.v1"
    assert body["run_id"] == run_id
    assert body["reaction"]["starting_materials"] == ["C=O"]
    assert body["accepted_path"] == [] and body["candidate_sets"] == []
    assert body["states"]["s0"]["species"] == ["C=O"]
    assert client.get("/api/runs/does-not-exist/observatory").status_code == 404
