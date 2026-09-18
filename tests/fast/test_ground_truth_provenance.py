"""Ground-truth exposure is declared, gates evidence/leaderboard use, and never leaks into bridge requests."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from mechanistic_agent.agent_bridge import (
    BRIDGE_DIR_ENV,
    BRIDGE_POLL_ENV,
    BRIDGE_SAW_GROUND_TRUTH_ENV,
    BRIDGE_TIMEOUT_ENV,
    build_origin_provenance,
    pending_requests,
    read_request,
    write_response,
)
from mechanistic_agent.core.coordinator import RunCoordinator
from mechanistic_agent.core.db import RunStore

GROUND_TRUTH_MARKER = "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] |mech:v1;lp:4>2;sigma:2-3>3| GROUND_TRUTH_MARKER_7f3a"


def test_origin_provenance_declares_ground_truth_exposure(monkeypatch) -> None:
    monkeypatch.delenv(BRIDGE_SAW_GROUND_TRUTH_ENV, raising=False)
    assert build_origin_provenance("agent-bridge")["responder_saw_ground_truth"] == "undeclared"
    monkeypatch.setenv(BRIDGE_SAW_GROUND_TRUTH_ENV, "true")
    assert build_origin_provenance("agent-bridge")["responder_saw_ground_truth"] is True
    monkeypatch.setenv(BRIDGE_SAW_GROUND_TRUTH_ENV, "false")
    assert build_origin_provenance("agent-bridge")["responder_saw_ground_truth"] is False


def _seed_eval_run(store: RunStore, *, saw_ground_truth: str, monkeypatch) -> str:
    monkeypatch.setenv(BRIDGE_SAW_GROUND_TRUTH_ENV, saw_ground_truth)
    eval_set_id = store.add_eval_set(
        name=f"set_{saw_ground_truth}",
        version="v1",
        source_path=None,
        sha256=None,
        cases=[{"case_id": "c1", "input": {"starting_materials": ["CCBr", "[Cl-]"], "products": ["CCCl", "[Br-]"]}, "expected": {"products": ["CCCl", "[Br-]"]}}],
    )
    run_id = store.create_run(
        mode="unverified",
        input_payload={"starting_materials": ["CCBr", "[Cl-]"], "products": ["CCCl", "[Br-]"]},
        config={"model": "agent-bridge", "model_name": "agent-bridge", "model_family": "agent"},
        prompt_bundle_hash="p",
        skill_bundle_hash="s",
    )
    eval_run_id = store.create_eval_run(
        eval_set_id=eval_set_id, run_group_name=f"grp_{saw_ground_truth}", model="agent-bridge", harness_bundle_hash="h"
    )
    store.record_eval_run_result(
        eval_run_id=eval_run_id, case_id="c1", run_id=run_id, score=0.99, passed=True, cost=None, latency_ms=10.0, summary={}
    )
    store.set_eval_run_status(eval_run_id, "completed")
    return eval_set_id


def test_leaderboard_rejects_rows_whose_responder_saw_ground_truth(tmp_path: Path, monkeypatch) -> None:
    store = RunStore(tmp_path / "data" / "mechanistic.db")
    replay_set = _seed_eval_run(store, saw_ground_truth="true", monkeypatch=monkeypatch)
    blind_set = _seed_eval_run(store, saw_ground_truth="false", monkeypatch=monkeypatch)

    assert store.leaderboard(replay_set) == []
    blind_rows = store.leaderboard(blind_set)
    assert len(blind_rows) == 1
    assert blind_rows[0]["model"] == "agent-bridge"


def _canned_answer(tool_name: str) -> dict:
    if tool_name == "assess_conditions_result":
        return {"environment": "neutral", "representative_ph": 7.0}
    if tool_name == "missing_reagents_result":
        return {"missing_reactants": [], "missing_products": []}
    if tool_name == "atom_mapping_result":
        return {"mapped_atoms": [], "unmapped_atoms": [], "confidence": 0.5, "reasoning": "n/a"}
    if tool_name == "reaction_type_selection_result":
        return {"selected_label_exact": "no_match", "confidence": 0.5, "rationale": "n/a"}
    if tool_name == "mechanism_step_proposal_result":
        return {
            "classification": "intermediate_step",
            "analysis": "SN2",
            "candidates": [
                {
                    "rank": 1,
                    "intermediate_smiles": "CCCl",
                    "reaction_description": "chloride displaces bromide",
                    "reaction_smirks": "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] |mech:v1;lp:4>2;sigma:2-3>3|",
                    "electron_pushes": [{"kind": "lone_pair", "source_atom": "4", "target_atom": "2", "electrons": 2}],
                    "resulting_state": ["CCCl", "[Br-]"],
                }
            ],
        }
    return {}


def test_bridge_requests_never_contain_verified_mechanism(tmp_path: Path, monkeypatch) -> None:
    """Drive a real run through the keyless bridge and assert no request carries the
    eval case's verified mechanism (or any expected/known-mechanism block)."""
    bridge_dir = tmp_path / "bridge"
    monkeypatch.setenv(BRIDGE_DIR_ENV, str(bridge_dir))
    monkeypatch.setenv(BRIDGE_POLL_ENV, "0.02")
    monkeypatch.setenv(BRIDGE_TIMEOUT_ENV, "60")
    monkeypatch.setenv(BRIDGE_SAW_GROUND_TRUTH_ENV, "false")
    monkeypatch.setenv("MECHANISTIC_ACTIVE_MODEL", "agent-bridge")

    store = RunStore(tmp_path / "data" / "mechanistic.db")
    # The verified mechanism lives only in the eval set's expected block.
    store.add_eval_set(
        name="leak_probe",
        version="v1",
        source_path=None,
        sha256=None,
        cases=[
            {
                "case_id": "probe_case",
                "input": {"starting_materials": ["CCBr", "[Cl-]"], "products": ["CCCl", "[Br-]"]},
                "expected": {
                    "products": ["CCCl", "[Br-]"],
                    "verified_mechanism": {"steps": [{"step_index": 1, "reaction_smirks": GROUND_TRUTH_MARKER, "resulting_state": ["CCCl", "[Br-]"]}]},
                    "known_mechanism": {"steps": [{"step_index": 1, "target_smiles": "CCCl", "note": "GROUND_TRUTH_MARKER_7f3a"}]},
                },
            }
        ],
    )
    # Mirror what the eval runner passes into a case run: inputs only.
    run_id = store.create_run(
        mode="unverified",
        input_payload={
            "starting_materials": ["CCBr", "[Cl-]"],
            "products": ["CCCl", "[Br-]"],
            "temperature_celsius": 25.0,
            "ph": 7.0,
            "example_id": "probe_case",
        },
        config={
            "model": "agent-bridge",
            "model_name": "agent-bridge",
            "model_family": "agent",
            "optional_llm_tools": ["attempt_atom_mapping", "predict_missing_reagents"],
            "functional_groups_enabled": True,
            "intermediate_prediction_enabled": True,
            "max_steps": 1,
            "max_runtime_seconds": 60.0,
            "harness_name": "default",
            "chemistry_backend": "python",
        },
        prompt_bundle_hash="p",
        skill_bundle_hash="s",
    )
    assert store.get_run_row(run_id)["config"]["origin"]["responder_saw_ground_truth"] is False

    coordinator = RunCoordinator(store)
    stop_event = threading.Event()
    worker = threading.Thread(target=coordinator.execute_run, args=(run_id, stop_event), daemon=True)
    worker.start()

    captured: list[str] = []
    deadline = time.time() + 60.0
    while worker.is_alive() and time.time() < deadline:
        for req_path in pending_requests(str(bridge_dir)):
            raw = req_path.read_text(encoding="utf-8")
            captured.append(raw)
            model_input = read_request(req_path)["model_input"]
            tool_name = model_input["tool_choice"]["function"]["name"]
            write_response(req_path, tool_calls=[{"name": tool_name, "arguments": _canned_answer(tool_name)}])
        time.sleep(0.02)
    stop_event.set()
    worker.join(timeout=10.0)

    assert captured, "the run never reached the bridge"
    for raw in captured:
        assert "GROUND_TRUTH_MARKER_7f3a" not in raw
        assert "verified_mechanism" not in raw
        assert "known_mechanism" not in raw
        payload = json.loads(raw)
        assert set(payload["model_input"].keys()) == {"messages", "tools", "tool_choice"}
    tools_seen = {json.loads(raw)["model_input"]["tool_choice"]["function"]["name"] for raw in captured}
    assert "mechanism_step_proposal_result" in tools_seen
