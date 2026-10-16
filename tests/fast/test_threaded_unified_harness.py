"""threaded_unified harness end to end through the keyless bridge.

One pre-analysis request replaces the four pre-loop calls; mechanism
proposals continue that chat; a rejected proposal is dropped from the thread
and comes back only as an avoid note.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

pytest.importorskip("rdkit")

from mechanistic_agent.agent_bridge import (
    BRIDGE_DIR_ENV,
    BRIDGE_POLL_ENV,
    BRIDGE_SAW_GROUND_TRUTH_ENV,
    BRIDGE_TIMEOUT_ENV,
    pending_requests,
    read_request,
    write_response,
)
from mechanistic_agent.core.coordinator import RunCoordinator
from mechanistic_agent.core.db import RunStore

GOOD_SMIRKS = "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] |mech:v1;lp:4>2;sigma:2-3>3|"
FIRST_TRY_MARKER = "FIRST_TRY_MARKER_91c2"

PREANALYSIS = {
    "conditions": {"environment": "neutral", "representative_ph": 7.0, "justification": "SN2 halide exchange"},
    "missing_reagents": {"missing_reactants": [], "missing_products": []},
    "atom_mapping": {
        "mapped_atoms": [
            {"product_atom": "CCCl#0", "source": {"molecule_index": 0, "smiles": "CCBr", "atom_index": 0}},
            {"product_atom": "CCCl#1", "source": {"molecule_index": 0, "smiles": "CCBr", "atom_index": 1}},
            {"product_atom": "CCCl#2", "source": {"molecule_index": 1, "smiles": "[Cl-]", "atom_index": 0}},
            {"product_atom": "[Br-]#0", "source": {"molecule_index": 0, "smiles": "CCBr", "atom_index": 2}},
        ],
        "unmapped_atoms": [],
        "confidence": 0.9,
        "reasoning": "chloride replaces bromide",
    },
    "reaction_type": {"selected_label_exact": "no_match", "confidence": 0.5, "rationale": "simple SN2"},
}


def _proposal(*, good: bool) -> Dict[str, Any]:
    candidate = {
        "rank": 1,
        "intermediate_smiles": "CCCl" if good else "C1CC",
        "reaction_description": "chloride displaces bromide",
        "reaction_smirks": GOOD_SMIRKS,
        "electron_pushes": [{"kind": "lone_pair", "source_atom": "4", "target_atom": "2", "electrons": 2}],
        "resulting_state": ["CCCl", "[Br-]"] if good else ["C1CC", "[Br-]"],
    }
    return {
        "classification": "intermediate_step",
        "analysis": "SN2" if good else FIRST_TRY_MARKER,
        "candidates": [candidate],
    }


def _answer(tool_name: str, proposal_calls: List[int]) -> Dict[str, Any]:
    if tool_name == "unified_preanalysis_result":
        return PREANALYSIS
    if tool_name == "mechanism_step_proposal_result":
        proposal_calls.append(1)
        return _proposal(good=len(proposal_calls) > 1)
    if tool_name == "missing_reagents_result":
        return {"missing_reactants": [], "missing_products": []}
    return {}


def test_threaded_unified_run_through_bridge(tmp_path: Path, monkeypatch) -> None:
    bridge_dir = tmp_path / "bridge"
    monkeypatch.setenv(BRIDGE_DIR_ENV, str(bridge_dir))
    monkeypatch.setenv(BRIDGE_POLL_ENV, "0.02")
    monkeypatch.setenv(BRIDGE_TIMEOUT_ENV, "60")
    monkeypatch.setenv(BRIDGE_SAW_GROUND_TRUTH_ENV, "false")
    monkeypatch.setenv("MECHANISTIC_ACTIVE_MODEL", "agent-bridge")

    store = RunStore(tmp_path / "data" / "mechanistic.db")
    run_id = store.create_run(
        mode="unverified",
        input_payload={"starting_materials": ["CCBr", "[Cl-]"], "products": ["CCCl", "[Br-]"], "ph": 7.0},
        config={
            "model": "agent-bridge",
            "model_name": "agent-bridge",
            "model_family": "agent",
            "optional_llm_tools": ["attempt_atom_mapping", "predict_missing_reagents"],
            "functional_groups_enabled": True,
            "intermediate_prediction_enabled": True,
            "max_steps": 3,
            "max_runtime_seconds": 90.0,
            "harness_name": "threaded_unified",
            "chemistry_backend": "python",
        },
        prompt_bundle_hash="p",
        skill_bundle_hash="s",
    )

    coordinator = RunCoordinator(store)
    stop_event = threading.Event()
    worker = threading.Thread(target=coordinator.execute_run, args=(run_id, stop_event), daemon=True)
    worker.start()

    requests: List[Dict[str, Any]] = []
    proposal_calls: List[int] = []
    deadline = time.time() + 90.0
    while worker.is_alive() and time.time() < deadline:
        for req_path in pending_requests(str(bridge_dir)):
            payload = json.loads(req_path.read_text(encoding="utf-8"))
            requests.append(payload)
            model_input = read_request(req_path)["model_input"]
            tool_name = model_input["tool_choice"]["function"]["name"]
            write_response(req_path, tool_calls=[{"name": tool_name, "arguments": _answer(tool_name, proposal_calls)}])
        time.sleep(0.02)
    stop_event.set()
    worker.join(timeout=10.0)

    tools = [r["model_input"]["tool_choice"]["function"]["name"] for r in requests]
    for r in requests:
        assert set(r["model_input"].keys()) == {"messages", "tools", "tool_choice"}
    assert tools[0] == "unified_preanalysis_result"
    for old_tool in ("assess_conditions_result", "atom_mapping_result", "reaction_type_selection_result"):
        assert old_tool not in tools
    proposals = [r for r in requests if r["model_input"]["tool_choice"]["function"]["name"] == "mechanism_step_proposal_result"]
    assert len(proposals) >= 2

    events = store.list_events(run_id)
    kinds = [e["event_type"] for e in events]
    checks = next(e["payload"] for e in events if e["event_type"] == "unified_preanalysis_checks")
    assert checks["sections"] == ["conditions", "missing_reagents", "atom_mapping", "reaction_type"]

    first_messages = proposals[0]["model_input"]["messages"]
    last_messages = proposals[-1]["model_input"]["messages"]
    if checks["passed"]:
        # The pre-analysis exchange is the root of the mechanism thread.
        assert first_messages[1]["role"] == "user" and "Pre-analysis" in first_messages[1]["content"]
        assert first_messages[2]["role"] == "assistant"
        assert "unified_preanalysis_result" in first_messages[2]["content"]
    # The rejected first proposal is not in the thread; its avoid note is.
    joined = "\n".join(m["content"] for m in last_messages)
    assert FIRST_TRY_MARKER not in joined
    assert "Already rejected for this step" in last_messages[-1]["content"]
    assert "thread_forked" in kinds
    assert "thread_turn_appended" in kinds

    steps = {row["step_name"]: row for row in store.list_step_outputs(run_id)}
    for derived in ("initial_conditions", "missing_reagents", "atom_mapping", "reaction_type_mapping"):
        assert steps[derived]["source"] == "deterministic"
        assert steps[derived]["output"]["derived_from"] == "unified_preanalysis"
    assert steps["unified_preanalysis"]["source"] == "llm"
    assert "step_atom_mapping" not in steps
    accepted = [e for e in events if e["event_type"] == "mechanism_step_accepted"]
    assert accepted and "CCCl" in accepted[-1]["payload"]["resulting_state"]
    assert store.get_run_row(run_id)["status"] == "completed"
