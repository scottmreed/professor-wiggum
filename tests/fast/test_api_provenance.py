"""API surface for provenance (Observatory PRD §3.7.5, §14.4, §27 M0).

The verified-step route writes rows outside the coordinator, so it must use
the same normalization; the run snapshot must expose a replayable provenance
summary derived from persisted events.
"""
from __future__ import annotations

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("starlette")
pytest.importorskip("rdkit")

from fastapi.testclient import TestClient  # noqa: E402

from mechanistic_agent.api.app import create_app  # noqa: E402
from tests.fast.test_api_runtime import _prepare_base  # noqa: E402


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(_prepare_base(tmp_path)))


def test_verified_human_step_is_recorded_with_human_provenance(tmp_path: Path) -> None:
    client = _client(tmp_path)
    run_id = client.post(
        "/api/runs",
        json={"mode": "verified", "starting_materials": ["CCBr", "[Cl-]"], "products": ["CCCl", "[Br-]"], "model": "gpt-5", "max_steps": 1},
    ).json()["run_id"]

    resp = client.post(
        f"/api/runs/{run_id}/mechanism_steps",
        json={
            "step_index": 1,
            "current_state": ["[CH3:1][CH2:2][Br:3]", "[Cl-:4]"],
            "resulting_state": ["[CH3:1][CH2:2][Cl:4]", "[Br-:3]"],
            "predicted_intermediate": "[CH3:1][CH2:2][Cl:4]",
            "target_products": ["[CH3:1][CH2:2][Cl:4]", "[Br-:3]"],
            "electron_pushes": [
                {"kind": "lone_pair", "source_atom": "4", "target_atom": "2", "electrons": 2},
                {"kind": "sigma_bond", "source_bond": ["2", "3"], "through_atom": "3", "target_atom": "3", "electrons": 2},
            ],
            "reaction_smirks": "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] |mech:v1;lp:4>2;sigma:2-3>3|",
        },
    )
    assert resp.status_code == 200, resp.text

    snapshot = client.get(f"/api/runs/{run_id}").json()
    prov = snapshot["provenance"]
    assert prov["steps"]["mechanism_synthesis"]["engine"] == "human"
    assert prov["steps"]["mechanism_synthesis"]["resolved_model"] is None
    assert prov["inventory"]["models_by_engine"] == {}
    assert prov["inventory"]["call_counts"] == {}

    verbose = client.get(f"/api/runs/{run_id}?verbose=true").json()
    assert verbose["provenance"] == prov
    events = verbose["events"]
    started = [e for e in events if e["event_type"] == "step_started"]
    human_started = [e for e in started if e["payload"]["tool_name"] == "human_submitted_mechanistic_step"]
    assert human_started and human_started[-1]["payload"]["planned_engine"] == "human"
    assert human_started[-1]["payload"]["planned_model"] is None
    validator_started = [e for e in started if e["payload"]["step_name"].endswith("_validation")]
    assert validator_started and all(e["payload"]["planned_engine"] == "deterministic" for e in validator_started)
    outputs = [e for e in events if e["event_type"] == "step_output"]
    assert outputs and outputs[-1]["payload"]["provenance"]["engine"] == "human"
    assert outputs[-1]["payload"]["source"] == "human"


def test_snapshot_provenance_present_for_fresh_run(tmp_path: Path) -> None:
    client = _client(tmp_path)
    run_id = client.post(
        "/api/runs",
        json={"mode": "unverified", "starting_materials": ["C=O"], "products": ["CO"], "model": "gpt-5", "max_steps": 1},
    ).json()["run_id"]
    snapshot = client.get(f"/api/runs/{run_id}").json()
    assert snapshot["provenance"] == {
        "event_schema_version": "mechanism_observatory_event.v1",
        "steps": {},
        "inventory": {"models_by_engine": {}, "call_counts": {}, "failed_calls": 0},
    }
