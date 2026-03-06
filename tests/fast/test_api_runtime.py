from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("starlette")

from fastapi.testclient import TestClient

from mechanistic_agent.api.app import create_app
from mechanistic_agent.core.db import RunStore


def _prepare_base(tmp_path: Path) -> Path:
    (tmp_path / "prompt_versions" / "shared").mkdir(parents=True)
    (tmp_path / "prompt_versions" / "calls" / "assess_initial_conditions").mkdir(parents=True)
    (tmp_path / "prompt_versions" / "calls" / "predict_missing_reagents").mkdir(parents=True)
    (tmp_path / "prompt_versions" / "calls" / "attempt_atom_mapping").mkdir(parents=True)
    (tmp_path / "prompt_versions" / "calls" / "propose_mechanism_step").mkdir(parents=True)
    (tmp_path / "prompt_versions" / "calls" / "evaluate_run_judge").mkdir(parents=True)
    (tmp_path / "skills" / "demo").mkdir(parents=True)
    (tmp_path / "memory_packs").mkdir(parents=True)
    (tmp_path / "mechanistic_agent" / "ui").mkdir(parents=True)
    (tmp_path / "data").mkdir(parents=True)
    (tmp_path / "training_data").mkdir(parents=True)
    (tmp_path / "traces" / "runs").mkdir(parents=True)
    (tmp_path / "traces" / "evidence").mkdir(parents=True)

    (tmp_path / "prompt_versions" / "shared" / "base_system.md").write_text("shared base", encoding="utf-8")
    for call_name in [
        "assess_initial_conditions",
        "predict_missing_reagents",
        "attempt_atom_mapping",
        "propose_mechanism_step",
        "evaluate_run_judge",
    ]:
        call_dir = tmp_path / "prompt_versions" / "calls" / call_name
        (call_dir / "base.md").write_text(f"{call_name} base", encoding="utf-8")
        (call_dir / "few_shot.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "skills" / "demo" / "SKILL.md").write_text("# demo", encoding="utf-8")
    (tmp_path / "memory_packs" / "pack.md").write_text("memory", encoding="utf-8")
    (tmp_path / "mechanistic_agent" / "ui" / "index.html").write_text("ok", encoding="utf-8")
    (tmp_path / "training_data" / "flower_mechanisms_100.json").write_text(
        '[{"id":"flower_000001","name":"FlowER Example 1","starting_materials":["C=O"],"products":["CO"],"n_mechanistic_steps":1,"source":"FlowER 100","verified_mechanism":{"version":"1.0.0","provisional":true,"source_refs":["https://github.com/schwallergroup/ChRIMP"],"steps":[{"step_index":1,"current_state":["C=O"],"resulting_state":["CO"],"predicted_intermediate":"CO","target_products":["CO"],"electron_pushes":[{"kind":"lone_pair","source_atom":"1","target_atom":"2","electrons":2,"notation":"lp:1>2"}],"reaction_smirks":"[O-:1].[H+:2]>>[O:1][H:2] |mech:v1;lp:1>2|","confidence":1.0,"note":"fixture"}]}}]',
        encoding="utf-8",
    )
    (tmp_path / "data" / "mechanism_examples.json").write_text(
        '[{\"id\":\"ex1\",\"name\":\"Example 1\",\"starting_materials\":[\"C=O\"],\"products\":[\"CO\"]}]',
        encoding="utf-8",
    )
    return tmp_path


def test_api_run_lifecycle_and_memory(tmp_path: Path) -> None:
    app = create_app(_prepare_base(tmp_path))
    client = TestClient(app)

    create_resp = client.post(
        "/api/runs",
        json={
            "mode": "unverified",
            "starting_materials": ["C=O"],
            "products": ["CO"],
            "model": "gpt-5",
            "max_steps": 1,
        },
    )
    assert create_resp.status_code == 200
    run_id = create_resp.json()["run_id"]

    snapshot_resp = client.get(f"/api/runs/{run_id}")
    assert snapshot_resp.status_code == 200
    assert snapshot_resp.json()["status"] == "pending"

    start_resp = client.post(f"/api/runs/{run_id}/start")
    assert start_resp.status_code == 200

    # Allow background thread at least one scheduling cycle.
    time.sleep(0.2)
    snapshot_resp = client.get(f"/api/runs/{run_id}")
    assert snapshot_resp.status_code == 200
    assert snapshot_resp.json()["status"] in {"running", "paused", "failed", "completed", "stopped"}

    memory_add = client.post(
        "/api/memory/items",
        json={
            "scope": "local",
            "key": "test-key",
            "value": {"hello": "world"},
            "source": "test",
            "tags": ["demo"],
            "active": True,
        },
    )
    assert memory_add.status_code == 200

    memory_query = client.post("/api/memory/query", json={"scope": "local", "tags": ["demo"]})
    assert memory_query.status_code == 200
    assert memory_query.json()["counts"]["total"] >= 1

    examples_resp = client.get("/api/examples")
    assert examples_resp.status_code == 200
    assert examples_resp.json()[0]["id"] == "flower_000001"

    parse_resp = client.post("/api/parse_smirks", json={"smirks": "C=O.O>>CO"})
    assert parse_resp.status_code == 200
    assert parse_resp.json()["reactants"] == ["C=O", "O"]
    assert parse_resp.json()["products"] == ["CO"]


def test_create_run_accepts_example_id_and_deterministic_template_mapping(tmp_path: Path) -> None:
    app = create_app(_prepare_base(tmp_path))
    client = TestClient(app)

    create_resp = client.post(
        "/api/runs",
        json={
            "mode": "unverified",
            "starting_materials": ["C=O"],
            "products": ["CO"],
            "model": "gpt-5",
            "max_steps": 1,
            "example_id": "flower_000001",
            "reaction_template_margin_threshold": 0.07,
        },
    )
    assert create_resp.status_code == 200
    run_id = create_resp.json()["run_id"]

    snapshot = client.get(f"/api/runs/{run_id}").json()
    assert snapshot["input_payload"]["example_id"] == "flower_000001"
    assert snapshot["config"]["reaction_template_margin_threshold"] == 0.07

    start_resp = client.post(f"/api/runs/{run_id}/start")
    assert start_resp.status_code == 200


def test_examples_include_multistep_eval_cases_with_stripped_known_steps(tmp_path: Path) -> None:
    app = create_app(_prepare_base(tmp_path))
    client = TestClient(app)

    import_resp = client.post(
        "/api/eval_sets/import_template",
        json={
            "name": "multistep_examples",
            "version": "v1",
            "cases": [
                {
                    "id": "multistep_eval_case",
                    "starting_materials": ["[CH2:1]=[CH2:2]"],
                    "products": ["[CH3:1][CH2:2][OH:3]"],
                    "verified_mechanism": {
                        "version": "1.0.0",
                        "steps": [
                            {
                                "step_index": 1,
                                "current_state": ["[CH2:1]=[CH2:2]"],
                                "resulting_state": ["[CH3:1][CH2:2][Br:3]"],
                                "predicted_intermediate": "[CH3:1][CH2:2][Br:3]",
                                "target_products": ["[CH3:1][CH2:2][OH:3]"],
                                "reaction_smirks": "[CH2:1]=[CH2:2].[Br:3][Br:4]>>[CH3:1][CH2:2][Br:3].[Br-:4]",
                            },
                            {
                                "step_index": 2,
                                "current_state": ["[CH3:1][CH2:2][Br:3]"],
                                "resulting_state": ["[CH3:1][CH2:2][OH:3]"],
                                "predicted_intermediate": "[CH3:1][CH2:2][OH:3]",
                                "target_products": ["[CH3:1][CH2:2][OH:3]"],
                                "reaction_smirks": "[CH3:1][CH2:2][Br:3].[OH-:4]>>[CH3:1][CH2:2][OH:4].[Br-:3]",
                            },
                        ],
                    },
                }
            ],
        },
    )
    assert import_resp.status_code == 200

    examples_resp = client.get("/api/examples")
    assert examples_resp.status_code == 200
    rows = examples_resp.json()
    row = next(item for item in rows if item["id"] == "multistep_eval_case")

    assert row["n_mechanistic_steps"] == 2
    assert row["name"] == "alkene -> alcohol"
    known = row["known_mechanism"]
    assert known["final_products"] == ["CCO"]
    assert len(known["steps"]) == 2
    assert known["steps"][0]["current_state"] == ["C=C"]
    assert known["steps"][0]["resulting_state"] == ["CCBr"]
    assert known["steps"][1]["current_state"] == ["CCBr"]
    assert known["steps"][1]["resulting_state"] == ["CCO"]


def test_create_run_and_preview_accept_model_name_and_thinking_level(tmp_path: Path) -> None:
    base = _prepare_base(tmp_path)
    app = create_app(base)
    client = TestClient(app)

    preview_resp = client.get("/api/catalog/preview_step_models?model_name=gpt-5.1&thinking_level=high")
    assert preview_resp.status_code == 200
    preview = preview_resp.json()["step_models"]
    assert preview["mechanism_synthesis"] == "gpt-5.1"
    assert preview["initial_conditions"] == "gpt-5.1"

    create_resp = client.post(
        "/api/runs",
        json={
            "mode": "unverified",
            "starting_materials": ["C=O"],
            "products": ["CO"],
            "model_name": "gpt-5.1",
            "thinking_level": "high",
            "max_steps": 1,
        },
    )
    assert create_resp.status_code == 200
    run_id = create_resp.json()["run_id"]

    store = RunStore(base / "data" / "mechanistic.db")
    run_row = store.get_run_row(run_id)
    assert run_row is not None
    assert run_row["config"]["model_name"] == "gpt-5.1"
    assert run_row["config"]["thinking_level"] == "high"
    assert run_row["config"]["model_family"] == "openai"


def test_verified_mode_requires_acceptance_to_complete(tmp_path: Path) -> None:
    base = _prepare_base(tmp_path)
    app = create_app(base)
    client = TestClient(app)

    create_resp = client.post(
        "/api/runs",
        json={
            "mode": "verified",
            "starting_materials": ["C=O"],
            "products": ["CO"],
            "model": "gpt-5",
            "max_steps": 1,
        },
    )
    assert create_resp.status_code == 200
    run_id = create_resp.json()["run_id"]

    store = RunStore(base / "data" / "mechanistic.db")
    store.set_run_status(run_id, "running")
    store.record_step_output(
        run_id=run_id,
        step_name="mechanism_synthesis",
        attempt=1,
        model="gpt-5",
        reasoning_level="low",
        tool_name="predict_mechanistic_step",
        output={"contains_target_product": True},
        validation={"passed": True, "checks": []},
        accepted_bool=None,
    )

    verify_resp = client.post(
        f"/api/runs/{run_id}/steps/mechanism_synthesis/verify",
        json={"decision": "accept", "attempt": 1},
    )
    assert verify_resp.status_code == 200
    assert verify_resp.json()["status"] == "completed"


def test_eval_leaderboard_and_few_shot_endpoints(tmp_path: Path) -> None:
    base = _prepare_base(tmp_path)
    app = create_app(base)
    client = TestClient(app)
    store = RunStore(base / "data" / "mechanistic.db")

    eval_set_id = store.add_eval_set(
        name="default_examples",
        version="v1",
        source_path=str(base / "data" / "mechanism_examples.json"),
        sha256=None,
        cases=[
            {
                "case_id": "case-1",
                "input": {"starting_materials": ["C=O"], "products": ["CO"]},
                "expected": {"products": ["CO"]},
                "tags": ["default"],
            }
        ],
        active=True,
    )
    eval_run_id = store.create_eval_run(
        eval_set_id=eval_set_id,
        run_group_name="smoke",
        model="gpt-5-mini",
        model_name="gpt-5-mini",
        model_family="openai",
        thinking_level="low",
        harness_bundle_hash="bundle-a",
        status="completed",
    )
    store.record_eval_run_result(
        eval_run_id=eval_run_id,
        case_id="case-1",
        run_id=None,
        score=0.75,
        passed=True,
        cost={"total_cost": 0.0},
        latency_ms=1000.0,
        summary={"ok": True},
    )

    leaderboard_resp = client.get(f"/api/evals/leaderboard?eval_set_id={eval_set_id}")
    assert leaderboard_resp.status_code == 200
    items = leaderboard_resp.json()["items"]
    assert items
    assert items[0]["eval_run_id"] == eval_run_id
    assert items[0]["model_name"] == "gpt-5-mini"
    assert items[0]["thinking_level"] == "low"

    trace_id = store.add_trace_record(
        step_name="mechanism_synthesis",
        trace={"tool_name": "predict_mechanistic_step", "output": {"contains_target_product": True}},
        source="baseline",
        score=1.0,
    )
    few_shot_resp = client.post("/api/few_shot/from_trace", json={"trace_id": trace_id, "approved": True})
    assert few_shot_resp.status_code == 200
    list_few_shot = client.get("/api/few_shot?step_name=mechanism_synthesis")
    assert list_few_shot.status_code == 200
    assert list_few_shot.json()["items"]


def test_catalog_models_and_families_include_single_model_metadata(tmp_path: Path) -> None:
    app = create_app(_prepare_base(tmp_path))
    client = TestClient(app)

    models_resp = client.get("/api/catalog/models")
    assert models_resp.status_code == 200
    models = models_resp.json()
    minimax = next(item for item in models if item["id"] == "minimax/minimax-m2.1")
    assert minimax["family"] == "minimax"
    assert minimax["provider"] == "openrouter"

    families_resp = client.get("/api/catalog/families")
    assert families_resp.status_code == 200
    family_ids = [item["id"] for item in families_resp.json()]
    assert "minimax" in family_ids


def test_resume_and_verified_step_submission(tmp_path: Path) -> None:
    base = _prepare_base(tmp_path)
    app = create_app(base)
    client = TestClient(app)

    create_resp = client.post(
        "/api/runs",
        json={
            "mode": "verified",
            "starting_materials": ["C=O", "OCCO"],
            "products": ["C1OCOC1"],
            "model": "gpt-5",
            "max_steps": 3,
        },
    )
    assert create_resp.status_code == 200
    run_id = create_resp.json()["run_id"]

    start_resp = client.post(f"/api/runs/{run_id}/start")
    assert start_resp.status_code == 200
    time.sleep(0.2)

    step_submit = client.post(
        f"/api/runs/{run_id}/mechanism_steps",
        json={
            "step_index": 1,
            "current_state": ["C=O", "OCCO"],
            "resulting_state": ["C1OCOC1", "O"],
            "predicted_intermediate": "C1OCOC1",
            "target_products": ["C1OCOC1"],
            "electron_pushes": [{"start_atom": "1", "end_atom": "2", "electrons": 2}],
            "reaction_smirks": "C=O.OCCO>>C1OCOC1.O |dbe:1-2:+2;1-1:-2|",
        },
    )
    assert step_submit.status_code == 200
    body = step_submit.json()
    assert body["run_id"] == run_id
    assert body["status"] in {"running", "completed"}

    store = RunStore(base / "data" / "mechanistic.db")
    annotations = store.list_arrow_push_annotations(run_id)
    assert annotations
    assert annotations[-1]["source"] == "verified_submission"
    assert str((annotations[-1]["prediction"] or {}).get("annotation_suffix", "")).startswith("|aps:v1;")

    snapshot_resp = client.get(f"/api/runs/{run_id}")
    assert snapshot_resp.status_code == 200
    snapshot = snapshot_resp.json()
    assert "arrow_push_annotations" not in snapshot
    assert all("aps:v1;" not in json.dumps(row) for row in snapshot.get("step_outputs", []))
    verbose_snapshot = client.get(f"/api/runs/{run_id}?verbose=true").json()
    assert all("aps:v1;" not in json.dumps(ev) for ev in verbose_snapshot.get("events", []))


def test_trace_approval_and_curation_export(tmp_path: Path) -> None:
    base = _prepare_base(tmp_path)
    app = create_app(base)
    client = TestClient(app)
    store = RunStore(base / "data" / "mechanistic.db")

    trace_id = store.add_trace_record(
        step_name="mechanism_synthesis",
        trace={"output": {"contains_target_product": True}},
        source="baseline",
        score=1.0,
    )
    approve_resp = client.post(
        f"/api/traces/{trace_id}/approve",
        json={"approved": True, "label": "baseline", "notes": "approved in test"},
    )
    assert approve_resp.status_code == 200
    assert approve_resp.json()["item"]["approved_bool"] is True

    export_resp = client.post("/api/curation/export", json={"eval_set_id": "default", "created_by": "pytest"})
    assert export_resp.status_code == 200
    export_body = export_resp.json()
    assert export_body["id"]
    list_resp = client.get("/api/curation/exports")
    assert list_resp.status_code == 200
    assert list_resp.json()["items"]


def test_trace_evidence_export_requires_linkage_and_exports_when_valid(tmp_path: Path) -> None:
    base = _prepare_base(tmp_path)
    app = create_app(base)
    client = TestClient(app)
    store = RunStore(base / "data" / "mechanistic.db")

    bad_trace_id = store.add_trace_record(
        step_name="initial_conditions",
        trace={"output": {"environment": "acidic"}},
        source="run",
        approved=True,
    )
    bad_resp = client.post("/api/traces/export_evidence", json={"trace_ids": [bad_trace_id]})
    assert bad_resp.status_code == 400

    prompt_ids = store.upsert_prompt_versions(
        [
            {
                "name": "assess_initial_conditions",
                "call_name": "assess_initial_conditions",
                "step": "initial_conditions",
                "version": "bundle",
                "path": "prompt_versions/calls/assess_initial_conditions/base.md",
                "sha256": "bundle-sha",
                "prompt_bundle_sha256": "bundle-sha",
                "shared_base_sha256": "shared",
                "call_base_sha256": "base",
                "few_shot_sha256": "few",
                "template": "template",
            }
        ]
    )
    model_version_id = store.upsert_model_version(model_name="gpt-5", reasoning_level="lowest")
    good_trace_id = store.add_trace_record(
        step_name="initial_conditions",
        trace={"output": {"environment": "acidic"}},
        source="run",
        model="gpt-5",
        reasoning_level="lowest",
        prompt_version_id=prompt_ids.get("initial_conditions"),
        model_version_id=model_version_id,
        approved=True,
    )
    good_resp = client.post("/api/traces/export_evidence", json={"trace_ids": [good_trace_id]})
    assert good_resp.status_code == 200
    payload = good_resp.json()
    assert payload["items"]


def test_convert_inputs_endpoint(tmp_path: Path) -> None:
    app = create_app(_prepare_base(tmp_path))
    client = TestClient(app)
    resp = client.post(
        "/api/convert_inputs",
        json={"starting_materials": ["C=O", "CC(=O)O"], "products": ["CCOC(=O)C"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "starting_materials" in data
    assert "canonical_starting_materials" in data
    assert "all_converted" in data
    assert data["all_converted"] is True
    assert len(data["starting_materials"]) == 2
    assert len(data["products"]) == 1
    for item in data["starting_materials"]:
        assert item["success"] is True
        assert item["canonical_smiles"] is not None


def test_convert_inputs_with_invalid_smiles(tmp_path: Path) -> None:
    app = create_app(_prepare_base(tmp_path))
    client = TestClient(app)
    resp = client.post(
        "/api/convert_inputs",
        json={"starting_materials": ["C=O", "INVALID!!!"], "products": ["CO"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["all_converted"] is False
    assert data["starting_materials"][1]["success"] is False


def test_import_template_eval_set(tmp_path: Path) -> None:
    app = create_app(_prepare_base(tmp_path))
    client = TestClient(app)
    resp = client.post(
        "/api/eval_sets/import_template",
        json={
            "name": "test_template",
            "version": "v1",
            "auto_convert": False,
            "cases": [
                {
                    "id": "test_case_1",
                    "starting_materials": ["C=O", "OCCO"],
                    "products": ["C1OCOC1"],
                    "temperature_celsius": 40.0,
                    "ph": 3.5,
                }
            ],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["case_count"] == 1
    assert data["eval_set_id"]
    assert data["name"] == "test_template"

    # Verify cases are retrievable.
    cases_resp = client.get(f"/api/eval_sets/{data['eval_set_id']}/cases")
    assert cases_resp.status_code == 200
    items = cases_resp.json()["items"]
    assert len(items) == 1
    assert items[0]["case_id"] == "test_case_1"


def test_import_template_with_verified_mechanism(tmp_path: Path) -> None:
    app = create_app(_prepare_base(tmp_path))
    client = TestClient(app)
    resp = client.post(
        "/api/eval_sets/import_template",
        json={
            "name": "verified_test",
            "version": "v1",
            "auto_convert": False,
            "cases": [
                {
                    "id": "verified_case_1",
                    "starting_materials": ["C=O", "OCCO"],
                    "products": ["C1OCOC1"],
                    "verified_mechanism": {
                        "version": "0.1.0",
                        "provisional": True,
                        "source_refs": ["https://example.com"],
                        "steps": [
                            {
                                "step_index": 1,
                                "current_state": ["C=O", "OCCO"],
                                "resulting_state": ["C1OCOC1"],
                                "predicted_intermediate": "C1OCOC1",
                                "target_products": ["C1OCOC1"],
                                "electron_pushes": [],
                                "reaction_smirks": "C=O.OCCO>>C1OCOC1",
                                "confidence": 0.9,
                            }
                        ],
                    },
                }
            ],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["case_count"] == 1

    cases_resp = client.get(f"/api/eval_sets/{data['eval_set_id']}/cases")
    items = cases_resp.json()["items"]
    expected = items[0].get("expected") or {}
    assert "verified_mechanism" in expected


def test_import_template_empty_cases_returns_400(tmp_path: Path) -> None:
    app = create_app(_prepare_base(tmp_path))
    client = TestClient(app)
    resp = client.post(
        "/api/eval_sets/import_template",
        json={"name": "empty", "version": "v1", "cases": []},
    )
    assert resp.status_code == 400


def test_eval_tiers_endpoint_and_case_selection_precedence(tmp_path: Path) -> None:
    base = _prepare_base(tmp_path)
    training = base / "training_data"
    training.mkdir(parents=True, exist_ok=True)
    (training / "eval_tiers.json").write_text(
        json.dumps(
            {
                "_meta": {"difficulty_criteria": {}, "source": "pytest"},
                "easy": ["case_a"],
                "medium": ["case_b"],
                "hard": ["case_c"],
            }
        ),
        encoding="utf-8",
    )

    app = create_app(base)
    client = TestClient(app)
    store = RunStore(base / "data" / "mechanistic.db")
    eval_set_id = store.add_eval_set(
        name="hb350_test",
        version="v1",
        source_path=None,
        sha256=None,
        cases=[
            {"case_id": "case_a", "input": {"starting_materials": [], "products": []}, "expected": {"products": []}},
            {"case_id": "case_b", "input": {"starting_materials": [], "products": []}, "expected": {"products": []}},
            {"case_id": "case_c", "input": {"starting_materials": [], "products": []}, "expected": {"products": []}},
        ],
        active=True,
    )

    tiers_resp = client.get(f"/api/evals/tiers?eval_set_id={eval_set_id}")
    assert tiers_resp.status_code == 200
    tiers_payload = tiers_resp.json()
    assert tiers_payload["tiers"]["easy"] == ["case_a"]

    run_resp = client.post(
        "/api/evals/runset",
        json={
            "eval_set_id": eval_set_id,
            "case_ids": ["case_b"],
            "tier_name": "easy",
            "async_mode": False,
            "max_cases": 25,
        },
    )
    assert run_resp.status_code == 200
    eval_run_id = run_resp.json()["eval_run_id"]
    results_resp = client.get(f"/api/evals/runs/{eval_run_id}/results")
    assert results_resp.status_code == 200
    result_items = results_resp.json()["items"]
    assert len(result_items) == 1
    assert result_items[0]["case_id"] == "case_b"


def test_flow_template_includes_reaction_type_mapping_node(tmp_path: Path) -> None:
    app = create_app(_prepare_base(tmp_path))
    client = TestClient(app)
    resp = client.get("/api/catalog/flow_template")
    assert resp.status_code == 200
    payload = resp.json()
    nodes = payload.get("nodes") or []
    node_ids = {str(item.get("id") or "") for item in nodes if isinstance(item, dict)}
    assert "reaction_type_mapping" in node_ids


def test_ralph_vote_endpoints_accept_advisory_votes(tmp_path: Path) -> None:
    app = create_app(_prepare_base(tmp_path))
    client = TestClient(app)

    create_resp = client.post(
        "/api/runs",
        json={
            "mode": "unverified",
            "orchestration_mode": "ralph",
            "starting_materials": ["C=O"],
            "products": ["CO"],
            "model": "gpt-5",
            "max_steps": 1,
            "ralph": {"max_iterations": 1},
        },
    )
    assert create_resp.status_code == 200
    run_id = create_resp.json()["run_id"]

    vote_resp = client.post(
        f"/api/runs/{run_id}/votes",
        json={
            "attempt_index": 1,
            "step_index": 1,
            "candidate_a": {"smiles": "CCO"},
            "candidate_b": {"smiles": "CCBr"},
            "vote": "A",
            "confidence": 0.8,
            "source": "test",
        },
    )
    assert vote_resp.status_code == 200
    assert vote_resp.json().get("vote_id")

    list_resp = client.get(f"/api/runs/{run_id}/votes")
    assert list_resp.status_code == 200
    items = list_resp.json()["items"]
    assert len(items) == 1
    assert items[0]["vote"] == "A"


def test_holdout_eval_set_is_hidden_and_restricted_to_official_endpoints(tmp_path: Path) -> None:
    base = _prepare_base(tmp_path)
    app = create_app(base)
    client = TestClient(app)
    store = RunStore(base / "data" / "mechanistic.db")

    general_eval_set_id = store.add_eval_set(
        name="general_visible_eval",
        version="v1",
        source_path="training_data/eval_set.json",
        sha256=None,
        cases=[
            {
                "case_id": "general_case",
                "input": {"starting_materials": ["C=O"], "products": ["CO"]},
                "expected": {"products": ["CO"], "n_mechanistic_steps": 1},
                "tags": ["general"],
            }
        ],
        active=True,
        purpose="general",
        exposed_in_ui=True,
    )
    holdout_eval_set_id = store.add_eval_set(
        name="official_holdout_eval",
        version="v1",
        source_path="training_data/leaderboard_holdout/eval_set_holdout.json",
        sha256=None,
        cases=[
            {
                "case_id": "holdout_case",
                "input": {"starting_materials": ["CCO"], "products": ["CC=O"]},
                "expected": {"products": ["CC=O"], "n_mechanistic_steps": 2},
                "tags": ["leaderboard_holdout"],
            }
        ],
        active=True,
        purpose="leaderboard_holdout",
        exposed_in_ui=False,
    )
    assert general_eval_set_id
    assert holdout_eval_set_id

    visible_sets = client.get("/api/eval_sets")
    assert visible_sets.status_code == 200
    visible_ids = {str(item.get("id") or "") for item in visible_sets.json()["items"]}
    assert general_eval_set_id in visible_ids
    assert holdout_eval_set_id not in visible_ids

    all_sets = client.get("/api/eval_sets?include_hidden=true")
    assert all_sets.status_code == 200
    all_ids = {str(item.get("id") or "") for item in all_sets.json()["items"]}
    assert holdout_eval_set_id in all_ids

    examples_resp = client.get("/api/examples")
    assert examples_resp.status_code == 200
    example_ids = {str(row.get("id") or "") for row in examples_resp.json()}
    assert "holdout_case" not in example_ids

    user_runset_resp = client.post(
        "/api/evals/runset",
        json={
            "eval_set_id": holdout_eval_set_id,
            "async_mode": True,
            "max_cases": 1,
            "mode": "unverified",
        },
    )
    assert user_runset_resp.status_code == 403

    baseline_resp = client.post(
        "/api/evals/baseline-runset",
        json={
            "eval_set_id": holdout_eval_set_id,
            "async_mode": True,
            "max_cases": 1,
        },
    )
    assert baseline_resp.status_code == 403

    official_runset_resp = client.post(
        "/api/evals/official-runset",
        json={
            "eval_set_id": holdout_eval_set_id,
            "async_mode": True,
            "max_cases": 1,
            "mode": "unverified",
        },
    )
    assert official_runset_resp.status_code == 200

    official_lb = client.get("/api/evals/leaderboard/official")
    assert official_lb.status_code == 200
    assert official_lb.json()["eval_set_id"] == holdout_eval_set_id
