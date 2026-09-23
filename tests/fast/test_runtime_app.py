"""Runtime-only product API (Observatory PRD §21–§25, M3 first slice).

``create_runtime_app`` exposes only what a product needs to run, watch and
replay a mechanism prediction, behind server-to-server bearer auth. Research,
evaluation, leaderboard, harness-editing and curriculum routes are absent by
construction; the app fails closed when no token is configured.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("starlette")
pytest.importorskip("rdkit")

from fastapi.testclient import TestClient  # noqa: E402

from mechanistic_agent.api.runtime_app import (  # noqa: E402
    RUNTIME_ALLOWED_PATHS,
    RUNTIME_TOKEN_ENV,
    create_runtime_app,
)
from tests.fast.test_api_runtime import _prepare_base  # noqa: E402

TOKEN = "runtime-test-token-not-a-secret"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, token: str | None = TOKEN) -> TestClient:
    if token is None:
        monkeypatch.delenv(RUNTIME_TOKEN_ENV, raising=False)
    else:
        monkeypatch.setenv(RUNTIME_TOKEN_ENV, token)
    return TestClient(create_runtime_app(_prepare_base(tmp_path)))


def test_health_is_open_and_everything_else_needs_a_bearer_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch)
    assert client.get("/healthz").status_code == 200
    assert client.get("/healthz").json()["status"] == "ok"
    assert client.post("/api/runs", json={"mode": "unverified", "starting_materials": ["C=O"], "products": ["CO"], "model": "gpt-5", "max_steps": 1}).status_code == 401
    assert client.post("/api/runs", headers={"Authorization": "Bearer wrong"}, json={"mode": "unverified", "starting_materials": ["C=O"], "products": ["CO"], "model": "gpt-5", "max_steps": 1}).status_code == 401
    assert client.get("/v1/mechanism/version").status_code == 401


def test_runtime_fails_closed_without_a_configured_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch, token=None)
    assert client.get("/healthz").status_code == 200
    assert client.get("/healthz").json()["auth_configured"] is False
    resp = client.get("/v1/mechanism/version", headers=AUTH)
    assert resp.status_code == 503
    assert "token" in resp.json()["detail"].lower()


def test_product_run_lifecycle_and_projections_are_exposed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch)
    created = client.post("/api/runs", headers=AUTH, json={"mode": "verified", "starting_materials": ["CCBr", "[Cl-]"], "products": ["CCCl", "[Br-]"], "model": "gpt-5", "max_steps": 1})
    assert created.status_code == 200, created.text
    run_id = created.json()["run_id"]
    assert client.get(f"/api/runs/{run_id}", headers=AUTH).status_code == 200
    assert client.get(f"/api/runs/{run_id}/observatory", headers=AUTH).json()["schema_version"] == "mechanism_observatory.v1"
    assert client.get(f"/api/runs/{run_id}/flow", headers=AUTH).status_code == 200
    assert client.post("/api/molecules/render", headers=AUTH, json={"smiles": ["CCBr"]}).status_code == 200
    step = client.post(f"/api/runs/{run_id}/mechanism_steps", headers=AUTH, json={
        "step_index": 1, "current_state": ["[CH3:1][CH2:2][Br:3]", "[Cl-:4]"], "resulting_state": ["[CH3:1][CH2:2][Cl:4]", "[Br-:3]"],
        "predicted_intermediate": "[CH3:1][CH2:2][Cl:4]", "target_products": ["[CH3:1][CH2:2][Cl:4]", "[Br-:3]"],
        "electron_pushes": [{"kind": "lone_pair", "source_atom": "4", "target_atom": "2", "electrons": 2},
                            {"kind": "sigma_bond", "source_bond": ["2", "3"], "through_atom": "3", "target_atom": "3", "electrons": 2}],
        "reaction_smirks": "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] |mech:v1;lp:4>2;sigma:2-3>3|"})
    assert step.status_code == 200, step.text
    assert client.post(f"/api/runs/{run_id}/stop", headers=AUTH).status_code in {200, 400}


def test_research_and_mutation_routes_are_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch)
    for path in ("/api/evals/leaderboard", "/api/evals/leaderboard/official", "/api/examples", "/api/traces",
                 "/api/harness/config", "/api/harness/configs", "/api/memory", "/api/curation/exports", "/ui/app.js", "/"):
        assert client.get(path, headers=AUTH).status_code == 404, path
    for path in ("/api/evals/runset", "/api/evals/official-runset", "/api/harness/config", "/api/curation/export",
                 "/api/runs/x/evaluate", "/api/runs/x/harness/apply", "/api/runs/x/votes", "/api/eval_sets/import_template"):
        assert client.post(path, headers=AUTH, json={}).status_code == 404, path


def test_allow_list_matches_the_mounted_routes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = create_runtime_app(_prepare_base(tmp_path))
    mounted = {getattr(r, "path", None) for r in app.routes}
    for path in RUNTIME_ALLOWED_PATHS:
        assert path in mounted, path
    assert "/api/evals/leaderboard" not in mounted
    assert "/healthz" in mounted and "/v1/mechanism/version" in mounted


def test_version_manifest_identifies_the_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch)
    manifest = client.get("/v1/mechanism/version", headers=AUTH).json()
    assert manifest["schema_version"] == "runtime_manifest.v1"
    assert manifest["runtime_version"]
    assert set(manifest["hashes"]) >= {"prompt_bundle_hash", "skill_bundle_hash", "harness_bundle_hash"}
    assert manifest["harness_name"] == "default"
    assert manifest["event_schema_version"] == "mechanism_observatory_event.v1"
    assert manifest["observatory_schema_version"] == "mechanism_observatory.v1"
    assert manifest["reaction_focus_version"] == "reaction_focus.v1"
    assert manifest["be_convention"] == "ugi_flower_kekule_v1"
    assert isinstance(manifest["git_sha"], (str, type(None)))
    assert manifest["excluded_surfaces"] == ["curation", "evals", "examples", "harness_editing", "leaderboard", "memory", "ralph", "traces", "ui"]
