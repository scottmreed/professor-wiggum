from __future__ import annotations

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("starlette")

from fastapi.testclient import TestClient

from mechanistic_agent.api.app import _MOLECULE_IMAGE_CACHE, _render_molecule, create_app


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
        '[{"id":"flower_000001","name":"FlowER Example 1","starting_materials":["C=O"],"products":["CO"],"n_mechanistic_steps":1}]',
        encoding="utf-8",
    )
    return tmp_path


def test_render_molecule_cache_keys_include_atom_number_flag() -> None:
    _MOLECULE_IMAGE_CACHE.clear()

    plain = _render_molecule("CCO", show_atom_numbers=False)
    numbered = _render_molecule("CCO", show_atom_numbers=True)

    assert ("CCO", False) in _MOLECULE_IMAGE_CACHE
    assert ("CCO", True) in _MOLECULE_IMAGE_CACHE
    assert plain["smiles"] == "CCO"
    assert numbered["smiles"] == "CCO"


def test_render_molecules_endpoint_accepts_atom_number_flag(tmp_path: Path) -> None:
    app = create_app(_prepare_base(tmp_path))
    client = TestClient(app)

    response = client.post("/api/molecules/render", json={"smiles": ["CCO"], "show_atom_numbers": True})

    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) == 1
    assert items[0]["smiles"] == "CCO"


def test_render_molecules_endpoint_defaults_atom_number_flag_to_false(tmp_path: Path) -> None:
    app = create_app(_prepare_base(tmp_path))
    client = TestClient(app)

    response = client.post("/api/molecules/render", json={"smiles": ["CCO"]})

    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) == 1
    assert items[0]["smiles"] == "CCO"


def test_render_molecules_invalid_smiles_returns_payload(tmp_path: Path) -> None:
    app = create_app(_prepare_base(tmp_path))
    client = TestClient(app)

    response = client.post("/api/molecules/render", json={"smiles": ["not_a_smiles!!!"]})

    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) == 1
    assert items[0]["smiles"] == "not_a_smiles!!!"
    assert "image_data" in items[0]
