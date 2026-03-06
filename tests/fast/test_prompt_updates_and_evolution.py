from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from mechanistic_agent.api.app import create_app
from mechanistic_agent.core.db import RunStore
from mechanistic_agent.prompt_assets import replace_prompt_in_skill_md


def _load_evolve_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "evolve_harness.py"
    spec = importlib.util.spec_from_file_location("evolve_harness_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prepare_runtime_base(tmp_path: Path) -> Path:
    (tmp_path / "skills" / "mechanistic" / "base_system").mkdir(parents=True)
    (tmp_path / "skills" / "mechanistic" / "base_system" / "SKILL.md").write_text(
        "---\nkind: shared_base\ncall_name: base_system\n---\n<!-- PROMPT_START -->\nbase\n<!-- PROMPT_END -->\n",
        encoding="utf-8",
    )
    (tmp_path / "skills" / "mechanistic" / "propose_mechanism_step").mkdir(parents=True)
    (tmp_path / "skills" / "mechanistic" / "propose_mechanism_step" / "SKILL.md").write_text(
        "---\nkind: llm\ncall_name: propose_mechanism_step\nsteps: [mechanism_step_proposal]\n---\n"
        "# Skill\n\n"
        "<!-- PROMPT_START -->\nOriginal prompt.\n<!-- PROMPT_END -->\n\n"
        "## Docs\n\nKeep docs.\n",
        encoding="utf-8",
    )
    (tmp_path / "skills" / "mechanistic" / "propose_mechanism_step" / "few_shot.jsonl").write_text(
        "",
        encoding="utf-8",
    )
    (tmp_path / "memory_packs").mkdir()
    (tmp_path / "memory_packs" / "pack.md").write_text("memory", encoding="utf-8")
    (tmp_path / "mechanistic_agent" / "ui").mkdir(parents=True)
    (tmp_path / "mechanistic_agent" / "ui" / "index.html").write_text("ok", encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "mechanism_examples.json").write_text("[]", encoding="utf-8")
    (tmp_path / "harness_versions" / "default").mkdir(parents=True)
    (tmp_path / "harness_versions" / "default" / "harness.json").write_text(
        '{"schema_version":"2.0","modules":[],"loop_module":"mechanism_step_proposal","metadata":{"version":"test"}}',
        encoding="utf-8",
    )
    return tmp_path


def test_replace_prompt_in_skill_md_only_changes_prompt_block() -> None:
    original = """---
skill_type: mechanistic
call_name: propose_mechanism_step
kind: llm
---

# Proposal Skill

Docs stay here.

<!-- PROMPT_START -->
Original prompt text.
<!-- PROMPT_END -->

## Notes

Leave this alone.
"""
    updated = replace_prompt_in_skill_md(
        original,
        prompt_text="New prompt text.",
        append_mode=False,
    )

    assert "New prompt text." in updated
    assert "Original prompt text." not in updated
    assert "Docs stay here." in updated
    assert "Leave this alone." in updated
    assert updated.count("<!-- PROMPT_START -->") == 1
    assert updated.count("<!-- PROMPT_END -->") == 1


def test_apply_harness_update_dry_run_returns_diff_without_mutating_skill(tmp_path: Path) -> None:
    base = _prepare_runtime_base(tmp_path)
    app = create_app(base)
    client = TestClient(app)
    store = RunStore(base / "data" / "mechanistic.db")
    run_id = store.create_run(
        mode="unverified",
        input_payload={"starting_materials": ["C=O"], "products": ["CO"]},
        config={"dry_run": True},
        prompt_bundle_hash="",
        skill_bundle_hash="",
        memory_bundle_hash="",
    )

    response = client.post(
        f"/api/runs/{run_id}/harness/apply",
        json={
            "call_name": "propose_mechanism_step",
            "component": "base",
            "recommendation": "Use stricter formatting.",
            "append_mode": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    skill_path = base / "skills" / "mechanistic" / "propose_mechanism_step" / "SKILL.md"
    assert "diff" in payload
    assert payload["dry_run"] is True
    assert "Use stricter formatting." not in skill_path.read_text(encoding="utf-8")
    assert (base / payload["diff_path"]).exists()


def test_apply_harness_update_non_dry_run_updates_only_prompt_block(tmp_path: Path) -> None:
    base = _prepare_runtime_base(tmp_path)
    app = create_app(base)
    client = TestClient(app)
    store = RunStore(base / "data" / "mechanistic.db")
    run_id = store.create_run(
        mode="unverified",
        input_payload={"starting_materials": ["C=O"], "products": ["CO"]},
        config={"dry_run": False},
        prompt_bundle_hash="",
        skill_bundle_hash="",
        memory_bundle_hash="",
    )

    response = client.post(
        f"/api/runs/{run_id}/harness/apply",
        json={
            "call_name": "propose_mechanism_step",
            "component": "base",
            "recommendation": "Use stricter formatting.",
            "append_mode": False,
        },
    )

    assert response.status_code == 200
    updated = (base / "skills" / "mechanistic" / "propose_mechanism_step" / "SKILL.md").read_text(encoding="utf-8")
    assert "Use stricter formatting." in updated
    assert "Keep docs." in updated
    assert updated.count("<!-- PROMPT_START -->") == 1
    assert updated.count("<!-- PROMPT_END -->") == 1


def test_setup_workspace_copies_runtime_assets_for_dry_run(tmp_path: Path) -> None:
    evolve = _load_evolve_module()
    (tmp_path / "skills" / "mechanistic" / "base_system").mkdir(parents=True)
    (tmp_path / "skills" / "mechanistic" / "base_system" / "SKILL.md").write_text(
        "<!-- PROMPT_START -->\nbase\n<!-- PROMPT_END -->\n",
        encoding="utf-8",
    )
    (tmp_path / "skills" / "mechanistic" / "propose_mechanism_step").mkdir(parents=True)
    (tmp_path / "skills" / "mechanistic" / "propose_mechanism_step" / "SKILL.md").write_text(
        "<!-- PROMPT_START -->\ncall\n<!-- PROMPT_END -->\n",
        encoding="utf-8",
    )
    (tmp_path / "skills" / "mechanistic" / "propose_mechanism_step" / "few_shot.jsonl").write_text(
        "",
        encoding="utf-8",
    )
    (tmp_path / "memory_packs").mkdir()
    (tmp_path / "memory_packs" / "pack.md").write_text("memory", encoding="utf-8")
    (tmp_path / "harness_versions" / "default").mkdir(parents=True)
    (tmp_path / "harness_versions" / "default" / "harness.json").write_text("{}", encoding="utf-8")
    (tmp_path / "training_data").mkdir()
    (tmp_path / "training_data" / "reaction_type_templates.json").write_text("{}", encoding="utf-8")
    store = RunStore(tmp_path / "data" / "mechanistic.db")
    store.list_eval_sets()

    workspace = evolve.setup_workspace(tmp_path, True)

    assert (workspace / "skills" / "mechanistic" / "propose_mechanism_step" / "SKILL.md").exists()
    assert (workspace / "memory_packs" / "pack.md").exists()
    assert (workspace / "harness_versions" / "default" / "harness.json").exists()
    assert (workspace / "training_data" / "reaction_type_templates.json").exists()
    assert (workspace / "data" / "mechanistic.db").exists()
    assert (workspace / "traces" / "runs").exists()


def test_mine_and_apply_examples_require_structured_outputs_and_keep_dry_run_unapproved(tmp_path: Path) -> None:
    evolve = _load_evolve_module()
    (tmp_path / "skills" / "mechanistic" / "predict_missing_reagents").mkdir(parents=True)
    (tmp_path / "skills" / "mechanistic" / "predict_missing_reagents" / "few_shot.jsonl").write_text(
        "",
        encoding="utf-8",
    )
    store = RunStore(tmp_path / "data" / "mechanistic.db")
    store.list_eval_sets()

    config = evolve.EvolutionConfig(model_name="gpt-5.2", mining_score_threshold=0.5)
    mined = evolve.mine_few_shots(
        [
            {
                "score": 0.9,
                "passed": True,
                "run_status": "completed",
                "graded_details": {"final_product_reached": True},
                "subagent_scores": {
                    "missing_reagents": {"quality_score": 0.9},
                },
                "step_outputs": [
                    {
                        "step_name": "missing_reagents",
                        "output": {
                            "status": "success",
                            "missing_reactants": ["O"],
                            "schema_validation": {"status": "ok", "source": "tool_call"},
                        },
                    },
                    {
                        "step_name": "missing_reagents",
                        "output": {
                            "status": "success",
                            "missing_reactants": ["Cl"],
                            "schema_validation": {"status": "fallback", "source": "tool_call"},
                        },
                    },
                ],
                "input_payload": {"starting_materials": ["C=O"], "products": ["CO"]},
            }
        ],
        config,
        {"predict_missing_reagents": set()},
        {"predict_missing_reagents": 0.0},
    )

    assert list(mined) == ["predict_missing_reagents"]
    assert len(mined["predict_missing_reagents"]) == 1

    counts = evolve.apply_mined_examples(
        mined,
        store,
        tmp_path,
        tmp_path,
        True,
        "tier_0",
        {"case-1": 0.9},
    )

    assert counts["predict_missing_reagents"] == 1
    rows = store.list_few_shot_examples(step_name="missing_reagents", approved_only=False)
    assert len(rows) == 1
    assert rows[0]["approved_bool"] is False


def test_mine_few_shots_requires_score_at_least_current_best(tmp_path: Path) -> None:
    evolve = _load_evolve_module()
    call_dir = tmp_path / "skills" / "mechanistic" / "predict_missing_reagents"
    call_dir.mkdir(parents=True)
    (call_dir / "few_shot.jsonl").write_text(
        '{"input":"stored","output":"{\\"missing_reactants\\":[\\"O\\"],\\"missing_products\\":[]}","score":0.8}\n',
        encoding="utf-8",
    )

    config = evolve.EvolutionConfig(model_name="gpt-5.2", mining_score_threshold=0.5)
    mined = evolve.mine_few_shots(
        [
            {
                "score": 0.9,
                "passed": True,
                "run_status": "completed",
                "graded_details": {"final_product_reached": True},
                "subagent_scores": {
                    "missing_reagents": {"quality_score": 0.7},
                },
                "step_outputs": [
                    {
                        "step_name": "missing_reagents",
                        "output": {
                            "status": "success",
                            "missing_reactants": ["Cl"],
                            "missing_products": [],
                            "schema_validation": {"status": "ok", "source": "tool_call"},
                        },
                    }
                ],
                "input_payload": {"starting_materials": ["C=O"], "products": ["CO"]},
            }
        ],
        config,
        {"predict_missing_reagents": set()},
        {"predict_missing_reagents": 0.8},
    )

    assert mined == {}


def test_mine_few_shots_rejects_non_passing_results_even_with_high_score(tmp_path: Path) -> None:
    evolve = _load_evolve_module()
    config = evolve.EvolutionConfig(model_name="gpt-5.2", mining_score_threshold=0.3)
    mined = evolve.mine_few_shots(
        [
            {
                "score": 0.55,
                "passed": False,
                "run_status": "completed",
                "graded_details": {"final_product_reached": False},
                "step_outputs": [
                    {
                        "step_name": "missing_reagents",
                        "output": {
                            "status": "success",
                            "missing_reactants": ["O"],
                            "schema_validation": {"status": "ok", "source": "tool_call"},
                        },
                    }
                ],
                "input_payload": {"starting_materials": ["C=O"], "products": ["CO"]},
            }
        ],
        config,
        {"predict_missing_reagents": set()},
        {"predict_missing_reagents": 0.0},
    )
    assert mined == {}


def test_extract_validation_errors_reads_checks_payload() -> None:
    evolve = _load_evolve_module()
    errors = evolve.extract_validation_errors(
        [
            {
                "step_name": "mechanism_synthesis",
                "attempt": 3,
                "retry_index": 1,
                "validation": {
                    "passed": False,
                    "checks": [
                        {"name": "bond_electron_validation", "passed": True, "details": {}},
                        {
                            "name": "atom_balance_validation",
                            "passed": False,
                            "details": {"error": "atom mismatch", "extra": {"left_only": ["Cl"]}},
                        },
                    ],
                },
            }
        ]
    )
    assert len(errors) == 1
    assert errors[0]["check_name"] == "atom_balance_validation"
    assert errors[0]["validation_error"] == "atom mismatch"
    assert errors[0]["attempt"] == 3
    assert errors[0]["retry_index"] == 1
    assert errors[0]["step_index"] == 3


def test_extract_validation_errors_falls_back_without_checks() -> None:
    evolve = _load_evolve_module()
    errors = evolve.extract_validation_errors(
        [
            {
                "step_name": "mechanism_synthesis",
                "attempt": 2,
                "retry_index": 0,
                "validation": {
                    "passed": False,
                    "error": "legacy_error",
                    "details": {"message": "legacy details"},
                },
            }
        ]
    )
    assert len(errors) == 1
    assert errors[0]["check_name"] is None
    assert errors[0]["validation_error"] == "legacy_error"
    assert errors[0]["step_index"] == 2
