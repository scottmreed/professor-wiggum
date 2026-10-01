"""Eval-created runs must bind prompt versions so their traces can become evidence.

`POST /api/traces/export_evidence` rejects a trace whose ``prompt_version_id`` is
empty, and the coordinator fills that field from ``run_step_prompts``. Runs
created by ``main.py eval`` used to skip the binding, so no eval trace could
satisfy the prompt-trace evidence gate.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import main as main_module
from mechanistic_agent.core.db import RunStore
from mechanistic_agent.core.registries import RegistrySet
from mechanistic_agent.core import select_step_models

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL = "anthropic/claude-opus-4.6"


def _create_bare_run(store: RunStore) -> str:
    return store.create_run(
        mode="unverified",
        input_payload={"starting_materials": ["CCBr", "[Cl-]"], "products": ["CCCl", "[Br-]"]},
        config={"model": MODEL, "model_name": MODEL},
        prompt_bundle_hash="",
        skill_bundle_hash="",
    )


def test_bind_run_prompts_binds_proposal_step(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "m.db")
    registry = RegistrySet(REPO_ROOT)
    run_id = _create_bare_run(store)
    plan = select_step_models(
        model_name=MODEL,
        thinking_level=None,
        functional_groups_enabled=True,
        intermediate_prediction_enabled=True,
        optional_llm_tools=["attempt_atom_mapping", "predict_missing_reagents"],
    )

    prompt_ids = registry.bind_run_prompts(
        store, run_id, model_name=plan.model_name, step_names=plan.step_models
    )

    proposal_id = store.resolve_run_step_prompt_id(
        run_id=run_id, step_name="mechanism_step_proposal", attempt=1
    )
    assert proposal_id and proposal_id == prompt_ids["mechanism_step_proposal"]
    bound = {row["step_name"] for row in store.list_run_step_prompts(run_id)}
    assert "mechanism_step_proposal" in bound


def test_harness_eval_run_binds_prompt_versions(tmp_path: Path, monkeypatch) -> None:
    store = RunStore(tmp_path / "m.db")
    created: list[str] = []

    class _NoopCoordinator:
        def __init__(self, _store: RunStore) -> None:
            pass

        def execute_run(self, run_id: str, _stop) -> None:
            created.append(run_id)

    monkeypatch.setattr(main_module, "RunCoordinator", _NoopCoordinator)
    cases = [
        {
            "case_id": "case_1",
            "input": {"starting_materials": ["CCBr", "[Cl-]"], "products": ["CCCl", "[Br-]"]},
            "expected": {},
        }
    ]
    eval_set_id = store.add_eval_set(
        name="binding_eval", version="v1", source_path=None, sha256=None, cases=cases
    )
    eval_set = SimpleNamespace(eval_set_id=eval_set_id, cases=cases)

    main_module._execute_harness_eval_run(
        store=store,
        registry=RegistrySet(REPO_ROOT),
        resolved_eval_set=eval_set,
        model_name=MODEL,
        thinking_level=None,
        harness="default",
        run_group="test_binding",
        max_cases=1,
        max_steps=1,
        max_runtime=10.0,
        chemistry_backend="python",
        rdkit_cli_command=None,
        chemistry_backend_parity=False,
        json_output=True,
        trace_runtime=False,
    )

    assert created, "eval did not create a run"
    assert store.resolve_run_step_prompt_id(
        run_id=created[0], step_name="mechanism_step_proposal", attempt=1
    )
