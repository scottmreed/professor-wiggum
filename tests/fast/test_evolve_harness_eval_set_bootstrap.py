from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from mechanistic_agent.core import RunStore


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "evolve_harness.py"
    spec = importlib.util.spec_from_file_location("evolve_harness_module", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ensure_default_flower_eval_set_imports_repo_default(tmp_path: Path) -> None:
    module = _load_module()
    (tmp_path / "training_data").mkdir(parents=True)
    (tmp_path / "data").mkdir(parents=True)
    (tmp_path / "training_data" / "eval_set.json").write_text(
        json.dumps(
            [
                {
                    "id": "flower_000001",
                    "starting_materials": ["C=O"],
                    "products": ["CO"],
                    "known_mechanism": {"min_steps": 1, "steps": [{"step_index": 1, "target_smiles": "CO"}]},
                }
            ]
        ),
        encoding="utf-8",
    )
    store = RunStore(tmp_path / "data" / "mechanistic.db")

    eval_set_id = module.ensure_default_flower_eval_set(store, base_dir=tmp_path)

    assert eval_set_id
    eval_sets = store.list_eval_sets()
    assert eval_sets[0]["name"] == "flower_100_default"
    cases = store.list_eval_set_cases(eval_set_id)
    assert len(cases) == 1
    assert cases[0]["case_id"] == "flower_000001"


def test_resolve_eval_set_alias_maps_eval_set_to_flower_default(tmp_path: Path) -> None:
    module = _load_module()
    store = RunStore(tmp_path / "mechanistic.db")
    eval_set_id = store.add_eval_set(
        name="flower_100_default",
        version="flower100_v1",
        source_path="training_data/eval_set.json",
        sha256=None,
        cases=[],
        active=True,
    )

    resolved = module.resolve_eval_set_id(store, "eval_set")

    assert resolved == eval_set_id
