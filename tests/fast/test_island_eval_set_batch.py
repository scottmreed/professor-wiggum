"""Island evolution can draw its cases from a stored (non-default) eval set."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from mechanistic_agent.core.archive import DEFAULT_ISLANDS
from mechanistic_agent.core.db import RunStore

_ROOT = Path(__file__).resolve().parents[2]


def _load_evolve_module():
    spec = importlib.util.spec_from_file_location("evolve_harness_mod_islands", _ROOT / "scripts" / "evolve_harness.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("evolve_harness_mod_islands", module)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _case(case_id: str, steps: int) -> dict:
    return {
        "case_id": case_id,
        "input": {"starting_materials": ["CCBr", "[Cl-]"], "products": ["CCCl", "[Br-]"]},
        "expected": {
            "products": ["CCCl", "[Br-]"],
            "known_mechanism": {"steps": [{"step_index": i + 1, "target_smiles": "CCCl"} for i in range(steps)]},
        },
    }


def test_prepare_eval_set_batch_filters_by_tier_and_rotates(tmp_path: Path) -> None:
    mod = _load_evolve_module()
    store = RunStore(tmp_path / "data" / "mechanistic.db")
    eval_set_id = store.add_eval_set(
        name="practice_set",
        version="v1",
        source_path=None,
        sha256=None,
        cases=[_case("easy_1", 1), _case("hard_a", 4), _case("hard_b", 5)],
    )
    config = mod.EvolutionConfig(model_name="gpt-5", group_size=1)
    hard_island = next(isl for isl in DEFAULT_ISLANDS if isl.eval_tier_filter == "hard")
    mapping_island = next(isl for isl in DEFAULT_ISLANDS if isl.id == "mapping")

    batch = mod.prepare_eval_set_batch(store=store, eval_set_id=eval_set_id, config=config, island=hard_island, generation=1)
    assert [c["case_id"] for c in batch["runnable_cases"]] == ["hard_a"]
    entry = batch["runnable_cases"][0]["entry"]
    assert entry["step_count"] == 4 and entry["global_rank"] == 1 and entry["source"] == "eval_set"
    assert batch["runnable_cases"][0]["expected"]["products"] == ["CCCl", "[Br-]"]

    batch2 = mod.prepare_eval_set_batch(store=store, eval_set_id=eval_set_id, config=config, island=hard_island, generation=2)
    assert [c["case_id"] for c in batch2["runnable_cases"]] == ["hard_b"]

    easy_only = mod.prepare_eval_set_batch(
        store=store, eval_set_id=eval_set_id, config=config, island=mapping_island, generation=1, step_count_override=1
    )
    assert [c["case_id"] for c in easy_only["runnable_cases"]] == ["easy_1"]
    assert easy_only["current_step_count"] == 1
