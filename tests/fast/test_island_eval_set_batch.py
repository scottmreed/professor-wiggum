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


def test_prior_failing_cases_are_scheduled_first(tmp_path: Path) -> None:
    """Mutations conditioned on failure traces must be evaluated on the cases that produced them."""
    mod = _load_evolve_module()
    store = RunStore(tmp_path / "data" / "mechanistic.db")
    eval_set_id = store.add_eval_set(
        name="practice_set",
        version="v1",
        source_path=None,
        sha256=None,
        cases=[_case("hard_a", 4), _case("hard_b", 5), _case("hard_c", 6)],
    )
    older = store.create_eval_run(eval_set_id=eval_set_id, run_group_name="g0", model="m", harness_bundle_hash=None)
    store.record_eval_run_result(eval_run_id=older, case_id="hard_a", run_id=None, score=0.4, passed=False, cost={}, latency_ms=None, summary={})
    store.record_eval_run_result(eval_run_id=older, case_id="hard_c", run_id=None, score=0.3, passed=False, cost={}, latency_ms=None, summary={})
    newer = store.create_eval_run(eval_set_id=eval_set_id, run_group_name="g1", model="m", harness_bundle_hash=None)
    store.record_eval_run_result(eval_run_id=newer, case_id="hard_a", run_id=None, score=0.99, passed=True, cost={}, latency_ms=None, summary={})
    store.record_eval_run_result(eval_run_id=newer, case_id="hard_b", run_id=None, score=0.5, passed=False, cost={}, latency_ms=None, summary={})

    # Most recent verdict per case wins: hard_a was fixed, hard_b and hard_c still fail.
    assert mod.prior_failing_case_ids(store, eval_set_id) == ["hard_b", "hard_c"]

    config = mod.EvolutionConfig(model_name="gpt-5", group_size=1)
    hard_island = next(isl for isl in DEFAULT_ISLANDS if isl.eval_tier_filter == "hard")
    picks = [
        mod.prepare_eval_set_batch(
            store=store,
            eval_set_id=eval_set_id,
            config=config,
            island=hard_island,
            generation=gen,
            prioritize_case_ids=["hard_b", "hard_c"],
        )["runnable_cases"][0]["case_id"]
        for gen in (1, 2, 3)
    ]
    assert picks == ["hard_b", "hard_c", "hard_a"]

    # Without prior failures the deterministic step-count order is unchanged.
    plain = mod.prepare_eval_set_batch(store=store, eval_set_id=eval_set_id, config=config, island=hard_island, generation=1)
    assert plain["runnable_cases"][0]["case_id"] == "hard_a"
