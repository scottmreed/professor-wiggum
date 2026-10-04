"""Append-only growth of the development tiers: dataset extension + in-place eval-set append."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

pytest.importorskip("rdkit")

from mechanistic_agent.core.db import RunStore  # noqa: E402
from mechanistic_agent.flower_curriculum import ConversionError  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_module(PROJECT_ROOT / "scripts" / "build_flower_mechanism_dataset.py", "flower_builder_extend")
sync_script = _load_module(PROJECT_ROOT / "scripts" / "sync_dev_tier_eval_sets.py", "sync_dev_tier_eval_sets")


# ---------------------------------------------------------------------------
# Dataset extension (scripts/build_flower_mechanism_dataset.py --mode extend)
# ---------------------------------------------------------------------------


def _case(mechanism_id: int, steps: int) -> Dict[str, Any]:
    return {
        "id": f"flower_{mechanism_id:06d}",
        "starting_materials": ["[CH4:1]"],
        "products": ["[CH4:1]"],
        "tags": ["flower", "train", "multistep"],
        "n_mechanistic_steps": steps,
        "verified_mechanism": {"steps": [{"step_index": i + 1} for i in range(steps)]},
    }


def _index(step: int, mechanism_ids: List[int]) -> List[Dict[str, Any]]:
    return [
        {"mechanism_id": mid, "case_id": f"flower_{mid:06d}", "step_count": step, "rank_within_step_count": rank}
        for rank, mid in enumerate(mechanism_ids, start=1)
    ]


def _fake_converter(step_of: Dict[int, int], failing: set[int]):
    calls: List[int] = []

    def convert(mechanism_id: int) -> Dict[str, Any]:
        calls.append(mechanism_id)
        if mechanism_id in failing:
            raise ConversionError("state_discontinuity")
        return _case(mechanism_id, step_of[mechanism_id])

    return convert, calls


def test_extend_appends_next_lowest_ranked_and_keeps_existing_rows() -> None:
    existing = [_case(1, 3), _case(2, 3), _case(10, 4)]
    report = {
        "selected_case_ids": ["flower_000001", "flower_000002", "flower_000010"],
        "skipped_case_ids": ["flower_000003"],
        "tier_summary": {"3": {"attempted": 3, "selected": 2, "skipped": 1, "available_in_index": 9}},
        "conversion_failures_by_reason": {"state_discontinuity": 1},
        "max_step": 4,
    }
    index = _index(3, [1, 3, 2, 4, 5, 6, 7, 8, 9]) + _index(7, [70, 71, 72])
    step_of = {mid: 3 for mid in range(1, 10)} | {70: 7, 71: 7, 72: 7}
    convert, calls = _fake_converter(step_of, failing={5, 71})

    dataset, new_report, added = builder.extend_stratified_dataset(
        dataset=existing,
        report=report,
        index_entries=index,
        step_targets={3: 5, 7: 5},
        exclude_ids={"flower_000004"},
        convert=convert,
    )

    assert dataset[:3] == existing  # existing rows verbatim, in place
    # 3-step: 1,3,2 already handled; 4 excluded; 5 fails; 6,7,8 fill the tier to 5.
    assert added["3"] == ["flower_000006", "flower_000007", "flower_000008"]
    # 7-step: index exhausted, one failure -> short.
    assert added["7"] == ["flower_000070", "flower_000072"]
    assert 3 not in calls and 1 not in calls and 4 not in calls  # never re-attempted / excluded
    ext = new_report["extensions"][-1]
    assert ext["tiers"]["7"]["short_by"] == 3
    assert ext["tiers"]["3"]["excluded_case_ids_passed_over"] == ["flower_000004"]
    assert new_report["selected_case_ids"][:3] == report["selected_case_ids"]
    assert new_report["skipped_case_ids"] == ["flower_000003", "flower_000005", "flower_000071"]
    assert new_report["conversion_failures_by_reason"] == {"state_discontinuity": 3}
    assert new_report["tier_summary"]["3"]["selected"] == 5
    assert new_report["step_count_distribution_sampled_set"] == {"3": 5, "4": 1, "7": 2}
    assert new_report["max_step"] == 7

    # Idempotent: a second run at the same targets converts nothing new in full tiers.
    again, _, added_again = builder.extend_stratified_dataset(
        dataset=dataset, report=new_report, index_entries=index, step_targets={3: 5}, convert=convert
    )
    assert again == dataset and added_again == {"3": []}


def test_extend_never_selects_holdout_namespace_ids() -> None:
    index = [{"mechanism_id": 1, "case_id": "flower_test_000001", "step_count": 3, "rank_within_step_count": 1}]
    convert, calls = _fake_converter({1: 3}, failing=set())
    dataset, _, added = builder.extend_stratified_dataset(
        dataset=[], report={}, index_entries=index, step_targets={3: 1}, convert=convert
    )
    assert dataset == [] and added == {"3": []} and calls == []


def test_extend_rejects_step_count_mismatch() -> None:
    convert, _ = _fake_converter({1: 4}, failing=set())
    with pytest.raises(ValueError, match="3-step tier"):
        builder.extend_stratified_dataset(
            dataset=[], report={}, index_entries=_index(3, [1]), step_targets={3: 1}, convert=convert
        )


def test_collect_case_ids_reads_datasets_and_tier_files() -> None:
    assert builder.collect_case_ids([{"id": "a"}, {"id": "b"}]) == ["a", "b"]
    assert builder.collect_case_ids({"_meta": {"x": ["no"]}, "easy": ["a"], "hard": ["b"]}) == ["a", "b"]


# ---------------------------------------------------------------------------
# In-place eval-set append (RunStore.append_eval_set_cases + sync script)
# ---------------------------------------------------------------------------


def _record(mechanism_id: int, steps: int) -> Dict[str, Any]:
    return _case(mechanism_id, steps) | {"temperature_celsius": None, "ph": None}


def test_append_eval_set_cases_keeps_id_and_skips_existing(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "mechanistic.db")
    first = sync_script.tier_eval_case(_record(1, 3))
    eval_set_id = store.add_eval_set(name="t", version="v1", source_path=None, sha256=None, cases=[first])

    result = store.append_eval_set_cases(
        eval_set_id,
        [first, sync_script.tier_eval_case(_record(2, 3))],
        version="v2",
        sha256="abc",
    )

    assert result["added_case_ids"] == ["flower_000002"]
    assert result["skipped_existing_case_ids"] == ["flower_000001"]
    assert result["case_count"] == 2
    meta = store.get_eval_set(eval_set_id)
    assert meta["version"] == "v2" and meta["sha256"] == "abc"
    assert [c["case_id"] for c in store.list_eval_set_cases(eval_set_id)] == ["flower_000001", "flower_000002"]


def test_append_eval_set_cases_refuses_holdout_and_unknown_sets(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "mechanistic.db")
    holdout_id = store.add_eval_set(
        name="h", version="v1", source_path=None, sha256=None, cases=[], purpose="leaderboard_holdout"
    )
    with pytest.raises(ValueError, match="holdout"):
        store.append_eval_set_cases(holdout_id, [sync_script.tier_eval_case(_record(1, 3))])
    with pytest.raises(KeyError):
        store.append_eval_set_cases("missing", [])


def test_sync_appends_missing_tier_cases_in_place(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "mechanistic.db")
    dataset = [_record(1, 3), _record(2, 3), _record(3, 3)]
    eval_set_id = store.add_eval_set(
        name="flower_multistep_medium_clawdiator",
        version="v1",
        source_path=None,
        sha256=None,
        cases=[sync_script.tier_eval_case(dataset[0])],
    )
    tiers = {"medium": ["flower_000001", "flower_000002", "flower_000003"]}

    plan = sync_script.sync(store, tier_ids_by_name=tiers, eval_set_ids={"medium": eval_set_id}, dataset=dataset)
    assert plan[0]["to_add"] == ["flower_000002", "flower_000003"] and plan[0]["applied"] is False
    assert len(store.list_eval_set_cases(eval_set_id)) == 1  # dry run wrote nothing

    applied = sync_script.sync(
        store, tier_ids_by_name=tiers, eval_set_ids={"medium": eval_set_id}, dataset=dataset, apply=True
    )
    assert applied[0]["applied"] is True and applied[0]["case_count"] == 3
    meta = store.get_eval_set(eval_set_id)
    assert meta["version"] == "v2"
    cases = store.list_eval_set_cases(eval_set_id)
    assert meta["sha256"] == sync_script.cases_sha256(cases)
    added = {c["case_id"]: c for c in cases}["flower_000002"]
    assert added["input"]["n_mechanistic_steps"] == 3
    assert added["expected"]["verified_mechanism"] == dataset[1]["verified_mechanism"]
    assert added["tags"] == ["flower", "train", "multistep", "flower", "multistep", "clawdiator_planned"]

    # Re-running is a no-op (nothing missing, version unchanged).
    rerun = sync_script.sync(
        store, tier_ids_by_name=tiers, eval_set_ids={"medium": eval_set_id}, dataset=dataset, apply=True
    )
    assert rerun[0]["to_add"] == [] and store.get_eval_set(eval_set_id)["version"] == "v2"


@pytest.mark.parametrize(
    "tier_ids, existing, message",
    [
        (["flower_000002"], ["flower_000001"], "does not list"),
        (["flower_000001", "flower_test_000009"], ["flower_000001"], "holdout"),
        (["flower_000001", "flower_000099"], ["flower_000001"], "missing from the dataset"),
    ],
)
def test_plan_tier_append_rejects_inconsistent_inputs(tier_ids, existing, message) -> None:
    dataset_by_id = {r["id"]: r for r in [_record(1, 3), _record(2, 3)]}
    existing_cases = [sync_script.tier_eval_case(dataset_by_id[cid]) for cid in existing]
    with pytest.raises(ValueError, match=message):
        sync_script.plan_tier_append(
            tier="medium", tier_ids=tier_ids, existing_cases=existing_cases, dataset_by_id=dataset_by_id
        )


def test_plan_tier_append_rejects_drifted_existing_case() -> None:
    record = _record(1, 3)
    stale = sync_script.tier_eval_case(record)
    stale["input"]["products"] = ["[OH2:1]"]
    with pytest.raises(ValueError, match="differ"):
        sync_script.plan_tier_append(
            tier="medium", tier_ids=["flower_000001"], existing_cases=[stale], dataset_by_id={record["id"]: record}
        )


def test_bump_version() -> None:
    assert sync_script.bump_version("v1") == "v2"
    assert sync_script.bump_version("flower100_v1") == "flower100_v1+1"
