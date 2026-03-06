from __future__ import annotations

import json
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TRAINING = _PROJECT_ROOT / "training_data"


def test_humanbenchmark_artifacts_exist() -> None:
    assert (_TRAINING / "eval_set.json").exists()
    assert (_TRAINING / "eval_tiers.json").exists()
    report_path = _TRAINING / "eval_quality_report.json"
    assert report_path.exists()
    json.loads(report_path.read_text(encoding="utf-8"))


def test_eval_set_has_known_mechanism_for_each_entry() -> None:
    data = json.loads((_TRAINING / "eval_set.json").read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) == 100
    for row in data:
        known = row.get("known_mechanism")
        assert isinstance(known, dict)
        assert known.get("source") == "FlowER flower_new_dataset train.txt"
        steps = known.get("steps")
        assert isinstance(steps, list)
        assert isinstance(row.get("verified_mechanism"), dict)


def test_eval_set_includes_multi_step_mechanisms() -> None:
    data = json.loads((_TRAINING / "eval_set.json").read_text(encoding="utf-8"))
    step_counts = [
        len(((row.get("known_mechanism") or {}).get("steps") or []))
        for row in data
    ]
    assert any(count >= 1 for count in step_counts)
    assert all(count == int(row.get("n_mechanistic_steps") or 0) for count, row in zip(step_counts, data))


def test_tiers_are_ordered_step_bands() -> None:
    tiers = json.loads((_TRAINING / "eval_tiers.json").read_text(encoding="utf-8"))
    data = json.loads((_TRAINING / "eval_set.json").read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in data}
    for tier in ("easy", "medium", "hard"):
        assert isinstance(tiers[tier], list)
        for rid in tiers[tier]:
            assert str(rid).startswith("flower_")
    all_ids = tiers["easy"] + tiers["medium"] + tiers["hard"]
    assert len(all_ids) == len(set(all_ids))
    assert all_ids == [row["id"] for row in data]
    assert all(1 <= int(by_id[rid]["n_mechanistic_steps"]) <= 2 for rid in tiers["easy"])
    assert all(int(by_id[rid]["n_mechanistic_steps"]) == 3 for rid in tiers["medium"])
    assert all(int(by_id[rid]["n_mechanistic_steps"]) >= 4 for rid in tiers["hard"])
