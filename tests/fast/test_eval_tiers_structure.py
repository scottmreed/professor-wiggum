"""Fast tests for the default FlowER eval set and tier structure."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EVAL_SET_PATH = _PROJECT_ROOT / "training_data" / "eval_set.json"
_EVAL_TIERS_PATH = _PROJECT_ROOT / "training_data" / "eval_tiers.json"

_skip_no_files = pytest.mark.skipif(
    not (_EVAL_SET_PATH.exists() and _EVAL_TIERS_PATH.exists()),
    reason="Eval files not found — run convert_training_data.py first",
)


@_skip_no_files
class TestEvalSetStructure:
    """Validate eval_set.json is well-formed."""

    def test_eval_set_is_list(self) -> None:
        with open(_EVAL_SET_PATH) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) > 0

    def test_eval_set_entries_have_required_fields(self) -> None:
        with open(_EVAL_SET_PATH) as f:
            data = json.load(f)
        required = {"id", "name", "starting_materials", "products"}
        for entry in data:
            missing = required - set(entry.keys())
            assert not missing, f"Entry {entry.get('id', '?')} missing fields: {missing}"

    def test_eval_set_ids_unique(self) -> None:
        with open(_EVAL_SET_PATH) as f:
            data = json.load(f)
        ids = [r["id"] for r in data]
        assert len(ids) == len(set(ids)), "Duplicate IDs in eval_set.json"

    def test_eval_set_has_100_reactions(self) -> None:
        with open(_EVAL_SET_PATH) as f:
            data = json.load(f)
        assert len(data) == 100, f"Expected 100 reactions, got {len(data)}"


@_skip_no_files
class TestEvalTiersStructure:
    """Validate eval_tiers.json is well-formed and consistent with eval_set.json."""

    def test_tiers_has_meta(self) -> None:
        with open(_EVAL_TIERS_PATH) as f:
            tiers = json.load(f)
        assert "_meta" in tiers
        assert "difficulty_criteria" in tiers["_meta"]
        assert "source" in tiers["_meta"]

    def test_all_three_tiers_present(self) -> None:
        with open(_EVAL_TIERS_PATH) as f:
            tiers = json.load(f)
        for tier in ("easy", "medium", "hard"):
            assert tier in tiers, f"Missing tier: {tier}"

    def test_each_tier_is_a_list(self) -> None:
        with open(_EVAL_TIERS_PATH) as f:
            tiers = json.load(f)
        for tier in ("easy", "medium", "hard"):
            assert isinstance(tiers[tier], list), f"Tier {tier}: expected list"

    def test_tier_ids_are_strings(self) -> None:
        with open(_EVAL_TIERS_PATH) as f:
            tiers = json.load(f)
        for tier in ("easy", "medium", "hard"):
            for rid in tiers[tier]:
                assert isinstance(rid, str), f"Tier {tier}: expected string ID, got {type(rid)}"

    def test_no_duplicates_across_tiers(self) -> None:
        with open(_EVAL_TIERS_PATH) as f:
            tiers = json.load(f)
        all_ids = tiers["easy"] + tiers["medium"] + tiers["hard"]
        assert len(all_ids) == len(set(all_ids)), "Duplicate IDs across tiers"
        with open(_EVAL_SET_PATH) as f:
            eval_set = json.load(f)
        assert all_ids == [row["id"] for row in eval_set], "Tiers should preserve eval_set ranked order"

    def test_all_tier_ids_exist_in_eval_set(self) -> None:
        with open(_EVAL_TIERS_PATH) as f:
            tiers = json.load(f)
        with open(_EVAL_SET_PATH) as f:
            eval_set = json.load(f)
        eval_ids = {r["id"] for r in eval_set}
        for tier in ("easy", "medium", "hard"):
            for rid in tiers[tier]:
                assert rid in eval_ids, f"Tier {tier} ID '{rid}' not in eval_set.json"

    def test_easy_tier_step_counts(self) -> None:
        with open(_EVAL_TIERS_PATH) as f:
            tiers = json.load(f)
        with open(_EVAL_SET_PATH) as f:
            by_id = {r["id"]: r for r in json.load(f)}
        for rid in tiers["easy"]:
            steps = by_id[rid].get("n_mechanistic_steps", 0)
            assert 1 <= steps <= 2, f"Easy tier {rid} has {steps} steps (expected 1-2)"

    def test_medium_tier_step_counts(self) -> None:
        with open(_EVAL_TIERS_PATH) as f:
            tiers = json.load(f)
        with open(_EVAL_SET_PATH) as f:
            by_id = {r["id"]: r for r in json.load(f)}
        for rid in tiers["medium"]:
            steps = by_id[rid].get("n_mechanistic_steps", 0)
            assert steps == 3, f"Medium tier {rid} has {steps} steps (expected 3)"

    def test_hard_tier_step_counts(self) -> None:
        with open(_EVAL_TIERS_PATH) as f:
            tiers = json.load(f)
        with open(_EVAL_SET_PATH) as f:
            by_id = {r["id"]: r for r in json.load(f)}
        for rid in tiers["hard"]:
            steps = by_id[rid].get("n_mechanistic_steps", 0)
            assert steps >= 4, f"Hard tier {rid} has {steps} steps (expected 4+)"
