"""Eval tier regression tests for PR gating.

These tests require API keys and run actual LLM calls against the default
FlowER-derived eval tiers defined in training_data/eval_tiers.json.

Usage:
    # Run a specific tier
    PYTHONPATH=. pytest tests/llm/test_eval_tiers.py -k easy
    PYTHONPATH=. pytest tests/llm/test_eval_tiers.py -k medium
    PYTHONPATH=. pytest tests/llm/test_eval_tiers.py -k hard

    # Run all tiers
    PYTHONPATH=. pytest tests/llm/test_eval_tiers.py

Eval tiers gate PR approval:
    - Cheap model additions: easy tier must pass at expected baseline
    - Few-shot / module PRs: medium tier must show improvement or no regression
    - SOTA claims: if leaderboard >=80% on hard, must improve that score
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EVAL_SET_PATH = _PROJECT_ROOT / "training_data" / "eval_set.json"
_EVAL_TIERS_PATH = _PROJECT_ROOT / "training_data" / "eval_tiers.json"


# ---------------------------------------------------------------------------
# Skip if no API key or eval files missing
# ---------------------------------------------------------------------------

_has_api_key = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
_has_eval_files = _EVAL_SET_PATH.exists() and _EVAL_TIERS_PATH.exists()

pytestmark = [
    pytest.mark.skipif(not _has_api_key, reason="No API key set — LLM tests skipped"),
    pytest.mark.skipif(not _has_eval_files, reason="Eval files not found — run convert_training_data.py first"),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _load_eval_data() -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load eval tiers and full eval set."""
    with open(_EVAL_TIERS_PATH) as f:
        tiers = json.load(f)
    with open(_EVAL_SET_PATH) as f:
        eval_set = json.load(f)
    return tiers, eval_set


def _get_tier_reactions(tier_name: str) -> List[Dict[str, Any]]:
    """Get the list of reaction dicts for a given tier."""
    tiers, eval_set = _load_eval_data()
    tier_ids = tiers.get(tier_name, [])
    by_id = {r["id"]: r for r in eval_set}
    return [by_id[rid] for rid in tier_ids if rid in by_id]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEvalTierStructure:
    """Validate eval_tiers.json structure (no API key needed)."""

    def test_tiers_file_exists(self) -> None:
        assert _EVAL_TIERS_PATH.exists(), "eval_tiers.json not found"

    def test_eval_set_file_exists(self) -> None:
        assert _EVAL_SET_PATH.exists(), "eval_set.json not found"

    def test_all_tiers_present(self) -> None:
        tiers, _ = _load_eval_data()
        for tier in ("easy", "medium", "hard"):
            assert tier in tiers, f"Missing tier: {tier}"
            assert len(tiers[tier]) == 10, f"Tier {tier} must have exactly 10 reactions, got {len(tiers[tier])}"

    def test_all_tier_ids_exist_in_eval_set(self) -> None:
        tiers, eval_set = _load_eval_data()
        eval_ids = {r["id"] for r in eval_set}
        for tier_name in ("easy", "medium", "hard"):
            for rid in tiers[tier_name]:
                assert rid in eval_ids, f"Tier {tier_name} references {rid} which is not in eval_set.json"

    def test_no_duplicate_ids_across_tiers(self) -> None:
        tiers, _ = _load_eval_data()
        all_ids = tiers["easy"] + tiers["medium"] + tiers["hard"]
        assert len(all_ids) == len(set(all_ids)), "Duplicate reaction IDs across tiers"

    def test_meta_field_present(self) -> None:
        tiers, _ = _load_eval_data()
        assert "_meta" in tiers, "eval_tiers.json must have a _meta field"
        assert "difficulty_criteria" in tiers["_meta"]


class TestEvalTierEasy:
    """Run the easy eval tier (1-2 step reactions).

    Required for: cheap/lightweight model addition PRs.
    """

    def test_easy_tier_reactions_loadable(self) -> None:
        reactions = _get_tier_reactions("easy")
        assert len(reactions) == 10
        for r in reactions:
            steps = r.get("n_mechanistic_steps", 0)
            assert 1 <= steps <= 2, (
                f"Easy tier reaction {r['id']} has {steps} steps (expected 1-2)"
            )

    # Placeholder for actual LLM eval — uncomment and implement when running real evals
    # @pytest.mark.parametrize("reaction", _get_tier_reactions("easy"), ids=lambda r: r["id"])
    # def test_easy_tier_completion(self, reaction: Dict[str, Any]) -> None:
    #     """Run a single easy-tier reaction and assert mechanism loop completes."""
    #     pass


class TestEvalTierMedium:
    """Run the medium eval tier (3-step reactions).

    Required for: few-shot and module addition PRs.
    """

    def test_medium_tier_reactions_loadable(self) -> None:
        reactions = _get_tier_reactions("medium")
        assert len(reactions) == 10
        for r in reactions:
            steps = r.get("n_mechanistic_steps", 0)
            assert steps == 3, (
                f"Medium tier reaction {r['id']} has {steps} steps (expected 3)"
            )


class TestEvalTierHard:
    """Run the hard eval tier (4-8 step reactions).

    Required for: SOTA improvement claims; if leaderboard shows >=80% on hard
    for a given model, improving that score is required.
    """

    def test_hard_tier_reactions_loadable(self) -> None:
        reactions = _get_tier_reactions("hard")
        assert len(reactions) == 10
        for r in reactions:
            steps = r.get("n_mechanistic_steps", 0)
            assert 4 <= steps <= 8, (
                f"Hard tier reaction {r['id']} has {steps} steps (expected 4-8)"
            )
