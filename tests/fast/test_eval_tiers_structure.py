"""Fast tests for the default FlowER eval set and tier structure."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EVAL_SET_PATH = _PROJECT_ROOT / "training_data" / "eval_set.json"
_EVAL_TIERS_PATH = _PROJECT_ROOT / "training_data" / "eval_tiers.json"
_BASELINE_TIERS_PATH = _PROJECT_ROOT / "training_data" / "baseline_tiers_clawdiator.json"
# Gitignored, generated, local/CI-optional: holds the actual medium/hard (3-step /
# 4+-step) records that training_data/eval_tiers.json and
# training_data/baseline_tiers_clawdiator.json reference by ID. Not shipped in git
# (training_data/* is gitignored except an explicit allow-list — see .gitignore), so
# any check that reads it must skip cleanly when it is absent (fresh checkout / CI
# without training_data/REGENERATE.md having been run).
_MULTISTEP_PATH = _PROJECT_ROOT / "training_data" / "flower_mechanisms_multistep.json"

_skip_no_files = pytest.mark.skipif(
    not (_EVAL_SET_PATH.exists() and _EVAL_TIERS_PATH.exists()),
    reason="Eval files not found — run convert_training_data.py first",
)

_skip_no_multistep = pytest.mark.skipif(
    not _MULTISTEP_PATH.exists(),
    reason=(
        "training_data/flower_mechanisms_multistep.json is gitignored/generated and "
        "not present in this checkout — see training_data/REGENERATE.md"
    ),
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
    """Validate eval_tiers.json is well-formed and internally consistent.

    `easy` is sourced from `training_data/eval_set.json` (single-step FlowER
    defaults, always tracked in git). `medium`/`hard` are sourced from
    `training_data/flower_mechanisms_multistep.json` (3-step / 4+-step FlowER
    conversions), which is a gitignored, generated artifact — see
    `training_data/REGENERATE.md`. Checks that need that file are marked with
    `_skip_no_multistep` and skip cleanly when it is absent; checks that only need
    the always-tracked `eval_tiers.json` / `baseline_tiers_clawdiator.json` run
    unconditionally.
    """

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

    def test_no_tier_is_empty(self) -> None:
        """Every tier must have at least one case.

        A tier silently resolving to 0 cases is exactly the failure mode this
        guards against: `eval_tiers.json` is the CODEOWNERS-protected arbiter cited
        by docs/change_evidence_policy.md ("the eval tiers are the arbiter"), so an
        empty tier here means "improve medium" is unenforceable for that tier.
        """
        with open(_EVAL_TIERS_PATH) as f:
            tiers = json.load(f)
        for tier in ("easy", "medium", "hard"):
            assert tiers[tier], f"Tier '{tier}' is empty in {_EVAL_TIERS_PATH}"

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

    def test_easy_tier_ids_preserve_eval_set_order(self) -> None:
        with open(_EVAL_TIERS_PATH) as f:
            tiers = json.load(f)
        with open(_EVAL_SET_PATH) as f:
            eval_set = json.load(f)
        assert tiers["easy"] == [row["id"] for row in eval_set], (
            "easy tier should preserve eval_set.json ranked order"
        )

    def test_easy_tier_ids_exist_in_eval_set(self) -> None:
        with open(_EVAL_TIERS_PATH) as f:
            tiers = json.load(f)
        with open(_EVAL_SET_PATH) as f:
            eval_set = json.load(f)
        eval_ids = {r["id"] for r in eval_set}
        for rid in tiers["easy"]:
            assert rid in eval_ids, f"Easy tier ID '{rid}' not in eval_set.json"

    def test_easy_tier_step_counts(self) -> None:
        with open(_EVAL_TIERS_PATH) as f:
            tiers = json.load(f)
        with open(_EVAL_SET_PATH) as f:
            by_id = {r["id"]: r for r in json.load(f)}
        for rid in tiers["easy"]:
            steps = by_id[rid].get("n_mechanistic_steps", 0)
            assert 1 <= steps <= 2, f"Easy tier {rid} has {steps} steps (expected 1-2)"

    def test_medium_and_hard_match_baseline_tiers_clawdiator(self) -> None:
        """medium/hard are meant to be synchronized views with baseline_tiers_clawdiator.json.

        docs/development_leaderboard_routes.md describes eval_tiers.json and
        baseline_tiers_clawdiator.json as "synchronized views over the same
        development mechanism pool." medium/hard here were populated by copying
        baseline_tiers_clawdiator.json's already-established, deterministic
        selection verbatim, so they must match exactly. This check needs no
        gitignored multistep data — both files are tracked in git.
        """
        if not _BASELINE_TIERS_PATH.exists():
            pytest.skip("training_data/baseline_tiers_clawdiator.json not found")
        with open(_EVAL_TIERS_PATH) as f:
            tiers = json.load(f)
        with open(_BASELINE_TIERS_PATH) as f:
            baseline = json.load(f)
        assert tiers["medium"] == baseline["medium"], (
            "eval_tiers.json medium should match baseline_tiers_clawdiator.json medium"
        )
        assert tiers["hard"] == baseline["hard"], (
            "eval_tiers.json hard should match baseline_tiers_clawdiator.json hard"
        )

    @_skip_no_multistep
    def test_medium_tier_ids_exist_in_multistep_file(self) -> None:
        with open(_EVAL_TIERS_PATH) as f:
            tiers = json.load(f)
        with open(_MULTISTEP_PATH) as f:
            multistep_ids = {r["id"] for r in json.load(f)}
        for rid in tiers["medium"]:
            assert rid in multistep_ids, f"Medium tier ID '{rid}' not in flower_mechanisms_multistep.json"

    @_skip_no_multistep
    def test_hard_tier_ids_exist_in_multistep_file(self) -> None:
        with open(_EVAL_TIERS_PATH) as f:
            tiers = json.load(f)
        with open(_MULTISTEP_PATH) as f:
            multistep_ids = {r["id"] for r in json.load(f)}
        for rid in tiers["hard"]:
            assert rid in multistep_ids, f"Hard tier ID '{rid}' not in flower_mechanisms_multistep.json"

    @_skip_no_multistep
    def test_medium_tier_step_counts(self) -> None:
        with open(_EVAL_TIERS_PATH) as f:
            tiers = json.load(f)
        with open(_MULTISTEP_PATH) as f:
            by_id = {r["id"]: r for r in json.load(f)}
        for rid in tiers["medium"]:
            steps = by_id[rid].get("n_mechanistic_steps", 0)
            assert steps == 3, f"Medium tier {rid} has {steps} steps (expected 3)"

    @_skip_no_multistep
    def test_hard_tier_step_counts(self) -> None:
        with open(_EVAL_TIERS_PATH) as f:
            tiers = json.load(f)
        with open(_MULTISTEP_PATH) as f:
            by_id = {r["id"]: r for r in json.load(f)}
        for rid in tiers["hard"]:
            steps = by_id[rid].get("n_mechanistic_steps", 0)
            assert steps >= 4, f"Hard tier {rid} has {steps} steps (expected 4+)"

    def test_medium_and_hard_ids_disjoint_from_holdout(self) -> None:
        """medium/hard must not leak official holdout cases into the development tiers.

        The holdout (training_data/leaderboard_holdout/eval_set_holdout.json, resolved
        via mechanistic_agent.data_paths under the sibling wiggum-data data root) is
        FlowER test-split derived and uses a disjoint 'flower_test_*' ID namespace, while
        eval_tiers.json / flower_mechanisms_multistep.json use train-split 'flower_*'
        IDs (no '_test_'). This is a structural invariant, not per-ID lookup, so it
        does not require the holdout file to be present.
        """
        with open(_EVAL_TIERS_PATH) as f:
            tiers = json.load(f)
        for tier in ("medium", "hard"):
            for rid in tiers[tier]:
                assert "_test_" not in rid, (
                    f"{tier} tier ID '{rid}' looks like a holdout ID (contains '_test_')"
                )
