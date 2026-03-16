"""Tests for island-based evolution archive."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import pytest

from mechanistic_agent.core.archive import (
    DEFAULT_ISLANDS,
    EvolutionArchive,
    _sigmoid,
)
from mechanistic_agent.core.db import RunStore
from mechanistic_agent.core.types import ArchiveEntry


def _make_store(tmp_path: Path) -> RunStore:
    """Create a fresh RunStore in a temp directory."""
    db_path = tmp_path / "data" / "mechanistic.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return RunStore(db_path)


def _make_entry(
    *,
    island_id: str = "mapping",
    generation: int = 1,
    score: float = 0.5,
    parent_id: str | None = None,
    children_count: int = 0,
    mutation_type: str = "seed",
) -> ArchiveEntry:
    """Create a minimal ArchiveEntry for testing."""
    return ArchiveEntry(
        id="",  # Will be assigned by insert
        generation=generation,
        island_id=island_id,
        parent_id=parent_id,
        archive_inspiration_ids=[],
        harness_name="default",
        harness_config_json="{}",
        prompt_bundle_hash="abc123",
        skill_bundle_hash="def456",
        few_shot_snapshot_json="{}",
        topology_profile_json="{}",
        mutation_type=mutation_type,
        mutation_summary="test entry",
        mean_quality_score=score,
        weighted_pass_rate=score,
        per_subagent_scores_json="{}",
        total_cost=0.01,
        eval_run_id=None,
        eval_tier="mixed",
        case_count=4,
        children_count=children_count,
        score_delta=0.0,
        migration_history_json="[]",
        created_at=time.time(),
    )


class TestArchiveCRUD:
    def test_insert_and_get(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        archive = EvolutionArchive(store=store)
        entry = _make_entry(score=0.8)
        entry_id = archive.insert(entry)
        assert entry_id
        fetched = archive.get(entry_id)
        assert fetched is not None
        assert fetched.mean_quality_score == 0.8
        assert fetched.island_id == "mapping"

    def test_list_island(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        archive = EvolutionArchive(store=store)
        archive.insert(_make_entry(island_id="mapping", score=0.6))
        archive.insert(_make_entry(island_id="mapping", score=0.9))
        archive.insert(_make_entry(island_id="topology", score=0.7))
        mapping_entries = archive.list_island("mapping")
        assert len(mapping_entries) == 2
        # Should be sorted by score DESC
        assert mapping_entries[0].mean_quality_score >= mapping_entries[1].mean_quality_score

    def test_island_best_score(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        archive = EvolutionArchive(store=store)
        archive.insert(_make_entry(island_id="mapping", score=0.3))
        archive.insert(_make_entry(island_id="mapping", score=0.9))
        assert archive.island_best_score("mapping") == pytest.approx(0.9)
        assert archive.island_best_score("topology") == 0.0

    def test_increment_children(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        archive = EvolutionArchive(store=store)
        entry_id = archive.insert(_make_entry(score=0.5))
        archive.increment_children(entry_id)
        archive.increment_children(entry_id)
        fetched = archive.get(entry_id)
        assert fetched is not None
        assert fetched.children_count == 2

    def test_is_empty_and_max_generation(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        archive = EvolutionArchive(store=store)
        assert archive.is_empty()
        assert archive.max_generation() == -1
        archive.insert(_make_entry(generation=3))
        assert not archive.is_empty()
        assert archive.max_generation() == 3

    def test_population_cap_eviction(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        # Small island with cap=3
        from mechanistic_agent.core.types import IslandConfig

        small_island = IslandConfig(
            id="tiny",
            label="Tiny Island",
            mutation_target="mapping",
            allowed_lanes=["prompt"],
            eval_tier_filter=None,
            population_cap=3,
            stagnation_threshold=5,
            description="test island",
        )
        archive = EvolutionArchive(store=store, islands=[small_island])
        archive.insert(_make_entry(island_id="tiny", score=0.2))
        archive.insert(_make_entry(island_id="tiny", score=0.5))
        archive.insert(_make_entry(island_id="tiny", score=0.8))
        # This should evict the worst (0.2)
        archive.insert(_make_entry(island_id="tiny", score=0.6))
        entries = archive.list_island("tiny")
        assert len(entries) == 3
        scores = [e.mean_quality_score for e in entries]
        assert min(scores) >= 0.5  # 0.2 was evicted


class TestWeightedParentSelection:
    def test_single_entry_returns_it(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        archive = EvolutionArchive(store=store)
        archive.insert(_make_entry(island_id="mapping", score=0.5))
        parent = archive.select_parent("mapping")
        assert parent.mean_quality_score == pytest.approx(0.5)

    def test_empty_island_raises(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        archive = EvolutionArchive(store=store)
        with pytest.raises(ValueError, match="no archive entries"):
            archive.select_parent("mapping")

    def test_higher_score_selected_more_often(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        archive = EvolutionArchive(store=store)
        # Insert entries with different scores
        archive.insert(_make_entry(island_id="mapping", score=0.1))
        archive.insert(_make_entry(island_id="mapping", score=0.9))

        rng = random.Random(42)
        selections = [archive.select_parent("mapping", rng=rng).mean_quality_score for _ in range(100)]
        high_count = sum(1 for s in selections if s > 0.5)
        # The 0.9-score entry should be selected significantly more often
        assert high_count > 60

    def test_underexplored_gets_boost(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        archive = EvolutionArchive(store=store)
        # Two entries with same score, but one has many children
        id_a = archive.insert(_make_entry(island_id="mapping", score=0.7))
        id_b = archive.insert(_make_entry(island_id="mapping", score=0.7))
        # Give entry A many children
        for _ in range(10):
            archive.increment_children(id_a)

        rng = random.Random(42)
        selections = []
        for _ in range(100):
            parent = archive.select_parent("mapping", rng=rng)
            selections.append(parent.id)
        b_count = sum(1 for s in selections if s == id_b)
        # Entry B (fewer children) should be selected more often
        assert b_count > 60


class TestMigration:
    def test_migration_succeeds_when_beating_target(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        archive = EvolutionArchive(store=store)
        archive.insert(_make_entry(island_id="topology", score=0.5))
        entry = _make_entry(island_id="mapping", score=0.8)
        entry.id = archive.insert(entry)
        entry = archive.get(entry.id)

        result = archive.attempt_migration("mapping", "topology", entry)
        assert result is True
        # Check target island now has the migrated entry
        topo_entries = archive.list_island("topology")
        assert any(e.mean_quality_score == pytest.approx(0.8) for e in topo_entries)

    def test_migration_fails_when_not_beating_target(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        archive = EvolutionArchive(store=store)
        archive.insert(_make_entry(island_id="topology", score=0.9))
        entry = _make_entry(island_id="mapping", score=0.5)
        entry.id = archive.insert(entry)
        entry = archive.get(entry.id)

        result = archive.attempt_migration("mapping", "topology", entry)
        assert result is False

    def test_migration_records_event(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        archive = EvolutionArchive(store=store)
        entry = _make_entry(island_id="mapping", score=0.8)
        entry.id = archive.insert(entry)
        entry = archive.get(entry.id)

        archive.attempt_migration("mapping", "topology", entry)
        migrations = store.list_archive_migrations()
        assert len(migrations) == 1
        assert migrations[0]["from_island"] == "mapping"
        assert migrations[0]["to_island"] == "topology"


class TestStagnation:
    def test_no_stagnation_with_recent_improvement(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        archive = EvolutionArchive(store=store)
        archive.insert(_make_entry(island_id="mapping", generation=1, score=0.3))
        archive.insert(_make_entry(island_id="mapping", generation=5, score=0.5))
        archive.insert(_make_entry(island_id="mapping", generation=9, score=0.8))
        assert archive.stagnation_check("mapping", window=5) is False

    def test_stagnation_when_no_improvement(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        archive = EvolutionArchive(store=store)
        # Old entries with high score
        archive.insert(_make_entry(island_id="mapping", generation=1, score=0.9))
        # Recent entries with lower scores
        for g in range(6, 11):
            archive.insert(_make_entry(island_id="mapping", generation=g, score=0.5))
        assert archive.stagnation_check("mapping", window=5) is True


class TestSeedFromLeaderboard:
    def test_seed_creates_entries(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        archive = EvolutionArchive(store=store)

        # Create a mock leaderboard file
        lb_dir = tmp_path / "curriculum" / "generated"
        lb_dir.mkdir(parents=True)
        lb_data = [
            {"item": {"mean_quality_score": 0.85, "weighted_pass_rate": 0.9, "case_count": 4}},
            {"item": {"mean_quality_score": 0.72, "weighted_pass_rate": 0.8, "case_count": 4}},
        ]
        (lb_dir / "leaderboard_anthropic_test.json").write_text(
            json.dumps(lb_data), encoding="utf-8"
        )

        count = archive.seed_from_leaderboard(lb_dir)
        # 2 items * 4 islands = 8 entries
        assert count == 2 * len(DEFAULT_ISLANDS)
        assert not archive.is_empty()

    def test_seed_skips_zero_score(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        archive = EvolutionArchive(store=store)

        lb_dir = tmp_path / "curriculum" / "generated"
        lb_dir.mkdir(parents=True)
        lb_data = [{"item": {"mean_quality_score": 0.0}}]
        (lb_dir / "leaderboard_anthropic_test.json").write_text(
            json.dumps(lb_data), encoding="utf-8"
        )
        count = archive.seed_from_leaderboard(lb_dir)
        assert count == 0


class TestSigmoid:
    def test_sigmoid_zero(self) -> None:
        assert _sigmoid(0.0) == pytest.approx(0.5)

    def test_sigmoid_large_positive(self) -> None:
        assert _sigmoid(100.0) == pytest.approx(1.0, abs=1e-6)

    def test_sigmoid_large_negative(self) -> None:
        assert _sigmoid(-100.0) == pytest.approx(0.0, abs=1e-6)
