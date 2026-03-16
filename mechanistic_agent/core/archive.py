"""Island-based evolution archive with weighted parent selection.

Inspired by ShinkaEvolve (arxiv:2509.19349). Maintains a searchable archive
of prior prompt bundles, few-shot sets, topology profiles, and model lanes.
Parents are sampled by a mix of score and underexploredness; islands are
defined by mutation target / chemistry regime, not random seed.
"""
from __future__ import annotations

import json
import math
import random
import statistics
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .db import RunStore
from .types import ArchiveEntry, IslandConfig

# ------------------------------------------------------------------
# Default island definitions
# ------------------------------------------------------------------

DEFAULT_ISLANDS: List[IslandConfig] = [
    IslandConfig(
        id="mapping",
        label="Atom Mapping & Step Mapping",
        mutation_target="mapping",
        allowed_lanes=["prompt", "few_shot"],
        eval_tier_filter=None,
        population_cap=50,
        stagnation_threshold=10,
        description="Evolves atom_mapping and select_reaction_type prompts/few-shots",
    ),
    IslandConfig(
        id="reagent_conditions",
        label="Reagent & Condition Inference",
        mutation_target="reagent_conditions",
        allowed_lanes=["prompt", "few_shot"],
        eval_tier_filter=None,
        population_cap=50,
        stagnation_threshold=10,
        description="Evolves predict_missing_reagents and assess_initial_conditions",
    ),
    IslandConfig(
        id="topology",
        label="Topology & Harness Structure",
        mutation_target="topology",
        allowed_lanes=["topology", "harness"],
        eval_tier_filter=None,
        population_cap=30,
        stagnation_threshold=8,
        description="Evolves agent_count, peer_rounds, module enabled flags",
    ),
    IslandConfig(
        id="hard_multistep",
        label="Hard Multi-Step Reactions",
        mutation_target="hard_multistep",
        allowed_lanes=["prompt", "few_shot", "topology"],
        eval_tier_filter="hard",
        population_cap=40,
        stagnation_threshold=12,
        description="All mutation types, evaluated on 9-19 step reactions only",
    ),
]

# Map island mutation targets to the skill call_names they may mutate.
ISLAND_CALL_NAMES: Dict[str, List[str]] = {
    "mapping": ["attempt_atom_mapping", "select_reaction_type"],
    "reagent_conditions": ["predict_missing_reagents", "assess_initial_conditions"],
    "topology": [],  # topology/harness mutators don't target a single call_name
    "hard_multistep": ["propose_mechanism_step"],
}


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ez = math.exp(x)
    return ez / (1.0 + ez)


def _row_to_entry(row: Dict[str, Any]) -> ArchiveEntry:
    """Convert a DB row dict to an ArchiveEntry dataclass."""
    return ArchiveEntry(
        id=row["id"],
        generation=int(row["generation"]),
        island_id=row["island_id"],
        parent_id=row.get("parent_id"),
        archive_inspiration_ids=(
            json.loads(row["archive_inspiration_ids_json"])
            if isinstance(row.get("archive_inspiration_ids_json"), str)
            else row.get("archive_inspiration_ids_json") or []
        ),
        harness_name=row["harness_name"],
        harness_config_json=row["harness_config_json"],
        prompt_bundle_hash=row["prompt_bundle_hash"],
        skill_bundle_hash=row["skill_bundle_hash"],
        few_shot_snapshot_json=row.get("few_shot_snapshot_json", "{}"),
        topology_profile_json=row.get("topology_profile_json", "{}"),
        mutation_type=row["mutation_type"],
        mutation_summary=row.get("mutation_summary", ""),
        mean_quality_score=float(row.get("mean_quality_score", 0.0)),
        weighted_pass_rate=float(row.get("weighted_pass_rate", 0.0)),
        per_subagent_scores_json=(
            json.dumps(row["per_subagent_scores_json"])
            if isinstance(row.get("per_subagent_scores_json"), dict)
            else row.get("per_subagent_scores_json", "{}")
        ),
        total_cost=float(row.get("total_cost", 0.0)),
        eval_run_id=row.get("eval_run_id", ""),
        eval_tier=row.get("eval_tier", "mixed"),
        case_count=int(row.get("case_count", 0)),
        children_count=int(row.get("children_count", 0)),
        score_delta=float(row.get("score_delta", 0.0)),
        migration_history_json=(
            json.dumps(row["migration_history_json"])
            if isinstance(row.get("migration_history_json"), (dict, list))
            else row.get("migration_history_json", "[]")
        ),
        created_at=float(row.get("created_at", 0.0)),
    )


class EvolutionArchive:
    """Searchable archive of harness configurations with island-based evolution."""

    def __init__(self, store: RunStore, islands: Optional[List[IslandConfig]] = None) -> None:
        self.store = store
        self.islands = {isl.id: isl for isl in (islands or DEFAULT_ISLANDS)}

    def is_empty(self) -> bool:
        return self.store.archive_generation_max() < 0

    def max_generation(self) -> int:
        return self.store.archive_generation_max()

    def insert(self, entry: ArchiveEntry) -> str:
        """Insert an archive entry, enforcing island population cap."""
        island = self.islands.get(entry.island_id)
        if island:
            existing = self.store.list_archive_entries(
                island_id=entry.island_id, limit=island.population_cap + 1,
            )
            if len(existing) >= island.population_cap:
                # Evict worst entry (list is sorted by score DESC, so last is worst)
                worst = existing[-1]
                self._delete_entry(worst["id"])

        return self.store.insert_archive_entry(
            generation=entry.generation,
            island_id=entry.island_id,
            parent_id=entry.parent_id,
            archive_inspiration_ids=entry.archive_inspiration_ids,
            harness_name=entry.harness_name,
            harness_config_json=entry.harness_config_json,
            prompt_bundle_hash=entry.prompt_bundle_hash,
            skill_bundle_hash=entry.skill_bundle_hash,
            few_shot_snapshot_json=entry.few_shot_snapshot_json,
            topology_profile_json=entry.topology_profile_json,
            mutation_type=entry.mutation_type,
            mutation_summary=entry.mutation_summary,
            mean_quality_score=entry.mean_quality_score,
            weighted_pass_rate=entry.weighted_pass_rate,
            per_subagent_scores_json=entry.per_subagent_scores_json,
            total_cost=entry.total_cost,
            eval_run_id=entry.eval_run_id,
            eval_tier=entry.eval_tier,
            case_count=entry.case_count,
            children_count=entry.children_count,
            score_delta=entry.score_delta,
            migration_history_json=entry.migration_history_json,
        )

    def get(self, entry_id: str) -> Optional[ArchiveEntry]:
        row = self.store.get_archive_entry(entry_id)
        return _row_to_entry(row) if row else None

    def list_island(self, island_id: str, *, limit: int = 100) -> List[ArchiveEntry]:
        rows = self.store.list_archive_entries(island_id=island_id, limit=limit)
        return [_row_to_entry(r) for r in rows]

    def island_best(self, island_id: str) -> Optional[ArchiveEntry]:
        entries = self.list_island(island_id, limit=1)
        return entries[0] if entries else None

    def island_best_score(self, island_id: str) -> float:
        return self.store.island_best_score(island_id)

    def global_elites(self, limit: int = 10) -> List[ArchiveEntry]:
        rows = self.store.list_archive_entries(island_id=None, limit=limit)
        return [_row_to_entry(r) for r in rows]

    # ------------------------------------------------------------------
    # Weighted parent selection (ShinkaEvolve WeightedSamplingStrategy)
    # ------------------------------------------------------------------

    def select_parent(self, island_id: str, *, rng: Optional[random.Random] = None) -> ArchiveEntry:
        """Sample a parent from the island archive.

        Weight = sigmoid((score - median) / MAD) * (1 / (1 + children_count))
        This balances exploitation (high score) with exploration (underexplored).
        """
        entries = self.list_island(island_id, limit=200)
        if not entries:
            raise ValueError(f"Island '{island_id}' has no archive entries")
        if len(entries) == 1:
            return entries[0]

        rng = rng or random.Random()
        scores = [e.mean_quality_score for e in entries]
        median_score = statistics.median(scores)
        deviations = [abs(s - median_score) for s in scores]
        mad = statistics.median(deviations) if deviations else 0.01
        if mad < 1e-9:
            mad = 0.01

        weights: List[float] = []
        for entry in entries:
            z = (entry.mean_quality_score - median_score) / mad
            perf_weight = _sigmoid(z)
            novelty_weight = 1.0 / (1.0 + entry.children_count)
            weights.append(perf_weight * novelty_weight)

        total = sum(weights)
        if total < 1e-12:
            return rng.choice(entries)
        probs = [w / total for w in weights]
        selected = rng.choices(entries, weights=probs, k=1)[0]
        return selected

    def increment_children(self, entry_id: str) -> None:
        self.store.increment_archive_children(entry_id)

    # ------------------------------------------------------------------
    # Migration
    # ------------------------------------------------------------------

    def attempt_migration(
        self,
        source_island: str,
        target_island: str,
        entry: ArchiveEntry,
    ) -> bool:
        """Migrate entry to target island only if it beats the target's best score."""
        target_best = self.island_best_score(target_island)
        if entry.mean_quality_score <= target_best:
            return False

        # Create a copy on the target island
        migration_history = json.loads(entry.migration_history_json) if isinstance(
            entry.migration_history_json, str
        ) else list(entry.migration_history_json or [])
        migration_history.append({
            "from_island": source_island,
            "to_island": target_island,
            "generation": entry.generation,
            "timestamp": time.time(),
        })

        migrated_id = self.store.insert_archive_entry(
            generation=entry.generation,
            island_id=target_island,
            parent_id=entry.id,
            archive_inspiration_ids=entry.archive_inspiration_ids,
            harness_name=entry.harness_name,
            harness_config_json=entry.harness_config_json,
            prompt_bundle_hash=entry.prompt_bundle_hash,
            skill_bundle_hash=entry.skill_bundle_hash,
            few_shot_snapshot_json=entry.few_shot_snapshot_json,
            topology_profile_json=entry.topology_profile_json,
            mutation_type="migration",
            mutation_summary=f"migrated from {source_island} -> {target_island}",
            mean_quality_score=entry.mean_quality_score,
            weighted_pass_rate=entry.weighted_pass_rate,
            per_subagent_scores_json=entry.per_subagent_scores_json,
            total_cost=entry.total_cost,
            eval_run_id=entry.eval_run_id,
            eval_tier=entry.eval_tier,
            case_count=entry.case_count,
            score_delta=entry.mean_quality_score - target_best,
            migration_history_json=json.dumps(migration_history),
        )
        self.store.insert_archive_migration(
            entry_id=migrated_id,
            from_island=source_island,
            to_island=target_island,
            generation=entry.generation,
        )
        return True

    # ------------------------------------------------------------------
    # Stagnation detection
    # ------------------------------------------------------------------

    def stagnation_check(self, island_id: str, *, window: int = 5) -> bool:
        """Return True if island has not improved in the last *window* generations."""
        entries = self.list_island(island_id, limit=200)
        if len(entries) < 2:
            return False
        current_gen = max(e.generation for e in entries)
        best_score = max(e.mean_quality_score for e in entries)
        recent_best = max(
            (e.mean_quality_score for e in entries if e.generation > current_gen - window),
            default=0.0,
        )
        older_best = max(
            (e.mean_quality_score for e in entries if e.generation <= current_gen - window),
            default=0.0,
        )
        return recent_best <= older_best

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def seed_from_leaderboard(self, leaderboard_dir: Path) -> int:
        """Seed the archive from existing leaderboard JSON files.

        Returns the number of entries created.
        """
        count = 0
        for lb_path in sorted(leaderboard_dir.glob("leaderboard_*.json")):
            try:
                data = json.loads(lb_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            items = data if isinstance(data, list) else [data]
            for item_wrapper in items:
                item = item_wrapper.get("item", item_wrapper) if isinstance(item_wrapper, dict) else item_wrapper
                if not isinstance(item, dict):
                    continue
                score = float(item.get("mean_quality_score", 0.0))
                if score <= 0:
                    continue
                for island_id in self.islands:
                    self.store.insert_archive_entry(
                        generation=0,
                        island_id=island_id,
                        parent_id=None,
                        harness_name=str(item.get("harness_name", "default")),
                        harness_config_json="{}",
                        prompt_bundle_hash=str(item.get("prompt_bundle_hash", "")),
                        skill_bundle_hash=str(item.get("skill_bundle_hash", "")),
                        mutation_type="seed",
                        mutation_summary=f"seeded from {lb_path.name}",
                        mean_quality_score=score,
                        weighted_pass_rate=float(item.get("weighted_pass_rate", 0.0)),
                        per_subagent_scores_json=json.dumps(
                            item.get("per_subagent_scores", {})
                        ),
                        total_cost=float(item.get("total_cost", 0.0)),
                        eval_tier=str(item.get("eval_tier", "mixed")),
                        case_count=int(item.get("case_count", 0)),
                    )
                    count += 1
        return count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _delete_entry(self, entry_id: str) -> None:
        """Remove an archive entry (used for population cap eviction)."""
        with self.store._lock, self.store._connect() as conn:
            conn.execute("DELETE FROM archive_entries WHERE id = ?", (entry_id,))
            conn.commit()
