#!/usr/bin/env python3
"""Build a practice eval set for contributors from FlowER train.txt.

The practice set is disjoint from both the dev eval set (eval_set.json)
and the leaderboard holdout (eval_set_holdout.json).  It draws from the
same FlowER train.txt source so contributors can test their changes
locally without touching the real eval or leaderboard data.

Usage:
    python scripts/build_practice_eval.py

Prerequisites:
    - FlowER train.txt at ../FlowER/data/flower_new_dataset/train.txt
    - SQLite lookup cache (run `python main.py curriculum build-lookup` first)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mechanistic_agent.flower_curriculum import (
    ConversionError,
    DEFAULT_FLOWER_INPUT,
    DEFAULT_LOOKUP_CACHE,
    SOURCE_LABEL,
    convert_mechanism_id_to_case,
    eval_case_from_case,
)

from mechanistic_agent.data_paths import (
    flower_mechanism_index_path,
    holdout_eval_set_path,
    repo_training_dir,
)

TRAINING_DIR = repo_training_dir(PROJECT_ROOT)
INDEX_PATH = flower_mechanism_index_path(PROJECT_ROOT)
EVAL_SET_PATH = TRAINING_DIR / "eval_set.json"
HOLDOUT_PATH = holdout_eval_set_path(PROJECT_ROOT)

PRACTICE_DIR = TRAINING_DIR / "practice_eval"
PRACTICE_SET_PATH = PRACTICE_DIR / "practice_set.json"
PRACTICE_TIERS_PATH = PRACTICE_DIR / "practice_tiers.json"

TARGET_COUNT = 20

TIER_BANDS: dict[str, tuple[int, int]] = {
    "easy": (1, 2),
    "medium": (3, 3),
    "hard": (4, 99),
}


def _load_excluded_ids() -> set[str]:
    """Collect all IDs already used by dev eval and holdout sets."""
    excluded: set[str] = set()
    if EVAL_SET_PATH.exists():
        for entry in json.loads(EVAL_SET_PATH.read_text(encoding="utf-8")):
            excluded.add(str(entry.get("id") or ""))
    if HOLDOUT_PATH.exists():
        for entry in json.loads(HOLDOUT_PATH.read_text(encoding="utf-8")):
            excluded.add(str(entry.get("id") or ""))
    excluded.discard("")
    return excluded


def _load_candidate_ids(excluded: set[str]) -> List[Dict[str, Any]]:
    """Read the JSONL index and return candidate rows not in excluded sets."""
    if not INDEX_PATH.exists():
        raise SystemExit(
            f"Missing {INDEX_PATH}\n"
            "Run: python main.py curriculum build-lookup && "
            "python scripts/build_flower_mechanism_dataset.py"
        )
    candidates: List[Dict[str, Any]] = []
    with INDEX_PATH.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            mechanism_id = int(row["mechanism_id"])
            flower_id = f"flower_{mechanism_id:06d}"
            if flower_id in excluded:
                continue
            candidates.append(row)
    return candidates


def _sample_across_steps(
    candidates: List[Dict[str, Any]], target: int
) -> List[Dict[str, Any]]:
    """Sample reactions spread across step counts for difficulty coverage.

    Over-samples by 5x to account for conversion failures.
    """
    by_steps: dict[int, List[Dict[str, Any]]] = {}
    for row in candidates:
        sc = int(row["step_count"])
        by_steps.setdefault(sc, []).append(row)

    # Take from step counts 1-6 to cover easy/medium/hard tiers
    target_steps = sorted(sc for sc in by_steps if sc <= 6)
    if not target_steps:
        target_steps = sorted(by_steps.keys())[:6]

    # Allocate per step count with generous over-sampling for conversion failures
    # Higher step counts have much higher failure rates, so give them more candidates
    per_step_target = max(1, target // len(target_steps))
    selected: List[Dict[str, Any]] = []
    for sc in target_steps:
        pool = by_steps[sc]
        # Higher steps need more over-sampling due to conversion failures
        oversample_factor = 10 if sc <= 2 else 50
        n_candidates = min(len(pool), per_step_target * oversample_factor)
        step = max(1, len(pool) // n_candidates)
        picks = pool[::step][:n_candidates]
        selected.extend(picks)

    return selected


def _convert_candidate(row: Dict[str, Any]) -> Dict[str, Any] | None:
    """Convert a JSONL index row to a full eval case."""
    mechanism_id = int(row["mechanism_id"])
    try:
        case = convert_mechanism_id_to_case(
            mechanism_id,
            input_path=DEFAULT_FLOWER_INPUT,
            cache_path=DEFAULT_LOOKUP_CACHE,
        )
    except (ConversionError, Exception) as exc:
        print(f"  skip {mechanism_id}: {exc}")
        return None

    case["id"] = f"flower_{mechanism_id:06d}"
    case["name"] = f"FlowER practice mechanism {mechanism_id}"
    case["description"] = f"Practice eval case from FlowER train.txt group {mechanism_id}."
    tags = list(case.get("tags") or [])
    if "practice_eval" not in tags:
        tags.append("practice_eval")
    case["tags"] = tags
    return eval_case_from_case(case)


def _build_tiers(entries: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    tiers: Dict[str, List[str]] = {"easy": [], "medium": [], "hard": []}
    for entry in entries:
        case_id = str(entry.get("id") or "")
        step_count = int(entry.get("n_mechanistic_steps") or 0)
        for tier_name, (lower, upper) in TIER_BANDS.items():
            if lower <= step_count <= upper:
                tiers[tier_name].append(case_id)
                break
    return tiers


def main() -> None:
    excluded = _load_excluded_ids()
    print(f"Excluded IDs: {len(excluded)} (eval={len([i for i in excluded if not i.startswith('flower_test_')])}, holdout={len([i for i in excluded if i.startswith('flower_test_')])})")

    candidates = _load_candidate_ids(excluded)
    print(f"Available candidates: {len(candidates)}")

    # Group candidates by step count and convert per-group
    by_steps: dict[int, List[Dict[str, Any]]] = {}
    for row in candidates:
        sc = int(row["step_count"])
        if sc <= 6:  # Cap at 6 steps for practice set
            by_steps.setdefault(sc, []).append(row)

    target_steps = sorted(by_steps.keys())
    per_step_target = max(3, -(-TARGET_COUNT // max(1, len(target_steps))))  # ceiling division
    entries: List[Dict[str, Any]] = []
    conversion_failures = 0

    for sc in target_steps:
        pool = by_steps[sc]
        step_entries: List[Dict[str, Any]] = []
        # Try candidates evenly spread across the pool
        stride = max(1, len(pool) // (per_step_target * 20))
        for row in pool[::stride]:
            case = _convert_candidate(row)
            if case is not None:
                step_entries.append(case)
                if len(step_entries) >= per_step_target:
                    break
            else:
                conversion_failures += 1
        entries.extend(step_entries)
        print(f"  step={sc}: {len(step_entries)} converted from {len(pool)} available")

    # Trim to target
    entries = entries[:TARGET_COUNT]
    print(f"Successfully converted: {len(entries)} (failures: {conversion_failures})")

    tiers = _build_tiers(entries)
    tier_payload = {
        "_meta": {
            "description": "Practice eval tiers (NOT the leaderboard eval set).",
            "source": SOURCE_LABEL,
            "difficulty_criteria": {
                "easy": "Mechanistic steps 1-2",
                "medium": "Mechanistic steps 3",
                "hard": "Mechanistic steps 4+",
            },
        },
        **tiers,
    }

    step_dist = Counter(int(e.get("n_mechanistic_steps", 0)) for e in entries)
    print(f"Step distribution: {dict(sorted(step_dist.items()))}")
    print(f"Tiers: easy={len(tiers['easy'])}, medium={len(tiers['medium'])}, hard={len(tiers['hard'])}")

    PRACTICE_DIR.mkdir(parents=True, exist_ok=True)
    PRACTICE_SET_PATH.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    PRACTICE_TIERS_PATH.write_text(json.dumps(tier_payload, indent=2) + "\n", encoding="utf-8")

    print(f"\nWrote {PRACTICE_SET_PATH} ({len(entries)} entries)")
    print(f"Wrote {PRACTICE_TIERS_PATH}")


if __name__ == "__main__":
    main()
