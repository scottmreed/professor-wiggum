#!/usr/bin/env python3
"""Build an isolated leaderboard holdout eval suite from FlowER test.txt."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mechanistic_agent.flower_curriculum import (  # noqa: E402
    ConversionError,
    build_curriculum_index,
    build_lookup_cache,
    convert_mechanism_id_to_case,
    known_mechanism_from_case,
)
from mechanistic_agent.flower_rendering import render_cases  # noqa: E402

DEFAULT_INPUT_PATH = PROJECT_ROOT.parent / "FlowER" / "data" / "flower_new_dataset" / "test.txt"
DEFAULT_CACHE_PATH = PROJECT_ROOT / "data" / "flower_test_lookup.sqlite"
HOLDOUT_DIR = PROJECT_ROOT / "training_data" / "leaderboard_holdout"
DEFAULT_EVAL_SET_PATH = HOLDOUT_DIR / "eval_set_holdout.json"
DEFAULT_BUCKET_PATH = HOLDOUT_DIR / "eval_step_buckets_holdout.json"
DEFAULT_REPORT_PATH = HOLDOUT_DIR / "eval_quality_report_holdout.json"
DEFAULT_INDEX_PATH = HOLDOUT_DIR / "flow_index_holdout.jsonl"
DEFAULT_INDEX_REPORT_PATH = HOLDOUT_DIR / "flow_index_report_holdout.json"
DEFAULT_PNG_DIR = HOLDOUT_DIR / "pngs"
SOURCE_LABEL = "FlowER flower_new_dataset test.txt"


def _json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _split_terciles(items: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    n = len(items)
    if n <= 0:
        return [], [], []
    if n <= 2:
        return list(items), [], []
    first_end = max(1, int(math.ceil(n / 3)))
    second_end = max(first_end + 1, int(math.ceil(2 * n / 3)))
    easy = list(items[:first_end])
    medium = list(items[first_end:second_end])
    hard = list(items[second_end:])
    if not medium:
        medium = list(items[first_end:first_end + 1])
    if not hard:
        hard = list(items[-1:])
    return easy, medium, hard


def _select_with_difficulty_mix(items: Sequence[Dict[str, Any]], target: int) -> List[Dict[str, Any]]:
    pool = list(items)
    if target <= 0 or not pool:
        return []
    if target >= len(pool):
        out: List[Dict[str, Any]] = []
        for item in pool:
            clone = dict(item)
            clone["difficulty_band"] = clone.get("difficulty_band") or "mixed"
            out.append(clone)
        return out
    if target == 1:
        mid = len(pool) // 2
        selected = dict(pool[mid])
        selected["difficulty_band"] = "medium"
        return [selected]
    if target == 2:
        first = dict(pool[0])
        second = dict(pool[-1])
        first["difficulty_band"] = "easy"
        second["difficulty_band"] = "hard"
        if first.get("id") == second.get("id"):
            return [first]
        return [first, second]

    easy, medium, hard = _split_terciles(pool)
    queues: Dict[str, List[Dict[str, Any]]] = {
        "easy": list(easy),
        "medium": list(medium),
        "hard": list(hard),
    }
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    while len(out) < target:
        progressed = False
        for band in ("easy", "medium", "hard"):
            queue = queues[band]
            while queue:
                candidate = dict(queue.pop(0))
                cid = str(candidate.get("id") or "")
                if not cid or cid in seen:
                    continue
                candidate["difficulty_band"] = band
                seen.add(cid)
                out.append(candidate)
                progressed = True
                break
            if len(out) >= target:
                break
        if not progressed:
            break
    return out[:target]


def _build_holdout_eval_case(base_case: Dict[str, Any], *, source_label: str) -> Dict[str, Any]:
    out = dict(base_case)
    out["source"] = source_label
    out["n_mechanistic_steps"] = int(
        out.get("n_mechanistic_steps")
        or len((((out.get("verified_mechanism") or {}).get("steps")) or []))
        or 0
    )
    known = known_mechanism_from_case(base_case)
    known["source"] = source_label
    out["known_mechanism"] = known
    return out


def build_holdout_suite(
    *,
    input_path: Path,
    cache_path: Path,
    target_per_step: int,
    only_missing_pngs: bool = False,
) -> Dict[str, Any]:
    entries, index_report = build_curriculum_index(input_path=input_path)
    build_lookup_cache(input_path=input_path, cache_path=cache_path)

    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    conversion_failures: Counter[str] = Counter()
    raw_distribution: Counter[int] = Counter()
    for entry in entries:
        row = entry.as_dict()
        step_count = int(row["step_count"])
        raw_distribution[step_count] += 1
        try:
            converted = convert_mechanism_id_to_case(
                int(row["mechanism_id"]),
                input_path=input_path,
                cache_path=cache_path,
            )
        except ConversionError as exc:
            conversion_failures[exc.reason] += 1
            continue

        mechanism_id = int(row["mechanism_id"])
        holdout_case = dict(converted)
        holdout_case["id"] = f"flower_test_{mechanism_id:06d}"
        holdout_case["name"] = f"FlowER test mechanism {mechanism_id}"
        holdout_case["description"] = f"Converted from FlowER test.txt group {mechanism_id}."
        holdout_case["source"] = SOURCE_LABEL
        tags = list(holdout_case.get("tags") or [])
        tags = [str(tag) for tag in tags if str(tag).strip()]
        if "leaderboard_holdout" not in tags:
            tags.append("leaderboard_holdout")
        holdout_case["tags"] = tags

        grouped[step_count].append(
            {
                "id": str(holdout_case.get("id") or ""),
                "case": holdout_case,
                "step_count": step_count,
                "mechanism_id": mechanism_id,
                "global_rank": int(row["global_rank"]),
                "rank_within_step_count": int(row["rank_within_step_count"]),
            }
        )

    selected_rows: List[Dict[str, Any]] = []
    bucket_payload: Dict[str, Any] = {}
    selected_distribution: Counter[int] = Counter()
    convertible_distribution: Dict[int, int] = {}

    for step_count in sorted(grouped):
        candidates = grouped[step_count]
        convertible_distribution[step_count] = len(candidates)
        target = min(int(target_per_step), len(candidates))
        selected = _select_with_difficulty_mix(candidates, target)
        selected_ids: List[str] = []
        for row in selected:
            selected_rows.append(row)
            selected_ids.append(str(row["id"]))
            selected_distribution[step_count] += 1

        bucket_payload[str(step_count)] = {
            "step_count": int(step_count),
            "target_cases": int(target),
            "weight": int(step_count),
            "case_ids": selected_ids,
            "convertible_count": int(len(candidates)),
        }

    selected_rows.sort(key=lambda row: (int(row["step_count"]), int(row["global_rank"])))

    holdout_eval_cases: List[Dict[str, Any]] = []
    render_cases_payload: List[Dict[str, Any]] = []
    metadata_by_case_id: Dict[str, Dict[str, Any]] = {}
    for row in selected_rows:
        case_payload = _build_holdout_eval_case(row["case"], source_label=SOURCE_LABEL)
        case_payload["leaderboard_holdout"] = {
            "suite": "leaderboard_holdout",
            "step_count": int(row["step_count"]),
            "global_rank": int(row["global_rank"]),
            "rank_within_step_count": int(row["rank_within_step_count"]),
            "difficulty_band": str(row.get("difficulty_band") or "mixed"),
        }
        holdout_eval_cases.append(case_payload)
        render_cases_payload.append(case_payload)
        metadata_by_case_id[str(case_payload["id"])] = {
            "suite": "leaderboard_holdout",
            "step_count": int(row["step_count"]),
            "global_rank": int(row["global_rank"]),
            "rank_within_step_count": int(row["rank_within_step_count"]),
            "difficulty_band": str(row.get("difficulty_band") or "mixed"),
        }

    render_index = render_cases(
        cases=render_cases_payload,
        output_dir=DEFAULT_PNG_DIR,
        metadata_by_case_id=metadata_by_case_id,
        only_missing=bool(only_missing_pngs),
        source=str(input_path),
    )

    step_bucket_output = {
        "_meta": {
            "description": "Leaderboard-only holdout buckets from FlowER test.txt.",
            "suite": "leaderboard_holdout",
            "source": SOURCE_LABEL,
            "aggregate_weighting": "linear_step_count",
            "aggregate_gate_cases": 6,
            "target_cases_per_step_max": int(target_per_step),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "step_buckets": bucket_payload,
    }

    quality_report = {
        "_meta": {
            "suite": "leaderboard_holdout",
            "source": SOURCE_LABEL,
            "generated_by": "scripts/build_flower_holdout_leaderboard_set.py",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "selection_policy": "per step_count: min(12, convertible_count) with rank-tercile difficulty mixing",
        },
        "summary": {
            "raw_step_distribution": {str(k): int(raw_distribution[k]) for k in sorted(raw_distribution)},
            "convertible_step_distribution": {str(k): int(convertible_distribution.get(k, 0)) for k in sorted(convertible_distribution)},
            "selected_step_distribution": {str(k): int(selected_distribution[k]) for k in sorted(selected_distribution)},
            "selected_case_count": int(len(holdout_eval_cases)),
            "step_bucket_count": int(len(bucket_payload)),
            "png_rendered_count": int(render_index.get("rendered_count") or 0),
        },
        "conversion_failures_by_reason": dict(sorted(conversion_failures.items())),
        "notes": [
            "This holdout suite is leaderboard-only and not used by UI examples or evolve_harness.",
            "Sparse high-step buckets are expected due conversion constraints in deterministic step conversion.",
        ],
    }

    _write_jsonl((entry.as_dict() for entry in entries), DEFAULT_INDEX_PATH)
    _json_dump(index_report, DEFAULT_INDEX_REPORT_PATH)
    _json_dump(holdout_eval_cases, DEFAULT_EVAL_SET_PATH)
    _json_dump(step_bucket_output, DEFAULT_BUCKET_PATH)
    _json_dump(quality_report, DEFAULT_REPORT_PATH)

    return {
        "index_count": len(entries),
        "holdout_case_count": len(holdout_eval_cases),
        "step_bucket_count": len(bucket_payload),
        "paths": {
            "eval_set": str(DEFAULT_EVAL_SET_PATH),
            "step_buckets": str(DEFAULT_BUCKET_PATH),
            "quality_report": str(DEFAULT_REPORT_PATH),
            "index_jsonl": str(DEFAULT_INDEX_PATH),
            "index_report": str(DEFAULT_INDEX_REPORT_PATH),
            "png_index": str(DEFAULT_PNG_DIR / "index.json"),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build isolated leaderboard holdout artifacts from FlowER test.txt")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="Path to FlowER test.txt")
    parser.add_argument("--lookup-cache", default=str(DEFAULT_CACHE_PATH), help="Path to lookup-cache sqlite")
    parser.add_argument("--target-per-step", type=int, default=12, help="Max selected reactions per unique step count")
    parser.add_argument("--only-missing-pngs", action="store_true", help="Only render missing PNGs")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = build_holdout_suite(
        input_path=Path(args.input),
        cache_path=Path(args.lookup_cache),
        target_per_step=max(1, int(args.target_per_step)),
        only_missing_pngs=bool(args.only_missing_pngs),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
