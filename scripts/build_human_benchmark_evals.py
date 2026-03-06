#!/usr/bin/env python3
"""Build FlowER-derived eval artifacts from the ranked default dataset."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mechanistic_agent.flower_curriculum import DEFAULT_DATASET_PATH, SOURCE_LABEL, eval_case_from_case

TRAINING_DIR = PROJECT_ROOT / "training_data"
FLOWER_SOURCE_PATH = TRAINING_DIR / "flower_mechanisms_100.json"
EVAL_SET_PATH = TRAINING_DIR / "eval_set.json"
EVAL_TIERS_PATH = TRAINING_DIR / "eval_tiers.json"
QUALITY_REPORT_PATH = TRAINING_DIR / "eval_quality_report.json"

TIER_BANDS: dict[str, tuple[int, int]] = {
    "easy": (1, 2),
    "medium": (3, 3),
    "hard": (4, 99),
}


def _load_flower_cases() -> List[Dict[str, Any]]:
    source_path = DEFAULT_DATASET_PATH if DEFAULT_DATASET_PATH.exists() else FLOWER_SOURCE_PATH
    if not source_path.exists():
        raise SystemExit(f"Missing FlowER source file: {source_path}")
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit("flower_mechanisms_100.json must contain a JSON list")
    return [eval_case_from_case(case) for case in payload if isinstance(case, dict)]


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


def _quality_report(entries: List[Dict[str, Any]], tiers: Dict[str, List[str]]) -> Dict[str, Any]:
    by_steps = Counter(int(entry.get("n_mechanistic_steps") or 0) for entry in entries)
    tier_step_ranges = {
        name: {
            "step_range": TIER_BANDS[name],
            "count": len(ids),
            "first_case_id": ids[0] if ids else None,
            "last_case_id": ids[-1] if ids else None,
        }
        for name, ids in tiers.items()
    }
    return {
        "_meta": {
            "source": SOURCE_LABEL,
            "generated_by": "scripts/build_human_benchmark_evals.py",
            "tier_policy": {
                "bands": TIER_BANDS,
                "ordering": "Preserve ranked order from training_data/flower_mechanisms_100.json",
            },
        },
        "summary": {
            "total_rows": len(entries),
            "step_count_distribution": dict(sorted(by_steps.items())),
            "tier_summary": tier_step_ranges,
        },
        "issues": [],
        "suggestions": [
            "Use training_data/flower_mechanisms_100.json as the upstream source of truth for default eval updates.",
            "Curriculum ordering is preserved in eval_set.json and tier membership is now full-band rather than fixed 10-case quotas.",
            "scripts/evolve_harness.py reads training_data/flower_mechanism_index.jsonl directly and no longer depends on eval_tiers.json.",
        ],
    }


def main() -> None:
    entries = _load_flower_cases()
    tiers = _build_tiers(entries)
    report = _quality_report(entries, tiers)

    tier_payload = {
        "_meta": {
            "description": "Step-band groupings derived from the ranked FlowER 100 defaults.",
            "source": SOURCE_LABEL,
            "difficulty_criteria": {
                "easy": "Mechanistic steps 1-2",
                "medium": "Mechanistic steps 3",
                "hard": "Mechanistic steps 4+",
            },
            "ordering": "IDs remain in ranked order from eval_set.json",
        },
        **tiers,
    }

    EVAL_SET_PATH.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    EVAL_TIERS_PATH.write_text(json.dumps(tier_payload, indent=2) + "\n", encoding="utf-8")
    QUALITY_REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {EVAL_SET_PATH} ({len(entries)} entries)")
    print(f"Wrote {EVAL_TIERS_PATH}")
    print(f"Wrote {QUALITY_REPORT_PATH}")


if __name__ == "__main__":
    main()
