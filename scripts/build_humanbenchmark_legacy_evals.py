#!/usr/bin/env python3
"""Build legacy HumanBenchmark eval artifacts into ignored local-only paths.

This script preserves the old CSV-based eval generation workflow for local use.
It intentionally writes under training_data/local_legacy/humanbenchmark/ and is
not used by the default menu or eval harness anymore.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TierQuota:
    step_count: int
    count: int


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "training_data"
LEGACY_DIR = TRAINING_DIR / "local_legacy" / "humanbenchmark"
CSV_PATH = LEGACY_DIR / "350_HumanBenchmark.csv"
EVAL_SET_PATH = LEGACY_DIR / "eval_set.json"
EVAL_TIERS_PATH = LEGACY_DIR / "eval_tiers.json"
QUALITY_REPORT_PATH = LEGACY_DIR / "eval_quality_report.json"

TIER_BANDS: dict[str, tuple[int, int]] = {
    "easy": (1, 2),
    "medium": (3, 5),
    "hard": (6, 7),
}

TIER_QUOTAS: dict[str, list[TierQuota]] = {
    "easy": [TierQuota(1, 5), TierQuota(2, 5)],
    "medium": [TierQuota(3, 3), TierQuota(4, 4), TierQuota(5, 3)],
    "hard": [TierQuota(6, 7), TierQuota(7, 3)],
}

STEP_COLUMNS = [f"Step {i} Target" for i in range(1, 8)]


def _parse_steps(value: str) -> int:
    text = str(value or "").strip()
    if not text:
        return 0
    return int(float(text))


def _parse_temperature(value: str) -> float | None:
    text = str(value or "").strip()
    if not text or text.lower() == "n/a":
        return None
    match = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if not match:
        return None
    return float(match.group(1))


def _split_species(species: str) -> list[str]:
    return [item.strip() for item in str(species or "").split(".") if item.strip()]


def _entry_from_row(row: dict[str, Any], idx: int) -> dict[str, Any]:
    min_steps = _parse_steps(str(row.get("Min Steps", "")))
    step_targets = [str(row.get(col, "") or "").strip() for col in STEP_COLUMNS]
    known_steps = []
    for step_index in range(1, 8):
        target = step_targets[step_index - 1]
        if target:
            known_steps.append({"step_index": step_index, "target_smiles": target})
    missing_required_steps = [i for i in range(1, min_steps + 1) if not step_targets[i - 1]]

    entry: dict[str, Any] = {
        "id": f"hb350_{idx:03d}",
        "name": f"HumanBenchmark Reaction {idx:03d}",
        "description": f"HumanBenchmark row {idx}. Min steps: {min_steps}. Source: 350_HumanBenchmark.csv.",
        "starting_materials": _split_species(str(row.get("Reactants", "") or "").strip()),
        "products": _split_species(str(row.get("Final Product", "") or "").strip()),
        "temperature_celsius": _parse_temperature(str(row.get("Temp (≠ RT)", ""))),
        "ph": None,
        "source": "350_HumanBenchmark.csv",
        "n_mechanistic_steps": min_steps,
        "known_mechanism": {
            "source": "350_HumanBenchmark.csv",
            "min_steps": min_steps,
            "citation": str(row.get("Clayden, 2nd Ed Textbook Citation", "") or "").strip(),
            "steps": known_steps,
        },
        "_quality": {"row_index": idx, "missing_required_steps": missing_required_steps},
    }
    return entry


def _select_tier_ids(entries: list[dict[str, Any]]) -> tuple[dict[str, list[str]], dict[str, Any]]:
    by_steps: dict[int, list[dict[str, Any]]] = defaultdict(list)
    complete_pool: list[dict[str, Any]] = []
    for entry in entries:
        if ((entry.get("_quality") or {}).get("missing_required_steps") or []):
            continue
        complete_pool.append(entry)
        by_steps[int(entry.get("n_mechanistic_steps") or 0)].append(entry)
    for rows in by_steps.values():
        rows.sort(key=lambda item: int((item.get("_quality") or {}).get("row_index") or 0))

    selected: dict[str, list[str]] = {"easy": [], "medium": [], "hard": []}
    diagnostics: dict[str, Any] = {"replacement_candidates": {}}
    for tier_name, quotas in TIER_QUOTAS.items():
        tier_ids: list[str] = []
        used: set[str] = set()
        for quota in quotas:
            for entry in by_steps.get(quota.step_count, []):
                eid = str(entry.get("id") or "")
                if not eid or eid in used:
                    continue
                tier_ids.append(eid)
                used.add(eid)
                if len([rid for rid in tier_ids if rid in {str(row.get('id') or '') for row in by_steps.get(quota.step_count, [])}]) >= quota.count:
                    break
        if tier_name == "hard" and len(tier_ids) < 10:
            for entry in by_steps.get(6, []):
                eid = str(entry.get("id") or "")
                if not eid or eid in used:
                    continue
                tier_ids.append(eid)
                used.add(eid)
                if len(tier_ids) >= 10:
                    break
        selected[tier_name] = tier_ids[:10]

        replacements: list[str] = []
        lo, hi = TIER_BANDS[tier_name]
        for entry in complete_pool:
            steps = int(entry.get("n_mechanistic_steps") or 0)
            eid = str(entry.get("id") or "")
            if lo <= steps <= hi and eid not in selected[tier_name]:
                replacements.append(eid)
        diagnostics["replacement_candidates"][tier_name] = replacements[:25]
    return selected, diagnostics


def _quality_report(entries: list[dict[str, Any]], tiers: dict[str, list[str]], diagnostics: dict[str, Any]) -> dict[str, Any]:
    by_steps = Counter(int(entry.get("n_mechanistic_steps") or 0) for entry in entries)
    issues: list[dict[str, Any]] = []
    for entry in entries:
        missing = list(((entry.get("_quality") or {}).get("missing_required_steps") or []))
        if missing:
            issues.append(
                {
                    "id": entry.get("id"),
                    "row_index": (entry.get("_quality") or {}).get("row_index"),
                    "min_steps": entry.get("n_mechanistic_steps"),
                    "issue": "missing_required_step_targets",
                    "missing_steps": missing,
                }
            )
    return {
        "_meta": {
            "source": "350_HumanBenchmark.csv",
            "generated_by": "scripts/build_humanbenchmark_legacy_evals.py",
        },
        "summary": {
            "total_rows": len(entries),
            "step_count_distribution": dict(sorted(by_steps.items())),
        },
        "tier_selection": {"selected_ids": tiers, **diagnostics},
        "issues": issues,
    }


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"Missing local HumanBenchmark CSV: {CSV_PATH}")
    LEGACY_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    entries = [_entry_from_row(row, idx) for idx, row in enumerate(rows, start=1)]
    tiers, diagnostics = _select_tier_ids(entries)
    report = _quality_report(entries, tiers, diagnostics)
    eval_entries = []
    for entry in entries:
        row = dict(entry)
        row.pop("_quality", None)
        eval_entries.append(row)
    tier_payload = {
        "_meta": {
            "description": "Fixed deterministic 10-case tiers selected from local HumanBenchmark legacy data.",
            "source": "350_HumanBenchmark.csv",
            "difficulty_criteria": {
                "easy": "Min Steps 1-2",
                "medium": "Min Steps 3-5",
                "hard": "Min Steps 6-7",
            },
        },
        **tiers,
    }
    EVAL_SET_PATH.write_text(json.dumps(eval_entries, indent=2) + "\n", encoding="utf-8")
    EVAL_TIERS_PATH.write_text(json.dumps(tier_payload, indent=2) + "\n", encoding="utf-8")
    QUALITY_REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {EVAL_SET_PATH}")
    print(f"Wrote {EVAL_TIERS_PATH}")
    print(f"Wrote {QUALITY_REPORT_PATH}")


if __name__ == "__main__":
    main()
