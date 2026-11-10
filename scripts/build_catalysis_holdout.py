#!/usr/bin/env python3
"""Build the catalysis holdout from FlowER test.txt (PRD_catalysis_handling R5.3).

A reserve, not a leaderboard tier: nothing ranks on it and no tier map, eval set or
leaderboard references it. It exists so a catalysis board can be added later without the
cases having been seen during development. It is written next to the main holdout under
``holdout_dir()/catalysis/`` (the bulk data checkout, excluded from Docker images and never
committed). Cases carry ``flower_test_`` ids and ``purpose = leaderboard_holdout``, so the
dev-set builders, ``evolve_harness.py`` and the non-official runsets refuse them.

Selection mirrors the train-side development set (``mechanistic_agent.catalysis_dataset``):
mechanisms in which a d-block atom changes its bonded neighbours, at most 22 steps,
stratified by coupling family and by step band, curriculum order within each stratum.
The script prints counts only, never structures, so running it does not expose holdout
chemistry to an agent's context.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mechanistic_agent.catalysis_dataset import TM_FAMILIES, metal_active, tm_family  # noqa: E402
from mechanistic_agent.data_paths import flower_test_lookup_path, holdout_dir  # noqa: E402
from mechanistic_agent.flower_curriculum import (  # noqa: E402
    ConversionError,
    _convert_group,
    known_mechanism_from_case,
    load_mechanism_reactions,
)

SUITE = "catalysis_holdout"
SOURCE_LABEL = "FlowER flower_new_dataset test.txt"
DEFAULT_INPUT = PROJECT_ROOT.parent / "FlowER" / "data" / "flower_new_dataset" / "test.txt"
STEP_BANDS = (("3-4", 3, 4), ("5-8", 5, 8), ("9-22", 9, 22))


def _band(steps: int) -> str | None:
    for name, low, high in STEP_BANDS:
        if low <= steps <= high:
            return name
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(*, input_path: Path, index_path: Path, cache_path: Path, out_dir: Path,
          per_family_band: int, max_steps: int) -> Dict[str, Any]:
    if input_path.name != "test.txt":
        raise SystemExit("The catalysis holdout is built from FlowER test.txt only.")
    rows = [json.loads(line) for line in index_path.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda r: int(r.get("global_rank") or 0))
    picked: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    counts: Counter = Counter()
    for row in rows:
        steps = int(row.get("step_count") or 0)
        band = _band(steps)
        if band is None or steps > max_steps:
            continue
        mechanism_id = int(row["mechanism_id"])
        reactions = load_mechanism_reactions(mechanism_id, input_path=input_path, cache_path=cache_path)
        if not reactions or not metal_active(reactions):
            continue
        counts["metal_active"] += 1
        family, metals = tm_family(reactions[0].partition(">>")[0].split("."))
        if len(picked[(family, band)]) >= per_family_band:
            continue
        try:
            case = _convert_group(mechanism_id, reactions)
        except ConversionError as exc:
            counts[f"conversion_failed:{exc}"] += 1
            continue
        case["id"] = f"flower_test_{mechanism_id:06d}"
        case["name"] = f"FlowER test catalysis mechanism {mechanism_id}"
        case["description"] = f"Converted from FlowER test.txt group {mechanism_id}."
        case["source"] = SOURCE_LABEL
        case["tags"] = ["flower", "test", "leaderboard_holdout", "catalysis", family]
        case["purpose"] = "leaderboard_holdout"
        known = known_mechanism_from_case(case)
        known["source"] = SOURCE_LABEL
        case["known_mechanism"] = known
        case["leaderboard_holdout"] = {"suite": SUITE, "family": family, "metals": metals,
                                       "step_band": band, "step_count": steps,
                                       "global_rank": int(row.get("global_rank") or 0)}
        picked[(family, band)].append(case)
    cases = [c for family in TM_FAMILIES for name, _, _ in STEP_BANDS for c in picked[(family, name)]]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_set_catalysis_holdout.json").write_text(json.dumps(cases, indent=1) + "\n")
    report = {
        "suite": SUITE,
        "purpose": "leaderboard_holdout",
        "status": "reserve: not a leaderboard tier; nothing ranks on it",
        "generated_by": "scripts/build_catalysis_holdout.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": SOURCE_LABEL,
        "source_sha256": _sha256(input_path),
        "selection": {"per_family_band": per_family_band, "max_steps": max_steps,
                      "step_bands": [b[0] for b in STEP_BANDS], "families": list(TM_FAMILIES)},
        "selected_case_count": len(cases),
        "selected_by_family_band": {f"{f}/{b}": len(v) for (f, b), v in sorted(picked.items())},
        "counts": dict(sorted(counts.items())),
    }
    (out_dir / "catalysis_holdout_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="FlowER test.txt")
    parser.add_argument("--index", type=Path, default=holdout_dir(PROJECT_ROOT) / "flow_index_holdout.jsonl",
                        help="test-split curriculum index (built by build_flower_holdout_leaderboard_set.py)")
    parser.add_argument("--lookup-cache", type=Path, default=flower_test_lookup_path(PROJECT_ROOT))
    parser.add_argument("--out-dir", type=Path, default=holdout_dir(PROJECT_ROOT) / "catalysis")
    parser.add_argument("--per-family-band", type=int, default=5,
                        help="cases per (coupling family, step band); 5 gives 30 cases from the current test split")
    parser.add_argument("--max-steps", type=int, default=22)
    args = parser.parse_args()
    report = build(input_path=args.input, index_path=args.index, cache_path=args.lookup_cache,
                   out_dir=args.out_dir, per_family_band=args.per_family_band, max_steps=args.max_steps)
    # Counts only: never print structures from the holdout.
    print(json.dumps({k: report[k] for k in ("selected_case_count", "selected_by_family_band", "counts")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
