#!/usr/bin/env python3
"""Append newly tiered cases to the existing development tier eval sets in the DB.

`eval --tier medium|hard` and `baseline --all-tiers` run against the DB eval sets
named in ``training_data/baseline_tier_eval_set_map.json``; the tier file only
picks which of that set's cases to run. When a tier list grows (append-only), its
eval set must grow too or the new IDs are silently dropped by the planner's
intersection. This script appends the missing cases **in place** so the
``eval_set_id`` — and the leaderboard history keyed on it — stays the same.

Case payloads use the shape the existing multistep tier sets were imported with:
``input`` = starting_materials / products / temperature_celsius / ph /
n_mechanistic_steps, ``expected`` = products / verified_mechanism /
n_mechanistic_steps, ``tags`` = the case's tags + flower, multistep,
clawdiator_planned.

Dry run by default; ``--apply`` writes. Back up the DB first.

    python scripts/sync_dev_tier_eval_sets.py            # plan
    python scripts/sync_dev_tier_eval_sets.py --apply    # append + bump version
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mechanistic_agent.core.db import RunStore  # noqa: E402
from mechanistic_agent.data_paths import bulk_training_dir, db_path, repo_training_dir  # noqa: E402

TRAINING = repo_training_dir(PROJECT_ROOT)
DEFAULT_TIER_FILES = (TRAINING / "baseline_tiers_clawdiator.json", TRAINING / "eval_tiers.json")
DEFAULT_MAP = TRAINING / "baseline_tier_eval_set_map.json"
TIER_CASE_TAGS = ["flower", "multistep", "clawdiator_planned"]
HOLDOUT_ID_PREFIX = "flower_test_"


def default_dataset_path() -> Path:
    legacy = TRAINING / "flower_mechanisms_multistep.json"
    return legacy if legacy.exists() else bulk_training_dir(PROJECT_ROOT) / "flower_mechanisms_multistep.json"


def tier_eval_case(record: Mapping[str, Any]) -> Dict[str, Any]:
    """One multistep dataset record -> the eval-set case shape used by the tier sets."""

    n_steps = record.get("n_mechanistic_steps")
    return {
        "case_id": str(record["id"]),
        "input": {
            "starting_materials": record.get("starting_materials"),
            "products": record.get("products"),
            "temperature_celsius": record.get("temperature_celsius"),
            "ph": record.get("ph"),
            "n_mechanistic_steps": n_steps,
        },
        "expected": {
            "products": record.get("products"),
            "verified_mechanism": record.get("verified_mechanism"),
            "n_mechanistic_steps": n_steps,
        },
        "tags": list(record.get("tags") or []) + list(TIER_CASE_TAGS),
    }


def cases_sha256(cases: Iterable[Mapping[str, Any]]) -> str:
    """Content hash of an eval set: (case_id, input, expected, tags), sorted by case_id."""

    rows = sorted(
        (
            [str(c.get("case_id")), c.get("input") or {}, c.get("expected"), list(c.get("tags") or [])]
            for c in cases
        ),
        key=lambda row: row[0],
    )
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode("utf-8")).hexdigest()


def bump_version(version: str) -> str:
    match = re.fullmatch(r"v(\d+)", str(version or "").strip())
    return f"v{int(match.group(1)) + 1}" if match else f"{version}+1"


def plan_tier_append(
    *,
    tier: str,
    tier_ids: Sequence[str],
    existing_cases: Sequence[Mapping[str, Any]],
    dataset_by_id: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Validate a tier against its eval set and build the cases to append.

    Raises ``ValueError`` when the set holds a case the tier does not list, a
    missing tier ID has no dataset record or is a holdout-namespace ID, or an
    existing case's payload no longer matches the dataset record.
    """

    tier_ids = [str(cid) for cid in tier_ids]
    if len(set(tier_ids)) != len(tier_ids):
        raise ValueError(f"{tier}: duplicate IDs in tier list")
    existing_by_id = {str(c["case_id"]): c for c in existing_cases}
    stray = sorted(set(existing_by_id) - set(tier_ids))
    if stray:
        raise ValueError(f"{tier}: eval set holds cases the tier does not list: {stray[:5]}")
    drifted = []
    for case_id, case in existing_by_id.items():
        record = dataset_by_id.get(case_id)
        if record is None:
            continue
        rebuilt = tier_eval_case(record)
        if case.get("input") != rebuilt["input"] or case.get("expected") != rebuilt["expected"]:
            drifted.append(case_id)
    if drifted:
        raise ValueError(f"{tier}: existing eval cases differ from the dataset records: {drifted[:5]}")
    missing = [cid for cid in tier_ids if cid not in existing_by_id]
    bad = [cid for cid in missing if cid.startswith(HOLDOUT_ID_PREFIX) or "_test_" in cid]
    if bad:
        raise ValueError(f"{tier}: refusing holdout-namespace IDs: {bad[:5]}")
    unresolved = [cid for cid in missing if cid not in dataset_by_id]
    if unresolved:
        raise ValueError(f"{tier}: tier IDs missing from the dataset: {unresolved[:5]}")
    new_cases = [tier_eval_case(dataset_by_id[cid]) for cid in missing]
    return {
        "tier": tier,
        "existing_count": len(existing_by_id),
        "tier_count": len(tier_ids),
        "new_cases": new_cases,
        "final_sha256": cases_sha256(list(existing_cases) + new_cases),
    }


def _load_tiers(paths: Sequence[Path], tiers: Sequence[str]) -> Dict[str, List[str]]:
    loaded = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths if Path(p).exists()]
    if not loaded:
        raise SystemExit(f"No tier file found among {[str(p) for p in paths]}")
    first = loaded[0]
    for other, path in zip(loaded[1:], paths[1:]):
        for tier in tiers:
            if list(other.get(tier) or []) != list(first.get(tier) or []):
                raise SystemExit(f"Tier '{tier}' differs between {paths[0]} and {path}; sync them first")
    return {tier: list(first.get(tier) or []) for tier in tiers}


def sync(
    store: RunStore,
    *,
    tier_ids_by_name: Mapping[str, Sequence[str]],
    eval_set_ids: Mapping[str, str],
    dataset: Sequence[Mapping[str, Any]],
    apply: bool = False,
) -> List[Dict[str, Any]]:
    dataset_by_id = {str(row["id"]): row for row in dataset}
    results: List[Dict[str, Any]] = []
    plans = []
    for tier, tier_ids in tier_ids_by_name.items():
        eval_set_id = str(eval_set_ids[tier])
        eval_set = store.get_eval_set(eval_set_id)
        if eval_set is None:
            raise SystemExit(f"{tier}: eval set {eval_set_id} not found in {store.db_path}")
        if str(eval_set.get("purpose")) == "leaderboard_holdout":
            raise SystemExit(f"{tier}: eval set {eval_set_id} is a leaderboard holdout; refusing")
        plan = plan_tier_append(
            tier=tier,
            tier_ids=tier_ids,
            existing_cases=store.list_eval_set_cases(eval_set_id),
            dataset_by_id=dataset_by_id,
        )
        plans.append((tier, eval_set, plan))
    for tier, eval_set, plan in plans:
        new_version = bump_version(str(eval_set.get("version") or "v1")) if plan["new_cases"] else eval_set.get("version")
        result: Dict[str, Any] = {
            "tier": tier,
            "eval_set_id": eval_set["id"],
            "name": eval_set.get("name"),
            "existing_count": plan["existing_count"],
            "to_add": [c["case_id"] for c in plan["new_cases"]],
            "version": f"{eval_set.get('version')} -> {new_version}",
            "sha256": plan["final_sha256"],
            "applied": False,
        }
        if apply and plan["new_cases"]:
            outcome = store.append_eval_set_cases(
                str(eval_set["id"]),
                plan["new_cases"],
                version=str(new_version),
                sha256=plan["final_sha256"],
            )
            result["applied"] = True
            result["case_count"] = outcome["case_count"]
        results.append(result)
    return results


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tiers", nargs="+", default=["medium", "hard"])
    parser.add_argument("--tier-file", action="append", help="Tier JSON (repeatable; all must agree). Default: baseline_tiers_clawdiator.json + eval_tiers.json")
    parser.add_argument("--map", default=str(DEFAULT_MAP), help="Tier -> eval_set_id map JSON.")
    parser.add_argument("--dataset", default=None, help="Multistep dataset JSON (default: training_data/flower_mechanisms_multistep.json).")
    parser.add_argument("--db", default=None, help="SQLite DB path (default: data_paths.db_path()).")
    parser.add_argument("--apply", action="store_true", help="Write the appends (default: dry run).")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    tier_files = [Path(p) for p in args.tier_file] if args.tier_file else list(DEFAULT_TIER_FILES)
    tier_ids = _load_tiers(tier_files, args.tiers)
    mapping = json.loads(Path(args.map).read_text(encoding="utf-8"))
    eval_set_ids = {tier: str((mapping.get(tier) or {}).get("eval_set_id") or "") for tier in args.tiers}
    missing_map = [tier for tier, value in eval_set_ids.items() if not value]
    if missing_map:
        raise SystemExit(f"No eval_set_id mapped for tiers {missing_map} in {args.map}")
    dataset_path = Path(args.dataset) if args.dataset else default_dataset_path()
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    database = Path(args.db) if args.db else db_path()
    if not database.exists():
        raise SystemExit(f"DB not found: {database}")
    store = RunStore(database)
    results = sync(store, tier_ids_by_name=tier_ids, eval_set_ids=eval_set_ids, dataset=dataset, apply=args.apply)
    print(json.dumps({"db": str(database), "dataset": str(dataset_path), "apply": args.apply, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
