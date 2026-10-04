#!/usr/bin/env python3
"""Build the deterministic FlowER curriculum index and ranked top-100 dataset."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mechanistic_agent.flower_curriculum import (
    ConversionError,
    DEFAULT_DATASET_PATH,
    DEFAULT_DATASET_REPORT_PATH,
    DEFAULT_FLOWER_INPUT,
    DEFAULT_INDEX_PATH,
    DEFAULT_INDEX_REPORT_PATH,
    DEFAULT_LOOKUP_CACHE,
    build_curriculum_index,
    build_ranked_dataset,
    build_stratified_dataset,
    build_lookup_cache,
    convert_elementary_step,
    convert_mechanism_id_to_case,
    ensure_lookup_cache,
    load_curriculum_index,
    normalize_electron_pushes,
    write_curriculum_index,
    _json_dump,
)

from mechanistic_agent.data_paths import bulk_training_dir, repo_training_dir

DEFAULT_MULTISTEP_DATASET_PATH = bulk_training_dir(PROJECT_ROOT) / "flower_mechanisms_multistep.json"
DEFAULT_MULTISTEP_REPORT_PATH = bulk_training_dir(PROJECT_ROOT) / "flower_mechanisms_multistep_report.json"

# Files whose case IDs an `--mode extend` run must never re-select: the dev eval
# set and its source, the practice set, and every tier list (all tiers). The
# official holdout is FlowER *test* split (`flower_test_*`) and is excluded by
# prefix as well as by file when present.
DEFAULT_EXTEND_EXCLUDE_FILES: Tuple[str, ...] = (
    "eval_set.json",
    "flower_mechanisms_100.json",
    "practice_eval/practice_set.json",
    "eval_tiers.json",
    "baseline_tiers_clawdiator.json",
)
HOLDOUT_ID_PREFIX = "flower_test_"


def _default_multistep_paths() -> Tuple[Path, Path]:
    """Existing multistep dataset/report: the in-repo legacy copy wins when present."""

    legacy = repo_training_dir(PROJECT_ROOT) / "flower_mechanisms_multistep.json"
    if legacy.exists():
        return legacy, legacy.with_name("flower_mechanisms_multistep_report.json")
    return DEFAULT_MULTISTEP_DATASET_PATH, DEFAULT_MULTISTEP_REPORT_PATH


def collect_case_ids(payload: Any) -> List[str]:
    """Case IDs from a dataset list (rows with ``id``) or a tier file (``{tier: [ids]}``)."""

    ids: List[str] = []
    if isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict) and row.get("id"):
                ids.append(str(row["id"]))
            elif isinstance(row, str):
                ids.append(row)
    elif isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).startswith("_"):
                continue
            ids.extend(collect_case_ids(value))
    return ids


def load_exclusion_ids(paths: Iterable[Path]) -> Tuple[set[str], List[str]]:
    excluded: set[str] = set()
    used: List[str] = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        excluded.update(collect_case_ids(json.loads(path.read_text(encoding="utf-8"))))
        used.append(str(path))
    return excluded, used


def extend_stratified_dataset(
    *,
    dataset: Sequence[Dict[str, Any]],
    report: Mapping[str, Any],
    index_entries: Sequence[Mapping[str, Any]],
    step_targets: Mapping[int, int],
    exclude_ids: Iterable[str] = (),
    convert: Optional[Callable[[int], Dict[str, Any]]] = None,
    input_path: Optional[Path] = None,
    cache_path: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, List[str]]]:
    """Append-only extension of a ``--mode stratified`` dataset.

    For each ``step -> target`` pair, keeps converting the next lowest-ranked
    mechanisms of that step-count tier (index order, i.e. ``rank_within_step_count``)
    until the dataset holds ``target`` cases of that step count. Existing rows are
    kept verbatim and in place; new rows are appended in ascending step order, so
    the selection is the same policy as the original stratified build, continued.

    Never re-attempted: IDs already selected or previously skipped (conversion
    failures, recorded in the report). Never selected: ``exclude_ids`` and any
    ``flower_test_*`` (holdout-namespace) ID. Every appended case must have
    ``n_mechanistic_steps`` equal to its tier, otherwise ``ValueError``.

    Returns ``(dataset, report, added_ids_by_step)``.
    """

    if convert is None:
        def convert(mechanism_id: int) -> Dict[str, Any]:
            kwargs: Dict[str, Any] = {}
            if input_path is not None:
                kwargs["input_path"] = Path(input_path)
            if cache_path is not None:
                kwargs["cache_path"] = Path(cache_path)
            return convert_mechanism_id_to_case(int(mechanism_id), **kwargs)

    out_dataset: List[Dict[str, Any]] = [dict(row) for row in dataset]
    out_report: Dict[str, Any] = json.loads(json.dumps(dict(report)))
    existing_ids = {str(row.get("id")) for row in out_dataset}
    prior_skipped = {str(cid) for cid in out_report.get("skipped_case_ids") or []}
    excluded = {str(cid) for cid in exclude_ids}

    by_step: Dict[int, List[Mapping[str, Any]]] = {}
    for entry in index_entries:
        by_step.setdefault(int(entry.get("step_count") or 0), []).append(entry)
    for entries in by_step.values():
        entries.sort(key=lambda e: int(e.get("rank_within_step_count") or 0))

    have_by_step = Counter(int(row.get("n_mechanistic_steps") or 0) for row in out_dataset)
    failures: Counter[str] = Counter()
    added_by_step: Dict[str, List[str]] = {}
    tier_log: Dict[str, Any] = {}

    for step in sorted(int(s) for s in step_targets):
        target = int(step_targets[step])
        tier_entries = by_step.get(step, [])
        added: List[str] = []
        skipped: List[str] = []
        excluded_seen: List[str] = []
        attempted = 0
        for entry in tier_entries:
            if have_by_step[step] + len(added) >= target:
                break
            case_id = str(entry["case_id"])
            if case_id in existing_ids or case_id in prior_skipped:
                continue
            if case_id in excluded or case_id.startswith(HOLDOUT_ID_PREFIX):
                excluded_seen.append(case_id)
                continue
            attempted += 1
            try:
                case = convert(int(entry["mechanism_id"]))
            except ConversionError as exc:
                failures[exc.reason] += 1
                skipped.append(case_id)
                continue
            n_steps = int(case.get("n_mechanistic_steps") or 0)
            n_verified = len(((case.get("verified_mechanism") or {}).get("steps")) or [])
            if n_steps != step or n_verified != step:
                raise ValueError(
                    f"{case_id}: converted to {n_steps} steps ({n_verified} verified) "
                    f"but sits in the {step}-step tier of the index"
                )
            out_dataset.append(case)
            existing_ids.add(case_id)
            added.append(case_id)

        added_by_step[str(step)] = added
        tier_log[str(step)] = {
            "target_total": target,
            "attempted": attempted,
            "added": len(added),
            "skipped": len(skipped),
            "excluded_case_ids_passed_over": excluded_seen,
            "available_in_index": len(tier_entries),
            "short_by": max(0, target - have_by_step[step] - len(added)),
        }
        out_report.setdefault("selected_case_ids", []).extend(added)
        out_report.setdefault("skipped_case_ids", []).extend(skipped)
        summary = dict((out_report.get("tier_summary") or {}).get(str(step)) or {})
        summary["attempted"] = int(summary.get("attempted") or 0) + attempted
        summary["selected"] = int(summary.get("selected") or 0) + len(added)
        summary["skipped"] = int(summary.get("skipped") or 0) + len(skipped)
        summary["available_in_index"] = len(tier_entries)
        out_report.setdefault("tier_summary", {})[str(step)] = summary

    merged_failures = Counter({str(k): int(v) for k, v in (out_report.get("conversion_failures_by_reason") or {}).items()})
    merged_failures.update(failures)
    out_report["conversion_failures_by_reason"] = dict(sorted(merged_failures.items()))
    distribution = Counter(int(row.get("n_mechanistic_steps") or 0) for row in out_dataset)
    out_report["step_count_distribution_sampled_set"] = {str(k): distribution[k] for k in sorted(distribution)}
    out_report["total_selected"] = len(out_dataset)
    out_report["max_step"] = max([int(out_report.get("max_step") or 0), *[int(s) for s in step_targets]])
    out_report["tier_summary"] = dict(sorted(out_report["tier_summary"].items(), key=lambda kv: int(kv[0])))
    out_report.setdefault("extensions", []).append(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "step_targets": {str(k): int(v) for k, v in sorted(step_targets.items())},
            "added_case_ids_by_step": added_by_step,
            "tiers": tier_log,
            "conversion_failures_by_reason": dict(sorted(failures.items())),
            "excluded_id_count": len(excluded),
            "selection_policy": (
                "append-only continuation of the stratified policy: next lowest-ranked "
                "successful conversions per step-count tier (index rank_within_step_count "
                "order), skipping IDs already selected, previously skipped, in an "
                "exclusion file, or in the flower_test_ holdout namespace"
            ),
        }
    )
    return out_dataset, out_report, added_by_step


def _parse_step_targets(values: Sequence[str]) -> Dict[int, int]:
    targets: Dict[int, int] = {}
    for raw in values:
        step_text, sep, count_text = str(raw).partition("=")
        if not sep:
            raise SystemExit(f"--extend-step expects STEP=TOTAL, got {raw!r}")
        step, count = int(step_text), int(count_text)
        if step < 1 or count < 1:
            raise SystemExit(f"--extend-step values must be positive, got {raw!r}")
        targets[step] = count
    if not targets:
        raise SystemExit("--mode extend needs at least one --extend-step STEP=TOTAL")
    return targets


def _run_extend(args: argparse.Namespace) -> int:
    default_dataset, default_report = _default_multistep_paths()
    dataset_path = Path(args.dataset_output) if args.dataset_output else default_dataset
    report_path = Path(args.dataset_report) if args.dataset_report else default_report
    if not dataset_path.exists() or not report_path.exists():
        raise SystemExit(
            f"--mode extend needs an existing stratified dataset and report: {dataset_path}, {report_path}"
        )
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    index_path = Path(args.index_output)
    if not index_path.exists():
        raise SystemExit(f"Curriculum index not found: {index_path} (run --mode stratified first)")
    index_entries = load_curriculum_index(index_path)

    input_path = Path(args.input)
    cache_path = ensure_lookup_cache(input_path=input_path, cache_path=Path(args.lookup_cache))

    exclude_files = [Path(p) for p in (args.exclude_ids_from or [])] or [
        repo_training_dir(PROJECT_ROOT) / name for name in DEFAULT_EXTEND_EXCLUDE_FILES
    ]
    from mechanistic_agent.data_paths import holdout_eval_set_path

    exclude_files.append(holdout_eval_set_path(PROJECT_ROOT))
    excluded, used_files = load_exclusion_ids(exclude_files)

    new_dataset, new_report, added = extend_stratified_dataset(
        dataset=dataset,
        report=report,
        index_entries=index_entries,
        step_targets=_parse_step_targets(args.extend_step or []),
        exclude_ids=excluded,
        input_path=input_path,
        cache_path=cache_path,
    )
    new_report["extensions"][-1]["exclusion_files"] = used_files
    _json_dump(new_dataset, dataset_path)
    _json_dump(new_report, report_path)
    for step, ids in added.items():
        tier = new_report["extensions"][-1]["tiers"][step]
        note = f" (short by {tier['short_by']})" if tier["short_by"] else ""
        print(f"step {step}: added {len(ids)}{note}: {', '.join(ids)}")
    print(f"Wrote {len(new_dataset)} FlowER mechanisms to {dataset_path}")
    print(f"Wrote dataset report to {report_path}")
    return 0


def build_dataset(
    *,
    input_path: Path,
    sample_size: int = 100,
    index_output: Optional[Path] = None,
    index_report: Optional[Path] = None,
    cache_path: Optional[Path] = None,
    seed: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Compatibility wrapper around the deterministic curriculum builder.

    `seed` and `batch_size` are accepted for backward compatibility but ignored.
    """

    _ = seed
    _ = batch_size
    cache = Path(cache_path) if cache_path is not None else DEFAULT_LOOKUP_CACHE
    build_lookup_cache(input_path=Path(input_path), cache_path=cache)
    entries, report = build_curriculum_index(input_path=Path(input_path))
    if index_output is not None:
        write_curriculum_index(
            entries,
            output_path=Path(index_output),
            report_path=Path(index_report) if index_report is not None else None,
            report=report if index_report is not None else None,
        )
    dataset, dataset_report = build_ranked_dataset(
        input_path=Path(input_path),
        cache_path=cache,
        index_entries=entries,
        sample_size=int(sample_size),
    )
    dataset_report["index_report"] = report
    return dataset, dataset_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the deterministic FlowER curriculum dataset.")
    parser.add_argument(
        "--mode",
        choices=["ranked", "stratified", "extend"],
        default="ranked",
        help=(
            "ranked: take the lowest-ranked N mechanisms (all 1-step); "
            "stratified: sample --per-step mechanisms from each step-count tier; "
            "extend: append-only growth of an existing stratified dataset to the "
            "--extend-step STEP=TOTAL targets (existing rows kept verbatim, in place)."
        ),
    )
    parser.add_argument("--input", default=str(DEFAULT_FLOWER_INPUT), help="Path to FlowER train.txt file.")
    parser.add_argument("--index-output", default=str(DEFAULT_INDEX_PATH), help="Output JSONL curriculum index path.")
    parser.add_argument("--index-report", default=str(DEFAULT_INDEX_REPORT_PATH), help="Output curriculum index report JSON path.")
    parser.add_argument("--dataset-output", default=None, help="Output dataset JSON path (default depends on --mode).")
    parser.add_argument("--dataset-report", default=None, help="Output dataset report JSON path (default depends on --mode).")
    parser.add_argument("--lookup-cache", default=str(DEFAULT_LOOKUP_CACHE), help="Local lookup cache SQLite path.")
    # ranked-mode options
    parser.add_argument("--sample-size", type=int, default=100, help="[ranked] Number of successfully converted mechanisms to emit.")
    # stratified-mode options
    parser.add_argument("--per-step", type=int, default=20, help="[stratified] Examples per step-count tier.")
    parser.add_argument("--max-step", type=int, default=8, help="[stratified] Maximum step count tier to sample.")
    # extend-mode options
    parser.add_argument(
        "--extend-step",
        action="append",
        metavar="STEP=TOTAL",
        help="[extend] Grow the STEP-step tier to TOTAL cases (repeatable), e.g. --extend-step 3=40 --extend-step 7=20.",
    )
    parser.add_argument(
        "--exclude-ids-from",
        action="append",
        metavar="JSON",
        help=(
            "[extend] Dataset or tier JSON whose IDs must never be selected (repeatable). "
            "Default: training_data/{" + ",".join(DEFAULT_EXTEND_EXCLUDE_FILES) + "}; "
            "the holdout eval set is always added when present."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.mode == "extend":
        return _run_extend(args)
    input_path = Path(args.input)
    cache_path = Path(args.lookup_cache)

    # Resolve output paths based on mode
    if args.mode == "stratified":
        dataset_output = Path(args.dataset_output) if args.dataset_output else DEFAULT_MULTISTEP_DATASET_PATH
        dataset_report_path = Path(args.dataset_report) if args.dataset_report else DEFAULT_MULTISTEP_REPORT_PATH
    else:
        dataset_output = Path(args.dataset_output) if args.dataset_output else DEFAULT_DATASET_PATH
        dataset_report_path = Path(args.dataset_report) if args.dataset_report else DEFAULT_DATASET_REPORT_PATH

    build_lookup_cache(input_path=input_path, cache_path=cache_path)
    entries, index_report = build_curriculum_index(input_path=input_path)
    write_curriculum_index(
        entries,
        output_path=Path(args.index_output),
        report_path=Path(args.index_report),
        report=index_report,
    )

    if args.mode == "stratified":
        dataset, dataset_report = build_stratified_dataset(
            input_path=input_path,
            cache_path=cache_path,
            index_entries=entries,
            per_step=max(1, int(args.per_step)),
            max_step=max(1, int(args.max_step)),
        )
    else:
        dataset, dataset_report = build_ranked_dataset(
            input_path=input_path,
            cache_path=cache_path,
            index_entries=entries,
            sample_size=max(1, int(args.sample_size)),
        )

    _json_dump(dataset, dataset_output)
    _json_dump(dataset_report, dataset_report_path)

    print(f"Wrote {len(entries)} curriculum index rows to {args.index_output}")
    print(f"Wrote curriculum index report to {args.index_report}")
    print(f"Wrote {len(dataset)} FlowER mechanisms to {dataset_output}")
    print(f"Wrote dataset report to {dataset_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
