#!/usr/bin/env python3
"""Build the deterministic FlowER curriculum index and ranked top-100 dataset."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    normalize_electron_pushes,
    write_curriculum_index,
    _json_dump,
)

DEFAULT_MULTISTEP_DATASET_PATH = Path(__file__).resolve().parents[1] / "training_data" / "flower_mechanisms_multistep.json"
DEFAULT_MULTISTEP_REPORT_PATH = Path(__file__).resolve().parents[1] / "training_data" / "flower_mechanisms_multistep_report.json"


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
        choices=["ranked", "stratified"],
        default="ranked",
        help=(
            "ranked: take the lowest-ranked N mechanisms (all 1-step); "
            "stratified: sample --per-step mechanisms from each step-count tier."
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
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
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
