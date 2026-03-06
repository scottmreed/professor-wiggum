#!/usr/bin/env python3
"""Render PNG panels for FlowER-derived datasets or the ranked curriculum.

Modes
-----
curriculum  (default)
    Render PNGs directly from the FlowER train.txt via the curriculum index.
    Use --top-n to limit output to the N lowest-ranked cases.
    Output: training_data/flower_curriculum_pngs/

dataset
    Render PNGs from a pre-built JSON dataset file (e.g. flower_mechanisms_multistep.json).
    Output: training_data/flower_curriculum_pngs/  (merged with existing training PNGs)
    Override with --output.

eval
    Render PNGs from training_data/eval_set.json.
    Output: training_data/eval_set_pngs/
    Override with --output.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mechanistic_agent.flower_curriculum import (
    DEFAULT_DATASET_PATH,
    DEFAULT_FLOWER_INPUT,
    DEFAULT_INDEX_PATH,
    DEFAULT_LOOKUP_CACHE,
)
from mechanistic_agent.flower_rendering import render_curriculum_pngs, render_pngs

_TRAINING_DATA = Path(__file__).resolve().parents[1] / "training_data"
DEFAULT_DATASET_OUTPUT = _TRAINING_DATA / "flower_mechanisms_100_pngs"
DEFAULT_CURRICULUM_OUTPUT = _TRAINING_DATA / "flower_curriculum_pngs"
DEFAULT_EVAL_OUTPUT = _TRAINING_DATA / "eval_set_pngs"
DEFAULT_EVAL_INPUT = _TRAINING_DATA / "eval_set.json"
DEFAULT_MULTISTEP_INPUT = _TRAINING_DATA / "flower_mechanisms_multistep.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render PNGs for FlowER-derived mechanism cases.")
    parser.add_argument(
        "--mode",
        choices=["curriculum", "dataset", "eval"],
        default="curriculum",
        help=(
            "curriculum: render from FlowER train.txt via curriculum index (use --top-n); "
            "dataset: render from a JSON dataset file into flower_curriculum_pngs/; "
            "eval: render from eval_set.json into eval_set_pngs/."
        ),
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Input path. For 'curriculum' mode: FlowER train.txt (default: DEFAULT_FLOWER_INPUT). "
            "For 'dataset' mode: JSON dataset file (default: flower_mechanisms_multistep.json). "
            "For 'eval' mode: JSON eval set (default: eval_set.json)."
        ),
    )
    parser.add_argument("--train-input", default=str(DEFAULT_FLOWER_INPUT), help="Raw FlowER train.txt path for curriculum rendering.")
    parser.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH), help="Curriculum index JSONL path.")
    parser.add_argument("--lookup-cache", default=str(DEFAULT_LOOKUP_CACHE), help="Lookup cache SQLite path.")
    parser.add_argument("--output", default=None, help="Output directory for PNGs (overrides mode default).")
    parser.add_argument("--max-reactions", type=int, default=None, help="Optional limit on number of cases to render.")
    parser.add_argument("--top-n", type=int, default=None, help="[curriculum] Render only the N lowest-ranked curriculum cases.")
    parser.add_argument("--only-missing", action="store_true", help="Only render cases whose PNG file is missing.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        if args.mode == "curriculum":
            output_dir = Path(args.output) if args.output else DEFAULT_CURRICULUM_OUTPUT
            train_input = Path(args.input) if args.input else Path(args.train_input)
            index = render_curriculum_pngs(
                input_path=train_input,
                index_path=Path(args.index_path),
                cache_path=Path(args.lookup_cache),
                output_dir=output_dir,
                top_n=max(1, int(args.top_n)) if args.top_n is not None else None,
                only_missing=bool(args.only_missing),
            )
            print(f"Rendered curriculum PNGs to {output_dir} ({index['rendered_count']} newly rendered)")

        elif args.mode == "eval":
            input_path = Path(args.input) if args.input else DEFAULT_EVAL_INPUT
            output_dir = Path(args.output) if args.output else DEFAULT_EVAL_OUTPUT
            index = render_pngs(
                input_path=input_path,
                output_dir=output_dir,
                max_reactions=args.max_reactions,
            )
            print(f"Rendered {index['rendered_count']} eval PNG(s) to {output_dir}")

        else:  # dataset
            input_path = Path(args.input) if args.input else DEFAULT_MULTISTEP_INPUT
            output_dir = Path(args.output) if args.output else DEFAULT_CURRICULUM_OUTPUT
            index = render_pngs(
                input_path=input_path,
                output_dir=output_dir,
                max_reactions=args.max_reactions,
            )
            print(f"Rendered {index['rendered_count']} training PNG(s) to {output_dir}")

        return 0
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
