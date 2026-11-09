#!/usr/bin/env python
"""Build training_data/catalysis_development/ from FlowER train.txt (PRD_catalysis_handling R5.1/R5.2).

Train split only: refuses any source not named train.txt or inside a
test/validation/holdout directory. Reads the curriculum index and lookup cache
built by ``main.py curriculum build-lookup``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mechanistic_agent.catalysis_dataset import build_catalysis_cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True, help="flower_mechanism_index.jsonl")
    parser.add_argument("--cache", type=Path, required=True, help="flower_train_lookup.sqlite")
    parser.add_argument("--per-family", type=int, default=6)
    parser.add_argument("--organo-cases", type=int, default=12)
    parser.add_argument("--organo-scan", type=int, default=20000)
    parser.add_argument("--max-steps", type=int, default=22)
    parser.add_argument("--output", type=Path, default=Path("training_data/catalysis_development/cases.json"))
    parser.add_argument("--report", type=Path, default=Path("training_data/catalysis_development/extraction_report.json"))
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.index.read_text().splitlines() if line.strip()]
    cases, report = build_catalysis_cases(
        args.train,
        rows,
        cache_path=args.cache,
        per_family=args.per_family,
        organo_cases=args.organo_cases,
        organo_scan=args.organo_scan,
        max_steps=args.max_steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cases, indent=1, sort_keys=True) + "\n")
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["selected"]))


if __name__ == "__main__":
    main()
