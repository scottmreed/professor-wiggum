#!/usr/bin/env python3
"""Extract a separate train-derived radical step development set; never reads test."""
import argparse
import json
from pathlib import Path

from mechanistic_agent.radical_dataset import extract_radical_training_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, help='keep the first N cases overall (after --per-label)')
    parser.add_argument('--per-label', type=int, help='keep the first N cases of each label (RC, RS)')
    parser.add_argument('--report', type=Path, help='report path (default: <output>.report.json)')
    args = parser.parse_args()
    cases, report = extract_radical_training_cases(args.train, limit=args.limit, per_label=args.per_label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cases, indent=2) + '\n')
    report_path = args.report or args.output.with_suffix('.report.json')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rejections'}, indent=2))


if __name__ == '__main__':
    main()
