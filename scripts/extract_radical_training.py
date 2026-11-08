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
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    cases, report = extract_radical_training_cases(args.train, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cases, indent=2) + '\n')
    report_path = args.output.with_suffix('.report.json')
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rejections'}, indent=2))


if __name__ == '__main__':
    main()
