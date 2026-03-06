#!/usr/bin/env python3
"""Validate that prompt call changes have approved linked trace evidence."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mechanistic_agent.prompt_trace_validator import (
    discover_changed_calls,
    validate_evidence_for_calls,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", default="origin/main", help="Base git ref for change detection")
    parser.add_argument("--head-ref", default="HEAD", help="Head git ref for change detection")
    parser.add_argument(
        "--call",
        dest="calls",
        action="append",
        default=[],
        help="Explicit changed call name (can be provided multiple times)",
    )
    parser.add_argument("--repo", default=".", help="Repository root")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo = Path(args.repo).resolve()
    if args.calls:
        changed_calls = sorted({str(item).strip() for item in args.calls if str(item).strip()})
    else:
        changed_calls = discover_changed_calls(base_ref=args.base_ref, head_ref=args.head_ref, cwd=repo)

    if not changed_calls:
        print("No prompt call changes detected; evidence gate passed.")
        return 0

    result = validate_evidence_for_calls(changed_calls=changed_calls, base_dir=repo)
    if result.ok:
        print("Prompt trace evidence gate passed.")
        for call_name in result.changed_calls:
            files = result.valid_evidence_by_call.get(call_name, [])
            print(f"- {call_name}: {len(files)} valid evidence file(s)")
        return 0

    print("Prompt trace evidence gate failed.")
    for err in result.errors:
        print(f"- {err}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
