#!/usr/bin/env python3
"""Validate that prompt call changes have approved linked trace evidence.

Without ``--call`` the changed calls are discovered from ``git diff`` between
``--base-ref`` and ``--head-ref`` (paths under ``skills/mechanistic/``). With
``--call`` the named calls are validated directly; add ``--model`` to scope them
to one model lane (``skills/mechanistic/<call>/models/<slug>/``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mechanistic_agent.prompt_trace_validator import (
    PromptChange,
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
    parser.add_argument(
        "--model",
        default=None,
        help="Scope every --call to this model lane (e.g. anthropic/claude-opus-4.6)",
    )
    parser.add_argument("--repo", default=".", help="Repository root")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo = Path(args.repo).resolve()
    model = str(args.model or "").strip() or None
    if args.calls:
        names = sorted({str(item).strip() for item in args.calls if str(item).strip()})
        changes = [PromptChange(call_name=name, model_name=model) for name in names]
    else:
        if model:
            print("--model only applies together with --call", file=sys.stderr)
            return 2
        changes = discover_changed_calls(base_ref=args.base_ref, head_ref=args.head_ref, cwd=repo)

    if not changes:
        print("No prompt call changes detected; evidence gate passed.")
        return 0

    print("Detected prompt changes:")
    for change in changes:
        components = ",".join(sorted(change.components)) or "explicit"
        print(f"- {change.label} [{components}]")

    result = validate_evidence_for_calls(changed_calls=changes, base_dir=repo)
    if result.ok:
        print("Prompt trace evidence gate passed.")
        for label in result.changed_calls:
            files = result.valid_evidence_by_call.get(label, [])
            print(f"- {label}: {len(files)} valid evidence file(s)")
        return 0

    print("Prompt trace evidence gate failed.")
    for err in result.errors:
        print(f"- {err}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
