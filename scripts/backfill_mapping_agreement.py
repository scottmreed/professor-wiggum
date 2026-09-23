#!/usr/bin/env python3
"""Report benchmark mapping agreement over existing eval traces (read-only).

Opens the runtime DB with SQLite ``mode=ro``: nothing is written. For every
eval run result with a stored trace whose eval case carries a benchmark atom
mapping, scores the global ``atom_mapping`` output and each accepted step's
``step_atom_mapping`` output against it and prints a summary table.

Examples::

    PYTHONPATH=. python scripts/backfill_mapping_agreement.py
    PYTHONPATH=. python scripts/backfill_mapping_agreement.py --db-path ../wiggum-data/data/mechanistic.db --json

When the DB (``data/mechanistic.db`` or the sibling ``../wiggum-data``) is not
available the script says so and exits 0.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mechanistic_agent.mapping_audit import audit_mapping_agreement, format_audit_table  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db-path", type=Path, default=None, help="SQLite DB (default: resolved data root)")
    parser.add_argument("--eval-set-id", default=None, help="Restrict to one eval set")
    parser.add_argument("--eval-run-id", action="append", default=[], help="Restrict to eval run(s)")
    parser.add_argument("--include-hydrogens", action="store_true", help="Explicit-H policy (default heavy atoms)")
    parser.add_argument("--all", action="store_true", help="Also list eval runs with no benchmark-mapped traces")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    args = parser.parse_args(argv)

    report = audit_mapping_agreement(
        args.db_path,
        eval_set_id=args.eval_set_id,
        eval_run_ids=args.eval_run_id or None,
        include_hydrogens=args.include_hydrogens,
    )
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        if report.available:
            print(f"Mapping agreement (read-only) over {report.db_path}\n")
        print(format_audit_table(report, include_empty=args.all))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
