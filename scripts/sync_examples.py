#!/usr/bin/env python3
"""Sync approved few-shot examples between the local SQLite DB and version-controlled .jsonl files.

Commands
--------
export  DB → files
    Reads approved examples from the DB and merges them into the appropriate
    few_shot.jsonl files under skills/mechanistic/. New or higher-scoring
    examples are written; existing file entries beat equal-score newcomers.
    The resulting files are git-committable.

import  files → DB
    Reads every few_shot.jsonl under skills/mechanistic/ and seeds the DB
    using INSERT OR IGNORE.  Local DB data always wins on key conflict, so
    a ``git pull`` followed by ``sync import`` will never overwrite locally-
    approved examples.

Usage
-----
    python scripts/sync_examples.py export [--db data/mechanistic.db] [--step STEP]
    python scripts/sync_examples.py import [--db data/mechanistic.db] [--step STEP]

The ``--step`` filter accepts the DB step_name (e.g. ``mechanism_step_proposal``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _key_for(entry: dict) -> str:
    explicit = entry.get("example_key") or entry.get("key")
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    return hashlib.sha256(
        f"{entry.get('input', '')}\n\0{entry.get('output', '')}".encode("utf-8")
    ).hexdigest()[:16]


# ---------------------------------------------------------------------------
# export: DB → files
# ---------------------------------------------------------------------------

def cmd_export(args: argparse.Namespace) -> int:
    repo = _repo_root()
    db_path = (repo / args.db).resolve()
    if not db_path.exists():
        print(f"DB not found: {db_path}", file=sys.stderr)
        return 1

    from mechanistic_agent.core.db import RunStore
    from mechanistic_agent.prompt_assets import (
        mechanistic_skills_root,
        resolve_call_name_from_step,
        write_call_few_shot_examples,
    )

    store = RunStore(db_path)

    # Fetch all approved examples (optionally filtered by step).
    rows = store.list_few_shot_examples(
        step_name=args.step or None,
        approved_only=True,
        limit=100_000,
    )

    if not rows:
        print("No approved examples found in DB.")
        return 0

    # Resolve model_name for each example via its prompt_version.
    # We cache prompt_version lookups to avoid N+1 queries.
    pv_cache: dict[str, dict] = {}

    def _model_for_pv(pv_id: str | None) -> str | None:
        if not pv_id:
            return None
        if pv_id not in pv_cache:
            pv = store.get_prompt_version(pv_id)
            pv_cache[pv_id] = pv or {}
        return pv_cache[pv_id].get("model_name") or None

    # Group by (call_name, model_name).
    groups: dict[tuple[str, str | None], list[dict]] = {}
    skipped = 0
    for row in rows:
        step_name = str(row.get("step_name") or "")
        call_name = resolve_call_name_from_step(step_name)
        if not call_name:
            # step_name IS the call_name when it doesn't map through STEP_TO_CALL_NAME
            # (e.g. a directly-named call). Fall back to step_name itself.
            call_name = step_name or None
        if not call_name:
            skipped += 1
            continue
        model_name = _model_for_pv(row.get("prompt_version_id"))
        key = (call_name, model_name)
        groups.setdefault(key, []).append(
            {
                "input": row["input_text"],
                "output": row["output_text"],
                "score": row.get("score"),
                "example_key": row.get("example_key") or "",
            }
        )

    files_written: list[Path] = []
    total_examples = 0
    # Pass repo root as base_dir; write_call_few_shot_examples resolves to
    # <base_dir>/skills/mechanistic/<call_name>/few_shot.jsonl internally.
    for (call_name, model_name), examples in groups.items():
        path = write_call_few_shot_examples(
            call_name,
            examples,
            base_dir=repo,
            model_name=model_name,
        )
        files_written.append(path)
        total_examples += len(examples)
        label = f"{call_name}" + (f" [{model_name}]" if model_name else "")
        print(f"  {label}: {len(examples)} examples → {path.relative_to(repo)}")

    print(
        f"\nExported {total_examples} approved examples to {len(files_written)} file(s)."
        + (f"  ({skipped} skipped: unknown step)" if skipped else "")
    )
    return 0


# ---------------------------------------------------------------------------
# import: files → DB
# ---------------------------------------------------------------------------

def cmd_import(args: argparse.Namespace) -> int:
    repo = _repo_root()
    db_path = (repo / args.db).resolve()
    if not db_path.exists():
        print(f"DB not found: {db_path}", file=sys.stderr)
        return 1

    from mechanistic_agent.core.db import RunStore
    from mechanistic_agent.prompt_assets import (
        mechanistic_skills_root,
        normalize_call_name,
        resolve_call_name_from_step,
        CALL_TO_STEPS,
    )

    store = RunStore(db_path)
    skills_root = mechanistic_skills_root(repo)

    # Discover all few_shot.jsonl files.
    jsonl_files = sorted(skills_root.rglob("few_shot.jsonl"))

    if not jsonl_files:
        print(f"No few_shot.jsonl files found under {skills_root}", file=sys.stderr)
        return 0

    total_inserted = 0
    total_skipped = 0

    for jsonl_path in jsonl_files:
        # Infer call_name and model_name from path.
        # Expected layouts:
        #   skills/mechanistic/<call_name>/few_shot.jsonl
        #   skills/mechanistic/<call_name>/models/<model-slug>/few_shot.jsonl
        rel = jsonl_path.relative_to(skills_root)
        parts = rel.parts  # e.g. ('propose_mechanism_step', 'few_shot.jsonl')
                           # or ('propose_mechanism_step', 'models', 'anthropic__claude-opus-4-5', 'few_shot.jsonl')
        if len(parts) == 2:
            call_name = parts[0]
            model_name = None
        elif len(parts) == 4 and parts[1] == "models":
            call_name = parts[0]
            model_name = parts[2]
        else:
            # Unrecognised layout — skip.
            continue

        # Validate call_name.
        try:
            call_name = normalize_call_name(call_name)
        except ValueError:
            continue

        # Optionally filter by step.
        if args.step:
            # Accept either step_name or call_name in the --step filter.
            steps_for_call = CALL_TO_STEPS.get(call_name, [call_name])
            if args.step not in steps_for_call and args.step != call_name:
                continue

        # The step_name stored in the DB is from CALL_TO_STEPS.
        # Use the first mapped step_name (or call_name itself as fallback).
        step_names = CALL_TO_STEPS.get(call_name, [call_name])
        step_name = step_names[0]

        # Parse the JSONL file.
        examples: list[dict] = []
        try:
            raw_text = jsonl_path.read_text(encoding="utf-8")
        except OSError:
            continue
        for raw_line in raw_text.splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            input_text = obj.get("input")
            output_text = obj.get("output")
            if not isinstance(input_text, str) or not isinstance(output_text, str):
                continue
            example_key = _key_for(obj)
            examples.append(
                {
                    "step_name": step_name,
                    "example_key": example_key,
                    "input_text": input_text,
                    "output_text": output_text,
                    "score": obj.get("score"),
                }
            )

        if not examples:
            continue

        inserted = store.seed_few_shot_examples(examples)
        skipped = len(examples) - inserted
        total_inserted += inserted
        total_skipped += skipped
        label = call_name + (f" [{model_name}]" if model_name else "")
        print(f"  {label}: {inserted} imported, {skipped} already present")

    print(f"\nTotal: {total_inserted} new examples imported, {total_skipped} already present (skipped).")
    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo", default=".", help="Repository root (default: auto-detected)")
    parser.add_argument("--db", default="data/mechanistic.db", help="SQLite path relative to repo root")
    sub = parser.add_subparsers(dest="command", required=True)

    p_export = sub.add_parser("export", help="DB → files: export approved examples to .jsonl")
    p_export.add_argument("--step", default="", help="Filter by step_name or call_name")

    p_import = sub.add_parser("import", help="files → DB: seed DB from .jsonl (INSERT OR IGNORE)")
    p_import.add_argument("--step", default="", help="Filter by step_name or call_name")

    args = parser.parse_args()

    # Allow --repo to override the auto-detected root.
    if args.repo != ".":
        import os
        os.chdir(Path(args.repo).resolve())

    if args.command == "export":
        return cmd_export(args)
    if args.command == "import":
        return cmd_import(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
