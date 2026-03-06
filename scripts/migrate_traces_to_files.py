#!/usr/bin/env python3
"""One-time migration helper for DB traces -> traces/runs and traces/evidence."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mechanistic_agent.core.db import RunStore
from mechanistic_agent.prompt_assets import resolve_call_name_from_step, traces_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="Repository root")
    parser.add_argument("--db", default="data/mechanistic.db", help="SQLite path relative to repo")
    parser.add_argument("--promote-approved", action="store_true", help="Also write approved evidence files")
    parser.add_argument("--limit", type=int, default=50000, help="Maximum traces to migrate")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    db_path = (repo / args.db).resolve()
    store = RunStore(db_path)
    rows = store.list_trace_records(limit=max(1, args.limit), approved_only=False)

    migrated = 0
    promoted = 0
    for row in rows:
        trace_id = str(row.get("id") or "")
        step_name = str(row.get("step_name") or "")
        run_id = str(row.get("run_id") or "unassigned")
        call_name = resolve_call_name_from_step(step_name)
        stamp = int((float(row.get("created_at") or time.time())) * 1000)

        run_dir = traces_root(repo) / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = run_dir / f"{stamp}_{step_name}_{trace_id}.json"

        prompt_version = store.get_prompt_version(str(row.get("prompt_version_id"))) if row.get("prompt_version_id") else None
        model_version = store.get_model_version(str(row.get("model_version_id"))) if row.get("model_version_id") else None
        payload = {
            "trace_id": trace_id,
            "run_id": row.get("run_id"),
            "step_name": step_name,
            "call_name": call_name,
            "approved_bool": bool(row.get("approved_bool")),
            "prompt_version_id": row.get("prompt_version_id"),
            "model_version_id": row.get("model_version_id"),
            "prompt_version": prompt_version,
            "model_version": model_version,
            "trace": row.get("trace") or {},
            "migrated_at": time.time(),
        }
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        migrated += 1

        if not args.promote_approved:
            continue
        if not bool(row.get("approved_bool")):
            continue
        if not call_name:
            continue
        if not prompt_version or not model_version:
            continue
        bundle_sha = str(prompt_version.get("prompt_bundle_sha256") or prompt_version.get("sha256") or "")
        if not bundle_sha:
            continue
        evidence_dir = traces_root(repo) / "evidence" / call_name / bundle_sha
        evidence_dir.mkdir(parents=True, exist_ok=True)
        evidence_path = evidence_dir / f"{trace_id}.json"
        evidence_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        promoted += 1

    legacy_path = repo / "data" / "evaluation_traces.json"
    if legacy_path.exists():
        try:
            legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
        except Exception:
            legacy = {}
        entries = legacy.get("entries", {}) if isinstance(legacy, dict) else {}
        if isinstance(entries, dict):
            for key, entry in entries.items():
                if not isinstance(entry, dict):
                    continue
                step_name = str(entry.get("step") or "")
                call_name = resolve_call_name_from_step(step_name)
                trace = entry.get("trace") if isinstance(entry.get("trace"), dict) else {}
                stamp = int((float(entry.get("stored_at") or time.time())) * 1000)
                trace_id = str(key).replace("::", "_")
                run_dir = traces_root(repo) / "runs" / "legacy_baseline"
                run_dir.mkdir(parents=True, exist_ok=True)
                out_path = run_dir / f"{stamp}_{step_name}_{trace_id}.json"
                payload = {
                    "trace_id": trace_id,
                    "run_id": None,
                    "step_name": step_name,
                    "call_name": call_name,
                    "approved_bool": False,
                    "prompt_version_id": None,
                    "model_version_id": None,
                    "prompt_version": None,
                    "model_version": None,
                    "trace": trace,
                    "migrated_from": "data/evaluation_traces.json",
                    "migrated_at": time.time(),
                }
                out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                migrated += 1

    print(f"Migrated run traces: {migrated}")
    if args.promote_approved:
        print(f"Promoted approved evidence traces: {promoted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
