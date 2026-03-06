"""Validate that PR run traces use verified (non-manual) model strategies.

This script checks changed trace files in a PR to ensure:
1. The run's model_strategy is not "manual" (custom model calls forbidden for external PRs)
2. The run's harness version + step models match a known verification result

Usage:
    python scripts/validate_pr_model_strategy.py --base-ref <sha> --head-ref <sha>
    python scripts/validate_pr_model_strategy.py  # defaults to origin/main vs HEAD
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_changed_files(base_ref: str, head_ref: str) -> list[str]:
    """Get list of files changed between two refs."""
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...{head_ref}"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"Warning: git diff failed: {result.stderr.strip()}")
        return []
    return [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]


def load_trace_file(path: Path) -> dict | None:
    """Load and parse a JSON trace file."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def validate_trace(trace: dict, db_path: Path) -> tuple[bool, str]:
    """Validate a single trace against verification requirements.

    Returns (passed, message).
    """
    config = trace.get("config") or {}
    strategy = config.get("model_strategy", "")

    # Rule 1: manual strategy is forbidden for external PRs
    if strategy == "manual":
        return False, f"Manual model strategy is not allowed in PR submissions (run {trace.get('id', '?')})"

    # Rule 2: Check harness version + model set matches verification results
    step_models = config.get("step_models") or {}
    model_family = config.get("model_family", "")
    prompt_hash = trace.get("prompt_bundle_hash", "")
    skill_hash = trace.get("skill_bundle_hash", "")
    memory_hash = trace.get("memory_bundle_hash", "")

    if not (prompt_hash and skill_hash and memory_hash):
        return False, f"Missing bundle hashes in trace (run {trace.get('id', '?')})"

    # Compute harness version
    import hashlib
    combined = "|".join([prompt_hash, skill_hash, memory_hash])
    harness_version = hashlib.sha256(combined.encode("utf-8")).hexdigest()

    # Look up verification results
    if db_path.exists():
        from mechanistic_agent.core.db import RunStore
        store = RunStore(db_path)
        verified = store.get_verified_step_models(
            harness_version=harness_version, model_family=model_family
        )
        if not verified:
            return False, (
                f"No verification results found for harness version {harness_version[:12]} "
                f"and family {model_family} (run {trace.get('id', '?')})"
            )

        # Check that step models match verified set
        for step, expected in verified.items():
            actual_model = step_models.get(step)
            if actual_model and actual_model != expected.get("model"):
                return False, (
                    f"Step '{step}' uses model '{actual_model}' but verified model is "
                    f"'{expected['model']}' for version {harness_version[:12]} "
                    f"(run {trace.get('id', '?')})"
                )

    return True, "OK"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate PR model strategies")
    parser.add_argument("--base-ref", default="origin/main", help="Base git ref")
    parser.add_argument("--head-ref", default="HEAD", help="Head git ref")
    args = parser.parse_args()

    changed = get_changed_files(args.base_ref, args.head_ref)
    trace_files = [
        f for f in changed
        if f.startswith("traces/") and f.endswith(".json")
    ]

    if not trace_files:
        print("No trace files changed in this PR. Skipping validation.")
        return 0

    db_path = PROJECT_ROOT / "data" / "mechanistic.db"
    failures: list[str] = []

    for rel_path in trace_files:
        full_path = PROJECT_ROOT / rel_path
        if not full_path.exists():
            continue

        trace = load_trace_file(full_path)
        if trace is None:
            print(f"Warning: Could not parse {rel_path}")
            continue

        passed, message = validate_trace(trace, db_path)
        if not passed:
            failures.append(message)
            print(f"FAIL: {message}")
        else:
            print(f"OK: {rel_path}")

    if failures:
        print(f"\n{len(failures)} trace(s) failed validation.")
        return 1

    print(f"\nAll {len(trace_files)} trace(s) passed validation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
