"""Pre-release verification check.

Ensures that verification results exist for the current harness version
across all supported model families before a release.

Usage:
    python scripts/pre_release_verification.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED_FAMILIES = ["openai", "claude", "gemini"]


def main() -> int:
    from mechanistic_agent.core.registries import RegistrySet
    from mechanistic_agent.core.db import RunStore

    db_path = PROJECT_ROOT / "data" / "mechanistic.db"
    if not db_path.exists():
        print(f"FAIL: Database not found at {db_path}")
        return 1

    registry = RegistrySet(PROJECT_ROOT)
    harness_version = registry.harness_version()
    store = RunStore(db_path)

    print(f"Harness version: {harness_version[:12]}...")

    missing: list[str] = []
    for family in REQUIRED_FAMILIES:
        results = store.get_verified_step_models(
            harness_version=harness_version, model_family=family
        )
        if results:
            steps = ", ".join(sorted(results.keys()))
            print(f"  {family}: verified ({len(results)} steps: {steps})")
        else:
            print(f"  {family}: MISSING verification results")
            missing.append(family)

    if missing:
        print(f"\nFAIL: Missing verification for families: {', '.join(missing)}")
        print("Run verification for these families before releasing.")
        return 1

    print("\nAll families verified. Ready for release.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
