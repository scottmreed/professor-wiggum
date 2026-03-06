#!/usr/bin/env python3
"""Validate that AGENTS.md stays in sync with the codebase.

Two checks:

  1. Forward check — every *_TOOL constant defined in tool_schemas.py must be
     mentioned somewhere in AGENTS.md.  Fires when a contributor adds a new
     tool schema without documenting it.

  2. Reverse check — every /api/... path referenced in AGENTS.md must exist as
     a route in api/app.py.  Fires when a route is removed but AGENTS.md is not
     updated, leaving stale documentation.

Run locally:
    python scripts/validate_agents_md.py

Exit code 0 = OK, exit code 1 = violations found.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
TOOL_SCHEMAS = ROOT / "mechanistic_agent" / "tool_schemas.py"
APP_PY = ROOT / "mechanistic_agent" / "api" / "app.py"
AGENTS_MD = ROOT / "AGENTS.md"


def _tool_variable_names() -> list[str]:
    """Extract *_TOOL constant names from tool_schemas.py."""
    src = TOOL_SCHEMAS.read_text(encoding="utf-8")
    return re.findall(r"^([A-Z][A-Z0-9_]*_TOOL)\s*:", src, re.MULTILINE)


def _app_routes() -> set[str]:
    """Extract normalised route paths from api/app.py."""
    src = APP_PY.read_text(encoding="utf-8")
    paths = re.findall(r'@app\.(?:get|post|put|delete|patch)\("([^"]+)"', src)
    return {_normalise(p) for p in paths}


def _agents_md_api_paths() -> list[str]:
    """Extract /api/... paths quoted in backticks in AGENTS.md.

    Handles both plain paths (``/api/runs``) and paths with an HTTP method
    prefix (``POST /api/runs``, ``GET /api/runs/{id}``).
    """
    src = AGENTS_MD.read_text(encoding="utf-8")
    # Match backtick-quoted strings that contain /api/, with optional METHOD prefix
    raw = re.findall(r"`(?:(?:GET|POST|PUT|DELETE|PATCH)\s+)?(/api/[^`\s]+)[^`]*`", src)
    return raw


def _normalise(path: str) -> str:
    """Replace all {param} segments with a canonical placeholder."""
    return re.sub(r"\{[^}]+\}", "{param}", path)


def main() -> int:
    agents_text = AGENTS_MD.read_text(encoding="utf-8")
    errors: list[str] = []

    # --- Check 1: forward — all tool schema constants appear in AGENTS.md ---
    for name in _tool_variable_names():
        if name not in agents_text:
            errors.append(
                f"  '{name}' is defined in tool_schemas.py but not mentioned in AGENTS.md.\n"
                f"  Add a reference to it in the 'Forced Tool Calling' section."
            )

    # --- Check 2: reverse — all /api/... paths in AGENTS.md exist in app.py ---
    app_routes = _app_routes()
    for raw_path in _agents_md_api_paths():
        normalised = _normalise(raw_path)
        if normalised not in app_routes:
            errors.append(
                f"  '{raw_path}' is documented in AGENTS.md but no matching route was found "
                f"in api/app.py.\n"
                f"  Either restore the route or remove the stale reference from AGENTS.md."
            )

    if errors:
        print("AGENTS.md validation FAILED\n")
        for e in errors:
            print(e)
        return 1

    tool_count = len(_tool_variable_names())
    path_count = len(_agents_md_api_paths())
    print(
        f"AGENTS.md validation passed  "
        f"({tool_count} tool schemas checked, {path_count} documented routes verified)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
