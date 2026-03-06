"""Structural tests enforcing module layer boundaries.

Layer rules enforced here:

  core/     → may import: config, types, model_registry, prompt_assets, llm, tools (siblings)
  core/     → must NOT import: api/
  types.py  → must NOT import from within mechanistic_agent (leaf node)
  config.py → must NOT import from core/ or api/

These rules mirror the harness engineering principle: architectural constraints
are what allows speed without decay or drift as contributors submit PRs.

When a rule fires the assertion message names the offending file, line, and the
violated constraint so the contributor knows exactly what to fix.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Tuple

PACKAGE = Path(__file__).resolve().parents[2] / "mechanistic_agent"


def _resolve_relative_import(filepath: Path, level: int, module: str) -> str:
    """Resolve a relative import to its absolute mechanistic_agent module name.

    e.g. in core/coordinator.py:  ``from ..api import x``  (level=2, module='api')
    resolves to ``mechanistic_agent.api``.
    """
    rel_parts = list(filepath.relative_to(PACKAGE).with_suffix("").parts)
    # Strip `level` components from the right (level=1 means same package)
    base_parts = rel_parts[: len(rel_parts) - level + 1] if level <= len(rel_parts) else []
    suffix = [module] if module else []
    return ".".join(["mechanistic_agent"] + base_parts + suffix)


def _get_internal_imports(filepath: Path) -> List[Tuple[str, int]]:
    """Return (absolute_module, lineno) for every mechanistic_agent import in filepath."""
    src = filepath.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(filepath))
    results: List[Tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("mechanistic_agent"):
                    results.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level > 0:
                abs_mod = _resolve_relative_import(filepath, node.level, module)
                results.append((abs_mod, node.lineno))
            elif module.startswith("mechanistic_agent"):
                results.append((module, node.lineno))
    return results


def test_core_does_not_import_api():
    """No module in core/ may import from mechanistic_agent.api.

    Dependency direction is strictly: api/ → core/.  Reversing it creates a
    circular dependency and couples the service boundary into the runtime.
    """
    violations: List[str] = []
    for py_file in (PACKAGE / "core").rglob("*.py"):
        for module, lineno in _get_internal_imports(py_file):
            if module.startswith("mechanistic_agent.api"):
                rel = py_file.relative_to(PACKAGE.parent)
                violations.append(f"  {rel}:{lineno}  imports '{module}'")

    assert not violations, (
        "Layer violation — core/ must not import from api/.\n"
        "The allowed dependency direction is api/ → core/, not the reverse.\n"
        "Move shared logic to a sibling module (e.g. config.py, model_registry.py) "
        "if both layers need it.\n"
        "Violations:\n" + "\n".join(violations)
    )


def test_types_is_leaf():
    """core/types.py must not import from within mechanistic_agent.

    types.py is the leaf node of the dependency graph — pure stdlib + dataclasses.
    Nothing in the package should depend on a higher layer just to get a type.
    """
    types_file = PACKAGE / "core" / "types.py"
    violations = [
        f"  types.py:{ln}  imports '{m}'"
        for m, ln in _get_internal_imports(types_file)
        if m.startswith("mechanistic_agent")
    ]
    assert not violations, (
        "Layer violation — core/types.py is a leaf module.\n"
        "It must not import from within mechanistic_agent.\n"
        "Define new types here using only stdlib (dataclasses, typing, etc.).\n"
        "Violations:\n" + "\n".join(violations)
    )


def test_config_does_not_import_core_or_api():
    """config.py must not import from core/ or api/.

    config.py sits above model_registry in the dependency graph but below core/.
    Importing from core/ or api/ would create a cycle.
    """
    config_file = PACKAGE / "config.py"
    violations = [
        f"  config.py:{ln}  imports '{m}'"
        for m, ln in _get_internal_imports(config_file)
        if m.startswith("mechanistic_agent.core") or m.startswith("mechanistic_agent.api")
    ]
    assert not violations, (
        "Layer violation — config.py must not import from core/ or api/.\n"
        "It should only depend on model_registry and stdlib/pydantic.\n"
        "Violations:\n" + "\n".join(violations)
    )
