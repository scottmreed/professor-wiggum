"""The installed package must carry ``skills`` (Observatory PRD rev 3 §2.3.2).

``core/validators.py`` imports ``skills.mechanistic.<name>.validator`` and the
registries read ``skills/**/SKILL.md``. An embedding product installs this repo
with ``pip install -e /opt/wiggum`` and imports it from another working
directory, so ``skills`` has to be a discovered package, not a repo-root
side effect. Regression for the 2026-09-29 Railway build failure
(``ModuleNotFoundError: No module named 'skills'``).
"""
from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_pyproject_discovers_skills_as_a_namespace_package() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    find = config["tool"]["setuptools"]["packages"]["find"]
    assert "skills*" in find["include"]
    assert find.get("namespaces") is True, "skill directories have no __init__.py"
    package_data = config["tool"]["setuptools"]["package-data"]["skills"]
    assert "**/*.md" in package_data and "**/*.py" in package_data and "**/*.jsonl" in package_data


def test_validators_import_path_matches_package_layout() -> None:
    assert (ROOT / "skills" / "__init__.py").exists()
    assert (ROOT / "skills" / "mechanistic" / "atom_balance_validation" / "validator.py").exists()
