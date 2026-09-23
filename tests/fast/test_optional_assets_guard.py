"""Targeted tests for the optional-repo-asset skip guard.

These lock in the behaviour that lets the fast suite stay green on a fresh
clone: present assets resolve to a real path, absent assets skip the calling
test instead of erroring.  See ``tests/fast/_optional_assets.py``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from _optional_assets import PROJECT_ROOT, require_repo_asset


def test_returns_absolute_path_for_tracked_asset() -> None:
    # eval_set.json is explicitly allow-listed in .gitignore, so it is always
    # present in a checkout and must resolve to a real, absolute path.
    path = require_repo_asset("training_data/eval_set.json")
    assert isinstance(path, Path)
    assert path.is_absolute()
    assert path.exists()
    assert path.name == "eval_set.json"
    assert path == (PROJECT_ROOT / "training_data" / "eval_set.json").resolve()


def test_skips_calling_test_for_missing_asset() -> None:
    # An absent asset must raise Skipped (what pytest.skip raises) rather than
    # FileNotFoundError, so the suite degrades to a skip on a fresh checkout.
    with pytest.raises(pytest.skip.Exception):
        require_repo_asset("training_data/__does_not_exist_optional_asset__.py")
