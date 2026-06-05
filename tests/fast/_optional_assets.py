"""Helpers for fast tests that depend on optional, untracked repo assets.

A handful of fast tests exercise authored helper scripts / notebooks that live
under ``training_data/``.  The repository's ``.gitignore`` uses a blanket
``training_data/*`` rule with an explicit allow-list of specific data files, so
those helper *source* files are not tracked in version control.  On a fresh
clone (and therefore in CI and for any new contributor) the files are simply
absent, which previously turned the import-time load into a hard
``FileNotFoundError`` and left the fast suite red.

The fast suite is the project's merge gate (see SOUL.md, "Guardrail 4: The fast
test suite is the merge gate").  A gate that cannot go green on a clean checkout
blocks the whole automated-evolution loop, so these tests must degrade to a
*skip* when their optional asset is missing -- while still running and asserting
normally wherever the asset is present (a maintainer machine, or CI that
regenerates it).

``require_repo_asset`` centralises that "load it, or skip with a clear reason"
behaviour.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# tests/fast/_optional_assets.py -> parents[2] is the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def require_repo_asset(relpath: str) -> Path:
    """Return the absolute path to a repo asset, or skip the calling test.

    Parameters
    ----------
    relpath:
        Path to the asset relative to the repository root, e.g.
        ``"training_data/template_confidence_calibrator.py"``.

    Returns
    -------
    Path
        The resolved absolute path when the asset exists.

    Notes
    -----
    When the asset is absent this calls :func:`pytest.skip`, which raises
    ``Skipped`` and marks the calling test as skipped (not failed).  This keeps
    the fast suite green on a fresh checkout where intentionally-untracked
    ``training_data/`` helpers are unavailable.
    """
    path = (PROJECT_ROOT / relpath).resolve()
    if not path.exists():
        pytest.skip(
            f"optional repo asset '{relpath}' is not present "
            "(not tracked in git; see tests/fast/_optional_assets.py); skipping"
        )
    return path
