"""Tests for bulk data directory resolution."""
from __future__ import annotations

from pathlib import Path

from mechanistic_agent.data_paths import (
    bulk_training_dir,
    data_root,
    db_path,
    repo_root,
    traces_root,
)


def test_data_root_defaults_to_repo_when_no_sibling(tmp_path: Path) -> None:
    assert data_root(tmp_path) == tmp_path.resolve()
    assert db_path(tmp_path) == tmp_path / "data" / "mechanistic.db"
    assert traces_root(tmp_path) == tmp_path / "traces"
    assert bulk_training_dir(tmp_path) == tmp_path / "training_data"


def test_data_root_uses_sibling_wiggum_data(tmp_path: Path) -> None:
    repo = tmp_path / "professor-wiggum"
    sibling = tmp_path / "wiggum-data"
    repo.mkdir()
    sibling.mkdir()
    assert data_root(repo) == sibling.resolve()
    assert db_path(repo) == sibling / "data" / "mechanistic.db"
    assert repo_root(repo) == repo.resolve()
