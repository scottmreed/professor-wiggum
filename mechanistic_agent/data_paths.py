"""Resolve repo vs bulk-data directory layout.

Committed benchmark JSON and skills live in the Mechanistic repo (``repo_root``).
Large generated artifacts, runtime SQLite, traces, and PNG caches live in a
separate data checkout (``data_root``), defaulting to sibling ``../wiggum-data``.

Override with ``MECHANISTIC_DATA_DIR`` or ``WIGGUM_DATA_DIR``. Forks that keep
data in-repo can omit the sibling directory; resolution falls back to ``repo_root``.
"""
from __future__ import annotations

import os
from pathlib import Path

_DATA_DIR_ENV_KEYS = ("MECHANISTIC_DATA_DIR", "WIGGUM_DATA_DIR")
_PROJECT_ROOT_ENV = "MECHANISTIC_PROJECT_ROOT"


def repo_root(base_dir: Path | None = None) -> Path:
    """Mechanistic application repository root (code, committed training JSON)."""
    if base_dir is not None:
        return base_dir.resolve()
    env_root = str(os.getenv(_PROJECT_ROOT_ENV) or "").strip()
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parents[1]


def data_root(base_dir: Path | None = None) -> Path:
    """Root for bulk/local data (DB, traces, large training artifacts)."""
    for key in _DATA_DIR_ENV_KEYS:
        env = str(os.getenv(key) or "").strip()
        if env:
            return Path(env).expanduser().resolve()
    root = repo_root(base_dir)
    sibling = root.parent / "wiggum-data"
    if sibling.is_dir():
        return sibling.resolve()
    return root


def uses_external_data_root(base_dir: Path | None = None) -> bool:
    """True when bulk data is resolved outside ``repo_root``."""
    root = repo_root(base_dir)
    return data_root(base_dir) != root.resolve()


def runtime_data_dir(base_dir: Path | None = None) -> Path:
    return data_root(base_dir) / "data"


def db_path(base_dir: Path | None = None) -> Path:
    return runtime_data_dir(base_dir) / "mechanistic.db"


def data_root_for_db(db_path: Path) -> Path:
    """Infer bulk-data root from an on-disk SQLite path."""
    parent = db_path.parent.resolve()
    if parent.name == "data":
        return parent.parent
    return parent


def traces_root_for_db(db_path: Path) -> Path:
    return data_root_for_db(db_path) / "traces"


def traces_root(base_dir: Path | None = None) -> Path:
    """Runtime run traces (scratchpads, per-step JSON)."""
    return data_root(base_dir) / "traces"


def evidence_root(base_dir: Path | None = None) -> Path:
    """PR-gated prompt evidence traces (stay in the code repo)."""
    return repo_root(base_dir) / "traces" / "evidence"


def repo_training_dir(base_dir: Path | None = None) -> Path:
    """Committed eval/benchmark JSON under the code repo."""
    return repo_root(base_dir) / "training_data"


def bulk_training_dir(base_dir: Path | None = None) -> Path:
    """Large or generated training artifacts (index, PNGs, holdout)."""
    return data_root(base_dir) / "training_data"


def flower_mechanism_index_path(base_dir: Path | None = None) -> Path:
    return bulk_training_dir(base_dir) / "flower_mechanism_index.jsonl"


def flower_mechanism_index_report_path(base_dir: Path | None = None) -> Path:
    return bulk_training_dir(base_dir) / "flower_mechanism_index_report.json"


def flower_train_lookup_path(base_dir: Path | None = None) -> Path:
    return runtime_data_dir(base_dir) / "flower_train_lookup.sqlite"


def flower_test_lookup_path(base_dir: Path | None = None) -> Path:
    return runtime_data_dir(base_dir) / "flower_test_lookup.sqlite"


def flower_curriculum_index_path(base_dir: Path | None = None) -> Path:
    return runtime_data_dir(base_dir) / "flower_curriculum_index.jsonl"


def holdout_dir(base_dir: Path | None = None) -> Path:
    return bulk_training_dir(base_dir) / "leaderboard_holdout"


def holdout_eval_set_path(base_dir: Path | None = None) -> Path:
    return holdout_dir(base_dir) / "eval_set_holdout.json"


def flower_curriculum_pngs_dir(base_dir: Path | None = None) -> Path:
    return bulk_training_dir(base_dir) / "flower_curriculum_pngs"


def local_contributions_runs_dir(base_dir: Path | None = None) -> Path:
    return data_root(base_dir) / "local_contributions" / "runs"
