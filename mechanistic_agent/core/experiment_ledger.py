"""Experiment ledger for overnight Ralph (SQLite + JSONL)."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .db import RunStore
from .types import ExperimentRecord


class ExperimentLedger:
    """Persists overnight experiments to both DB and trace JSONL."""

    def __init__(self, *, base_dir: Path, store: RunStore) -> None:
        self.base_dir = base_dir
        self.store = store
        self._dir = base_dir / "traces" / "overnight_ralph"
        self._dir.mkdir(parents=True, exist_ok=True)

    def append(self, record: ExperimentRecord) -> str:
        if record.timestamp <= 0:
            record.timestamp = time.time()

        row_id = self.store.create_ralph_experiment(
            parent_sha=record.parent_checkpoint_sha,
            mutation_lane=record.mutation_lane,
            mutation_summary=record.mutation_summary,
            mutated_asset_path=record.mutated_asset_path,
            eval_slice_id=record.eval_slice_id,
            completion_pct=record.completion_pct,
            validator_pass_pct=record.validator_pass_pct,
            avg_retries=record.avg_retries,
            avg_backtracks=record.avg_backtracks,
            token_cost_usd=record.token_cost_usd,
            keep=record.keep,
            revert_reason=record.revert_reason,
        )
        record.experiment_id = row_id

        payload = self._record_to_dict(record)
        with self._jsonl_path(record.timestamp).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        return row_id

    def list(self, *, eval_slice_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.store.list_ralph_experiments(eval_slice_id=eval_slice_id)

    def tail(self, *, limit: int = 20, eval_slice_id: Optional[str] = None) -> List[Dict[str, Any]]:
        rows = self.list(eval_slice_id=eval_slice_id)
        return rows[: max(1, int(limit))]

    def _jsonl_path(self, ts: float) -> Path:
        day = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        return self._dir / f"{day}_ledger.jsonl"

    @staticmethod
    def _record_to_dict(record: ExperimentRecord) -> Dict[str, Any]:
        return {
            "experiment_id": record.experiment_id,
            "parent_checkpoint_sha": record.parent_checkpoint_sha,
            "mutated_asset_path": record.mutated_asset_path,
            "mutation_lane": record.mutation_lane,
            "mutation_summary": record.mutation_summary,
            "eval_slice_id": record.eval_slice_id,
            "completion_pct": record.completion_pct,
            "validator_pass_pct": record.validator_pass_pct,
            "avg_retries": record.avg_retries,
            "avg_backtracks": record.avg_backtracks,
            "token_cost_usd": record.token_cost_usd,
            "keep": record.keep,
            "revert_reason": record.revert_reason,
            "timestamp": record.timestamp,
        }
