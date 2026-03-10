"""Fixed-slice micro-evaluation runner for overnight Ralph."""
from __future__ import annotations

import statistics
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .coordinator import RunCoordinator
from .db import RunStore
from .registries import RegistrySet
from .types import MicroEvalResult


class MicroEvalRunner:
    """Runs a frozen reaction slice and aggregates quality/cost signals."""

    def __init__(
        self,
        *,
        base_dir: Path,
        store: RunStore,
        coordinator: Optional[RunCoordinator] = None,
    ) -> None:
        self.base_dir = base_dir
        self.store = store
        self.coordinator = coordinator or RunCoordinator(store)
        self.registry = RegistrySet(base_dir)

    def run_slice(
        self,
        *,
        eval_slice_id: str,
        cases: Iterable[Dict[str, Any]],
        base_config: Dict[str, Any],
        harness_config_path: Optional[str] = None,
    ) -> MicroEvalResult:
        case_list = list(cases)
        if not case_list:
            return MicroEvalResult(
                eval_slice_id=eval_slice_id,
                case_count=0,
                completion_pct=0.0,
                validator_pass_pct=0.0,
                avg_retries=0.0,
                avg_backtracks=0.0,
                token_cost_usd=0.0,
                median_steps_to_completion=0.0,
                completed_runs=0,
                validator_pass_runs=0,
            )

        completions = 0
        validator_passes = 0
        retries: List[float] = []
        backtracks: List[float] = []
        costs: List[float] = []
        completion_steps: List[int] = []

        for case in case_list:
            run_id = self._create_case_run(case=case, base_config=base_config, harness_config_path=harness_config_path)
            self.coordinator.execute_run(run_id, threading.Event())
            snapshot = self.store.get_run_snapshot(run_id) or {}

            completed = self._is_completed(snapshot)
            validator_pass = self._validator_passed(snapshot)
            retry_count = self._event_count(snapshot, "mechanism_retry_started")
            backtrack_count = self._event_count(snapshot, "backtrack")
            total_cost = self._snapshot_total_cost(snapshot)

            if completed:
                completions += 1
                completion_steps.append(self._mechanism_step_count(snapshot))
            if validator_pass:
                validator_passes += 1
            retries.append(float(retry_count))
            backtracks.append(float(backtrack_count))
            costs.append(total_cost)

        case_count = len(case_list)
        return MicroEvalResult(
            eval_slice_id=eval_slice_id,
            case_count=case_count,
            completion_pct=(completions / case_count),
            validator_pass_pct=(validator_passes / case_count),
            avg_retries=(sum(retries) / case_count),
            avg_backtracks=(sum(backtracks) / case_count),
            token_cost_usd=sum(costs),
            median_steps_to_completion=(
                float(statistics.median(completion_steps)) if completion_steps else 0.0
            ),
            completed_runs=completions,
            validator_pass_runs=validator_passes,
        )

    def _create_case_run(
        self,
        *,
        case: Dict[str, Any],
        base_config: Dict[str, Any],
        harness_config_path: Optional[str],
    ) -> str:
        starting = list(case.get("starting_materials") or case.get("input", {}).get("starting_materials") or [])
        products = list(case.get("products") or case.get("input", {}).get("products") or [])
        if not starting or not products:
            raise ValueError(f"Invalid eval case payload: missing starting/products in {case}")

        config = dict(base_config)
        if harness_config_path:
            config["harness_config_path"] = harness_config_path
        config["orchestration_mode"] = "standard"

        model_name = str(config.get("model_name") or config.get("model") or "")
        hashes = self.registry.bundle_hashes(model_name=model_name or None)
        run_id = self.store.create_run(
            mode="unverified",
            input_payload={
                "starting_materials": starting,
                "products": products,
                "temperature_celsius": case.get("temperature_celsius"),
                "ph": case.get("ph"),
                "example_id": case.get("id") or case.get("case_id"),
            },
            config=config,
            prompt_bundle_hash=hashes.get("prompt_bundle_hash", ""),
            skill_bundle_hash=hashes.get("skill_bundle_hash", ""),
            memory_bundle_hash=hashes.get("memory_bundle_hash", ""),
            harness_bundle_hash=hashes.get("harness_bundle_hash", ""),
        )
        return run_id

    @staticmethod
    def _snapshot_total_cost(snapshot: Dict[str, Any]) -> float:
        summary = snapshot.get("cost_summary") or {}
        total = (summary.get("total_cost") or {}).get("total_cost")
        try:
            return float(total or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _event_count(snapshot: Dict[str, Any], event_type: str) -> int:
        events = list(snapshot.get("events") or [])
        return sum(1 for row in events if str(row.get("event_type") or "") == event_type)

    @staticmethod
    def _is_completed(snapshot: Dict[str, Any]) -> bool:
        status = str(snapshot.get("status") or "")
        if status == "completed":
            return True
        events = list(snapshot.get("events") or [])
        return any(str(row.get("event_type") or "") == "run_completed" for row in events)

    @staticmethod
    def _mechanism_step_count(snapshot: Dict[str, Any]) -> int:
        rows = list(snapshot.get("step_outputs") or [])
        return sum(1 for row in rows if str(row.get("step_name") or "") == "mechanism_synthesis")

    @staticmethod
    def _validator_passed(snapshot: Dict[str, Any]) -> bool:
        rows = [
            row
            for row in (snapshot.get("step_outputs") or [])
            if str(row.get("step_name") or "") == "mechanism_synthesis"
        ]
        if not rows:
            return False
        for row in rows:
            validation = row.get("validation") or {}
            if isinstance(validation, dict) and validation.get("passed") is False:
                return False
        return True
