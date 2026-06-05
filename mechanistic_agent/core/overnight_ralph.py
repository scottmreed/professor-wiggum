"""Overnight Ralph hill-climbing orchestrator."""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .coordinator import RunCoordinator
from .db import RunStore
from .experiment_ledger import ExperimentLedger
from .lane_mutator import (
    FewShotLaneMutator,
    HarnessLaneMutator,
    PromptLaneMutator,
    TopologyLaneMutator,
)
from .micro_eval_runner import MicroEvalRunner
from .types import ExperimentRecord, MicroEvalResult, OvernightRalphConfig, RalphLane


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class OvernightRalphStatus:
    running: bool = False
    stop_requested: bool = False
    started_at: float = 0.0
    finished_at: Optional[float] = None
    current_experiment: int = 0
    max_experiments: int = 0
    eval_slice_id: str = ""
    baseline: Optional[Dict[str, Any]] = None
    summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "stop_requested": self.stop_requested,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "current_experiment": self.current_experiment,
            "max_experiments": self.max_experiments,
            "eval_slice_id": self.eval_slice_id,
            "baseline": self.baseline,
            "summary": self.summary,
            "error": self.error,
        }


class OvernightRalphOrchestrator:
    """Runs nightly lane-scoped hill-climbing on a frozen eval slice."""

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
        self.micro_eval = MicroEvalRunner(base_dir=base_dir, store=store, coordinator=self.coordinator)
        self.ledger = ExperimentLedger(base_dir=base_dir, store=store)
        self.stop_event = threading.Event()
        self.status = OvernightRalphStatus()

    def request_stop(self) -> None:
        self.stop_event.set()
        self.status.stop_requested = True

    def run(self, *, config: OvernightRalphConfig, run_config: Dict[str, Any]) -> Dict[str, Any]:
        self.status = OvernightRalphStatus(
            running=True,
            stop_requested=False,
            started_at=time.time(),
            current_experiment=0,
            max_experiments=max(1, int(config.max_experiments)),
            eval_slice_id=config.eval_slice_id,
        )
        self.stop_event.clear()

        try:
            slice_cases = self._load_eval_slice(config)
            baseline_result = self.micro_eval.run_slice(
                eval_slice_id=config.eval_slice_id,
                cases=slice_cases,
                base_config=run_config,
                harness_config_path=run_config.get("harness_config_path"),
            )
            baseline_payload = self._result_to_dict(baseline_result)
            self.status.baseline = baseline_payload

            parent_asset = self._resolve_parent_harness_asset(run_config)
            parent_sha = _sha256_text(parent_asset.read_text(encoding="utf-8"))
            spent_cost = 0.0
            keep_count = 0
            discard_count = 0

            allowed_lanes = list(config.allowed_lanes or ["topology", "harness"])
            if not allowed_lanes:
                allowed_lanes = ["topology"]

            from mechanistic_agent.prompt_assets import traces_root

            candidates_dir = traces_root(self.base_dir) / "overnight_ralph" / "candidates"
            candidates_dir.mkdir(parents=True, exist_ok=True)

            for idx in range(1, int(config.max_experiments) + 1):
                if self.stop_event.is_set():
                    break
                if config.max_cost_usd > 0 and spent_cost >= float(config.max_cost_usd):
                    break

                lane: RalphLane = allowed_lanes[(idx - 1) % len(allowed_lanes)]  # type: ignore[assignment]
                self.status.current_experiment = idx

                mutated = self._propose_mutation(lane=lane, parent_asset=parent_asset)
                harness_override: Optional[str] = None
                if lane in {"topology", "harness"}:
                    harness_override = str(mutated.asset_path)

                result = self.micro_eval.run_slice(
                    eval_slice_id=config.eval_slice_id,
                    cases=slice_cases,
                    base_config=run_config,
                    harness_config_path=harness_override,
                )
                spent_cost += float(result.token_cost_usd)

                keep, revert_reason = self._should_keep(
                    baseline=baseline_result,
                    candidate=result,
                    acceptance_threshold_pct=float(config.acceptance_threshold_pct),
                )
                if keep:
                    keep_count += 1
                    baseline_result = result
                    parent_asset = mutated.asset_path if lane in {"topology", "harness"} else parent_asset
                    parent_sha = _sha256_text(parent_asset.read_text(encoding="utf-8"))
                    candidate_path = candidates_dir / f"exp_{idx:03d}_{lane}_{mutated.asset_path.name}"
                    shutil.copyfile(mutated.asset_path, candidate_path)
                    mutated_path = str(candidate_path)
                else:
                    discard_count += 1
                    mutated_path = str(mutated.asset_path)

                self.ledger.append(
                    ExperimentRecord(
                        parent_checkpoint_sha=parent_sha,
                        mutated_asset_path=mutated_path,
                        mutation_lane=lane,
                        mutation_summary=mutated.summary,
                        eval_slice_id=config.eval_slice_id,
                        completion_pct=result.completion_pct,
                        validator_pass_pct=result.validator_pass_pct,
                        avg_retries=result.avg_retries,
                        avg_backtracks=result.avg_backtracks,
                        token_cost_usd=result.token_cost_usd,
                        keep=keep,
                        revert_reason=revert_reason,
                        timestamp=time.time(),
                    )
                )

            summary = {
                "status": "stopped" if self.stop_event.is_set() else "completed",
                "eval_slice_id": config.eval_slice_id,
                "experiments_attempted": self.status.current_experiment,
                "max_experiments": config.max_experiments,
                "keep_count": keep_count,
                "discard_count": discard_count,
                "spent_cost_usd": round(spent_cost, 6),
                "max_cost_usd": config.max_cost_usd,
                "baseline": baseline_payload,
                "final_parent_sha": parent_sha,
            }
            self.status.summary = summary
            self.status.running = False
            self.status.finished_at = time.time()
            return summary
        except Exception as exc:
            self.status.running = False
            self.status.finished_at = time.time()
            self.status.error = str(exc)
            raise

    def _resolve_parent_harness_asset(self, run_config: Dict[str, Any]) -> Path:
        harness_path = str(run_config.get("harness_config_path") or "").strip()
        if harness_path:
            path = Path(harness_path)
            if path.exists():
                return path
        harness_name = str(run_config.get("harness_name") or "default").strip() or "default"
        path = self.base_dir / "harness_versions" / harness_name / "harness.json"
        if not path.exists():
            path = self.base_dir / "harness_versions" / "default" / "harness.json"
        if not path.exists():
            raise FileNotFoundError(f"Unable to resolve harness asset for overnight Ralph: {path}")
        return path

    def _propose_mutation(self, *, lane: RalphLane, parent_asset: Path):
        if lane == "topology":
            return TopologyLaneMutator().propose(parent_asset)
        if lane == "harness":
            return HarnessLaneMutator().propose(parent_asset)
        if lane == "prompt":
            return PromptLaneMutator(base_dir=self.base_dir).propose(parent_asset)
        if lane == "few_shot":
            return FewShotLaneMutator(base_dir=self.base_dir).propose(parent_asset)
        raise ValueError(f"Unsupported overnight lane: {lane}")

    @staticmethod
    def _composite(result: MicroEvalResult) -> float:
        quality = (float(result.completion_pct) + float(result.validator_pass_pct)) / 2.0
        cost_penalty = float(result.token_cost_usd) * 0.01 / max(1, int(result.case_count))
        return quality - cost_penalty

    def _should_keep(
        self,
        *,
        baseline: MicroEvalResult,
        candidate: MicroEvalResult,
        acceptance_threshold_pct: float,
    ) -> tuple[bool, Optional[str]]:
        threshold = max(0.0, float(acceptance_threshold_pct))
        checks = {
            "completion_pct": float(candidate.completion_pct) >= (float(baseline.completion_pct) + threshold),
            "validator_pass_pct": float(candidate.validator_pass_pct)
            >= (float(baseline.validator_pass_pct) + threshold),
            "composite": self._composite(candidate) >= (self._composite(baseline) + threshold),
        }
        if all(checks.values()):
            return True, None
        failed = [name for name, ok in checks.items() if not ok]
        return False, f"threshold_not_met:{','.join(failed)}"

    @staticmethod
    def _result_to_dict(result: MicroEvalResult) -> Dict[str, Any]:
        return {
            "eval_slice_id": result.eval_slice_id,
            "case_count": result.case_count,
            "completion_pct": result.completion_pct,
            "validator_pass_pct": result.validator_pass_pct,
            "avg_retries": result.avg_retries,
            "avg_backtracks": result.avg_backtracks,
            "token_cost_usd": result.token_cost_usd,
            "median_steps_to_completion": result.median_steps_to_completion,
            "completed_runs": result.completed_runs,
            "validator_pass_runs": result.validator_pass_runs,
        }

    def _load_eval_slice(self, config: OvernightRalphConfig) -> List[Dict[str, Any]]:
        eval_set_path = self.base_dir / "training_data" / "eval_set.json"
        raw = json.loads(eval_set_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("training_data/eval_set.json must be a list of cases")

        if config.eval_case_ids:
            wanted = {str(item) for item in config.eval_case_ids}
            selected = [case for case in raw if str(case.get("id") or case.get("case_id") or "") in wanted]
            return selected

        size = max(1, int(config.eval_slice_size))
        if size >= len(raw):
            return list(raw)

        seed = int(hashlib.sha256(config.eval_slice_id.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed)
        pool = list(raw)
        rng.shuffle(pool)
        return pool[:size]


def load_overnight_program(path: Path) -> OvernightRalphConfig:
    """Parse a lightweight key/value markdown program file."""
    text = path.read_text(encoding="utf-8")
    lines = [line.rstrip() for line in text.splitlines()]

    data: Dict[str, Any] = {}
    current_list_key: Optional[str] = None
    list_keys = {"allowed_lanes", "frozen_surfaces", "eval_case_ids"}

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-") and current_list_key:
            value = line[1:].strip()
            if value:
                data.setdefault(current_list_key, []).append(value)
            continue
        if ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        normalized = key.lower().replace(" ", "_")
        if normalized in list_keys:
            current_list_key = normalized
            data.setdefault(normalized, [])
            if value:
                data[normalized].append(value)
            continue

        current_list_key = None
        if value.startswith("[") and value.endswith("]"):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = [item.strip() for item in value.strip("[]").split(",") if item.strip()]
            data[normalized] = parsed
            continue

        lowered = value.lower()
        if lowered in {"true", "false"}:
            data[normalized] = lowered == "true"
            continue
        try:
            if "." in value:
                data[normalized] = float(value)
            else:
                data[normalized] = int(value)
            continue
        except ValueError:
            data[normalized] = value

    allowed_raw = data.get("allowed_lanes") or ["topology", "harness"]
    allowed_lanes: List[RalphLane] = []
    for lane in allowed_raw:
        text_lane = str(lane).strip()
        if text_lane in {"topology", "harness", "prompt", "few_shot"}:
            allowed_lanes.append(text_lane)  # type: ignore[arg-type]

    return OvernightRalphConfig(
        eval_slice_id=str(data.get("eval_slice_id") or "default"),
        eval_slice_size=max(1, int(data.get("eval_slice_size") or 10)),
        eval_case_ids=[str(item) for item in (data.get("eval_case_ids") or [])],
        max_experiments=max(1, int(data.get("max_experiments") or data.get("mutation_budget_per_night_experiments") or 20)),
        max_cost_usd=float(data.get("max_cost_usd") or data.get("mutation_budget_per_night_cost_usd") or 15.0),
        acceptance_threshold_pct=float(data.get("acceptance_threshold_pct") or 0.02),
        allowed_lanes=allowed_lanes or ["topology", "harness"],
        program_path=str(path),
        mutation_budget_per_night=(
            data.get("mutation_budget_per_night")
            if isinstance(data.get("mutation_budget_per_night"), dict)
            else {}
        ),
        frozen_surfaces=[str(item) for item in (data.get("frozen_surfaces") or [])],
        anti_overfitting_rules=(
            data.get("anti_overfitting_rules")
            if isinstance(data.get("anti_overfitting_rules"), dict)
            else {}
        ),
        rollback_rules=(
            data.get("rollback_rules") if isinstance(data.get("rollback_rules"), dict) else {}
        ),
        stop_conditions=(
            data.get("stop_conditions") if isinstance(data.get("stop_conditions"), dict) else {}
        ),
        trace_artifact_requirements=(
            data.get("trace_artifact_requirements")
            if isinstance(data.get("trace_artifact_requirements"), dict)
            else {}
        ),
    )
