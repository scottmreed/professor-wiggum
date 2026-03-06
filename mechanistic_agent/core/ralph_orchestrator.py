"""Outer-loop Ralph orchestration for full-flow retries with harness mutation."""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from .db import RunStore
from .registries import HarnessRegistry
from .types import HarnessConfig, RalphConfig, RunConfig, RunState

if TYPE_CHECKING:  # pragma: no cover
    from .coordinator import RunCoordinator


class RalphOrchestrator:
    """Runs full-pipeline child attempts and mutates harness config between attempts."""

    def __init__(self, *, store: RunStore, coordinator: "RunCoordinator") -> None:
        self.store = store
        self.coordinator = coordinator

    def run(
        self,
        *,
        parent_run_id: str,
        parent_row: Dict[str, Any],
        state: RunState,
        stop_event: Any,
    ) -> None:
        config = self._build_config(state.run_config)
        harness_pool = self._resolve_harness_pool(state.run_config)
        if not harness_pool:
            harness_pool = ["default"]

        start = time.monotonic()
        cumulative_cost = 0.0
        repeat_signature_counts: Dict[str, int] = {}
        warned_levels: set[float] = set()
        latest_snapshot: Dict[str, Any] = {}
        last_harness_sha = ""
        tmp_paths: List[Path] = []

        attempt_index = 0
        while True:
            attempt_index += 1
            if config.max_iterations > 0 and attempt_index > int(config.max_iterations):
                self._terminal_stop(
                    parent_run_id,
                    "max_iterations",
                    attempt_index=int(config.max_iterations),
                    cost=cumulative_cost,
                )
                self._cleanup_tmp_files(tmp_paths)
                return

            if stop_event.is_set():
                self._terminal_stop(parent_run_id, "manual_stop")
                return

            elapsed = time.monotonic() - start
            if elapsed > config.max_runtime_seconds:
                self._terminal_stop(parent_run_id, "max_runtime_seconds")
                return

            base_harness_name = harness_pool[(attempt_index - 1) % len(harness_pool)]
            harness = self._load_harness(base_harness_name)
            parent_harness_sha = str(harness.version or "")

            mutation_actions: List[Dict[str, Any]] = []
            diff_summary: Dict[str, Any] = {}
            if attempt_index > 1:
                previous_votes = [
                    item
                    for item in self.store.list_ralph_votes(parent_run_id)
                    if int(item.get("attempt_index") or 0) == (attempt_index - 1)
                ]
                harness, mutation_actions, diff_summary = self._mutate_harness(
                    harness=harness,
                    latest_snapshot=latest_snapshot,
                    allow_validator_mutation=config.allow_validator_mutation,
                    votes=previous_votes,
                )
                if not mutation_actions:
                    self._terminal_stop(parent_run_id, "no_mutation_actions_remaining")
                    return

            harness_path, harness_sha = self._write_ephemeral_harness(
                parent_run_id=parent_run_id,
                attempt_index=attempt_index,
                harness=harness,
            )
            tmp_paths.append(harness_path)
            last_harness_sha = harness_sha

            child_config = dict(parent_row.get("config") or {})
            child_config["orchestration_mode"] = "standard"
            child_config["ralph_parent_run_id"] = parent_run_id
            child_config["harness_name"] = base_harness_name
            child_config["harness_config_path"] = str(harness_path)

            child_run_id = self.store.create_run(
                mode=str(parent_row.get("mode") or "unverified"),
                input_payload=dict(parent_row.get("input_payload") or {}),
                config=child_config,
                prompt_bundle_hash=str(parent_row.get("prompt_bundle_hash") or ""),
                skill_bundle_hash=str(parent_row.get("skill_bundle_hash") or ""),
                memory_bundle_hash=str(parent_row.get("memory_bundle_hash") or ""),
            )
            attempt_id = self.store.create_ralph_attempt(
                parent_run_id=parent_run_id,
                attempt_index=attempt_index,
                child_run_id=child_run_id,
                harness_name=base_harness_name,
                parent_harness_sha=parent_harness_sha,
                harness_sha=harness_sha,
                mutation_actions=mutation_actions,
                diff_summary=diff_summary,
            )
            self.store.append_event(
                parent_run_id,
                "ralph_iteration_started",
                {
                    "attempt_index": attempt_index,
                    "child_run_id": child_run_id,
                    "harness_name": base_harness_name,
                    "parent_harness_sha": parent_harness_sha,
                    "harness_sha": harness_sha,
                    "mutation_actions": mutation_actions,
                    "diff_summary": diff_summary,
                },
            )
            if mutation_actions:
                self.store.append_event(
                    parent_run_id,
                    "ralph_harness_mutated",
                    {
                        "attempt_index": attempt_index,
                        "child_run_id": child_run_id,
                        "actions": mutation_actions,
                        "diff_summary": diff_summary,
                    },
                )

            self.coordinator.execute_run(child_run_id, stop_event)
            latest_snapshot = self.store.get_run_snapshot(child_run_id) or {"id": child_run_id}

            child_cost = self._snapshot_total_cost(latest_snapshot)
            cumulative_cost += child_cost
            promise_met, promise_details = self._evaluate_completion_promise(
                completion_promise=config.completion_promise,
                snapshot=latest_snapshot,
            )
            self.store.complete_ralph_attempt(
                attempt_id=attempt_id,
                stop_reason=self._attempt_reason(latest_snapshot),
                completion_promise_met=promise_met,
                cost_usd=child_cost,
            )
            self.store.append_event(
                parent_run_id,
                "ralph_completion_promise_check",
                {
                    "attempt_index": attempt_index,
                    "child_run_id": child_run_id,
                    "completion_promise": config.completion_promise,
                    "met": promise_met,
                    "details": promise_details,
                },
            )

            for level in (0.5, 0.8, 1.0):
                if level in warned_levels:
                    continue
                if config.max_cost_usd and config.max_cost_usd > 0:
                    if cumulative_cost >= config.max_cost_usd * level:
                        warned_levels.add(level)
                        self.store.append_event(
                            parent_run_id,
                            "ralph_budget_warning",
                            {
                                "level": level,
                                "cumulative_cost_usd": cumulative_cost,
                                "max_cost_usd": config.max_cost_usd,
                                "attempt_index": attempt_index,
                            },
                        )

            failure_sig = self._failure_signature(latest_snapshot)
            repeat_signature_counts[failure_sig] = repeat_signature_counts.get(failure_sig, 0) + 1

            if promise_met:
                self.store.set_run_status(parent_run_id, "completed")
                self.store.append_event(
                    parent_run_id,
                    "run_completed",
                    {
                        "reason": "ralph_completion_promise_met",
                        "attempt_index": attempt_index,
                        "child_run_id": child_run_id,
                        "completion_promise": config.completion_promise,
                        "cumulative_cost_usd": cumulative_cost,
                    },
                )
                self.store.append_event(
                    parent_run_id,
                    "ralph_stopped",
                    {
                        "reason": "completed",
                        "attempt_index": attempt_index,
                        "child_run_id": child_run_id,
                        "harness_sha": last_harness_sha,
                        "cumulative_cost_usd": cumulative_cost,
                    },
                )
                self._cleanup_tmp_files(tmp_paths)
                return

            if stop_event.is_set():
                self._terminal_stop(parent_run_id, "manual_stop")
                self._cleanup_tmp_files(tmp_paths)
                return

            if config.max_cost_usd is not None and cumulative_cost >= float(config.max_cost_usd):
                self._terminal_stop(parent_run_id, "max_cost_usd", attempt_index=attempt_index, cost=cumulative_cost)
                self._cleanup_tmp_files(tmp_paths)
                return

            if repeat_signature_counts.get(failure_sig, 0) >= max(1, int(config.repeat_failure_signature_limit)):
                self._terminal_stop(
                    parent_run_id,
                    "repeat_failure_signature_limit",
                    attempt_index=attempt_index,
                    cost=cumulative_cost,
                )
                self._cleanup_tmp_files(tmp_paths)
                return

    @staticmethod
    def _build_config(run_config: RunConfig) -> RalphConfig:
        return RalphConfig(
            max_iterations=max(0, int(run_config.max_iterations or 0)),
            completion_promise="target_products_reached && flow_node:run_complete",
            max_runtime_seconds=max(1.0, float(run_config.ralph_max_runtime_seconds or 6000.0)),
            max_cost_usd=(
                float(run_config.max_cost_usd)
                if run_config.max_cost_usd is not None
                else None
            ),
            repeat_failure_signature_limit=max(
                1, int(run_config.repeat_failure_signature_limit or 2)
            ),
            harness_strategy=run_config.harness_strategy,
            harness_list=list(run_config.harness_list or []),
            babysit_mode=run_config.babysit_mode,
            allow_validator_mutation=bool(run_config.allow_validator_mutation),
        )

    @staticmethod
    def _resolve_harness_pool(run_config: RunConfig) -> List[str]:
        explicit = [str(x).strip() for x in (run_config.harness_list or []) if str(x).strip()]
        if run_config.harness_strategy == "portfolio" and explicit:
            return explicit
        name = str(run_config.harness_name or "default").strip() or "default"
        if explicit and run_config.harness_strategy == "portfolio":
            return explicit
        return [name]

    @staticmethod
    def _load_harness(name: str) -> HarnessConfig:
        base_dir = Path(__file__).resolve().parents[2]
        registry = HarnessRegistry(base_dir / "harness_versions")
        try:
            return registry.load(name)
        except Exception:
            return registry.load("default")

    @staticmethod
    def _write_ephemeral_harness(
        *,
        parent_run_id: str,
        attempt_index: int,
        harness: HarnessConfig,
    ) -> Tuple[Path, str]:
        payload = harness.as_dict()
        text = json.dumps(payload, sort_keys=True, indent=2)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix=f"ralph_{parent_run_id}_{attempt_index}_",
            delete=False,
            encoding="utf-8",
        )
        with handle:
            handle.write(text)
        return Path(handle.name), digest

    @staticmethod
    def _snapshot_total_cost(snapshot: Dict[str, Any]) -> float:
        summary = snapshot.get("cost_summary") or {}
        total = (summary.get("total_cost") or {}).get("total_cost")
        try:
            return float(total or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _evaluate_completion_promise(
        *,
        completion_promise: str,
        snapshot: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, bool]]:
        expr = str(completion_promise or "").strip()
        atoms = [item.strip() for item in expr.split("&&") if item.strip()]
        if not atoms:
            atoms = ["target_products_reached", "flow_node:run_complete"]
        results: Dict[str, bool] = {}
        for atom in atoms:
            results[atom] = RalphOrchestrator._evaluate_promise_atom(atom, snapshot)
        return all(results.values()), results

    @staticmethod
    def _evaluate_promise_atom(atom: str, snapshot: Dict[str, Any]) -> bool:
        events = snapshot.get("events") or []
        event_types = {str(ev.get("event_type") or "") for ev in events if isinstance(ev, dict)}
        step_outputs = snapshot.get("step_outputs") or []
        status = str(snapshot.get("status") or "")

        if atom == "run_complete":
            return status == "completed" or "run_completed" in event_types
        if atom == "target_products_reached":
            if "target_products_detected" in event_types:
                return True
            for row in step_outputs:
                output = row.get("output") if isinstance(row, dict) else {}
                if isinstance(output, dict) and bool(output.get("contains_target_product")):
                    return True
            return False
        if atom.startswith("event:"):
            target_event = atom.split(":", 1)[1].strip()
            return bool(target_event) and target_event in event_types
        if atom.startswith("flow_node:"):
            node_id = atom.split(":", 1)[1].strip()
            if node_id == "run_complete":
                return status == "completed" or "run_completed" in event_types
            return any(str(row.get("step_name") or "") == node_id for row in step_outputs)
        if atom.startswith("exact:"):
            marker = atom.split(":", 1)[1]
            summary_text = (
                f"status={status} "
                f"events={','.join(sorted(event_types))} "
                f"failed_reason={RalphOrchestrator._latest_failed_reason(snapshot)}"
            )
            return marker in summary_text
        return False

    @staticmethod
    def _latest_failed_reason(snapshot: Dict[str, Any]) -> str:
        events = list(snapshot.get("events") or [])
        events.sort(key=lambda x: int(x.get("seq") or 0), reverse=True)
        for event in events:
            if str(event.get("event_type") or "") == "run_failed":
                payload = event.get("payload") or {}
                return str(payload.get("reason") or "")
        return ""

    @staticmethod
    def _progress_metric(snapshot: Dict[str, Any]) -> int:
        step_outputs = snapshot.get("step_outputs") or []
        validated_steps = 0
        target_bonus = 0
        for row in step_outputs:
            if not isinstance(row, dict):
                continue
            if row.get("step_name") != "mechanism_synthesis":
                continue
            validation = row.get("validation") or {}
            if isinstance(validation, dict) and validation.get("passed"):
                validated_steps += 1
            output = row.get("output") or {}
            if isinstance(output, dict) and bool(output.get("contains_target_product")):
                target_bonus = 1000
        return validated_steps + target_bonus

    @staticmethod
    def _failure_signature(snapshot: Dict[str, Any]) -> str:
        reason = RalphOrchestrator._latest_failed_reason(snapshot).strip()
        if reason:
            return f"failed:{reason}"
        pause = snapshot.get("latest_pause") or {}
        if isinstance(pause, dict):
            pause_reason = str(pause.get("reason") or "").strip()
            if pause_reason:
                return f"paused:{pause_reason}"
        return f"status:{snapshot.get('status')}"

    @staticmethod
    def _attempt_reason(snapshot: Dict[str, Any]) -> str:
        reason = RalphOrchestrator._latest_failed_reason(snapshot)
        if reason:
            return reason
        latest_pause = snapshot.get("latest_pause") or {}
        if isinstance(latest_pause, dict) and latest_pause.get("reason"):
            return str(latest_pause.get("reason"))
        return str(snapshot.get("status") or "unknown")

    @staticmethod
    def _failed_validator_counts(snapshot: Dict[str, Any]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        events = snapshot.get("events") or []
        for event in events:
            if not isinstance(event, dict):
                continue
            if str(event.get("event_type") or "") != "mechanism_retry_failed":
                continue
            payload = event.get("payload") or {}
            checks = payload.get("failed_checks") or []
            for name in checks:
                check_name = str(name or "").strip()
                if not check_name:
                    continue
                counts[check_name] = counts.get(check_name, 0) + 1
        return counts

    @staticmethod
    def _contains_failure_reason(snapshot: Dict[str, Any], *needles: str) -> bool:
        reason = RalphOrchestrator._latest_failed_reason(snapshot).lower()
        if any(needle.lower() in reason for needle in needles):
            return True
        events = snapshot.get("events") or []
        for event in events:
            if not isinstance(event, dict):
                continue
            if str(event.get("event_type") or "") != "mechanism_reproposal_requested":
                continue
            payload = event.get("payload") or {}
            event_reason = str(payload.get("reason") or "").lower()
            if any(needle.lower() in event_reason for needle in needles):
                return True
        return False

    @staticmethod
    def _module_by_id(harness: HarnessConfig, module_id: str):
        for module in harness.pre_loop_modules + harness.post_step_modules:
            if module.id == module_id:
                return module
        return None

    def _mutate_harness(
        self,
        *,
        harness: HarnessConfig,
        latest_snapshot: Dict[str, Any],
        allow_validator_mutation: bool,
        votes: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[HarnessConfig, List[Dict[str, Any]], Dict[str, Any]]:
        mutated = copy.deepcopy(harness)
        actions: List[Dict[str, Any]] = []

        if self._contains_failure_reason(latest_snapshot, "invalid_smiles", "invalid_valence"):
            loop_module = dict(mutated.loop_module or {})
            old_candidates = int(loop_module.get("max_candidates") or 3)
            old_retries = int(loop_module.get("max_retries_per_candidate") or 3)
            new_candidates = min(6, old_candidates + 1)
            new_retries = min(5, old_retries + 1)
            if new_candidates != old_candidates:
                loop_module["max_candidates"] = new_candidates
                actions.append(
                    {
                        "type": "loop_tuning",
                        "field": "max_candidates",
                        "from": old_candidates,
                        "to": new_candidates,
                    }
                )
            if new_retries != old_retries:
                loop_module["max_retries_per_candidate"] = new_retries
                actions.append(
                    {
                        "type": "loop_tuning",
                        "field": "max_retries_per_candidate",
                        "from": old_retries,
                        "to": new_retries,
                    }
                )
            mutated.loop_module = loop_module

        vote_rows = list(votes or [])
        if vote_rows:
            vote_a = 0
            vote_b = 0
            for row in vote_rows:
                choice = str(row.get("vote") or "").upper()
                if choice == "A":
                    vote_a += 1
                elif choice == "B":
                    vote_b += 1
            if vote_b > vote_a:
                loop_module = dict(mutated.loop_module or {})
                old_candidates = int(loop_module.get("max_candidates") or 3)
                new_candidates = min(6, old_candidates + 1)
                if new_candidates != old_candidates:
                    loop_module["max_candidates"] = new_candidates
                    mutated.loop_module = loop_module
                    actions.append(
                        {
                            "type": "babysit_bias",
                            "field": "max_candidates",
                            "from": old_candidates,
                            "to": new_candidates,
                            "votes_a": vote_a,
                            "votes_b": vote_b,
                        }
                    )

        if self._contains_failure_reason(latest_snapshot, "all_candidates_rejected", "incomplete"):
            for module_id in ("missing_reagents", "atom_mapping"):
                module = self._module_by_id(mutated, module_id)
                if module is not None and not module.enabled:
                    module.enabled = True
                    actions.append(
                        {
                            "type": "module_toggle",
                            "module_id": module_id,
                            "enabled": True,
                        }
                    )

        if allow_validator_mutation:
            failed = self._failed_validator_counts(latest_snapshot)
            if failed:
                candidate = max(failed.items(), key=lambda item: item[1])[0]
                module = self._module_by_id(mutated, candidate)
                if module is not None and module.enabled:
                    module.enabled = False
                    actions.append(
                        {
                            "type": "validator_toggle",
                            "module_id": candidate,
                            "enabled": False,
                            "reason": "high_failure_frequency",
                            "fail_count": failed.get(candidate, 0),
                        }
                    )

        if not actions:
            current = str(mutated.tool_calling_mode or "forced").strip().lower() or "forced"
            new_mode = "auto" if current == "forced" else "forced"
            mutated.tool_calling_mode = new_mode
            actions.append(
                {
                    "type": "tool_calling_mode",
                    "from": current,
                    "to": new_mode,
                }
            )

        diff_summary = {
            "action_count": len(actions),
            "pre_loop_enabled": [m.id for m in mutated.pre_loop_modules if m.enabled],
            "post_step_enabled": [m.id for m in mutated.post_step_modules if m.enabled],
            "tool_calling_mode": mutated.tool_calling_mode,
            "loop_module": dict(mutated.loop_module or {}),
        }
        return mutated, actions, diff_summary

    def _terminal_stop(
        self,
        parent_run_id: str,
        reason: str,
        *,
        attempt_index: Optional[int] = None,
        cost: Optional[float] = None,
    ) -> None:
        if reason == "manual_stop":
            self.store.set_run_status(parent_run_id, "stopped")
            self.store.append_event(
                parent_run_id,
                "run_stopped",
                {"reason": "ralph_manual_stop", "attempt_index": attempt_index, "cumulative_cost_usd": cost},
            )
        else:
            self.store.set_run_status(parent_run_id, "failed")
            self.store.append_event(
                parent_run_id,
                "run_failed",
                {
                    "reason": f"ralph_{reason}",
                    "attempt_index": attempt_index,
                    "cumulative_cost_usd": cost,
                },
            )
        self.store.append_event(
            parent_run_id,
            "ralph_stopped",
            {
                "reason": reason,
                "attempt_index": attempt_index,
                "cumulative_cost_usd": cost,
            },
        )

    @staticmethod
    def _cleanup_tmp_files(paths: Sequence[Path]) -> None:
        for path in paths:
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                continue
