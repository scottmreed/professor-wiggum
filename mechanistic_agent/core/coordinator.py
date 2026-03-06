"""Explicit run coordinator for the local-first mechanistic runtime."""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from mechanistic_agent.model_registry import get_default_model

from . import model_context
from .arrow_push import predict_arrow_push_annotation
from .db import RunStore
from .mechanism_moves import normalize_electron_pushes, repair_candidate_reaction_smirks
from .reaction_type_templates import (
    compact_template_for_prompt,
    example_mapping_for_reaction_id,
    load_reaction_type_catalog_for_runtime,
    suggest_reaction_type_for_example,
)
from .subagents import (
    BalanceAgent,
    ConditionsAgent,
    FunctionalGroupsAgent,
    IntermediateAgent,
    MappingAgent,
    MechanismAgent,
    MissingReagentsAgent,
    ReactionTypeAgent,
    ReflectionAgent,
)
from .tool_executor import ToolExecutor
from .types import (
    BranchCandidate,
    BranchPoint,
    FailedPath,
    HarnessConfig,
    ModuleSpec,
    RunConfig,
    RunInput,
    RunMode,
    RunState,
    StepResult,
    TemplateGuidanceState,
    TopologyProfile,
    StepValidationCheck,
    StepValidationResult,
)
from .validators import ALL_VALIDATOR_IDS, validate_mechanism_step_output


class _RunPaused(Exception):
    """Raised when a run is intentionally paused awaiting user decision."""


class RunCoordinator:
    """Coordinates run execution through deterministic and LLM-backed subagents."""

    def __init__(self, store: RunStore) -> None:
        self.store = store
        executor = ToolExecutor()
        self.balance_agent = BalanceAgent(executor)
        self.conditions_agent = ConditionsAgent(executor)
        self.functional_groups_agent = FunctionalGroupsAgent(executor)
        self.missing_reagents_agent = MissingReagentsAgent(executor)
        self.mapping_agent = MappingAgent(executor)
        self.reaction_type_agent = ReactionTypeAgent(executor)
        self.intermediate_agent = IntermediateAgent(executor)
        self.mechanism_agent = MechanismAgent(executor)
        self.reflection_agent = ReflectionAgent()
        self._agent_registry: Dict[str, Any] = {
            "BalanceAgent": self.balance_agent,
            "ConditionsAgent": self.conditions_agent,
            "FunctionalGroupsAgent": self.functional_groups_agent,
            "MissingReagentsAgent": self.missing_reagents_agent,
            "MappingAgent": self.mapping_agent,
            "ReactionTypeAgent": self.reaction_type_agent,
            "IntermediateAgent": self.intermediate_agent,
            "MechanismAgent": self.mechanism_agent,
            "ReflectionAgent": self.reflection_agent,
        }

    def _resolve_harness(self, state: RunState) -> HarnessConfig:
        """Resolve harness config from run config, falling back to default."""
        from .registries import HarnessRegistry
        from pathlib import Path
        base_dir = Path(__file__).resolve().parents[2]
        registry = HarnessRegistry(base_dir / "harness_versions")
        return registry.resolve_from_run_config(state.run_config)

    def _enabled_validators(self, harness: HarnessConfig) -> set[str]:
        """Return the set of enabled validator module IDs from harness."""
        return {
            m.id for m in harness.post_step_modules
            if m.enabled and m.group_key == "validators"
        }

    @staticmethod
    def _coerce_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y", "on"}:
                return True
            if lowered in {"false", "0", "no", "n", "off"}:
                return False
        return default

    @staticmethod
    def _coerce_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_literal(value: Any, allowed: set[str], default: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized in allowed:
            return normalized
        return default

    def _build_state(self, run_row: Dict[str, Any]) -> RunState:
        payload = run_row.get("input_payload", {})
        config = run_row.get("config", {})

        run_input = RunInput(
            starting_materials=list(payload.get("starting_materials") or []),
            products=list(payload.get("products") or []),
            temperature_celsius=float(payload.get("temperature_celsius", 25.0)),
            ph=payload.get("ph"),
            example_id=str(payload.get("example_id") or "").strip() or None,
        )
        run_config = RunConfig(
            model=str(config.get("model") or get_default_model()),
            model_name=str(config.get("model_name") or config.get("model") or get_default_model()),
            model_family=str(config.get("model_family") or "openai"),
            step_models=dict(config.get("step_models") or {}),
            step_reasoning=dict(config.get("step_reasoning") or {}),
            thinking_level=config.get("thinking_level"),
            reasoning_level=config.get("reasoning_level"),
            optional_llm_tools=list(config.get("optional_llm_tools") or []),
            functional_groups_enabled=self._coerce_bool(
                config.get("functional_groups_enabled", False),
                False,
            ),
            intermediate_prediction_enabled=self._coerce_bool(
                config.get("intermediate_prediction_enabled", True),
                True,
            ),
            max_steps=self._coerce_int(config.get("max_steps", 10), 10),
            max_runtime_seconds=self._coerce_float(config.get("max_runtime_seconds", 600.0), 600.0),
            api_keys=dict(config.get("api_keys") or {}),
            retry_same_candidate_max=self._coerce_int(config.get("retry_same_candidate_max", 1), 1),
            max_reproposals_per_step=self._coerce_int(config.get("max_reproposals_per_step", 4), 4),
            reproposal_on_repeat_failure=self._coerce_bool(
                config.get("reproposal_on_repeat_failure", True),
                True,
            ),
            candidate_rescue_enabled=self._coerce_bool(config.get("candidate_rescue_enabled", True), True),
            step_mapping_enabled=self._coerce_bool(config.get("step_mapping_enabled", True), True),
            arrow_push_annotation_enabled=self._coerce_bool(
                config.get("arrow_push_annotation_enabled", True),
                True,
            ),
            dbe_policy=self._coerce_literal(config.get("dbe_policy"), {"strict", "soft"}, "soft"),  # type: ignore[arg-type]
            reaction_template_policy=self._coerce_literal(
                config.get("reaction_template_policy"),
                {"off", "auto"},
                "auto",
            ),  # type: ignore[arg-type]
            reaction_template_confidence_threshold=self._coerce_float(
                config.get("reaction_template_confidence_threshold", 0.65),
                0.65,
            ),
            reaction_template_margin_threshold=self._coerce_float(
                config.get("reaction_template_margin_threshold", 0.10),
                0.10,
            ),
            reaction_template_disable_step_window=self._coerce_int(
                config.get("reaction_template_disable_step_window", 3),
                3,
            ),
            reaction_template_disable_consecutive_mismatch=self._coerce_int(
                config.get("reaction_template_disable_consecutive_mismatch", 2),
                2,
            ),
            orchestration_mode=self._coerce_literal(
                config.get("orchestration_mode"),
                {"standard", "ralph"},
                "standard",
            ),  # type: ignore[arg-type]
            coordination_topology=self._coerce_literal(
                config.get("coordination_topology"),
                {"sas", "centralized_mas", "independent_mas", "decentralized_mas"},
                "centralized_mas",
            ),  # type: ignore[arg-type]
            harness_name=str(config.get("harness_name") or "default"),
            harness_config_path=(
                str(config.get("harness_config_path"))
                if config.get("harness_config_path")
                else None
            ),
            harness_strategy=self._coerce_literal(
                config.get("harness_strategy"),
                {"latest", "portfolio", "mutate"},
                "latest",
            ),  # type: ignore[arg-type]
            harness_list=(
                [
                    str(item).strip()
                    for item in (config.get("harness_list") or [])
                    if str(item).strip()
                ]
                if isinstance(config.get("harness_list"), list)
                else []
            ),
            max_iterations=max(0, self._coerce_int(config.get("max_iterations", 0), 0)),
            completion_promise="target_products_reached && flow_node:run_complete",
            ralph_max_runtime_seconds=self._coerce_float(
                config.get("ralph_max_runtime_seconds", 6000.0),
                6000.0,
            ),
            max_cost_usd=(
                self._coerce_float(config.get("max_cost_usd"), 0.0)
                if config.get("max_cost_usd") is not None
                else None
            ),
            repeat_failure_signature_limit=max(
                1,
                self._coerce_int(config.get("repeat_failure_signature_limit", 2), 2),
            ),
            babysit_mode=self._coerce_literal(
                config.get("babysit_mode"),
                {"off", "advisory"},
                "off",
            ),  # type: ignore[arg-type]
            allow_validator_mutation=self._coerce_bool(
                config.get("allow_validator_mutation", False),
                False,
            ),
            ralph_parent_run_id=(
                str(config.get("ralph_parent_run_id"))
                if config.get("ralph_parent_run_id")
                else None
            ),
        )
        mode: RunMode = str(run_row.get("mode") or "unverified")  # type: ignore[assignment]

        state = RunState(
            run_id=run_row["id"],
            mode=mode,
            run_input=run_input,
            run_config=run_config,
        )
        state.initialise()
        self._hydrate_state_from_outputs(state)
        return state

    def _hydrate_state_from_outputs(self, state: RunState) -> None:
        outputs = self.store.list_step_outputs(state.run_id)
        mechanism_rows = [
            row
            for row in outputs
            if row.get("step_name") == "mechanism_synthesis"
            and isinstance(row.get("validation"), dict)
            and bool(row["validation"].get("passed"))
        ]
        mechanism_rows.sort(
            key=lambda row: (int(row.get("attempt") or 0), int(row.get("retry_index") or 0))
        )
        if not mechanism_rows:
            return

        latest = mechanism_rows[-1]
        latest_output = latest.get("output") or {}
        resulting_state = latest_output.get("resulting_state")
        if isinstance(resulting_state, list) and resulting_state:
            state.current_state = [str(item) for item in resulting_state]

        state.step_index = max(int(latest.get("attempt") or 0), state.step_index)

        intermediates: List[str] = []
        for row in mechanism_rows:
            output = row.get("output") or {}
            intermediate = output.get("predicted_intermediate")
            if isinstance(intermediate, str) and intermediate and intermediate not in intermediates:
                intermediates.append(intermediate)
        state.previous_intermediates = intermediates

        step_mapping_rows = [
            row
            for row in outputs
            if row.get("step_name") == "step_atom_mapping"
        ]
        if step_mapping_rows:
            step_mapping_rows.sort(
                key=lambda row: (int(row.get("attempt") or 0), int(row.get("retry_index") or 0))
            )
            latest_mapping = step_mapping_rows[-1].get("output") or {}
            if isinstance(latest_mapping, dict):
                state.latest_step_mapping = {
                    "step_index": int(step_mapping_rows[-1].get("attempt") or 0),
                    "mapped_atoms": list(latest_mapping.get("compact_mapped_atoms") or [])[:12],
                    "unmapped_atoms": list(latest_mapping.get("unmapped_atoms") or [])[:12],
                    "confidence": latest_mapping.get("confidence"),
                }

        reaction_type_rows = [
            row
            for row in outputs
            if row.get("step_name") == "reaction_type_mapping"
        ]
        if reaction_type_rows:
            reaction_type_rows.sort(
                key=lambda row: (int(row.get("attempt") or 0), int(row.get("retry_index") or 0))
            )
            latest_reaction_type = reaction_type_rows[-1].get("output") or {}
            if isinstance(latest_reaction_type, dict):
                self._apply_reaction_type_selection(
                    state,
                    latest_reaction_type,
                    emit_event=False,
                )

        # Reconstruct branch points and failed paths from stored events.
        events = self.store.list_events(state.run_id) if hasattr(self.store, "list_events") else []
        for ev in events:
            ev_type = ev.get("event_type") or ""
            payload = ev.get("payload") or {}
            if ev_type == "branch_point_created":
                bp = BranchPoint(
                    step_index=int(payload.get("step_index") or 0),
                    current_state=list(payload.get("current_state") or []),
                    previous_intermediates=list(payload.get("previous_intermediates") or []),
                    template_guidance_snapshot=(
                        dict(payload.get("template_guidance_snapshot") or {})
                        if isinstance(payload.get("template_guidance_snapshot"), dict)
                        else None
                    ),
                )
                # Restore alternative count (alternatives themselves are lost on
                # serialisation but the branch point existence is preserved for
                # backtracking decisions).
                state.branch_points.append(bp)
            elif ev_type == "failed_path_recorded":
                fp = FailedPath(
                    branch_step_index=int(payload.get("branch_step_index") or 0),
                    candidate_rank=int(payload.get("candidate_rank") or 0),
                    steps_taken=list(payload.get("steps_taken") or []),
                    failure_reason=str(payload.get("failure_reason") or ""),
                )
                state.failed_paths.append(fp)
            elif ev_type == "template_guidance_state_updated":
                if isinstance(payload, dict):
                    state.template_guidance_state = TemplateGuidanceState.from_dict(payload)

    def _step_model(self, state: RunState, step_name: str) -> Optional[str]:
        return state.run_config.step_models.get(step_name, state.run_config.model)

    def _step_reasoning(self, state: RunState, step_name: str) -> Optional[str]:
        return state.run_config.step_reasoning.get(step_name)

    def _record_step(self, state: RunState, result: StepResult) -> None:
        validation_payload = result.validation.as_dict() if result.validation else None
        resolved_model = result.model or self._step_model(state, result.step_name)
        resolved_reasoning = result.reasoning_level or self._step_reasoning(state, result.step_name)
        self.store.record_step_output(
            run_id=state.run_id,
            step_name=result.step_name,
            attempt=result.attempt,
            retry_index=result.retry_index,
            source=result.source,
            model=resolved_model,
            reasoning_level=resolved_reasoning,
            tool_name=result.tool_name,
            output=result.output,
            validation=validation_payload,
            accepted_bool=None,
            usage=result.token_usage,
            cost=result.cost,
        )
        self.store.append_event(
            state.run_id,
            "step_output",
            {
                "step_name": result.step_name,
                "tool_name": result.tool_name,
                "attempt": result.attempt,
                "retry_index": result.retry_index,
                "source": result.source,
                "output": result.output,
                "validation": validation_payload,
            },
            step_name=result.step_name,
        )
        trace_score: Optional[float] = None
        if isinstance(validation_payload, dict):
            trace_score = 1.0 if bool(validation_payload.get("passed")) else 0.0
        prompt_version_id = self.store.resolve_run_step_prompt_id(
            run_id=state.run_id,
            step_name=result.step_name,
            attempt=result.attempt,
        )
        model_version_id: Optional[str] = None
        if result.source == "llm" and resolved_model:
            model_version_id = self.store.upsert_model_version(
                model_name=resolved_model,
                reasoning_level=resolved_reasoning,
            )
        # Calculate step duration
        step_key = f"{result.step_name}_{result.attempt}_{result.retry_index}"
        start_time = state.step_start_times.get(step_key)
        captured_at = time.time()
        duration_seconds = captured_at - start_time if start_time else None

        # Format duration in human readable format
        duration_human = None
        if duration_seconds is not None:
            if duration_seconds < 60:
                duration_human = ".1fs"
            elif duration_seconds < 3600:
                minutes = int(duration_seconds // 60)
                seconds = duration_seconds % 60
                duration_human = f"{minutes}m {seconds:.1f}s"
            else:
                hours = int(duration_seconds // 3600)
                minutes = int((duration_seconds % 3600) // 60)
                duration_human = f"{hours}h {minutes}m"

        self.store.add_trace_record(
            run_id=state.run_id,
            step_name=result.step_name,
            model=resolved_model,
            reasoning_level=resolved_reasoning,
            prompt_version_id=prompt_version_id,
            model_version_id=model_version_id,
            score=trace_score,
            source="run",
            trace={
                "tool_name": result.tool_name,
                "attempt": result.attempt,
                "retry_index": result.retry_index,
                "output": result.output,
                "validation": validation_payload,
                "captured_at": captured_at,
                "duration_seconds": duration_seconds,
                "duration_human": duration_human,
                "token_usage": result.token_usage,
                "cost": result.cost,
            },
        )

        validation = validation_payload
        if isinstance(validation, dict) and validation.get("passed") is False:
            self.store.append_event(
                state.run_id,
                "step_failed",
                {
                    "step_name": result.step_name,
                    "attempt": result.attempt,
                    "retry_index": result.retry_index,
                    "validation": validation,
                },
                step_name=result.step_name,
            )
        else:
            self.store.append_event(
                state.run_id,
                "step_completed",
                {
                    "step_name": result.step_name,
                    "attempt": result.attempt,
                    "retry_index": result.retry_index,
                },
                step_name=result.step_name,
            )

    def _mark_step_started(
        self,
        state: RunState,
        *,
        step_name: str,
        tool_name: str,
        attempt: int = 1,
        retry_index: int = 0,
    ) -> None:
        start_time = time.time()
        state.step_start_times[f"{step_name}_{attempt}_{retry_index}"] = start_time
        self.store.append_event(
            state.run_id,
            "step_started",
            {
                "step_name": step_name,
                "tool_name": tool_name,
                "attempt": attempt,
                "retry_index": retry_index,
                "start_time": start_time,
            },
            step_name=step_name,
        )

    def _existing_steps(self, run_id: str) -> set[str]:
        outputs = self.store.list_step_outputs(run_id)
        return {str(row.get("step_name") or "") for row in outputs}

    def _latest_output_by_step(self, run_id: str, step_name: str) -> Optional[Dict[str, Any]]:
        rows = [
            row
            for row in self.store.list_step_outputs(run_id)
            if row.get("step_name") == step_name
        ]
        if not rows:
            return None
        rows.sort(key=lambda row: (int(row.get("attempt") or 0), int(row.get("retry_index") or 0)))
        output = rows[-1].get("output")
        return dict(output) if isinstance(output, dict) else None

    def _initial_context_by_step(self, state: RunState) -> Dict[str, Optional[Dict[str, Any]]]:
        return {
            "balance_analysis": self._latest_output_by_step(state.run_id, "balance_analysis"),
            "functional_groups": self._latest_output_by_step(state.run_id, "functional_groups"),
            "ph_recommendation": self._latest_output_by_step(state.run_id, "ph_recommendation"),
            "initial_conditions": self._latest_output_by_step(state.run_id, "initial_conditions"),
            "missing_reagents": self._latest_output_by_step(state.run_id, "missing_reagents"),
            "atom_mapping": self._latest_output_by_step(state.run_id, "atom_mapping"),
        }

    def _emit_template_guidance_state(self, state: RunState) -> None:
        if state.template_guidance_state is None:
            return
        self.store.append_event(
            state.run_id,
            "template_guidance_state_updated",
            state.template_guidance_state.as_dict(),
        )

    @staticmethod
    def _selection_confidence_gap(output: Dict[str, Any]) -> Optional[float]:
        selected = output.get("selected_label_exact")
        selected_label = str(selected or "").strip()
        selected_conf = float(output.get("confidence") or 0.0)
        top_candidates = output.get("top_candidates")
        if not isinstance(top_candidates, list) or not top_candidates:
            return None

        selected_in_top = False
        best_alt: Optional[float] = None
        for row in top_candidates:
            if not isinstance(row, dict):
                continue
            label = str(row.get("label_exact") or "").strip()
            confidence = row.get("confidence")
            if not isinstance(confidence, (int, float)):
                continue
            score = float(confidence)
            if selected_label and label == selected_label:
                selected_in_top = True
                continue
            if best_alt is None or score > best_alt:
                best_alt = score

        # When selected label is absent from top_candidates, treat selected_conf as top-1.
        if best_alt is None:
            return None
        if not selected_in_top and selected_label and selected_label != "no_match":
            return max(0.0, selected_conf - best_alt)
        return max(0.0, selected_conf - best_alt)

    @staticmethod
    def _guidance_mode_for_selection(
        *,
        confidence: float,
        confidence_threshold: float,
        confidence_gap: Optional[float],
        margin_threshold: float,
    ) -> Tuple[str, Optional[str]]:
        if confidence < confidence_threshold:
            return "disabled", f"selection_confidence_below_threshold_{confidence_threshold:.2f}"
        if confidence_gap is not None and confidence_gap < margin_threshold:
            return "weak", f"selection_confidence_gap_below_margin_{margin_threshold:.2f}"
        return "active", None

    def _apply_reaction_type_selection(
        self,
        state: RunState,
        output: Dict[str, Any],
        *,
        emit_event: bool,
    ) -> None:
        state.reaction_type_selection = {
            "selected_label_exact": output.get("selected_label_exact"),
            "selected_type_id": output.get("selected_type_id"),
            "confidence": output.get("confidence"),
            "rationale": output.get("rationale"),
            "top_candidates": list(output.get("top_candidates") or []),
        }
        selected_template = output.get("selected_template")
        state.selected_reaction_template = (
            dict(selected_template) if isinstance(selected_template, dict) else None
        )
        selected_label = str(output.get("selected_label_exact") or "").strip()
        selected_type_id = str(output.get("selected_type_id") or "").strip() or None
        confidence = float(output.get("confidence") or 0.0)
        _thresh_cfg = state.run_config.reaction_template_confidence_threshold
        confidence_threshold = float(_thresh_cfg if _thresh_cfg is not None else 0.65)
        _margin_cfg = state.run_config.reaction_template_margin_threshold
        margin_threshold = float(_margin_cfg if _margin_cfg is not None else 0.10)
        confidence_gap = self._selection_confidence_gap(output)

        if selected_label and selected_label != "no_match" and state.selected_reaction_template:
            guidance_mode, disable_reason = self._guidance_mode_for_selection(
                confidence=confidence,
                confidence_threshold=confidence_threshold,
                confidence_gap=confidence_gap,
                margin_threshold=margin_threshold,
            )
            decision_reason = disable_reason or "selection_passed_thresholds"
            state.template_guidance_state = TemplateGuidanceState(
                mode=guidance_mode,  # type: ignore[arg-type]
                selected_type_id=selected_type_id,
                selected_label_exact=selected_label,
                selection_confidence=confidence,
                selection_confidence_gap=confidence_gap,
                selection_confidence_threshold=confidence_threshold,
                selection_margin_threshold=margin_threshold,
                suitable_step_count=int(
                    state.selected_reaction_template.get("suitable_step_count") or 0
                ),
                current_template_step_index=1,
                completed_steps_count=0,
                alignment_history=[],
                disable_reason=disable_reason,
                selection_decision_reason=decision_reason,
            )
        elif selected_label == "no_match":
            state.template_guidance_state = TemplateGuidanceState(
                mode="no_match",
                selection_confidence=confidence,
                selection_confidence_gap=confidence_gap,
                selection_confidence_threshold=confidence_threshold,
                selection_margin_threshold=margin_threshold,
                selection_decision_reason="no_template_match",
            )
            state.selected_reaction_template = None
        else:
            state.template_guidance_state = TemplateGuidanceState(
                mode="disabled",
                selection_confidence=confidence,
                selection_confidence_gap=confidence_gap,
                selection_confidence_threshold=confidence_threshold,
                selection_margin_threshold=margin_threshold,
                disable_reason="invalid_template_selection",
                selection_decision_reason="invalid_template_selection",
            )
            state.selected_reaction_template = None

        if emit_event:
            self._emit_template_guidance_state(state)

    def _build_template_guidance_payload(self, state: RunState) -> Optional[Dict[str, Any]]:
        guidance_state = state.template_guidance_state
        if guidance_state is None or guidance_state.mode not in {"active", "weak"}:
            return None
        template = state.selected_reaction_template
        if not isinstance(template, dict):
            return None
        guidance_strength = "strong" if guidance_state.mode == "active" else "weak"
        confidence_gap = None
        if isinstance(state.reaction_type_selection, dict):
            confidence_gap = self._selection_confidence_gap(state.reaction_type_selection)

        history = list(guidance_state.alignment_history or [])
        recent = history[-3:]
        alignment_summary = "no prior alignment signal"
        if recent:
            alignment_summary = "; ".join(
                f"step {int(item.get('step_index') or 0)}: {item.get('alignment')}"
                for item in recent
            )

        steps = []
        for item in list(template.get("generic_mechanism_steps") or [])[:8]:
            if not isinstance(item, dict):
                continue
            steps.append(
                {
                    "step_index": int(item.get("step_index") or 0),
                    "reaction_generic": str(item.get("reaction_generic") or ""),
                    "note": str(item.get("note") or ""),
                }
            )

        return {
            "selected_type_id": guidance_state.selected_type_id,
            "selected_label_exact": guidance_state.selected_label_exact,
            "selection_confidence": guidance_state.selection_confidence,
            "confidence_gap": confidence_gap,
            "guidance_strength": guidance_strength,
            "suitable_step_count": guidance_state.suitable_step_count,
            "current_template_step_index": guidance_state.current_template_step_index,
            "completed_steps_count": guidance_state.completed_steps_count,
            "alignment_summary": alignment_summary,
            "recent_alignment": recent,
            "template_steps": steps,
            "advisory_only": True,
        }

    @staticmethod
    def _build_example_mapping_output(
        *,
        mapping: Dict[str, Any],
        catalog: Dict[str, Any],
        reason: str,
        example_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        selected_type_id = str(mapping.get("mechanism_type_id") or mapping.get("selected_type_id") or "").strip()
        selected_label = str(mapping.get("mechanism_type_label") or mapping.get("selected_label_exact") or "").strip()
        by_id = dict(catalog.get("by_id") or {})
        by_label = dict(catalog.get("by_label") or {})

        template = None
        if selected_type_id and selected_type_id in by_id:
            template = by_id[selected_type_id]
        elif selected_label and selected_label in by_label:
            template = by_label[selected_label]
            selected_type_id = str(template.get("type_id") or "")
            selected_label = str(template.get("label_exact") or selected_label)
        if not isinstance(template, dict):
            return None

        confidence_raw = mapping.get("confidence")
        confidence = 0.99
        if isinstance(confidence_raw, (int, float)):
            confidence = max(0.0, min(1.0, float(confidence_raw)))

        rationale_text = str(mapping.get("rationale") or "").strip()
        if not rationale_text:
            rationale_text = reason

        return {
            "status": "success",
            "selected_label_exact": selected_label,
            "selected_type_id": selected_type_id,
            "confidence": confidence,
            "rationale": rationale_text,
            "top_candidates": [
                {
                    "label_exact": selected_label,
                    "type_id": selected_type_id,
                    "confidence": confidence,
                }
            ],
            "selected_template": compact_template_for_prompt(template),
            "available_reaction_type_count": len(list(catalog.get("templates") or [])),
            "model_used": "deterministic_example_mapping",
            "tool_calling_used": False,
            "example_id": example_id,
        }

    def _run_initial_phase(self, state: RunState, harness: Optional[HarnessConfig] = None) -> None:
        """Run pre-loop analysis modules. Driven by harness config when provided."""
        existing = self._existing_steps(state.run_id)
        context: Dict[str, Optional[Dict[str, Any]]] = {}

        if harness is None:
            # Backward-compatible path: run the legacy hardcoded sequence.
            self._run_initial_phase_legacy(state, existing)
            return

        # Iterate over enabled pre-loop modules from harness config.
        processed_groups: set[str] = set()
        for module in harness.enabled_pre_loop():
            # Skip already-completed steps (for resume).
            if module.step_name in existing and module.group_key not in ("conditions_pair",):
                # Load existing output into context for downstream modules.
                context[module.id] = self._latest_output_by_step_name(state.run_id, module.step_name)
                continue

            # Handle grouped modules (conditions_pair dispatches once).
            if module.group_key and module.group_key in processed_groups:
                continue

            if module.group_key == "conditions_pair":
                # ConditionsAgent returns both ph_recommendation and initial_conditions.
                group_modules = [
                    m for m in harness.enabled_pre_loop() if m.group_key == "conditions_pair"
                ]
                group_step_names = {m.step_name for m in group_modules}
                if not group_step_names.issubset(existing):
                    for gm in group_modules:
                        self._mark_step_started(state, step_name=gm.step_name, tool_name=gm.tool_name)
                    conditions_results = self.conditions_agent.run(state)
                    for conditions_result in conditions_results:
                        if conditions_result.step_name not in existing:
                            self._record_step(state, conditions_result)
                            context[conditions_result.step_name] = conditions_result.output
                # Load from DB for downstream.
                for gm in group_modules:
                    if gm.id not in context:
                        context[gm.id] = self._latest_output_by_step_name(state.run_id, gm.step_name)
                processed_groups.add("conditions_pair")
                continue

            result = self._dispatch_pre_loop_module(state, module, context, existing)
            if result is not None:
                self._handle_module_side_effects(state, module, result)

        # If reaction_type_mapping is disabled, set template guidance to disabled.
        rtm_enabled = any(
            m.id == "reaction_type_mapping" and m.enabled
            for m in harness.pre_loop_modules
        )
        if not rtm_enabled:
            state.template_guidance_state = TemplateGuidanceState(
                mode="disabled",
                disable_reason="reaction_type_mapping_disabled_in_harness",
            )
            self._emit_template_guidance_state(state)

    def _dispatch_pre_loop_module(
        self,
        state: RunState,
        module: ModuleSpec,
        context: Dict[str, Optional[Dict[str, Any]]],
        existing: set[str],
    ) -> Optional[StepResult]:
        """Dispatch a single pre-loop module to its agent."""
        if module.step_name in existing:
            context[module.id] = self._latest_output_by_step_name(state.run_id, module.step_name)
            return None

        self._mark_step_started(state, step_name=module.step_name, tool_name=module.tool_name)

        if module.id == "balance_analysis":
            result = self.balance_agent.run(state)
        elif module.id == "functional_groups":
            result = self.functional_groups_agent.run(state)
        elif module.id == "missing_reagents":
            latest_initial = context.get("initial_conditions")
            result = self.missing_reagents_agent.run(state, latest_initial)
        elif module.id == "atom_mapping":
            result = self.mapping_agent.run(state)
        elif module.id == "reaction_type_mapping":
            result = self._run_reaction_type_module(state, context)
        elif module.id == "inject_canonical_byproducts":
            result = self._run_inject_canonical_byproducts(state, context)
        elif module.custom:
            result = self._run_custom_module(state, module, context)
        else:
            agent = self._agent_registry.get(module.agent_class)
            if agent and hasattr(agent, "run"):
                result = agent.run(state)
            else:
                return None

        self._record_step(state, result)
        context[module.id] = result.output
        return result

    def _run_reaction_type_module(
        self,
        state: RunState,
        context: Dict[str, Optional[Dict[str, Any]]],
    ) -> StepResult:
        """Run reaction type mapping with catalog lookup and fallback logic."""
        catalog = load_reaction_type_catalog_for_runtime()
        # Build context from whatever prior steps have run.
        full_context = self._initial_context_by_step(state)
        # Merge in anything from harness context not already in full_context.
        for key, val in context.items():
            if key not in full_context and val is not None:
                full_context[key] = val

        mapped_output: Optional[Dict[str, Any]] = None
        if state.run_input.example_id:
            mapping = example_mapping_for_reaction_id(catalog, state.run_input.example_id)
            if isinstance(mapping, dict):
                mapped_output = self._build_example_mapping_output(
                    mapping=mapping,
                    catalog=catalog,
                    reason="Deterministic mapping from reaction_type_templates example_mappings.",
                    example_id=state.run_input.example_id,
                )

        if mapped_output is not None:
            reaction_type_result = StepResult(
                step_name="reaction_type_mapping",
                tool_name="select_reaction_type",
                output=mapped_output,
                source="deterministic",
            )
        else:
            reaction_type_result = self.reaction_type_agent.run(
                state,
                balance_analysis=full_context.get("balance_analysis"),
                functional_groups=full_context.get("functional_groups"),
                ph_recommendation=full_context.get("ph_recommendation"),
                initial_conditions=full_context.get("initial_conditions"),
                missing_reagents=full_context.get("missing_reagents"),
                atom_mapping=full_context.get("atom_mapping"),
            )
            output = reaction_type_result.output or {}
            selected_label = str(output.get("selected_label_exact") or "").strip().lower()
            if state.run_input.example_id and selected_label == "no_match":
                fallback_mapping = suggest_reaction_type_for_example(
                    catalog,
                    starting_materials=list(state.run_input.starting_materials),
                    products=list(state.run_input.products),
                )
                if isinstance(fallback_mapping, dict):
                    mapped_output = self._build_example_mapping_output(
                        mapping=fallback_mapping,
                        catalog=catalog,
                        reason="Example fallback heuristic applied after no_match selection.",
                        example_id=state.run_input.example_id,
                    )
                    if isinstance(mapped_output, dict):
                        reaction_type_result.output = mapped_output

        output = reaction_type_result.output or {}
        self._apply_reaction_type_selection(state, output, emit_event=True)
        return reaction_type_result

    def _run_inject_canonical_byproducts(
        self,
        state: RunState,
        context: Dict[str, Optional[Dict[str, Any]]],
    ) -> StepResult:
        """Inject canonical byproducts from the selected reaction template.

        Reads the canonical_byproducts list from the template that reaction_type_mapping
        selected, and returns them as a missing_reagents-compatible StepResult.  The
        step_name is intentionally set to 'missing_reagents' so that downstream consumers
        (mechanism proposer prompt context, atom-balance validators) find the data via the
        standard 'missing_reagents' step output lookup.
        """
        rtm_output = context.get("reaction_type_mapping") or {}
        type_id = str(rtm_output.get("selected_type_id") or "").strip()

        byproducts: List[str] = []
        source_note = "no_template_selected"
        if type_id:
            try:
                catalog = load_reaction_type_catalog_for_runtime()
                template = (catalog.get("by_id") or {}).get(type_id)
                if isinstance(template, dict):
                    raw = template.get("canonical_byproducts")
                    if isinstance(raw, list):
                        byproducts = [str(s) for s in raw if isinstance(s, str) and str(s).strip()]
                    source_note = f"template:{type_id}"
            except Exception:
                source_note = "catalog_load_error"

        output = {
            "missing_reactants": [],
            "missing_products": byproducts,
            "status": "ok",
            "source": "inject_canonical_byproducts",
            "notes": source_note,
        }
        return StepResult(
            step_name="missing_reagents",
            tool_name="inject_canonical_byproducts",
            output=output,
            source="deterministic",
        )

    def _run_custom_module(
        self,
        state: RunState,
        module: ModuleSpec,
        context: Dict[str, Optional[Dict[str, Any]]],
    ) -> StepResult:
        """Execute a custom user-defined module (LLM or deterministic)."""
        input_data: Dict[str, Any] = {}
        for dep_id in module.inputs:
            if dep_id in context and context[dep_id] is not None:
                input_data[dep_id] = context[dep_id]

        if module.kind == "deterministic" and module.code_text:
            namespace: Dict[str, Any] = {"inputs": input_data, "output": {}}
            exec(module.code_text, {"__builtins__": __builtins__}, namespace)  # noqa: S102
            return StepResult(
                step_name=module.step_name or module.id,
                tool_name=module.id,
                output=namespace.get("output") or {},
                source="deterministic",
            )

        # Custom LLM modules are a future extension; placeholder.
        return StepResult(
            step_name=module.step_name or module.id,
            tool_name=module.id,
            output={"status": "custom_module_placeholder", "inputs": list(input_data.keys())},
            source="llm" if module.kind == "llm" else "deterministic",
        )

    def _handle_module_side_effects(
        self,
        state: RunState,
        module: ModuleSpec,
        result: StepResult,
    ) -> None:
        """Handle module-specific post-dispatch side effects."""
        if module.id == "missing_reagents":
            missing_output = result.output or {}
            if bool(missing_output.get("should_abort_mechanism")):
                pause_payload = {
                    "reason": "missing_reagents_unbalanced_abort",
                    "attempt": 0,
                    "has_alternative": False,
                    "details": {"message": missing_output.get("message")},
                }
                pause_id = self.store.create_run_pause(
                    run_id=state.run_id,
                    reason="missing_reagents_unbalanced_abort",
                    details=pause_payload,
                )
                state.paused = True
                self.store.set_run_status(state.run_id, "paused")
                self.store.append_event(
                    state.run_id,
                    "run_paused",
                    {**pause_payload, "pause_id": pause_id},
                    step_name="missing_reagents",
                )
                raise _RunPaused()
            if str(missing_output.get("status") or "").lower() == "failed":
                self.store.append_event(
                    state.run_id,
                    "missing_reagents_warning",
                    {
                        "attempt": 0,
                        "status": missing_output.get("status"),
                        "error": missing_output.get("error"),
                        "message": missing_output.get("message"),
                        "abort_severity": missing_output.get("abort_severity"),
                        "balance_issues": missing_output.get("balance_issues") or {},
                    },
                    step_name="missing_reagents",
                )

    def _latest_output_by_step_name(
        self, run_id: str, step_name: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve the latest output for a given step_name from the store."""
        for item in self.store.list_step_outputs(run_id):
            if item.get("step_name") == step_name:
                return item.get("output")
        return None

    def _run_initial_phase_legacy(self, state: RunState, existing: set[str]) -> None:
        """Legacy hardcoded initial phase for backward compatibility."""
        if "balance_analysis" not in existing:
            self._mark_step_started(state, step_name="balance_analysis", tool_name="analyse_balance")
            result = self.balance_agent.run(state)
            self._record_step(state, result)

        if state.run_config.functional_groups_enabled and "functional_groups" not in existing:
            self._mark_step_started(
                state,
                step_name="functional_groups",
                tool_name="fingerprint_functional_groups",
            )
            fg_result = self.functional_groups_agent.run(state)
            self._record_step(state, fg_result)

        if "ph_recommendation" not in existing or "initial_conditions" not in existing:
            self._mark_step_started(state, step_name="ph_recommendation", tool_name="recommend_ph")
            self._mark_step_started(state, step_name="initial_conditions", tool_name="assess_initial_conditions")
            conditions_results = self.conditions_agent.run(state)
            for conditions_result in conditions_results:
                if conditions_result.step_name in existing:
                    continue
                self._record_step(state, conditions_result)

        latest_initial = None
        existing_outputs = self.store.list_step_outputs(state.run_id)
        for item in existing_outputs:
            if item.get("step_name") == "initial_conditions":
                latest_initial = item.get("output")

        optional_tools = set(state.run_config.optional_llm_tools)
        if "predict_missing_reagents" in optional_tools and "missing_reagents" not in existing:
            self._mark_step_started(
                state,
                step_name="missing_reagents",
                tool_name="predict_missing_reagents",
            )
            missing_result = self.missing_reagents_agent.run(state, latest_initial)
            self._record_step(state, missing_result)
            missing_output = missing_result.output or {}
            if bool(missing_output.get("should_abort_mechanism")):
                pause_payload = {
                    "reason": "missing_reagents_unbalanced_abort",
                    "attempt": 0,
                    "has_alternative": False,
                    "details": {"message": missing_output.get("message")},
                }
                pause_id = self.store.create_run_pause(
                    run_id=state.run_id,
                    reason="missing_reagents_unbalanced_abort",
                    details=pause_payload,
                )
                state.paused = True
                self.store.set_run_status(state.run_id, "paused")
                self.store.append_event(
                    state.run_id,
                    "run_paused",
                    {**pause_payload, "pause_id": pause_id},
                    step_name="missing_reagents",
                )
                raise _RunPaused()
            if str(missing_output.get("status") or "").lower() == "failed":
                self.store.append_event(
                    state.run_id,
                    "missing_reagents_warning",
                    {
                        "attempt": 0,
                        "status": missing_output.get("status"),
                        "error": missing_output.get("error"),
                        "message": missing_output.get("message"),
                        "abort_severity": missing_output.get("abort_severity"),
                        "balance_issues": missing_output.get("balance_issues") or {},
                    },
                    step_name="missing_reagents",
                )

        if "attempt_atom_mapping" in optional_tools and "atom_mapping" not in existing:
            self._mark_step_started(
                state,
                step_name="atom_mapping",
                tool_name="attempt_atom_mapping",
            )
            mapping_result = self.mapping_agent.run(state)
            self._record_step(state, mapping_result)

        existing = self._existing_steps(state.run_id)
        if (
            state.run_config.reaction_template_policy != "off"
            and "reaction_type_mapping" not in existing
        ):
            catalog = load_reaction_type_catalog_for_runtime()
            context = self._initial_context_by_step(state)
            self._mark_step_started(
                state,
                step_name="reaction_type_mapping",
                tool_name="select_reaction_type",
            )
            mapped_output: Optional[Dict[str, Any]] = None
            if state.run_input.example_id:
                mapping = example_mapping_for_reaction_id(catalog, state.run_input.example_id)
                if isinstance(mapping, dict):
                    mapped_output = self._build_example_mapping_output(
                        mapping=mapping,
                        catalog=catalog,
                        reason="Deterministic mapping from reaction_type_templates example_mappings.",
                        example_id=state.run_input.example_id,
                    )

            if mapped_output is not None:
                reaction_type_result = StepResult(
                    step_name="reaction_type_mapping",
                    tool_name="select_reaction_type",
                    output=mapped_output,
                    source="deterministic",
                )
            else:
                reaction_type_result = self.reaction_type_agent.run(
                    state,
                    balance_analysis=context.get("balance_analysis"),
                    functional_groups=context.get("functional_groups"),
                    ph_recommendation=context.get("ph_recommendation"),
                    initial_conditions=context.get("initial_conditions"),
                    missing_reagents=context.get("missing_reagents"),
                    atom_mapping=context.get("atom_mapping"),
                )
                output = reaction_type_result.output or {}
                selected_label = str(output.get("selected_label_exact") or "").strip().lower()
                if state.run_input.example_id and selected_label == "no_match":
                    fallback_mapping = suggest_reaction_type_for_example(
                        catalog,
                        starting_materials=list(state.run_input.starting_materials),
                        products=list(state.run_input.products),
                    )
                    if isinstance(fallback_mapping, dict):
                        mapped_output = self._build_example_mapping_output(
                            mapping=fallback_mapping,
                            catalog=catalog,
                            reason="Example fallback heuristic applied after no_match selection.",
                            example_id=state.run_input.example_id,
                        )
                        if isinstance(mapped_output, dict):
                            reaction_type_result.output = mapped_output

            self._record_step(state, reaction_type_result)
            output = reaction_type_result.output or {}
            self._apply_reaction_type_selection(state, output, emit_event=True)
        elif state.run_config.reaction_template_policy == "off":
            state.template_guidance_state = TemplateGuidanceState(
                mode="disabled",
                disable_reason="reaction_template_policy_off",
            )
            self._emit_template_guidance_state(state)

    @staticmethod
    def _retry_feedback_for_validation(validation_payload: Dict[str, Any]) -> Dict[str, Any]:
        failed_checks: List[str] = []
        guidance_parts: List[str] = []
        checks = validation_payload.get("checks")
        if isinstance(checks, list):
            for check in checks:
                if not isinstance(check, dict):
                    continue
                if check.get("passed") is True:
                    continue
                name = str(check.get("name") or "unknown")
                failed_checks.append(name)
                details = check.get("details")
                if isinstance(details, dict):
                    error_text = details.get("error") or details.get("message")
                    if isinstance(error_text, str) and error_text.strip():
                        # Provide specific guidance for SMILES validation errors
                        if "Invalid SMILES" in error_text:
                            guidance_parts.append(f"{name}: Invalid SMILES detected - ensure all chemical structures are valid and parseable by RDKit. Avoid excessive radicals, unclosed rings, and invalid atom symbols.")
                        elif "balance_check_failed" in error_text:
                            guidance_parts.append(f"{name}: Atom balance analysis failed - check that all SMILES strings are properly formatted and represent valid chemical structures.")
                        else:
                            guidance_parts.append(f"{name}: {error_text.strip()}")
        return {
            "failed_checks": failed_checks,
            "guidance": "; ".join(guidance_parts) if guidance_parts else "",
        }

    @staticmethod
    def _validation_signature(validation_payload: Dict[str, Any]) -> str:
        checks = validation_payload.get("checks")
        compact: List[Dict[str, Any]] = []
        if isinstance(checks, list):
            for check in checks:
                if not isinstance(check, dict):
                    continue
                if bool(check.get("passed")):
                    continue
                details = check.get("details") if isinstance(check.get("details"), dict) else {}
                compact.append(
                    {
                        "name": str(check.get("name") or ""),
                        "error": str(details.get("error") or ""),
                        "message": str(details.get("message") or ""),
                        "total_delta": details.get("total_delta"),
                        "balanced": details.get("balanced"),
                    }
                )
        payload = json.dumps(compact, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _attempt_candidate_rescue(
        self,
        state: RunState,
        *,
        mechanism_result: StepResult,
        failed_checks: List[str],
        candidate_rank: Any,
    ) -> Optional[StepResult]:
        if not state.run_config.candidate_rescue_enabled:
            return None
        if not any(name in {"atom_balance", "dbe_metadata"} for name in failed_checks):
            return None
        output = mechanism_result.output or {}
        current_state = [str(x) for x in output.get("current_state") or []]
        resulting_state = [str(x) for x in output.get("resulting_state") or []]
        if not current_state or not resulting_state:
            return None
        self.store.append_event(
            state.run_id,
            "candidate_rescue_started",
            {
                "attempt": state.step_index + 1,
                "candidate_rank": candidate_rank,
                "failed_checks": failed_checks,
            },
            step_name="candidate_rescue",
        )
        self._mark_step_started(
            state,
            step_name="candidate_rescue",
            tool_name="predict_missing_reagents_for_candidate",
            attempt=state.step_index + 1,
        )
        rescue_result = self.missing_reagents_agent.rescue_candidate(
            state,
            current_state=current_state,
            resulting_state=resulting_state,
            failed_checks=failed_checks,
            validation_details=mechanism_result.validation.as_dict() if mechanism_result.validation else {},
        )
        self._record_step(state, rescue_result)
        self.store.append_event(
            state.run_id,
            "candidate_rescue_completed",
            {
                "attempt": state.step_index + 1,
                "candidate_rank": candidate_rank,
                "status": (rescue_result.output or {}).get("status"),
                "add_reactants": (rescue_result.output or {}).get("add_reactants", []),
                "add_products": (rescue_result.output or {}).get("add_products", []),
            },
            step_name="candidate_rescue",
        )
        return rescue_result

    def _record_validation_checks(
        self,
        state: RunState,
        *,
        mechanism_result: Optional[StepResult] = None,
        validation_result: Optional[StepValidationResult] = None,
        attempt: Optional[int] = None,
        retry_index: Optional[int] = None,
    ) -> None:
        validation = validation_result
        recorded_attempt = attempt
        recorded_retry = retry_index

        if mechanism_result is not None:
            validation = mechanism_result.validation
            recorded_attempt = mechanism_result.attempt
            recorded_retry = mechanism_result.retry_index

        if not validation:
            return

        mapping = {
            "dbe_metadata": "bond_electron_validation",
            "atom_balance": "atom_balance_validation",
            "state_progress": "state_progress_validation",
        }
        for check in validation.checks:
            step_name = mapping.get(check.name)
            if not step_name:
                continue
            self._mark_step_started(
                state,
                step_name=step_name,
                tool_name=step_name,
                attempt=recorded_attempt,
                retry_index=recorded_retry,
            )
            result = StepResult(
                step_name=step_name,
                tool_name=step_name,
                output={"check": check.name, "passed": check.passed, "details": check.details},
                attempt=recorded_attempt,
                retry_index=recorded_retry,
                source="deterministic",
                validation=StepValidationResult(checks=[StepValidationCheck(name=check.name, passed=check.passed, details=check.details)]),
            )
            self._record_step(state, result)

    def _record_arrow_push_annotation(
        self,
        state: RunState,
        *,
        mechanism_output: Dict[str, Any],
        attempt: int,
        retry_index: int,
        candidate_rank: Optional[int],
        source: str,
    ) -> None:
        if not state.run_config.arrow_push_annotation_enabled:
            return
        if not isinstance(mechanism_output, dict):
            return

        current_state = [str(item) for item in mechanism_output.get("current_state") or []]
        resulting_state = [str(item) for item in mechanism_output.get("resulting_state") or []]
        if not current_state or not resulting_state:
            return

        try:
            prediction = predict_arrow_push_annotation(
                current_state=current_state,
                resulting_state=resulting_state,
                reaction_smirks=str(mechanism_output.get("reaction_smirks") or ""),
                raw_reaction_smirks=str(mechanism_output.get("raw_reaction_smirks") or ""),
                electron_pushes=mechanism_output.get("electron_pushes"),
                step_index=int(mechanism_output.get("step_index") or attempt),
                candidate_rank=candidate_rank,
            )
            self.store.record_arrow_push_annotation(
                run_id=state.run_id,
                step_index=int(prediction.get("step_index") or attempt),
                attempt=attempt,
                retry_index=retry_index,
                candidate_rank=candidate_rank,
                source=source,
                prediction=prediction,
            )
        except Exception:
            # Annotation is best-effort and must not affect mechanism execution.
            return

    def _pause_for_retry_exhaustion(
        self,
        state: RunState,
        *,
        attempt: int,
        last_validation: Dict[str, Any],
        failed_checks: Optional[List[str]] = None,
        validation_signature: Optional[str] = None,
        candidate_rank: Optional[int] = None,
        rescue_attempted: bool = False,
        rescue_outcome: str = "none",
    ) -> None:
        pause_payload = {
            "reason": "mechanism_retry_exhausted",
            "attempt": attempt,
            "max_retries": max(1, int(state.run_config.retry_same_candidate_max or 1)),
            "validation": last_validation,
            "has_alternative": False,
            "failed_checks": failed_checks or [],
            "validation_signature": validation_signature or "",
            "candidate_rank": candidate_rank,
            "rescue_attempted": rescue_attempted,
            "rescue_outcome": rescue_outcome,
        }
        pause_id = self.store.create_run_pause(
            run_id=state.run_id,
            reason="mechanism_retry_exhausted",
            details=pause_payload,
        )
        state.paused = True
        self.store.set_run_status(state.run_id, "paused")
        self.store.append_event(
            state.run_id,
            "mechanism_retry_exhausted",
            {
                **pause_payload,
                "pause_id": pause_id,
            },
            step_name="mechanism_synthesis",
        )
        self.store.append_event(
            state.run_id,
            "run_paused",
            {
                "pause_id": pause_id,
                "reason": "mechanism_retry_exhausted",
                "attempt": attempt,
                "has_alternative": False,
            },
            step_name="mechanism_synthesis",
        )
        raise _RunPaused()

    def _extract_candidates_from_proposal(
        self, proposal_output: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract ranked candidate dicts from a proposal output.

        Handles both the new ``candidates[]`` schema and the legacy
        ``intermediates[]`` / ``proposed_intermediates[]`` formats.
        """
        # New multi-candidate schema
        candidates = proposal_output.get("candidates")
        if isinstance(candidates, list) and candidates:
            return sorted(candidates, key=lambda c: int(c.get("rank") or 99))

        # Legacy: build single-candidate list from old fields
        legacy: List[Dict[str, Any]] = []
        proposed = proposal_output.get("proposed_intermediates")
        if isinstance(proposed, list):
            for idx, smiles in enumerate(proposed):
                if isinstance(smiles, str) and smiles.strip():
                    legacy.append({"rank": idx + 1, "intermediate_smiles": smiles, "reaction_description": ""})

        if not legacy:
            intermediates = proposal_output.get("intermediates")
            if isinstance(intermediates, list):
                for idx, item in enumerate(intermediates):
                    if isinstance(item, dict) and item.get("smiles"):
                        legacy.append({
                            "rank": idx + 1,
                            "intermediate_smiles": str(item["smiles"]),
                            "reaction_description": item.get("note", ""),
                        })
        return legacy

    @staticmethod
    def _parse_candidate_pushes(candidate: Dict[str, Any]) -> List[Tuple[int, int, int]]:
        pushes = candidate.get("electron_pushes")
        if not isinstance(pushes, list) or not pushes:
            return []
        try:
            normalized = normalize_electron_pushes(pushes)
        except Exception:
            normalized = []
        if normalized:
            parsed_pushes: List[Tuple[int, int, int]] = []
            for move in normalized:
                if move.kind == "lone_pair" and move.source_atom is not None:
                    parsed_pushes.append((int(move.source_atom), int(move.target_atom), 2))
                    continue
                source_ref = move.through_atom if move.through_atom is not None else move.bond_end
                if source_ref is None:
                    continue
                parsed_pushes.append((int(source_ref), int(move.target_atom), 2))
            if parsed_pushes:
                candidate["electron_pushes"] = [move.as_dict() for move in normalized]
                return parsed_pushes
        parsed_pushes: List[Tuple[int, int, int]] = []
        for push in pushes:
            if not isinstance(push, dict):
                continue
            start_raw = push.get("start_atom")
            end_raw = push.get("end_atom")
            if start_raw is None or end_raw is None:
                continue
            start_match = re.search(r"\d+", str(start_raw))
            end_match = re.search(r"\d+", str(end_raw))
            if start_match is None or end_match is None:
                continue
            electrons = push.get("electrons")
            try:
                electrons_int = int(electrons)
            except (TypeError, ValueError):
                continue
            if electrons_int in {1, 2}:
                parsed_pushes.append((int(start_match.group(0)), int(end_match.group(0)), electrons_int))
        return parsed_pushes

    @classmethod
    def _candidate_ready_for_execution(cls, candidate: Dict[str, Any]) -> Tuple[bool, str]:
        repaired_reaction_smirks, error_reason = repair_candidate_reaction_smirks(
            reaction_smirks=candidate.get("reaction_smirks"),
            electron_pushes=candidate.get("electron_pushes"),
        )
        if not repaired_reaction_smirks:
            return False, error_reason
        candidate["reaction_smirks"] = repaired_reaction_smirks

        pushes = cls._parse_candidate_pushes(candidate)
        if not pushes:
            return False, "invalid_electron_pushes"
        return True, ""

    @staticmethod
    def _validation_error_strings(validation_payload: Dict[str, Any]) -> List[str]:
        texts: List[str] = []
        for check in list(validation_payload.get("checks") or []):
            if not isinstance(check, dict):
                continue
            details = check.get("details")
            if not isinstance(details, dict):
                continue
            for key in ("error", "message"):
                value = details.get(key)
                if isinstance(value, str) and value.strip():
                    texts.append(value.strip())
        return texts

    def _summarize_proposal_quality(
        self,
        *,
        attempt: int,
        candidates: List[Dict[str, Any]],
        rejected_candidate_count: int,
        candidate_attempts: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "attempt": attempt,
            "candidate_count": len(candidates),
            "rejected_candidate_count": int(rejected_candidate_count),
            "incomplete_candidate_count": 0,
            "failed_candidate_count": 0,
            "invalid_smiles_count": 0,
            "rdkit_parse_error_count": 0,
            "rdkit_valence_error_count": 0,
            "structurally_off_template_count": 0,
            "first_invalid_detail": None,
        }
        for candidate, attempt_result in candidate_attempts:
            if str(candidate.get("template_alignment") or "").strip() == "not_aligned":
                summary["structurally_off_template_count"] += 1
            status = str(attempt_result.get("status") or "")
            if status == "validated":
                continue
            reason = str(attempt_result.get("reason") or "").strip()
            if status == "incomplete":
                summary["incomplete_candidate_count"] += 1
                if summary["first_invalid_detail"] is None and reason:
                    summary["first_invalid_detail"] = reason
                continue
            summary["failed_candidate_count"] += 1
            validation_payload = attempt_result.get("last_validation")
            if not isinstance(validation_payload, dict):
                validation_payload = {}
            texts = self._validation_error_strings(validation_payload)
            joined = " ".join(texts)
            has_parse = bool(
                re.search(r"SMILES Parse Error|Failed parsing SMILES|could not parse", joined)
            )
            has_valence = bool(re.search(r"Explicit valence .* greater than permitted", joined))
            if has_parse or has_valence:
                summary["invalid_smiles_count"] += 1
                if summary["first_invalid_detail"] is None and texts:
                    summary["first_invalid_detail"] = texts[0]
            if has_parse:
                summary["rdkit_parse_error_count"] += 1
            if has_valence:
                summary["rdkit_valence_error_count"] += 1
            if summary["first_invalid_detail"] is None and texts:
                summary["first_invalid_detail"] = texts[0]

        total = len(candidates)
        summary["all_candidates_incomplete"] = total > 0 and summary["incomplete_candidate_count"] == total
        summary["all_candidates_invalid_smiles"] = total > 0 and summary["invalid_smiles_count"] == total
        summary["all_candidates_invalid_valence"] = total > 0 and summary["rdkit_valence_error_count"] == total
        summary["all_candidates_unassessable"] = total > 0 and (
            summary["incomplete_candidate_count"] + summary["invalid_smiles_count"] == total
        )
        summary["all_candidates_not_aligned"] = (
            total > 0 and summary["structurally_off_template_count"] == total
        )
        return summary

    def _record_template_guidance_preaccept_observation(
        self,
        state: RunState,
        *,
        attempt: int,
        alignment: str,
        reason: str,
        proposal_quality_summary: Dict[str, Any],
    ) -> None:
        guidance = state.template_guidance_state
        if guidance is None or guidance.mode not in {"active", "weak"}:
            return
        observation = {
            "step_index": attempt,
            "alignment": alignment,
            "reason": reason,
            "source": "preaccept",
        }
        guidance.alignment_history.append(observation)
        self.store.append_event(
            state.run_id,
            "template_guidance_preaccept_observation",
            {
                "attempt": attempt,
                "alignment": alignment,
                "reason": reason,
                "proposal_quality_summary": dict(proposal_quality_summary),
            },
            step_name="mechanism_step_proposal",
        )
        self._emit_template_guidance_state(state)

    def _try_candidate_with_retries(
        self,
        state: RunState,
        candidate: Dict[str, Any],
        proposal_output: Dict[str, Any],
        enabled_validators: Optional[set[str]] = None,
    ) -> Dict[str, Any]:
        """Validate a single candidate with up to 3 retry attempts.

        Returns a dict with keys:
        - ``status``: ``validated`` | ``failed`` | ``incomplete``
        - ``branch_candidate``: BranchCandidate when validated
        - ``last_validation``: latest validation payload for failures
        """
        smiles = candidate.get("intermediate_smiles", "")
        ready, incomplete_reason = self._candidate_ready_for_execution(candidate)
        if not ready:
            self.store.append_event(
                state.run_id,
                "mechanism_candidate_incomplete",
                {
                    "attempt": state.step_index + 1,
                    "candidate_rank": candidate.get("rank"),
                    "candidate_smiles": smiles,
                    "reason": incomplete_reason,
                },
                step_name="mechanism_step_proposal",
            )
            return {
                "status": "incomplete",
                "branch_candidate": None,
                "last_validation": None,
                "reason": incomplete_reason,
            }

        # Build a scoped intermediate output for MechanismAgent
        scoped_output: Dict[str, Any] = {
            **proposal_output,
            "selected_candidate": dict(candidate),
            "intermediates": [{"smiles": smiles}],
        }

        retry_feedback: Optional[Dict[str, Any]] = None
        last_validation: Dict[str, Any] = {}
        last_failed_checks: List[str] = []
        last_signature: str = ""
        repeated_signatures: Dict[str, int] = {}
        rescue_attempted = False
        rescue_outcome = "none"

        max_retries = max(1, int(state.run_config.retry_same_candidate_max or 1))
        for retry_index in range(max_retries):
            self._mark_step_started(
                state,
                step_name="mechanism_synthesis",
                tool_name="predict_mechanistic_step",
                attempt=state.step_index + 1,
                retry_index=retry_index,
            )

            mechanism_result = self.mechanism_agent.run(
                state,
                scoped_output,
                retry_feedback=retry_feedback,
            )
            mechanism_result.attempt = state.step_index + 1
            mechanism_result.retry_index = retry_index
            mechanism_result.validation = validate_mechanism_step_output(
                mechanism_result.output,
                dbe_policy=state.run_config.dbe_policy,
                enabled_validators=enabled_validators,
            )
            self._record_step(state, mechanism_result)
            self._record_validation_checks(state, mechanism_result=mechanism_result)

            validation_payload = mechanism_result.validation.as_dict()
            last_validation = validation_payload
            retry_feedback = self._retry_feedback_for_validation(validation_payload)
            last_failed_checks = list(retry_feedback.get("failed_checks", []))
            last_signature = self._validation_signature(validation_payload)
            repeated_signatures[last_signature] = repeated_signatures.get(last_signature, 0) + 1

            if validation_payload.get("passed"):
                self._record_arrow_push_annotation(
                    state,
                    mechanism_output=mechanism_result.output,
                    attempt=state.step_index + 1,
                    retry_index=retry_index,
                    candidate_rank=int(candidate.get("rank") or 0),
                    source="mechanism_loop",
                )
                return {
                    "status": "validated",
                    "branch_candidate": BranchCandidate(
                        rank=int(candidate.get("rank") or 0),
                        intermediate_smiles=smiles,
                        intermediate_output=candidate,
                        mechanism_output=mechanism_result.output,
                        resulting_state=list(mechanism_result.output.get("resulting_state") or []),
                        validation_summary=validation_payload,
                    ),
                    "last_validation": validation_payload,
                    "reason": "",
                    "failed_checks": [],
                    "validation_signature": "",
                    "candidate_rank": int(candidate.get("rank") or 0),
                    "rescue_attempted": rescue_attempted,
                    "rescue_outcome": rescue_outcome,
                }

            rescue_result = self._attempt_candidate_rescue(
                state,
                mechanism_result=mechanism_result,
                failed_checks=last_failed_checks,
                candidate_rank=candidate.get("rank"),
            )
            if rescue_result is not None:
                rescue_attempted = True
                rescue_output = rescue_result.output or {}
                add_reactants = [str(x) for x in rescue_output.get("add_reactants") or []]
                add_products = [str(x) for x in rescue_output.get("add_products") or []]
                if add_reactants or add_products:
                    rescue_outcome = "applied"
                    maybe_output = dict(mechanism_result.output or {})
                    base_current = [str(x) for x in maybe_output.get("current_state") or []]
                    base_resulting = [str(x) for x in maybe_output.get("resulting_state") or []]
                    def _merge_unique(base: List[str], additions: List[str]) -> List[str]:
                        merged = list(base)
                        seen = set(base)
                        for item in additions:
                            if item in seen:
                                continue
                            merged.append(item)
                            seen.add(item)
                        return merged
                    # Reagents are consumed from the current side; byproducts are added to resulting side.
                    maybe_output["current_state"] = _merge_unique(base_current, add_reactants)
                    maybe_output["resulting_state"] = _merge_unique(base_resulting, add_products)
                    maybe_output["rescue_additions"] = {
                        "add_reactants": add_reactants,
                        "add_products": add_products,
                        "dbe_adjustment_hint": rescue_output.get("dbe_adjustment_hint"),
                    }
                    rescued_validation_result = validate_mechanism_step_output(
                        maybe_output,
                        dbe_policy=state.run_config.dbe_policy,
                        enabled_validators=enabled_validators,
                    )
                    rescued_validation = rescued_validation_result.as_dict()
                    if rescued_validation.get("passed"):
                        # Persist a successful mechanism_synthesis row so terminal
                        # completion gates count rescue-validated candidates.
                        rescued_step_result = StepResult(
                            step_name="mechanism_synthesis",
                            tool_name="predict_mechanistic_step",
                            output=maybe_output,
                            model=mechanism_result.model,
                            reasoning_level=mechanism_result.reasoning_level,
                            attempt=state.step_index + 1,
                            retry_index=retry_index,
                            source=mechanism_result.source,
                            validation=rescued_validation_result,
                            token_usage=mechanism_result.token_usage,
                            cost=mechanism_result.cost,
                        )
                        self._record_step(state, rescued_step_result)
                        self._record_validation_checks(
                            state,
                            validation_result=rescued_validation_result,
                            attempt=state.step_index + 1,
                            retry_index=retry_index,
                        )
                        self._record_arrow_push_annotation(
                            state,
                            mechanism_output=maybe_output,
                            attempt=state.step_index + 1,
                            retry_index=retry_index,
                            candidate_rank=int(candidate.get("rank") or 0),
                            source="mechanism_loop_rescue",
                        )
                        rescue_outcome = "validated"
                        self.store.append_event(
                            state.run_id,
                            "candidate_rescue_completed",
                            {
                                "attempt": state.step_index + 1,
                                "candidate_rank": candidate.get("rank"),
                                "status": "validated",
                            },
                            step_name="candidate_rescue",
                        )
                        return {
                            "status": "validated",
                            "branch_candidate": BranchCandidate(
                                rank=int(candidate.get("rank") or 0),
                                intermediate_smiles=smiles,
                                intermediate_output=candidate,
                                mechanism_output=maybe_output,
                                resulting_state=list(maybe_output.get("resulting_state") or []),
                                validation_summary=rescued_validation,
                            ),
                            "last_validation": rescued_validation,
                            "reason": "",
                            "failed_checks": [],
                            "validation_signature": "",
                            "candidate_rank": int(candidate.get("rank") or 0),
                            "rescue_attempted": rescue_attempted,
                            "rescue_outcome": rescue_outcome,
                        }
                else:
                    rescue_outcome = "no_changes"

            self.store.append_event(
                state.run_id,
                "mechanism_retry_failed",
                {
                    "attempt": state.step_index + 1,
                    "retry_index": retry_index,
                    "candidate_rank": candidate.get("rank"),
                    "candidate_smiles": smiles,
                    "failed_checks": last_failed_checks,
                    "validation_signature": last_signature,
                    "rescue_attempted": rescue_attempted,
                    "rescue_outcome": rescue_outcome,
                    "validation": validation_payload,
                },
                step_name="mechanism_synthesis",
            )

            repeat_failure_signature_limit = max(
                2,
                int(state.run_config.repeat_failure_signature_limit or 2),
            )
            repeat_count = repeated_signatures.get(last_signature, 0)
            if (
                state.run_config.reproposal_on_repeat_failure
                and bool(last_signature)
                and repeat_count >= repeat_failure_signature_limit
            ):
                return {
                    "status": "failed",
                    "branch_candidate": None,
                    "last_validation": last_validation,
                    "reason": "repeat_failure_signature",
                    "force_reproposal": True,
                    "failed_checks": last_failed_checks,
                    "validation_signature": last_signature,
                    "repeat_failure_signature_limit": repeat_failure_signature_limit,
                    "candidate_rank": int(candidate.get("rank") or 0),
                    "rescue_attempted": rescue_attempted,
                    "rescue_outcome": rescue_outcome,
                }

            if retry_index < max_retries - 1:
                self.store.append_event(
                    state.run_id,
                    "mechanism_retry_started",
                    {
                        "attempt": state.step_index + 1,
                        "retry_index": retry_index + 1,
                        "candidate_rank": candidate.get("rank"),
                        "retry_guidance": retry_feedback.get("guidance", ""),
                    },
                    step_name="mechanism_synthesis",
                )

        return {
            "status": "failed",
            "branch_candidate": None,
            "last_validation": last_validation,
            "reason": "",
            "failed_checks": last_failed_checks,
            "validation_signature": last_signature,
            "candidate_rank": int(candidate.get("rank") or 0),
            "rescue_attempted": rescue_attempted,
            "rescue_outcome": rescue_outcome,
        }

    def _apply_candidate(self, state: RunState, candidate: BranchCandidate) -> None:
        """Apply a validated candidate to the run state."""
        previous_state = list(state.current_state)
        resulting = candidate.resulting_state
        if isinstance(resulting, list) and resulting:
            state.current_state = [str(s) for s in resulting]
        if candidate.intermediate_smiles and candidate.intermediate_smiles not in state.previous_intermediates:
            state.previous_intermediates.append(candidate.intermediate_smiles)
        state.step_index += 1

        guidance = state.template_guidance_state
        if guidance is not None:
            guidance.completed_steps_count = max(0, int(guidance.completed_steps_count or 0)) + 1
            if guidance.mode in {"active", "weak"}:
                alignment = str(
                    (candidate.intermediate_output or {}).get("template_alignment") or "unknown"
                ).strip() or "unknown"
                reason = str(
                    (candidate.intermediate_output or {}).get("template_alignment_reason") or ""
                ).strip()
                guidance.alignment_history.append(
                    {
                        "step_index": state.step_index,
                        "alignment": alignment,
                        "reason": reason,
                    }
                )
                if alignment in {"aligned", "partial"}:
                    next_index = guidance.current_template_step_index + 1
                    max_steps = int(guidance.suitable_step_count or 0)
                    guidance.current_template_step_index = (
                        min(next_index, max_steps) if max_steps > 0 else next_index
                    )
                disable_window = max(1, int(state.run_config.reaction_template_disable_step_window or 3))
                disable_consecutive = max(
                    1, int(state.run_config.reaction_template_disable_consecutive_mismatch or 2)
                )
                recent = guidance.alignment_history[-disable_consecutive:]
                if (
                    state.step_index <= disable_window
                    and len(recent) >= disable_consecutive
                    and all(item.get("alignment") == "not_aligned" for item in recent)
                ):
                    guidance.mode = "disabled"
                    guidance.disable_reason = "early_consecutive_template_mismatch"
            self._emit_template_guidance_state(state)

        self.store.append_event(
            state.run_id,
            "mechanism_step_accepted",
            {
                "step_index": state.step_index,
                "candidate_rank": candidate.rank,
                "current_state": previous_state,
                "resulting_state": list(state.current_state),
                "predicted_intermediate": candidate.intermediate_smiles,
                "contains_target_product": bool((candidate.mechanism_output or {}).get("contains_target_product")),
                "validation_summary": dict(candidate.validation_summary or {}),
            },
            step_name="mechanism_synthesis",
        )

    def _collect_failed_path_steps(self, state: RunState, from_step_index: int) -> List[Dict[str, Any]]:
        """Gather mechanism step outputs from the DB for steps after *from_step_index*."""
        outputs = self.store.list_step_outputs(state.run_id)
        steps: List[Dict[str, Any]] = []
        for row in outputs:
            if row.get("step_name") != "mechanism_synthesis":
                continue
            attempt = int(row.get("attempt") or 0)
            if attempt > from_step_index:
                steps.append({
                    "attempt": attempt,
                    "output": row.get("output"),
                    "validation": row.get("validation"),
                })
        return steps

    def _backtrack(self, state: RunState) -> bool:
        """Revert to the most recent branch point with untried alternatives.

        Records the current failed path into ``state.failed_paths`` for UI
        display, then reverts ``current_state``, ``previous_intermediates``,
        and ``step_index`` to the branch point snapshot.  The next alternative
        candidate is applied so the loop can continue with a clean slate.

        Returns ``True`` if backtracking succeeded, ``False`` if no options remain.
        """
        for i in range(len(state.branch_points) - 1, -1, -1):
            bp = state.branch_points[i]
            if bp.exhausted or not bp.alternatives:
                continue

            # Record the failed path for UI display
            failed_steps = self._collect_failed_path_steps(state, bp.step_index)
            chosen_rank = bp.chosen_candidate.rank if bp.chosen_candidate else -1
            state.failed_paths.append(FailedPath(
                branch_step_index=bp.step_index,
                candidate_rank=chosen_rank,
                steps_taken=failed_steps,
                failure_reason="validation_retry_exhausted",
            ))

            self.store.append_event(
                state.run_id,
                "failed_path_recorded",
                {
                    "branch_step_index": bp.step_index,
                    "candidate_rank": chosen_rank,
                    "steps_in_path": len(failed_steps),
                },
            )

            # Pop the next alternative
            next_alt = bp.alternatives.pop(0)
            if not bp.alternatives:
                bp.exhausted = True

            # CLEAN SLATE: revert state to the snapshot at this branch point
            state.current_state = list(bp.current_state)
            state.previous_intermediates = list(bp.previous_intermediates)
            state.step_index = bp.step_index
            if bp.template_guidance_snapshot is not None:
                state.template_guidance_state = TemplateGuidanceState.from_dict(
                    bp.template_guidance_snapshot
                )
                self._emit_template_guidance_state(state)

            # Remove branch points created after this one (belong to failed path)
            state.branch_points = state.branch_points[: i + 1]

            # Update the chosen candidate at this branch point
            bp.chosen_candidate = next_alt

            # Apply the alternative candidate
            self._apply_candidate(state, next_alt)

            self.store.append_event(
                state.run_id,
                "backtrack",
                {
                    "reverted_to_step": bp.step_index,
                    "alternative_rank": next_alt.rank,
                    "intermediate": next_alt.intermediate_smiles,
                    "remaining_alternatives": len(bp.alternatives),
                },
            )
            return True

        return False

    def _peek_next_alternative(
        self, state: RunState
    ) -> Optional[Tuple["BranchPoint", "BranchCandidate"]]:
        """Return (branch_point, candidate) for the next viable alternative, or None."""
        for i in range(len(state.branch_points) - 1, -1, -1):
            bp = state.branch_points[i]
            if not bp.exhausted and bp.alternatives:
                return bp, bp.alternatives[0]
        return None

    def _pause_for_last_chance(
        self,
        state: RunState,
        bp: "BranchPoint",
        alt: "BranchCandidate",
        *,
        attempt: int,
    ) -> None:
        """Pause with the last viable alternative stored for resume.

        Records the current failed path, then pauses with reason='last_chance_backtrack'
        and the alternative candidate serialised into pause details for replay on resume.
        """
        failed_steps = self._collect_failed_path_steps(state, bp.step_index)
        chosen_rank = bp.chosen_candidate.rank if bp.chosen_candidate else -1
        state.failed_paths.append(FailedPath(
            branch_step_index=bp.step_index,
            candidate_rank=chosen_rank,
            steps_taken=failed_steps,
            failure_reason="validation_retry_exhausted",
        ))
        self.store.append_event(
            state.run_id,
            "failed_path_recorded",
            {
                "branch_step_index": bp.step_index,
                "candidate_rank": chosen_rank,
                "steps_in_path": len(failed_steps),
            },
        )

        pause_payload: Dict[str, Any] = {
            "reason": "last_chance_backtrack",
            "attempt": attempt,
            "has_alternative": True,
            "pending_alternative": {
                "rank": alt.rank,
                "intermediate_smiles": alt.intermediate_smiles,
                "intermediate_output": dict(alt.intermediate_output),
                "mechanism_output": dict(alt.mechanism_output),
                "resulting_state": list(alt.resulting_state),
            },
            "revert_to_step": bp.step_index,
            "revert_current_state": list(bp.current_state),
            "revert_previous_intermediates": list(bp.previous_intermediates),
            "revert_template_guidance_state": (
                dict(bp.template_guidance_snapshot)
                if isinstance(bp.template_guidance_snapshot, dict)
                else None
            ),
        }
        pause_id = self.store.create_run_pause(
            run_id=state.run_id,
            reason="last_chance_backtrack",
            details=pause_payload,
        )
        state.paused = True
        self.store.set_run_status(state.run_id, "paused")
        self.store.append_event(
            state.run_id,
            "mechanism_retry_exhausted",
            {**pause_payload, "pause_id": pause_id},
            step_name="mechanism_synthesis",
        )
        self.store.append_event(
            state.run_id,
            "run_paused",
            {
                "pause_id": pause_id,
                "reason": "last_chance_backtrack",
                "attempt": attempt,
                "has_alternative": True,
            },
            step_name="mechanism_synthesis",
        )
        raise _RunPaused()

    # ------------------------------------------------------------------
    # Topology-aware proposal strategies
    # ------------------------------------------------------------------

    def _propose_for_topology(
        self,
        state: RunState,
        harness: Optional[HarnessConfig],
        proposal_hints: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Dispatch proposal to the appropriate topology strategy.

        Returns (proposal_output, candidates) regardless of topology.
        """
        topology = state.run_config.coordination_topology
        profile = harness.get_topology_profile(topology) if harness else TopologyProfile()
        template_guidance = self._build_template_guidance_payload(state)
        if proposal_hints and isinstance(proposal_hints, dict):
            merged = dict(template_guidance or {})
            merged.update(proposal_hints)
            template_guidance = merged

        self.store.append_event(
            state.run_id,
            "topology_dispatch",
            {
                "topology": topology,
                "agent_count": profile.agent_count,
                "max_candidates_per_agent": profile.max_candidates_per_agent,
                "peer_rounds": profile.peer_rounds,
                "aggregation_mode": profile.aggregation_mode,
            },
            step_name="mechanism_step_proposal",
        )

        if topology == "sas":
            return self._propose_sas(state, profile, template_guidance)
        elif topology == "independent_mas":
            return self._propose_independent(state, profile, template_guidance)
        elif topology == "decentralized_mas":
            return self._propose_decentralized(state, profile, template_guidance)
        else:  # centralized_mas (default / current behavior)
            return self._propose_centralized(state, profile, template_guidance)

    def _single_proposal_call(
        self,
        state: RunState,
        template_guidance: Optional[Dict[str, Any]],
        *,
        record: bool = True,
    ) -> Tuple[Optional["StepResult"], Dict[str, Any]]:
        """Execute one IntermediateAgent.run() call and optionally record it."""
        proposal_result = None
        proposal_output: Dict[str, Any] = {}
        if state.run_config.intermediate_prediction_enabled:
            if record:
                self._mark_step_started(
                    state,
                    step_name="mechanism_step_proposal",
                    tool_name="propose_mechanism_step",
                    attempt=state.step_index + 1,
                )
            try:
                proposal_result = self.intermediate_agent.run(
                    state,
                    template_guidance=template_guidance,
                )
            except TypeError:
                proposal_result = self.intermediate_agent.run(state)  # type: ignore[misc]
            proposal_result.attempt = state.step_index + 1
            if record:
                self._record_step(state, proposal_result)
            proposal_output = proposal_result.output if proposal_result else {}
        return proposal_result, proposal_output

    @staticmethod
    def _aggregate_usage_cost_from_results(
        results: List[Tuple[int, Optional["StepResult"]]],
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, int]], Optional[Dict[str, float]]]:
        """Aggregate per-agent usage/cost metadata from proposal calls."""
        from mechanistic_agent.model_registry import update_cost_totals, update_usage_totals

        per_agent: List[Dict[str, Any]] = []
        usage_totals: Dict[str, int] = {}
        cost_totals: Dict[str, float] = {}
        for agent_idx, result in results:
            usage = result.token_usage if result else None
            cost = result.cost if result else None
            per_agent.append({"agent_idx": agent_idx, "usage": usage, "cost": cost})
            if isinstance(usage, dict):
                update_usage_totals(usage_totals, usage)
            if isinstance(cost, dict):
                update_cost_totals(cost_totals, cost)
        return per_agent, (usage_totals or None), (cost_totals or None)

    def _propose_sas(
        self,
        state: RunState,
        profile: TopologyProfile,
        template_guidance: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Single Agent System: one call, top-1 candidate only."""
        _, proposal_output = self._single_proposal_call(state, template_guidance)
        candidates = self._extract_candidates_from_proposal(proposal_output)
        # SAS: keep only the top-ranked candidate
        if candidates:
            candidates = candidates[:1]
        return proposal_output, candidates

    def _propose_centralized(
        self,
        state: RunState,
        profile: TopologyProfile,
        template_guidance: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Centralized MAS: current behavior extracted as-is."""
        _, proposal_output = self._single_proposal_call(state, template_guidance)
        candidates = self._extract_candidates_from_proposal(proposal_output)
        return proposal_output, candidates

    def _propose_independent(
        self,
        state: RunState,
        profile: TopologyProfile,
        template_guidance: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Independent MAS: N parallel calls, synthesis-only merge."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        agent_count = max(1, profile.agent_count)

        # Record step start once for the overall proposal
        self._mark_step_started(
            state,
            step_name="mechanism_step_proposal",
            tool_name="propose_mechanism_step",
            attempt=state.step_index + 1,
        )

        def _call_agent(agent_idx: int) -> Tuple[int, Optional["StepResult"]]:
            try:
                result = self.intermediate_agent.run(
                    state, template_guidance=template_guidance,
                )
            except TypeError:
                result = self.intermediate_agent.run(state)  # type: ignore[misc]
            if result is not None:
                result.attempt = state.step_index + 1
            return agent_idx, result

        results: List[Tuple[int, Optional["StepResult"]]] = []
        with ThreadPoolExecutor(max_workers=agent_count) as pool:
            futures = {pool.submit(_call_agent, i): i for i in range(agent_count)}
            for f in as_completed(futures):
                results.append(f.result())

        results.sort(key=lambda r: r[0])

        # Merge candidate pools
        merged_candidates: List[Dict[str, Any]] = []
        first_output: Dict[str, Any] = {}
        for agent_idx, step_result in results:
            output = step_result.output if step_result else {}
            if not first_output:
                first_output = output
            self.store.append_event(
                state.run_id,
                "independent_agent_result",
                {"agent_idx": agent_idx, "candidate_count": len(self._extract_candidates_from_proposal(output))},
                step_name="mechanism_step_proposal",
            )
            for c in self._extract_candidates_from_proposal(output):
                c["source_agent"] = agent_idx
                merged_candidates.append(c)

        # Interleave by original rank then agent index
        merged_candidates.sort(key=lambda c: (c.get("rank", 99), c.get("source_agent", 0)))
        for i, c in enumerate(merged_candidates):
            c["rank"] = i + 1

        per_agent_usage_cost, aggregated_usage, aggregated_cost = self._aggregate_usage_cost_from_results(results)

        # Record a synthetic step result from the first agent's output
        if first_output:
            merged_output = dict(first_output)
            merged_output["candidates"] = merged_candidates
            merged_output["topology"] = "independent_mas"
            merged_output["agent_count"] = len(results)
            merged_output["agent_usage_cost"] = per_agent_usage_cost
            merged_output["aggregated_usage_cost"] = {
                "usage": aggregated_usage,
                "cost": aggregated_cost,
            }
            result_record = StepResult(
                step_name="mechanism_step_proposal",
                tool_name="propose_mechanism_step",
                output=merged_output,
                source="llm",
                attempt=state.step_index + 1,
                token_usage=aggregated_usage,
                cost=aggregated_cost,
            )
            self._record_step(state, result_record)
        else:
            merged_output = {
                "candidates": merged_candidates,
                "topology": "independent_mas",
                "agent_count": len(results),
                "agent_usage_cost": per_agent_usage_cost,
                "aggregated_usage_cost": {
                    "usage": aggregated_usage,
                    "cost": aggregated_cost,
                },
            }

        return merged_output, merged_candidates

    def _propose_decentralized(
        self,
        state: RunState,
        profile: TopologyProfile,
        template_guidance: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Decentralized MAS: N agents x D rounds with peer summaries, consensus merge."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        agent_count = max(1, profile.agent_count)
        rounds = max(1, profile.peer_rounds)

        self._mark_step_started(
            state,
            step_name="mechanism_step_proposal",
            tool_name="propose_mechanism_step",
            attempt=state.step_index + 1,
        )

        round_outputs: List[List[Dict[str, Any]]] = [[] for _ in range(agent_count)]
        all_results: List[Tuple[int, Optional["StepResult"], int]] = []
        first_output: Dict[str, Any] = {}

        for d in range(rounds):
            peer_summaries = self._build_peer_summaries(round_outputs, d) if d > 0 else None

            def _call(agent_idx: int, peer_ctx: Optional[List[List[Dict[str, Any]]]]) -> Tuple[int, Optional["StepResult"]]:
                tg = dict(template_guidance or {})
                if peer_ctx and agent_idx < len(peer_ctx):
                    tg["peer_proposals"] = peer_ctx[agent_idx]
                try:
                    result = self.intermediate_agent.run(state, template_guidance=tg)
                except TypeError:
                    result = self.intermediate_agent.run(state)  # type: ignore[misc]
                if result is not None:
                    result.attempt = state.step_index + 1
                return agent_idx, result

            round_results: List[Tuple[int, Optional["StepResult"]]] = []
            with ThreadPoolExecutor(max_workers=agent_count) as pool:
                futures = {
                    pool.submit(_call, i, peer_summaries): i
                    for i in range(agent_count)
                }
                for f in as_completed(futures):
                    round_results.append(f.result())

            for agent_idx, step_result in round_results:
                output = step_result.output if step_result else {}
                if not first_output:
                    first_output = output
                all_results.append((agent_idx, step_result, d + 1))
                round_outputs[agent_idx] = self._extract_candidates_from_proposal(output)
                self.store.append_event(
                    state.run_id,
                    "independent_agent_result",
                    {"agent_idx": agent_idx, "round": d + 1, "candidate_count": len(round_outputs[agent_idx])},
                    step_name="mechanism_step_proposal",
                )

            self.store.append_event(
                state.run_id,
                "peer_round_complete",
                {"round": d + 1, "total_rounds": rounds, "agents": agent_count},
                step_name="mechanism_step_proposal",
            )

        # Consensus merge from final round
        merged = self._consensus_merge(round_outputs, profile)

        self.store.append_event(
            state.run_id,
            "consensus_merge_result",
            {"candidate_count": len(merged), "agents": agent_count, "rounds": rounds},
            step_name="mechanism_step_proposal",
        )

        # Record synthetic step result
        merged_output = dict(first_output) if first_output else {}
        merged_output["candidates"] = merged
        merged_output["topology"] = "decentralized_mas"
        merged_output["agent_count"] = agent_count
        merged_output["peer_rounds"] = rounds

        agg_input = [(agent_idx, result) for agent_idx, result, _ in all_results]
        per_agent_usage_cost, aggregated_usage, aggregated_cost = self._aggregate_usage_cost_from_results(agg_input)
        if all_results:
            # Round is included only in decentralized mode to disambiguate repeated agent indexes.
            for i, (_, _, round_idx) in enumerate(all_results):
                per_agent_usage_cost[i]["round"] = round_idx
        merged_output["agent_usage_cost"] = per_agent_usage_cost
        merged_output["aggregated_usage_cost"] = {
            "usage": aggregated_usage,
            "cost": aggregated_cost,
        }

        if first_output:
            result_record = StepResult(
                step_name="mechanism_step_proposal",
                tool_name="propose_mechanism_step",
                output=merged_output,
                source="llm",
                attempt=state.step_index + 1,
                token_usage=aggregated_usage,
                cost=aggregated_cost,
            )
            self._record_step(state, result_record)

        return merged_output, merged

    @staticmethod
    def _build_peer_summaries(
        round_outputs: List[List[Dict[str, Any]]],
        current_round: int,
    ) -> List[List[Dict[str, Any]]]:
        """Build per-agent peer context from other agents' prior-round candidates."""
        agent_count = len(round_outputs)
        summaries: List[List[Dict[str, Any]]] = []
        for agent_idx in range(agent_count):
            peer_items: List[Dict[str, Any]] = []
            for other_idx in range(agent_count):
                if other_idx == agent_idx:
                    continue
                for c in round_outputs[other_idx]:
                    peer_items.append({
                        "smiles": c.get("intermediate_smiles", ""),
                        "reaction": c.get("reaction_description", ""),
                        "rank": c.get("rank", 99),
                        "agent": other_idx,
                    })
            summaries.append(peer_items)
        return summaries

    @staticmethod
    def _consensus_merge(
        round_outputs: List[List[Dict[str, Any]]],
        profile: TopologyProfile,
    ) -> List[Dict[str, Any]]:
        """Merge final-round candidates with consensus bonus."""
        key_field = profile.consensus_key or "reaction_smirks"
        fallback_field = profile.consensus_fallback_key or "intermediate_smiles"

        # Collect all final-round candidates with source agent
        all_candidates: List[Dict[str, Any]] = []
        for agent_idx, candidates in enumerate(round_outputs):
            for c in candidates:
                entry = dict(c)
                entry["source_agent"] = agent_idx
                all_candidates.append(entry)

        if not all_candidates:
            return []

        # Group by consensus key
        from collections import Counter
        key_support: Counter = Counter()
        for c in all_candidates:
            consensus_val = str(c.get(key_field) or c.get(fallback_field) or "").strip()
            if consensus_val:
                key_support[consensus_val] += 1

        # Assign consensus bonus: candidates with 2+ agent support get lower rank
        for c in all_candidates:
            consensus_val = str(c.get(key_field) or c.get(fallback_field) or "").strip()
            support = key_support.get(consensus_val, 1)
            original_rank = c.get("rank", 99)
            # Consensus bonus: subtract 100 * (support - 1) so multi-agent consensus sorts first
            c["_sort_key"] = (-(support - 1), original_rank, c.get("source_agent", 0))

        all_candidates.sort(key=lambda c: c["_sort_key"])

        # De-duplicate by consensus key (keep first/best)
        seen_keys: set = set()
        merged: List[Dict[str, Any]] = []
        for c in all_candidates:
            consensus_val = str(c.get(key_field) or c.get(fallback_field) or "").strip()
            if consensus_val and consensus_val in seen_keys:
                continue
            if consensus_val:
                seen_keys.add(consensus_val)
            c.pop("_sort_key", None)
            c["rank"] = len(merged) + 1
            merged.append(c)

        return merged

    def _run_mechanism_loop(
        self,
        state: RunState,
        stop_event: threading.Event,
        harness: Optional[HarnessConfig] = None,
    ) -> None:
        start = time.monotonic()
        max_steps = max(1, state.run_config.max_steps)
        reproposals_by_step: Dict[int, int] = {}
        reproposal_hints: Dict[int, Dict[str, Any]] = {}

        while state.step_index < max_steps:
            if stop_event.is_set() or state.stop_requested:
                break

            self.store.append_event(
                state.run_id,
                "loop_iteration_started",
                {"step_index": state.step_index + 1},
            )

            elapsed = time.monotonic() - start
            if elapsed > state.run_config.max_runtime_seconds:
                self.store.append_event(
                    state.run_id,
                    "runtime_limit",
                    {
                        "max_runtime_seconds": state.run_config.max_runtime_seconds,
                        "elapsed_seconds": elapsed,
                    },
                )
                break

            # --- Resume path: apply the stored alternative from a last_chance pause ---
            if state.pending_resume_candidate is not None:
                chosen = state.pending_resume_candidate
                state.pending_resume_candidate = None
                self.store.append_event(
                    state.run_id,
                    "backtrack",
                    {
                        "reverted_to_step": state.step_index,
                        "alternative_rank": chosen.rank,
                        "intermediate": chosen.intermediate_smiles,
                        "remaining_alternatives": 0,
                        "resumed_from_pause": True,
                    },
                )
                self._apply_candidate(state, chosen)
            else:
                # --- Step A: Propose mechanism step candidates (topology-aware) ---
                proposal_output, candidates = self._propose_for_topology(
                    state,
                    harness,
                    proposal_hints=reproposal_hints.get(state.step_index + 1),
                )
                rejected_candidates = proposal_output.get("rejected_candidates")
                rejected_candidate_count = (
                    len(rejected_candidates)
                    if isinstance(rejected_candidates, list)
                    else 0
                )
                all_candidates_rejected = not candidates and rejected_candidate_count > 0

                # --- Step B: Validate each candidate (up to 3 retries per candidate) ---
                validated: List[BranchCandidate] = []
                had_incomplete_candidate = False
                had_retryable_failure = False
                had_repeat_signature_failure = False
                incomplete_reasons: List[str] = []
                last_failed_validation: Dict[str, Any] = {}
                last_failed_checks: List[str] = []
                last_validation_signature = ""
                last_repeat_failure_signature_limit = max(
                    2,
                    int(state.run_config.repeat_failure_signature_limit or 2),
                )
                last_candidate_rank: Optional[int] = None
                last_rescue_attempted = False
                last_rescue_outcome = "none"
                candidate_attempts: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
                for candidate in candidates:
                    _ev = self._enabled_validators(harness) if harness else None
                    attempt_result = self._try_candidate_with_retries(
                        state, candidate, proposal_output, enabled_validators=_ev,
                    )
                    candidate_attempts.append((dict(candidate), dict(attempt_result)))
                    status = str(attempt_result.get("status") or "")
                    if status == "validated":
                        branch_candidate = attempt_result.get("branch_candidate")
                        if isinstance(branch_candidate, BranchCandidate):
                            validated.append(branch_candidate)
                    elif status == "incomplete":
                        had_incomplete_candidate = True
                        reason = str(attempt_result.get("reason") or "")
                        if reason and reason not in incomplete_reasons:
                            incomplete_reasons.append(reason)
                    else:
                        had_retryable_failure = True
                        if bool(attempt_result.get("force_reproposal")):
                            had_repeat_signature_failure = True
                        maybe_validation = attempt_result.get("last_validation")
                        if isinstance(maybe_validation, dict) and maybe_validation:
                            last_failed_validation = maybe_validation
                        last_failed_checks = list(attempt_result.get("failed_checks") or [])
                        last_validation_signature = str(attempt_result.get("validation_signature") or "")
                        maybe_repeat_limit = attempt_result.get("repeat_failure_signature_limit")
                        if isinstance(maybe_repeat_limit, int):
                            last_repeat_failure_signature_limit = max(2, maybe_repeat_limit)
                        maybe_rank = attempt_result.get("candidate_rank")
                        if isinstance(maybe_rank, int):
                            last_candidate_rank = maybe_rank
                        last_rescue_attempted = bool(attempt_result.get("rescue_attempted"))
                        last_rescue_outcome = str(attempt_result.get("rescue_outcome") or "none")
                proposal_quality_summary = self._summarize_proposal_quality(
                    attempt=state.step_index + 1,
                    candidates=candidates,
                    rejected_candidate_count=rejected_candidate_count,
                    candidate_attempts=candidate_attempts,
                )
                self.store.append_event(
                    state.run_id,
                    "proposal_quality_summary",
                    proposal_quality_summary,
                    step_name="mechanism_step_proposal",
                )

                # --- Step C: Handle validation results ---
                if not validated:
                    if proposal_quality_summary.get("all_candidates_unassessable"):
                        self._record_template_guidance_preaccept_observation(
                            state,
                            attempt=state.step_index + 1,
                            alignment="unassessable",
                            reason=(
                                "All proposed candidates were incomplete or chemically invalid before "
                                "any candidate could be accepted."
                            ),
                            proposal_quality_summary=proposal_quality_summary,
                        )
                    elif proposal_quality_summary.get("all_candidates_not_aligned"):
                        self._record_template_guidance_preaccept_observation(
                            state,
                            attempt=state.step_index + 1,
                            alignment="not_aligned",
                            reason="All proposed candidates were structurally off-template before acceptance.",
                            proposal_quality_summary=proposal_quality_summary,
                        )
                    if had_repeat_signature_failure:
                        step_key = state.step_index + 1
                        reproposals_by_step[step_key] = reproposals_by_step.get(step_key, 0) + 1
                        current_reproposals = reproposals_by_step[step_key]
                        reproposal_hints[step_key] = {
                            "avoid_signatures": [last_validation_signature] if last_validation_signature else [],
                            "avoid_failed_checks": list(last_failed_checks),
                        }
                        self.store.append_event(
                            state.run_id,
                            "mechanism_reproposal_requested",
                            {
                                "attempt": state.step_index + 1,
                                "reason": "repeat_failure_signature",
                                "reproposal_count": current_reproposals,
                                "validation_signature": last_validation_signature,
                                "repeat_failure_signature_limit": last_repeat_failure_signature_limit,
                                "avoid_signatures": [last_validation_signature] if last_validation_signature else [],
                                "avoid_failed_checks": list(last_failed_checks),
                                "candidate_rank": last_candidate_rank,
                            },
                            step_name="mechanism_step_proposal",
                        )
                        if current_reproposals >= max(1, int(state.run_config.max_reproposals_per_step or 4)):
                            self.store.append_event(
                                state.run_id,
                                "mechanism_reproposal_limit_reached",
                                {
                                    "step_index": state.step_index + 1,
                                    "reproposal_count": current_reproposals,
                                    "max_reproposals_per_step": max(
                                        1, int(state.run_config.max_reproposals_per_step or 4)
                                    ),
                                    "reason": "repeat_failure_signature",
                                    "validation_signature": last_validation_signature,
                                    "repeat_failure_signature_limit": last_repeat_failure_signature_limit,
                                    "candidate_rank": last_candidate_rank,
                                },
                                step_name="mechanism_step_proposal",
                            )
                            self.store.set_run_status(state.run_id, "failed")
                            self.store.append_event(
                                state.run_id,
                                "run_failed",
                                {
                                    "reason": "proposal_repeat_failure_loop",
                                    "step_index": state.step_index + 1,
                                    "reproposal_count": current_reproposals,
                                    "last_reproposal_reason": "repeat_failure_signature",
                                    "validation_signature": last_validation_signature,
                                    "repeat_failure_signature_limit": last_repeat_failure_signature_limit,
                                    "candidate_rank": last_candidate_rank,
                                },
                            )
                            return
                        continue
                    if all_candidates_rejected:
                        step_key = state.step_index + 1
                        reproposals_by_step[step_key] = reproposals_by_step.get(step_key, 0) + 1
                        current_reproposals = reproposals_by_step[step_key]
                        self.store.append_event(
                            state.run_id,
                            "mechanism_reproposal_requested",
                            {
                                "attempt": state.step_index + 1,
                                "reason": "all_candidates_rejected",
                                "candidate_count": len(candidates),
                                "rejected_candidate_count": rejected_candidate_count,
                                "reproposal_count": current_reproposals,
                            },
                            step_name="mechanism_step_proposal",
                        )
                        if current_reproposals >= max(1, int(state.run_config.max_reproposals_per_step or 4)):
                            self.store.append_event(
                                state.run_id,
                                "mechanism_reproposal_limit_reached",
                                {
                                    "step_index": state.step_index + 1,
                                    "reproposal_count": current_reproposals,
                                    "max_reproposals_per_step": max(
                                        1, int(state.run_config.max_reproposals_per_step or 4)
                                    ),
                                    "reason": "all_candidates_rejected",
                                    "candidate_count": len(candidates),
                                    "rejected_candidate_count": rejected_candidate_count,
                                },
                                step_name="mechanism_step_proposal",
                            )
                            self.store.set_run_status(state.run_id, "failed")
                            self.store.append_event(
                                state.run_id,
                                "run_failed",
                                {
                                    "reason": "proposal_all_candidates_rejected",
                                    "step_index": state.step_index + 1,
                                    "reproposal_count": current_reproposals,
                                    "last_reproposal_reason": "all_candidates_rejected",
                                    "candidate_count": len(candidates),
                                    "rejected_candidate_count": rejected_candidate_count,
                                },
                            )
                            return
                        continue
                    if proposal_quality_summary.get("all_candidates_invalid_valence"):
                        step_key = state.step_index + 1
                        reproposals_by_step[step_key] = reproposals_by_step.get(step_key, 0) + 1
                        current_reproposals = reproposals_by_step[step_key]
                        self.store.append_event(
                            state.run_id,
                            "mechanism_reproposal_requested",
                            {
                                "attempt": state.step_index + 1,
                                "reason": "proposal_invalid_valence_loop",
                                "candidate_count": len(candidates),
                                "reproposal_count": current_reproposals,
                                "proposal_quality_summary": proposal_quality_summary,
                            },
                            step_name="mechanism_step_proposal",
                        )
                        if current_reproposals >= max(1, int(state.run_config.max_reproposals_per_step or 4)):
                            self.store.append_event(
                                state.run_id,
                                "mechanism_reproposal_limit_reached",
                                {
                                    "step_index": state.step_index + 1,
                                    "reproposal_count": current_reproposals,
                                    "max_reproposals_per_step": max(
                                        1, int(state.run_config.max_reproposals_per_step or 4)
                                    ),
                                    "reason": "proposal_invalid_valence_loop",
                                    "candidate_count": len(candidates),
                                    "proposal_quality_summary": proposal_quality_summary,
                                },
                                step_name="mechanism_step_proposal",
                            )
                            self.store.set_run_status(state.run_id, "failed")
                            self.store.append_event(
                                state.run_id,
                                "run_failed",
                                {
                                    "reason": "proposal_invalid_valence_loop",
                                    "step_index": state.step_index + 1,
                                    "reproposal_count": current_reproposals,
                                    "last_reproposal_reason": "proposal_invalid_valence_loop",
                                    "candidate_count": len(candidates),
                                    "proposal_quality_summary": proposal_quality_summary,
                                },
                            )
                            return
                        continue
                    if proposal_quality_summary.get("all_candidates_invalid_smiles"):
                        step_key = state.step_index + 1
                        reproposals_by_step[step_key] = reproposals_by_step.get(step_key, 0) + 1
                        current_reproposals = reproposals_by_step[step_key]
                        self.store.append_event(
                            state.run_id,
                            "mechanism_reproposal_requested",
                            {
                                "attempt": state.step_index + 1,
                                "reason": "proposal_invalid_smiles_loop",
                                "candidate_count": len(candidates),
                                "reproposal_count": current_reproposals,
                                "proposal_quality_summary": proposal_quality_summary,
                            },
                            step_name="mechanism_step_proposal",
                        )
                        if current_reproposals >= max(1, int(state.run_config.max_reproposals_per_step or 4)):
                            self.store.append_event(
                                state.run_id,
                                "mechanism_reproposal_limit_reached",
                                {
                                    "step_index": state.step_index + 1,
                                    "reproposal_count": current_reproposals,
                                    "max_reproposals_per_step": max(
                                        1, int(state.run_config.max_reproposals_per_step or 4)
                                    ),
                                    "reason": "proposal_invalid_smiles_loop",
                                    "candidate_count": len(candidates),
                                    "proposal_quality_summary": proposal_quality_summary,
                                },
                                step_name="mechanism_step_proposal",
                            )
                            self.store.set_run_status(state.run_id, "failed")
                            self.store.append_event(
                                state.run_id,
                                "run_failed",
                                {
                                    "reason": "proposal_invalid_smiles_loop",
                                    "step_index": state.step_index + 1,
                                    "reproposal_count": current_reproposals,
                                    "last_reproposal_reason": "proposal_invalid_smiles_loop",
                                    "candidate_count": len(candidates),
                                    "proposal_quality_summary": proposal_quality_summary,
                                },
                            )
                            return
                        continue
                    # If all candidates were structurally incomplete, request a
                    # fresh proposal instead of retrying deterministic failures.
                    if had_incomplete_candidate and not had_retryable_failure:
                        step_key = state.step_index + 1
                        reproposals_by_step[step_key] = reproposals_by_step.get(step_key, 0) + 1
                        current_reproposals = reproposals_by_step[step_key]
                        reproposal_hints[step_key] = {
                            "incomplete_payload_reasons": list(incomplete_reasons),
                            "require_reaction_smirks": True,
                            "require_electron_pushes": True,
                        }
                        self.store.append_event(
                            state.run_id,
                            "mechanism_reproposal_requested",
                            {
                                "attempt": state.step_index + 1,
                                "reason": "incomplete_candidate_payload",
                                "candidate_count": len(candidates),
                                "reproposal_count": current_reproposals,
                                "incomplete_payload_reasons": list(incomplete_reasons),
                            },
                            step_name="mechanism_step_proposal",
                        )
                        if current_reproposals >= max(1, int(state.run_config.max_reproposals_per_step or 4)):
                            self.store.append_event(
                                state.run_id,
                                "mechanism_reproposal_limit_reached",
                                {
                                    "step_index": state.step_index + 1,
                                    "reproposal_count": current_reproposals,
                                    "max_reproposals_per_step": max(
                                        1, int(state.run_config.max_reproposals_per_step or 4)
                                    ),
                                    "reason": "incomplete_candidate_payload",
                                    "candidate_count": len(candidates),
                                },
                                step_name="mechanism_step_proposal",
                            )
                            self.store.set_run_status(state.run_id, "failed")
                            self.store.append_event(
                                state.run_id,
                                "run_failed",
                                {
                                    "reason": "proposal_incomplete_loop",
                                    "step_index": state.step_index + 1,
                                    "reproposal_count": current_reproposals,
                                    "last_reproposal_reason": "incomplete_candidate_payload",
                                    "candidate_count": len(candidates),
                                    "proposal_quality_summary": proposal_quality_summary,
                                },
                            )
                            return
                        continue
                    # No candidate passed — pause for user decision.
                    # If a branch point with an alternative exists, offer it as last chance.
                    alt_result = self._peek_next_alternative(state)
                    if alt_result is not None:
                        alt_bp, alt_candidate = alt_result
                        self._pause_for_last_chance(
                            state,
                            alt_bp,
                            alt_candidate,
                            attempt=state.step_index + 1,
                        )
                    # No alternatives remain — dead end.
                    self._pause_for_retry_exhaustion(
                        state,
                        attempt=state.step_index + 1,
                        last_validation=last_failed_validation,
                        failed_checks=last_failed_checks,
                        validation_signature=last_validation_signature,
                        candidate_rank=last_candidate_rank,
                        rescue_attempted=last_rescue_attempted,
                        rescue_outcome=last_rescue_outcome,
                    )

                # Sort by rank and pick the top-ranked validated candidate
                validated.sort(key=lambda bc: bc.rank)
                chosen = validated[0]
                alternatives = validated[1:]
                reproposal_hints.pop(state.step_index + 1, None)

                # --- Step D: Store branch point if alternatives exist ---
                if alternatives:
                    bp = BranchPoint(
                        step_index=state.step_index,
                        current_state=list(state.current_state),
                        previous_intermediates=list(state.previous_intermediates),
                        template_guidance_snapshot=(
                            state.template_guidance_state.as_dict()
                            if state.template_guidance_state is not None
                            else None
                        ),
                        chosen_candidate=chosen,
                        alternatives=alternatives,
                    )
                    state.branch_points.append(bp)
                    self.store.append_event(
                        state.run_id,
                        "branch_point_created",
                        {
                            "step_index": state.step_index,
                            "current_state": list(state.current_state),
                            "previous_intermediates": list(state.previous_intermediates),
                            "template_guidance_snapshot": (
                                bp.template_guidance_snapshot if bp.template_guidance_snapshot is not None else {}
                            ),
                            "chosen_rank": chosen.rank,
                            "alternative_count": len(alternatives),
                            "alternative_ranks": [a.rank for a in alternatives],
                        },
                    )

                # --- Step E: Apply the chosen candidate ---
                self._apply_candidate(state, chosen)

            # --- Step F: Post-step modules (reflection, step mapping, etc.) ---
            self._run_post_step_modules(state, chosen, harness)

            # --- Step G: Completion check ---
            contains_target = bool(chosen.mechanism_output.get("contains_target_product"))
            self.store.append_event(
                state.run_id,
                "completion_check",
                {
                    "step_index": state.step_index,
                    "contains_target_product": contains_target,
                    "validation_passed": True,
                },
                step_name="completion_check",
            )
            if contains_target:
                self.store.append_event(
                    state.run_id,
                    "target_products_detected",
                    {"step_index": state.step_index},
                    step_name="mechanism_synthesis",
                )
                break

            self.store.append_event(
                state.run_id,
                "loop_iteration_completed",
                {"step_index": state.step_index},
            )

    def _run_post_step_modules(
        self,
        state: RunState,
        chosen: BranchCandidate,
        harness: Optional[HarnessConfig] = None,
    ) -> None:
        """Run post-step modules after a validated mechanism step."""
        if harness is not None:
            enabled_post = [
                m for m in harness.enabled_post_step()
                if m.group_key != "validators"  # Validators run during _try_candidate_with_retries
            ]
        else:
            enabled_post = None

        if enabled_post is not None:
            for module in enabled_post:
                if module.id == "reflection":
                    self._mark_step_started(
                        state,
                        step_name="reflection",
                        tool_name="reflection_agent",
                        attempt=state.step_index,
                    )
                    reflection = self.reflection_agent.run(state, chosen.mechanism_output)
                    reflection.attempt = state.step_index
                    self._record_step(state, reflection)
                elif module.id == "step_atom_mapping":
                    self._run_step_mapping(state, chosen)
                elif module.custom:
                    context: Dict[str, Any] = {"mechanism_output": chosen.mechanism_output}
                    custom_result = self._run_custom_module(state, module, context)
                    custom_result.attempt = state.step_index
                    self._record_step(state, custom_result)
        else:
            # Legacy path: reflection always, step_mapping if enabled.
            self._mark_step_started(
                state,
                step_name="reflection",
                tool_name="reflection_agent",
                attempt=state.step_index,
            )
            reflection = self.reflection_agent.run(state, chosen.mechanism_output)
            reflection.attempt = state.step_index
            self._record_step(state, reflection)
            if state.run_config.step_mapping_enabled:
                self._run_step_mapping(state, chosen)

    def _run_step_mapping(self, state: RunState, chosen: BranchCandidate) -> None:
        """Run step atom mapping for the chosen candidate."""
        mapping_current = [str(x) for x in (chosen.mechanism_output or {}).get("current_state") or []]
        mapping_resulting = [str(x) for x in (chosen.mechanism_output or {}).get("resulting_state") or []]
        if mapping_current and mapping_resulting:
            self._mark_step_started(
                state,
                step_name="step_atom_mapping",
                tool_name="attempt_atom_mapping_for_step",
                attempt=state.step_index,
            )
            step_mapping = self.mapping_agent.run_step_mapping(
                state,
                current_state=mapping_current,
                resulting_state=mapping_resulting,
            )
            step_mapping.attempt = state.step_index
            self._record_step(state, step_mapping)
            compact = (step_mapping.output or {}).get("compact_mapped_atoms") or []
            state.latest_step_mapping = {
                "step_index": state.step_index,
                "mapped_atoms": compact[:12],
                "unmapped_atoms": (step_mapping.output or {}).get("unmapped_atoms", [])[:12],
                "confidence": (step_mapping.output or {}).get("confidence"),
            }
            self.store.append_event(
                state.run_id,
                "step_mapping_generated",
                {
                    "step_index": state.step_index,
                    "mapped_atom_count": len(compact),
                    "confidence": (step_mapping.output or {}).get("confidence"),
                },
                step_name="step_atom_mapping",
            )

    def execute_run(self, run_id: str, stop_event: threading.Event) -> None:
        run_row = self.store.get_run_row(run_id)
        if run_row is None:
            return

        state = self._build_state(run_row)

        # Ralph mode is an outer orchestration loop that spawns full child attempts.
        if (
            state.run_config.orchestration_mode == "ralph"
            and not state.run_config.ralph_parent_run_id
        ):
            prior_status = str(run_row.get("status") or "pending")
            self.store.set_run_status(run_id, "running")
            if prior_status == "paused":
                self.store.append_event(
                    run_id,
                    "run_resumed",
                    {
                        "mode": state.mode,
                        "input": asdict(state.run_input),
                        "config": asdict(state.run_config),
                        "resume_step_index": state.step_index,
                    },
                )
            else:
                self.store.append_event(
                    run_id,
                    "run_started",
                    {
                        "mode": state.mode,
                        "input": asdict(state.run_input),
                        "config": asdict(state.run_config),
                    },
                )
            try:
                from .ralph_orchestrator import RalphOrchestrator

                RalphOrchestrator(store=self.store, coordinator=self).run(
                    parent_run_id=run_id,
                    parent_row=run_row,
                    state=state,
                    stop_event=stop_event,
                )
                return
            except Exception as exc:  # pragma: no cover - defensive
                self.store.set_run_status(run_id, "failed")
                self.store.append_event(
                    run_id,
                    "run_failed",
                    {"reason": "ralph_uncaught_exception", "error": str(exc)},
                )
                return

        harness = self._resolve_harness(state)

        # Set thread-local model context so tool functions can read model config.
        model_context.set_run_context(
            step_models=state.run_config.step_models,
            step_reasoning=state.run_config.step_reasoning,
            active_model=state.run_config.model,
            model_family=state.run_config.model_family,
            reasoning_level=state.run_config.reasoning_level,
            api_keys=state.run_config.api_keys,
            few_shot_policies=harness.few_shot_policies_by_call(),
        )

        prior_status = str(run_row.get("status") or "pending")
        self.store.set_run_status(run_id, "running")
        if prior_status == "paused":
            # If resuming from a last_chance_backtrack pause with decision="continue",
            # reconstruct the pending alternative and revert state to the branch point.
            latest_pause = self.store.get_latest_run_pause(run_id)
            if (
                latest_pause
                and latest_pause.get("reason") == "last_chance_backtrack"
                and latest_pause.get("decision") == "continue"
            ):
                details = latest_pause.get("details") or {}
                alt_data = details.get("pending_alternative") or {}
                if alt_data:
                    state.pending_resume_candidate = BranchCandidate(
                        rank=int(alt_data.get("rank") or 99),
                        intermediate_smiles=str(alt_data.get("intermediate_smiles") or ""),
                        intermediate_output=dict(alt_data.get("intermediate_output") or {}),
                        mechanism_output=dict(alt_data.get("mechanism_output") or {}),
                        resulting_state=list(alt_data.get("resulting_state") or []),
                    )
                    state.current_state = list(details.get("revert_current_state") or state.current_state)
                    state.previous_intermediates = list(
                        details.get("revert_previous_intermediates") or state.previous_intermediates
                    )
                    state.step_index = int(details.get("revert_to_step") or state.step_index)
                    revert_template_guidance = details.get("revert_template_guidance_state")
                    if isinstance(revert_template_guidance, dict):
                        state.template_guidance_state = TemplateGuidanceState.from_dict(
                            revert_template_guidance
                        )
                        self._emit_template_guidance_state(state)

            self.store.append_event(
                run_id,
                "run_resumed",
                {
                    "mode": state.mode,
                    "input": asdict(state.run_input),
                    "config": asdict(state.run_config),
                    "resume_step_index": state.step_index,
                },
            )
        else:
            self.store.append_event(
                run_id,
                "run_started",
                {
                    "mode": state.mode,
                    "input": asdict(state.run_input),
                    "config": asdict(state.run_config),
                },
            )

        try:
            self._run_initial_phase(state, harness)

            if state.mode == "verified":
                self.store.append_event(
                    run_id,
                    "awaiting_manual_steps",
                    {"next_step_index": state.step_index + 1},
                )
                # verified mode now requires human step submission and validation.
                self.store.set_run_status(run_id, "running")
                return

            self._run_mechanism_loop(state, stop_event, harness)

            if state.paused:
                return
            if hasattr(self.store, "get_run_row"):
                post_loop = self.store.get_run_row(run_id)  # type: ignore[attr-defined]
                if isinstance(post_loop, dict) and str(post_loop.get("status") or "") in {"failed", "completed", "stopped"}:
                    return

            if stop_event.is_set() or state.stop_requested:
                self.store.set_run_status(run_id, "stopped")
                self.store.append_event(run_id, "run_stopped", {"reason": "stop_requested"})
                return

            events = self.store.list_events(run_id)
            reached_runtime_limit = any(
                str(event.get("event_type") or "") == "runtime_limit"
                for event in events
            )
            step_outputs = self.store.list_step_outputs(run_id)
            mechanism_steps = [
                row
                for row in step_outputs
                if row.get("step_name") == "mechanism_synthesis"
                and isinstance(row.get("validation"), dict)
                and bool(row["validation"].get("passed"))
            ]

            if not mechanism_steps:
                self.store.set_run_status(run_id, "failed")
                self.store.append_event(
                    run_id,
                    "run_failed",
                    {
                        "reason": (
                            "runtime_limit_reached"
                            if reached_runtime_limit
                            else "no_valid_mechanism_steps_generated"
                        )
                    },
                )
                return

            has_completion = any(
                bool((row.get("output") or {}).get("contains_target_product"))
                for row in mechanism_steps
            )
            if has_completion:
                self.store.set_run_status(run_id, "completed")
                self.store.append_event(
                    run_id,
                    "run_completed",
                    {
                        "mode": state.mode,
                        "mechanism_steps": len(mechanism_steps),
                        "validated_only": True,
                    },
                )
                return

            self.store.set_run_status(run_id, "failed")
            self.store.append_event(
                run_id,
                "run_failed",
                {
                    "reason": (
                        "runtime_limit_reached"
                        if reached_runtime_limit
                        else "completion_not_reached_within_limits"
                    ),
                    "validated_mechanism_steps": len(mechanism_steps),
                },
            )
        except _RunPaused:
            return
        except Exception as exc:  # pragma: no cover - defensive
            self.store.set_run_status(run_id, "failed")
            self.store.append_event(
                run_id,
                "run_failed",
                {"reason": "uncaught_exception", "error": str(exc)},
            )
        finally:
            model_context.clear_run_context()


class RunManager:
    """Background execution manager for run coordinator threads."""

    def __init__(self, coordinator: RunCoordinator) -> None:
        self.coordinator = coordinator
        self._threads: Dict[str, threading.Thread] = {}
        self._stops: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def start(self, run_id: str) -> None:
        with self._lock:
            existing = self._threads.get(run_id)
            if existing and existing.is_alive():
                return
            stop_event = threading.Event()
            thread = threading.Thread(
                target=self.coordinator.execute_run,
                args=(run_id, stop_event),
                daemon=True,
            )
            self._stops[run_id] = stop_event
            self._threads[run_id] = thread
            thread.start()

    def stop(self, run_id: str) -> bool:
        with self._lock:
            stop_event = self._stops.get(run_id)
            if stop_event is None:
                return False
            stop_event.set()
            return True

    def is_running(self, run_id: str) -> bool:
        with self._lock:
            thread = self._threads.get(run_id)
            return bool(thread and thread.is_alive())
