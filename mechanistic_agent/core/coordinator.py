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
from mechanistic_agent.smiles_utils import canonicalize_if_valid

from . import model_context
from .arrow_push import predict_arrow_push_annotation
from .baseline_runner import BaselineRunner
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
            chemistry_backend=self._coerce_literal(
                config.get("chemistry_backend"),
                {"auto", "rdkit_cli", "python"},
                "auto",
            ),  # type: ignore[arg-type]
            chemistry_backend_parity=self._coerce_bool(
                config.get("chemistry_backend_parity", False),
                False,
            ),
            rdkit_cli_command=str(config.get("rdkit_cli_command") or "rdkit_cli"),
            rdkit_cli_timeout_seconds=self._coerce_float(
                config.get("rdkit_cli_timeout_seconds", 5.0),
                5.0,
            ),
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
            adaptive_harness_mode=self._coerce_literal(
                config.get("adaptive_harness_mode")
                or (
                    "conservative"
                    if str(config.get("harness_name") or "default").strip() == "adaptive_default"
                    else "off"
                ),
                {"off", "conservative"},
                "off",
            ),  # type: ignore[arg-type]
            babysit_mode=self._coerce_literal(
                config.get("babysit_mode"),
                {"off", "advisory"},
                "off",
            ),  # type: ignore[arg-type]
            allow_validator_mutation=self._coerce_bool(
                config.get("allow_validator_mutation", False),
                False,
            ),
            mutation_lane=(
                str(config.get("mutation_lane")).strip().lower()
                if str(config.get("mutation_lane") or "").strip().lower()
                in {"topology", "harness", "prompt", "few_shot"}
                else None
            ),
            ralph_parent_run_id=(
                str(config.get("ralph_parent_run_id"))
                if config.get("ralph_parent_run_id")
                else None
            ),
            proceed_on_validation_failure=self._coerce_bool(
                config.get("proceed_on_validation_failure", False),
                False,
            ),
            proceed_only_on_arrow_push_failure=self._coerce_bool(
                config.get("proceed_only_on_arrow_push_failure", True),
                True,
            ),
            runtime_trace_enabled=self._coerce_bool(
                config.get("runtime_trace_enabled", False),
                False,
            ),
            runtime_trace_label=(
                str(config.get("runtime_trace_label")).strip()
                if config.get("runtime_trace_label")
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
                    "current_state": list(latest_mapping.get("current_state") or []),
                    "resulting_state": list(latest_mapping.get("resulting_state") or []),
                    "mapped_atoms": list(latest_mapping.get("compact_mapped_atoms") or [])[:12],
                    "species_lineage_summary": list(latest_mapping.get("species_lineage_summary") or [])[:8],
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

    @staticmethod
    def _runtime_trace_enabled(state: RunState) -> bool:
        return bool(getattr(state.run_config, "runtime_trace_enabled", False))

    @staticmethod
    def _trace_run_label(state: RunState) -> str:
        label = str(getattr(state.run_config, "runtime_trace_label", "") or "").strip()
        if label:
            return label
        return str(state.run_id or "")[:8]

    @staticmethod
    def _short_text(value: Any, limit: int = 120) -> str:
        text = str(value or "").strip()
        text = re.sub(r"\s+", " ", text)
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)] + "..."

    def _short_smiles_list(self, values: Any, *, limit: int = 3) -> str:
        if not isinstance(values, list):
            return ""
        items = [self._short_text(item, 48) for item in values if str(item or "").strip()]
        if not items:
            return ""
        head = items[:limit]
        suffix = ""
        if len(items) > limit:
            suffix = f", +{len(items) - limit} more"
        return "[" + ", ".join(head) + suffix + "]"

    def _summarize_candidate(self, candidate: Dict[str, Any]) -> str:
        rank = int(candidate.get("rank") or 0)
        smiles = self._short_text(candidate.get("intermediate_smiles"), 64) or "n/a"
        smirks = self._short_text(candidate.get("reaction_smirks"), 96)
        pushes = candidate.get("electron_pushes")
        push_count = len(pushes) if isinstance(pushes, list) else 0
        parts = [f"rank={rank}", f"smiles={smiles}"]
        if smirks:
            parts.append(f"smirks={smirks}")
        if push_count:
            parts.append(f"pushes={push_count}")
        resulting = self._short_smiles_list(candidate.get("resulting_state"))
        if resulting:
            parts.append(f"resulting={resulting}")
        return " ".join(parts)

    def _summarize_validation_details(self, details: Any) -> str:
        if not isinstance(details, dict):
            return ""
        parts: List[str] = []
        for key in ("code", "message", "error", "field", "summary"):
            value = details.get(key)
            if value:
                parts.append(f"{key}={self._short_text(value, 96)}")
        chemistry_backend = details.get("chemistry_backend")
        if isinstance(chemistry_backend, dict):
            backend_used = str(chemistry_backend.get("backend_used") or "").strip()
            if backend_used:
                parts.append(f"backend={backend_used}")
            fallback_reason = str(chemistry_backend.get("fallback_reason") or "").strip()
            if fallback_reason:
                parts.append(f"fallback={fallback_reason}")
            rdkit_cli_error_code = str(chemistry_backend.get("rdkit_cli_error_code") or "").strip()
            if rdkit_cli_error_code:
                parts.append(f"rdkit_cli_error={rdkit_cli_error_code}")
        invalid_smiles = details.get("invalid_smiles")
        if isinstance(invalid_smiles, list) and invalid_smiles:
            parts.append(f"invalid_smiles={self._short_smiles_list(invalid_smiles, limit=2)}")
        return " ".join(parts[:5])

    def _summarize_validation(self, validation: Optional[Dict[str, Any]]) -> str:
        if not isinstance(validation, dict):
            return ""
        checks = validation.get("checks")
        if not isinstance(checks, list):
            return "validation=pass" if validation.get("passed") else "validation=fail"
        failed = []
        for item in checks:
            if not isinstance(item, dict):
                continue
            if item.get("passed"):
                continue
            name = str(item.get("name") or "").strip()
            detail = self._summarize_validation_details(item.get("details"))
            failed.append(f"{name}({detail})" if detail else name)
        if validation.get("passed"):
            return "validation=pass"
        if failed:
            return "validation=fail " + "; ".join(failed[:3])
        return "validation=fail"

    def _summarize_step_output(
        self,
        step_name: str,
        output: Dict[str, Any],
        *,
        validation: Optional[Dict[str, Any]] = None,
    ) -> str:
        parts: List[str] = []
        if step_name == "balance_analysis":
            deficits = output.get("remaining_deficit")
            if isinstance(deficits, dict) and deficits:
                parts.append(f"deficit={self._short_text(json.dumps(deficits, sort_keys=True), 96)}")
        elif step_name == "initial_conditions":
            env = self._short_text(output.get("environment"), 24)
            acid = self._short_smiles_list([item.get("smiles") for item in output.get("acid_candidates", []) if isinstance(item, dict)])
            base = self._short_smiles_list([item.get("smiles") for item in output.get("base_candidates", []) if isinstance(item, dict)])
            if env:
                parts.append(f"environment={env}")
            if acid:
                parts.append(f"acid={acid}")
            if base:
                parts.append(f"base={base}")
        elif step_name == "missing_reagents":
            reactants = self._short_smiles_list(output.get("missing_reactants") or output.get("missing_reagents"))
            products = self._short_smiles_list(output.get("missing_products"))
            if reactants:
                parts.append(f"missing_reactants={reactants}")
            if products:
                parts.append(f"missing_products={products}")
        elif step_name in {"atom_mapping", "step_atom_mapping"}:
            for key in ("mapped_reaction", "reaction_smirks", "mapping"):
                value = self._short_text(output.get(key), 120)
                if value:
                    parts.append(f"{key}={value}")
                    break
        elif step_name == "reaction_type_mapping":
            selected = self._short_text(output.get("selected_label_exact"), 48)
            confidence = output.get("confidence")
            if selected:
                parts.append(f"selected={selected}")
            if isinstance(confidence, (int, float)):
                parts.append(f"confidence={float(confidence):.2f}")
        elif step_name == "mechanism_step_proposal":
            candidates = output.get("candidates")
            rejected = output.get("rejected_candidates")
            topology = self._short_text(output.get("topology"), 32)
            if topology:
                parts.append(f"topology={topology}")
            if isinstance(candidates, list):
                parts.append(f"candidates={len(candidates)}")
                if candidates:
                    parts.append(
                        " | ".join(
                            self._summarize_candidate(dict(item))
                            for item in candidates[:3]
                            if isinstance(item, dict)
                        )
                    )
            if isinstance(rejected, list) and rejected:
                parts.append(f"rejected={len(rejected)}")
        elif step_name == "mechanism_synthesis":
            predicted = self._short_text(output.get("predicted_intermediate"), 64)
            smirks = self._short_text(output.get("reaction_smirks") or output.get("raw_reaction_smirks"), 120)
            pushes = output.get("electron_pushes")
            push_count = len(pushes) if isinstance(pushes, list) else 0
            current = self._short_smiles_list(output.get("current_state"))
            resulting = self._short_smiles_list(output.get("resulting_state"))
            if predicted:
                parts.append(f"predicted={predicted}")
            if smirks:
                parts.append(f"smirks={smirks}")
            parts.append(f"pushes={push_count}")
            if current:
                parts.append(f"current={current}")
            if resulting:
                parts.append(f"resulting={resulting}")
            if output.get("contains_target_product") is not None:
                parts.append(f"contains_target={bool(output.get('contains_target_product'))}")
        elif step_name == "candidate_rescue":
            status = self._short_text(output.get("status"), 32)
            error = self._short_text(output.get("error"), 64)
            add_reactants = self._short_smiles_list(output.get("add_reactants"))
            add_products = self._short_smiles_list(output.get("add_products"))
            if status:
                parts.append(f"status={status}")
            if error:
                parts.append(f"error={error}")
            if add_reactants:
                parts.append(f"add_reactants={add_reactants}")
            if add_products:
                parts.append(f"add_products={add_products}")
        elif step_name in ALL_VALIDATOR_IDS:
            passed = output.get("passed")
            if passed is not None:
                parts.append(f"passed={bool(passed)}")
            detail = self._summarize_validation_details(output.get("details"))
            if detail:
                parts.append(detail)

        validation_summary = self._summarize_validation(validation)
        if validation_summary:
            parts.append(validation_summary)
        return " ".join(part for part in parts if part).strip()

    def _trace(
        self,
        state: RunState,
        message: str,
        *,
        step_name: Optional[str] = None,
        attempt: Optional[int] = None,
        retry_index: Optional[int] = None,
    ) -> None:
        if not self._runtime_trace_enabled(state):
            return
        prefix_parts = [f"TRACE[{self._trace_run_label(state)}"]
        if step_name:
            prefix_parts.append(f"step={step_name}")
        if attempt is not None:
            prefix_parts.append(f"attempt={attempt}")
        if retry_index is not None:
            prefix_parts.append(f"retry={retry_index}")
        prefix = " ".join(prefix_parts) + "]"
        print(f"{prefix} {message}", flush=True)

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
        duration_text = f" duration={duration_seconds:.2f}s" if duration_seconds is not None else ""
        summary = self._summarize_step_output(
            result.step_name,
            result.output if isinstance(result.output, dict) else {},
            validation=validation_payload,
        )
        self._trace(
            state,
            f"DONE tool={result.tool_name} source={result.source}{duration_text} {summary}".strip(),
            step_name=result.step_name,
            attempt=result.attempt,
            retry_index=result.retry_index,
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
        context_parts = [f"START tool={tool_name}"]
        if step_name in {"mechanism_step_proposal", "mechanism_synthesis"}:
            current = self._short_smiles_list(state.current_state)
            targets = self._short_smiles_list(state.run_input.products)
            if current:
                context_parts.append(f"current={current}")
            if targets:
                context_parts.append(f"targets={targets}")
        self._trace(
            state,
            " ".join(context_parts),
            step_name=step_name,
            attempt=attempt,
            retry_index=retry_index,
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
    def _merge_unique_species(base: List[str], additions: List[str]) -> List[str]:
        merged = list(base)
        seen = set(base)
        for item in additions:
            if item in seen:
                continue
            merged.append(item)
            seen.add(item)
        return merged

    @staticmethod
    def _constraint_replacements_for_species(
        species: str,
        proposal_constraints: Dict[str, Any],
    ) -> List[str]:
        replacements: List[str] = []
        for pair in proposal_constraints.get("conjugate_pairs") or []:
            if not isinstance(pair, dict):
                continue
            left = str(pair.get("left") or "").strip()
            right = str(pair.get("right") or "").strip()
            if species == left and right:
                replacements.append(right)
            elif species == right and left:
                replacements.append(left)
        deduped: List[str] = []
        seen: set[str] = set()
        for item in replacements:
            if item in seen:
                continue
            deduped.append(item)
            seen.add(item)
        return deduped

    @staticmethod
    def _canonicalize_constraint_species_list(items: Any) -> List[str]:
        canonical: List[str] = []
        seen: set[str] = set()
        if not isinstance(items, list):
            return canonical
        for item in items:
            text = str(item or "").strip()
            if not text:
                continue
            normalized = canonicalize_if_valid(text) or text
            if normalized in seen:
                continue
            canonical.append(normalized)
            seen.add(normalized)
        return canonical

    def _build_proposal_constraint_guidance(self, state: RunState) -> Optional[Dict[str, Any]]:
        missing_output = self._latest_output_by_step(state.run_id, "missing_reagents") or {}
        proposal_constraints = (
            dict(missing_output.get("proposal_constraints") or {})
            if isinstance(missing_output.get("proposal_constraints"), dict)
            else {}
        )
        species_registry = [
            dict(item)
            for item in (missing_output.get("species_registry") or [])
            if isinstance(item, dict)
        ]
        participant_summary = (
            dict(missing_output.get("participant_summary") or {})
            if isinstance(missing_output.get("participant_summary"), dict)
            else {}
        )

        if state.selected_reaction_template and isinstance(state.selected_reaction_template, dict):
            raw_byproducts = state.selected_reaction_template.get("canonical_byproducts") or []
            canonical_byproducts = self._canonicalize_constraint_species_list(
                [str(item) for item in raw_byproducts if isinstance(item, str)]
            )
            proposal_constraints["canonical_byproducts"] = self._merge_unique_species(
                self._canonicalize_constraint_species_list(
                    list(proposal_constraints.get("canonical_byproducts") or [])
                ),
                canonical_byproducts,
            )
            proposal_constraints["allowed_generated_species"] = self._merge_unique_species(
                self._canonicalize_constraint_species_list(
                    list(proposal_constraints.get("allowed_generated_species") or [])
                ),
                canonical_byproducts,
            )

        atom_mapping_output = self._latest_output_by_step(state.run_id, "atom_mapping") or {}
        atom_mapping_summary = {}
        if isinstance(atom_mapping_output, dict):
            llm_response = atom_mapping_output.get("llm_response")
            atom_mapping_summary = {
                "confidence": atom_mapping_output.get("confidence"),
                "unmapped_atoms": (
                    list(llm_response.get("unmapped_atoms") or [])[:12]
                    if isinstance(llm_response, dict)
                    else []
                ),
            }

        if not proposal_constraints and not species_registry and not participant_summary and not atom_mapping_summary:
            return None

        return {
            "proposal_constraints": proposal_constraints,
            "species_registry": species_registry[:12],
            "participant_summary": participant_summary,
            "atom_mapping_summary": atom_mapping_summary,
        }

    def _apply_candidate_constraint_repairs(
        self,
        state: RunState,
        candidate: Dict[str, Any],
        proposal_constraints: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[str]]:
        resulting_state = candidate.get("resulting_state")
        if not isinstance(resulting_state, list):
            return dict(candidate), []

        repaired = dict(candidate)
        repaired_state = [str(item) for item in resulting_state if str(item).strip()]
        repair_notes: List[str] = []
        persistent_species = set(
            self._canonicalize_constraint_species_list(
                list(proposal_constraints.get("persistent_species") or [])
            )
            + self._canonicalize_constraint_species_list(
                list(proposal_constraints.get("spectator_species") or [])
            )
            + self._canonicalize_constraint_species_list(
                list(proposal_constraints.get("counterion_species") or [])
            )
        )

        for species in list(state.current_state):
            if species not in persistent_species:
                continue
            if species in repaired_state:
                continue
            replacements = self._constraint_replacements_for_species(species, proposal_constraints)
            if replacements and any(item in repaired_state for item in replacements):
                continue
            repaired_state.append(species)
            repair_notes.append(f"carried_persistent_species:{species}")

        if repair_notes:
            repaired["resulting_state"] = repaired_state
            repaired["constraint_repairs"] = list(repair_notes)
        return repaired, repair_notes

    def _prevalidate_candidate_against_constraints(
        self,
        state: RunState,
        candidate: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        guidance = self._build_proposal_constraint_guidance(state)
        if not isinstance(guidance, dict):
            return dict(candidate), None
        proposal_constraints = guidance.get("proposal_constraints")
        if not isinstance(proposal_constraints, dict) or not proposal_constraints:
            return dict(candidate), None

        repaired_candidate, repair_notes = self._apply_candidate_constraint_repairs(
            state,
            candidate,
            proposal_constraints,
        )
        resulting_state = repaired_candidate.get("resulting_state")
        if not isinstance(resulting_state, list):
            return repaired_candidate, None

        current_state = [str(item) for item in state.current_state if str(item).strip()]
        resulting_clean = [str(item) for item in resulting_state if str(item).strip()]
        introduced = [item for item in resulting_clean if item not in current_state]
        forbidden_new = set(
            self._canonicalize_constraint_species_list(
                list(proposal_constraints.get("forbidden_new_species") or [])
            )
        )
        violating_species = sorted(item for item in introduced if item in forbidden_new)
        if violating_species:
            return repaired_candidate, {
                "reason": "forbidden_new_species",
                "species": violating_species,
                "environment": proposal_constraints.get("environment"),
                "repair_notes": repair_notes,
            }

        persistent_missing: List[str] = []
        for species in self._canonicalize_constraint_species_list(
            list(proposal_constraints.get("persistent_species") or [])
        ):
            if species not in current_state:
                continue
            if species in resulting_clean:
                continue
            replacements = self._constraint_replacements_for_species(species, proposal_constraints)
            if replacements and any(item in resulting_clean for item in replacements):
                continue
            persistent_missing.append(species)

        if persistent_missing:
            return repaired_candidate, {
                "reason": "persistent_species_removed",
                "species": persistent_missing,
                "environment": proposal_constraints.get("environment"),
                "repair_notes": repair_notes,
            }

        return repaired_candidate, None

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
        warning_checks: List[str] = []
        validator_hints: Dict[str, str] = {}
        checks = validation_payload.get("checks")
        if isinstance(checks, list):
            for check in checks:
                if not isinstance(check, dict):
                    continue
                name = str(check.get("name") or "unknown")
                details = check.get("details")
                if check.get("passed") is True:
                    if isinstance(details, dict) and bool(details.get("warning_only")):
                        warning_checks.append(name)
                        warnings = details.get("warnings")
                        if isinstance(warnings, list):
                            for warning in warnings:
                                if isinstance(warning, str) and warning.strip():
                                    guidance_parts.append(f"{name}: {warning.strip()}")
                                    break
                        elif isinstance(warnings, str) and warnings.strip():
                            guidance_parts.append(f"{name}: {warnings.strip()}")
                        if bool(details.get("retry_recommended")):
                            guidance_parts.append(
                                f"{name}: retry recommended with explicit spectator/byproduct carry-through."
                            )
                    continue
                failed_checks.append(name)
                if isinstance(details, dict):
                    error_code = details.get("error_code")
                    fix_suggestions = details.get("fix_suggestions")
                    if isinstance(error_code, str) and error_code.strip():
                        guidance_parts.append(f"{name} [{error_code.strip()}]")
                    if isinstance(fix_suggestions, list):
                        for suggestion in fix_suggestions:
                            if isinstance(suggestion, str) and suggestion.strip():
                                guidance_parts.append(f"{name}: {suggestion.strip()}")
                                break
                    error_text = details.get("error") or details.get("message")
                    if isinstance(error_text, str) and error_text.strip():
                        # Provide specific guidance for SMILES validation errors
                        if "Invalid SMILES" in error_text:
                            guidance_parts.append(f"{name}: Invalid SMILES detected - ensure all chemical structures are valid and parseable by RDKit. Avoid excessive radicals, unclosed rings, and invalid atom symbols.")
                        elif "balance_check_failed" in error_text:
                            guidance_parts.append(f"{name}: Atom balance analysis failed - check that all SMILES strings are properly formatted and represent valid chemical structures.")
                        else:
                            guidance_parts.append(f"{name}: {error_text.strip()}")
                if name == "atom_balance":
                    validator_hints[name] = (
                        "Ensure current_state and resulting_state conserve atom counts; include expected "
                        "counterions/byproducts and proton transfer species when required."
                    )
                elif name == "dbe_metadata":
                    validator_hints[name] = (
                        "Ensure reaction_smirks contains parseable |mech:v1;...| metadata and that "
                        "bond/electron deltas are consistent with electron_pushes."
                    )
                elif name == "state_progress":
                    validator_hints[name] = (
                        "Ensure resulting_state is not identical to current_state and reflects forward "
                        "mechanistic progress toward target products."
                    )
        return {
            "failed_checks": failed_checks,
            "warning_checks": warning_checks,
            "guidance": "; ".join(guidance_parts) if guidance_parts else "",
            "validator_hints": validator_hints,
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
        rescue_output = rescue_result.output or {}
        if str(rescue_output.get("error") or "") == "candidate_rescue_invalid_species":
            self.store.append_event(
                state.run_id,
                "invalid_species_in_rescue_input",
                {
                    "attempt": state.step_index + 1,
                    "candidate_rank": candidate_rank,
                    "invalid_species": list(rescue_output.get("invalid_species") or []),
                    "current_state_size": len(current_state),
                    "resulting_state_size": len(resulting_state),
                    "balance_diagnostics": rescue_output.get("balance_diagnostics") or {},
                },
                step_name="candidate_rescue",
            )
            self.store.append_event(
                state.run_id,
                "rescue_skipped_due_to_invalid_species",
                {
                    "attempt": state.step_index + 1,
                    "candidate_rank": candidate_rank,
                    "invalid_species": list(rescue_output.get("invalid_species") or []),
                },
                step_name="candidate_rescue",
            )
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
            self._update_chemistry_backend_telemetry(state, check)
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

    def _update_chemistry_backend_telemetry(self, state: RunState, check: StepValidationCheck) -> None:
        details = check.details if isinstance(check.details, dict) else {}
        backend_meta = details.get("chemistry_backend")
        if not isinstance(backend_meta, dict):
            return

        telemetry = state.adaptive_runtime_state.setdefault(
            "chemistry_backend_telemetry",
            {
                "calls": 0,
                "backend_used_counts": {},
                "backend_requested_counts": {},
                "fallback_count": 0,
                "fallback_reasons": {},
                "rdkit_cli_error_counts": {},
                "first_rdkit_cli_error": None,
            },
        )
        telemetry["calls"] = int(telemetry.get("calls") or 0) + 1

        backend_used = str(backend_meta.get("backend_used") or "python")
        used_counts = telemetry.setdefault("backend_used_counts", {})
        used_counts[backend_used] = int(used_counts.get(backend_used) or 0) + 1

        backend_requested = str(backend_meta.get("backend_requested") or "auto")
        requested_counts = telemetry.setdefault("backend_requested_counts", {})
        requested_counts[backend_requested] = int(requested_counts.get(backend_requested) or 0) + 1

        fallback_used = bool(backend_meta.get("fallback_used"))
        if fallback_used:
            telemetry["fallback_count"] = int(telemetry.get("fallback_count") or 0) + 1
            fallback_reason = str(backend_meta.get("fallback_reason") or "unknown")
            fallback_reasons = telemetry.setdefault("fallback_reasons", {})
            fallback_reasons[fallback_reason] = int(fallback_reasons.get(fallback_reason) or 0) + 1

        rdkit_cli_error_code = str(backend_meta.get("rdkit_cli_error_code") or "").strip()
        rdkit_cli_error = str(backend_meta.get("rdkit_cli_error") or "").strip()
        error_key = rdkit_cli_error_code or (rdkit_cli_error[:120] if rdkit_cli_error else "")
        if error_key:
            error_counts = telemetry.setdefault("rdkit_cli_error_counts", {})
            error_counts[error_key] = int(error_counts.get(error_key) or 0) + 1
            if telemetry.get("first_rdkit_cli_error") is None:
                telemetry["first_rdkit_cli_error"] = {
                    "code": rdkit_cli_error_code or None,
                    "message": rdkit_cli_error or None,
                    "check": check.name,
                }
            self.store.append_event(
                state.run_id,
                "chemistry_backend_error",
                {
                    "check": check.name,
                    "backend_requested": backend_requested,
                    "backend_used": backend_used,
                    "fallback_used": fallback_used,
                    "fallback_reason": backend_meta.get("fallback_reason"),
                    "rdkit_cli_error_code": rdkit_cli_error_code or None,
                    "rdkit_cli_error": rdkit_cli_error or None,
                },
                step_name=check.name,
            )
        if bool(details.get("warning_only")):
            warnings = details.get("warnings")
            warning_text = ""
            if isinstance(warnings, list):
                warning_text = next(
                    (str(item).strip() for item in warnings if isinstance(item, str) and item.strip()),
                    "",
                )
            elif isinstance(warnings, str):
                warning_text = warnings.strip()
            self.store.append_event(
                state.run_id,
                "chemistry_backend_soft_warning",
                {
                    "check": check.name,
                    "backend_requested": backend_requested,
                    "backend_used": backend_used,
                    "fallback_used": fallback_used,
                    "fallback_reason": backend_meta.get("fallback_reason"),
                    "rdkit_cli_error_code": rdkit_cli_error_code or None,
                    "rdkit_cli_error": rdkit_cli_error or None,
                    "warning": warning_text or None,
                    "soft_pass_reason": details.get("soft_pass_reason"),
                    "retry_recommended": bool(details.get("retry_recommended")),
                    "known_soft_pass": bool(details.get("known_soft_pass")),
                },
                step_name=check.name,
            )

        if fallback_used:
            self.store.append_event(
                state.run_id,
                "chemistry_backend_fallback",
                {
                    "check": check.name,
                    "backend_requested": backend_requested,
                    "backend_used": backend_used,
                    "fallback_reason": backend_meta.get("fallback_reason"),
                    "rdkit_cli_error_code": rdkit_cli_error_code or None,
                },
                step_name=check.name,
            )

    def _emit_chemistry_backend_summary_event(self, state: RunState) -> None:
        telemetry = state.adaptive_runtime_state.get("chemistry_backend_telemetry")
        if not isinstance(telemetry, dict):
            return
        if telemetry.get("_emitted"):
            return
        payload = {
            "calls": int(telemetry.get("calls") or 0),
            "backend_used_counts": dict(telemetry.get("backend_used_counts") or {}),
            "backend_requested_counts": dict(telemetry.get("backend_requested_counts") or {}),
            "fallback_count": int(telemetry.get("fallback_count") or 0),
            "fallback_reasons": dict(telemetry.get("fallback_reasons") or {}),
            "rdkit_cli_error_counts": dict(telemetry.get("rdkit_cli_error_counts") or {}),
            "first_rdkit_cli_error": telemetry.get("first_rdkit_cli_error"),
        }
        self.store.append_event(
            state.run_id,
            "chemistry_backend_summary",
            payload,
            step_name="mechanism_synthesis",
        )
        telemetry["_emitted"] = True

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

    # ── Proceed-on-failure helpers ────────────────────────────────────────

    @staticmethod
    def _is_arrow_push_only_failure(
        failed_checks: List[str],
        last_validation: Dict[str, Any],
        incomplete_reasons: List[str],
    ) -> bool:
        """Return True when the only failures are arrow-push / DBE related.

        Arrow-push failures come from ``bond_electron_validation`` (dbe_metadata check)
        or from the incomplete-candidate path where ``reaction_smirks`` or
        ``electron_pushes`` are missing/invalid.
        """
        arrow_incomplete_reasons = {
            "missing_reaction_smirks",
            "missing_electron_pushes",
            "reaction_smirks_invalid_mech_block",
            "reaction_smirks_missing_mech_block",
            "invalid_electron_pushes",
        }
        if any(r in arrow_incomplete_reasons for r in (incomplete_reasons or [])):
            # Even if there are non-arrow reasons, arrow-push-only means we only
            # have arrow-push reasons.
            non_arrow_reasons = [r for r in (incomplete_reasons or []) if r not in arrow_incomplete_reasons]
            if not non_arrow_reasons:
                return True

        if failed_checks:
            non_arrow_checks = [c for c in failed_checks if c not in {"dbe_metadata", "bond_electron_validation"}]
            if not non_arrow_checks:
                # Only DBE/bond_electron checks failed.
                return True
        return False

    def _best_soft_candidate(
        self,
        candidates: List[Dict[str, Any]],
        candidate_attempts: List[Tuple[Dict[str, Any], Dict[str, Any]]],
        proposal_output: Dict[str, Any],
        soft_reason: str = "proceed_on_validation_failure",
    ) -> Optional[BranchCandidate]:
        """Pick the best available candidate for a soft-advance step.

        Priority: candidate that has a valid intermediate_smiles with passing
        atom_balance/state_progress, regardless of bond_electron / reaction_smirks.
        Falls back to the rank-1 candidate from the proposal.
        """
        best_smiles: Optional[str] = None
        best_resulting: Optional[List[str]] = None
        best_output: Optional[Dict[str, Any]] = None

        for candidate_data, attempt_result in candidate_attempts:
            smiles = str(candidate_data.get("intermediate_smiles") or "").strip()
            if not smiles:
                continue
            last_validation = attempt_result.get("last_validation") or {}
            if not last_validation:
                continue
            checks = last_validation.get("checks") or []
            # Accept if atom_balance and state_progress pass (even if dbe fails).
            atom_ok = any(
                c.get("name") == "atom_balance" and c.get("passed")
                for c in checks
                if isinstance(c, dict)
            )
            state_ok = any(
                c.get("name") == "state_progress" and c.get("passed")
                for c in checks
                if isinstance(c, dict)
            )
            if atom_ok and state_ok and best_smiles is None:
                # Find the corresponding mechanism_output for this candidate.
                step_outputs = attempt_result.get("mechanism_output") or {}
                if not isinstance(step_outputs, dict):
                    step_outputs = {}
                resulting = step_outputs.get("resulting_state") or []
                if resulting:
                    best_smiles = smiles
                    best_resulting = list(resulting)
                    best_output = dict(step_outputs)

        # Fallback: use rank-1 candidate from proposal output as soft step.
        if best_smiles is None:
            for candidate_data, _ in candidate_attempts:
                smiles = str(candidate_data.get("intermediate_smiles") or "").strip()
                if smiles:
                    best_smiles = smiles
                    best_resulting = list(candidate_data.get("resulting_state") or [])
                    break

        if not best_smiles:
            return None

        if best_output is None:
            best_output = {}

        return BranchCandidate(
            rank=99,
            intermediate_smiles=best_smiles,
            intermediate_output={},
            mechanism_output={
                **(best_output or {}),
                "soft_advance": True,
                "soft_advance_reason": soft_reason,
                "contains_target_product": bool((best_output or {}).get("contains_target_product")),
            },
            resulting_state=best_resulting or [],
            validation_summary={
                "passed": False,
                "soft_advance": True,
                "checks": [],
            },
        )

    @staticmethod
    def _validation_check_passed(validation: Dict[str, Any], check_name: str) -> bool:
        checks = validation.get("checks") if isinstance(validation, dict) else None
        if not isinstance(checks, list):
            return False
        return any(
            isinstance(check, dict)
            and str(check.get("name") or "") == check_name
            and bool(check.get("passed"))
            for check in checks
        )

    def _best_balance_pending_candidate(
        self,
        *,
        candidate_attempts: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    ) -> Optional[BranchCandidate]:
        for candidate_data, attempt_result in candidate_attempts:
            validation = attempt_result.get("last_validation")
            if not isinstance(validation, dict) or not validation:
                continue
            failed_checks = {str(item) for item in attempt_result.get("failed_checks") or []}
            if failed_checks != {"atom_balance"}:
                continue
            if not self._validation_check_passed(validation, "state_progress"):
                continue

            mechanism_output = attempt_result.get("mechanism_output")
            if not isinstance(mechanism_output, dict) or not mechanism_output:
                continue
            resulting_state = [str(item) for item in mechanism_output.get("resulting_state") or []]
            if not resulting_state:
                continue

            contains_target = bool(mechanism_output.get("contains_target_product"))
            current_state = [str(item) for item in mechanism_output.get("current_state") or []]
            advances_state = bool(resulting_state) and set(resulting_state) != set(current_state)
            if not contains_target and not advances_state:
                continue

            smiles = str(candidate_data.get("intermediate_smiles") or "").strip()
            if not smiles:
                continue

            return BranchCandidate(
                rank=int(candidate_data.get("rank") or 0),
                intermediate_smiles=smiles,
                intermediate_output=dict(candidate_data),
                mechanism_output={
                    **mechanism_output,
                    "soft_advance": True,
                    "soft_advance_reason": "balance_pending",
                    "balance_pending_validation": validation,
                },
                resulting_state=resulting_state,
                validation_summary=validation,
            )
        return None

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

    @staticmethod
    def _normalise_chemistry_error_code(error_text: str) -> Optional[str]:
        text = str(error_text or "").strip()
        if not text:
            return None
        lowered = text.lower()
        if "explicit valence" in lowered and "greater than permitted" in lowered:
            return "explicit_valence"
        if "can't kekulize" in lowered or "unkekulized" in lowered:
            return "kekulize_fail"
        if "non-ring atom" in lowered and "aromatic" in lowered:
            return "aromatic_non_ring"
        if "unclosed ring" in lowered:
            return "unclosed_ring"
        if (
            "smiles parse error" in lowered
            or "failed parsing smiles" in lowered
            or "could not parse" in lowered
            or "invalid smiles" in lowered
        ):
            return "smiles_parse"
        return None

    @classmethod
    def _extract_chemistry_error_codes(cls, errors: List[str]) -> List[str]:
        ordered: List[str] = []
        for error in errors:
            code = cls._normalise_chemistry_error_code(error)
            if not code:
                continue
            if code in ordered:
                continue
            ordered.append(code)
        return ordered

    @staticmethod
    def _runtime_guard_window(
        max_runtime_seconds: float,
        *,
        fraction: float = 0.08,
        min_seconds: float = 0.01,
        max_seconds: float = 30.0,
    ) -> float:
        return min(max_seconds, max(min_seconds, float(max_runtime_seconds) * float(fraction)))

    def _runtime_budget_guard_triggered(
        self,
        state: RunState,
        *,
        loop_start: Optional[float],
        step_name: str,
        fraction: float,
    ) -> bool:
        if loop_start is None:
            return False
        elapsed = time.monotonic() - float(loop_start)
        remaining = float(state.run_config.max_runtime_seconds) - float(elapsed)
        guard_seconds = self._runtime_guard_window(
            state.run_config.max_runtime_seconds,
            fraction=fraction,
        )
        if remaining > guard_seconds:
            return False
        self.store.append_event(
            state.run_id,
            "runtime_budget_guard_triggered",
            {
                "step_index": state.step_index + 1,
                "step_name": step_name,
                "elapsed_seconds": elapsed,
                "remaining_seconds": max(0.0, remaining),
                "guard_seconds": guard_seconds,
                "max_runtime_seconds": state.run_config.max_runtime_seconds,
            },
        )
        return True

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
            "execution_exception_count": 0,
            "invalid_smiles_count": 0,
            "rdkit_parse_error_count": 0,
            "rdkit_valence_error_count": 0,
            "kekulize_error_count": 0,
            "aromatic_non_ring_error_count": 0,
            "unclosed_ring_error_count": 0,
            "structurally_off_template_count": 0,
            "first_invalid_detail": None,
            "first_chemistry_error_code": None,
            "chemistry_error_codes": {},
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
            if reason in {"candidate_execution_exception", "mechanism_validation_exception"}:
                summary["execution_exception_count"] += 1
                if summary["first_invalid_detail"] is None:
                    summary["first_invalid_detail"] = reason
            validation_payload = attempt_result.get("last_validation")
            if not isinstance(validation_payload, dict):
                validation_payload = {}
            texts = self._validation_error_strings(validation_payload)
            chemistry_error_codes = self._extract_chemistry_error_codes(texts)
            for code in chemistry_error_codes:
                summary["chemistry_error_codes"][code] = int(summary["chemistry_error_codes"].get(code, 0)) + 1
            if summary["first_chemistry_error_code"] is None and chemistry_error_codes:
                summary["first_chemistry_error_code"] = chemistry_error_codes[0]
            joined = " ".join(texts)
            has_parse = bool(
                re.search(r"SMILES Parse Error|Failed parsing SMILES|could not parse", joined)
            )
            has_valence = bool(re.search(r"Explicit valence .* greater than permitted", joined))
            if chemistry_error_codes:
                summary["invalid_smiles_count"] += 1
                if summary["first_invalid_detail"] is None and texts:
                    summary["first_invalid_detail"] = texts[0]
            elif has_parse or has_valence:
                summary["invalid_smiles_count"] += 1
                if summary["first_invalid_detail"] is None and texts:
                    summary["first_invalid_detail"] = texts[0]
            if has_parse:
                summary["rdkit_parse_error_count"] += 1
            if has_valence:
                summary["rdkit_valence_error_count"] += 1
            if "kekulize_fail" in chemistry_error_codes:
                summary["kekulize_error_count"] += 1
            if "aromatic_non_ring" in chemistry_error_codes:
                summary["aromatic_non_ring_error_count"] += 1
            if "unclosed_ring" in chemistry_error_codes:
                summary["unclosed_ring_error_count"] += 1
            if summary["first_invalid_detail"] is None and texts:
                summary["first_invalid_detail"] = texts[0]
            elif summary["first_invalid_detail"] is None and reason:
                summary["first_invalid_detail"] = reason

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
        loop_start: Optional[float] = None,
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

        candidate, constraint_violation = self._prevalidate_candidate_against_constraints(
            state,
            candidate,
        )
        smiles = candidate.get("intermediate_smiles", "")
        if constraint_violation is not None:
            self.store.append_event(
                state.run_id,
                "mechanism_candidate_constraint_rejected",
                {
                    "attempt": state.step_index + 1,
                    "candidate_rank": candidate.get("rank"),
                    "candidate_smiles": smiles,
                    "details": dict(constraint_violation),
                },
                step_name="mechanism_step_proposal",
            )
            validation_payload = {
                "passed": False,
                "checks": [
                    {
                        "name": "proposal_constraints",
                        "passed": False,
                        "details": dict(constraint_violation),
                    }
                ],
            }
            return {
                "status": "failed",
                "branch_candidate": None,
                "last_validation": validation_payload,
                "reason": str(constraint_violation.get("reason") or "proposal_constraints"),
                "failed_checks": ["proposal_constraints"],
                "validation_signature": self._validation_signature(validation_payload),
                "candidate_rank": int(candidate.get("rank") or 0),
                "rescue_attempted": False,
                "rescue_outcome": "not_applicable",
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
        last_mechanism_output: Dict[str, Any] = {}

        max_retries = max(1, int(state.run_config.retry_same_candidate_max or 1))
        for retry_index in range(max_retries):
            self._trace(
                state,
                (
                    f"CANDIDATE rank={int(candidate.get('rank') or 0)} "
                    f"smiles={self._short_text(smiles, 64) or 'n/a'} "
                    f"smirks={self._short_text(candidate.get('reaction_smirks'), 120) or 'n/a'} "
                    f"retry_budget={retry_index + 1}/{max_retries}"
                ),
                step_name="mechanism_synthesis",
                attempt=state.step_index + 1,
                retry_index=retry_index,
            )
            self._mark_step_started(
                state,
                step_name="mechanism_synthesis",
                tool_name="predict_mechanistic_step",
                attempt=state.step_index + 1,
                retry_index=retry_index,
            )

            try:
                mechanism_result = self.mechanism_agent.run(
                    state,
                    scoped_output,
                    retry_feedback=retry_feedback,
                )
            except Exception as exc:
                self.store.append_event(
                    state.run_id,
                    "mechanism_candidate_execution_exception",
                    {
                        "attempt": state.step_index + 1,
                        "retry_index": retry_index,
                        "candidate_rank": candidate.get("rank"),
                        "candidate_smiles": smiles,
                        "error": str(exc),
                    },
                    step_name="mechanism_synthesis",
                )
                return {
                    "status": "failed",
                    "branch_candidate": None,
                    "last_validation": {},
                    "reason": "candidate_execution_exception",
                    "failed_checks": [],
                    "validation_signature": "",
                    "candidate_rank": int(candidate.get("rank") or 0),
                    "rescue_attempted": rescue_attempted,
                    "rescue_outcome": rescue_outcome,
                }
            mechanism_result.attempt = state.step_index + 1
            mechanism_result.retry_index = retry_index
            last_mechanism_output = dict(mechanism_result.output or {})
            try:
                mechanism_result.validation = validate_mechanism_step_output(
                    mechanism_result.output,
                    dbe_policy=state.run_config.dbe_policy,
                    enabled_validators=enabled_validators,
                    run_config=state.run_config,
                )
            except Exception as exc:
                self.store.append_event(
                    state.run_id,
                    "mechanism_validation_exception",
                    {
                        "attempt": state.step_index + 1,
                        "retry_index": retry_index,
                        "candidate_rank": candidate.get("rank"),
                        "candidate_smiles": smiles,
                        "error": str(exc),
                    },
                    step_name="mechanism_synthesis",
                )
                return {
                    "status": "failed",
                    "branch_candidate": None,
                    "last_validation": {},
                    "reason": "mechanism_validation_exception",
                    "failed_checks": [],
                    "validation_signature": "",
                    "candidate_rank": int(candidate.get("rank") or 0),
                    "rescue_attempted": rescue_attempted,
                    "rescue_outcome": rescue_outcome,
                    "mechanism_output": last_mechanism_output,
                }
            self._record_step(state, mechanism_result)
            self._record_validation_checks(state, mechanism_result=mechanism_result)

            validation_payload = mechanism_result.validation.as_dict()
            last_validation = validation_payload
            retry_feedback = self._retry_feedback_for_validation(validation_payload)
            last_failed_checks = list(retry_feedback.get("failed_checks", []))
            last_signature = self._validation_signature(validation_payload)
            repeated_signatures[last_signature] = repeated_signatures.get(last_signature, 0) + 1
            invalid_species_errors = [
                text
                for text in self._validation_error_strings(validation_payload)
                if "Invalid SMILES" in text or "Invalid SMILES strings" in text
            ]
            if invalid_species_errors:
                self.store.append_event(
                    state.run_id,
                    "invalid_species_in_candidate",
                    {
                        "attempt": state.step_index + 1,
                        "candidate_rank": candidate.get("rank"),
                        "candidate_smiles": smiles,
                        "errors": invalid_species_errors,
                    },
                    step_name="mechanism_synthesis",
                )

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

            rescue_result: Optional[StepResult] = None
            if self._runtime_budget_guard_triggered(
                state,
                loop_start=loop_start,
                step_name="candidate_rescue",
                fraction=0.08,
            ):
                rescue_outcome = "skipped_runtime_guard"
                self.store.append_event(
                    state.run_id,
                    "candidate_rescue_skipped_runtime_guard",
                    {
                        "attempt": state.step_index + 1,
                        "candidate_rank": candidate.get("rank"),
                    },
                    step_name="candidate_rescue",
                )
            else:
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
                error_code = str(rescue_output.get("error") or "")
                if error_code == "candidate_rescue_invalid_species":
                    rescue_outcome = "invalid_species"
                elif error_code == "candidate_rescue_exception":
                    rescue_outcome = "exception"
                if add_reactants or add_products:
                    rescue_outcome = "applied"
                    maybe_output = dict(mechanism_result.output or {})
                    base_current = [str(x) for x in maybe_output.get("current_state") or []]
                    base_resulting = [str(x) for x in maybe_output.get("resulting_state") or []]
                    # Reagents are consumed from the current side; byproducts are added to resulting side.
                    maybe_output["current_state"] = self._merge_unique_species(base_current, add_reactants)
                    maybe_output["resulting_state"] = self._merge_unique_species(base_resulting, add_products)
                    maybe_output["rescue_additions"] = {
                        "add_reactants": add_reactants,
                        "add_products": add_products,
                        "dbe_adjustment_hint": rescue_output.get("dbe_adjustment_hint"),
                    }
                    rescued_validation_result = validate_mechanism_step_output(
                        maybe_output,
                        dbe_policy=state.run_config.dbe_policy,
                        enabled_validators=enabled_validators,
                        run_config=state.run_config,
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
                            "mechanism_output": maybe_output,
                        }
                else:
                    rescue_outcome = rescue_outcome if rescue_outcome == "invalid_species" else "no_changes"

            self.store.append_event(
                state.run_id,
                "mechanism_retry_failed",
                {
                    "attempt": state.step_index + 1,
                    "retry_index": retry_index,
                    "candidate_rank": candidate.get("rank"),
                    "candidate_smiles": smiles,
                    "failed_checks": last_failed_checks,
                    "validator_hints": dict(retry_feedback.get("validator_hints", {})),
                    "validation_signature": last_signature,
                    "rescue_attempted": rescue_attempted,
                    "rescue_outcome": rescue_outcome,
                    "chemistry_error_codes": self._extract_chemistry_error_codes(
                        self._validation_error_strings(validation_payload)
                    ),
                    "validation": validation_payload,
                },
                step_name="mechanism_synthesis",
            )
            self._trace(
                state,
                (
                    f"RETRY_FAILED rank={int(candidate.get('rank') or 0)} "
                    f"failed_checks={','.join(last_failed_checks) or 'none'} "
                    f"signature={self._short_text(last_signature, 80) or 'n/a'} "
                    f"rescue={rescue_outcome}"
                ),
                step_name="mechanism_synthesis",
                attempt=state.step_index + 1,
                retry_index=retry_index,
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
                    "mechanism_output": last_mechanism_output,
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
                        "validator_hints": dict(retry_feedback.get("validator_hints", {})),
                    },
                    step_name="mechanism_synthesis",
                )
                self._trace(
                    state,
                    (
                        f"RETRY_NEXT rank={int(candidate.get('rank') or 0)} "
                        f"next_retry={retry_index + 1} "
                        f"failed_checks={','.join(last_failed_checks) or 'none'} "
                        f"guidance={self._short_text(retry_feedback.get('guidance'), 120) or 'n/a'}"
                    ),
                    step_name="mechanism_synthesis",
                    attempt=state.step_index + 1,
                    retry_index=retry_index + 1,
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
            "mechanism_output": last_mechanism_output,
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
        self._trace(
            state,
            (
                f"ACCEPT rank={candidate.rank} intermediate={self._short_text(candidate.intermediate_smiles, 64) or 'n/a'} "
                f"resulting={self._short_smiles_list(state.current_state)}"
            ),
            step_name="mechanism_synthesis",
            attempt=state.step_index,
            retry_index=0,
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
            self._trace(
                state,
                (
                    f"BACKTRACK reverted_to_step={bp.step_index} "
                    f"alt_rank={next_alt.rank} "
                    f"intermediate={self._short_text(next_alt.intermediate_smiles, 64) or 'n/a'} "
                    f"remaining_alternatives={len(bp.alternatives)}"
                ),
                step_name="mechanism_synthesis",
                attempt=bp.step_index + 1,
                retry_index=0,
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
        constraint_guidance = self._build_proposal_constraint_guidance(state)
        if constraint_guidance and isinstance(constraint_guidance, dict):
            merged = dict(template_guidance or {})
            merged.update(constraint_guidance)
            template_guidance = merged
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

    @staticmethod
    def _adaptive_enabled(state: RunState) -> bool:
        return str(state.run_config.adaptive_harness_mode or "off") == "conservative"

    @staticmethod
    def _adaptive_state(state: RunState) -> Dict[str, Any]:
        runtime = state.adaptive_runtime_state
        if not runtime:
            runtime.update(
                {
                    "mode": "standard",
                    "reason": None,
                    "activated_step": None,
                    "fallback_attempted_steps": [],
                    "failure_counts": {},
                }
            )
        return runtime

    def _activate_low_risk_mode(
        self,
        state: RunState,
        *,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        runtime = self._adaptive_state(state)
        if runtime.get("mode") == "low_risk":
            return
        runtime["mode"] = "low_risk"
        runtime["reason"] = reason
        runtime["activated_step"] = state.step_index + 1
        state.run_config.coordination_topology = "sas"
        state.run_config.candidate_rescue_enabled = False
        if state.template_guidance_state is not None:
            state.template_guidance_state = TemplateGuidanceState(
                mode="disabled",
                disable_reason=f"adaptive_low_risk:{reason}",
            )
            self._emit_template_guidance_state(state)
        self.store.append_event(
            state.run_id,
            "adaptive_harness_mode_changed",
            {
                "mode": "low_risk",
                "reason": reason,
                "activated_step": state.step_index + 1,
                "coordination_topology": "sas",
                "candidate_rescue_enabled": False,
                "step_mapping_enabled": state.run_config.step_mapping_enabled,
                "details": dict(details or {}),
            },
        )

    def _classify_failure_for_adaptation(
        self,
        *,
        state: RunState,
        proposal_output: Dict[str, Any],
        proposal_quality_summary: Dict[str, Any],
        all_candidates_rejected: bool,
        had_repeat_signature_failure: bool,
        last_failed_checks: List[str],
        last_rescue_outcome: str,
    ) -> Optional[str]:
        runtime = self._adaptive_state(state)
        failure_counts = runtime.setdefault("failure_counts", {})
        step_key = str(state.step_index + 1)
        per_step = failure_counts.setdefault(step_key, {})

        classification: Optional[str] = None
        if last_rescue_outcome == "invalid_species":
            classification = "rescue_invalid_species"
        elif had_repeat_signature_failure:
            classification = "repeat_signature_loop"
        elif proposal_quality_summary.get("candidate_count", 0) == 0 or bool(proposal_output.get("non_executable_fallback")):
            classification = "proposal_empty"
        elif proposal_quality_summary.get("all_candidates_invalid_smiles") or proposal_quality_summary.get("all_candidates_unassessable"):
            classification = "proposal_invalid_smiles"
        elif proposal_quality_summary.get("execution_exception_count", 0) > 0:
            classification = "candidate_execution_exception"
        elif (
            proposal_quality_summary.get("failed_candidate_count", 0) > 0
            and not proposal_quality_summary.get("invalid_smiles_count")
            and not proposal_quality_summary.get("incomplete_candidate_count")
            and not all_candidates_rejected
            and last_failed_checks
            and set(last_failed_checks) == {"atom_balance"}
        ):
            classification = "atom_balance_dead_end"

        if classification:
            per_step[classification] = int(per_step.get(classification, 0)) + 1
            self.store.append_event(
                state.run_id,
                "failure_classified",
                {
                    "step_index": state.step_index + 1,
                    "failure_class": classification,
                    "count_for_step": per_step[classification],
                },
                step_name="mechanism_step_proposal",
            )
        return classification

    def _fallback_candidate_from_remaining_mechanism(
        self,
        state: RunState,
        *,
        timeout: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        runner = BaselineRunner()
        result = runner.run_case(
            list(state.run_input.starting_materials),
            list(state.run_input.products),
            state.run_config.model,
            thinking_level=state.run_config.thinking_level,
            api_keys=state.run_config.api_keys,
            temperature_celsius=state.run_input.temperature_celsius,
            ph=state.run_input.ph,
            timeout=float(timeout or 90.0),
            current_state=list(state.current_state),
            accepted_path_summary=list(state.previous_intermediates),
        )
        if result.get("error"):
            self.store.append_event(
                state.run_id,
                "remaining_mechanism_fallback_failed",
                {
                    "step_index": state.step_index + 1,
                    "error": str(result.get("error") or ""),
                },
                step_name="mechanism_step_proposal",
            )
            return None
        raw_steps = list(result.get("raw_steps") or [])
        if not raw_steps:
            self.store.append_event(
                state.run_id,
                "remaining_mechanism_fallback_failed",
                {
                    "step_index": state.step_index + 1,
                    "error": "no_remaining_steps_returned",
                },
                step_name="mechanism_step_proposal",
            )
            return None
        first_step = dict(raw_steps[0] or {})
        resulting_state = [str(item) for item in first_step.get("resulting_state") or [] if str(item).strip()]
        predicted = str(first_step.get("predicted_intermediate") or "").strip()
        if not predicted and resulting_state:
            predicted = resulting_state[0]
        candidate = {
            "rank": 1,
            "intermediate_smiles": predicted,
            "reaction_description": str(first_step.get("step_label") or "remaining_mechanism_fallback"),
            "reaction_smirks": str(first_step.get("reaction_smirks") or ""),
            "electron_pushes": list(first_step.get("electron_pushes") or []),
            "resulting_state": resulting_state,
            "note": "Generated by remaining mechanism fallback.",
            "source": "remaining_mechanism_fallback",
        }
        self.store.append_event(
            state.run_id,
            "remaining_mechanism_fallback_generated",
            {
                "step_index": state.step_index + 1,
                "candidate_smiles": predicted,
                "resulting_state_size": len(resulting_state),
            },
            step_name="mechanism_step_proposal",
        )
        return candidate

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
                proposal_hints = dict(reproposal_hints.get(state.step_index + 1) or {})
                if self._adaptive_enabled(state) and self._adaptive_state(state).get("mode") == "low_risk":
                    proposal_hints.update(
                        {
                            "low_risk_mode": True,
                            "require_explicit_resulting_state": True,
                            "require_balance_consistency": True,
                            "max_candidates": 1,
                        }
                    )
                try:
                    proposal_output, candidates = self._propose_for_topology(
                        state,
                        harness,
                        proposal_hints=proposal_hints or None,
                    )
                except Exception as exc:
                    self.store.append_event(
                        state.run_id,
                        "proposal_dispatch_exception",
                        {
                            "step_index": state.step_index + 1,
                            "coordination_topology": state.run_config.coordination_topology,
                            "error": str(exc),
                        },
                        step_name="mechanism_step_proposal",
                    )
                    self.store.set_run_status(state.run_id, "failed")
                    self.store.append_event(
                        state.run_id,
                        "run_failed",
                        {
                            "reason": "proposal_dispatch_exception",
                            "step_index": state.step_index + 1,
                            "coordination_topology": state.run_config.coordination_topology,
                            "error": str(exc),
                        },
                    )
                    return
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
                    try:
                        attempt_result = self._try_candidate_with_retries(
                            state, candidate, proposal_output, enabled_validators=_ev, loop_start=start,
                        )
                    except Exception as exc:
                        attempt_result = {
                            "status": "failed",
                            "branch_candidate": None,
                            "last_validation": {},
                            "reason": "candidate_execution_exception",
                            "failed_checks": [],
                            "validation_signature": "",
                            "candidate_rank": int(candidate.get("rank") or 0),
                            "rescue_attempted": False,
                            "rescue_outcome": "none",
                        }
                        self.store.append_event(
                            state.run_id,
                            "mechanism_candidate_uncaught_exception",
                            {
                                "attempt": state.step_index + 1,
                                "candidate_rank": candidate.get("rank"),
                                "candidate_smiles": candidate.get("intermediate_smiles"),
                                "error": str(exc),
                            },
                            step_name="mechanism_synthesis",
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

                step_number = state.step_index + 1
                proposal_hint_payload = reproposal_hints.get(step_number, {}) or {}

                # Extract SMILES error context from validation failures for reproposal
                smiles_error_context = None
                if isinstance(proposal_quality_summary, dict):
                    first_invalid_detail = proposal_quality_summary.get("first_invalid_detail")
                    if first_invalid_detail and "rdkit_errors" in str(first_invalid_detail):
                        # Extract RDKit error messages from the validation details
                        import json
                        try:
                            if isinstance(first_invalid_detail, dict) and "rdkit_errors" in first_invalid_detail:
                                rdkit_errors = first_invalid_detail["rdkit_errors"]
                                if isinstance(rdkit_errors, dict) and rdkit_errors:
                                    error_lines = []
                                    for smiles, error in rdkit_errors.items():
                                        error_lines.append(f"{smiles}: {error}")
                                    smiles_error_context = "\n".join(error_lines)
                        except (json.JSONDecodeError, KeyError, TypeError):
                            pass  # Fall back to no error context if parsing fails

                failure_class: Optional[str] = None
                if not validated and self._adaptive_enabled(state):
                    runtime = self._adaptive_state(state)
                    failure_class = self._classify_failure_for_adaptation(
                        state=state,
                        proposal_output=proposal_output,
                        proposal_quality_summary=proposal_quality_summary,
                        all_candidates_rejected=all_candidates_rejected,
                        had_repeat_signature_failure=had_repeat_signature_failure,
                        last_failed_checks=last_failed_checks,
                        last_rescue_outcome=last_rescue_outcome,
                    )
                    should_activate_low_risk = False
                    if failure_class in {
                        "proposal_empty",
                        "proposal_invalid_smiles",
                        "repeat_signature_loop",
                        "rescue_invalid_species",
                        "candidate_execution_exception",
                    }:
                        should_activate_low_risk = True
                    elif failure_class == "atom_balance_dead_end":
                        per_step_failures = (
                            runtime.get("failure_counts", {})
                            .get(str(step_number), {})
                        )
                        atom_balance_failures = int(per_step_failures.get("atom_balance_dead_end", 0))
                        should_activate_low_risk = atom_balance_failures >= 2
                        if atom_balance_failures < 2 and runtime.get("mode") != "low_risk":
                            reproposal_hints[step_number] = {
                                **proposal_hint_payload,
                                "prior_failure_class": failure_class,
                                "require_explicit_resulting_state": True,
                                "require_balance_consistency": True,
                                "avoid_failed_checks": list(last_failed_checks),
                                "smiles_error_context": smiles_error_context,
                            }
                            self.store.append_event(
                                state.run_id,
                                "mechanism_reproposal_requested",
                                {
                                    "attempt": step_number,
                                    "reason": "atom_balance_dead_end",
                                    "reproposal_count": int(atom_balance_failures),
                                    "failed_checks": list(last_failed_checks),
                                },
                                step_name="mechanism_step_proposal",
                            )
                            continue

                    if should_activate_low_risk and runtime.get("mode") != "low_risk":
                        self._activate_low_risk_mode(
                            state,
                            reason=str(failure_class or "adaptive_low_risk"),
                            details={
                                "proposal_quality_summary": proposal_quality_summary,
                                "failed_checks": list(last_failed_checks),
                            },
                        )
                        reproposal_hints[step_number] = {
                            **proposal_hint_payload,
                            "low_risk_mode": True,
                            "prior_failure_class": failure_class,
                            "require_explicit_resulting_state": True,
                            "require_balance_consistency": True,
                            "max_candidates": 1,
                        }
                        continue

                    if runtime.get("mode") == "low_risk":
                        fallback_steps = runtime.setdefault("fallback_attempted_steps", [])
                        if step_number not in fallback_steps and failure_class:
                            fallback_steps.append(step_number)
                            fallback_candidate = self._fallback_candidate_from_remaining_mechanism(
                                state,
                                timeout=min(
                                    90.0,
                                    max(15.0, state.run_config.max_runtime_seconds / 3.0),
                                ),
                            )
                            if fallback_candidate is not None:
                                _ev = self._enabled_validators(harness) if harness else None
                                fallback_attempt = self._try_candidate_with_retries(
                                    state,
                                    fallback_candidate,
                                    {
                                        "classification": "intermediate_step",
                                        "analysis": "remaining_mechanism_fallback",
                                        "selected_candidate": fallback_candidate,
                                    },
                                    enabled_validators=_ev,
                                    loop_start=start,
                                )
                                candidate_attempts.append((dict(fallback_candidate), dict(fallback_attempt)))
                                if str(fallback_attempt.get("status") or "") == "validated":
                                    branch_candidate = fallback_attempt.get("branch_candidate")
                                    if isinstance(branch_candidate, BranchCandidate):
                                        validated.append(branch_candidate)
                                        reproposal_hints.pop(step_number, None)
                                else:
                                    maybe_validation = fallback_attempt.get("last_validation")
                                    if isinstance(maybe_validation, dict) and maybe_validation:
                                        last_failed_validation = maybe_validation
                                    last_failed_checks = list(fallback_attempt.get("failed_checks") or last_failed_checks)
                                    last_validation_signature = str(
                                        fallback_attempt.get("validation_signature") or last_validation_signature
                                    )
                                    last_candidate_rank = 1
                                    last_rescue_attempted = bool(fallback_attempt.get("rescue_attempted"))
                                    last_rescue_outcome = str(
                                        fallback_attempt.get("rescue_outcome") or last_rescue_outcome
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
                            "smiles_error_context": smiles_error_context,
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
                    if proposal_quality_summary.get("candidate_count", 0) == 0 and not all_candidates_rejected:
                        step_key = state.step_index + 1
                        reproposals_by_step[step_key] = reproposals_by_step.get(step_key, 0) + 1
                        current_reproposals = reproposals_by_step[step_key]
                        reproposal_hints[step_key] = {
                            "require_reaction_smirks": True,
                            "require_electron_pushes": True,
                            "smiles_error_context": smiles_error_context,
                            "non_executable_fallback": bool(proposal_output.get("non_executable_fallback")),
                        }
                        self.store.append_event(
                            state.run_id,
                            "mechanism_reproposal_requested",
                            {
                                "attempt": state.step_index + 1,
                                "reason": "incomplete_candidate_payload",
                                "candidate_count": 0,
                                "reproposal_count": current_reproposals,
                                "non_executable_fallback": bool(proposal_output.get("non_executable_fallback")),
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
                                    "candidate_count": 0,
                                    "non_executable_fallback": bool(proposal_output.get("non_executable_fallback")),
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
                                    "candidate_count": 0,
                                    "non_executable_fallback": bool(proposal_output.get("non_executable_fallback")),
                                    "proposal_quality_summary": proposal_quality_summary,
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
                            "smiles_error_context": smiles_error_context,
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
                    # ── Deferred atom-balance soft-advance (unverified only) ─────
                    if state.mode == "unverified":
                        balance_pending_candidate = self._best_balance_pending_candidate(
                            candidate_attempts=candidate_attempts,
                        )
                        if balance_pending_candidate is not None:
                            self.store.append_event(
                                state.run_id,
                                "mechanism_step_soft_advance",
                                {
                                    "step_index": state.step_index + 1,
                                    "reason": "balance_pending",
                                    "failed_checks": last_failed_checks,
                                    "soft_intermediate_smiles": balance_pending_candidate.intermediate_smiles,
                                    "soft_resulting_state": list(balance_pending_candidate.resulting_state),
                                    "balance_pending_validation": balance_pending_candidate.validation_summary,
                                },
                                step_name="mechanism_synthesis",
                            )
                            soft_validation = StepValidationResult(
                                checks=[
                                    StepValidationCheck(
                                        name="soft_advance",
                                        passed=False,
                                        details={
                                            "reason": "balance_pending",
                                            "failed_checks": list(last_failed_checks),
                                            "validation": dict(balance_pending_candidate.validation_summary or {}),
                                        },
                                    )
                                ]
                            )
                            soft_step = StepResult(
                                step_name="mechanism_synthesis",
                                tool_name="predict_mechanistic_step",
                                output={
                                    **(balance_pending_candidate.mechanism_output or {}),
                                    "soft_advance": True,
                                    "soft_advance_reason": "balance_pending",
                                    "failed_checks": list(last_failed_checks),
                                    "balance_pending_validation": dict(balance_pending_candidate.validation_summary or {}),
                                },
                                attempt=state.step_index + 1,
                                retry_index=0,
                                source="deterministic",
                                validation=soft_validation,
                            )
                            self._record_step(state, soft_step)
                            chosen = balance_pending_candidate
                            validated = [chosen]

                    # ── Proceed-on-failure: generic soft-advance when configured ──
                    # When proceed_on_validation_failure is enabled, instead of
                    # pausing/failing we accept the best available candidate as a
                    # "soft" step (validation_passed = False, marked for post-loop
                    # re-evaluation) and continue the harness.
                    if not validated and state.mode != "verified" and state.run_config.proceed_on_validation_failure:
                        can_proceed = True
                        if state.run_config.proceed_only_on_arrow_push_failure:
                            can_proceed = self._is_arrow_push_only_failure(
                                failed_checks=last_failed_checks,
                                last_validation=last_failed_validation,
                                incomplete_reasons=incomplete_reasons,
                            )
                        if can_proceed:
                            soft_candidate = self._best_soft_candidate(
                                candidates=candidates,
                                candidate_attempts=candidate_attempts,
                                proposal_output=proposal_output,
                                soft_reason="proceed_on_validation_failure",
                            )
                            if soft_candidate is not None:
                                self.store.append_event(
                                    state.run_id,
                                    "mechanism_step_soft_advance",
                                    {
                                        "step_index": state.step_index + 1,
                                        "reason": "proceed_on_validation_failure",
                                        "only_arrow_push": state.run_config.proceed_only_on_arrow_push_failure,
                                        "failed_checks": last_failed_checks,
                                        "incomplete_reasons": list(incomplete_reasons),
                                        "soft_intermediate_smiles": soft_candidate.intermediate_smiles,
                                        "soft_resulting_state": list(soft_candidate.resulting_state),
                                    },
                                    step_name="mechanism_synthesis",
                                )
                                # Persist a step_output row for this soft step so
                                # it appears in step_outputs and the post-loop
                                # re-evaluation phase can find it.
                                soft_validation = StepValidationResult(
                                    checks=[
                                        StepValidationCheck(
                                            name="soft_advance",
                                            passed=False,
                                            details={
                                                "reason": "proceed_on_validation_failure",
                                                "failed_checks": list(last_failed_checks),
                                                "incomplete_reasons": list(incomplete_reasons),
                                            },
                                        )
                                    ]
                                )
                                soft_step = StepResult(
                                    step_name="mechanism_synthesis",
                                    tool_name="predict_mechanistic_step",
                                    output={
                                        **(soft_candidate.mechanism_output or {}),
                                        "soft_advance": True,
                                        "soft_advance_reason": "proceed_on_validation_failure",
                                        "failed_checks": list(last_failed_checks),
                                        "incomplete_reasons": list(incomplete_reasons),
                                    },
                                    attempt=state.step_index + 1,
                                    retry_index=0,
                                    source="deterministic",
                                    validation=soft_validation,
                                )
                                self._record_step(state, soft_step)
                                chosen = soft_candidate
                                validated = [chosen]
                                # Fall through to Steps D-G with soft candidate.
                                # The post-loop re-evaluation phase will attempt to
                                # fix these steps after the run completes.
                    if not validated:
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
            self._run_post_step_modules(state, chosen, harness, loop_start=start)

            # --- Step G: Completion check ---
            contains_target = bool(chosen.mechanism_output.get("contains_target_product"))
            self.store.append_event(
                state.run_id,
                "completion_check",
                {
                    "step_index": state.step_index,
                    "contains_target_product": contains_target,
                    "validation_passed": not bool(chosen.mechanism_output.get("soft_advance")),
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

    # ── Post-loop phase ──────────────────────────────────────────────────

    def _run_post_loop_phase(
        self,
        state: RunState,
        harness: Optional[HarnessConfig] = None,
    ) -> None:
        """Execute post_loop modules after the mechanism loop finishes.

        Currently dispatches the ``past_failure_reevaluation`` module, which
        attempts to validate/fix soft-accepted (proceed-on-failure) steps using
        the context of later successful steps.
        """
        if harness is None:
            return
        enabled_post_loop = harness.enabled_post_loop()
        if not enabled_post_loop:
            return

        self.store.append_event(
            state.run_id,
            "post_loop_phase_started",
            {"module_count": len(enabled_post_loop)},
        )

        for module in enabled_post_loop:
            if module.id == "past_failure_reevaluation":
                self._run_past_failure_reevaluation(state)
            elif module.custom:
                context: Dict[str, Any] = {}
                result = self._run_custom_module(state, module, context)
                result.attempt = 0
                self._record_step(state, result)

        self.store.append_event(
            state.run_id,
            "post_loop_phase_completed",
            {"module_count": len(enabled_post_loop)},
        )

    def _accepted_mechanism_rows(self, run_id: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for row in self.store.list_step_outputs(run_id):
            if row.get("step_name") != "mechanism_synthesis":
                continue
            output = row.get("output")
            if not isinstance(output, dict):
                continue
            validation = row.get("validation") if isinstance(row.get("validation"), dict) else {}
            if bool(validation.get("passed")):
                rows.append(row)
                continue
            if bool(output.get("soft_advance")) and str(output.get("soft_advance_reason") or "") == "balance_pending":
                rows.append(row)
        rows.sort(key=lambda item: (int(item.get("attempt") or 0), int(item.get("retry_index") or 0)))
        return rows

    def _run_overall_balance_reconciliation(self, state: RunState) -> None:
        from mechanistic_agent.balance import assess_balance_diagnostics

        accepted_rows = self._accepted_mechanism_rows(state.run_id)
        final_state = list(state.current_state or [])
        if accepted_rows:
            final_output = accepted_rows[-1].get("output") if isinstance(accepted_rows[-1].get("output"), dict) else {}
            if isinstance(final_output, dict):
                resulting_state = [str(item) for item in final_output.get("resulting_state") or []]
                if resulting_state:
                    final_state = resulting_state

        left_species = [str(item) for item in state.run_input.starting_materials or []]
        right_species = list(final_state)
        for row in accepted_rows:
            output = row.get("output") if isinstance(row.get("output"), dict) else {}
            if not isinstance(output, dict):
                continue
            rescue_additions = output.get("rescue_additions") if isinstance(output.get("rescue_additions"), dict) else {}
            add_reactants = [str(item) for item in rescue_additions.get("add_reactants") or []]
            add_products = [str(item) for item in rescue_additions.get("add_products") or []]
            left_species = self._merge_unique_species(left_species, add_reactants)
            right_species = self._merge_unique_species(right_species, add_products)

        initial_diagnostics = assess_balance_diagnostics(
            left_species,
            right_species,
            include_hydrogens=True,
            backend_config=state.run_config,
            left_label="overall_left",
            right_label="overall_right",
        )

        add_reactants: List[str] = []
        add_products: List[str] = []
        reconciliation_step_output: Dict[str, Any] = {
            "status": "not_needed" if initial_diagnostics.get("balanced") else "pending",
            "accepted_step_count": len(accepted_rows),
            "initial_left_species": list(left_species),
            "initial_right_species": list(right_species),
            "initial_balance": initial_diagnostics,
        }
        final_diagnostics = dict(initial_diagnostics)
        grade = "exact" if bool(initial_diagnostics.get("balanced")) else str(initial_diagnostics.get("classification") or "approximate")

        if (
            state.mode == "unverified"
            and not bool(initial_diagnostics.get("balanced"))
            and str(initial_diagnostics.get("classification") or "") != "invalid_species"
        ):
            self.store.append_event(
                state.run_id,
                "overall_balance_reconciliation_started",
                {
                    "accepted_step_count": len(accepted_rows),
                    "initial_balance": initial_diagnostics,
                },
                step_name="overall_balance_reconciliation",
            )
            rescue_output = self.missing_reagents_agent.executor.run_missing_reagents(
                starting=left_species,
                products=right_species,
                conditions_guidance={
                    "rescue_mode": True,
                    "overall_balance_reconciliation": True,
                    "accepted_step_count": len(accepted_rows),
                    "final_state": final_state,
                },
            )
            add_reactants = [str(item) for item in rescue_output.get("missing_reactants") or rescue_output.get("suggested_reactants") or []]
            add_products = [str(item) for item in rescue_output.get("missing_products") or rescue_output.get("suggested_products") or []]
            reconciled_left = self._merge_unique_species(left_species, add_reactants)
            reconciled_right = self._merge_unique_species(right_species, add_products)
            final_diagnostics = assess_balance_diagnostics(
                reconciled_left,
                reconciled_right,
                include_hydrogens=True,
                backend_config=state.run_config,
                left_label="overall_left",
                right_label="overall_right",
            )
            reconciliation_step_output = {
                **reconciliation_step_output,
                **rescue_output,
                "status": str(rescue_output.get("status") or "completed"),
                "add_reactants": add_reactants,
                "add_products": add_products,
                "reconciled_left_species": reconciled_left,
                "reconciled_right_species": reconciled_right,
                "final_balance": final_diagnostics,
            }
            if bool(final_diagnostics.get("balanced")) and (add_reactants or add_products):
                grade = "reconciled"
            elif str(final_diagnostics.get("classification") or "") == "invalid_species":
                grade = "invalid_species"
            else:
                grade = "approximate"

            reconciliation_step = StepResult(
                step_name="overall_balance_reconciliation",
                tool_name="predict_missing_reagents",
                output=reconciliation_step_output,
                source="llm",
            )
            self._record_step(state, reconciliation_step)
        elif str(initial_diagnostics.get("classification") or "") == "invalid_species":
            grade = "invalid_species"
        elif not bool(initial_diagnostics.get("balanced")):
            grade = "approximate"

        overall_balance = {
            "grade": grade,
            "balanced": bool(final_diagnostics.get("balanced")),
            "accepted_step_count": len(accepted_rows),
            "initial_balance": initial_diagnostics,
            "final_balance": final_diagnostics,
            "add_reactants": add_reactants,
            "add_products": add_products,
            "final_state": final_state,
        }
        self.store.append_event(
            state.run_id,
            "overall_balance_reconciled",
            overall_balance,
            step_name="overall_balance_reconciliation",
        )

    def _run_past_failure_reevaluation(self, state: RunState) -> None:
        """Re-evaluate soft-accepted mechanism steps after the loop completes.

        For each step_output row where ``output.soft_advance == True``, this
        method attempts to re-run the deterministic validators (atom_balance,
        state_progress) with any corrections that can be inferred from the
        subsequent accepted steps. If the step now passes validation it is
        marked with ``reevaluated_passed = True`` in its output so the
        completion logic can count it.

        Algorithm
        ---------
        1. Collect all soft-advance rows from step_outputs in attempt order.
        2. For each soft row, build a corrected ``resulting_state`` from the
           *next* validated step's ``current_state`` snapshot (i.e. what the
           harness had when the following step started).  If the atom balance
           passes with that state, mark the step corrected.
        3. Persist an updated step_output row and an event for observability.
        """
        step_outputs = self.store.list_step_outputs(state.run_id)
        soft_rows = [
            row for row in step_outputs
            if row.get("step_name") == "mechanism_synthesis"
            and isinstance(row.get("output"), dict)
            and bool((row.get("output") or {}).get("soft_advance"))
        ]
        if not soft_rows:
            return

        soft_rows.sort(key=lambda r: (int(r.get("attempt") or 0), int(r.get("retry_index") or 0)))

        # Build a list of confirmed mechanism states (from fully-validated steps).
        validated_rows = [
            row for row in step_outputs
            if row.get("step_name") == "mechanism_synthesis"
            and isinstance(row.get("validation"), dict)
            and bool((row.get("validation") or {}).get("passed"))
        ]
        validated_rows.sort(
            key=lambda r: (int(r.get("attempt") or 0), int(r.get("retry_index") or 0))
        )

        self.store.append_event(
            state.run_id,
            "past_failure_reevaluation_started",
            {
                "soft_step_count": len(soft_rows),
                "validated_step_count": len(validated_rows),
            },
            step_name="past_failure_reevaluation",
        )

        fixed_count = 0
        for soft_row in soft_rows:
            soft_attempt = int(soft_row.get("attempt") or 0)
            soft_output = dict(soft_row.get("output") or {})
            soft_smiles = str(soft_output.get("intermediate_smiles") or "").strip()
            soft_resulting = list(soft_output.get("resulting_state") or [])

            # Strategy: find the next validated step that has a higher attempt index.
            # Use the current_state of that next step as the corrected resulting_state.
            next_validated = next(
                (
                    r for r in validated_rows
                    if int(r.get("attempt") or 0) > soft_attempt
                ),
                None,
            )

            corrected_resulting: Optional[List[str]] = None
            if next_validated is not None:
                next_output = dict(next_validated.get("output") or {})
                prev_state = list(next_output.get("current_state") or [])
                if prev_state:
                    corrected_resulting = prev_state

            candidate_resulting = corrected_resulting or soft_resulting
            if not candidate_resulting:
                continue

            # Re-run atom_balance and state_progress validators.
            current_state_for_step = list(
                soft_output.get("current_state") or state.run_input.starting_materials or []
            )
            validation_payload: Dict[str, Any] = {
                "current_state": current_state_for_step,
                "intermediate_smiles": soft_smiles,
                "resulting_state": candidate_resulting,
                "reaction_smirks": "",
                "electron_pushes": [],
            }
            validation_result = validate_mechanism_step_output(
                validation_payload,
                dbe_policy="soft",
                enabled_validators={"atom_balance_validation", "state_progress_validation"},
                run_config=state.run_config,
            )
            passed = validation_result.passed if validation_result else False

            # Update the soft step output with reevaluation result.
            updated_output = {
                **soft_output,
                "reevaluated": True,
                "reevaluated_passed": passed,
                "reevaluated_resulting_state": candidate_resulting,
                "reevaluated_from_next_step_attempt": int(next_validated.get("attempt") or 0)
                if next_validated
                else None,
            }
            if passed:
                updated_output["soft_advance"] = False
                fixed_count += 1

            self.store.append_event(
                state.run_id,
                "past_failure_step_reevaluated",
                {
                    "soft_attempt": soft_attempt,
                    "reevaluated_passed": passed,
                    "corrected_resulting_state": candidate_resulting,
                    "original_soft_smiles": soft_smiles,
                },
                step_name="past_failure_reevaluation",
            )

            # Persist updated step_output using the store's upsert method.
            self.store.upsert_step_output(
                run_id=state.run_id,
                step_name="mechanism_synthesis",
                attempt=soft_attempt,
                retry_index=int(soft_row.get("retry_index") or 0),
                output=updated_output,
                validation={"passed": passed, "soft_advance_reevaluated": True},
                source="deterministic",
            )

        self.store.append_event(
            state.run_id,
            "past_failure_reevaluation_completed",
            {
                "soft_step_count": len(soft_rows),
                "fixed_count": fixed_count,
                "remaining_unfixed": len(soft_rows) - fixed_count,
            },
            step_name="past_failure_reevaluation",
        )

    def _run_post_step_modules(
        self,
        state: RunState,
        chosen: BranchCandidate,
        harness: Optional[HarnessConfig] = None,
        loop_start: Optional[float] = None,
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
                    if self._runtime_budget_guard_triggered(
                        state,
                        loop_start=loop_start,
                        step_name="step_atom_mapping",
                        fraction=0.10,
                    ):
                        self.store.append_event(
                            state.run_id,
                            "step_mapping_skipped_runtime_guard",
                            {"step_index": state.step_index},
                            step_name="step_atom_mapping",
                        )
                        continue
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
                if self._runtime_budget_guard_triggered(
                    state,
                    loop_start=loop_start,
                    step_name="step_atom_mapping",
                    fraction=0.10,
                ):
                    self.store.append_event(
                        state.run_id,
                        "step_mapping_skipped_runtime_guard",
                        {"step_index": state.step_index},
                        step_name="step_atom_mapping",
                    )
                    return
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
                "current_state": list((step_mapping.output or {}).get("current_state") or mapping_current),
                "resulting_state": list((step_mapping.output or {}).get("resulting_state") or mapping_resulting),
                "mapped_atoms": compact[:12],
                "species_lineage_summary": list((step_mapping.output or {}).get("species_lineage_summary") or [])[:8],
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
        self._trace(
            state,
            (
                f"RUN_START mode={state.mode} topology={state.run_config.coordination_topology} "
                f"starting={self._short_smiles_list(state.run_input.starting_materials)} "
                f"products={self._short_smiles_list(state.run_input.products)}"
            ),
        )
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

            if not state.paused:
                # Run post-loop phase (e.g. past-failure re-evaluation) before
                # making final pass/fail judgement.
                self._run_post_loop_phase(state, harness)
                if not state.paused and state.mode == "unverified":
                    self._run_overall_balance_reconciliation(state)

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
            all_accepted_steps = self._accepted_mechanism_rows(run_id)

            if not all_accepted_steps and not mechanism_steps:
                # No valid steps at all — check if we have only soft steps to at
                # least count the attempt for diagnostics.
                soft_only = [
                    row
                    for row in step_outputs
                    if row.get("step_name") == "mechanism_synthesis"
                    and isinstance(row.get("output"), dict)
                    and bool((row.get("output") or {}).get("soft_advance"))
                ]
                if not soft_only:
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
                # Has only unvalidated soft steps — still treat as failed but
                # include count for diagnostics.
                self.store.set_run_status(run_id, "failed")
                self.store.append_event(
                    run_id,
                    "run_failed",
                    {
                        "reason": "no_validated_mechanism_steps",
                        "soft_advance_steps": len(soft_only),
                    },
                )
                return

            if not mechanism_steps and not all_accepted_steps:
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
                for row in all_accepted_steps
            )
            if has_completion:
                self.store.set_run_status(run_id, "completed")
                self.store.append_event(
                    run_id,
                    "run_completed",
                    {
                        "mode": state.mode,
                        "mechanism_steps": len(mechanism_steps),
                        "validated_only": len(all_accepted_steps) == len(mechanism_steps),
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
                {
                    "reason": "uncaught_exception",
                    "error": str(exc),
                    "step_index": state.step_index + 1,
                    "phase": "coordinator_execute",
                },
            )
        finally:
            try:
                terminal_row = self.store.get_run_row(run_id)
                terminal_status = str((terminal_row or {}).get("status") or "")
                self._trace(
                    state,
                    f"RUN_END status={terminal_status or 'unknown'} final_state={self._short_smiles_list(state.current_state)}",
                )
                if terminal_status in {"completed", "failed", "stopped"}:
                    self._emit_chemistry_backend_summary_event(state)
            except Exception:
                # Telemetry summary must never break run completion/failure handling.
                pass
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
