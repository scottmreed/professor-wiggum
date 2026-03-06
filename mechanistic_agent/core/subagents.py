"""Subagent modules for the explicit mechanistic runtime state machine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .tool_executor import ToolExecutor
from .types import RunState, StepResult


def _extract_step_cost(
    output: Dict[str, Any],
    model: Optional[str],
) -> Tuple[Optional[Dict[str, int]], Optional[Dict[str, float]]]:
    """Pop ``_llm_usage`` from *output*, normalise, and compute cost."""
    raw_usage = output.pop("_llm_usage", None)
    if not raw_usage or not isinstance(raw_usage, dict):
        return None, None
    from mechanistic_agent.model_registry import calculate_cost, normalise_token_usage

    usage = normalise_token_usage(raw_usage)
    cost = None
    if model:
        try:
            cost = calculate_cost(model, usage)
        except Exception:
            pass
    return usage, cost


@dataclass(slots=True)
class BalanceAgent:
    executor: ToolExecutor

    def run(self, state: RunState) -> StepResult:
        output = self.executor.run_balance(state.run_input.starting_materials, state.run_input.products)
        return StepResult(
            step_name="balance_analysis",
            tool_name="analyse_balance",
            output=output,
            source="deterministic",
        )


@dataclass(slots=True)
class ConditionsAgent:
    executor: ToolExecutor

    def run(self, state: RunState) -> List[StepResult]:
        results: List[StepResult] = []
        ph_output = self.executor.run_ph_recommendation(
            state.run_input.starting_materials,
            state.run_input.products,
            state.run_input.ph,
        )
        results.append(
            StepResult(
                step_name="ph_recommendation",
                tool_name="recommend_ph",
                output=ph_output,
                source="deterministic",
            )
        )

        conditions_output = self.executor.run_conditions(
            state.run_input.starting_materials,
            state.run_input.products,
            state.run_input.ph,
        )
        model = state.run_config.step_models.get("initial_conditions", state.run_config.model)
        usage, cost = _extract_step_cost(conditions_output, model)
        results.append(
            StepResult(
                step_name="initial_conditions",
                tool_name="assess_initial_conditions",
                output=conditions_output,
                source="llm",
                token_usage=usage,
                cost=cost,
            )
        )
        return results


@dataclass(slots=True)
class FunctionalGroupsAgent:
    executor: ToolExecutor

    def run(self, state: RunState) -> StepResult:
        smiles = list(state.run_input.starting_materials) + list(state.run_input.products)
        output = self.executor.run_functional_groups(smiles)
        return StepResult(
            step_name="functional_groups",
            tool_name="fingerprint_functional_groups",
            output=output,
            source="deterministic",
        )


@dataclass(slots=True)
class MissingReagentsAgent:
    executor: ToolExecutor

    def run(self, state: RunState, conditions_output: Optional[Dict[str, Any]]) -> StepResult:
        output = self.executor.run_missing_reagents(
            starting=state.run_input.starting_materials,
            products=state.run_input.products,
            conditions_guidance=conditions_output,
        )
        model = state.run_config.step_models.get("missing_reagents", state.run_config.model)
        usage, cost = _extract_step_cost(output, model)
        return StepResult(
            step_name="missing_reagents",
            tool_name="predict_missing_reagents",
            output=output,
            source="llm",
            token_usage=usage,
            cost=cost,
        )

    def rescue_candidate(
        self,
        state: RunState,
        *,
        current_state: List[str],
        resulting_state: List[str],
        failed_checks: Optional[List[str]] = None,
        validation_details: Optional[Dict[str, Any]] = None,
    ) -> StepResult:
        output = self.executor.run_candidate_rescue(
            current_state=current_state,
            resulting_state=resulting_state,
            failed_checks=failed_checks,
            validation_details=validation_details,
        )
        model = state.run_config.step_models.get("missing_reagents", state.run_config.model)
        usage, cost = _extract_step_cost(output, model)
        return StepResult(
            step_name="candidate_rescue",
            tool_name="predict_missing_reagents_for_candidate",
            output=output,
            source="llm",
            token_usage=usage,
            cost=cost,
        )


@dataclass(slots=True)
class MappingAgent:
    executor: ToolExecutor

    def run(self, state: RunState) -> StepResult:
        output = self.executor.run_mapping(state.run_input.starting_materials, state.run_input.products)
        model = state.run_config.step_models.get("atom_mapping", state.run_config.model)
        usage, cost = _extract_step_cost(output, model)
        return StepResult(
            step_name="atom_mapping",
            tool_name="attempt_atom_mapping",
            output=output,
            source="llm",
            token_usage=usage,
            cost=cost,
        )

    def run_step_mapping(self, state: RunState, *, current_state: List[str], resulting_state: List[str]) -> StepResult:
        output = self.executor.run_step_mapping(
            current_state=current_state,
            resulting_state=resulting_state,
        )
        model = state.run_config.step_models.get("atom_mapping", state.run_config.model)
        usage, cost = _extract_step_cost(output, model)
        return StepResult(
            step_name="step_atom_mapping",
            tool_name="attempt_atom_mapping_for_step",
            output=output,
            source="llm",
            token_usage=usage,
            cost=cost,
        )


@dataclass(slots=True)
class ReactionTypeAgent:
    executor: ToolExecutor

    def run(
        self,
        state: RunState,
        *,
        balance_analysis: Optional[Dict[str, Any]] = None,
        functional_groups: Optional[Dict[str, Any]] = None,
        ph_recommendation: Optional[Dict[str, Any]] = None,
        initial_conditions: Optional[Dict[str, Any]] = None,
        missing_reagents: Optional[Dict[str, Any]] = None,
        atom_mapping: Optional[Dict[str, Any]] = None,
    ) -> StepResult:
        output = self.executor.run_reaction_type_mapping(
            starting=state.run_input.starting_materials,
            products=state.run_input.products,
            balance_analysis=balance_analysis,
            functional_groups=functional_groups,
            ph_recommendation=ph_recommendation,
            initial_conditions=initial_conditions,
            missing_reagents=missing_reagents,
            atom_mapping=atom_mapping,
        )
        model = state.run_config.step_models.get("reaction_type_mapping", state.run_config.model)
        usage, cost = _extract_step_cost(output, model)
        return StepResult(
            step_name="reaction_type_mapping",
            tool_name="select_reaction_type",
            output=output,
            source="llm",
            token_usage=usage,
            cost=cost,
        )


@dataclass(slots=True)
class IntermediateAgent:
    executor: ToolExecutor

    def run(
        self,
        state: RunState,
        *,
        template_guidance: Optional[Dict[str, Any]] = None,
    ) -> StepResult:
        output = self.executor.run_intermediates(
            starting=state.run_input.starting_materials,
            products=state.run_input.products,
            current_state=state.current_state,
            previous_intermediates=state.previous_intermediates,
            ph=state.run_input.ph,
            temperature=state.run_input.temperature_celsius,
            step_index=state.step_index,
            step_mapping_context=state.latest_step_mapping,
            template_guidance=template_guidance,
        )
        model = state.run_config.step_models.get("intermediates", state.run_config.model)
        usage, cost = _extract_step_cost(output, model)
        return StepResult(
            step_name="mechanism_step_proposal",
            tool_name="propose_mechanism_step",
            output=output,
            source="llm",
            token_usage=usage,
            cost=cost,
        )


@dataclass(slots=True)
class MechanismAgent:
    executor: ToolExecutor

    def run(
        self,
        state: RunState,
        intermediate_output: Optional[Dict[str, Any]],
        *,
        retry_feedback: Optional[Dict[str, Any]] = None,
    ) -> StepResult:
        if intermediate_output is None:
            intermediate_output = {}

        selected_candidate: Dict[str, Any] = {}
        maybe_selected = intermediate_output.get("selected_candidate")
        if isinstance(maybe_selected, dict):
            selected_candidate = maybe_selected

        candidates: List[str] = []
        if isinstance(intermediate_output.get("proposed_intermediates"), list):
            candidates = [
                str(item)
                for item in intermediate_output["proposed_intermediates"]
                if isinstance(item, str) and str(item).strip()
            ]

        if not candidates and isinstance(intermediate_output.get("intermediates"), list):
            for item in intermediate_output["intermediates"]:
                if isinstance(item, dict) and item.get("smiles"):
                    candidates.append(str(item["smiles"]))

        predicted: Optional[str] = None
        if isinstance(selected_candidate.get("intermediate_smiles"), str):
            text = str(selected_candidate.get("intermediate_smiles") or "").strip()
            if text:
                predicted = text
        if not predicted:
            predicted = candidates[0] if candidates else None

        resulting_state: Optional[List[str]] = None
        if isinstance(selected_candidate.get("resulting_state"), list):
            resulting_state = [
                str(item) for item in selected_candidate["resulting_state"] if str(item).strip()
            ]
        elif predicted:
            resulting_state = list(state.current_state)
            if predicted not in resulting_state:
                resulting_state.append(predicted)

        electron_pushes: Optional[List[Dict[str, object]]] = None
        raw_pushes = selected_candidate.get("electron_pushes")
        if isinstance(raw_pushes, list):
            parsed_pushes: List[Dict[str, object]] = []
            for item in raw_pushes:
                if isinstance(item, dict):
                    parsed_pushes.append(dict(item))
            if parsed_pushes:
                electron_pushes = parsed_pushes

        reaction_smirks: Optional[str] = None
        raw_smirks = selected_candidate.get("reaction_smirks")
        if isinstance(raw_smirks, str) and raw_smirks.strip():
            reaction_smirks = raw_smirks.strip()

        note = intermediate_output.get("llm_reasoning") or intermediate_output.get("message")
        candidate_note = selected_candidate.get("note")
        if isinstance(candidate_note, str) and candidate_note.strip():
            if isinstance(note, str) and note.strip():
                note = f"{note.strip()} | Candidate note: {candidate_note.strip()}"
            else:
                note = candidate_note.strip()

        if retry_feedback:
            failure_bits = retry_feedback.get("failed_checks")
            guidance = retry_feedback.get("guidance")
            note_parts: List[str] = []
            if isinstance(note, str) and note.strip():
                note_parts.append(note.strip())
            if isinstance(failure_bits, list):
                note_parts.append(f"Retry context failed checks: {', '.join(str(x) for x in failure_bits)}")
            if isinstance(guidance, str) and guidance.strip():
                note_parts.append(guidance.strip())
            note = " | ".join(note_parts) if note_parts else note

        output = self.executor.run_mechanism_step(
            step_index=state.step_index + 1,
            current_state=state.current_state,
            target_products=state.run_input.products,
            predicted_intermediate=predicted,
            resulting_state=resulting_state,
            electron_pushes=electron_pushes,
            reaction_smirks=reaction_smirks,
            previous_intermediates=state.previous_intermediates,
            starting_materials=state.run_input.starting_materials,
            note=str(note) if note else None,
        )

        return StepResult(
            step_name="mechanism_synthesis",
            tool_name="predict_mechanistic_step",
            output=output,
            attempt=state.step_index + 1,
            source="llm",
        )


@dataclass(slots=True)
class ReflectionAgent:
    """Collects critique signals without mutating prompts or memory automatically."""

    def run(self, state: RunState, latest_output: Dict[str, Any]) -> StepResult:
        critique: Dict[str, Any] = {
            "run_mode": state.mode,
            "step_index": state.step_index,
            "contains_target_product": latest_output.get("contains_target_product"),
            "warnings": [],
        }
        if latest_output.get("unchanged_starting_materials_detected"):
            critique["warnings"].append("step_returned_unchanged_state")
        if latest_output.get("bond_electron_validation", {}).get("valid") is False:
            critique["warnings"].append("missing_or_invalid_dbe_metadata")

        return StepResult(
            step_name="reflection",
            tool_name="reflection_agent",
            output=critique,
            attempt=state.step_index + 1,
            source="deterministic",
        )
