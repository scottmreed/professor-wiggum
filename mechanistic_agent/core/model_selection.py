"""Exact-model selection for local runtime runs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional

from mechanistic_agent.config import LLM_STEP_KEYS
from mechanistic_agent.model_registry import (
    get_model_family,
    get_reasoning_levels,
    model_supports_tools,
    resolve_model_key,
    to_internal_reasoning_level,
    to_public_reasoning_level,
)

ModelFamily = Literal["openai", "claude", "gemini", "olmo", "minimax"]
ThinkingLevel = Literal["low", "high"]

# Steps that require native tool-calling support.
_TOOL_REQUIRED_STEPS: set[str] = set()


@dataclass(slots=True)
class ModelSelectionResult:
    family: ModelFamily
    model_name: str
    thinking_level: Optional[ThinkingLevel]
    step_models: Dict[str, str]
    step_reasoning: Dict[str, str]
    notes: Dict[str, str]


def _enabled_steps(
    *,
    functional_groups_enabled: bool,
    intermediate_prediction_enabled: bool,
    optional_llm_tools: Iterable[str],
) -> List[str]:
    enabled = set(LLM_STEP_KEYS)
    optional = set(optional_llm_tools)
    if not functional_groups_enabled:
        enabled.discard("functional_groups")
    if not intermediate_prediction_enabled:
        enabled.discard("intermediates")
    if "predict_missing_reagents" not in optional:
        enabled.discard("missing_reagents")
    if "attempt_atom_mapping" not in optional:
        enabled.discard("atom_mapping")
    return sorted(enabled)


def _validate_model(model_name: str) -> str:
    return resolve_model_key(model_name)


def _resolve_step_reasoning(
    *,
    model_name: str,
    thinking_level: Optional[str],
    steps: Iterable[str],
) -> tuple[Optional[ThinkingLevel], Dict[str, str]]:
    if not thinking_level:
        return None, {}

    public_level = to_public_reasoning_level(thinking_level)
    internal_level = to_internal_reasoning_level(thinking_level)
    if public_level not in {"low", "high"} or not internal_level:
        return None, {}

    supported_levels = set(get_reasoning_levels(model_name))
    if internal_level not in supported_levels:
        return None, {}

    return public_level, {step: internal_level for step in steps}


def select_step_models(
    *,
    model_name: str,
    thinking_level: Optional[str],
    functional_groups_enabled: bool,
    intermediate_prediction_enabled: bool,
    optional_llm_tools: Iterable[str],
) -> ModelSelectionResult:
    """Select a single exact model for every enabled LLM-backed step."""

    selected_model = _validate_model(model_name)
    family = get_model_family(selected_model)
    enabled_steps = _enabled_steps(
        functional_groups_enabled=functional_groups_enabled,
        intermediate_prediction_enabled=intermediate_prediction_enabled,
        optional_llm_tools=optional_llm_tools,
    )
    chosen = {step: selected_model for step in enabled_steps}
    if "mechanism_synthesis" not in chosen:
        chosen["mechanism_synthesis"] = selected_model

    public_thinking, step_reasoning = _resolve_step_reasoning(
        model_name=selected_model,
        thinking_level=thinking_level,
        steps=chosen.keys(),
    )

    notes: Dict[str, str] = {
        "mode": "single_model",
        "family": family,
        "model_name": selected_model,
        "thinking_level": public_thinking or "none",
    }
    if _TOOL_REQUIRED_STEPS and not model_supports_tools(selected_model):
        notes["tool_warning"] = "selected model lacks native tool support for some required steps"

    return ModelSelectionResult(
        family=family,  # type: ignore[arg-type]
        model_name=selected_model,
        thinking_level=public_thinking,
        step_models=chosen,
        step_reasoning=step_reasoning,
        notes=notes,
    )


def preview_step_models(
    *,
    model_name: str,
    thinking_level: Optional[str] = None,
    functional_groups_enabled: bool = True,
    intermediate_prediction_enabled: bool = True,
    optional_llm_tools: Optional[Iterable[str]] = None,
) -> Dict[str, str]:
    """Preview the derived uniform step model map for the selected model."""
    result = select_step_models(
        model_name=model_name,
        thinking_level=thinking_level,
        functional_groups_enabled=functional_groups_enabled,
        intermediate_prediction_enabled=intermediate_prediction_enabled,
        optional_llm_tools=optional_llm_tools or (
            "attempt_atom_mapping",
            "predict_missing_reagents",
        ),
    )
    return result.step_models


__all__ = [
    "ModelFamily",
    "ThinkingLevel",
    "ModelSelectionResult",
    "select_step_models",
    "preview_step_models",
]
