"""Configuration models and helpers for the mechanistic agent."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .model_registry import (
    get_default_model,
    get_default_reasoning_level,
    get_model_catalog,
    get_reasoning_levels,
    resolve_model_key,
    to_internal_reasoning_level,
)


DEFAULT_STARTING_MATERIALS = ["C=O", "OCCO"]  # Formaldehyde and ethylene glycol
DEFAULT_PRODUCTS = ["C1OCOC1"]  # 1,3-dioxolane


LLM_STEP_KEYS = {
    "functional_groups",
    "mechanism_synthesis",
    "missing_reagents",
    "atom_mapping",
    "reaction_type_mapping",
    "intermediates",
    "initial_conditions",
}

OPTIONAL_LLM_TOOL_NAMES: tuple[str, ...] = (
    "attempt_atom_mapping",
    "predict_missing_reagents",
)

_OPTIONAL_TOOL_LOOKUP = {name.lower(): name for name in OPTIONAL_LLM_TOOL_NAMES}


def _default_step_models() -> Dict[str, str]:
    """Return the default model configuration for each LLM-powered step."""

    default = get_default_model()
    return {
        "functional_groups": default,
        "mechanism_synthesis": default,
        "intermediates": default,
        "missing_reagents": default,
        "initial_conditions": default,
        "atom_mapping": default,
        "reaction_type_mapping": default,
    }


class ReactionInputs(BaseModel):
    """Strongly typed reaction description derived from CLI inputs."""

    model_config = ConfigDict(validate_assignment=True)

    starting_materials: List[str] = Field(
        default_factory=lambda: DEFAULT_STARTING_MATERIALS.copy(),
        description="Comma-separated SMILES codes for reactants",
    )
    products: List[str] = Field(
        default_factory=lambda: DEFAULT_PRODUCTS.copy(),
        description="Comma-separated SMILES codes for target products",
    )
    temperature_celsius: float = Field(
        25.0,
        description="Reaction temperature in Celsius",
        ge=-273.15,
        le=500.0,
    )
    ph: Optional[float] = Field(
        None,
        description="Observed or expected reaction pH (optional)",
        ge=0.0,
        le=14.0,
    )
    model: str = Field(
        default_factory=get_default_model,
        description="Identifier of the language model used for analysis",
    )
    thinking_level: Optional[str] = Field(
        None,
        description="Public thinking level applied uniformly to enabled LLM-backed steps",
    )
    step_models: Dict[str, str] = Field(
        default_factory=_default_step_models,
        description="Mapping of workflow step identifiers to model choices",
    )
    step_reasoning: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of workflow step identifiers to reasoning levels",
    )
    optional_llm_tools: List[str] = Field(
        default_factory=lambda: list(OPTIONAL_LLM_TOOL_NAMES),
        description="Optional LLM-driven tool names to enable during analysis",
    )
    functional_groups_enabled: bool = Field(
        True,
        description="Include the functional group analysis step in the workflow",
    )
    intermediate_prediction_enabled: bool = Field(
        True,
        description="Include the intermediate prediction step in the workflow",
    )
    max_turns_override: Optional[int] = Field(
        None,
        description="Override maximum runtime loop turns (None uses CLI default)",
        ge=1,
        le=100,
    )
    max_runtime_seconds_override: Optional[float] = Field(
        None,
        description="Override maximum runtime in seconds (None uses CLI default)",
        ge=10.0,
        le=3600.0,
    )
    max_retries_override: Optional[int] = Field(
        None,
        description="Override maximum retries for unchanged materials (None uses default)",
        ge=0,
        le=10,
    )

    @field_validator("starting_materials", "products", mode="before")
    @classmethod
    def _split_materials(cls, value: object) -> List[str]:  # type: ignore[override]
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            parts = [token.strip() for token in value.split(",") if token.strip()]
            if parts:
                return parts
            raise ValueError("Provide at least one SMILES code")
        raise TypeError("Materials must be provided as a list or comma separated string")

    @field_validator("model")
    @classmethod
    def _validate_model(cls, value: str) -> str:  # type: ignore[override]
        try:
            return resolve_model_key(value)
        except ValueError as exc:
            raise ValueError(f"Unsupported model '{value}'") from exc

    @field_validator("step_models", mode="before")
    @classmethod
    def _normalise_step_models(cls, value: object) -> Dict[str, str]:
        if value is None:
            return {}
        if isinstance(value, dict):
            normalised: Dict[str, str] = {}
            for key, raw in value.items():
                key_str = str(key)
                if not key_str:
                    continue
                normalised[key_str] = str(raw)
            return normalised
        raise TypeError("step_models must be provided as a mapping of step names to model identifiers")

    @field_validator("optional_llm_tools", mode="before")
    @classmethod
    def _normalise_optional_tools(cls, value: object) -> List[str]:
        if value is None:
            return list(OPTIONAL_LLM_TOOL_NAMES)

        if isinstance(value, str):
            raw_items = [value]
        elif isinstance(value, (list, tuple, set)):
            raw_items = list(value)
        else:
            raise TypeError("optional_llm_tools must be a sequence or comma separated string of tool names")

        selected: List[str] = []
        for raw in raw_items:
            text = str(raw).strip()
            if not text:
                continue
            canonical = _OPTIONAL_TOOL_LOOKUP.get(text.lower())
            if canonical is None:
                raise ValueError(f"Unsupported optional LLM tool '{text}'")
            if canonical not in selected:
                selected.append(canonical)
        return selected

    @field_validator("step_reasoning", mode="before")
    @classmethod
    def _normalise_step_reasoning(cls, value: object) -> Dict[str, str]:
        if value is None:
            return {}
        if isinstance(value, dict):
            normalised: Dict[str, str] = {}
            for key, raw in value.items():
                key_str = str(key)
                if not key_str:
                    continue
                if raw is None:
                    continue
                raw_str = str(raw).strip()
                if raw_str:
                    normalised[key_str] = raw_str
            return normalised
        raise TypeError("step_reasoning must be provided as a mapping of step names to reasoning levels")

    @model_validator(mode="after")
    def _finalise_models(self) -> "ReactionInputs":
        catalog = get_model_catalog()
        defaults = _default_step_models()
        combined = {**defaults, **self.step_models}

        primary_model = self.model or combined["mechanism_synthesis"]
        combined["mechanism_synthesis"] = primary_model

        canonical_combined: Dict[str, str] = {}
        for step_name, model_name in combined.items():
            if step_name not in LLM_STEP_KEYS:
                raise ValueError(f"Unsupported step model key '{step_name}'")
            try:
                canonical_combined[step_name] = resolve_model_key(model_name)
            except ValueError as exc:
                known = ", ".join(sorted(catalog.keys()))
                raise ValueError(
                    f"Unsupported model '{model_name}' for step '{step_name}'. Known models: {known}"
                ) from exc

        reasoning_settings: Dict[str, str] = dict(self.step_reasoning)
        if self.thinking_level:
            internal_level = to_internal_reasoning_level(self.thinking_level)
            if internal_level:
                reasoning_settings = {step_name: internal_level for step_name in combined}
        for step_name, level in list(reasoning_settings.items()):
            if step_name not in LLM_STEP_KEYS:
                raise ValueError(f"Unsupported reasoning step key '{step_name}'")
            model_name = canonical_combined.get(step_name, primary_model)
            levels = get_reasoning_levels(model_name)
            if not levels:
                # Remove reasoning level for models that don't support it
                reasoning_settings.pop(step_name, None)
                continue
            if level not in levels:
                default_level = get_default_reasoning_level(model_name)
                if default_level and default_level in levels:
                    reasoning_settings[step_name] = default_level
                else:
                    raise ValueError(
                        f"Unsupported reasoning level '{level}' for model '{model_name}' (step '{step_name}')"
                    )

        object.__setattr__(self, "step_models", canonical_combined)
        object.__setattr__(self, "model", primary_model)
        object.__setattr__(self, "step_reasoning", reasoning_settings)
        return self

    @property
    def all_optional_llm_tools(self) -> List[str]:
        """Return the canonical list of optional LLM tool names."""

        return list(OPTIONAL_LLM_TOOL_NAMES)

    @property
    def reaction_summary(self) -> str:
        start = ", ".join(self.starting_materials)
        prod = ", ".join(self.products)
        temp = f"{self.temperature_celsius:.1f} °C"
        ph = "unspecified" if self.ph is None else f"pH {self.ph:.2f}"
        return f"{start} -> {prod} @ {temp}, {ph}"


@dataclass(slots=True)
class AgentLimits:
    """Bounds to keep runtime execution constrained."""

    max_turns: int = 30
    max_runtime_seconds: float = 600.0
    max_retries: int = 1


__all__ = [
    "ReactionInputs",
    "AgentLimits",
    "DEFAULT_STARTING_MATERIALS",
    "DEFAULT_PRODUCTS",
    "LLM_STEP_KEYS",
    "OPTIONAL_LLM_TOOL_NAMES",
]
