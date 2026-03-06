"""Model catalog and pricing utilities for the mechanistic agent."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

_MODEL_DATA_PATH = Path(__file__).with_name("model_pricing.json")

try:
    with _MODEL_DATA_PATH.open("r", encoding="utf-8") as handle:
        _MODEL_CATALOG: Dict[str, Dict[str, Any]] = json.load(handle)
except FileNotFoundError as exc:  # pragma: no cover - configuration error
    raise RuntimeError(
        f"Model pricing file not found at {_MODEL_DATA_PATH}. Did you remove it?"
    ) from exc


def get_model_catalog() -> Dict[str, Dict[str, Any]]:
    """Return the loaded model catalog."""
    return _MODEL_CATALOG


def get_model_options() -> Iterable[Dict[str, Any]]:
    """Provide model options with metadata suitable for UI rendering."""
    for model_id, spec in _MODEL_CATALOG.items():
        reasoning = spec.get("reasoning") or {}
        levels: List[str] = []
        if isinstance(reasoning, dict):
            raw_levels = reasoning.get("levels", {})
            if isinstance(raw_levels, dict):
                levels = list(raw_levels.keys())
            elif isinstance(raw_levels, list):
                levels = list(raw_levels)
        yield {
            "id": model_id,
            "label": spec.get("label", model_id),
            "description": spec.get("description", ""),
            "family": spec.get("family", "openai"),
            "provider": spec.get("provider", "openai"),
            "supports_tools": spec.get("supports_tools", True),
            "best_in_class": spec.get("best_in_class", False),
            "pricing_per_million": spec.get("pricing_per_million", {}),
            "reasoning_levels": [
                public_level
                for public_level in (to_public_reasoning_level(level) for level in levels)
                if public_level
            ],
        }


# ---------------------------------------------------------------------------
# Family-level utilities
# ---------------------------------------------------------------------------

_FAMILY_ORDER = ["openai", "claude", "gemini", "olmo", "minimax"]
_FAMILY_LABELS = {
    "openai": "OpenAI",
    "claude": "Claude",
    "gemini": "Gemini",
    "olmo": "OLMo",
    "minimax": "Minimax",
}
_PUBLIC_REASONING_LEVELS = {
    "lowest": "low",
    "highest": "high",
}
_INTERNAL_REASONING_LEVELS = {value: key for key, value in _PUBLIC_REASONING_LEVELS.items()}
_FAMILY_DEFAULT_PRIORITY: Dict[str, List[str]] = {
    "openai": ["gpt-5.4", "gpt-5", "gpt-5.2", "gpt-5.1", "gpt-4o"],
}


def get_model_family(model_id: str) -> str:
    """Return the family name for a model ID."""
    spec = _MODEL_CATALOG.get(model_id, {})
    return str(spec.get("family", "openai"))


def get_model_provider(model_id: str) -> str:
    """Return the provider name for a model ID."""
    spec = _MODEL_CATALOG.get(model_id, {})
    return str(spec.get("provider", "openai"))


def model_supports_tools(model_id: str) -> bool:
    """Return True when the model natively supports tool/function calling."""
    spec = _MODEL_CATALOG.get(model_id, {})
    return bool(spec.get("supports_tools", True))


def get_family_models(family: str) -> List[str]:
    """Return all model IDs belonging to the given family, cheapest-first."""
    members = [
        model_id
        for model_id, spec in _MODEL_CATALOG.items()
        if spec.get("family") == family
    ]

    def _cost(mid: str) -> float:
        pricing = _MODEL_CATALOG.get(mid, {}).get("pricing_per_million", {})
        try:
            return float(pricing.get("input", 0)) + float(pricing.get("output", 0))
        except Exception:
            return float("inf")

    return sorted(members, key=_cost)


def get_cheapest_family_model(family: str) -> Optional[str]:
    """Return the cheapest model in the given family."""
    members = get_family_models(family)
    return members[0] if members else None


def get_top_family_model(family: str) -> Optional[str]:
    """Return the most capable (most expensive) model in the given family."""
    members = get_family_models(family)
    return members[-1] if members else None


def get_family_supports_reasoning(family: str) -> bool:
    """Return True when at least one model in the family supports reasoning levels."""
    for model_id, spec in _MODEL_CATALOG.items():
        if spec.get("family") == family:
            reasoning = spec.get("reasoning")
            if reasoning and isinstance(reasoning, dict) and reasoning.get("levels"):
                return True
    return False


def get_all_families() -> List[Dict[str, Any]]:
    """Return metadata for all known model families."""
    seen: Dict[str, Dict[str, Any]] = {}
    for model_id, spec in _MODEL_CATALOG.items():
        family = str(spec.get("family", "openai"))
        if family not in seen:
            seen[family] = {
                "id": family,
                "label": _FAMILY_LABELS.get(family, family.title()),
                "provider": spec.get("provider", "openai"),
                "supports_reasoning": False,
                "models": [],
                "cheapest_model": None,
                "top_model": None,
                "api_key_env": _family_api_key_env(family),
            }
        entry = seen[family]
        entry["models"].append(model_id)
        reasoning = spec.get("reasoning")
        if reasoning and isinstance(reasoning, dict) and reasoning.get("levels"):
            entry["supports_reasoning"] = True

    for family_id, entry in seen.items():
        members = get_family_models(family_id)
        entry["cheapest_model"] = members[0] if members else None
        entry["top_model"] = members[-1] if members else None
        entry["models"] = members  # sorted cheapest-first

    return [seen[f] for f in _FAMILY_ORDER if f in seen]


def _family_api_key_env(family: str) -> str:
    mapping = {
        "openai": "OPENAI_API_KEY",
        "claude": "OPENROUTER_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "olmo": "OPENROUTER_API_KEY",
        "minimax": "OPENROUTER_API_KEY",
    }
    return mapping.get(family, "OPENAI_API_KEY")


def to_public_reasoning_level(level: Optional[str]) -> Optional[str]:
    """Map internal reasoning labels to the public low/high vocabulary."""
    if level is None:
        return None
    return _PUBLIC_REASONING_LEVELS.get(level, level)


def to_internal_reasoning_level(level: Optional[str]) -> Optional[str]:
    """Map public low/high reasoning labels to provider-facing internal names."""
    if level is None:
        return None
    return _INTERNAL_REASONING_LEVELS.get(level, level)


def _resolve_catalog_key(model_name: str) -> Optional[str]:
    """Resolve catalogue key for model names that include provider-specific suffixes."""

    if not model_name:
        return None

    candidate = model_name.strip()
    if not candidate:
        return None

    if candidate in _MODEL_CATALOG:
        return candidate

    # Backward-compatible aliases for renamed provider IDs.
    alias_map = {
        "anthropic/claude-opus-4-5": "anthropic/claude-opus-4.6",
        "anthropic/claude-opus-4-6": "anthropic/claude-opus-4.6",
        "anthropic/claude-sonnet-4-5": "anthropic/claude-sonnet-4.5",
        "anthropic/claude-sonnet-4-6": "anthropic/claude-sonnet-4.6",
        "gpt-5-4": "gpt-5.4",
        "openai/gpt-5-4": "openai/gpt-5.4",
    }
    aliased = alias_map.get(candidate)
    if aliased and aliased in _MODEL_CATALOG:
        return aliased

    # Prefer the longest known model identifier that matches the start of the name. This
    # handles cases like ``gpt-5-nano-2025-01-01`` where providers append version tags.
    prefix_matches = [
        known
        for known in _MODEL_CATALOG
        if candidate.startswith(known)
    ]
    if prefix_matches:
        return max(prefix_matches, key=len)

    # Fall back to progressively trimming hyphen-separated suffixes to cover names that
    # don't strictly start with the base identifier (e.g. additional metadata segments).
    parts = candidate.split("-")
    for index in range(len(parts) - 1, 0, -1):
        shortened = "-".join(parts[:index])
        if shortened in _MODEL_CATALOG:
            return shortened

    # Some providers use colon separators for versions. Retain the base identifier when
    # present.
    if ":" in candidate:
        prefix = candidate.split(":", 1)[0]
        if prefix in _MODEL_CATALOG:
            return prefix

    return None


def get_model_spec(model_name: str) -> Dict[str, Any]:
    """Look up metadata for a specific model."""

    resolved = _resolve_catalog_key(model_name)
    if resolved is None:
        raise ValueError(f"Unknown model '{model_name}'")
    return _MODEL_CATALOG[resolved]


def resolve_model_key(model_name: str) -> str:
    """Resolve a model identifier to a canonical catalog key."""
    resolved = _resolve_catalog_key(model_name)
    if resolved is None:
        raise ValueError(f"Unknown model '{model_name}'")
    return resolved


def get_reasoning_levels(model_name: str) -> List[str]:
    """Return supported reasoning levels for the given model."""
    spec = get_model_spec(model_name)
    reasoning = spec.get("reasoning", {})
    if not isinstance(reasoning, dict):
        return []
    levels = reasoning.get("levels", {})
    if isinstance(levels, dict):
        return list(levels.keys())
    if isinstance(levels, list):
        return list(levels)
    return []


def get_default_reasoning_level(model_name: str) -> Optional[str]:
    """Return the default reasoning level for the given model, if any."""
    spec = get_model_spec(model_name)
    reasoning = spec.get("reasoning", {})
    if not isinstance(reasoning, dict):
        return None
    default = reasoning.get("default")
    if isinstance(default, str) and default:
        return default
    return None


def build_reasoning_payload(
    model_name: str,
    level: Optional[str],
) -> Dict[str, Any]:
    """Build provider-specific reasoning payload for the given model and level."""
    if not level or level == "auto":
        return {}

    spec = get_model_spec(model_name)
    reasoning = spec.get("reasoning", {})
    if not isinstance(reasoning, dict):
        return {}

    parameter = reasoning.get("parameter")
    if not isinstance(parameter, str) or not parameter:
        return {}

    levels = reasoning.get("levels", {})
    payload = None
    if isinstance(levels, dict):
        payload = levels.get(level)
    if payload is None:
        return {}
    return {parameter: payload}


def get_default_model(family: Optional[str] = None) -> str:
    """Return the default model identifier, optionally filtered by family."""
    if family:
        for preferred in _FAMILY_DEFAULT_PRIORITY.get(family, []):
            if preferred in _MODEL_CATALOG:
                return preferred
        cheapest = get_cheapest_family_model(family)
        if cheapest:
            return cheapest

    if "gpt-5.4" in _MODEL_CATALOG:
        return "gpt-5.4"
    if "gpt-5" in _MODEL_CATALOG:
        return "gpt-5"
    # Fall back to the first available model when GPT-5 is missing.
    return next(iter(_MODEL_CATALOG))


def get_fallback_model(family: Optional[str] = None) -> str:
    """Return a fallback model identifier for when the primary model fails."""
    if family:
        cheapest = get_cheapest_family_model(family)
        if cheapest:
            return cheapest

    if "gpt-4o" in _MODEL_CATALOG:
        return "gpt-4o"
    if "gpt-4" in _MODEL_CATALOG:
        return "gpt-4"
    return next(iter(_MODEL_CATALOG))


def normalise_token_usage(token_usage: Mapping[str, Any]) -> Dict[str, int]:
    """Normalise raw token usage metadata into standard fields."""
    prompt_tokens = int(token_usage.get("prompt_tokens") or 0)
    completion_tokens = int(
        token_usage.get("completion_tokens")
        or token_usage.get("output_tokens")
        or 0
    )
    cache_hits = int(token_usage.get("prompt_cache_hit_tokens") or 0)
    cache_misses = int(token_usage.get("prompt_cache_miss_tokens") or 0)

    if prompt_tokens == 0 and (cache_hits or cache_misses):
        prompt_tokens = cache_hits + cache_misses

    # Some providers report cached tokens separately but not prompt totals.
    if prompt_tokens < cache_hits:
        prompt_tokens = cache_hits

    input_tokens = max(prompt_tokens - cache_hits, 0)
    cached_tokens = min(cache_hits, prompt_tokens)

    total_tokens = int(token_usage.get("total_tokens") or 0)
    if total_tokens == 0:
        total_tokens = input_tokens + cached_tokens + completion_tokens

    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def calculate_cost(model_name: str, usage: Mapping[str, Any]) -> Dict[str, float]:
    """Calculate estimated cost for a usage breakdown."""
    spec = get_model_spec(model_name)
    pricing = spec.get("pricing_per_million", {})
    input_rate = float(pricing.get("input", 0.0))
    cached_rate = float(pricing.get("cached_input", 0.0))
    output_rate = float(pricing.get("output", 0.0))

    input_tokens = int(usage.get("input_tokens", 0))
    cached_tokens = int(usage.get("cached_input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))

    factor = 1_000_000.0
    input_cost = (input_tokens / factor) * input_rate
    cached_cost = (cached_tokens / factor) * cached_rate
    output_cost = (output_tokens / factor) * output_rate

    return {
        "input_cost": input_cost,
        "cached_input_cost": cached_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + cached_cost + output_cost,
    }


def update_usage_totals(
    totals: MutableMapping[str, int],
    usage: Mapping[str, Any],
) -> None:
    """Accumulate usage counts into the provided totals mapping."""
    for key, value in usage.items():
        if not isinstance(value, (int, float)):
            continue
        totals[key] = int(totals.get(key, 0)) + int(value)


def update_cost_totals(
    totals: MutableMapping[str, float],
    cost: Mapping[str, Any],
) -> None:
    """Accumulate cost values into the provided totals mapping."""
    for key, value in cost.items():
        if not isinstance(value, (int, float)):
            continue
        totals[key] = float(totals.get(key, 0.0)) + float(value)


__all__ = [
    "get_model_catalog",
    "get_model_options",
    "get_model_spec",
    "resolve_model_key",
    "get_default_model",
    "get_fallback_model",
    "get_reasoning_levels",
    "get_default_reasoning_level",
    "to_public_reasoning_level",
    "to_internal_reasoning_level",
    "build_reasoning_payload",
    "normalise_token_usage",
    "calculate_cost",
    "update_usage_totals",
    "update_cost_totals",
    # Family utilities
    "get_model_family",
    "get_model_provider",
    "model_supports_tools",
    "get_family_models",
    "get_cheapest_family_model",
    "get_top_family_model",
    "get_family_supports_reasoning",
    "get_all_families",
]
