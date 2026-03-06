"""Thread-local model context for passing per-run model configuration to tool functions.

Each run executes in its own background thread. Before calling tools, the
coordinator sets the active step models and API keys on this thread-local store.
Tool functions read from here (falling back to environment variables when
nothing is set) so that model selection actually propagates to LLM calls.
"""
from __future__ import annotations

import threading
from typing import Dict, Optional

_local = threading.local()


# ---------------------------------------------------------------------------
# Setters (called by RunCoordinator before executing each run)
# ---------------------------------------------------------------------------

def set_run_context(
    *,
    step_models: Dict[str, str],
    step_reasoning: Dict[str, str],
    active_model: str,
    model_family: str = "openai",
    reasoning_level: Optional[str] = None,
    api_keys: Optional[Dict[str, str]] = None,
    few_shot_policies: Optional[Dict[str, Dict[str, object]]] = None,
) -> None:
    """Set the full model context for the current thread (one call per run)."""
    _local.step_models = dict(step_models)
    _local.step_reasoning = dict(step_reasoning)
    _local.active_model = active_model
    _local.model_family = model_family
    _local.reasoning_level = reasoning_level
    _local.api_keys = dict(api_keys or {})
    _local.few_shot_policies = dict(few_shot_policies or {})


def clear_run_context() -> None:
    """Clear thread-local context (called after run completes)."""
    for attr in ("step_models", "step_reasoning", "active_model", "model_family",
                 "reasoning_level", "api_keys", "few_shot_policies"):
        try:
            delattr(_local, attr)
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# Getters (called by tools.py functions)
# ---------------------------------------------------------------------------

def get_step_model(step_name: str, default: Optional[str] = None) -> Optional[str]:
    """Return the model assigned to a specific step, or default."""
    step_models: Dict[str, str] = getattr(_local, "step_models", {})
    return step_models.get(step_name, default)


def get_active_model(default: Optional[str] = None) -> Optional[str]:
    """Return the run-level active model (primary/mechanism model)."""
    return getattr(_local, "active_model", default)


def get_active_model_slug(default: Optional[str] = None) -> Optional[str]:
    """Return a filesystem-safe model slug for the active model."""
    model_name = get_active_model(default=None)
    if not model_name:
        return default
    return str(model_name).replace("/", "__")


def get_model_family() -> str:
    """Return the model family selected for the current run."""
    return getattr(_local, "model_family", "openai")


def get_reasoning_level() -> Optional[str]:
    """Return the run-level reasoning level (lowest/highest/None)."""
    return getattr(_local, "reasoning_level", None)


def get_step_reasoning(step_name: str) -> Optional[str]:
    """Return the reasoning level for a specific step."""
    step_reasoning: Dict[str, str] = getattr(_local, "step_reasoning", {})
    return step_reasoning.get(step_name) or getattr(_local, "reasoning_level", None)


def get_api_key(provider: str) -> Optional[str]:
    """Return a user-provided API key for the given provider (openai/openrouter/gemini)."""
    api_keys: Dict[str, str] = getattr(_local, "api_keys", {})
    return api_keys.get(provider) or None


def get_few_shot_policy(call_name: str) -> Optional[Dict[str, object]]:
    """Return the active harness few-shot policy for a prompt call, if any."""
    policies: Dict[str, Dict[str, object]] = getattr(_local, "few_shot_policies", {})
    return policies.get(str(call_name or "").strip()) or None
