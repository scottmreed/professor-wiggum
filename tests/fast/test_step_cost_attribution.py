"""Cost is attributed to the model that actually answered (Observatory PRD §3.7.2 follow-up)."""
from __future__ import annotations

import pytest

pytest.importorskip("rdkit")

from mechanistic_agent.core.subagents import _extract_step_cost
from mechanistic_agent.model_registry import calculate_cost, normalise_token_usage


def test_extract_step_cost_prefers_output_model_used_over_configured_model() -> None:
    raw = {"prompt_tokens": 100_000, "completion_tokens": 10_000}
    output = {"_llm_usage": dict(raw), "model_used": "gpt-4o"}

    usage, cost = _extract_step_cost(output, "anthropic/claude-opus-4.6")

    assert usage == normalise_token_usage(raw)
    assert cost == calculate_cost("gpt-4o", usage)
    assert cost != calculate_cost("anthropic/claude-opus-4.6", usage)
    assert "_llm_usage" not in output


def test_extract_step_cost_falls_back_to_configured_model_without_model_used() -> None:
    raw = {"prompt_tokens": 1000, "completion_tokens": 100}
    usage, cost = _extract_step_cost({"_llm_usage": dict(raw)}, "gpt-4o")
    assert cost == calculate_cost("gpt-4o", normalise_token_usage(raw))


def test_extract_step_cost_ignores_unknown_model_used_gracefully() -> None:
    raw = {"prompt_tokens": 1000, "completion_tokens": 100}
    usage, cost = _extract_step_cost({"_llm_usage": dict(raw), "model_used": "not-a-real-model"}, "gpt-4o")
    assert usage == normalise_token_usage(raw)
    assert cost == calculate_cost("gpt-4o", usage)
