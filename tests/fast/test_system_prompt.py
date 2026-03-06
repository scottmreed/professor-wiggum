"""Tests for the dynamic system prompt builder."""

from __future__ import annotations

from types import SimpleNamespace

from mechanistic_agent.system_prompt import build_system_prompt


def _default_step_models() -> dict[str, str]:
    return {
        "balance_analysis": "gpt-5",
        "functional_groups": "gpt-5",
        "mechanism_synthesis": "gpt-5",
        "intermediates": "gpt-5",
        "missing_reagents": "gpt-5-nano",
        "atom_mapping": "gpt-5-nano",
    }


def _tool_stub(name: str):
    return SimpleNamespace(name=name)


def test_prompt_includes_budget_hint_for_missing_reagents() -> None:
    tools = [
        _tool_stub("analyse_balance"),
        _tool_stub("fingerprint_functional_groups"),
        _tool_stub("predict_missing_reagents"),
        _tool_stub("attempt_atom_mapping"),
        _tool_stub("recommend_ph"),
        _tool_stub("propose_intermediates"),
        _tool_stub("predict_mechanistic_step"),
    ]
    prompt = build_system_prompt(
        tools,
        _default_step_models(),
        primary_model="gpt-5",
        include_recommend_ph=True,
    )

    assert "Call `analyse_balance`" in prompt
    assert "predict_missing_reagents" in prompt
    assert "budget tier" in prompt  # cheaper model should be highlighted
    assert "Use `recommend_ph`" in prompt


def test_prompt_skips_missing_tools_and_reindexes() -> None:
    tools = [
        _tool_stub("analyse_balance"),
        _tool_stub("fingerprint_functional_groups"),
        _tool_stub("predict_missing_reagents"),
        _tool_stub("propose_intermediates"),
        _tool_stub("predict_mechanistic_step"),
    ]

    prompt = build_system_prompt(
        tools,
        _default_step_models(),
        primary_model="gpt-5",
        include_recommend_ph=False,
    )

    assert "attempt_atom_mapping" not in prompt
    # Ensure numbering is sequential despite the removed step.
    assert "1. Call `analyse_balance`" in prompt
    assert "2. Run `fingerprint_functional_groups`" in prompt
    # Missing step 3 should renumber predict_missing_reagents to 3
    assert "3. When imbalances remain" in prompt
    # recommend_ph instruction excluded when pH provided
    assert "recommend_ph" not in prompt
