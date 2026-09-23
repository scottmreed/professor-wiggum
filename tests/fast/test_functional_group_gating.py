"""Tests for the functional-group prompt-injection gate.

Before this fix, whether the conditions/mapping/proposal prompt builders
included a functional-group context block was controlled *only* by the
``MECHANISTIC_FUNCTIONAL_GROUPS_ENABLED`` env var — nothing set that var
outside tests, and the harness ``functional_groups`` module's ``enabled``
state (wired from ``RunConfig.functional_groups_enabled`` via its
``config_gate``) had no effect on prompt content at all.

These tests pin the fixed behavior directly at the tool-function level
(the layer the harness/config actually threads a resolved boolean into via
``functional_groups_enabled=...``), and confirm the env var still works as
an explicit override.
"""

import json

import pytest

from mechanistic_agent import tools

pytest.importorskip("rdkit")


class _FakeAIMessage:
    """Minimal stand-in for a provider response; carries no usable content."""

    def __init__(self):
        self.tool_calls = []
        self.content = ""
        self.usage = None


class _FakeChatModel:
    """Captures every ``invoke()`` call's messages without hitting a network."""

    def __init__(self, sink):
        self._sink = sink

    def invoke(self, messages, **kwargs):
        self._sink.append(messages)
        return _FakeAIMessage()


@pytest.fixture
def captured_prompts(monkeypatch):
    """Patch out the LLM client so prompt-builder calls can run offline.

    Returns the list of ``messages`` passed to each ``invoke()`` call, in
    order, so a test can inspect the human/user message content.
    """

    captured: list = []
    monkeypatch.setattr(tools, "get_chat_model", lambda *a, **k: _FakeChatModel(captured))
    monkeypatch.setattr(tools, "get_model_api_key", lambda *a, **k: "sk-test-key")
    monkeypatch.delenv("MECHANISTIC_FUNCTIONAL_GROUPS_ENABLED", raising=False)
    return captured


def _user_content(messages) -> str:
    for message in messages:
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


# --- conditions (assess_initial_conditions) ---------------------------------


def test_conditions_prompt_includes_fg_block_when_enabled(captured_prompts):
    tools.assess_initial_conditions(
        ["CC=O"], ["CCO"], functional_groups_enabled=True
    )
    assert captured_prompts, "expected the fake LLM to have been invoked"
    content = _user_content(captured_prompts[-1])
    assert "Functional group compatibility checks" in content


def test_conditions_prompt_omits_fg_block_when_disabled(captured_prompts):
    tools.assess_initial_conditions(
        ["CC=O"], ["CCO"], functional_groups_enabled=False
    )
    assert captured_prompts, "expected the fake LLM to have been invoked"
    content = _user_content(captured_prompts[-1])
    assert "Functional group compatibility checks" not in content


# --- mapping (attempt_atom_mapping) ------------------------------------------


def test_mapping_prompt_includes_fg_block_when_enabled(captured_prompts):
    tools.attempt_atom_mapping(["CC=O"], ["CCO"], functional_groups_enabled=True)
    assert captured_prompts, "expected the fake LLM to have been invoked"
    content = _user_content(captured_prompts[-1])
    assert "Functional group analysis (reactive sites) for context" in content


def test_mapping_prompt_omits_fg_block_when_disabled(captured_prompts):
    tools.attempt_atom_mapping(["CC=O"], ["CCO"], functional_groups_enabled=False)
    assert captured_prompts, "expected the fake LLM to have been invoked"
    content = _user_content(captured_prompts[-1])
    assert "Functional group analysis (reactive sites) for context" not in content


# --- proposal (propose_intermediates) ----------------------------------------


def test_proposal_prompt_includes_fg_block_when_enabled(captured_prompts):
    tools.propose_intermediates(
        starting_materials=["CC=O"],
        products=["CCO"],
        current_state=["CC=O"],
        functional_groups_enabled=True,
    )
    assert captured_prompts, "expected the fake LLM to have been invoked"
    content = _user_content(captured_prompts[-1])
    assert "Functional group analysis:" in content


def test_proposal_prompt_omits_fg_block_when_disabled(captured_prompts):
    tools.propose_intermediates(
        starting_materials=["CC=O"],
        products=["CCO"],
        current_state=["CC=O"],
        functional_groups_enabled=False,
    )
    assert captured_prompts, "expected the fake LLM to have been invoked"
    content = _user_content(captured_prompts[-1])
    assert "Functional group analysis:" not in content


# --- env var override ---------------------------------------------------------


def test_env_var_forces_enabled_even_when_harness_disabled(captured_prompts, monkeypatch):
    monkeypatch.setenv("MECHANISTIC_FUNCTIONAL_GROUPS_ENABLED", "1")
    tools.assess_initial_conditions(
        ["CC=O"], ["CCO"], functional_groups_enabled=False
    )
    content = _user_content(captured_prompts[-1])
    assert "Functional group compatibility checks" in content


def test_env_var_forces_disabled_even_when_harness_enabled(captured_prompts, monkeypatch):
    monkeypatch.setenv("MECHANISTIC_FUNCTIONAL_GROUPS_ENABLED", "0")
    tools.assess_initial_conditions(
        ["CC=O"], ["CCO"], functional_groups_enabled=True
    )
    content = _user_content(captured_prompts[-1])
    assert "Functional group compatibility checks" not in content


def test_default_is_enabled_without_env_var_or_explicit_flag(captured_prompts):
    """No env var and no explicit harness flag (None) now defaults to enabled."""
    tools.assess_initial_conditions(["CC=O"], ["CCO"])
    content = _user_content(captured_prompts[-1])
    assert "Functional group compatibility checks" in content
