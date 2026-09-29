"""Claude Opus 5.5 in the model catalog and the adapters (Observatory PRD rev 3, product default model).

Opus 5.5 (`claude-opus-5-5`; OpenRouter `anthropic/claude-opus-5.5`) differs
from the 4.x line in two ways the harness must handle: forced ``tool_choice``
(``any`` / a named tool) is rejected with a 400, and thinking cannot be
disabled (only ``effort`` controls depth; default ``medium``). The harness
keeps its forced-tool contract by sending ``tool_choice: auto`` plus an
explicit instruction to call the named tool, and validates the returned
tool call name downstream as before.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from mechanistic_agent.llm import _OpenAIChatAdapter, steer_forced_tool_choice
from mechanistic_agent.model_registry import (
    build_reasoning_payload,
    calculate_cost,
    get_default_reasoning_level,
    get_model_spec,
    get_reasoning_levels,
    model_supports_forced_tool_choice,
)
from mechanistic_agent.tool_schemas import MECHANISM_STEP_PROPOSAL_TOOL, build_tool_choice

OPUS55 = "anthropic/claude-opus-5.5"
OPUS46 = "anthropic/claude-opus-4.6"


def test_catalog_entry_exists_with_openrouter_provider_and_pricing() -> None:
    spec = get_model_spec(OPUS55)
    assert spec["id"] == OPUS55 and spec["family"] == "claude" and spec["provider"] == "openrouter"
    assert spec["supports_tools"] is True
    assert spec["forced_tool_choice"] is False
    cost = calculate_cost(OPUS55, {"input_tokens": 1_000_000, "cached_input_tokens": 1_000_000, "output_tokens": 1_000_000})
    assert cost["total_cost"] == pytest.approx(4.0 + 0.20 + 20.0)


def test_forced_tool_choice_flag_defaults_true_for_other_models() -> None:
    assert model_supports_forced_tool_choice(OPUS46) is True
    assert model_supports_forced_tool_choice(OPUS55) is False
    assert model_supports_forced_tool_choice("gpt-4o") is True


def test_opus_55_reasoning_levels_never_disable_thinking() -> None:
    levels = get_reasoning_levels(OPUS55)
    assert levels and "lowest" not in levels
    assert get_default_reasoning_level(OPUS55) == "medium"
    for level in levels:
        payload = build_reasoning_payload(OPUS55, level)
        assert payload.get("thinking", {}).get("type") == "adaptive", level
        assert payload.get("effort") in {"low", "medium", "high", "xhigh", "max"}, level


def test_steering_helper_rewrites_forced_choice_only_for_unsupported_models() -> None:
    messages: List[Dict[str, Any]] = [{"role": "system", "content": "sys"}, {"role": "user", "content": "propose"}]
    forced = build_tool_choice("mechanism_step_proposal_result")

    kept_choice, kept_messages = steer_forced_tool_choice(OPUS46, messages, forced)
    assert kept_choice == forced and kept_messages == messages

    choice, steered = steer_forced_tool_choice(OPUS55, messages, forced)
    assert choice == "auto"
    assert steered[0] == messages[0]
    assert steered[-1]["role"] == "user"
    assert steered[-1]["content"].startswith("propose")
    assert "mechanism_step_proposal_result" in steered[-1]["content"]
    assert messages[-1]["content"] == "propose", "input must not be mutated"


def test_steering_helper_handles_list_content_and_non_forced_choices() -> None:
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    choice, steered = steer_forced_tool_choice(OPUS55, messages, build_tool_choice("t"))
    assert choice == "auto" and steered[-1]["content"][-1]["type"] == "text" and "`t`" in steered[-1]["content"][-1]["text"]
    assert steer_forced_tool_choice(OPUS55, messages, "auto") == ("auto", messages)
    assert steer_forced_tool_choice(OPUS55, messages, None) == (None, messages)
    assert steer_forced_tool_choice(OPUS55, messages, "required") == ("auto", messages)


class _FakeCompletions:
    def __init__(self) -> None:
        self.params: Dict[str, Any] = {}

    def create(self, **params: Any) -> Any:
        self.params = params
        msg = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id="c1", function=SimpleNamespace(name="mechanism_step_proposal_result", arguments="{}"))])
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)


def _adapter(model: str) -> tuple[_OpenAIChatAdapter, _FakeCompletions]:
    adapter = object.__new__(_OpenAIChatAdapter)
    fake = _FakeCompletions()
    adapter._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    adapter._model = model
    adapter._temperature = None
    adapter._model_kwargs = {}
    adapter._timeout = 30
    return adapter, fake


def test_openai_adapter_sends_auto_plus_instruction_for_opus_55() -> None:
    adapter, fake = _adapter(OPUS55)
    result = adapter.invoke([{"role": "user", "content": "propose"}], tools=[MECHANISM_STEP_PROPOSAL_TOOL], tool_choice=build_tool_choice("mechanism_step_proposal_result"))
    assert fake.params["tool_choice"] == "auto"
    assert "mechanism_step_proposal_result" in fake.params["messages"][-1]["content"]
    assert result.tool_calls[0]["name"] == "mechanism_step_proposal_result"


def test_openai_adapter_keeps_forced_choice_for_opus_46() -> None:
    adapter, fake = _adapter(OPUS46)
    adapter.invoke([{"role": "user", "content": "propose"}], tools=[MECHANISM_STEP_PROPOSAL_TOOL], tool_choice=build_tool_choice("mechanism_step_proposal_result"))
    assert fake.params["tool_choice"] == build_tool_choice("mechanism_step_proposal_result")
    assert fake.params["messages"][-1]["content"] == "propose"


# ---------------------------------------------------------------------------
# Retry with a harder push, then a human-readable error (Scott, 2026-09-29):
# on models that reject forced tool_choice the prompt must push hard for the
# tool; if the model still answers in prose, retry once with a stronger
# instruction, and if that fails raise an error the product can show and the
# logs can be mined for.
# ---------------------------------------------------------------------------

from mechanistic_agent.llm import ForcedToolCallUnavailableError  # noqa: E402


class _ScriptedCompletions:
    """Returns scripted responses in order: 'text' → prose only, 'tool' → the tool call."""

    def __init__(self, script: List[str]) -> None:
        self.script = list(script)
        self.requests: List[Dict[str, Any]] = []

    def create(self, **params: Any) -> Any:
        self.requests.append(params)
        kind = self.script.pop(0)
        if kind == "tool":
            msg = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id="c1", function=SimpleNamespace(name="mechanism_step_proposal_result", arguments="{}"))])
        else:
            msg = SimpleNamespace(content="Here is my answer in prose.", tool_calls=[])
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})


def _scripted(model: str, script: List[str]) -> tuple[_OpenAIChatAdapter, _ScriptedCompletions]:
    adapter = object.__new__(_OpenAIChatAdapter)
    fake = _ScriptedCompletions(script)
    adapter._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    adapter._model = model
    adapter._temperature = None
    adapter._model_kwargs = {}
    adapter._timeout = 30
    return adapter, fake


def test_prose_answer_triggers_one_harder_retry_that_succeeds() -> None:
    adapter, fake = _scripted(OPUS55, ["text", "tool"])
    result = adapter.invoke([{"role": "user", "content": "propose"}], tools=[MECHANISM_STEP_PROPOSAL_TOOL], tool_choice=build_tool_choice("mechanism_step_proposal_result"))
    assert result.tool_calls[0]["name"] == "mechanism_step_proposal_result"
    assert len(fake.requests) == 2
    first, second = fake.requests
    assert first["tool_choice"] == "auto" and second["tool_choice"] == "auto"
    assert "mechanism_step_proposal_result" in second["messages"][-1]["content"]
    assert "must" in second["messages"][-1]["content"].lower()
    assert second["messages"][-1]["content"] != first["messages"][-1]["content"], "retry must push harder"


def test_prose_twice_raises_human_readable_error() -> None:
    adapter, fake = _scripted(OPUS55, ["text", "text"])
    with pytest.raises(ForcedToolCallUnavailableError) as info:
        adapter.invoke([{"role": "user", "content": "propose"}], tools=[MECHANISM_STEP_PROPOSAL_TOOL], tool_choice=build_tool_choice("mechanism_step_proposal_result"))
    err = info.value
    assert len(fake.requests) == 2
    assert err.model == OPUS55 and err.tool_name == "mechanism_step_proposal_result" and err.attempts == 2
    text = str(err)
    assert OPUS55 in text and "mechanism_step_proposal_result" in text and "did not call" in text
    assert "prose" in text.lower() or "plain text" in text.lower()
    assert err.error_code == "tool_call_missing"


def test_wrong_tool_name_counts_as_missing() -> None:
    class _WrongTool(_ScriptedCompletions):
        def create(self, **params: Any) -> Any:
            self.requests.append(params)
            msg = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id="c1", function=SimpleNamespace(name="other_tool", arguments="{}"))])
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)

    adapter = object.__new__(_OpenAIChatAdapter)
    fake = _WrongTool([])
    adapter._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    adapter._model = OPUS55
    adapter._temperature = None
    adapter._model_kwargs = {}
    adapter._timeout = 30
    with pytest.raises(ForcedToolCallUnavailableError):
        adapter.invoke([{"role": "user", "content": "propose"}], tools=[MECHANISM_STEP_PROPOSAL_TOOL], tool_choice=build_tool_choice("mechanism_step_proposal_result"))
    assert len(fake.requests) == 2


def test_models_with_forced_tool_choice_never_retry_on_prose() -> None:
    adapter, fake = _scripted(OPUS46, ["text"])
    result = adapter.invoke([{"role": "user", "content": "propose"}], tools=[MECHANISM_STEP_PROPOSAL_TOOL], tool_choice=build_tool_choice("mechanism_step_proposal_result"))
    assert result.tool_calls == [] and len(fake.requests) == 1


def test_recorder_logs_the_missing_tool_call_as_a_failed_call() -> None:
    from mechanistic_agent.core.call_recorder import RecordingChatAdapter, call_context

    events: List[Dict[str, Any]] = []
    adapter, _ = _scripted(OPUS55, ["text", "text"])
    with call_context(run_id="r", step_name="mechanism_step_proposal", sink=lambda t, p, step_name=None: events.append({"type": t, **p})):
        with pytest.raises(ForcedToolCallUnavailableError):
            RecordingChatAdapter(adapter, OPUS55).invoke([{"role": "user", "content": "propose"}], tools=[MECHANISM_STEP_PROPOSAL_TOOL], tool_choice=build_tool_choice("mechanism_step_proposal_result"))
    failed = [e for e in events if e["type"] == "inference_call_failed"]
    assert len(failed) == 1
    assert "ForcedToolCallUnavailableError" in failed[0]["error"] and "tool_call_missing" in failed[0]["error"]
