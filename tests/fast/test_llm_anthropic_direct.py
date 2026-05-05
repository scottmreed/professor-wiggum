from __future__ import annotations

from mechanistic_agent.llm import _AnthropicChatAdapter, get_chat_model


def test_get_chat_model_routes_direct_anthropic_models_to_anthropic_adapter(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured = {}

    class _FakeAnthropicAdapter:
        def __init__(self, **kwargs):  # noqa: ANN003
            captured.update(kwargs)

    monkeypatch.setattr("mechanistic_agent.llm._AnthropicChatAdapter", _FakeAnthropicAdapter)

    adapter = get_chat_model(
        "claude-opus-4-7",
        temperature=0.0,
        timeout=10.0,
        user_api_key="anthropic-key",
    )

    assert isinstance(adapter, _FakeAnthropicAdapter)
    assert captured["model"] == "claude-opus-4-7"
    assert captured["api_key"] == "anthropic-key"


def test_anthropic_adapter_returns_openai_style_tool_calls(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls = []

    class _TextBlock:
        type = "text"
        text = "reasoning"

    class _ToolUseBlock:
        type = "tool_use"
        id = "toolu_1"
        name = "predict_full_mechanism"
        input = {"steps": [], "mechanism_type": "oxidation"}

    class _Usage:
        input_tokens = 11
        output_tokens = 7
        cache_read_input_tokens = 3
        cache_creation_input_tokens = 2

    class _Response:
        content = [_TextBlock(), _ToolUseBlock()]
        usage = _Usage()

    class _FakeMessages:
        def create(self, **kwargs):  # noqa: ANN003
            calls.append(kwargs)
            return _Response()

    class _FakeAnthropic:
        def __init__(self, **_kwargs):  # noqa: ANN003
            self.messages = _FakeMessages()

    monkeypatch.setattr("mechanistic_agent.llm.Anthropic", _FakeAnthropic)

    adapter = _AnthropicChatAdapter(
        model="claude-opus-4-7",
        temperature=0.0,
        timeout=10.0,
        model_kwargs={"max_tokens": 2048},
        api_key="anthropic-key",
    )
    message = adapter.invoke(
        [{"role": "system", "content": "system"}, {"role": "user", "content": "hello"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "predict_full_mechanism",
                    "description": "Predict mechanism",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "predict_full_mechanism"}},
    )

    assert message.content == "reasoning"
    assert message.tool_calls == [
        {
            "id": "toolu_1",
            "name": "predict_full_mechanism",
            "arguments": '{"steps": [], "mechanism_type": "oxidation"}',
        }
    ]
    assert message.usage == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "prompt_cache_hit_tokens": 3,
        "prompt_cache_miss_tokens": 0,
        "cache_creation_tokens": 2,
    }
    assert calls[0]["system"] == "system"
    assert calls[0]["tool_choice"] == {"type": "tool", "name": "predict_full_mechanism"}
