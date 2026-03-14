from __future__ import annotations

from mechanistic_agent.llm import _OpenAIChatAdapter


def test_openai_adapter_retries_unknown_model_kwargs_with_extra_body(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls = []

    class _FakeMessage:
        content = "ok"
        tool_calls = []

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeResponse:
        choices = [_FakeChoice()]
        usage = None

    class _FakeCompletions:
        def create(self, **kwargs):  # noqa: ANN003
            calls.append(kwargs)
            if "thinking" in kwargs:
                raise TypeError("Completions.create() got an unexpected keyword argument 'thinking'")
            return _FakeResponse()

    class _FakeClient:
        def __init__(self) -> None:
            self.chat = type("_Chat", (), {"completions": _FakeCompletions()})()

    monkeypatch.setattr("mechanistic_agent.llm.OpenAI", lambda **_kwargs: _FakeClient())

    adapter = _OpenAIChatAdapter(
        model="anthropic/claude-opus-4.6",
        temperature=0.0,
        timeout=5.0,
        model_kwargs={
            "thinking": {"type": "adaptive"},
            "effort": "low",
        },
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
    )
    message = adapter.invoke([{"role": "user", "content": "hello"}])

    assert message.content == "ok"
    assert len(calls) == 2
    assert "thinking" in calls[0]
    assert "thinking" not in calls[1]
    assert calls[1]["extra_body"]["thinking"] == {"type": "adaptive"}
    assert calls[1]["extra_body"]["effort"] == "low"
