from mechanistic_agent.llm import get_provider_label, is_openrouter_model


def test_bare_claude_alias_routes_through_openrouter():
    assert is_openrouter_model("claude-opus-4-6") is True
    assert get_provider_label("claude-opus-4-6") == "OpenRouter"


def test_bare_claude_dot_alias_routes_through_openrouter():
    assert is_openrouter_model("claude-sonnet-4.6") is True
    assert get_provider_label("claude-sonnet-4.6") == "OpenRouter"
