from __future__ import annotations

from mechanistic_agent.tools import _prompt_char_cap_for_model


def test_gpt_55_uses_large_gpt5_prompt_cap(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("MECHANISTIC_PROMPT_CHAR_CAP", raising=False)
    monkeypatch.delenv("MECHANISTIC_GPT5_PROMPT_CHAR_CAP", raising=False)

    assert _prompt_char_cap_for_model("gpt-5.5") == 2_400_000


def test_gpt5_prompt_cap_can_be_overridden_for_new_and_legacy_names(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("MECHANISTIC_GPT5_PROMPT_CHAR_CAP", "123456")
    monkeypatch.setenv("MECHANISTIC_GPT54_PROMPT_CHAR_CAP", "654321")

    assert _prompt_char_cap_for_model("gpt-5.5") == 123456
    assert _prompt_char_cap_for_model("gpt-5.4") == 123456
