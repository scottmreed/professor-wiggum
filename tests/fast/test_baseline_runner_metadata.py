from __future__ import annotations

import json

import pytest

from mechanistic_agent.core.baseline_runner import BaselineRunner


def test_baseline_runner_records_prompt_hashes_and_seed_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubResponse:
        usage = None
        content = "analysis"

        def __init__(self) -> None:
            payload = {
                "mechanism_type": "sn2",
                "steps": [
                    {
                        "step_index": 1,
                        "step_label": "SN2 attack",
                        "current_state": ["CCBr", "[Cl-]"],
                        "resulting_state": ["CCCl", "[Br-]"],
                        "predicted_intermediate": "CCCl",
                        "reaction_smirks": "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3]",
                        "electron_pushes": [{"kind": "lone_pair", "source_atom": "4", "target_atom": "2", "electrons": 2}],
                        "contains_target_product": True,
                    }
                ],
            }
            self.tool_calls = [{"name": "predict_full_mechanism", "arguments": json.dumps(payload)}]

    class _StubAdapter:
        def __init__(self, *, should_fail: bool) -> None:
            self._should_fail = should_fail

        def invoke(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
            if self._should_fail:
                raise RuntimeError("seed unsupported")
            return _StubResponse()

    calls = []

    def _fake_get_chat_model(*_args, **kwargs):  # noqa: ANN002
        model_kwargs = dict(kwargs.get("model_kwargs") or {})
        calls.append({"temperature": kwargs.get("temperature"), "model_kwargs": model_kwargs})
        should_fail = len(calls) == 1 and "seed" in model_kwargs
        return _StubAdapter(should_fail=should_fail)

    monkeypatch.setattr("mechanistic_agent.core.baseline_runner.get_chat_model", _fake_get_chat_model)
    monkeypatch.setattr("mechanistic_agent.core.baseline_runner.adapter_supports_forced_tools", lambda _model: True)

    runner = BaselineRunner()
    result = runner.run_case(
        starting_materials=["CCBr", "[Cl-]"],
        products=["CCCl", "[Br-]"],
        model="gpt-4o-mini",
        llm_seed=7,
        llm_temperature=0.15,
        sampling_policy="fixed",
    )

    assert calls[0]["model_kwargs"]["seed"] == 7
    assert "seed" not in calls[1]["model_kwargs"]
    assert calls[0]["temperature"] == 0.15
    assert result["llm_seed_requested"] == 7
    assert result["llm_seed_applied"] is None
    assert result["sampling_policy"] == "fixed"
    assert isinstance(result["prompt_hash"], str) and len(result["prompt_hash"]) == 64
    assert isinstance(result["prompt_system_hash"], str) and len(result["prompt_system_hash"]) == 64
    assert isinstance(result["prompt_user_hash"], str) and len(result["prompt_user_hash"]) == 64


def test_baseline_runner_uses_provider_specific_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubResponse:
        usage = None
        content = ""
        tool_calls = [
            {
                "name": "predict_full_mechanism",
                "arguments": json.dumps({"mechanism_type": "sn2", "steps": []}),
            }
        ]

    class _StubAdapter:
        def invoke(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
            return _StubResponse()

    captured_user_keys: list[str | None] = []

    def _fake_get_chat_model(*_args, **kwargs):  # noqa: ANN002
        captured_user_keys.append(kwargs.get("user_api_key"))
        return _StubAdapter()

    monkeypatch.setattr("mechanistic_agent.core.baseline_runner.get_chat_model", _fake_get_chat_model)
    monkeypatch.setattr("mechanistic_agent.core.baseline_runner.adapter_supports_forced_tools", lambda _model: True)

    runner = BaselineRunner()
    runner.run_case(
        starting_materials=["CCBr"],
        products=["CCCl"],
        model="anthropic/claude-opus-4.6",
        api_keys={
            "openai_api_key": "openai-key",
            "openrouter_api_key": "openrouter-key",
        },
    )

    assert captured_user_keys == ["openrouter-key"]


def test_baseline_runner_normalizes_openrouter_reasoning_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubResponse:
        usage = None
        content = ""
        tool_calls = [
            {
                "name": "predict_full_mechanism",
                "arguments": json.dumps({"mechanism_type": "sn2", "steps": []}),
            }
        ]

    class _StubAdapter:
        def invoke(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
            return _StubResponse()

    captured_kwargs: list[dict] = []

    def _fake_get_chat_model(*_args, **kwargs):  # noqa: ANN002
        captured_kwargs.append(dict(kwargs.get("model_kwargs") or {}))
        return _StubAdapter()

    monkeypatch.setattr("mechanistic_agent.core.baseline_runner.get_chat_model", _fake_get_chat_model)
    monkeypatch.setattr("mechanistic_agent.core.baseline_runner.adapter_supports_forced_tools", lambda _model: True)

    runner = BaselineRunner()
    runner.run_case(
        starting_materials=["CCBr"],
        products=["CCCl"],
        model="anthropic/claude-opus-4.6",
        thinking_level="max",
        llm_seed=None,
    )

    assert captured_kwargs
    kwargs = captured_kwargs[-1]
    assert kwargs.get("effort") == "xhigh"
    assert "thinking" not in kwargs
