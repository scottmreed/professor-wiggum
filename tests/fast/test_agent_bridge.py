"""Tests for the keyless agent-bridge LLM provider.

The most important guarantee here is the **privacy contract**: a bridged
responder (an external agent/subagent standing in for the model) must receive
exactly the inputs a hosted model would for the call — the chat messages, the
tool schema, and the forced tool_choice — and nothing else.
"""
from __future__ import annotations

import json
import threading
import time

import pytest

from mechanistic_agent.agent_bridge import (
    MODEL_INPUT_KEYS,
    REQUEST_SCHEMA,
    AgentBridgeAdapter,
    pending_requests,
    read_request,
    write_response,
)
from mechanistic_agent.llm import (
    get_chat_model,
    get_model_api_key,
    get_provider_label,
    serialise_chat_messages,
)

SAMPLE_MESSAGES = [
    {"role": "system", "content": "You are proposing one elementary mechanism step."},
    {"role": "user", "content": "Starting: CCBr, [I-]. Target: CCI, [Br-]."},
]
SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "mechanism_step_proposal_result",
            "description": "Return one elementary step.",
            "parameters": {"type": "object", "properties": {"intermediate_smiles": {"type": "string"}}},
        },
    }
]
SAMPLE_TOOL_CHOICE = {"type": "function", "function": {"name": "mechanism_step_proposal_result"}}


def _adapter(tmp_path, **kwargs):
    return AgentBridgeAdapter(model="agent-bridge", bridge_dir=str(tmp_path), **kwargs)


def test_request_exposes_only_model_visible_inputs(tmp_path) -> None:
    """The request must contain exactly the model-visible inputs and nothing else."""
    adapter = _adapter(tmp_path)
    request_path = adapter._write_request(SAMPLE_MESSAGES, SAMPLE_TOOLS, SAMPLE_TOOL_CHOICE)
    payload = read_request(request_path)

    # Envelope holds only non-task routing metadata + the model_input block.
    assert set(payload.keys()) == {"schema", "request_id", "model", "model_input"}
    assert payload["schema"] == REQUEST_SCHEMA
    assert payload["model"] == "agent-bridge"

    # The model-visible block is EXACTLY messages/tools/tool_choice — no run
    # state, ground truth, atom maps, or scoring hints may be smuggled in.
    model_input = payload["model_input"]
    assert set(model_input.keys()) == set(MODEL_INPUT_KEYS)
    assert model_input["tools"] == SAMPLE_TOOLS
    assert model_input["tool_choice"] == SAMPLE_TOOL_CHOICE


def test_messages_serialised_identically_to_hosted_adapter(tmp_path) -> None:
    """Bridged messages must match the exact serialisation a keyed model receives."""
    adapter = _adapter(tmp_path)
    request_path = adapter._write_request(SAMPLE_MESSAGES, SAMPLE_TOOLS, SAMPLE_TOOL_CHOICE)
    payload = read_request(request_path)
    assert payload["model_input"]["messages"] == serialise_chat_messages(SAMPLE_MESSAGES)


def test_replay_response_is_parsed_into_tool_calls(tmp_path) -> None:
    """A pre-seeded response is parsed into the standard tool_calls shape."""
    adapter = _adapter(tmp_path)
    request_path = adapter._write_request(SAMPLE_MESSAGES, SAMPLE_TOOLS, SAMPLE_TOOL_CHOICE)
    write_response(
        request_path,
        tool_calls=[{"name": "mechanism_step_proposal_result", "arguments": {"intermediate_smiles": "CCI"}}],
    )
    message = adapter._await_response(request_path)
    assert len(message.tool_calls) == 1
    call = message.tool_calls[0]
    assert call["name"] == "mechanism_step_proposal_result"
    # arguments must be a JSON *string* (object inputs are encoded) so the
    # existing tools.py parsing path (json.loads) works unchanged.
    assert call["arguments"] == json.dumps({"intermediate_smiles": "CCI"})


def test_full_invoke_roundtrip_via_responder_thread(tmp_path) -> None:
    """The public invoke() path: a concurrent responder answers the request."""
    adapter = _adapter(tmp_path, timeout=5.0)

    def _responder():
        for _ in range(100):
            reqs = pending_requests(str(tmp_path))
            if reqs:
                model_input = read_request(reqs[0])["model_input"]
                # Responder answers using only model_input — proves sufficiency.
                assert "messages" in model_input
                write_response(
                    reqs[0],
                    tool_calls=[{"name": "mechanism_step_proposal_result", "arguments": {"ok": True}}],
                )
                return
            time.sleep(0.02)

    t = threading.Thread(target=_responder)
    t.start()
    message = adapter.invoke(SAMPLE_MESSAGES, tools=SAMPLE_TOOLS, tool_choice=SAMPLE_TOOL_CHOICE)
    t.join(timeout=5)
    assert message.tool_calls[0]["arguments"] == json.dumps({"ok": True})


def test_timeout_fails_loud(tmp_path) -> None:
    """No silent degradation: a missing response raises rather than hanging forever."""
    adapter = _adapter(tmp_path, timeout=0.3)
    with pytest.raises(RuntimeError, match="timed out"):
        adapter.invoke(SAMPLE_MESSAGES, tools=SAMPLE_TOOLS, tool_choice=SAMPLE_TOOL_CHOICE)


def test_routing_selects_bridge_without_api_key(tmp_path, monkeypatch) -> None:
    """get_chat_model must return the bridge adapter with no provider key set."""
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MECHANISTIC_AGENT_BRIDGE_DIR", str(tmp_path))
    adapter = get_chat_model("agent-bridge")
    assert isinstance(adapter, AgentBridgeAdapter)


def test_missing_bridge_dir_raises_clearly(monkeypatch) -> None:
    monkeypatch.delenv("MECHANISTIC_AGENT_BRIDGE_DIR", raising=False)
    with pytest.raises(RuntimeError, match="MECHANISTIC_AGENT_BRIDGE_DIR"):
        AgentBridgeAdapter(model="agent-bridge")


def test_bridge_reports_available_without_a_real_key(monkeypatch) -> None:
    """Tools gate on get_model_api_key; the bridge must report a non-empty
    sentinel so they don't short-circuit to a 'no key configured' fallback."""
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    key = get_model_api_key("agent-bridge")
    assert key  # non-empty sentinel, never used as a real credential
    assert get_provider_label("agent-bridge") == "Agent Bridge"


def test_pending_requests_clears_after_response(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    request_path = adapter._write_request(SAMPLE_MESSAGES, SAMPLE_TOOLS, SAMPLE_TOOL_CHOICE)
    assert [p.name for p in pending_requests(str(tmp_path))] == [request_path.name]
    write_response(request_path, tool_calls=[{"name": "x", "arguments": {}}])
    assert pending_requests(str(tmp_path)) == []
