"""Keyless "agent bridge" LLM provider.

This provider lets the harness run with an **external agent or subagent acting
as the model**, instead of a hosted provider API key. It is selected through the
model catalog (``provider: "agent_bridge"``) and routed by
:func:`mechanistic_agent.llm.get_chat_model`.

Why it exists
-------------
The whole runtime reaches a model through exactly one seam —
``get_chat_model(model).invoke(messages, tools=..., tool_choice=...)``. A hosted
adapter (OpenAI / Anthropic / Gemini) turns that into an HTTP call. The agent
bridge instead writes the request to a directory and waits for a response file,
so any agent loop that cannot expose an API key (for example an orchestrator
that answers via spawned subagents) can drive the harness and contribute.

Privacy contract (enforced by tests/fast/test_agent_bridge.py)
--------------------------------------------------------------
The request handed to the responder contains a ``model_input`` block with
**exactly** the inputs a hosted model would receive for the call:

    model_input = {"messages": [...], "tools": ..., "tool_choice": ...}

``messages`` are serialised with :func:`mechanistic_agent.llm.serialise_chat_messages`
— the *same* helper the OpenAI adapter uses — so the responder sees byte-for-byte
the message view a keyed model would, and nothing else. No run state, eval
ground truth, atom-map context, or scoring information is added here (the harness
already strips privileged context before this seam; see
``test_tool_executor_does_not_forward_raw_mapped_prompt_context``). The envelope
adds only non-task routing metadata (``schema``, ``request_id``, ``model``).

Protocol
--------
Bridge directory (``MECHANISTIC_AGENT_BRIDGE_DIR``) layout::

    <dir>/requests/<seq>-<uuid>.json     # written by this adapter
    <dir>/responses/<seq>-<uuid>.json    # written by the responder

A responder reads a request, produces a tool call from ``model_input`` alone,
and writes the response. Pre-seeding the matching response file makes runs
deterministic / replayable in CI. If no response arrives within the timeout
(``MECHANISTIC_AGENT_BRIDGE_TIMEOUT`` seconds, default 1800) the adapter raises —
failing loud rather than silently degrading.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .llm import _SimpleMessage, serialise_chat_messages

BRIDGE_DIR_ENV = "MECHANISTIC_AGENT_BRIDGE_DIR"
BRIDGE_TIMEOUT_ENV = "MECHANISTIC_AGENT_BRIDGE_TIMEOUT"
BRIDGE_POLL_ENV = "MECHANISTIC_AGENT_BRIDGE_POLL_SECONDS"

DEFAULT_TIMEOUT_SECONDS = 1800.0
DEFAULT_POLL_SECONDS = 0.2
REQUEST_SCHEMA = "mechanistic.agent_bridge/request@1"

# The only keys allowed inside the model-visible request block. Kept as a module
# constant so the privacy invariant is asserted against a single source of truth.
MODEL_INPUT_KEYS = ("messages", "tools", "tool_choice")


def _resolve_bridge_dir(bridge_dir: Optional[str]) -> Path:
    target = bridge_dir or os.getenv(BRIDGE_DIR_ENV)
    if not target:
        raise RuntimeError(
            "Agent bridge not configured. Set the "
            f"{BRIDGE_DIR_ENV} environment variable to a writable directory "
            "(or pass bridge_dir=) so requests/responses can be exchanged."
        )
    root = Path(target).expanduser()
    (root / "requests").mkdir(parents=True, exist_ok=True)
    (root / "responses").mkdir(parents=True, exist_ok=True)
    return root


def build_model_input(messages: Any, tools: Any, tool_choice: Any) -> Dict[str, Any]:
    """Return the model-visible request block — and nothing else.

    This is the single place that decides what a bridged responder may see. It
    intentionally contains only what a hosted provider would receive.
    """
    return {
        "messages": serialise_chat_messages(messages),
        "tools": tools,
        "tool_choice": tool_choice,
    }


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


class AgentBridgeAdapter:
    """Chat-model adapter that delegates each call to an external agent/subagent.

    Mirrors the ``.invoke(messages, *, tools, tool_choice)`` interface of the
    hosted adapters and returns the same ``_SimpleMessage`` shape.
    """

    def __init__(
        self,
        *,
        model: str,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        bridge_dir: Optional[str] = None,
    ) -> None:
        self._model = model
        # temperature/model_kwargs are recorded for parity with hosted adapters
        # but carry no extra task information to the responder.
        self._temperature = temperature
        self._model_kwargs = dict(model_kwargs or {})
        self._dir = _resolve_bridge_dir(bridge_dir)
        timeout_raw = timeout if timeout is not None else os.getenv(BRIDGE_TIMEOUT_ENV)
        self._timeout = float(timeout_raw) if timeout_raw is not None else DEFAULT_TIMEOUT_SECONDS
        self._poll = float(os.getenv(BRIDGE_POLL_ENV) or DEFAULT_POLL_SECONDS)

    # -- protocol helpers (kept small and individually testable) ---------------

    def _response_path_for(self, request_path: Path) -> Path:
        return self._dir / "responses" / request_path.name

    def _write_request(self, messages: Any, tools: Any, tool_choice: Any) -> Path:
        request_id = uuid.uuid4().hex
        seq = time.time_ns()
        name = f"{seq:020d}-{request_id}.json"
        path = self._dir / "requests" / name
        payload = {
            "schema": REQUEST_SCHEMA,
            "request_id": request_id,
            "model": self._model,
            "model_input": build_model_input(messages, tools, tool_choice),
        }
        _write_json_atomic(path, payload)
        return path

    def _await_response(self, request_path: Path) -> _SimpleMessage:
        response_path = self._response_path_for(request_path)
        deadline = time.monotonic() + self._timeout
        while True:
            if response_path.exists():
                return _parse_response(response_path)
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Agent bridge timed out after "
                    f"{self._timeout:.0f}s waiting for a response at {response_path}. "
                    "No silent degradation: provide a response file or raise the "
                    f"timeout via {BRIDGE_TIMEOUT_ENV}."
                )
            time.sleep(self._poll)

    def invoke(
        self,
        messages: Any,
        config: Any = None,  # noqa: ARG002 - parity with other adapters
        *,
        tools: Any = None,
        tool_choice: Any = None,
    ) -> _SimpleMessage:
        request_path = self._write_request(messages, tools, tool_choice)
        return self._await_response(request_path)


def _parse_response(response_path: Path) -> _SimpleMessage:
    data = json.loads(response_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Agent bridge response {response_path} is not a JSON object")

    raw_tool_calls = data.get("tool_calls") or []
    tool_calls: List[Dict[str, Any]] = []
    for index, call in enumerate(raw_tool_calls):
        if not isinstance(call, dict):
            continue
        arguments = call.get("arguments", "")
        if isinstance(arguments, (dict, list)):
            arguments = json.dumps(arguments)
        tool_calls.append(
            {
                "id": str(call.get("id") or f"bridge-{index}"),
                "name": str(call.get("name") or ""),
                "arguments": arguments if isinstance(arguments, str) else str(arguments),
            }
        )

    content = data.get("content") or ""
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
    return _SimpleMessage(str(content), tool_calls=tool_calls, usage=usage)


# ---------------------------------------------------------------------------
# Responder-side helpers (used by external drivers / subagent loops and by the
# reference demonstration). These never read anything beyond the request file.
# ---------------------------------------------------------------------------


def pending_requests(bridge_dir: Optional[str] = None) -> List[Path]:
    """Return request files that do not yet have a response, oldest first."""
    root = _resolve_bridge_dir(bridge_dir)
    out: List[Path] = []
    for req in sorted((root / "requests").glob("*.json")):
        if not (root / "responses" / req.name).exists():
            out.append(req)
    return out


def read_request(request_path: str | Path) -> Dict[str, Any]:
    """Load a request file. The responder should use only its ``model_input``."""
    return json.loads(Path(request_path).read_text(encoding="utf-8"))


def write_response(
    request_path: str | Path,
    *,
    tool_calls: List[Dict[str, Any]],
    content: str = "",
    usage: Optional[Dict[str, Any]] = None,
    bridge_dir: Optional[str] = None,
) -> Path:
    """Write the responder's answer next to the request, matching by basename."""
    request_path = Path(request_path)
    root = _resolve_bridge_dir(bridge_dir) if bridge_dir else request_path.parent.parent
    response_path = root / "responses" / request_path.name
    response_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"tool_calls": tool_calls, "content": content}
    if usage is not None:
        payload["usage"] = usage
    _write_json_atomic(response_path, payload)
    return response_path


__all__ = [
    "AgentBridgeAdapter",
    "build_model_input",
    "pending_requests",
    "read_request",
    "write_response",
    "BRIDGE_DIR_ENV",
    "BRIDGE_TIMEOUT_ENV",
    "MODEL_INPUT_KEYS",
    "REQUEST_SCHEMA",
]
