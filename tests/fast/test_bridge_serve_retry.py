"""`bridge-serve` survives responder failures.

A responder that exits non-zero or prints non-JSON once (e.g. a CLI agent that
hit a usage limit) must not end the serve loop: the request is retried with
backoff, other requests keep being served, and a request that runs out of
retries gets an error response so the waiting harness call fails fast instead
of blocking until MECHANISTIC_AGENT_BRIDGE_TIMEOUT.
"""
from __future__ import annotations

import json
import shlex
import sys
import time

import pytest
from typer.testing import CliRunner

import main as cli
from mechanistic_agent.agent_bridge import AgentBridgeAdapter, AgentBridgeResponderError

TOOL = {"type": "function", "function": {"name": "reaction_type_selection_result"}}

# Fails the first time it is called for a request (marker file absent), then answers.
FLAKY_RESPONDER = """
import json, pathlib, sys
request = json.load(sys.stdin)
marker = pathlib.Path(sys.argv[1]) / request["request_id"]
if not marker.exists():
    marker.write_text("failed once")
    print("You've hit your session limit")
    sys.exit(1)
print(json.dumps({"selected_label_exact": "Finkelstein halide exchange"}))
"""

# Fails for any request whose user message says "fail"; answers the rest.
SELECTIVE_RESPONDER = """
import json, sys
request = json.load(sys.stdin)
if "fail" in json.dumps(request["model_input"]["messages"]):
    print("not json at all")
    sys.exit(0)
print(json.dumps({"selected_label_exact": "SN2"}))
"""


def _command(tmp_path, source: str, *args: str) -> str:
    script = tmp_path / "responder.py"
    script.write_text(source)
    return " ".join(shlex.quote(part) for part in (sys.executable, str(script), *args))


def _request(adapter: AgentBridgeAdapter, text: str):
    return adapter._write_request([{"role": "user", "content": text}], [TOOL], TOOL)


def _serve(bridge_dir, command: str, *extra: str):
    return CliRunner().invoke(
        cli.app,
        ["bridge-serve", "--bridge-dir", str(bridge_dir), "--command", command,
         "--poll-seconds", "0.01", *extra],
    )


def test_responder_failing_once_is_retried_and_answered(tmp_path) -> None:
    bridge_dir = tmp_path / "bridge"
    markers = tmp_path / "markers"
    markers.mkdir()
    adapter = AgentBridgeAdapter(model="agent-bridge", bridge_dir=str(bridge_dir))
    req = _request(adapter, "x")

    result = _serve(
        bridge_dir, _command(tmp_path, FLAKY_RESPONDER, str(markers)),
        "--retries", "2", "--retry-wait", "0", "--once",
    )

    assert result.exit_code == 0, result.output
    assert f"responder failed for {req.name} (attempt 1/3)" in result.output
    assert "session limit" in result.output  # stdout detail surfaced when stderr is empty
    assert f"answered {req.name}" in result.output
    message = adapter._await_response(req)
    assert json.loads(message.tool_calls[0]["arguments"]) == {
        "selected_label_exact": "Finkelstein halide exchange"
    }


def test_exhausted_retries_write_error_response_that_fails_fast(tmp_path) -> None:
    bridge_dir = tmp_path / "bridge"
    adapter = AgentBridgeAdapter(model="agent-bridge", bridge_dir=str(bridge_dir), timeout=600)
    req = _request(adapter, "please fail")

    result = _serve(
        bridge_dir, _command(tmp_path, SELECTIVE_RESPONDER),
        "--retries", "1", "--retry-wait", "0", "--once",
    )

    assert result.exit_code == 0, result.output
    assert "attempt 2/2" in result.output and "wrote error response" in result.output
    payload = json.loads((bridge_dir / "responses" / req.name).read_text())
    assert payload["tool_calls"] == [] and "did not emit valid JSON" in payload["error"]
    started = time.monotonic()
    with pytest.raises(AgentBridgeResponderError, match="responder could not answer"):
        adapter._await_response(req)
    assert time.monotonic() - started < 5  # not the 600 s timeout


def test_other_requests_are_served_while_one_backs_off(tmp_path) -> None:
    bridge_dir = tmp_path / "bridge"
    adapter = AgentBridgeAdapter(model="agent-bridge", bridge_dir=str(bridge_dir))
    failing = _request(adapter, "please fail")  # older, so it is tried first
    healthy = _request(adapter, "fine")

    result = _serve(
        bridge_dir, _command(tmp_path, SELECTIVE_RESPONDER),
        "--retries", "3", "--retry-wait", "60", "--max-requests", "1",
    )

    assert result.exit_code == 0, result.output
    assert f"responder failed for {failing.name} (attempt 1/4)" in result.output
    assert "retrying in 60s" in result.output
    assert (bridge_dir / "responses" / healthy.name).exists()
    assert not (bridge_dir / "responses" / failing.name).exists()  # still pending, backing off


# Records the exact stdin payload it was handed, then answers.
RECORDING_RESPONDER = """
import json, pathlib, sys
raw = sys.stdin.read()
pathlib.Path(sys.argv[1]).write_text(raw)
print(json.dumps({"selected_label_exact": "SN2"}))
"""


def test_command_responder_never_sees_request_context(tmp_path) -> None:
    from mechanistic_agent.core.call_recorder import call_context

    bridge_dir = tmp_path / "bridge"
    seen = tmp_path / "seen.json"
    adapter = AgentBridgeAdapter(model="agent-bridge", bridge_dir=str(bridge_dir))
    with call_context(run_id="secret-run-id", step_name="reaction_type_mapping"):
        req = _request(adapter, "x")
    assert json.loads(req.read_text())["context"]["run_id"] == "secret-run-id"

    result = _serve(bridge_dir, _command(tmp_path, RECORDING_RESPONDER, str(seen)), "--once")

    assert result.exit_code == 0, result.output
    handed = json.loads(seen.read_text())
    assert "context" not in handed
    assert "secret-run-id" not in seen.read_text()
    assert handed["model_input"]["tool_choice"] == TOOL
