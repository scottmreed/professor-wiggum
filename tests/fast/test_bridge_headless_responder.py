from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("bridge_headless_responder", ROOT / "scripts" / "bridge_headless_responder.py")
hr = importlib.util.module_from_spec(_SPEC)
sys.modules["bridge_headless_responder"] = hr
_SPEC.loader.exec_module(hr)

MODEL_INPUT = {
    "messages": [{"role": "user", "content": "Propose the next step."}],
    "tools": [{"type": "function", "function": {"name": "propose_step", "parameters": {"required": ["steps"]}}}],
    "tool_choice": {"type": "function", "function": {"name": "propose_step"}},
}


def test_command_always_carries_isolation_flags() -> None:
    command = hr.build_command("claude-opus-5-5", "claude")
    assert command[:2] == ["claude", "-p"]
    joined = " ".join(command)
    assert "--strict-mcp-config" in command
    assert "--restricted" in command
    assert "--no-session-persistence" in command
    assert command[command.index("--tools") + 1] == ""
    assert "--mcp-config" not in joined  # strict with no config file = no MCP servers at all


def test_answer_runs_isolated_in_empty_tmp_dir_and_sees_only_model_input(monkeypatch) -> None:
    calls: List[Dict[str, Any]] = []

    def fake_run(command, **kwargs):
        calls.append({"command": command, **kwargs, "cwd_listing": sorted(Path(kwargs["cwd"]).iterdir())})
        return SimpleNamespace(stdout='```json\n{"steps": []}\n```', stderr="", returncode=0)

    monkeypatch.delenv("RESPONDER_LOG", raising=False)
    request = {"request_id": "r1", "model_input": MODEL_INPUT, "context": {"run_id": "SECRET-RUN", "case_id": "flower_1"}}
    out = hr.answer(request, run=fake_run, sleep=lambda s: None)
    assert out["tool_calls"] == [{"name": "propose_step", "arguments": {"steps": []}}]
    (call,) = calls
    assert "--strict-mcp-config" in call["command"]
    assert call["cwd_listing"] == []
    assert "SECRET-RUN" not in call["input"] and "flower_1" not in call["input"]
    assert "Propose the next step." in call["input"]


def test_usage_limit_waits_instead_of_failing(monkeypatch) -> None:
    replies = iter(["You've hit your session limit", '{"steps": [1]}'])
    sleeps: List[float] = []
    out = hr.answer(
        {"model_input": MODEL_INPUT},
        run=lambda command, **kw: SimpleNamespace(stdout=next(replies), stderr="", returncode=0),
        sleep=sleeps.append,
    )
    assert out["tool_calls"][0]["arguments"] == {"steps": [1]}
    assert sleeps == [60]
