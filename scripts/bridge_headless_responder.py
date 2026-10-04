#!/usr/bin/env python3
"""``bridge-serve --command`` responder: answer one agent-bridge request with a blind headless ``claude -p``.

    python main.py bridge-serve --command "python scripts/bridge_headless_responder.py"

Reads the request JSON on stdin, renders ONLY ``model_input`` (messages + tool schema; never the
envelope's ``context``) and prints ``{"tool_calls":[{"name":..., "arguments":{...}}]}`` on stdout.

Isolation is fixed in code, not left to whoever launches it (``HEADLESS_ISOLATION_FLAGS``):
``--tools ""`` turns off every built-in tool (no Read/Grep/Bash/WebSearch/WebFetch),
``--restricted`` ignores user, project and local settings files (hooks, plugins, permissions),
``--strict-mcp-config`` with no ``--mcp-config`` loads no MCP server (user or project),
``--no-session-persistence`` keeps nothing between calls, and the process runs in a fresh empty
temporary directory. Nothing the model could use to look up a reference mechanism is reachable.

Environment: ``RESPONDER_MODEL`` (default ``claude-opus-5-5``), ``CLAUDE_BIN`` (default ``claude``),
``RESPONDER_LOG`` (optional JSONL of per-call timings), ``RESPONDER_TIMEOUT`` (seconds, default 900),
``RESPONDER_EFFORT`` (``--effort`` level; match the run's ``--thinking-level``).
The headless CLI shares the account's usage window; on a usage-limit reply it waits for the reset
instead of failing the run. Prefer in-session blind subagents (``.claude/skills/bridge-responder``);
use this only when the user has asked for a headless responder.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bridge_responder import render_prompt, tool_spec  # noqa: E402

HEADLESS_ISOLATION_FLAGS: Tuple[str, ...] = (
    "--tools", "", "--restricted", "--strict-mcp-config", "--no-session-persistence",
)
LIMIT_MARKERS = ("session limit", "usage limit")
MAX_ATTEMPTS = 3
MAX_LIMIT_WAITS = 420  # one per minute: up to 7 h for the usage window to reset

PREAMBLE = (
    "You are the model behind a chemistry-mechanism harness. Below is the exact request the harness "
    "sent you, followed by the tool you must call. Answer from your own knowledge. Respond with ONLY a "
    "single JSON object that is the arguments for the `{name}` tool, conforming to its schema. "
    "No prose, no code fences.\n\n"
)


def build_command(model: str, claude_bin: str = "claude", effort: str = "") -> List[str]:
    """``effort`` (low/medium/high/...) sets the thinking effort; record the same level on the run
    with ``--thinking-level`` so harness and baseline rows compare at equal thinking."""
    command = [claude_bin, "-p", "--model", model, "--output-format", "text", *HEADLESS_ISOLATION_FLAGS]
    if effort:
        command += ["--effort", effort]
    return command


def build_prompt(model_input: Dict[str, Any]) -> Tuple[str, str]:
    name, _required = tool_spec(model_input)
    name = str(name or "")
    return name, PREAMBLE.format(name=name) + render_prompt(model_input)


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in reply")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise ValueError("reply is not a JSON object")
    return obj


def answer(request: Dict[str, Any], *, run=subprocess.run, sleep=time.sleep) -> Dict[str, Any]:
    name, prompt = build_prompt(request["model_input"])
    command = build_command(
        os.environ.get("RESPONDER_MODEL", "claude-opus-5-5"),
        os.environ.get("CLAUDE_BIN", "claude"),
        os.environ.get("RESPONDER_EFFORT", ""),
    )
    timeout = float(os.environ.get("RESPONDER_TIMEOUT", "900"))
    log = os.environ.get("RESPONDER_LOG")
    last_err = ""
    attempt = limit_waits = 0
    while attempt < MAX_ATTEMPTS:
        started = time.time()
        with tempfile.TemporaryDirectory(prefix="bridge-headless-") as cwd:
            proc = run(command, input=prompt, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        try:
            arguments = extract_json(proc.stdout)
        except ValueError as exc:
            last_err = f"{exc}; stdout={proc.stdout[:300]!r} stderr={proc.stderr[:300]!r}"
            if any(marker in proc.stdout for marker in LIMIT_MARKERS) and limit_waits < MAX_LIMIT_WAITS:
                limit_waits += 1
                sleep(60)
                continue
            attempt += 1
            continue
        if log:
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"request_id": request.get("request_id"), "tool": name, "attempt": attempt,
                                     "secs": round(time.time() - started, 1), "prompt_chars": len(prompt)}) + "\n")
        return {"tool_calls": [{"name": name, "arguments": arguments}], "content": ""}
    raise RuntimeError(f"responder failed: {last_err}")


def main() -> int:
    try:
        print(json.dumps(answer(json.load(sys.stdin))))
    except RuntimeError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
