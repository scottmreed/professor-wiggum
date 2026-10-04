#!/usr/bin/env python3
"""Blind-subagent responder for the keyless agent bridge, plus the responder-integrity audit.

An orchestrator (for example a Claude Code session) answers each agent-bridge
request with a fresh, blind subagent that may read only that call's
``prompt.md`` and write only its ``answer.json``. This tool does the file work
around those subagents and then audits their transcripts, so a result produced
by a subagent that looked anywhere else (the repo, ``training_data/``, the DB,
the web) is caught and disregarded.

Layout (``--bridge-dir`` or ``MECHANISTIC_AGENT_BRIDGE_DIR``; calls dir is the
sibling ``calls/`` unless ``--calls-dir`` is given)::

    <root>/bridge/requests/<stem>.json     written by the harness
    <root>/bridge/responses/<stem>.json    written by ``respond``
    <root>/calls/<stem>/prompt.md          ``model_input`` only (``prep``)
    <root>/calls/<stem>/answer.json        written by the subagent

Subcommands::

    pending                 stems still waiting for a response
    prep STEM...            write calls/<stem>/prompt.md from model_input only
    prompt STEM             print the standard blind-subagent prompt for a call
    respond STEM...         validate answer.json against the tool schema and respond
    cycle                   one pass: respond to answers, prep new requests, print DISPATCH lines
    serve                   run ``cycle`` forever (put it under a Monitor)
    audit --transcripts ... classify every subagent tool call: clean / procedural / contaminated

Standard library only (``audit --mark`` additionally imports the repo's RunStore).
See ``.claude/skills/bridge-responder/SKILL.md`` and ``docs/agent_bridge.md``.
"""
from __future__ import annotations

import argparse
import contextlib
import glob as globlib
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

BRIDGE_DIR_ENV = "MECHANISTIC_AGENT_BRIDGE_DIR"
AUDIT_SCHEMA = "mechanistic.bridge_responder_audit@1"
PROMPT_FILE = "prompt.md"
ANSWER_FILE = "answer.json"
DISPATCHED_FILE = "dispatched"
REJECTED_FILE = "rejected_transcripts.txt"
DEFAULT_MAX_ATTEMPTS = 3

CLEAN, PROCEDURAL, UNAUDITED, CONTAMINATED = "clean", "procedural", "unaudited", "contaminated"
_SEVERITY = {CLEAN: 0, PROCEDURAL: 1, UNAUDITED: 2, CONTAMINATED: 3}
ALLOWED, PROC, CONTAM = "allowed", "procedural", "contamination"

SUBAGENT_PROMPT = """You are acting as a language model answering one API request. Read ONLY this file: {prompt}

It contains the system/user messages and the JSON schema of the tool you must answer with. Follow those instructions as if given to you directly, reasoning from your own chemistry knowledge.

Rules:
- Use the Read tool on that one file. Do not read, list, search or open any other file or directory.
- Do not run code or shell commands (no Bash, no python, not even to check your JSON). Do not search or fetch the web. Do not start other agents or use any other tool.
- Write your answer with the Write tool to: {answer}
  It must be a single JSON object: the tool's `arguments`, conforming to the tool's parameters schema, with every required field. Valid JSON only, no comments or markdown fences.
- Reply "done" when it is written.

Your tool calls are audited after the run. Any other file access or tool use voids your answer."""


# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------


def resolve_bridge_dir(bridge_dir: Optional[str]) -> Path:
    target = bridge_dir or os.getenv(BRIDGE_DIR_ENV)
    if not target:
        raise SystemExit(f"Set --bridge-dir or {BRIDGE_DIR_ENV}.")
    return Path(target).expanduser()


def resolve_calls_dir(bridge: Path, calls_dir: Optional[str]) -> Path:
    return Path(calls_dir).expanduser() if calls_dir else bridge.parent / "calls"


def _real(path: Any) -> str:
    return os.path.realpath(os.path.expanduser(str(path)))


# ---------------------------------------------------------------------------
# Responder side
# ---------------------------------------------------------------------------


def pending(bridge: Path) -> List[str]:
    requests = bridge / "requests"
    if not requests.is_dir():
        return []
    return [req.stem for req in sorted(requests.glob("*.json")) if not (bridge / "responses" / req.name).exists()]


def _load_request(bridge: Path, stem: str) -> Dict[str, Any]:
    return json.loads((bridge / "requests" / f"{stem}.json").read_text(encoding="utf-8"))


def render_prompt(model_input: Dict[str, Any]) -> str:
    """prompt.md text built from ``model_input`` alone (messages, tools, tool_choice)."""
    parts: List[str] = []
    for message in model_input.get("messages") or []:
        role = message.get("role") or message.get("type") or "message"
        content = message.get("content")
        if not isinstance(content, str):
            content = json.dumps(content, indent=2)
        parts.append(f"## {role}\n\n{content}\n")
    tools = model_input.get("tools") or []
    parts.append("## Tool you must answer with\n\n```json\n" + json.dumps(tools, indent=2) + "\n```\n")
    parts.append("## tool_choice\n\n```json\n" + json.dumps(model_input.get("tool_choice"), indent=2) + "\n```\n")
    return "\n".join(parts)


def prep(bridge: Path, calls: Path, stem: str) -> Path:
    """Write ``calls/<stem>/prompt.md``. Only ``model_input`` is copied: never the
    envelope's ``context`` (run/step attribution) or anything else."""
    model_input = _load_request(bridge, stem)["model_input"]
    call_dir = calls / stem
    call_dir.mkdir(parents=True, exist_ok=True)
    (call_dir / PROMPT_FILE).write_text(render_prompt(model_input), encoding="utf-8")
    return call_dir


def subagent_prompt(calls: Path, stem: str) -> str:
    call_dir = calls.resolve() / stem
    return SUBAGENT_PROMPT.format(prompt=call_dir / PROMPT_FILE, answer=call_dir / ANSWER_FILE)


def tool_spec(model_input: Dict[str, Any]) -> Tuple[Optional[str], List[str]]:
    """Forced tool name and its required argument keys."""
    tools = model_input.get("tools") or []
    choice = model_input.get("tool_choice")
    wanted = None
    if isinstance(choice, dict):
        wanted = (choice.get("function") or {}).get("name") or choice.get("name")
    chosen: Dict[str, Any] = {}
    for tool in tools:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        if not chosen or fn.get("name") == wanted:
            chosen = fn
        if fn.get("name") == wanted:
            break
    name = wanted or chosen.get("name")
    required = list((chosen.get("parameters") or {}).get("required") or [])
    return name, required


def load_answer(calls: Path, stem: str, model_input: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Parse and validate ``answer.json``; raise ValueError with the reason when unusable."""
    try:
        answer = json.loads((calls / stem / ANSWER_FILE).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"answer.json is not valid JSON: {exc}") from exc
    if isinstance(answer, dict) and "arguments" in answer and "name" in answer:
        answer = answer["arguments"]
        if isinstance(answer, str):
            answer = json.loads(answer)
    if not isinstance(answer, dict):
        raise ValueError("answer.json must be a JSON object (the tool arguments)")
    name, required = tool_spec(model_input)
    missing = [key for key in required if key not in answer]
    if missing:
        raise ValueError(f"answer missing required keys: {missing}")
    return str(name or ""), answer


def write_bridge_response(
    bridge: Path, stem: str, *, tool_calls: List[Dict[str, Any]], error: Optional[str] = None
) -> Path:
    """Same response shape as ``mechanistic_agent.agent_bridge.write_response``."""
    path = bridge / "responses" / f"{stem}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"tool_calls": tool_calls, "content": ""}
    if error:
        payload["error"] = str(error)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def respond(bridge: Path, calls: Path, stem: str) -> Path:
    model_input = _load_request(bridge, stem)["model_input"]
    name, arguments = load_answer(calls, stem, model_input)
    return write_bridge_response(bridge, stem, tool_calls=[{"name": name, "arguments": arguments}])


def _attempts(call_dir: Path) -> int:
    try:
        return int((call_dir / DISPATCHED_FILE).read_text().strip() or "1")
    except (OSError, ValueError):
        return 0


def _reject_answer(call_dir: Path, label: str) -> None:
    answer = call_dir / ANSWER_FILE
    if answer.exists():
        answer.rename(call_dir / f"answer.{label}.{time.time_ns()}.json")


def cycle(
    bridge: Path,
    calls: Path,
    *,
    transcripts: Optional[Sequence[str]] = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> List[str]:
    """One responder pass. Returns event lines: DISPATCH / BAD / CONTAMINATED / PROCEDURAL / FAILED / RESPONDED.

    With ``transcripts``, an answer whose subagent transcript is already
    contaminated is rejected (and the subagent re-dispatched) instead of being
    handed to the harness. The post-run ``audit`` remains mandatory: a
    transcript that is not on disk yet cannot be checked here.
    """
    events: List[str] = []
    index = TranscriptIndex(transcripts or [], [calls]) if transcripts else None
    for stem in pending(bridge):
        call_dir = calls / stem
        if not (call_dir / PROMPT_FILE).exists():
            prep(bridge, calls, stem)
        if not (call_dir / ANSWER_FILE).exists():
            if not (call_dir / DISPATCHED_FILE).exists():
                (call_dir / DISPATCHED_FILE).write_text("1")
                events.append(f"DISPATCH {stem}")
            continue
        problem: Optional[str] = None
        label = "bad"
        if index is not None:
            rejected = _rejected_ids(call_dir)
            for transcript in index.for_stem(stem):
                if transcript.agent_id in rejected:
                    continue
                result = audit_transcript(transcript)
                if result["verdict"] == CONTAMINATED:
                    with (call_dir / REJECTED_FILE).open("a", encoding="utf-8") as handle:
                        handle.write(transcript.agent_id + "\n")
                    problem = "; ".join(v["reason"] for v in result["violations"] if v["category"] == CONTAM)
                    label = "contaminated"
                    events.append(f"CONTAMINATED {stem} ({transcript.agent_id}): {problem}")
                elif result["verdict"] == PROCEDURAL:
                    reasons = "; ".join(v["reason"] for v in result["violations"])
                    events.append(f"PROCEDURAL {stem} ({transcript.agent_id}): {reasons}")
        if problem is None:
            try:
                respond(bridge, calls, stem)
                events.append(f"RESPONDED {stem}")
                continue
            except (ValueError, OSError) as exc:
                problem = str(exc)
                events.append(f"BAD {stem}: {problem}")
        _reject_answer(call_dir, label)
        attempts = _attempts(call_dir)
        if attempts >= max_attempts:
            write_bridge_response(
                bridge, stem, tool_calls=[], error=f"no usable answer after {attempts} subagent attempt(s): {problem}"
            )
            events.append(f"FAILED {stem}: gave up after {attempts} attempt(s)")
            continue
        (call_dir / DISPATCHED_FILE).write_text(str(attempts + 1))
        events.append(f"DISPATCH {stem}")
    return events


def serve(bridge: Path, calls: Path, *, interval: float, transcripts: Sequence[str], max_attempts: int) -> None:
    print(f"bridge_responder serve: bridge={bridge} calls={calls}", flush=True)
    while True:
        for line in cycle(bridge, calls, transcripts=transcripts, max_attempts=max_attempts):
            print(line, flush=True)
        time.sleep(interval)


# ---------------------------------------------------------------------------
# Audit: subagent transcripts (Claude Code JSONL) -> per-call verdicts
# ---------------------------------------------------------------------------

_PROMPT_PATH_RE = re.compile(r"""(/[^\s'"`<>]+?)/prompt\.md""")


def _iter_records(path: Path) -> Iterable[Dict[str, Any]]:
    try:
        handle = path.open(encoding="utf-8", errors="replace")
    except OSError:
        return
    with handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def _message_text(record: Dict[str, Any]) -> str:
    content = (record.get("message") or {}).get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(block.get("text") or "") for block in content if isinstance(block, dict))
    return ""


def _first_prompt(path: Path) -> str:
    for record in _iter_records(path):
        if record.get("type") == "user" and not record.get("isMeta"):
            return _message_text(record)
    return ""


class Transcript:
    """One subagent transcript and the call(s) its first prompt names."""

    def __init__(self, path: Path, call_dirs: List[str]) -> None:
        self.path = path
        self.agent_id = path.stem
        self.call_dirs = call_dirs  # realpaths of calls/<stem>
        self.stems = [os.path.basename(d) for d in call_dirs]

    def tool_uses(self) -> Iterable[Dict[str, Any]]:
        for record in _iter_records(self.path):
            if record.get("type") != "assistant":
                continue
            for block in (record.get("message") or {}).get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    yield block


def expand_transcripts(specs: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for spec in specs:
        spec = os.path.expanduser(spec)
        if os.path.isdir(spec):
            matches = sorted(globlib.glob(os.path.join(spec, "*.output")) + globlib.glob(os.path.join(spec, "*.jsonl")))
        else:
            matches = sorted(globlib.glob(spec)) or ([spec] if os.path.exists(spec) else [])
        out.extend(Path(match) for match in matches)
    seen: set = set()
    unique = []
    for path in out:
        key = _real(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


class TranscriptIndex:
    """Maps call stems to the subagent transcripts that answered them (first prompt names the prompt.md)."""

    _cache: Dict[str, Tuple[float, List[str]]] = {}

    def __init__(self, specs: Sequence[str], calls_dirs: Sequence[Path]) -> None:
        roots = {_real(c) for c in calls_dirs}
        self.transcripts: List[Transcript] = []
        self.unmatched = 0
        self.by_stem: Dict[str, List[Transcript]] = {}
        for path in expand_transcripts(specs):
            key = _real(path)
            try:
                mtime = os.path.getmtime(key)
            except OSError:
                continue
            cached = self._cache.get(key)
            if cached and cached[0] == mtime:
                dirs = cached[1]
            else:
                dirs = []
                for match in _PROMPT_PATH_RE.finditer(_first_prompt(path)):
                    call_dir = _real(match.group(1))
                    if call_dir not in dirs:
                        dirs.append(call_dir)
                self._cache[key] = (mtime, dirs)
            dirs = [d for d in dirs if os.path.dirname(d) in roots]
            if not dirs:
                self.unmatched += 1
                continue
            transcript = Transcript(path, dirs)
            self.transcripts.append(transcript)
            for call_dir in dirs:
                self.by_stem.setdefault(call_dir, []).append(transcript)

    def for_stem(self, stem: str, calls: Optional[Path] = None) -> List[Transcript]:
        if calls is not None:
            return self.by_stem.get(_real(Path(calls) / stem), [])
        return [t for d, ts in self.by_stem.items() if os.path.basename(d) == stem for t in ts]


# -- Bash command analysis ----------------------------------------------------

_SHELL_OK = {"cat", "python", "python3", "echo", "printf", "jq", "head", "tail", "wc", "test", "[", "true", "false", ":"}
_INTERPRETER_RE = re.compile(r"^(python3?(\.\d+)?|jq|cat)$")
_PY_RISK_RE = re.compile(
    r"\b(?:import|from)\s+(os|glob|subprocess|sqlite3|urllib|urllib3|requests|httpx|socket|http|shutil|pathlib|"
    r"importlib|rdkit|mechanistic_agent|pandas|numpy|webbrowser|ftplib|pickle)\b"
    r"|__import__|\bos\.|\bPath\s*\(|\bglob\.|\bsubprocess\b|\burlopen\b|\bexec\s*\(|\beval\s*\(|\bscandir\b|\blistdir\b"
)
_PY_MODULE_FLAG_RE = re.compile(r"\s-m\s+([\w.]+)")
_HEREDOC_RE = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_]\w*)\1")
_BOUNDARY = r"(?<![^\s'\"(=<>`,])"
_ABS_PATH_RE = re.compile(_BOUNDARY + r"(/[\w.\-~+%@]+(?:/[\w.\-~+%@]*)+)")
_HOME_RE = re.compile(_BOUNDARY + r"(~/\S*|\$\{?(?:HOME|PWD|OLDPWD|TMPDIR|USER)\b\S*)")
_REL_PATH_RE = re.compile(_BOUNDARY + r"((?:\.\.?/|[\w.\-]+/)[\w.\-/]*)")
_PATHY_EXT_RE = re.compile(r"\.(?:json|jsonl|db|sqlite3?|py|md|txt|csv|tsv|gz|npz|ya?ml|toml|log|output)$")
# Repo / data-checkout directory names a relative path could reach the reference mechanism through.
_SENSITIVE_DIRS = {
    "training_data", "traces", "results", "data", "wiggum-data", "skills", "mechanistic_agent", "harness_versions",
    "calls", "bridge", "scripts", "tests", "docs", "novelty_index", "templates", "requests", "responses",
}
_DEVICE_PATHS = {"/dev/null", "/dev/stdin", "/dev/stdout", "/dev/stderr"}
_QUOTED_RE = re.compile(r"'[^']*'|\"(?:\\.|[^\"\\])*\"", re.S)
_BARE_FILE_RE = re.compile(
    r"\b([\w\-]+\.(?:json|jsonl|db|sqlite3?|py|md|txt|csv|tsv|gz|npz|ya?ml|toml|log|output|smi|sdf|mol))\b"
)
_OPEN_RE = re.compile(r"\bopen\s*\(\s*(['\"])(.*?)\1")
_SEGMENT_SPLIT_RE = re.compile(r"\|\||&&|[;|\n()`]|\$\(")


def _split_heredocs(command: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Return (shell text without heredoc bodies, [(kind, body)]) where kind is python|data."""
    shell_lines: List[str] = []
    bodies: List[Tuple[str, str]] = []
    lines = command.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        shell_lines.append(line)
        match = _HEREDOC_RE.search(line)
        i += 1
        if not match:
            continue
        delimiter = match.group(2)
        body: List[str] = []
        while i < len(lines) and lines[i].strip() != delimiter:
            body.append(lines[i])
            i += 1
        i += 1  # skip the delimiter line
        kind = "python" if re.search(r"\bpython[\d.]*\b", line[: match.start()]) else "data"
        bodies.append((kind, "\n".join(body)))
    return "\n".join(shell_lines), bodies


def _pathy_relative(token: str) -> bool:
    """A relative token that looks like a file path (not prose such as "and/or" or a SMILES "C/C")."""
    if token.startswith(("./", "../")):
        return True
    segments = [seg for seg in token.split("/") if seg]
    if not segments:
        return False
    if any(seg in _SENSITIVE_DIRS for seg in segments):
        return True
    return (token.count("/") >= 2 and len(segments) >= 2) or bool(_PATHY_EXT_RE.search(segments[-1]))


def analyse_bash(command: str, call_paths: Dict[str, set]) -> Tuple[str, str]:
    """Classify a Bash command: (procedural|contamination, reason)."""
    allowed_paths = call_paths["prompt"] | call_paths["answer"] | call_paths["call_dir"]
    shell_text, bodies = _split_heredocs(command)
    problems: List[str] = []

    # 1) Command words: only a small set of harmless commands may run. Quoted
    #    strings (python -c code, echo text) are not commands.
    exempt: set = set(_DEVICE_PATHS)
    commands: List[str] = []
    for segment in _SEGMENT_SPLIT_RE.split(_QUOTED_RE.sub(" Q ", shell_text)):
        words = segment.strip().split()
        while words and (re.match(r"^\w+=", words[0]) or words[0] in {"!", "{", "}", "then", "do", "else", "fi"}):
            words = words[1:]
        if not words or words[0].startswith((">", "<", "2>", "&")):
            continue
        word = words[0]
        base = os.path.basename(word)
        commands.append(base)
        if "/" in word and _INTERPRETER_RE.match(base):
            exempt.add(word)
        if base not in _SHELL_OK and not _INTERPRETER_RE.match(base):
            problems.append(f"runs `{base}`")
    for module in _PY_MODULE_FLAG_RE.findall(shell_text):
        if module != "json.tool":
            problems.append(f"runs python module `{module}`")

    # 2) Python code (python -c in the shell text, or a heredoc fed to python).
    code_texts = [shell_text] + [body for kind, body in bodies if kind == "python"]
    for code in code_texts:
        for match in _PY_RISK_RE.finditer(code):
            problems.append(f"python uses `{match.group(0).strip()}`")
        for match in _OPEN_RE.finditer(code):
            target = match.group(2)
            if target.startswith(("/", "~")):
                ok = _real(target) in allowed_paths
            else:  # relative: only a bare prompt.md / answer.json name is harmless
                ok = target in (PROMPT_FILE, ANSWER_FILE)
            if not ok:
                problems.append(f"opens `{target}`")

    # 3) Paths anywhere in the command (heredoc bodies included).
    for match in _ABS_PATH_RE.finditer(command):
        token = match.group(1).rstrip(".,:")
        if token in exempt:
            continue
        first = token.strip("/").split("/")[0]
        if len(first) < 3 and not first[:1].islower():
            continue  # SMILES-like fragment such as /C/C
        if _real(token) not in allowed_paths:
            problems.append(f"touches `{token}`")
    for match in _HOME_RE.finditer(shell_text):
        problems.append(f"touches `{match.group(1)}`")
    scrubbed_texts = [shell_text] + [body for kind, body in bodies if kind == "python"]
    for text in scrubbed_texts:
        text = _ABS_PATH_RE.sub(" ", text)
        for match in _REL_PATH_RE.finditer(text):
            if _pathy_relative(match.group(1)):
                problems.append(f"touches relative path `{match.group(1)}`")
        for match in _BARE_FILE_RE.finditer(text):
            if match.group(1) not in (PROMPT_FILE, ANSWER_FILE):
                problems.append(f"names file `{match.group(1)}`")

    if problems:
        unique = list(dict.fromkeys(problems))
        return CONTAM, "Bash " + "; ".join(unique[:6])
    used = ", ".join(dict.fromkeys(commands)) or "shell"
    return PROC, f"Bash ({used}) touching only this call's prompt/answer"


# -- tool-use classification ----------------------------------------------------

_ALLOWED_TOOLS = {"SubagentHandback"}
_PROCEDURAL_TOOLS = {"ToolSearch", "TodoWrite"}


def classify_tool_use(name: str, tool_input: Dict[str, Any], call_paths: Dict[str, set]) -> Tuple[str, str]:
    """Classify one subagent tool call as allowed / procedural / contamination, with a reason."""
    tool_input = tool_input if isinstance(tool_input, dict) else {}
    prompts, answers, call_dirs = call_paths["prompt"], call_paths["answer"], call_paths["call_dir"]

    def target(*keys: str) -> Optional[str]:
        for key in keys:
            value = tool_input.get(key)
            if value:
                return _real(value)
        return None

    if name in _ALLOWED_TOOLS:
        return ALLOWED, f"{name}"
    if name in _PROCEDURAL_TOOLS:
        return PROC, f"used {name}"
    if name == "Read":
        path = target("file_path", "path")
        if path in prompts:
            return ALLOWED, "read prompt.md"
        if path in answers:
            return PROC, "re-read its own answer.json"
        return CONTAM, f"Read `{tool_input.get('file_path') or tool_input.get('path')}`"
    if name == "Write":
        path = target("file_path", "path")
        if path in answers:
            return ALLOWED, "wrote answer.json"
        return CONTAM, f"Write `{tool_input.get('file_path')}`"
    if name in {"Edit", "MultiEdit", "NotebookEdit"}:
        path = target("file_path", "notebook_path", "path")
        if path in answers:
            return PROC, f"{name} of its own answer.json"
        return CONTAM, f"{name} `{tool_input.get('file_path') or tool_input.get('notebook_path')}`"
    if name == "Grep":
        path = target("path")
        if path and (path in prompts or path in answers):
            return PROC, "grepped its own prompt/answer"
        return CONTAM, f"Grep `{tool_input.get('pattern')}` in `{tool_input.get('path') or '<cwd>'}`"
    if name == "Glob":
        path = target("path")
        if path and path in call_dirs and ".." not in str(tool_input.get("pattern") or ""):
            return PROC, "globbed its own call directory"
        return CONTAM, f"Glob `{tool_input.get('pattern')}` in `{tool_input.get('path') or '<cwd>'}`"
    if name == "Bash":
        return analyse_bash(str(tool_input.get("command") or ""), call_paths)
    if name in {"WebFetch", "WebSearch"}:
        return CONTAM, f"{name} `{tool_input.get('url') or tool_input.get('query')}`"
    if name in {"Agent", "Task"}:
        return CONTAM, f"spawned a subagent ({name})"
    if name.startswith("mcp__"):
        return CONTAM, f"called MCP tool `{name}`"
    return CONTAM, f"used tool `{name}`"


def _call_paths(call_dirs: Sequence[str]) -> Dict[str, set]:
    return {
        "prompt": {_real(os.path.join(d, PROMPT_FILE)) for d in call_dirs},
        "answer": {_real(os.path.join(d, ANSWER_FILE)) for d in call_dirs},
        "call_dir": {_real(d) for d in call_dirs},
    }


def _worst(statuses: Iterable[str], default: str = CLEAN) -> str:
    worst = default
    for status in statuses:
        if _SEVERITY.get(status, 0) > _SEVERITY.get(worst, 0):
            worst = status
    return worst


def audit_transcript(transcript: Transcript) -> Dict[str, Any]:
    paths = _call_paths(transcript.call_dirs)
    violations: List[Dict[str, Any]] = []
    tool_counts: Dict[str, int] = {}
    wrote_answer = False
    for block in transcript.tool_uses():
        name = str(block.get("name") or "")
        tool_input = block.get("input") or {}
        tool_counts[name] = tool_counts.get(name, 0) + 1
        category, reason = classify_tool_use(name, tool_input, paths)
        if name == "Write" and category == ALLOWED:
            wrote_answer = True
        if category != ALLOWED:
            violations.append({"tool": name, "category": category, "reason": reason})
    if any(v["category"] == CONTAM for v in violations):
        verdict = CONTAMINATED
    elif violations:
        verdict = PROCEDURAL
    else:
        verdict = CLEAN
    return {
        "agent_id": transcript.agent_id,
        "path": str(transcript.path),
        "verdict": verdict,
        "violations": violations,
        "tool_counts": tool_counts,
        "wrote_answer_with_write_tool": wrote_answer,
    }


def _rejected_ids(call_dir: Path) -> set:
    try:
        return {line.strip() for line in (call_dir / REJECTED_FILE).read_text().splitlines() if line.strip()}
    except OSError:
        return set()


def _bridge_for(calls: Path, explicit: Optional[Path], n_calls_dirs: int) -> Optional[Path]:
    if explicit is not None and n_calls_dirs == 1:
        return explicit
    sibling = calls.parent / "bridge"
    return sibling if sibling.is_dir() else explicit


def _response_state(bridge: Optional[Path], stem: str) -> Optional[str]:
    if bridge is None:
        return None
    path = bridge / "responses" / f"{stem}.json"
    if not path.exists():
        return None
    try:
        return "error" if json.loads(path.read_text(encoding="utf-8")).get("error") else "answered"
    except (OSError, json.JSONDecodeError, AttributeError):
        return "answered"


def _request_context(bridge: Optional[Path], stem: str) -> Dict[str, Any]:
    if bridge is None:
        return {}
    try:
        context = _load_request(bridge, stem).get("context")
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(context) if isinstance(context, dict) else {}


def _eval_refs_readonly(db: Optional[Path], run_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """run_id -> {eval_run_id, case_id} via a read-only sqlite connection (never writes)."""
    if not db or not run_ids or not Path(db).exists():
        return {}
    refs: Dict[str, Dict[str, Any]] = {}
    try:
        conn = sqlite3.connect(f"file:{Path(db).resolve()}?mode=ro", uri=True)
    except sqlite3.Error:
        return {}
    try:
        for run_id in run_ids:
            row = conn.execute(
                "SELECT eval_run_id, case_id FROM eval_run_results WHERE run_id = ? LIMIT 1", (run_id,)
            ).fetchone()
            if row:
                refs[run_id] = {"eval_run_id": row[0], "case_id": row[1]}
    except sqlite3.Error:
        return refs
    finally:
        conn.close()
    return refs


def audit(
    calls_dirs: Sequence[Path],
    transcript_specs: Sequence[str],
    *,
    bridge_dir: Optional[Path] = None,
    db: Optional[Path] = None,
) -> Dict[str, Any]:
    """Audit every answered call under ``calls_dirs`` against the subagent transcripts."""
    index = TranscriptIndex(transcript_specs, calls_dirs)
    calls_out: List[Dict[str, Any]] = []
    for calls in calls_dirs:
        bridge = _bridge_for(calls, bridge_dir, len(calls_dirs))
        stems = sorted(p.name for p in Path(calls).iterdir() if p.is_dir()) if Path(calls).is_dir() else []
        for stem in stems:
            call_dir = Path(calls) / stem
            transcripts = index.for_stem(stem, calls)
            response = _response_state(bridge, stem)
            answered = response == "answered" or (call_dir / ANSWER_FILE).exists()
            if not transcripts and not answered:
                continue  # never dispatched / still pending
            rejected = _rejected_ids(call_dir)
            results = [audit_transcript(t) for t in transcripts]
            counted = [r for r in results if r["agent_id"] not in rejected]
            if counted:
                verdict = _worst(r["verdict"] for r in counted)
            else:
                verdict = UNAUDITED if answered else CLEAN
            violations = [
                {**v, "agent_id": r["agent_id"]} for r in counted for v in r["violations"]
            ]
            calls_out.append(
                {
                    "stem": stem,
                    "calls_dir": str(calls),
                    "verdict": verdict,
                    "answered": answered,
                    "response": response,
                    "context": _request_context(bridge, stem),
                    "transcripts": results,
                    "rejected_transcripts": sorted(rejected),
                    "violations": violations,
                }
            )
    run_ids = sorted({c["context"]["run_id"] for c in calls_out if c["context"].get("run_id")})
    refs = _eval_refs_readonly(db, run_ids)
    for call in calls_out:
        ref = refs.get(str(call["context"].get("run_id") or ""))
        if ref:
            call["context"] = {**call["context"], **ref}
    summary = {status: sum(1 for c in calls_out if c["verdict"] == status) for status in _SEVERITY}
    summary.update(
        {
            "calls": len(calls_out),
            "transcripts_matched": len(index.transcripts),
            "transcripts_unmatched": index.unmatched,
            "attributed_runs": run_ids,
            "unattributed_calls": sum(1 for c in calls_out if not c["context"].get("run_id")),
        }
    )
    return {"schema": AUDIT_SCHEMA, "audited_at": time.time(), "calls": calls_out, "summary": summary}


def integrity_by_run(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Fold call verdicts into one ``responder_integrity`` record per run id."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for call in report["calls"]:
        run_id = call["context"].get("run_id")
        if run_id:
            grouped.setdefault(str(run_id), []).append(call)
    out: Dict[str, Dict[str, Any]] = {}
    for run_id, calls in grouped.items():
        out[run_id] = _integrity_record(calls, report["audited_at"])
    return out


def _integrity_record(calls: List[Dict[str, Any]], audited_at: float) -> Dict[str, Any]:
    counts = {status: sum(1 for c in calls if c["verdict"] == status) for status in _SEVERITY}
    violations = [
        {"stem": c["stem"], "step_name": c["context"].get("step_name"), "tool": v["tool"],
         "category": v["category"], "reason": v["reason"][:300]}
        for c in calls for v in c["violations"]
    ]
    return {
        "status": _worst(c["verdict"] for c in calls),
        "audited_at": audited_at,
        "auditor": "scripts/bridge_responder.py audit",
        "calls_audited": len(calls),
        "calls_by_verdict": counts,
        "violations": violations[:50],
    }


def mark(report: Dict[str, Any], db: Optional[Path]) -> Dict[str, Any]:
    """Write ``responder_integrity`` onto affected runs (config.origin) and their eval runs (metadata)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mechanistic_agent.core.db import RunStore
    from mechanistic_agent.data_paths import db_path

    store = RunStore(Path(db) if db else db_path())
    per_run = integrity_by_run(report)
    marked_runs = [run_id for run_id, record in per_run.items() if store.set_run_responder_integrity(run_id, record)]
    by_eval: Dict[str, List[Dict[str, Any]]] = {}
    for ref in store.eval_refs_for_runs(marked_runs):
        calls = [c for c in report["calls"] if c["context"].get("run_id") == ref["run_id"]]
        by_eval.setdefault(str(ref["eval_run_id"]), []).extend(calls)
    marked_evals = [
        eval_run_id
        for eval_run_id, calls in by_eval.items()
        if store.set_eval_run_responder_integrity(eval_run_id, _integrity_record(calls, report["audited_at"]))
    ]
    return {
        "runs": {run_id: per_run[run_id]["status"] for run_id in marked_runs},
        "eval_runs": marked_evals,
        "missing_runs": sorted(set(per_run) - set(marked_runs)),
    }


def format_table(report: Dict[str, Any]) -> str:
    lines = [f"{'call':<30} {'verdict':<13} {'run/step':<34} {'transcripts':<11} detail"]
    for call in report["calls"]:
        stem = call["stem"]
        short = f"{stem[:20][-8:]}-{stem[21:29]}" if len(stem) > 29 else stem
        ctx = call["context"]
        run_step = f"{str(ctx.get('run_id') or '-')[:12]}/{ctx.get('step_name') or '-'}"
        if ctx.get("case_id"):
            run_step += f" [{ctx['case_id']}]"
        detail = "; ".join(dict.fromkeys(v["reason"] for v in call["violations"]))
        if call["verdict"] == UNAUDITED:
            detail = "answered but no matching subagent transcript"
        lines.append(f"{short:<30} {call['verdict']:<13} {run_step[:34]:<34} {len(call['transcripts']):<11} {detail[:160]}")
    s = report["summary"]
    lines.append(
        f"summary: {s['calls']} calls - clean {s[CLEAN]}, procedural {s[PROCEDURAL]}, "
        f"contaminated {s[CONTAMINATED]}, unaudited {s[UNAUDITED]}; transcripts matched {s['transcripts_matched']}, "
        f"unmatched {s['transcripts_unmatched']}; unattributed calls {s['unattributed_calls']}"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    # SUPPRESS defaults so the options work before or after the subcommand.
    common.add_argument("--bridge-dir", default=argparse.SUPPRESS,
                        help=f"bridge exchange dir (default: ${BRIDGE_DIR_ENV})")
    common.add_argument("--calls-dir", action="append", default=argparse.SUPPRESS,
                        help="calls dir (default: <bridge-dir>/../calls); audit accepts several")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0], parents=[common])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pending", parents=[common])
    p_prep = sub.add_parser("prep", parents=[common])
    p_prep.add_argument("stems", nargs="+")
    p_prompt = sub.add_parser("prompt", parents=[common])
    p_prompt.add_argument("stem")
    p_respond = sub.add_parser("respond", parents=[common])
    p_respond.add_argument("stems", nargs="+")
    for name in ("cycle", "serve"):
        p = sub.add_parser(name, parents=[common])
        p.add_argument("--transcripts", action="append", default=[],
                       help="subagent transcript dir/glob: reject answers whose transcript is already contaminated")
        p.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
        if name == "serve":
            p.add_argument("--interval", type=float, default=3.0)
    p_audit = sub.add_parser("audit", parents=[common])
    p_audit.add_argument("--transcripts", action="append", required=True,
                         help="subagent transcript dir (*.output / *.jsonl), glob or file; repeatable")
    p_audit.add_argument("--json", dest="json_out", help="also write the full report to this file")
    p_audit.add_argument("--db", help="SQLite DB (default: data_paths.db_path()); read-only unless --mark")
    p_audit.add_argument("--mark", action="store_true",
                         help="record responder_integrity on the audited runs and their eval runs")
    args = parser.parse_args(argv)
    args.bridge_dir = getattr(args, "bridge_dir", None)
    args.calls_dir = getattr(args, "calls_dir", None)

    def dirs() -> Tuple[Path, Path]:
        bridge = resolve_bridge_dir(args.bridge_dir)
        return bridge, resolve_calls_dir(bridge, (args.calls_dir or [None])[0])

    if args.cmd == "pending":
        print("\n".join(pending(dirs()[0])))
    elif args.cmd == "prep":
        bridge, calls = dirs()
        for stem in args.stems:
            print(prep(bridge, calls, stem))
    elif args.cmd == "prompt":
        calls = Path(args.calls_dir[0]).expanduser() if args.calls_dir else dirs()[1]
        print(subagent_prompt(calls, args.stem))
    elif args.cmd == "respond":
        bridge, calls = dirs()
        status = 0
        for stem in args.stems:
            try:
                print(f"responded {stem} -> {respond(bridge, calls, stem).name}")
            except (ValueError, OSError) as exc:
                print(f"BAD {stem}: {exc}")
                status = 1
        return status
    elif args.cmd == "cycle":
        bridge, calls = dirs()
        for line in cycle(bridge, calls, transcripts=args.transcripts, max_attempts=args.max_attempts):
            print(line)
    elif args.cmd == "serve":
        bridge, calls = dirs()
        with contextlib.suppress(KeyboardInterrupt):
            serve(bridge, calls, interval=args.interval, transcripts=args.transcripts, max_attempts=args.max_attempts)
    elif args.cmd == "audit":
        # The env bridge dir only applies when no calls dir is named; with explicit
        # --calls-dir each one pairs with --bridge-dir or its sibling bridge/.
        explicit_bridge = Path(args.bridge_dir).expanduser() if args.bridge_dir else None
        if explicit_bridge is None and not args.calls_dir and os.getenv(BRIDGE_DIR_ENV):
            explicit_bridge = Path(os.environ[BRIDGE_DIR_ENV]).expanduser()
        if args.calls_dir:
            calls_dirs = [Path(c).expanduser() for c in args.calls_dir]
        elif explicit_bridge is not None:
            calls_dirs = [resolve_calls_dir(explicit_bridge, None)]
        else:
            raise SystemExit("audit needs --calls-dir or --bridge-dir")
        db = Path(args.db).expanduser() if args.db else None
        if db is None:
            with contextlib.suppress(Exception):
                sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
                from mechanistic_agent.data_paths import db_path

                db = db_path()
        report = audit(calls_dirs, args.transcripts, bridge_dir=explicit_bridge, db=db)
        if args.mark:
            report["marked"] = mark(report, db)
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(format_table(report))
        if args.mark:
            marked = report["marked"]
            print(f"marked {len(marked['runs'])} run(s), {len(marked['eval_runs'])} eval run(s); "
                  f"runs not found: {len(marked['missing_runs'])}")
        return 1 if report["summary"][CONTAMINATED] else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
