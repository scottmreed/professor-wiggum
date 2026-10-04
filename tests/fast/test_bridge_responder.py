"""scripts/bridge_responder.py: blind-subagent responder + responder-integrity audit.

The audit reads Claude Code subagent transcripts (JSONL), maps each to the
bridge call named in its first prompt, and classifies every tool call: reading
the call's prompt.md / writing its answer.json is allowed, touching only those
files another way is procedural, and anything else (repo, DB, web, other calls,
subagents) contaminates the result.
"""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("bridge_responder", ROOT / "scripts" / "bridge_responder.py")
br = importlib.util.module_from_spec(_spec)
sys.modules["bridge_responder"] = br
_spec.loader.exec_module(br)

TOOL = {
    "type": "function",
    "function": {
        "name": "reaction_type_selection_result",
        "parameters": {
            "type": "object",
            "properties": {"selected_label_exact": {"type": "string"}, "confidence": {"type": "number"}},
            "required": ["selected_label_exact", "confidence"],
        },
    },
}
CHOICE = {"type": "function", "function": {"name": "reaction_type_selection_result"}}
SECRET = "RUN_CONTEXT_SECRET_91"


def _request(bridge: Path, n: int, *, context: Optional[Dict[str, Any]] = None) -> str:
    stem = f"{n:020d}-{n:032x}"
    (bridge / "requests").mkdir(parents=True, exist_ok=True)
    (bridge / "responses").mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "mechanistic.agent_bridge/request@1",
        "request_id": f"{n:032x}",
        "model": "agent-bridge",
        "model_input": {
            "messages": [{"role": "system", "content": "Pick a reaction type."},
                         {"role": "user", "content": "CCBr + [I-] -> CCI + [Br-]"}],
            "tools": [TOOL],
            "tool_choice": CHOICE,
        },
        "context": context if context is not None else {"run_id": SECRET, "step_name": "reaction_type_mapping"},
    }
    (bridge / "requests" / f"{stem}.json").write_text(json.dumps(payload))
    return stem


@pytest.fixture
def layout(tmp_path: Path):
    bridge = tmp_path / "bridge"
    calls = tmp_path / "calls"
    return bridge, calls


# --- responder ----------------------------------------------------------------------------------


def test_prep_copies_only_model_input_into_prompt(layout) -> None:
    bridge, calls = layout
    stem = _request(bridge, 1)
    call_dir = br.prep(bridge, calls, stem)
    text = (call_dir / "prompt.md").read_text()
    assert "CCBr + [I-]" in text and "reaction_type_selection_result" in text
    assert SECRET not in text and "reaction_type_mapping" not in text and "context" not in text
    request = json.loads((bridge / "requests" / f"{stem}.json").read_text())
    assert text == br.render_prompt(request["model_input"])


def test_subagent_prompt_names_only_this_calls_files(layout) -> None:
    bridge, calls = layout
    stem = _request(bridge, 2)
    text = br.subagent_prompt(calls, stem)
    assert str(calls.resolve() / stem / "prompt.md") in text
    assert str(calls.resolve() / stem / "answer.json") in text
    assert "Write tool" in text and "audited" in text
    assert "no Bash" in text


def test_respond_validates_required_keys_and_writes_response(layout) -> None:
    bridge, calls = layout
    stem = _request(bridge, 3)
    br.prep(bridge, calls, stem)
    (calls / stem / "answer.json").write_text(json.dumps({"selected_label_exact": "SN2"}))
    with pytest.raises(ValueError, match="confidence"):
        br.respond(bridge, calls, stem)
    (calls / stem / "answer.json").write_text(
        json.dumps({"name": "reaction_type_selection_result", "arguments": {"selected_label_exact": "SN2", "confidence": 0.9}})
    )
    br.respond(bridge, calls, stem)
    response = json.loads((bridge / "responses" / f"{stem}.json").read_text())
    assert response["tool_calls"] == [
        {"name": "reaction_type_selection_result", "arguments": {"selected_label_exact": "SN2", "confidence": 0.9}}
    ]
    assert br.pending(bridge) == []


def test_cycle_dispatches_redispatches_bad_answers_and_gives_up(layout) -> None:
    bridge, calls = layout
    stem = _request(bridge, 4)
    assert br.cycle(bridge, calls) == [f"DISPATCH {stem}"]
    assert br.cycle(bridge, calls) == []  # already dispatched, still waiting
    for attempt in (1, 2):
        (calls / stem / "answer.json").write_text("{not json")
        events = br.cycle(bridge, calls, max_attempts=3)
        assert events[0].startswith(f"BAD {stem}") and events[1] == f"DISPATCH {stem}"
    (calls / stem / "answer.json").write_text("{not json")
    events = br.cycle(bridge, calls, max_attempts=3)
    assert events[-1].startswith(f"FAILED {stem}")
    assert json.loads((bridge / "responses" / f"{stem}.json").read_text())["error"]
    assert len(list((calls / stem).glob("answer.bad.*.json"))) == 3


# --- audit --------------------------------------------------------------------------------------


def _transcript(tasks: Path, agent_id: str, prompt: str, tool_uses: List[Dict[str, Any]]) -> Path:
    tasks.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = [
        {"type": "user", "isSidechain": True, "agentId": agent_id, "message": {"role": "user", "content": prompt}},
        {"type": "user", "isMeta": True, "message": {"role": "user", "content": "<system-reminder>x</system-reminder>"}},
    ]
    for i, (name, tool_input) in enumerate(tool_uses):
        records.append({"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "thinking"},
            {"type": "tool_use", "id": f"t{i}", "name": name, "input": tool_input},
        ]}})
    path = tasks / f"{agent_id}.output"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n" + "{truncated line")
    return path


def _answered(bridge: Path, calls: Path, n: int, **kw) -> str:
    stem = _request(bridge, n, **kw)
    br.prep(bridge, calls, stem)
    (calls / stem / "answer.json").write_text(json.dumps({"selected_label_exact": "SN2", "confidence": 0.8}))
    br.respond(bridge, calls, stem)
    return stem


def _files(calls: Path, stem: str) -> Dict[str, str]:
    d = calls.resolve() / stem
    return {"prompt": str(d / "prompt.md"), "answer": str(d / "answer.json")}


def test_audit_classifies_clean_procedural_contaminated_and_unaudited(layout, tmp_path: Path) -> None:
    bridge, calls = layout
    tasks = tmp_path / "tasks"
    stems = {name: _answered(bridge, calls, i + 10) for i, name in enumerate(
        ["clean", "heredoc", "jsoncheck", "grep_repo", "read_training", "web", "other_call", "spawn", "unaudited",
         "sqlite", "ls_cwd"]
    )}

    def tx(name: str, extra: List[Any]) -> None:
        f = _files(calls, stems[name])
        _transcript(tasks, f"a_{name}", br.subagent_prompt(calls, stems[name]),
                    [("Read", {"file_path": f["prompt"]})] + extra + [("SubagentHandback", {"message": "done"})])

    f = {name: _files(calls, stem) for name, stem in stems.items()}
    tx("clean", [("Write", {"file_path": f["clean"]["answer"], "content": "{}"})])
    tx("heredoc", [("Bash", {"command": f"cat > {f['heredoc']['answer']} <<'EOF'\n"
                                        '{"selected_label_exact": "E1/SN1 and/or C/C=C/C", "confidence": 0.5}\nEOF'})])
    tx("jsoncheck", [("Write", {"file_path": f["jsoncheck"]["answer"], "content": "{}"}),
                     ("Bash", {"command": f"python3 -c \"import json; json.load(open('{f['jsoncheck']['answer']}'))\""
                                          " && echo OK 2>/dev/null"})])
    tx("grep_repo", [("Grep", {"pattern": "SN2", "path": str(ROOT / "training_data")}),
                     ("Write", {"file_path": f["grep_repo"]["answer"], "content": "{}"})])
    tx("read_training", [("Read", {"file_path": str(ROOT / "training_data" / "eval_set.json")}),
                         ("Write", {"file_path": f["read_training"]["answer"], "content": "{}"})])
    tx("web", [("WebSearch", {"query": "Finkelstein mechanism"}),
               ("Write", {"file_path": f["web"]["answer"], "content": "{}"})])
    tx("other_call", [("Read", {"file_path": f["clean"]["answer"]}),
                      ("Write", {"file_path": f["other_call"]["answer"], "content": "{}"})])
    tx("spawn", [("Agent", {"prompt": "find the answer"}),
                 ("Write", {"file_path": f["spawn"]["answer"], "content": "{}"})])
    tx("sqlite", [("Bash", {"command": "sqlite3 ../wiggum-data/data/mechanistic.db 'select * from eval_sets'"})])
    tx("ls_cwd", [("Bash", {"command": "ls"}), ("Write", {"file_path": f["ls_cwd"]["answer"], "content": "{}"})])
    # A transcript for something else entirely (e.g. the orchestrator's own helper) is ignored.
    _transcript(tasks, "a_unrelated", "Refactor the scoring module", [("Bash", {"command": "ls"})])

    report = br.audit([calls], [str(tasks)], bridge_dir=bridge)
    verdicts = {c["stem"]: c for c in report["calls"]}
    v = {name: verdicts[stem]["verdict"] for name, stem in stems.items()}
    assert v == {
        "clean": "clean", "heredoc": "procedural", "jsoncheck": "procedural",
        "grep_repo": "contaminated", "read_training": "contaminated", "web": "contaminated",
        "other_call": "contaminated", "spawn": "contaminated", "unaudited": "unaudited",
        "sqlite": "contaminated", "ls_cwd": "contaminated",
    }
    assert verdicts[stems["clean"]]["context"] == {"run_id": SECRET, "step_name": "reaction_type_mapping"}
    reasons = " ".join(x["reason"] for x in verdicts[stems["sqlite"]]["violations"])
    assert "sqlite3" in reasons and "wiggum-data" in reasons
    summary = report["summary"]
    assert (summary["clean"], summary["procedural"], summary["contaminated"], summary["unaudited"]) == (1, 2, 7, 1)
    assert summary["transcripts_unmatched"] == 1 and summary["attributed_runs"] == [SECRET]
    table = br.format_table(report)
    assert "contaminated" in table and "summary:" in table


@pytest.mark.parametrize(
    "command, category",
    [
        ("cat {answer}", "procedural"),
        ("jq . {answer}", "procedural"),
        ("python3 -m json.tool {answer} > /dev/null", "procedural"),
        ("python3 - <<'EOF'\nimport json\nopen('{answer}','w').write(json.dumps({{'a': 1}}))\nEOF", "procedural"),
        ("python3 - <<'EOF'\nimport os\nprint(os.listdir('.'))\nEOF", "contamination"),
        ("python3 -c \"print(open('training_data/eval_set.json').read())\"", "contamination"),
        ("python3 -c \"from rdkit import Chem\"", "contamination"),
        ("cd /tmp && cat {answer}", "contamination"),
        ("cat ~/notes.txt", "contamination"),
        ("grep -r SN2 .", "contamination"),
        ("find / -name '*.db'", "contamination"),
        ("curl https://example.com", "contamination"),
        ("cat {prompt} | head -5", "procedural"),
        ("cat {other}", "contamination"),
        ("cat ../bridge/requests/x.json", "contamination"),
    ],
)
def test_bash_classification(command: str, category: str, tmp_path: Path) -> None:
    call_dir = tmp_path / "calls" / "stem1"
    other = tmp_path / "calls" / "stem2" / "answer.json"
    paths = br._call_paths([str(call_dir)])
    cmd = command.format(answer=call_dir / "answer.json", prompt=call_dir / "prompt.md", other=other)
    got, reason = br.analyse_bash(cmd, paths)
    assert got == category, reason


def test_live_gate_rejects_contaminated_answer_and_redispatches(layout, tmp_path: Path) -> None:
    bridge, calls = layout
    tasks = tmp_path / "tasks"
    stem = _request(bridge, 30)
    assert br.cycle(bridge, calls, transcripts=[str(tasks)]) == [f"DISPATCH {stem}"]
    f = _files(calls, stem)
    (calls / stem / "answer.json").write_text(json.dumps({"selected_label_exact": "SN2", "confidence": 0.8}))
    _transcript(tasks, "a_cheat", br.subagent_prompt(calls, stem), [
        ("Read", {"file_path": f["prompt"]}),
        ("Grep", {"pattern": "flower", "path": str(ROOT)}),
        ("Write", {"file_path": f["answer"], "content": "{}"}),
    ])
    events = br.cycle(bridge, calls, transcripts=[str(tasks)])
    assert events[0].startswith(f"CONTAMINATED {stem} (a_cheat)") and events[1] == f"DISPATCH {stem}"
    assert not (bridge / "responses" / f"{stem}.json").exists()
    assert list((calls / stem).glob("answer.contaminated.*.json"))

    # A fresh, clean subagent answers; the rejected transcript no longer counts.
    (calls / stem / "answer.json").write_text(json.dumps({"selected_label_exact": "SN2", "confidence": 0.7}))
    _transcript(tasks, "a_clean", br.subagent_prompt(calls, stem), [
        ("Read", {"file_path": f["prompt"]}),
        ("Write", {"file_path": f["answer"], "content": "{}"}),
    ])
    assert br.cycle(bridge, calls, transcripts=[str(tasks)]) == [f"RESPONDED {stem}"]
    report = br.audit([calls], [str(tasks)], bridge_dir=bridge)
    call = report["calls"][0]
    assert call["verdict"] == "clean" and call["rejected_transcripts"] == ["a_cheat"]


def test_audit_cli_exit_code_json_and_readonly_lookup(layout, tmp_path: Path, capsys) -> None:
    bridge, calls = layout
    tasks = tmp_path / "tasks"
    stem = _answered(bridge, calls, 40, context={"run_id": "run-x", "step_name": "atom_mapping"})
    f = _files(calls, stem)
    _transcript(tasks, "a1", br.subagent_prompt(calls, stem), [
        ("Read", {"file_path": f["prompt"]}), ("WebFetch", {"url": "https://x"}),
        ("Write", {"file_path": f["answer"], "content": "{}"})])
    db = tmp_path / "ro.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE eval_run_results (id TEXT, eval_run_id TEXT, case_id TEXT, run_id TEXT)")
        conn.execute("INSERT INTO eval_run_results VALUES ('1', 'ev-9', 'flower_7', 'run-x')")
    out_json = tmp_path / "report.json"
    code = br.main(["audit", "--calls-dir", str(calls), "--bridge-dir", str(bridge), "--transcripts", str(tasks),
                    "--db", str(db), "--json", str(out_json)])
    assert code == 1
    report = json.loads(out_json.read_text())
    assert report["calls"][0]["context"] == {
        "run_id": "run-x", "step_name": "atom_mapping", "eval_run_id": "ev-9", "case_id": "flower_7"
    }
    assert "flower_7" in capsys.readouterr().out


def test_audit_mark_records_integrity_on_runs_and_eval_runs(layout, tmp_path: Path, monkeypatch) -> None:
    from mechanistic_agent.core.db import RunStore

    monkeypatch.setenv("MECHANISTIC_ACTIVE_MODEL", "agent-bridge")
    store = RunStore(tmp_path / "mechanistic.db")
    eval_set_id = store.add_eval_set(name="s", version="1", source_path=None, sha256=None, cases=[
        {"case_id": "c1", "input": {"starting_materials": ["CCBr"], "products": ["CCI"]}, "expected": {}},
        {"case_id": "c2", "input": {"starting_materials": ["CCBr"], "products": ["CCI"]}, "expected": {}}])
    run_dirty = store.create_run(mode="unverified", input_payload={}, config={"model": "agent-bridge"},
                                 prompt_bundle_hash="p", skill_bundle_hash="s")
    run_clean = store.create_run(mode="unverified", input_payload={}, config={"model": "agent-bridge"},
                                 prompt_bundle_hash="p", skill_bundle_hash="s")
    eval_run_id = store.create_eval_run(eval_set_id=eval_set_id, run_group_name="g", model="agent-bridge",
                                        harness_bundle_hash="h")
    for case_id, run_id in (("c1", run_dirty), ("c2", run_clean)):
        store.record_eval_run_result(eval_run_id=eval_run_id, case_id=case_id, run_id=run_id, score=0.5,
                                     passed=False, cost=None, latency_ms=1.0, summary={})

    bridge, calls = layout
    tasks = tmp_path / "tasks"
    dirty = _answered(bridge, calls, 50, context={"run_id": run_dirty, "step_name": "mechanism_step_proposal"})
    clean = _answered(bridge, calls, 51, context={"run_id": run_clean, "step_name": "mechanism_step_proposal"})
    for agent, stem, extra in (("a_d", dirty, [("Bash", {"command": "grep -r CCI training_data"})]), ("a_c", clean, [])):
        f = _files(calls, stem)
        _transcript(tasks, agent, br.subagent_prompt(calls, stem),
                    [("Read", {"file_path": f["prompt"]})] + extra + [("Write", {"file_path": f["answer"], "content": "{}"})])

    code = br.main(["audit", "--calls-dir", str(calls), "--bridge-dir", str(bridge), "--transcripts", str(tasks),
                    "--db", str(tmp_path / "mechanistic.db"), "--mark"])
    assert code == 1
    dirty_integrity = store.get_run_row(run_dirty)["config"]["origin"]["responder_integrity"]
    assert dirty_integrity["status"] == "contaminated"
    assert dirty_integrity["violations"][0]["step_name"] == "mechanism_step_proposal"
    assert store.get_run_row(run_clean)["config"]["origin"]["responder_integrity"]["status"] == "clean"
    eval_integrity = store.get_eval_run(eval_run_id)["metadata"]["responder_integrity"]
    assert eval_integrity["status"] == "contaminated" and eval_integrity["calls_audited"] == 2


def test_cli_options_work_before_or_after_subcommand(layout, capsys) -> None:
    bridge, calls = layout
    stem = _request(bridge, 60)
    assert br.main(["--bridge-dir", str(bridge), "prompt", stem]) == 0
    before = capsys.readouterr().out
    assert br.main(["prompt", stem, "--bridge-dir", str(bridge)]) == 0
    assert capsys.readouterr().out == before
    assert str(calls.resolve() / stem / "prompt.md") in before


def test_atom_labels_in_heredoc_are_not_paths(tmp_path: Path) -> None:
    call = tmp_path / "calls" / "stem"
    paths = br._call_paths([str(call)])
    answer = str(call / "answer.json")
    labels = f"python3 - <<'EOF'\nsym=['C14 tert-butyl CH3 (equivalent with C15/C16)', 'N9/O7']\nopen('{answer}','w')\nEOF"
    category, reason = br.analyse_bash(labels, paths)
    assert category == br.PROC, reason
    category, reason = br.analyse_bash("cat training_data/C1/eval.json", paths)
    assert category == br.CONTAM, reason
