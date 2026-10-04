"""Responder-integrity verdicts (``responder_integrity``) make consumers disregard contaminated bridge runs.

A blind bridge responder that looked beyond its prompt (grepped the repo, read
the DB, searched the web) produced a contaminated answer. ``scripts/bridge_responder.py
audit --mark`` records that on the run's ``config.origin`` and the eval run's
``metadata``; the leaderboard, the prompt-evidence gate and results publishing
must then ignore those runs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from mechanistic_agent import results_publish as rp
from mechanistic_agent.agent_bridge import BRIDGE_SAW_GROUND_TRUTH_ENV
from mechanistic_agent.core.db import RunStore
from mechanistic_agent.prompt_assets import get_call_prompt_version
from mechanistic_agent.prompt_trace_validator import validate_evidence_for_calls

CONTAMINATED = {"status": "contaminated", "audited_at": 1.0, "violations": [{"tool": "Grep", "reason": "repo"}]}
PROCEDURAL = {"status": "procedural", "audited_at": 1.0, "violations": []}


def _seed(store: RunStore, name: str, monkeypatch) -> Dict[str, str]:
    monkeypatch.setenv(BRIDGE_SAW_GROUND_TRUTH_ENV, "false")
    eval_set_id = store.add_eval_set(
        name=f"set_{name}", version="v1", source_path=None, sha256=None,
        cases=[{"case_id": "c1", "input": {"starting_materials": ["CCBr", "[Cl-]"], "products": ["CCCl", "[Br-]"]},
                "expected": {"products": ["CCCl", "[Br-]"]}}],
    )
    run_id = store.create_run(
        mode="unverified",
        input_payload={"starting_materials": ["CCBr", "[Cl-]"], "products": ["CCCl", "[Br-]"]},
        config={"model": "agent-bridge", "model_name": "agent-bridge", "model_family": "agent"},
        prompt_bundle_hash="p", skill_bundle_hash="s",
    )
    eval_run_id = store.create_eval_run(
        eval_set_id=eval_set_id, run_group_name=f"grp_{name}", model="agent-bridge", harness_bundle_hash="h",
        metadata={"origin": {"responder": "agent-bridge", "responder_saw_ground_truth": False}},
    )
    store.record_eval_run_result(
        eval_run_id=eval_run_id, case_id="c1", run_id=run_id, score=0.9, passed=True, cost=None,
        latency_ms=10.0, summary={},
    )
    store.set_eval_run_status(eval_run_id, "completed")
    return {"eval_set_id": eval_set_id, "run_id": run_id, "eval_run_id": eval_run_id}


def test_store_records_integrity_on_run_origin_and_eval_metadata(tmp_path: Path, monkeypatch) -> None:
    store = RunStore(tmp_path / "mechanistic.db")
    ids = _seed(store, "a", monkeypatch)
    assert store.get_run_row(ids["run_id"])["config"]["origin"]["responder"] == "agent-bridge"

    assert store.eval_refs_for_runs([ids["run_id"], "missing"]) == [
        {"run_id": ids["run_id"], "eval_run_id": ids["eval_run_id"], "case_id": "c1"}
    ]
    assert store.set_run_responder_integrity(ids["run_id"], CONTAMINATED) is True
    assert store.set_run_responder_integrity("missing", CONTAMINATED) is False
    assert store.set_eval_run_responder_integrity(ids["eval_run_id"], PROCEDURAL) is True
    assert store.set_eval_run_responder_integrity("missing", PROCEDURAL) is False

    origin = store.get_run_row(ids["run_id"])["config"]["origin"]
    assert origin["responder_integrity"] == CONTAMINATED
    assert origin["responder_saw_ground_truth"] is False  # rest of the origin is untouched
    metadata = store.get_eval_run(ids["eval_run_id"])["metadata"]
    assert metadata["responder_integrity"] == PROCEDURAL
    assert metadata["origin"]["responder_integrity"] == PROCEDURAL


def test_leaderboard_drops_contaminated_runs(tmp_path: Path, monkeypatch) -> None:
    store = RunStore(tmp_path / "mechanistic.db")
    dirty = _seed(store, "dirty", monkeypatch)
    procedural = _seed(store, "procedural", monkeypatch)
    dirty_meta = _seed(store, "dirty_meta", monkeypatch)
    assert len(store.leaderboard(dirty["eval_set_id"])) == 1

    store.set_run_responder_integrity(dirty["run_id"], CONTAMINATED)
    store.set_run_responder_integrity(procedural["run_id"], PROCEDURAL)
    store.set_eval_run_responder_integrity(dirty_meta["eval_run_id"], CONTAMINATED)

    assert store.leaderboard(dirty["eval_set_id"]) == []
    assert store.leaderboard(dirty_meta["eval_set_id"]) == []
    assert len(store.leaderboard(procedural["eval_set_id"])) == 1  # procedural slips are still usable


# --- prompt-evidence gate ---------------------------------------------------------------------

CALL = "assess_initial_conditions"


def _skill_md(kind: str, call: str, prompt: str) -> str:
    return f"---\nkind: {kind}\ncall_name: {call}\n---\n<!-- PROMPT_START -->\n{prompt}\n<!-- PROMPT_END -->\n"


def _evidence(base: Path, bundle: str, name: str, origin: Dict[str, Any]) -> None:
    folder = base / "traces" / "evidence" / CALL / bundle
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "approved_bool": True,
        "responder_saw_ground_truth": False,
        "origin": origin,
        "prompt_version": {"prompt_bundle_sha256": bundle, "model_name": None},
        "model_version": {"model_version_id": "abc", "resolved_model_key": "gpt-5", "provider": "openai",
                          "family": "openai", "pricing_sha256": "123"},
    }
    (folder / name).write_text(json.dumps(payload), encoding="utf-8")


def test_evidence_gate_rejects_contaminated_origin(tmp_path: Path) -> None:
    mech = tmp_path / "skills" / "mechanistic"
    (mech / "base_system").mkdir(parents=True)
    (mech / "base_system" / "SKILL.md").write_text(_skill_md("shared_base", "base_system", "base"), encoding="utf-8")
    (mech / CALL).mkdir(parents=True)
    (mech / CALL / "SKILL.md").write_text(_skill_md("llm", CALL, "call prompt"), encoding="utf-8")
    (mech / CALL / "few_shot.jsonl").write_text('{"input": "q", "output": "a"}\n', encoding="utf-8")
    bundle = str(get_call_prompt_version(CALL, base_dir=tmp_path)["prompt_bundle_sha256"])

    blind = {"responder": "agent-bridge", "responder_saw_ground_truth": False}
    _evidence(tmp_path, bundle, "dirty.json", {**blind, "responder_integrity": CONTAMINATED})
    result = validate_evidence_for_calls(changed_calls=[CALL], base_dir=tmp_path)
    assert not result.ok
    assert any("responder_integrity" in err for err in result.errors)

    _evidence(tmp_path, bundle, "procedural.json", {**blind, "responder_integrity": PROCEDURAL})
    result_ok = validate_evidence_for_calls(changed_calls=[CALL], base_dir=tmp_path)
    assert result_ok.ok
    assert result_ok.valid_evidence_by_call[CALL] == [f"traces/evidence/{CALL}/{bundle}/procedural.json"]


# --- results publishing ------------------------------------------------------------------------


class _FakeStore:
    def __init__(self, *, case_integrity: Any = None, run_integrity: Any = None) -> None:
        self.case_integrity = case_integrity
        self.run_integrity = run_integrity

    def get_eval_run(self, eval_run_id: str) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {"tier_name": "hard"}
        if self.run_integrity:
            metadata["responder_integrity"] = self.run_integrity
        return {"id": eval_run_id, "eval_set_id": "set1", "run_group_name": "grp", "model": "agent-bridge",
                "model_name": "agent-bridge", "created_at": 1790000000.0, "metadata": metadata}

    def list_eval_run_results(self, eval_run_id: str) -> List[Dict[str, Any]]:
        return [{"case_id": "c1", "run_id": "r1", "score": 1.0, "pass_bool": True, "latency_ms": 1000}]

    def get_eval_set(self, eval_set_id: str) -> Dict[str, Any]:
        return {"id": eval_set_id, "purpose": "general"}

    def get_run_snapshot(self, run_id: str) -> Dict[str, Any]:
        origin: Dict[str, Any] = {"responder": "agent-bridge", "responder_saw_ground_truth": False}
        if self.case_integrity:
            origin["responder_integrity"] = self.case_integrity
        return {"id": run_id, "status": "completed", "config": {"harness_name": "default", "origin": origin},
                "input_payload": {"starting_materials": ["CO"], "products": ["C=O"]}, "events": [], "step_outputs": []}


@pytest.fixture
def _stub_grading(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rp, "score_snapshot_against_known", lambda snap, expected, **_: {
        "score": 1.0, "passed": True, "final_product_reached": True, "known_alignment_component": 1.0,
        "step_validity_component": 1.0, "accepted_path_step_count": 1})
    monkeypatch.setattr(rp, "_accepted_path_record", lambda snap: [])
    monkeypatch.setattr(rp, "_git_commit", lambda base: "deadbee")


def _resolver(result: Dict[str, Any], run: Dict[str, Any]) -> Dict[str, Any]:
    return {"n_mechanistic_steps": 1}


@pytest.mark.usefixtures("_stub_grading")
def test_publish_refuses_contaminated_runs() -> None:
    with pytest.raises(rp.PublishError, match="contaminated"):
        rp.export_eval_run(_FakeStore(case_integrity=CONTAMINATED), "ev1", expected_resolver=_resolver)
    with pytest.raises(rp.PublishError, match="contaminated"):
        rp.export_eval_run(_FakeStore(run_integrity=CONTAMINATED), "ev1", expected_resolver=_resolver)
    record = rp.export_eval_run(_FakeStore(case_integrity=PROCEDURAL), "ev1", expected_resolver=_resolver)
    assert record["summary"]["cases"] == 1


def test_publish_refuses_contaminated_baseline_runs(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "b.db")
    set_id = store.add_eval_set(name="s", version="1", source_path=None, sha256=None, cases=[
        {"case_id": "flower_1", "input": {"starting_materials": ["CC"], "products": ["C=C"]},
         "expected": {"n_mechanistic_steps": 1}}])
    origin = {"responder": "agent-bridge", "responder_saw_ground_truth": False, "responder_integrity": CONTAMINATED}
    eval_run_id = store.create_eval_run(
        eval_set_id=set_id, run_group_name="harness_free_baseline_easy", model="agent-bridge",
        model_name="agent-bridge", harness_bundle_hash=None, metadata={"origin": origin}, status="completed",
    )
    store.record_eval_run_result(eval_run_id=eval_run_id, case_id="flower_1", run_id=None, score=0.5, passed=False,
                                 cost=None, latency_ms=10.0, summary={"eval_mode": "baseline"})
    with pytest.raises(rp.PublishError, match="contaminated"):
        rp.export_eval_run(store, eval_run_id)
