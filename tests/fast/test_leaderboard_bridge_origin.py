"""Keyless agent-bridge eval runs rank under the responder's declared model, marked as bridged."""
from __future__ import annotations

from pathlib import Path

import pytest

from mechanistic_agent.agent_bridge import (
    BRIDGE_DECLARED_MODEL_ENV,
    BRIDGE_SAW_GROUND_TRUTH_ENV,
    build_origin_provenance,
)
from mechanistic_agent.core.db import RunStore

_INPUT = {"starting_materials": ["CCBr", "[Cl-]"], "products": ["CCCl", "[Br-]"]}


def _eval_set(store: RunStore) -> str:
    return store.add_eval_set(
        name="bridge_set",
        version="v1",
        source_path=None,
        sha256=None,
        cases=[{"case_id": "c1", "input": _INPUT, "expected": {"products": ["CCCl", "[Br-]"]}}],
    )


def _harness_eval_run(store: RunStore, eval_set_id: str, *, model: str, group: str, score: float) -> str:
    run_id = store.create_run(
        mode="unverified",
        input_payload=_INPUT,
        config={"model": model, "model_name": model, "model_family": "agent"},
        prompt_bundle_hash="p",
        skill_bundle_hash="s",
    )
    eval_run_id = store.create_eval_run(
        eval_set_id=eval_set_id, run_group_name=group, model=model, model_name=model, harness_bundle_hash="h"
    )
    store.record_eval_run_result(
        eval_run_id=eval_run_id, case_id="c1", run_id=run_id, score=score, passed=True, cost=None, latency_ms=10.0, summary={}
    )
    store.set_eval_run_status(eval_run_id, "completed")
    return eval_run_id


def _rows_by_group(store: RunStore, eval_set_id: str) -> dict:
    return {row["run_group_name"]: row for row in store.leaderboard(eval_set_id)}


def test_harness_bridge_run_ranks_under_declared_model(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(BRIDGE_DECLARED_MODEL_ENV, "claude-opus-5-5 (headless claude -p, blind per call)")
    monkeypatch.setenv(BRIDGE_SAW_GROUND_TRUTH_ENV, "false")
    store = RunStore(tmp_path / "data" / "mechanistic.db")
    eval_set_id = _eval_set(store)
    _harness_eval_run(store, eval_set_id, model="agent-bridge", group="bridge_grp", score=0.9)
    _harness_eval_run(store, eval_set_id, model="anthropic/claude-opus-5.5", group="api_grp", score=0.8)

    rows = _rows_by_group(store, eval_set_id)
    bridge, api = rows["bridge_grp"], rows["api_grp"]
    assert bridge["model_name"] == api["model_name"] == "anthropic/claude-opus-5.5"
    assert bridge["model"] == "agent-bridge"  # raw value kept for filters
    assert bridge["via_bridge"] is True
    assert bridge["bridge_model"] == "agent-bridge"
    assert api["via_bridge"] is False
    assert "bridge_model" not in api


def test_baseline_bridge_run_reads_origin_from_eval_run_metadata(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(BRIDGE_DECLARED_MODEL_ENV, "claude-opus-5-5")
    monkeypatch.setenv(BRIDGE_SAW_GROUND_TRUTH_ENV, "false")
    store = RunStore(tmp_path / "data" / "mechanistic.db")
    eval_set_id = _eval_set(store)
    # Baseline eval runs have no run rows: the origin lives on the eval run.
    eval_run_id = store.create_eval_run(
        eval_set_id=eval_set_id,
        run_group_name="harness_free_baseline_easy",
        model="agent-bridge",
        model_name="agent-bridge",
        harness_bundle_hash="h",
        metadata={"origin": build_origin_provenance("agent-bridge")},
    )
    store.record_eval_run_result(
        eval_run_id=eval_run_id, case_id="c1", run_id=None, score=0.7, passed=True, cost=None, latency_ms=10.0, summary={}
    )
    store.set_eval_run_status(eval_run_id, "completed")

    (row,) = store.leaderboard(eval_set_id)
    assert row["is_baseline"] is True
    assert row["model_name"] == "anthropic/claude-opus-5.5"
    assert row["via_bridge"] is True
    assert row["bridge_model"] == "agent-bridge"


def test_baseline_bridge_run_that_saw_ground_truth_is_not_ranked(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(BRIDGE_DECLARED_MODEL_ENV, "claude-opus-5-5")
    monkeypatch.setenv(BRIDGE_SAW_GROUND_TRUTH_ENV, "true")
    store = RunStore(tmp_path / "data" / "mechanistic.db")
    eval_set_id = _eval_set(store)
    eval_run_id = store.create_eval_run(
        eval_set_id=eval_set_id,
        run_group_name="harness_free_baseline_easy",
        model="agent-bridge",
        harness_bundle_hash="h",
        metadata={"origin": build_origin_provenance("agent-bridge")},
    )
    store.record_eval_run_result(
        eval_run_id=eval_run_id, case_id="c1", run_id=None, score=0.99, passed=True, cost=None, latency_ms=10.0, summary={}
    )
    store.set_eval_run_status(eval_run_id, "completed")

    assert store.leaderboard(eval_set_id) == []


def test_undeclared_bridge_run_keeps_agent_bridge_label(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(BRIDGE_DECLARED_MODEL_ENV, raising=False)
    monkeypatch.setenv(BRIDGE_SAW_GROUND_TRUTH_ENV, "false")
    store = RunStore(tmp_path / "data" / "mechanistic.db")
    eval_set_id = _eval_set(store)
    _harness_eval_run(store, eval_set_id, model="agent-bridge", group="bridge_grp", score=0.9)

    (row,) = store.leaderboard(eval_set_id)
    assert row["model_name"] == "agent-bridge"
    assert row["via_bridge"] is True


@pytest.fixture(autouse=True)
def _no_forced_bridge(monkeypatch) -> None:
    monkeypatch.delenv("MECHANISTIC_ACTIVE_MODEL", raising=False)
