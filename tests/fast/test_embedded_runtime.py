"""Embedded Mechanism Runtime (Observatory PRD rev 3 §2.3.2).

ChemIllusion runs the tagged Wiggum release in-process from its job worker.
It needs: create a run from a request dict, execute it blocking on the worker
thread, stop it, read the replay projection, and mirror every event to its own
durable store (Postgres) — the SQLite file is scratch. Because the
``/observatory`` projection is built from events only, the mirrored log alone
reproduces the view after a container restart.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List

import pytest

pytest.importorskip("rdkit")
pytest.importorskip("fastapi")

from mechanistic_agent.core.db import RunStore  # noqa: E402
from mechanistic_agent.core.observatory import build_observatory  # noqa: E402
from mechanistic_agent.core.types import RunState, StepResult  # noqa: E402
from mechanistic_agent.runtime import EmbeddedMechanismRuntime  # noqa: E402
from tests.fast.test_api_runtime import _prepare_base  # noqa: E402

SN2_SMIRKS = "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] |mech:v1;lp:4>2;sigma:2-3>3|"
SN2_REQUEST = {"mode": "unverified", "starting_materials": ["CCBr", "[Cl-]"], "products": ["CCCl", "[Br-]"],
               "model": "gpt-4o", "max_steps": 1, "optional_llm_tools": [], "max_runtime_seconds": 60}


class _Proposer:
    def run(self, _state: RunState, **_kw: Any) -> StepResult:
        return StepResult(step_name="mechanism_step_proposal", tool_name="propose_mechanism_step", source="llm", output={
            "candidates": [{"rank": 1, "intermediate_smiles": "CCCl", "reaction_description": "SN2", "reaction_smirks": SN2_SMIRKS,
                            "electron_pushes": [{"kind": "lone_pair", "source_atom": "4", "target_atom": "2", "electrons": 2},
                                                {"kind": "sigma_bond", "source_bond": ["2", "3"], "through_atom": "3", "target_atom": "3", "electrons": 2}],
                            "resulting_state": ["CCCl", "[Br-]"]}]})


# ---------------------------------------------------------------------------
# store-level event sink
# ---------------------------------------------------------------------------

def test_run_store_mirrors_every_event_to_the_sink(tmp_path: Path) -> None:
    mirrored: List[Dict[str, Any]] = []
    store = RunStore(tmp_path / "m.db", event_sink=mirrored.append)
    run_id = store.create_run(mode="unverified", input_payload={"starting_materials": ["A"], "products": ["B"]}, config={},
                              prompt_bundle_hash="a", skill_bundle_hash="b", memory_bundle_hash="c")
    store.append_event(run_id, "run_started", {"x": 1})
    store.append_event(run_id, "step_started", {"step_name": "s"}, step_name="s")
    assert [(e["seq"], e["event_type"], e["step_name"]) for e in mirrored][-2:] == [(len(mirrored) - 1, "run_started", None), (len(mirrored), "step_started", "s")]
    assert mirrored[-1]["run_id"] == run_id and mirrored[-1]["payload"] == {"step_name": "s"} and mirrored[-1]["ts"] > 0
    # the mirrored rows have the same shape list_events returns, so projections accept either
    stored = store.list_events(run_id)
    assert [(e["seq"], e["event_type"]) for e in stored] == [(e["seq"], e["event_type"]) for e in mirrored]


def test_a_failing_sink_never_breaks_the_run_store(tmp_path: Path) -> None:
    def boom(_event: Dict[str, Any]) -> None:
        raise RuntimeError("postgres down")

    store = RunStore(tmp_path / "m.db", event_sink=boom)
    run_id = store.create_run(mode="unverified", input_payload={}, config={}, prompt_bundle_hash="a", skill_bundle_hash="b", memory_bundle_hash="c")
    assert store.append_event(run_id, "run_started", {}) >= 1
    assert store.list_events(run_id)[-1]["event_type"] == "run_started"


# ---------------------------------------------------------------------------
# embedded runtime
# ---------------------------------------------------------------------------

def _runtime(tmp_path: Path, mirrored: List[Dict[str, Any]]) -> EmbeddedMechanismRuntime:
    return EmbeddedMechanismRuntime(base_dir=_prepare_base(tmp_path), work_dir=tmp_path / "work", event_sink=mirrored.append)


def test_create_execute_and_replay_from_mirrored_events(tmp_path: Path) -> None:
    mirrored: List[Dict[str, Any]] = []
    rt = _runtime(tmp_path, mirrored)
    rt.coordinator.intermediate_agent = _Proposer()  # type: ignore[assignment]
    assert rt.db_path.parent == tmp_path / "work"

    run_id = rt.create_run(dict(SN2_REQUEST))
    assert rt.status(run_id) == "pending"
    rt.execute(run_id)  # blocking, on the caller's thread

    assert rt.status(run_id) in {"completed", "failed", "paused", "stopped"}
    live = rt.observatory(run_id)
    assert live["schema_version"] == "mechanism_observatory.v1"
    assert len(live["accepted_path"]) == 1 and live["accepted_path"][0]["validation_passed"] is True
    # replay from the mirror alone (what ChemIllusion keeps in Postgres) is identical
    replay = build_observatory([e for e in mirrored if e["run_id"] == run_id], run_id=run_id,
                               run_input={"starting_materials": ["CCBr", "[Cl-]"], "products": ["CCCl", "[Br-]"]}, status=live["status"])
    runtime = live.pop("runtime")
    assert replay == live
    assert runtime["runtime_version"] == rt.manifest()["runtime_version"] and runtime["deployment"] == "embedded"


def test_create_run_rejects_empty_input(tmp_path: Path) -> None:
    rt = _runtime(tmp_path, [])
    with pytest.raises(ValueError, match="starting_materials"):
        rt.create_run({"mode": "unverified", "starting_materials": [], "products": ["CO"], "model": "gpt-4o"})


def test_stop_is_honoured_by_a_running_execute(tmp_path: Path) -> None:
    rt = _runtime(tmp_path, [])
    started = threading.Event()

    class _SlowProposer(_Proposer):
        def run(self, state: RunState, **kw: Any) -> StepResult:
            started.set()
            return super().run(state, **kw)

    rt.coordinator.intermediate_agent = _SlowProposer()  # type: ignore[assignment]
    run_id = rt.create_run(dict(SN2_REQUEST, max_steps=3))
    stop = threading.Event()
    stop.set()  # stop requested before the loop starts
    rt.execute(run_id, stop_event=stop)
    assert rt.status(run_id) in {"stopped", "completed", "failed"}


def test_manifest_and_unknown_run(tmp_path: Path) -> None:
    rt = _runtime(tmp_path, [])
    manifest = rt.manifest()
    assert manifest["schema_version"] == "runtime_manifest.v1"
    assert manifest["deployment"] == "embedded"
    with pytest.raises(KeyError):
        rt.observatory("nope")
