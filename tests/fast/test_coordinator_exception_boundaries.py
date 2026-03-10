from __future__ import annotations

import threading

from mechanistic_agent.core.coordinator import RunCoordinator
from mechanistic_agent.core.db import RunStore


def test_proposal_dispatch_exception_sets_structured_run_failed_reason(tmp_path) -> None:
    store = RunStore(tmp_path / "data" / "mechanistic.db")
    run_id = store.create_run(
        mode="unverified",
        input_payload={
            "starting_materials": ["CCBr", "[Cl-]"],
            "products": ["CCCl", "[Br-]"],
            "temperature_celsius": 25.0,
            "ph": 7.0,
        },
        config={
            "model": "gpt-4o-mini",
            "model_family": "openai",
            "max_steps": 1,
            "max_runtime_seconds": 0.2,
            "intermediate_prediction_enabled": True,
        },
        prompt_bundle_hash="p",
        skill_bundle_hash="s",
        memory_bundle_hash="m",
    )
    coordinator = RunCoordinator(store)
    row = store.get_run_row(run_id)
    assert row is not None
    state = coordinator._build_state(row)

    def _explode(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("dispatch exploded")

    coordinator._propose_for_topology = _explode  # type: ignore[method-assign]
    coordinator._run_mechanism_loop(state, threading.Event(), harness=None)

    events = store.list_events(run_id)
    failed_events = [ev for ev in events if ev.get("event_type") == "run_failed"]
    assert failed_events
    assert failed_events[-1]["payload"]["reason"] == "proposal_dispatch_exception"
    run_row = store.get_run_row(run_id) or {}
    assert run_row.get("status") == "failed"
