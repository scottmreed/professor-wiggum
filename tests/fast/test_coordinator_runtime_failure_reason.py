from __future__ import annotations

import threading

from mechanistic_agent.core.coordinator import RunCoordinator
from mechanistic_agent.core.db import RunStore


def test_runtime_limit_takes_precedence_over_no_valid_steps(tmp_path) -> None:
    store = RunStore(tmp_path / "data" / "mechanistic.db")
    run_id = store.create_run(
        mode="unverified",
        input_payload={
            "starting_materials": ["CCBr", "[Cl-]"],
            "products": ["CCCl", "[Br-]"],
            "temperature_celsius": 25.0,
            "ph": 7.0,
        },
        config={"model": "gpt-5", "model_family": "openai", "max_steps": 1, "max_runtime_seconds": 0.01},
        prompt_bundle_hash="p",
        skill_bundle_hash="s",
        memory_bundle_hash="m",
    )
    coordinator = RunCoordinator(store)
    coordinator._run_initial_phase = lambda *_args, **_kwargs: None  # type: ignore[method-assign]

    def _fake_loop(state, _stop_event, _harness):  # type: ignore[no-untyped-def]
        store.append_event(
            state.run_id,
            "runtime_limit",
            {"elapsed_seconds": 1.0, "max_runtime_seconds": 0.01},
        )

    coordinator._run_mechanism_loop = _fake_loop  # type: ignore[method-assign]
    coordinator.execute_run(run_id, threading.Event())

    events = store.list_events(run_id)
    run_failed = [e for e in events if e.get("event_type") == "run_failed"]
    assert run_failed
    assert run_failed[-1]["payload"]["reason"] == "runtime_limit_reached"
