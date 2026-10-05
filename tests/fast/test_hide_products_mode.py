"""No-product mode: targets are withheld from every LLM call and the model declares completion."""

from __future__ import annotations

import threading

from mechanistic_agent.core.baseline_runner import _build_user_message
from mechanistic_agent.core.coordinator import RunCoordinator
from mechanistic_agent.core.db import RunStore

PRODUCT = "CCCl"


def _state(tmp_path, *, hide: bool):
    store = RunStore(tmp_path / "data" / "mechanistic.db")
    run_id = store.create_run(
        mode="unverified",
        input_payload={"starting_materials": ["CCBr", "[Cl-]"], "products": [PRODUCT, "[Br-]"],
                       "temperature_celsius": 25.0, "ph": 7.0},
        config={"model": "gpt-4o-mini", "model_family": "openai", "max_steps": 3, "max_runtime_seconds": 30.0,
                "intermediate_prediction_enabled": True, "hide_products": hide},
        prompt_bundle_hash="p", skill_bundle_hash="s", memory_bundle_hash="m",
    )
    coordinator = RunCoordinator(store)
    return store, run_id, coordinator, coordinator._build_state(store.get_run_row(run_id))


def test_hidden_products_never_reach_the_run_state(tmp_path) -> None:
    store, run_id, _coordinator, state = _state(tmp_path, hide=True)
    assert state.run_config.hide_products is True
    assert state.run_input.products == []
    # The real targets stay on the stored input for scoring.
    assert store.get_run_row(run_id)["input_payload"]["products"] == [PRODUCT, "[Br-]"]


def test_visible_products_are_unchanged(tmp_path) -> None:
    _store, _run_id, _coordinator, state = _state(tmp_path, hide=False)
    assert state.run_config.hide_products is False
    assert state.run_input.products == [PRODUCT, "[Br-]"]


def _declare_final(*_args, **_kwargs):  # noqa: ANN002, ANN003
    return {"step_classification": "final_step"}, []


def test_model_declared_final_step_ends_the_loop_in_no_product_mode(tmp_path) -> None:
    store, run_id, coordinator, state = _state(tmp_path, hide=True)
    state.step_index = 1  # one step already accepted
    coordinator._propose_for_topology = _declare_final  # type: ignore[method-assign]
    coordinator._run_mechanism_loop(state, threading.Event(), harness=None)
    assert state.declared_complete is True
    events = [e["event_type"] for e in store.list_events(run_id)]
    assert "mechanism_declared_complete" in events


def test_declared_final_is_ignored_when_products_are_visible(tmp_path) -> None:
    store, run_id, coordinator, state = _state(tmp_path, hide=False)
    state.step_index = 1
    coordinator._propose_for_topology = _declare_final  # type: ignore[method-assign]
    coordinator._run_mechanism_loop(state, threading.Event(), harness=None)
    assert state.declared_complete is False
    assert "mechanism_declared_complete" not in [e["event_type"] for e in store.list_events(run_id)]


def test_baseline_prompt_withholds_products() -> None:
    hidden = _build_user_message(["CCBr", "[Cl-]"], [PRODUCT, "[Br-]"], hide_products=True)
    assert PRODUCT not in hidden and "not given" in hidden
    visible = _build_user_message(["CCBr", "[Cl-]"], [PRODUCT, "[Br-]"])
    assert PRODUCT in visible
