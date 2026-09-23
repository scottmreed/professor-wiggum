"""M0b: live inference-call recording (Observatory PRD §13, "M0b — live").

A chat adapter obtained inside a run step emits ``inference_call_started``
before the request and ``inference_call_completed`` / ``inference_call_failed``
after it, with measured latency. ``_record_step`` then uses those live records
instead of deriving a call after the fact, so no call is counted twice.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

pytest.importorskip("rdkit")

from mechanistic_agent.core.call_recorder import (
    RecordingChatAdapter,
    call_context,
    current_call_context,
)
from mechanistic_agent.core.coordinator import RunCoordinator
from mechanistic_agent.core.db import RunStore
from mechanistic_agent.core.types import RunConfig, RunInput, RunState, StepResult
from mechanistic_agent.llm import _SimpleMessage, get_chat_model

CONFIGURED = "gpt-4o"


class _FakeInner:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: List[Dict[str, Any]] = []
        self._model = CONFIGURED

    def invoke(self, messages: Any, config: Any = None, *, tools: Any = None, tool_choice: Any = None) -> Any:
        self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
        if self.fail:
            raise RuntimeError("provider down")
        return _SimpleMessage("ok", usage={"prompt_tokens": 120, "completion_tokens": 30})


class _Sink:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def __call__(self, event_type: str, payload: Dict[str, Any], *, step_name: str | None = None) -> None:
        self.events.append({"event_type": event_type, "payload": payload, "step_name": step_name})

    def of(self, kind: str) -> List[Dict[str, Any]]:
        return [e["payload"] for e in self.events if e["event_type"] == kind]


# ---------------------------------------------------------------------------
# Recorder unit behaviour
# ---------------------------------------------------------------------------

def test_recording_adapter_emits_started_then_completed_with_latency() -> None:
    sink = _Sink()
    inner = _FakeInner()
    with call_context(run_id="r1", step_name="mechanism_step_proposal", attempt=2, retry_index=1, sink=sink):
        adapter = RecordingChatAdapter(inner, CONFIGURED)
        response = adapter.invoke([{"role": "user", "content": "hi"}], tools=[{"x": 1}], tool_choice="auto")

    assert response.content == "ok"
    assert inner.calls[0]["tools"] == [{"x": 1}]
    assert [e["event_type"] for e in sink.events] == ["inference_call_started", "inference_call_completed"]
    started, completed = sink.of("inference_call_started")[0], sink.of("inference_call_completed")[0]
    assert started["call_id"] == completed["call_id"]
    assert started["engine"] == "llm"
    assert started["role"] == "mechanism_proposal"
    assert started["requested_model"] == CONFIGURED
    assert started["attempt"] == 2 and started["retry_index"] == 1
    assert started["started_at"] > 0
    assert completed["status"] == "completed"
    assert completed["resolved_model"] == CONFIGURED
    assert completed["latency_ms"] is not None and completed["latency_ms"] >= 0
    assert completed["usage"]["input_tokens"] == 120
    assert completed["usage"]["output_tokens"] == 30
    assert sink.events[0]["step_name"] == "mechanism_step_proposal"


def test_recording_adapter_emits_failed_and_reraises() -> None:
    sink = _Sink()
    with call_context(run_id="r1", step_name="missing_reagents", attempt=1, retry_index=0, sink=sink):
        adapter = RecordingChatAdapter(_FakeInner(fail=True), CONFIGURED)
        with pytest.raises(RuntimeError, match="provider down"):
            adapter.invoke([])
    kinds = [e["event_type"] for e in sink.events]
    assert kinds == ["inference_call_started", "inference_call_failed"]
    failed = sink.of("inference_call_failed")[0]
    assert failed["status"] == "failed"
    assert failed["error"] == "RuntimeError: provider down"
    assert failed["role"] == "missing_chemistry"


def test_recording_adapter_is_a_passthrough_without_context() -> None:
    assert current_call_context() is None
    sink = _Sink()
    adapter = RecordingChatAdapter(_FakeInner(), CONFIGURED)
    adapter.invoke([])
    assert sink.events == []


def test_context_collects_live_calls_and_clears_on_exit() -> None:
    sink = _Sink()
    with call_context(run_id="r1", step_name="atom_mapping", attempt=1, retry_index=0, sink=sink) as ctx:
        RecordingChatAdapter(_FakeInner(), CONFIGURED).invoke([])
        RecordingChatAdapter(_FakeInner(), "gpt-5").invoke([])
        assert [c.requested_model for c in ctx.calls] == [CONFIGURED, "gpt-5"]
    assert current_call_context() is None


def test_recording_adapter_delegates_other_attributes() -> None:
    inner = _FakeInner()
    adapter = RecordingChatAdapter(inner, CONFIGURED)
    assert adapter._model == CONFIGURED
    assert adapter.inner is inner


# ---------------------------------------------------------------------------
# Factory wrap: only inside a run step
# ---------------------------------------------------------------------------

def test_get_chat_model_wraps_only_inside_call_context(monkeypatch: pytest.MonkeyPatch) -> None:
    import mechanistic_agent.llm as llm_mod

    monkeypatch.setattr(llm_mod, "_build_chat_model", lambda model_name, **_kw: _FakeInner())
    bare = get_chat_model(CONFIGURED)
    assert isinstance(bare, _FakeInner)
    sink = _Sink()
    with call_context(run_id="r1", step_name="initial_conditions", attempt=1, retry_index=0, sink=sink):
        wrapped = get_chat_model(CONFIGURED)
        assert isinstance(wrapped, RecordingChatAdapter)
        assert isinstance(wrapped.inner, _FakeInner)
        wrapped.invoke([])
    assert [e["event_type"] for e in sink.events] == ["inference_call_started", "inference_call_completed"]


# ---------------------------------------------------------------------------
# Coordinator integration: live records replace the derived call
# ---------------------------------------------------------------------------

def _setup(tmp_path: Path):
    store = RunStore(tmp_path / "mechanistic.db")
    run_input = RunInput(starting_materials=["CCBr", "[Cl-]"], products=["CCCl", "[Br-]"])
    run_config = RunConfig(model=CONFIGURED, model_family="openai", max_steps=1, intermediate_prediction_enabled=True)
    run_id = store.create_run(
        mode="unverified",
        input_payload={"starting_materials": run_input.starting_materials, "products": run_input.products},
        config={"model": CONFIGURED},
        prompt_bundle_hash="a", skill_bundle_hash="b", memory_bundle_hash="c",
    )
    state = RunState(run_id=run_id, mode="unverified", run_input=run_input, run_config=run_config)
    state.initialise()
    return RunCoordinator(store=store), store, state


def _events(store: RunStore, run_id: str, kind: str) -> List[Dict[str, Any]]:
    return [e for e in store.list_events(run_id) if e.get("event_type") == kind]


def test_step_started_opens_context_and_record_step_uses_live_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import mechanistic_agent.llm as llm_mod

    monkeypatch.setattr(llm_mod, "_build_chat_model", lambda model_name, **_kw: _FakeInner())
    coordinator, store, state = _setup(tmp_path)

    coordinator._mark_step_started(state, step_name="mechanism_step_proposal", tool_name="propose_mechanism_step", attempt=1)
    ctx = current_call_context()
    assert ctx is not None and ctx.step_name == "mechanism_step_proposal" and ctx.run_id == state.run_id
    # what tools.py does inside the step:
    get_chat_model(CONFIGURED).invoke([{"role": "user", "content": "propose"}])

    coordinator._record_step(
        state,
        StepResult(step_name="mechanism_step_proposal", tool_name="propose_mechanism_step", output={"model_used": CONFIGURED}, source="llm"),
    )

    started = _events(store, state.run_id, "inference_call_started")
    completed = _events(store, state.run_id, "inference_call_completed")
    assert len(started) == 1 and len(completed) == 1, "live call must not be re-derived"
    assert started[0]["payload"]["call_id"] == completed[0]["payload"]["call_id"]
    assert completed[0]["payload"]["latency_ms"] is not None
    prov = _events(store, state.run_id, "step_output")[0]["payload"]["provenance"]
    assert prov["primary_call_id"] == completed[0]["payload"]["call_id"]
    assert prov["call_ids"] == [completed[0]["payload"]["call_id"]]
    assert prov["resolved_model"] == CONFIGURED
    assert current_call_context() is None, "record_step closes the step's context"


def test_provider_fallback_inside_step_yields_failed_then_completed_live_calls(tmp_path: Path) -> None:
    coordinator, store, state = _setup(tmp_path)
    coordinator._mark_step_started(state, step_name="mechanism_step_proposal", tool_name="propose_mechanism_step", attempt=1)
    with pytest.raises(RuntimeError):
        RecordingChatAdapter(_FakeInner(fail=True), CONFIGURED).invoke([])
    RecordingChatAdapter(_FakeInner(), "gpt-5").invoke([])
    coordinator._record_step(
        state,
        StepResult(step_name="mechanism_step_proposal", tool_name="propose_mechanism_step", output={"model_used": "gpt-5"}, source="llm"),
    )
    failed = _events(store, state.run_id, "inference_call_failed")
    completed = _events(store, state.run_id, "inference_call_completed")
    assert len(failed) == 1 and len(completed) == 1
    assert completed[0]["payload"]["fallback_from_call_id"] == failed[0]["payload"]["call_id"]
    prov = _events(store, state.run_id, "step_output")[0]["payload"]["provenance"]
    assert prov["resolved_model"] == "gpt-5"
    assert prov["model_fallback"] is True
    assert prov["fallback_chain"] == ["llm", "llm"]
    assert store.list_step_outputs(state.run_id)[0]["model"] == "gpt-5"


def test_deterministic_step_does_not_inherit_a_stale_context(tmp_path: Path) -> None:
    coordinator, store, state = _setup(tmp_path)
    coordinator._mark_step_started(state, step_name="mechanism_synthesis", tool_name="predict_mechanistic_step", attempt=1)
    coordinator._record_step(
        state, StepResult(step_name="mechanism_synthesis", tool_name="predict_mechanistic_step", output={}, source="deterministic"),
    )
    assert _events(store, state.run_id, "inference_call_started") == []
    assert current_call_context() is None
