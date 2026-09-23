"""M0 provenance tests (docs/PRD_live_mechanism_observatory_chemillusion.md §12–§13, §39).

Every step output must say which engine did the work, deterministic steps must
not inherit the run's LLM model, Jev and LLM calls inside one step stay
separate, and a replay from persisted events reproduces the same answer.
"""
from __future__ import annotations

from typing import Any, Dict

import pytest

pytest.importorskip("rdkit")

from mechanistic_agent.core.provenance import (
    EVENT_SCHEMA_VERSION,
    InferenceCall,
    planned_provenance,
    provenance_inventory_from_events,
    resolve_step_provenance,
    step_provenance_from_events,
)
from mechanistic_agent.core.types import StepResult


CONFIGURED = "anthropic/claude-opus-4.6"


def _llm_result(step_name: str = "mechanism_step_proposal", **kw: Any) -> StepResult:
    base: Dict[str, Any] = dict(step_name=step_name, tool_name="propose_mechanism_step", output={}, source="llm")
    base.update(kw)
    return StepResult(**base)


# ---------------------------------------------------------------------------
# resolve_step_provenance
# ---------------------------------------------------------------------------

def test_llm_step_without_explicit_model_resolves_to_configured_model() -> None:
    prov = resolve_step_provenance(_llm_result(), configured_model=CONFIGURED, configured_reasoning="high")
    assert prov.engine == "llm"
    assert prov.resolved_model == CONFIGURED
    assert prov.resolved_reasoning == "high"
    assert len(prov.calls) == 1
    call = prov.calls[0]
    assert call.engine == "llm"
    assert call.role == "mechanism_proposal"
    assert call.requested_model == CONFIGURED
    assert call.resolved_model == CONFIGURED
    assert call.status == "completed"
    assert call.call_id.startswith("call_")


def test_llm_step_reports_fallback_model_from_output_model_used() -> None:
    """tools.py sets output.model_used = fallback_model after a provider fallback (§3.7.2)."""
    result = _llm_result(output={"model_used": "openai/gpt-5.6-sol"}, model=CONFIGURED)
    prov = resolve_step_provenance(result, configured_model=CONFIGURED, configured_reasoning=None)
    assert prov.resolved_model == "openai/gpt-5.6-sol"
    call = prov.calls[0]
    assert call.requested_model == CONFIGURED
    assert call.resolved_model == "openai/gpt-5.6-sol"
    assert call.model_fallback is True


def test_deterministic_step_has_no_model_and_no_model_calls() -> None:
    result = StepResult(
        step_name="bond_electron_validation",
        tool_name="bond_electron_validation",
        output={"check": "dbe_metadata", "passed": True},
        source="deterministic",
    )
    prov = resolve_step_provenance(result, configured_model=CONFIGURED, configured_reasoning="high")
    assert prov.engine == "deterministic"
    assert prov.resolved_model is None
    assert prov.resolved_reasoning is None
    assert prov.tool == "bond_electron_validation"
    assert prov.calls == []


def test_mechanism_synthesis_regression_no_llm_model() -> None:
    """Regression guard from PRD §35: mechanism_synthesis is deterministic."""
    result = StepResult(step_name="mechanism_synthesis", tool_name="predict_mechanistic_step", output={}, source="deterministic")
    prov = resolve_step_provenance(result, configured_model=CONFIGURED, configured_reasoning=None)
    assert prov.resolved_model is None
    assert prov.engine == "deterministic"


def test_human_step_normalizes_to_human_engine() -> None:
    result = StepResult(
        step_name="mechanism_synthesis",
        tool_name="human_submitted_mechanistic_step",
        output={},
        model="human_input",
        source="human",
    )
    prov = resolve_step_provenance(result, configured_model=CONFIGURED, configured_reasoning=None)
    assert prov.engine == "human"
    assert prov.resolved_model is None
    assert prov.calls == []


def _jev_trace(*, failure: str | None = None, request_id: str = "gen-abc123") -> Dict[str, Any]:
    return {
        "decision_engine": "jev",
        "question_id": "reaction_type",
        "decision_type": "choice",
        "model": "typesafe/jev-1.13",
        "model_version": "typesafe/jev-1.13-20260917",
        "provider": "openrouter",
        "selected": None if failure else "rt_042",
        "confidence": None if failure else 0.78,
        "latency_ms": 110.0,
        "usage": {"total_tokens": 900},
        "cost_breakdown": {"total_cost": 0.0004},
        "request_id": request_id,
        "called": True,
        "failure": failure,
    }


def test_jev_step_maps_decision_record_onto_provenance_schema() -> None:
    result = StepResult(
        step_name="reaction_type_mapping",
        tool_name="select_reaction_type",
        output={"model_used": "typesafe/jev-1.13", "decision_engine": "jev", "decision_trace": [_jev_trace()]},
        model="typesafe/jev-1.13",
        source="jev",
    )
    prov = resolve_step_provenance(result, configured_model=CONFIGURED, configured_reasoning="high")
    assert prov.engine == "jev"
    assert prov.resolved_model == "typesafe/jev-1.13-20260917"
    assert prov.resolved_reasoning is None  # decision models have no reasoning level
    assert len(prov.calls) == 1
    call = prov.calls[0]
    assert call.engine == "jev"
    assert call.role == "reaction_type"
    assert call.call_id == "gen-abc123"
    assert call.provider == "openrouter"
    assert call.requested_model == "typesafe/jev-1.13"
    assert call.resolved_model == "typesafe/jev-1.13-20260917"
    assert call.decision_type == "choice"
    assert call.status == "completed"
    assert call.latency_ms == 110.0
    assert call.usage == {"total_tokens": 900}


def test_jev_failure_then_llm_fallback_yields_two_ordered_calls() -> None:
    """One logical step, two engines (PRD §12.1, §13.1). The LLM call points back at the failed Jev call."""
    result = StepResult(
        step_name="reaction_type_mapping",
        tool_name="select_reaction_type",
        output={
            "model_used": CONFIGURED,
            "decision_engine": "llm",
            "jev_fallback": {"mode": "llm", "reason": "timeout"},
            "decision_trace": [_jev_trace(failure="timeout", request_id="local-deadbeef")],
        },
        model=CONFIGURED,
        source="llm",
    )
    prov = resolve_step_provenance(result, configured_model=CONFIGURED, configured_reasoning="low")
    assert prov.engine == "llm"
    assert prov.resolved_model == CONFIGURED
    assert [c.engine for c in prov.calls] == ["jev", "llm"]
    jev_call, llm_call = prov.calls
    assert jev_call.status == "failed"
    assert jev_call.error == "timeout"
    assert llm_call.status == "completed"
    assert llm_call.fallback_from_call_id == jev_call.call_id == "local-deadbeef"
    assert prov.primary_call_id == llm_call.call_id


def test_decision_trace_entry_that_never_called_is_not_a_call() -> None:
    trace = _jev_trace(failure="missing_api_key")
    trace["called"] = False
    result = StepResult(
        step_name="reaction_type_mapping",
        tool_name="select_reaction_type",
        output={"decision_trace": [trace], "decision_engine": "llm"},
        source="llm",
    )
    prov = resolve_step_provenance(result, configured_model=CONFIGURED, configured_reasoning=None)
    assert [c.engine for c in prov.calls] == ["llm"]
    assert prov.calls[0].fallback_from_call_id is None


def test_step_provenance_summary_dict_shape() -> None:
    prov = resolve_step_provenance(_llm_result(), configured_model=CONFIGURED, configured_reasoning="high")
    summary = prov.as_dict()
    assert summary["engine"] == "llm"
    assert summary["resolved_model"] == CONFIGURED
    assert summary["primary_call_id"] == prov.calls[0].call_id
    assert summary["call_ids"] == [prov.calls[0].call_id]
    assert summary["event_schema_version"] == EVENT_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# planned_provenance (step_started)
# ---------------------------------------------------------------------------

def test_planned_provenance_for_llm_step_uses_step_models() -> None:
    planned = planned_provenance(
        "mechanism_step_proposal",
        step_models={"intermediates": "openai/gpt-5.6-sol"},
        default_model=CONFIGURED,
        step_reasoning={"intermediates": "high"},
        reaction_type_policy="llm",
    )
    assert planned == {"planned_engine": "llm", "planned_model": "openai/gpt-5.6-sol", "planned_reasoning": "high"}


def test_planned_provenance_for_deterministic_step_has_no_model() -> None:
    planned = planned_provenance(
        "atom_balance_validation",
        step_models={"mechanism_synthesis": CONFIGURED},
        default_model=CONFIGURED,
        step_reasoning={},
        reaction_type_policy="llm",
    )
    assert planned == {"planned_engine": "deterministic", "planned_model": None, "planned_reasoning": None}


def test_planned_provenance_for_mechanism_synthesis_is_deterministic_despite_step_models_entry() -> None:
    """LLM_STEP_KEYS lists mechanism_synthesis; that must not make it look like a model step (§3.7.1)."""
    planned = planned_provenance(
        "mechanism_synthesis",
        step_models={"mechanism_synthesis": CONFIGURED},
        default_model=CONFIGURED,
        step_reasoning={},
        reaction_type_policy="llm",
    )
    assert planned["planned_engine"] == "deterministic"
    assert planned["planned_model"] is None


def test_planned_provenance_reaction_type_follows_decision_policy() -> None:
    planned = planned_provenance(
        "reaction_type_mapping",
        step_models={"reaction_type_mapping": CONFIGURED},
        default_model=CONFIGURED,
        step_reasoning={},
        reaction_type_policy="jev",
        jev_model="typesafe/jev-1.13",
    )
    assert planned == {"planned_engine": "jev", "planned_model": "typesafe/jev-1.13", "planned_reasoning": None}


# ---------------------------------------------------------------------------
# Replay from events
# ---------------------------------------------------------------------------

def _events_for(prov_by_step: Dict[str, Any]) -> list[Dict[str, Any]]:
    events = []
    seq = 0
    for step_name, prov in prov_by_step.items():
        for call in prov.calls:
            seq += 1
            events.append({
                "seq": seq,
                "event_type": "inference_call_completed" if call.status == "completed" else "inference_call_failed",
                "step_name": step_name,
                "payload": call.to_dict(),
            })
        seq += 1
        events.append({
            "seq": seq,
            "event_type": "step_output",
            "step_name": step_name,
            "payload": {"step_name": step_name, "attempt": 1, "retry_index": 0, "source": prov.engine, "provenance": prov.as_dict()},
        })
    return events


def test_replay_from_events_reproduces_step_engines_and_models() -> None:
    llm = resolve_step_provenance(_llm_result(), configured_model=CONFIGURED, configured_reasoning="high")
    det = resolve_step_provenance(
        StepResult(step_name="mechanism_synthesis", tool_name="predict_mechanistic_step", output={}, source="deterministic"),
        configured_model=CONFIGURED,
        configured_reasoning="high",
    )
    events = _events_for({"mechanism_step_proposal": llm, "mechanism_synthesis": det})

    by_step = step_provenance_from_events(events)
    assert by_step["mechanism_step_proposal"]["engine"] == "llm"
    assert by_step["mechanism_step_proposal"]["resolved_model"] == CONFIGURED
    assert by_step["mechanism_synthesis"]["engine"] == "deterministic"
    assert by_step["mechanism_synthesis"]["resolved_model"] is None


def test_inventory_lists_models_by_engine_from_completed_calls_only() -> None:
    jev_ok = resolve_step_provenance(
        StepResult(
            step_name="reaction_type_mapping", tool_name="select_reaction_type", source="jev",
            output={"decision_trace": [_jev_trace()]},
        ),
        configured_model=CONFIGURED, configured_reasoning=None,
    )
    llm_fb = resolve_step_provenance(
        StepResult(
            step_name="missing_reagents", tool_name="predict_missing_reagents", source="llm",
            output={"model_used": "openai/gpt-5.6-sol"},
        ),
        configured_model=CONFIGURED, configured_reasoning=None,
    )
    inv = provenance_inventory_from_events(_events_for({"reaction_type_mapping": jev_ok, "missing_reagents": llm_fb}))
    assert inv["models_by_engine"] == {
        "jev": ["typesafe/jev-1.13-20260917"],
        "llm": ["openai/gpt-5.6-sol"],
    }
    assert inv["call_counts"] == {"jev": 1, "llm": 1}
    assert inv["failed_calls"] == 0


def test_inference_call_to_dict_round_trips() -> None:
    call = InferenceCall(call_id="call_x", engine="llm", role="mechanism_proposal", step_name="mechanism_step_proposal")
    d = call.to_dict()
    assert d["event_schema_version"] == EVENT_SCHEMA_VERSION
    assert InferenceCall.from_dict(d) == call


def test_llm_step_whose_call_failed_is_a_failed_call() -> None:
    """tools.py catches the provider error and returns status=failed/fallback with an error string."""
    for status in ("failed", "fallback"):
        result = _llm_result(output={"status": status, "error": "LLM call failed: provider down", "model_used": CONFIGURED})
        prov = resolve_step_provenance(result, configured_model=CONFIGURED, configured_reasoning=None)
        assert len(prov.calls) == 1, status
        call = prov.calls[0]
        assert call.status == "failed", status
        assert call.error == "LLM call failed: provider down"
        assert prov.resolved_model == CONFIGURED  # the model that was asked is still the step's model


def test_llm_step_with_fallback_status_but_no_error_is_completed() -> None:
    result = _llm_result(output={"status": "fallback", "model_used": CONFIGURED})
    prov = resolve_step_provenance(result, configured_model=CONFIGURED, configured_reasoning=None)
    assert prov.calls[0].status == "completed"
