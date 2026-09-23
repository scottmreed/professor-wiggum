"""Instrumentation tests for the per-run LLM-call counter.

Covers PRD_jev_atom_identity_mechanistic.md §18 "Instrumentation gaps" / §19
Phase 0: an explicit tally of real model invocations (by step/call name and
by engine), correct exclusion of the deterministic ``mechanism_synthesis``
step from that tally, and retry token usage being folded into the step that
made the retry rather than silently dropped.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pytest

pytest.importorskip("rdkit")

from mechanistic_agent.core.db import RunStore
from mechanistic_agent.core.subagents import MechanismAgent
from mechanistic_agent.core.types import RunConfig, RunInput, RunState
from mechanistic_agent.tools import predict_missing_reagents


def _state() -> RunState:
    run_input = RunInput(starting_materials=["C=O", "OCCO"], products=["C1OCOC1"], ph=3.5, temperature_celsius=40.0)
    run_config = RunConfig(
        model="gpt-4",
        model_family="openai",
        max_steps=1,
        intermediate_prediction_enabled=True,
    )
    state = RunState(run_id="run-test-id", mode="unverified", run_input=run_input, run_config=run_config)
    state.initialise()
    return state


def test_mechanism_agent_reports_deterministic_source() -> None:
    """`mechanism_synthesis` makes no model call; it must not be tagged "llm"."""

    @dataclass
    class _StubExecutor:
        def run_mechanism_step(self, **kwargs: Any) -> Dict[str, Any]:
            return {"contains_target_product": False}

    agent = MechanismAgent(executor=_StubExecutor())  # type: ignore[arg-type]
    state = _state()
    output = {
        "selected_candidate": {
            "rank": 1,
            "intermediate_smiles": "C1OCOC1",
            "resulting_state": ["C1OCOC1", "O"],
        }
    }

    result = agent.run(state, output)

    assert result.step_name == "mechanism_synthesis"
    assert result.source == "deterministic"


def test_run_store_call_summary_counts_llm_steps_and_excludes_deterministic(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "mechanistic.db")

    run_id = store.create_run(
        mode="unverified",
        input_payload={"starting_materials": ["C=O"], "products": ["CO"]},
        config={"model": "gpt-5", "max_steps": 2},
        prompt_bundle_hash="a",
        skill_bundle_hash="b",
        memory_bundle_hash="c",
    )

    # Step 1: a real LLM pre-loop call.
    store.record_step_output(
        run_id=run_id,
        step_name="initial_conditions",
        attempt=1,
        source="llm",
        model="gpt-5",
        reasoning_level="low",
        tool_name="assess_initial_conditions",
        output={"environment": "neutral"},
        validation=None,
        usage={"input_tokens": 100, "cached_input_tokens": 0, "output_tokens": 40, "total_tokens": 140},
        cost={"input_cost": 0.001, "cached_input_cost": 0.0, "output_cost": 0.002, "total_cost": 0.003},
    )

    # Step 2: missing_reagents — usage here mirrors a step whose internal
    # retry usage has already been merged by tools.py into one usage blob
    # (see test_predict_missing_reagents_retry_merges_usage below); at the
    # step_outputs granularity this is still exactly one recorded call.
    store.record_step_output(
        run_id=run_id,
        step_name="missing_reagents",
        attempt=1,
        source="llm",
        model="gpt-5",
        reasoning_level="low",
        tool_name="predict_missing_reagents",
        output={"status": "success"},
        validation=None,
        usage={"input_tokens": 160, "cached_input_tokens": 0, "output_tokens": 35, "total_tokens": 195},
        cost={"input_cost": 0.0016, "cached_input_cost": 0.0, "output_cost": 0.0007, "total_cost": 0.0023},
    )

    # mechanism_synthesis: deterministic, no model call, no usage. Must not
    # be counted as an "llm" call even though it was historically mislabeled.
    store.record_step_output(
        run_id=run_id,
        step_name="mechanism_synthesis",
        attempt=1,
        source="deterministic",
        model="gpt-5",
        reasoning_level="low",
        tool_name="predict_mechanistic_step",
        output={"contains_target_product": True},
        validation={"passed": True, "checks": []},
    )

    summary = store.get_run_cost_summary(run_id)
    call_summary = summary["call_summary"]

    assert call_summary["total_calls"] == 2
    assert call_summary["llm_calls"] == 2
    assert call_summary["llm_tokens"] == 140 + 195

    assert call_summary["by_step"]["initial_conditions"]["calls"] == 1
    assert call_summary["by_step"]["missing_reagents"]["calls"] == 1
    assert "mechanism_synthesis" not in call_summary["by_step"]

    assert call_summary["by_engine"]["llm"]["calls"] == 2
    assert call_summary["by_engine"]["llm"]["usage"]["total_tokens"] == 335
    assert "jev" not in call_summary["by_engine"]

    # Existing totals (unaffected by the new call_summary) still aggregate
    # across every step that carried usage/cost.
    assert summary["total_usage"]["total_tokens"] == 140 + 195
    assert summary["total_cost"]["total_cost"] == pytest.approx(0.003 + 0.0023)


def test_predict_missing_reagents_retry_merges_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reagent-retry re-prompt is a second real model call; its tokens
    must be merged into the step's `_llm_usage`, not discarded."""

    class _StubResponse:
        def __init__(self, payload: Dict[str, Any], usage: Dict[str, int]) -> None:
            self.tool_calls = [{"arguments": json.dumps(payload)}]
            self.usage = usage

    class _StubLLM:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, *_args: Any, **_kwargs: Any) -> _StubResponse:
            self.calls += 1
            if self.calls == 1:
                return _StubResponse(
                    {"missing_reactants": ["Cl"], "missing_products": []},
                    {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                )
            return _StubResponse(
                {"missing_reactants": ["Cl"], "missing_products": ["O"]},
                {"prompt_tokens": 60, "completion_tokens": 15, "total_tokens": 75},
            )

    validation_calls: List[Dict[str, Any]] = []

    def _stub_validate(reactants, missing_products, _starting, _products):  # noqa: ANN001
        validation_calls.append({"reactants": list(reactants), "missing_products": list(missing_products)})
        if len(validation_calls) == 1:
            return json.dumps(
                {
                    "status": "failed",
                    "is_balanced": False,
                    "invalid_reagents": [],
                    "remaining_deficit": {"O": 1},
                    "remaining_surplus": {},
                    "reason": "not balanced",
                }
            )
        return json.dumps(
            {
                "status": "success",
                "is_balanced": True,
                "valid_reagents": list(reactants) + list(missing_products),
            }
        )

    def _stub_cli(command, args, **_kwargs):  # noqa: ANN001
        if command == "balance":
            return {
                "command": "balance",
                "status": "ok",
                "output": {"fix_suggestions": ["add O to products"], "remaining_deficit": {"O": 1}},
            }
        return {"command": command, "status": "ok", "output": {}}

    llm = _StubLLM()
    monkeypatch.setattr("mechanistic_agent.tools.adapter_supports_forced_tools", lambda _model: True)
    monkeypatch.setattr("mechanistic_agent.tools.get_model_api_key", lambda *_args, **_kwargs: "test-key")
    monkeypatch.setattr("mechanistic_agent.tools.get_chat_model", lambda *_args, **_kwargs: llm)
    monkeypatch.setattr("mechanistic_agent.tools.validate_proposed_reagents", _stub_validate)
    monkeypatch.setattr("mechanistic_agent.tools._run_rdkit_cli_command", _stub_cli)

    raw = predict_missing_reagents(
        starting_materials=["CCBr"],
        products=["CCCl"],
    )
    payload = json.loads(raw)

    assert payload["status"] == "success"
    assert llm.calls == 2  # confirms a retry actually happened

    usage = payload["_llm_usage"]
    assert usage["prompt_tokens"] == 100 + 60
    assert usage["completion_tokens"] == 20 + 15
    assert usage["total_tokens"] == 120 + 75
