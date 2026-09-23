"""Jev decision layer: adapter, catalog/routing, policies (PRD §16.1-16.2, Phase A).

All tests use a fake transport; no network and no API key.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import pytest

from mechanistic_agent.decisions.jev import (
    DecisionQuestion,
    DecisionRecord,
    JevDecisionClient,
    encode_question,
    validate_question,
)
from mechanistic_agent.decisions import policies
from mechanistic_agent.core.types import DecisionPolicy, JevConfig

JEV = "typesafe/jev-1.13"


class FakeTransport:
    """Records requests and returns a canned (status, body) or raises."""

    def __init__(self, response: Any = None, *, status: int = 200, raises: Optional[Exception] = None) -> None:
        self.response = response
        self.status = status
        self.raises = raises
        self.calls: List[Tuple[str, Dict[str, str], Dict[str, Any], float]] = []

    def __call__(self, url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float):
        self.calls.append((url, headers, payload, timeout))
        if self.raises is not None:
            raise self.raises
        body = self.response if isinstance(self.response, str) else json.dumps(self.response)
        return self.status, body


def _client(transport: FakeTransport, **kwargs: Any) -> JevDecisionClient:
    return JevDecisionClient(JEV, api_key="sk-test", transport=transport, clock=_Clock(), **kwargs)


class _Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        self.t += 0.125
        return self.t


CHOICE = DecisionQuestion.choice(
    "team", "Which team?", {"billing": "payments", "technical": "bugs", "none": "none of these"}
)
SCORE = DecisionQuestion.score("severity", "How severe?", ["cosmetic", "degraded", "down"])
NOUL = DecisionQuestion.noul("refund", "Refund requested?", true="asks for money back", false="no")

OK_BODY = {
    "id": "gen-dec-1",
    "model": "typesafe/jev-1.13-20260917",
    "provider": "TypeSafe",
    "answers": {
        "team": {"type": "choice", "choice": "billing", "confidence": 0.8,
                 "probabilities": {"billing": 0.87, "technical": 0.13, "none": 0.0}},
        "severity": {"type": "score", "score": 1.05, "confidence": 0.92,
                     "legend": {"0": "cosmetic", "1": "degraded", "2": "down"},
                     "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05}},
        "refund": {"type": "noul", "noul": 0.98},
    },
    "usage": {"cost": 0.000019992, "input_tokens": 476, "output_tokens": 70},
}


# ---------------------------------------------------------------------------
# Encoding and validation
# ---------------------------------------------------------------------------
def test_encode_question_wire_format() -> None:
    assert encode_question(CHOICE) == {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "payments", "technical": "bugs", "none": "none of these"},
    }
    assert encode_question(SCORE)["criteria"] == ["cosmetic", "degraded", "down"]
    assert encode_question(NOUL)["criteria"] == {"true": "asks for money back", "false": "no"}
    assert "criteria" not in encode_question(DecisionQuestion.noul("q", "Is it?"))


def test_choice_requires_none_option() -> None:
    with pytest.raises(ValueError, match="none"):
        validate_question(DecisionQuestion.choice("q", "pick", {"a": "A", "b": "B"}))
    validate_question(DecisionQuestion.choice("q", "pick", {"a": "A", "no_match": "neither"}))


def test_choice_option_limit_and_score_level_limits() -> None:
    options = {f"o{i}": "x" for i in range(255)}
    options["none"] = "none"
    with pytest.raises(ValueError, match="limit"):
        validate_question(DecisionQuestion.choice("q", "pick", options))
    for levels in (["only"], [str(i) for i in range(11)]):
        with pytest.raises(ValueError, match="levels"):
            validate_question(DecisionQuestion.score("s", "rate", levels))
    validate_question(DecisionQuestion.score("s", "rate", ["lo", "hi"]))


def test_unknown_question_type_and_duplicate_keys_raise() -> None:
    client = _client(FakeTransport(OK_BODY))
    with pytest.raises(ValueError):
        client.build_payload("s", [DecisionQuestion(key="x", type="rank", instructions="?")])
    with pytest.raises(ValueError, match="duplicate"):
        client.build_payload("s", [CHOICE, CHOICE])


# ---------------------------------------------------------------------------
# decide_many / parsing
# ---------------------------------------------------------------------------
def test_decide_many_sends_one_request_and_parses_all_types() -> None:
    transport = FakeTransport(OK_BODY)
    records = _client(transport, timeout=7.5).decide_many({"text": "payouts failing"}, [CHOICE, SCORE, NOUL])

    assert len(transport.calls) == 1
    url, headers, payload, timeout = transport.calls[0]
    assert url == "https://openrouter.ai/api/alpha/decisions"
    assert headers["Authorization"] == "Bearer sk-test"
    assert timeout == 7.5
    assert payload["model"] == JEV
    assert payload["state"] == {"text": "payouts failing"}
    assert set(payload["questions"]) == {"team", "severity", "refund"}

    team = records["team"]
    assert team.ok and team.selected == "billing"
    assert team.confidence == 0.8
    assert math.isclose(sum(team.probabilities.values()), 1.0)
    assert team.model == JEV
    assert team.model_version == "typesafe/jev-1.13-20260917"
    assert team.provider == "openrouter"
    assert team.latency_ms == 125.0
    assert team.request_id == "gen-dec-1" and team.called is True
    assert team.usage == {"input_tokens": 476, "cached_input_tokens": 0, "output_tokens": 70, "total_tokens": 546}
    assert math.isclose(team.cost["total_cost"], 0.000019992)

    severity = records["severity"]
    assert severity.selected == 1 and severity.score == 1.05
    assert severity.probabilities == {"0": 0.0, "1": 0.95, "2": 0.05}
    assert severity.legend["2"] == "down"

    refund = records["refund"]
    assert refund.selected is True
    assert refund.confidence is None  # Noul has no confidence
    assert math.isclose(refund.probabilities["true"], 0.98)
    assert math.isclose(refund.probabilities["false"], 0.02)


def test_cost_falls_back_to_catalog_input_only_pricing() -> None:
    body = dict(OK_BODY, usage={"input_tokens": 1_000_000, "output_tokens": 500})
    record = _client(FakeTransport(body)).decide_many("s", [CHOICE])["team"]
    assert math.isclose(record.cost["input_cost"], 0.042)
    assert record.cost["output_cost"] == 0.0
    assert math.isclose(record.cost["total_cost"], 0.042)


def test_missing_api_key_makes_no_request() -> None:
    transport = FakeTransport(OK_BODY)
    record = JevDecisionClient(JEV, api_key=None, transport=transport).choice(
        "s", "team", "Which team?", {"billing": "b", "none": "n"}
    )
    assert transport.calls == []
    assert record.failure == "missing_api_key" and record.called is False


@pytest.mark.parametrize(
    "transport,failure",
    [
        (FakeTransport(raises=TimeoutError("read timed out")), "timeout"),
        (FakeTransport(raises=ConnectionError("refused")), "transport_error"),
        (FakeTransport({"error": "boom"}, status=500), "http_500"),
        (FakeTransport("<html>not json</html>"), "malformed_response"),
        (FakeTransport(["not", "an", "object"]), "malformed_response"),
        (FakeTransport({"id": "x", "usage": {}}), "malformed_response"),
    ],
)
def test_runtime_failures_become_failed_records(transport: FakeTransport, failure: str) -> None:
    records = _client(transport).decide_many("s", [CHOICE, NOUL])
    for record in records.values():
        assert record.failure == failure
        assert record.called is True
        assert record.selected is None


def test_missing_probabilities_and_missing_answer() -> None:
    body = {
        "id": "r",
        "answers": {
            "team": {"type": "choice", "choice": "billing", "confidence": 0.9},
            "severity": {"type": "score", "score": 1.0, "probabilities": {}},
            # refund missing entirely
        },
    }
    records = _client(FakeTransport(body)).decide_many("s", [CHOICE, SCORE, NOUL])
    assert records["team"].failure == "missing_probabilities"
    assert records["severity"].failure == "missing_probabilities"
    assert records["refund"].failure == "missing_answer"


def test_noul_without_value_is_missing_probabilities() -> None:
    body = {"id": "r", "answers": {"refund": {"type": "noul"}}}
    record = _client(FakeTransport(body)).decide_many("s", [NOUL])["refund"]
    assert record.failure == "missing_probabilities"


def test_choice_distribution_is_normalised_and_unknown_choice_rejected() -> None:
    body = {"id": "r", "answers": {"team": {
        "type": "choice", "choice": "technical",
        "probabilities": {"billing": 0.2, "technical": 0.6, "sales": 0.3},
    }}}
    record = _client(FakeTransport(body)).decide_many("s", [CHOICE])["team"]
    assert record.ok
    assert set(record.probabilities) == {"billing", "technical", "none"}
    assert math.isclose(sum(record.probabilities.values()), 1.0)
    assert math.isclose(record.probabilities["technical"], 0.75)
    assert record.confidence is None and "missing_confidence" in record.notes
    assert any(note.startswith("dropped_unknown_keys") for note in record.notes)

    bad = {"id": "r", "answers": {"team": {"type": "choice", "choice": "sales",
                                            "probabilities": {"billing": 1.0}}}}
    assert _client(FakeTransport(bad)).decide_many("s", [CHOICE])["team"].failure == "unknown_choice"


def test_request_too_large_is_refused_before_sending() -> None:
    transport = FakeTransport(OK_BODY)
    record = _client(transport).decide_many("x" * 200_000, [CHOICE])["team"]
    assert transport.calls == []
    assert record.failure == "request_too_large"


def test_trace_has_prd_18_fields() -> None:
    records = _client(FakeTransport(OK_BODY)).decide_many("s", [CHOICE, NOUL])
    trace = records["team"].to_trace()
    for key in ("decision_engine", "model", "decision_type", "question_id", "selected", "probabilities",
                "confidence", "fallback_triggered", "fallback_reason", "latency_ms", "cost"):
        assert key in trace
    assert trace["decision_engine"] == "jev"
    assert trace["decision_type"] == "choice"
    assert trace["confidence"] == 0.8
    assert math.isclose(trace["cost"], 0.000019992)
    noul_trace = records["refund"].to_trace(fallback_triggered=True, fallback_reason="x")
    assert noul_trace["confidence"] is None
    assert noul_trace["fallback_triggered"] is True and noul_trace["fallback_reason"] == "x"
    json.dumps(trace)  # serialisable


# ---------------------------------------------------------------------------
# Catalog and routing
# ---------------------------------------------------------------------------
def test_catalog_entry_is_a_decision_model() -> None:
    from mechanistic_agent.model_registry import (
        get_default_decision_model,
        get_model_kind,
        get_model_options,
        get_model_provider,
        get_model_spec,
        is_decision_model,
        model_supports_tools,
    )

    spec = get_model_spec(JEV)
    assert spec["provider"] == "openrouter"
    assert spec["model_kind"] == "decision"
    assert spec["supports_tools"] is False
    assert spec["pricing_per_million"]["output"] == 0.0
    assert get_model_provider(JEV) == "openrouter"
    assert get_model_kind(JEV) == "decision"
    assert get_model_kind("typesafe/jev-1.13-20260917") == "decision"  # versioned response id resolves
    assert get_model_kind("gpt-5") == "chat"
    assert is_decision_model(JEV) and not is_decision_model("gpt-5")
    assert model_supports_tools(JEV) is False
    assert get_default_decision_model() == JEV
    assert JEV not in {row["id"] for row in get_model_options()}
    assert JEV in {row["id"] for row in get_model_options(include_decision=True)}


def test_input_only_cost_calculation() -> None:
    from mechanistic_agent.model_registry import calculate_cost

    cost = calculate_cost(JEV, {"input_tokens": 2_000_000, "cached_input_tokens": 0, "output_tokens": 999})
    assert math.isclose(cost["input_cost"], 0.084)
    assert cost["output_cost"] == 0.0
    assert math.isclose(cost["total_cost"], 0.084)


def test_get_chat_model_refuses_decision_model() -> None:
    from mechanistic_agent.llm import get_chat_model

    with pytest.raises(ValueError, match="decision model"):
        get_chat_model(JEV, user_api_key="sk-test")


def test_get_decision_model_routes_to_jev_client_with_openrouter_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from mechanistic_agent.llm import get_decision_model

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env")
    client = get_decision_model(timeout=3.0)
    assert isinstance(client, JevDecisionClient)
    assert client.model == JEV and client.api_model_id == JEV
    assert client.api_key == "sk-or-env" and client.timeout == 3.0
    assert get_decision_model(JEV, user_api_key="sk-user").api_key == "sk-user"
    with pytest.raises(ValueError):
        get_decision_model("gpt-5")


def test_client_refuses_ids_outside_the_catalog() -> None:
    with pytest.raises(ValueError):
        JevDecisionClient("~typesafe/jev-latest", api_key="k")
    with pytest.raises(ValueError, match="not a decision model"):
        JevDecisionClient("gpt-5", api_key="k")


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
def test_thresholds_default_to_observational() -> None:
    cfg = JevConfig()
    assert all(value is None for value in cfg.thresholds.values())
    assert policies.resolve_threshold(cfg, "reaction_type_active_probability") is None
    assert policies.passes_threshold(0.9, None) is None
    assert policies.passes_threshold(0.9, 0.8) is True
    assert policies.passes_threshold(0.7, 0.8) is False


def test_reaction_type_gates() -> None:
    llm = policies.resolve_reaction_type_gates(
        decision_engine="llm", jev=JevConfig(thresholds={"reaction_type_active_probability": 0.4}),
        run_confidence_threshold=0.65, run_margin_threshold=0.1,
    )
    assert llm == {"confidence_threshold": 0.65, "margin_threshold": 0.1, "source": "run_config"}
    jev_unset = policies.resolve_reaction_type_gates(
        decision_engine="jev", jev=JevConfig(), run_confidence_threshold=0.65, run_margin_threshold=0.1,
    )
    assert jev_unset["confidence_threshold"] == 0.65 and jev_unset["source"] == "run_config"
    jev_set = policies.resolve_reaction_type_gates(
        decision_engine="jev",
        jev=JevConfig(thresholds={"reaction_type_active_probability": 0.4, "reaction_type_min_margin": None}),
        run_confidence_threshold=0.65, run_margin_threshold=0.1,
    )
    assert jev_set == {"confidence_threshold": 0.4, "margin_threshold": 0.1, "source": "harness_jev"}


def test_example_bypass_precedence() -> None:
    off = DecisionPolicy(example_reaction_type_bypass=False)
    assert policies.resolve_example_bypass(run_value=None, policy=None, env={}) == (True, "harness")
    assert policies.resolve_example_bypass(run_value=None, policy=off, env={}) == (False, "harness")
    env = {policies.EXAMPLE_BYPASS_ENV: "0"}
    assert policies.resolve_example_bypass(run_value=None, policy=None, env=env) == (False, "env")
    assert policies.resolve_example_bypass(run_value=True, policy=off, env=env) == (True, "run_config")


def test_choice_margin_and_top_candidates() -> None:
    probs = {"a": 0.5, "b": 0.3, "c": 0.3, "none": 0.0}
    assert math.isclose(policies.choice_margin(probs), 0.2)
    assert policies.top_candidates(probs, 2) == [("a", 0.5), ("b", 0.3)]
    assert policies.choice_margin({"a": 1.0}) is None


def test_decision_record_ok_property() -> None:
    assert DecisionRecord(question_id="q", decision_type="noul", model=JEV).ok
    assert not DecisionRecord(question_id="q", decision_type="noul", model=JEV, failure="timeout").ok
