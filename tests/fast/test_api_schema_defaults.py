from __future__ import annotations

from mechanistic_agent.api.schemas import EvaluateRunRequest, SaveEvaluationRequest


def test_evaluation_requests_default_to_current_gpt_model() -> None:
    assert EvaluateRunRequest().judge_model == "gpt-5.5"
    assert SaveEvaluationRequest().judge_model == "gpt-5.5"
