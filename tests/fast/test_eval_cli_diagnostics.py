import sys

_maybe_stub = sys.modules.get("mechanistic_agent")
if _maybe_stub is not None and not hasattr(_maybe_stub, "OPTIONAL_LLM_TOOL_NAMES"):
    sys.modules.pop("mechanistic_agent", None)

from main import _build_eval_case_summary, _extract_eval_run_diagnostics, _format_eval_case_result_line


def test_extract_eval_run_diagnostics_prefers_run_failure_and_first_llm_step_error():
    snapshot = {
        "status": "failed",
        "events": [
            {
                "event_type": "run_failed",
                "payload": {"reason": "no_valid_mechanism_steps_generated"},
            }
        ],
        "step_outputs": [
            {
                "step_name": "initial_conditions",
                "source": "llm",
                "output": {
                    "status": "failed",
                    "error": "LLM call failed: OpenRouter API key not configured.",
                },
                "validation": None,
            }
        ],
    }

    diagnostics = _extract_eval_run_diagnostics(snapshot)

    assert diagnostics == {
        "run_status": "failed",
        "failure_reason": "no_valid_mechanism_steps_generated",
        "first_step_error": (
            "initial_conditions: LLM call failed: OpenRouter API key not configured."
        ),
    }


def test_eval_case_summary_and_output_line_include_failure_details():
    snapshot = {
        "status": "failed",
        "events": [
            {"event_type": "run_failed", "payload": {"reason": "no_valid_mechanism_steps_generated"}}
        ],
        "step_outputs": [
            {
                "step_name": "initial_conditions",
                "source": "llm",
                "output": {
                    "status": "failed",
                    "error": "LLM call failed: OpenRouter API key not configured.",
                },
                "validation": None,
            }
        ],
    }
    step_outputs = list(snapshot["step_outputs"])

    summary = _build_eval_case_summary(
        snapshot=snapshot,
        score=0.0,
        passed=False,
        step_outputs=step_outputs,
        case_step_count=4,
        subagent_scores={},
        scored_error=None,
    )

    assert summary["run_status"] == "failed"
    assert summary["failure_reason"] == "no_valid_mechanism_steps_generated"
    assert summary["first_step_error"] == (
        "initial_conditions: LLM call failed: OpenRouter API key not configured."
    )
    assert summary["error"] == "initial_conditions: LLM call failed: OpenRouter API key not configured."

    line = _format_eval_case_result_line(
        index=1,
        case_id="flower_test_000001",
        score=0.0,
        passed=False,
        total_cost=0.0,
        latency_ms=0.0,
        summary=summary,
    )

    assert "status=failed" in line
    assert "reason=no_valid_mechanism_steps_generated" in line
