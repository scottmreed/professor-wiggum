"""Harness-free single-shot baseline mechanism evaluation.

This module provides ``BaselineRunner``, which asks a selected model for the
complete reaction mechanism in a *single* tool-calling step (no multi-step
harness pipeline).  Results are stored in the same eval framework so they can
be compared directly against harness runs on the leaderboard.

The run_group_name convention for baseline eval runs is ``"harness_free_baseline"``.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from mechanistic_agent.llm import (
    adapter_supports_forced_tools,
    extract_text_content,
    get_chat_model,
)
from mechanistic_agent.tool_schemas import PREDICT_FULL_MECHANISM_TOOL, build_tool_choice

# Sentinel group name for baseline runs – checked by leaderboard() to set is_baseline.
BASELINE_GROUP_PREFIX = "harness_free_baseline"
SIMULATED_GROUP_PREFIX = "[SIMULATED]"


def _load_baseline_system_prompt() -> str:
    """Load and compose the system prompt for the baseline single-shot call."""
    base_path = Path(__file__).resolve().parent.parent.parent / "prompt_versions" / "shared" / "base_system.md"
    call_path = (
        Path(__file__).resolve().parent.parent.parent
        / "prompt_versions"
        / "calls"
        / "baseline_mechanism"
        / "base.md"
    )
    parts: List[str] = []
    if base_path.exists():
        parts.append(base_path.read_text(encoding="utf-8").strip())
    if call_path.exists():
        parts.append(call_path.read_text(encoding="utf-8").strip())
    return "\n\n".join(parts) if parts else "You are an expert organic chemist."


def _build_user_message(
    starting_materials: List[str],
    products: List[str],
    temperature_celsius: float = 25.0,
    ph: Optional[float] = None,
) -> str:
    lines = [
        "Predict the complete stepwise mechanism for the following reaction.",
        "",
        f"Starting materials: {', '.join(starting_materials)}",
        f"Target products: {', '.join(products)}",
    ]
    if ph is not None:
        lines.append(f"pH: {ph}")
    lines.append(f"Temperature: {temperature_celsius} °C")
    lines.append("")
    lines.append(
        "Call predict_full_mechanism with all elementary steps from starting materials "
        "to target products. Each step must be a single bond-level elementary event."
    )
    return "\n".join(lines)


def _parse_tool_call_arguments(tool_calls: List[Any]) -> Optional[Dict[str, Any]]:
    """Extract the first predict_full_mechanism call arguments."""
    for tc in tool_calls or []:
        name = ""
        args_raw = ""
        if isinstance(tc, dict):
            name = str(tc.get("name") or "")
            args_raw = str(tc.get("arguments") or "")
        else:
            name = str(getattr(tc, "name", "") or "")
            args_raw = str(getattr(tc, "arguments", "") or "")

        if name == "predict_full_mechanism":
            try:
                return json.loads(args_raw) if args_raw else {}
            except json.JSONDecodeError:
                return None
    return None


def _steps_to_synthetic_snapshot(
    steps: List[Dict[str, Any]],
    starting_materials: List[str],
    products: List[str],
) -> Dict[str, Any]:
    """Convert baseline steps into a snapshot compatible with score_snapshot_against_known."""
    events: List[Dict[str, Any]] = []
    step_outputs: List[Dict[str, Any]] = []

    for seq_i, step in enumerate(steps):
        step_index = int(step.get("step_index") or (seq_i + 1))
        current_state = list(step.get("current_state") or [])
        resulting_state = list(step.get("resulting_state") or [])
        predicted = str(step.get("predicted_intermediate") or "").strip() or None
        contains_product = bool(step.get("contains_target_product", False))

        # If the model didn't set contains_target_product, infer from products overlap.
        if not contains_product:
            for p in products:
                if p in resulting_state:
                    contains_product = True
                    break

        events.append(
            {
                "seq": seq_i,
                "event_type": "mechanism_step_accepted",
                "payload": {
                    "step_index": step_index,
                    "candidate_rank": 1,
                    "current_state": current_state,
                    "resulting_state": resulting_state,
                    "predicted_intermediate": predicted,
                    "contains_target_product": contains_product,
                    "validation_summary": None,
                },
            }
        )

        step_outputs.append(
            {
                "step_name": "baseline_mechanism_step",
                "attempt": step_index,
                "retry_index": 0,
                "source": "llm",
                "output": {
                    "step_index": step_index,
                    "step_label": str(step.get("step_label") or ""),
                    "current_state": current_state,
                    "resulting_state": resulting_state,
                    "predicted_intermediate": predicted,
                    "reaction_smirks": str(step.get("reaction_smirks") or ""),
                    "electron_pushes": list(step.get("electron_pushes") or []),
                    "contains_target_product": contains_product,
                },
            }
        )

    return {
        "events": events,
        "step_outputs": step_outputs,
        "starting_materials": list(starting_materials),
        "products": list(products),
    }


class BaselineRunner:
    """Single-shot harness-free mechanism evaluator.

    Makes exactly one LLM tool call per case asking for the complete mechanism,
    then scores the result using the same ``score_snapshot_against_known`` scorer
    as the harness pipeline.
    """

    def run_case(
        self,
        starting_materials: List[str],
        products: List[str],
        model: str,
        *,
        thinking_level: Optional[str] = None,
        api_keys: Optional[Dict[str, str]] = None,
        temperature_celsius: float = 25.0,
        ph: Optional[float] = None,
        timeout: float = 180.0,
    ) -> Dict[str, Any]:
        """Run a single baseline case.

        Returns a dict with:
          - ``snapshot``: synthetic snapshot compatible with score_snapshot_against_known
          - ``raw_steps``: parsed steps list from the LLM response
          - ``mechanism_type``: model's stated mechanism type
          - ``llm_text``: free-form reasoning text from the model
          - ``token_usage``: raw usage dict or None
          - ``latency_ms``: wall-clock time in milliseconds
          - ``error``: error message string if the call failed, else None
        """
        start_ts = time.time()

        model_kwargs: Dict[str, Any] = {}
        if thinking_level:
            from mechanistic_agent.model_registry import to_internal_reasoning_level
            internal = to_internal_reasoning_level(thinking_level)
            if internal:
                model_kwargs["reasoning_effort"] = internal

        user_key: Optional[str] = None
        if api_keys:
            for k in ("openai_api_key", "OPENAI_API_KEY", "google_api_key", "GOOGLE_API_KEY",
                      "openrouter_api_key", "OPENROUTER_API_KEY"):
                if api_keys.get(k):
                    user_key = api_keys[k]
                    break

        system_prompt = _load_baseline_system_prompt()
        user_message = _build_user_message(
            starting_materials, products, temperature_celsius, ph
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            adapter = get_chat_model(
                model,
                timeout=timeout,
                model_kwargs=model_kwargs if model_kwargs else None,
                user_api_key=user_key,
            )

            supports_forced = adapter_supports_forced_tools(model)
            tool_choice = build_tool_choice("predict_full_mechanism") if supports_forced else None

            response = adapter.invoke(
                messages,
                tools=[PREDICT_FULL_MECHANISM_TOOL],
                tool_choice=tool_choice,
            )

            latency_ms = (time.time() - start_ts) * 1000.0
            token_usage = getattr(response, "usage", None)
            llm_text = extract_text_content(response) or ""
            tool_calls = getattr(response, "tool_calls", [])
            parsed = _parse_tool_call_arguments(tool_calls)

            if parsed is None:
                return {
                    "snapshot": _steps_to_synthetic_snapshot([], starting_materials, products),
                    "raw_steps": [],
                    "mechanism_type": None,
                    "llm_text": llm_text,
                    "token_usage": token_usage,
                    "latency_ms": latency_ms,
                    "error": "No predict_full_mechanism tool call returned by model.",
                }

            steps = list(parsed.get("steps") or [])
            mechanism_type = str(parsed.get("mechanism_type") or "").strip() or None
            if not llm_text:
                llm_text = str(parsed.get("text") or "")

            snapshot = _steps_to_synthetic_snapshot(steps, starting_materials, products)
            return {
                "snapshot": snapshot,
                "raw_steps": steps,
                "mechanism_type": mechanism_type,
                "llm_text": llm_text,
                "token_usage": token_usage,
                "latency_ms": latency_ms,
                "error": None,
            }

        except Exception as exc:
            latency_ms = (time.time() - start_ts) * 1000.0
            return {
                "snapshot": _steps_to_synthetic_snapshot([], starting_materials, products),
                "raw_steps": [],
                "mechanism_type": None,
                "llm_text": "",
                "token_usage": None,
                "latency_ms": latency_ms,
                "error": str(exc),
            }


def score_baseline_result(
    result: Dict[str, Any],
    expected: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Score a baseline run result using the standard harness scorer."""
    from mechanistic_agent.scoring import score_snapshot_against_known

    snapshot = result.get("snapshot") or {}
    graded = score_snapshot_against_known(snapshot, expected)
    return {
        "score": graded["score"],
        "passed": graded["passed"],
        "scoring_breakdown": graded,
        "step_count": len(result.get("raw_steps") or []),
        "mechanism_type": result.get("mechanism_type"),
        "error": result.get("error"),
    }
