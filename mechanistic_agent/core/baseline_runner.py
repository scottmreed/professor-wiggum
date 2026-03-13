"""Harness-free single-shot baseline mechanism evaluation.

This module provides ``BaselineRunner``, which asks a selected model for the
complete reaction mechanism in a *single* tool-calling step (no multi-step
harness pipeline).  Results are stored in the same eval framework so they can
be compared directly against harness runs on the leaderboard.

The run_group_name convention for baseline eval runs is ``"harness_free_baseline"``.
"""
from __future__ import annotations

import json
import hashlib
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from mechanistic_agent.llm import (
    adapter_supports_forced_tools,
    extract_text_content,
    get_chat_model,
    is_anthropic_model,
    is_gemini_model,
    is_openrouter_model,
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
    *,
    current_state: Optional[List[str]] = None,
    accepted_path_summary: Optional[List[str]] = None,
) -> str:
    if current_state:
        lines = [
            "Predict the remaining stepwise mechanism for the following reaction.",
            "",
            f"Original starting materials: {', '.join(starting_materials)}",
            f"Current accepted state: {', '.join(current_state)}",
            f"Target products: {', '.join(products)}",
        ]
    else:
        lines = [
            "Predict the complete stepwise mechanism for the following reaction.",
            "",
            f"Starting materials: {', '.join(starting_materials)}",
            f"Target products: {', '.join(products)}",
        ]
    if ph is not None:
        lines.append(f"pH: {ph}")
    lines.append(f"Temperature: {temperature_celsius} °C")
    if accepted_path_summary:
        lines.append(f"Accepted path so far: {' | '.join(accepted_path_summary)}")
    lines.append("")
    if current_state:
        lines.append(
            "Call predict_full_mechanism with only the remaining elementary steps from the current "
            "accepted state to the target products. Do not repeat already accepted earlier steps."
        )
    else:
        lines.append(
            "Call predict_full_mechanism with all elementary steps from starting materials "
            "to target products. Each step must be a single bond-level elementary event."
        )
    return "\n".join(lines)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _resolve_user_api_key_for_model(model: str, api_keys: Optional[Dict[str, str]]) -> Optional[str]:
    """Select the correct provider API key from a mixed key dictionary."""
    if not api_keys:
        return None

    def _first(*names: str) -> Optional[str]:
        for name in names:
            value = api_keys.get(name)
            if value:
                return str(value)
        return None

    if is_gemini_model(model):
        return _first(
            "google_api_key",
            "GOOGLE_API_KEY",
            "gemini_api_key",
            "GEMINI_API_KEY",
            "vertex_api_key",
            "VERTEX_API_KEY",
        )
    if is_anthropic_model(model):
        return _first("anthropic_api_key", "ANTHROPIC_API_KEY")
    if is_openrouter_model(model):
        # Compatibility: when users only configure ANTHROPIC_API_KEY for OpenRouter-routed
        # Anthropic models, keep that as fallback.
        return _first(
            "openrouter_api_key",
            "OPENROUTER_API_KEY",
            "anthropic_api_key",
            "ANTHROPIC_API_KEY",
        )
    return _first("openai_api_key", "OPENAI_API_KEY")


def _normalise_baseline_reasoning_payload(model: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Clamp reasoning payload to provider-supported options for baseline calls."""
    normalized = dict(payload or {})
    if not normalized:
        return normalized

    effort_value = normalized.get("effort")
    if isinstance(effort_value, str):
        effort_map = {
            "max": "xhigh",
            "lowest": "none",
            "disabled": "none",
            "off": "none",
        }
        normalized["effort"] = effort_map.get(effort_value.strip().lower(), effort_value.strip().lower())

    # OpenRouter baseline calls are most stable with effort-only controls.
    if is_openrouter_model(model) and isinstance(normalized.get("thinking"), dict):
        thinking_type = str(normalized["thinking"].get("type") or "").strip().lower()
        if thinking_type in {"disabled", "none", "off"}:
            normalized["effort"] = "none"
        normalized.pop("thinking", None)
    return normalized


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
        current_state: Optional[List[str]] = None,
        accepted_path_summary: Optional[List[str]] = None,
        llm_seed: Optional[int] = 42,
        llm_temperature: Optional[float] = 0.0,
        sampling_policy: str = "fixed",
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
        from mechanistic_agent.model_registry import (
            build_reasoning_payload,
            get_default_reasoning_level,
            to_internal_reasoning_level,
        )
        effective_level: Optional[str] = None
        if thinking_level:
            effective_level = to_internal_reasoning_level(thinking_level)
        if not effective_level:
            effective_level = get_default_reasoning_level(model)
        if effective_level:
            payload = build_reasoning_payload(model, effective_level)
            if payload:
                model_kwargs.update(_normalise_baseline_reasoning_payload(model, payload))
        policy = str(sampling_policy or "fixed").strip().lower()
        if policy not in {"fixed", "provider_default"}:
            policy = "fixed"
        requested_seed = llm_seed if llm_seed is not None else None
        if requested_seed is not None:
            model_kwargs["seed"] = int(requested_seed)
        requested_temperature = float(llm_temperature) if (policy == "fixed" and llm_temperature is not None) else None

        user_key = _resolve_user_api_key_for_model(model, api_keys)

        system_prompt = _load_baseline_system_prompt()
        user_message = _build_user_message(
            starting_materials,
            products,
            temperature_celsius,
            ph,
            current_state=current_state,
            accepted_path_summary=accepted_path_summary,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        prompt_system_hash = _sha256_text(system_prompt)
        prompt_user_hash = _sha256_text(user_message)
        prompt_hash = _sha256_text(f"{system_prompt}\n\n{user_message}")

        try:
            effective_model_kwargs = dict(model_kwargs)
            seed_applied = requested_seed

            supports_forced = adapter_supports_forced_tools(model)
            tool_choice = build_tool_choice("predict_full_mechanism") if supports_forced else None

            try:
                adapter = get_chat_model(
                    model,
                    temperature=requested_temperature,
                    timeout=timeout,
                    model_kwargs=effective_model_kwargs if effective_model_kwargs else None,
                    user_api_key=user_key,
                )
                response = adapter.invoke(
                    messages,
                    tools=[PREDICT_FULL_MECHANISM_TOOL],
                    tool_choice=tool_choice,
                )
            except Exception:
                # Retry once without seed for providers/adapters that reject it.
                if "seed" not in effective_model_kwargs:
                    raise
                fallback_kwargs = dict(effective_model_kwargs)
                fallback_kwargs.pop("seed", None)
                adapter = get_chat_model(
                    model,
                    temperature=requested_temperature,
                    timeout=timeout,
                    model_kwargs=fallback_kwargs if fallback_kwargs else None,
                    user_api_key=user_key,
                )
                response = adapter.invoke(
                    messages,
                    tools=[PREDICT_FULL_MECHANISM_TOOL],
                    tool_choice=tool_choice,
                )
                seed_applied = None

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
                    "prompt_hash": prompt_hash,
                    "prompt_system_hash": prompt_system_hash,
                    "prompt_user_hash": prompt_user_hash,
                    "sampling_policy": policy,
                    "llm_seed_requested": requested_seed,
                    "llm_seed_applied": seed_applied,
                    "llm_temperature": requested_temperature,
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
                "prompt_hash": prompt_hash,
                "prompt_system_hash": prompt_system_hash,
                "prompt_user_hash": prompt_user_hash,
                "sampling_policy": policy,
                "llm_seed_requested": requested_seed,
                "llm_seed_applied": seed_applied,
                "llm_temperature": requested_temperature,
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
                "prompt_hash": prompt_hash,
                "prompt_system_hash": prompt_system_hash,
                "prompt_user_hash": prompt_user_hash,
                "sampling_policy": policy,
                "llm_seed_requested": requested_seed,
                "llm_seed_applied": None,
                "llm_temperature": requested_temperature,
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
