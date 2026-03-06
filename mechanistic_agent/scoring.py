"""Deterministic mechanism scoring for eval runs."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional


# Canonical ordered list of subagent IDs used in the harness.
SUBAGENT_IDS: List[str] = [
    "balance_analysis",
    "functional_groups",
    "ph_recommendation",
    "initial_conditions",
    "missing_reagents",
    "atom_mapping",
    "reaction_type_mapping",
    "mechanism_step_proposal",
    "bond_electron_validation",
    "atom_balance_validation",
    "state_progress_validation",
    "reflection",
    "step_atom_mapping",
]

_VALIDATION_SUBAGENTS = frozenset(
    {"bond_electron_validation", "atom_balance_validation", "state_progress_validation"}
)
_LLM_SUBAGENTS = frozenset(
    {
        "initial_conditions",
        "missing_reagents",
        "atom_mapping",
        "reaction_type_mapping",
        "mechanism_step_proposal",
    }
)
_DETERMINISTIC_SUBAGENTS = frozenset(
    {"balance_analysis", "functional_groups", "ph_recommendation", "reflection"}
)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalise_mapping_confidence(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
    if isinstance(value, (int, float)):
        score = float(value)
    elif isinstance(value, str):
        text = value.strip().lower()
        if not text:
            score = default
        elif text in {"high", "very_high", "strong"}:
            score = 0.9
        elif text in {"medium", "moderate"}:
            score = 0.6
        elif text in {"low", "weak"}:
            score = 0.3
        else:
            score = _as_float(text, default=default)
    else:
        score = default
    if score is None:
        return None
    return max(0.0, min(score, 1.0))


def _normalized_species(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    for item in values:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _validation_check_score(validation: Mapping[str, Any] | None) -> float:
    if not isinstance(validation, Mapping):
        return 0.0
    checks = validation.get("checks")
    if not isinstance(checks, list) or not checks:
        return 0.0
    wanted = {"dbe_metadata", "atom_balance", "state_progress"}
    values: List[float] = []
    for item in checks:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or "")
        if name not in wanted:
            continue
        values.append(1.0 if bool(item.get("passed")) else 0.0)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _build_mapping_confidence_by_attempt(step_outputs: List[Dict[str, Any]]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for row in step_outputs:
        if row.get("step_name") != "step_atom_mapping":
            continue
        attempt = int(row.get("attempt") or 0)
        output = row.get("output") or {}
        confidence = _normalise_mapping_confidence((output or {}).get("confidence"), default=0.0)
        if attempt <= 0:
            continue
        out[attempt] = 0.0 if confidence is None else confidence
    return out


def extract_accepted_path(snapshot: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Extract accepted mechanism pathway from events; fallback to synthesis rows."""
    events = list(snapshot.get("events") or [])
    events.sort(key=lambda item: int(item.get("seq") or 0))

    accepted: List[Dict[str, Any]] = []
    for event in events:
        if str(event.get("event_type") or "") != "mechanism_step_accepted":
            continue
        payload = event.get("payload") or {}
        accepted.append(
            {
                "step_index": int(payload.get("step_index") or 0),
                "candidate_rank": int(payload.get("candidate_rank") or 0),
                "current_state": _normalized_species(payload.get("current_state")),
                "resulting_state": _normalized_species(payload.get("resulting_state")),
                "predicted_intermediate": str(payload.get("predicted_intermediate") or "").strip() or None,
                "contains_target_product": bool(payload.get("contains_target_product")),
                "validation": payload.get("validation_summary"),
            }
        )
    if accepted:
        accepted.sort(key=lambda item: int(item.get("step_index") or 0))
        return accepted

    # Backward-compatible fallback for older runs without mechanism_step_accepted.
    rows = []
    for row in list(snapshot.get("step_outputs") or []):
        if row.get("step_name") != "mechanism_synthesis":
            continue
        validation = row.get("validation") or {}
        if isinstance(validation, Mapping) and validation.get("passed") is False:
            continue
        rows.append(row)

    by_attempt: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        attempt = int(row.get("attempt") or 0)
        existing = by_attempt.get(attempt)
        if existing is None or int(row.get("retry_index") or 0) > int(existing.get("retry_index") or 0):
            by_attempt[attempt] = row

    fallback: List[Dict[str, Any]] = []
    for attempt in sorted(by_attempt):
        row = by_attempt[attempt]
        output = row.get("output") or {}
        fallback.append(
            {
                "step_index": attempt,
                "candidate_rank": 1,
                "current_state": _normalized_species(output.get("current_state")),
                "resulting_state": _normalized_species(output.get("resulting_state")),
                "predicted_intermediate": str(output.get("predicted_intermediate") or "").strip() or None,
                "contains_target_product": bool(output.get("contains_target_product")),
                "validation": row.get("validation"),
            }
        )
    return fallback


def _known_targets(expected: Mapping[str, Any] | None) -> List[Dict[str, Any]]:
    if not isinstance(expected, Mapping):
        return []
    known = expected.get("known_mechanism")
    if isinstance(known, Mapping):
        steps = known.get("steps")
        if isinstance(steps, list):
            out: List[Dict[str, Any]] = []
            for step in steps:
                if not isinstance(step, Mapping):
                    continue
                target = str(step.get("target_smiles") or "").strip()
                if not target:
                    continue
                out.append({"step_index": int(step.get("step_index") or 0), "target_smiles": target})
            if out:
                out.sort(key=lambda x: int(x["step_index"]))
                return out

    verified = expected.get("verified_mechanism")
    if isinstance(verified, Mapping):
        steps = verified.get("steps")
        if isinstance(steps, list):
            out = []
            for step in steps:
                if not isinstance(step, Mapping):
                    continue
                resulting = _normalized_species(step.get("resulting_state"))
                if not resulting:
                    continue
                # Use first resulting species as known target proxy.
                out.append({"step_index": int(step.get("step_index") or 0), "target_smiles": resulting[0]})
            out.sort(key=lambda x: int(x["step_index"]))
            return out
    return []


def score_snapshot_against_known(
    snapshot: Mapping[str, Any],
    expected: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """Return deterministic score + breakdown for leaderboard/eval use."""
    accepted = extract_accepted_path(snapshot)
    step_outputs = list(snapshot.get("step_outputs") or [])
    mapping_conf = _build_mapping_confidence_by_attempt(step_outputs)
    known_steps = _known_targets(expected)

    final_known_product = known_steps[-1]["target_smiles"] if known_steps else None
    final_resulting = accepted[-1]["resulting_state"] if accepted else []
    final_reached = bool(final_known_product and final_known_product in final_resulting)
    final_component = 1.0 if final_reached else 0.0

    step_breakdown: List[Dict[str, Any]] = []
    validity_scores: List[float] = []
    alignment_scores: List[float] = []

    prior_resulting: List[str] | None = None
    seen_signatures: set[str] = set()
    penalty_items: List[Dict[str, Any]] = []
    penalty_total = 0.0

    expected_by_idx = {int(item["step_index"]): str(item["target_smiles"]) for item in known_steps}
    expected_future: Dict[int, List[str]] = {}
    for idx in expected_by_idx:
        expected_future[idx] = [expected_by_idx[j] for j in sorted(expected_by_idx) if j > idx]

    min_steps = int(((expected or {}).get("known_mechanism") or {}).get("min_steps") or len(known_steps) or 0)

    for step in accepted:
        step_index = int(step.get("step_index") or 0)
        resulting_state = _normalized_species(step.get("resulting_state"))
        validation_score = _validation_check_score(step.get("validation"))
        map_conf = mapping_conf.get(step_index, 0.5 if step_index > 0 else 0.0)
        validity = (0.8 * validation_score) + (0.2 * map_conf)
        validity_scores.append(validity)

        expected_target = expected_by_idx.get(step_index)
        future_targets = expected_future.get(step_index, [])
        if expected_target and expected_target in resulting_state:
            alignment = 1.0
            alignment_label = "exact_step_match"
        elif any(target in resulting_state for target in future_targets):
            alignment = 0.75
            alignment_label = "future_step_match"
        elif resulting_state and prior_resulting is not None and set(resulting_state) != set(prior_resulting):
            alignment = 0.55
            alignment_label = "reasonable_non_identical"
        elif resulting_state:
            alignment = 0.35
            alignment_label = "weak_alignment"
        else:
            alignment = 0.0
            alignment_label = "no_alignment"
        alignment_scores.append(alignment)

        signature = "|".join(sorted(set(resulting_state)))
        if prior_resulting is not None and set(resulting_state) == set(prior_resulting):
            penalty_total += 0.08
            penalty_items.append({"type": "circular_step", "step_index": step_index, "value": 0.08})
        if signature and signature in seen_signatures:
            penalty_total += 0.05
            penalty_items.append({"type": "repeated_state", "step_index": step_index, "value": 0.05})
        if signature:
            seen_signatures.add(signature)

        step_breakdown.append(
            {
                "step_index": step_index,
                "candidate_rank": int(step.get("candidate_rank") or 0),
                "validity_score": round(validity, 4),
                "alignment_score": round(alignment, 4),
                "alignment_label": alignment_label,
                "mapping_confidence": round(map_conf, 4),
                "validation_score": round(validation_score, 4),
                "resulting_state": resulting_state,
            }
        )
        prior_resulting = resulting_state

    if min_steps and len(accepted) > min_steps:
        extra = len(accepted) - min_steps
        value = min(0.03 * extra, 0.2)
        penalty_total += value
        penalty_items.append({"type": "extra_steps", "count": extra, "value": round(value, 4)})

    validity_component = (sum(validity_scores) / len(validity_scores)) if validity_scores else 0.0
    alignment_component = (sum(alignment_scores) / len(alignment_scores)) if alignment_scores else 0.0
    penalty_total = min(penalty_total, 0.4)

    overall = (0.45 * validity_component) + (0.35 * alignment_component) + (0.20 * final_component) - penalty_total
    overall = max(0.0, min(overall, 1.0))
    if not final_reached:
        overall = min(overall, 0.55)

    passed = final_reached and overall >= 0.70
    return {
        "score": round(overall, 6),
        "passed": passed,
        "final_product_component": final_component,
        "final_product_reached": final_reached,
        "final_known_product": final_known_product,
        "step_validity_component": round(validity_component, 6),
        "known_alignment_component": round(alignment_component, 6),
        "efficiency_penalty_total": round(penalty_total, 6),
        "penalties": penalty_items,
        "accepted_path_step_count": len(accepted),
        "known_step_count": len(known_steps),
        "step_breakdown": step_breakdown,
    }


def _parse_output(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def score_subagents_from_step_outputs(
    step_outputs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return per-subagent quality_score and pass_rate from a run's step_outputs.

    Each subagent entry contains:
      - quality_score (0.0–1.0): composite quality signal
      - pass_rate (0.0–1.0): fraction of calls that passed their check
      - calls: total number of calls observed for this subagent
    """
    known_ids = set(SUBAGENT_IDS)
    by_name: Dict[str, List[Dict[str, Any]]] = {}
    for row in step_outputs:
        name = str(row.get("step_name") or "")
        if name in known_ids:
            by_name.setdefault(name, []).append(row)

    result: Dict[str, Any] = {}
    validated_mechanism_attempts: set[int] = set()
    for row in step_outputs:
        if str(row.get("step_name") or "") != "mechanism_synthesis":
            continue
        validation = row.get("validation")
        if not isinstance(validation, Mapping):
            continue
        if bool(validation.get("passed")):
            validated_mechanism_attempts.add(int(row.get("attempt") or 0))

    for name, rows in by_name.items():
        if name in _VALIDATION_SUBAGENTS:
            # Score by the latest outcome per (attempt, retry, check).
            latest_by_check: Dict[tuple[int, int, str], bool] = {}
            for row in rows:
                output = _parse_output(row.get("output"))
                passed = output.get("passed")
                if passed is None:
                    continue
                key = (
                    int(row.get("attempt") or 0),
                    int(row.get("retry_index") or 0),
                    str(output.get("check") or ""),
                )
                latest_by_check[key] = bool(passed)
            passes = [1.0 if item else 0.0 for item in latest_by_check.values()]
            rate = sum(passes) / len(passes) if passes else 0.0
            result[name] = {
                "quality_score": round(rate, 4),
                "pass_rate": round(rate, 4),
                "calls": len(latest_by_check),
            }

        elif name in _LLM_SUBAGENTS:
            # Valid call = structured output present; retries reduce quality.
            # For mechanism proposals, quality tracks downstream mechanism validation.
            valid = 0
            retry_calls = 0
            proposal_attempts: List[int] = []
            for row in rows:
                output = _parse_output(row.get("output"))
                if int(row.get("retry_index") or 0) > 0:
                    retry_calls += 1
                if output and any(k for k in output if not k.startswith("_")):
                    valid += 1
                if name == "mechanism_step_proposal":
                    proposal_attempts.append(int(row.get("attempt") or 0))
            n = len(rows)
            pass_rate = valid / n if n else 0.0
            retry_rate = retry_calls / n if n else 0.0
            quality = max(0.0, pass_rate - min(retry_rate * 0.4, 0.3))
            if name == "mechanism_step_proposal":
                validated_calls = sum(
                    1
                    for attempt in proposal_attempts
                    if attempt in validated_mechanism_attempts
                )
                downstream_pass_rate = (validated_calls / n) if n else 0.0
                # Blend structured-output quality with chemistry-valid downstream outcome.
                quality = max(0.0, (0.4 * quality) + (0.6 * downstream_pass_rate))
            result[name] = {
                "quality_score": round(quality, 4),
                "pass_rate": round(pass_rate, 4),
                "calls": n,
                "retry_calls": retry_calls,
            }

        elif name == "step_atom_mapping":
            confidences: List[float] = []
            for row in rows:
                output = _parse_output(row.get("output"))
                conf = _normalise_mapping_confidence(output.get("confidence"), default=None)
                if conf is not None:
                    confidences.append(conf)
            mean_conf = sum(confidences) / len(confidences) if confidences else 0.5
            pass_rate = (
                sum(1.0 for c in confidences if c >= 0.5) / len(confidences)
                if confidences
                else 0.0
            )
            result[name] = {
                "quality_score": round(mean_conf, 4),
                "pass_rate": round(pass_rate, 4),
                "calls": len(rows),
            }

        else:
            # Deterministic subagents: pass = output present and no error key.
            passed_count = sum(
                1
                for row in rows
                if row.get("output") is not None
                and not _parse_output(row.get("output")).get("error")
            )
            rate = passed_count / len(rows) if rows else 0.0
            result[name] = {
                "quality_score": round(rate, 4),
                "pass_rate": round(rate, 4),
                "calls": len(rows),
            }

    return result
