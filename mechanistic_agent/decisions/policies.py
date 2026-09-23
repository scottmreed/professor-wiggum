"""Decision-policy and threshold resolution (PRD §15, §17).

Thresholds are harness values that may be ``None``. ``None`` means
*observational*: the decision is recorded but no Jev-specific gate is
applied, so the existing (RunConfig) behaviour stands. Thresholds are per
question and per Jev version; nothing here copies a threshold from one
question to another.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

from mechanistic_agent.core.types import (
    DECISION_POLICY_ENUMS,
    DecisionPolicy,
    JevConfig,
)

EXAMPLE_BYPASS_ENV = "MECHANISTIC_EXAMPLE_REACTION_TYPE_BYPASS"


def decision_engine_for(policy: Optional[DecisionPolicy], key: str) -> str:
    """Return the engine configured for ``key`` (e.g. ``reaction_type``)."""
    if key not in DECISION_POLICY_ENUMS:
        raise KeyError(f"unknown decision_policy key {key!r}")
    policy = policy if isinstance(policy, DecisionPolicy) else DecisionPolicy()
    return str(getattr(policy, key))


def resolve_threshold(jev: Optional[JevConfig], key: str) -> Optional[float]:
    """Harness threshold for ``key``, or None when unset (observational)."""
    if not isinstance(jev, JevConfig):
        return None
    value = jev.thresholds.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def passes_threshold(value: Optional[float], threshold: Optional[float]) -> Optional[bool]:
    """Tri-state gate: None when observational (no threshold) or no value."""
    if threshold is None or value is None:
        return None
    return float(value) >= float(threshold)


def choice_margin(probabilities: Mapping[str, float]) -> Optional[float]:
    """Top-1 minus top-2 probability, or None with fewer than two options."""
    values = sorted((float(v) for v in probabilities.values()), reverse=True)
    if len(values) < 2:
        return None
    return max(0.0, values[0] - values[1])


def top_candidates(probabilities: Mapping[str, float], n: int) -> List[Tuple[str, float]]:
    """``n`` most probable labels, ties broken by input order."""
    ordered = sorted(enumerate(probabilities.items()), key=lambda item: (-float(item[1][1]), item[0]))
    return [(label, float(prob)) for _, (label, prob) in ordered[: max(0, int(n))]]


def resolve_reaction_type_gates(
    *,
    decision_engine: Optional[str],
    jev: Optional[JevConfig],
    run_confidence_threshold: Optional[float],
    run_margin_threshold: Optional[float],
) -> Dict[str, Any]:
    """Confidence/margin thresholds for template guidance.

    LLM selections keep the RunConfig thresholds (0.65 / 0.10 by default).
    Jev selections use ``jev.thresholds.reaction_type_active_probability`` /
    ``reaction_type_min_margin`` when those are set, and otherwise fall back
    to the RunConfig values, so the existing gates keep working on Jev
    probabilities until Phase D calibration produces Jev-specific ones.
    """
    confidence = float(run_confidence_threshold if run_confidence_threshold is not None else 0.65)
    margin = float(run_margin_threshold if run_margin_threshold is not None else 0.10)
    source = "run_config"
    if decision_engine == "jev":
        jev_conf = resolve_threshold(jev, "reaction_type_active_probability")
        jev_margin = resolve_threshold(jev, "reaction_type_min_margin")
        if jev_conf is not None:
            confidence = jev_conf
            source = "harness_jev"
        if jev_margin is not None:
            margin = jev_margin
            source = "harness_jev"
    return {"confidence_threshold": confidence, "margin_threshold": margin, "source": source}


def _parse_flag(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def resolve_example_bypass(
    *,
    run_value: Optional[bool],
    policy: Optional[DecisionPolicy],
    env: Optional[Mapping[str, str]] = None,
) -> Tuple[bool, str]:
    """Whether the curated ``example_id`` reaction-type shortcut may run.

    Precedence: run config ``example_reaction_type_bypass`` > env
    ``MECHANISTIC_EXAMPLE_REACTION_TYPE_BYPASS`` > harness
    ``decision_policy.example_reaction_type_bypass`` (default True, today's
    behaviour). Returns ``(enabled, source)``.
    """
    if run_value is not None:
        return bool(run_value), "run_config"
    environ = os.environ if env is None else env
    env_value = _parse_flag(environ.get(EXAMPLE_BYPASS_ENV))
    if env_value is not None:
        return env_value, "env"
    policy = policy if isinstance(policy, DecisionPolicy) else DecisionPolicy()
    return bool(policy.example_reaction_type_bypass), "harness"


__all__ = [
    "EXAMPLE_BYPASS_ENV",
    "choice_margin",
    "decision_engine_for",
    "passes_threshold",
    "resolve_example_bypass",
    "resolve_reaction_type_gates",
    "resolve_threshold",
    "top_candidates",
]
