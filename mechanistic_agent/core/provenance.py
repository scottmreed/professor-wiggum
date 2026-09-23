"""Model/engine provenance for run steps (Observatory PRD §12–§13, M0).

A logical step can involve several engines: a Jev decision that fails and an
LLM that answers instead, a deterministic validator, a human submission. This
module normalizes a ``StepResult`` into

* a :class:`StepProvenance` — which engine owns the step's answer and which
  model (if any) produced it; and
* a list of :class:`InferenceCall` records — one per real model/decision
  request, derived from the result and from ``output.decision_trace``.

Rules that matter for the UI:

* Deterministic and human steps never carry a model, even though the run
  has a configured fallback model (PRD §12.3, §3.7.1).
* ``output.model_used`` wins over the configured model, because ``tools.py``
  falls back to another provider model silently (PRD §3.7.2).
* Jev calls are populated from the existing ``DecisionRecord`` trace rather
  than re-described (PRD §12.2 mapping table).

This is the derived path ("M0a"): events are emitted after the fact from
``RunCoordinator._record_step``. The live hook in ``llm.py`` is a later slice.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .types import StepResult

EVENT_SCHEMA_VERSION = "mechanism_observatory_event.v1"

ENGINE_LLM = "llm"
ENGINE_JEV = "jev"
ENGINE_DETERMINISTIC = "deterministic"
ENGINE_HUMAN = "human"
MODEL_ENGINES = frozenset({ENGINE_LLM, ENGINE_JEV})

# PRD §12.2 roles, keyed by coordinator step name.
ROLE_BY_STEP: Dict[str, str] = {
    "initial_conditions": "conditions_decision",
    "ph_recommendation": "conditions_decision",
    "balance_analysis": "balance_analysis",
    "functional_groups": "functional_groups",
    "missing_reagents": "missing_chemistry",
    "atom_mapping": "global_mapping",
    "reaction_type_mapping": "reaction_type",
    "mechanism_step_proposal": "mechanism_proposal",
    "candidate_rescue": "candidate_rescue",
    "step_atom_mapping": "step_mapping",
    "mechanism_synthesis": "mapped_state_executor",
    "bond_electron_validation": "validator",
    "atom_balance_validation": "validator",
    "state_progress_validation": "validator",
    "reflection": "reflection",
    "completion_check": "completion_check",
}

# Steps whose default engine is a chat model. Everything else is deterministic
# unless the decision policy routes it to Jev. Note ``LLM_STEP_KEYS`` in
# ``config.py`` is a *model-selection* key set and lists deterministic steps;
# it must not be used to decide provenance.
LLM_PLANNED_STEPS = frozenset({
    "initial_conditions",
    "missing_reagents",
    "atom_mapping",
    "reaction_type_mapping",
    "mechanism_step_proposal",
    "candidate_rescue",
    "step_atom_mapping",
})

# ``RunConfig.step_models`` keys differ from step names for two steps.
_STEP_MODEL_KEY: Dict[str, str] = {
    "mechanism_step_proposal": "intermediates",
    "initial_conditions": "initial_conditions",
}


def new_candidate_set_id(step_number: int) -> str:
    return f"cs{int(step_number)}-{uuid.uuid4().hex[:8]}"


def assign_candidate_ids(candidates: List[Dict[str, Any]], *, step_number: int) -> List[Dict[str, Any]]:
    """Give every proposed candidate a stable ``candidate_id`` (PRD §13.4).

    Ranks repeat across reproposal rounds, retries and topology rounds, so the
    id is ``c<step>-r<rank>-<random>``. Existing ids are kept, and the dicts are
    mutated in place because ``BranchCandidate.intermediate_output`` holds the
    same dict and persists it (PRD §3.7.4).
    """
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        existing = candidate.get("candidate_id")
        if isinstance(existing, str) and existing.strip():
            continue
        rank = candidate.get("rank")
        try:
            rank_text = str(int(rank))
        except (TypeError, ValueError):
            rank_text = "x"
        candidate["candidate_id"] = f"c{int(step_number)}-r{rank_text}-{uuid.uuid4().hex[:8]}"
    return candidates


def role_for_step(step_name: str) -> str:
    return ROLE_BY_STEP.get(step_name, step_name)


def new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:16]}"


@dataclass
class InferenceCall:
    """One real model or decision request (PRD §12.2)."""

    call_id: str
    engine: str
    role: str
    step_name: str
    attempt: int = 1
    retry_index: int = 0
    provider: Optional[str] = None
    requested_model: Optional[str] = None
    resolved_model: Optional[str] = None
    reasoning_level: Optional[str] = None
    decision_type: Optional[str] = None
    status: str = "completed"
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    usage: Optional[Dict[str, Any]] = None
    cost: Optional[Dict[str, Any]] = None
    fallback_from_call_id: Optional[str] = None
    model_fallback: bool = False
    candidate_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["event_schema_version"] = EVENT_SCHEMA_VERSION
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceCall":
        payload = {k: v for k, v in dict(data or {}).items() if k in cls.__dataclass_fields__}
        return cls(**payload)


@dataclass
class StepProvenance:
    """Which engine answered a step and which calls it made."""

    engine: str
    step_name: str
    resolved_model: Optional[str] = None
    resolved_reasoning: Optional[str] = None
    requested_model: Optional[str] = None
    tool: Optional[str] = None
    calls: List[InferenceCall] = field(default_factory=list)

    @property
    def primary_call_id(self) -> Optional[str]:
        for call in reversed(self.calls):
            if call.engine == self.engine and call.status == "completed":
                return call.call_id
        return self.calls[-1].call_id if self.calls else None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "event_schema_version": EVENT_SCHEMA_VERSION,
            "engine": self.engine,
            "resolved_model": self.resolved_model,
            "resolved_reasoning": self.resolved_reasoning,
            "requested_model": self.requested_model,
            "tool": self.tool,
            "primary_call_id": self.primary_call_id,
            "call_ids": [c.call_id for c in self.calls],
            "model_fallback": any(c.model_fallback for c in self.calls),
            "fallback_chain": [c.engine for c in self.calls] if len(self.calls) > 1 else [],
        }


def _decision_calls(result: StepResult) -> List[InferenceCall]:
    """Jev calls recorded in ``output.decision_trace`` (``DecisionRecord.to_trace()``)."""
    output = result.output if isinstance(result.output, dict) else {}
    trace = output.get("decision_trace")
    if not isinstance(trace, list):
        return []
    calls: List[InferenceCall] = []
    seen: set[str] = set()
    for index, entry in enumerate(trace):
        if not isinstance(entry, dict) or not entry.get("called"):
            continue
        engine = str(entry.get("decision_engine") or "")
        if engine != ENGINE_JEV:
            continue
        request_id = str(entry.get("request_id") or f"local-{index}")
        if request_id in seen:
            continue
        seen.add(request_id)
        failure = entry.get("failure")
        cost = entry.get("cost_breakdown") if isinstance(entry.get("cost_breakdown"), dict) else None
        if cost is None and isinstance(entry.get("cost"), (int, float)):
            cost = {"total_cost": float(entry["cost"])}
        calls.append(
            InferenceCall(
                call_id=request_id,
                engine=ENGINE_JEV,
                role=role_for_step(result.step_name),
                step_name=result.step_name,
                attempt=result.attempt,
                retry_index=result.retry_index,
                provider=entry.get("provider"),
                requested_model=entry.get("model"),
                resolved_model=entry.get("model_version") or None,
                decision_type=entry.get("decision_type"),
                status="failed" if failure else "completed",
                error=str(failure) if failure else None,
                latency_ms=entry.get("latency_ms"),
                usage=entry.get("usage") if isinstance(entry.get("usage"), dict) else None,
                cost=cost,
            )
        )
    return calls


def resolve_step_provenance(
    result: StepResult,
    *,
    configured_model: Optional[str],
    configured_reasoning: Optional[str],
) -> StepProvenance:
    """Normalize a ``StepResult`` into engine + model + call records."""
    source = str(result.source or ENGINE_LLM)
    output = result.output if isinstance(result.output, dict) else {}
    model_used = output.get("model_used")
    model_used = str(model_used) if isinstance(model_used, str) and model_used.strip() else None

    if source == ENGINE_HUMAN:
        return StepProvenance(engine=ENGINE_HUMAN, step_name=result.step_name, tool=result.tool_name)

    if source == ENGINE_DETERMINISTIC:
        return StepProvenance(engine=ENGINE_DETERMINISTIC, step_name=result.step_name, tool=result.tool_name)

    decision_calls = _decision_calls(result)

    if source == ENGINE_JEV:
        completed = [c for c in decision_calls if c.status == "completed"]
        primary = completed[-1] if completed else (decision_calls[-1] if decision_calls else None)
        requested = (primary.requested_model if primary else None) or model_used or result.model
        resolved = (primary.resolved_model if primary else None) or requested
        return StepProvenance(
            engine=ENGINE_JEV,
            step_name=result.step_name,
            resolved_model=resolved,
            resolved_reasoning=None,
            requested_model=requested,
            tool=result.tool_name,
            calls=decision_calls,
        )

    # source == "llm" (or unknown → treated as a chat-model step)
    requested = configured_model or result.model
    resolved = model_used or result.model or configured_model
    failed_jev = [c for c in decision_calls if c.status == "failed"]
    # tools.py swallows provider errors and returns ``status: failed|fallback``
    # plus an ``error`` string instead of raising; that is a failed call, not a
    # completed one (PRD §13.2 ``inference_call_failed``).
    call_error = output.get("error")
    call_failed = bool(call_error) and str(output.get("status") or "").lower() in {"failed", "fallback", "error"}
    llm_call = InferenceCall(
        call_id=new_call_id(),
        engine=ENGINE_LLM,
        role=role_for_step(result.step_name),
        step_name=result.step_name,
        attempt=result.attempt,
        retry_index=result.retry_index,
        requested_model=requested,
        resolved_model=resolved,
        reasoning_level=result.reasoning_level or configured_reasoning,
        status="failed" if call_failed else "completed",
        error=str(call_error) if call_failed else None,
        usage=dict(result.token_usage) if isinstance(result.token_usage, dict) else None,
        cost=dict(result.cost) if isinstance(result.cost, dict) else None,
        fallback_from_call_id=failed_jev[-1].call_id if failed_jev else None,
        model_fallback=bool(requested and resolved and requested != resolved),
    )
    return StepProvenance(
        engine=ENGINE_LLM,
        step_name=result.step_name,
        resolved_model=resolved,
        resolved_reasoning=result.reasoning_level or configured_reasoning,
        requested_model=requested,
        tool=result.tool_name,
        calls=[*decision_calls, llm_call],
    )


def planned_provenance(
    step_name: str,
    *,
    step_models: Optional[Dict[str, str]],
    default_model: Optional[str],
    step_reasoning: Optional[Dict[str, str]],
    reaction_type_policy: str = "llm",
    jev_model: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """What ``step_started`` can honestly say before the step runs (PRD §13.3)."""
    if step_name == "reaction_type_mapping" and reaction_type_policy == ENGINE_JEV:
        return {"planned_engine": ENGINE_JEV, "planned_model": jev_model, "planned_reasoning": None}
    if step_name not in LLM_PLANNED_STEPS:
        return {"planned_engine": ENGINE_DETERMINISTIC, "planned_model": None, "planned_reasoning": None}
    key = _STEP_MODEL_KEY.get(step_name, step_name)
    models = step_models or {}
    reasoning = step_reasoning or {}
    return {
        "planned_engine": ENGINE_LLM,
        "planned_model": models.get(key) or models.get(step_name) or default_model,
        "planned_reasoning": reasoning.get(key) or reasoning.get(step_name),
    }


# ---------------------------------------------------------------------------
# Replay helpers (PRD §17): reconstruct provenance from persisted events only.
# ---------------------------------------------------------------------------

def _sorted(events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(events, key=lambda e: int(e.get("seq") or 0))


def step_provenance_from_events(events: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Latest ``step_output.provenance`` per step name, in event order."""
    by_step: Dict[str, Dict[str, Any]] = {}
    for event in _sorted(events):
        if str(event.get("event_type") or "") != "step_output":
            continue
        payload = event.get("payload") or {}
        prov = payload.get("provenance")
        step_name = str(payload.get("step_name") or event.get("step_name") or "")
        if isinstance(prov, dict) and step_name:
            by_step[step_name] = dict(prov)
    return by_step


def provenance_inventory_from_events(events: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Run-header inventory (PRD §14.4): models actually used, from completed calls."""
    models: Dict[str, List[str]] = {}
    counts: Dict[str, int] = {}
    failed = 0
    for event in _sorted(events):
        kind = str(event.get("event_type") or "")
        if kind not in {"inference_call_completed", "inference_call_failed"}:
            continue
        payload = event.get("payload") or {}
        engine = str(payload.get("engine") or "")
        if not engine:
            continue
        counts[engine] = counts.get(engine, 0) + 1
        if kind == "inference_call_failed":
            failed += 1
            continue
        model = payload.get("resolved_model") or payload.get("requested_model")
        if model:
            bucket = models.setdefault(engine, [])
            if model not in bucket:
                bucket.append(str(model))
    return {"models_by_engine": models, "call_counts": counts, "failed_calls": failed}


def build_run_provenance(events: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Snapshot projection (PRD §17, §22 ``/observatory``): everything from events."""
    events = list(events)
    return {
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "steps": step_provenance_from_events(events),
        "inventory": provenance_inventory_from_events(events),
    }


__all__ = [
    "EVENT_SCHEMA_VERSION",
    "ENGINE_DETERMINISTIC",
    "ENGINE_HUMAN",
    "ENGINE_JEV",
    "ENGINE_LLM",
    "InferenceCall",
    "LLM_PLANNED_STEPS",
    "MODEL_ENGINES",
    "ROLE_BY_STEP",
    "StepProvenance",
    "assign_candidate_ids",
    "build_run_provenance",
    "new_call_id",
    "new_candidate_set_id",
    "planned_provenance",
    "provenance_inventory_from_events",
    "resolve_step_provenance",
    "role_for_step",
    "step_provenance_from_events",
]
