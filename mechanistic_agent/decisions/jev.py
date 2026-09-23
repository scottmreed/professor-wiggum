"""Generic client for TypeSafe Jev through OpenRouter's Decisions API.

No chemistry lives here. Callers build a machine-only ``state`` and a set of
typed questions; the client encodes one request, calls the endpoint, and
returns one :class:`DecisionRecord` per question. The client never raises for
runtime failures (missing key, timeout, HTTP error, malformed response,
missing probabilities): those come back as records with ``failure`` set, so
the caller decides the fallback. Programming errors (unknown question type, a
Choice without a ``none``/``no_match`` option, a model id that is not a
decision model in the catalog) raise ``ValueError``.

Wire format (OpenRouter ``POST /api/alpha/decisions``)::

    request  {"model": "<api_model_id>", "state": <str|dict|list>,
              "questions": {"<key>": {"type": "choice|score|noul",
                                      "instructions": "...",
                                      "criteria": ...}}}
    criteria choice -> {"<label>": "<description>", ...}   (object, one entry per option)
             score  -> ["<level 0 (lowest)>", ..., "<level n-1>"]  (ordered array)
             noul   -> {"true": "<when yes>", "false": "<when no>"}
    response {"id": ..., "model": ..., "provider": ..., "answers": {"<key>": {...}},
              "usage": {"cost": <usd>, "input_tokens": n, "output_tokens": n}}
    answers  noul   -> {"type": "noul", "noul": p_true}                      (no confidence)
             choice -> {"type": "choice", "choice": label, "probabilities": {...}, "confidence": c}
             score  -> {"type": "score", "score": weighted_mean_index, "legend": {...},
                        "probabilities": {"0": p, ...}, "confidence": c}

The encoder is :func:`encode_question`; keep all wire-format knowledge in it
and in :func:`parse_answer`.
"""
from __future__ import annotations

import json
import math
import socket
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
PROVIDER = "openrouter"
DECISION_ENGINE = "jev"
QUESTION_TYPES = ("choice", "score", "noul")
NONE_OPTION_KEYS = ("none", "no_match")

# Published limits (PRD §7.0). A catalog entry may override them via ``limits``.
DEFAULT_LIMITS: Dict[str, int] = {
    "choice_max_options": 255,
    "score_min_levels": 2,
    "score_max_levels": 10,
    "max_state_plus_question_tokens": 32000,
}

# (url, headers, json_payload, timeout_seconds) -> (http_status, body_text)
Transport = Callable[[str, Dict[str, str], Dict[str, Any], float], Tuple[int, str]]


class JevResponseError(ValueError):
    """A response (or one answer in it) could not be parsed."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


# ---------------------------------------------------------------------------
# Questions and request encoding
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DecisionQuestion:
    """One typed question. Build with :meth:`choice`, :meth:`score` or :meth:`noul`."""

    key: str
    type: str
    instructions: str
    # choice: ordered label -> description. Must contain "none" or "no_match".
    options: Tuple[Tuple[str, str], ...] = ()
    # score: ordered rubric levels, index 0 = lowest.
    levels: Tuple[str, ...] = ()
    # noul: optional descriptions of when the proposition is true / false.
    true_criterion: Optional[str] = None
    false_criterion: Optional[str] = None

    @classmethod
    def choice(cls, key: str, instructions: str, options: Mapping[str, str]) -> "DecisionQuestion":
        return cls(key=key, type="choice", instructions=instructions,
                   options=tuple((str(k), str(v)) for k, v in options.items()))

    @classmethod
    def score(cls, key: str, instructions: str, levels: Sequence[str]) -> "DecisionQuestion":
        return cls(key=key, type="score", instructions=instructions,
                   levels=tuple(str(level) for level in levels))

    @classmethod
    def noul(
        cls,
        key: str,
        instructions: str,
        *,
        true: Optional[str] = None,
        false: Optional[str] = None,
    ) -> "DecisionQuestion":
        return cls(key=key, type="noul", instructions=instructions,
                   true_criterion=true, false_criterion=false)

    @property
    def option_labels(self) -> List[str]:
        return [label for label, _ in self.options]


def validate_question(question: DecisionQuestion, limits: Optional[Mapping[str, int]] = None) -> None:
    """Raise ``ValueError`` when the question violates the API contract."""
    lim = {**DEFAULT_LIMITS, **dict(limits or {})}
    if not question.key or not str(question.key).strip():
        raise ValueError("question key must be non-empty")
    if question.type not in QUESTION_TYPES:
        raise ValueError(f"unknown question type {question.type!r}; expected one of {QUESTION_TYPES}")
    if not str(question.instructions or "").strip():
        raise ValueError(f"question {question.key!r} needs instructions")
    if question.type == "choice":
        labels = question.option_labels
        if len(labels) < 2:
            raise ValueError(f"choice {question.key!r} needs at least 2 options")
        if len(set(labels)) != len(labels):
            raise ValueError(f"choice {question.key!r} has duplicate option labels")
        if len(labels) > int(lim["choice_max_options"]):
            raise ValueError(
                f"choice {question.key!r} has {len(labels)} options; limit is {lim['choice_max_options']}"
            )
        if not any(label in NONE_OPTION_KEYS for label in labels):
            raise ValueError(
                f"choice {question.key!r} must include an explicit 'none' or 'no_match' option"
            )
    elif question.type == "score":
        n = len(question.levels)
        if n < int(lim["score_min_levels"]) or n > int(lim["score_max_levels"]):
            raise ValueError(
                f"score {question.key!r} has {n} levels; allowed "
                f"{lim['score_min_levels']}-{lim['score_max_levels']}"
            )


def encode_question(question: DecisionQuestion) -> Dict[str, Any]:
    """Encode one question in the Decisions API wire format (see module docstring)."""
    encoded: Dict[str, Any] = {"type": question.type, "instructions": question.instructions}
    if question.type == "choice":
        encoded["criteria"] = {label: description for label, description in question.options}
    elif question.type == "score":
        encoded["criteria"] = list(question.levels)
    elif question.type == "noul":
        criteria: Dict[str, str] = {}
        if question.true_criterion:
            criteria["true"] = question.true_criterion
        if question.false_criterion:
            criteria["false"] = question.false_criterion
        if criteria:
            encoded["criteria"] = criteria
    return encoded


def encode_request(api_model_id: str, state: Any, questions: Sequence[DecisionQuestion]) -> Dict[str, Any]:
    return {
        "model": api_model_id,
        "state": state,
        "questions": {q.key: encode_question(q) for q in questions},
    }


def estimate_tokens(value: Any) -> int:
    """Conservative token estimate (~4 characters per token) for limit checks."""
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
    return int(math.ceil(len(text) / 4.0))


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
@dataclass
class DecisionRecord:
    """Outcome of one question. ``failure`` is None on success."""

    question_id: str
    decision_type: str
    model: str
    model_version: Optional[str] = None
    provider: str = PROVIDER
    selected: Any = None
    probabilities: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None  # always None for noul
    score: Optional[float] = None  # score questions: probability-weighted mean level index
    legend: Optional[Dict[str, str]] = None
    latency_ms: Optional[float] = None
    # Request-level usage/cost. decide_many shares one request across several
    # records; group by request_id to avoid double counting.
    usage: Optional[Dict[str, int]] = None
    cost: Optional[Dict[str, float]] = None
    request_id: Optional[str] = None
    called: bool = False  # True when an HTTP request was actually sent
    failure: Optional[str] = None
    failure_detail: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.failure is None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_trace(
        self,
        *,
        fallback_triggered: bool = False,
        fallback_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Trace entry in the PRD §18 shape (plus request bookkeeping fields)."""
        total_cost = None
        if isinstance(self.cost, dict):
            total_cost = self.cost.get("total_cost")
        return {
            "decision_engine": DECISION_ENGINE,
            "model": self.model,
            "model_version": self.model_version,
            "provider": self.provider,
            "decision_type": self.decision_type,
            "question_id": self.question_id,
            "selected": self.selected,
            "probabilities": dict(self.probabilities) if self.probabilities is not None else None,
            "confidence": None if self.decision_type == "noul" else self.confidence,
            "score": self.score,
            "fallback_triggered": bool(fallback_triggered),
            "fallback_reason": fallback_reason,
            "latency_ms": self.latency_ms,
            "cost": total_cost,
            "cost_breakdown": dict(self.cost) if isinstance(self.cost, dict) else None,
            "usage": dict(self.usage) if isinstance(self.usage, dict) else None,
            "request_id": self.request_id,
            "called": self.called,
            "failure": self.failure,
            "failure_detail": self.failure_detail,
        }


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------
def _as_probability(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return None
    return min(1.0, max(0.0, number))


def _normalise_distribution(raw: Any, keys: Sequence[str]) -> Tuple[Dict[str, float], List[str]]:
    """Map a raw probability object onto ``keys``; missing keys get 0.

    Raises ``JevResponseError("missing_probabilities")`` when the object is
    absent, empty, or sums to zero. Renormalises a distribution that does not
    sum to 1 and reports that (and dropped unknown keys) in the notes.
    """
    notes: List[str] = []
    if not isinstance(raw, Mapping) or not raw:
        raise JevResponseError("missing_probabilities", "no probabilities object in answer")
    known = set(keys)
    dist: Dict[str, float] = {key: 0.0 for key in keys}
    dropped: List[str] = []
    for raw_key, raw_value in raw.items():
        key = str(raw_key)
        prob = _as_probability(raw_value)
        if key not in known:
            if prob:
                dropped.append(key)
            continue
        if prob is None:
            continue
        dist[key] = prob
    if dropped:
        notes.append(f"dropped_unknown_keys:{','.join(sorted(dropped))}")
    total = sum(dist.values())
    if total <= 0.0:
        raise JevResponseError("missing_probabilities", "probabilities sum to zero over known options")
    if abs(total - 1.0) > 1e-6:
        notes.append(f"renormalised_from:{total:.6f}")
        dist = {key: value / total for key, value in dist.items()}
    return dist, notes


def _argmax(dist: Mapping[str, float], order: Sequence[str]) -> str:
    best = order[0]
    for key in order:
        if dist.get(key, 0.0) > dist.get(best, 0.0):
            best = key
    return best


def parse_answer(question: DecisionQuestion, answer: Any) -> Dict[str, Any]:
    """Parse one answer object; returns record fields or raises JevResponseError."""
    if not isinstance(answer, Mapping):
        raise JevResponseError("missing_answer", f"no answer for question {question.key!r}")
    answer_type = str(answer.get("type") or question.type)
    if answer_type != question.type:
        raise JevResponseError(
            "malformed_response", f"answer type {answer_type!r} != question type {question.type!r}"
        )

    if question.type == "noul":
        p_true = _as_probability(answer.get("noul"))
        if p_true is None:
            raise JevResponseError("missing_probabilities", "noul answer has no numeric 'noul'")
        return {
            "selected": p_true >= 0.5,
            "probabilities": {"true": p_true, "false": 1.0 - p_true},
            "confidence": None,
            "notes": [],
        }

    confidence = _as_probability(answer.get("confidence"))
    notes: List[str] = []
    if confidence is None:
        notes.append("missing_confidence")

    if question.type == "choice":
        labels = question.option_labels
        dist, dist_notes = _normalise_distribution(answer.get("probabilities"), labels)
        notes.extend(dist_notes)
        raw_choice = answer.get("choice")
        argmax_label = _argmax(dist, labels)
        if raw_choice is None or str(raw_choice) == "":
            selected = argmax_label
            notes.append("choice_missing_used_argmax")
        elif str(raw_choice) not in dist:
            raise JevResponseError("unknown_choice", f"choice {raw_choice!r} is not an option")
        else:
            selected = str(raw_choice)
            if selected != argmax_label and dist[argmax_label] > dist[selected]:
                notes.append(f"choice_differs_from_argmax:{argmax_label}")
        return {"selected": selected, "probabilities": dist, "confidence": confidence, "notes": notes}

    # score
    level_keys = [str(index) for index in range(len(question.levels))]
    dist, dist_notes = _normalise_distribution(answer.get("probabilities"), level_keys)
    notes.extend(dist_notes)
    weighted = sum(int(key) * value for key, value in dist.items())
    raw_score = answer.get("score")
    score = float(raw_score) if isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool) else weighted
    if not isinstance(raw_score, (int, float)) or isinstance(raw_score, bool):
        notes.append("score_missing_used_weighted_mean")
    legend_raw = answer.get("legend")
    legend = (
        {str(k): str(v) for k, v in legend_raw.items()}
        if isinstance(legend_raw, Mapping) and legend_raw
        else {key: level for key, level in zip(level_keys, question.levels)}
    )
    return {
        "selected": int(_argmax(dist, level_keys)),
        "probabilities": dist,
        "confidence": confidence,
        "score": score,
        "legend": legend,
        "notes": notes,
    }


def _usage_from_body(body: Mapping[str, Any]) -> Optional[Dict[str, int]]:
    raw = body.get("usage")
    if not isinstance(raw, Mapping):
        return None
    input_tokens = int(raw.get("input_tokens") or raw.get("prompt_tokens") or 0)
    output_tokens = int(raw.get("output_tokens") or raw.get("completion_tokens") or 0)
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": 0,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------
def urllib_transport(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float) -> Tuple[int, str]:
    """Default transport (stdlib only). Raises TimeoutError on timeout."""
    import urllib.error
    import urllib.request

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https URL
            return int(response.status), response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read().decode("utf-8", errors="replace")
    except socket.timeout as exc:
        raise TimeoutError(str(exc)) from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, socket.timeout):
            raise TimeoutError(str(exc.reason)) from exc
        raise


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
def _resolve_decision_spec(model: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    from mechanistic_agent.model_registry import (
        get_default_decision_model,
        get_model_kind,
        get_model_spec,
        resolve_model_key,
    )

    name = model or get_default_decision_model()
    if not name:
        raise ValueError("no decision model in the catalog (model_kind='decision')")
    key = resolve_model_key(name)  # raises ValueError for unknown ids (SOUL Guardrail 3)
    if get_model_kind(key) != "decision":
        raise ValueError(f"model {name!r} is not a decision model (catalog model_kind != 'decision')")
    return key, dict(get_model_spec(key))


class JevDecisionClient:
    """Client for Jev decision questions. See the module docstring for the wire format."""

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        transport: Optional[Transport] = None,
        clock: Callable[[], float] = time.perf_counter,
        title: str = "mechanistic-agent",
    ) -> None:
        self.model, spec = _resolve_decision_spec(model)
        self.spec = spec
        self.api_model_id = str(spec.get("api_model_id") or self.model)
        self.url = str(spec.get("decision_endpoint") or DECISIONS_URL)
        self.provider = str(spec.get("provider") or PROVIDER)
        self.limits = {**DEFAULT_LIMITS, **dict(spec.get("limits") or {})}
        self.api_key = api_key
        self.timeout = float(timeout)
        self._transport = transport or urllib_transport
        self._clock = clock
        self._title = title

    # -- convenience wrappers --------------------------------------------
    def choice(self, state: Any, key: str, instructions: str, options: Mapping[str, str]) -> DecisionRecord:
        return self.decide_many(state, [DecisionQuestion.choice(key, instructions, options)])[key]

    def score(self, state: Any, key: str, instructions: str, levels: Sequence[str]) -> DecisionRecord:
        return self.decide_many(state, [DecisionQuestion.score(key, instructions, levels)])[key]

    def noul(
        self,
        state: Any,
        key: str,
        instructions: str,
        *,
        true: Optional[str] = None,
        false: Optional[str] = None,
    ) -> DecisionRecord:
        return self.decide_many(state, [DecisionQuestion.noul(key, instructions, true=true, false=false)])[key]

    # -- core ----------------------------------------------------------------
    def build_payload(self, state: Any, questions: Sequence[DecisionQuestion]) -> Dict[str, Any]:
        """Validate questions and return the request body (no network)."""
        keys = [q.key for q in questions]
        if not questions:
            raise ValueError("at least one question is required")
        if len(set(keys)) != len(keys):
            raise ValueError(f"duplicate question keys: {keys}")
        for question in questions:
            validate_question(question, self.limits)
        return encode_request(self.api_model_id, state, questions)

    def _records(self, questions: Sequence[DecisionQuestion], **common: Any) -> Dict[str, DecisionRecord]:
        return {
            q.key: DecisionRecord(question_id=q.key, decision_type=q.type, model=self.model,
                                  provider=self.provider, **common)
            for q in questions
        }

    def _cost(self, body: Mapping[str, Any], usage: Optional[Dict[str, int]]) -> Optional[Dict[str, float]]:
        raw = body.get("usage") if isinstance(body.get("usage"), Mapping) else {}
        provider_cost = raw.get("cost") if isinstance(raw, Mapping) else None
        if isinstance(provider_cost, (int, float)) and not isinstance(provider_cost, bool):
            value = float(provider_cost)
            return {"input_cost": value, "cached_input_cost": 0.0, "output_cost": 0.0, "total_cost": value}
        if usage is None:
            return None
        from mechanistic_agent.model_registry import calculate_cost

        return calculate_cost(self.model, usage)

    def decide_many(self, state: Any, questions: Sequence[DecisionQuestion]) -> Dict[str, DecisionRecord]:
        """Ask all questions in one request; one record per question key."""
        payload = self.build_payload(state, questions)

        if not self.api_key:
            return self._records(questions, failure="missing_api_key",
                                 failure_detail="OpenRouter API key not configured. Set OPENROUTER_API_KEY.")

        state_tokens = estimate_tokens(state)
        longest = max(estimate_tokens(q) for q in payload["questions"].values())
        max_tokens = int(self.limits["max_state_plus_question_tokens"])
        if state_tokens + longest > max_tokens:
            return self._records(
                questions,
                failure="request_too_large",
                failure_detail=f"estimated state+longest question {state_tokens + longest} tokens > {max_tokens}",
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": self._title,
        }
        local_request_id = f"local-{uuid.uuid4().hex}"
        started = self._clock()
        try:
            status, text = self._transport(self.url, headers, payload, self.timeout)
        except TimeoutError as exc:
            latency = round((self._clock() - started) * 1000.0, 1)
            return self._records(questions, failure="timeout", failure_detail=str(exc)[:300],
                                 latency_ms=latency, called=True, request_id=local_request_id)
        except Exception as exc:  # network errors
            latency = round((self._clock() - started) * 1000.0, 1)
            return self._records(questions, failure="transport_error",
                                 failure_detail=f"{type(exc).__name__}: {str(exc)[:300]}",
                                 latency_ms=latency, called=True, request_id=local_request_id)
        latency = round((self._clock() - started) * 1000.0, 1)

        if int(status) != 200:
            return self._records(questions, failure=f"http_{int(status)}",
                                 failure_detail=str(text or "")[:300], latency_ms=latency,
                                 called=True, request_id=local_request_id)
        try:
            body = json.loads(text) if isinstance(text, str) else text
        except (TypeError, ValueError) as exc:
            return self._records(questions, failure="malformed_response",
                                 failure_detail=f"non-JSON body: {exc}", latency_ms=latency,
                                 called=True, request_id=local_request_id)
        if not isinstance(body, Mapping):
            return self._records(questions, failure="malformed_response",
                                 failure_detail="response body is not a JSON object",
                                 latency_ms=latency, called=True, request_id=local_request_id)

        usage = _usage_from_body(body)
        cost = self._cost(body, usage)
        common = {
            "model_version": str(body.get("model")) if body.get("model") else None,
            "latency_ms": latency,
            "usage": usage,
            "cost": cost,
            "request_id": str(body.get("id") or local_request_id),
            "called": True,
        }
        provider = str(body.get("provider") or "").strip()
        answers = body.get("answers")
        records: Dict[str, DecisionRecord] = {}
        for question in questions:
            record = DecisionRecord(question_id=question.key, decision_type=question.type,
                                    model=self.model, provider=self.provider, **common)
            if provider:
                record.notes.append(f"upstream_provider:{provider}")
            if not isinstance(answers, Mapping):
                record.failure = "malformed_response"
                record.failure_detail = "response has no 'answers' object"
                records[question.key] = record
                continue
            try:
                parsed = parse_answer(question, answers.get(question.key))
            except JevResponseError as exc:
                record.failure = exc.code
                record.failure_detail = exc.detail[:300]
                records[question.key] = record
                continue
            record.selected = parsed["selected"]
            record.probabilities = parsed["probabilities"]
            record.confidence = parsed.get("confidence")
            record.score = parsed.get("score")
            record.legend = parsed.get("legend")
            record.notes.extend(parsed.get("notes") or [])
            records[question.key] = record
        return records


__all__ = [
    "DECISIONS_URL",
    "DEFAULT_LIMITS",
    "DecisionQuestion",
    "DecisionRecord",
    "JevDecisionClient",
    "JevResponseError",
    "encode_question",
    "encode_request",
    "estimate_tokens",
    "parse_answer",
    "urllib_transport",
    "validate_question",
]
