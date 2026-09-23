"""Live inference-call recording (Observatory PRD §13, "M0b — live").

``RunCoordinator._mark_step_started`` opens a :func:`call_context` for the
step that is about to run. While it is open, ``llm.get_chat_model`` returns a
:class:`RecordingChatAdapter` that emits ``inference_call_started`` before the
provider request and ``inference_call_completed`` / ``inference_call_failed``
after it, with measured latency, and appends the :class:`InferenceCall` to the
context. ``_record_step`` then uses those live records instead of deriving a
call after the fact (the "M0a" path in ``provenance.py``), so a request is
never counted twice, and closes the context.

Outside a context (CLI scripts, tests, the API thread) the adapter is a plain
pass-through and ``get_chat_model`` returns the bare provider adapter.

The context is a :class:`contextvars.ContextVar`, so each run thread sees only
its own step.
"""
from __future__ import annotations

import contextlib
import contextvars
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

from .provenance import ENGINE_LLM, InferenceCall, new_call_id, role_for_step

EventSink = Callable[..., None]  # sink(event_type, payload, *, step_name=None)


@dataclass
class CallContext:
    """The step currently executing on this thread, plus its live call records."""

    run_id: str
    step_name: str
    attempt: int = 1
    retry_index: int = 0
    sink: Optional[EventSink] = None
    planned_model: Optional[str] = None
    planned_reasoning: Optional[str] = None
    calls: List[InferenceCall] = field(default_factory=list)

    @property
    def last_failed_call_id(self) -> Optional[str]:
        for call in reversed(self.calls):
            if call.status == "failed":
                return call.call_id
        return None

    def emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        if self.sink is None:
            return
        try:
            self.sink(event_type, payload, step_name=self.step_name)
        except Exception:  # pragma: no cover - recording must never break a run
            pass


_current: contextvars.ContextVar[Optional[CallContext]] = contextvars.ContextVar(
    "mechanistic_call_context", default=None
)


def current_call_context() -> Optional[CallContext]:
    return _current.get()


def open_call_context(
    *,
    run_id: str,
    step_name: str,
    attempt: int = 1,
    retry_index: int = 0,
    sink: Optional[EventSink] = None,
    planned_model: Optional[str] = None,
    planned_reasoning: Optional[str] = None,
) -> CallContext:
    """Replace the thread's current context with a fresh one for *step_name*."""
    ctx = CallContext(
        run_id=run_id,
        step_name=step_name,
        attempt=int(attempt or 1),
        retry_index=int(retry_index or 0),
        sink=sink,
        planned_model=planned_model,
        planned_reasoning=planned_reasoning,
    )
    _current.set(ctx)
    return ctx


def close_call_context() -> Optional[CallContext]:
    ctx = _current.get()
    _current.set(None)
    return ctx


@contextlib.contextmanager
def call_context(**kwargs: Any) -> Iterator[CallContext]:
    previous = _current.get()
    ctx = open_call_context(**kwargs)
    try:
        yield ctx
    finally:
        _current.set(previous)


def _normalised_usage(response: Any) -> Optional[Dict[str, int]]:
    raw = getattr(response, "usage", None)
    if not isinstance(raw, dict) or not raw:
        return None
    try:
        from mechanistic_agent.model_registry import normalise_token_usage

        return normalise_token_usage(raw)
    except Exception:  # pragma: no cover - defensive
        return None


class RecordingChatAdapter:
    """Proxy around a provider chat adapter that records each ``invoke``."""

    def __init__(self, inner: Any, model_name: str) -> None:
        self.inner = inner
        self.model_name = str(model_name)

    def __getattr__(self, name: str) -> Any:  # delegate everything else
        return getattr(self.inner, name)

    def invoke(
        self,
        messages: Any,
        config: Any = None,
        *,
        tools: Any = None,
        tool_choice: Any = None,
    ) -> Any:
        ctx = _current.get()
        if ctx is None:
            return self.inner.invoke(messages, config, tools=tools, tool_choice=tool_choice)

        call = InferenceCall(
            call_id=new_call_id(),
            engine=ENGINE_LLM,
            role=role_for_step(ctx.step_name),
            step_name=ctx.step_name,
            attempt=ctx.attempt,
            retry_index=ctx.retry_index,
            requested_model=self.model_name,
            resolved_model=None,
            reasoning_level=ctx.planned_reasoning,
            status="started",
            # A request that follows a failed one in the same step is a
            # provider/model fallback (tools.py retries on fallback_model).
            fallback_from_call_id=ctx.last_failed_call_id,
            model_fallback=bool(ctx.planned_model and self.model_name != ctx.planned_model),
        )
        started_at = time.time()
        started_payload = call.to_dict()
        started_payload["started_at"] = started_at
        ctx.emit("inference_call_started", started_payload)
        clock = time.monotonic()
        try:
            response = self.inner.invoke(messages, config, tools=tools, tool_choice=tool_choice)
        except Exception as exc:
            call.status = "failed"
            call.error = f"{type(exc).__name__}: {exc}"
            call.latency_ms = round((time.monotonic() - clock) * 1000.0, 1)
            ctx.calls.append(call)
            ctx.emit("inference_call_failed", call.to_dict())
            raise
        call.status = "completed"
        call.resolved_model = str(getattr(response, "model", None) or self.model_name)
        call.latency_ms = round((time.monotonic() - clock) * 1000.0, 1)
        call.usage = _normalised_usage(response)
        ctx.calls.append(call)
        ctx.emit("inference_call_completed", call.to_dict())
        return response


def maybe_record(adapter: Any, model_name: str) -> Any:
    """Wrap *adapter* for recording only while a step context is open."""
    if _current.get() is None or isinstance(adapter, RecordingChatAdapter):
        return adapter
    return RecordingChatAdapter(adapter, model_name)


__all__ = [
    "CallContext",
    "RecordingChatAdapter",
    "call_context",
    "close_call_context",
    "current_call_context",
    "maybe_record",
    "open_call_context",
]
