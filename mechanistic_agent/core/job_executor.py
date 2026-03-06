"""Swappable background-job execution interface.

The default implementation uses daemon threads (ThreadJobExecutor), matching
the existing runtime behaviour.  A cloud implementation could dispatch to
Cloudflare Queues, Durable Objects, or a task queue instead.

Usage
-----
    executor = ThreadJobExecutor()
    executor.start("my-job-id", my_fn, arg1, arg2)
    if executor.is_running("my-job-id"):
        ...
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Dict

try:
    from typing import Protocol, runtime_checkable
except ImportError:  # Python < 3.8
    from typing_extensions import Protocol, runtime_checkable  # type: ignore[assignment]


@runtime_checkable
class JobExecutor(Protocol):
    """Minimal interface for launching and tracking background jobs."""

    def start(self, job_id: str, fn: Callable[..., Any], /, *args: Any) -> None:
        """Start *fn* as a background job identified by *job_id*.

        *fn* is called with *args* in the background.  If *job_id* is already
        running, behaviour is implementation-defined (ThreadJobExecutor allows
        duplicate IDs — the old thread continues and the new one replaces it in
        the tracking dict).
        """
        ...

    def is_running(self, job_id: str) -> bool:
        """Return True if the job with *job_id* is still running."""
        ...


class ThreadJobExecutor:
    """JobExecutor backed by daemon threads — default local implementation."""

    def __init__(self) -> None:
        self._threads: Dict[str, threading.Thread] = {}

    def start(self, job_id: str, fn: Callable[..., Any], /, *args: Any) -> None:
        thread = threading.Thread(target=fn, args=args, daemon=True)
        self._threads[job_id] = thread
        thread.start()

    def is_running(self, job_id: str) -> bool:
        thread = self._threads.get(job_id)
        return thread is not None and thread.is_alive()
