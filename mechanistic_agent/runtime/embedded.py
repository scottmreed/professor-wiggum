"""Embedded Mechanism Runtime (Observatory PRD rev 3 §2.3.2).

The product (ChemIllusion) runs a tagged Wiggum release **inside its existing
API service** instead of a separate Railway service. Its job worker does::

    rt = EmbeddedMechanismRuntime(base_dir="/opt/wiggum", work_dir=tmp, event_sink=mirror_to_postgres)
    run_id = rt.create_run(request)        # same validation/normalization as POST /api/runs
    rt.execute(run_id, stop_event=stop)    # blocking, on the worker thread; heartbeat the lease meanwhile
    view = rt.observatory(run_id)          # or build_observatory(<mirrored events>) after a restart

Everything goes through the research app's own handlers and coordinator, so the
behaviour is identical to what this repository evaluates. The SQLite database
under ``work_dir`` is scratch; durability comes from ``event_sink``, and because
the ``/observatory`` projection is built from events alone, the mirrored log
reproduces the view after a container restart.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from fastapi import HTTPException
from fastapi.routing import APIRoute

from mechanistic_agent.api.app import create_app
from mechanistic_agent.api.runtime_app import build_runtime_manifest
from mechanistic_agent.api.schemas import CreateRunRequest
from mechanistic_agent.core.observatory import build_observatory

EventSink = Callable[[Dict[str, Any]], None]


class EmbeddedMechanismRuntime:
    def __init__(
        self,
        *,
        base_dir: Path | str,
        work_dir: Path | str,
        event_sink: Optional[EventSink] = None,
    ) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._app = create_app(self.base_dir, db_path=self.work_dir / "mechanistic.db", event_sink=event_sink)
        self.store = self._app.state.store
        self.coordinator = self._app.state.coordinator
        self.db_path: Path = Path(self._app.state.db_path)
        self._create_endpoint = self._endpoint("/api/runs", "POST")
        manifest = build_runtime_manifest(self.base_dir)
        manifest["deployment"] = "embedded"
        self._manifest = manifest

    def _endpoint(self, path: str, method: str) -> Callable[..., Any]:
        for route in self._app.routes:
            if isinstance(route, APIRoute) and route.path == path and method in route.methods:
                return route.endpoint
        raise RuntimeError(f"research app has no {method} {path}")  # pragma: no cover

    # -- lifecycle ---------------------------------------------------------

    def create_run(self, request: Dict[str, Any]) -> str:
        """Create a run from a ``CreateRunRequest``-shaped dict. Raises ValueError on bad input."""
        try:
            payload = CreateRunRequest(**dict(request))
            response = self._create_endpoint(payload)
        except HTTPException as exc:
            raise ValueError(str(exc.detail)) from exc
        return str(response.run_id if hasattr(response, "run_id") else response["run_id"])

    def execute(self, run_id: str, *, stop_event: Optional[threading.Event] = None) -> str:
        """Run to completion on the calling thread; returns the final status."""
        if self.store.get_run_row(run_id) is None:
            raise KeyError(run_id)
        self.store.append_event(run_id, "run_start_requested", {"deployment": "embedded"})
        self.coordinator.execute_run(run_id, stop_event or threading.Event())
        return self.status(run_id)

    def status(self, run_id: str) -> str:
        row = self.store.get_run_row(run_id)
        if row is None:
            raise KeyError(run_id)
        return str(row.get("status") or "")

    # -- read side ---------------------------------------------------------

    def observatory(self, run_id: str) -> Dict[str, Any]:
        row = self.store.get_run_row(run_id)
        if row is None:
            raise KeyError(run_id)
        events = self.store.list_events(run_id, after_seq=0, limit=20000)
        run_input = row.get("input_payload") if isinstance(row.get("input_payload"), dict) else {}
        view = build_observatory(events, run_id=run_id, run_input=run_input, status=row.get("status"))
        view["runtime"] = {k: self._manifest.get(k) for k in ("runtime_version", "git_sha", "harness_name", "deployment")}
        return view

    def events(self, run_id: str, *, after_seq: int = 0, limit: int = 500) -> list:
        return self.store.list_events(run_id, after_seq=after_seq, limit=limit)

    def manifest(self) -> Dict[str, Any]:
        return dict(self._manifest)


__all__ = ["EmbeddedMechanismRuntime"]
