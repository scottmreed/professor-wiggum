"""Runtime-only product API (Observatory PRD §21–§25, M3).

The research app in ``app.py`` exposes ~80 routes: evals, leaderboards,
harness editing, curation, memory, RAlph, the local UI. A product deployment
needs only what it takes to create, run, watch and replay a mechanism
prediction. ``create_runtime_app`` therefore builds the research app and
re-mounts **only** the allow-listed routes onto a new FastAPI app, so the
handlers stay identical to the ones Wiggum evaluates, and everything else is
absent by construction (no static UI, no research data routes).

Security (§25.1): every request except ``/healthz`` must carry
``Authorization: Bearer <MECHANISTIC_RUNTIME_TOKEN>``. With no token
configured the app fails closed (503) instead of serving unauthenticated.

Run it with ``uvicorn mechanistic_agent.api.runtime_app:create_runtime_app --factory``
(see ``Dockerfile.runtime``). ``/v1/mechanism/version`` returns the release
manifest (§21.4) so every product run can record the runtime identity.
"""
from __future__ import annotations

import hmac
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from mechanistic_agent.core.bond_electron import BE_CONVENTION
from mechanistic_agent.core.observatory import OBSERVATORY_SCHEMA
from mechanistic_agent.core.provenance import EVENT_SCHEMA_VERSION
from mechanistic_agent.core.reaction_focus import REACTION_FOCUS_SCHEMA
from mechanistic_agent.core.registries import RegistrySet

from .app import create_app

RUNTIME_VERSION = "0.1.0"
RUNTIME_MANIFEST_SCHEMA = "runtime_manifest.v1"
RUNTIME_TOKEN_ENV = "MECHANISTIC_RUNTIME_TOKEN"
RUNTIME_GIT_SHA_ENV = "MECHANISTIC_RUNTIME_GIT_SHA"
RUNTIME_HARNESS_ENV = "MECHANISTIC_RUNTIME_HARNESS"

# Product surface (§22). Paths are the research app's; a product facade may
# prefix them (e.g. ``/v1/mechanism``). Order is documentation only.
RUNTIME_ALLOWED_PATHS: tuple[str, ...] = (
    "/api/runs",
    "/api/runs/{run_id}",
    "/api/runs/{run_id}/start",
    "/api/runs/{run_id}/stop",
    "/api/runs/{run_id}/resume",
    "/api/runs/{run_id}/events",
    "/api/runs/{run_id}/flow",
    "/api/runs/{run_id}/observatory",
    "/api/runs/{run_id}/mechanism_steps",
    "/api/runs/{run_id}/steps/{step_name}/verify",
    "/api/molecules/render",
)

# Surfaces deliberately not mounted (§21.3), named so the manifest can state them.
RUNTIME_EXCLUDED_SURFACES: tuple[str, ...] = (
    "curation", "evals", "examples", "harness_editing", "leaderboard", "memory", "ralph", "traces", "ui",
)

OPEN_PATHS = frozenset({"/healthz"})


def _git_sha(base: Path) -> Optional[str]:
    env_value = os.getenv(RUNTIME_GIT_SHA_ENV)
    if env_value:
        return env_value.strip()
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(base), capture_output=True, text=True, timeout=3, check=False
        )
        sha = out.stdout.strip()
        return sha or None
    except Exception:
        return None


def build_runtime_manifest(base: Path) -> Dict[str, Any]:
    registry = RegistrySet(base)
    try:
        hashes = registry.bundle_hashes()
    except Exception:  # pragma: no cover - defensive
        hashes = {}
    harness_name = os.getenv(RUNTIME_HARNESS_ENV, "default")
    return {
        "schema_version": RUNTIME_MANIFEST_SCHEMA,
        "runtime_version": RUNTIME_VERSION,
        "git_sha": _git_sha(base),
        "harness_name": harness_name,
        "hashes": hashes,
        "harness_version": registry.harness_version() if hashes else None,
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "observatory_schema_version": OBSERVATORY_SCHEMA,
        "reaction_focus_version": REACTION_FOCUS_SCHEMA,
        "be_convention": BE_CONVENTION,
        "allowed_paths": list(RUNTIME_ALLOWED_PATHS),
        "excluded_surfaces": sorted(RUNTIME_EXCLUDED_SURFACES),
    }


def _authorized(request: Request) -> Optional[JSONResponse]:
    """Return an error response when the request may not proceed."""
    if request.url.path in OPEN_PATHS:
        return None
    expected = os.getenv(RUNTIME_TOKEN_ENV, "").strip()
    if not expected:
        return JSONResponse(
            status_code=503,
            content={"detail": f"Runtime auth token not configured; set {RUNTIME_TOKEN_ENV}. Refusing to serve unauthenticated."},
        )
    header = request.headers.get("authorization", "")
    scheme, _, presented = header.partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(presented.strip(), expected):
        return JSONResponse(status_code=401, content={"detail": "Missing or invalid bearer token."}, headers={"WWW-Authenticate": "Bearer"})
    return None


def create_runtime_app(base_dir: Path | None = None) -> FastAPI:
    base = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parents[2]
    research_app = create_app(base)

    app = FastAPI(
        title="Mechanistic Runtime",
        version=RUNTIME_VERSION,
        description="Product-facing mechanism runtime: run, watch and replay predictions. Research surfaces are not mounted.",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    allowed = set(RUNTIME_ALLOWED_PATHS)
    copied: List[str] = []
    for route in research_app.routes:
        if isinstance(route, APIRoute) and route.path in allowed:
            app.router.routes.append(route)
            copied.append(route.path)
    missing = allowed - set(copied)
    if missing:  # pragma: no cover - guards against the research app renaming a route
        raise RuntimeError(f"Runtime allow-list references routes the research app no longer defines: {sorted(missing)}")

    manifest = build_runtime_manifest(base)

    @app.middleware("http")
    async def _require_bearer(request: Request, call_next):  # type: ignore[no-untyped-def]
        denied = _authorized(request)
        if denied is not None:
            return denied
        return await call_next(request)

    @app.get("/healthz")
    def healthz() -> Dict[str, Any]:
        return {
            "status": "ok",
            "runtime_version": RUNTIME_VERSION,
            "auth_configured": bool(os.getenv(RUNTIME_TOKEN_ENV, "").strip()),
        }

    @app.get("/v1/mechanism/version")
    def version() -> Dict[str, Any]:
        return manifest

    @app.exception_handler(HTTPException)
    async def _http_exc(request: Request, exc: HTTPException):  # type: ignore[no-untyped-def]
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    app.state.runtime_manifest = manifest
    app.state.research_app = research_app
    return app


__all__ = [
    "RUNTIME_ALLOWED_PATHS",
    "RUNTIME_EXCLUDED_SURFACES",
    "RUNTIME_MANIFEST_SCHEMA",
    "RUNTIME_TOKEN_ENV",
    "RUNTIME_VERSION",
    "build_runtime_manifest",
    "create_runtime_app",
]
