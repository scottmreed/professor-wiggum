"""Shared pytest fixtures and shims for the Mechanistic agent test suite."""

from __future__ import annotations

import sys
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


if "openai" not in sys.modules:  # pragma: no cover - import shim for optional dependency
    openai_stub = types.ModuleType("openai")

    class _OpenAIStub:  # pragma: no cover - defensive guard
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("OpenAI client access is disabled during tests")

    openai_stub.OpenAI = _OpenAIStub  # type: ignore[attr-defined]
    sys.modules["openai"] = openai_stub
