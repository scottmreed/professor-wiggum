"""Core package exports for the local-first mechanistic runtime."""

try:  # pragma: no cover - optional dependency surface
    from .config import AgentLimits, OPTIONAL_LLM_TOOL_NAMES, ReactionInputs
except Exception:  # pragma: no cover
    AgentLimits = None  # type: ignore[assignment]
    OPTIONAL_LLM_TOOL_NAMES = []  # type: ignore[assignment]
    ReactionInputs = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency surface
    from .core import RegistrySet, RunCoordinator, RunManager, RunStore
except Exception:  # pragma: no cover
    RegistrySet = None  # type: ignore[assignment]
    RunCoordinator = None  # type: ignore[assignment]
    RunManager = None  # type: ignore[assignment]
    RunStore = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency surface
    from .tools import TOOLKIT
except Exception:  # pragma: no cover
    TOOLKIT = []  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency surface
    from .api import create_app
except Exception:  # pragma: no cover
    create_app = None  # type: ignore[assignment]

__all__ = [
    "ReactionInputs",
    "AgentLimits",
    "OPTIONAL_LLM_TOOL_NAMES",
    "TOOLKIT",
    "create_app",
    "RunCoordinator",
    "RunManager",
    "RunStore",
    "RegistrySet",
]
