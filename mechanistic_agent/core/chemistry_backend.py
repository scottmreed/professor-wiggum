"""Chemistry backend adapter (rdkit-agent subprocess with Python fallback)."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar


T = TypeVar("T")
logger = logging.getLogger(__name__)


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _read_cfg(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


@dataclass(frozen=True)
class ChemistryBackendConfig:
    backend: str = "auto"
    parity_mode: bool = False
    rdkit_cli_command: str = "rdkit-agent"
    rdkit_cli_command_custom: bool = False
    rdkit_cli_timeout_seconds: float = 5.0

    @classmethod
    def from_config(cls, config: Any = None) -> "ChemistryBackendConfig":
        backend = str(
            _read_cfg(
                config,
                "chemistry_backend",
                os.getenv("MECHANISTIC_CHEMISTRY_BACKEND", "auto"),
            )
            or "auto"
        ).strip().lower()
        if backend not in {"auto", "rdkit_cli", "rdkit_agent", "python"}:
            backend = "auto"
        # rdkit_agent is the canonical name; rdkit_cli is kept for backward compat
        if backend == "rdkit_agent":
            backend = "rdkit_cli"

        parity_mode = _truthy(
            _read_cfg(
                config,
                "chemistry_backend_parity",
                os.getenv("MECHANISTIC_CHEMISTRY_BACKEND_PARITY", "0"),
            ),
            default=False,
        )

        command_override = _read_cfg(config, "rdkit_cli_command", None)
        if command_override is None:
            command_override = os.getenv("MECHANISTIC_RDKIT_CLI_COMMAND", "rdkit-agent")
        rdkit_cli_command = str(command_override or "rdkit-agent").strip()
        rdkit_cli_command_custom = rdkit_cli_command not in {"", "rdkit_cli", "rdkit-agent"}

        timeout_raw = _read_cfg(
            config,
            "rdkit_cli_timeout_seconds",
            os.getenv("MECHANISTIC_RDKIT_CLI_TIMEOUT_SECONDS", "5"),
        )
        try:
            timeout_seconds = max(0.5, float(timeout_raw))
        except Exception:
            timeout_seconds = 5.0

        return cls(
            backend=backend,
            parity_mode=parity_mode,
            rdkit_cli_command=rdkit_cli_command,
            rdkit_cli_command_custom=rdkit_cli_command_custom,
            rdkit_cli_timeout_seconds=timeout_seconds,
        )


@dataclass(frozen=True)
class RdkitCliResolution:
    command_parts: Optional[list[str]]
    source: str
    executable_path: Optional[str] = None
    rejected: bool = False
    rejection_reason: Optional[str] = None
    warning: Optional[str] = None
    custom_override: bool = False


def _resolve_command(command: str) -> Optional[list[str]]:
    if not command:
        return None
    parts = shlex.split(command)
    if not parts:
        return None
    head = parts[0]
    if os.path.isabs(head) and os.path.exists(head):
        return parts
    if shutil.which(head):
        return parts
    return None


def _is_within(path: str, root: str) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except Exception:
        return False


def _local_npm_bin() -> str:
    return str((Path.cwd() / "node_modules" / ".bin" / "rdkit-agent").resolve())


def _local_npm_package() -> str:
    return str((Path.cwd() / "node_modules" / "rdkit-agent").resolve())


@lru_cache(maxsize=1)
def _global_npm_root() -> Optional[str]:
    npm = shutil.which("npm")
    if not npm:
        return None
    try:
        proc = subprocess.run(
            [npm, "root", "-g"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    root = str((proc.stdout or "").strip())
    if not root:
        return None
    return str(Path(root).resolve())


def _global_npm_package() -> Optional[str]:
    root = _global_npm_root()
    if not root:
        return None
    return str((Path(root) / "rdkit-agent").resolve())


def _allowed_rdkit_cli_roots() -> Tuple[list[str], list[str]]:
    """Return (allowed_roots, rejected_linked_roots)."""
    allowed: list[str] = []
    rejected: list[str] = []

    local_root = str((Path.cwd() / "node_modules").resolve())
    local_pkg = _local_npm_package()
    if os.path.exists(local_pkg):
        local_real = os.path.realpath(local_pkg)
        if _is_within(local_real, local_root):
            allowed.append(local_real)
        else:
            rejected.append(local_real)

    global_root = _global_npm_root()
    global_pkg = _global_npm_package()
    if global_root and global_pkg and os.path.exists(global_pkg):
        global_real = os.path.realpath(global_pkg)
        if _is_within(global_real, global_root):
            allowed.append(global_real)
        else:
            rejected.append(global_real)

    return allowed, rejected


def resolve_rdkit_cli_command(cfg: ChemistryBackendConfig) -> RdkitCliResolution:
    """Resolve rdkit-agent command using npm-first policy.

    Resolution order for default command:
      1) ./node_modules/.bin/rdkit-agent
      2) npx --no-install rdkit-agent
      3) PATH rdkit-agent
    """
    if cfg.rdkit_cli_command_custom:
        parts = _resolve_command(cfg.rdkit_cli_command)
        if not parts:
            return RdkitCliResolution(
                command_parts=None,
                source="custom_override",
                rejected=True,
                rejection_reason="custom_command_not_found",
                custom_override=True,
            )
        head = parts[0]
        resolved_path = os.path.realpath(head) if os.path.exists(head) else shutil.which(head)
        return RdkitCliResolution(
            command_parts=parts,
            source="custom_override",
            executable_path=str(resolved_path) if resolved_path else None,
            custom_override=True,
        )

    allowed_roots, rejected_roots = _allowed_rdkit_cli_roots()
    warning = None
    if rejected_roots:
        warning = (
            "linked rdkit-agent package(s) detected outside npm roots; "
            f"ignored: {', '.join(rejected_roots[:2])}"
        )

    local_bin = _local_npm_bin()
    if os.path.exists(local_bin):
        real_local_bin = os.path.realpath(local_bin)
        if any(_is_within(real_local_bin, root) for root in allowed_roots):
            return RdkitCliResolution(
                command_parts=[local_bin],
                source="npm_local_bin",
                executable_path=real_local_bin,
                warning=warning,
            )
        return RdkitCliResolution(
            command_parts=None,
            source="npm_local_bin",
            executable_path=real_local_bin,
            rejected=True,
            rejection_reason="linked_local_binary_outside_npm_roots",
            warning=warning,
        )

    npx = shutil.which("npx")
    if npx:
        global_pkg = _global_npm_package()
        global_pkg_real = os.path.realpath(global_pkg) if global_pkg else None
        if global_pkg and global_pkg_real and any(_is_within(global_pkg_real, root) for root in allowed_roots):
            return RdkitCliResolution(
                command_parts=[npx, "--no-install", "rdkit-agent"],
                source="npx_no_install",
                executable_path=npx,
                warning=warning,
            )
        # allow npx even without a known global package when local package exists.
        if any("node_modules/rdkit-agent" in root for root in allowed_roots):
            return RdkitCliResolution(
                command_parts=[npx, "--no-install", "rdkit-agent"],
                source="npx_no_install",
                executable_path=npx,
                warning=warning,
            )

    path_binary = shutil.which("rdkit-agent")
    if path_binary:
        real_path_binary = os.path.realpath(path_binary)
        if any(_is_within(real_path_binary, root) for root in allowed_roots):
            return RdkitCliResolution(
                command_parts=[path_binary],
                source="path",
                executable_path=real_path_binary,
                warning=warning,
            )
        return RdkitCliResolution(
            command_parts=None,
            source="path",
            executable_path=real_path_binary,
            rejected=True,
            rejection_reason="path_binary_outside_npm_roots",
            warning=warning,
        )

    return RdkitCliResolution(
        command_parts=None,
        source="none",
        rejected=False,
        warning=warning,
    )


def _invoke_rdkit_cli(
    *,
    command_parts: list[str],
    payload: Dict[str, Any],
    timeout_seconds: float,
) -> Dict[str, Any]:
    cmd = list(command_parts) + ["check", "--json", "-", "--output", "json"]
    proc = subprocess.run(
        cmd,
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    text = (proc.stdout or "").strip()
    parsed: Optional[Dict[str, Any]] = None
    if text:
        try:
            loaded = json.loads(text)
            if isinstance(loaded, dict):
                parsed = loaded
            else:
                raise RuntimeError("rdkit-agent check output must be a JSON object")
        except Exception as exc:
            if proc.returncode == 0:
                raise RuntimeError(f"rdkit-agent returned invalid JSON: {exc}") from exc

    # rdkit-agent uses exit code 1 for validation failure; when structured JSON is
    # present this is expected and should not trigger Python fallback.
    if proc.returncode in (0, 1):
        if parsed is not None:
            return parsed
        if proc.returncode == 0:
            raise RuntimeError("rdkit-agent check returned empty stdout")

    err_text = (proc.stderr or proc.stdout or "").strip() or f"exit_code={proc.returncode}"
    raise RuntimeError(f"rdkit-agent check failed: {err_text}")


def _first_cli_error_code(output: Dict[str, Any]) -> Optional[str]:
    failed = output.get("failed_checks")
    if not isinstance(failed, list):
        return None
    for item in failed:
        if not isinstance(item, dict):
            continue
        code = item.get("error_code")
        if isinstance(code, str) and code.strip():
            return code.strip()
    return None


def execute_chemistry_check(
    *,
    mode: str,
    payload: Dict[str, Any],
    config: Any,
    python_callable: Callable[[], T],
    cli_to_result: Callable[[Dict[str, Any]], T],
    python_signature: Optional[Callable[[T], Tuple[Any, ...]]] = None,
    cli_signature: Optional[Callable[[Dict[str, Any]], Tuple[Any, ...]]] = None,
) -> Tuple[T, Dict[str, Any]]:
    """Execute a chemistry check via rdkit-agent when available, with Python fallback.

    Returns:
      - result (authoritative output for caller)
      - metadata dict with backend/fallback/parity details
    """

    cfg = ChemistryBackendConfig.from_config(config)
    resolution = resolve_rdkit_cli_command(cfg)
    command_parts = resolution.command_parts
    requested = cfg.backend

    metadata: Dict[str, Any] = {
        "mode": mode,
        "backend_requested": requested,
        "backend_used": "python",
        "fallback_used": False,
        "fallback_reason": None,
        "rdkit_cli_command": cfg.rdkit_cli_command,
        "rdkit_cli_resolution_source": resolution.source,
        "rdkit_cli_resolution_warning": resolution.warning,
        "rdkit_cli_resolution_rejected": bool(resolution.rejected),
        "rdkit_cli_resolution_reason": resolution.rejection_reason,
        "rdkit_cli_available": bool(command_parts),
        "rdkit_cli_timeout_seconds": cfg.rdkit_cli_timeout_seconds,
        "parity_mode": bool(cfg.parity_mode),
    }
    if resolution.warning:
        logger.warning("rdkit-agent resolution warning (%s): %s", mode, resolution.warning)
    if resolution.rejected:
        logger.warning(
            "rdkit-agent resolution rejected (%s): source=%s reason=%s",
            mode,
            resolution.source,
            resolution.rejection_reason,
        )

    # Explicit Python mode, or auto/rdkit_cli when command is unavailable.
    if requested == "python" or not command_parts:
        if requested == "rdkit_cli" and not command_parts:
            metadata["fallback_used"] = True
            if resolution.rejected:
                metadata["fallback_reason"] = "rdkit_cli_policy_rejected"
                metadata["rdkit_cli_error"] = str(
                    resolution.rejection_reason or "rdkit-agent command rejected by source policy"
                )
            else:
                metadata["fallback_reason"] = "rdkit_cli_unavailable"
                metadata["rdkit_cli_error"] = "rdkit-agent command not found"
        elif resolution.rejected:
            metadata["fallback_used"] = True
            metadata["fallback_reason"] = "rdkit_cli_policy_rejected"
            metadata["rdkit_cli_error"] = str(
                resolution.rejection_reason or "rdkit-agent command rejected by source policy"
            )
        result = python_callable()
        metadata["backend_used"] = "python"
        return result, metadata

    # rdkit_cli path.
    try:
        cli_output = _invoke_rdkit_cli(
            command_parts=command_parts,
            payload=payload,
            timeout_seconds=cfg.rdkit_cli_timeout_seconds,
        )
        metadata["backend_used"] = "rdkit_cli"
        metadata["rdkit_cli_error_code"] = _first_cli_error_code(cli_output)
        metadata["rdkit_cli_overall_pass"] = bool(cli_output.get("overall_pass"))
        metadata["rdkit_cli_failed_check_names"] = list(cli_output.get("failed_check_names") or [])
        authoritative = cli_to_result(cli_output)
    except Exception as exc:
        # Automatic fallback to Python for per-call subprocess failures.
        metadata["fallback_used"] = True
        metadata["fallback_reason"] = "rdkit_cli_error"
        metadata["rdkit_cli_error"] = str(exc)
        authoritative = python_callable()
        metadata["backend_used"] = "python"
        return authoritative, metadata

    # Optional dual-run parity mode with Python-authoritative mismatch handling.
    if cfg.parity_mode:
        parity: Dict[str, Any] = {"enabled": True}
        try:
            python_result = python_callable()
            if python_signature is not None and cli_signature is not None:
                py_sig = tuple(python_signature(python_result))
                cli_sig = tuple(cli_signature(cli_output))
                parity["python_signature"] = list(py_sig)
                parity["rdkit_cli_signature"] = list(cli_sig)
                parity["match"] = py_sig == cli_sig
                if py_sig != cli_sig:
                    # Python remains authoritative in parity mismatch mode.
                    authoritative = python_result
                    metadata["backend_used"] = "python"
                    metadata["fallback_used"] = True
                    metadata["fallback_reason"] = "parity_mismatch"
            else:
                parity["match"] = None
        except Exception as exc:
            parity["match"] = None
            parity["python_error"] = str(exc)
        metadata["parity"] = parity

    return authoritative, metadata
