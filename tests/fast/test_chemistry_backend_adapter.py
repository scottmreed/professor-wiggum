from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, Tuple

from mechanistic_agent.core import chemistry_backend
from mechanistic_agent.core.chemistry_backend import (
    ChemistryBackendConfig,
    RdkitCliResolution,
    execute_chemistry_check,
    resolve_rdkit_cli_command,
)
from mechanistic_agent.core.validators import validate_mechanism_step_output


def _python_result() -> Tuple[str, Dict[str, Any]]:
    return "python", {"source": "python"}


def _cli_to_result(output: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    return "rdkit_cli", {"source": "rdkit_cli", "output": output}


def test_backend_auto_uses_python_when_cli_unavailable(monkeypatch):
    monkeypatch.setattr(
        "mechanistic_agent.core.chemistry_backend.resolve_rdkit_cli_command",
        lambda _cfg: RdkitCliResolution(command_parts=None, source="none"),
    )
    result, meta = execute_chemistry_check(
        mode="smiles",
        payload={"smiles": "CCO"},
        config={"chemistry_backend": "auto"},
        python_callable=_python_result,
        cli_to_result=_cli_to_result,
    )
    assert result[0] == "python"
    assert meta["backend_used"] == "python"
    assert meta["fallback_used"] is False


def test_backend_rdkit_cli_unavailable_falls_back(monkeypatch):
    monkeypatch.setattr(
        "mechanistic_agent.core.chemistry_backend.resolve_rdkit_cli_command",
        lambda _cfg: RdkitCliResolution(command_parts=None, source="none"),
    )
    result, meta = execute_chemistry_check(
        mode="smiles",
        payload={"smiles": "CCO"},
        config={"chemistry_backend": "rdkit_cli"},
        python_callable=_python_result,
        cli_to_result=_cli_to_result,
    )
    assert result[0] == "python"
    assert meta["backend_used"] == "python"
    assert meta["fallback_used"] is True
    assert meta["fallback_reason"] == "rdkit_cli_unavailable"


def test_backend_rdkit_cli_policy_rejected_falls_back(monkeypatch):
    monkeypatch.setattr(
        "mechanistic_agent.core.chemistry_backend.resolve_rdkit_cli_command",
        lambda _cfg: RdkitCliResolution(
            command_parts=None,
            source="path",
            rejected=True,
            rejection_reason="path_binary_outside_npm_roots",
        ),
    )
    result, meta = execute_chemistry_check(
        mode="smiles",
        payload={"smiles": "CCO"},
        config={"chemistry_backend": "auto"},
        python_callable=_python_result,
        cli_to_result=_cli_to_result,
    )
    assert result[0] == "python"
    assert meta["backend_used"] == "python"
    assert meta["fallback_used"] is True
    assert meta["fallback_reason"] == "rdkit_cli_policy_rejected"
    assert meta["rdkit_cli_resolution_rejected"] is True


def test_backend_rdkit_cli_error_falls_back(monkeypatch):
    monkeypatch.setattr(
        "mechanistic_agent.core.chemistry_backend.resolve_rdkit_cli_command",
        lambda _cfg: RdkitCliResolution(command_parts=["rdkit_cli"], source="path"),
    )

    def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "mechanistic_agent.core.chemistry_backend._invoke_rdkit_cli",
        _raise,
    )

    result, meta = execute_chemistry_check(
        mode="smiles",
        payload={"smiles": "CCO"},
        config={"chemistry_backend": "auto"},
        python_callable=_python_result,
        cli_to_result=_cli_to_result,
    )
    assert result[0] == "python"
    assert meta["backend_used"] == "python"
    assert meta["fallback_used"] is True
    assert meta["fallback_reason"] == "rdkit_cli_error"
    assert "boom" in str(meta.get("rdkit_cli_error") or "")


def test_backend_uses_rdkit_cli_on_success(monkeypatch):
    monkeypatch.setattr(
        "mechanistic_agent.core.chemistry_backend.resolve_rdkit_cli_command",
        lambda _cfg: RdkitCliResolution(command_parts=["rdkit_cli"], source="path"),
    )
    monkeypatch.setattr(
        "mechanistic_agent.core.chemistry_backend._invoke_rdkit_cli",
        lambda **kwargs: {"overall_pass": True, "checks": [], "failed_checks": [], "failed_check_names": []},
    )
    result, meta = execute_chemistry_check(
        mode="smiles",
        payload={"smiles": "CCO"},
        config={"chemistry_backend": "auto"},
        python_callable=_python_result,
        cli_to_result=_cli_to_result,
    )
    assert result[0] == "rdkit_cli"
    assert meta["backend_used"] == "rdkit_cli"
    assert meta["fallback_used"] is False


def test_invoke_rdkit_cli_accepts_validation_failure_json(monkeypatch):
    cli_payload = {
        "overall_pass": False,
        "checks": [{"name": "state_progress", "pass": False}],
        "failed_checks": [{"name": "state_progress", "error_code": "state_progress_unchanged_starting_materials"}],
        "failed_check_names": ["state_progress"],
    }

    def _fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout=json.dumps(cli_payload), stderr="")

    monkeypatch.setattr("mechanistic_agent.core.chemistry_backend.subprocess.run", _fake_run)

    parsed = chemistry_backend._invoke_rdkit_cli(
        command_parts=["rdkit_cli"],
        payload={"mechanism-step": True},
        timeout_seconds=5.0,
    )
    assert parsed == cli_payload


def test_backend_uses_rdkit_cli_on_validation_failure(monkeypatch):
    monkeypatch.setattr(
        "mechanistic_agent.core.chemistry_backend.resolve_rdkit_cli_command",
        lambda _cfg: RdkitCliResolution(command_parts=["rdkit_cli"], source="path"),
    )
    monkeypatch.setattr(
        "mechanistic_agent.core.chemistry_backend._invoke_rdkit_cli",
        lambda **kwargs: {
            "overall_pass": False,
            "checks": [{"name": "atom_balance", "pass": False}],
            "failed_checks": [{"name": "atom_balance", "error_code": "atom_balance_mismatch"}],
            "failed_check_names": ["atom_balance"],
        },
    )
    result, meta = execute_chemistry_check(
        mode="mechanism_step",
        payload={"mechanism-step": True},
        config={"chemistry_backend": "rdkit_cli"},
        python_callable=_python_result,
        cli_to_result=_cli_to_result,
    )
    assert result[0] == "rdkit_cli"
    assert meta["backend_used"] == "rdkit_cli"
    assert meta["fallback_used"] is False
    assert meta["rdkit_cli_error_code"] == "atom_balance_mismatch"


def test_backend_parity_mismatch_uses_python(monkeypatch):
    monkeypatch.setattr(
        "mechanistic_agent.core.chemistry_backend.resolve_rdkit_cli_command",
        lambda _cfg: RdkitCliResolution(command_parts=["rdkit_cli"], source="path"),
    )
    monkeypatch.setattr(
        "mechanistic_agent.core.chemistry_backend._invoke_rdkit_cli",
        lambda **kwargs: {"overall_pass": True, "checks": [{"name": "atom_balance", "pass": True}], "failed_checks": [], "failed_check_names": []},
    )

    result, meta = execute_chemistry_check(
        mode="mechanism_step",
        payload={"mechanism-step": True},
        config={"chemistry_backend": "auto", "chemistry_backend_parity": True},
        python_callable=lambda: ("python", {"passed": False}),
        cli_to_result=lambda output: ("rdkit_cli", {"passed": True}),
        python_signature=lambda result_value: (result_value[0], False),
        cli_signature=lambda output: ("rdkit_cli", True),
    )
    assert result[0] == "python"
    assert meta["backend_used"] == "python"
    assert meta["fallback_used"] is True
    assert meta["fallback_reason"] == "parity_mismatch"


def test_resolve_rdkit_cli_prefers_npm_local_over_path(monkeypatch):
    monkeypatch.setattr(
        "mechanistic_agent.core.chemistry_backend._allowed_rdkit_cli_roots",
        lambda: (["/repo/node_modules/rdkit-agent"], []),
    )
    monkeypatch.setattr(
        "mechanistic_agent.core.chemistry_backend._local_npm_bin",
        lambda: "/repo/node_modules/.bin/rdkit-agent",
    )

    def _exists(path: str) -> bool:
        return path == "/repo/node_modules/.bin/rdkit-agent"

    monkeypatch.setattr("mechanistic_agent.core.chemistry_backend.os.path.exists", _exists)
    monkeypatch.setattr(
        "mechanistic_agent.core.chemistry_backend._is_within",
        lambda path, root: str(path).startswith(str(root)),
    )
    monkeypatch.setattr(
        "mechanistic_agent.core.chemistry_backend.os.path.realpath",
        lambda path: "/repo/node_modules/rdkit-agent/bin/rdkit_cli.js"
        if path == "/repo/node_modules/.bin/rdkit-agent"
        else path,
    )
    monkeypatch.setattr(
        "mechanistic_agent.core.chemistry_backend.shutil.which",
        lambda name: "/usr/local/bin/rdkit-agent" if name == "rdkit-agent" else None,
    )
    cfg = ChemistryBackendConfig.from_config({"rdkit_cli_command": "rdkit-agent"})
    resolution = resolve_rdkit_cli_command(cfg)
    assert resolution.source == "npm_local_bin"
    assert resolution.command_parts == ["/repo/node_modules/.bin/rdkit-agent"]
    assert resolution.rejected is False


def test_resolve_rdkit_cli_rejects_linked_path_binary(monkeypatch):
    monkeypatch.setattr(
        "mechanistic_agent.core.chemistry_backend._allowed_rdkit_cli_roots",
        lambda: (["/repo/node_modules/rdkit-agent"], ["/Users/scott/PycharmProjects/rdkit-agent"]),
    )
    monkeypatch.setattr(
        "mechanistic_agent.core.chemistry_backend._local_npm_bin",
        lambda: "/repo/node_modules/.bin/rdkit-agent",
    )
    monkeypatch.setattr("mechanistic_agent.core.chemistry_backend.os.path.exists", lambda _path: False)
    monkeypatch.setattr(
        "mechanistic_agent.core.chemistry_backend.shutil.which",
        lambda name: "/opt/homebrew/bin/rdkit-agent" if name == "rdkit-agent" else None,
    )
    monkeypatch.setattr(
        "mechanistic_agent.core.chemistry_backend.os.path.realpath",
        lambda path: "/Users/scott/PycharmProjects/rdkit-agent/bin/rdkit_cli.js"
        if path == "/opt/homebrew/bin/rdkit-agent"
        else path,
    )
    cfg = ChemistryBackendConfig.from_config({"rdkit_cli_command": "rdkit-agent"})
    resolution = resolve_rdkit_cli_command(cfg)
    assert resolution.command_parts is None
    assert resolution.source == "path"
    assert resolution.rejected is True
    assert resolution.rejection_reason == "path_binary_outside_npm_roots"
    assert "linked rdkit-agent package" in str(resolution.warning or "")


def test_resolve_rdkit_cli_honors_explicit_override(monkeypatch):
    monkeypatch.setattr(
        "mechanistic_agent.core.chemistry_backend._resolve_command",
        lambda command: ["node", "bin/rdkit_cli.js"] if command == "node bin/rdkit_cli.js" else None,
    )
    monkeypatch.setattr(
        "mechanistic_agent.core.chemistry_backend.shutil.which",
        lambda name: "/usr/local/bin/node" if name == "node" else None,
    )
    cfg = ChemistryBackendConfig.from_config({"rdkit_cli_command": "node bin/rdkit_cli.js"})
    resolution = resolve_rdkit_cli_command(cfg)
    assert resolution.source == "custom_override"
    assert resolution.custom_override is True
    assert resolution.command_parts == ["node", "bin/rdkit_cli.js"]
    assert resolution.rejected is False


def test_validator_attaches_backend_metadata():
    payload = {
        "current_state": ["CCO"],
        "resulting_state": ["CC=O"],
        "unchanged_starting_materials_detected": False,
        "resulting_state_changed": True,
        "bond_electron_validation": {"valid": True, "message": "ok", "total_delta": 0},
    }
    result = validate_mechanism_step_output(
        payload,
        dbe_policy="soft",
        run_config={"chemistry_backend": "python"},
    )
    assert result.checks
    for check in result.checks:
        assert "chemistry_backend" in check.details
        assert check.details["chemistry_backend"]["backend_used"] == "python"
