"""Tests for deterministic atom-mapping validation via rdkit-agent atom-map check."""
from __future__ import annotations

import json
import subprocess
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from mechanistic_agent.core.chemistry_backend import RdkitCliResolution
from mechanistic_agent.core.types import StepValidationResult
from mechanistic_agent.tools import validate_atom_mapping_via_rdkit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_resolution(available: bool = True) -> RdkitCliResolution:
    if available:
        return RdkitCliResolution(
            command_parts=["rdkit-agent"],
            source="test",
        )
    return RdkitCliResolution(command_parts=None, source="none")


def _make_proc(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["rdkit-agent", "atom-map", "--json", "{}"],
        returncode=returncode,
        stdout=stdout,
        stderr="",
    )


SAMPLE_REACTANTS = ["CCO", "O"]
SAMPLE_PRODUCTS = ["CC=O", "O"]
SAMPLE_MAPPED_ATOMS: List[Dict[str, Any]] = [
    {
        "product_atom": "CC=O#0",
        "source": {"molecule_index": 0, "smiles": "CCO", "atom_index": 0},
    },
]


# ---------------------------------------------------------------------------
# Tests: validate_atom_mapping_via_rdkit
# ---------------------------------------------------------------------------

class TestValidateAtomMappingViaRdkit:

    def test_passes_for_valid_mapping(self, monkeypatch):
        """When rdkit-agent returns valid=True, StepValidationResult.passed is True."""
        monkeypatch.setattr(
            "mechanistic_agent.tools.resolve_rdkit_cli_command",
            lambda _cfg: _fake_resolution(True),
        )
        monkeypatch.setattr(
            "mechanistic_agent.tools.subprocess.run",
            lambda *a, **kw: _make_proc(json.dumps({
                "valid": True,
                "errors": [],
                "warnings": [],
            })),
        )
        result = validate_atom_mapping_via_rdkit(
            starting_materials=SAMPLE_REACTANTS,
            products=SAMPLE_PRODUCTS,
            mapped_atoms=SAMPLE_MAPPED_ATOMS,
        )
        assert result is not None
        assert result.passed is True
        assert len(result.checks) == 1
        assert result.checks[0].name == "atom_map_check"

    def test_catches_invalid_atom_index(self, monkeypatch):
        """When rdkit-agent returns valid=False with errors, validation fails."""
        errors = [{"type": "invalid_atom_index", "detail": "atom index 99 out of range"}]
        monkeypatch.setattr(
            "mechanistic_agent.tools.resolve_rdkit_cli_command",
            lambda _cfg: _fake_resolution(True),
        )
        monkeypatch.setattr(
            "mechanistic_agent.tools.subprocess.run",
            lambda *a, **kw: _make_proc(json.dumps({
                "valid": False,
                "errors": errors,
                "warnings": [],
            }), returncode=1),
        )
        result = validate_atom_mapping_via_rdkit(
            starting_materials=SAMPLE_REACTANTS,
            products=SAMPLE_PRODUCTS,
            mapped_atoms=SAMPLE_MAPPED_ATOMS,
        )
        assert result is not None
        assert result.passed is False
        assert result.checks[0].details["errors"] == errors

    def test_skips_when_rdkit_agent_unavailable(self, monkeypatch):
        """When rdkit-agent is not installed, returns None (graceful skip)."""
        monkeypatch.setattr(
            "mechanistic_agent.tools.resolve_rdkit_cli_command",
            lambda _cfg: _fake_resolution(False),
        )
        result = validate_atom_mapping_via_rdkit(
            starting_materials=SAMPLE_REACTANTS,
            products=SAMPLE_PRODUCTS,
            mapped_atoms=SAMPLE_MAPPED_ATOMS,
        )
        assert result is None

    def test_skips_on_subprocess_timeout(self, monkeypatch):
        """When subprocess times out, returns None (graceful skip)."""
        monkeypatch.setattr(
            "mechanistic_agent.tools.resolve_rdkit_cli_command",
            lambda _cfg: _fake_resolution(True),
        )

        def _raise_timeout(*a, **kw):
            raise subprocess.TimeoutExpired(cmd="rdkit-agent", timeout=5)

        monkeypatch.setattr("mechanistic_agent.tools.subprocess.run", _raise_timeout)
        result = validate_atom_mapping_via_rdkit(
            starting_materials=SAMPLE_REACTANTS,
            products=SAMPLE_PRODUCTS,
            mapped_atoms=SAMPLE_MAPPED_ATOMS,
        )
        assert result is None

    def test_skips_when_no_mapped_atoms(self):
        """When mapped_atoms is empty/None, returns a trivial pass (nothing to validate)."""
        result = validate_atom_mapping_via_rdkit(
            starting_materials=SAMPLE_REACTANTS,
            products=SAMPLE_PRODUCTS,
            mapped_atoms=None,
        )
        assert result is not None
        assert result.passed is True
        assert result.checks[0].details.get("skipped") is True

        result2 = validate_atom_mapping_via_rdkit(
            starting_materials=SAMPLE_REACTANTS,
            products=SAMPLE_PRODUCTS,
            mapped_atoms=[],
        )
        assert result2 is not None
        assert result2.passed is True

    def test_handles_unexpected_response_shape(self, monkeypatch):
        """When response lacks 'valid' key, treat as skip rather than crash."""
        monkeypatch.setattr(
            "mechanistic_agent.tools.resolve_rdkit_cli_command",
            lambda _cfg: _fake_resolution(True),
        )
        monkeypatch.setattr(
            "mechanistic_agent.tools.subprocess.run",
            lambda *a, **kw: _make_proc(json.dumps({"status": "unknown_format"})),
        )
        result = validate_atom_mapping_via_rdkit(
            starting_materials=SAMPLE_REACTANTS,
            products=SAMPLE_PRODUCTS,
            mapped_atoms=SAMPLE_MAPPED_ATOMS,
        )
        assert result is not None
        assert result.passed is True
        assert result.checks[0].details.get("skipped") is True


# ---------------------------------------------------------------------------
# Tests: MappingAgent integration
# ---------------------------------------------------------------------------

class TestMappingAgentValidation:

    def _make_state(self):
        """Build a minimal RunState-like object for MappingAgent."""
        from types import SimpleNamespace
        return SimpleNamespace(
            run_input=SimpleNamespace(
                starting_materials=SAMPLE_REACTANTS,
                products=SAMPLE_PRODUCTS,
            ),
            run_config=SimpleNamespace(
                step_models={},
                model="test-model",
            ),
        )

    def test_run_attaches_validation(self, monkeypatch):
        """MappingAgent.run() attaches validation to the StepResult."""
        from mechanistic_agent.core.subagents import MappingAgent

        fake_output = {
            "llm_response": {
                "mapped_atoms": SAMPLE_MAPPED_ATOMS,
                "confidence": 0.9,
            },
        }
        mock_executor = MagicMock()
        mock_executor.run_mapping.return_value = fake_output

        validation_result = StepValidationResult(checks=[])

        monkeypatch.setattr(
            "mechanistic_agent.core.subagents.MappingAgent._validate_mapping",
            staticmethod(lambda output, **kw: validation_result),
        )

        agent = MappingAgent(executor=mock_executor)
        step = agent.run(self._make_state())
        assert step.validation is validation_result
        assert step.step_name == "atom_mapping"

    def test_confidence_clamped_on_failure(self, monkeypatch):
        """When validation fails, confidence is clamped to 0.3."""
        from mechanistic_agent.core.subagents import MappingAgent
        from mechanistic_agent.core.types import StepValidationCheck

        fake_output = {
            "llm_response": {
                "mapped_atoms": SAMPLE_MAPPED_ATOMS,
                "confidence": 0.9,
            },
        }
        mock_executor = MagicMock()
        mock_executor.run_mapping.return_value = fake_output

        failing_validation = StepValidationResult(checks=[
            StepValidationCheck(name="atom_map_check", passed=False, details={"errors": [{"type": "bad"}]}),
        ])

        # Use the real _validate_mapping but mock the underlying rdkit call
        monkeypatch.setattr(
            "mechanistic_agent.tools.validate_atom_mapping_via_rdkit",
            lambda **kw: failing_validation,
        )

        agent = MappingAgent(executor=mock_executor)
        step = agent.run(self._make_state())
        assert step.validation is failing_validation
        assert step.validation.passed is False
        # Confidence should be clamped
        assert fake_output["llm_response"]["confidence"] <= 0.3

    def test_run_step_mapping_attaches_validation(self, monkeypatch):
        """MappingAgent.run_step_mapping() also attaches validation."""
        from mechanistic_agent.core.subagents import MappingAgent

        fake_output = {
            "llm_response": {
                "mapped_atoms": SAMPLE_MAPPED_ATOMS,
                "confidence": 0.8,
            },
        }
        mock_executor = MagicMock()
        mock_executor.run_step_mapping.return_value = fake_output

        validation_result = StepValidationResult(checks=[])

        monkeypatch.setattr(
            "mechanistic_agent.core.subagents.MappingAgent._validate_mapping",
            staticmethod(lambda output, **kw: validation_result),
        )

        agent = MappingAgent(executor=mock_executor)
        step = agent.run_step_mapping(
            self._make_state(),
            current_state=["CCO"],
            resulting_state=["CC=O"],
        )
        assert step.validation is validation_result
        assert step.step_name == "step_atom_mapping"
