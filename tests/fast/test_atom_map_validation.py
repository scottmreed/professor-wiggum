"""Tests for deterministic atom-mapping validation via ``rdkit-agent atom-map check``.

The real CLI contract (rdkit-agent 0.1.1):

    rdkit-agent atom-map check --json '{"smirks": "<mapped smirks>"}'

* ``atom-map`` requires a sub-command; without one it exits 2 with
  ``{"error": "No sub-command provided. ..."}``.
* ``check`` takes a mapped reaction SMIRKS, not index pairs, and returns
  ``{"valid": bool, "balanced": bool, "mapped_atoms": int, "unmapped_atoms": int,
  "map_numbers_only_in_reactants": [...], "map_numbers_only_in_products": [...]}``
  with exit code 0 even when ``valid``/``balanced`` are false.
"""
from __future__ import annotations

import json
import logging
import subprocess
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from mechanistic_agent.core.chemistry_backend import (
    ChemistryBackendConfig,
    RdkitCliResolution,
    resolve_rdkit_cli_command,
)
from mechanistic_agent.core.types import StepValidationCheck, StepValidationResult
from mechanistic_agent.tools import validate_atom_mapping_via_rdkit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_resolution(available: bool = True) -> RdkitCliResolution:
    if available:
        return RdkitCliResolution(command_parts=["rdkit-agent"], source="test")
    return RdkitCliResolution(command_parts=None, source="none")


def _make_proc(stdout: str, returncode: int = 0, argv: List[str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=argv or ["rdkit-agent"],
        returncode=returncode,
        stdout=stdout,
        stderr="",
    )


def _cli_check_response(**overrides: Any) -> str:
    """A realistic ``atom-map check`` stdout payload."""
    payload: Dict[str, Any] = {
        "smirks": EXPECTED_SMIRKS,
        "valid": True,
        "mapped_atoms": 3,
        "unmapped_atoms": 2,
        "balanced": True,
        "map_numbers_only_in_reactants": [],
        "map_numbers_only_in_products": [],
        "reactant_atom_count": 4,
        "product_atom_count": 4,
    }
    payload.update(overrides)
    return json.dumps(payload)


class _RecordingRunner:
    """Stand-in for ``subprocess.run`` that records argv and returns a canned process."""

    def __init__(self, stdout: str, returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.calls: List[List[str]] = []

    def __call__(self, cmd: List[str], *args: Any, **kwargs: Any) -> subprocess.CompletedProcess:
        self.calls.append(list(cmd))
        return _make_proc(self.stdout, self.returncode, argv=list(cmd))


SAMPLE_REACTANTS = ["CCO", "O"]
SAMPLE_PRODUCTS = ["CC=O", "O"]
# Full mapping of ethanol -> acetaldehyde; water is a spectator.
SAMPLE_MAPPED_ATOMS: List[Dict[str, Any]] = [
    {"product_atom": "CC=O#0", "source": {"molecule_index": 0, "smiles": "CCO", "atom_index": 0}},
    {"product_atom": "CC=O#1", "source": {"molecule_index": 0, "smiles": "CCO", "atom_index": 1}},
    {"product_atom": "CC=O#2", "source": {"molecule_index": 0, "smiles": "CCO", "atom_index": 2}},
]
# RDKit canonical rendering of the pairs above with fresh map numbers 1..3.
EXPECTED_SMIRKS = "[CH3:1][CH2:2][OH:3].O>>[CH3:1][CH:2]=[O:3].O"

# Third pair maps the product O onto a reactant C: element mismatch, must be dropped.
MISMATCHED_MAPPED_ATOMS: List[Dict[str, Any]] = SAMPLE_MAPPED_ATOMS[:2] + [
    {"product_atom": "CC=O#2", "source": {"molecule_index": 0, "smiles": "CCO", "atom_index": 1}},
]


def _checks_by_name(result: StepValidationResult) -> Dict[str, StepValidationCheck]:
    return {check.name: check for check in result.checks}


@pytest.fixture
def cli_available(monkeypatch):
    monkeypatch.setattr(
        "mechanistic_agent.tools.resolve_rdkit_cli_command",
        lambda _cfg: _fake_resolution(True),
    )


# ---------------------------------------------------------------------------
# Tests: invocation shape
# ---------------------------------------------------------------------------

class TestInvocationShape:

    def test_invokes_atom_map_check_subcommand_with_mapped_smirks(self, monkeypatch, cli_available):
        """The CLI must be called as ``atom-map check --json {"smirks": <mapped>}``."""
        runner = _RecordingRunner(_cli_check_response())
        monkeypatch.setattr("mechanistic_agent.tools.subprocess.run", runner)

        validate_atom_mapping_via_rdkit(
            starting_materials=SAMPLE_REACTANTS,
            products=SAMPLE_PRODUCTS,
            mapped_atoms=SAMPLE_MAPPED_ATOMS,
        )

        assert len(runner.calls) == 1
        argv = runner.calls[0]
        assert argv[:3] == ["rdkit-agent", "atom-map", "check"]
        assert argv[3] == "--json"
        assert len(argv) == 5
        payload = json.loads(argv[4])
        assert payload == {"smirks": EXPECTED_SMIRKS}

    def test_result_records_mapped_smirks_and_pair_counts(self, monkeypatch, cli_available):
        monkeypatch.setattr(
            "mechanistic_agent.tools.subprocess.run", _RecordingRunner(_cli_check_response())
        )
        result = validate_atom_mapping_via_rdkit(
            starting_materials=SAMPLE_REACTANTS,
            products=SAMPLE_PRODUCTS,
            mapped_atoms=SAMPLE_MAPPED_ATOMS,
        )
        assert result is not None
        checks = _checks_by_name(result)
        assert checks["atom_map_pairs_resolved"].details["pairs_total"] == 3
        assert checks["atom_map_pairs_resolved"].details["pairs_used"] == 3
        assert checks["atom_map_pairs_resolved"].details["pairs_dropped"] == 0
        assert checks["atom_map_check"].details["mapped_smirks"] == EXPECTED_SMIRKS


# ---------------------------------------------------------------------------
# Tests: interpreting the CLI verdict
# ---------------------------------------------------------------------------

class TestCliVerdict:

    def test_passes_when_valid_and_balanced(self, monkeypatch, cli_available):
        monkeypatch.setattr(
            "mechanistic_agent.tools.subprocess.run", _RecordingRunner(_cli_check_response())
        )
        result = validate_atom_mapping_via_rdkit(
            starting_materials=SAMPLE_REACTANTS,
            products=SAMPLE_PRODUCTS,
            mapped_atoms=SAMPLE_MAPPED_ATOMS,
        )
        assert result is not None
        assert result.passed is True
        checks = _checks_by_name(result)
        assert set(checks) == {"atom_map_pairs_resolved", "atom_map_check"}
        assert checks["atom_map_check"].details["valid"] is True
        assert checks["atom_map_check"].details["balanced"] is True

    def test_fails_when_map_numbers_are_unbalanced(self, monkeypatch, cli_available):
        monkeypatch.setattr(
            "mechanistic_agent.tools.subprocess.run",
            _RecordingRunner(_cli_check_response(
                balanced=False,
                map_numbers_only_in_reactants=[3],
                map_numbers_only_in_products=[4],
            )),
        )
        result = validate_atom_mapping_via_rdkit(
            starting_materials=SAMPLE_REACTANTS,
            products=SAMPLE_PRODUCTS,
            mapped_atoms=SAMPLE_MAPPED_ATOMS,
        )
        assert result is not None
        assert result.passed is False
        check = _checks_by_name(result)["atom_map_check"]
        assert check.passed is False
        assert check.details["map_numbers_only_in_reactants"] == [3]
        assert check.details["map_numbers_only_in_products"] == [4]

    def test_fails_when_smirks_is_invalid(self, monkeypatch, cli_available):
        monkeypatch.setattr(
            "mechanistic_agent.tools.subprocess.run",
            _RecordingRunner(_cli_check_response(valid=False)),
        )
        result = validate_atom_mapping_via_rdkit(
            starting_materials=SAMPLE_REACTANTS,
            products=SAMPLE_PRODUCTS,
            mapped_atoms=SAMPLE_MAPPED_ATOMS,
        )
        assert result is not None
        assert result.passed is False
        assert _checks_by_name(result)["atom_map_check"].details["valid"] is False

    def test_unexpected_response_shape_is_a_failure_not_a_skip(self, monkeypatch, cli_available):
        """A CLI reply without ``valid``/``balanced`` means the contract drifted; fail loudly."""
        monkeypatch.setattr(
            "mechanistic_agent.tools.subprocess.run",
            _RecordingRunner(json.dumps({"status": "unknown_format"})),
        )
        result = validate_atom_mapping_via_rdkit(
            starting_materials=SAMPLE_REACTANTS,
            products=SAMPLE_PRODUCTS,
            mapped_atoms=SAMPLE_MAPPED_ATOMS,
        )
        assert result is not None
        assert result.passed is False
        check = _checks_by_name(result)["atom_map_check"]
        assert check.details["error_code"] == "rdkit_cli_unexpected_response"
        assert check.details.get("skipped") is not True


# ---------------------------------------------------------------------------
# Tests: Python-side pair resolution (before the CLI is consulted)
# ---------------------------------------------------------------------------

class TestPairResolution:

    def test_element_mismatch_pair_fails_validation(self, monkeypatch, cli_available):
        """A pair mapping O onto C cannot be rendered; the mapping fails even if the CLI is happy."""
        runner = _RecordingRunner(_cli_check_response(mapped_atoms=2, unmapped_atoms=3))
        monkeypatch.setattr("mechanistic_agent.tools.subprocess.run", runner)
        result = validate_atom_mapping_via_rdkit(
            starting_materials=SAMPLE_REACTANTS,
            products=SAMPLE_PRODUCTS,
            mapped_atoms=MISMATCHED_MAPPED_ATOMS,
        )
        assert result is not None
        assert result.passed is False
        pairs = _checks_by_name(result)["atom_map_pairs_resolved"]
        assert pairs.passed is False
        assert pairs.details["pairs_total"] == 3
        assert pairs.details["pairs_used"] == 2
        assert pairs.details["pairs_dropped"] == 1
        assert pairs.details["error_code"] == "unresolvable_mapping_pairs"
        # The CLI is still consulted for the pairs that did resolve.
        assert len(runner.calls) == 1
        payload = json.loads(runner.calls[0][4])
        assert ":1]" in payload["smirks"] and ":3]" not in payload["smirks"]

    def test_no_resolvable_pairs_fails_without_calling_cli(self, monkeypatch, cli_available):
        def _should_not_run(*a, **kw):  # pragma: no cover - asserts non-invocation
            raise AssertionError("rdkit-agent must not be invoked when no pair resolves")

        monkeypatch.setattr("mechanistic_agent.tools.subprocess.run", _should_not_run)
        garbage = [{"product_atom": "CC=O#99", "source": {"molecule_index": 0, "smiles": "CCO", "atom_index": 0}}]
        result = validate_atom_mapping_via_rdkit(
            starting_materials=SAMPLE_REACTANTS,
            products=SAMPLE_PRODUCTS,
            mapped_atoms=garbage,
        )
        assert result is not None
        assert result.passed is False
        checks = _checks_by_name(result)
        assert checks["atom_map_pairs_resolved"].details["pairs_used"] == 0
        assert "atom_map_check" not in checks

    def test_skips_when_no_mapped_atoms(self):
        """Empty/None mapped_atoms is a trivial pass: nothing to validate."""
        for value in (None, []):
            result = validate_atom_mapping_via_rdkit(
                starting_materials=SAMPLE_REACTANTS,
                products=SAMPLE_PRODUCTS,
                mapped_atoms=value,
            )
            assert result is not None
            assert result.passed is True
            assert result.checks[0].details.get("skipped") is True


# ---------------------------------------------------------------------------
# Tests: tool missing vs tool rejected input
# ---------------------------------------------------------------------------

class TestToolAvailability:

    def test_returns_none_and_warns_when_tool_missing(self, monkeypatch, caplog):
        monkeypatch.setattr(
            "mechanistic_agent.tools.resolve_rdkit_cli_command",
            lambda _cfg: _fake_resolution(False),
        )
        with caplog.at_level(logging.WARNING, logger="mechanistic_agent.tools"):
            result = validate_atom_mapping_via_rdkit(
                starting_materials=SAMPLE_REACTANTS,
                products=SAMPLE_PRODUCTS,
                mapped_atoms=SAMPLE_MAPPED_ATOMS,
            )
        assert result is None
        assert any("unavailable" in rec.getMessage() for rec in caplog.records)

    def test_returns_none_on_subprocess_timeout(self, monkeypatch, cli_available):
        def _raise_timeout(*a, **kw):
            raise subprocess.TimeoutExpired(cmd="rdkit-agent", timeout=5)

        monkeypatch.setattr("mechanistic_agent.tools.subprocess.run", _raise_timeout)
        result = validate_atom_mapping_via_rdkit(
            starting_materials=SAMPLE_REACTANTS,
            products=SAMPLE_PRODUCTS,
            mapped_atoms=SAMPLE_MAPPED_ATOMS,
        )
        assert result is None

    def test_usage_error_is_reported_as_failed_check(self, monkeypatch, cli_available, caplog):
        """Exit 2 with a usage error is a malformed invocation, never a silent skip."""
        runner = _RecordingRunner(
            json.dumps({"error": "No sub-command provided. Use: atom-map add | remove | check | list"}),
            returncode=2,
        )
        monkeypatch.setattr("mechanistic_agent.tools.subprocess.run", runner)
        with caplog.at_level(logging.ERROR, logger="mechanistic_agent.tools"):
            result = validate_atom_mapping_via_rdkit(
                starting_materials=SAMPLE_REACTANTS,
                products=SAMPLE_PRODUCTS,
                mapped_atoms=SAMPLE_MAPPED_ATOMS,
            )
        assert result is not None
        assert result.passed is False
        check = _checks_by_name(result)["atom_map_check"]
        assert check.details["error_code"] == "rdkit_cli_rejected_input"
        assert check.details["returncode"] == 2
        assert "No sub-command provided" in check.details["error"]
        assert check.details.get("skipped") is not True
        assert any("rejected" in rec.getMessage() for rec in caplog.records)

    def test_rdkit_error_exit_is_reported_as_failed_check(self, monkeypatch, cli_available):
        runner = _RecordingRunner(json.dumps({"error": "WASM not loaded"}), returncode=3)
        monkeypatch.setattr("mechanistic_agent.tools.subprocess.run", runner)
        result = validate_atom_mapping_via_rdkit(
            starting_materials=SAMPLE_REACTANTS,
            products=SAMPLE_PRODUCTS,
            mapped_atoms=SAMPLE_MAPPED_ATOMS,
        )
        assert result is not None
        assert result.passed is False
        check = _checks_by_name(result)["atom_map_check"]
        assert check.details["error_code"] == "rdkit_cli_error"
        assert check.details["returncode"] == 3


# ---------------------------------------------------------------------------
# Tests: live CLI regression guard (skipped when rdkit-agent is not resolvable)
# ---------------------------------------------------------------------------

def _live_cli_resolvable() -> bool:
    cfg = ChemistryBackendConfig.from_config({"chemistry_backend": "rdkit_cli"})
    return bool(resolve_rdkit_cli_command(cfg).command_parts)


@pytest.mark.skipif(not _live_cli_resolvable(), reason="rdkit-agent CLI not resolvable")
class TestLiveCli:

    def test_real_cli_accepts_invocation_and_passes_good_mapping(self):
        result = validate_atom_mapping_via_rdkit(
            starting_materials=SAMPLE_REACTANTS,
            products=SAMPLE_PRODUCTS,
            mapped_atoms=SAMPLE_MAPPED_ATOMS,
        )
        assert result is not None, "validation must execute when the CLI is installed"
        checks = _checks_by_name(result)
        assert "error_code" not in checks["atom_map_check"].details
        assert result.passed is True

    def test_real_cli_flags_element_mismatch(self):
        result = validate_atom_mapping_via_rdkit(
            starting_materials=SAMPLE_REACTANTS,
            products=SAMPLE_PRODUCTS,
            mapped_atoms=MISMATCHED_MAPPED_ATOMS,
        )
        assert result is not None
        assert result.passed is False


# ---------------------------------------------------------------------------
# Tests: MappingAgent integration
# ---------------------------------------------------------------------------

class TestMappingAgentValidation:

    def _make_state(self):
        from types import SimpleNamespace
        return SimpleNamespace(
            run_input=SimpleNamespace(starting_materials=SAMPLE_REACTANTS, products=SAMPLE_PRODUCTS),
            run_config=SimpleNamespace(step_models={}, model="test-model"),
        )

    def _output(self, confidence: float = 0.9) -> Dict[str, Any]:
        return {"llm_response": {"mapped_atoms": SAMPLE_MAPPED_ATOMS, "confidence": confidence}}

    def test_run_attaches_validation(self, monkeypatch):
        from mechanistic_agent.core.subagents import MappingAgent

        mock_executor = MagicMock()
        mock_executor.run_mapping.return_value = self._output()
        validation_result = StepValidationResult(checks=[])
        monkeypatch.setattr(
            "mechanistic_agent.core.subagents.MappingAgent._validate_mapping",
            staticmethod(lambda output, **kw: validation_result),
        )
        step = MappingAgent(executor=mock_executor).run(self._make_state())
        assert step.validation is validation_result
        assert step.step_name == "atom_mapping"

    def test_confidence_clamped_on_failure(self, monkeypatch):
        from mechanistic_agent.core.subagents import MappingAgent

        fake_output = self._output(0.9)
        mock_executor = MagicMock()
        mock_executor.run_mapping.return_value = fake_output
        failing = StepValidationResult(checks=[
            StepValidationCheck(name="atom_map_check", passed=False, details={"balanced": False}),
        ])
        monkeypatch.setattr("mechanistic_agent.tools.validate_atom_mapping_via_rdkit", lambda **kw: failing)

        step = MappingAgent(executor=mock_executor).run(self._make_state())
        assert step.validation is failing
        assert step.validation.passed is False
        assert fake_output["llm_response"]["confidence"] <= 0.3
        assert fake_output["atom_map_validation"]["passed"] is False

    def test_confidence_untouched_on_pass(self, monkeypatch):
        from mechanistic_agent.core.subagents import MappingAgent

        fake_output = self._output(0.9)
        mock_executor = MagicMock()
        mock_executor.run_mapping.return_value = fake_output
        passing = StepValidationResult(checks=[
            StepValidationCheck(name="atom_map_check", passed=True, details={"balanced": True}),
        ])
        monkeypatch.setattr("mechanistic_agent.tools.validate_atom_mapping_via_rdkit", lambda **kw: passing)

        MappingAgent(executor=mock_executor).run(self._make_state())
        assert fake_output["llm_response"]["confidence"] == 0.9
        assert fake_output["atom_map_validation"]["passed"] is True

    def test_unavailable_tool_leaves_confidence_and_marks_skip(self, monkeypatch):
        """When the CLI is missing, the output says so instead of silently omitting validation."""
        from mechanistic_agent.core.subagents import MappingAgent

        fake_output = self._output(0.9)
        mock_executor = MagicMock()
        mock_executor.run_mapping.return_value = fake_output
        monkeypatch.setattr("mechanistic_agent.tools.validate_atom_mapping_via_rdkit", lambda **kw: None)

        step = MappingAgent(executor=mock_executor).run(self._make_state())
        assert step.validation is None
        assert fake_output["llm_response"]["confidence"] == 0.9
        marker = fake_output["atom_map_validation"]
        assert marker["skipped"] is True
        assert marker["reason"] == "rdkit_cli_unavailable"

    def test_run_step_mapping_attaches_validation(self, monkeypatch):
        from mechanistic_agent.core.subagents import MappingAgent

        mock_executor = MagicMock()
        mock_executor.run_step_mapping.return_value = self._output(0.8)
        validation_result = StepValidationResult(checks=[])
        monkeypatch.setattr(
            "mechanistic_agent.core.subagents.MappingAgent._validate_mapping",
            staticmethod(lambda output, **kw: validation_result),
        )
        step = MappingAgent(executor=mock_executor).run_step_mapping(
            self._make_state(), current_state=["CCO"], resulting_state=["CC=O"],
        )
        assert step.validation is validation_result
        assert step.step_name == "step_atom_mapping"
