from __future__ import annotations

import json
import threading
from typing import Any, Dict, List, Optional

import pytest

from mechanistic_agent.balance import assess_balance_diagnostics
from mechanistic_agent.core.coordinator import RunCoordinator, _RunPaused
from mechanistic_agent.core.types import (
    RunConfig,
    RunInput,
    RunState,
    StepResult,
    StepValidationCheck,
    StepValidationResult,
)
from mechanistic_agent.core.validators import validate_mechanism_step_output
from mechanistic_agent.tools import predict_missing_reagents_for_candidate, validate_proposed_reagents
from skills.mechanistic.atom_balance_validation.validator import validate_atom_balance


class _MemoryStore:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.step_outputs: List[Dict[str, Any]] = []

    def append_event(
        self,
        run_id: str,
        event_type: str,
        payload: Dict[str, Any],
        *,
        step_name: Optional[str] = None,
    ) -> None:
        self.events.append(
            {
                "run_id": run_id,
                "event_type": event_type,
                "payload": payload,
                "step_name": step_name,
            }
        )

    def create_run_pause(self, *, run_id: str, reason: str, details: Dict[str, Any]) -> str:
        self.events.append(
            {
                "run_id": run_id,
                "event_type": "run_pause_created",
                "payload": {"reason": reason, "details": details},
                "step_name": None,
            }
        )
        return "pause-id"

    def set_run_status(self, run_id: str, status: str) -> None:
        self.events.append(
            {
                "run_id": run_id,
                "event_type": "status_update",
                "payload": {"status": status},
                "step_name": None,
            }
        )

    def record_step_output(self, **kwargs: Any) -> None:
        self.step_outputs.append(dict(kwargs))

    def upsert_step_output(self, **kwargs: Any) -> None:
        for idx, row in enumerate(self.step_outputs):
            if (
                row.get("run_id") == kwargs.get("run_id")
                and row.get("step_name") == kwargs.get("step_name")
                and int(row.get("attempt") or 0) == int(kwargs.get("attempt") or 0)
                and int(row.get("retry_index") or 0) == int(kwargs.get("retry_index") or 0)
            ):
                self.step_outputs[idx] = dict(row) | dict(kwargs)
                return
        self.step_outputs.append(dict(kwargs))

    def add_trace_record(self, *args: Any, **kwargs: Any) -> None:
        return

    def resolve_run_step_prompt_id(self, *args: Any, **kwargs: Any) -> Optional[str]:
        return None

    def upsert_model_version(self, *args: Any, **kwargs: Any) -> str:
        return "model-version"

    def list_step_outputs(self, run_id: str) -> List[Dict[str, Any]]:
        return [row for row in self.step_outputs if row.get("run_id") == run_id]

    def record_arrow_push_annotation(self, **kwargs: Any) -> str:
        return "arrow-id"


def _state(
    *,
    starting_materials: Optional[List[str]] = None,
    products: Optional[List[str]] = None,
    mode: str = "unverified",
) -> RunState:
    run_input = RunInput(
        starting_materials=starting_materials or ["CCBr", "[Cl-]"],
        products=products or ["CCCl", "[Br-]"],
        ph=7.0,
        temperature_celsius=25.0,
    )
    run_config = RunConfig(
        model="gpt-4",
        model_family="openai",
        max_steps=1,
        max_runtime_seconds=0.2,
        intermediate_prediction_enabled=True,
        retry_same_candidate_max=1,
        reproposal_on_repeat_failure=True,
        step_mapping_enabled=False,
        dbe_policy="soft",
    )
    state = RunState(run_id="run-test-id", mode=mode, run_input=run_input, run_config=run_config)
    state.initialise()
    return state


def test_assess_balance_diagnostics_records_repairs(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_canonicalise(smiles: str, *, backend_config: Any = None):  # noqa: ANN001
        if smiles == "broken-solvent":
            return "ClCCl", {"repair_smiles_applied": True, "validated_from": "rdkit_cli:repair-smiles"}
        return smiles, {"validated_from": smiles}

    monkeypatch.setattr("mechanistic_agent.tools._canonicalise_candidate_smiles", _fake_canonicalise)

    diagnostics = assess_balance_diagnostics(
        ["broken-solvent"],
        ["ClCCl"],
        include_hydrogens=True,
    )
    assert diagnostics["classification"] == "exact"
    assert diagnostics["invalid_species"] == []
    assert diagnostics["repaired_species"][0]["after"] == "ClCCl"


def test_validate_atom_balance_reports_structured_invalid_species(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_canonicalise(smiles: str, *, backend_config: Any = None):  # noqa: ANN001
        if smiles == "bad-smiles":
            return None, {"error": "still invalid after repair", "error_code": "invalid_species"}
        return smiles, {"validated_from": smiles}

    monkeypatch.setattr("mechanistic_agent.tools._canonicalise_candidate_smiles", _fake_canonicalise)

    result = validate_atom_balance(["CCO"], ["bad-smiles"])
    assert result.passed is False
    assert result.details["classification"] == "invalid_species"
    assert result.details["invalid_species"][0]["species"] == "bad-smiles"
    assert "error" in result.details


def test_validate_mechanism_step_output_marks_rdkit_invalid_species_as_soft_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_atom_balance(_current_state: List[str], _resulting_state: List[str]) -> StepValidationCheck:
        return StepValidationCheck(
            name="atom_balance",
            passed=True,
            details={"classification": "exact", "balanced": True},
        )

    def _fake_execute_chemistry_check(**kwargs: Any):  # noqa: ANN401
        python_result = kwargs["python_callable"]()
        return python_result, {
            "backend_requested": "rdkit_cli",
            "backend_used": "rdkit_cli",
            "fallback_used": False,
            "fallback_reason": None,
            "rdkit_cli_available": True,
            "rdkit_cli_command": "rdkit_cli",
            "rdkit_cli_error_code": "atom_balance_invalid_species",
            "rdkit_cli_error": "invalid species reported by rdkit_cli",
            "rdkit_cli_failed_check_names": ["atom_balance"],
            "parity": None,
        }

    monkeypatch.setattr("mechanistic_agent.core.validators.validate_atom_balance", _fake_atom_balance)
    monkeypatch.setattr(
        "mechanistic_agent.core.validators.execute_chemistry_check",
        _fake_execute_chemistry_check,
    )

    result = validate_mechanism_step_output(
        {
            "current_state": ["CCBr", "[Cl-]"],
            "resulting_state": ["CCCl", "[Br-]"],
            "reaction_smirks": "[C:1][Br:2].[Cl-:3]>>[C:1][Cl:3].[Br-:2]",
            "dbe": "",
        },
        enabled_validators={"atom_balance_validation"},
        run_config={"chemistry_backend": "rdkit_cli"},
    )

    assert result.passed is True
    atom_check = result.checks[0]
    assert atom_check.name == "atom_balance"
    assert atom_check.details["warning_only"] is True
    assert atom_check.details["known_soft_pass"] is True
    assert atom_check.details["soft_pass_reason"] == "rdkit_cli_invalid_species"
    assert atom_check.details["retry_recommended"] is True
    assert atom_check.details["chemistry_backend"]["rdkit_cli_error_code"] == "atom_balance_invalid_species"
    assert "known soft pass" in atom_check.details["warnings"][0]


def test_record_validation_checks_emits_soft_backend_warning_event() -> None:
    store = _MemoryStore()
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]
    state = _state()
    validation_result = StepValidationResult(
        checks=[
            StepValidationCheck(
                name="atom_balance",
                passed=True,
                details={
                    "warning_only": True,
                    "known_soft_pass": True,
                    "soft_pass_reason": "rdkit_cli_invalid_species",
                    "retry_recommended": True,
                    "warnings": [
                        "rdkit_cli reported invalid species; treating as a known soft pass because Python atom-balance validation succeeded."
                    ],
                    "chemistry_backend": {
                        "backend_requested": "rdkit_cli",
                        "backend_used": "rdkit_cli",
                        "fallback_used": False,
                        "fallback_reason": None,
                        "rdkit_cli_error_code": "atom_balance_invalid_species",
                        "rdkit_cli_error": "invalid species reported by rdkit_cli",
                    },
                },
            )
        ]
    )

    coordinator._record_validation_checks(
        state,
        validation_result=validation_result,
        attempt=1,
        retry_index=0,
    )

    event_types = [ev["event_type"] for ev in store.events]
    assert "chemistry_backend_soft_warning" in event_types
    soft_warning = next(ev for ev in store.events if ev["event_type"] == "chemistry_backend_soft_warning")
    assert soft_warning["payload"]["soft_pass_reason"] == "rdkit_cli_invalid_species"
    assert soft_warning["payload"]["retry_recommended"] is True
    assert soft_warning["payload"]["known_soft_pass"] is True


def test_validate_proposed_reagents_and_validator_share_balance_semantics() -> None:
    validator = validate_atom_balance(["CCBr", "[Cl-]"], ["CCCl", "[Br-]"])
    proposed = json.loads(
        validate_proposed_reagents(["[Cl-]"], ["[Br-]"], ["CCBr"], ["CCCl"])
    )
    assert validator.passed is True
    assert proposed["is_balanced"] is True
    assert proposed["balance_diagnostics"]["classification"] == "exact"


def test_candidate_rescue_repairs_invalid_species_before_delegating(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def _fake_canonicalise(smiles: str, *, backend_config: Any = None):  # noqa: ANN001
        if smiles == "broken-solvent":
            return "ClCCl", {"repair_smiles_applied": True, "validated_from": "rdkit_cli:repair-smiles"}
        return smiles, {"validated_from": smiles}

    def _fake_missing_reagents(*, starting_materials: List[str], products: List[str], conditions_guidance: Optional[str] = None) -> str:
        captured["starting_materials"] = list(starting_materials)
        captured["products"] = list(products)
        captured["conditions_guidance"] = json.loads(conditions_guidance or "{}")
        return json.dumps({"status": "success", "missing_reactants": [], "missing_products": []})

    monkeypatch.setattr("mechanistic_agent.tools._canonicalise_candidate_smiles", _fake_canonicalise)
    monkeypatch.setattr("mechanistic_agent.tools.predict_missing_reagents", _fake_missing_reagents)

    raw = predict_missing_reagents_for_candidate(
        current_state=["broken-solvent"],
        resulting_state=["ClCCl", "O"],
        failed_checks=["atom_balance"],
        validation_details={"passed": False},
    )
    payload = json.loads(raw)
    assert payload["status"] == "success"
    assert captured["starting_materials"] == ["ClCCl"]
    assert captured["products"] == ["ClCCl", "O"]
    assert payload["balance_diagnostics"]["repaired_species"][0]["before"] == "broken-solvent"


def test_unverified_balance_pending_soft_advance_records_reason() -> None:
    store = _MemoryStore()
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]
    state = _state(mode="unverified")

    class _IntermediateAgent:
        def run(self, _state: RunState, template_guidance: Optional[Dict[str, Any]] = None) -> StepResult:
            return StepResult(
                step_name="mechanism_step_proposal",
                tool_name="propose_mechanism_step",
                output={
                    "classification": "intermediate_step",
                    "candidates": [
                        {
                            "rank": 1,
                            "intermediate_smiles": "CCCl",
                            "resulting_state": ["CCCl", "[Br-]", "O"],
                        }
                    ],
                },
                source="llm",
            )

    coordinator.intermediate_agent = _IntermediateAgent()  # type: ignore[assignment]

    def _failed_balance_pending(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        return {
            "status": "failed",
            "last_validation": {
                "passed": False,
                "checks": [
                    {"name": "atom_balance", "passed": False, "details": {"classification": "imbalanced", "deficit": {"O": 1}}},
                    {"name": "state_progress", "passed": True, "details": {}},
                ],
            },
            "failed_checks": ["atom_balance"],
            "validation_signature": "atom-balance",
            "candidate_rank": 1,
            "rescue_attempted": True,
            "rescue_outcome": "no_changes",
            "mechanism_output": {
                "current_state": ["CCBr", "[Cl-]"],
                "resulting_state": ["CCCl", "[Br-]", "O"],
                "contains_target_product": True,
            },
        }

    coordinator._try_candidate_with_retries = _failed_balance_pending  # type: ignore[method-assign]
    coordinator._run_mechanism_loop(state, threading.Event())

    soft_events = [ev for ev in store.events if ev["event_type"] == "mechanism_step_soft_advance"]
    assert soft_events
    assert soft_events[-1]["payload"]["reason"] == "balance_pending"
    assert state.current_state == ["CCCl", "[Br-]", "O"]


def test_verified_mode_does_not_soft_advance_balance_pending() -> None:
    store = _MemoryStore()
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]
    state = _state(mode="verified")

    class _IntermediateAgent:
        def run(self, _state: RunState, template_guidance: Optional[Dict[str, Any]] = None) -> StepResult:
            return StepResult(
                step_name="mechanism_step_proposal",
                tool_name="propose_mechanism_step",
                output={
                    "classification": "intermediate_step",
                    "candidates": [
                        {
                            "rank": 1,
                            "intermediate_smiles": "CCCl",
                            "resulting_state": ["CCCl", "[Br-]", "O"],
                        }
                    ],
                },
                source="llm",
            )

    coordinator.intermediate_agent = _IntermediateAgent()  # type: ignore[assignment]

    def _failed_balance_pending(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        return {
            "status": "failed",
            "last_validation": {
                "passed": False,
                "checks": [
                    {"name": "atom_balance", "passed": False, "details": {"classification": "imbalanced", "deficit": {"O": 1}}},
                    {"name": "state_progress", "passed": True, "details": {}},
                ],
            },
            "failed_checks": ["atom_balance"],
            "validation_signature": "atom-balance",
            "candidate_rank": 1,
            "rescue_attempted": True,
            "rescue_outcome": "no_changes",
            "mechanism_output": {
                "current_state": ["CCBr", "[Cl-]"],
                "resulting_state": ["CCCl", "[Br-]", "O"],
                "contains_target_product": True,
            },
        }

    coordinator._try_candidate_with_retries = _failed_balance_pending  # type: ignore[method-assign]
    with pytest.raises(_RunPaused):
        coordinator._run_mechanism_loop(state, threading.Event())

    soft_events = [ev for ev in store.events if ev["event_type"] == "mechanism_step_soft_advance"]
    assert soft_events == []


def test_overall_balance_reconciliation_emits_reconciled_payload() -> None:
    store = _MemoryStore()
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]
    state = _state(
        starting_materials=["N#CO", "O"],
        products=["N#CO"],
        mode="unverified",
    )
    state.current_state = ["N#CO"]

    coordinator._record_step(
        state,
        StepResult(
            step_name="mechanism_synthesis",
            tool_name="predict_mechanistic_step",
                output={
                    "current_state": ["N#CO", "O"],
                    "resulting_state": ["N#CO"],
                    "contains_target_product": True,
                    "soft_advance": True,
                    "soft_advance_reason": "balance_pending",
                },
            attempt=1,
            retry_index=0,
            source="deterministic",
        ),
    )

    class _Executor:
        @staticmethod
        def run_missing_reagents(*, starting: List[str], products: List[str], conditions_guidance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            assert starting == ["N#CO", "O"]
            assert products == ["N#CO"]
            return {
                "status": "success",
                "missing_reactants": [],
                "missing_products": ["O"],
            }

    coordinator.missing_reagents_agent.executor = _Executor()  # type: ignore[assignment]
    coordinator._run_overall_balance_reconciliation(state)

    reconciliation_events = [ev for ev in store.events if ev["event_type"] == "overall_balance_reconciled"]
    assert reconciliation_events
    payload = reconciliation_events[-1]["payload"]
    assert payload["grade"] == "reconciled"
    assert payload["final_balance"]["balanced"] is True
    assert payload["add_products"] == ["O"]
