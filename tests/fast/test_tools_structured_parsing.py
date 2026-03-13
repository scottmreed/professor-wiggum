"""Regression tests for tool-response parsing and SMILES token filtering."""

import json
from typing import Any, Dict, List
import pytest
from pydantic import ValidationError

from mechanistic_agent.core.tool_executor import ToolExecutor
from mechanistic_agent.tools import (
    AtomMappingPayload,
    MechanismIntermediate,
    MechanismStepCandidate,
    MechanismStepPrediction,
    MissingReagentsPayload,
    RdkitCliCommandSpec,
    ToolRuntimeError,
    _apply_smiles_correction,
    _canonicalise_candidate_smiles,
    _canonicalise_candidate_smiles_python,
    _execute_rdkit_cli_plan_for_candidate,
    _looks_like_smiles,
    _mol_from_smiles,
    attempt_atom_mapping_for_step,
    predict_missing_reagents,
    predict_missing_reagents_for_candidate,
    propose_intermediates,
    select_reaction_type,
)


def test_looks_like_smiles_rejects_natural_language_descriptors() -> None:
    assert not _looks_like_smiles("acid-catalyzed")
    assert not _looks_like_smiles("basic")
    assert not _looks_like_smiles("reaction-step")


def test_looks_like_smiles_rejects_diels_alder_notation() -> None:
    # Pericyclic electron-count notation must never be treated as SMILES.
    assert not _looks_like_smiles("[4+2]")
    assert not _looks_like_smiles("[2+2]")
    assert not _looks_like_smiles("[3+3]")
    # Parenthesised role descriptors must be rejected.
    assert not _looks_like_smiles("(diene)")
    assert not _looks_like_smiles("(dienophile)")
    assert not _looks_like_smiles("(solvent)")


def test_looks_like_smiles_accepts_valid_tokens() -> None:
    assert _looks_like_smiles("C1=CC=CC=C1")
    assert _looks_like_smiles("[OH-]")
    assert _looks_like_smiles("[NH4+]")
    assert _looks_like_smiles("[C@@H]")
    assert _looks_like_smiles("C-C")
    # Short branch groups are valid SMILES, not descriptive words.
    assert _looks_like_smiles("C1(CC1)")


def test_mol_from_smiles_rejects_descriptor_tokens_before_rdkit_parse() -> None:
    with pytest.raises(ToolRuntimeError):
        _mol_from_smiles("acid-catalyzed")


def test_mechanism_step_prediction_accepts_candidates_schema() -> None:
    payload = {
        "classification": "intermediate_step",
        "analysis": "Nucleophilic attack forms a tetrahedral intermediate.",
        "candidates": [
            {
                "rank": 1,
                "intermediate_smiles": "CCO",
                "reaction_description": "Attack at carbonyl carbon.",
                "reaction_smirks": "[CH2:1]=[O:2]>>[CH3:1][O:2] |mech:v1;pi:1-2>2|",
                "electron_pushes": [{"kind": "pi_bond", "source_bond": ["1", "2"], "through_atom": "2", "target_atom": "2", "electrons": 2}],
                "confidence": "medium",
            }
        ],
    }

    parsed = MechanismStepPrediction.model_validate(payload)
    assert parsed.candidates
    assert parsed.candidates[0].intermediate_smiles == "CCO"


def test_mechanism_step_prediction_accepts_legacy_intermediates_schema() -> None:
    payload = {
        "classification": "intermediate_step",
        "analysis": "Legacy format remains valid.",
        "intermediates": [{"smiles": "CCO", "type": "legacy"}],
    }

    parsed = MechanismStepPrediction.model_validate(payload)
    assert parsed.intermediates
    assert parsed.intermediates[0].smiles == "CCO"


def test_atom_mapping_payload_coerces_legacy_string_confidence() -> None:
    payload = AtomMappingPayload.model_validate(
        {
            "mapped_atoms": [],
            "unmapped_atoms": [],
            "confidence": "medium",
            "reasoning": "mapping is plausible",
        }
    )
    assert payload.confidence == 0.6


def test_attempt_atom_mapping_for_step_emits_numeric_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mechanistic_agent.tools.attempt_atom_mapping",
        lambda *_args, **_kwargs: json.dumps(
            {
                "status": "success",
                "llm_response": {
                    "confidence": "high",
                    "mapped_atoms": [],
                    "unmapped_atoms": [],
                },
            }
        ),
    )
    raw = attempt_atom_mapping_for_step(current_state=["CCO"], resulting_state=["CCO"])
    payload = json.loads(raw)
    assert payload["confidence"] == 0.9
    assert payload["raw_confidence"] == "high"


# ---------------------------------------------------------------------------
# SMILES correction map tests
# ---------------------------------------------------------------------------


def test_common_smiles_corrections_h2o() -> None:
    """[H2O] should be auto-corrected to 'O'."""
    corrected, was_corrected = _apply_smiles_correction("[H2O]")
    assert was_corrected
    assert corrected == "O"


def test_common_smiles_corrections_h2so4() -> None:
    corrected, was_corrected = _apply_smiles_correction("[H2SO4]")
    assert was_corrected
    assert corrected == "OS(=O)(=O)O"


def test_common_smiles_corrections_case_insensitive() -> None:
    corrected, was_corrected = _apply_smiles_correction("[h2o]")
    assert was_corrected
    assert corrected == "O"


def test_no_correction_for_valid_smiles() -> None:
    corrected, was_corrected = _apply_smiles_correction("CCO")
    assert not was_corrected
    assert corrected == "CCO"


def test_common_smiles_corrections_ethanol() -> None:
    corrected, was_corrected = _apply_smiles_correction("EtOH")
    assert was_corrected
    assert corrected == "CCO"


def test_canonicalise_python_path_uses_repair_smiles_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stub_repair(*_args, **_kwargs):
        return "CCO", {"used": True, "response": {"canonical_smiles": "CCO"}}

    monkeypatch.setattr("mechanistic_agent.tools._attempt_repair_smiles_via_rdkit_cli", _stub_repair)
    canonical, details = _canonicalise_candidate_smiles_python("definitely_not_smiles")
    assert canonical == "CCO"
    assert details.get("repair_smiles_applied") is True


def test_canonicalise_backend_path_uses_repair_smiles_on_cli_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stub_backend(*, mode, payload, config, python_callable, cli_to_result, python_signature, cli_signature):
        _ = (mode, payload, config, python_callable, python_signature, cli_signature)
        cli_output = {
            "overall_pass": False,
            "summary": "Validation failed: rdkit_parse",
            "checks": [],
            "failed_checks": [{"error_code": "smiles_rdkit_parse_failed", "message": "RDKit parse failed"}],
            "failed_check_names": ["rdkit_parse"],
            "fix_suggestions": [],
        }
        return cli_to_result(cli_output), {"backend_used": "rdkit_cli", "fallback_used": False}

    def _stub_repair(*_args, **_kwargs):
        return "O", {"used": True, "response": {"canonical_smiles": "O"}}

    monkeypatch.setattr("mechanistic_agent.tools.execute_chemistry_check", _stub_backend)
    monkeypatch.setattr("mechanistic_agent.tools._attempt_repair_smiles_via_rdkit_cli", _stub_repair)

    canonical, details = _canonicalise_candidate_smiles("not_a_smiles", backend_config={"chemistry_backend": "rdkit_cli"})
    assert canonical == "O"
    assert details.get("repair_smiles_applied") is True


# ---------------------------------------------------------------------------
# Per-item filtering in MissingReagentsPayload
# ---------------------------------------------------------------------------


def test_missing_reagents_payload_corrects_h2o() -> None:
    """[H2O] should be auto-corrected to canonical water SMILES."""
    payload = MissingReagentsPayload.model_validate(
        {
            "missing_reactants": [],
            "missing_products": ["[H2O]"],
        }
    )
    assert payload.missing_products == ["O"]


def test_missing_reagents_payload_filters_invalid_smiles() -> None:
    """Invalid SMILES items should be silently dropped, not reject the whole payload."""
    payload = MissingReagentsPayload.model_validate(
        {
            "missing_reactants": ["O", "definitely-not-smiles"],
            "missing_products": ["CCO"],
        }
    )
    assert "O" in payload.missing_reactants
    assert "definitely-not-smiles" not in payload.missing_reactants
    assert payload.missing_products == ["CCO"]


def test_missing_reagents_payload_does_not_raise_on_all_invalid() -> None:
    """Even when all items are invalid, the payload should not raise — just return empty lists."""
    payload = MissingReagentsPayload.model_validate(
        {
            "missing_reactants": ["not-a-molecule"],
            "missing_products": ["also-not-valid"],
        }
    )
    assert payload.missing_reactants == []
    assert payload.missing_products == []


def test_missing_reagents_payload_accepts_cli_command_plan() -> None:
    payload = MissingReagentsPayload.model_validate(
        {
            "missing_reactants": ["Cl"],
            "missing_products": ["O"],
            "rdkit_cli_commands": [
                {
                    "command": "balance",
                    "args": {"reactants": ["CCO", "Cl"], "products": ["CCCl", "O"]},
                    "run_on": "retry",
                    "reason": "check atom deficit on retry",
                }
            ],
        }
    )
    assert payload.rdkit_cli_commands
    assert payload.rdkit_cli_commands[0].command == "balance"
    assert payload.rdkit_cli_commands[0].run_on == "retry"


def test_missing_reagents_payload_rejects_invalid_cli_command_plan() -> None:
    with pytest.raises(ValidationError):
        MissingReagentsPayload.model_validate(
            {
                "missing_reactants": ["Cl"],
                "missing_products": ["O"],
                "rdkit_cli_commands": [
                    {
                        "command": "rm -rf /",
                        "args": {"input": "CCO"},
                    }
                ],
            }
        )


def test_rdkit_cli_command_spec_rejects_invalid_edit_operation() -> None:
    with pytest.raises(ValidationError):
        RdkitCliCommandSpec.model_validate(
            {
                "command": "edit",
                "args": {"smiles": "CCO", "operation": "arbitrary"},
                "run_on": "retry",
            }
        )


def test_invalid_cli_plan_never_executes_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = {
        "intermediate_smiles": "CCO",
        "rdkit_cli_commands": [
            {"command": "repair-smiles", "args": "not-an-object"},
        ],
    }
    called = {"count": 0}

    def _should_not_run(*_args, **_kwargs):  # noqa: ANN002, ANN003
        called["count"] += 1
        return {"status": "ok", "output": {}}

    monkeypatch.setattr("mechanistic_agent.tools._run_rdkit_cli_command", _should_not_run)
    monkeypatch.setattr("mechanistic_agent.tools._load_rdkit_cli_available_commands", lambda **_kwargs: set())
    result = _execute_rdkit_cli_plan_for_candidate(
        candidate=candidate,
        run_on="retry",
        backend_config={"chemistry_backend": "rdkit_cli"},
    )
    assert called["count"] == 0
    assert result["cli_failures"]


# ---------------------------------------------------------------------------
# Mechanism intermediate/candidate validators with correction + RDKit
# ---------------------------------------------------------------------------


def test_mechanism_step_candidate_corrects_known_errors() -> None:
    """[H2O] in intermediate_smiles should be auto-corrected to 'O'."""
    candidate = MechanismStepCandidate.model_validate(
        {
            "rank": 1,
            "intermediate_smiles": "[H2O]",
            "reaction_description": "Water formation",
            "reaction_smirks": "[CH2:1]=[O:2]>>[CH3:1][O:2] |mech:v1;pi:1-2>2|",
            "electron_pushes": [{"kind": "pi_bond", "source_bond": ["1", "2"], "through_atom": "2", "target_atom": "2", "electrons": 2}],
        }
    )
    assert candidate.intermediate_smiles == "O"


def test_mechanism_step_candidate_keeps_execution_fields() -> None:
    candidate = MechanismStepCandidate.model_validate(
        {
            "rank": 1,
            "intermediate_smiles": "CCO",
            "reaction_description": "Hydride addition",
            "reaction_smirks": "[CH2:1]=[O:2]>>[CH3:1][O:2] |mech:v1;pi:1-2>2|",
            "electron_pushes": [{"kind": "pi_bond", "source_bond": ["1", "2"], "through_atom": "2", "target_atom": "2", "electrons": 2}],
            "resulting_state": ["CCO"],
        }
    )
    assert candidate.reaction_smirks is not None
    assert candidate.electron_pushes == [{"kind": "pi_bond", "target_atom": "2", "electrons": 2, "source_bond": ["1", "2"], "through_atom": "2", "notation": "pi:1-2>2"}]
    assert candidate.resulting_state == ["CCO"]


def test_mechanism_step_candidate_accepts_cli_command_plan() -> None:
    candidate = MechanismStepCandidate.model_validate(
        {
            "rank": 1,
            "intermediate_smiles": "CCO",
            "reaction_description": "Hydride addition",
            "reaction_smirks": "[CH2:1]=[O:2]>>[CH3:1][O:2] |mech:v1;pi:1-2>2|",
            "electron_pushes": [{"kind": "pi_bond", "source_bond": ["1", "2"], "through_atom": "2", "target_atom": "2", "electrons": 2}],
            "rdkit_cli_commands": [
                {
                    "command": "repair-smiles",
                    "args": {"input": "CCO"},
                    "run_on": "retry",
                    "apply_to": "intermediate_smiles",
                }
            ],
        }
    )
    assert candidate.rdkit_cli_commands
    assert candidate.rdkit_cli_commands[0].command == "repair-smiles"
    assert candidate.rdkit_cli_commands[0].apply_to == "intermediate_smiles"


def test_mechanism_step_candidate_rejects_prose() -> None:
    """Natural-language descriptors should raise ValidationError."""
    with pytest.raises(ValidationError):
        MechanismStepCandidate.model_validate(
            {
                "rank": 1,
                "intermediate_smiles": "acid-catalyzed",
                "reaction_description": "Invalid",
            }
        )


def test_mechanism_intermediate_corrects_h2o() -> None:
    intermediate = MechanismIntermediate.model_validate(
        {"smiles": "[H2O]"}
    )
    assert intermediate.smiles == "O"


def test_mechanism_intermediate_rejects_prose() -> None:
    with pytest.raises(ValidationError):
        MechanismIntermediate.model_validate(
            {"smiles": "acid-catalyzed"}
        )


def test_propose_intermediates_repairs_candidates_missing_mech(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubResponse:
        usage = None

        def __init__(self) -> None:
            payload = {
                "classification": "intermediate_step",
                "analysis": "analysis",
                "candidates": [
                    {
                        "rank": 1,
                        "intermediate_smiles": "CCCl",
                        "reaction_description": "no mech block",
                        "reaction_smirks": "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3]",
                        "electron_pushes": [{"kind": "lone_pair", "source_atom": "4", "target_atom": "2", "electrons": 2}],
                    }
                ],
            }
            self.tool_calls = [{"arguments": json.dumps(payload)}]

    class _StubLLM:
        def invoke(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
            return _StubResponse()

    monkeypatch.setattr("mechanistic_agent.tools.adapter_supports_forced_tools", lambda _model: True)
    monkeypatch.setattr("mechanistic_agent.tools.get_model_api_key", lambda *_args, **_kwargs: "test-key")
    monkeypatch.setattr("mechanistic_agent.tools.get_chat_model", lambda *_args, **_kwargs: _StubLLM())

    raw = propose_intermediates(
        starting_materials=["CCBr", "[Cl-]"],
        products=["CCCl", "[Br-]"],
        current_state=["CCBr", "[Cl-]"],
    )
    payload = json.loads(raw)
    assert payload.get("candidates")
    repaired = payload["candidates"][0]["reaction_smirks"]
    assert "|mech:v1;" in repaired
    assert payload["candidates"][0].get("mechanism_move_repair") == "synthesized_mech_from_electron_pushes"


def test_propose_intermediates_repairs_candidates_with_invalid_mech_block(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubResponse:
        usage = None

        def __init__(self) -> None:
            payload = {
                "classification": "intermediate_step",
                "analysis": "analysis",
                "candidates": [
                    {
                        "rank": 1,
                        "intermediate_smiles": "CCCl",
                        "reaction_description": "invalid mech block",
                        "reaction_smirks": "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] |mech:v1;broken-token|",
                        "electron_pushes": [{"kind": "lone_pair", "source_atom": "4", "target_atom": "2", "electrons": 2}],
                    }
                ],
            }
            self.tool_calls = [{"arguments": json.dumps(payload)}]

    class _StubLLM:
        def invoke(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
            return _StubResponse()

    monkeypatch.setattr("mechanistic_agent.tools.adapter_supports_forced_tools", lambda _model: True)
    monkeypatch.setattr("mechanistic_agent.tools.get_model_api_key", lambda *_args, **_kwargs: "test-key")
    monkeypatch.setattr("mechanistic_agent.tools.get_chat_model", lambda *_args, **_kwargs: _StubLLM())

    raw = propose_intermediates(
        starting_materials=["CCBr", "[Cl-]"],
        products=["CCCl", "[Br-]"],
        current_state=["CCBr", "[Cl-]"],
    )
    payload = json.loads(raw)
    assert "candidates" not in payload
    assert payload["rejected_candidates"][0]["reason"] == "reaction_smirks_invalid_mech_block"


def test_propose_intermediates_marks_non_executable_candidates_unvalidated(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubResponse:
        usage = None

        def __init__(self) -> None:
            payload = {
                "classification": "intermediate_step",
                "analysis": "analysis",
                "candidates": [
                    {
                        "rank": 1,
                        "intermediate_smiles": "CCCl",
                        "reaction_description": "missing mechanism fields",
                        "reaction_smirks": "",
                        "electron_pushes": [],
                    }
                ],
            }
            self.tool_calls = [{"arguments": json.dumps(payload)}]

    class _StubLLM:
        def invoke(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
            return _StubResponse()

    monkeypatch.setattr("mechanistic_agent.tools.adapter_supports_forced_tools", lambda _model: True)
    monkeypatch.setattr("mechanistic_agent.tools.get_model_api_key", lambda *_args, **_kwargs: "test-key")
    monkeypatch.setattr("mechanistic_agent.tools.get_chat_model", lambda *_args, **_kwargs: _StubLLM())

    raw = propose_intermediates(
        starting_materials=["CCBr", "[Cl-]"],
        products=["CCCl", "[Br-]"],
        current_state=["CCBr", "[Cl-]"],
    )
    payload = json.loads(raw)
    assert payload["validation_status"] == "unvalidated"
    assert payload["has_executable_candidates"] is False
    assert payload["executable_candidate_count"] == 0
    assert payload["proposed_intermediates"] == []
    assert payload["rejected_candidates"][0]["reason"] in {
        "missing_reaction_smirks",
        "reaction_smirks_missing",
        "reaction_smirks_invalid",
    }


def test_propose_intermediates_rejects_invalid_reaction_smirks_species(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubResponse:
        usage = None

        def __init__(self) -> None:
            payload = {
                "classification": "intermediate_step",
                "analysis": "analysis",
                "candidates": [
                    {
                        "rank": 1,
                        "intermediate_smiles": "CCCl",
                        "reaction_description": "invalid mapped fragment",
                        "reaction_smirks": "[Qq:2][C:1](=[O:3])>>[Qq:2][C:1](=[O:3])",
                        "electron_pushes": [{"kind": "lone_pair", "source_atom": "2", "target_atom": "1", "electrons": 2}],
                    }
                ],
            }
            self.tool_calls = [{"arguments": json.dumps(payload)}]

    class _StubLLM:
        def invoke(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
            return _StubResponse()

    monkeypatch.setattr("mechanistic_agent.tools.adapter_supports_forced_tools", lambda _model: True)
    monkeypatch.setattr("mechanistic_agent.tools.get_model_api_key", lambda *_args, **_kwargs: "test-key")
    monkeypatch.setattr("mechanistic_agent.tools.get_chat_model", lambda *_args, **_kwargs: _StubLLM())

    raw = propose_intermediates(
        starting_materials=["CCBr", "[Cl-]"],
        products=["CCCl", "[Br-]"],
        current_state=["CCBr", "[Cl-]"],
    )
    payload = json.loads(raw)
    assert payload["has_executable_candidates"] is False
    assert payload["rejected_candidates"][0]["reason"] == "reaction_smirks_invalid"
    assert payload["rejected_candidates"][0]["chemistry_error_code"] in {"smiles_parse", "unknown"}


def test_propose_intermediates_rejects_disconnected_intermediate(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubResponse:
        usage = None

        def __init__(self) -> None:
            payload = {
                "classification": "intermediate_step",
                "analysis": "analysis",
                "candidates": [
                    {
                        "rank": 1,
                        "intermediate_smiles": "C.C.C",
                        "reaction_description": "degenerate candidate",
                        "reaction_smirks": "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] |mech:v1;lp:4>2;sigma:2-3>3|",
                        "electron_pushes": [{"kind": "lone_pair", "source_atom": "4", "target_atom": "2", "electrons": 2}],
                    }
                ],
            }
            self.tool_calls = [{"arguments": json.dumps(payload)}]

    class _StubLLM:
        def invoke(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
            return _StubResponse()

    monkeypatch.setattr("mechanistic_agent.tools.adapter_supports_forced_tools", lambda _model: True)
    monkeypatch.setattr("mechanistic_agent.tools.get_model_api_key", lambda *_args, **_kwargs: "test-key")
    monkeypatch.setattr("mechanistic_agent.tools.get_chat_model", lambda *_args, **_kwargs: _StubLLM())

    raw = propose_intermediates(
        starting_materials=["CCBr", "[Cl-]"],
        products=["CCCl", "[Br-]"],
        current_state=["CCBr", "[Cl-]"],
    )
    payload = json.loads(raw)
    assert payload["has_executable_candidates"] is False
    assert payload["rejected_candidates"][0]["reason"] == "intermediate_disconnected_species"


def test_candidate_rescue_dedupes_drops_existing_and_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mechanistic_agent.tools.predict_missing_reagents",
        lambda **_kwargs: json.dumps(
            {
                "status": "success",
                "suggested_reactants": ["O", "O", "CCO", "N", "Cl", "Br"],
                "suggested_products": ["CO", "CO", "CCO", "O", "Cl", "Br", "[Na+]"],
            }
        ),
    )

    raw = predict_missing_reagents_for_candidate(
        current_state=["CCO", "O"],
        resulting_state=["CO", "O", "CCO"],
        failed_checks=["atom_balance"],
        validation_details={"passed": False},
    )
    payload = json.loads(raw)
    assert payload["status"] == "success"
    assert payload["add_reactants"] == ["N", "Cl"]
    assert payload["add_products"] == ["Cl", "Br"]
    dropped = payload.get("dropped_additions", [])
    assert any(item["reason"] == "duplicate_suggestion" for item in dropped)
    assert any(item["reason"] == "already_present" for item in dropped)
    assert any(item["reason"] == "cap_exceeded" for item in dropped)


def test_select_reaction_type_supports_no_match(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubResponse:
        usage = None

        def __init__(self) -> None:
            payload = {
                "selected_label_exact": "no_match",
                "selected_type_id": None,
                "confidence": 0.22,
                "rationale": "No close taxonomy fit.",
                "top_candidates": [],
            }
            self.tool_calls = [{"arguments": json.dumps(payload)}]

    class _StubLLM:
        def invoke(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
            return _StubResponse()

    monkeypatch.setattr("mechanistic_agent.tools.adapter_supports_forced_tools", lambda _model: True)
    monkeypatch.setattr("mechanistic_agent.tools.get_model_api_key", lambda *_args, **_kwargs: "test-key")
    monkeypatch.setattr("mechanistic_agent.tools.get_chat_model", lambda *_args, **_kwargs: _StubLLM())

    raw = select_reaction_type(
        starting_materials=["CCBr", "[Cl-]"],
        products=["CCCl", "[Br-]"],
    )
    payload = json.loads(raw)
    assert payload["selected_label_exact"] == "no_match"
    assert payload["selected_type_id"] is None
    assert payload["selected_template"] is None


def test_propose_intermediates_receives_template_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {"messages": None}

    class _StubResponse:
        usage = None

        def __init__(self) -> None:
            payload = {
                "classification": "intermediate_step",
                "analysis": "SN2 substitution step.",
                "candidates": [
                    {
                        "rank": 1,
                        "intermediate_smiles": "CCCl",
                        "reaction_description": "SN2 substitution",
                        "reaction_smirks": "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] |mech:v1;lp:4>2;sigma:2-3>3|",
                        "electron_pushes": [{"kind": "lone_pair", "source_atom": "4", "target_atom": "2", "electrons": 2}],
                        "template_alignment": "aligned",
                        "template_alignment_reason": "Matches expected SN2 displacement.",
                    }
                ],
            }
            self.tool_calls = [{"arguments": json.dumps(payload)}]

    class _StubLLM:
        def invoke(self, messages, **_kwargs):  # noqa: ANN001, ANN003
            captured["messages"] = messages
            return _StubResponse()

    monkeypatch.setattr("mechanistic_agent.tools.adapter_supports_forced_tools", lambda _model: True)
    monkeypatch.setattr("mechanistic_agent.tools.get_model_api_key", lambda *_args, **_kwargs: "test-key")
    monkeypatch.setattr("mechanistic_agent.tools.get_chat_model", lambda *_args, **_kwargs: _StubLLM())

    raw = propose_intermediates(
        starting_materials=["CCBr", "[Cl-]"],
        products=["CCCl", "[Br-]"],
        current_state=["CCBr", "[Cl-]"],
        template_guidance={
            "selected_label_exact": "SN2 reaction",
            "selection_confidence": 0.91,
            "current_template_step_index": 1,
            "template_steps": [{"step_index": 1, "reaction_generic": "R-Br.[Cl-]>>R-Cl.[Br-]"}],
        },
    )
    payload = json.loads(raw)
    assert payload.get("candidates")
    message_text = "\n".join(str(msg.get("content") or "") for msg in (captured["messages"] or []))
    assert "Optional deterministic harness guidance" in message_text
    assert "SN2 reaction" in message_text


def test_tool_executor_does_not_forward_raw_mapped_prompt_context(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def _stub_propose_intermediates(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return json.dumps({"classification": "intermediate_step", "candidates": []})

    monkeypatch.setattr("mechanistic_agent.core.tool_executor.propose_intermediates", _stub_propose_intermediates)

    executor = ToolExecutor()
    executor.run_intermediates(
        starting=["[CH3:1][Br:2]", "[Cl-:3]"],
        products=["[CH3:1][Cl:3]", "[Br-:2]"],
        current_state=["[CH3:1][Br:2]", "[Cl-:3]"],
        previous_intermediates=[],
        ph=7.0,
        temperature=25.0,
        step_index=0,
        step_mapping_context=None,
        template_guidance=None,
    )

    assert captured["starting_materials"] == ["CBr", "[Cl-]"]
    assert captured["current_state"] == ["CBr", "[Cl-]"]
    assert captured["mapped_starting_materials"] == []
    assert captured["mapped_products"] == []
    assert captured["mapped_current_state"] == []


def test_propose_intermediates_retry_records_cli_command_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubResponse:
        usage = None

        def __init__(self) -> None:
            payload = {
                "classification": "intermediate_step",
                "analysis": "retry-safe candidate",
                "candidates": [
                    {
                        "rank": 1,
                        "intermediate_smiles": "CCCl",
                        "reaction_description": "SN2 substitution",
                        "reaction_smirks": "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] |mech:v1;lp:4>2;sigma:2-3>3|",
                        "electron_pushes": [{"kind": "lone_pair", "source_atom": "4", "target_atom": "2", "electrons": 2}],
                        "rdkit_cli_commands": [
                            {
                                "command": "repair-smiles",
                                "args": {"input": "CCCl"},
                                "run_on": "retry",
                                "apply_to": "intermediate_smiles",
                            }
                        ],
                    }
                ],
            }
            self.tool_calls = [{"arguments": json.dumps(payload)}]

    class _StubLLM:
        def invoke(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
            return _StubResponse()

    def _stub_cli(command, args, **_kwargs):  # noqa: ANN001, ANN003
        if command == "repair-smiles":
            return {
                "command": command,
                "status": "ok",
                "output": {"canonical_smiles": "CCCl"},
            }
        if command == "check":
            return {
                "command": command,
                "status": "ok",
                "output": {
                    "overall_pass": False,
                    "failed_check_names": ["atom_balance"],
                    "fix_suggestions": ["add missing byproduct"],
                },
            }
        return {"command": command, "status": "ok", "output": {}}

    monkeypatch.setattr("mechanistic_agent.tools.adapter_supports_forced_tools", lambda _model: True)
    monkeypatch.setattr("mechanistic_agent.tools.get_model_api_key", lambda *_args, **_kwargs: "test-key")
    monkeypatch.setattr("mechanistic_agent.tools.get_chat_model", lambda *_args, **_kwargs: _StubLLM())
    monkeypatch.setattr(
        "mechanistic_agent.tools._load_rdkit_cli_available_commands",
        lambda **_kwargs: {"repair-smiles", "check"},
    )
    monkeypatch.setattr("mechanistic_agent.tools._run_rdkit_cli_command", _stub_cli)

    raw = propose_intermediates(
        starting_materials=["CCBr", "[Cl-]"],
        products=["CCCl", "[Br-]"],
        current_state=["CCBr", "[Cl-]"],
        template_guidance={"retry_mode": True},
    )
    payload = json.loads(raw)
    executed = payload.get("executed_cli_commands", [])
    commands = [entry.get("command") for entry in executed if isinstance(entry, dict)]
    assert "repair-smiles" in commands
    assert "check" in commands
    assert payload["candidates"][0].get("cli_retry_check", {}).get("failed_check_names") == ["atom_balance"]


def test_predict_missing_reagents_retry_uses_balance_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubResponse:
        usage = None

        def __init__(self, payload: Dict[str, Any]) -> None:
            self.tool_calls = [{"arguments": json.dumps(payload)}]

    class _StubLLM:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
            self.calls += 1
            if self.calls == 1:
                return _StubResponse({"missing_reactants": ["Cl"], "missing_products": []})
            return _StubResponse({"missing_reactants": ["Cl"], "missing_products": ["O"]})

    validation_calls: List[Dict[str, Any]] = []

    def _stub_validate(reactants, missing_products, _starting, _products):  # noqa: ANN001, ANN002
        validation_calls.append({"reactants": list(reactants), "missing_products": list(missing_products)})
        if len(validation_calls) == 1:
            return json.dumps(
                {
                    "status": "failed",
                    "is_balanced": False,
                    "invalid_reagents": [],
                    "remaining_deficit": {"O": 1},
                    "remaining_surplus": {},
                    "reason": "not balanced",
                }
            )
        return json.dumps(
            {
                "status": "success",
                "is_balanced": True,
                "valid_reagents": list(reactants) + list(missing_products),
            }
        )

    def _stub_cli(command, args, **_kwargs):  # noqa: ANN001, ANN003
        if command == "balance":
            return {
                "command": "balance",
                "status": "ok",
                "output": {"fix_suggestions": ["add O to products"], "remaining_deficit": {"O": 1}},
            }
        return {"command": command, "status": "ok", "output": {}}

    llm = _StubLLM()
    monkeypatch.setattr("mechanistic_agent.tools.adapter_supports_forced_tools", lambda _model: True)
    monkeypatch.setattr("mechanistic_agent.tools.get_model_api_key", lambda *_args, **_kwargs: "test-key")
    monkeypatch.setattr("mechanistic_agent.tools.get_chat_model", lambda *_args, **_kwargs: llm)
    monkeypatch.setattr("mechanistic_agent.tools.validate_proposed_reagents", _stub_validate)
    monkeypatch.setattr("mechanistic_agent.tools._run_rdkit_cli_command", _stub_cli)

    raw = predict_missing_reagents(
        starting_materials=["CCBr"],
        products=["CCCl"],
    )
    payload = json.loads(raw)
    assert payload["status"] == "success"
    assert payload.get("balance_retry_diagnostics", {}).get("remaining_deficit") == {"O": 1}
    executed = payload.get("executed_cli_commands", [])
    assert any(entry.get("command") == "balance" for entry in executed if isinstance(entry, dict))


def test_predict_missing_reagents_retry_executes_repair_smiles_and_applies_fix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StubResponse:
        usage = None

        def __init__(self) -> None:
            self.tool_calls = [{"arguments": json.dumps({"missing_reactants": ["C=C"], "missing_products": []})}]

    class _StubLLM:
        def invoke(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
            return _StubResponse()

    validation_calls: List[List[str]] = []

    def _stub_validate(reactants, _missing_products, _starting, _products):  # noqa: ANN001, ANN002
        validation_calls.append(list(reactants))
        if len(validation_calls) == 1:
            return json.dumps(
                {
                    "status": "failed",
                    "is_balanced": False,
                    "invalid_reagents": [{"molecule": "C=C", "error": "needs repair"}],
                    "remaining_deficit": {},
                    "remaining_surplus": {},
                }
            )
        return json.dumps(
            {
                "status": "success",
                "is_balanced": True,
                "valid_reagents": list(reactants),
            }
        )

    def _stub_cli(command, args, **_kwargs):  # noqa: ANN001, ANN003
        if command == "repair-smiles":
            return {
                "command": "repair-smiles",
                "status": "ok",
                "output": {"canonical_smiles": "CC"},
            }
        return {"command": command, "status": "ok", "output": {}}

    monkeypatch.setattr("mechanistic_agent.tools.adapter_supports_forced_tools", lambda _model: True)
    monkeypatch.setattr("mechanistic_agent.tools.get_model_api_key", lambda *_args, **_kwargs: "test-key")
    monkeypatch.setattr("mechanistic_agent.tools.get_chat_model", lambda *_args, **_kwargs: _StubLLM())
    monkeypatch.setattr("mechanistic_agent.tools.validate_proposed_reagents", _stub_validate)
    monkeypatch.setattr("mechanistic_agent.tools._run_rdkit_cli_command", _stub_cli)

    raw = predict_missing_reagents(
        starting_materials=["CCBr"],
        products=["CCCl"],
    )
    payload = json.loads(raw)
    assert payload["status"] == "success"
    assert validation_calls[0] == ["C=C"]
    assert validation_calls[1] == ["CC"]
    executed = payload.get("executed_cli_commands", [])
    assert any(entry.get("command") == "repair-smiles" for entry in executed if isinstance(entry, dict))
    applied = payload.get("cli_applied_fixes", [])
    assert any(item.get("command") == "repair-smiles" for item in applied if isinstance(item, dict))


def test_predict_missing_reagents_emits_participant_constraints(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubResponse:
        usage = None

        def __init__(self) -> None:
            self.tool_calls = [{"arguments": json.dumps({"missing_reactants": ["[OH3+]"], "missing_products": ["O"]})}]

    class _StubLLM:
        def invoke(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
            return _StubResponse()

    monkeypatch.setattr("mechanistic_agent.tools.adapter_supports_forced_tools", lambda _model: True)
    monkeypatch.setattr("mechanistic_agent.tools.get_model_api_key", lambda *_args, **_kwargs: "test-key")
    monkeypatch.setattr("mechanistic_agent.tools.get_chat_model", lambda *_args, **_kwargs: _StubLLM())
    monkeypatch.setattr(
        "mechanistic_agent.tools.validate_proposed_reagents",
        lambda reactants, products, *_args, **_kwargs: json.dumps(
            {"status": "success", "is_balanced": True, "valid_reagents": list(reactants) + list(products)}
        ),
    )

    raw = predict_missing_reagents(
        starting_materials=["CCO"],
        products=["CCCl"],
        conditions_guidance=json.dumps(
            {
                "environment": "acidic",
                "representative_ph": 2.0,
                "acid_candidates": [{"name": "hydronium", "smiles": "[OH3+]", "role": "acid"}],
            }
        ),
    )
    payload = json.loads(raw)
    constraints = payload.get("proposal_constraints", {})
    registry = payload.get("species_registry", [])
    assert payload["missing_reactants"] == ["[OH3+]"]
    assert payload["missing_products"] == ["O"]
    assert constraints.get("environment") == "acidic"
    assert "[OH-]" in constraints.get("forbidden_new_species", [])
    assert "O" in constraints.get("allowed_generated_species", [])
    assert any(entry.get("species") == "[OH3+]" and "acid" in entry.get("roles", []) for entry in registry)
