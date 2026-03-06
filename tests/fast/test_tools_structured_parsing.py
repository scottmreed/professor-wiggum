"""Regression tests for tool-response parsing and SMILES token filtering."""

import json
import pytest
from pydantic import ValidationError

from mechanistic_agent.tools import (
    AtomMappingPayload,
    MechanismIntermediate,
    MechanismStepCandidate,
    MechanismStepPrediction,
    MissingReagentsPayload,
    ToolRuntimeError,
    _apply_smiles_correction,
    _looks_like_smiles,
    _mol_from_smiles,
    attempt_atom_mapping_for_step,
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
    assert "Optional reaction-type template guidance" in message_text
    assert "SN2 reaction" in message_text
