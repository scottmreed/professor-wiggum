"""Unified pre-analysis: schema, preset replay through the old tools, checks."""
from __future__ import annotations

import json

import pytest

from mechanistic_agent.core import preanalysis_checks as checks
from mechanistic_agent.llm import PresetAnswerConsumedError, get_chat_model, preset_tool_answers
from mechanistic_agent.tool_schemas import UNIFIED_PREANALYSIS_TOOL_NAME, build_unified_preanalysis_tool
from mechanistic_agent.unified_preanalysis import (
    assistant_turn_text,
    build_user_prompt,
    repair_prompt,
    section_presets,
    sections_for,
)


def test_schema_has_all_sections_and_hidden_variant_drops_reagents_and_mapping() -> None:
    full = build_unified_preanalysis_tool()["function"]
    hidden = build_unified_preanalysis_tool(products_hidden=True)["function"]
    assert full["name"] == UNIFIED_PREANALYSIS_TOOL_NAME
    assert full["parameters"]["required"] == ["conditions", "missing_reagents", "atom_mapping", "reaction_type"]
    assert hidden["parameters"]["required"] == ["conditions", "reaction_type"]
    assert "text" not in full["parameters"]["properties"]["conditions"]["properties"]
    assert "environment" in full["parameters"]["properties"]["conditions"]["properties"]


def test_preset_adapter_is_single_use_and_needs_no_model() -> None:
    with preset_tool_answers({"assess_conditions_result": {"environment": "basic"}}):
        llm = get_chat_model("no-such-model")
        msg = llm.invoke([], tools=[], tool_choice={"type": "function", "function": {"name": "assess_conditions_result"}})
        assert json.loads(msg.tool_calls[0]["arguments"]) == {"environment": "basic"}
        with pytest.raises(PresetAnswerConsumedError):
            llm.invoke([], tool_choice={"type": "function", "function": {"name": "assess_conditions_result"}})


def test_conditions_section_replays_through_assess_initial_conditions() -> None:
    pytest.importorskip("rdkit")
    from mechanistic_agent.tools import assess_initial_conditions

    answer = {
        "conditions": {
            "environment": "basic",
            "representative_ph": 11.0,
            "justification": "alkoxide base",
            "base_candidates": [{"name": "sodium ethoxide", "smiles": "CC[O-].[Na+]", "role": "base"}],
        }
    }
    with preset_tool_answers(section_presets(answer, ["conditions"])):
        output = json.loads(assess_initial_conditions(["CCBr", "CC[O-]"], ["CCOCC", "[Br-]"]))
    assert output["status"] == "success"
    assert output["environment"] == "basic"
    assert checks.check_conditions(output)["status"] == checks.PASS


def test_invented_reaction_type_label_is_a_soft_failure() -> None:
    pytest.importorskip("rdkit")
    from mechanistic_agent.tools import select_reaction_type

    raw = {"selected_label_exact": "Totally made up", "confidence": 0.9, "rationale": "x"}
    with preset_tool_answers({"reaction_type_selection_result": raw}):
        output = json.loads(select_reaction_type(starting_materials=["CCBr", "[I-]"], products=["CCI", "[Br-]"]))
    assert output["selected_label_exact"] == "no_match"
    result = checks.check_sections({"reaction_type": output}, ["reaction_type"], {"reaction_type": raw})
    assert result["reaction_type"]["status"] == checks.SOFT_FAIL


def test_missing_reagent_deficit_is_hard_failure() -> None:
    output = {
        "status": "failed",
        "error": "unbalanced",
        "should_abort_mechanism": True,
        "balance_issues": {"remaining_deficit": {"O": 1}, "remaining_surplus": {}},
    }
    result = checks.check_missing_reagents(output)
    assert result["status"] == checks.HARD_FAIL
    assert any("deficit" in e for e in result["errors"])
    assert checks.check_missing_reagents({"status": "balanced"})["status"] == checks.PASS


def test_atom_mapping_check_reads_rdkit_validation() -> None:
    base = {"llm_response": {"mapped_atoms": [{"product_atom": "CCI#0"}]}}
    assert checks.check_atom_mapping({**base, "atom_map_validation": {"passed": True}})["status"] == checks.PASS
    assert checks.check_atom_mapping({**base, "atom_map_validation": {"skipped": True}})["status"] == checks.PASS
    failed = checks.check_atom_mapping(
        {**base, "atom_map_validation": {"passed": False, "checks": [{"name": "atom_map_pairs_resolved", "passed": False, "details": {"errors": ["bad idx"]}}]}}
    )
    assert failed["status"] == checks.SOFT_FAIL
    assert "atom_map_pairs_resolved" in failed["errors"][0]
    assert checks.check_atom_mapping({"llm_response": {"mapped_atoms": []}})["status"] == checks.SOFT_FAIL


def test_prompt_mentions_only_requested_sections_when_products_hidden() -> None:
    pytest.importorskip("rdkit")
    prompt = build_user_prompt(
        starting_materials=["CCBr"], products=[], products_hidden=True, context={"ph_recommendation": {"ph": 7}}
    )
    assert "missing_reagents:" not in prompt and "atom_mapping:" not in prompt
    assert "reaction_type:" in prompt and "Products: not given" in prompt
    assert sections_for(True) == ["conditions", "reaction_type"]


def test_repair_prompt_and_assistant_text() -> None:
    text = repair_prompt({"atom_mapping": ["atom_map_pairs_resolved: bad idx"]})
    assert "atom_mapping: atom_map_pairs_resolved" in text
    assert assistant_turn_text({"a": 1}).startswith(f"Called `{UNIFIED_PREANALYSIS_TOOL_NAME}` with:")
