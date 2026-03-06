import pytest

pytest.importorskip("pydantic", minversion="2.0")

from mechanistic_agent.config import (
    ReactionInputs,
    OPTIONAL_LLM_TOOL_NAMES,
)


def test_reaction_inputs_optional_tools_default():
    reaction = ReactionInputs()
    assert reaction.optional_llm_tools == list(OPTIONAL_LLM_TOOL_NAMES)


def test_reaction_inputs_optional_tools_subset():
    reaction = ReactionInputs(optional_llm_tools=["predict_missing_reagents"])
    assert reaction.optional_llm_tools == ["predict_missing_reagents"]


def test_reaction_inputs_optional_tools_invalid():
    with pytest.raises(ValueError):
        ReactionInputs(optional_llm_tools=["not_a_tool"])



def test_model_and_step_models_are_normalized_to_canonical_keys():
    reaction = ReactionInputs(
        model="anthropic/claude-sonnet-4-5",
        step_models={"functional_groups": "anthropic/claude-sonnet-4-6"},
    )

    assert reaction.model == "anthropic/claude-sonnet-4.5"
    assert reaction.step_models["functional_groups"] == "anthropic/claude-sonnet-4.6"


def test_invalid_step_model_still_fails_validation():
    with pytest.raises(ValueError):
        ReactionInputs(step_models={"functional_groups": "bad-model-name"})
