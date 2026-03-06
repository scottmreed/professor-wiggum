"""Tests for exact-model runtime selection."""

from mechanistic_agent.core.model_selection import preview_step_models, select_step_models


def test_single_model_selection_assigns_exact_model_to_all_enabled_steps():
    result = select_step_models(
        model_name="gpt-5.1",
        thinking_level=None,
        functional_groups_enabled=False,
        intermediate_prediction_enabled=True,
        optional_llm_tools=["attempt_atom_mapping", "predict_missing_reagents"],
    )

    assert result.model_name == "gpt-5.1"
    assert result.family == "openai"
    assert result.step_models["mechanism_synthesis"] == "gpt-5.1"
    assert result.step_models["intermediates"] == "gpt-5.1"
    assert result.step_models["initial_conditions"] == "gpt-5.1"
    assert result.step_models["reaction_type_mapping"] == "gpt-5.1"
    assert "functional_groups" not in result.step_models


def test_thinking_level_maps_to_internal_reasoning_levels():
    result = select_step_models(
        model_name="gpt-5.1",
        thinking_level="high",
        functional_groups_enabled=True,
        intermediate_prediction_enabled=False,
        optional_llm_tools=["predict_missing_reagents"],
    )

    assert result.thinking_level == "high"
    assert set(result.step_reasoning.values()) == {"highest"}
    assert "atom_mapping" not in result.step_models
    assert "intermediates" not in result.step_models


def test_unsupported_thinking_level_is_ignored_for_non_reasoning_model():
    result = select_step_models(
        model_name="allenai/olmo-3.1-32b-instruct",
        thinking_level="low",
        functional_groups_enabled=True,
        intermediate_prediction_enabled=True,
        optional_llm_tools=["attempt_atom_mapping", "predict_missing_reagents"],
    )

    assert result.family == "olmo"
    assert result.thinking_level is None
    assert result.step_reasoning == {}


def test_preview_step_models_returns_uniform_map():
    preview = preview_step_models(
        model_name="minimax/minimax-m2.1",
        thinking_level=None,
        functional_groups_enabled=True,
        intermediate_prediction_enabled=False,
        optional_llm_tools=["predict_missing_reagents"],
    )

    assert set(preview.values()) == {"minimax/minimax-m2.1"}
    assert "intermediates" not in preview



def test_single_model_selection_normalizes_dash_variant_to_dot_canonical_model():
    result = select_step_models(
        model_name="anthropic/claude-sonnet-4-5",
        thinking_level=None,
        functional_groups_enabled=True,
        intermediate_prediction_enabled=True,
        optional_llm_tools=["attempt_atom_mapping", "predict_missing_reagents"],
    )

    assert result.model_name == "anthropic/claude-sonnet-4.5"
    assert all(model == "anthropic/claude-sonnet-4.5" for model in result.step_models.values())
