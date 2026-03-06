"""Tests for model registry helpers."""

import importlib.util
import math
import pathlib
import sys
import types

import pytest


if "pydantic" not in sys.modules:
    stub = types.ModuleType("pydantic")

    class BaseModel:  # type: ignore[too-few-public-methods]
        """Minimal stand-in for pydantic.BaseModel used only for imports."""

    def Field(*args, default=None, default_factory=None, **kwargs):  # type: ignore[unused-argument]
        if default_factory is not None:
            try:
                return default_factory()
            except Exception:
                return None
        return default

    def field_validator(*args, **kwargs):  # type: ignore[unused-argument]
        def decorator(func):
            return func

        return decorator

    stub.BaseModel = BaseModel
    stub.Field = Field
    stub.field_validator = field_validator
    sys.modules["pydantic"] = stub


MODULE_PATH = pathlib.Path(__file__).resolve().parents[2] / "mechanistic_agent" / "model_registry.py"
SPEC = importlib.util.spec_from_file_location("_model_registry", MODULE_PATH)
assert SPEC and SPEC.loader
_module = importlib.util.module_from_spec(SPEC)
package = sys.modules.setdefault("mechanistic_agent", types.ModuleType("mechanistic_agent"))
if not getattr(package, "__path__", None):
    package.__path__ = [str(MODULE_PATH.parent)]  # type: ignore[attr-defined]
SPEC.loader.exec_module(_module)  # type: ignore[arg-type]

calculate_cost = _module.calculate_cost
get_all_families = _module.get_all_families
get_model_family = _module.get_model_family
get_model_provider = _module.get_model_provider
get_model_spec = _module.get_model_spec
resolve_model_key = _module.resolve_model_key


def test_get_model_spec_handles_versioned_names():
    """Versioned model identifiers should resolve to their base catalogue entry."""

    base_spec = get_model_spec("gpt-5")
    versioned_spec = get_model_spec("gpt-5-2025-08-07")

    assert versioned_spec == base_spec


def test_calculate_cost_uses_pricing_for_versioned_model():
    """Costs are derived from pricing data even when the model includes a suffix."""

    usage = {
        "input_tokens": 1_000,
        "cached_input_tokens": 0,
        "output_tokens": 2_000,
    }

    cost = calculate_cost("gpt-5-2025-08-07", usage)

    expected_input = 1_000 / 1_000_000 * 1.25
    expected_output = 2_000 / 1_000_000 * 10.0

    assert math.isclose(cost["input_cost"], expected_input)
    assert math.isclose(cost["output_cost"], expected_output)
    assert math.isclose(cost["total_cost"], expected_input + expected_output)


def test_unknown_model_raises_value_error():
    """Unknown models should still raise a clear ValueError."""

    with pytest.raises(ValueError):
        get_model_spec("unknown-model")


def test_minimax_model_resolves_to_expected_family_and_provider():
    spec = get_model_spec("minimax/minimax-m2.1")

    assert spec["family"] == "minimax"
    assert get_model_family("minimax/minimax-m2.1") == "minimax"
    assert get_model_provider("minimax/minimax-m2.1") == "openrouter"


def test_catalog_families_includes_minimax():
    family_ids = [item["id"] for item in get_all_families()]

    assert "minimax" in family_ids


def test_sonnet_dot_and_dash_variants_resolve_to_dot_canonical_key():
    assert resolve_model_key("anthropic/claude-sonnet-4-5") == "anthropic/claude-sonnet-4.5"
    assert resolve_model_key("anthropic/claude-sonnet-4.5") == "anthropic/claude-sonnet-4.5"


def test_sonnet_46_model_is_available():
    spec = get_model_spec("anthropic/claude-sonnet-4.6")
    assert spec["id"] == "anthropic/claude-sonnet-4.6"
