from __future__ import annotations

import pytest

from mechanistic_agent.llm import _openai_schema_to_gemini


def test_integer_enum_preserved_for_gemini_schema() -> None:
    pytest.importorskip("google.genai.types")
    schema = {
        "type": "object",
        "properties": {
            "electrons": {"type": "integer", "enum": [1, 2]},
            "confidence": {"type": "string", "enum": ["high", "low"]},
        },
    }
    converted = _openai_schema_to_gemini(schema)
    electrons = converted.properties["electrons"]
    confidence = converted.properties["confidence"]
    assert electrons.enum in (None, [])
    assert list(confidence.enum) == ["high", "low"]
