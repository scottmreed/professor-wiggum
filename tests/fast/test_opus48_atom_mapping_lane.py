"""Fast checks for the Claude Opus 4.8 enablement + attempt_atom_mapping lane seed.

Keyless agent-bridge contribution: a new catalog entry and the first model lane for
the previously-empty `attempt_atom_mapping` few-shot file. These assert the artifacts
are present and structurally valid (no network / no API key required).
"""
from __future__ import annotations

import json
from pathlib import Path

from mechanistic_agent.prompt_assets import (
    load_call_few_shot_examples,
    score_few_shot_example,
)

REPO = Path(__file__).resolve().parents[2]
MODEL = "anthropic/claude-opus-4.8"
CALL = "attempt_atom_mapping"


def test_opus_4_8_in_model_catalog() -> None:
    catalog = json.loads((REPO / "mechanistic_agent" / "model_pricing.json").read_text())
    assert MODEL in catalog, "claude-opus-4.8 must be present in the model catalog"
    entry = catalog[MODEL]
    assert entry["supports_tools"] is True
    assert entry["family"] == "claude"
    assert entry["provider"]  # non-empty
    assert "pricing_per_million" in entry


def test_opus_4_8_atom_mapping_lane_exists_and_is_valid() -> None:
    lane = REPO / "skills" / "mechanistic" / CALL / "models" / "anthropic__claude-opus-4.8" / "few_shot.jsonl"
    assert lane.exists(), "opus-4.8 attempt_atom_mapping lane file must exist"

    examples = load_call_few_shot_examples(CALL, REPO, model_name=MODEL)
    assert len(examples) >= 1, "lane must contain at least one seeded example"

    for ex in examples:
        # output must be a JSON object carrying the required atom_mapping_result fields
        payload = json.loads(ex["output"])
        assert isinstance(payload, dict)
        assert "confidence" in payload and "reasoning" in payload
        assert isinstance(payload.get("mapped_atoms"), list)
        # the project's own scorer should rate the seeded example as usable
        score = score_few_shot_example(
            CALL, input_text=ex["input"], output_text=ex["output"]
        )
        assert score >= 0.5, f"seeded example scored too low: {score}"


def test_base_atom_mapping_lane_untouched() -> None:
    """The model-agnostic base lane stays empty; this change is opus-4.8 scoped."""
    base_examples = load_call_few_shot_examples(CALL, REPO)
    assert base_examples == [], "base attempt_atom_mapping lane should remain empty"


def test_opus_4_8_propose_mechanism_step_lane_valid() -> None:
    """Medium-tier seed: the new multi-step proposal lane loads schema-valid examples."""
    lane = REPO / "skills" / "mechanistic" / "propose_mechanism_step" / "models" / "anthropic__claude-opus-4.8" / "few_shot.jsonl"
    assert lane.exists(), "opus-4.8 propose_mechanism_step lane file must exist"

    examples = load_call_few_shot_examples("propose_mechanism_step", REPO, model_name=MODEL)
    # at least the seeded model-lane examples (plus any base example)
    assert len(examples) >= 4, "propose lane should hold the seeded multi-step examples"

    saw_candidates = False
    for ex in examples:
        payload = json.loads(ex["output"])
        assert isinstance(payload, dict)
        assert payload.get("classification") in {"intermediate_step", "final_step"}
        cands = payload.get("candidates")
        if isinstance(cands, list) and cands:
            saw_candidates = True
            cand = cands[0]
            # each candidate carries an intermediate SMILES + arrow-pushing electron moves
            assert cand.get("intermediate_smiles")
            assert isinstance(cand.get("electron_pushes"), list) and cand["electron_pushes"]
        score = score_few_shot_example(
            "propose_mechanism_step", input_text=ex["input"], output_text=ex["output"]
        )
        assert score >= 0.5, f"seeded propose example scored too low: {score}"
    assert saw_candidates, "expected at least one example with a candidate block"
