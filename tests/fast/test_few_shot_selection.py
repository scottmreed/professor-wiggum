from __future__ import annotations

import json
from pathlib import Path

from mechanistic_agent.core.types import FewShotSelectionConfig
from mechanistic_agent.prompt_assets import (
    append_call_few_shot_example,
    best_few_shot_score,
    format_few_shot_block,
    get_call_prompt_version,
    score_few_shot_example,
    select_few_shot_examples,
)


def _write_examples(base: Path, call_name: str, rows: list[dict[str, object]]) -> None:
    call_dir = base / "skills" / "mechanistic" / call_name
    call_dir.mkdir(parents=True, exist_ok=True)
    (call_dir / "few_shot.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_select_few_shot_examples_prefers_highest_scores(tmp_path: Path) -> None:
    _write_examples(
        tmp_path,
        "predict_missing_reagents",
        [
            {"input": "older", "output": json.dumps({"missing_reactants": ["O"], "missing_products": []}), "score": 0.4},
            {"input": "mid", "output": json.dumps({"missing_reactants": ["Cl"], "missing_products": []}), "score": 0.8},
            {"input": "newest", "output": json.dumps({"missing_reactants": ["Br"], "missing_products": []}), "score": 0.8},
        ],
    )

    selected = select_few_shot_examples(
        "predict_missing_reagents",
        tmp_path,
        policy=FewShotSelectionConfig(max_examples=2, selection_strategy="top_score"),
    )

    assert [row["input"] for row in selected] == ["newest", "mid"]


def test_select_few_shot_examples_respects_most_recent_strategy(tmp_path: Path) -> None:
    _write_examples(
        tmp_path,
        "assess_initial_conditions",
        [
            {"input": "first", "output": json.dumps({"environment": "neutral", "representative_ph": 7.0}), "score": 0.9},
            {"input": "second", "output": json.dumps({"environment": "acidic", "representative_ph": 2.0}), "score": 0.2},
        ],
    )

    selected = select_few_shot_examples(
        "assess_initial_conditions",
        tmp_path,
        policy=FewShotSelectionConfig(max_examples=1, selection_strategy="most_recent"),
    )

    assert [row["input"] for row in selected] == ["second"]


def test_derived_scoring_works_for_legacy_rows_without_score(tmp_path: Path) -> None:
    output_text = json.dumps(
        {
            "classification": "intermediate_step",
            "candidates": [
                {
                    "rank": 1,
                    "intermediate_smiles": "CO",
                    "reaction_description": "Attack then proton transfer",
                    "reaction_smirks": "[CH3:1][OH:2]>>[CH3:1][OH2+:2] |mech:v1;lp:2>1|",
                    "electron_pushes": [{"kind": "lone_pair", "target_atom": "1", "electrons": 2}],
                }
            ],
            "analysis": "Plausible next step.",
        }
    )
    _write_examples(
        tmp_path,
        "propose_mechanism_step",
        [{"input": "legacy", "output": output_text}],
    )

    score = score_few_shot_example(
        "propose_mechanism_step",
        input_text="legacy",
        output_text=output_text,
    )
    assert score > 0.7
    assert best_few_shot_score("propose_mechanism_step", tmp_path) == score

    block = format_few_shot_block(
        "propose_mechanism_step",
        tmp_path,
        policy=FewShotSelectionConfig(max_examples=1),
    )
    assert "Example 1 input:" in block
    assert "legacy" in block


def test_exact_model_lane_merges_override_and_shared_examples(tmp_path: Path) -> None:
    _write_examples(
        tmp_path,
        "predict_missing_reagents",
        [{"input": "shared", "output": json.dumps({"missing_reactants": ["O"], "missing_products": []}), "score": 0.4}],
    )
    override_dir = tmp_path / "skills" / "mechanistic" / "predict_missing_reagents" / "models" / "anthropic__claude-opus-4.6"
    override_dir.mkdir(parents=True, exist_ok=True)
    (override_dir / "few_shot.jsonl").write_text(
        json.dumps({"input": "opus", "output": json.dumps({"missing_reactants": ["Cl"], "missing_products": []}), "score": 0.9}) + "\n",
        encoding="utf-8",
    )

    selected = select_few_shot_examples(
        "predict_missing_reagents",
        tmp_path,
        policy=FewShotSelectionConfig(max_examples=2, selection_strategy="most_recent"),
        model_name="anthropic/claude-opus-4.6",
    )

    assert [row["input"] for row in selected] == ["shared", "opus"] or [row["input"] for row in selected] == ["opus", "shared"]
    append_call_few_shot_example(
        "predict_missing_reagents",
        input_text="new-opus",
        output_text=json.dumps({"missing_reactants": ["Br"], "missing_products": []}),
        base_dir=tmp_path,
        model_name="anthropic/claude-opus-4.6",
    )
    override_text = (override_dir / "few_shot.jsonl").read_text(encoding="utf-8")
    assert "new-opus" in override_text


def test_get_call_prompt_version_prefers_model_override_when_present(tmp_path: Path) -> None:
    base_system = tmp_path / "skills" / "mechanistic" / "base_system"
    base_system.mkdir(parents=True, exist_ok=True)
    (base_system / "SKILL.md").write_text(
        "---\nkind: shared_base\ncall_name: base_system\n---\n<!-- PROMPT_START -->\nshared base\n<!-- PROMPT_END -->\n",
        encoding="utf-8",
    )
    call_dir = tmp_path / "skills" / "mechanistic" / "propose_mechanism_step"
    call_dir.mkdir(parents=True, exist_ok=True)
    (call_dir / "SKILL.md").write_text(
        "---\nkind: llm\ncall_name: propose_mechanism_step\nsteps: [mechanism_step_proposal]\n---\n<!-- PROMPT_START -->\nshared prompt\n<!-- PROMPT_END -->\n",
        encoding="utf-8",
    )
    (call_dir / "few_shot.jsonl").write_text("", encoding="utf-8")
    override_dir = call_dir / "models" / "anthropic__claude-opus-4.6"
    override_dir.mkdir(parents=True, exist_ok=True)
    (override_dir / "SKILL.md").write_text(
        "---\nkind: llm\ncall_name: propose_mechanism_step\nsteps: [mechanism_step_proposal]\n---\n<!-- PROMPT_START -->\nopus prompt\n<!-- PROMPT_END -->\n",
        encoding="utf-8",
    )

    shared = get_call_prompt_version("propose_mechanism_step", tmp_path)
    opus = get_call_prompt_version("propose_mechanism_step", tmp_path, model_name="anthropic/claude-opus-4.6")

    assert opus["asset_scope"] == "exact_model"
    assert opus["resolved_call_base_path"].endswith("anthropic__claude-opus-4.6/SKILL.md")
    assert shared["prompt_bundle_sha256"] != opus["prompt_bundle_sha256"]
