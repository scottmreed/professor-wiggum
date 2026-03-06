from __future__ import annotations

import json
from pathlib import Path

import pytest

from mechanistic_agent.prompt_assets import get_call_prompt_version
from mechanistic_agent.prompt_trace_validator import validate_evidence_for_calls


def _seed_prompt_versions(base: Path) -> None:
    (base / "prompt_versions" / "shared").mkdir(parents=True, exist_ok=True)
    (base / "prompt_versions" / "calls" / "assess_initial_conditions").mkdir(parents=True, exist_ok=True)
    (base / "prompt_versions" / "shared" / "base_system.md").write_text("shared", encoding="utf-8")
    (base / "prompt_versions" / "calls" / "assess_initial_conditions" / "base.md").write_text(
        "call base",
        encoding="utf-8",
    )
    (base / "prompt_versions" / "calls" / "assess_initial_conditions" / "few_shot.jsonl").write_text(
        "",
        encoding="utf-8",
    )


def test_validator_passes_with_approved_linked_evidence(tmp_path: Path) -> None:
    _seed_prompt_versions(tmp_path)
    version = get_call_prompt_version("assess_initial_conditions", tmp_path)
    bundle = str(version["prompt_bundle_sha256"])
    evidence_dir = tmp_path / "traces" / "evidence" / "assess_initial_conditions" / bundle
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "trace1.json").write_text(
        json.dumps(
            {
                "approved_bool": True,
                "prompt_version": {"prompt_bundle_sha256": bundle},
                "model_version": {
                    "model_version_id": "abc",
                    "resolved_model_key": "gpt-5",
                    "provider": "openai",
                    "family": "openai",
                    "pricing_sha256": "123",
                },
            }
        ),
        encoding="utf-8",
    )

    result = validate_evidence_for_calls(
        changed_calls=["assess_initial_conditions"],
        base_dir=tmp_path,
    )
    assert result.ok


def test_validator_fails_without_evidence(tmp_path: Path) -> None:
    _seed_prompt_versions(tmp_path)
    result = validate_evidence_for_calls(
        changed_calls=["assess_initial_conditions"],
        base_dir=tmp_path,
    )
    assert not result.ok
    assert any("no evidence files found" in err for err in result.errors)


def test_get_call_prompt_version_rejects_path_traversal_call_name(tmp_path: Path) -> None:
    _seed_prompt_versions(tmp_path)
    with pytest.raises(ValueError):
        get_call_prompt_version("../../etc/passwd", tmp_path)
