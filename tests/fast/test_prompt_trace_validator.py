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
                "responder_saw_ground_truth": False,
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


def _write_evidence(tmp_path: Path, bundle: str, name: str, extra: dict) -> None:
    evidence_dir = tmp_path / "traces" / "evidence" / "assess_initial_conditions" / bundle
    evidence_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "approved_bool": True,
        "prompt_version": {"prompt_bundle_sha256": bundle},
        "model_version": {
            "model_version_id": "abc",
            "resolved_model_key": "agent-bridge",
            "provider": "agent_bridge",
            "family": "agent",
            "pricing_sha256": "123",
        },
    }
    payload.update(extra)
    (evidence_dir / name).write_text(json.dumps(payload), encoding="utf-8")


def test_validator_rejects_evidence_that_saw_ground_truth_or_is_undeclared(tmp_path: Path) -> None:
    _seed_prompt_versions(tmp_path)
    version = get_call_prompt_version("assess_initial_conditions", tmp_path)
    bundle = str(version["prompt_bundle_sha256"])

    # Declared replay of the verified mechanism: not evidence of capability.
    _write_evidence(tmp_path, bundle, "replay.json", {"origin": {"responder_saw_ground_truth": True}})
    # Missing declaration: also rejected until someone states it.
    _write_evidence(tmp_path, bundle, "undeclared.json", {})

    result = validate_evidence_for_calls(changed_calls=["assess_initial_conditions"], base_dir=tmp_path)
    assert not result.ok
    assert any("ground-truth exposure" in err for err in result.errors)

    # A blind run declared via the origin block is accepted.
    _write_evidence(tmp_path, bundle, "blind.json", {"origin": {"responder_saw_ground_truth": False}})
    result_ok = validate_evidence_for_calls(changed_calls=["assess_initial_conditions"], base_dir=tmp_path)
    assert result_ok.ok
    assert result_ok.valid_evidence_by_call["assess_initial_conditions"] == [
        str((tmp_path / "traces" / "evidence" / "assess_initial_conditions" / bundle / "blind.json").resolve().relative_to(tmp_path))
    ]


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
