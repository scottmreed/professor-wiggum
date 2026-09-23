from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from mechanistic_agent.prompt_assets import CALL_TO_STEPS, get_call_prompt_version
from mechanistic_agent.prompt_trace_validator import (
    PromptChange,
    calls_from_changed_paths,
    discover_changed_calls,
    validate_evidence_for_calls,
)

CALL = "assess_initial_conditions"
OPUS = "anthropic/claude-opus-4.6"
OPUS_SLUG = "anthropic__claude-opus-4.6"


def _skill_md(kind: str, call: str, prompt: str) -> str:
    return (
        f"---\nkind: {kind}\ncall_name: {call}\n---\n"
        f"<!-- PROMPT_START -->\n{prompt}\n<!-- PROMPT_END -->\n"
    )


def _seed_skills(base: Path, *, call_prompt: str = "call prompt", model_slug: str | None = None) -> None:
    """Seed the real skills/mechanistic layout the runtime and the gate read."""
    mech = base / "skills" / "mechanistic"
    (mech / "base_system").mkdir(parents=True, exist_ok=True)
    (mech / "base_system" / "SKILL.md").write_text(_skill_md("shared_base", "base_system", "shared base"), encoding="utf-8")
    call_dir = mech / CALL
    call_dir.mkdir(parents=True, exist_ok=True)
    (call_dir / "SKILL.md").write_text(_skill_md("llm", CALL, call_prompt), encoding="utf-8")
    (call_dir / "few_shot.jsonl").write_text('{"input": "q", "output": "a"}\n', encoding="utf-8")
    if model_slug:
        override = call_dir / "models" / model_slug
        override.mkdir(parents=True, exist_ok=True)
        (override / "SKILL.md").write_text(_skill_md("llm", CALL, "model-specific prompt"), encoding="utf-8")


def _write_evidence(base: Path, bundle: str, name: str, extra: dict | None = None, *, model_name: str | None = None) -> Path:
    evidence_dir = base / "traces" / "evidence" / CALL / bundle
    evidence_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "approved_bool": True,
        "responder_saw_ground_truth": False,
        "prompt_version": {"prompt_bundle_sha256": bundle, "model_name": model_name},
        "model_version": {
            "model_version_id": "abc",
            "resolved_model_key": model_name or "gpt-5",
            "provider": "openai",
            "family": "openai",
            "pricing_sha256": "123",
        },
    }
    payload.update(extra or {})
    path = evidence_dir / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _bundle(base: Path, model_name: str | None = None) -> str:
    return str(get_call_prompt_version(CALL, base, model_name=model_name)["prompt_bundle_sha256"])


# --- change detection (pure) -------------------------------------------------


def test_calls_from_changed_paths_maps_call_files() -> None:
    changes = calls_from_changed_paths(
        [f"skills/mechanistic/{CALL}/SKILL.md", f"skills/mechanistic/{CALL}/few_shot.jsonl", "README.md"]
    )
    assert changes == [PromptChange(CALL, None, frozenset({"call_base", "few_shot"}))]


def test_calls_from_changed_paths_expands_shared_base_to_all_gated_calls() -> None:
    changes = calls_from_changed_paths(["skills/mechanistic/base_system/SKILL.md"])
    assert [c.call_name for c in changes] == sorted(CALL_TO_STEPS)
    assert all(c.model_name is None and c.components == frozenset({"shared_base"}) for c in changes)


def test_calls_from_changed_paths_extracts_model_lane() -> None:
    changes = calls_from_changed_paths([f"skills/mechanistic/propose_mechanism_step/models/{OPUS_SLUG}/few_shot.jsonl"])
    assert changes == [PromptChange("propose_mechanism_step", OPUS, frozenset({"few_shot"}))]
    assert changes[0].label == f"propose_mechanism_step@{OPUS}"


def test_calls_from_changed_paths_ignores_deterministic_unrelated_and_legacy_paths() -> None:
    assert (
        calls_from_changed_paths(
            [
                "skills/mechanistic/atom_balance_validation/validator.py",
                "skills/mechanistic/atom_balance_validation/SKILL.md",
                "skills/mechanistic/baseline_mechanism/SKILL.md",
                "skills/mechanistic/__init__.py",
                f"skills/mechanistic/{CALL}/notes.md",
                "prompt_versions/calls/assess_initial_conditions/base.md",
                "skills/project/alpha/SKILL.md",
            ]
        )
        == []
    )


def test_calls_from_changed_paths_detects_real_pr_history_layout() -> None:
    # Layout touched by PR #19 on this repo; the pre-repair regex matched nothing here.
    changes = calls_from_changed_paths(
        [
            "skills/mechanistic/attempt_atom_mapping/models/anthropic__claude-opus-4.8/few_shot.jsonl",
            "skills/mechanistic/propose_mechanism_step/models/anthropic__claude-opus-4.8/few_shot.jsonl",
        ]
    )
    assert [c.label for c in changes] == [
        "attempt_atom_mapping@anthropic/claude-opus-4.8",
        "propose_mechanism_step@anthropic/claude-opus-4.8",
    ]


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_discover_changed_calls_reads_git_diff(tmp_path: Path) -> None:
    _seed_skills(tmp_path)
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x"}

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True, env={**env, "PATH": __import__("os").environ["PATH"]})

    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "seed")
    (tmp_path / "skills" / "mechanistic" / CALL / "few_shot.jsonl").write_text('{"input": "q2", "output": "a2"}\n', encoding="utf-8")
    git("commit", "-q", "-am", "edit few-shot")

    assert discover_changed_calls(base_ref="HEAD~1", head_ref="HEAD", cwd=tmp_path) == [
        PromptChange(CALL, None, frozenset({"few_shot"}))
    ]


# --- evidence validation ---------------------------------------------------------


def test_validator_reads_seeded_prompt_and_passes_with_approved_linked_evidence(tmp_path: Path) -> None:
    _seed_skills(tmp_path)
    version = get_call_prompt_version(CALL, tmp_path)
    assert version["template"] == "call prompt"  # the seed is really being read
    bundle = str(version["prompt_bundle_sha256"])
    _write_evidence(tmp_path, bundle, "trace1.json")

    result = validate_evidence_for_calls(changed_calls=[CALL], base_dir=tmp_path)
    assert result.ok
    assert result.changed_calls == [CALL]
    assert result.valid_evidence_by_call[CALL] == [f"traces/evidence/{CALL}/{bundle}/trace1.json"]


def test_validator_rejects_evidence_that_saw_ground_truth_or_is_undeclared(tmp_path: Path) -> None:
    _seed_skills(tmp_path)
    bundle = _bundle(tmp_path)

    # Declared replay of the verified mechanism: not evidence of capability.
    _write_evidence(tmp_path, bundle, "replay.json", {"responder_saw_ground_truth": None, "origin": {"responder_saw_ground_truth": True}})
    # Missing declaration: also rejected until someone states it.
    _write_evidence(tmp_path, bundle, "undeclared.json", {"responder_saw_ground_truth": None})

    result = validate_evidence_for_calls(changed_calls=[CALL], base_dir=tmp_path)
    assert not result.ok
    assert any("responder_saw_ground_truth" in err for err in result.errors)

    # A blind run declared via the origin block is accepted.
    _write_evidence(tmp_path, bundle, "blind.json", {"responder_saw_ground_truth": None, "origin": {"responder_saw_ground_truth": False}})
    result_ok = validate_evidence_for_calls(changed_calls=[CALL], base_dir=tmp_path)
    assert result_ok.ok
    assert result_ok.valid_evidence_by_call[CALL] == [f"traces/evidence/{CALL}/{bundle}/blind.json"]


def test_validator_fails_without_evidence(tmp_path: Path) -> None:
    _seed_skills(tmp_path)
    result = validate_evidence_for_calls(changed_calls=[CALL], base_dir=tmp_path)
    assert not result.ok
    assert any("no evidence files found" in err for err in result.errors)


def test_validator_rejects_stale_bundle_with_reason(tmp_path: Path) -> None:
    _seed_skills(tmp_path)
    stale = _bundle(tmp_path)
    _write_evidence(tmp_path, stale, "old.json")
    _seed_skills(tmp_path, call_prompt="edited prompt")  # prompt changed after the evidence was exported

    result = validate_evidence_for_calls(changed_calls=[CALL], base_dir=tmp_path)
    assert not result.ok
    assert any("old.json: bundle" in err and "!= current" in err for err in result.errors)


def test_validator_matches_model_scoped_evidence_for_shared_change(tmp_path: Path) -> None:
    """Runtime records prompt versions per model; evidence must be matched in that scope."""
    _seed_skills(tmp_path)
    shared_bundle = _bundle(tmp_path)
    model_bundle = _bundle(tmp_path, "openai/gpt-5")
    assert shared_bundle != model_bundle

    _write_evidence(tmp_path, model_bundle, "gpt5.json", model_name="openai/gpt-5")
    result = validate_evidence_for_calls(changed_calls=[PromptChange(CALL, None, frozenset({"few_shot"}))], base_dir=tmp_path)
    assert result.ok, result.errors


def test_validator_model_lane_change_requires_matching_model_evidence(tmp_path: Path) -> None:
    _seed_skills(tmp_path, model_slug=OPUS_SLUG)
    _write_evidence(tmp_path, _bundle(tmp_path), "shared.json")  # shared-scope evidence

    lane_change = PromptChange(CALL, OPUS, frozenset({"call_base"}))
    result = validate_evidence_for_calls(changed_calls=[lane_change], base_dir=tmp_path)
    assert not result.ok
    assert result.changed_calls == [f"{CALL}@{OPUS}"]
    assert any("does not match changed lane" in err for err in result.errors)

    _write_evidence(tmp_path, _bundle(tmp_path, OPUS), "opus.json", model_name=OPUS)
    result_ok = validate_evidence_for_calls(changed_calls=[lane_change], base_dir=tmp_path)
    assert result_ok.ok, result_ok.errors
    assert list(result_ok.valid_evidence_by_call) == [f"{CALL}@{OPUS}"]


def test_validator_rejects_evidence_from_model_whose_override_shadows_changed_shared_prompt(tmp_path: Path) -> None:
    _seed_skills(tmp_path, model_slug=OPUS_SLUG)
    # Opus never reads the shared SKILL.md (it has its own), so its evidence cannot vouch for a shared SKILL.md change.
    _write_evidence(tmp_path, _bundle(tmp_path, OPUS), "opus.json", model_name=OPUS)

    result = validate_evidence_for_calls(changed_calls=[PromptChange(CALL, None, frozenset({"call_base"}))], base_dir=tmp_path)
    assert not result.ok
    assert any("overrides SKILL.md" in err for err in result.errors)

    # But the same evidence is fine for a few-shot change: the shared few_shot.jsonl still feeds the Opus bundle.
    result_ok = validate_evidence_for_calls(changed_calls=[PromptChange(CALL, None, frozenset({"few_shot"}))], base_dir=tmp_path)
    assert result_ok.ok, result_ok.errors


def test_validator_accepts_plain_strings_and_merges_duplicates(tmp_path: Path) -> None:
    _seed_skills(tmp_path)
    _write_evidence(tmp_path, _bundle(tmp_path), "t.json")
    result = validate_evidence_for_calls(
        changed_calls=[CALL, PromptChange(CALL, None, frozenset({"few_shot"})), CALL],
        base_dir=tmp_path,
    )
    assert result.ok
    assert result.changed_calls == [CALL]
    assert result.changes[0].components == frozenset({"few_shot"})


def test_get_call_prompt_version_rejects_path_traversal_call_name(tmp_path: Path) -> None:
    _seed_skills(tmp_path)
    with pytest.raises(ValueError):
        get_call_prompt_version("../../etc/passwd", tmp_path)


# --- CI constraint -----------------------------------------------------------------


def test_validator_import_chain_is_stdlib_only() -> None:
    """The CI workflow runs the gate without installing project dependencies.

    Static check: every module-level import reachable from the validator must be
    either standard library or one of the allow-listed lightweight modules. This
    holds in any environment, unlike a runtime check that depends on what the
    package ``__init__`` happens to swallow.
    """
    import ast

    repo_root = Path(__file__).resolve().parents[2]
    allowed_internal = {
        "mechanistic_agent.prompt_trace_validator",
        "mechanistic_agent.prompt_assets",
        "mechanistic_agent.data_paths",
    }
    pending = ["mechanistic_agent.prompt_trace_validator"]
    seen: set[str] = set()
    offenders: list[str] = []
    while pending:
        module = pending.pop()
        if module in seen:
            continue
        seen.add(module)
        source = (repo_root / (module.replace(".", "/") + ".py")).read_text(encoding="utf-8")
        for node in ast.parse(source).body:  # module level only; lazy imports inside functions are fine
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            for name in names:
                root = name.split(".")[0]
                if root == "__future__" or root in sys.stdlib_module_names:
                    continue
                if root == "mechanistic_agent":
                    if name in allowed_internal:
                        pending.append(name)
                    else:
                        offenders.append(f"{module} -> {name}")
                    continue
                offenders.append(f"{module} -> {name}")
    assert not offenders, "non-stdlib module-level imports in the gate chain: " + ", ".join(sorted(offenders))
