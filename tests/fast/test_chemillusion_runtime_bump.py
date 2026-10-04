"""scripts/chemillusion_runtime_bump.py: plan and apply ChemIllusion runtime bumps."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("chemillusion_runtime_bump", ROOT / "scripts" / "chemillusion_runtime_bump.py")
bump = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bump)  # type: ignore[union-attr]

DOCKERFILE = """FROM python:3.11-bookworm AS wiggum
ARG WIGGUM_RUNTIME_REF={ref}
RUN git clone https://github.com/scottmreed/professor-wiggum.git /opt/wiggum
"""
CONFIG = """class Settings:
    MECHANISM_PREDICTOR_WIGGUM_DIR: str = "/opt/wiggum"
    MECHANISM_PREDICTOR_MODEL: str = "anthropic/claude-opus-5.5"
    MECHANISM_PREDICTOR_MAX_CONCURRENT: int = 1
"""


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def _commit(repo: Path, files: dict[str, str], message: str) -> str:
    for rel, text in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _record(tier: str, model: str, points: int, *, date: str = "2026-10-01", **extra: object) -> str:
    return json.dumps({"tier": tier, "model": model, "date": date, "holdout": False, "summary": {"points": points}, **extra})


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "wiggum"
    repo.mkdir()
    _git(repo, "init", "-q")
    catalog = {"anthropic/claude-opus-5.5": {}, "anthropic/claude-fable-5.1": {}, "agent-bridge": {}}
    _commit(repo, {"mechanistic_agent/model_pricing.json": json.dumps(catalog), "README.md": "x"}, "base")
    return repo


def test_plan_lists_runtime_commits_by_kind_and_ignores_docs(repo: Path) -> None:
    old = _git(repo, "rev-parse", "HEAD")
    _commit(repo, {"README.md": "docs only"}, "docs: readme")
    _commit(repo, {"skills/mechanistic/propose_mechanism_step/few_shot.jsonl": "{}\n"}, "feat(prompt): few-shot")
    _commit(repo, {"harness_versions/default/harness.json": "{}"}, "feat(harness): tweak")
    new = _commit(repo, {"mechanistic_agent/core/coordinator.py": "# c"}, "fix(core): coordinator")

    plan = bump.plan_bump(repo, "HEAD", DOCKERFILE.format(ref=old), CONFIG)

    assert plan["old_ref"] == old and plan["new_ref"] == new and plan["changed"] is True
    assert [c["subject"] for c in plan["commits"]] == [
        "fix(core): coordinator", "feat(harness): tweak", "feat(prompt): few-shot",
    ]
    assert plan["kinds"] == ["harness", "prompts_fewshots", "runtime_code"]
    assert plan["model_recommendation"] is None
    dockerfile, config = bump.apply_plan(plan, DOCKERFILE.format(ref=old), CONFIG)
    assert f"ARG WIGGUM_RUNTIME_REF={new}\n" in dockerfile
    assert config == CONFIG
    body = bump.pr_body(plan)
    assert "Prompts / few-shots" in body and "Harness configs" in body and "Product model" not in body


def test_docs_only_changes_do_not_bump(repo: Path) -> None:
    old = _git(repo, "rev-parse", "HEAD")
    _commit(repo, {"README.md": "docs only", "docs/x.md": "y"}, "docs")
    plan = bump.plan_bump(repo, "HEAD", DOCKERFILE.format(ref=old), CONFIG)
    assert plan["changed"] is False and plan["new_ref"] == old and plan["commits"] == []


def test_new_leader_on_medium_and_hard_proposes_product_model(repo: Path) -> None:
    old = _git(repo, "rev-parse", "HEAD")
    bridge = {"origin": {"responder": "agent-bridge", "declared_underlying_model": "claude-fable-5-1 (blind)"}}
    _commit(
        repo,
        {
            "results/runs/a.json": _record("medium", "anthropic/claude-opus-5.5", 904),
            "results/runs/b.json": _record("hard", "anthropic/claude-opus-5.5", 823),
            "results/runs/c.json": _record("medium", "agent-bridge", 950, date="2026-10-05", **bridge),
            "results/runs/d.json": _record("hard", "agent-bridge", 870, date="2026-10-05", **bridge),
            # Baselines and holdout never crown a leader.
            "results/runs/e.json": _record("hard", "gpt-5", 999, kind="baseline"),
        },
        "results: fable",
    )
    plan = bump.plan_bump(repo, "HEAD", DOCKERFILE.format(ref=old), CONFIG)
    assert plan["changed"] is True
    assert plan["model_recommendation"]["model"] == "anthropic/claude-fable-5.1"
    _, config = bump.apply_plan(plan, DOCKERFILE.format(ref=old), CONFIG)
    assert 'MECHANISM_PREDICTOR_MODEL: str = "anthropic/claude-fable-5.1"' in config
    assert "Product model" in bump.pr_body(plan)


def test_split_leadership_or_current_model_leading_keeps_the_product_model(repo: Path) -> None:
    old = _git(repo, "rev-parse", "HEAD")
    _commit(
        repo,
        {
            "results/runs/a.json": _record("medium", "anthropic/claude-fable-5.1", 950),
            "results/runs/b.json": _record("hard", "anthropic/claude-opus-5.5", 900),
        },
        "results: split",
    )
    plan = bump.plan_bump(repo, "HEAD", DOCKERFILE.format(ref=old), CONFIG)
    assert plan["model_recommendation"] is None and plan["changed"] is False


def test_unknown_pin_line_is_an_error(repo: Path) -> None:
    with pytest.raises(ValueError):
        bump.plan_bump(repo, "HEAD", "FROM python\nARG WIGGUM_RUNTIME_REF=main\n", CONFIG)
