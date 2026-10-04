"""scripts/chemillusion_runtime_bump.py: when to bump ChemIllusion's runtime pin and product model."""

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
PRODUCT_MODEL = '"""Product model."""\n\nPRODUCT_MODEL = "anthropic/claude-opus-5.5"\n'
LEGACY_CONFIG = """class Settings:
    MECHANISM_PREDICTOR_MODEL: str = "anthropic/claude-opus-5.5"
"""
OPUS_LANE = "skills/mechanistic/propose_mechanism_step/models/anthropic__claude-opus-5.5/few_shot.jsonl"
MINOR_LANE = "skills/mechanistic/propose_mechanism_step/models/openai__gpt-4o-mini/few_shot.jsonl"


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
    catalog = {"anthropic/claude-opus-5.5": {}, "anthropic/claude-fable-5.1": {}, "openai/gpt-4o-mini": {}}
    _commit(
        repo,
        {
            "mechanistic_agent/model_pricing.json": json.dumps(catalog),
            "README.md": "x",
            "results/runs/m.json": _record("medium", "anthropic/claude-opus-5.5", 904),
            "results/runs/h.json": _record("hard", "anthropic/claude-opus-5.5", 823),
        },
        "base",
    )
    return repo


def _plan(repo: Path, old: str, model_text: str = PRODUCT_MODEL) -> dict:
    return bump.plan_bump(repo, "HEAD", DOCKERFILE.format(ref=old), model_text)


def test_harness_changes_bump_and_are_grouped_by_kind(repo: Path) -> None:
    old = _git(repo, "rev-parse", "HEAD")
    _commit(repo, {"README.md": "docs", "mechanistic_agent/results_publish.py": "# board"}, "docs + leaderboard code")
    _commit(repo, {"skills/mechanistic/propose_mechanism_step/few_shot.jsonl": "{}\n"}, "feat(prompt): shared few-shot")
    _commit(repo, {OPUS_LANE: "{}\n"}, "feat(prompt): opus 5.5 lane")
    _commit(repo, {"harness_versions/default/harness.json": "{}"}, "feat(harness): tweak")
    new = _commit(repo, {"mechanistic_agent/core/coordinator.py": "# c"}, "fix(core): coordinator")

    plan = _plan(repo, old)

    assert plan["changed"] is True and plan["new_ref"] == new and plan["reasons"] == ["harness_change"]
    assert [c["subject"] for c in plan["commits"]] == [
        "fix(core): coordinator", "feat(harness): tweak", "feat(prompt): opus 5.5 lane", "feat(prompt): shared few-shot",
    ]
    assert plan["kinds"] == ["frontier_lane", "harness_code", "harness_config", "prompts_fewshots"]
    dockerfile, model = bump.apply_plan(plan, DOCKERFILE.format(ref=old), PRODUCT_MODEL)
    assert f"ARG WIGGUM_RUNTIME_REF={new}\n" in dockerfile and model == PRODUCT_MODEL
    assert "Harness changes" in bump.pr_body(plan)


def test_minor_model_lanes_docs_and_non_harness_code_do_not_bump(repo: Path) -> None:
    old = _git(repo, "rev-parse", "HEAD")
    _commit(repo, {MINOR_LANE: "{}\n"}, "feat(prompt): gpt-4o-mini lane")
    _commit(repo, {"mechanistic_agent/results_publish.py": "# x", "docs/x.md": "y"}, "leaderboard + docs")
    _commit(repo, {"mechanistic_agent/core/db.py": "# leaderboard storage"}, "fix(db): leaderboard rows")
    _commit(repo, {"results/runs/minor.json": _record("hard", "openai/gpt-4o-mini", 700)}, "results: minor model")
    plan = _plan(repo, old)
    assert plan["changed"] is False and plan["new_ref"] == old and plan["reasons"] == []


def test_frontier_improvement_on_medium_or_hard_bumps(repo: Path) -> None:
    old = _git(repo, "rev-parse", "HEAD")
    _commit(repo, {"results/runs/h2.json": _record("hard", "anthropic/claude-opus-5.5", 880, date="2026-10-04")}, "results")
    plan = _plan(repo, old)
    assert plan["reasons"] == ["frontier_improvement"] and plan["changed"] is True
    assert plan["frontier_improvements"] == {"hard": {"before": 823, "after": 880}}
    assert "hard: 823 → 880/1000" in bump.pr_body(plan)


def test_frontier_regression_or_easy_only_result_does_not_bump(repo: Path) -> None:
    old = _git(repo, "rev-parse", "HEAD")
    _commit(
        repo,
        {
            "results/runs/h2.json": _record("hard", "anthropic/claude-opus-5.5", 800),
            "results/runs/e.json": _record("easy", "anthropic/claude-opus-5.5", 990),
        },
        "results",
    )
    assert _plan(repo, old)["changed"] is False


def test_new_leader_on_medium_and_hard_switches_the_product_model(repo: Path) -> None:
    old = _git(repo, "rev-parse", "HEAD")
    bridge = {"origin": {"responder": "agent-bridge", "declared_underlying_model": "claude-fable-5-1 (blind)"}}
    _commit(
        repo,
        {
            "results/runs/c.json": _record("medium", "agent-bridge", 950, date="2026-10-05", **bridge),
            "results/runs/d.json": _record("hard", "agent-bridge", 870, date="2026-10-05", **bridge),
            "results/runs/e.json": _record("hard", "openai/gpt-4o-mini", 999, kind="baseline"),
        },
        "results: fable",
    )
    plan = _plan(repo, old)
    assert "new_frontier_model" in plan["reasons"] and plan["frontier_model"] == "anthropic/claude-fable-5.1"
    _, model = bump.apply_plan(plan, DOCKERFILE.format(ref=old), PRODUCT_MODEL)
    assert 'PRODUCT_MODEL = "anthropic/claude-fable-5.1"' in model
    body = bump.pr_body(plan)
    assert "Product model" in body and "Railway" not in body
    # The legacy settings field is rewritten the same way until ChemIllusion moves to PRODUCT_MODEL.
    _, legacy = bump.apply_plan(_plan(repo, old, LEGACY_CONFIG), DOCKERFILE.format(ref=old), LEGACY_CONFIG)
    assert 'MECHANISM_PREDICTOR_MODEL: str = "anthropic/claude-fable-5.1"' in legacy


def test_split_leadership_keeps_the_product_model(repo: Path) -> None:
    old = _git(repo, "rev-parse", "HEAD")
    _commit(repo, {"results/runs/c.json": _record("medium", "anthropic/claude-fable-5.1", 950)}, "results: split")
    plan = _plan(repo, old)
    assert plan["model_recommendation"] is None and plan["changed"] is False


def test_unknown_pin_line_is_an_error(repo: Path) -> None:
    with pytest.raises(ValueError):
        bump.plan_bump(repo, "HEAD", "FROM python\nARG WIGGUM_RUNTIME_REF=main\n", PRODUCT_MODEL)
