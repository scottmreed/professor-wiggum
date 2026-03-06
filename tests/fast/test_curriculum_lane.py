from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

pytest.importorskip("rdkit")
from mechanistic_agent.core.db import RunStore
from mechanistic_agent.curriculum import build_curriculum_status, publish_curriculum_release, render_curriculum_readme


def _seed_curriculum_base(tmp_path: Path) -> Path:
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "curriculum").mkdir(parents=True, exist_ok=True)
    (tmp_path / "skills" / "mechanistic" / "base_system").mkdir(parents=True, exist_ok=True)
    (tmp_path / "training_data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "harness_versions" / "default").mkdir(parents=True, exist_ok=True)
    (tmp_path / "skills" / "mechanistic" / "base_system" / "SKILL.md").write_text(
        "---\nkind: shared_base\ncall_name: base_system\n---\n<!-- PROMPT_START -->\nbase\n<!-- PROMPT_END -->\n",
        encoding="utf-8",
    )
    (tmp_path / "harness_versions" / "default" / "harness.json").write_text(
        json.dumps(
            {
                "name": "default",
                "schema_version": "2.0",
                "tool_calling_mode": "forced",
                "few_shot_defaults": {"enabled": True, "max_examples": 4, "selection_strategy": "top_score"},
                "pre_loop_modules": [],
                "loop_module": {"id": "mechanism_step_proposal", "step_name": "mechanism_step_proposal", "prompt_call_name": "propose_mechanism_step"},
                "post_step_modules": [],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "training_data" / "flower_mechanism_index.jsonl").write_text("", encoding="utf-8")
    return tmp_path


def test_curriculum_queue_and_publish_creates_checkpoint_manifest(tmp_path: Path) -> None:
    base = _seed_curriculum_base(tmp_path)
    store = RunStore(base / "data" / "mechanistic.db")
    queue_id = store.queue_curriculum_release(
        release_date="2026-03-04",
        model_name="anthropic/claude-opus-4.6",
        module_id="module_01",
        release_kind="lesson",
        scheduled_publish_at=0.0,
        eval_run_id=None,
        payload={
            "module": {"id": "module_01", "number": 1, "label": "1-step reactions"},
            "mean_quality_score": 0.75,
            "pass_count": 3,
            "case_count": 4,
            "selected_case_ids": ["flower_000001"],
            "quiz_passed": True,
            "prompt_assets": {},
            "harness_sha": "abc123",
            "leaderboard_row": {},
        },
    )

    published = publish_curriculum_release(base, store, queue_id=queue_id, force=True)
    manifest_path = base / str(published["manifest_path"])
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["release_date"] == "2026-03-04"
    assert manifest["summary"]["mean_quality_score"] == 0.75

    second = publish_curriculum_release(base, store, queue_id=queue_id, force=False)
    assert second["id"] == published["id"]


def test_curriculum_status_and_readme_render_include_queued_release(tmp_path: Path) -> None:
    base = _seed_curriculum_base(tmp_path)
    store = RunStore(base / "data" / "mechanistic.db")
    store.queue_curriculum_release(
        release_date="2099-01-05",
        model_name="anthropic/claude-opus-4.6",
        module_id="module_01",
        release_kind="lesson",
        scheduled_publish_at=4102444800.0,
        eval_run_id=None,
        payload={"module": {"id": "module_01", "number": 1, "label": "1-step reactions"}},
    )

    status = build_curriculum_status(base, store, model_name="anthropic/claude-opus-4.6")
    assert status["current_module"]["id"] == "module_01"

    content = render_curriculum_readme(base, store, model_name="anthropic/claude-opus-4.6")
    assert "# Mechanistic Curriculum" in content
    assert "How to Inspect Any Past Milestone" in content


def test_curriculum_uses_march_11_launch_and_two_week_calendar(tmp_path: Path) -> None:
    base = _seed_curriculum_base(tmp_path)
    store = RunStore(base / "data" / "mechanistic.db")
    before_launch = datetime(2026, 3, 4, 12, 0, tzinfo=ZoneInfo("America/Denver"))

    status = build_curriculum_status(base, store, model_name="anthropic/claude-opus-4.6", now=before_launch)

    assert status["today_slot"] is None
    assert status["next_slot"]["release_date"] == "2026-03-11"
    assert status["next_slot"]["countdown"]["days"] >= 6
    assert len(status["weekly_checklist"]) == 10

    content = render_curriculum_readme(base, store, model_name="anthropic/claude-opus-4.6", now=before_launch)
    assert "Public launch date: `2026-03-11`" in content
    assert "## Current Two-Week Calendar" in content
    assert "Countdown:" not in content
