from __future__ import annotations

import pytest

pytest.importorskip("rdkit")
from main import _filter_leaderboard_rows, _render_leaderboard_markdown


def test_filter_leaderboard_rows_excludes_incomplete_runs_by_default() -> None:
    items = [
        {"status": "completed", "model_name": "gpt-5"},
        {"status": "running", "model_name": "gpt-5-mini"},
    ]

    filtered = _filter_leaderboard_rows(items, completed_only=True)

    assert filtered == [{"status": "completed", "model_name": "gpt-5"}]


def test_render_leaderboard_markdown_includes_sota_and_table() -> None:
    markdown = _render_leaderboard_markdown(
        "eval-123",
        [
            {
                "model_name": "gpt-5",
                "thinking_level": "high",
                "mean_quality_score": 0.8123,
                "deterministic_pass_rate": 0.66,
                "run_group_name": "medium_default_2026-03-02",
                "case_count": 10,
                "is_baseline": False,
            }
        ],
        generated_at="2026-03-02 12:00:00",
    )

    assert "# Mechanistic Agent Leaderboard" in markdown
    assert "Current SOTA" in markdown
    assert "`gpt-5`" in markdown
    assert "804/1000" in markdown
    assert "66.0%" in markdown
    assert "| Rank | Model | Thinking | Type | Score | Outcome | Pass | Cases | Group |" in markdown


def _row(**overrides):
    row = {
        "model_name": "anthropic/claude-opus-5.5",
        "thinking_level": "medium",
        "mean_quality_score": 0.8,
        "deterministic_pass_rate": 0.6,
        "run_group_name": "grp",
        "case_count": 10,
        "is_baseline": False,
    }
    row.update(overrides)
    return row


def test_render_leaderboard_markdown_marks_bridge_rows_under_declared_model() -> None:
    markdown = _render_leaderboard_markdown(
        "eval-123",
        [
            _row(model="agent-bridge", via_bridge=True, bridge_model="agent-bridge", run_group_name="bridge"),
            _row(model="anthropic/claude-opus-5.5", via_bridge=False, run_group_name="api"),
        ],
        generated_at="2026-03-02 12:00:00",
    )

    table_rows = [line for line in markdown.splitlines() if line.startswith("| ") and "`anthropic/claude-opus-5.5`" in line]
    bridge_row = next(line for line in table_rows if "`bridge`" in line)
    api_row = next(line for line in table_rows if "`api`" in line)
    assert "`anthropic/claude-opus-5.5` †" in bridge_row
    assert "†" not in api_row
    assert "listed under the responder's declared model" in markdown
    assert "opaque" in markdown


def test_render_leaderboard_markdown_falls_back_to_bridge_model_name() -> None:
    markdown = _render_leaderboard_markdown("eval-123", [_row(model_name="agent-bridge")])

    assert "`agent-bridge` †" in markdown
    assert "Agent-bridge origin" in markdown


def test_render_leaderboard_markdown_has_no_bridge_footnote_for_api_rows() -> None:
    markdown = _render_leaderboard_markdown("eval-123", [_row(via_bridge=False)])

    assert "†" not in markdown
