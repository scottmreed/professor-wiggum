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
    assert "81.23" not in markdown
    assert "0.812" in markdown
    assert "66.0%" in markdown
    assert "| Rank | Model | Thinking | Type | Quality | Pass | Cases | Group |" in markdown
