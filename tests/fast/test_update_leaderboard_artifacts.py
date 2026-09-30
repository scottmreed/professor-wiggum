from __future__ import annotations

from pathlib import Path

from main import (
    _ARENA_TABLE_HEADER,
    _arena_table_from_leaderboard_items,
    _find_arena_table_span,
    _replace_arena_table_in_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LEADERBOARD_MD = REPO_ROOT / "LEADERBOARD.md"

MINIMAL_FIXTURE = """\
# Mechanistic Agent Leaderboard

## Arena Thresholds (mechanistic-easy, 1000 pts max)

| Score | Outcome |
|---|---|
| ≥ 700 | **WIN** |

---

## Arena Submissions (Harness Eval — 1000-pt scale)

*Auto-generated note — do not remove.*

| Date | Model | Score | Outcome | Pass Rate | Avg Latency | Run Group |
|---|---|---|---|---|---|---|
| 2026-01-01 | `old-model` | 100/1000 | LOSS | 0.0% | — | `old_group` |

† Hand-written footnote that must survive table refresh.

- Bullet detail that must also survive.

### Speed Calibration

Speed scoring uses calibration constants.

## Lessons Learned

Do not touch this section.
"""


def test_find_arena_table_span_parses_committed_leaderboard_md() -> None:
    content = LEADERBOARD_MD.read_text(encoding="utf-8")
    span = _find_arena_table_span(content)
    assert span is not None
    start, end = span
    table_block = content[start:end]
    assert table_block.startswith(_ARENA_TABLE_HEADER)
    assert "2026-03-16" in table_block
    assert "anthropic/claude-opus-4.6" in table_block
    assert "† **agent-bridge" not in table_block


def test_replace_arena_table_preserves_hand_written_sections() -> None:
    new_table = _arena_table_from_leaderboard_items(
        [
            {
                "created_at": 1704067200.0,
                "model_name": "new-model",
                "weighted_pass_rate": 0.5,
                "avg_latency_ms": 4200.0,
                "run_group_name": "new_group",
                "mean_quality_score": 0.5,
            }
        ]
    )
    updated = _replace_arena_table_in_markdown(MINIMAL_FIXTURE, new_table)
    assert updated is not None
    assert "## Arena Thresholds" in updated
    assert "† Hand-written footnote" in updated
    assert "- Bullet detail that must also survive." in updated
    assert "### Speed Calibration" in updated
    assert "## Lessons Learned" in updated
    assert "*Auto-generated note — do not remove.*" in updated
    assert "`new-model`" in updated
    assert "`old-model`" not in updated
    assert "Do not touch this section." in updated


def test_replace_arena_table_idempotent_on_fixture() -> None:
    new_table = _arena_table_from_leaderboard_items([])
    first = _replace_arena_table_in_markdown(MINIMAL_FIXTURE, new_table)
    assert first is not None
    second = _replace_arena_table_in_markdown(first, new_table)
    assert second == first
