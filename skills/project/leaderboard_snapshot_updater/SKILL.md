---
skill_type: project
call_name: leaderboard_snapshot_updater
kind: workflow
phase: maintenance
version: 1
---

# Leaderboard Snapshot Updater

Update the leaderboard snapshot PNG in the README when leaderboard data changes.

## When to Use

- After running `python main.py eval-runset-official` or `python main.py baseline-runset-official`
- After running `python main.py update-leaderboard-artifacts`
- When asked to refresh the leaderboard image in the README

## Steps

1. Run `python scripts/update_leaderboard_snapshot.py`
2. Verify `docs/leaderboard_snapshot.png` is updated
3. Commit the new image if desired

## Dependencies

- `matplotlib` must be installed (`pip install matplotlib`)
- Script reads from `data/mechanistic.db` — same database used by the API server
- Uses the same official holdout resolution as `GET /api/evals/leaderboard/official`

## Output

Generates `docs/leaderboard_snapshot.png` — a grouped bar chart (harness vs baseline) organized by model, sorted by best harness score. Referenced in the README under the "Leaderboard Snapshot" section.
