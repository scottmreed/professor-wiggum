# Local data checkout (wiggum-data)

Bulk artifacts (SQLite DB, FlowER indices, PNG caches, run traces) live in a
**separate sibling repository** by default:

```
../wiggum-data/
```

The Mechanistic app resolves this automatically when that directory exists.
Forks and CI can keep data in-repo instead — see fallbacks in
`mechanistic_agent/data_paths.py`.

## Override path

```bash
export MECHANISTIC_DATA_DIR=/path/to/your/data
```

Also accepted: `WIGGUM_DATA_DIR`.

## What stays in this repo

- Committed eval JSON under `training_data/` (`eval_set.json`, `eval_tiers.json`, …)
- PR evidence traces under `traces/evidence/`
- Small chemistry helpers under `data/` (`checkmol_smarts_part1.csv`, …)

## What moves to wiggum-data

| Artifact | wiggum-data path |
|----------|------------------|
| Runtime DB | `data/mechanistic.db` |
| Lookup caches | `data/flower_*_lookup.sqlite` |
| Full mechanism index | `training_data/flower_mechanism_index.jsonl` |
| Holdout suite | `training_data/leaderboard_holdout/` |
| Curriculum PNGs | `training_data/flower_curriculum_pngs/` |
| Run traces | `traces/runs/` |

## Setup for maintainers

1. Clone `wiggum-data` as a sibling of this repo.
2. Copy or regenerate `data/mechanistic.db` locally — it is **gitignored** in
   wiggum-data because it exceeds GitHub’s 100 MB per-file limit (~160 MB).
3. Run the app from `professor-wiggum` as usual; no code changes needed when
   the sibling layout is present.

## Setup for forkers

No separate checkout needed. The app falls back to `data/` and `training_data/`
inside this repo when no sibling `wiggum-data` directory exists. Generate the
SQLite lookups and mechanism index from the FlowER dataset as described in
[training_data/REGENERATE.md](../training_data/REGENERATE.md).

Running evals and the fast test suite works immediately after clone — no bulk
data is required for those.

Regeneration instructions: [training_data/REGENERATE.md](../training_data/REGENERATE.md).
