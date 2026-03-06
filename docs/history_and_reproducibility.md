# History and Reproducibility

Published curriculum checkpoints are recorded in two places:

- SQLite via `curriculum_checkpoints`
- repo manifests under `curriculum/checkpoints/<YYYY>/<YYYY-MM-DD>.json`

Each checkpoint captures:

- release date and kind
- exact trainee lane
- eval run id
- harness hash
- resolved prompt and few-shot asset paths and hashes
- git branch, tag, and commit when available
- manifest path suitable for direct linking from the README and history UI

## Inspecting upcoming curriculum

- The next 2 weeks of scheduled releases are written to `curriculum/generated/readme_context.json`.
- The public README links directly to the upcoming curriculum reactions index in `training_data/flower_curriculum_pngs/index.json`.
- Prompt and few-shot override locations are documented in `docs/model_asset_overrides.md` so observers can compare how a trainee lane changed over time.

## Replaying a milestone

1. Open the checkpoint manifest for the date you care about.
2. Check out the recorded git tag or commit.
3. Inspect the manifest's prompt and harness hashes to confirm the runtime state.
4. Re-run local evals from that checkout if deeper verification is needed.
