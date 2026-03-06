# Curriculum Operations

The trainee-first curriculum is driven locally. Phase 1 focuses on the Claude Opus trainee lane and starts on `2026-03-11`.

## Daily flow

1. Submit the day's lesson or quiz:
   `python main.py curriculum submit --model-name anthropic/claude-opus-4.6`
2. Publish due releases at or after 5:00 PM America/Denver:
   `python main.py curriculum publish-due`
3. Regenerate the course dashboard manually if needed:
   `python main.py curriculum render-readme`

## Public cadence

- The public curriculum calendar shows the next 2 weeks of weekday releases.
- Monday through Thursday are lesson releases.
- Friday is the quiz release that determines whether the trainee advances.
- Before launch, the calendar points forward to the `2026-03-11` start date.

## Queue and publish

- `submit` runs the scheduled batch and stores a queued release in SQLite.
- `publish-due` writes the checkpoint manifest, updates `curriculum/generated/`, renders `README.md`, and records git metadata when available.
- `publish --checkpoint-id <queued-release-id> --force` bypasses the due-time check for recovery workflows.

## Contribution paths

- Single reaction: submit one reaction through the UI or API and use it as evidence for a later PR.
- Few-shot contribution: update a trainee-specific or shared `few_shot.jsonl` file and link evidence.
- Prompt contribution: update `SKILL.md` for a shared skill or a trainee override.
- Harness contribution: add or revise a harness under `harness_versions/` and link eval deltas.
- New trainee lane: add exact-model overrides under `skills/mechanistic/<call_name>/models/<model-slug>/`.

## Local scheduler

Generate a launchd example plist with:

`python main.py curriculum install-launchd`

The command writes a plist example to `/tmp/mechanistic_curriculum_publish.plist` by default so it can be reviewed before installation.

## GitHub README refresh

The repo includes a manual GitHub Action that can refresh `README.md` and generated curriculum JSON files on demand.
