# Professor Wiggum — Mechanistic Agent

<img align="right" src="docs/readme_ralph.png" alt="Ralph" width="260" />

A local-first agent that predicts **arrow-pushing (elementary-step) mechanisms** for organic reactions. An LLM proposes each step; deterministic RDKit validators (bond/electron conservation, atom balance, state progress) decide whether the step is accepted. The harness — prompts, few-shots, module graph, validators — evolves only on eval evidence. See [SOUL.md](SOUL.md) for the philosophy.

## Current focus: Claude Opus 5.5

**Claude Opus 5.5** (`anthropic/claude-opus-5.5`) is the model we are developing against, and the default model for the product runtime. Two API differences from the 4.x line are handled in the adapters: forced `tool_choice` is rejected (the harness sends `auto` plus an explicit instruction and retries once before failing with `tool_call_missing`), and thinking cannot be disabled (`effort` only, default `medium`).

Latest development-tier results (FlowER-derived tiers, 10 cases each; not the official holdout):

| Date | How it ran | Harness | Easy | Medium | Hard |
|---|---|---|---|---|---|
| 2026-09-29 | Opus 5.5 via API | `default` | 940 · 10/10 | — | — |
| 2026-09-29 | Opus 5.5 blind via agent bridge † | `jev_reaction_type` | 939 · 10/10 | 904 · 10/10 | 823 · 10/10 targets, 9/10 pass |

Scores are out of 1000 (WIN ≥ 700). The one hard-tier fail reached the product with a correct acid-catalysed path; it was marked down because the end-of-run balance check does not yet recognise a catalyst that is regenerated (a second equivalent of acetic acid used as a proton shuttle). Fixing that reconciliation is the next harness item.

† Bridge runs are attributed to the `agent-bridge` model with a declared origin (`claude-opus-5-5`, headless `claude -p`, one fresh blind session per call, `responder_saw_ground_truth: false`). Their cost is opaque, so they are not eligible for cost-class claims. Older Opus 4.8 bridge rows were ground-truth replays, not capability measurements.

## Run types

All results land in the local SQLite database (`../wiggum-data/data/mechanistic.db` when the sibling data checkout exists; see [docs/DATA_SETUP.md](docs/DATA_SETUP.md)). Nothing is committed to git by running them.

| Run type | Invoke | Leaderboard |
|---|---|---|
| Single reaction | `python main.py run --starting "CCO" --products "CC=O"` (add `--mode verified` to submit your own steps) | none |
| Web UI | `python main.py serve` → `http://127.0.0.1:8010/` (unverified mode only; `/?run=<id>` replays a run) | none |
| Harness dev eval | `python main.py eval --tier easy\|medium\|hard --model <id> [--harness <name>]` or `--all-tiers` | development leaderboard (local): `python main.py leaderboard --eval-set-id <id>`, UI, `GET /api/evals/leaderboard` |
| Harness-free baseline | `python main.py baseline --tier <tier> --model <id>` (or `--all-tiers`) | development leaderboard, type `Baseline` |
| Official holdout | `python main.py eval-runset-official --model-name <id>` and `baseline-runset-official` | official leaderboard: `python main.py leaderboard-official`; `python main.py update-leaderboard-artifacts` writes [LEADERBOARD.md](LEADERBOARD.md) |
| Keyless (agent bridge) | any of the above with `--model agent-bridge`, answered by `python main.py bridge-serve --command "<responder>"` | same leaderboards, model column `agent-bridge` † |
| Curriculum checkpoints | `python main.py curriculum submit\|publish\|render-readme --model-name <id>` | `curriculum/generated/leaderboard_*.json` |
| Harness evolution | `python scripts/evolve_harness.py [--island-mode]`, `python main.py overnight-ralph`, `python main.py vote` | evolution archive (holdout sets are rejected) |

Notes:
- `eval --tier` goes through the development-leaderboard planner, which may re-route a run (for example to repeat the current tier). Pass `--leaderboard-route next` to move up a tier or `custom` to keep your own selection; see [docs/development_leaderboard_routes.md](docs/development_leaderboard_routes.md).
- Only official-holdout runs reach the committed `LEADERBOARD.md` table. Development evals — including the Opus 5.5 rows above — live in your local database.
- Leaderboards drop eval runs whose responder declares it saw the ground truth.
- Custom eval sets without FlowER data: [docs/custom_eval_sets.md](docs/custom_eval_sets.md). Keyless runs: [docs/agent_bridge.md](docs/agent_bridge.md).

## Contributing

There are three ways to contribute (details in [CONTRIBUTING.md](CONTRIBUTING.md)):

1. **Report a chemistry problem** — open a *Chemistry failure* issue with the starting materials, products, and what went wrong. That alone is a complete contribution.
2. **Report a bug or idea** — open a *Bug report* or *Feature request* issue.
3. **Submit code** — PRs are welcome; the only check you need is `python -m pytest tests/fast/ -q`.

Changes to prompts, few-shots, models, validators, or harness behaviour are evidence-gated by the maintainers before merge ([docs/change_evidence_policy.md](docs/change_evidence_policy.md)); you do not need to run model evals yourself. If you work through an agent with no API key, you can still produce runs and traces via the [agent bridge](docs/agent_bridge.md) — declare the responder's origin and ground-truth exposure so the runs can be used as evidence. Maintainer playbooks: [docs/agent_playbooks.md](docs/agent_playbooks.md).

## Curriculum status

<!-- curriculum-status:start -->
## Program Status

- Course: `Mechanistic Curriculum`
- Launch: `2026-03-11`
- Module: `Module 1` — climbing the difficulty chain (easy 1–2 step → medium 3-step → hard 4+ step)

**Trainees:** [anthropic__claude-opus-4-5](skills/mechanistic/propose_mechanism_step/models/anthropic__claude-opus-4-5/) | [anthropic__claude-opus-4.6](skills/mechanistic/propose_mechanism_step/models/anthropic__claude-opus-4.6/) | [anthropic__claude-opus-4.8](skills/mechanistic/propose_mechanism_step/models/anthropic__claude-opus-4.8/)

Quick links: [Checkpoints](curriculum/checkpoints/) | [Reactions](training_data/flower_curriculum_pngs/index.json) | [Prompt guide](docs/model_asset_overrides.md) | [History](docs/history_and_reproducibility.md)

Curriculum checkpoints and trainee lanes advance **as time permits**. There is no public release clock; use the CLI below when you are ready to queue or publish work.

## Trainee Progress Snapshot

- See [curriculum/generated/](curriculum/generated/) for per-lane leaderboard rows.

## Checkpoints
<!-- curriculum-status:end -->

`python main.py curriculum render-readme --model-name <id>` refreshes only the block above. To inspect a past milestone, open its manifest under `curriculum/checkpoints/`, check out the recorded tag or commit, and compare the resolved prompt and few-shot hashes to the current trainee lane.

## Developer

### Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e . --no-deps
python -m pytest tests/fast/ -q
python main.py serve
```

Full setup: [SETUP.md](SETUP.md). Data layout (sibling `wiggum-data` checkout or in-repo fallback): [docs/DATA_SETUP.md](docs/DATA_SETUP.md). Runtime architecture, endpoints, and conventions: [AGENTS.md](AGENTS.md).

### Harness workflow

![Harness flow diagram](docs/diagrams/Harness_Configuration_Flowchart.png)

- **Pre-loop** (runs once): check atom balance → identify functional groups → recommend pH → assess conditions → predict missing reagents → map atoms → map to reaction type (LLM, or a Jev decision in the `jev_reaction_type` harness)
- **Loop**: propose next mechanism step (LLM) → bond/electron, atom-balance, and state-progress validators → retry, backtrack, or continue → target products reached?
- **Post-loop**: overall balance reconciliation and re-evaluation of any soft-advanced steps

Harness variants live under `harness_versions/` (`default`, `jev_reaction_type`, `permissive_default`, ablations). Regenerate the diagram with `python scripts/capture_harness_mermaid.py`.

### Docs

- Prompt/few-shot overrides: [docs/model_asset_overrides.md](docs/model_asset_overrides.md)
- History and reproducibility: [docs/history_and_reproducibility.md](docs/history_and_reproducibility.md)
- Evidence policy: [docs/change_evidence_policy.md](docs/change_evidence_policy.md)

The 1000-point score rubric originated with the Clawdiators arena; no arena data is currently coming in, and the historical arena material is kept in [LEADERBOARD.md](LEADERBOARD.md).
