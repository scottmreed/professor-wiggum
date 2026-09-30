# Mechanistic Curriculum

<img align="right" src="../docs/readme_ralph.png" alt="Ralph" width="260" />

## Orchestration Modes

RAlph mode provides iterative multi-attempt orchestration with budget controls for enhanced mechanism prediction reliability.

## Program Status

- Course: `Mechanistic Curriculum`
- Launch: `2026-03-11`
- Module: `Module 1` — 1-step reactions

**Trainees:** [anthropic__claude-opus-4-5](../skills/mechanistic/propose_mechanism_step/models/anthropic__claude-opus-4-5/) | [anthropic__claude-opus-4.6](../skills/mechanistic/propose_mechanism_step/models/anthropic__claude-opus-4.6/) | [anthropic__claude-opus-4.8](../skills/mechanistic/propose_mechanism_step/models/anthropic__claude-opus-4.8/)

Quick links: [Checkpoints](../curriculum/checkpoints/) | [Reactions](../training_data/flower_curriculum_pngs/index.json) | [Prompt guide](../docs/model_asset_overrides.md) | [History](../docs/history_and_reproducibility.md)

Curriculum checkpoints and trainee lanes advance **as time permits**. There is no public release clock; use the CLI below when you are ready to queue or publish work.

## Trainee Progress Snapshot

- [`anthropic__claude-opus-4-5`](../skills/mechanistic/propose_mechanism_step/models/anthropic__claude-opus-4-5/) — quality: `—` pass-rate: `—` [leaderboard](../curriculum/generated/leaderboard_anthropic_claude-opus-4-5.json)
- [`anthropic__claude-opus-4.6`](../skills/mechanistic/propose_mechanism_step/models/anthropic__claude-opus-4.6/) — quality: `—` pass-rate: `—` [leaderboard](../curriculum/generated/leaderboard_anthropic_claude-opus-4.6.json)
- [`anthropic__claude-opus-4.8`](../skills/mechanistic/propose_mechanism_step/models/anthropic__claude-opus-4.8/) — quality: `—` pass-rate: `—` [leaderboard](../curriculum/generated/leaderboard_anthropic_claude-opus-4.8.json)

## Checkpoints


## How to Inspect Any Past Milestone

1. Open the linked checkpoint manifest under `curriculum/checkpoints/`.
2. Check out the recorded git tag or commit.
3. Inspect the manifest for harness metadata plus resolved prompt and few-shot asset hashes.
4. Compare the linked skill directory to the current trainee lane if you want to see prompt or few-shot drift.

---

## Developer

### Harness Workflow Diagram

The default mechanistic harness orchestrates pre-loop analysis, an iterative mechanism-step proposal loop, and post-step validation. The diagram below matches the flow shown in the frontend app's Progress panel:

![Harness flow diagram](../docs/diagrams/Harness_Configuration_Flowchart.png)

- **Pre-loop** (runs once): Check Atom Balance -> Identify Functional Groups -> Recommend pH -> Assess Reaction Conditions -> Predict Missing Reagents -> Map Atoms -> Map To Reaction Type
- **Loop**: Propose Next Mechanism Step (LLM) -> Validate Mechanism Step -> Bond/Electron, Atom Balance, State Progress validators -> Retry or Continue? -> Target Products Reached? (yes -> Run Complete; no -> loop back)
- **Decision gates**: Retry/Backtrack routing when validation fails; Paused when no branch points remain

Regenerate the diagram with `python scripts/capture_harness_mermaid.py` (writes docs/diagrams/Harness_Configuration_Flowchart.mmd and .png).

### Quick Start

- Start the app: `python main.py serve`
- Queue a trainee curriculum batch when ready: `python main.py curriculum submit --model-name anthropic/claude-opus-4.6`
- Publish a queued batch when ready: `python main.py curriculum publish --checkpoint-id <queue-id>` (add `--force` to skip any stored publish timestamp)
- Optionally publish every queued batch whose timestamp has passed: `python main.py curriculum publish-due`
- Refresh this README and `curriculum/generated/`: `python main.py curriculum render-readme --model-name anthropic/claude-opus-4.6`
- Optional: `python main.py curriculum install-launchd` writes a sample plist if you automate `publish-due` locally

### Contributing

- Found a reaction the agent gets wrong? Open a **Chemistry failure** issue with reactants, products, and what went wrong.
- Software bug or idea? Open an issue. Code PRs are welcome; you only need the fast tests, not model evals.
- Prompt, few-shot, model, validator, and harness changes are evidence-gated internally: see [CONTRIBUTING.md](../CONTRIBUTING.md) and [docs/change_evidence_policy.md](../docs/change_evidence_policy.md).

### Docs

- Prompt/few-shot overrides: [docs/model_asset_overrides.md](../docs/model_asset_overrides.md)
- History and reproducibility: [docs/history_and_reproducibility.md](../docs/history_and_reproducibility.md)
