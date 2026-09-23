# Mechanistic Curriculum

<img align="right" src="docs/readme_ralph.png" alt="Ralph" width="260" />

## Orchestration Modes

RAlph mode provides iterative multi-attempt orchestration with budget controls for enhanced mechanism prediction reliability.

## Program Status

- Course: `Mechanistic Curriculum`
- Launch: `2026-03-11`
- Module: `Module 1` — climbing the difficulty chain (easy 1–2 step ✓ → **medium 3-step ✓** → hard 4+ step)
- Active model: **Claude Opus 4.8** (catalog id `anthropic/claude-opus-4.8`), driven **keyless** through the [agent bridge](docs/agent_bridge.md). Runs are attributed to `agent-bridge` with declared origin `Claude Opus 4.8` per the bridge provenance contract.

**Trainees:** [anthropic__claude-opus-4-5](skills/mechanistic/propose_mechanism_step/models/anthropic__claude-opus-4-5/) | [anthropic__claude-opus-4.6](skills/mechanistic/propose_mechanism_step/models/anthropic__claude-opus-4.6/) | [anthropic__claude-opus-4.8](skills/mechanistic/propose_mechanism_step/models/anthropic__claude-opus-4.8/) ← current (easy + medium)

Quick links: [Checkpoints](curriculum/checkpoints/) | [Reactions](training_data/flower_curriculum_pngs/index.json) | [Prompt guide](docs/model_asset_overrides.md) | [History](docs/history_and_reproducibility.md)

Curriculum checkpoints and trainee lanes advance **as time permits**. There is no public release clock; use the CLI below when you are ready to queue or publish work.

## Trainee Progress Snapshot

- Trainee: `agent-bridge` †  (declared model: **Claude Opus 4.8**, orchestrator + subagents) — [leaderboard](curriculum/generated/leaderboard_agent-bridge.json)
- Mean quality: `0.997`
- Pass rate: `100.0%`
- Cases: `4`
- Run group: `cli_eval_opus48_medium`  (FlowER **medium** tier, 3-step mechanisms)
- Subagent quality this tier: `mechanism_step_proposal` `1.00`, `step_atom_mapping` `0.95` (the historical weak point: GPT-5.5 `0.58`, Opus 4.6 `0.965`)

† Produced **keyless** via the agent bridge (no provider API key); cost is `budget_observability: opaque`, so this row is **not** eligible for a cost-class SOTA claim. **Caveat:** these runs replayed FlowER's verified mechanism steps through the responder (`responder_saw_ground_truth: true`, see `local_contributions/opus48_medium_evidence.md`), so the snapshot measures harness/validator acceptance of correct chemistry, not blind model skill. The first blind hard-tier runs (Fable 5.1 via the bridge, 2026-09-16) are recorded under run group `cli_eval_fable51_hard_blind`. Medium tier = 3-step FlowER mechanisms (carbonate formation, carbamate aminolysis, sulfonylation); every elementary step passed the deterministic RDKit validators (bond/electron balance, atom balance, state progress). Easy-tier (1-step SN2/Menshutkin) WINs remain in the history.

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

![Harness flow diagram](docs/diagrams/Harness_Configuration_Flowchart.png)

- **Pre-loop** (runs once): Check Atom Balance -> Identify Functional Groups -> Recommend pH -> Assess Reaction Conditions -> Predict Missing Reagents -> Map Atoms -> Map To Reaction Type
- **Loop**: Propose Next Mechanism Step (LLM) -> Validate Mechanism Step -> Bond/Electron, Atom Balance, State Progress validators -> Retry or Continue? -> Target Products Reached? (yes -> Run Complete; no -> loop back)
- **Decision gates**: Retry/Backtrack routing when validation fails; Paused when no branch points remain

Regenerate the diagram with `python scripts/capture_harness_mermaid.py` (writes docs/diagrams/Harness_Configuration_Flowchart.mmd and .png).

### Quick Start

- Start the app: `python main.py serve`
- Queue a trainee curriculum batch when ready: `python main.py curriculum submit --model-name agent-bridge`
- Publish a queued batch when ready: `python main.py curriculum publish --checkpoint-id <queue-id>` (add `--force` to skip any stored publish timestamp)
- Optionally publish every queued batch whose timestamp has passed: `python main.py curriculum publish-due`
- Refresh this README and `curriculum/generated/`: `python main.py curriculum render-readme --model-name agent-bridge`
- Optional: `python main.py curriculum install-launchd` writes a sample plist if you automate `publish-due` locally

### Contributing

- Found a reaction the agent gets wrong? Open a **Chemistry failure** issue with reactants, products, and what went wrong.
- Software bug or idea? Open an issue. Code PRs are welcome; you only need the fast tests, not model evals.
- Prompt, few-shot, model, validator, and harness changes are evidence-gated internally: see [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/change_evidence_policy.md](docs/change_evidence_policy.md).

### Docs

- Prompt/few-shot overrides: [docs/model_asset_overrides.md](docs/model_asset_overrides.md)
- History and reproducibility: [docs/history_and_reproducibility.md](docs/history_and_reproducibility.md)
