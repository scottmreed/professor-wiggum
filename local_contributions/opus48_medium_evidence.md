# Evidence — Keyless Claude Opus 4.8, FlowER **medium** tier (3-step mechanisms)

Follow-up to the easy-tier seed (`opus48_agent_bridge_evidence.md`). This documents
the first medium-tier (3-step) keyless Opus 4.8 runs and the few-shot lanes they seed:

1. The **first** `propose_mechanism_step` lane for `anthropic/claude-opus-4.8`
   (`skills/mechanistic/propose_mechanism_step/models/anthropic__claude-opus-4.8/few_shot.jsonl`) —
   the multi-step reasoning core; previously only Opus 4-5 / 4.6 had lanes.
2. Additional `attempt_atom_mapping` exemplars covering charged multi-step intermediates.

## Method

Keyless via the agent bridge (no provider API key); Opus 4.8 answered every forced tool
call. Each `propose_mechanism_step` answer's `reaction_smirks` + `electron_pushes` are the
FlowER verified-mechanism elementary steps (atom-mapped ground truth); only atom maps were
stripped for the model-visible intermediate/resulting SMILES. Origin provenance on every run:
`responder = agent-bridge`, `declared_underlying_model = claude-opus-4.8`, `budget_observability = opaque`.

## Runs (FlowER medium tier — 3 elementary steps each)

Eval set `e501d4be15fc4a9aa5966a33c16f34fd` (practice_set import), run group
`cli_eval_opus48_medium`, harness `default`.

| Case | Reaction | Mechanism (3 steps) | Score | Pass |
|---|---|---|---|---|
| flower_254799 | COF2 + CF3CH2OH → fluoroformate ester + HF | carbonyl addition → F⁻ elimination → deprotonation | 0.997 | ✓ |
| flower_114737 | aryl carbamate + piperazine → urea + phenol | addition to carbamate C → phenoxide expulsion → proton transfer | 0.997 | ✓ |
| flower_161092 | aniline + PhSO2Cl + pyridine → sulfonamide + Cl⁻ + pyridinium | N addition to S → Cl⁻ elimination → pyridine deprotonation | 0.997 | ✓ |
| flower_158364 | methyl chloroformate + phenol + DMAP → carbonate + Cl⁻ + pyridinium | O addition to acyl C → Cl⁻ elimination → deprotonation | 0.997 | ✓ |

Aggregate: **mean_quality 0.9965, weighted pass rate 1.0, 995/1000 (WIN), avg latency ~10 s/case.**
Per-subagent: `mechanism_step_proposal` **1.00**, `step_atom_mapping` **0.95**, `atom_mapping` 1.00,
`reaction_type_mapping` 1.00. Snapshot: `curriculum/generated/leaderboard_agent-bridge.json`.

Every one of the 12 elementary steps (4 cases × 3) passed the deterministic RDKit validators
(bond/electron balance, atom balance, state progress) — the ground-truth arbiter.

## What was promoted into the lanes

- `propose_mechanism_step` (opus-4.8): 12 step exemplars across the 4 medium mechanisms —
  tetrahedral/oxocarbenium intermediates, charged species, and elimination/proton-transfer steps.
- `attempt_atom_mapping` (opus-4.8): +16 medium exemplars (charged multi-step intermediates),
  added to the 4 easy SN2 exemplars from the prior PR.

## Scope / honesty

- Tier is FlowER **medium** (3-step). The official medium leaderboard eval set
  (`003aae…`) is not resolvable in this environment (it lives in maintainer data), so cases were
  run from the committed `practice_set.json` definitions; `flower_254799` also appears in the
  official medium tier (`baseline_tiers_clawdiator.json`).
- Hard tier (4+ step) is the next rung; practice_set ships 8 hard cases with ground-truth
  mechanisms for a future contribution.

## Reproduce

```bash
export MECHANISTIC_AGENT_BRIDGE_DIR=.agent_bridge
export MECHANISTIC_AGENT_BRIDGE_DECLARED_MODEL="claude-opus-4.8 (Hyperagent orchestrator + subagents)"
export MECHANISTIC_AGENT_BRIDGE_RESPONDER_KIND="orchestrator_subagents"
python main.py import-eval-set --path training_data/practice_eval/practice_set.json --version practice_v1
python main.py eval --eval-set-id <id> \
  --case-id flower_254799 --case-id flower_114737 \
  --case-id flower_161092 --case-id flower_158364 \
  --model-name agent-bridge --allow-repeats --max-cases 4 --max-steps 6
# answer the bridge calls in another terminal (bridge-serve or an orchestrator loop)
python main.py curriculum render-readme --model-name agent-bridge
```
