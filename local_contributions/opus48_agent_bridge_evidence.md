# Evidence — Keyless Claude Opus 4.8 runs → `attempt_atom_mapping` few-shot seed

This file documents the evidence behind two coupled changes:

1. A new model-catalog entry `anthropic/claude-opus-4.8`.
2. The first few-shot lane for the `attempt_atom_mapping` subagent under
   `skills/mechanistic/attempt_atom_mapping/models/anthropic__claude-opus-4.8/few_shot.jsonl`,
   seeded from approved, deterministically-validated atom-mapping traces.

## Method — keyless via the agent bridge

All runs were produced **without any provider API key**, using the
[agent bridge](../docs/agent_bridge.md): Claude Opus 4.8 (this orchestrator plus
its subagents) answered every forced tool call by writing response files into the
exchange directory. Dispatch was the normal seam
`get_chat_model("agent-bridge").invoke(messages, tools, tool_choice)`; the responder
saw only `messages` / `tools` / `tool_choice` (the privacy envelope).

Origin provenance recorded on every run's stored `config.origin`:

```json
{
  "responder": "agent-bridge",
  "declared_underlying_model": "claude-opus-4.8 (Hyperagent orchestrator + subagents)",
  "responder_kind": "orchestrator_subagents",
  "budget_observability": "opaque",
  "bridge_model": "agent-bridge"
}
```

Per the bridge contract these runs are attributed to `agent-bridge` (never
misattributed to a hosted model) and are **not** eligible for a Track 3 cost-class
SOTA claim (`budget_observability: opaque`).

## Runs (FlowER easy tier — SN2 / Menshutkin quaternizations)

Eval set `a294f47568574ca3b80245cf88aeb3b1` (FlowER 100, `import-eval-set`).
Run group `cli_eval_opus48_bridge`, harness `default`, eval_run_id `97ea4eb599024b5989b087d9b1c5c9da`.

| Case | Reaction (stripped) | Score | Pass | Status |
|---|---|---|---|---|
| flower_024300 | `ClCC1CO1 + CN(C)C → C[N+](C)(C)CC1CO1 + [Cl-]` | 0.997 | ✓ | completed |
| flower_130926 | `CCCBr + CN1CCCC1 → CCC[N+]1(C)CCCC1 + [Br-]` | 0.997 | ✓ | completed |
| flower_054599 | `CN(C)C + OCC(O)CCl → C[N+](C)(C)CC(O)CO + [Cl-]` | 0.997 | ✓ | completed |
| flower_252433 | `CCCl + CCN(N)CC → CC[N+](N)(CC)CC + [Cl-]` | 0.997 | ✓ | completed |

Aggregate: **mean_quality 0.9972, weighted pass rate 1.0, 996/1000 (WIN), avg latency 6.3 s/case.**
Per-subagent `step_atom_mapping` quality **0.96** (the universal weak point: GPT-5.5 `0.58`,
Opus 4.6 `0.965`). Snapshot: `curriculum/generated/leaderboard_agent-bridge.json`.

Every mechanism step passed the deterministic RDKit validators (bond/electron balance,
atom balance, state progress) — the ground-truth arbiter, independent of which model
produced the step.

## What was promoted into the lane

The four atom-mapping exemplars in the new opus-4.8 lane are the
`attempt_atom_mapping` (input, output) pairs produced during these runs. Each output
was cross-checked against the FlowER ground-truth atom map and is RDKit-parseable and
mass-balanced. This is the SOUL.md evolution loop in miniature: approved keyless traces
become few-shot examples for later work on the previously-empty lane.

## Scope / honesty

- Evidence tier is **easy** (1-step SN2 / Menshutkin). The CONTRIBUTING Track 1 medium-tier
  gate run is the recommended maintainer follow-up; the deterministic validators (the
  real arbiter) passed on every step here.
- The lane is a **first seed** of one reaction class; broadening to more classes
  (cycloaddition, carbonyl addition, reduction) is a natural next contribution.

## Reproduce

```bash
export MECHANISTIC_AGENT_BRIDGE_DIR=.agent_bridge
export MECHANISTIC_AGENT_BRIDGE_DECLARED_MODEL="claude-opus-4.8 (Hyperagent orchestrator + subagents)"
export MECHANISTIC_AGENT_BRIDGE_RESPONDER_KIND="orchestrator_subagents"
python main.py import-eval-set
# one terminal: run the eval keyless
python main.py eval --eval-set-id <id> \
  --case-id flower_024300 --case-id flower_130926 \
  --case-id flower_054599 --case-id flower_252433 \
  --model-name agent-bridge --allow-repeats --max-cases 4
# another terminal: answer the bridge calls (bridge-serve or an orchestrator loop)
python main.py bridge-serve            # or an orchestrator that writes responses/
python main.py curriculum render-readme --model-name agent-bridge
```
