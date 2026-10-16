# Threaded unified harness (`threaded_unified`) — design

Date: 2026-10-16. Status: approved ("do it").

## Goal

Make the mechanism harness faster and cheaper, and possibly more accurate, by
(1) replacing the four pre-loop decision calls (conditions, missing reagents,
atom mapping, reaction type — three LLM calls and one Jev decision) with one
LLM call, and (2) running the mechanism-step proposals as turns of one chat
thread that starts from that call, instead of a fresh call per step.

## Decisions taken with the user

| Question | Answer |
|---|---|
| Channel for the comparison | keyless agent bridge only |
| Rewind target after a failed proposal | last validated turn (keep steps 1..k-1, drop the failed turn, add a short error note) |
| Comparison cases | 10 hard cases, products hidden, low thinking (eval set `4d5405b940884e469c01a9fdef161b80`) |
| Side calls | drop `step_atom_mapping`; keep candidate rescue as a separate stateless call |

## Unified pre-analysis

One call, `unified_preanalysis` (skill `skills/mechanistic/unified_preanalysis/`,
tool `unified_preanalysis_result`). The tool arguments hold one object per
section, each with the same fields as the old tool:
`conditions`, `missing_reagents`, `atom_mapping`, `reaction_type`. When
products are hidden the reagent and mapping sections are omitted from the
schema and the prompt.

The answer is replayed through the old tools (`assess_initial_conditions`,
`predict_missing_reagents`, `attempt_atom_mapping`, `select_reaction_type`)
with `llm.preset_tool_answers`, a context manager that makes `get_chat_model`
return the stored answer instead of calling a model. Every existing parser,
normaliser, cache write (conditions summary, species registry, proposal
constraints) and output shape is therefore unchanged, and the old step names
are still recorded (`source = deterministic`, `derived_from =
unified_preanalysis`). A preset answer is single-use, so the old tools'
internal retries fail fast instead of re-reading the same answer.

### Checks (`core/preanalysis_checks.py`)

| Section | Pass | Soft fail | Hard fail |
|---|---|---|---|
| conditions | status success, environment set, 0 ≤ pH ≤ 14, candidate SMILES parse | anything else | — |
| missing_reagents | status `success` / `balanced` | other failures | still unbalanced with a deficit (`should_abort_mechanism`) |
| atom_mapping | rdkit-agent check passed, or skipped (CLI missing) | check failed, no mapped atoms | — |
| reaction_type | valid taxonomy label or `no_match` | parse/status failure | — |

Any failure gets one repair turn in the same thread (feedback lists each
failed section and its errors). Failed sections are re-derived from the
repaired answer; passing sections keep the first answer. A hard failure that
survives repair pauses the run exactly like the old reagent abort. If a soft
failure survives, the run continues but the loop thread does **not** inherit
the pre-analysis turns (the failure is not carried into the mechanism chat).

## Conversation thread (`core/conversation_thread.py`)

`ConversationThread` keeps `turns` (role, content, kind, step) plus
checkpoints `accepted:<k>` = turn count after step k was accepted
(`accepted:0` = end of the root). Bridge requests already carry
`model_input.messages`, so a thread needs no transport change. Assistant
turns are stored as text (`Called <tool> with: <json>`) because
`serialise_chat_messages` keeps only role and content.

- **Turn modes.** The first proposal (and the first after a compaction) uses
  the full proposal prompt. Later turns send a delta: step number, current
  state, mapped state, template guidance, reproposal hints, reagent line.
- **Rewind.** Each proposal for step k+1 starts by forking to
  `accepted:<k>`, so failed proposals for the step never stay in the thread.
  Their candidates and failed checks become short avoid notes (≤ 6 notes,
  ≤ 300 chars each) prepended to the next turn for that step.
- **Branching.** On backtrack to a branch point at step k, the thread forks
  to `accepted:<k+1>` (the exchange that produced the alternatives) and the
  next turn notes which alternative was applied and why the later path
  dead-ended. Notes and avoid lists for later steps are dropped.
- **Accept notes.** After a step is applied the next turn opens with the
  applied rank, intermediate, SMIRKS (truncated) and resulting state.
- **Compaction.** When the thread exceeds `max_prompt_chars` (default
  120 000), it becomes: root + one deterministic path-summary user turn
  (steps 1..n-2) + an acknowledgement + the last two accepted exchanges
  verbatim. The next turn is a full prompt. A fork into the compacted range
  rebuilds the summary for steps 1..k only.
- **Fallback.** If a threaded proposal returns no turn (provider failure,
  `tool_call_missing`), the step is retried stateless and a
  `thread_fallback` event is recorded.
- **Persistence.** The thread is stored in `run_resume_state` so resume
  keeps it.

Events: `unified_preanalysis_checks`, `thread_forked`,
`thread_turn_appended`, `thread_compacted`, `thread_fallback`,
`thread_root_reset`.

## Harness `harness_versions/threaded_unified/`

Built from `jev_reaction_type`: `decision_policy.reaction_type = llm` (no
Jev), pre-loop = balance → functional groups → pH → `unified_preanalysis`;
`loop_module.conversation_mode = "threaded"`; `step_atom_mapping` disabled;
`balance_mode = proton_deferred`; `centralized_mas` topology.

## Evaluation

Bridge, `bridge-model-low` subagents, hidden products, the hard-10 eval set,
`--max-steps 14`, then `bridge_responder.py audit --mark`. Compare against
the `jev_reaction_type` run (quality_v2 938; eval runs `db71026865a3…` +
`6d403f921952…`) and the one-shot baseline (900) with
`scripts/compare_harness_runs.py`: quality_v2 and axes per case, LLM calls
per case and per accepted step, prompt characters, wall time, forks and
compactions. Focus cases: `flower_020948`, `flower_067967` (targets missed
before).

## Out of scope

Publishing, PRs and the ChemIllusion bump (a `harness_versions/` change on
`main` triggers it) wait for the user. Prompt-evidence traces for the new
skill come from the eval runs.
