# PRD: Keyless Agent Contributions via the Agent Bridge

## TL;DR

**Goal.** Let a keyless agent surface (this Hyperagent — repo access, no provider API key) advance the project's goal and contribute *mergeable* PRs, by acting as the model behind subagent calls.

**Finding (tested, not assumed).** The capability already exists in the open PR #16 *agent bridge*, and an Opus 4.8 agent standing in as the model **works end-to-end through the real deterministic validators** — including the system's documented weak point (atom mapping). See *Empirical Validation*.

**Recommendation — keep it lightweight.** Treat the agent bridge as what it already is: a zero‑cost entry in the model catalog. Route agent contributions through the **existing** Tracks 1–4 (plus the existing Track 5 evidence lane), on the **one** existing leaderboard, with a small **origin‑provenance** record. Do **not** add a separate non‑mergeable lane, separate leaderboards, an executor registry, or a temp‑worktree sandbox subsystem. Those add friction and contradict the explicit goal of making contribution *easier*.

**Why this is safe.** SOUL.md Guardrail 1 already makes deterministic RDKit validation the final arbiter of correctness. Who or what produced a step is irrelevant to whether it is *correct* — it matters only for *attribution* and *cost framing*. So enabling a keyless contributor is a **labeling problem, not an architecture problem.**

**Net change for a contributor:** *fewer* steps (no API key, no provider account), *not more*.

## Problem & Goal

**Problem.** The harness reaches a model through exactly one seam — `get_chat_model(model).invoke(messages, tools=…, tool_choice=…)` — and every hosted adapter (OpenAI / Anthropic / OpenRouter / Gemini) hard‑requires a provider API key. A capable agent surface that has *no* API key (but can reason about chemistry and read/write the repo) is therefore locked out of producing runs, traces, and evidence — even though it could improve the system.

**Goal.** A keyless agent can:
1. drive real harness runs (single reactions, eval tiers) by answering each model call itself;
2. produce traces and eval‑tier evidence indistinguishable in *kind* from a keyed model's;
3. open **mergeable** Track 1–4 PRs that pass the same eval gates as anyone else;
4. do so with **less** ceremony than today, while the origin of the work stays auditable.

**Non‑negotiables preserved.** Deterministic chemistry as arbiter; evidence‑gated prompt changes; the model catalog as the source of model truth; the fast suite as the merge gate; fail‑loud over silent degradation (SOUL.md Guardrails 1–5).

## What Already Exists (build on this)

PR #16 (`feat/agent-bridge-keyless-provider`) already implements the core mechanism. The PRD's job is mostly to **decide what *not* to add** on top of it.

**The bridge is a catalog model.** `mechanistic_agent/model_pricing.json` →
```
id: agent-bridge | provider: agent_bridge | family: agent
supports_tools: true | label: "Agent Bridge (keyless)" | pricing: 0
```
Routed by `get_chat_model` exactly like the hosted providers. No new code path constructs model identity outside the catalog (Guardrail 3 intact).

**Single file‑based seam.** `MECHANISTIC_AGENT_BRIDGE_DIR/{requests,responses}/<seq>-<uuid>.json`. The adapter writes a request and waits; a responder writes the matching response. Pre‑seeding `responses/` gives deterministic CI replay. Timeout → **raises** (no silent degradation).

**Privacy contract (enforced by `tests/fast/test_agent_bridge.py`).** The responder receives only `model_input = {messages, tools, tool_choice}`, serialised with the *same* `serialise_chat_messages` helper the OpenAI adapter uses — byte‑for‑byte the view a keyed model gets, and nothing else. The harness strips privileged context before this seam.

**Keyless gating.** `get_model_api_key("agent-bridge")` returns a non‑empty *sentinel* so tool‑gated steps proceed; it is never used as a credential. `_resolve_step_model` resolves thread‑local → per‑call env → `MECHANISTIC_ACTIVE_MODEL` → default, so `MECHANISTIC_ACTIVE_MODEL=agent-bridge` forces every step through the bridge.

**Attribution today.** Traces and the leaderboard already record `model = agent-bridge` (family `agent`, pricing 0), so bridge runs are never misattributed to a hosted model.

## Empirical Validation (the test the user asked for)

I ran this, rather than reasoning about it abstractly. Environment: portable CPython 3.12.7 + RDKit 2026.03.2; branch `feat/agent-bridge-keyless-provider`.

**Setup.** Launched a real run — `main.py run --starting "CCBr.[I-]" --products "CCI.[Br-]" --model-name agent-bridge` (Finkelstein SN2) — with `MECHANISTIC_ACTIVE_MODEL=agent-bridge`. The orchestrator polled the bridge `requests/` dir, copied each call's `model_input` (and **only** that) into an isolated dir, and spawned an **Opus 4.8 subagent** to produce the forced tool's arguments from the prompt alone. No subagent was given repo, eval, or ground‑truth access — exactly the privacy contract.

**Calls answered correctly by subagents (model=agent-bridge in the trace):** `assess_initial_conditions`, `predict_missing_reagents`, `attempt_atom_mapping`, `select_reaction_type` (→ *Finkelstein halide exchange*, rt_001), `propose_mechanism_step`, `attempt_atom_mapping_for_step`. The **atom‑mapping** step (the documented universal weak point) was handled rigorously: correct heavy‑atom tracking, explicit symmetry/spectator reasoning, and a valid `|mech:v1;lp:4>2;sigma:2-3>3|` electron‑flow block.

**Deterministic validators — the arbiter — PASSED.** Recorded in the run DB (`data/mechanistic.db → step_outputs.validation_json`):
```
mechanism_synthesis (predict_mechanistic_step): passed=True
   checks = atom_balance:True, dbe_metadata:True, state_progress:True
atom_balance_validation:  passed=True
bond_electron_validation: passed=True (dbe_metadata)
state_progress_validation:passed=True
```
The harness **accepted** the step and advanced (`Step index: 1 … Current state ['CCI','[Br-]'] = target`). A failed step would have triggered a proposal *retry*, not advanced.

**Independent re‑confirmation (keyless, pure RDKit).** Re‑ran `predict_mechanistic_step` + `validate_mechanism_step_output` on the Opus‑authored step outside the run loop → aggregate `passed=True`, all three checks True.

**Conclusion.** The subagent‑as‑model path is not a thought experiment — it completes the full pipeline and clears the deterministic gate. This *strengthens* the case for the lightweight design: if the validators already certify correctness regardless of producer, no parallel correctness regime is warranted.

## The Key Insight

**Correctness is producer‑agnostic; only attribution and cost framing are producer‑specific.**

SOUL.md Guardrail 1 says RDKit validation is ground truth and is *not* configurable by model output. The validators ran identically whether the bytes came from a keyed Claude/GPT call or an Opus 4.8 subagent via the bridge. Therefore:

- We do **not** need separate validators, separate gates, or a quarantine lane to trust agent output. The existing gate already does the trusting.
- We **do** need to (a) label *where the output came from* (provenance), and (b) make sure a zero‑cost delegated *system* isn't mistaken for a cheap raw *model* in cost‑class rankings.

The other agent's proposal solves a real concern (don't misrepresent a delegated system as a raw model) but solves it with heavy architecture (execution channels, executor registry, Track 6, three leaderboard views, manifests, eight guardrails). Given the insight above, the same concern is met with **one provenance field + one leaderboard‑eligibility rule.** That is the whole delta this PRD proposes beyond PR #16.

## Decisions on the Other Agent's Proposal

| Proposal | Verdict | Rationale |
|---|---|---|
| `ExecutionChannel` abstraction (direct_model vs delegated_agent) wrapping the adapter | **Reject** | The catalog‑model approach (PR #16) already isolates the path cleanly. A second top‑level concept duplicates what `provider: agent_bridge` already expresses. |
| `executor_registry.json` separate from the model catalog | **Reject** | Guardrail 3 says the catalog is the single source of model truth. The bridge is already one catalog entry; a parallel registry is a second source of truth to keep in sync. |
| **Track 6: non‑mergeable agent‑candidate lane** + mandatory promotion to Track 1–4 | **Reject** | **Track 5 already is the non‑mergeable evidence lane.** A second one + a promotion ceremony is pure friction — the opposite of the goal. Agent work goes through Tracks 1–4 directly. |
| Separate delegated‑system leaderboard (View 2) + paired report (View 3) as required | **Reject as required; keep paired report as optional** | The one leaderboard already keys on `model_version_id`; `agent-bridge` is already a distinct key. Origin is legible without a parallel board. A paired direct‑vs‑bridge report is a useful *optional* analysis, not a gate. |
| Delegated‑agent manifest (`.json` with command/stdout/stderr hashes) | **Reject heavy form; keep a light record** | Hash‑of‑everything provenance is overkill. Keep a small, honest origin record (below). |
| Guardrail A — no silent fallback to a keyed model | **Keep (already satisfied)** | The bridge is its own provider and **fails loud** on timeout; it never falls back. Add one explicit regression test to lock it. |
| Guardrail B — runtime responder can't get repo write / tokens | **Keep (structural, already true)** | A runtime responder only ever receives `model_input` and only writes a response file. No new permission subsystem needed. |
| Guardrail C — PR‑authoring in temp worktree + write‑path allowlist | **Reject bespoke sandbox; use standard review** | CI + CODEOWNERS + the fast‑suite gate already cover this. |
| Guardrail D — no holdout ground‑truth exposure | **Keep (already true); add a guard test** | Privacy contract limits the responder to `model_input`; holdout *answers* never enter a request. Add a test asserting it. |
| Guardrail E — nested provenance | **Keep, simplified** | This is the user's "evidence of origin." One small optional block (below). |
| Guardrail F — structured‑contract gate | **Keep (already enforced)** | Response → `tool_calls` → existing `json.loads` + tool‑schema + chemistry validators. |
| Guardrail G — budget accounting | **Keep as observability label only** | `agent-bridge` pricing 0 is honest (no API spend), but inner agent cost is unknown → mark `budget_observability: opaque` so cost‑adjusted views don't imply free SOTA. |
| Guardrail H — no self‑modifying guardrail PRs without human review | **Keep — generalize to all contributors** | Implement via CODEOWNERS/required‑review on sensitive paths. Applies to humans and agents alike; directly supports the SOUL.md Stage 3/4 autonomy vision. |

## Proposed Design (lightweight)

Five small additions on top of PR #16. Nothing here introduces a new merge path or a second source of truth.

**1. Contributions flow through the existing tracks.** A keyless agent that improves the required eval tier opens a normal **Track 1** (few‑shot), **Track 2** (subagent/validator/schema), or **Track 4** (harness) PR, gated on the same tier as everyone. **Track 5** remains the place for non‑mergeable single‑reaction evidence. No Track 6.

**2. One leaderboard, origin already keyed.** Bridge runs already appear under `model_version_id = agent-bridge`. Surface two read‑only facts next to such rows: the **declared origin** (below) and **`budget_observability: opaque`** so a zero‑cost delegated system is never read as a cheap raw model.

**3. Origin‑provenance record (the one real addition).** A small, optional block written into the existing `runs.config_json` (no schema migration):
```json
{"responder": "agent-bridge",
 "declared_underlying_model": "opus-4.8 (Hyperagent orchestrator + subagents)",
 "responder_kind": "orchestrator_subagents | cli | script | replay",
 "budget_observability": "opaque",
 "notes": "optional free text"}
```
Defaults to `declared_underlying_model: "undeclared"`. Honest about opacity; no command/stdout hashes required.

**4. Responder ergonomics — the actual "make it easier" work.** Ship `python main.py bridge-serve` that runs the polling responder loop and hands each `model_input` to a configured agent command, plus `--replay <dir>` for pre‑seeded CI. (For Hyperagent, the orchestrator is the responder, as demonstrated — document that pattern too.) This removes the only genuinely fiddly part: hand‑rolling file polling.

**5. Governance on sensitive paths (covers B/C/H cheaply, for everyone).** Add CODEOWNERS / required‑review on: `mechanistic_agent/core/validators.py`, `skills/mechanistic/*/validator.py`, `scripts/validate_prompt_trace_evidence.py`, `training_data/eval_tiers.json`, `training_data/leaderboard_holdout/**`, `mechanistic_agent/model_pricing.json`. No PR (human or agent) merges changes to the arbiter without human core review.

**Eligibility rule (sharp, important).** `agent-bridge` is a *delegated system*, **not** a raw model, so it is **not eligible for Track 3 cost‑class SOTA claims** (Track 3 = new models, gated on easy‑tier SOTA *for a cost class*; a pricing‑0 system would falsely win every cost class). It contributes via Tracks 1/2/4, where the artifact is chemistry/structure, not a model‑cost claim. This is the one place the other agent's "it's a system, not a model" caution is honored — as a one‑line rule, not an architecture.

## Guardrails: Kept / Dropped Summary

**Kept because already present (lock with a test if missing):**
- *No silent fallback* — bridge fails loud; never reaches a keyed model.
- *Structured‑contract gate* — `tool_calls` → schema + RDKit validators, unchanged.
- *Holdout isolation* — responder sees only `model_input`; answers never sent.
- *Catalog is source of truth* — bridge is one catalog entry; no parallel registry.

**Kept, simplified:**
- *Provenance* → one optional origin block in `runs.config_json`.
- *Budget* → `budget_observability: opaque` label, not an accounting subsystem.
- *No self‑modifying guardrails* → CODEOWNERS/required‑review for all contributors.

**Dropped (friction without safety gain, given producer‑agnostic correctness):**
- ExecutionChannel abstraction; executor_registry.json.
- Track 6 quarantine lane + promotion ceremony (Track 5 already covers non‑mergeable evidence).
- Separate delegated leaderboard + required paired‑comparison views.
- Heavy manifest with command/stdout/stderr hashes.
- Bespoke temp‑worktree executor sandbox with per‑track write allowlists (standard CI + CODEOWNERS instead).

**New tests (small):**
- `test_no_fallback_to_keyed_model_when_bridge_selected` (regression‑lock Guardrail A).
- `test_bridge_request_excludes_holdout_ground_truth` (substrate allowed; answers/products never present).

## Contribution Flow: Before vs After

**Today, for a keyless agent:** *blocked* — every adapter requires a provider key.

**After this PRD, Track 1 (few‑shot) example:**
1. `export MECHANISTIC_AGENT_BRIDGE_DIR=.agent_bridge` and run `main.py bridge-serve` (or let the orchestrator respond).
2. `MECHANISTIC_ACTIVE_MODEL=agent-bridge python main.py eval --tier medium --model-name agent-bridge` — the agent answers each call; RDKit validates each step.
3. If the medium tier improves, add the few‑shot line(s) under `skills/mechanistic/<call_name>/few_shot.jsonl`, link the approved evidence trace, regenerate `LEADERBOARD.md`.
4. Open a normal **Track 1** PR. Fill the new PR‑template origin fields (`responder`, `declared_underlying_model`, `budget_observability`). Reviewer sees a standard Track 1 diff + medium‑tier delta + an origin tag.

**Reviewer burden:** identical to any Track 1 PR, plus a glance at the origin tag. **Contributor burden:** *lower* than a keyed contributor (no account, no key). That is the make‑it‑easier win, achieved by *subtraction*.

**PR‑template additions (all tracks):** `responder` · `declared_underlying_model` · `budget_observability` · `official_holdout_exposed_to_agent: false`. Track 5 also gains `submitted_by: human | agent`.

## Rollout / Milestones

Each milestone is independently shippable and (M1–M4) qualifies as an **Infra Exception** under CONTRIBUTING.md (workflow plumbing, no harness‑quality claim, ships with targeted fast tests, track gates intact).

- **M1 — Land the bridge.** Merge PR #16 (already verified: full fast suite green; this PRD adds an end‑to‑end validator pass).
- **M2 — Responder ergonomics.** `main.py bridge-serve` + `--replay`; update `docs/agent_bridge.md`. *This is the core "easier" deliverable.*
- **M3 — Origin provenance + leaderboard surfacing + PR‑template fields.** Write the origin block into `runs.config_json`; show declared origin + `budget_observability` next to `agent-bridge` rows; add the Track‑3 ineligibility note.
- **M4 — Governance + guard tests.** CODEOWNERS/required‑review on sensitive paths; add the two small tests.
- **M5 — First real agent‑bridge contribution.** A keyless Opus 4.8 run produces a **Track 1** few‑shot PR that improves the **medium** tier — the proof the evolution loop closes for a keyless contributor. Strongest first target per the leaderboard: **`step_atom_mapping`** few‑shots (the universal weak point; GPT‑5.5 scores 0.58 there, opus‑4.6 0.965).

**Maps to SOUL.md evolution:** M1–M4 enable Stage 1–2 (agents proposing evidence‑backed PRs) **without** weakening the Stage 4 safety model — validators, evidence gate, catalog, and fast suite stay intact, and M4 *hardens* the sensitive paths that autonomy will eventually touch.

## Risks & Mitigations

- **Stochasticity.** Agent responders vary run‑to‑run. *Mitigation:* the eval tiers are already the arbiter (single‑reaction wins don't merge); `--replay` gives reproducible CI; optional repeated runs + paired report for anyone who wants a confidence interval.
- **Zero‑cost looks like free SOTA.** *Mitigation:* `budget_observability: opaque` label + Track‑3 ineligibility rule. Treat `agent-bridge` as a *system*, not a raw model, in any cost framing.
- **Provenance honesty.** A responder could under‑declare its underlying model. *Mitigation:* the field is explicitly *declared* (defaults to `undeclared`); we never imply a raw‑model comparison from a bridge row. Correctness is still fully gated regardless.
- **Sensitive‑path tampering by an autonomous agent.** *Mitigation:* M4 governance — guardrail/arbiter files require human core review for *any* author.
- **Over‑trust of the bridge.** *Mitigation:* none of the guardrails that make autonomy safe are removed; this PRD only *adds* a keyless producer and *subtracts* unnecessary process.

## Non‑Goals

- Not building a parallel correctness regime or a new merge gate — the existing eval‑tier gate + RDKit validators stand.
- Not creating separate leaderboards — one board, origin‑keyed.
- Not building an executor sandbox / write‑path allowlist subsystem — standard CI + CODEOWNERS.
- Not measuring inner agent token cost — declared opaque.
- Not letting `agent-bridge` make Track‑3 cost‑class SOTA claims.
- Not changing prompts, few‑shot content, schemas, or the validation chain as part of *this* enabling work (those are normal tracked contributions a keyless agent can then make).

## Appendix: Reproduction

```bash
# portable py3.12 + rdkit venv, branch feat/agent-bridge-keyless-provider
export MECHANISTIC_AGENT_BRIDGE_DIR=$PWD/.agent_bridge
export MECHANISTIC_ACTIVE_MODEL=agent-bridge
python main.py run --starting "CCBr.[I-]" --products "CCI.[Br-]" --model-name agent-bridge
# responder: read each requests/<f>.json -> hand model_input to an Opus 4.8 agent ->
# write responses/<f>.json with the forced tool's arguments
```
Evidence captured this session: 6 LLM calls answered by Opus 4.8 subagents (all `model=agent-bridge`); deterministic validators `atom_balance / dbe_metadata / state_progress` all `passed=True` in `data/mechanistic.db`; independent `predict_mechanistic_step` re‑run confirmed the same. The atom‑mapping weak point was handled correctly from the prompt alone.

*Owner:* TBD · *Status:* Draft for review · *Prereq:* PR #16.
