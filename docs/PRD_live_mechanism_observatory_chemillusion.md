# PRD: Live Mechanism Observatory and Production Mechanism Runtime

**Status:** Draft for implementation  
**Date:** 2026-09-23  
**Primary research repository:** `scottmreed/professor-wiggum`  
**Primary product repository:** `scottmreed/chem-art-generator` (ChemIllusion)  
**Wiggum baseline inspected:** `main` through `bf56e43c598f4c33200430d4733c4a588e0f5c87`  
**ChemIllusion baseline inspected:** `main` at `8f8e89e841eadac703197012fcee2fb37a4671ff`  
**Related Wiggum work:** PRs #27, #28, #29, #31, #32, #33, #35, #36 and `docs/PRD_jev_atom_identity_mechanistic.md`  
**Related ChemIllusion work:** Mechanism Explorer stack and commit `dc70d22ae141add4025a849028d7621fe78cddc2` (`typesafe/jev-latest` pricing/catalog stub)  
**Re-audited against Wiggum `main`:** `c9248b8` (2026-09-23, includes PRs #37–#41) — corrections in §0 and §3.7  
**Rev 3 (2026-09-24):** deployment on the existing Railway services (§2.3) and the ChemIllusion product surface — standalone URL, Generator right dock, mobile, beta labeling, Jev-vs-LLM labeling (§8.5). Checked against ChemIllusion `origin/main` `f7a0f57d6` (includes Adaptive Right Dock PR #3390) and the Railway project `triumphant-elegance` / `production`.

---

## 0. Revision log

### 2026-09-24 (rev 3) — no new Railway service; ChemIllusion surface

| Section | Change |
|---|---|
| §1, §2.1 | The Mechanism Runtime is a **pinned Wiggum release embedded in the existing ChemIllusion API service**, not a new Railway service. |
| §2.3 (new) | Service-by-service review of the five services in `triumphant-elegance/production` and the chosen placement: in-process in the API, run through the existing `async_jobs` lease ledger, events in the existing Postgres. |
| §8.5 (new) | Product surface: standalone URL `/mechanism-predictor` + shareable `/mechanism-predictor/runs/:runId`; Generator right-dock entry as a tool tile now and a context card later; mobile "answer" layout on the web page (not in the MyMol native wrapper); Beta · In development labeling with a link to this repo; three disclosure levels (Answer / Steps / Details); explicit Jev-vs-LLM-vs-deterministic chips on every step. |
| §21–§24 | Re-scoped: `runtime_app.py` (PR #50) stays as the Wiggum-side contract and dev surface; production consumption is in-process. Storage is a `RunStateStore` adapter on ChemIllusion's Postgres. |
| §33 M3/M4 | Updated to the embedded plan and the surface in §8.5. |
| §37 | Decisions 15 and 18–22 added/changed. |


The first draft inspected `bf56e43`. Five PRs landed before implementation started, and a line-level re-read of the runtime changed several premises. Corrections applied in this revision:

| Section | Change |
|---|---|
| §3.6 | **Jev is live on Wiggum `main` for one decision.** PR #40 wired `ReactionTypeAgent` to the OpenRouter Decisions API behind `decision_policy.reaction_type == "jev"`. Rewritten to describe what exists and what is still reserved. |
| §3.7 (new) | Gaps found in the re-audit that the first draft missed: internal provider fallback not surfaced, soft-advance acceptances, human path bypasses the coordinator, `LLM_STEP_KEYS` lists deterministic steps. |
| §10.1 | Names the existing bond-electron substrate (`bond_electron_deltas`, `reaction_bond_deltas`, `\|mech:v1\|` notation) so BE(t)/ΔBE/BE(t+1) is a projection, not a new engine. |
| §12.2 | Provenance schema is now mapped field-by-field onto the existing `DecisionRecord` so Jev calls do not get a second, incompatible record. |
| §13 | Splits call-level events into M0a (derived at `_record_step` from `StepResult` + `decision_trace`) and M0b (live hook in `llm.py`). |
| §15.2 | Reaction-type Choice on `typesafe/jev-1.13-20260917` **is** calibrated (n=72, ECE 0.056, `docs/calibration/jev_reaction_type_2026-09-23.md`); `calibrated`/`calibration_version` must reflect that. |
| §16.6 (new) | `mechanism_step_accepted` gains `acceptance_kind`; soft-advanced steps must not render as validated. |
| §17.2 | PR #37 already persists branch points with full alternatives in `run_resume_state`; replay gap narrowed to per-candidate history. |
| §27, §35, §36, §39 | M0 file list and tests updated to the actual code sites; `make test` does not exist (no Makefile), use `python -m pytest tests/fast -q`. |
| §38 | Q1/Q2 partially answered by PR #40 and the calibration run. |

---

## 1. Executive summary

Professor Wiggum has evolved from a linear mechanism predictor into a traceable chemistry harness with deterministic validation, retry, branching, backtracking, atom-mapping work, persistent atom identity, and recent work toward a Jev-first decision layer. Its current browser UI still presents the **harness flow** as the dominant visualization. That is useful for debugging the agent, but it is not the best way for a chemist to understand the mechanism search.

Build a **Live Mechanism Observatory** whose visual center is the **chemical pathway being explored**. The main view shows accepted intermediates, candidates under consideration, rejected candidates, abandoned branches, and backtracking in real time. Peripheral views explain the active elementary step using:

1. a deterministic reaction-focus mask that suppresses unchanged chemistry;
2. explicit bond-electron matrices `BE(t)`, `ΔBE`, and `BE(t+1)`;
3. persistent atom-identity / atom-mapping visualization;
4. deterministic validator diagnostics;
5. typed candidate probabilities or confidence signals when available; and
6. **explicit model/engine provenance for every computational step**.

The product architecture should **not copy the full Professor Wiggum repository into ChemIllusion**. Professor Wiggum remains the research, evaluation, curriculum, and evolution environment. An immutable, tagged **Mechanism Runtime** release of this repository is pinned by ChemIllusion and **runs inside the existing ChemIllusion API service** (rev 3, §2.3) — no new Railway service. ChemIllusion owns the React visualization, the product URL, the Generator right-dock entry, and the user-facing authorization/billing layer (§8.5).

The existing ChemIllusion **Mechanism Explorer** is already the correct deterministic chemistry/display foundation. It has stable atom IDs, semantic electron actions, deterministic state transitions, state diffs, validator findings, path-graph schemas, SVG rendering, and a React interactive workspace. The Observatory should extend and reuse that system rather than create a second molecule-state representation.

The immediate prerequisite is **M0: live provenance**. The current Wiggum runtime already stores a `model` and `reasoning_level` on `step_outputs`, but its live `step_output` SSE event omits the resolved model, and `step_started` does not identify the planned engine/model. This becomes increasingly misleading as Jev, frontier LLMs, deterministic chemistry, fallbacks, and potentially multiple model calls coexist in one logical step.

---

# 2. Product decision

## 2.1 Adopt a three-layer architecture

```text
┌───────────────────────────────────────────────────────────────────────┐
│                    PROFESSOR WIGGUM — LAB                            │
│                                                                       │
│ prompts • harness evolution • eval tiers • Jev experiments           │
│ mapping experiments • scoring • curriculum • leaderboards             │
│                                                                       │
│             promote an evaluated, versioned runtime release           │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│              MECHANISM RUNTIME — PRIVATE RAILWAY SERVICE              │
│                                                                       │
│ prediction loop • mapping • candidate selection • validators          │
│ mapped-state executor • branching/backtracking • BE matrices          │
│ reaction focus • probabilities • provenance • SSE                     │
│                                                                       │
│ NO curriculum • NO training/eval corpus • NO harness mutation UI      │
└───────────────────────────────┬───────────────────────────────────────┘
                                │ authenticated structured API/SSE
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│                       CHEMILLUSION PRODUCT                            │
│                                                                       │
│ auth • entitlement • billing • proxy • persistence metadata           │
│ React Mechanism Observatory                                           │
│ existing Mechanism Explorer deterministic state/display components    │
└───────────────────────────────────────────────────────────────────────┘
```

## 2.1.1 Rev 3 note

The middle layer in the diagram above is a **release artifact, not a service**: a git tag of this repository (`runtime-vX.Y.Z`) that ChemIllusion's API image checks out at build time and imports. Its boundary (what is included and excluded, §21.2–§21.3) and its contract (events, `/observatory` projection, manifest) are unchanged; only its deployment moves (§2.3).

## 2.2 Do not merge the full Wiggum codebase into ChemIllusion

The two repositories have different responsibilities.

**Professor Wiggum owns:**
- mechanism-prediction research;
- model and harness evaluation;
- prompt/few-shot evolution;
- Jev experiments;
- RAlph / island evolution;
- benchmark and scoring logic;
- training/eval data;
- promotion of approved runtime releases.

**ChemIllusion owns:**
- product authentication;
- user entitlements and quotas;
- billing;
- presentation;
- accessibility;
- user-facing run history;
- the Mechanism Explorer state/display framework;
- the production Observatory.

**Mechanism Runtime owns:**
- only what is required to predict, validate, trace, and explain a live mechanism run.

This boundary allows Professor Wiggum to continue changing rapidly while ChemIllusion consumes a stable, versioned runtime contract.

---

## 2.3 Deployment on the existing Railway services (rev 3)

**Requirement:** do not add a Railway service for the Mechanism Runtime.

### 2.3.1 The five services in `triumphant-elegance` / `production` (read 2026-09-24 via `railway status --json`)

| Service | Source | What it is | Host the runtime? |
|---|---|---|---|
| `spirited-liberation` | `scottmreed/chem-art-generator`, `railway.json` → `backend/Dockerfile.api` | The ChemIllusion API (FastAPI, `python minimal_start.py`, port 8080, public domain). Already has RDKit 2023.9.1, OpenAI/Anthropic/Gemini SDKs, the Postgres connection and the `async_jobs` lease ledger. | **Yes — chosen.** |
| `ts-worker-spike` | same repo, `railway.ts-worker.json` → `Dockerfile.ts-worker` | Transition-state compute worker; claims work from the API over `ts_worker_internal.py` (`claim_job`) with `TS_WORKER_TOKEN`, reached at `ts-worker-spike.railway.internal`. | Fallback only (§2.3.4). It is a spike with a slim TS-specific image and its own watch patterns. |
| `steel-man-args` | image `ghcr.io/steel-dev/steel-browser` | Headless browser for Local Assist. | No. |
| `privatemode-proxy` | image `ghcr.io/edgelesssys/privatemode-proxy` | Confidential-inference proxy (latest deploy FAILED). | No. |
| `oscar-blues-scan` | no source connected | Unconnected service (latest deploy FAILED). | No — not ours to repurpose without an owner decision. |

### 2.3.2 Placement: in-process in the API, through the job ledger

1. **Code.** `backend/Dockerfile.api` checks out this repository at a pinned tag (build arg `WIGGUM_RUNTIME_REF=runtime-v0.1.0`) into `/opt/wiggum` and runs `pip install --no-deps /opt/wiggum`, adding `dimorphite-dl` to `requirements-railway.txt`. A tagged checkout, not a bare wheel, because prompts (`skills/mechanistic/`), harnesses (`harness_versions/`) and reaction templates live outside the Python package; the runtime is constructed with `base_dir=/opt/wiggum`. The repository is public, so no token is needed; ChemIllusion already pins `xyzrender @ git+https://…@v0.3.1` the same way. The image copies no `training_data/leaderboard_holdout/`, `traces/` or `data/` (same exclusion list as `.dockerignore` from PR #50).
2. **Execution.** A mechanism run is an `async_jobs` row (`job_type = "mechanism_run"`) leased by the API's existing job worker (`FOR UPDATE SKIP LOCKED`, `lease_expires_at`, `heartbeat_at`). The worker calls the embedded runtime on a thread (runs are dominated by model latency, not CPU; RDKit validation is milliseconds per candidate). Concurrency is capped (default **1**, env `MECHANISM_RUNTIME_MAX_CONCURRENT`), watched with the existing `event_loop_lag_monitor`. No request thread ever runs a mechanism loop.
3. **Storage.** A `RunStateStore` adapter (the seam named in §24) writes runs, step outputs and ordered events to ChemIllusion's existing Postgres (new tables `mechanism_runs`, `mechanism_run_events`; alembic migration run by the existing `preDeployCommand`). No SQLite file on Railway's ephemeral disk.
4. **Transport.** The API serves `GET /api/mechanism-predictor/runs/{id}/events` (SSE, `after_seq` resume) and `/observatory` directly from that store — the same projection code as Wiggum's `build_observatory`. The browser talks only to the API it already uses; there is no second host, no service token and no private-network hop.
5. **Keys and cost.** Model calls use the API's existing provider keys; every `inference_call_*` event is metered into the existing AI-action ledger. Jev pricing stays a placeholder until confirmed (§26).

### 2.3.2.1 Embedding API (delivered 2026-09-24, `mechanistic_agent/runtime/embedded.py`)

```python
from mechanistic_agent.runtime import EmbeddedMechanismRuntime
rt = EmbeddedMechanismRuntime(base_dir="/opt/wiggum", work_dir=job_tmp, event_sink=mirror_event_to_postgres)
run_id = rt.create_run(request_dict)        # CreateRunRequest shape; ValueError on bad input
rt.execute(run_id, stop_event=stop)         # blocking on the job-worker thread; returns final status
rt.observatory(run_id)                      # mechanism_observatory.v1 + {"runtime": {...}}
rt.manifest()                               # runtime_manifest.v1 with deployment="embedded"
```

- It drives the research app's own `POST /api/runs` handler and `RunCoordinator.execute_run`, so validation, normalization and the mechanism loop are the evaluated ones.
- `RunStore(db_path, event_sink=...)` mirrors every event, after commit and with the `list_events` row shape, to the caller. The SQLite file in `work_dir` is scratch. Because `build_observatory` needs only events, `build_observatory(<mirrored rows>)` equals the live projection (tested), so the durable store in ChemIllusion is one append-only table (`mechanism_run_events(run_id, seq, ts, event_type, step_name, payload jsonb)`), not a port of `RunStore`. A mirror failure never breaks a run.
- Not yet covered: resuming a run interrupted by a container restart (the lease expires, the job is marked failed, the mirrored events still replay). Acceptable for beta.

### 2.3.3 Why not a separate service now

- The runtime is a Python library whose dependencies are already in the API image (RDKit, FastAPI, pydantic, the three provider SDKs); the only addition is `dimorphite-dl`.
- The durable job ledger, Postgres, auth, entitlements and billing already live in the API. A separate service would duplicate all five and add a service token, a private-network client and a second deploy for a **beta, in-training** tool with low expected traffic.
- The HTTP runtime from PR #50 (`runtime_app.py`) stays useful: it is the Wiggum-side contract test for the product surface and a dev host, and it keeps the option to split later without redesign.

### 2.3.4 Risks and the fallback

| Risk | Check before enabling | Fallback |
|---|---|---|
| `openai` SDK major version: API pins `openai==3.16.0`; Wiggum declares `openai>=1.40` and calls `client.chat.completions.create` | **Checked 2026-09-24:** in a venv with ChemIllusion's pins (openai 3.16.0, rdkit 2023.9.1, numpy 1.26.4, fastapi 0.141.1, httpx 0.28.1) Wiggum's full fast suite passed (853/853) and the OpenAI adapter completed a forced tool call against a local mock server with usage parsed and live `inference_call_*` recorded. Keep this as a ChemIllusion CI job on every tag bump. | Wiggum adapter shim; do not bump ChemIllusion's pin |
| Memory/latency pressure on the API | `event_loop_lag_monitor` and container memory during a 3-case smoke run | Keep concurrency 1; if still hot, let the existing `ts-worker-spike` claim `mechanism_run` jobs through the same internal claim endpoint (a job type on an existing service, still no new service) — requires an explicit owner decision because that service is a spike |
| Prompt/harness drift between Wiggum `main` and the pinned tag | Runtime manifest (`/v1/mechanism/version` fields) recorded on every product run | Bump only after Wiggum eval evidence (§8.5.8) |
| Hard-coded fallback to `gpt-4o` inside `tools.py` spends tokens silently (found in rev 2) | Provenance shows `model_fallback`; product surfaces it | Disable provider fallback in the production harness |

# 3. Codebase audit: what exists now

## 3.1 Professor Wiggum runtime is FastAPI, not Flask

`mechanistic_agent/api/app.py` creates a FastAPI application. The local UI is served from:

- `mechanistic_agent/ui/index.html`
- `mechanistic_agent/ui/app.js`
- `mechanistic_agent/ui/styles.css`

The browser UI is vanilla JavaScript/CSS. The central harness diagram is Mermaid. Chart.js is also loaded for analytical plots.

The live path already exists:

```text
GET /api/runs/{run_id}/events
```

This is an SSE stream. `mechanistic_agent/ui/app.js` opens it with `EventSource`, while a run snapshot is refreshed on a roughly 1.5-second interval.

This is a strong basis for a live observatory. The new product does not need a new streaming protocol.

## 3.2 Wiggum already persists per-step model fields

`mechanistic_agent/core/types.py::RunConfig` contains:

```python
model
model_name
model_family
step_models
step_reasoning
thinking_level
reasoning_level
```

`mechanistic_agent/core/model_selection.py::select_step_models()` currently creates a uniform exact-model map for enabled LLM-backed steps.

`mechanistic_agent/core/coordinator.py` already resolves a model and reasoning level per `StepResult`:

```python
resolved_model = result.model or self._step_model(state, result.step_name)
resolved_reasoning = result.reasoning_level or self._step_reasoning(state, result.step_name)
```

and writes those values to `store.record_step_output(...)`.

Therefore this PRD does **not** require inventing model provenance from scratch. The first requirement is to make the existing provenance accurate and live.

## 3.3 Current live provenance gap

`_record_step()` emits `step_output`, but the event payload currently contains:

```text
step_name
tool_name
attempt
retry_index
source
output
validation
```

It does **not** include `resolved_model` or `resolved_reasoning`.

`_mark_step_started()` similarly emits:

```text
step_name
tool_name
attempt
retry_index
start_time
```

with no engine/model identity.

The compact `GET /api/runs/{id}` snapshot does expose the run-level `model`, `step_models`, and `step_reasoning`, and the returned `step_outputs` have persisted metadata. That is enough for post-hoc reconstruction, but it is not adequate for a live UI, especially once one logical step can use Jev, an LLM fallback, or more than one inference call.

## 3.4 `mechanism_step_accepted` is emitted but not subscribed to by the current Wiggum UI

The coordinator emits `mechanism_step_accepted`, and scoring code treats it as the preferred accepted-path record.

The current `eventKinds` array in `mechanistic_agent/ui/app.js` does not include it.

This is a concrete bug/omission for the Observatory and must be fixed in M0.

The gap is wider than one event. The coordinator emits roughly 70 event types; the UI subscribes to 38. Mechanism-loop events that a chemist would want to see and that the UI currently drops include:

```text
mechanism_step_accepted
mechanism_candidate_incomplete
mechanism_candidate_constraint_rejected
mechanism_candidate_execution_exception
mechanism_candidate_uncaught_exception
mechanism_validation_exception
invalid_species_in_candidate
candidate_rescue_started / candidate_rescue_completed / candidate_rescue_skipped_*
proposal_quality_summary
mechanism_reproposal_requested / mechanism_reproposal_limit_reached
mechanism_step_soft_advance
topology_dispatch / independent_agent_result / peer_round_complete / consensus_merge_result
step_mapping_generated
remaining_mechanism_fallback_generated / remaining_mechanism_fallback_failed
```

M0 subscribes to the mechanism-loop subset; the RAlph/evolution kinds stay research-only.

## 3.5 Recent Wiggum changes materially improve the Observatory substrate

The 2026-09-23 changes are directly relevant:

### PR #27 / commit `0d041bee...`
Global atom mapping now actually reaches the mechanism proposal:
- mapping confidence path corrected;
- mapped starting/product/current-state context is generated instead of hard-coded empty lists;
- `global_mapping_context.py` added;
- `no_mapping` ablation harness added.

### PR #33 / commit `23479a99...`
`rdkit-agent atom-map` validation was repaired so mapping checks actually run instead of silently becoming “tool unavailable.”

### PR #35 / commit `d608daf6...`
A persistent mapped-state executor was added:
- persistent atom identities;
- mapped SMIRKS execution;
- electron-move execution;
- identity propagation;
- `smirks_state_agreement`;
- optional mapped loop state.

This is the key technical precursor to a useful atom-lineage visualization.

### PR #31 / commit `ad8715d9...`
Benchmark mapping agreement became a measured metric.

### PR #36 / commit `521309e2...`
Scoring v2 stopped relying on the mapping LLM's self-reported confidence as the mapping-quality component and instead uses measured evidence where available.

This is important for the Observatory: **do not visually equate self-reported model confidence with measured mapping quality.**

### PR #28 / commit `e65353ab...`
Real model-call and token accounting was added. It also corrected `mechanism_synthesis` from `source="llm"` to `source="deterministic"` because the synthesis/validation function itself makes no model call.

This is the immediate provenance foundation.

## 3.6 Jev status must be represented accurately

There are two separate “Jev has been added” facts in the current codebase.

### Professor Wiggum

`docs/PRD_jev_atom_identity_mechanistic.md` defines the Jev-first decision-layer architecture and the Phase-0 instrumentation explicitly reserves an engine label for future `jev` calls.

**Rev 2 correction.** PR #40 (`4d35737`, 2026-09-23) made one Jev decision live on `main`:

- `mechanistic_agent/decisions/jev.py` — `JevDecisionClient` over the OpenRouter Decisions API (`PROVIDER = "openrouter"`), returning a `DecisionRecord` per question with `model`, `model_version`, `provider`, `decision_type`, `selected`, `probabilities`, `confidence`, `latency_ms`, `usage`, `cost`, `request_id`, `called`, `failure`.
- `mechanistic_agent/core/reaction_type_jev.py` — `select_reaction_type_jev` behind `decision_policy.reaction_type == "jev"`; on Jev failure with `jev.fallback == "llm"` it calls the LLM selector and records the failed decision in `output.decision_trace`.
- `mechanistic_agent/core/subagents.py::ReactionTypeAgent._run_jev` — returns `StepResult(source="jev", model=output.model_used)`; the fallback case returns `source="llm"` with both engines' usage billed to the step.
- `mechanistic_agent/core/db.py::get_run_cost_summary` — `engine_by_source = {"llm", "jev"}`; `call_summary` exposes `jev_calls`, `jev_tokens`, `by_engine`, and counts nested `decision_trace` requests by `request_id`.
- `harness_versions/jev_reaction_type/harness.json` — the opt-in harness variant; the default harness still uses `reaction_type: "llm"`.
- `docs/calibration/jev_reaction_type_2026-09-23.md` — shadow calibration on 72 curated labels: top-1 accuracy 0.917, ECE 0.056, provider reported revision `typesafe/jev-1.13-20260917`.

Still reserved (accepted and round-tripped by `DecisionPolicy`, ignored by the runtime): `conditions`, `global_mapping`, `step_mapping`, `missing_reagents_gate`, `candidate_ranker`. `DECISION_POLICY_WIRED_KEYS == ("reaction_type",)`.

Therefore:

> The Observatory contract must represent the reaction-type Jev decision as live today, including the Jev→LLM fallback inside one logical step, and must treat the other Jev roles as reserved. The provenance schema in §12 is mapped onto `DecisionRecord` rather than defined beside it.

### ChemIllusion

Commit `dc70d22ae141add4025a849028d7621fe78cddc2` added:

```text
typesafe/jev-latest
```

to `backend/app/services/model_pricing.json`.

That commit explicitly describes it as a **catalog stub only**:
- not wired into model allow-lists;
- not wired into model routers;
- placeholder `$0` pricing;
- adapter verification still required.

Therefore:

> A pricing-catalog entry is not inference integration.

The model-provenance schema defined below is designed so the Jev runtime can land without another UI/data-model redesign.

---

## 3.7 Additional gaps found in the 2026-09-23 re-audit

These are confirmed at line level on `c9248b8` and are M0 requirements.

### 3.7.1 Deterministic steps inherit the run model — exact mechanism

All deterministic `StepResult`s (`balance_analysis`, `ph_recommendation`, `functional_groups`, `mechanism_synthesis`, `reflection`, and the three validator rows written by `_record_validation_checks`) are created with `model=None`. `_record_step()` then does:

```python
resolved_model = result.model or self._step_model(state, result.step_name)
# _step_model: state.run_config.step_models.get(step_name, state.run_config.model)
```

so every deterministic row is stored with the run's LLM model. `mechanistic_agent/config.py::LLM_STEP_KEYS` makes this worse by listing `functional_groups` and `mechanism_synthesis`, so `select_step_models()` writes explicit entries for two deterministic steps. `GET /api/runs/{id}` progress rows then fall back to `cfg.step_models[step_name]` for pending steps. Fix at `_record_step` (normalize by `source`) and stop treating `LLM_STEP_KEYS` as "steps that have a model".

### 3.7.2 Internal provider fallback is invisible to provenance

`tools.py::propose_mechanism_step` and `select_reaction_type` retry on a `fallback_model` and set `output["model_used"] = fallback_model`, but `IntermediateAgent.run` (and the other LLM agents) build `StepResult(model=<configured>)` and call `_extract_step_cost(output, <configured>)`. Only `_run_jev` propagates `model_used`. The stored `model` and the cost attribution are therefore wrong on every fallback. M0 must resolve `resolved_model` from `output.model_used` first. Cost re-attribution landed with M0b: `_extract_step_cost` prices usage against `output.model_used` when present, falling back to the configured model (and to the configured model again when `model_used` is not in the catalog).

**Observed live (2026-09-23, M0 branch).** A keyless `agent-bridge` run with no bridge directory configured produced `reaction_type_mapping` with `requested_model: agent-bridge`, `resolved_model: gpt-4o`, `model_fallback: true`: the selector fell back to a hard-coded OpenAI model and spent real tokens on a run the user believed was keyless. Before M0 the stored row said only `model: gpt-4o` with no trace of what was requested.

**Swallowed errors are failed calls.** `tools.py` catches provider exceptions and returns `status: failed|fallback` plus an `error` string while the `StepResult` keeps `source="llm"`. Provenance SHALL emit `inference_call_failed` for that shape; a chat step that never got a model response must not be counted as a completed call. Seven of eight LLM steps in the run above were of this kind and would otherwise have inflated `llm_calls`.

### 3.7.3 Accepted ≠ validated: soft advance

`harness_versions/default/harness.json` sets `run_config_defaults.proceed_on_validation_failure: true`. When every candidate fails, the loop emits `mechanism_step_soft_advance`, writes a `mechanism_synthesis` row with a failing `soft_advance` check, and then calls `_apply_candidate`, which emits `mechanism_step_accepted` with `validation_summary.passed == false`. A pathway view that draws every accepted edge the same way would show a chemically unvalidated step as accepted. See §16.6.

### 3.7.4 Candidate identity carrier already exists

Candidates are plain dicts keyed by `rank`; `BranchCandidate.intermediate_output` holds that dict and PR #37 persists it via `to_persisted_dict()`. A `candidate_id` written into the candidate dict at extraction time therefore flows through validation, branch points, resume snapshots, and backtracking without a schema migration. Ranks repeat across `mechanism_reproposal_requested` rounds, so rank alone is not an identity.

### 3.7.5 The human path bypasses the coordinator

`POST /api/runs/{id}/mechanism_steps` writes `step_started` and `step_outputs` rows directly in `api/app.py` with `source="human"`, `model="human_input"`, and never emits `step_output`. Provenance normalization must be a shared helper (`core/provenance.py`) used by both `_record_step` and the verified-step route.

### 3.7.6 No candidates-proposed event

The only record of what was proposed is `output.candidates` inside the `mechanism_step_proposal` `step_output`. Rejected-at-proposal candidates appear as a count (`rejected_candidates`). §16.1 `mechanism_candidates_proposed` is therefore new, not an enrichment.

---

# 4. ChemIllusion already has the correct deterministic visualization substrate

Do not create another canonical chemistry-state format in ChemIllusion.

The existing Mechanism Explorer stack includes:

```text
backend/app/api/mechanism_explorer.py
backend/app/schemas/mechanism_explorer.py
backend/app/services/mechanism_transaction_service.py
backend/app/services/mechanism_validation_service.py
backend/app/services/mechanism_display_service.py

frontend/src/types/mechanismExplorer.ts
frontend/src/services/mechanismExplorerApi.ts
frontend/src/components/activity-renderers/MechanismExplorerRenderer.tsx
frontend/src/components/activity-renderers/mechanismExplorerReducer.ts
```

## 4.1 Existing canonical features to reuse

`MechanismState` already provides stable atom identity.

`MechanismAtom` already carries:
- `atom_id`;
- element/isotope;
- formal charge;
- lone pairs;
- radical electrons;
- implicit H count;
- optional atom map.

`MechanismStateDiff` already carries:
- bonds formed;
- bonds broken;
- bond-order changes;
- formal-charge changes;
- lone-pair changes;
- atoms added/removed;
- total charge before/after.

`MechanismPathGraph` already supports:
- a state dictionary;
- a step dictionary;
- outgoing/incoming adjacency;
- product states;
- `abandoned_branch_ids`.

This graph is structurally well matched to Wiggum's candidate branching/backtracking.

`mechanism_display_service.py` already renders deterministic SVG plus atom/bond coordinates from a `MechanismState` without a SMILES identity round trip.

`MechanismExplorerRenderer.tsx` already overlays semantic atom/bond hotspots on that SVG and supports curved-arrow interaction.

## 4.2 Preserve the AI-free canonical state contract

`backend/app/schemas/mechanism_explorer.py` explicitly states that its canonical chemistry schemas carry no AI fields.

Keep that rule.

Do **not** add model names, probabilities, tokens, prompts, or runtime provider details directly to `MechanismState`, `MechanismAtom`, or the deterministic transaction engine.

Create a separate Observatory metadata layer that references canonical state and step IDs.

---

# 5. Problem statement

A user watching a mechanism prediction currently cannot easily answer:

1. **Where is the chemically active region?**
2. **What electron/bond changes are being proposed?**
3. **Which atoms in the new state correspond to which atoms in the old state?**
4. **Which candidates were explored and why were they rejected?**
5. **How confident is the prediction/selection?**
6. **Was this operation performed by Jev, a frontier LLM, a deterministic RDKit tool, or a human?**
7. **Did a different model become involved because of a retry or fallback?**
8. **What did the deterministic chemistry actually validate?**

The current Mermaid view instead answers mainly:

> “Which harness module is executing?”

That remains useful but should become secondary.

---

# 6. Goals

## G1. Chemical progression is the dominant live view

The accepted path is visually central. Candidates, failed candidates, branches, and backtracking appear around it in real time.

## G2. Every view focuses on changing chemistry

A deterministic `ReactionFocus` object defines the active region once. Molecule rendering, BE matrices, atom mapping, validator details, and electron-flow overlays all consume that same focus.

## G3. Every model-backed operation identifies its actual model

A user can see which model/engine is responsible for:
- conditions decisions;
- atom mapping;
- reaction-type selection;
- mechanism proposal;
- candidate ranking/selection;
- rescue/fallback;
- any future probability calculation.

## G4. Deterministic work is labeled as deterministic

A deterministic validator must never display the default LLM merely because a run-level fallback model exists.

## G5. Confidence is scientifically typed

Probability, confidence, score, mapping agreement, and validator pass/fail are distinct concepts and remain distinct in data and UI.

## G6. Live display is replayable

A reload or historical replay reconstructs the same pathway, provenance, failures, and probabilities from persisted events/snapshots.

## G7. ChemIllusion can consume the same stable runtime that Wiggum evaluates

No manual copying of evolving mechanism code into the product.

---

# 7. Non-goals

This PRD does not require, for the first release:

- replacing the full Wiggum research UI;
- exposing chain-of-thought or hidden model reasoning;
- showing raw prompts to ChemIllusion users;
- exposing training/eval/benchmark data;
- calculating DFT energies;
- calling energy “energy” if only a heuristic progress metric exists;
- generating a probability when the runtime does not actually have one;
- merging Professor Wiggum into the ChemIllusion monorepo;
- making Jev a production dependency before its adapter and chemistry performance are validated;
- changing the deterministic Mechanism Explorer student grading semantics.

---

# 8. Core UX

## 8.1 Primary layout

```text
┌────────────────────────────────────────────────────────────────────────────┐
│ Reaction                  Runtime v0.x   Models: Jev · Opus · RDKit        │
├──────────────────────────────────────────────────────┬─────────────────────┤
│                                                      │ ACTIVE STEP         │
│             LIVE MECHANISM SEARCH                    │                     │
│                                                      │ Model / engine      │
│ R ───── I1 ───── I2 ───── I3                        │ Proposal: Opus      │
│          \          \                                │ Select: Jev         │
│           ×          ○ Candidate B                   │ Validate: RDKit     │
│                       \ × Candidate C                │                     │
│                                                      ├─────────────────────┤
│ molecule structures stay dominant                    │ ΔBE                 │
│ active reaction center highlighted                   │ reaction-center     │
│ remote unchanged chemistry dimmed                    │ matrix              │
│                                                      ├─────────────────────┤
│                                                      │ ATOM LINEAGE        │
│                                                      │ focused mappings    │
├──────────────────────────────────────────────────────┴─────────────────────┤
│ compact harness: Analyze ✓ → Decide ✓ → Propose ◉ → Validate ◉ → …       │
└────────────────────────────────────────────────────────────────────────────┘
```

The center answers **where is the mechanism going?**

The side panels answer:
- what changed;
- which electrons/bonds changed;
- how atom identity propagated;
- who/what made the decision;
- how strong the model signal was;
- why deterministic validation accepted/rejected it.

## 8.2 Accepted pathway

Accepted states form the main horizontal spine.

Each accepted edge can show a compact provenance line:

```text
I2 ───────────────────────────────→ I3
    Proposal: Claude Opus 5.5 · high
    Selected: Jev 1.13 · Choice  P=0.81
    Validated: RDKit ✓
```

Exact labels depend on what actually happened. Never invent a selection model if none ran.

## 8.3 Candidate branches

At an active step:

```text
                    0.81
                ┌────────→ candidate A  ✓
                │
current ────────┼── 0.14 ─→ candidate B  ✓
                │
                └── 0.05 ─→ candidate C  × atom balance
```

Rules:
- candidate status and model probability are separate;
- rejected chemistry uses red/desaturated styling and `×`;
- confidence/probability uses a separate continuous scale;
- low probability must not look the same as invalid chemistry.

## 8.4 Harness view becomes secondary

Keep the current harness topology as:
- a compact bottom strip;
- a collapsible “Harness view”;
- or an expert/debug tab.

Do not delete it. It answers a different question.

---

## 8.5 ChemIllusion product surface (rev 3)

Facts this section relies on (ChemIllusion `origin/main` `f7a0f57d6`):

- There is **no** mechanism page URL today; mechanism features are an LMS activity workspace (`MechanismExplorerRenderer`), a chat capability (`mechanism` in `config/capabilities/visual-capabilities.json`, adapter `chemed.predict_mechanism`), and the paid `/api/mechanisms/predict-full-mechanism` route (5 AI actions) backed by the simplified `mechanism_prediction_service.py`.
- Standalone tool pages are lazy `<Route>`s in `frontend/src/App.tsx` (e.g. `/drawing-tools/ocsr`, `/drawing-tools/symmetry`, `/generator/tool/:capabilityId`).
- The Adaptive Right Dock (PR #3390) is **desktop-only**, behind `VITE_GENERATOR_RIGHT_DOCK` (on in previews/dev, **off in production**). Its Phase 1 non-goals forbid new panes, new flags and more than two context cards. Tools are tiles in `features/generatorDock/dockToolGroups.ts`; a "Mechanism & resonance" tile already exists (`resonance_explorer`).
- Jev is labeled only in Suggestions ("Legacy engine" subtitle, "Jev (recommended)" setting). There is **no shared model/engine badge component**.
- Beta patterns exist: `Badge` "Admin · Beta", the orange `role="note"` "Experimental — not a validated transition state" box (`TSCreatorNotice.tsx`), `in_development: "In development"`, fail-closed per-tool flags (`toolFeatureFlags.ts::isToolFlagEnabled`), and registry `status: experimental` (hidden from the creation hub).
- MyMol is a host-gated section of the same web app (`/mymol…`) plus a Capacitor native wrapper.

### 8.5.1 One tool, one URL

- **`/mechanism-predictor`** — standalone page: reactants + products input (SMILES or drawn), optional conditions, Run, and the live result.
- **`/mechanism-predictor/runs/:runId`** — shareable, reloadable replay of a run (owner-scoped; same data as the live view, from `/observatory`). Every entry point below lands on these URLs, so there is exactly one implementation of the tool UI.
- Registered in `visual-capabilities.json` as `mechanism_predictor` with `status: experimental` (kept out of the creation hub until promoted), distinct from the existing `mechanism` chat capability and `resonance_explorer` tile so users never confuse the three.

### 8.5.2 Generator right-dock entry

- **Now (dock Phase 1 constraints respected):** a tile **"Mechanism predictor · Beta"** in the *AI tools* group of `dockToolGroups.ts`. It opens `/mechanism-predictor` in a new tab, prefilled from the canvas (reactant/product species of the current reaction or selection; nothing sent until the user presses Run). No new pane, card or dock flag. The tile is hidden unless the tool flag (§8.5.5) is on.
- **Later (when the dock PRD allows a third card type, and the tool leaves beta):** a `GeneratorDockContextCard` "Mechanism" in the contextual stack showing the *Answer* level (§8.5.4) for the canvas reaction, with "Open full view" linking to `/mechanism-predictor/runs/:runId`. This is where the tool is expected to live once it is a real predictor.
- Because the dock is off in production, **the standalone URL is the only production entry point during beta.** That is intended: training-stage output should not appear inside Generator's main workflow by default.

### 8.5.3 Mobile

- The standalone page is responsive. At `max-width: 767px` (the existing `mobileEntryEligibility` breakpoint) it shows the **Answer** level only: final status, the accepted path as a vertical list of structures with one-line step labels and engine chips, and a "Show steps" expander. BE matrices, candidate trees and atom lineage are desktop-only (they need width).
- Input on mobile is SMILES / example picker only (no Ketcher).
- **Not** added to the MyMol Capacitor app; the mobile web page is the mobile surface. MyMol hosts may link to it later.

### 8.5.4 Three disclosure levels ("just the answer" vs "show me the steps")

| Level | Who | Shows | Default |
|---|---|---|---|
| **Answer** | "just tell me" | Overall status (completed / stopped / failed; **target reached** or not), the accepted path as structures, count of steps accepted **without validation** (never hidden), total engines used | yes, and always on mobile |
| **Steps** | students, instructors | Per step: structures before/after, electron-push arrows, deterministic validation result (✓ / × with check names), engine chips, Jev probability with calibration status | one click |
| **Details** | researchers | Candidate sets with rejected/abandoned branches, focus mask, `BE(t) | ΔBE | BE(t+1)`, atom lineage, the full call chain per step (model, latency, fallbacks) | one click; remembered per user |

The level is a view preference, never a different computation: all three render from the same `/observatory` payload.

### 8.5.5 Beta · In development labeling

- Header badge **"Beta · In development"** (orange, the `in_development` style) next to the tool name, on the page and on the dock tile.
- A persistent `role="note"` box (the `TSCreatorNotice` pattern): *"Research preview. Predictions come from Professor Wiggum, a mechanism harness that is still being trained and evaluated. Steps are checked by deterministic chemistry validators, but the mechanism is not guaranteed. [How it works](https://github.com/scottmreed/professor-wiggum)"*. The link points to this repository.
- Footer on every result: runtime release (`runtime_version`, `git_sha` short, harness name) from the manifest, so a user report can be traced to the exact Wiggum release.
- Gating: backend `MECHANISM_PREDICTOR_ENABLED` (config.py boolean, default false) and a fail-closed tool flag `mechanism_predictor` in `toolFeatureFlags.ts`; beta audience = admins + an allow-list of testers. Registry `status: experimental`. Promotion out of beta is a flag and label change, not a rebuild.

### 8.5.6 Jev vs LLM vs deterministic, on every step

- New shared component **`EngineBadge`** (ChemIllusion has none): text-first chip, neutral colors (§14.1), one per call in the step's chain, built from `step_output.provenance` / `inference_call_*`:
  - **"Jev · Choice"** (decision model; revision on hover, e.g. `typesafe/jev-1.13-20260917`; "calibrated" or "not calibrated" per §15.2);
  - **"LLM · <model>"** (e.g. `LLM · claude-opus-5.5 · high`; "fallback from X" when `model_fallback` is set);
  - **"Deterministic · RDKit"** for validators and the mapped-state executor;
  - **"Human"** for verified submissions.
- A Jev→LLM fallback shows both chips in order, the Jev chip marked "failed → fell back". The run header lists the inventory ("Models used: Jev 1.13 · Claude Opus 5.5 · Deterministic: RDKit").
- At the **Answer** level the chips collapse to a per-step summary ("LLM proposed · RDKit validated", "Jev chose · LLM proposed"), never hidden.

### 8.5.7 Relationship to the existing paid predictor

`/api/mechanisms/predict-full-mechanism` and the `mechanism` chat capability keep their current behavior during beta. §20's `MECHANISM_RUNTIME_V2_ENABLED` adapter is the later migration path; the new page does not reuse the old service.

### 8.5.8 Harness evolution stays in this repository

- ChemIllusion never edits prompts, few-shots, harnesses, validators or topology. It consumes a tag.
- Promotion: a Wiggum maintainer tags `runtime-vX.Y.Z` after the change clears the evidence policy here (`docs/change_evidence_policy.md`, eval tiers). ChemIllusion bumps `WIGGUM_RUNTIME_REF` in one PR; the manifest makes the bump visible on every run.
- Product runs are **not** fed back into evolution automatically. If product traces are later used for training, that is a separate, consented export into this repository's evidence flow — never an in-product mutation loop.

# 9. ReactionFocus: one deterministic active-region definition

## 9.1 Requirement

Every elementary candidate transition SHALL produce a deterministic `ReactionFocus`.

Proposed schema:

```json
{
  "schema_version": "reaction_focus.v1",
  "source_state_id": "s2",
  "target_state_id": "s3",
  "core_atom_ids": ["a4", "a7", "a9"],
  "context_atom_ids": ["a1", "a5", "a6", "a8"],
  "unchanged_atom_ids": ["a2", "a3", "a10"],
  "changed_bonds": [
    {
      "atom_ids": ["a4", "a7"],
      "order_before": 2,
      "order_after": 1
    }
  ],
  "changed_formal_charges": ["a7"],
  "changed_lone_pairs": ["a7"],
  "changed_hydrogens": [],
  "electron_flow_atom_ids": ["a4", "a7", "a9"],
  "matrix_atom_ids": ["a1", "a4", "a5", "a6", "a7", "a8", "a9"]
}
```

## 9.2 Derivation

Core atoms are the union of atoms involved in:
- formed/broken bonds;
- bond-order changes;
- formal-charge changes;
- lone-pair changes;
- materialized proton/H changes;
- explicit electron actions;
- non-zero `ΔBE` entries.

Context defaults to graph radius 1 around the core, with deterministic completion rules such as:
- include the rest of a ring when the core intersects that ring and clipping it would obscure interpretation;
- retain atoms required to show the immediate functional group;
- retain species identity labels where multiple components exist.

The full molecule remains available.

## 9.2.1 Implementation (delivered 2026-09-23, `core/reaction_focus.py`)

`build_reaction_focus(reaction_smirks, electron_pushes=, bond_electron_deltas=, context_radius=1)` implements `reaction_focus.v1` on Wiggum:

- atom ids are `a<map>` from the candidate's mapped SMIRKS; persistent ids (`mapped_state.MappedState.map_to_pid`) map onto them 1:1 when the mapped loop state is on;
- core = changed bonds ∪ changed formal charge ∪ changed hydrogen count ∪ changed non-bonding electrons ∪ electron-push participants ∪ non-zero ΔBE entries ∪ atoms present on one side only (`unbalanced_atom_ids`);
- context = BFS radius 1 over the union of both sides' adjacency, plus every ring (SSSR on either side) that intersects the core, taken whole;
- output is sorted by map number, so it is independent of species order and byte-identical on replay;
- an unparsable SMIRKS returns an empty focus with `error` set instead of raising.

## 9.3 Reuse ChemIllusion `MechanismStateDiff`

ChemIllusion already computes most of this information in `MechanismStateDiff`.

The runtime-to-product adapter should translate Wiggum's mapped/persistent state into canonical ChemIllusion `MechanismState` objects, then either:

1. use the ChemIllusion deterministic diff engine as the product projection; or
2. verify that the runtime-provided diff agrees and use the runtime result.

Do not implement independent “changed atom” heuristics in React.

---

# 10. Bond-electron matrix visualization

## 10.1 Data model

For the selected candidate transition expose:

```json
{
  "schema_version": "bond_electron_view.v1",
  "convention": "ugi_flower_kekule_v1",
  "atom_ids": ["a4", "a7", "a9"],
  "before": [[...], [...], [...]],
  "delta": [[...], [...], [...]],
  "after": [[...], [...], [...]],
  "electron_delta_sum": 0,
  "full_atom_count": 31,
  "is_focus_projection": true
}
```

The implementation must document the matrix convention:
- atom ordering;
- non-bonding/lone-pair representation;
- shared bonding-electron representation;
- aromatic/Kekulé convention;
- explicit-H policy;
- charge/radical handling.

The first implementation should follow the Ugi/FlowER-style electron-accounting vocabulary already motivating this feature, with a versioned convention so it can evolve safely.

**Existing substrate (rev 2).** Wiggum already computes per-step bond-electron deltas; the BE view is a projection of them, not a new engine:

- `tools.py::predict_mechanistic_step` returns `bond_electron_deltas` and `bond_electron_validation` (`dbe_source: inferred_from_electron_pushes | explicit`) for every candidate.
- `core/mechanism_moves.py::reaction_bond_deltas(reaction_smirks)` and `implied_bond_deltas(moves)` derive deltas from a mapped SMIRKS or from the `|mech:v1;lp:4>2;sigma:2-3>3|` CXSMILES move block (`docs/mechanism_move_notation.md`); PR #39 fixed explicit-hydrogen handling.
- `core/mapped_state.py::MappedState` / `execute_candidate` give the persistent-ID atom order for both `BE(t)` and `BE(t+1)`.

`bond_electron_view.v1` SHALL be built from these three inputs with the persistent atom ID as the row/column key.

## 10.1.1 Convention `ugi_flower_kekule_v1` as implemented (`core/bond_electron.py`)

| Item | Decision |
|---|---|
| Atom ordering | rows/columns follow the requested `atom_ids` (default: all mapped atoms sorted by map number) |
| Off-diagonal `(i, j)` | shared bonding electrons = 2 × Kekulé bond order; 0 when not bonded |
| Aromatic / Kekulé | molecules are Kekulized with aromatic flags cleared before counting, so ring bonds are 2 or 4 electrons, never 3 |
| Diagonal `(i, i)` | non-bonding valence electrons with hydrogens folded in: `outer − formal_charge − Σ bond orders − n_H`; radical electrons are included |
| Explicit H | mapped `[H:n]` atoms are ordinary rows; implicit H is folded into the heavy atom's diagonal |
| Charge | enters the diagonal through the formula above, so a lone-pair donation shows as −2 on the donor diagonal and +2 on the new bond cell |
| Unmapped atoms | not rows; a bond to one still counts in the mapped atom's diagonal |
| Conservation | `electron_delta_sum = Σ_diag Δ + Σ_{i<j} Δ`; `conserved` additionally requires every atom with a changed row to be inside the projection (`missing_changed_atom_ids == []`), because a partial projection can sum to zero by coincidence (C2 + Br3 alone in an SN2) |
| `is_focus_projection` | true whenever the rows were chosen by a `ReactionFocus`, even when the focus covers every atom of a small step; `full_atom_count` carries the total |

Worked example (SN2 `[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3]`): `BE(a2,a3)` 2→0, `BE(a2,a4)` 0→2, `BE(a3,a3)` 6→8, `BE(a4,a4)` 8→6, `Σ ΔBE = 0`.

## 10.2 Default display

Do not show the entire `N × N` matrix for a large molecule.

Default to `ReactionFocus.matrix_atom_ids`.

Show:

```text
BE(t)       ΔBE       BE(t+1)
```

with `ΔBE` visually dominant.

Only non-zero or changed cells should have high visual weight.

**Delivered in the Wiggum UI (2026-09-23):** each candidate row in the Mechanism-search panel is a keyboard-accessible `<details>`; expanding it shows the candidate's mapped SMIRKS, the focus summary (changed bonds with orders, charge / lone-pair / H-count atoms, context atoms, unchanged count), the changed entries as text (`a2–a3: 2 → 0 (Δ −2)`, `a3 (non-bonding): 6 → 8 (Δ +2)`), and three tables `BE(t) | ΔBE | BE(t+1)` over `matrix_atom_ids` with row/column headers and captions. Zero cells are dimmed; changed cells are bold with the sign colour-coded, and every value stays numeric (§30). The conservation line shows `Σ ΔBE`, the convention id and `projected/full` atom counts. Expanded rows stay expanded across live refreshes. Spine states carry RDKit structure thumbnails (cached per SMILES, one render call per refresh for new species). Still open: linked hover between matrix cells and structure atoms (§10.3) and the atom-lineage panel (§11.4).

## 10.3 Linked interaction

Hover/click behavior:
- matrix cell → corresponding atom/bond highlighted in molecule;
- atom/bond → corresponding matrix row/column/cell highlighted;
- electron arrow → corresponding non-zero `ΔBE` entries highlighted.

Show deterministic conservation:

```text
Σ ΔBE = 0  ✓
```

when the matrix convention makes this exact.

---

# 11. Atom identity and mapping visualization

## 11.1 Persistent identity is primary

Use the persistent identity work in Wiggum PR #35 and ChemIllusion's existing stable `MechanismAtom.atom_id`.

Atom identity should be stable across the pathway, not reconstructed as an incidental label for each frame.

## 11.2 Product adapter

The adapter SHALL map runtime persistent IDs deterministically onto ChemIllusion `atom_id` values.

Example:

```text
Wiggum persistent atom 17 → ChemIllusion atom_id "a17"
```

The exact string convention may differ, but it must be:
- stable for the run;
- collision-free;
- replayable;
- independent of current SMILES position.

## 11.3 Default mapping display

The default molecule view emphasizes mappings only in the reaction focus.

Remote unchanged atoms:
- retain identity internally;
- are visually dimmed;
- do not need a visible atom number by default.

A full “show atom identities” toggle remains available.

## 11.4 Atom-lineage panel

For the focused atoms:

```text
Reactant      I1           I2           Product

a4  ───────── a4 ───────── a4 ───────── a4
a7  ───────── a7 ───────── a7 ───────── a7
a9  ───────── a9 ───────── a9 ───────── a9
```

Hovering one lineage highlights that atom in every displayed state.

## 11.4.1 As implemented (2026-09-23)

- `mechanism_step_accepted.atom_identity` (`atom_identity.v1`): built in `RunCoordinator._atom_identity_payload` from the mapped loop state that `_record_smirks_state_agreement` advances for every accepted candidate — `mapped_species` (mapped SMILES), `atoms` (`pid`, `map_number`, `element`, `component`), and the identity counters from the executor record (`preserved_id_count`, `new_ids`, `lost_ids`, `identity_resynced`, `smirks_state_agreement`, `executed`). `identity_source` is `persistent` only when `loop_state_mapping == "mapped"`; in the default `stripped` loop it is `derived`, because the ids come from re-executing the accepted candidate, so per-candidate `a<map>` ids in the ReactionFocus are not guaranteed to coincide with them.
- `GET /api/runs/{id}/observatory` adds `atom_lineage` (`atom_lineage.v1`): rows are persistent ids with element, columns are the accepted states; each cell is the atom's map number or absent; `new_at_step` / `lost_at_step` come from the counters (falling back to first/last presence); `changed_pids` lists atoms that appeared or disappeared. Accepted-path entries carry an `identity` summary and states carry `mapped_species` / `atoms`. Legacy runs without identity project an empty lineage.
- UI: an **Atom lineage** `<details>` under the search spine shows the table (changed atoms by default, "show all" toggle) and states the identity source. Hover-linking a lineage row to the structures (§11.4 last paragraph) and the focus-only default (§11.3) remain open; both need the `a<map>` ↔ pid bridge, i.e. the mapped loop.

## 11.5 Mapping confidence must not be conflated with mapping quality

Potential values include:
- LLM self-reported mapping confidence;
- Jev candidate-selection probability;
- RDKit map-check pass/fail;
- benchmark mapping agreement in eval runs;
- persistent-state execution agreement.

They SHALL be separately typed.

A production user should normally see:
- mapping status;
- ambiguity if present;
- measured deterministic checks.

Benchmark agreement is an evaluation metric and should not be exposed when no benchmark is available.

---

# 12. Model and engine provenance — M0 requirement

This is the first implementation milestone.

## 12.1 Why a single `model` string is no longer enough

A logical operation can become:

```text
Jev decision
   ↓ low/ambiguous result
frontier LLM fallback
   ↓
deterministic validator
```

or:

```text
frontier LLM generates 3 candidates
   ↓
Jev ranks candidates
   ↓
RDKit / mapped-state executor validates each
```

The user should not be told that “Jev made the step” when Jev only ranked it, or that “Claude validated it” when RDKit performed the validation.

## 12.2 Canonical provenance schema

Add an Observatory/runtime metadata type:

```json
{
  "call_id": "call_01J...",
  "engine": "llm",
  "role": "mechanism_proposal",
  "provider": "openrouter",
  "requested_model": "anthropic/claude-opus-5.5",
  "resolved_model": "anthropic/claude-opus-5.5",
  "resolved_model_version": null,
  "reasoning_level": "high",
  "decision_type": null,
  "attempt": 1,
  "retry_index": 0,
  "fallback_from_call_id": null,
  "status": "completed",
  "latency_ms": 1840,
  "usage": {
    "input_tokens": 4200,
    "output_tokens": 650
  }
}
```

Supported `engine` values at minimum:

```text
llm
jev
deterministic
human
agent_bridge        # Wiggum/research compatibility
```

Supported `role` values should be extensible, including:

```text
conditions_decision
missing_chemistry
global_mapping
reaction_type
mechanism_proposal
candidate_ranking
candidate_rescue
step_mapping
validator
mapped_state_executor
completion_check
```

For Jev:

```json
{
  "engine": "jev",
  "role": "reaction_type",
  "requested_model": "typesafe/jev-latest",
  "resolved_model": "jev-1.13.0",
  "decision_type": "choice"
}
```

If the provider cannot report the resolved revision, do not fabricate one:

```json
{
  "requested_model": "typesafe/jev-latest",
  "resolved_model": null
}
```

Production should prefer a pinned Jev model/revision before confidence thresholds are treated as stable.

**Mapping onto the existing `DecisionRecord` (rev 2).** Jev calls already produce a typed record; the provenance schema is populated from it rather than duplicated:

| Provenance field | `DecisionRecord` / `decision_trace` source |
|---|---|
| `call_id` | `request_id` (provider id when returned, else `local-<uuid>`) |
| `engine` | `decision_engine` (`"jev"`) |
| `role` | from the question: `reaction_type`, later `conditions_decision`, `global_mapping`, `candidate_ranking` |
| `provider` | `provider` (`"openrouter"`) |
| `requested_model` | `JevConfig.model` / catalog id (`typesafe/jev-1.13`) |
| `resolved_model` | `model_version` as reported by the provider (`typesafe/jev-1.13-20260917`); `null` if not returned |
| `decision_type` | `decision_type` (`choice` / `score` / `noul`) |
| `status` | `completed` when `called and not failure`; `failed` when `called and failure`; no record when `called == False` (no request sent) |
| `latency_ms`, `usage` | same-named fields |
| `fallback_from_call_id` | set on the LLM call that answered after a failed Jev call in the same step |

## 12.3 Deterministic provenance

Deterministic operations have no LLM model.

Example:

```json
{
  "engine": "deterministic",
  "role": "validator",
  "provider": null,
  "requested_model": null,
  "resolved_model": null,
  "tool": "rdkit",
  "tool_version": "..."
}
```

This requirement specifically fixes a risk in the current Wiggum `_record_step()` behavior: `resolved_model` falls back to the run's `step_models`/default model even for steps whose `source` is deterministic.

**Requirement:** normalize provenance based on the actual `source`/engine. A deterministic `StepResult` SHALL NOT acquire a user-visible model merely because `_step_model()` has a configured fallback.

## 12.4 Human provenance

Verified/manual steps use:

```json
{
  "engine": "human",
  "role": "mechanism_proposal",
  "model": null
}
```

The existing verified-step API already stores `source="human"` and `model="human_input"`; the Observatory adapter should normalize this to a human provenance record.

---

# 13. New inference-call events

## 13.1 Do not rely only on `step_output`

`step_output` is a logical-step completion record. It cannot accurately represent:
- retries;
- multiple calls inside one step;
- Jev followed by LLM fallback;
- model-provider fallback;
- candidate ranking after proposal.

Introduce call-level events.

**Two delivery stages (rev 2).**

- **M0a — derived.** `_record_step()` derives `inference_call_completed` / `inference_call_failed` from the `StepResult` (`source`, `model`, `output.model_used`, `token_usage`) and from `output.decision_trace` entries, then emits `step_output` with a `provenance` summary pointing at those call IDs. Events are emitted after the fact, so `started_at` is the step start time and `latency_ms` is the step duration unless the record carries its own.
- **M0b — live (delivered 2026-09-23).** `core/call_recorder.py`: `_mark_step_started` opens a per-thread `CallContext`; `llm.get_chat_model` returns a `RecordingChatAdapter` while it is open, which emits `inference_call_started` before the provider request and `inference_call_completed` / `inference_call_failed` after it with measured `latency_ms` and normalized usage. A request that follows a failed one in the same step (the `tools.py` `fallback_model` retry) carries `fallback_from_call_id` and `model_fallback: true`. `_record_step` uses the live records instead of deriving a call, so nothing is counted twice, and closes the context; `execute_run` clears any stale context. Live evidence (run `66130e90…`, keyless bridge with a 3 s timeout and invalid provider keys): every LLM step produced `started → failed (latency ≈ 3050 ms)`, and `reaction_type_mapping` produced the chain `agent-bridge failed → gpt-4o failed (401)` with the link set — the previously hidden fallback now visible without spending tokens.
- **Still derived.** Jev calls come from `DecisionRecord.to_trace()` (which already carries provider latency and `request_id`); a live `inference_call_started` for Jev is a small follow-up in `JevDecisionClient`. Adapters constructed outside a step (CLI utilities, calibration scripts) stay unwrapped by design.

The UI must not depend on which stage produced an event; both carry `event_schema_version`.

## 13.2 Events

### `inference_call_started`

```json
{
  "call_id": "call_...",
  "step_name": "reaction_type_mapping",
  "attempt": 1,
  "retry_index": 0,
  "engine": "jev",
  "role": "reaction_type",
  "requested_model": "typesafe/jev-latest",
  "resolved_model": null,
  "reasoning_level": null,
  "decision_type": "choice",
  "candidate_id": null,
  "started_at": 1790170000.12
}
```

### `inference_call_completed`

```json
{
  "call_id": "call_...",
  "step_name": "reaction_type_mapping",
  "engine": "jev",
  "role": "reaction_type",
  "status": "completed",
  "resolved_model": "jev-1.13.0",
  "latency_ms": 110,
  "usage": {},
  "result_summary": {
    "selected": "rt_042",
    "selected_probability": 0.78
  }
}
```

### `inference_call_failed`

Contains:
- call ID;
- error class/code safe for client display;
- fallback planned/triggered;
- no secret/provider credentials.

## 13.3 Step-level events also gain a provenance summary

`step_started` should include the planned engine/model if known.

`step_output` should include:

```json
{
  "provenance": {
    "primary_call_id": "call_...",
    "call_ids": ["call_...", "call_..."],
    "primary_engine": "llm",
    "primary_model": "anthropic/claude-opus-5.5"
  }
}
```

This is a summary pointer, not the only source of truth.

## 13.4 Accepted candidate correlation

Introduce a stable `candidate_id`.

Do not rely solely on `candidate_rank`, because rank can repeat across:
- retries;
- reproposals;
- topology rounds.

Propagate `candidate_id` through:
- proposed event;
- validation result;
- branch point;
- accepted step;
- failed path;
- backtrack event;
- probability result.

---

# 14. User-facing model display rules

## 14.1 Model identity must not consume the confidence color channel

Confidence/probability already needs color.

Therefore model provenance should use:
- labeled neutral badges;
- small provider/model icon only as secondary treatment;
- text as the primary identifier.

Example:

```text
Proposal   Claude Opus 5.5 · high
Selection  Jev 1.13 · Choice
Validation RDKit · deterministic
```

Do not make “Claude = blue, Jev = green” the primary encoding.

## 14.2 Central path edge

Show the model that **generated the chemical candidate** as the primary edge provenance.

If another engine selected it, add a secondary chip.

Example:

```text
[Opus 5.5] ── P 0.81 [Jev Choice] ──→
```

## 14.3 Inspector

Clicking a step opens the full provenance chain:

```text
Step 3

1. Candidate generation
   Claude Opus 5.5
   reasoning: high
   1.84 s

2. Candidate selection
   Jev 1.13
   Choice
   P(selected) = 0.81
   probability not yet calibrated

3. Mapped-state execution
   deterministic

4. Bond/electron validation
   RDKit / deterministic
   passed

5. Atom-balance validation
   deterministic
   passed
```

## 14.4 Run header

Show a compact inventory:

```text
Models used: Jev 1.13 · Claude Opus 5.5
Deterministic: RDKit · mapped-state executor
```

This is derived from actual completed call events, not only run configuration.

---

# 15. Confidence and probability

## 15.1 Canonical typed value

Never store a naked `confidence: 0.81` without meaning.

Use:

```json
{
  "kind": "candidate_probability",
  "value": 0.81,
  "source_engine": "jev",
  "source_model": "jev-1.13.0",
  "calibrated": false,
  "calibration_version": null,
  "candidate_set_id": "cs_...",
  "question_id": "candidate_choice"
}
```

Initial `kind` values:

```text
candidate_probability
choice_probability
score_probability
noul_probability
llm_self_confidence
mapping_agreement
ensemble_vote_fraction
heuristic_score
```

Deterministic pass/fail is not a probability.

## 15.2 Color

Use a continuous confidence gradient for numeric values.

Do not initially hard-code labels such as:
- “high” above 0.8;
- “medium” above 0.5.

The Wiggum Jev PRD already warns that Jev probability calibration may depend on:
- question type;
- model version;
- chemistry domain.

Calibration is per question and per Jev revision. On `c9248b8` exactly one surface is calibrated: reaction-type `Choice` on `typesafe/jev-1.13-20260917` (`docs/calibration/jev_reaction_type_2026-09-23.md`, n=72, ECE 0.056). For that surface emit `calibrated: true`, `calibration_version: "jev_reaction_type_2026-09-23"`. Every other Jev question and every LLM self-confidence emits `calibrated: false` and shows:
- number;
- relative color;
- tooltip `Model probability — not calibrated`.

A Jev revision change invalidates `calibrated` until the calibration script is re-run.

## 15.3 Candidate-set normalization

If Jev Choice produces a probability distribution over candidates, display probabilities in that candidate set.

Do not compare an 0.81 from one decision surface directly against an 0.81 from another without calibration.

## 15.4 No fake step probability

The mechanism proposal LLM may provide:
- a rank;
- self-reported confidence;
- nothing.

Do not turn rank into probability.

If the runtime has no legitimate candidate probability, omit the probability encoding for that edge.

## 15.5 Future path probability

Path probability is deferred.

A naive product of per-step probabilities is length-dependent and may imply more statistical meaning than the system has earned.

If added later, it requires:
- a documented probabilistic interpretation;
- calibration;
- dependency assumptions or an alternative path-score definition.

---

# 16. Live mechanism event contract

The runtime should emit chemistry-oriented events in addition to generic harness events.

## 16.1 `mechanism_candidates_proposed`

```json
{
  "step_index": 3,
  "proposal_round": 1,
  "current_state_id": "s2",
  "candidate_set_id": "cs_3_1",
  "candidates": [
    {
      "candidate_id": "c_3_1_a",
      "rank": 1,
      "proposed_state_id": "s3a",
      "reaction_smirks": "...",
      "electron_actions": [],
      "proposal_call_id": "call_..."
    }
  ]
}
```

## 16.2 `candidate_probability_updated`

Optional and only when a real score/probability exists.

```json
{
  "candidate_set_id": "cs_3_1",
  "candidate_id": "c_3_1_a",
  "confidence": {
    "kind": "candidate_probability",
    "value": 0.81,
    "source_engine": "jev",
    "source_model": "jev-1.13.0",
    "calibrated": false
  }
}
```

## 16.3 `candidate_validation_started`

References candidate ID.

## 16.4 `candidate_validation_result`

Contains:
- canonical state diff;
- ReactionFocus;
- BE view;
- deterministic findings;
- mapping/identity summary;
- `smirks_state_agreement`;
- accepted/rejected status.

### 16.4.1 As implemented (2026-09-23)

`RunCoordinator._emit_candidate_validation_result` emits one event per candidate attempt right after the deterministic validators run, for validated and rejected candidates alike:

```json
{
  "event_schema_version": "mechanism_observatory_event.v1",
  "step_index": 3, "attempt": 3, "retry_index": 0,
  "candidate_id": "c3-r1-9f2a1c0d", "candidate_rank": 1,
  "accepted": true, "validation": {"passed": true, "checks": []}, "failed_checks": [],
  "reaction_smirks": "...", "current_state": [], "resulting_state": [], "predicted_intermediate": "...",
  "reaction_focus": {"schema_version": "reaction_focus.v1"},
  "bond_electron_view": {"schema_version": "bond_electron_view.v1", "is_focus_projection": true},
  "smirks_state_agreement": null, "projection_error": null
}
```

`accepted` here means "passed the validators"; the eventual `mechanism_step_accepted.acceptance_kind` says whether the step was applied and how. Projection failures set `projection_error` instead of failing the step. The mapping-lineage payload (§11.4) and the persistent-id adapter remain open for the rest of M1.

## 16.5 Existing events retained

Continue using:
- `mechanism_step_accepted`;
- `mechanism_retry_started`;
- `mechanism_retry_failed`;
- `mechanism_retry_exhausted`;
- `branch_point_created`;
- `failed_path_recorded`;
- `backtrack`;
- `completion_check`.

Enrich rather than replace unless migration requires a versioned event schema.

## 16.6 `mechanism_step_accepted` acceptance kind (rev 2)

Add to the accepted event:

```json
{
  "candidate_id": "c3-r1-9f2a1c0d",
  "acceptance_kind": "validated"
}
```

`acceptance_kind` values:

```text
validated              all enabled validators passed
soft_advance           proceed_on_validation_failure / balance_pending; validation failed
backtrack_alternative  a stored branch alternative applied after backtracking
human                  verified-mode submission
```

The pathway view SHALL render `soft_advance` edges with a distinct "unvalidated" treatment and SHALL never count them toward a "validated steps" total. `branch_point_created` gains `chosen_candidate_id` and `alternative_candidate_ids`; `backtrack` and `failed_path_recorded` gain `candidate_id`.

---

# 17. Wiggum event persistence and replay

## 17.1 Replay is a product requirement

The Observatory should be reconstructable from:
- a run snapshot;
- ordered persisted events.

A frontend reload must not lose rejected branches or model provenance.

## 17.2 Branch details

PR #37 (`aff1b3f`) persists branch points with their full untried alternatives and the mapped loop state in `run_resume_state` snapshots, so resume/backtrack no longer depends on process memory. What remains missing for replay is the per-candidate history: which candidates were proposed each round, which failed which check, and which call produced them. Snapshots are latest-state; the Observatory needs the ordered event record.

For every candidate that becomes a visible branch, persist:
- candidate ID;
- source/target state references;
- candidate rank;
- proposal provenance;
- selection probability if any;
- validation result;
- final branch status.

`failed_path_recorded` should carry enough information to reconstitute the path without relying on process memory.

## 17.3 Schema version

Every Observatory event SHALL include:

```text
event_schema_version
```

Start at:

```text
mechanism_observatory_event.v1
```

---

# 18. ChemIllusion Observatory metadata schema

Create a new schema rather than modifying the AI-free Mechanism Explorer canonical schema.

Recommended files:

```text
backend/app/schemas/mechanism_observatory.py
frontend/src/types/mechanismObservatory.ts
```

High-level shape:

```python
class ObservatoryRun:
    run_id
    runtime_version
    harness_version
    graph: MechanismPathGraph
    step_metadata: dict[str, ObservatoryStepMetadata]
    candidate_metadata: dict[str, CandidateMetadata]
    calls: dict[str, InferenceCall]
    active_step_id
```

`ObservatoryStepMetadata` references a `MechanismStep.step_id` but contains non-canonical metadata:

```text
proposal provenance
selection provenance
confidence values
ReactionFocus
BondElectronMatrixView
runtime validation details
branch status
timestamps
```

This preserves the strong existing separation:

```text
Mechanism Explorer schemas = chemistry truth/state
Observatory schemas        = runtime/search/provenance metadata
```

---

# 19. Reuse and refactor the existing React Mechanism Explorer

## 19.1 Do not duplicate its SVG chemistry renderer

`MechanismExplorerRenderer.tsx` already has:
- deterministic SVG;
- atom coordinates;
- atom hotspots;
- bond hotspots;
- curved-arrow interaction;
- canonical state handling.

Extract reusable presentation components where necessary.

Recommended extraction:

```text
frontend/src/components/mechanism/
    MechanismStateCanvas.tsx
    ElectronArrowOverlay.tsx
    AtomHighlightOverlay.tsx
```

Then both:
- the student Mechanism Explorer;
- the prediction Observatory

can reuse them.

## 19.2 New Observatory components

Recommended:

```text
frontend/src/features/mechanism-observatory/
    MechanismObservatory.tsx
    MechanismPathView.tsx
    MechanismCandidateBranch.tsx
    ActiveStepInspector.tsx
    BondElectronMatrix.tsx
    AtomLineageView.tsx
    ValidatorSummary.tsx
    ModelProvenanceBadge.tsx
    ConfidenceIndicator.tsx
    HarnessMiniMap.tsx
    observatoryReducer.ts
    observatoryEvents.ts
```

Do not put this functionality into the already-large student `MechanismExplorerRenderer.tsx`.

---

# 20. Existing ChemIllusion mechanism predictor: migration plan

ChemIllusion currently has:

```text
backend/app/services/mechanism_prediction_service.py
```

It contains a simplified Wiggum-inspired loop with fixed model constants including:
- `gpt-4o-2024-08-06`;
- `openai/gpt-5.6-sol`;
- `gpt-4o-mini`.

It performs useful product functionality but does **not** represent the current Professor Wiggum architecture:
- no Wiggum branch/backtrack search;
- no current persistent mapped-state executor;
- no Wiggum validator stack;
- only lightweight SMILES validation in the full-mechanism loop;
- no current live SSE event contract.

Do not expand this service into a second implementation of Professor Wiggum.

## 20.1 Compatibility strategy

Keep its public interfaces temporarily because current callers include:
- `backend/app/api/mechanisms.py`;
- ChemEd tool execution;
- MCP authenticated tooling;
- possibly other internal flows.

Add a feature-gated adapter:

```text
MECHANISM_RUNTIME_V2_ENABLED
```

When enabled, a full-mechanism request is sent to the Mechanism Runtime and the final accepted path is translated back to `FullMechanismResult` for legacy callers.

The Observatory uses the new runtime contract directly.

After parity and adoption, the simplified predictor can be deprecated rather than maintained in parallel.

---

# 21. Production Mechanism Runtime service

> **Rev 3:** read "service" in this section as "release artifact". Production runs the runtime in-process in the ChemIllusion API (§2.3). §21.2–§21.4 (include/exclude lists, assets, manifest) apply unchanged to the tagged checkout.

## 21.1 First deployment can remain in the Wiggum repo

Do not split repositories prematurely.

Add a runtime-specific build target, for example:

```text
Dockerfile.runtime
mechanistic_agent/api/runtime_app.py
runtime_assets/
```

The runtime container should copy/import only production-required code and assets.

## 21.1.1 As implemented (2026-09-23, first slice)

- `mechanistic_agent/api/runtime_app.py::create_runtime_app(base_dir)` builds the research app and **re-mounts only the allow-listed routes** (`RUNTIME_ALLOWED_PATHS`: create/start/stop/resume/get run, `events` SSE, `flow`, `observatory`, verified `mechanism_steps` + `steps/{step}/verify`, `molecules/render`) onto a fresh FastAPI app with no docs/OpenAPI and no static UI. Handlers are therefore byte-identical to the ones Wiggum evaluates; evals, leaderboards, examples, traces, curation, memory, harness editing and RAlph are absent by construction (`RUNTIME_EXCLUDED_SURFACES`), and a renamed research route fails the build at import time rather than silently dropping out of the product surface.
- **Auth (§25.1):** every request except `/healthz` requires `Authorization: Bearer $MECHANISTIC_RUNTIME_TOKEN` (constant-time compare); with no token configured the app answers 503 to everything but `/healthz` — fail closed.
- **Manifest (§21.4):** `GET /v1/mechanism/version` → `runtime_manifest.v1` with `runtime_version`, `git_sha` (from `MECHANISTIC_RUNTIME_GIT_SHA`, set as a Docker build arg, else `git rev-parse`), `harness_name`, prompt/skill/harness bundle hashes, combined `harness_version`, `event_schema_version`, `observatory_schema_version`, `reaction_focus_version`, `be_convention`, the allow-list and the excluded surfaces.
- **Image:** `Dockerfile.runtime` (python:3.11-slim, `uvicorn … --factory`, healthcheck on `/healthz`, `MECHANISTIC_DATA_DIR=/data`); `.dockerignore` excludes `.git`, `.env*`, `data/`, `traces/`, `local_contributions/`, `training_data/leaderboard_holdout/`, `training_data/local_legacy/`, tests, docs and scripts (§34 "runtime image contains no holdout/eval corpus").
- **Still open for M3:** a durable `RunStateStore` adapter (Postgres) instead of SQLite in `/data`, the `runtime_assets/` manifest-driven bundle so the image does not need general `training_data/` access (today `create_app` still loads the FlowER example menu at startup), path prefixing under `/v1/mechanism` (left to the ChemIllusion facade), and a leaner factory that does not instantiate the research executors.

## 21.2 Include

- run coordinator / mechanism loop;
- approved model adapter(s);
- approved Jev adapter when implemented;
- production prompts/few-shots required for inference;
- production harness;
- mapping;
- persistent mapped-state executor;
- RDKit chemistry;
- arrow/electron action conversion;
- deterministic validators;
- retries / branches / backtracking;
- ReactionFocus;
- BE calculation;
- confidence/probability records;
- provenance;
- runtime event store;
- SSE API.

## 21.3 Exclude

- training corpus;
- eval sets;
- known benchmark mechanisms;
- scoring against ground truth;
- leaderboard routes;
- curriculum publication;
- island evolution;
- RAlph mutation/evolution controls;
- prompt/harness editing;
- PR creation;
- evidence export UI;
- local API-key configuration UI;
- research dashboards.

Some files currently stored under `training_data/` may actually be inference assets, such as reaction-template data. Move required production assets into a clearly named, versioned runtime bundle rather than giving the production image general access to `training_data/`.

Recommended:

```text
runtime_assets/
    manifest.json
    reaction_templates.json
    harness.json
    prompts/
    few_shot/
```

## 21.4 Runtime release manifest

Example:

```json
{
  "runtime_version": "0.1.0",
  "git_sha": "...",
  "harness_name": "production",
  "harness_sha256": "...",
  "prompt_bundle_hash": "...",
  "validator_version": "...",
  "reaction_focus_version": "reaction_focus.v1",
  "be_convention": "ugi_flower_kekule_v1",
  "approved_models": [
    "anthropic/claude-opus-5.5",
    "jev-1.13.0"
  ]
}
```

Every run records this manifest identity.

---

# 22. Runtime API

> **Rev 3:** in production these routes are served by the ChemIllusion API under `/api/mechanism-predictor/…` (owner-scoped, user-authenticated), backed by the embedded runtime. Wiggum's `runtime_app.py` exposes the same shapes for dev and contract tests.

Suggested product-oriented API:

```text
POST /v1/mechanism/runs
POST /v1/mechanism/runs/{id}/start
POST /v1/mechanism/runs/{id}/stop
GET  /v1/mechanism/runs/{id}
GET  /v1/mechanism/runs/{id}/events
GET  /v1/mechanism/runs/{id}/observatory
```

`/observatory` is an aggregate replay/snapshot projection, not a replacement for SSE.

**Delivered on Wiggum (2026-09-23):** `GET /api/runs/{id}/observatory` → `core/observatory.py::build_observatory(events)`, schema `mechanism_observatory.v1`, built from persisted events only:

- `states` — `s0` (reactants) plus `st:<candidate_id>` for every candidate's resulting state, each with `kind` (`initial | candidate | validated | rejected | incomplete | constraint_rejected | branch_alternative | accepted | soft_advance | abandoned`);
- `accepted_path` — ordered edges `from_state_id → to_state_id` with `candidate_id`, `acceptance_kind`, `validation_passed`, `contains_target_product` and the proposal step's `provenance` summary; a `backtrack` truncates the path and the dropped candidates/states become `abandoned`;
- `candidate_sets` — one per `mechanism_candidates_proposed`, anchored on the state it was proposed from, each candidate carrying `status`, `failed_checks`, `validation_attempts`, the latest `reaction_focus` and `bond_electron_view`;
- `branch_points`, `backtracks`, `failed_paths`, `abandoned_candidate_ids`, `unvalidated_step_count`, `completed`, `active_step`, `provenance` (§14.4 inventory), `event_count`, `last_seq`.

Index conventions it relies on: `mechanism_step_accepted.step_index` and `mechanism_candidates_proposed.step_index` are the 1-based step number (`_apply_candidate` increments before emitting); `branch_point_created.step_index` and `backtrack.reverted_to_step` are the 0-based index of the step being decided. Legacy events without candidate ids get `legacy-<n>` / `soft-<n>` ids so old runs still project.

ChemIllusion exposes its own authenticated facade, e.g.:

```text
POST /api/mechanism-observatory/runs
GET  /api/mechanism-observatory/runs/{id}
GET  /api/mechanism-observatory/runs/{id}/events
```

The browser should not need a permanent credential for the Railway service.

---

# 23. ChemIllusion backend integration

> **Rev 3:** `mechanism_runtime_client.py` becomes an in-process adapter (`mechanism_runtime_service.py`: enqueue `async_jobs` row, worker handler calls the embedded runtime, SSE/observatory read from the Postgres store). No server-to-server token.

Recommended new files:

```text
backend/app/api/mechanism_observatory.py
backend/app/services/mechanism_runtime_client.py
backend/app/schemas/mechanism_observatory.py
```

Responsibilities:

## `mechanism_runtime_client.py`

- server-to-server authentication;
- create/start/stop/get run;
- SSE proxy/stream parsing;
- timeouts;
- version checks;
- error normalization.

## `mechanism_observatory.py`

- authenticate ChemIllusion user;
- entitlement/quota checks;
- create product run record;
- proxy runtime request;
- expose safe SSE;
- enforce ownership;
- record final usage/cost;
- hide runtime secrets.

## `mechanism_observatory.py` schema

- frontend-safe event types;
- no prompts;
- no hidden reasoning;
- no API keys;
- no benchmark answers;
- no research-only metadata.

---

# 24. Storage

> **Rev 3:** the store is ChemIllusion's existing Postgres, via a `RunStateStore` adapter shipped in the Wiggum release (§2.3.2 item 3).

The Wiggum local runtime currently uses SQLite through the `RunStateStore` abstraction.

Do not depend on a local SQLite file as the durable production event store on Railway.

The existing storage interface is the right seam for a production adapter.

Production requirements:
- durable run/event history for replay;
- ordered event sequence numbers;
- reconnect after process restart;
- ownership mapping;
- expiration/retention policy;
- no reliance on ephemeral local filesystem state.

An implementation may use:
- a dedicated Railway Postgres store; or
- the existing Supabase/Postgres infrastructure,

but the store choice must preserve the `RunStateStore` abstraction.

ChemIllusion should separately retain user-facing run metadata even if the runtime owns detailed event history.

---

# 25. Security and privacy

## 25.1 Service access

Mechanism Runtime is private or requires server-to-server authentication.

Do not expose an unrestricted public “use our model keys” endpoint.

## 25.2 Browser events

Allowed:
- chemical structures/states;
- electron actions;
- BE data;
- mapping data;
- model name/provenance;
- model probability;
- validator diagnostics;
- latency/usage if product policy allows.

Disallowed:
- provider API keys;
- system prompts;
- hidden chain-of-thought;
- raw internal reasoning traces;
- holdout benchmark answers;
- training data;
- mutation/evolution internals.

## 25.3 Model transparency

Showing users the actual model is a feature requirement.

It is not permission to expose provider secrets or raw traces.

---

# 26. Billing and usage

ChemIllusion already has AI-action/usage infrastructure. The Runtime should return actual call records/usage so ChemIllusion can meter the user-facing feature.

Do **not** assume Jev is free because `typesafe/jev-latest` currently has placeholder `$0` values in `model_pricing.json`.

The commit itself states that pricing is placeholder and adapter verification is pending.

Billing rules must use confirmed provider pricing before production charging decisions are based on Jev usage.

---

# 27. Detailed implementation changes — Professor Wiggum

## M0: model provenance

### `mechanistic_agent/core/coordinator.py`

1. Introduce a call-level provenance recorder.
2. Emit `inference_call_started/completed/failed`.
3. Include provenance summary in `step_output`.
4. Include planned source/model in `step_started` only where it is genuinely known.
5. Do not inherit an LLM model for deterministic `StepResult`s.
6. Add stable `candidate_id`.
7. Correlate candidate events, call events, validation events, accepted events.
8. Emit `mechanism_candidates_proposed` after `_propose_for_topology` and add `acceptance_kind` in `_apply_candidate` (§16.6).
9. Resolve `resolved_model` from `output.model_used` before the configured model (§3.7.2).

### `mechanistic_agent/core/subagents.py` / `mechanistic_agent/config.py`

- Stop treating `LLM_STEP_KEYS` membership as "has a model": `functional_groups` and `mechanism_synthesis` are deterministic (§3.7.1).
- Follow-up (not M0): pass `output.model_used` into `_extract_step_cost` so fallback cost is attributed to the model that ran.

### `mechanistic_agent/core/types.py`

Add/centralize typed provenance structures, or add a dedicated:

```text
mechanistic_agent/core/provenance.py
```

Do not make the event schema depend on an untyped arbitrary metadata dictionary.

### `mechanistic_agent/core/db.py`

1. Extend call accounting to recognize `jev` when live.
2. Persist inference-call records/events.
3. Aggregate by:
   - engine;
   - model;
   - role;
   - step.
4. Keep `llm_calls` compatibility fields.
5. Add:
   - `jev_calls`;
   - `inference_calls`;
   - or a general `by_engine` summary.

### `mechanistic_agent/api/app.py`

Expose the provenance in:
- snapshot;
- SSE;
- Observatory aggregate endpoint.

### `mechanistic_agent/ui/app.js`

Immediately:
- subscribe to `mechanism_step_accepted`;
- subscribe to inference-call events;
- add model/engine badges to current local debug summaries.

This local Wiggum UI does not need the final ChemIllusion visual design to validate the event contract.

## M1: Observatory chemistry projections

New recommended modules:

```text
mechanistic_agent/core/reaction_focus.py
mechanistic_agent/core/bond_electron.py
mechanistic_agent/core/observatory.py
```

Responsibilities:
- canonical focus mask;
- BE/ΔBE calculation;
- product-facing candidate/event projection;
- no UI concerns.

## M2: replay completeness

Enrich candidate/branch/failure events so a run can be reconstructed from storage after restart.

---

# 28. Detailed implementation changes — ChemIllusion

## Keep unchanged as canonical chemistry core

```text
backend/app/schemas/mechanism_explorer.py
backend/app/services/mechanism_transaction_service.py
backend/app/services/mechanism_validation_service.py
backend/app/services/mechanism_display_service.py
```

Only make compatible extensions when needed for a deterministic chemistry concept, not AI provenance.

## Add

```text
backend/app/schemas/mechanism_observatory.py
backend/app/services/mechanism_runtime_client.py
backend/app/api/mechanism_observatory.py

frontend/src/types/mechanismObservatory.ts
frontend/src/services/mechanismObservatoryApi.ts
frontend/src/features/mechanism-observatory/*
```

## Refactor reusable display elements out of

```text
frontend/src/components/activity-renderers/MechanismExplorerRenderer.tsx
```

without changing released student behavior.

## Adapt, then eventually deprecate the duplicated predictor

```text
backend/app/services/mechanism_prediction_service.py
```

Use a runtime-backed compatibility mode rather than extending its separate full-mechanism loop.

---

# 29. Jev integration expectations

The Observatory SHALL be ready for these planned Jev roles even before they are live on Wiggum main:

1. reaction conditions decision;
2. reaction-type Choice;
3. bounded atom-mapping candidate Choice;
4. candidate ranking/selection;
5. future narrow binary/score decisions.

A Jev decision result must preserve:
- decision type (`choice`, `score`, `noul`);
- full or safely filtered probability distribution when useful;
- selected option;
- model/revision;
- calibration status;
- candidate-set/question identity.

A Jev call does not become the “step model” merely because it participates in the step.

Example:

```text
Step 4 chemical proposal:
  generator = Claude Opus 5.5

Candidate selection:
  decision model = Jev 1.13

Chemistry execution:
  mapped-state executor

Validation:
  deterministic/RDKit
```

This distinction is a core product requirement.

---

# 30. Accessibility

Confidence and state cannot be encoded by color alone.

Every edge/status needs redundant encoding:

```text
P=0.81
✓ accepted
× rejected
◉ exploring
○ queued
```

Model identity is text.

BE matrix changed cells need:
- numeric values;
- accessible row/column labels;
- keyboard focus;
- textual changed-entry summary.

Atom lineage needs a nonvisual equivalent:

```text
Atom a7: O
Step 2 → 3:
  formal charge 0 → -1
  bond a4-a7 order 2 → 1
  lone pairs 2 → 3
```

Reuse the accessibility philosophy already present in Mechanism Explorer rather than making the Observatory canvas-only.

---

# 31. Performance

## 31.1 Avoid full-molecule redraws when not necessary

The focus system should make the UI perceptually lightweight even when the canonical state contains a large molecule.

## 31.2 BE matrix

Do not transmit/render the entire matrix by default for large systems if it is expensive.

The runtime can provide:
- focused matrix;
- full matrix on request or in compressed sparse form.

## 31.3 SSE

SSE remains the primary live transport.

Snapshot polling is:
- recovery;
- reconciliation;
- fallback.

It should not be the mechanism by which the React view learns the active model.

## 31.4 Render caching

ChemIllusion `mechanism_display_service.py` already caches deterministic renders. Preserve/reuse this behavior.

---

# 32. Observability and debugging

Every product run should record:

```text
runtime_version
runtime_git_sha
harness_version/hash
prompt_bundle_hash
validator version
BE convention version
ReactionFocus version
requested model(s)
resolved model(s)
reasoning levels
Jev revision
provider
inference call IDs
candidate IDs
event schema version
```

An admin/debug mode may additionally show:
- tokens;
- cost;
- latency;
- retry count;
- provider fallback.

Normal users need:
- model/engine;
- probability/confidence when meaningful;
- validation status.

---

# 33. Release phases

## Phase M0 — Provenance before prettier UI

**Professor Wiggum**

- inference-call event schema;
- actual model/engine on live events;
- deterministic source normalization;
- stable candidate IDs;
- `mechanism_step_accepted` current UI subscription;
- replay tests;
- no new product UI required.

**Exit criterion:** a live Wiggum run can truthfully answer “what model/tool is running right now?” and a replay produces the same answer.

---

## Phase M1 — ReactionFocus + chemistry projection

- persistent identity adapter;
- ReactionFocus v1;
- BE/ΔBE v1;
- candidate validation event;
- mapping lineage payload;
- focus/replay tests.

**Exit criterion:** every accepted or rejected candidate has a deterministic focus mask and chemistry-delta payload.

**Status (2026-09-23):** ReactionFocus v1, BE/ΔBE v1, `candidate_validation_result`, `atom_identity` on accepted steps and the `atom_lineage` projection are delivered. The persistent-id ↔ `a<map>` bridge for candidates requires the mapped loop (`loop_state_mapping: mapped`) and is the remaining M1 item.

---

## Phase M2 — Wiggum local Observatory prototype

Replace or supplement the dominant Mermaid panel with a simple chemical search tree.

Purpose:
- validate event contract;
- validate branch behavior;
- validate focus/matrix/mapping display;
- not production styling.

**Exit criterion:** easy/medium/hard test runs can be watched live without inspecting terminal JSON.

**Status (2026-09-23, prototype delivered):** the Wiggum UI now shows a **Mechanism search** panel above the harness diagram, rendered from `/observatory` on every snapshot refresh: reactant → accepted intermediates → product spine, each step's candidates with glyph + text status (◉ accepted, ✓ validated, ○ alternative, × rejected with failed checks, ⊘ abandoned, ⚠ accepted without validation), focus core atoms, `ΣΔBE` with conservation mark, and a neutral proposal-provenance chip per accepted edge. The harness diagram stays available behind a "Harness view" toggle (§8.4). `?run=<id>` re-attaches the page to an existing run (snapshot + projection + live stream when running), which is the §17 reload/replay requirement. Structure thumbnails in the spine and per-candidate `BE(t) | ΔBE | BE(t+1)` tables landed next (§10.2). Still open for M2: matrix↔structure hover linking (§10.3) and the atom-lineage panel (§11.4).

---

## Phase M3 — Runtime release embedded in the ChemIllusion API (rev 3; formerly "Runtime-only Railway build")

Rev 3 scope: tag `runtime-vX.Y.Z`; embeddable entry point that runs one mechanism job against a supplied `RunStateStore`; Postgres `RunStateStore` adapter; asset manifest so the checkout needs no general `training_data/` access; `Dockerfile.api` checkout + `dimorphite-dl`; `async_jobs` handler; `openai` SDK compatibility check. No new Railway service.

Original list (still the capability checklist):

- runtime-specific app/build;
- runtime asset manifest;
- no research endpoints/data;
- durable RunStateStore;
- server-to-server auth;
- health/readiness;
- immutable version manifest.

**Exit criterion:** ChemIllusion backend can create a run, stream it, recover it after reconnect, and retrieve final accepted path without the Wiggum research app.

**Status (2026-09-24):** runtime-only app, bearer auth, version manifest, `Dockerfile.runtime` and `.dockerignore` delivered (§21.1.1); embedding API with event mirroring delivered (§2.3.2.1); ChemIllusion-pinned dependency compatibility verified (§2.3.4). Open: tag `runtime-v0.1.0`, the ChemIllusion side (Dockerfile checkout, `async_jobs` handler, `mechanism_run_events` table, page and dock tile).

---

## Phase M4 — ChemIllusion Observatory

- new backend proxy/client;
- new Observatory schemas;
- React search-tree view;
- reused Mechanism Explorer state canvas;
- model badges;
- validator inspector;
- atom lineage;
- BE matrix;
- responsive layout;
- accessibility.

**Exit criterion:** authenticated ChemIllusion users can run and inspect a live mechanism prediction.

**Rev 3 surface:** `/mechanism-predictor` + `/mechanism-predictor/runs/:runId`, Generator dock tile, mobile Answer layout, Beta · In development labeling with the repo link, `EngineBadge` Jev/LLM/deterministic chips, three disclosure levels (§8.5). Beta audience only.

---

## Phase M5 — Jev decisions + probability display

Only after actual Jev runtime calls are implemented and evaluated.

- adapter;
- pinned/reported Jev revision;
- reaction-type and/or candidate-choice shadow mode;
- probability payloads;
- calibration measurement;
- UI `not calibrated` state;
- fallback chain display.

**Exit criterion:** Jev decisions are traceable and probabilities are never presented without their source/type/calibration status.

---

## Phase M6 — Migration of existing ChemIllusion full-mechanism predictor

- compatibility adapter from Runtime result → `FullMechanismResult`;
- feature flag;
- parity tests for API/MCP/ChemEd callers;
- progressive migration;
- eventual deprecation of duplicate loop.

---

# 34. Acceptance criteria

## Provenance

- [ ] Every actual model/decision call has a unique `call_id`.
- [ ] Every live call event includes engine and requested model where applicable.
- [ ] Resolved model/revision is recorded when provider/runtime can know it.
- [ ] Deterministic steps expose no fake LLM model.
- [ ] Jev and full-LLM calls are independently visible.
- [ ] A Jev→LLM fallback shows both calls in order.
- [ ] Human-supplied steps display as human.
- [ ] Page reload reproduces the same call chain.
- [ ] `mechanism_step_accepted` is consumed by the live UI.

## Chemical focus

- [ ] Bond changes are in the core focus.
- [ ] Charge/lone-pair changes are in the core focus.
- [ ] Electron-push participants are in the core focus.
- [ ] Default context is bounded and deterministic.
- [ ] Full molecule can always be restored.
- [ ] Same focus IDs drive molecule, BE, mapping, and diagnostics.

## Bond-electron matrix

- [ ] BE convention is versioned and documented.
- [ ] `BE(t)`, `ΔBE`, `BE(t+1)` share identical atom order.
- [ ] Default matrix is reaction-focused.
- [ ] Full matrix is available.
- [ ] Matrix hover is linked to molecular atom/bond highlights.
- [ ] Electron conservation is shown only when valid under the chosen convention.

## Atom identity

- [ ] Stable identity survives all accepted steps.
- [ ] Branch candidates have stable identities within their own state.
- [ ] Backtracking does not reuse identity incorrectly.
- [ ] Mapping uncertainty is distinct from measured mapping validation.
- [ ] Focused atom-lineage view is keyboard accessible.

## Probability

- [ ] No rank is displayed as probability.
- [ ] No deterministic pass/fail is displayed as probability.
- [ ] Every numeric confidence has a `kind`.
- [ ] Every model-derived probability identifies source model/engine.
- [ ] Uncalibrated values say so.
- [ ] Color is not the only confidence cue.
- [ ] Red/rejection styling is not reused as the low-probability color.

## Deployment

- [ ] Runtime image contains no holdout/eval corpus.
- [ ] Runtime image exposes no mutation/leaderboard/training routes.
- [ ] Runtime is immutable between deployments.
- [ ] Runtime release manifest is queryable/recorded.
- [ ] Railway deployment does not rely on ephemeral SQLite for durable replay.
- [ ] ChemIllusion owns user auth/entitlement.
- [ ] Provider keys never reach browser.
- [ ] Existing Mechanism Explorer deterministic endpoints remain AI-free.

---

# 35. Test plan

## Professor Wiggum fast tests

Add:

```text
tests/fast/test_inference_provenance.py
tests/fast/test_observatory_events.py
tests/fast/test_reaction_focus.py
tests/fast/test_bond_electron_view.py
tests/fast/test_observatory_replay.py
```

Cases:
1. single LLM proposal;
2. deterministic validator;
3. human submitted step;
4. retry with same model;
5. provider/model fallback;
6. Jev call when adapter lands;
7. Jev → LLM escalation;
8. 3-candidate branch;
9. rejected candidate;
10. backtrack to alternative;
11. process reload/replay.

Add a regression assertion that `mechanism_synthesis` does not show an LLM model when it is deterministic. `tests/fast/test_llm_call_counter.py` already asserts `source == "deterministic"` for that step; extend it rather than duplicating.

Additional M0 cases (rev 2):

12. LLM step whose `output.model_used` differs from the configured model records the fallback model as `resolved_model`.
13. Jev→LLM fallback within `reaction_type_mapping` yields one failed `jev` call and one completed `llm` call with `fallback_from_call_id` set.
14. Validator rows (`bond_electron_validation` etc.) expose `engine: deterministic` and `model: null`.
15. Soft-advanced step emits `mechanism_step_accepted` with `acceptance_kind: soft_advance`.
16. Two proposal rounds with the same ranks produce distinct `candidate_id`s; the id survives `BranchCandidate.to_persisted_dict()` round trip.

Run with `python -m pytest tests/fast -q` (the `make test` target referenced in `AGENTS.md` has no Makefile in the checkout). Baseline on `c9248b8`: 772 passed.

## ChemIllusion backend tests

Add:
- runtime client contract tests;
- auth/ownership tests;
- SSE safe-field tests;
- Observatory schema mirror tests;
- runtime-version mismatch tests;
- compatibility adapter tests for existing `FullMechanismResult`.

## ChemIllusion frontend tests

Test:
- live model badge changes;
- mixed-model call chain;
- no-model deterministic badges;
- confidence labels;
- branch/rejection;
- focus synchronization;
- BE/molecule hover synchronization;
- reload/replay;
- keyboard navigation;
- reduced-motion behavior.

---

# 36. File-by-file implementation map

## Professor Wiggum

| File | Change |
|---|---|
| `mechanistic_agent/core/coordinator.py` | call-level provenance events, candidate IDs, enriched mechanism events |
| `mechanistic_agent/core/db.py` | persist/aggregate inference calls; activate Jev engine counting when live |
| `mechanistic_agent/core/types.py` | provenance/event typing if kept with core types |
| `mechanistic_agent/core/provenance.py` | **new**, preferred home for provenance model/helpers |
| `mechanistic_agent/core/reaction_focus.py` | **new**, deterministic focus |
| `mechanistic_agent/core/bond_electron.py` | **new**, BE/ΔBE projection |
| `mechanistic_agent/core/observatory.py` | **new**, aggregate/replay projection |
| `mechanistic_agent/api/app.py` | enriched snapshot/SSE and Observatory endpoint |
| `mechanistic_agent/ui/app.js` | subscribe accepted/inference events; local model badges |
| `mechanistic_agent/core/subagents.py` | (follow-up) attribute fallback cost to `output.model_used` |
| `mechanistic_agent/config.py` | `LLM_STEP_KEYS` no longer implies model provenance for deterministic steps |
| `mechanistic_agent/api/app.py` (verified-step route) | use the shared provenance helper for `source="human"` rows |
| `Dockerfile.runtime` | **new**, stripped production build |
| `mechanistic_agent/api/runtime_app.py` | **new**, narrow product API |
| `runtime_assets/*` | **new**, versioned production inference bundle |

## ChemIllusion

| File | Change |
|---|---|
| `backend/app/schemas/mechanism_observatory.py` | **new**, AI/runtime metadata separate from canonical chemistry |
| `backend/app/services/mechanism_runtime_client.py` | **new**, private service client/SSE |
| `backend/app/api/mechanism_observatory.py` | **new**, auth/entitlement/proxy |
| `backend/app/services/mechanism_prediction_service.py` | feature-gated compatibility adapter; do not expand duplicate loop |
| `frontend/src/types/mechanismObservatory.ts` | **new**, typed mirror |
| `frontend/src/services/mechanismObservatoryApi.ts` | **new**, run + SSE client |
| `frontend/src/features/mechanism-observatory/*` | **new**, product UI |
| `frontend/src/components/activity-renderers/MechanismExplorerRenderer.tsx` | extract reusable state canvas/overlays without changing student behavior |
| `backend/app/services/model_pricing.json` | replace Jev placeholder pricing when confirmed; catalog entry alone remains insufficient |

---

# 37. Decisions locked by this PRD

1. **Chemistry path, not harness flow, is the primary visualization.**
2. **One deterministic ReactionFocus controls all “show me what changed” views.**
3. **BE matrices are explicit and reaction-focused by default.**
4. **Persistent atom identity is a first-class visualization dimension.**
5. **Model/engine provenance is visible to the user for every model-backed operation.**
6. **Deterministic stages never inherit a misleading model label.**
7. **A logical step can have multiple provenance records.**
8. **Jev is shown according to the decision it actually made, not as the generator unless it generated.**
9. **Probability/confidence and deterministic validity remain separate.**
10. **No probability is fabricated from rank.**
11. **ChemIllusion's AI-free `MechanismState` schema remains AI-free.**
12. **Observatory metadata lives in a parallel schema.**
13. **Existing ChemIllusion Mechanism Explorer rendering/state infrastructure is reused.**
14. **The existing simplified ChemIllusion full-mechanism loop is not expanded into a second Wiggum implementation.**
15. **Professor Wiggum remains the lab; an evaluated, tagged runtime release is promoted into the existing ChemIllusion API — no new Railway service (rev 3).**
16. **The production runtime excludes training/eval/evolution machinery.**
17. **SSE is the primary live transport; snapshots provide recovery/replay.**
18. **The tool has its own URL (`/mechanism-predictor`, `/mechanism-predictor/runs/:runId`); every entry point, including the Generator right dock, lands there (rev 3).**
19. **While in training it is labeled "Beta · In development", links to this repository, and is gated to a beta audience (rev 3).**
20. **Every step labels Jev, LLM, deterministic and human work explicitly, at every disclosure level (rev 3).**
21. **One payload, three views: Answer, Steps, Details (rev 3).**
22. **Harness evolution happens only in this repository; ChemIllusion consumes tags (rev 3).**

---

# 38. Open questions

These should not block M0.

### Q1. Which Jev endpoint/adapter becomes canonical?
**Partially answered (rev 2).** Wiggum's live adapter is the OpenRouter Decisions API (`mechanistic_agent/decisions/jev.py`, `PROVIDER = "openrouter"`, catalog id `typesafe/jev-1.13`). ChemIllusion's `typesafe/jev-latest` catalog stub points at chat completions and is not equivalent. The runtime service should ship the Wiggum adapter; ChemIllusion should not add a second one.

### Q2. What exact Jev revision is available through the chosen provider?
**Answered (rev 2).** The Decisions API returns the revision (`typesafe/jev-1.13-20260917` on 2026-09-23). Store it as `resolved_model`; the `resolved_model: null` case applies only when the response omits it.

### Q3. Which service owns durable detailed run events?
Recommendation: Mechanism Runtime owns the detailed event log; ChemIllusion owns user/run ownership and product metadata.

### Q4. Should ChemIllusion re-run deterministic validation on runtime candidate states?
Recommended for critical compatibility boundaries where cheap, especially when converting to canonical `MechanismState`; avoid silently having two divergent validators claim authority.

### Q5. How much candidate history should a normal product user see?
Default: accepted path + recent/meaningful rejected alternatives. Expert mode may show all candidates.

### Q6. How should the BE convention handle aromatic systems, radicals, and implicit H?
This must be locked and tested before `bond_electron_view.v1` is declared stable.

---

# 39. First implementation slice

The first PR should be intentionally narrow:

## Wiggum M0 PR

1. Add typed `InferenceProvenance`.
2. Add `inference_call_started/completed/failed`.
3. Include resolved source/model/reasoning in `step_output` summary.
4. Prevent deterministic steps from inheriting misleading LLM provenance.
5. Add stable candidate IDs.
6. Add `mechanism_step_accepted` to current UI subscriptions.
7. Add provenance/replay tests.
8. Add a small current-engine/model badge in the Wiggum local UI.
9. Emit `mechanism_candidates_proposed` and `acceptance_kind` (§16.1, §16.6).
10. Use the same provenance helper in the verified-step API route.

Order of work inside the PR (rev 2): provenance helper + tests → `_record_step` / `_mark_step_started` → candidate IDs and event enrichment → API/UI subscription and badge. M0b (live `llm.py` hook) is a second PR.

Do **not** build the entire ChemIllusion Observatory in this PR.

The second PR can add ReactionFocus/BE projection.

This sequencing creates a trustworthy event stream first. Once model identity, candidate identity, and event replay are correct, the richer visualization becomes primarily a rendering problem rather than another round of instrumentation repair.

---

# 40. Success condition

A chemist watching a live prediction should be able to understand the run without reading terminal JSON.

At any moment they can answer:

> **What chemical transformation is being explored?**

> **What atoms/electrons are changing?**

> **Which pathway alternatives were considered?**

> **How strong is the model's preference, if a real probability exists?**

> **Which model or deterministic engine made each part of the decision?**

> **Why did the system accept or reject this candidate?**

And the answer remains reproducible when the same run is reopened later.

That is the standard the Mechanism Observatory should meet.
