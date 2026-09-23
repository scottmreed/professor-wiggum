# PRD: Jev-First Decision Layer and Persistent Atom Identity for Mechanistic

**Status:** Proposed (v2, revised 2026-09-23 against the current codebase)  
**Project:** MechanisticWiggum / Mechanistic Agent  
**Primary scope:** Mechanistic prediction harness  
**Target implementation area:** `mechanistic_agent/core/`, `mechanistic_agent/tools.py`, `skills/mechanistic/`, `harness_versions/`, eval/trace infrastructure  
**Design principle:** Use Jev to avoid full LLM calls. Reserve full LLMs for genuinely generative chemistry, ambiguity escalation, and opportunistic review of upstream Jev decisions.

---

## 0. What changed in v2

v1 was written from the harness diagram and the SOUL, not from the code. An audit of the runtime on 2026-09-23 found that several of its premises do not hold. v2 keeps the direction and corrects the foundations:

| v1 assumption | Reality (with location) | Consequence for the PRD |
| --- | --- | --- |
| pH recommendation is an LLM call worth replacing with a Jev `Score` | `recommend_ph` (`tools.py:4542`) is deterministic. Its Dimorphite branch is dead code (installed `dimorphite_dl` 2.0.2 no longer exports `DimorphiteDL`, import fails silently at `tools.py:28`). Its output does **not** feed the conditions LLM despite the harness description; only `select_reaction_type` sees it (`coordinator.py:870, 1521`). | §7.1 is no longer a call-reduction target. pH becomes one question inside the conditions Jev request; the standalone module gets a bug fix, not a model. |
| A "deterministic mechanism transform" applies graph edits, and the spike asks whether it preserves atom identity | No transform exists. `predict_mechanistic_step` (`tools.py:6571`) accepts the **LLM-written** `resulting_state` and parses `reaction_smirks` only for metadata. `RunReactants` / `ReactionFromSmarts` are unused. Nothing checks that the SMIRKS left side matches `current_state`. | §9 is reframed: the spike is "build a mapped-state SMIRKS executor", not "test the existing one". |
| Atom identity is re-inferred after each step | True, and worse: maps are stripped from every LLM input (`tool_executor.py:37-42, 63-178`, `db.py:20-49`), so the proposal LLM re-invents map numbers per step (`[C:17]` in step 1 becomes `[C:11]` in step 2 of run `06cb9708…`). | Persistent identity requires stopping the ingress strip on the loop state, or executing SMIRKS on a mapped state. |
| Benchmark mapping ground truth may not exist | Every eval record is fully atom-mapped **with explicit H** (`training_data/eval_set.json`, `flower_mechanisms_100.json`, practice set, holdout). No metric compares a predicted mapping to it. | Mapping recall against ground truth is definable today and becomes the first deliverable. |
| Step mapping is bookkeeping | Its self-reported `confidence` is 20% of per-step validity (`scoring.py:345`), the whole `step_atom_mapping` leaderboard subagent score (`scoring.py:557-574`), and ~7% of the final eval score. Absent mapping scores 0.5. | Replacing it with a deterministic method changes eval scores unless the metric is redefined first. |
| Reaction type needs pre-filtering to fit a bounded choice | The taxonomy has 86 templates, all listed on every call (`tools.py:4198-4235`). `no_match`, a 0.65 confidence threshold and a 0.10 margin threshold already exist (`coordinator.py:886-1006`, RunConfig fields `types.py:529-530`). Jev `Choice` accepts up to 255 options. | Pre-filtering is optional, not required. Existing thresholds move to the harness. |
| An LLM currently ranks candidates | Only the generator's self-assigned `rank` is used (`coordinator.py:4952-4955`). `evaluate_run_judge` is a prompt asset that nothing invokes. Multi-agent topologies already have a consensus re-ranker (`coordinator.py:3922-3972`). | §14 compares Jev against generator rank and the consensus merge; the "LLM judge ranker" is new code. |
| Eval tiers easy/medium/hard exist for acceptance tests | `training_data/eval_tiers.json` had easy=100, medium=0, hard=0, but `eval --tier` resolves tiers through `development_leaderboard_policy.json`, which sourced medium/hard from `baseline_tiers_clawdiator.json` (20 and 60 cases). The empty lists were masked, and a requested tier resolving to zero cases silently rerouted instead of failing. (Corrected 2026-09-23; PR #29 synchronizes the two files, fails loudly on empty tiers, and documents the resolution in `docs/eval_tiers.md`.) | Acceptance criteria can cite medium/hard now; §19 Phase 0 keeps the loud-failure guard and the tier documentation. |
| Harness config can carry a `decision_policy` block | `HarnessConfig.from_dict` (`types.py:411-443`) reads only known keys and `save()` erases unknown ones. The evolver whitelists editable keys and coerces values to bool/int (`llm_mutator.py:34-41, 386-393`). | §17 specifies the dataclass and evolver changes explicitly. |
| Jev is a hypothetical decision model | Jev is TypeSafe AI's System One model, released 2026-09-15. Typed `Choice` (≤255 options), `Score` (2–10 levels), `Noul` (yes/no probability, **no separate confidence**). State + longest question ≤ 32k tokens. Reachable directly (`api.typesafe.ai/v1/systemone`) and via OpenRouter's Decisions API (`POST openrouter.ai/api/alpha/decisions`, model `~typesafe/jev-latest`). Pin `jev-1.13.0`. | §7 gains an API-constraints section and a calibration section drawn from published third-party evaluations. |

Measured baseline (local DB, 246 runs with ≥1 accepted step, 414 accepted steps, default harness):

| Quantity | Value |
| --- | --- |
| Pre-loop LLM calls per run | 4 (`initial_conditions`, `missing_reagents`, `atom_mapping`, `reaction_type_mapping`) |
| Proposal calls per accepted step | 1.38 |
| Step-mapping calls per accepted step | 0.88 |
| All in-loop LLM calls per accepted step (incl. `candidate_rescue` 214, `overall_balance_reconciliation` 75) | 2.95 |

For a 3-step run that is roughly 13 LLM calls, of which this PRD can remove up to 3 pre-loop calls and ~2.6 step-mapping calls (about 40% of calls). The proposal call dominates tokens, so the honest framing is **latency, determinism and auditability first, cost second**. Token-weighted savings must be measured, not assumed (§18).

---

## 1. Summary

Mechanistic combines deterministic chemistry checks with four LLM-backed pre-loop analyses, one generative LLM step inside the mechanism loop, and one LLM-backed post-step mapping call. This PRD moves bounded judgment and bookkeeping away from full LLMs and into:

1. deterministic RDKit logic,
2. Jev `Choice`, `Score` and `Noul` decisions,
3. explicit, calibrated uncertainty thresholds with fallback behavior,
4. full LLM calls only when new chemical content must be generated or bounded methods remain ambiguous.

Opportunities, in the order the evidence supports pursuing them:

1. **Reaction-type classification → Jev `Choice`** over the existing 86-template taxonomy plus `no_match`. Lowest risk: the output contract, thresholds and `no_match` already exist.
2. **Reaction-condition assessment → Jev `Choice` + `Score`** (environment, pH band, compatibility) in one request, **while keeping a bounded path for acid/base candidate species**, which Jev must not generate.
3. **Mapped-state SMIRKS execution** so atom identity propagates through steps and `resulting_state` is derived rather than trusted. This is the technical spike and the prerequisite for retiring step mapping.
4. **Deterministic global mapping candidates + Jev `Choice`**, gated on a ground-truth recall metric that does not exist yet.
5. **Missing-reagent `Noul` gate** in front of the existing (already partially gated) LLM call.
6. **Candidate-ranking shadow experiment**: Jev rank vs generator rank vs consensus merge vs a new LLM judge.
7. **Opportunistic upstream review** by the proposal LLM via a compact `context_review` field.

Everything lands as harness variants and advances through the normal evidence-gated process (`docs/change_evidence_policy.md`).

---

## 2. Why this change

Target architecture:

> **deterministic chemistry → bounded Jev decisions → full LLM only for open-ended chemistry**

Jev must not become an extra judge bolted onto every LLM output. Its value is replacing full-model calls and making the remaining decisions typed, logged and calibratable.

The harness already has the right seams: explicit pre-loop modules with `enabled` flags and `config_gate`s, a single generative loop module, deterministic post-step validators, branch points and backtracking, and eval-backed harness evolution. Candidate rank is operationally important (`validated.sort(key=rank)`, top applied, rest become branch alternatives), which gives a clean ranker-comparison surface.

What the harness does **not** yet have, and this PRD adds: a mapped-state executor, any deterministic mapping, any ground-truth mapping metric, an LLM call counter, and a place in the harness schema for decision policy.

---

## 3. Goals

### 3.1 Product goals

- Reduce unnecessary full-LLM calls; measure the reduction in calls **and** tokens.
- Reduce latency without reducing mechanism quality.
- Make uncertainty explicit, typed and calibrated per question.
- Preserve deterministic chemistry validators as final arbiters.
- Make atom mapping reproducible and checkable against benchmark mappings.
- Turn Jev-vs-LLM decisions into measurable harness experiments that the evolver can operate on.

### 3.2 Technical goals

- A Jev adapter (`Choice`, `Score`, `Noul`, multi-question requests, distributions, model/version metadata, timeout and failure fallback) registered in the model catalog with its own provider route.
- A mapped-state SMIRKS executor and persistent atom identity through steps.
- Deterministic mapping-candidate generation with symmetry collapse.
- A mapping-recall metric against benchmark mappings.
- Decision policy and thresholds in the harness schema, mutable by the evolver.
- Jev outputs and distributions in traces; an explicit LLM-call counter.
- Shadow-mode ranker comparison.
- `context_review` fields without an additional LLM call.

---

## 4. Non-goals

This PRD does **not** propose:

- replacing deterministic chemistry validators with Jev,
- using Jev to generate intermediate SMILES, reagents, electron pushes or SMIRKS,
- removing full-LLM mechanism proposal,
- treating Jev probabilities as proof of chemical correctness,
- changing production candidate ordering before shadow evidence exists,
- using raw RDKit atom indices as persistent identity,
- trusting a Jev threshold tuned for one question type on another (thresholds do not transfer; see §7.0).

---

## 5. Current architecture (verified 2026-09-23)

### Pre-loop (`harness_versions/default/harness.json`, `pre_loop_modules`)

| # | module id | kind | function | notes |
| --- | --- | --- | --- | --- |
| 1 | `balance_analysis` | deterministic | `analyse_balance` | RDKit formula counts; `_atom_counter` differs from the hydrogen-aware counter used by missing-reagents |
| 2 | `functional_groups` | deterministic | `fingerprint_functional_groups` | ~43 SMARTS. Injected into LLM prompts **only** when env var `MECHANISTIC_FUNCTIONAL_GROUPS_ENABLED` is set (`tools.py:528-532`); the harness flag does not control it, and `coordinator._build_state` defaults `functional_groups_enabled` to False (`coordinator.py:196`) |
| 3 | `ph_recommendation` | deterministic | `recommend_ph` | user pH → Dimorphite (dead) → SMARTS heuristic returning a range; consumed only by `select_reaction_type` |
| 4 | `initial_conditions` | **LLM** | `assess_initial_conditions` | `ASSESS_CONDITIONS_TOOL`: `environment` ∈ {acidic, basic, neutral}, `representative_ph`, `ph_range`, `acid_candidates`/`base_candidates` (≤3 each, name + SMILES), `warnings`, `justification`. Feeds `predict_missing_reagents` as `conditions_guidance`, `select_reaction_type`, and the proposal prompt via an in-memory deque (`_INITIAL_CONDITION_HISTORY`, `tools.py:5085-5144`) that is lost on resume in a new process |
| 5 | `missing_reagents` | **LLM** | `predict_missing_reagents` | one deterministic gate already: returns `status:"balanced"` with no call when heavy-atom and H counts balance (`tools.py:3356-3364`); retry-loop usage not captured in `_llm_usage` |
| 6 | `atom_mapping` | **LLM** | `attempt_atom_mapping` | `ATOM_MAPPING_TOOL` returns `product_smiles#index → source atom_index` pairs plus self-reported `confidence`. Post-hoc check via external `rdkit-agent atom-map check` (`tools.py:1273`), which caps confidence at 0.3 on failure. Downstream use is nearly nil: the proposal prompt reads `confidence` from the wrong level (`coordinator.py:1146`, always `None`) and `run_intermediates` passes empty mapped-state arguments (`tool_executor.py:115-117`) |
| 7 | `reaction_type_mapping` | **LLM** | `select_reaction_type` | 86 templates, no pre-filter, `no_match` supported, `top_candidates` ≤3, confidence/margin thresholds 0.65/0.10 in RunConfig. **If `run_input.example_id` is in the catalog's 72 `example_mappings`, the LLM is skipped and the curated label is used at confidence 0.99** (`coordinator.py:1282, 1484-1546`). Any Jev-vs-LLM comparison on curated evals must disable this path |

### Mechanism loop

```text
LLM proposes candidates (mapped reaction_smirks, electron_pushes, UNMAPPED resulting_state)
→ predict_mechanistic_step: validates LLM-supplied resulting_state, parses SMIRKS for metadata only
→ deterministic validators (bond/electron via mech block, atom balance, state progress)
→ validated.sort(key=rank); top applied; rest → BranchPoint alternatives (persisted in `run_resume_state`, restored on resume)
→ post-step: reflection (deterministic), step_atom_mapping (LLM)
→ continue / backtrack / terminate
```

Corrections to v1's picture: there is no "deterministic mechanism synthesis/execution"; `MechanismAgent` is recorded as `source="llm"` although it makes no model call (`subagents.py:404`); the SMIRKS is never applied to the state; atom maps are stripped from every LLM input.

### Post-step mapping

`attempt_atom_mapping_for_step` (`tools.py:4092`) calls the same global mapping LLM on `current_state`/`resulting_state` with maps stripped. Consumers: (a) `state.latest_step_mapping` → advisory "atom-lineage context" text in the next proposal prompt (`tools.py:5513-5522`); (b) scoring, as above. No validator reads it.

---

## 6. Target architecture

```text
DETERMINISTIC   balance · functional groups (actually injected) · pH heuristic (fixed)
      ▼
JEV (one request) environment Choice · pH-band Score · compatibility Score · needs_missing_chemistry Noul
      ▼
DETERMINISTIC   acid/base candidate species from a bounded reagent table keyed on environment
                (LLM fallback only when the table has no entry and the Noul says chemistry is missing)
      ▼
DETERMINISTIC   global mapping candidates → rdkit-agent atom-map check as hard filter
      ├── 1 candidate → accept
      ├── 2..K → JEV Choice (+ none)
      └── 0 or > K_hard → FULL LLM mapping fallback
      ▼
JEV             reaction-type Choice over 86 templates + no_match
      ▼
OPTIONAL LLM    missing chemistry, only if Noul says so and balance gate did not already pass
═══════════════ MECHANISM LOOP ═══════════════
FULL LLM        candidate elementary steps + context_review of upstream decisions
      ▼
RANK            generator rank (production) · shadow: Jev Choice · consensus merge · LLM judge
      ▼
DETERMINISTIC   execute mapped SMIRKS on mapped current_state → derived resulting_state
                compare with LLM resulting_state (agreement is a new validator signal)
                validators · identity propagates via map numbers
      ▼
STEP MAPPING    identity preserved → none needed
                else local deterministic candidates → 1: accept · few: Jev · unresolved: LLM
      ▼
DETERMINISTIC   retry / branch / backtrack / completion
```

---

## 7. Jev integration surfaces

### 7.0 Jev API constraints and calibration facts (new)

Facts to design against (sources: TypeSafe docs, Pydantic AI provider docs, OpenRouter Decisions API harness, independent calibration write-ups; verify against the pinned version before implementation):

- Request = one `state` (text or JSON) + one or more typed `questions`; all questions are answered in one forward pass. Ask related questions in one request.
- `Choice`: ≤255 options; returns `choice`, `probabilities` over all options (sum to 1) and `confidence`.
- `Score`: 2–10 ordered rubric levels described in words; returns probability-weighted `score`, `probabilities` per level and `confidence`.
- `Noul`: returns probability that a proposition is true; **no separate confidence field**. v1's `confidence` on Noul results must be dropped.
- State + longest question ≤ 32k tokens (64k combined). Filter state in code first; irrelevant detail measurably degrades answers.
- Pin the model version (`jev-1.13.0`); thresholds do not transfer across versions or across question types.
- Every `Choice` must include an explicit `none`/`no_match` option.
- Published calibration on out-of-distribution data: ECE ≈ 0.107; `Choice`/`Score` overconfident, `Noul` underconfident; on unanswerable questions Jev stayed ~45% accurate while assigning ~0.74 probability. Treat confidence bands as hypotheses to be measured per question (§19 Phase D).
- A single broad question underperformed a frontier LLM (62.6% vs 81.3%); five narrow questions combined by a fitted logistic layer reached 95%. Prefer several narrow questions plus a small fitted combiner over one broad question.
- Jev treats state as data, not instructions, but is still susceptible to injected text in state. Our state is machine-built (SMILES, counts, group labels), never user prose. Keep it that way.
- Cost: input-only billing (~$0.042/M tokens), latency ~70–500 ms. Available via OpenRouter, which this repo already routes hosted models through (`llm.py:617-697`).

Chemistry competence is unmeasured. Jev has no published chemistry evaluation; discriminating atom mappings from SMILES text is a hard, unusual task. Every surface below therefore starts in shadow mode with a ground-truth metric.

### 7.1 pH: fix the deterministic module, fold the judgment into the conditions request

No LLM call is saved here; v1's framing was wrong. Do this instead:

1. Fix `recommend_ph`: either update the `dimorphite_dl` import to the 2.0 API (`protonate_smiles`) or delete the dead branch. Add a fast test that asserts which branch runs.
2. Wire its output into whatever replaces `assess_initial_conditions` as state, as the harness description already (falsely) claims.
3. Add a **pH-band `Score`** question to the conditions Jev request (§7.2), five levels: strongly acidic (0–3), mildly acidic (3–6), near neutral (6–8), mildly basic (8–11), strongly basic (11–14). Derive `representative_ph` from level midpoints only for legacy consumers (`propose_intermediates` reads `representative_ph`); the distribution is canonical.
4. If the user supplied a pH, it is authoritative; Jev only assesses compatibility.

Acceptance: downstream effects on the eval tiers (completion, reaction-type accuracy, retries), not agreement with a single numeric pH.

### 7.2 Reaction conditions → Jev `Choice` + `Score`, plus a bounded species path

One request, shared state, at least three questions:

- `environment` Choice: acidic, basic, neutral, mixed, unclear. (Current enum lacks `mixed`/`unclear`; downstream consumers must accept them.)
- `ph_band` Score (§7.1).
- `compatibility` Score, 4 levels: strongly incompatible, questionable, broadly plausible, strongly supportive.
- `needs_missing_chemistry` Noul (§12).

**The species problem.** Today's LLM call also returns `acid_candidates`/`base_candidates` (name + SMILES) that feed `predict_missing_reagents` and the proposal prompt. Jev cannot generate them. Options, to be decided by Phase E data:

- (a) a deterministic reagent table keyed on `environment` × detected functional groups (e.g., basic + ester → hydroxide/alkoxide), reviewed by a chemist; or
- (b) keep a full-LLM species call only when the Noul says missing chemistry is likely **and** the table has no entry.

Fallback policy: no full LLM call merely because compatibility confidence is moderate. Escalate only when downstream logic needs a missing chemical object, when ambiguity blocks reaction-type classification or proposal, or when a configured threshold is crossed and the harness permits escalation.

Also fix: the in-memory `_INITIAL_CONDITION_HISTORY` deque must be replaced by a persisted per-run field so resume works.

### 7.3 Reaction type → Jev `Choice`

Strongest bounded-decision target. 86 options + `no_match` fits well under the 255 cap, so pre-filtering is an optimization, not a requirement. Start with the full taxonomy; test pre-filtering (by functional groups, bond-change signature, reagent class) as a variant.

Input state (machine-built JSON): normalized SMILES, balance analysis, functional groups (actually injected, see §5 row 2), pH/conditions Jev outputs, missing-reagent status, mapping summary if available.

Output contract preserved for the coordinator: `selected_type_id`, `selected_label_exact`, `confidence`, `top_candidates` (populate from the distribution, lift the cap of 3 to a configurable N).

Thresholds: reuse the existing confidence/margin gates (`_guidance_mode_for_selection`) but move `reaction_template_confidence_threshold` and `reaction_template_margin_threshold` from RunConfig into the harness `decision_policy` (§17) so the evolver can tune them, and recalibrate them for Jev probabilities rather than LLM self-reports.

**Evaluation hazard:** the `example_id` lookup (`_build_example_mapping_output`) bypasses the LLM on 72 curated cases at confidence 0.99. Any Jev-vs-LLM comparison must run with that path disabled, and the metric must be reaction-type accuracy against the curated label on cases where it exists.

---

## 8. Global atom mapping redesign

### 8.1 Principle

Do not ask Jev to invent a mapping. Generate a bounded set of chemically reasonable candidates deterministically, filter them with the existing `rdkit-agent atom-map check`, then use Jev only to select. Full-LLM mapping is the fallback.

### 8.2 First, decide whether global mapping matters

Today the global mapping barely reaches the proposal LLM (broken confidence path, empty mapped-state arguments). Before investing in candidate generation, run an ablation: `no_mapping_no_reagents` harness vs default on the eval tiers. If disabling global mapping does not hurt completion, the right first move is either to fix the consumers (`coordinator.py:1146`, `tool_executor.py:115-117`) so mapping has an effect worth optimizing, or to drop the module. Record the ablation in the PRD before Milestone 2.

### 8.3 Deterministic candidate generator (`mechanistic_agent/core/mapping_candidates.py`)

1. Parse reactants/products with RDKit; assign temporary identities (atom-map numbers on a working copy).
2. Hard elemental/isotopic compatibility; element mismatch is rejection, not a cost.
3. Lock unique correspondences; compare local environments (atomic number, charge, aromaticity, ring membership, degree, valence, neighbor signatures, Morgan-style environment).
4. MCS/substructure correspondence (`rdFMCS`) where appropriate; maximize preserved bonds; identify reaction-center changes.
5. Enumerate only unresolved chemically distinct correspondences; canonicalize; score; keep top-K.

None of this exists today (no `rdFMCS`, `FindMCS`, `rxnmapper` or `CanonicalRankAtoms` anywhere in the repo). Consider `rxnmapper` as a third candidate source if its license and dependency footprint are acceptable; it must still pass the same hard filter.

### 8.4 Collapse symmetry-equivalent mappings

Mappings are equivalent if they differ only by exchange of chemically equivalent atoms (`CanonicalRankAtoms(breakTies=False)`) and yield the same reaction center, bond-change set, participating identities and stereochemical outcome. Required to keep Jev option sets small.

### 8.5 Candidate-count policy (initial hypotheses, harness-tunable)

```text
0                      → full LLM fallback
1                      → accept
2..mapping_max_options → Jev Choice (+ none)
..mapping_hard_max     → further deterministic pruning, Jev only if reduced
> hard max             → full LLM fallback
```

Primary benchmark: **does the deterministic top-K contain the benchmark mapping?** Recall is measurable now because every eval record carries a full mapping. Only after top-K recall is strong does Jev selection accuracy become the target.

### 8.6 Jev mapping `Choice`

Each option is a compact description of the discriminating difference (map pairs near the reaction center, preserved-bond count, changed bonds, functional-group identities, ambiguity note), never a full atom table. Include `none`. The machine-readable mapping stays in state.

---

## 9. Persistent atom identity spike (reframed)

### 9.1 Key question

Not "does the existing transform preserve identity" (there is no transform) but:

> Can we execute the LLM's mapped `reaction_smirks` (or its `mech:` move block) against a persistently mapped `current_state` to **derive** `resulting_state` and carry identity through, and does the derived state agree with the LLM's stated `resulting_state`?

If yes, three things follow: `step_atom_mapping` becomes unnecessary for most steps; the SMIRKS-vs-state consistency that is currently unchecked becomes a validator; and `resulting_state` stops being an unverified LLM claim.

### 9.2 Two implementation routes

- **Route A — SMIRKS execution.** Keep a mapped copy of the loop state. Build an RDKit reaction from the candidate's `reaction_smirks`, run it on the mapped state, canonicalize, compare against the LLM's `resulting_state` by map-stripped signature. Map numbers on the product side are the identity carrier.
- **Route B — move execution.** Apply the parsed `electron_pushes` / `mech:` bond deltas (`mechanism_moves.py:247-284`) as `RWMol` edits on the mapped state. Independent of SMIRKS syntax quality.

Prerequisite for both: stop stripping maps from the **loop** state. Today `ToolExecutor._sanitize_species_list` strips maps on every LLM input, including `current_state` for the proposal call. Keep stripping for pre-loop inputs; for the loop, send the mapped state (the benchmark prompts and few-shots are mapped, so this is what the model was trained toward) or send a stripped copy while keeping the mapped copy internally and re-associating by canonical rank. Which of these the LLM handles better is itself a measurable variant.

Hydrogens: the benchmark is explicit-H mapped; the runtime is heavy-atom with `[H+]` species. Define heavy-atom identity as required and proton identity as optional (explicit H only where proton provenance matters), and document it.

### 9.3 Identity representation

Do **not** use `Atom.GetIdx()`. Use atom-map numbers as the serialization bridge and a sidecar `PersistentAtomId → {component, current_index, provenance, original_map}` as the canonical internal identity, allocated monotonically, never reused. Custom RDKit atom properties do not survive SMILES round-trips and must not be relied on.

### 9.4 Spike result (2026-09-23)

Code: `mechanistic_agent/core/mapped_state.py`. Tests: `tests/fast/test_persistent_atom_identity.py`.

- **`persistent_identity_viable: true`**. Identity is carried through RDKit's `react_atom_idx` (Route A) or through in-place `RWMol` edits (Route B), never through template map numbers or `GetIdx()`.
- **§10.14 pass rate, Route A (SMIRKS):** 580/580 (100%). That is eval_set 100/100, flower multistep 420/420 and practice 60/60. It also holds under the heavy-atom policy (deterministic 1-in-4 sample, 40/40), and on map-stripped states with fresh numbering (unconstrained match, 580/580, measured offline). There are no failures in any category: H handling, charges, aromaticity, multi-component, SMIRKS syntax.
- **§10.11:** every 3–8 step benchmark chain (practice 12, multistep 80) runs step to step with zero new, lost, duplicate or reused ids. The final mapping equals the benchmark mapping.
- **Route B (moves):** 569/580 (98.1%). All 11 misses are peracid N-oxidation steps. Their `mech:` block (`lp:N>O` only) implies fewer bond changes than their SMIRKS: the O–O cleavage and proton shift are missing. This is benchmark data, not an executor defect.
- **Recommendation: Route A primary, Route B as a cross-check.** Route A takes charges, H counts, atom creation and deletion, and stereo (`[C@]>>[C@@]` inversion) from the SMIRKS. It also works when the LLM invents its own numbering. Route B needs map numbers that match the state and a complete move block. Its moves carry no stereo information. Where the two routes disagree, that is useful evidence of mech-block/SMIRKS inconsistency.
- **Executor-level pitfalls (resolved in code, keep in mind):**
  - RDKit reaction SMARTS keep a reactant's charge when the product atom states none. The executor applies SMILES-semantics deltas instead.
  - `useSmiles=True` ignores product charges.
  - RDKit drops fragments the template does not touch.
  - Deleted atoms are invisible unless ghosted.
  - Hand-built reactions need `UpdateProductsStereochemistry`.
  - `MolFromSmiles` defaults drop mapped H.
- **Blockers before any variant ships:**
  1. The executor has only been validated on ground truth. LLM SMIRKS agreement is unmeasured. The coordinator now records `validation_summary.smirks_state_agreement` for every accepted step (non-blocking, harness `record_smirks_state_agreement`, default on). Collect a baseline from real runs before making it a validator.
  2. `loop_state_mapping: "mapped"` (opt-in; default `"stripped"`) changes proposal-prompt inputs. It needs eval-tier evidence per `docs/change_evidence_policy.md`.
  3. ~~Branch alternatives and `RunState.mapped_loop_state` are not persisted across resume.~~ **Resolved.** The coordinator now writes a `run_resume_state` row (`core/db.py`, migration `2026_09_run_resume_state_v1`) on every applied candidate, branch point and backtrack. Each row holds the branch points with their full alternatives, the loop cursor, `mapped_loop_state`, `mapped_state_history` and an allocator high-water mark. `_hydrate_state_from_outputs` restores it, so backtracking after resume re-applies the same alternatives with the same ids, and the allocator never reissues an id. Runs recorded before the migration fall back to event replay, which has no alternatives. §10.12 and §10.13 now also pass at run level (`tests/fast/test_resume_branch_identity_persistence.py`).
  4. Unconstrained matches on stripped states can bind any of several symmetry-equivalent atoms. The chemistry is identical, but the id assignment among equivalent atoms is arbitrary (`distinct_outcomes` is recorded).
  5. `reaction_bond_deltas` (`mechanism_moves.py`) parses with RDKit defaults and drops bonds to mapped H. As a result, `observed_bond_deltas` metadata omits proton moves.
  6. §11's scoring redefinition is still required before `step_atom_mapping` is downgraded.

---

## 10. Required atom-identity tests (`tests/fast/test_persistent_atom_identity.py`)

All prerequisites of §9.2 apply. Tests 10.12 and 10.13 additionally depend on persisting branch alternatives and the mapped loop state across resume. That landed with `run_resume_state` (§9.4 blocker 3). The run-level versions are in `tests/fast/test_resume_branch_identity_persistence.py`.

10.1 **No-op round trip.** Assign IDs, serialize (mapped SMILES and the actual state serialization path), reparse; every atom keeps its ID; document which custom properties do and do not survive.  
10.2 **Bond-order change** (`C-C→C=C`, `C=O→C-O`): 100% retention for surviving atoms.  
10.3 **Bond formation without atom creation**: components merge, IDs retained, new bond references IDs.  
10.4 **Bond cleavage / component split**: identity independent of molecule-array position.  
10.5 **Substitution** (SN2-like): simultaneous delete/add/merge/split/charge change.  
10.6 **Proton transfer**: implicit-H version (heavy-atom identity preserved; proton identity undefined) and explicit-H version (proton ID transfers donor→acceptor).  
10.7 **Atom addition/removal**: pre-existing IDs unchanged; new atoms get fresh monotonic IDs; no reuse.  
10.8 **Aromatic/Kekulé**: identity survives kekulization, edit, sanitize, canonical output.  
10.9 **Stereochemistry**: remote edit leaves a stereocenter alone; intentional inversion/retention; E/Z where represented. Identity and stereo state independently testable.  
10.10 **Canonicalization stress**: edit, canonicalize, reparse, compare by ID; demonstrates `index != identity`.  
10.11 **Multi-step persistence**: run a 3–8 step benchmark mechanism through the executor; per step verify retention, provenance of new atoms, no duplicates, no reuse; compare final mapping to the benchmark mapping. **Decisive test for retiring LLM step mapping.**  
10.12 **Backtracking identity**: apply A, B; backtrack; apply C; snapshot restores exact IDs; discarded path IDs do not leak.  
10.13 **Serialization/resume**: persist to SQLite and files, reload, resume; IDs restored; replay reproduces mapping state.  
10.14 **(new) SMIRKS-vs-state agreement**: for every benchmark step, executing the benchmark SMIRKS on the benchmark `current_state` reproduces the benchmark `resulting_state`. This validates the executor on ground truth before it is used on LLM output.

---

## 11. Step atom mapping target behavior

If §10 passes:

```text
1. derive mapping from persistent IDs (zero model calls)
2. if incomplete: local deterministic candidates (§8.3 on current→resulting)
3. exactly one → accept
4. small ambiguous set → Jev Choice
5. unresolved → full LLM mapping fallback
```

**Scoring must change first.** Today the `step_atom_mapping` subagent score and 20% of per-step validity are the LLM's self-reported confidence; a deterministic method reporting 1.0 would inflate eval scores by up to ~0.07, and reporting nothing scores 0.5. Before any variant ships, redefine the mapping component of validity as **agreement with the benchmark mapping** where ground truth exists (and `atom-map check` pass rate where it does not), and regenerate the leaderboard baseline under the new definition so old and new harnesses are compared fairly.

Expected end state: most accepted steps require **zero model calls for step mapping**.

---

## 12. Missing reagent behavior

A deterministic gate already exists (balanced → skip). Add a Jev `Noul` in front of the remaining calls:

> Does the reaction specification appear incomplete enough that missing chemistry must be generated before a coherent mechanism can be proposed?

```text
p low       → skip missing-reagent LLM
p high      → call it
p uncertain → harness-configurable; default: let the proposal LLM discover the gap
```

Because `Noul` has no confidence field and was measured underconfident, calibrate its threshold on labeled runs (cases where the LLM call did / did not change the species registry) before enabling. Candidate rescue keeps using a full LLM where species must be invented.

---

## 13. Full LLM responsibilities after this change

Generative roles: propose elementary steps; generate intermediates; generate SMIRKS/electron pushes; generate missing reagents/byproducts when no bounded set exists; resolve mapping only when deterministic + Jev fail.

### Opportunistic review

The proposal LLM receives provisional upstream decisions and returns a compact enum field:

```json
"context_review": {"ph": "accept|question", "conditions": "accept|question",
                   "atom_mapping": "accept|question", "reaction_type": "accept|question"}
```

Instruction: upstream decisions are provisional; use when consistent; flag `question` only on material conflict; no commentary on accepted items. A `question` is recorded in the trace, lowers trust in that decision, may trigger re-evaluation only if needed for execution, and becomes calibration data. Note that `template_alignment` per candidate already exists (`coordinator.py:3285-3300`); `context_review` generalizes it.

---

## 14. Candidate-ranking experiment

### 14.1 What is being compared (corrected)

There is no LLM ranker today. Production order is the generator's self-assigned `rank`. In multi-agent topologies `_consensus_merge` re-ranks by cross-agent agreement on `reaction_smirks`. So the rankers are:

```text
A. generator rank (production baseline)
B. consensus merge (existing, multi-agent topologies only)
C. Jev Choice rank over the candidate set + none
D. independent LLM judge (new; evaluate_run_judge is run-level and unimplemented, so this is a new per-step prompt)
```

The LLM judge sees the same context and candidates as Jev, but not generator rank, Jev rank or ordinal labels; use randomized candidate IDs.

### 14.2 Shadow mode

Production ordering unchanged; B/C/D recorded per step. No ranking experiment changes a mechanism path in the first phase.

### 14.3 Evaluation

Per ranker: expected next-intermediate agreement (benchmark), bond-change agreement, completion, retries, backtracks, path length, final score, cost, latency. Pairwise agreement rates; cases where Jev uniquely picks the successful candidate; cases where Jev would have failed; calibration of Jev top-choice probability (Brier, ECE by band).

### 14.4 Active variants

Only if shadow data justify it: harness variants `rank_generator`, `rank_consensus`, `rank_jev`, `rank_llm_judge`, evaluated on medium (and hard) tiers under the normal merge bar.

---

## 15. Probability and uncertainty policy

Jev probability controls **whether another model call is needed**, never chemistry.

- Mapping: one candidate → no model; small set + strong separation → Jev; weak separation → LLM; unbounded → LLM.
- Reaction type: high top probability + margin → active template guidance; moderate → weak; low / `no_match` → disabled.
- Candidate rank: recorded only until eval-backed.

All thresholds are harness values, unset (observational) until Phase D calibration produces them, calibrated **per question and per Jev version**, and never copied across question types.

---

## 16. Proposed code changes

### 16.1 Modules

```text
mechanistic_agent/
  decisions/jev.py            # generic client, no chemistry
  decisions/policies.py       # threshold/policy resolution from harness
  core/mapped_state.py        # SMIRKS/move executor + persistent identity (§9)
  core/mapping_candidates.py  # §8.3–8.4
  core/ranking.py             # CandidateRanker protocol, Generator/Consensus/Jev/LLMJudge
  core/llm_calls.py           # explicit LLM-call counter (§18)
```

### 16.2 Jev adapter and catalog

- `JevDecisionClient.choice/score/noul/decide_many`; captures model id + pinned version, provider, question type, selected output, full distribution, confidence where the type has one, latency, usage/cost, failure.
- **Catalog:** add a `model_pricing.json` entry (provider `typesafe` or `openrouter` with a `decision` capability flag; `supports_tools: false`; input-only pricing). SOUL Guardrail 3 forbids constructing model ids outside the catalog. The `agent-bridge` entry is the precedent for a non-chat responder.
- **Routing:** `get_model_provider` defaults to `openai` (`model_registry.py:81-85`) and `get_chat_model` (`llm.py:617-697`) only knows chat adapters. A decision model needs an explicit provider branch and a non-chat adapter; a chat wrapper is the wrong shape. Prefer OpenRouter's Decisions endpoint so the existing key and provenance plumbing apply.
- Fast tests with mocks only; no network.

### 16.3 Mapping agent

```text
MappingAgent
  generate_global_candidates()  choose_global_candidate()
  derive_step_mapping_from_identity()  generate_step_candidates()  choose_step_candidate()
  llm_fallback_global()  llm_fallback_step()
```

Fix the two dead consumers first (`coordinator.py:1146` confidence path; `tool_executor.py:115-117` empty mapped args).

### 16.4 Conditions agent

One Jev request for environment, pH band, compatibility, missing-chemistry Noul; bounded species table; persisted per-run summary instead of the process-global deque.

### 16.5 Reaction-type agent

Jev implementation behind the existing `selected_type_id / selected_label_exact / confidence / top_candidates` contract; `top_candidates` from the distribution; example-id bypass disabled in comparison mode.

### 16.6 Proposal schema

Optional `context_review` enum object; no rationale unless `question`.

### 16.7 Ranking abstraction

`CandidateRanker.rank(state, candidates) -> RankingResult`; harness fields `active_ranker` and `shadow_rankers[]`.

### 16.8 Executor and validator

`mapped_state.execute(candidate)` returns derived `resulting_state`, identity map and a `smirks_state_agreement` flag. Add `smirks_state_agreement` as a recorded (initially non-blocking) validator alongside the three existing ones.

---

## 17. Harness configuration

`HarnessConfig.from_dict` (`types.py:411-443`) silently drops unknown keys and `save()` erases them, so the schema must be extended explicitly (new dataclass fields, `schema_version` bump, `version` SHA will change). Proposed block:

```json
{
  "decision_policy": {
    "conditions": "llm|jev",
    "global_mapping": "llm|rdkit_jev_llm_fallback|disabled",
    "step_mapping": "llm|identity_rdkit_jev_llm_fallback|disabled",
    "reaction_type": "llm|jev",
    "missing_reagents_gate": "balance_only|balance_plus_noul",
    "candidate_ranker": "generator|consensus|jev|llm_judge",
    "shadow_rankers": ["jev", "llm_judge"],
    "loop_state_mapping": "stripped|mapped"
  },
  "jev": {
    "model": "typesafe/jev-1.13.0",
    "mapping_max_options": 6,
    "mapping_hard_max_options": 12,
    "thresholds": {
      "mapping_accept_probability": null, "mapping_min_margin": null,
      "reaction_type_active_probability": null, "reaction_type_min_margin": null,
      "missing_chemistry_noul": null
    }
  }
}
```

Evolver changes required: extend `_RUN_CONFIG_DEFAULT_KEYS` or add a `decision_policy` lane in `HARNESS_MUTATION_TOOL` and `LLMLaneMutator.apply` with enum-typed coercion (today values are forced to bool/int, `llm_mutator.py:386-393`). Add the new modules to every `harness_versions/*/harness.json` so `set_enabled` can toggle them. Reaction-template thresholds move here from RunConfig.

---

## 18. Trace schema additions and instrumentation

Record every Jev decision:

```json
{"decision_engine": "jev", "model": "typesafe/jev-1.13.0", "decision_type": "choice|score|noul",
 "question_id": "...", "selected": "...", "probabilities": {...}, "confidence": 0.0,
 "fallback_triggered": false, "fallback_reason": null, "latency_ms": 0, "cost": 0.0}
```

(`confidence` is null for `noul`.)

Mapping trace: `candidate_count_raw`, `after_equivalence_collapse`, `after_pruning`, `benchmark_mapping_in_top_k`, `selected_candidate_id`, `atom_map_check_passed`.

Identity trace: `identity_source: persistent|derived|inferred`, `preserved_atom_count`, `new_atom_count`, `lost_atom_ids`, `duplicate_atom_ids`, `smirks_state_agreement`.

**Instrumentation gaps to close before measuring "calls avoided":** there is no LLM-call counter; `MechanismAgent` is labeled `source="llm"` though deterministic; internal retry and fallback-model usage is not captured (`tools.py:3823`, reaction-type fallback); `duration_human` is a literal `".1fs"` string (`coordinator.py:730-770`). Add an explicit per-run counter of real model invocations with tokens, by call name and engine (`llm|jev`), and report it on the leaderboard.

---

## 19. Test and evaluation plan

### Phase 0 — instrumentation and baselines (new)

- LLM-call counter and token accounting; fix `source` labels; capture retry usage.
- Mapping-recall metric against benchmark mappings (global and per step).
- Tier resolution: keep `eval_tiers.json` and `baseline_tiers_clawdiator.json` synchronized, fail loudly when a requested tier resolves to zero cases, and keep `docs/eval_tiers.md` current (landed in PR #29).
- Redefine the mapping component of scoring (§11) and regenerate the baseline leaderboard rows.
- Ablation: `no_mapping_no_reagents` vs default (§8.2).
- Verify functional-group injection actually happens when the harness module is enabled (§5 row 2).

### Phase A — infrastructure

Jev client with mocks: Choice/Score/Noul parsing, multi-question, timeout, malformed response, missing probabilities, metadata capture, fallback. Catalog entry and routing branch tests. Import-chain and no-network guarantees in `tests/fast/`.

### Phase B — mapped-state executor and identity

Every test in §10, plus 10.14 on the full benchmark. **Gate:** do not remove or downgrade `step_atom_mapping` until 10.11–10.14 pass.

### Phase C — mapping candidate generator

Unit cases: identity reaction, substitution, elimination, addition, carbonyl, aromatic substitution, symmetric molecules, rearrangement, multi-component, explicit proton transfer, stereo. Metrics: top-1/3/6 recall vs benchmark, mean and p95 candidate count, fraction with one candidate, fraction requiring Jev, fraction requiring LLM. Do not evaluate Jev selection before recall is measured.

### Phase D — Jev calibration (per question)

For each Jev question (reaction type, environment, pH band, compatibility, missing-chemistry Noul, mapping Choice, ranking Choice), on 1–2k labeled historical decisions from the DB and benchmark: accuracy by confidence band, Brier, ECE, accuracy vs option count, and the threshold at which accuracy clears the current LLM. Pin the Jev version. Test narrow-question decomposition with a fitted combiner against the single broad question.

### Phase E — conditions and reaction type on eval tiers

Old vs new modules with the `example_id` bypass disabled: completion, pathway score, retries, backtracks, template disablement rate, reaction-type accuracy vs curated label, model calls and tokens avoided, cost, latency.

### Phase F — ranking shadow comparison

§14.3 across generator, consensus, Jev and LLM judge; active variants only afterward.

---

## 20. Acceptance criteria

**Persistent identity / executor:** 100% retention for surviving atoms in supported edit types; no duplicate or reused IDs; merge/split, pause/resume and branch/backtrack restoration correct; canonicalization round-trip passes; 10.11 and 10.14 pass on the benchmark; hydrogen policy documented.

**Global mapping:** ablation shows mapping matters (or the module is dropped); strong top-K recall against benchmark mappings; symmetry collapse working; option counts bounded on most eval reactions; LLM fallback safe. Research target, not merge requirement: ≥90% of eval reactions reduced to ≤6 meaningful options.

**Step mapping:** research target ≥95% of accepted steps need no inference or ≤4 residual choices; scoring redefinition landed and baseline regenerated first.

**Jev decisions:** per-question calibration measured; thresholds set from Phase D, not guessed; every Choice has `none`; version pinned.

**Candidate ranking:** Jev ranking does not become default until an eval-backed harness variant meets the merge bar (`docs/change_evidence_policy.md`: improve `medium`; `hard` preferred).

---

## 21. Rollout plan

| Milestone | Deliverable | Production change |
| --- | --- | --- |
| M0 instrumentation | call counter, mapping-recall metric, tier population, scoring redefinition, ablation report | none (scoring baseline regenerated) |
| M1 executor spike | `mapped_state.py`, §10 suite incl. 10.14, `persistent_identity_viable` decision, `smirks_state_agreement` recorded | none |
| M2 Jev foundation | adapter, catalog entry, routing, traces, mocks; reaction-type Choice as variant; Phase D calibration for it | variant only |
| M3 conditions | Jev conditions request, species table, persisted summary; pH heuristic fix | variant only |
| M4 mapping | candidate generator, collapse, recall benchmark; Jev selection + LLM fallback; residual step mapping | variant only |
| M5 context review | schema, prompt update, disagreement stats | variant only |
| M6 ranking shadow | rankers, neutralized labels, shadow traces, report | none |
| M7 evolution | defaults change where eval evidence supports; retire routine mapping LLM calls; `decision_policy` mutable by the evolver | via normal gates |

Reaction type is deliberately before conditions and mapping: it has an existing output contract, existing thresholds, ground-truth labels, and fits Jev's Choice type directly.

---

## 22. Failure modes and guardrails

- **Jev is not a chemist.** Mitigation: shadow first; ground-truth metrics; per-question calibration; narrow questions; deterministic candidates so Jev only discriminates.
- **Overconfidence** (measured for Choice/Score). Mitigation: probabilities never override chemistry; thresholds from Phase D; `none` option; fallback.
- **Candidate generator drops the correct mapping.** Mitigation: recall before accuracy; conservative pruning; LLM fallback.
- **Scoring inflation.** Mitigation: redefine mapping scoring before any deterministic mapping ships (§11).
- **Evaluation leakage.** Mitigation: disable the `example_id` reaction-type bypass in comparisons.
- **Atom-map numbers collide with chemistry-facing numbering.** Mitigation: sidecar identity canonical; maps as transport.
- **Canonicalization reorders atoms.** Mitigation: never index as identity; explicit tests.
- **Explicit/implicit H mismatch** between benchmark and runtime. Mitigation: separate heavy-atom and proton policies; explicit H only where provenance matters.
- **LLM trusts wrong Jev context.** Mitigation: provisional labeling; `context_review`; validators final.
- **Prompt bloat from review.** Mitigation: fixed enum, no rationale unless `question`.
- **Injection via state.** Mitigation: machine-built state only; never user prose.
- **Ranker experiment changes chemistry.** Mitigation: shadow mode; active ranker only via explicit variant.
- **Version drift.** Mitigation: pin `jev-1.13.0`; recalibrate on upgrade; record version in every trace.
- **Harness schema silently drops the policy block.** Mitigation: dataclass fields + round-trip test.

---

## 23. Open research questions

1. Does global atom mapping affect completion at all today (ablation)?
2. What fraction of global mappings can RDKit reduce to ≤6 meaningful candidates, measured against benchmark recall?
3. What fraction of step mappings disappear through executed-SMIRKS identity?
4. Does sending the LLM a mapped loop state help or hurt proposal quality?
5. Sidecar identity vs atom-map transport: is transport sufficient in practice?
6. How should implicit proton identity be represented?
7. How many mapping options can Jev reliably distinguish from compact chemistry descriptions, and at what calibration?
8. Does taxonomy pre-filtering improve Jev reaction-type accuracy enough to justify the logic?
9. Do Jev conditions outputs improve downstream proposal quality, or is the benefit only call reduction and latency?
10. How often does the proposal LLM `question` upstream Jev context, and who is right?
11. Does Jev ranking reduce retries/backtracks relative to generator rank and consensus merge?
12. What thresholds minimize full-LLM calls while preserving completion, per question and per Jev version?
13. Does executed-SMIRKS agreement predict step correctness well enough to become a blocking validator?

---

## 24. Definition of done

Mechanistic supports an eval-backed harness in which:

```text
reaction condition assessment = Jev Choice + Score (+ Noul gate), bounded species table
reaction type                 = Jev Choice over taxonomy + no_match
global atom mapping           = deterministic candidates + atom-map check + Jev Choice + LLM fallback
step atom mapping             = executed-SMIRKS identity, residual deterministic candidates, Jev, LLM fallback
mechanism generation          = full LLM, with derived resulting_state checked against the LLM's
candidate ranking             = pluggable (generator / consensus / Jev / LLM judge) with shadow evidence
hard chemistry validation     = deterministic (+ smirks_state_agreement)
retry/backtracking            = deterministic
```

and the eval/trace system quantifies: model calls and tokens avoided, mapping candidate recall, Jev selection quality and calibration per question, LLM/Jev disagreements, completion, retries, backtracks, leaderboard score, cost, latency.

The final architecture makes a full LLM call because **new chemistry must be invented**, not because the harness needs a classifier, a bookkeeping decision, or an atom-identity guess.

---

## Appendix A. Source references for §7.0

- TypeSafe Jev API docs: https://jevmodel.org/api/
- Pydantic AI TypeSafe provider (limits, model ids): https://pydantic.dev/docs/ai/models/typesafe/
- OpenRouter Decisions API harness and benchmark: https://github.com/souvikr/jev-test
- Calibration and decomposition findings: https://www.beri.net/article/typesafe-jev-typed-decision-model-calibration-decomposition-shadow-eval
- Launch coverage: https://www.marktechpost.com/2026/09/19/typesafe-ai-releases-jev/
