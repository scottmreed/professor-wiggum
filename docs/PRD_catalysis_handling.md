# PRD: Catalysis handling, longer catalytic mechanisms, and a shared organometallic package

Status: Wiggum-side implementation landed on branch `fix/metal-ligand-radical-artifact` (2026-11-08); see §10. Owner: Wiggum maintainers, with ChemIllusion backend.
Related: [PRD_live_mechanism_observatory_chemillusion.md](PRD_live_mechanism_observatory_chemillusion.md),
[scoring_quality_v2.md](scoring_quality_v2.md), [change_evidence_policy.md](change_evidence_policy.md).

## 1. Summary

ChemIllusion users ask Wiggum about transition-metal (TM) catalysed reactions (Suzuki,
Sonogashira, Stille, Buchwald, Heck, metathesis, hydrogenation) and about organocatalysis.
Today Wiggum treats a catalyst as one more species in the state. Nothing checks the metal
centre. Nothing tracks catalyst regeneration. The tiers never sample catalytic cases. And
the 14-step cap cuts off a fifth of FlowER's longer Pd mechanisms.

ChemIllusion already has authored-cycle chemistry. It has an MLX electron counter, an
organometallic SMILES linter with house conventions, a `catalytic_cycle` validator and a
cycle renderer. This PRD:

1. moves those modules into one versioned shared package that both repos pin;
2. adds a notation bridge, because Wiggum and FlowER draw ligands with covalent bonds
   while ChemIllusion draws them with dative bonds;
3. adds a soft metal-centre validator, catalyst tracking and a cycle projection to
   Wiggum;
4. raises the step cap for catalytic runs only;
5. builds catalysis eval data from FlowER train, plus a small expert-reviewed textbook
   set;
6. adapts scoring without changing the headline `quality_v2` formula.

The goal is predicted catalytic mechanisms. ChemIllusion's existing cycles are authored.
Every predicted cycle shown to users must say that it is predicted.

## 2. Evidence

All counts come from the FlowER **train** split (`flower_new_dataset/train.txt`) and the
local index of 257,167 mechanisms. No test or validation rows were read.

| Finding | Number |
|---|---|
| Train rows containing a TM | 246,763 of 1.93M (13%) |
| Index mechanisms containing Pd | 22,117 (8.6%) |
| Pd mechanisms whose first-step reagents include B / Cu / Sn / other (rough element tag) | 16,008 / 1,435 / 1,105 / 3,757 |
| Other metals (mechanisms): Fe / Cu / Zn / Ru / Ag / Ti | 2,863 / 1,818 / 218 / 195 / 195 / 166 |
| Sampled Pd-containing steps where bonds on Pd change | ~70% (phosphine loss, oxidative addition, transmetalation, reductive elimination) |
| Best-ranked Pd mechanism in the curriculum ordering | #2,423 (median #190,982), so tiers never sample one |
| Smallest 300 Pd mechanisms that convert to cases | 173 (failures: 126 `state_discontinuity`, 1 `no_cycle_found`) |
| Pd mechanisms with ≤14 / 15–20 / >20 steps | 19,320 (87%) / 1,695 (8%) / 1,102 (5%) |
| Current taxonomy labels for TM catalysis (86 labels) | 0 (only Grignard and "organometallic addition") |

Ru steps are mostly metal-active (metathesis). Cu is mixed: Cu acetylides are active, CuI
is often a spectator. Fe is almost always a FeCl3-type Lewis acid spectator.

**Already fixed (P0, branch `fix/metal-ligand-radical-artifact`).** FlowER draws Pd–PR3
as a covalent bond to a neutral P, and RDKit reads that P as having one radical electron.
`radical_species_changed` then sent ordinary Pd steps to strict radical validation, which
rejects metals. 63 of 290 ground-truth Pd steps (22%) lost arrow credit, and the same
gate ran on live predictions. Donor atoms bonded to a d-block metal no longer count as
radical (`core/radical_validation.py`, `tests/fast/test_metal_ligand_radical_artifact.py`).
After the fix, 290/290 Pd steps validate, and 72 of 72 multi-step ground-truth Pd
mechanisms score `quality_v2` = 1000.

**What FlowER cannot test.** FlowER mechanisms come from templates. They carry no
geometry, and ligands are idealized. FlowER can test bond, electron and catalyst
bookkeeping. It cannot test cis/trans relationships, open coordination sites or β-H
requirements. The ChemIllusion Chapter 23 material covers those: four validated cycles
(Suzuki, Heck, Wilkinson, Chauvin), 18 assessment items and structured `trouble_spots`.
Its location in the ChemIllusion repo still needs to be confirmed.

## 3. Goals and non-goals

**Goals**
- G1. One implementation of organometallic chemistry rules, shared by Wiggum and ChemIllusion.
- G2. Wiggum accepts, validates and scores correct TM-catalysed steps in either ligand notation.
- G3. Metal-centre problems produce specific, chemistry-language feedback for re-proposals.
- G4. Catalytic runs can take enough steps to finish the cycle (target: ≥95% of FlowER Pd
  mechanisms fit).
- G5. A catalysis eval tier that measures progress without polluting easy/medium/hard.
- G6. ChemIllusion can render a predicted cycle with the existing renderer and label it predicted.

**Non-goals (this PRD)**
- Multinuclear catalysts, η3-allyl or η6-arene bookkeeping beyond what the counter supports.
- Spin states, kinetics, enantioselectivity and ligand-effect prediction.
- Photoredox and radical-chain metal catalysis. Radical support exists, but metals stay
  out of the radical inventory.
- Changing the headline `quality_v2` weights.

## 4. Users and outcomes

- **ChemIllusion student or instructor.** Asks "how does this Suzuki coupling work?" and gets
  the organic product, a step-by-step mechanism, a cycle figure marked *predicted*, and
  warnings when metal bookkeeping is uncertain.
- **Wiggum maintainer.** Runs a `catalysis` tier and sees product, mechanism and
  metal-bookkeeping results separately.
- **Textbook author (ChemIllusion).** Keeps using the authored `catalytic_cycle` workflow,
  now backed by the same package that Wiggum validates against.

## 5. Requirements

### P1. Shared organometallic package

Source modules (ChemIllusion `backend/`). All are RDKit-only, with no app imports:

| Module | Role |
|---|---|
| `app/utils/organometallic_smiles.py` | House conventions v1 (dative L donors, `[O+]#[C-]->M` for CO, ionic Cp) and `lint_organometallic_smiles` |
| `app/utils/organometallic_electron_count.py` | Covalent MLX counter: `count_electrons` → VE, OS, d^n, CN, L/X/Z; `STEP_DELTAS` |
| `app/utils/catalytic_cycle.py` | `validate_catalytic_cycle` (one simple closed cycle, computed vs authored OS/VE, per-kind deltas), `describe_catalytic_cycle` |
| `scripts/interpret_tmc.py` | XYZ → TM graph via `xyz2mol_tm` (optional extra) |

Tests come along with them: `tests/unit/test_utils/test_organometallic_*.py`,
`test_catalytic_cycle.py`, and `tests/fixtures/catalytic_cycles/suzuki_miyaura.json`.
The render service (`catalytic_cycle_render_service.py`) stays in ChemIllusion: it is
presentation code.

- R1.1 Publish the package as `organometallic_chem` (working name), versioned with semver.
  Its import surface must not depend on either app. `xyz2mol_tm` is an optional extra.
- R1.2 Both repos pin an exact version. ChemIllusion keeps thin re-export shims at the old
  `app.utils.*` paths so existing imports and the offline topic-package validator keep working.
- R1.3 The conventions version (`house_conventions: v1`) is part of the package API. Any
  change to it is a major version bump.
- R1.4 Wiggum adds `organometallic_chem` to `requirements.txt` and `Dockerfile.runtime`.
  A package bump in Wiggum counts as a harness-engine change for the ChemIllusion runtime
  bump workflow.

### P1b. Notation bridge (covalent ↔ dative)

FlowER and most LLM output use covalent M–L bonds. The ChemIllusion linter rejects them:
they re-read as phantom-H phosphoranes and count L as X. Wiggum must accept what models
write and convert it before counting.

- R1b.1 `to_house_convention(smiles) -> (smiles, notes)`. For each bond between a d-block
  metal and a neutral main-group donor whose written valence exceeds its default
  (4-bonded P/As/N, 3-bonded O/S, a C≡O carbon), rewrite the bond as a dative donor→metal
  bond and keep atom maps.
  - Amido, alkoxide, halide, aryl and alkyl bonds stay X.
  - CO is rewritten to `[O+]#[C-]->M`.
- R1b.2 Some forms are ambiguous: an η2-alkene drawn as a metallacyclopropane (X2 vs L, an
  OS difference of 2), or bridging ligands. These return `ambiguous` with a reason. The metal
  validator then abstains rather than failing.
- R1b.3 `to_covalent(smiles)` is the inverse, for Wiggum's internal states and SMIRKS.
  Wiggum's states stay covalent so atom balance, SMIRKS replay and mapping are unchanged.
  The dative form is used only for counting and rendering.
- R1b.4 Round-trip property test: on 10k FlowER train Pd/Cu/Ru species, `to_covalent ∘
  to_house_convention` gives back the same canonical SMILES, and every converted species
  lints clean or is flagged `ambiguous`.

### P2. Metal-centre validation in Wiggum (soft by default)

New deterministic post-step validator skill `skills/mechanistic/metal_center_validation/`
(`SKILL.md` + `validator.py`), dispatched like the other three. It runs only when a step
changes a bond or formal charge on a d-block atom.

- R2.1 For each metal atom (by atom map) changed by the step, convert both states with
  P1b and run `count_electrons` on the species that contains the metal. Report OS, VE,
  d^n and CN before and after.
- R2.2 **Step-kind classification.** Prefer the step kind the proposal declares, a new
  optional `metal_step_kind` field in `MECHANISM_STEP_PROPOSAL_TOOL` using the
  `STEP_DELTAS` keys. Otherwise infer it from the bond changes at the metal: two new X
  bonds from one σ-bond means oxidative addition, and so on. Compare the observed
  (ΔOS, ΔVE) to `STEP_DELTAS`. A mismatch with a declared kind is reported. A mismatch with
  an inferred kind is reported only when inference was unambiguous.
- R2.3 **Policy** `RunConfig.metal_center_policy`: `off` | `soft` (default) | `strict`.

  | Check | soft | strict |
  |---|---|---|
  | Counter error or ambiguous notation | abstain (event) | abstain (event) |
  | OS outside the element's known range; VE > 18 for groups 8–10; negative d^n | **fail** | fail |
  | ΔOS/ΔVE disagree with the declared step kind | warn + re-proposal hint | fail |
  | Inferred-kind mismatch; CN > 6; 16/18-electron rule breach in groups 8–10 | warn | warn |
  | Geometry (cis/trans for reductive elimination, vacant site for β-H) | not checked (P5) | not checked |

  Following the `rdkit-agent` soft-pass pattern, a warning never fails a step on its own.
- R2.4 **Events.** `metal_center_check` per candidate (`metal_center.v1`: per-metal
  before/after counts, step kind and its source, findings, policy, abstain reason).
  `candidate_validation_result` gains `metal_center`. `/observatory` projects it, and the
  UI shows OS/VE badges per spine state.
- R2.5 **Feedback.** Findings render into `rejection_notes` and threaded VALIDATION FEEDBACK
  as chemistry sentences, e.g. "Pd goes from Pd(0) to Pd(II) but the step is labelled
  transmetalation (expected no OS change)". The phrasing library comes from the Chapter 23
  `trouble_spots` (P5).
- R2.6 Arrow-count preference and proton-transfer preference keep abstaining on metal
  steps. No change.

### P3. Catalyst tracking and cycle semantics

- R3.1 **Catalyst identification.** A starting-pool species that contains a d-block metal
  is a *catalyst candidate*. The pre-analysis `conditions` section gains
  `catalysts: [{smiles, role: catalyst|precatalyst|stoichiometric_reagent}]`, checked by
  `preanalysis_checks`. This keeps a stoichiometric metal reagent (organocuprate,
  Grignard, FeCl3 Lewis acid used in excess) from being judged as a cycle. Organocatalysts
  (DMAP, proline, NHC precursors, acids and bases) use the same field with no metal checks.
- R3.2 **Identity.** Metal atoms keep their persistent `pid` in `atom_identity.v1`. The
  catalyst lineage is the sequence of species that contain that pid.
- R3.3 **Regeneration.** After the loop, `mechanism_audit` adds
  `catalyst_regeneration: {metal_pid, start_species, end_species, regenerated, via}`.
  - If the end species matches the starting catalyst up to resonance key, or the active
    form after a precatalyst activation, it is `regenerated`.
  - If a declared catalyst is not regenerated, the result is an `unregenerated_catalyst`
    finding and the grade is `approximate`, so the case cannot pass.
  - The existing `resolved_catalyst` net-equation cancellation is unchanged.
- R3.4 **Completion.** `assess_target_product_state` adds to `tolerated_species`: catalyst
  species, released ligands (the free donor of a dissociated L) and boron, tin or zinc
  by-products that the reaction class declares. This applies only when R3.1 marked them,
  so leftover organic junk still blocks completion.
- R3.5 **Cycle projection** `catalytic_cycle_projection.v1` on `/observatory`, built from the
  chosen path:
  - `preamble`: precatalyst activation and ligand-loss steps before the first repeated
    metal species;
  - `cycle`: species and steps in the house `catalytic_cycle` spec shape, with species
    SMILES in house convention, `kind` from R2.2, and `enters`/`leaves` from the
    organic species consumed and released;
  - `closed: bool`;
  - `provenance: {source: "predicted", run_id, harness, model}`.

  The projection is passed to `validate_catalytic_cycle`. If it validates, ChemIllusion
  may render it with the existing renderer, with a visible "Predicted by Wiggum" label. If
  not, ChemIllusion shows the linear mechanism with the validator's errors as caveats.
  Branches and backtracks are never projected.

### P4. More steps for catalytic runs

Today `DEFAULT_MAX_STEPS = 14` (`core/types.py`), the API schemas default to 14, and
ChemIllusion clamps to 14 (`MAX_STEPS_LIMIT` in `backend/app/schemas/mechanism_submission.py`).
87% of Pd mechanisms fit in 14 steps, and 95% fit in 20.

- R4.1 Add `CATALYTIC_MAX_STEPS = 22` and apply it when a run's max_steps is unset and either
  pre-analysis marks a catalyst (R3.1) with a TM-catalysis or organocatalysis reaction type,
  or the user sets `catalytic: true`. Explicit `max_steps` still wins.
  `harness_run_config_defaults_applied.sources` records why the cap was chosen.
- R4.2 Add a `catalysis_step_cap_reached` field to the existing `step_cap_reached` event, so
  cap hits on catalytic runs are counted separately.
- R4.3 **Coarse-graining.** Prompts allow merging ligand dissociation or association into
  the next elementary event (for example "oxidative addition with loss of PPh3"), as long as
  atoms and electrons balance. Spectator carry already lets an untouched free ligand ride
  along. This brings typical Suzuki paths down from 8–10 FlowER steps to 4–6.
- R4.4 **ChemIllusion.** Replace the flat clamp with `MAX_STEPS_LIMIT = 14` and
  `MAX_CATALYTIC_STEPS_LIMIT = 22`, sending `max_steps` only when the user overrides it.
  Check the product cost budget and the `--max-runtime 600` official limit at 22 steps.
  Threaded thread compaction already bounds prompt size.
- R4.5 Do not raise the cap for non-catalytic runs in this PRD. Look at step-cap telemetry first.

### P5. Data and evaluation

- R5.1 **Catalysis dev set.** `scripts/build_catalysis_dataset.py` reads FlowER **train** only.
  It selects mechanisms where a d-block atom changes bonds in at least one step, with
  ≤22 steps, converted to cases. Output goes to `training_data/catalysis_development/`.
  Like `radical_development`, it records source line references and a hash of the input.
  It is stratified by metal and coupling family (Suzuki, Sonogashira, Stille, Buchwald,
  Heck, carbonylation, metathesis).
  - Fix the catalyst-related `state_discontinuity` conversion failures first: catalyst and
    ligand species dropped or redrawn between FlowER rows. Target ≥90% conversion on
    metal-active mechanisms.
- R5.2 **Organocatalysis slice.** Same builder, selected by catalyst role (DMAP, amine,
  NHC, Brønsted acids and bases regenerated by the end). These cases need no metal checks;
  they test R3.3/R3.4.
- R5.3 **Tier.** Add a `catalysis` tier to `eval_tiers.json` (append-only, 10-case slices),
  kept separate from easy/medium/hard. Add a FlowER **test** catalysis slice to
  `training_data/leaderboard_holdout/` through the existing holdout tooling only.
  `evolve_harness.py` and tuning tools keep refusing holdout sets.
- R5.4 **Textbook gold set.** Import the four Chapter 23 cycles and their assessment
  variants as `training_data/catalysis_textbook/` after independent expert review. Fix the
  reported cis/trans and β-H inconsistencies before any item counts as ground truth. Each
  `trouble_spot` becomes a negative case: an authored wrong step that `strict` policy must
  reject or `soft` policy must warn on. This set is small and gates geometry work. It is not
  a leaderboard tier.
- R5.5 **Taxonomy and few-shots.** Add reaction-type labels for the Pd cross-couplings,
  Heck, Buchwald–Hartwig, olefin metathesis, Wilkinson hydrogenation and the main
  organocatalytic classes. Add proposal few-shots from FlowER train and the textbook cycles,
  in house notation for display and covalent notation for SMIRKS. These are prompt changes
  and go through the evidence gate.

### P6. Scoring

`quality_v2` already handles catalytic paths. With P0, ground-truth Pd paths score 1000.
Connectivity is fine because regenerated Pd(0) counts as made earlier, and
`repeated_state` compares whole states. Changes:

- R6.1 **Product axis** compares only the main organic product(s). Catalyst, ligand and
  declared by-product species in the case's `products` are removed from the target list.
  The `products` list itself is unchanged.
- R6.2 **Reference axis** becomes ligand-shuttle agnostic, the way it is already
  proton-shuttle agnostic: ligand association and dissociation steps are skipped when the
  prediction is aligned with the FlowER sequence. A model that merges phosphine loss into
  oxidative addition (R4.3) loses nothing.
- R6.3 **Metal bookkeeping diagnostic.** `quality["metal_center"]` records the share of
  metal-active steps with consistent counts, the share with declared kinds that match, and
  catalyst regenerated (yes/no). This is reported in records and the radar SVG, not in
  the 1000-point score. It moves into a `quality_v3` axis only after it has agreed with the
  textbook gold set (R5.4) for one release.
- R6.4 The leaderboard shows `catalysis` as its own tier column. Harness-vs-baseline pairing
  is unchanged.

### Strictness when the user wants the organic product

| Layer | Treatment | Reason |
|---|---|---|
| Organic product correct | hard (product gate) | It is what the user asked for |
| Atom and charge balance including metal and ligands | hard (existing policy) | Notation-independent, never wrong |
| Electron conservation on mapped atoms | hard (existing) | Already metal-safe after P0 |
| Catalyst regenerated or declared consumed | grade gate (`approximate` cannot pass) | A catalytic mechanism that destroys the catalyst is wrong in a way users notice |
| Impossible metal states | hard | Cheap and unambiguous |
| OS/VE deltas vs step kind | soft, promoted per kind once evals agree | `STEP_DELTAS` is idealized; real steps combine events |
| Geometry and coordination sites | not checked until R5.4 exists | No data to validate it against |

## 6. Success metrics

- M1. P0 holds: no FlowER train Pd/Cu/Ru ground-truth step fails because of notation. Tracked
  by a 500-step replay test with fixtures from train.
- M2. Catalysis tier: products-given `quality_v2` and pass rate measured for `threaded_unified`
  vs the one-shot baseline at the same thinking level. The first goal is a measured baseline,
  not a target number.
- M3. Catalyst regeneration rate ≥90% on passed catalysis cases.
- M4. Cap hits on catalytic runs below 5% at 22 steps.
- M5. Metal-centre soft findings agree with the textbook gold set's labels on ≥90% of
  steps before any check is promoted to strict.
- M6. ChemIllusion: share of catalytic product runs whose cycle projection validates and
  renders.

## 7. Rollout

1. P0. Done on a branch. It is a validator-path change, so it is evidence-gated with a
   medium-tier paired eval per `change_evidence_policy.md`.
2. P1 + P1b. Package extraction with tests moved, ChemIllusion shims, round-trip property test.
3. P2 in `soft`, plus R3.1–R3.4, plus R6.1–R6.3. Evidence on the catalysis dev tier and an
   unchanged medium/hard tier (no regression on non-catalytic cases).
4. P4 cap and coarse-graining prompts (evidence-gated prompt change), then the ChemIllusion
   clamp change.
5. R3.5 projection and ChemIllusion rendering with the "predicted" label.
6. R5.4 textbook gold set. Only then promote any metal check to strict, or into `quality_v3`.

Merges that touch `core/`, validators, prompts or the pinned package trigger the
ChemIllusion runtime-bump PR (never auto-merged).

## 8. Risks

- **Notation drift.** Models mix covalent, dative and ionic forms. Mitigation: P1b converts
  with an explicit `ambiguous` result, and the validator abstains instead of guessing.
- **Over-rejection.** `STEP_DELTAS` would reject correct combined steps. Mitigation: soft by
  default; strict only per kind and only after M5.
- **Template bias.** FlowER Pd mechanisms are templated (always explicit L loss, idealized
  ligands). Mitigation: ligand-agnostic reference axis, plus the textbook set as an
  independent check.
- **Cost and latency.** 22-step runs cost more. Mitigation: coarse-graining, step-cap
  telemetry, and a ChemIllusion budget check before raising its clamp.
- **Two sources of truth.** Mitigation: one package, exact pins, shims, and a conventions
  version in the API.

## 9. Open questions

1. Package home: a standalone repo published to a private index (recommended, since neither
   app owns it), or a subdirectory of Wiggum that ChemIllusion imports through its existing
   pinned checkout. The second is simpler but ties textbook tooling to runtime pins.
2. Should `metal_step_kind` be required for catalytic proposals, or stay optional with
   inference as the fallback?
3. Where does the Chapter 23 package live, and who does the independent expert review?
4. Is 22 the right catalytic cap, or should the cap scale with the selected reaction type?
5. Should organocatalysis get its own tier, or a slice inside `catalysis`?

## 10. Implementation status (2026-11-08)

Done in this repo. Fast suite green except the six eval-tier data tests that also fail on `main`.

| Item | Status | Where |
|---|---|---|
| P0 radical misread | done | `core/radical_validation.py`, `tests/fast/test_metal_ligand_radical_artifact.py` |
| R1.1/R1.3 shared package | done, **local only** (see below) | `organometallic_chem/` (v1.0.0, conventions v1), 79 moved ChemIllusion tests pass |
| R1.4 dependency | done | `pyproject.toml` package list, `Dockerfile.runtime` |
| R1b notation bridge | done | `organometallic_chem/notation.py`; 10k-species round-trip measured, 400-species fixture test |
| P2 metal-centre validator | done (`soft` default) | `skills/mechanistic/metal_center_validation/`, `validators._append_metal_center_check`, `metal_center_check` event, `metal_step_kind` in `MECHANISM_STEP_PROPOSAL_TOOL` |
| R3.1 catalyst identification | done, deterministic only | `core/catalysis.identify_catalysts` (LLM `conditions.catalysts` field not added: prompt change, evidence-gated) |
| R3.3 regeneration | done | `mechanism_audit.catalyst_regeneration`, `unregenerated_catalyst` grades `approximate` |
| R3.4 completion tolerance | done | `smiles_utils._catalysis_tolerated` |
| R3.5 cycle projection | done | `/observatory` `catalytic_cycle` |
| R4.1/R4.2 catalytic cap | done | `CATALYTIC_MAX_STEPS = 22`, `_resolve_max_steps`, `step_cap_reached.cap_source` |
| R4.3 coarse-graining prompt | **not done** | prompt change; needs approved evidence traces |
| R4.4 ChemIllusion clamp | **not done** | ChemIllusion repo; needs the cost/runtime check |
| R5.1 dev set | done | 34 transition-metal cases, `training_data/catalysis_development/` |
| R5.2 organocatalysis slice | builder done, **slice empty** | FlowER has no covalent organocatalysis in 60k scanned mechanisms (DMAP and amines are spectators); needs another source |
| R5.3 `catalysis` tier + holdout slice | **not done** | needs a tier-file decision (the append-only `eval_tiers.json` tests already fail on `main`) and the official holdout tooling |
| R5.4 textbook gold set | **not done** | Chapter 23 package not found in the ChemIllusion checkout; needs expert review |
| R5.5 taxonomy + few-shots | **not done** | prompt and few-shot change; evidence-gated |
| R6.1 product targets | done | `quality_scoring._organic_targets`, completion targets |
| R6.2 ligand-agnostic reference | done | `skeleton_alignment._without_metal_l_donor_bonds` |
| R6.3 diagnostic | done (record field) | `quality["metal_center"]`; radar SVG not changed |
| R6.4 leaderboard column | **not done** | follows R5.3 |
| R1.2 ChemIllusion shims | **not done** | waits on open question 1 |

**Findings that changed the plan**
- FlowER contains no covalent organocatalysis to sample: where DMAP or amine catalysts appear, no step touches their atoms. Species that reappear unchanged are excess reagent copies. The organocatalysis slice must come from curated sources.
- The 126 `state_discontinuity` failures among small Pd mechanisms are not catalyst-related. In each, the next FlowER row starts from a different state, a gap in the source grouping. The builder skips and counts them; repairing them would mean inventing steps.
- Ground-truth replay on FlowER train Pd mechanisms after all changes: every metal-changing step passes the metal-centre check, every case regenerates its catalyst, audit grade `exact`, completion reached, `quality_v2` = 1000. Every closed projected cycle passes `validate_catalytic_cycle`.

**Package home (open question 1) still matters.** `organometallic_chem/` contains code copied from the private chem-art-generator repo into this public repo's working tree. It has not been committed or pushed. Pushing it publishes that code. Moving it to its own repository first means changing only the import root.
