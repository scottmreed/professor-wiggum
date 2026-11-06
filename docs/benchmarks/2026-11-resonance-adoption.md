# Resonance key and arrow-count tie-break: before / after (2026-11)

Change: `mechanistic_agent/resonance.py` (`resonance_key`) and
`Coordinator._apply_arrow_count_preference` (`core/arrow_count.py`), ported from ChemIllusion's
`backend/app/services/mechanism_resonance_service.py`. **Before** = `origin/main` `db53ba4`
(also ChemIllusion's pinned `WIGGUM_RUNTIME_REF`). **After** = branch
`feat/resonance-key-arrow-tiebreak`. No holdout set was used; nothing was published.

## 1. Offline counterfactual (stored runs, read-only)

`python scripts/arrow_count_counterfactual.py --created-before 1793900000` against the local store
(wiggum-data `mechanistic.db`; the cutoff leaves out this benchmark's own runs). A step
qualifies when it has a `branch_point_created` event (two or more validated candidates). The
validated list is rebuilt from that attempt's passed `mechanism_synthesis` outputs, in rank order.
(These stored runs predate `candidate_validation_result` / `mechanism_candidates_proposed`.)

| | Steps |
|---|---|
| Steps with ≥2 validated candidates | 93 |
| Rebuilt (distinct passed outputs = branch size) | 81 |
| Not rebuildable (count mismatch after reproposal/backtrack) | 12 |
| Proton-transfer preference would fire (pre-empts arrow count) | 0 |
| Top candidate unmeasurable (SMIRKS core-only, under-mapped, or not this step's species) | 61 |
| A lower rank has the same outcome up to resonance as the top | 5 |
| …and both sides measurable | 1 |
| **Arrow-count preference would fire** | **0** |
| Agrees with the rank-1 candidate | 81 / 81 |

Displaced vs chosen step quality (quality_v2 per-step axes): no firings, so nothing to compare.

On stored data the preference is a no-op. Most stored `reaction_smirks` map only the reaction
centre, and a core-only SMIRKS is not a measurement of the whole step (see the third bullet below),
so it never takes part. Where both sides are measurable, rank 1 already needs the fewest arrows.

Development history. Earlier versions of the counterfactual fired on 1–3 steps. Each firing came
from a defect, and each defect is now covered by a test:

1. **The resonance key dropped H counts.** ChemIllusion's original writes the erased skeleton with
   `MolToSmiles` and no `allHsExplicit`. The writer then prints an organic-subset atom bare
   whenever it has fewer H than its default valence. Acetone's key became `CC(C)O`, and an
   alkoxide-ketone and its enolate-alcohol tautomer (`C=C(C)C(C[O-])C(C)=O` vs
   `C=C(C)/C(CO)=C(\C)[O-]`) shared a key. Fixed with `allHsExplicit=True`. ChemIllusion's copy
   has the same bug.
2. **A proton transfer counted as one arrow.** `ugi_flower_kekule_v1` folds unmapped hydrogens into
   the diagonal, so the H–X bond is never an off-diagonal cell. The count now adds 2 per H gained
   or lost by a mapped heavy atom, so B: + H–A → B–H + A: is 2 arrows.
3. **The SMIRKS did not describe the step.** Arrows are measured on the SMIRKS while "same outcome"
   is judged on `resulting_state`. A validated top whose SMIRKS made a different tautomer from its
   own stated state was displaced. The code review then showed that an under-mapped SMIRKS (an
   unmapped leaving group, say) also loses electrons from the count and looks more parsimonious.
   A candidate is now measured only when both sides carry the same maps with every heavy atom
   mapped, its products are in `resulting_state` and its reactants in the current state, both up
   to resonance.

## 2. Scorer impact

Not applicable. The optional target-matching fallback was not added: 0 of 242 stored runs with
products reach a target only as another resonance form (`match_species`), so no case needs it.
No scorer code changed and no published number moves.

## 3. Live paired eval

Medium eval set `003aae8966184aa2a2874dbf3e0f510f` (`purpose=general`, not holdout). Ten fixed
cases, the first ten in `training_data/eval_tiers.json` medium order that are in the set:
`flower_254799 flower_077973 flower_184403 flower_156414 flower_102973 flower_122060
flower_138321 flower_058277 flower_103167 flower_175010`. Both arms used `threaded_unified`
(`PRIMARY_HARNESS`), `anthropic/claude-opus-5.5`, `--thinking-level low`, `--max-steps 14`,
`--max-runtime 10800`, products given, run in parallel from separate worktrees against the same
store. Answers came from the keyless headless responder (`claude -p --effort low`,
`scripts/bridge_headless_responder.py`), the same method as the threaded_unified evidence in
`AGENTS.md`. The OpenRouter key on this machine returned `401 User not found`; the two aborted
API attempts (run groups `resonance_bench_*`) made no billed calls. **API spend: $0.**
Not published.

| Arm | Eval run | quality_v2 mean | Pass | Product | Failed starts | Mean steps | LLM calls / case | Wall / case |
|---|---|---|---|---|---|---|---|---|
| before (`db53ba4`) | `c07ac137…` | 998.1 | 9/10 | 10/10 | 0 | 2.5 | 3.6 | 61.5 s |
| after (`fd5a88b`) | `263348b2…` | 992.1 | 9/10 | 10/10 | 0 | 2.9 | 4.3 | 73.8 s |

`arrow_count_preference_applied` events on the after arm: **0** (13 branch points). The before
arm had one `proton_transfer_preference_applied`, the after arm none.

Per case (quality_v2 points, steps):

| Case | Before | After | Steps before → after |
|---|---|---|---|
| flower_254799 | 1000.0 | 1000.0 | 3 → 3 |
| flower_077973 | 1000.0 | 1000.0 | 2 → 2 |
| flower_184403 | 1000.0 | 1000.0 | 3 → 3 |
| flower_156414 | 1000.0 | 1000.0 | 2 → 3 |
| flower_102973 | 1000.0 | 1000.0 | 2 → 2 |
| flower_122060 | 1000.0 | 1000.0 | 2 → 2 |
| flower_138321 | 1000.0 | 1000.0 | 3 → 3 |
| flower_058277 | 981.2 | 981.2 | 3 → 3 (fails on both: sequence 0.85) |
| flower_103167 | 1000.0 | 939.6 | 2 → 6 (sequence 1.0 → 0.52) |
| flower_175010 | 1000.0 | 1000.0 | 3 → 2 |

The −6.0 mean comes from one case, flower_103167. On the after arm the model proposed a 6-step
route, and every step was accepted at rank 1 with no arrow-count reorder. The difference is
run-to-run proposal variance, not this change, which did not alter any choice in either arm.

Counterfactual over the live runs (same script, no cutoff): 34 more steps with ≥2 validated
candidates. 13 had a measurable same-outcome alternative, which shows that live Opus 5.5 SMIRKS
are mostly fully mapped. None needed fewer arrows than rank 1. On this slice the preference is
inert, as intended: it reorders only when a measurably shorter equivalent exists.

## 4. Latency

| What | n | p50 | p95 |
|---|---|---|---|
| `resonance_key`, test fixtures (cold cache) | 1600 | 0.05 ms | 0.11 ms |
| `resonance_key`, real stored species (cold cache) | 355 | 0.10 ms | 0.26 ms |
| Preference per step, 81 real candidate sets (cold cache) | 81 | 1.4 ms | 2.7 ms |
| `step_arrow_count`, 400-atom fully mapped step (test) | — | — | < 50 ms (asserted) |

Well under the 50 ms per-step budget, and negligible next to an LLM call (seconds). Keys are
cached per SMILES string in a bounded cache (`resonance._KEY_CACHE_MAX`). The arrow count is
O(atoms + bonds): the first version built dense BE matrices and took about 190 ms on a 400-atom
step.
