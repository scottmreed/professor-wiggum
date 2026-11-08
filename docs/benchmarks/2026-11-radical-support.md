# Radical support: local implementation evidence

Date: 2026-11-07. Branch: `codex/radical-reactions`.

Implemented explicit `rad:`, `et:` and bond `:e1` notation; strict main-group
mapped electron-inventory validation; source capacity and state multiplicity
checks; radical-aware mapped replay preserving atom IDs; shared quality checks;
radical focus reporting; conservative abstention from pair-arrow preference;
and proposal/baseline instructions. Old pair notation is unchanged.

Data: local FlowER `flower_new_dataset/train.txt` contains 4,606 RC/RS rows,
4,400 unique labelled steps after 206 exact duplicates. All parse and contain
radical electrons. Extraction hashes the entire train file, never reads test or
validation, and retains source labels/line references. The 24-case development
sample is in `training_data/radical_development/`. Full local extraction:
a local file outside the repo (not committed). These are train-derived examples,
not held-out scores. RC/RS are not pathway IDs and no arrows are fabricated.

Verification: `PYTHONPATH=. python -m pytest tests/fast/ -q` completed with
**1,327 passed, 26 skipped**. New tests cover homolysis, recombination, H-atom
transfer, radical addition, closed-system single-electron transfer, invalid
sources/counts/mappings, and complete methane chlorination chain replay.
The chain preserves persistent atom identity. `git diff --check` passed.

This is authored deterministic contract evidence, not evidence of LLM
prediction quality. No benchmark score or prompt approval was fabricated.

## Release gates (2026-11-07)

**Code review.** Two fresh-context reviews found regressions in 8e1a536 that
broke the "pair arrows unchanged" invariant. They are fixed test-first in
7320eee and ba37d49 (`tests/fast/test_radical_regressions.py`):

- H-less SMIRKS atoms (`[C:1]`) were read as radicals and sent polar steps to
  strict radical validation. So did an untouched radical spectator (TEMPO, O2).
  The radical path now runs only for an explicit one-electron move or a radical
  species that the step changes.
- A malformed pair push raised inside the candidate model and dropped every
  candidate in the response. Pair pushes are skipped as on b0e6ee8 again. A bad
  fishhook rejects only its own candidate.
- Bond sources aimed at their own other endpoint, reversed `et` pairs and empty
  moves cancelled out and passed. They now fail.
- Radical multiplicity checks needed the whole state and lost duplicates
  (`[Cl].[Cl]` deduped). The SMIRKS sides are now sub-multisets with unchanged
  spectators, and radical candidates are not deduped.
- Metal and H atoms no longer count as radicals in scoring. Push/metadata
  agreement ignores bond order. The extraction command reproduces the
  committed `cases.json` byte for byte.

Fast suite at ba37d49: **1,395 passed, 22 skipped** (b0e6ee8 on the same
machine runs the pre-existing suite; the delta is the new radical tests).

**Medium-tier paired eval** (not published). Medium set `003aae89…` (not
holdout), the 10 fixed cases used for PR #101, `threaded_unified`,
`anthropic/claude-opus-5.5`, thinking low, products given, max-steps 14,
keyless headless responder (`claude -p --effort low`,
`responder_saw_ground_truth=false`, `official_holdout_exposed_to_agent=false`).

| Arm | Eval run | quality_v2 | Pass | LLM calls / case |
|---|---|---|---|---|
| before b0e6ee8 | `43551fcfc1f84a22bf6c160bda5344e2` | 995.4 | 9/10 | 3.7 |
| after 7320eee | `0a547647c3de46de9860a935a350caeb` | 1000.0 | 10/10 | 3.8 |

The +4.6 comes from one case: flower_102973 took a 5-step route on the before
arm (954.2) and 3 steps after (1000.0). The other nine score 1000 on both arms.
These polar cases never enter the radical path, so read this as **no regression,
with the gain inside proposal variance**, not as a radical-driven improvement.
ba37d49 changes only the radical path and was not re-run. The medium tier
carries no radical cases; radical capability is covered by the deterministic
contract tests above, not by a live score.

**Prompt evidence.** Ten approved `propose_mechanism_step` traces (one per
case, eval run `0a547647…`) are exported under
`traces/evidence/propose_mechanism_step/9ae9ac5b98b8…/`;
`validate_prompt_trace_evidence.py --base-ref origin/main` passes.
`baseline_mechanism` is not an evidence-gated call (`gated_call_names`).

**Still required:** human core review of the validator changes (CODEOWNERS).
The ChemIllusion runtime pin moves through the `chemillusion-runtime-bump`
workflow after merge (this is a harness-engine change), not by hand.

ChemIllusion integration is prepared separately on ChemIllusion branch
`codex/radical-mechanism-consumer`:
radical/fishhook SVG support, animation, replay source fields, explicit rejection
of radical conversion into closed-shell Lab/3D. Its runtime pin is unchanged.
A pin bump must reference a reviewed Wiggum commit after the evidence gate passes.

Scope limits: main-group organic electron bookkeeping; no spin/multiplicity or
excited-state feasibility model, no transition-metal photoredox validation.
Transferred hydrogen is explicit and mapped. Radical execution uses validated
mapped states; an arrow-only radical input without its mapped reaction is refused.
