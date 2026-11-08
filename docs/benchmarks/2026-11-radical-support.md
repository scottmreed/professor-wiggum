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
`/private/tmp/wiggum-radical-training.json`. These are train-derived examples,
not held-out scores. RC/RS are not pathway IDs and no arrows are fabricated.

Verification: `PYTHONPATH=. python -m pytest tests/fast/ -q` completed with
**1,327 passed, 26 skipped**. New tests cover homolysis, recombination, H-atom
transfer, radical addition, closed-system single-electron transfer, invalid
sources/counts/mappings, and complete methane chlorination chain replay.
The chain preserves persistent atom identity. `git diff --check` passed.

This is authored deterministic contract evidence, not evidence of LLM
prediction quality. No benchmark score or prompt approval was fabricated.
`validate_prompt_trace_evidence.py --call propose_mechanism_step --call
baseline_mechanism` fails as expected: no approved traces match the new prompt
bundles (baseline has none). Before merge, the existing change-evidence policy
requires approved linked traces, medium-tier improvement, and human core review
of deterministic validators. No tier definitions or evidence gates were relaxed.

ChemIllusion integration is prepared separately in
`/private/tmp/chemillusion-radicals`, branch `codex/radical-mechanism-consumer`:
radical/fishhook SVG support, animation, replay source fields, explicit rejection
of radical conversion into closed-shell Lab/3D. Its runtime pin is unchanged.
A pin bump must reference a reviewed Wiggum commit after the evidence gate passes.

Scope limits: main-group organic electron bookkeeping; no spin/multiplicity or
excited-state feasibility model, no transition-metal photoredox validation.
Transferred hydrogen is explicit and mapped. Radical execution uses validated
mapped states; an arrow-only radical input without its mapped reaction is refused.
