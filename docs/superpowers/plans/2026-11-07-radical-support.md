# Radical reaction support implementation plan

Approved by the user on 2026-11-07 for local FlowER testing and eventual ChemIllusion integration.

**Goal:** preserve and validate one-electron mechanisms end to end, expose the existing radical training subset, and prepare ChemIllusion consumption without claiming deployment or prediction-quality improvement.

**Architecture:** retain existing pair-arrow syntax and add explicit one-electron sources and transfers. Validate radical steps against mapped before/after electron inventories, rather than accepting a zero-sum annotation alone. Keep radical step data separate from numeric FlowER multistep curriculum and official holdout. Runtime traces carry the electron count to consumers.

**Execution:** use subagent-driven development for the core runtime unit, with independent integration inspection and root-owned dataset/test work. Review implementation against the requirements, then review code quality and run regressions.

- [x] Core: extend mechanism_moves parser, normalization, serialization, schemas and deterministic validation; fail closed on invalid electron counts and contradictory annotations. Cover homolysis, recombination, H abstraction, addition and explicit electron transfer, including negative tests and pair-arrow regressions.
- [x] Data: train-only RC/RS extraction with stable content IDs, source hashes and no guessed pathway grouping. Deterministic offline benchmark with curated elementary steps and a complete initiation/propagation/termination chain; no LLM-performance claims from replays.
- [x] Integration: audit scoring, mapped-state replay, arrow preferences and rendering; extend the product consumer contract where needed in an isolated ChemIllusion checkout.
- [x] Guidance: document radical notation and update proposal guidance to cover radical sources, conditions and closed-system electron accounting. Preserve prompt trace/medium-tier merge gates.
- [x] Verification: focused tests, full fast suite, evidence gate for changed prompts, consumer tests, diff review. Report required predictive evaluations and release gates honestly.

Acceptance: one-electron counts survive round trips; incorrect arrows fail even when totals cancel; ordinary pair reactions remain compatible; radical benchmark runs offline; training extraction never reads holdout; ChemIllusion can preserve and display new notation. Deployment and runtime pin promotion follow a later evidence-backed release.

Execution evidence and remaining release gates: `docs/benchmarks/2026-11-radical-support.md`. Verification includes recording the failed prompt evidence gate, not declaring merge readiness.
