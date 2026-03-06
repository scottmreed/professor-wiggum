# Soul of the Mechanistic Agent

**The Mechanistic Agent is an evolutionary AI system for organic reaction mechanism prediction that learns and improves through evidence-gated prompt evolution, combining LLM creativity with deterministic chemistry validation.**

## What This Project Is Trying to Become

This is a chemistry reasoning system in the form of an agent harness that can evolve to get better with use.

The long-term goal is **zero-human-input evolution**: a system where runs generate evidence, evidence gates prompt and module changes, and improved prompts and modules produce better runs — all with minimal oversight. Every guardrail in this codebase exists to make that loop safe, auditable, and reversible.

This document describes the philosophy behind the design choices, so that contributors understand not just what the rules are but why they exist — and so future automated agents have enough context to extend the system without breaking it.

---

## How the System Learns

### The feedback loop for agent evolution

```
User runs a test set of reactions
    ↓
Run produces traces and traces are evaluated
    ↓
User (or automated judge) approves good traces
    ↓
Approved traces become fewshot examples for later work
    ↓
Evidence files gate prompt changes (few-shot examples, base instructions)
    ↓
Better prompts produce better runs
    ↓
(repeat)
```

Every link in this chain is deterministic except the LLM-backed subagent calls themselves. The evidence file records which subagent version was in effect. The gate script checks that the current pull request matches the current version. This means:

- You cannot claim that a trace validates a prompt you have not actually tested.
- You cannot merge a prompt change without evidence that the new prompt produces better outputs.
- Every generation of the system is fully reproducible from its git history.

### What counts as "better"

The system uses concrete, measurable measures of improvement:

1. **The mechanism loop completed** (target products reached) — binary.
2. **All subagent steps passed deterministic validation** (bond balance, atom balance, state progress) — per-step binary.
3. **The model used fewer retries / backtrack steps** — numeric, tracked in the trace.
4. **The leaderboard score improved** (for eval runs) — numeric, tracked in SQLite, including per-subagent quality and pass-rate scores.

These measures apply regardless of the coordination topology used for the proposal step. A topology that produces more completions or fewer retries on the eval tiers is objectively better by these criteria.

A prompt change is "better" if it produces more completions, fewer retries, or better leaderboard scores on the eval tiers — not if it feels more elegant. Improvement on a single reaction is not sufficient; the eval tiers (easy/medium/hard) are the arbiter. The per-subagent leaderboard breakdown makes it possible to identify *which* subagent drove an improvement or regression.

### The eval set

The eval set is a 100-reaction benchmark derived from the PMechDB dataset (Tavakoli et al., *J. Chem. Inf. Model.* 2024, 64, 1975–1983; DOI: [10.1021/acs.jcim.3c01810](https://doi.org/10.1021/acs.jcim.3c01810)). It spans 33 mechanistic classes with 2–19 elementary steps each.

From this set, `eval_tiers.json` defines three difficulty tiers of 10 reactions each:

- **Easy** (2–4 steps): SN1, SN2, nucleophilic attack, carbonyl reduction
- **Medium** (5–8 steps): Friedel-Crafts, Wittig, SNAr, Jones oxidation, Boc deprotection
- **Hard** (9–19 steps): Mannich, Mitsunobu, DCC condensation, Ing-Manske, aldol condensation

PR approval gates on tier performance (see [CONTRIBUTING.md](CONTRIBUTING.md) for the specific requirements per contribution type). The tier definitions themselves are updatable by contributors following the same PR process as subagent additions.

---

## The Guardrails

### Guardrail 1: Deterministic chemistry is the final arbiter

Every mechanism step the LLM proposes is validated by RDKit before it is accepted. The validation chain is:

- Bond/electron balance (stoichiometry correct?)
- Atom balance (no atoms created or destroyed?)
- State progress (is the system actually moving toward the target products?)

These checks are **not configurable by prompts or model outputs**. They are ground truth. If an LLM-backed subagent output fails these checks, it is rejected regardless of how confident the model was. This is what makes the retry/backtrack loop safe — the system cannot accept a chemically wrong step.

### Guardrail 2: Prompt changes require trace evidence

A prompt change that is not backed by an approved trace is a guess. Guesses degrade the system over time because they have no accountability. The evidence gate enforces:

- Every changed subagent call has at least one approved evidence trace.
- The evidence SHA matches the current prompt SHA.
- The evidence file is linked to a specific model version (so you know which model produced it).

The gate is run locally before PR and can be run in CI without any API keys. There is no exception path. If a prompt change has no evidence, it does not merge.

### Guardrail 3: The model catalog is the only source of model truth

The `model_pricing.json` catalog is the single source of truth for which models exist, what they cost, and whether they support tool calling. No code path should construct a model identifier that is not in the catalog. This means:

- When a new model is added, it gets a catalog entry with explicit `supports_tools` and pricing.
- The `adapter_supports_forced_tools()` function reads from the catalog, not from hardcoded lists.
- The leaderboard tracks `model_version_id` from the catalog, so cost-adjusted performance comparisons are always valid.

### Guardrail 4: The fast test suite is the merge gate

`make test` runs the full fast suite in under 60 seconds with no API keys and no network. Every PR must pass this suite. The fast suite is the primary quality gate for automated merges. The LLM test suite (`make test-llm`) is for human review of new model behavior, not for automated gates.

### Guardrail 5: No silent degradation

The system is designed to fail loudly rather than silently degrade. Specifically:

- If a mechanism step fails validation, it is recorded as a failed path in the trace, not silently dropped.
- If all candidates for a step fail, the run pauses rather than inventing a step.
- If the evidence gate fails, the validation script exits with a non-zero code and a human-readable error message.
- If a model is removed from a provider, the trace for any run that used it still references the original `model_version_id`, so old traces remain interpretable.

---

## The Evolution Model

### Stage 0 — Human in the loop (current)

A human runs reactions, reviews traces, approves them, and opens PRs with evidence. The system records everything, but a person decides what to merge.

### Stage 1 — Automated approval with human review

An LLM judge (already scaffolded in `evaluate_run_judge`) scores runs against the eval set. Traces that score above a threshold are auto-approved. A human still reviews PRs, but they are reviewing a diff + leaderboard delta, not individual traces.

### Stage 2 — Automated PR generation

The system automatically generates PR candidates: "I found 3 atom mapping traces with score > 0.9 that differ from the current few-shot examples. Here is the proposed few-shot addition." A human reviews the proposed diff + evidence, clicks merge.

### Stage 3 — Automated merge with evidence gate

PRs that pass the evidence gate, the fast suite, and show a positive leaderboard delta merge automatically. A human can veto within a review window (default: 24 hours). No human action is required for the merge to proceed.

### Stage 4 — Fully autonomous evolution (target)

The system runs the eval set on a schedule, generates evidence, proposes changes, verifies them against the gate, and merges — all without human involvement. Human users can still veto any change and the audit trail is always available.

The guardrails described above are designed to make Stage 4 safe:

- Deterministic chemistry validation means the system cannot accept a wrong answer.
- Evidence gates mean the system cannot accept a change it has not tested.
- The model catalog means pricing and capability claims are always auditable.
- The fast test suite means changes that break the runtime are caught before merge.

---

## What Good Evolution Looks Like

A system that is evolving well shows these patterns in its git history:

- Few-shot examples accumulate in proportion to the variety of reaction classes being run.
- Leaderboard scores trend upward over time (more completions, fewer retries).
- The eval set grows as new reaction classes are added.
- Deprecated models stay in the catalog (for trace replay) but stop appearing in new runs.
- Prompt base files stay short — few-shot examples carry the task-specific knowledge, not the instruction text.
- Harness configurations are tested via dry runs and eval tiers before being proposed as PRs. The built-in reference point is `harness_versions/default.json`; changes to the pipeline (adding, removing, or reordering modules, or changing topology profiles) are versioned as saved harness variants in `harness_versions/` and gated on the same eval tiers as other contributions.
- Coordination topology experiments (SAS, centralized MAS, independent MAS, decentralized MAS) appear as leaderboard comparisons, not as guesses. If a topology produces better results on the eval tiers, it becomes the new default via the normal harness PR process.

A system that is evolving badly shows these patterns:

- Prompt base files grow longer and longer with special-case instructions.
- Evidence files exist but the leaderboard score is flat or declining.
- Models are removed from the catalog (breaking old traces).
- The fast test suite has large numbers of skipped or `xfail` tests.

If you see the second set of patterns, the right response is not to add more guardrails — it is to ask why the feedback loop is not producing good signal. Usually the answer is that the eval set is too small, or the judge scoring is miscalibrated.

---

## Design Principles

**Trust determinism over confidence.** A step that passes RDKit validation is correct by the definition we have agreed on. A step that the LLM says it is "highly confident" about but fails validation is wrong. The model's confidence score is informative for ranking candidates; it is not a substitute for the chemistry check.

**Evidence is not optional.** Any prompt change without evidence is a hypothesis, not an improvement. Hypotheses are fine to explore locally. They do not belong in main.

**Users are part of the curriculum, not just observers.** Single-reaction submissions, trace review, checkpoint inspection, and contribution PRs are all first-class inputs to the system's learning loop. A user who submits one failure case, reviews one trace, or links one checkpoint in a PR is helping define what the system should improve next. The curriculum should make that contribution path visible and easy to follow.

**Reversibility is a requirement.** Every change must be reversible: prompts can be rolled back (git), models can be deprecated (catalog flag), traces always reference the model and prompt SHA that produced them. If a change breaks the leaderboard, the revert is one `git revert` away.

**The eval set is the memory of what we care about.** The reactions in `training_data/eval_set.json` (100 reactions from PMechDB) and the tiered subset in `training_data/eval_tiers.json` (10 easy + 10 medium + 10 hard) define what "correct chemistry" means for this system. If you want the system to get better at a new reaction class, add that class to the eval set first. Otherwise the leaderboard cannot measure whether you succeeded. PR approval gates on eval tier performance, not individual reaction results.

**Structured outputs come first.** Tool schemas may include optional free-form commentary fields, but validation and leaderboard outcomes are driven by the structured payload and deterministic chemistry checks. Human reviewers can inspect supplemental commentary in traces when it is useful for debugging or prompt iteration.

**Small, composable changes win.** A PR that adds one few-shot example with one evidence trace and shows a 2% leaderboard improvement is more valuable than a PR that rewrites the base prompt and shows a 5% improvement, because the small PR is fully auditable and the large one is not. The system is designed to reward the small, incremental, evidence-backed improvement.

**Coordination topology is a harness-level knob, not a prompt change.** The `coordination_topology` field controls how the proposal step is orchestrated — single agent (SAS), centralized multi-candidate (default), independent parallel agents, or decentralized agents with peer debate. Changing this field changes how many LLM calls are made and how their outputs are merged, but it does not change the prompt content, the few-shot examples, or the validation chain. Topology experiments follow the Track 4 (harness change) contribution path and are gated on eval tiers like any other structural change.

**Contribution paths should stay legible.** A healthy project makes it obvious how someone can help: submit a single reaction, improve a trainee lane, add a few-shot example, patch the harness, or contribute a new model-specific override set. The curriculum, README, and history views should show where the system learned and how a person can extend that learning without needing to reverse-engineer the repo.
