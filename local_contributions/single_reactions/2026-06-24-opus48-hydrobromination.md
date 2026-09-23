## Track 5: Single Reaction Submission

Save this file under `local_contributions/single_reactions/`. Do not commit it.

### Classification
- Outcome: `failure` (run did not complete; correct chemistry rejected by bookkeeping)
- Why it matters: `harness_gap` (also `few_shot_candidate` for the per-step chemistry)

### Reaction
- Name: Markovnikov hydrobromination of isobutylene (2-methylpropene + HBr -> tert-butyl bromide)
- Starting materials: `CC(C)=C` (a.k.a. `C=C(C)C`), `Br` (HBr)
- Expected products: `CC(C)(C)Br`
- Observed result: Step 1 (Markovnikov protonation to the tertiary `C[C+](C)C` carbocation + `[Br-]`) is **chemically balanced** (C4H8 + HBr -> C4H9+ + Br-) but was rejected by `atom_balance_validation` and the run entered a re-analysis loop instead of advancing to step 2.
- Reaction class, if known: rt_046 Hydrobromination of alkene (electrophilic_addition, steps~=2)

### Runtime details
- Run ID: `020cab5cdd094fc887621a940a178aa5` (status left `running`; stopped manually)
- Trace IDs: step_outputs for run above; see `atom_balance_validation` -> `balanced: false`
- Model: `agent-bridge` (keyless); responder = Claude Opus 4.8
- Thinking level: none (bridge)
- Harness: `default`
- Verified or unverified mode: `unverified`

### What happened
- Short narrative: After Opus 4.8 proposed the correct step-1 protonation with `resulting_state = ["C[C+](C)C", "[Br-]"]`, the harness carried the reagent **HBr forward into the resulting state** (state became `['C[C+](C)C', '[Br-]', 'Br']`). The atom-balance validator then compared step reactants (`C=C(C)C` + `HBr`) against that state and reported a phantom deficit of `Br:1, H:1` — i.e. it double-counted the persistent reagent on the product side without a matching reactant-side copy. The same pattern appeared with `[Cl-]` in a Finkelstein smoke run (`208f5f31c30b43a1a00554699e5f63fd`).
- Why you believe this case deserves review: A *correct, balanced elementary step* is being rejected purely by persistent-species bookkeeping, which blocks every keyless multi-step run whose mechanism uses a reagent the species registry marks persistent (acids, counterions). This is a harness gap, not a model error.
- What change it might justify: When a species is tagged persistent/spectator and is appended to a step's resulting_state, the atom-balance validator should account for it on **both** sides (or exclude it from the per-step balance), so a balanced elementary step is not rejected. A fast regression test should cover "persistent reagent present in resulting_state must not create a phantom per-step deficit."

### Supporting artifacts
- Local trace path: `../wiggum-data/data/mechanistic.db` (run_ids above)
- Few-shot candidate (per-step chemistry is still valid): `local_contributions/few_shot_candidates/`

### Suggested follow-up
- [x] Consider as future few-shot candidate (step-1 carbocation formation is a good example)
- [ ] Consider as future subagent change
- [x] Consider as future harness change (persistent-species per-step balance double-count)
- [ ] Consider as future model comparison case

### Execution origin
- Execution channel: `agent-bridge` (keyless)
- Responder kind: `orchestrator_subagents`
- Declared underlying model/agent: `opus-4.8` (Claude Opus 4.8, Claude Code session)
- Budget observability: `opaque`
- Official holdout exposed to the responder? `no`

### Contact info for possible manuscript attribution
- Name: (Claude Opus 4.8 via Claude Code; maintainer to fill in)
- Email: scott.reed@ucdenver.edu
- ORCID / GitHub / preferred handle: scottmreed
- Contact before attribution? `yes`
