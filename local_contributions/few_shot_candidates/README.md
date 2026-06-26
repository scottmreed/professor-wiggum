# Opus 4.8 few-shot candidates (keyless agent-bridge, 2026-06-24)

These JSONL lines are **candidate** few-shot examples produced by running the harness
through the **keyless agent-bridge** with **Claude Opus 4.8** standing in as the model
(`MECHANISTIC_AGENT_BRIDGE_DECLARED_MODEL=opus-4.8`, `responder_kind=orchestrator_subagents`,
`budget_observability=opaque`). They are staged here (gitignored) for review — they are
**not** yet added to `skills/mechanistic/<call>/few_shot.jsonl` because Track 1 requires a
medium-tier eval improvement before merge (see CONTRIBUTING.md).

Format matches the on-disk few-shot format: each line is `{"input": <str>, "output": <str>}`
where `output` is a serialized JSON string (verified to parse). 

## Provenance (runs in ../wiggum-data/data/mechanistic.db)
- `select_reaction_type` / `propose_mechanism_step` / `assess_initial_conditions`
  Diels-Alder lines: run `1dc20e26e74d4b2faf00fb558ec7183a` (**validated, completed**, 0 failed validations).
- Hydrobromination lines: run `020cab5cdd094fc887621a940a178aa5` (step-1 chemistry is correct/balanced;
  the run itself was blocked by the persistent-reagent per-step balance double-count — see the
  Track 5 failure submission). The per-step example content is still valid.

## Why these are useful
- `propose_mechanism_step` currently ships only one example (an SN2). These add a **concerted
  pericyclic** step (Diels-Alder, 3 cyclic arrows, no ionic intermediate) and a **Markovnikov
  carbocation-forming** step — two mechanism families not yet represented.
- `select_reaction_type` gets a clean positive (`rt_046`) and a real-chemistry `no_match`
  (pericyclic is absent from the rt_001-rt_080 taxonomy).
- `assess_initial_conditions` currently has **zero** examples; the hydrobromination acidic
  example would be the first.

## Recommended promotion path (per CONTRIBUTING.md Track 1)
1. Add chosen lines to `skills/mechanistic/<call_name>/few_shot.jsonl`.
2. `PYTHONPATH=. python scripts/validate_prompt_trace_evidence.py --call <call_name>`
3. `python -m pytest tests/fast/ -q`
4. `PYTHONPATH=. pytest tests/llm/test_eval_tiers.py --tier medium -k medium` and show the
   before/after medium-tier leaderboard delta.
5. Open a Track 1 PR (`templates/contributions/track1_few_shot_pr.md`) with the origin fields filled in.
