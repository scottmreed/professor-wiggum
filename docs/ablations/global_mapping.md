# Ablation: does global atom mapping matter?

Status: **prepared, not run.** Running it costs money, so the maintainer decides when. Record the result in `docs/PRD_jev_atom_identity_mechanistic.md` §8.2 before Milestone 2.

## Question

PRD §8.2: before anyone builds deterministic mapping candidates (§8.3–8.6), find out whether the LLM atom-mapping modules change mechanism outcomes. If turning them off does not hurt completion, the next step is to drop the module or keep improving its consumers, not to optimize the mapping itself.

## What the proposal LLM sees from global mapping (after `m0-global-mapping-consumers`)

| Channel | Before | Now |
| --- | --- | --- |
| `atom_mapping_summary.confidence` in proposal guidance | always `null`. The coordinator read the top level, but the tool nests confidence under `llm_response` | `llm_response.confidence` normalised to [0, 1], falling back to the top level for legacy outputs. If the module did not run, the summary is `{}` |
| `atom_mapping_summary.unmapped_atoms` | first 12 entries | unchanged |
| `mapped_starting_materials` / `mapped_products` in `propose_intermediates` | hard-coded `[]` | atom-mapped SMILES built from the `mapped_atoms` pairs by `core/global_mapping_context.render_global_mapping`. Each consistent pair gets a map number. A pair is dropped if its species is unknown, its index is out of range, its elements do not match, or it reuses an atom |
| `mapped_current_state` | hard-coded `[]` | same as the mapped starting materials while `current_state` is still the starting materials (step 0), otherwise `[]`. Mapping the loop state after step 0 is outside this change |

`propose_intermediates` only adds its existing "Optional atom-mapped context" prompt section when at least one mapped list is non-empty. No new prompt text was added.

## Arms

| Arm | Harness | Run group (easy) | Difference |
| --- | --- | --- | --- |
| A (reference) | `default` | `easy_ablation_mapping_default` | none |
| B | `no_mapping` | `easy_ablation_mapping_off` | `atom_mapping` and `step_atom_mapping` have `enabled: false`. Everything else matches `default` v5 |

This uses `no_mapping` and not `no_mapping_no_reagents` for two reasons. The older harness also disables `missing_reagents` and keeps `step_atom_mapping` on. It also predates `run_config_defaults`, so it runs with a different `proceed_on_validation_failure`. Any delta from it would mix several changes.

## Commands

```bash
source .venv/bin/activate
scripts/ablate_global_mapping.sh plan                     # print exact commands, no model calls
MODEL=<model> THINKING=<level> scripts/ablate_global_mapping.sh run     # PAID: 2 x MAX_CASES cases
scripts/ablate_global_mapping.sh report                   # side-by-side readout from data/mechanistic.db
```

`plan` prints these commands, shown here with the defaults `TIER=easy` and `MAX_CASES=25`:

```bash
python main.py eval --eval-set-id ignored --tier easy --leaderboard-route custom --allow-repeats \
  --max-cases 25 --max-steps 10 --max-runtime 1200 [--model M --thinking-level T] \
  --harness default --run-group easy_ablation_mapping_default

python main.py eval --eval-set-id ignored --tier easy --leaderboard-route custom --allow-repeats \
  --max-cases 25 --max-steps 10 --max-runtime 1200 [--model M --thinking-level T] \
  --harness no_mapping --run-group easy_ablation_mapping_off
```

Both arms use `--leaderboard-route custom --allow-repeats`, so each takes the same first N case ids of the tier. That makes the comparison paired. Use the same `MODEL`/`THINKING` for both arms. Set `TIER=medium` once that tier is populated.

To read the results:

```bash
sqlite3 data/mechanistic.db "select id, run_group_name, eval_set_id from eval_runs \
  where run_group_name in ('easy_ablation_mapping_default','easy_ablation_mapping_off') order by created_at;"
python main.py compare-eval-runs --run-a <eval_run_id_A> --run-b <eval_run_id_B>   # same case_ids_hash / model / prompt hashes?
python main.py leaderboard --eval-set-id <eval_set_id> --limit 20
```

## Metrics to compare (B minus A)

| Metric | Source |
| --- | --- |
| Completion (run status `completed`) and product reached | `report`; the "Product Accuracy" line of each arm's Score Summary in `local_contributions/runs/ablation_global_mapping/<group>.log` |
| Pass rate, mean score, paired per-case deltas | `report` |
| Retries (`mechanism_retry_started`) and backtracks (`backtrack`) | `report` (from `run_events`) |
| Pathway score (`known_alignment_component`) | "Pathway Coverage" in each arm's Score Summary |
| Cost and latency | `report` (`total_cost`, `mean_latency_s`) |
| `atom_mapping_calls` / `step_mapping_calls` | `report`. Must be 0 in arm B, which confirms the ablation took effect |
| `mapping_agreement` (predicted vs ground-truth mapping) | `report` column. It stays empty until that metric lands (PRD Phase 0) |

**Scoring caveat.** Self-reported step-mapping confidence makes up 20% of per-step validity, and an absent mapping scores 0.5 (`scoring.py`, PRD §0). Because of that, arm B's `score` and `pass` move for metric reasons alone. Judge the ablation on product reached, completion, pathway, retries and backtracks first. Treat score and pass deltas as secondary.

## Known gap

`MappingAgent._validate_mapping` calls `rdkit-agent atom-map` with `{"action": "check", ...}`. The installed CLI rejects that ("No sub-command provided"), and its `check` sub-command takes a SMIRKS, not index pairs. The tool treats the failure as "unavailable" and skips validation, so the documented 0.3 confidence cap after a failed check never fires. The confidence that now reaches the proposal prompt has therefore not been checked.
