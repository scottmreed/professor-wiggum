# Jev reaction-type calibration — 2026-09-23

Shadow-mode calibration of the Jev `Choice` over the 86-template taxonomy + `no_match` (PRD §7.3, §19 Phase D), run with `scripts/calibrate_jev_reaction_type.py --live --limit 0` against the 72 curated `example_mappings` labels. No database writes; the `example_id` bypass is irrelevant here because the script calls Jev directly.

- Model version returned: `typesafe/jev-1.13-20260917` (catalog id `typesafe/jev-1.13`, OpenRouter Decisions API).
- Cases: 72. Top-1 accuracy: **0.917**. ECE: **0.056**. Brier (top label): 0.067. Multiclass Brier: 0.148. Failures: 0.
- LLM baseline on the same ids: none available in the local DB (n=0). A paired comparison needs `default` vs `jev_reaction_type` eval runs with `MECHANISTIC_EXAMPLE_REACTION_TYPE_BYPASS=0`.

| confidence band | n | accuracy | mean confidence |
| --- | --- | --- | --- |
| [0.00, 0.50) | 2 | 0.500 | 0.447 |
| [0.50, 0.65) | 8 | 0.625 | 0.552 |
| [0.65, 0.80) | 6 | 1.000 | 0.714 |
| [0.80, 0.90) | 12 | 0.917 | 0.846 |
| [0.90, 1.00] | 44 | 0.977 | 0.967 |

Above the existing 0.65 guidance threshold: 62 cases, accuracy 0.968. Below it: 10 cases, accuracy 0.600. Calibration is close to the diagonal in every band (no systematic overconfidence on this task, unlike the published out-of-distribution results), so the existing `reaction_template_confidence_threshold` of 0.65 transfers as `jev.thresholds.reaction_type_active_probability` for the `jev_reaction_type` harness variant. The two misses at ≥0.90 are `rxn_0301` (label `rt_040`, chose `rt_079` at 0.50 — below threshold, so guidance would be disabled) and one other; see the JSON for per-case rows.

Raw output (gitignored, maintainer machine): `local_contributions/runs/jev_reaction_type_calibration.json`.

Caveats: 72 cases is small; labels come from the same curated set that seeds the LLM's example lookup; per-case token counts were ~2–4k so the whole run cost well under one cent. Recalibrate on every Jev version change (PRD §7.0).
