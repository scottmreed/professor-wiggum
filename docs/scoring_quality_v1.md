# Mechanism quality rubric (`quality_v1`)

One 1000-point score for every mechanism: harness runs and harness-free baselines, API or agent bridge. It replaces both earlier published numbers, which are now labelled **legacy**:
- the Clawdiators-style harness rubric (product 300, pathway 300, pushes 200, speed 100, methodology 100);
- the baseline "mean case score × 1000".

Implementation: `mechanistic_agent/quality_scoring.py`. Tests: `tests/fast/test_quality_scoring.py`.

## Principle

A mechanism is scored from its accepted path alone: each step's `current_state`, `resulting_state`, `reaction_smirks` and `electron_pushes`. The rubric does not trust anything the harness did internally (retries, its own validation verdicts, rescue calls). Every step is re-checked by the same deterministic code, so a one-shot baseline answer and a validated harness path are measured the same way.

The rubric deliberately rewards following mechanism rules. A harness that checks every step should therefore score higher than a one-shot answer that skips or fudges steps. That is the intended advantage, and it is earned step by step.

## Components

| Component | Points | Per mechanism | Notes |
|---|---|---|---|
| `step_validity` | 250 | mean over steps of four checks: **atom and charge balance**; **arrows** parse into bond/electron deltas (the harness's own `bond_electron_validation`); the **SMIRKS sides match** the stated species; the **state changes** | Whole extra equivalents of a starting material, such as a second TFA drawn from the solvent, are accepted as excess reagent, exactly as in the harness validator |
| `sequence` | 200 | v3 alignment to the FlowER reference: `max(exact, proton-agnostic)` | Choosing a different proton shuttle is not penalized here |
| `electron_conservation` | 100 | fraction of steps whose mapped SMIRKS conserves electrons in the bond-electron matrix (`bond_electron_view.v1`) | |
| `proton_bookkeeping` | 100 | fraction of steps with no bare `[H+]`, × 0.5 if the net protons do not close | |
| `protonation_states` | 100 | fraction of steps with no free species implausible for the conditions | See [Conditions](#conditions) |
| `reagents_and_solvent` | 100 | 0.6 × fraction of steps whose new species were all supplied, + 0.4 × mass/charge closure | Supplied means a starting material or its conjugate, or water/H3O+/OH- when water is present. Closure: exact 1.0, reconciled 0.9, approximate 0.3 (`mechanism_audit`) |
| `efficiency` | 100 | 1 − 0.5 × each repeated or undone state − 0.2 × each heavy-atom step beyond the reference | Extra proton-transfer steps are not "excess" |
| `intermolecular` | 50 | fraction of proton transfers that use a shuttle, or are intramolecular only because no shuttle was available | |

When a case has no reference path, `sequence` is dropped and the other components are rescaled to 1000.

**Product is a gate, not points, when it is given.** The prompt supplies the target products, so reaching them earns nothing. A mechanism that misses a target scores half.

**No-product runs** (`eval/baseline --hide-products`) are different, because the model has to predict the product:
- The main product, in any protonation state, earns **300 points**. FlowER byproducts are not required, and the main product is never a starting material or its conjugate.
- The eight components share the other 700 in their usual proportions.

To compare runs across modes, every result reports:
- `mechanism_points`: the eight components on 1000, before any product gate or product points;
- `product_correct`.

The leaderboard shows these as the **Mechanism** and **Product** columns.

**Pass rule.** A case passes when all of these hold:
- every target is reached;
- every step is valid;
- closure is exact or reconciled;
- no state repeats;
- the total is at least 700.

## Conditions

FlowER cases carry no pH or conditions, so the rubric reads them from the starting materials.

| Class | When |
|---|---|
| acidic | carboxylic, sulfonic or mineral acids, or hydronium are present |
| basic | hydroxide, alkoxide, hydride, amide anions, carbonate, aliphatic amines or pyridines are present |
| buffered | both an acid and a base are present |
| neutral | neither is present |

The protonation rules flag only pKa extremes in free, net-charged species:
- **acidic:** hydroxide, alkoxide, amide anion or carbanion;
- **basic:** hydronium, oxonium or a protonated carbonyl;
- **buffered and neutral:** nothing is flagged.

Zwitterions, ammonium ions, carboxylates and halides are never flagged.

The FlowER references themselves score 950–1000. The rubric catches a few real oddities in them: a free amide anion under TFA (flower_151603), and hydroxide in acetic acid (flower_025913).

## Thinking parity

Each record carries `thinking_level`. Harness and baseline rows are compared at the same level.
- **Bridge runs:**
  - pass `--thinking-level` to both `eval` and `baseline`;
  - answer with the matching agent definition, `.claude/agents/bridge-model-<level>.md` (its `effort` frontmatter);
  - or use the headless responder with `RESPONDER_EFFORT=<level>`.
- **API runs:** pass the same `--thinking-level` to both.

## Commands

```bash
python main.py eval ... --thinking-level low                  # prints the quality_v1 summary; stores summary.quality per case
python main.py baseline ... --thinking-level low              # same, and stores baseline_steps so the run can be re-scored
python main.py rescore-quality --eval-run-id <id> [--apply]   # score stored runs; legacy baselines (no steps) are reported as legacy
python main.py publish-results --eval-run-id <id>             # records carry scoring: quality_v1, or legacy
```

## Legacy data

Nothing is deleted.
- **Harness runs:** these keep snapshots, so they can be re-scored and re-published under `quality_v1`. Their old rubric moves to the record's `legacy` block.
- **Baselines run before `baseline_steps` was stored:** these cannot be re-checked. They stay `scoring: legacy` and appear under **Legacy scores** on the leaderboard.
- **The DB case score:** `eval_run_results.score` (v1/v2/v3) is unchanged. It still drives the internal dev leaderboard and curriculum gating.
