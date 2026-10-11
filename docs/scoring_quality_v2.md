# Mechanism quality rubric `quality_v2`

`quality_v2` is the published score for every mechanism, harness or harness-free baseline. It
builds on [`quality_v1`](scoring_quality_v1.md), which re-checks every step, and adds the check
`quality_v1` lacked: **does each step start where the previous one ended?** A one-shot answer can
list balanced steps that never connect, for example an intermediate that vanishes or a product
that appears from nowhere, and still end on the right product. Under `quality_v1` such an answer
scored well. Under `quality_v2` it does not.

Code: `mechanistic_agent/quality_v2.py` (scoring) and `mechanistic_agent/quality_radar.py` (the
radar SVGs). `quality_scoring.score_snapshot_quality` attaches the case result as
`quality["quality_v2"]` next to the `quality_v1` fields.

## Six axes, each from 0 to 1

| Axis | Per case |
|---|---|
| Product correct | 1 when the `quality_v1` product gate is met. With the product given, every target is reached. With it hidden, the main product is reached in any protonation state. |
| Step connectivity | Fraction of steps whose start follows from what came before (see below). |
| Balanced, progressing | Mean over steps: atom and charge balanced **and** the state changes. |
| Arrows + SMIRKS | Mean over steps: arrows parse into a bond/electron change **and** the SMIRKS sides match the stated species. |
| Electron conservation | Mean over steps: the mapped SMIRKS conserves electrons in the bond-electron matrix. |
| Matches reference | Heavy-atom events in the FlowER reference order (`quality_v1` sequence ratio, proton-shuttle agnostic). |

The four step axes are averaged over the steps the mechanism actually made. A harness run that
makes five good steps and stops is not penalized on them. It just misses the product.

## Score

```
score = 1000 × (0.2 · product + 0.3 · connectivity + 0.5 · step quality)
step quality = mean of the four step axes
```

- **Failed start.** A mechanism with no step scores 0 on every axis.
- **Connectivity versus product.** Each broken link costs 0.3/n of the score, where n is the number of steps, and the product earns 0.2. Breaking more than two-thirds of the links therefore costs more than reaching the product earns. A single broken link in a long path costs less than the product.
- **Pass.** A case passes when it passes `quality_v1` (product reached, every step valid, mass and charge close, no repeated state), every step is connected, and it scores at least 700.

## Connectivity rule

Species are compared by neutral parent, so a protonation change is not a break. Two kinds of species are ignored:

- species with at most one heavy atom: water, H3O+, hydroxide, halides, metal cations, H2;
- inert solvents: THF, DCM, CHCl3, CCl4, DMF, DMSO, MeCN, toluene, benzene, Et2O, dioxane, DME, MTBE, hexane, cyclohexane and NMP.

Alcohols, acetone and ethyl acetate are not treated as solvents, because they are often reagents or products.

The first link is broken when step 1 does not start from a starting material, or when it adds a species that is neither a starting material nor a conjugate of one.

A later link (step k after step k−1) is broken when either of these happens:

- **vanished:** a species the path itself made, not in the starting pool and with more than four heavy atoms, is in step k−1's result but missing from step k's start (a small released byproduct such as SO2, CO2, isobutylene or a B(OH)3 fragment may leave);
- **appeared:** step k starts with a species that is neither in the starting pool nor made by an earlier step.

Dropping a spare starting material or reagent between steps is not a break.

## Aggregates, radar and records

For a run (one 10-case slice), the record's `summary` holds the following:

| Field | Meaning |
|---|---|
| `axes` | mean of each axis over cases |
| `points` | mean case score |
| `polygon_area` | Σ v_i·v_{i+1} / 6 around the axes in order; 1 means perfect on every axis |
| `failed_starts`, `failed_start_fraction` | cases with no step, as a count and a fraction |
| `fully_connected` | cases with every step connected |

The `quality_v1` summary is kept in the record as `quality_v1` and the pre-rubric numbers as `legacy`. Each case row carries `quality_points` (v2), `quality_v1_points`, `axes`, `connected_steps` and `broken_links`.

The harness-vs-baseline section of `LEADERBOARD.md` and the README shows one radar per model × product mode (`results/radar/*.svg`), with the baseline in orange and the harness in blue. Each radar has the score, polygon area and failed-start share next to it.

Re-scoring stored runs needs nothing new. Run `publish-results` again: `_case_quality` recomputes any case whose stored quality lacks `quality_v2`.
