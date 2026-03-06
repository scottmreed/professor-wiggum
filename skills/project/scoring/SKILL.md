# Scoring Skill

Deterministic mechanism grading rubric for eval runs and leaderboard scoring.

## What Is Scored

- Step-level deterministic validity:
  - bond/electron conservation (`mechanism_moves`)
  - atom balance
  - state progress
  - step atom-mapping confidence
- Alignment to known benchmark mechanism steps (`known_mechanism.steps`)
- Whether the accepted path reaches the known final product
- Efficiency penalties for circular/repeated/unnecessary steps

## Weighted Formula

`overall = clamp(0.45*validity + 0.35*alignment + 0.20*final - penalties, 0, 1)`

- `validity` = average accepted-step validity score
- `alignment` = average accepted-step benchmark alignment
- `final` = `1` if known final product reached, else `0`
- `penalties` = sum of efficiency penalties (capped)

If final product is not reached, score is capped at `0.55`.

Pass rule:
- `passed = (final product reached) and (score >= 0.70)`

## Alignment Rules

Per accepted step:
- Exact same step target reached: `1.00`
- Target appears in a future known step: `0.75`
- Pathway is reasonable but non-identical: high partial credit (`~0.55`)
- Weak/no alignment: low credit

This allows high (not perfect) scores for chemically reasonable alternate pathways.

## Efficiency Penalties

- Circular step (no state change): penalty
- Repeated previously seen state: penalty
- Extra accepted steps beyond known `min_steps`: penalty

Penalties reduce score even when the final product is reached.

## Interpretation Bands

- `0.90-1.00`: excellent
- `0.70-0.89`: good
- `<0.70`: needs work

## Review Guidance

- Investigate repeated or circular state transitions first.
- If final product is not reached, inspect early step validity/alignment drift.
- If pathway differs from benchmark but is chemically reasonable, expect high-but-not-perfect score.
