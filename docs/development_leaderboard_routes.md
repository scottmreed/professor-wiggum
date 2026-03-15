# Development Leaderboard Routes

This document describes the policy-driven route planner used by `python main.py eval --tier ...`.

## Purpose

The development leaderboard route planner gives contributors a persistent path onto the development leaderboard without hardcoding today's seed size or tier source into the CLI.

The planner compares within:

- exact `model_name`
- exact `thinking_level`

The planner applies only to harness eval tier flows run through `python main.py eval --tier ...`.

It does **not** apply to:

- `python main.py baseline ...`
- `python main.py eval --all-tiers ...`
- `python main.py eval-runset-official ...`
- arbitrary non-tier `--eval-set-id` runs

## Policy File

Route behavior is defined in [training_data/development_leaderboard_policy.json](../training_data/development_leaderboard_policy.json).

The policy controls:

- the initial qualifying tier
- the initial qualifying case count
- the same-tier extension increment
- the next-tier seed count
- which tier inventory view is currently active for each tier

This means "first qualifying slice is easy 10" is a data choice, not a code invariant.

## Tier Sources

The route planner treats these files as synchronized views over the same development mechanism pool:

- [training_data/eval_tiers.json](../training_data/eval_tiers.json)
- [training_data/baseline_tiers_clawdiator.json](../training_data/baseline_tiers_clawdiator.json)

The policy file decides which view is active per tier.

Current defaults:

- `easy` uses `eval_tiers`
- `medium` uses `baseline_tiers_clawdiator`
- `hard` uses `baseline_tiers_clawdiator`

This supports the current repo state where `easy` may intentionally expose more eval-visible mechanisms while `medium` and `hard` remain Clawdiator-backed until the eval-facing tiers are prepared.

## Routes

When no qualifying canonical row exists for the `model + thinking` scope:

- `seed`: run the policy-defined initial canonical slice

After a qualifying canonical row exists, the planner offers:

- `same`: rerun the current canonical slice and try to beat the current winner
- `extend`: run the current canonical slice plus the policy-defined increment and try to beat the current winner
- `next`: move to the next tier and run that tier's policy-defined seed slice
- `custom`: bypass planner case selection and preserve manual case control

The current winner is the best completed canonical row in the highest tier already reached by that `model + thinking` scope, using the existing leaderboard ordering:

- quality descending
- pass rate descending
- cost ascending

## CLI Usage

Show planner status only:

```bash
python main.py eval \
  --eval-set-id ignored \
  --tier easy \
  --model anthropic/claude-opus-4.6 \
  --thinking-level high \
  --leaderboard-status-only
```

Use the default recommended route:

```bash
python main.py eval \
  --eval-set-id ignored \
  --tier easy \
  --model anthropic/claude-opus-4.6 \
  --thinking-level high
```

Force a specific route:

```bash
python main.py eval \
  --eval-set-id ignored \
  --tier easy \
  --model anthropic/claude-opus-4.6 \
  --thinking-level high \
  --leaderboard-route extend
```

Bypass planner case selection:

```bash
python main.py eval \
  --eval-set-id ignored \
  --tier easy \
  --model anthropic/claude-opus-4.6 \
  --thinking-level high \
  --leaderboard-route custom \
  --case-id flower_135501 \
  --case-id flower_024300
```

## TTY vs Non-TTY

Interactive terminal:

- prints current status
- prints active tier sources and the current winner
- prompts for route confirmation unless `--yes` is passed

Non-interactive execution:

- prints the same status block in normal text mode
- auto-selects the recommended route

JSON mode preserves JSON-oriented output rather than mixing in the human-readable prompt flow.
