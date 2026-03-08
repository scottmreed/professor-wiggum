# Practice Eval Set

This directory contains a **practice evaluation set** for contributors who want to test their changes locally before submitting a PR.

**This is NOT the leaderboard evaluation set.** Results on this set do not count toward leaderboard rankings or merge gates.

## Contents

| File | Description |
| --- | --- |
| `practice_set.json` | 20 reactions with full verified mechanisms (same format as `eval_set.json`) |
| `practice_tiers.json` | Tier assignments: easy (1-2 steps), medium (3 steps), hard (4+ steps) |

## Source

These reactions are drawn from the same FlowER `train.txt` source as the development eval set but use **completely disjoint reaction IDs**. They do not overlap with:
- `training_data/eval_set.json` (the development eval set)
- The leaderboard holdout set (private)

## Usage

Run the practice eval to test your changes locally:

```bash
source .venv/bin/activate
python main.py eval --eval-set training_data/practice_eval/practice_set.json --tier easy
```

## Regeneration

To rebuild this set from the FlowER source (requires the FlowER dataset and lookup cache):

```bash
python scripts/build_practice_eval.py
```

See `training_data/REGENERATE.md` for prerequisites.
