# Sample Reaction PNGs

These 5 PNG files are samples showing the expected format for reaction mechanism visualizations. They are included in version control so contributors can see the output format without regenerating the full set.

## Full PNG sets (generated locally, not tracked)

| Directory | Contents | Approximate size |
| --- | --- | --- |
| `flower_curriculum_pngs/` | ~151 PNGs for the curriculum dataset | ~24 MB |
| `eval_set_pngs/` | ~101 PNGs for the dev eval set | ~8 MB |
| `pngs/` | ~351 PNGs for extended training data | ~63 MB |

## Regeneration

```bash
source .venv/bin/activate

# Curriculum PNGs
python scripts/render_flower_mechanism_pngs.py

# Eval set PNGs
python scripts/render_eval_set_pngs.py
```

Requires RDKit. See [SETUP.md](../../SETUP.md) for installation.
