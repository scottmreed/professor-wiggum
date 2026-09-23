# Regenerating Large Training Data Files

Some files are too large for version control and must be regenerated locally. This page documents what they are and how to rebuild them.

Eval and curriculum data are derived from the **FlowER** dataset: *Electron flow matching for generative reaction mechanism prediction.* Nature 645, 115–123 (2025). DOI: [10.1038/s41586-025-09426-9](https://doi.org/10.1038/s41586-025-09426-9). To recreate or obtain the source data, use the [FlowER dataset on figshare](https://figshare.com/articles/dataset/FlowER_-_Mechanistic_datasets_and_model_checkpoint/28359407/3).

## flower_mechanism_index.jsonl (65 MB, 257,167 mechanisms)

A ranked JSONL index of every mechanism group in the FlowER `train.txt` dataset. Used by `evolve_harness` and curriculum operations.

A 1,000-line sample is tracked at `flower_mechanism_index_sample.jsonl` so you can inspect the format without rebuilding.

### Prerequisites

1. Clone the FlowER dataset so that `../FlowER/data/flower_new_dataset/train.txt` exists relative to the project root.
2. Activate the project virtualenv: `source .venv/bin/activate`

### Rebuild steps

```bash
# Build the SQLite lookup cache (required first)
python main.py curriculum build-lookup

# Build the full JSONL index
python scripts/build_flower_mechanism_dataset.py
```

Expected output (under **wiggum-data** when using the default sibling layout):

- `training_data/flower_mechanism_index.jsonl` (~65 MB, ~257,167 lines)
- `training_data/flower_mechanism_index_report.json` (generation stats)
- `data/flower_train_lookup.sqlite` (via `main.py curriculum build-lookup`)

See [docs/DATA_SETUP.md](../docs/DATA_SETUP.md) for checkout layout.

### When is this needed?

- **Running evals or tests**: Not required. `eval_set.json` and `practice_eval/practice_set.json` are committed and self-contained.
- **Curriculum operations** (`python main.py curriculum ...`): Required.
- **Evolve harness** (`python scripts/evolve_harness.py`): Required.
- **Building new eval sets**: Required.

## Reaction PNGs

PNG visualizations live in the bulk data checkout (`flower_curriculum_pngs/`, `eval_set_pngs/`) — see [docs/DATA_SETUP.md](../docs/DATA_SETUP.md).

```bash
# Render curriculum PNGs
python scripts/render_flower_mechanism_pngs.py

# Render eval set PNGs
python scripts/render_eval_set_pngs.py
```

See `sample_pngs/README.md` for format examples.
