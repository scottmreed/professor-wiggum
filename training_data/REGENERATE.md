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

## Reaction novelty index and train-only reference mechanisms

**What it is, in plain language.** ChemIllusion needs to know whether a reaction a user submits is already in FlowER, so it never treats a FlowER reaction as a new private test case and never reveals the leaderboard holdout. The novelty index is a compact file of *fingerprints only*: for every FlowER train and test mechanism (start state → final state, atom maps ignored) it stores the split label, four 64-bit keys made from standard InChIKeys (exact, core, endpoint, family) and two 256-bit similarity fingerprints. It contains no structures, no mechanisms and no FlowER ids, and the loader can only answer "this matches the train/test split" or "this is X% similar to something in the train/test split".

The optional second file, the **train-only reference mechanism library**, powers ChemIllusion's free "reference mechanism from the FlowER dataset" feature: when a user's reactants and products match a FlowER *train* mechanism, ChemIllusion can show that mechanism's steps. It never contains test-split mechanisms (the builder refuses them and the loader checks).

Code: `mechanistic_agent/reaction_signatures.py` (the recipe, shared with ChemIllusion), `mechanistic_agent/novelty_index.py` (loaders), `scripts/build_reaction_novelty_index.py` (builder). Tests: `tests/fast/test_reaction_signatures.py`, `tests/fast/test_novelty_index.py`.

### Build

Use the RDKit version you want the deployed fingerprints to match (it is recorded in the manifest; ChemIllusion should run the same one). From the repo root, with `../FlowER/data/flower_new_dataset/{train,test}.txt` present:

```bash
python scripts/build_reaction_novelty_index.py \
  --flower-dir ../FlowER/data/flower_new_dataset \
  --with-mechanisms --workers 8 --out dist/novelty
```

Options: `--train-file/--test-file` instead of `--flower-dir`; `--no-fingerprints` (keys only, roughly half the size); `--max-bytes` (default 40,000,000); `--limit N` mechanisms per split for a smoke run. `--out` must not be inside `training_data/`; `dist/` is git-ignored. A single worker takes about 4–5 ms per mechanism.

Expected outputs in `dist/novelty/`:

- `reaction_novelty_index.npz` — about 20 MB for 300k rows with fingerprints (about 10 MB without). The build **fails with exit 2** above `--max-bytes`; rebuild with `--no-fingerprints` if that happens.
- `reaction_novelty_index.manifest.json` — recipe versions, RDKit/numpy versions, SHA-256 of `train.txt` and `test.txt`, rows per split, skipped counts, artifact bytes and SHA-256, and (with `--with-mechanisms`) the library's size, SHA-256 and per-mechanism averages.
- `flower_train_mechanisms.sqlite` and `flower_train_mechanisms.sqlite.gz` — train only, keyed by the core key. The core key ignores species found on both sides (regenerated catalysts, spectators) and one-heavy-atom fragments such as H+, water and counterions, so a user's bare substrate → product finds the FlowER mechanism that carries them. The builder prints the compressed and uncompressed size and the average steps and bytes per mechanism; there is no size budget yet (Scott sets one after the first real build).

The same inputs with the same RDKit and numpy give byte-identical artifacts (no timestamps inside them; `built_at` lives only in the manifest).

### Publish

1. Create a release on `scottmreed/professor-wiggum` with tag `novelty-index-reaction_corpus.v1` and upload the three files `reaction_novelty_index.npz`, `reaction_novelty_index.manifest.json` and `flower_train_mechanisms.sqlite.gz`:

   ```bash
   gh release create novelty-index-reaction_corpus.v1 \
     dist/novelty/reaction_novelty_index.npz \
     dist/novelty/reaction_novelty_index.manifest.json \
     dist/novelty/flower_train_mechanisms.sqlite.gz \
     --title "Reaction novelty index (reaction_corpus.v1)" \
     --notes "FlowER-derived novelty keys/fingerprints (train+test) and train-only reference mechanisms. FlowER: Nature 645, 115–123 (2025)."
   ```

2. The builder ends by printing the `assets` list (name, kind, url, sha256, bytes). Paste it into `novelty_index/manifest.json`, which then looks like:

   ```json
   {
     "schema": "novelty_index_manifest.v1",
     "release_tag": "novelty-index-reaction_corpus.v1",
     "corpus_recipe_version": "reaction_corpus.v1",
     "recipe_version": "mechanism_submission.v2",
     "rdkit_version": "2023.09.1",
     "assets": [
       {"name": "reaction_novelty_index.npz", "kind": "novelty_index", "url": "https://github.com/scottmreed/professor-wiggum/releases/download/novelty-index-reaction_corpus.v1/reaction_novelty_index.npz", "sha256": "…", "bytes": 0},
       {"name": "reaction_novelty_index.manifest.json", "kind": "novelty_index_manifest", "url": "…", "sha256": "…", "bytes": 0},
       {"name": "flower_train_mechanisms.sqlite.gz", "kind": "train_mechanisms", "url": "…", "sha256": "…", "bytes": 0}
     ]
   }
   ```

   While `assets` is `null`, `load_from_manifest` returns `None` and ChemIllusion reports reference screening as unavailable. A present file whose SHA-256 differs from this manifest raises `NoveltyIndexMismatch`.

3. Commit the manifest change only (never the assets). ChemIllusion picks it up when `WIGGUM_RUNTIME_REF` is bumped.

Before ChemIllusion shows reference mechanisms to users, confirm that the FlowER dataset license (figshare 28359407) allows it and add the attribution it requires.
