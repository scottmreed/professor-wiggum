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

## flower_mechanisms_multistep.json (medium/hard tier cases)

The records behind the `medium` (3-step) and `hard` (4-10-step) development tiers. Gitignored; generated from the index above. Built once with `--mode stratified --per-step 20 --max-step 6`, then grown append-only:

```bash
# Grow tiers to the given totals. Existing rows stay verbatim and in place; new rows are the
# next lowest-ranked successful conversions per step-count tier, skipping IDs in eval_set.json,
# flower_mechanisms_100.json, the practice set, both tier files and the holdout.
python scripts/build_flower_mechanism_dataset.py --mode extend \
  --dataset-output training_data/flower_mechanisms_multistep.json \
  --dataset-report training_data/flower_mechanisms_multistep_report.json \
  --extend-step 3=40 --extend-step 7=20 --extend-step 8=20 --extend-step 9=20 --extend-step 10=20

# Append the new tier IDs' cases to the existing medium/hard eval sets in the DB (same eval_set_id;
# version bumped, sha256 = content hash). Dry run without --apply. Back up the DB first.
python scripts/sync_dev_tier_eval_sets.py --apply
```

Each extension is recorded under `extensions` in the report (targets, IDs added per step, attempts, conversion failures). FlowER train has only 5 convertible 9-step and 7 convertible 10-step mechanisms (every indexed group was attempted), so those bands hold fewer than 20. Append the new IDs to the end of the tier lists in both `eval_tiers.json` and `baseline_tiers_clawdiator.json`.

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
- `flower_train_mechanisms.v2.sqlite` (with `--with-mechanisms`) — the reference mechanism library, train only, keyed by the core key. The core key ignores species found on both sides (regenerated catalysts, spectators) and one-heavy-atom fragments such as H+, water and counterions, so a user's bare substrate → product finds the FlowER mechanism that carries them. The builder prints its size and the average steps and bytes per mechanism.

#### Library format v2 (`train_mechanisms.v2`)

SQLite with `meta(key, value)` (`format = train_mechanisms.v2`, `split = train`, `corpus_recipe_version`, `source_sha256`, `rows`, `built_at`) and `mechanisms(core_key INTEGER, mechanism_id INTEGER, step_count INTEGER, payload BLOB)`, indexed on `core_key`, rows in `(core_key, mechanism_id)` order. Each `payload` is `zlib.compress(level 9)` of the compact JSON `{"initial_state": [...], "steps": [{"resulting_state": [...], "electron_pushes": ["lp:11>1", ...]}]}`. The state is stored once per step, and the fields that are derivable or constant are dropped: `current_state`, `target_products`, `predicted_intermediate`, `reaction_smirks`, `note`, `confidence` and the structured push dicts (only the `notation` string is kept). The file is shipped uncompressed and queried in place, because each row is already compressed. `built_at` is `SOURCE_DATE_EPOCH` when that is set, otherwise the source file's mtime, so the same input gives the same SHA-256.

The first real build wrote the older **v1** library (`flower_train_mechanisms.sqlite[.gz]`, full step JSON per row): 68 MB gzipped and 1.35 GB on disk for 118,888 mechanisms. Converted to v2 it is **60,297,216 bytes** (about 507 B per mechanism, with a 434 B payload), takes about 20 s, and gives identical lookups for all 112,214 core keys. `ReferenceMechanismLibrary.lookup(core_key)` reads both formats and returns the same shape: `[{"mechanism_id", "step_count", "initial_state", "steps": [{"index", "resulting_state", "electron_pushes"}]}]`.

**Known coverage limit.** Only mechanisms that `flower_curriculum._convert_group` can convert are in the library: 118,888 of 257,167 train mechanisms. About 121k (120,771) are skipped as `state_discontinuity`, and others as `no_cycle_found` (14,447) or as unsupported bond-order or charge patterns. The novelty index itself covers all of train and test, so plan B shows a reference mechanism for only about half of the train reactions it recognizes. The skip counts are in `reaction_novelty_index.manifest.json` under `train_mechanisms.skipped`.

The same inputs with the same RDKit, numpy, SQLite and zlib give byte-identical artifacts. The npz has no timestamps inside it, and the index's `built_at` is only in its manifest. The library's `built_at` is pinned as described above.

### Publish

1. Create a release on `scottmreed/professor-wiggum` with tag `novelty-index-reaction_corpus.v1` and upload the index files (done on 2026-10-01 for `reaction_corpus.v1`):

   ```bash
   gh release create novelty-index-reaction_corpus.v1 \
     dist/novelty/reaction_novelty_index.npz \
     dist/novelty/reaction_novelty_index.manifest.json \
     --title "Reaction novelty index (reaction_corpus.v1)" \
     --notes "FlowER-derived novelty keys/fingerprints (train+test) and train-only reference mechanisms. FlowER: Nature 645, 115–123 (2025)."
   ```

2. Upload the v2 mechanism library. A fresh `--with-mechanisms` build already writes `dist/novelty/flower_train_mechanisms.v2.sqlite`. If you have only the v1 file from the first build, convert it; you don't need to rebuild FlowER. The converter refuses a v1 file whose meta is not `split = train`:

   ```bash
   python scripts/compact_reference_mechanisms.py \
     --input dist/novelty/flower_train_mechanisms.sqlite.gz --out dist/novelty
   gh release upload novelty-index-reaction_corpus.v1 dist/novelty/flower_train_mechanisms.v2.sqlite
   ```

   The converter prints the row count, the size, per-mechanism averages, the SHA-256 and a ready-to-paste asset entry. It decompresses a `.gz` input into `--out`, or into `--tmp-dir`, which needs about 1.4 GB free.

3. Paste the printed asset entries into the `assets` list of `novelty_index/manifest.json`. The builder prints all of them, and the converter prints the `train_mechanisms` one. The list then looks like this:

   ```json
   {
     "schema": "novelty_index_manifest.v1",
     "release_tag": "novelty-index-reaction_corpus.v1",
     "assets": [
       {"name": "reaction_novelty_index.npz", "kind": "novelty_index", "url": "https://github.com/scottmreed/professor-wiggum/releases/download/novelty-index-reaction_corpus.v1/reaction_novelty_index.npz", "sha256": "…", "bytes": 0},
       {"name": "reaction_novelty_index.manifest.json", "kind": "novelty_index_manifest", "url": "…", "sha256": "…", "bytes": 0},
       {"name": "flower_train_mechanisms.v2.sqlite", "kind": "train_mechanisms", "url": "…", "sha256": "…", "bytes": 0}
     ]
   }
   ```

   `load_from_manifest(base_dir, assets_dir)` and `load_reference_library_from_manifest(base_dir, assets_dir)` return `None` when `assets` is `null`, when the kind is not listed, or when the file is not in `assets_dir`, and ChemIllusion then reports that feature as unavailable. A present file whose SHA-256 differs from this manifest raises `NoveltyIndexMismatch`. The committed manifest lists the two index assets now; the `train_mechanisms` entry is added after the v2 upload.

4. Commit the manifest change only (never the assets). ChemIllusion picks it up when `WIGGUM_RUNTIME_REF` is bumped.

`tests/fast/test_novelty_index.py::test_committed_manifest_loads_the_released_index` loads the released index through the committed manifest when the files are in `dist/novelty/` or in `$WIGGUM_NOVELTY_ASSETS_DIR`, and is skipped otherwise.

The FlowER dataset (figshare 28359407) is MIT-licensed (confirmed by Scott on 2026-10-01). Show the attribution with any reference mechanism.
