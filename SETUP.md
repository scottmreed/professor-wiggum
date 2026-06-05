# Setup

You can contribute without cloning the repo. See [CONTRIBUTING.md](CONTRIBUTING.md) for the no-clone path and the tracks that do require a local checkout.

## Prerequisites

- Python 3.10+
- At least one LLM API key (see [Environment Variables](#environment-variables) below)

## Install

```bash
git clone <repo>
cd Mechanistic
bash setup.sh
```

`setup.sh` creates a `.venv`, installs `requirements.txt`, and installs the project in editable mode. After it runs, activate the environment:

```bash
source .venv/bin/activate
```

## rdkit-agent Backend (Policy-Compliant Install)

If you plan to run with `MECHANISTIC_CHEMISTRY_BACKEND=rdkit_cli`, install the repo-local Node dependency (do not use `npm link`):

```bash
npm install
```

This installs `rdkit-agent` into `./node_modules` so backend resolution uses a package path inside the repository.

## Environment Variables

```bash
cp .env.example .env
# Then edit .env and add your API key(s)
```

See `.env.example` for details. At minimum, set one provider key matching the model you plan to run. For example, to run Claude models via OpenRouter you only need `OPENROUTER_API_KEY`.

## RDKit (Chemistry Features)

RDKit enables SMILES validation, atom mapping, and canonicalization. It is excluded from `requirements.txt` due to platform-specific install issues on some systems.

**Option A — pip extras (recommended):**
```bash
pip install -e ".[chemistry]"
```

**Option B — conda:**
```bash
conda install -c conda-forge rdkit
```

Without RDKit, chemistry validation is disabled and 3 fast tests will fail (this is expected; they are noted as pre-existing failures in MEMORY.md).

## Verify Setup

Run the fast test suite (no API keys required):

```bash
pytest tests/fast/ -q
```

Expected: all pass. A handful of chemistry tests require RDKit (see above); without it, those tests are skipped or xfail.

To run LLM tests (requires API keys):

```bash
pytest tests/llm/ -q
```

## Curriculum SQLite Lookups

Curriculum operations (`python main.py curriculum submit`, etc.) require two SQLite index files:

- `data/flower_train_lookup.sqlite`
- `data/flower_test_lookup.sqlite`

These are generated locally — they are not tracked in git. Build the train lookup from the full FlowER mechanism index (takes several minutes after downloading the index):

```bash
python main.py curriculum build-lookup
```

**Maintainers** keep bulk artifacts in a sibling `../wiggum-data` checkout (resolved automatically). **Forkers** generate everything locally; the app falls back to `data/` and `training_data/` inside this repo. See [docs/DATA_SETUP.md](docs/DATA_SETUP.md) for details and the `MECHANISTIC_DATA_DIR` override.

## Large Data Files

The full FlowER mechanism index (`flower_mechanism_index.jsonl`, ~65 MB) is not tracked in git. A 1,000-line sample is at `training_data/flower_mechanism_index_sample.jsonl` for format reference.

You only need the full index for curriculum operations and evolve harness runs — not for running evals or tests. See [training_data/REGENERATE.md](training_data/REGENERATE.md) for rebuild instructions (requires the FlowER dataset from figshare).

See [README.md](README.md) for the developer workflow and curriculum overview.
