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

## rdkit_cli Backend (Policy-Compliant Install)

If you plan to run with `MECHANISTIC_CHEMISTRY_BACKEND=rdkit_cli`, install the repo-local Node dependency (do not use `npm link`):

```bash
npm install
```

This installs `rdkit_cli` into `./node_modules` so backend resolution uses a package path inside the repository.

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

Expected: ~42/45 pass without RDKit, ~45/45 with RDKit installed.

To run LLM tests (requires API keys):

```bash
pytest tests/llm/ -q
```

## Curriculum SQLite Lookups

Curriculum operations (`python main.py curriculum submit`, etc.) require two SQLite index files that are not stored in git:

- `data/flower_train_lookup.sqlite`
- `data/flower_test_lookup.sqlite`

Build the train lookup from the committed `.jsonl` index (takes several minutes):

```bash
python main.py curriculum build-lookup
```

## Large Data Files

The full FlowER mechanism index (`flower_mechanism_index.jsonl`, ~65 MB) is **not tracked in git** to keep the repo lightweight. A 1,000-line sample is committed at `training_data/flower_mechanism_index_sample.jsonl` for format reference.

You only need the full index for curriculum operations and evolve harness runs — not for running evals or tests. See [training_data/REGENERATE.md](training_data/REGENERATE.md) for rebuild instructions.

See [README.md](README.md) for the developer workflow and curriculum overview.
