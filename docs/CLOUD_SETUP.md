# Cloud setup

Single-process FastAPI app + static UI. No Docker required unless you want it.

## Quick start

```bash
git clone <repo> && cd professor-wiggum
bash scripts/setup-cloud.sh
cp .env.cloud.example .env   # edit keys, or set vars in Railway/Render/Fly UI
bash scripts/run-cloud.sh
```

Open `http://<host>:<port>/` (root serves the UI; static assets under `/ui/`).

## Platform notes

| Platform | Start command | Notes |
|----------|---------------|-------|
| **Railway / Render / Fly** | `bash scripts/run-cloud.sh` | Set `PORT` in platform; script reads it automatically |
| **Cursor Cloud** | `bash scripts/run-cloud.sh` | Activate venv first; see AGENTS.md Cursor Cloud section |
| **Generic VM** | `bash scripts/setup-cloud.sh && bash scripts/run-cloud.sh` | Mount disk at `MECHANISTIC_DATA_DIR` |

### Persistent storage

Without a volume, SQLite and run traces live inside the container filesystem and are lost on redeploy.

```bash
export MECHANISTIC_DATA_DIR=/data/wiggum-data
mkdir -p /data/wiggum-data
bash scripts/setup-cloud.sh
```

Artifacts under that path:

- `data/mechanistic.db` — runtime DB
- `traces/runs/` — per-run scratchpads and step JSON

Committed eval JSON stays in the repo under `training_data/`; only bulk/generated data needs the volume.

## Recommended environment variables

### Minimum (hosted LLM)

Set **one** provider key matching your model:

| Variable | When needed |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI models (`gpt-4o`, `gpt-5.x`, …) |
| `OPENROUTER_API_KEY` | OpenRouter ids (`anthropic/claude-opus-4.6`, …) |
| `ANTHROPIC_API_KEY` | Direct Anthropic; also fallback for OpenRouter |
| `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Gemini models |

### Cloud runtime

| Variable | Default | Purpose |
|----------|---------|---------|
| `PORT` or `MECHANISTIC_PORT` | `8010` | Listen port (platforms inject `PORT`) |
| `MECHANISTIC_HOST` | `0.0.0.0` | Bind address |
| `MECHANISTIC_DATA_DIR` | repo root | Persistent bulk-data root |
| `MECH_ENABLE_GIT_OPS` | `true` | Set `false` in production |
| `MECH_ENABLE_SUBPROCESS_OPS` | `true` | Set `false` in production |
| `MECH_CORS_ORIGINS` | unset | Comma-separated origins for cross-origin UI |

### Chemistry

| Variable | Default | Purpose |
|----------|---------|---------|
| `MECHANISTIC_CHEMISTRY_BACKEND` | `auto` | `python`, `rdkit_cli`, or `auto` |
| `MECHANISTIC_RDKIT_CLI_COMMAND` | `rdkit-agent` | CLI when using `rdkit_cli` |
| `MECHANISTIC_RDKIT_CLI_TIMEOUT_SECONDS` | `5` | CLI timeout |

Run `npm install` in repo root (handled by `setup-cloud.sh`) before using `rdkit_cli`.

### Agent bridge (keyless)

See [agent_bridge.md](agent_bridge.md). Minimum:

```bash
MECHANISTIC_ACTIVE_MODEL=agent-bridge
MECHANISTIC_AGENT_BRIDGE_DIR=/data/agent_bridge
MECHANISTIC_AGENT_BRIDGE_DECLARED_MODEL="your-responder-label"
MECHANISTIC_AGENT_BRIDGE_RESPONDER_KIND=orchestrator_subagents
MECHANISTIC_AGENT_BRIDGE_SAW_GROUND_TRUTH=false
```

A separate process must answer bridge requests (`main.py bridge-serve` or your orchestrator).

### Advanced tuning

Per-step model overrides (`MECHANISTIC_INTERMEDIATE_MODEL`, `MECHANISTIC_ATOM_MAPPING_MODEL`, …), prompt caps, and reasoning env vars are documented in [.env.example](../.env.example).

## Verify

```bash
source .venv/bin/activate
make test          # fast suite, no API keys
python main.py serve --host 0.0.0.0 --port 8010
curl -s http://127.0.0.1:8010/api/examples | head
```

LLM evals require API keys: `make test-llm`.
