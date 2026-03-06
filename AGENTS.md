# Mechanistic Agent - Runtime Notes

This repository now runs on a local-first architecture:

- Runtime orchestration: explicit coordinator/state machine in `mechanistic_agent/core/`
- API service boundary: FastAPI app in `mechanistic_agent/api/`
- UI: static HTML/JS/CSS in `mechanistic_agent/ui/`
- Persistence: SQLite database at `data/mechanistic.db`
- Curated assets:
  - `skills/mechanistic/` (LLM prompts, few-shot files, and deterministic validators — formerly `prompt_versions/`)
  - `skills/project/` (project-level skills used outside mechanism prediction)
  - `memory_packs/`
  - `traces/` (run traces + PR evidence traces)
  - `harness_versions/` (per-harness subdirectories with `harness.json`)

## Verified vs Unverified

- `unverified`: runtime can auto-generate mechanism steps and validates each step.
- `verified`: runtime requires user-submitted mechanistic steps via API/UI, then validates each step deterministically.

## Key Endpoints

- Runs: `POST /api/runs`, `POST /api/runs/{id}/start`, `POST /api/runs/{id}/stop`, `POST /api/runs/{id}/resume`
- Verified step submission: `POST /api/runs/{id}/mechanism_steps`
- Snapshot/flow/events: `GET /api/runs/{id}`, `GET /api/runs/{id}/flow`, `GET /api/runs/{id}/events`
- Memory: `GET /api/memory`, `POST /api/memory/query`, `POST /api/memory/items`
- Traces/curation: `GET /api/traces`, `POST /api/traces/{trace_id}/approve`, `POST /api/curation/export`, `GET /api/curation/exports`
- Evals: `POST /api/evals/runset`, `GET /api/evals/leaderboard` (returns overall + per-subagent quality/pass scores)

## Leaderboard Holdout Isolation

- The train-derived eval/sample artifacts remain the user-facing development surface:
  - `training_data/eval_set.json`
  - `training_data/eval_tiers.json`
  - `training_data/flower_mechanisms_100.json`
- The leaderboard holdout suite is generated from `FlowER ... /test.txt` and stored separately under:
  - `training_data/leaderboard_holdout/`
- Holdout eval sets are marked `purpose=leaderboard_holdout` and hidden from UI example/eval menus.
- `POST /api/evals/runset` and `POST /api/evals/baseline-runset` reject holdout eval sets.
- Only official ranking flows use holdout:
  - `POST /api/evals/official-runset`
  - `GET /api/evals/leaderboard/official`
- `scripts/evolve_harness.py` explicitly rejects holdout eval sets to prevent tuning on leaderboard data.

## Skill and Prompt Architecture

All mechanistic prediction assets live under `skills/mechanistic/`, split from project-level skills in `skills/project/`.

### Mechanistic Skills (`skills/mechanistic/`)

Each subdirectory is a self-contained skill with a unified `SKILL.md`:

| Directory | Kind | Files |
|-----------|------|-------|
| `base_system/` | `shared_base` | `SKILL.md` (shared base system prompt) |
| `assess_initial_conditions/` | `llm` | `SKILL.md`, `few_shot.jsonl` |
| `predict_missing_reagents/` | `llm` | `SKILL.md`, `few_shot.jsonl` |
| `attempt_atom_mapping/` | `llm` | `SKILL.md`, `few_shot.jsonl` |
| `select_reaction_type/` | `llm` | `SKILL.md`, `few_shot.jsonl` |
| `propose_mechanism_step/` | `llm` | `SKILL.md`, `few_shot.jsonl` |
| `evaluate_run_judge/` | `llm` | `SKILL.md`, `few_shot.jsonl` |
| `baseline_mechanism/` | `llm` | `SKILL.md` |
| `bond_electron_validation/` | `deterministic` | `SKILL.md`, `validator.py` |
| `atom_balance_validation/` | `deterministic` | `SKILL.md`, `validator.py` |
| `state_progress_validation/` | `deterministic` | `SKILL.md`, `validator.py` |

**SKILL.md format**: Each file has YAML frontmatter (`skill_type`, `call_name`, `kind`, `phase`, `steps`, `version`), a human-readable description with I/O schema, and for LLM skills a prompt section delimited by `<!-- PROMPT_START -->` / `<!-- PROMPT_END -->` markers. The file is self-contained — a developer can read one file to understand and use the skill standalone.

**Deterministic validator skills** (`bond_electron_validation`, `atom_balance_validation`, `state_progress_validation`): The ground truth implementation is `validator.py` in each skill directory. `mechanistic_agent/core/validators.py` is a thin dispatcher that imports from these. Harness-specific validator overrides live in `harness_versions/<harness>/patches/<skill_name>.py`.

### Project Skills (`skills/project/`)

Skills used outside mechanism prediction: `base_runtime/`, `prepare_inputs/`, `scoring/`. These are human-facing documentation artifacts, not invoked during prediction.

### Prompt/Trace Evidence Gate

- Prompt text is embedded in `skills/mechanistic/<call_name>/SKILL.md` between the `<!-- PROMPT_START -->` / `<!-- PROMPT_END -->` markers.
- Few-shot examples live in `skills/mechanistic/<call_name>/few_shot.jsonl` (same directory as the skill).
- Run traces are written under `traces/runs/`.
- PR evidence traces are written under `traces/evidence/<call_name>/<prompt_bundle_sha>/`.
- Any PR changing `skills/mechanistic/<call_name>/(SKILL.md|few_shot.jsonl)` must include approved, linked evidence per changed call.
- Local validation: `PYTHONPATH=. python scripts/validate_prompt_trace_evidence.py --call <call_name>`.

## Subagent Architecture

Every LLM-calling step in the harness is a **subagent** — a named, independently scoreable unit defined in `mechanistic_agent/core/subagents.py`. A subagent either wraps a deterministic algorithm or an LLM call with a fixed OpenAI-format tool schema.

The nine canonical subagents are:

| Subagent class | Step name(s) | Kind |
|----------------|-------------|------|
| `BalanceAgent` | `balance_analysis` | deterministic |
| `ConditionsAgent` | `ph_recommendation`, `initial_conditions` | deterministic + llm |
| `FunctionalGroupsAgent` | `functional_groups` | deterministic |
| `MissingReagentsAgent` | `missing_reagents` | llm |
| `MappingAgent` | `atom_mapping`, `step_atom_mapping` | llm |
| `ReactionTypeAgent` | `reaction_type_mapping` | llm |
| `IntermediateAgent` | `mechanism_step_proposal` | llm |
| `MechanismAgent` | `mechanism_synthesis` | deterministic |
| `ReflectionAgent` | `reflection` | deterministic |

Post-step validators (`bond_electron_validation`, `atom_balance_validation`, `state_progress_validation`) run after every mechanism loop iteration and are scored alongside the canonical subagents in the eval leaderboard per-subagent breakdown.

## Forced Tool Calling Convention

LLM-backed subagents use **forced tool calling** to get structured responses. Key conventions:

- **Tool schemas** live in `mechanistic_agent/tool_schemas.py` — one OpenAI-format schema per LLM-backed subagent. Current schemas: `ASSESS_CONDITIONS_TOOL`, `MISSING_REAGENTS_TOOL`, `ATOM_MAPPING_TOOL`, `INTERMEDIATES_TOOL`, `MECHANISM_STEP_PROPOSAL_TOOL`, `REACTION_TYPE_SELECTION_TOOL`, `PREDICT_FULL_MECHANISM_TOOL`.
- **Routing**: Use `adapter_supports_forced_tools(model_name)` from `llm.py` to check if the adapter supports forced tools at runtime. Only OLMo falls back to text-based JSON parsing.
- **`text` field**: Every tool schema includes a `text` property (not required) so verbose models can provide reasoning without disrupting structured output. Extract and log it separately from the structured fields.
- **Gemini**: Uses `_GeminiChatAdapter` which converts OpenAI-format schemas via `_openai_tools_to_gemini()` and calls `generate_content()` with `ToolConfig(function_calling_config=FunctionCallingConfig(mode=FunctionCallingConfigMode.ANY, ...))` from the `google-genai` SDK.
- **Fallback**: Always maintain a text-based fallback path. OLMo prompts already include the expected JSON schema to encourage structured output.
- **New LLM-backed subagents** should follow this pattern: define a schema in `tool_schemas.py`, check `adapter_supports_forced_tools()`, and implement both tool-call and text-fallback paths.

### Deterministic subagents (no LLM call)

- `predict_mechanistic_step` — purely deterministic validation, no LLM call
- `validate_mechanism_step_output` — deterministic RDKit validation, no LLM call
- `analyse_balance`, `fingerprint_functional_groups`, `recommend_ph` — deterministic
- OLMo (`allenai/olmo-3.1-32b-instruct`) — `supports_tools: false` in registry, falls back to prompted text responses

### Mechanism Loop Flow

Within the mechanism loop, only **IntermediateAgent** (`mechanism_step_proposal` subagent) makes an LLM call. The subsequent subagents — **MechanismAgent** (`predict_mechanistic_step`), the three validation subagents (bond/electron, atom balance, state progress), and **ReflectionAgent** (`reflection`) — are all deterministic.

**Topology-aware proposals**: The coordinator dispatches proposal calls based on `coordination_topology` in `CreateRunRequest` (default: `centralized_mas`). The topology profile determines how many agent calls are made, how many candidates each returns, and how results are aggregated:

| Topology | Calls | Candidates | Aggregation |
|---|---|---|---|
| `sas` | 1 | 1 | top-1 only |
| `centralized_mas` | 1 | up to 3 | orchestrator selects |
| `independent_mas` | 3 parallel | 2 each | interleaved by rank |
| `decentralized_mas` | 3 × 2 rounds | 2 each | consensus merge |

In `decentralized_mas` mode, each agent receives summaries of the other agents' prior-round candidates (injected as `peer_proposals` in the prompt) before proposing in round 2+. Results are merged by consensus: candidates supported by 2+ agents rank above single-agent candidates.

All candidates from any topology are validated independently (up to 3 retries each) by the same post-step validators. The topology only affects Step A (proposal); Steps B–G (validation, branching, application, post-step, completion) are topology-invariant.

**Input mapping boundary**: User-entered reactions do not need to be pre-mapped. Before each LLM-backed tool call, atom-map labels are stripped and the SMILES are canonicalized. Deterministic mechanism validation still keeps mapped `reaction_smirks` and `electron_pushes` intact where explicit move notation is required.

**Branch points**: When multiple candidates pass validation, the top-ranked candidate is used and the alternatives are stored as branch points. This enables backtracking if later steps fail.

**Retry flow**: If validation of a single candidate fails and retries remain (up to 3 per candidate), the retry gate routes back to Validate Mechanism Step with retry_feedback injected. Each candidate gets its own set of 3 retries.

**Backtracking**: If all candidates for a step fail validation (or a later step exhausts retries), the system searches for the most recent branch point with untried alternatives. State is reverted to the branch point snapshot (current_state, previous_intermediates, step_index) providing a **clean slate** — no trace of the failed path is passed to the LLM. The next alternative is applied and the loop continues. Only after all alternatives at the most recent branch point are exhausted does backtracking go to earlier branch points.

**Failed path display**: Failed paths are recorded in the event log and displayed to the user with red styling in the UI. The user can see all paths explored, but only the successful path contributes to the LLM context.

**Completion flow**: When validation passes, the retry gate routes to the "Target Products Reached?" decision. If **yes** (target products found in resulting state), the run completes. If **no**, control passes through Collect Validation Warnings and loops back to Propose Next Mechanism Step for the next step. If no branch points remain and all candidates fail, the run pauses for user decision.

## Development

```bash
# Run the fast, free test suite
make test

# Run the full test suite including LLM tests
make test-llm

# Start the server
python main.py serve
```

Tests are split into two suites:
- `tests/fast/`: The free test suite and go-to for making updates and PR approval.
- `tests/llm/`: The LLM-calling test suite that requires API keys. Includes `test_eval_tiers.py` for running eval tier regression tests (easy/medium/hard) against the default FlowER-derived tiers.

RDKit is required for deterministic chemistry validators.

### Frontend Development

The UI consists of static HTML/JS/CSS files served from `mechanistic_agent/ui/`. During development, browsers aggressively cache JavaScript files, which can make testing changes frustrating.

**Cache Busting Strategy:**
- The main app script (`app.js`) includes a timestamp-based version parameter: `<script src="/ui/app.js?v=1734472800000" type="module"></script>`
- When making changes to `app.js`, update the timestamp to a new Unix timestamp (milliseconds since epoch) to force browsers to fetch the updated file
- Example: Change `1734472800000` to `1734472900000` (or use current timestamp from `date +%s%3N`)
- This ensures changes appear immediately without requiring users to disable browser cache or perform hard refreshes

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contribution workflow. There are five contribution tracks, each with its own template, test requirements, and acceptance gate:

| Track | What you're contributing | Key gate | Eval tier required |
|-------|--------------------------|----------|--------------------|
| **Few-Shot Examples** | New lines in `skills/mechanistic/<call_name>/few_shot.jsonl` | Approved evidence trace + medium-tier improvement | **medium** |
| **New Subagents** | New deterministic subagent, validator, or LLM-backed subagent | Fast tests + medium-tier improvement | **medium** + **hard** preferred |
| **New Models** | New entry in `model_pricing.json` or new adapter in `llm.py` | Catalog tests + easy-tier cost-class improvement | **easy** |
| **Harness Changes** | New or modified `harness_versions/<name>/harness.json` | Harness config tests + medium-tier improvement | **medium** |
| **Single Reaction Submission** | One success or failure case for local review | Not mergeable; reviewed for future changes | none |

Tracks 1-4 require `make test` before merge. PR approval is based on eval tier improvement, not the result of a single reaction. Track 5 submissions stay local and are evaluated as evidence for later tracked changes. See [CONTRIBUTING.md](CONTRIBUTING.md) for full details.

## Project Soul and Evolution Philosophy

See [SOUL.md](SOUL.md) for the philosophy behind how this system evolves. Key points:

- The long-term goal is **zero-human-input evolution**: runs generate evidence, evidence gates prompt changes, better prompts produce better runs.
- Deterministic chemistry validation (RDKit) is the final arbiter — LLM confidence does not override it.
- Prompt changes without approved trace evidence do not merge.
- The model catalog (`model_pricing.json`) is the single source of truth for model capabilities and pricing.
- The default eval set (`training_data/eval_set.json`, generated from `training_data/flower_mechanisms_100.json`) and eval tiers (`training_data/eval_tiers.json`, fixed 10 easy + 10 medium + 10 hard, with 1-2 / 3 / 4-8 step bands) are the memory of what the system cares about. PR approval gates on eval tier scores, not individual reaction results.

The system is currently at **Stage 0** (human approves traces, opens PRs). Stages 1–4 move progressively toward automated approval, automated PR generation, and fully autonomous merge. All guardrails are designed to make Stage 4 safe.

# Mechanistic Agent - Cursor Rules

## Project Overview
This is a local-first AI agent for chemical reaction mechanism prediction with evolutionary capabilities. The architecture supports two execution modes (verified/unverified), subagent orchestration, versioned prompt and skill assets in `skills/mechanistic/`, trace evidence gates in `traces/`, and future external validators without redesign. Built with FastAPI backend, SQLite persistence, and static HTML/JS/CSS frontend.

## Architecture Vision
A tool that evolves over time with use:
- **Verified Mode**: Requires user-submitted mechanistic steps with deterministic validation
- **Unverified Mode**: Auto-generates mechanism steps with validation
- **Hybrid Storage**: Local SQLite runtime records + repo-versioned skills/traces evidence artifacts
- **Performance Tracking**: Leaderboard and evidence surfaces for model + harness version performance
- **Evolutionary Learning**: Human feedback through PRs changing prompts, memories, or harness
- **Harness workflow**: one built-in default harness plus PR-editable saved harness variants under `harness_versions/`
- **Preloaded Examples**: Provisional verified steps for all dropdown examples as initial eval set
- **Eval Tiers**: PR gating via tiered eval (easy/medium/hard) from HumanBenchmark-derived data

## Virtual Environment Setup
**CRITICAL**: Always use a virtual environment to avoid dependency conflicts.

### Quick Setup Commands
```bash
# Navigate to project directory
cd /Users/scottreed/PycharmProjects/Mechanistic

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Verify setup
python -c "from mechanistic_agent.config import ReactionInputs; print('✅ Setup complete')"
```

### Environment Activation
Before running any Python commands, always activate the virtual environment:
```bash
# Activate virtual environment
source .venv/bin/activate

# Verify activation
which python  # Should show .venv/bin/python
```

### Required Environment Variables
```bash
# Set OpenAI API key (required)
export OPENAI_API_KEY=sk-your-api-key-here

# Optional: Enable auto-approval for headless operation
export MECHANISTIC_AUTO_APPROVE=1
```

### Cursor Remote Node Bootstrap Note
If Cursor Remote repeatedly fails while installing `exec-daemon` with:

`Error [ERR_MODULE_NOT_FOUND]: Cannot find package 'extensionless' imported from /workspace/`

the failure is in the remote Node bootstrap environment (typically a global `NODE_OPTIONS` preload), not in Mechanistic's Python runtime.

Use this in the remote shell before reopening Cursor:
```bash
echo "$NODE_OPTIONS"
unset NODE_OPTIONS
```

If your environment requires that preload, install the missing package in the workspace:
```bash
npm install --save-dev extensionless
```

See `docs/cursor_remote_bootstrap.md` for the full checklist.

## Project Structure
```
.
├── main.py                          # Typer CLI entrypoint with run/serve commands
├── mechanistic_agent/
│   ├── __init__.py                 # Package initialization
│   ├── config.py                   # Reaction input dataclasses
│   ├── tools.py                    # RDKit/Dimorphite tool implementations
│   ├── scoring.py                  # Deterministic mechanism scoring for leaderboard/evals
│   ├── model_registry.py           # Model selection and pricing
│   ├── core/                       # Local-first runtime orchestration
│   │   ├── __init__.py            # Core exports
│   │   ├── coordinator.py         # RunCoordinator state machine
│   │   ├── db.py                   # SQLite persistence layer
│   │   ├── registries.py           # Versioned prompt/skill/memory management
│   │   ├── model_selection.py      # Model strategy selection
│   │   ├── subagents.py            # Subagent orchestration
│   │   ├── validators.py           # Deterministic validation (RDKit-based)
│   │   ├── external_validation.py  # Future external validator registry
│   │   └── types.py                # Core type definitions
│   ├── api/                        # FastAPI service boundary
│   │   ├── __init__.py            # API exports
│   │   ├── app.py                  # FastAPI application with all endpoints
│   │   └── schemas.py              # API request/response schemas
│   └── ui/                         # Static HTML/JS/CSS frontend
│       ├── index.html              # Main UI with verified/unverified modes
│       ├── app.js                  # Frontend logic and API integration
│       └── styles.css              # UI styling
├── skills/                         # Versioned skill definitions (PR-editable)
│   ├── project/                    # Project-level skills (outside mechanism prediction)
│   │   ├── base_runtime/SKILL.md
│   │   ├── prepare_inputs/SKILL.md
│   │   └── scoring/SKILL.md
│   └── mechanistic/                # Mechanistic prediction assets (replaces prompt_versions/)
│       ├── base_system/SKILL.md    # Shared base system prompt
│       ├── assess_initial_conditions/
│       │   ├── SKILL.md            # Description + I/O + tool schema + prompt text
│       │   └── few_shot.jsonl
│       ├── <other_llm_skills>/     # Same structure per LLM call
│       ├── bond_electron_validation/
│       │   ├── SKILL.md            # Algorithm description + standalone usage
│       │   └── validator.py        # Ground truth implementation
│       └── <other_validators>/     # Same structure per deterministic validator
├── memory_packs/                   # Versioned memory artifacts (PR-editable)
│   ├── default_memory.md           # Default memory context
│   └── reagent_heuristics.json     # Reagent knowledge base
├── harness_versions/                # Pipeline config files (PR-editable)
│   ├── default/
│   │   ├── harness.json            # Schema v2: human-readable + machine-executable
│   │   └── patches/                # Optional harness-specific validator overrides
│   └── no_tools_baseline/
│       └── harness.json            # Control experiment: tool_calling_mode = "none"
├── data/                           # Local SQLite + baseline artifacts
│   ├── mechanistic.db              # Local SQLite database
├── training_data/                  # Eval dataset and user-provided reactions
│   ├── flower_mechanisms_100.json  # Repo-tracked default FlowER mechanism menu + eval source
│   ├── eval_set.json               # Default FlowER-derived eval benchmark
│   ├── eval_tiers.json             # Tiered eval definitions: fixed 10 easy + 10 medium + 10 hard (1-2 / 3 / 4-8 steps)
│   ├── local_legacy/               # Ignored local-only HumanBenchmark + old built-in example files
│   └── my_reactions_template.json  # Minimal template for hand-authored reactions
├── traces/                         # Run traces + PR evidence traces
├── pyproject.toml                  # Project configuration
├── requirements.txt                # Python dependencies
├── AGENTS.md                       # Runtime documentation
└── .cursorrules                    # This file
```

## Key Features

### Execution Modes
- **Verified Mode**: User submits mechanistic steps via API/UI, deterministic validation with RDKit
- **Unverified Mode**: Auto-generates mechanism steps, validates each step

### Core Components
1. **RunCoordinator** (`core/coordinator.py`): State machine orchestrating runs with explicit workflow control
2. **RunStore** (`core/db.py`): SQLite persistence for runs, traces, evaluations, and assets
3. **RegistrySet** (`core/registries.py`): Versioned management of prompt/skill/memory files
4. **Subagent Orchestration** (`core/subagents.py`): Distributed execution of mechanism steps
5. **Deterministic Validators** (`core/validators.py`): RDKit-based validation for mechanism steps
6. **FastAPI Service** (`api/app.py`): REST endpoints for runs, verification, memory, traces, evals
7. **Static Web UI** (`ui/`): HTML/JS/CSS interface with verified/unverified mode selection
8. **Evaluation Scoring** (`scoring.py`): Deterministic mechanism scoring for eval runs and leaderboard generation

### Harness Configuration

Each harness lives in its own subdirectory under `harness_versions/` with a single authoritative `harness.json`:

```
harness_versions/
├── default/
│   ├── harness.json              # schema v2 — human-readable + machine-executable
│   └── patches/                  # optional: harness-specific validator overrides
│       └── <validator_name>.py   # Python module override for ground truth in skills/
└── no_tools_baseline/            # control experiment: no tool calls
    └── harness.json
```

**Schema v2.1 fields** (current default):
- `schema_version: "2.1"` — backward-compat: `"2.0"` harnesses load without `topology_profiles` (defaults to centralized_mas behavior)
- `tool_calling_mode: "forced" | "auto" | "none"` — `"none"` enables no-tools control experiments
- `execution_note` — documents that array order = execution sequence (implicit → explicit)
- `step` integer on each module (1-based explicit ordering)
- `loop_module` — documents the main proposal step (previously hardcoded in coordinator)
- `validator_skill` on deterministic modules — references `skills/mechanistic/<name>/`
- `topology_profiles` — optional map from topology name to `TopologyProfile` config (see **Coordination Topology** below)
- `metadata.changelog` — human-readable change history for the harness

**Module pipeline:**
- **Pre-loop modules** (run once): balance_analysis, functional_groups, ph_recommendation, initial_conditions, missing_reagents, atom_mapping, reaction_type_mapping
- **Loop module**: mechanism_step_proposal (LLM, forced tool call)
- **Post-step modules** (run after each mechanism step): bond_electron_validation, atom_balance_validation, state_progress_validation, reflection, step_atom_mapping
- Each module has: `id`, `label`, `kind` (llm/deterministic/text_completion), `movable`, `removable`, `enabled`, `config_gate`, `step`
- Legacy `RunConfig` flags (`functional_groups_enabled`, `optional_llm_tools`, etc.) map to module enablement via `config_gate`
- The UI Pipeline Editor allows removing, reordering, and adding modules
- New harness variants for eval-backed PRs: add a subdirectory under `harness_versions/` with `harness.json`

**Patch system for validators:** When a harness needs a variant of a deterministic validator (e.g. soft DBE policy as default), add `harness_versions/<harness>/patches/<validator_name>.py`. `HarnessRegistry.load_validator()` checks for a patch first, falls back to `skills/mechanistic/<validator>/validator.py`.

### Coordination Topology

`coordination_topology` is a harness-level knob set at run time via `CreateRunRequest.coordination_topology` (default: `centralized_mas`). It controls how the coordinator calls IntermediateAgent during the mechanism loop proposal step and how results are aggregated.

**TopologyProfile config fields** (inside `topology_profiles` in `harness.json`):

| Field | Type | Meaning |
|---|---|---|
| `agent_count` | int | Number of independent agent calls per loop iteration |
| `max_candidates_per_agent` | int | Max candidates each agent call returns |
| `peer_rounds` | int | Number of debate rounds (0 = no peer context); only used by `decentralized_mas` |
| `aggregation_mode` | str | `"none"` / `"orchestrator_select"` / `"synthesis_only"` / `"consensus"` |
| `consensus_key` | str | Primary field for consensus grouping (default: `reaction_smirks`) |
| `consensus_fallback_key` | str | Secondary grouping field when primary is absent (default: `intermediate_smiles`) |

**Harness JSON example** (`topology_profiles` section):
```json
"topology_profiles": {
  "sas": {
    "agent_count": 1, "max_candidates_per_agent": 1, "peer_rounds": 0,
    "aggregation_mode": "none"
  },
  "centralized_mas": {
    "agent_count": 1, "max_candidates_per_agent": 3, "peer_rounds": 0,
    "aggregation_mode": "orchestrator_select"
  },
  "independent_mas": {
    "agent_count": 3, "max_candidates_per_agent": 2, "peer_rounds": 0,
    "aggregation_mode": "synthesis_only"
  },
  "decentralized_mas": {
    "agent_count": 3, "max_candidates_per_agent": 2, "peer_rounds": 2,
    "aggregation_mode": "consensus",
    "consensus_key": "reaction_smirks",
    "consensus_fallback_key": "intermediate_smiles"
  }
}
```

Harnesses without a `topology_profiles` section (schema v2.0) automatically fall back to `centralized_mas` defaults. To experiment with topology, set `coordination_topology` in the run request — no harness change is needed unless you want to tune per-harness defaults.

**API endpoints:**
- `GET /api/harness/config` (query param `name`, defaults to `default`) — load a harness config
- `POST /api/harness/config` — save a modified harness config
- `GET /api/harness/configs` — list available harness config files

### Evolutionary Architecture
- **Versioned Assets**: Prompt call assets, skills, memories, and harness configs are versioned files editable via PRs
- **Hybrid Storage**: Local SQLite for runtime data + repo artifacts for prompt/traces evidence
- **Feedback Loop**: Human feedback through git PRs modifying prompt bases/few-shot, memories, or harness configs
- **Performance Tracking**: Leaderboard comparing model+harness versions
- **Preloaded Examples**: Dropdown examples serve as initial evaluation set with provisional verified steps

## Development Guidelines

### Running Commands
Always ensure virtual environment is active:
```bash
# Activate environment first
source .venv/bin/activate

# CLI commands
python main.py --help                    # Show CLI help
python main.py run --help               # Show run command help
python main.py serve --help             # Show serve command help

# Run a single mechanistic prediction
python main.py run --starting "CCO" --products "CC=O"

# Run in verified mode (requires user input for steps)
python main.py run --mode verified --starting "CCO" --products "CC=O"

# Start the FastAPI server with web UI
python main.py serve --host 127.0.0.1 --port 8010
```

### Testing
```bash
# Activate environment
source .venv/bin/activate

# Run the fast, free test suite
make test

# Run the full test suite including LLM tests
make test-llm

# Test basic functionality
python -c "from mechanistic_agent.config import ReactionInputs; print('✅ Import successful')"

# Test CLI
mechanistic-agent --help
```

### Dependencies
- **Core**: pydantic, typer, fastapi, uvicorn, sqlalchemy
- **Chemical**: rdkit (install via conda), dimorphite-dl
- **Development**: pytest, ruff, mypy, httpx

### Common Issues & Solutions

#### Virtual Environment Not Active
```bash
# Check if active
which python  # Should show .venv/bin/python

# If not active, activate
source .venv/bin/activate
```

#### Import Errors
```bash
# Ensure in project directory
pwd  # Should show /Users/scottreed/PycharmProjects/Mechanistic

# Reinstall package
pip install -e . --force-reinstall
```

#### RDKit Issues
```bash
# Install via conda (recommended)
conda install -c conda-forge rdkit

# Or via pip
pip install rdkit-pypi
```

## File-Specific Notes

### Core Runtime (mechanistic_agent/core/)
- **`coordinator.py`**: RunCoordinator state machine orchestrating verified/unverified execution
- **`db.py`**: SQLite persistence layer for runs, traces, evaluations, and versioned assets
- **`registries.py`**: Management of versioned prompt/skills/memory files with SHA256 tracking
- **`subagents.py`**: Orchestration of distributed mechanism step execution
- **`validators.py`**: RDKit-based deterministic validation for mechanism steps
- **`model_selection.py`**: Single-model selection that derives a uniform step map from one exact model ID plus optional low/high thinking
- **`external_validation.py`**: Registry pattern for future external validators

### API Layer (mechanistic_agent/api/)
- **`app.py`**: FastAPI application with REST endpoints for runs, verification, memory, traces, evals
- **`schemas.py`**: Pydantic schemas for API request/response validation

### UI Layer (mechanistic_agent/ui/)
- **`index.html`**: Static HTML interface with verified/unverified mode selection
- **`app.js`**: Frontend logic handling API integration and real-time updates
- **`styles.css`**: Modern CSS styling for the web interface

### Configuration & Tools
- **`config.py`**: Reaction input dataclasses with validation
- **`tools.py`**: RDKit/Dimorphite chemical analysis tools + LLM-powered tools with forced tool calling
- **`tool_schemas.py`**: OpenAI-format tool schemas for forced tool calling (text field convention)
- **`scoring.py`**: Deterministic evaluation scoring and leaderboard quality metrics
- **`model_registry.py`**: Model selection and pricing information

### Versioned Assets (Repository Root)
- **`prompt_versions/`**: Versioned prompt assets editable via PRs
- **`skills/`**: Versioned skill definitions for capabilities
- **`memory_packs/`**: Versioned memory artifacts for context
- **`data/`**: Hybrid storage with SQLite + baseline evaluation artifacts

## Quick Start
```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Set API key
export OPENAI_API_KEY=sk-your-api-key-here

# 3. Start the web server
python main.py serve

# 4. Or run a single prediction
python main.py run --starting "CCO" --products "CC=O"

# 5. Run in verified mode (web UI required for step submission)
python main.py run --mode verified --starting "CCO" --products "CC=O"
```

## API Endpoints
When running `python main.py serve`, the following REST endpoints are available:

### Run Management
- `POST /api/runs` - Create new run
- `POST /api/runs/{id}/start` - Start run execution
- `POST /api/runs/{id}/stop` - Stop running run
- `POST /api/runs/{id}/resume` - Resume paused run

### Verified Mode
- `POST /api/runs/{id}/mechanism_steps` - Submit user-verified mechanism steps

### Monitoring & Snapshots
- `GET /api/runs/{id}` - Get run snapshot
- `GET /api/runs/{id}/flow` - Get workflow state
- `GET /api/runs/{id}/events` - Get run events

### Memory & Learning
- `GET /api/memory` - Query memory artifacts
- `POST /api/memory/query` - Semantic search memory
- `POST /api/memory/items` - Add memory items

### Evaluation & Performance
- `GET /api/traces` - Get evaluation traces
- `POST /api/traces/{trace_id}/approve` - Approve traces for learning
- `POST /api/evals/runset` - Run evaluation set
- `GET /api/evals/leaderboard` - Get performance leaderboard

## Adding Your Own Reactions

### How the dropdown is populated

At startup the server loads FlowER-backed examples from `training_data/flower_mechanisms_100.json`
for the default dropdown. The UI groups them by exact step count rather than by source.

Legacy built-in examples and HumanBenchmark artifacts should live only under ignored
`training_data/local_legacy/` paths so they stay off the menu by default.

### Quickstart: add your own reactions

**Option A — hand-write a JSON file** (simplest)

```bash
# Copy the minimal template
cp training_data/my_reactions_template.json training_data/my_project.json
# Edit it, then restart the server — your reactions appear in the dropdown.
python main.py serve
```

Minimum required fields per entry:
```json
[
  {
    "id": "my_unique_slug",
    "name": "My reaction name",
    "starting_materials": ["CCO", "CC(=O)O"],
    "products": ["CCOC(=O)C", "O"]
  }
]
```

For the full template including `run_config` overrides and `verified_mechanism`
seed steps, see `templates/test_case_template.json`.

**Option B — convert an Excel/CSV file** (for bulk datasets)

Use `scripts/convert_training_data.py` to convert a spreadsheet of reaction
SMILES into ready-to-use train/test JSON sets.  Input column must contain
`reactants>>products` SMILES (atom-mapping labels are auto-stripped):

```bash
source .venv/bin/activate

# Regenerate the bundled train/test sets from the included Excel file
python scripts/convert_training_data.py

# Convert your own file (80/20 split by default)
python scripts/convert_training_data.py --excel my_reactions.xlsx

# Custom column names, split ratio, and seed
python scripts/convert_training_data.py \
    --excel my_reactions.xlsx \
    --reaction-col smiles_column \
    --class-col reaction_type \
    --steps-col num_steps \
    --score-col expert_score \
    --train-frac 0.9 --seed 0

python scripts/convert_training_data.py --help   # all options
```

Output files land in `training_data/` and are picked up automatically.

**Option C — import via API** (auto-converts names/InChI to SMILES)

```bash
curl -X POST http://127.0.0.1:8010/api/eval_sets/import_template \
     -H "Content-Type: application/json" \
     -d @training_data/my_reactions_template.json
```

### Running a reaction (dry-run or real)

1. Start the server: `python main.py serve`
2. Open `http://127.0.0.1:8010/ui/`
3. Choose a reaction from the **Select example** dropdown
4. Optional: enable **Display atom numbers** to overlay RDKit atom indices on the rendered molecule cards
5. Optionally enable **Dry Run** (results are discarded after review)
6. Click **Start Run**

For systematic evaluation against the eval set:
```bash
curl -X POST http://127.0.0.1:8010/api/evals/runset \
     -H "Content-Type: application/json" \
     -d '{"eval_set_id": "eval_set", "model_family": "openai"}'
```

See `training_data/README.md` for full details.

## Important Notes
- Always activate virtual environment before running commands
- RDKit installation via conda is recommended over pip
- FastAPI server serves static UI automatically at http://127.0.0.1:8010
- Verified mode requires user step submission via API/UI endpoints
- Unverified mode auto-generates and validates mechanism steps
- Hybrid storage: local SQLite for runtime + repo artifacts for baselines
- Evolutionary design: prompt assets, skills, memories editable via PRs
- Performance tracking via evaluation harness and leaderboard
- Check `AGENTS.md` for comprehensive runtime documentation


## Cursor Cloud specific instructions

### Service overview

This is a single-process Python application (FastAPI + static UI). No Docker, no external database server (SQLite is embedded). The only service to run is `python main.py serve --host 127.0.0.1 --port 8010`.

### Environment activation

Always activate the venv before any Python command: `source .venv/bin/activate`.

### Gotchas

- **`pip install -e .` fails on `rdkit-pypi`**: The `pyproject.toml` lists `rdkit-pypi>=2022.9.5` but the current PyPI package is named `rdkit`. Use `pip install -e . --no-deps` after installing `requirements.txt` (which correctly installs `rdkit`).
- **Test symlink**: `tests/fast/test_model_registry.py` resolves `mechanistic_agent/model_registry.py` relative to `tests/`, so a symlink `tests/mechanistic_agent -> ../mechanistic_agent` must exist. The update script creates it.
- **Pre-existing test failures**: 10 tests in `make test` fail due to missing `training_data/eval_set.json` / `eval_tiers.json` files and API signature mismatches in `test_model_selection.py` and `test_functional_group_context.py`. These are pre-existing in the repo and not caused by the environment setup.
- **UI root path**: The web UI is served at `http://127.0.0.1:8010/` (root), not `/ui/`. Static assets (CSS/JS) are mounted at `/ui/`.
- **LLM tests require API keys**: `make test-llm` needs `OPENAI_API_KEY`. The fast suite (`make test`) does not need any API keys.
- **Lint**: `ruff check .` is the linter. Configuration is in `pyproject.toml` (`line-length = 100`).

## Cursor Cloud specific instructions

### Service overview

This is a single-process Python application (FastAPI + static UI). No Docker, no external database server (SQLite is embedded). The only service to run is `python main.py serve --host 127.0.0.1 --port 8010`.

### Environment activation

Always activate the venv before any Python command: `source .venv/bin/activate`.

### Gotchas

- **`pip install -e .` fails on `rdkit-pypi`**: The `pyproject.toml` lists `rdkit-pypi>=2022.9.5` but the current PyPI package is named `rdkit`. Use `pip install -e . --no-deps` after installing `requirements.txt` (which correctly installs `rdkit`).
- **Test symlink**: `tests/fast/test_model_registry.py` resolves `mechanistic_agent/model_registry.py` relative to `tests/`, so a symlink `tests/mechanistic_agent -> ../mechanistic_agent` must exist. The update script creates it.
- **Pre-existing test failures**: 10 tests in `make test` fail due to missing `training_data/eval_set.json` / `eval_tiers.json` files and API signature mismatches in `test_model_selection.py` and `test_functional_group_context.py`. These are pre-existing in the repo and not caused by the environment setup.
- **UI root path**: The web UI is served at `http://127.0.0.1:8010/` (root), not `/ui/`. Static assets (CSS/JS) are mounted at `/ui/`.
- **LLM tests require API keys**: `make test-llm` needs `OPENAI_API_KEY`. The fast suite (`make test`) does not need any API keys.
- **Lint**: `ruff check .` is the linter. Configuration is in `pyproject.toml` (`line-length = 100`).
