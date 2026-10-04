---
name: bridge-responder
description: Run keyless agent-bridge evals, baselines or single runs in this repo with fresh blind Claude Code subagents standing in as the model, then audit the subagent transcripts so contaminated results are disregarded. Use whenever you drive MECHANISTIC_AGENT_BRIDGE_DIR / agent-bridge runs by dispatching Agent subagents (orchestrator_subagents responder), or need to check whether a bridge run's answers were blind.
---

# Bridge responder: blind subagents + integrity audit

The keyless agent bridge (`mechanistic_agent/agent_bridge.py`, [docs/agent_bridge.md](../../../docs/agent_bridge.md)) writes each model call to `requests/<stem>.json` and waits for `responses/<stem>.json`. In a Claude Code session you answer each call with a **fresh blind subagent** that sees only that call's prompt. `scripts/bridge_responder.py` does the file work and audits the subagents afterwards.

A result counts as a blind capability measurement only if the subagent read nothing but its `prompt.md`. A subagent that greps the repo, `training_data/`, the DB or the web for the reference mechanism has contaminated the run. The audit catches this, and its verdict decides whether the result is used or disregarded.

## 1. Layout and environment

Keep everything in the session scratchpad, never in the repo:

```
<scratch>/<trial>/bridge/     # MECHANISTIC_AGENT_BRIDGE_DIR (requests/, responses/)
<scratch>/<trial>/calls/      # one dir per call: prompt.md, answer.json (sibling of bridge/)
```

Environment for the eval process. Set it inline or in a launch script, because env vars do not persist across Bash calls:

```bash
export MECHANISTIC_AGENT_BRIDGE_DIR=<scratch>/<trial>/bridge
export MECHANISTIC_ACTIVE_MODEL=agent-bridge              # every LLM step goes through the bridge
export MECHANISTIC_AGENT_BRIDGE_TIMEOUT=3600
export MECHANISTIC_AGENT_BRIDGE_DECLARED_MODEL=claude-opus-5-5   # the model the subagents run on
export MECHANISTIC_AGENT_BRIDGE_RESPONDER_KIND=orchestrator_subagents
export MECHANISTIC_AGENT_BRIDGE_SAW_GROUND_TRUTH=false     # true only for deliberate replays
export MECHANISTIC_AGENT_BRIDGE_NOTES="fresh blind Claude Code subagent per call; sees only model_input"
export OPENAI_API_KEY=blocked ANTHROPIC_API_KEY=blocked OPENROUTER_API_KEY=blocked   # no hosted fallback
```

Point `MECHANISTIC_DATA_DIR` at the data checkout you mean to write to (see `docs/DATA_SETUP.md`).

## 2. Launch the runs (background)

The eval blocks while it waits for responses, so run it with `run_in_background`:

```bash
python main.py eval --eval-set-id <id> [--case-id ...] --harness <harness> \
  --model-name anthropic/claude-opus-5.5 --run-group <group> --max-steps 14 > <scratch>/<trial>/log.txt 2>&1
# harness-free baseline:
python main.py baseline --tier easy --model-name anthropic/claude-opus-5.5 ...
```

## 3. Serve and dispatch

Run the responder loop in the background and watch its output with Monitor:

```bash
python scripts/bridge_responder.py serve --bridge-dir <scratch>/<trial>/bridge \
  --transcripts <session tasks dir>
```

`serve` repeats `cycle` every 3 s. Each pass preps `calls/<stem>/prompt.md` from `model_input` only (never the request's `context`), validates every `answer.json` against the forced tool's required keys, writes the response, and prints:

| Line | Meaning | Your action |
|---|---|---|
| `DISPATCH <stem>` | a call needs an answer (new, or the previous answer was rejected) | dispatch one fresh subagent |
| `RESPONDED <stem>` | answer accepted and handed to the harness | none |
| `BAD <stem>: ...` | answer unparseable or missing required keys; moved to `answer.bad.*.json` | a `DISPATCH` follows |
| `CONTAMINATED <stem> (<agent>)` | that subagent's transcript already shows contamination; answer rejected | a `DISPATCH` follows; note it |
| `PROCEDURAL <stem> (<agent>)` | rule slip that only touched its own files; answer accepted | note it |
| `FAILED <stem>` | `--max-attempts` (default 3) used up; error response written, the step fails fast | report it |

For every `DISPATCH <stem>`:

1. Get the exact prompt: `python scripts/bridge_responder.py prompt <stem> --bridge-dir <scratch>/<trial>/bridge`.
2. Launch **one new** `Agent` (general-purpose, `run_in_background: true`, the model you declared) with **exactly that text** as its prompt. Add nothing: no hints, no reaction context, no repo paths, no expected answer.
3. Never reuse a subagent (no `SendMessage`) and never have one subagent answer two calls. Do not read the subagent's answer and "fix" it yourself, because the answer must be the subagent's own.

Budgets: keep about 6 subagents in flight at most. Expect 40 to 140 s per call and 15 to 20 calls per 4 to 6 step case. If the account hits a usage limit, stop dispatching. Pending calls then time out and fail loudly, and you re-run those cases later. Never switch to a headless `claude -p` responder.

## 4. Mandatory audit after the runs finish

Once the eval processes have exited, audit every call before you report or use any result:

```bash
python scripts/bridge_responder.py audit \
  --bridge-dir <scratch>/<trial>/bridge \
  --transcripts <session tasks dir> \
  --json <scratch>/<trial>/audit.json --mark
```

- `<session tasks dir>` is the directory of the `output_file` that each background Agent call returns, e.g. `/private/tmp/claude-<uid>/<project>/<session-id>/tasks/`. Its `*.output` files link to `~/.claude/projects/<project>/<session-id>/subagents/agent-*.jsonl`. Pass several `--transcripts` and `--calls-dir` values to audit several trials at once.
- Each transcript is matched to its call by the `prompt.md` path in its first prompt. Every tool call is then classified:
  - **allowed**: Read of that call's `prompt.md`, Write of its `answer.json`, `SubagentHandback`.
  - **procedural**: touches only that call's own files in a forbidden way, for example a Bash heredoc into `answer.json`, `python3 -c json.load(open(answer.json))`, an Edit of `answer.json`, or re-reading the answer.
  - **contamination**: any other path (repo, `training_data/`, `wiggum-data`, `*.db`, `traces/`, `results/`, another call's files, a relative path, `~`), any command outside a small allow-list (`ls`, `grep`, `find`, `cd`, `git`, `sqlite3`, `curl` ...), python that imports `os`/`glob`/`subprocess`/`sqlite3`/`rdkit`/`mechanistic_agent` or opens other files, WebFetch/WebSearch, Agent/Task spawns, MCP tools, and any unknown tool.
- Per call verdict: `clean`, `procedural`, `contaminated`, or `unaudited` (answered, but no transcript found). The exit code is 1 when anything is contaminated.
- `--mark` (explicit opt-in) writes `responder_integrity = {status, audited_at, violations, ...}` onto each attributed run's `config.origin` and onto its eval run's `metadata`. Runs are attributed through the request's top-level `context.run_id`. Without `--mark` the audit reads the DB read-only, only to show eval case ids.

## 5. Acting on the verdicts

- **clean**: use the result.
- **procedural**: the result is usable. Mention the slip in your report, and tighten the dispatch if it keeps happening.
- **unaudited**: do not claim the result is blind. Find the missing transcript or re-run the case.
- **contaminated**: disregard the run. After `--mark`, `RunStore.leaderboard` skips it, `prompt_trace_validator` rejects evidence exported from it, and `results_publish` refuses to publish it, all through `agent_bridge.origin_integrity_contaminated`. Tell the user which call, subagent and tool call caused it, and re-run the affected cases with fresh subagents.

Calls with no `context.run_id` (requests written before attribution existed, or calls outside a run step) still get a verdict. They cannot be marked, so record the verdict in your report.
