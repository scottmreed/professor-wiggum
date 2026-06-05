## Track 1: Few-Shot PR

### Summary
- Call name:
- Reaction class:
- Why this example should become permanent:

### Files changed
- `skills/mechanistic/<call_name>/few_shot.jsonl`

### Evidence
- Approved trace path:
- Trace ID:
- Prompt bundle SHA:
- Model used:

### Tests run
- [ ] `PYTHONPATH=. python scripts/validate_prompt_trace_evidence.py --call <call_name>`
- [ ] `python -m pytest tests/fast/ -q`
- [ ] `PYTHONPATH=. pytest tests/llm/test_eval_tiers.py --tier medium -k medium`

### Leaderboard delta
- Baseline leaderboard row:
- Candidate leaderboard row:
- Medium-tier improvement:

### Execution origin
- Execution channel: `direct_model` (hosted API key) or `agent-bridge` (keyless)
- If `agent-bridge` — responder kind: `orchestrator_subagents` / `cli` / `script` / `replay`
- Declared underlying model/agent:
- Budget observability: `full` (hosted, real cost) or `opaque` (agent-bridge)
- Official holdout exposed to the responder? `no`

### Checklist
- [ ] Added serialized `input` and `output` strings only
- [ ] Linked approved evidence
- [ ] Medium tier improved
- [ ] Contact info included below

### Contact info for possible manuscript attribution
- Name:
- Email:
- ORCID / GitHub / preferred handle:
- Contact before attribution? `yes` / `no`
