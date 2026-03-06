## Track 2: Subagent PR

### Summary
- Subagent name:
- Type: `deterministic` / `llm_backed` / `validator`
- Problem solved:
- Why the existing system is insufficient:

### Files changed
- Core implementation:
- Skill directory, if any:
- Tool schema, if any:
- Coordinator or harness wiring:

### Required tests
- [ ] `python -m pytest tests/fast/ -q`
- [ ] `PYTHONPATH=. pytest tests/fast/test_<subagent_name>.py`
- [ ] `PYTHONPATH=. pytest tests/llm/test_eval_tiers.py --tier medium -k medium`

### Recommended tests
- [ ] `PYTHONPATH=. pytest tests/llm/test_eval_tiers.py --tier hard -k hard`

### Leaderboard delta
- Baseline medium-tier row:
- Candidate medium-tier row:
- Medium-tier improvement:
- Hard-tier delta, if run:
- Per-subagent change summary:

### Checklist
- [ ] Fast tests added for success and failure paths
- [ ] Tool fallback handled if LLM-backed
- [ ] Coordinator and harness wiring updated where needed
- [ ] Medium tier improved
- [ ] Contact info included below

### Contact info for possible manuscript attribution
- Name:
- Email:
- ORCID / GitHub / preferred handle:
- Contact before attribution? `yes` / `no`
