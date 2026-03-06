## Track 4: Harness PR

See [docs/harness_cookbook.md](../../docs/harness_cookbook.md) for step-by-step instructions on removing steps, adding steps, and evaluating your changes.

### Summary
- Harness name:
- Modules changed (added / removed / reordered):
- Hypothesis (why this pipeline configuration should perform differently):

### Files changed
- `harness_versions/<name>/harness.json`
- Patch files, if any (`harness_versions/<name>/patches/<validator_name>.py`):
- New skill files, if any (`skills/mechanistic/<call_name>/SKILL.md`):

### Required tests
- [ ] `python -m pytest tests/fast/ -q`
- [ ] `PYTHONPATH=. pytest tests/fast/test_harness_config.py`

### Local validation
- [ ] Dry-run reaction checks completed
- Dry-run notes:

### Eval delta (optional but encouraged)
If you ran the eval set against your harness, include a before/after comparison:
- Baseline harness:
- Candidate harness:
- Eval set used:
- Leaderboard delta (quality score, pass rate, or per-subagent breakdown):

No minimum tier improvement is required for Track 4. A neutral or inconclusive result is a valid contribution if the harness design and rationale are clear.

### Checklist
- [ ] Harness saved as a named variant under `harness_versions/<name>/`
- [ ] `metadata.changelog` updated in the harness JSON
- [ ] Module changes described precisely above
- [ ] For new steps: skill file exists with `<!-- PROMPT_START -->` / `<!-- PROMPT_END -->` markers
- [ ] Contact info included below

### Contact info for possible manuscript attribution
- Name:
- Email:
- ORCID / GitHub / preferred handle:
- Contact before attribution? `yes` / `no`
