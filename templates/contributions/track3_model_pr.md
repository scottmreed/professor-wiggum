## Track 3: Model PR

### Summary
- Model ID:
- Family:
- Provider:
- Cost class:
- Why this model is worth adding:

### Files changed
- `mechanistic_agent/model_pricing.json`
- `mechanistic_agent/llm.py`:
- Additional tests:

### Required tests
- [ ] `python -m pytest tests/fast/ -q`
- [ ] `PYTHONPATH=. pytest tests/fast/test_model_registry.py`
- [ ] `PYTHONPATH=. pytest tests/llm/test_eval_tiers.py --tier easy -k easy`

### Leaderboard delta
- Current easy-tier SOTA in this cost class:
- Candidate result:
- Improvement shown:

### Capability notes
- Supports forced tools? `yes` / `no`
- Text fallback required? `yes` / `no`
- Reasoning support:
- Context window:

### Checklist
- [ ] Model catalog entry complete
- [ ] Adapter added or verified
- [ ] Easy-tier result improves current cost-class SOTA
- [ ] Contact info included below

### Contact info for possible manuscript attribution
- Name:
- Email:
- ORCID / GitHub / preferred handle:
- Contact before attribution? `yes` / `no`
