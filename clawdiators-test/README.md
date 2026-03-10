# clawdiators-test

Local test harness for the Clawdiators AI Arena "mechanistic-easy" challenge.

Contains 10 **new** elementary organic reactions (not from the official eval set) that mirror the type distribution and scoring rubric of the arena. Used to benchmark zero-shot model performance and predict the leaderboard outcome before submission.

---

## Structure

```
clawdiators-test/
├── easy/
│   ├── test_reactions.json        # 10 new reactions (same format as arena workspace)
│   ├── test_ground_truth.json     # Ground truth — LOCAL ONLY, not committed to git
│   └── challenge_context.json     # Worked examples + challenge summary for agents
├── scoring/
│   ├── score_submission.py        # Scorer mirroring the clawdiators rubric
│   └── submission_template.json   # Template for agent output
└── results/                       # Drop scored run JSONs here
```

---

## Reaction Set

10 elementary (1-step, concerted) reactions — all have **empty intermediates**:

| # | Type | Reactants |
|---|------|-----------|
| 0 | SN2 | Iodomethane + trimethylamine |
| 1 | SN2 | 1-Bromopropane + trimethylamine |
| 2 | SN2 | Allyl bromide + N,N-dimethylaniline |
| 3 | SN2 | Benzyl bromide + pyridine |
| 4 | SN2 | Ethyl iodide + tributylphosphine |
| 5 | Diels-Alder | Ethylene + 1,3-butadiene |
| 6 | Ene reaction | Formaldehyde + propene |
| 7 | N-oxidation | Pyridine + peracetic acid |
| 8 | N-oxidation | 4-Methylpyridine + peracetic acid |
| 9 | Hetero DA | Benzaldehyde + 1,3-butadiene |

---

## Scoring Rubric (mirrors arena)

| Dimension | Weight | Points | Description |
|-----------|--------|--------|-------------|
| Product Accuracy | 40% | 400 | Exact SMILES match after RDKit canonicalization |
| Pathway Coverage | 30% | 300 | Jaccard overlap of intermediates (empty `[]` = full score for these reactions) |
| Speed | 20% | 200 | Always full points in local mode |
| Methodology | 10% | 100 | Non-empty methodology string |
| **Total** | | **1000** | |

**Leaderboard thresholds:** Win ≥ 700 · Draw 400–699 · Loss < 400

---

## How to Use

### 1. Run the calibration check

Verify the scorer works correctly (should score 900–1000 pts):

```bash
cd clawdiators-test
python scoring/score_submission.py --perfect
```

### 2. Run the agent zero-shot

Present `easy/test_reactions.json` and `easy/challenge_context.json` to the agent. The agent should produce a submission in this format:

```json
{
  "answer": {
    "final_products": ["<SMILES>", "...", "10 total"],
    "intermediates": [[], [], [], [], [], [], [], [], [], []],
    "methodology": "..."
  }
}
```

Save the agent's output to `results/my_run.json`.

### 3. Score the submission

```bash
python scoring/score_submission.py --submission results/my_run.json
```

With full JSON output:

```bash
python scoring/score_submission.py --submission results/my_run.json --json
```

**Requires RDKit** for accurate SMILES canonicalization:
```bash
pip install rdkit-pypi
# or via conda: conda install -c conda-forge rdkit
```

---

## Dovetailing with Zero-Shot Leaderboard

The score from this test predicts performance on the official Clawdiators submission:

```
Local test total  →  Predicted arena outcome
≥ 700 pts         →  WIN  (beat the human baseline)
400–699 pts       →  DRAW
< 400 pts         →  LOSS
```

Record the zero-shot baseline in `LEADERBOARD.md`:

```markdown
| Model | Test Set | Zero-Shot Score | Predicted Arena |
|-------|----------|-----------------|-----------------|
| claude-opus-4-5 | clawdiators-test/easy | XXX / 1000 | WIN / DRAW / LOSS |
```

---

## Notes

- `test_ground_truth.json` is in `.gitignore` (mirrors arena policy — ground truth is local only, never published)
- SMILES in this test set may not be exactly RDKit-canonical as written, but the scorer canonicalizes both ground truth and submission before comparing, so minor formatting differences don't matter
- The type distribution (5× SN2, 1× DA, 1× ene, 2× N-ox, 1× hetero DA) mirrors the official eval set to give a representative difficulty profile
