# Mechanistic Agent Leaderboard

Tracks zero-shot and harness-trained performance on the Clawdiators arena challenge set.
Local test scores use `clawdiators-test/scoring/score_submission.py` (same rubric, full speed points).

## Arena Thresholds (mechanistic-easy, 1000 pts max)

| Score | Outcome |
|---|---|
| ≥ 700 | **WIN** |
| 400–699 | **DRAW** |
| < 400 | **LOSS** |

---

## Zero-Shot Baselines

Scores from `clawdiators-test/easy/` (10 new reactions, same type distribution as the arena).
Run: `python local_contributions/clawdiators_test_run.py`
Official-holdout source: `python local_contributions/clawdiators_test_run.py --official-holdout`
Compare-set source by ID: `python local_contributions/clawdiators_test_run.py --eval-set-id <id>`
Score: `python clawdiators-test/scoring/score_submission.py --submission <path>`

| Date | Model | Test Set | Products Correct | Push Quality | Total Score | Predicted Outcome | Notes |
|---|---|---|---|---|---|---|---|
| 2026-03-09 | `gpt-4o-mini` | test_reactions.json | 10/10 | 0.95 | 990 | **WIN** | docker-validated steps=10/10, run=20260309_125630_gpt_4o_mini_docker |
| 2026-03-09 | `gpt-4o-mini` | test_reactions.json | 8/10 | 0.95 | 930 | **WIN** | docker-validated steps=0/10, run=20260309_125343_gpt_4o_mini_docker |
| 2026-03-09 | `gpt-4o-mini` | test_reactions.json | 8/10 | 0.95 | 930 | **WIN** | docker-validated steps=0/10, run=20260309_124138_gpt_4o_mini_docker |
| 2026-03-09 | `gpt-4o-mini` | test_reactions.json | 8/10 | 0.95 | 930 | **WIN** | zero-shot, no-rdkit, run=20260309_112347_gpt_4o_mini |

---

## Arena Submissions (Harness Eval — 1000-pt scale)

*Auto-generated from live leaderboard. Run `python main.py update-leaderboard-artifacts` after eval-runset-official to refresh this table and curriculum/generated/leaderboard_*.json.*

Official Clawdiators leaderboard results from `python main.py eval-runset-official`.
Official one-shot baseline on the same holdout set: `python main.py baseline-runset-official`.
Reproducibility comparison helper: `python main.py compare-eval-runs --run-a <id> --run-b <id>`.
Score computed by `_graded_to_clawdiators_pts()` in `main.py` using the same rubric as the arena.

| Date | Model | Score | Outcome | Pass Rate | Avg Latency | Run Group |
|---|---|---|---|---|---|---|
| 2026-03-16 | `anthropic/claude-opus-4.6` | 100/1000 | LOSS | 0.0% | 6.2s | `official_holdout_harness` |
| 2026-03-15 | `anthropic/claude-opus-4.6` | 100/1000 | LOSS | 0.0% | — | `official_holdout_harness` |
| 2026-03-15 | `anthropic/claude-opus-4.6` | 100/1000 | LOSS | 0.0% | 27.8s | `harness_free_baseline` |
| 2026-03-15 | `anthropic/claude-opus-4.6` | 367/1000 | LOSS | 10.0% | 119.3s | `official_holdout_harness` |

### Speed Calibration

Speed scoring uses `HARNESS_SPEED_CALIBRATION_MS` in `main.py` (top of file, visible in VC).

| Status | Value | Meaning |
|---|---|---|
| **Calibrated (current)** | `4200` | Opus-4.6 benchmark anchor (75 pts at ~4.2s/case) |
| Recalibration | `<measured_avg_ms>` | Update this value when a new official opus benchmark is run |

**Formula:** `T_max = 4 × HARNESS_SPEED_CALIBRATION_MS`
- At `HARNESS_SPEED_CALIBRATION_MS` ms/case → **75 pts** (opus-4.6 target)
- Faster models → > 75 pts; slower models → < 75 pts; >= T_max → 0 pts

**To recalibrate:** Run `python main.py eval-runset-official --model-name claude-opus-4-6`, note the
`avg Xs/case` value in the score summary output, then update `HARNESS_SPEED_CALIBRATION_MS` in `main.py`.

### Why 100/1000 LOSS?

If the leaderboard shows 100/1000 LOSS for a run, it means **pass_rate = 0** (no cases reached target products). The 100 pts come from the methodology component only; product/pathway/push/speed are zeroed when pass_rate is 0. Runs **interrupted with ^C** never get `status=completed` and are filtered out — only completed runs appear. To see your high-scoring runs, let `eval-runset-official` finish without interrupting.

### Harness Proxy Mapping

Harness rubric dimensions use proxies (not identical to single-shot zero-shot scoring):

| Dimension | Zero-Shot Source | Harness Proxy |
|---|---|---|
| Product Accuracy (30%) | Exact SMILES match | `final_product_reached` count / total cases |
| Pathway Coverage (30%) | Step count + intermediate Jaccard | `known_alignment_component` avg (step alignment) |
| Electron Push Quality (20%) | Push type (lp/sigma/pi) Jaccard | `step_validity_component` avg (validation + atom mapping) |
| Speed (10%) | Linear decay 0→600s | Per-case wall-clock latency via `_latency_to_speed_pts()` |
| Methodology (10%) | Non-empty string | Always 100 pts (harness always has methodology) |

---

## Lessons Learned (from clawdiators-test build)

Building the 10-reaction test harness revealed the following scoring dynamics:

### Score Structure (v2 — with electron pushes)
| Dimension | Weight | Max Points | Notes |
|---|---|---|---|
| Product Accuracy | 30% | 300 | Primary gate — correctness of final SMILES |
| Pathway Coverage | 30% | 300 | Step count + intermediate Jaccard |
| Electron Push Quality | 20% | 200 | Push type (lp/sigma/pi) Jaccard — partial credit |
| Speed | 10% | 100 | Linear decay 0→600 sec (full in local mode) |
| Methodology | 10% | 100 | Any non-empty string = full credit |

- **100 pts free**: Any non-empty methodology string → 100 pts. Never miss this.
- **Anti-gaming gate**: If `final_products` has 0 correct, pathway, electron_push, AND speed are all forced to 0. Getting any product correct unlocks 600 pts.
- **Electron push partial credit**: Push type distribution scoring (lp/sigma/pi counts) — agents without atom maps can still earn ~50-80% on this dimension by classifying reaction types correctly.
- **Product accuracy is the bottleneck**: Requires correct SMILES after RDKit canonicalization.

### Reaction Type Insights (eval set distribution)
- **3/10 are 1-step SN2**: Nucleophile (N or P) lone pair + alkyl halide → quaternary salt + X⁻. `steps` has 1 entry.
- **2/10 are 2-step**: Epoxide ring opening (SN2 + proton transfer) and Arbuzov reaction (2x SN2). `steps` has 2 entries with ionic intermediate.
- **2/10 are N-oxidation**: N-heterocycle + peracid → N-oxide + carboxylic acid. 1-step, 1 lp push.
- **1/10 Diels-Alder**: 3 concerted pushes (2pi + 1sigma). 1-step.
- **1/10 Ene reaction**: 3 concerted pushes (pi + sigma + pi). 1-step.
- **1/10 Hetero DA**: 3 concerted pushes (3pi). 1-step.

### Electron Push Type Patterns
| Reaction Type | n_steps | Push Types per Step |
|---|---|---|
| SN2 | 1 | `lp` + `sigma` |
| N-oxidation | 1 | `lp` |
| Diels-Alder | 1 | `pi` + `pi` + `sigma` |
| Ene reaction | 1 | `pi` + `sigma` + `pi` |
| Hetero DA | 1 | `pi` + `pi` + `pi` |
| Epoxide ring opening | 2 | Step1: `lp`+`sigma`; Step2: `lp`+`sigma` |
| Arbuzov reaction | 2 | Step1: `lp`+`sigma`; Step2: `lp`+`sigma` |

### SMILES Canonicalization
- Products can be submitted as a single dot-joined SMILES (`"A.B"`) or separate strings — scorer accepts both.
- Order of species in dot-joined SMILES doesn't matter (scorer sorts fragments).
- Invalid SMILES → 0 for that reaction. Validate with Docker validator or RDKit before submitting.

### Speed Strategy
- Arena: linear decay 1.0 → 0.0 over 600 seconds. 60-second single-shot submission → ~90/100 on speed.
- Local test: always full 100 pts (speed not penalized in `score_submission.py`).

---

## Updating Prompts and Few-Shot Examples

`eval-runset-official` is **read-only** for prompts and few-shot files. To mine and apply new few-shot examples from eval runs:

```bash
python scripts/evolve_harness.py --model-name anthropic/claude-opus-4.6 --harness default
```

Options: `--thinking-level` (low/high/max/auto; default: highest for model), `--step-count` (1–8 or `mixed`), `--loop` (when no candidates remain, allow repeats and continue). Use `--dry-run` to preview changes without writing. This runs the curriculum batch, mines high-scoring traces, and appends them to `skills/mechanistic/<call_name>/few_shot.jsonl`. For prompt edits, update `SKILL.md` files directly; see [CONTRIBUTING.md](CONTRIBUTING.md) for evidence gates.

---

## Planned First Entry

- Model: `anthropic/claude-opus-4-6`
- Thinking: none (zero-shot baseline)
- Type: Zero-shot (`local_contributions/clawdiators_test_run.py`)
- Group: `zero_shot_opus46_baseline`
