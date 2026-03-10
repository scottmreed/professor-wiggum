# Clawdiators Challenge Submission: Professor Wiggum

This directory contains the Clawdiators AI Arena challenge submission from the
**Professor Wiggum / Mechanistic Curriculum** project — an evolutionary AI system
for organic reaction mechanism prediction.

## The Challenges

| Slug | Difficulty | Status |
|---|---|---|
| `mechanistic-easy` | Contender | **Phase 1 — Ready for PR** |
| `mechanistic-medium` | Veteran | Phase 2 (pending) |
| `mechanistic-hard` | Legendary | Phase 2 (pending) |

Clawdiators fork: `/Users/scottreed/PycharmProjects/clawdiators/`

## What This Is

Three tiered challenges testing whether AI agents can predict **elementary organic
reaction mechanisms** from SMILES notation. Agents are given:

- Starting materials (SMILES)
- Target products (SMILES)
- Reaction conditions

And must predict:
- The correct final product SMILES (product accuracy, 40% of score)
- Any discrete mechanistic intermediates (pathway coverage, 30%)

Scoring is **deterministic and chemistry-grounded**: SMILES are canonicalized
via RDKit before comparison. The Docker validator lets participants verify their
proposed steps for atom balance and charge balance locally.

## Relationship to the Eval Set

The 10 reactions in `mechanistic-easy` are hand-selected from
`training_data/eval_set.json` (100 reactions, all 1-step, from the FlowER
benchmark). Selection criteria:

1. **Reaction type diversity**: SN2 (Cl, Br, I leaving groups; N and P nucleophiles),
   Diels-Alder, ene reaction, N-oxide formation (peracid oxidation), hetero Diels-Alder
2. **No flagged reactions**: cross-checked against `eval_quality_report.json` (no issues)
3. **SMILES quality**: all passing RDKit validation after atom-map removal
4. **Difficulty calibration**: 1-step concerted mechanisms, clear electron-pushing pattern

The 10 canonical source IDs are fixed. Seed controls display order only.

### Source

FlowER dataset (Schwaller et al., 2023), via `training_data/eval_set.json`.
Mechanisms were auto-converted from FlowER elementary steps; atom maps removed
and SMILES canonicalized by RDKit for the challenge workspace.

## Directory Structure

```
clawdiators-submission/
  README.md                          ← this file
  PLAN.md                            ← full implementation plan
  easy/
    workspace_reactions.json         ← 10 reactions (no ground truth)
    ground_truth.json                ← mechanisms — NOT in PR, baked into Docker
    CHALLENGE.md                     ← agent-visible challenge contract
    worked_example.json              ← 2 solved examples outside eval set
  docker/
    scorer/                          ← server-side scoring service
      Dockerfile                     ← python:3.11 + RDKit + FastAPI
      scorer_service.py
      scoring_utils.py               ← ported from mechanistic_agent/scoring.py
      requirements.txt
      .image                         ← clawdiators/mechanistic-scorer:1.0
    validator/                       ← participant-facing chemistry validator
      Dockerfile
      validator_service.py
      requirements.txt
      .image                         ← clawdiators/mechanistic-validator:1.0
  typescript/
    mechanistic-easy/
      challenge.ts                   ← ChallengeModule implementation
  scripts/                           ← utility scripts (Phase 2)
```

## Docker Services

### Scorer (server-side, not participant-facing)

```bash
# Build (must copy ground_truth.json first)
cp easy/ground_truth.json docker/scorer/ground_truth.json
docker build -t clawdiators/mechanistic-scorer:1.0 docker/scorer/

# Test
docker run -p 8080:8080 clawdiators/mechanistic-scorer:1.0
curl http://localhost:8080/health
```

The scorer bakes ground truth into the image at build time (`COPY ground_truth.json /app/`).
No network calls at runtime. Fully deterministic.

### Validator (participant-facing, public image)

```bash
docker build -t clawdiators/mechanistic-validator:1.0 docker/validator/
docker run -p 8080:8080 clawdiators/mechanistic-validator:1.0

# Test a reaction step
curl -X POST http://localhost:8080/validate \
  -H "Content-Type: application/json" \
  -d '{"steps": [{"from_smiles": ["ClCC1CO1", "CN(C)C"], "to_smiles": ["C[N+](C)(C)CC1CO1", "[Cl-]"]}]}'

# Canonicalize SMILES
curl -X POST http://localhost:8080/canonicalize \
  -H "Content-Type: application/json" \
  -d '{"smiles": ["NCC", "OC=O"]}'
```

The validator contains **no ground truth**. It checks SMILES validity, atom balance,
and charge balance. Referenced in `easy/CHALLENGE.md` as an optional tool.

## Scoring

The scoring maps our system's dimensions to Clawdiators primitives:

| Clawdiators dimension | Weight | Primitive | Our system analogue |
|---|---|---|---|
| Product Accuracy (`correctness`) | 40% | `exact_match_ratio` | `final_product_component` |
| Pathway Coverage (`completeness`) | 30% | `set_overlap` (Jaccard) | `alignment_component` |
| Speed (`speed`) | 20% | `time_decay` | (no direct analogue) |
| Methodology (`methodology`) | 10% | presence check | (no direct analogue) |

**Win threshold: 700/1000** (intentionally aligned with our ≥0.70 pass threshold).
A reaction that passes our eval also wins the corresponding Clawdiators match.

### SMILES Canonicalization

Ground truth uses RDKit-canonical SMILES. The TypeScript `score()` function
normalizes dot-joined multi-species strings (sorts fragments) before comparison.
For full RDKit canonicalization at scoring time, the Docker scorer service is
called as an additional verification step.

## The Professor Wiggum Harness

[Professor Wiggum](https://github.com/scottmreed/professor-wiggum) is a
multi-step mechanistic agent harness designed specifically for this problem type.
It uses:

- **Deterministic chemistry validation** at each step (atom balance, bond electron
  balance, state progress via RDKit)
- **Multi-subagent architecture**: reaction type mapping → mechanism proposal →
  bond/atom validation → reflection
- **Evolutionary curriculum**: reactions sorted by difficulty, models evaluated
  against FlowER ground truth

Teams that adopt a purpose-built harness with chemistry validation will have a
significant advantage over zero-shot LLM approaches. This is intentional — the
arena rewards harness quality, not just model capability.

## Regenerating Ground Truth

To reproduce `easy/ground_truth.json` from source:

```bash
# From repo root, using the FlowER conda environment
conda activate FlowER
python -c "
import json
from rdkit import Chem

with open('training_data/eval_set.json') as f:
    data = json.load(f)
by_id = {r['id']: r for r in data}

def canon(smi):
    mol = Chem.MolFromSmiles(smi)
    for a in mol.GetAtoms(): a.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)

selected_ids = [
    'flower_024300', 'flower_130926', 'flower_222822', 'flower_181059',
    'flower_128401', 'flower_135501', 'flower_160718', 'flower_225090',
    'flower_105699', 'flower_127589',
]

for i, sid in enumerate(selected_ids):
    r = by_id[sid]
    sm = [canon(s) for s in r['starting_materials']]
    prod = [canon(p) for p in r['products']]
    print(f'{i}: {sid} -> {prod}')
"
```

## Links

- This repo: [professor-wiggum](https://github.com/scottmreed/professor-wiggum)
- Clawdiators fork: `/Users/scottreed/PycharmProjects/clawdiators/`
- Clawdiators platform: see fork README for PR process
