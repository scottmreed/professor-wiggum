# Clawdiators Challenge Submission Plan
## Professor Wiggum → Clawdiators AI Arena

**Output location**: `clawdiators-submission/` (version-controlled in this repo)
**Clawdiators fork**: https://github.com/clawdiators-ai/clawdiators (submitted PR)

---

## Context

Professor Wiggum (Mechanistic Curriculum) is an evolutionary AI system for organic
reaction mechanism prediction. Our eval set — FlowER-derived reactions with verified
step-by-step mechanisms — is a rigorous benchmark that no existing public arena
challenge captures. Organic reaction mechanism prediction is real-world chemistry
reasoning: multi-step, deterministically verifiable, and deeply hard for current LLMs.

Submitting 3 difficulty-tiered challenges to Clawdiators gives us a public, competitive
venue, cross-harness benchmarking, and community visibility — while keeping our harness
as a competitive edge (not a requirement). The arena accumulates first-attempt verified
scores that serve as clean cross-agent comparisons of both model capability and harness
quality.

---

## Path Decision: PR Path (required)

| Factor | API Path | PR Path — **Our choice** |
|---|---|---|
| Language | JavaScript (ES5) | TypeScript ✓ |
| Sandbox | VM, 5s timeout, no imports | Full Node.js ✓ |
| Services | None | Docker Compose ✓ (we need RDKit) |
| Code visibility | Private (DB) | Transparent (repo) ✓ |
| Reproducibility | N/A | Auto-encrypted, pure-function scoring ✓ |

**Reason**: Our scorer requires Python + RDKit. The API path's sandboxed VM cannot
run external processes or imports. The PR path's Docker Compose support is the
only viable route for deterministic chemistry validation.

**Reproducibility requirement** (PR path): The scorer must be a pure function of
`(submission, groundTruth)` — no external randomness, no network calls, no state.
Ground truth is embedded in the Docker image at build time. Same inputs → same score,
always. This satisfies Clawdiators' reproducibility methodology and score encryption.

---

## Pre-work: Eval Set Status

**Current state** (`training_data/eval_tiers.json`):
- `easy`: All 100 reactions in the pool (1–2 step reactions)
- `medium`: empty — not yet built
- `hard`: empty — not yet built

**Phase 1 (Easy) uses**: 10 hand-selected reactions from the existing 100-reaction easy pool.
**Phase 2 (Medium/Hard)**: Build from `flower_mechanisms_multistep.json` using existing
scripts (`build_flower_mechanism_dataset.py`, `build_flower_holdout_leaderboard_set.py`).

---

## Reaction Diversity Requirement (Easy Tier — 10 reactions)

The 10 selected easy reactions must span distinct reaction types AND functional groups.
Select from `training_data/eval_set.json` (100 reactions, 1–2 steps each).

Target distribution:

| # | Reaction type | Functional group(s) |
|---|---|---|
| 1 | SN2 substitution | Alkyl halide + amine |
| 2 | Nucleophilic addition to aldehyde | Aldehyde + hydride |
| 3 | Nucleophilic addition to ketone | Ketone + organometallic |
| 4 | Acid/base proton transfer | Carboxylic acid + amine |
| 5 | Nucleophilic acyl substitution | Ester + amine |
| 6 | Carbonyl reduction (NaBH₄) | Ketone → alcohol |
| 7 | Electrophilic addition | Alkene + HX |
| 8 | E2 elimination | Beta-halo amine/alcohol |
| 9 | Enolization | Alpha-carbonyl |
| 10 | Oxidation | Alcohol → carbonyl |

Verify selection against `eval_set_pngs/` and `eval_quality_report.json` to avoid
flagged reactions. The 10 IDs are fixed in ground truth; seeds only control display order.

---

## Directory Structure: `clawdiators-submission/`

New top-level directory in professor-wiggum repo (version controlled, not gitignored).

```
clawdiators-submission/
  README.md                         # Relates this submission to Professor Wiggum
  easy/
    workspace_reactions.json        # 10 reactions (no ground truth, no hints)
    ground_truth.json               # Known mechanisms — NOT published, used for Docker build
    CHALLENGE.md                    # Agent-visible contract (final draft)
    worked_example.json             # One solved reaction OUTSIDE eval set
  medium/                           # Phase 2
    workspace_reactions.json
    ground_truth.json
    CHALLENGE.md
    worked_example.json
  hard/                             # Phase 2
    workspace_reactions.json
    ground_truth.json
    CHALLENGE.md
    worked_example.json
  docker/
    scorer/
      Dockerfile
      scorer_service.py
      scoring_utils.py              # Ported from mechanistic_agent/scoring.py
      requirements.txt
      .image                        # clawdiators/mechanistic-scorer:1.0
    validator/
      Dockerfile
      validator_service.py
      requirements.txt
      .image                        # clawdiators/mechanistic-validator:1.0
  typescript/
    mechanistic-easy/
      challenge.ts                  # ChallengeModule implementation
    mechanistic-medium/             # Phase 2
      challenge.ts
    mechanistic-hard/               # Phase 2
      challenge.ts
```

`ground_truth.json` files are committed to VC in this private repo but will NOT be
published in the PR. They are baked into the Docker image at build time via
`COPY ground_truth.json /app/` in the Dockerfile. The built image tag is in `.image`.

The `clawdiators-submission/README.md` explains:
- This is the Professor Wiggum challenge submission for Clawdiators
- Link to https://github.com/clawdiators-ai/clawdiators/README.md (submitted PR)
- How the eval set was built (FlowER dataset, PMechDB, deterministic tiers)
- How the Docker scorer relates to `mechanistic_agent/scoring.py`
- How to regenerate ground truth from source data

---

## Scoring Design: Using Platform Primitives

Three scoring primitives from `primitives/scoring.ts`:

### Dimensions (weights sum to 1.0)

| Dimension | Key | Weight | Primitive | Description |
|---|---|---|---|---|
| **Product Accuracy** | `product_accuracy` | 0.40 | `exact_match_ratio` | Fraction of reactions where submitted final product SMILES exactly matches known product |
| **Pathway Coverage** | `pathway_coverage` | 0.30 | `set_overlap` (intersection) | Recall of correct intermediates: how many known intermediates appear in submitted mechanism |
| **Speed** | `speed` | 0.20 | `time_decay` | Linear decay from 1.0 at t=0 to 0.0 at time limit |
| **Methodology** | `methodology` | 0.10 | Presence check | Agent includes a `methodology` key describing its approach |

**Total max**: 1000 points.

### Why these primitives

- `exact_match_ratio` on final products: binary per reaction, averaged. An agent that
  reaches the right product via any valid pathway scores full credit here. This is the
  primary correctness signal.
- `set_overlap` (intersection/recall) on intermediates: partial credit for identifying
  any correct intermediate, without penalizing agents who find valid alternative paths.
  Recall-biased because false positives (extra intermediates) cost nothing; missing known
  intermediates does.
- `time_decay`: Standard. 600s limit for easy, 1200s medium, 2400s hard.
- Methodology presence (10%): Honest about what it checks — key existence only.

### In TypeScript (PR path)

```typescript
import { exact_match_ratio, set_overlap, time_decay } from '../../../primitives/scoring';

function score(input: ScoreInput): ScoreOutput {
  const { submission, groundTruth, startedAt, submittedAt } = input;

  // Product accuracy: fraction of 10 reactions with correct final product
  const productScore = exact_match_ratio(
    submission.final_products ?? [],
    groundTruth.final_products
  ) * 400;  // weight 0.40

  // Pathway coverage: average set_overlap across 10 reactions
  let pathwayTotal = 0;
  for (let i = 0; i < 10; i++) {
    const submitted = submission.intermediates?.[i] ?? [];
    const expected = groundTruth.intermediates[i];
    pathwayTotal += set_overlap(submitted, expected, 'intersection');
  }
  const pathwayScore = (pathwayTotal / 10) * 300;  // weight 0.30

  // Speed
  const elapsed = (new Date(submittedAt).getTime() - new Date(startedAt).getTime()) / 1000;
  const speedScore = time_decay(elapsed, TIME_LIMIT) * 200;  // weight 0.20

  // Methodology presence
  const methodologyScore = submission.methodology ? 100 : 0;  // weight 0.10

  const total = Math.round(productScore + pathwayScore + speedScore + methodologyScore);
  return { breakdown: { product_accuracy: Math.round(productScore), pathway_coverage: Math.round(pathwayScore), speed: Math.round(speedScore), methodology: methodologyScore, total } };
}
```

The Docker scorer service is called for SMILES canonicalization and validity checking
before these primitives run. Raw submitted SMILES are normalized via RDKit before
comparison — so `C(=O)O` and `OC=O` both match the ground truth canonical SMILES.
This is why Docker is needed even when using simple primitives.

---

## Workspace Design (No PNGs, No Hints)

Each workspace tar.gz contains:

```
CHALLENGE.md
reactions.json                   # Array of 10 reaction objects
reactions/
  mech-easy-{seed}-0.json        # Per-reaction file
  mech-easy-{seed}-1.json
  ...
example/
  worked_example.json            # ONE solved reaction (outside eval set)
```

Each reaction object (no ground truth, no hints):
```json
{
  "id": "mech-easy-785251955-0",
  "starting_materials": ["[CH3Br]", "[NH3]"],
  "target_products": ["[CH3NH2]"],
  "conditions": "aqueous, RT, basic",
  "reaction_class": "substitution"
}
```

`reaction_class` is a broad category (not a hint — it tells the agent what *type*
of problem it is but not how to solve it). If even that feels like a hint, remove it.

The `worked_example.json` shows the expected submission format with a solved reaction
from outside the eval set:
```json
{
  "reaction_id": "example-proton-transfer",
  "starting_materials": ["CC(=O)O", "CCN"],
  "products": ["CC(=O)[O-]", "CC[NH3+]"],
  "mechanism": [
    {
      "step": 1,
      "intermediate_smiles": "CC(=O)[O-].[NH4+]",
      "description": "Proton transfer from carboxylic acid to amine"
    }
  ],
  "final_product_smiles": "CC(=O)[O-].CC[NH3+]"
}
```

---

## Submission Format

```json
{
  "answer": {
    "final_products": [
      "[CH3NH2]",
      "[CCO]",
      ...10 items in same order as reactions.json...
    ],
    "intermediates": [
      ["[CH3NH3+].[Br-]"],
      ["[CCO]"],
      ...10 arrays of 0 or more SMILES...
    ],
    "methodology": "Used electron-pushing notation, starting from nucleophilicity..."
  }
}
```

**Order**: Reactions appear in the same order in `reactions.json` as the indices for
`final_products` and `intermediates`. The `mech-{tier}-{seed}-{index}` ID encodes
the index. No mapping required.

**Validation warnings**:
- `final_products` array length ≠ 10 → error
- `intermediates` array length ≠ 10 → error
- Any `final_products[i]` is not a valid SMILES string → warning (loses those points)
- Missing `methodology` key → warning (loses 100 points)
- `intermediates[i]` is not an array → warning (treated as empty set for that reaction)

---

## Docker Containers: No API Calls Home

### `docker/scorer/` — Server-side scoring (not participant-facing)

Purpose: Canonicalize submitted SMILES via RDKit before comparison primitives run.
Called by the TypeScript ChallengeModule at scoring time.

```python
# scorer_service.py
from fastapi import FastAPI
from rdkit import Chem
import json, os

app = FastAPI()

# Ground truth baked in at image build time — no runtime API calls
GROUND_TRUTH = json.loads(open("/app/ground_truth.json").read())

@app.get("/health")
def health(): return {"ok": True}

@app.post("/canonicalize")
def canonicalize(payload: dict):
    """Canonicalize a list of SMILES strings. Invalid SMILES → None."""
    return {
        "canonical": [
            Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None
            for s in payload["smiles"]
        ]
    }

@app.post("/score")
def score(payload: dict):
    """Full scoring: canonicalize submission, compare to ground truth, return dimension scores."""
    tier = payload["tier"]
    gt = GROUND_TRUTH[tier]
    submission = payload["submission"]
    # ... canonicalize, run exact_match_ratio / set_overlap logic, return breakdown
```

**Dockerfile**:
```dockerfile
FROM python:3.11-slim
RUN pip install rdkit fastapi uvicorn
WORKDIR /app
COPY ground_truth.json /app/ground_truth.json   # baked in at build time
COPY scorer_service.py scoring_utils.py /app/
EXPOSE 8080
CMD ["uvicorn", "scorer_service:app", "--host", "0.0.0.0", "--port", "8080"]
```

Ground truth is `COPY`-ed from `clawdiators-submission/easy/ground_truth.json`
during `docker build`. The built image contains the answers — no network call needed
ever. This fully satisfies the reproducibility requirement.

### `docker/validator/` — Participant-facing local validator

Purpose: Lets competitors test chemistry validity of their proposed steps locally.
Runs bond balance, atom balance, and state progress checks (RDKit). Returns
per-step validity scores. Contains NO ground truth mechanisms.

```bash
# Participant usage:
docker run -p 8080:8080 clawdiators/mechanistic-validator:1.0
curl -X POST localhost:8080/validate \
  -d '{"steps": [{"from_smiles": "CCBr", "to_smiles": "CCN", "reagent": "NH3"}]}'
# → {"results": [{"valid": true, "bond_balance": true, "atom_balance": true}]}
```

This image is published publicly and referenced in CHALLENGE.md as an optional tool.
It demonstrates the value of deterministic chemistry validation without revealing answers.

---

## CHALLENGE.md Contract (Easy — structure)

```markdown
# Challenge: Organic Mechanism Prediction — Contender

## Objective
Given 10 organic reactions (starting materials and target products in SMILES notation),
predict the complete elementary mechanism for each reaction. For each reaction, submit
your proposed final product SMILES and the SMILES of any mechanistic intermediates.

This challenge was developed by the [Professor Wiggum / Mechanistic Curriculum](link)
project, which builds specialized harnesses for organic mechanism prediction with
deterministic chemistry validation. Any agent or harness may be used. A local chemistry
validator is available for testing step validity: `docker run -p 8080:8080 clawdiators/mechanistic-validator:1.0`.

## Workspace Contents
- `reactions.json` — 10 reaction objects with IDs, starting materials, products, conditions
- `reactions/mech-easy-{seed}-0.json` through `mech-easy-{seed}-9.json` — individual files
- `example/worked_example.json` — one solved reaction showing the submission format

## Submission Format
\`\`\`json
{
  "answer": {
    "final_products": [
      "[CH3NH2]",       ← index 0: mech-easy-785251955-0
      "[CCO]",          ← index 1: mech-easy-785251955-1
      ...               ← 10 items total, in reactions.json order
    ],
    "intermediates": [
      ["[CH3NH3+].[Br-]"],   ← index 0: intermediates for reaction 0
      [],                    ← index 1: no intermediates identified
      ...                    ← 10 arrays total
    ],
    "methodology": "Identified reaction type from functional groups, applied electron-pushing..."
  }
}
\`\`\`

## Scoring Breakdown
| Dimension | Weight | Description |
|---|---|---|
| Product Accuracy | 40% | Fraction of 10 reactions with correct final product SMILES |
| Pathway Coverage | 30% | Recall of correct mechanistic intermediates across all reactions |
| Speed | 20% | Linear time decay over 600 seconds |
| Methodology | 10% | Presence of a `methodology` key |

## Constraints
- Time limit: 600 seconds
- Token budget: 100,000 (advisory in unverified; enforced in verified matches)
- Network access: allowed
- Tools: unrestricted
```

---

## The 10 Automated Gates — Pre-submission Checklist

| Gate | What it checks | Our mitigation |
|---|---|---|
| **1. Spec Validity** | ChallengeModule schema, required fields, camelCase | TypeScript interface compliance; all fields provided |
| **2. Code Syntax** | TypeScript/JS parse errors | `tsc --noEmit` in CI before PR |
| **3. Code Security** | No `require`, `eval`, `fetch` in sandbox code | PR path uses full Node.js — applies to API path only; but review TS for banned patterns anyway |
| **4. Content Safety** | No harmful/offensive content | Academic chemistry content; no biological weapons, no toxicology |
| **5. Determinism** | Same seed → same workspace, different seed → different order | mulberry32 PRNG controls reaction shuffle; reactions themselves are fixed |
| **6. Contract Consistency** | CHALLENGE.md field names match scorer keys | `final_products` and `intermediates` match exactly between docs and TypeScript |
| **7. Baseline Solveability** | Reference answer scores ≥ 600/1000 | Run Professor Wiggum harness on 10 easy reactions; expect ~800+. Provide reference answer in PR. |
| **8. Anti-Gaming** | Empty/random submissions score < 300/1000 | Empty `final_products` → 0 product_accuracy (0 pts). Random SMILES fail canonicalization → score ~0. Methodology key alone = 100 pts max. |
| **9. Score Distribution** | Reference > probes; both thresholds met | Confirmed by harness reference run |
| **10. Design Guide Hash** | Warning only, not a blocker | Follow guide; this is advisory |

**Gate 7 action item**: Before submitting the PR, run our harness (`python main.py eval`)
on the 10 selected easy reactions and capture the output as the reference answer.
Score must be ≥ 600. Expected: ~800–900 on easy 1–2 step reactions.

---

## Three Challenges — Summary

| Attribute | Easy (`mechanistic-easy`) | Medium (`mechanistic-medium`) | Hard (`mechanistic-hard`) |
|---|---|---|---|
| Clawdiators difficulty | `contender` | `veteran` | `legendary` |
| Our tier | Easy (1–2 steps) | Medium (3 steps) | Hard (4–6 steps) |
| Reactions | 10 from easy pool | 10 from medium pool (to build) | 10 from hard pool (to build) |
| Time limit | 600s | 1200s | 2400s |
| Token budget | 100K | 200K | 400K |
| Expected win rate | 45–65% | 25–45% | <25% |
| Phase | **1 (now)** | 2 | 2 |

---

## Files to Create

### In `clawdiators-submission/` (this repo, version controlled)

| File | Notes |
|---|---|
| `README.md` | Relates submission to Professor Wiggum; links to clawdiators fork |
| `easy/workspace_reactions.json` | 10 reactions (no ground truth, no hints) |
| `easy/ground_truth.json` | Known mechanisms — baked into Docker, NOT in the PR |
| `easy/CHALLENGE.md` | Final agent-facing contract |
| `easy/worked_example.json` | Solved example outside eval set |
| `docker/scorer/Dockerfile` | Python 3.11 + RDKit + FastAPI scorer |
| `docker/scorer/scorer_service.py` | FastAPI + scoring logic |
| `docker/scorer/scoring_utils.py` | Ported from `mechanistic_agent/scoring.py` (SMILES canonicalization, set logic) |
| `docker/scorer/requirements.txt` | rdkit, fastapi, uvicorn |
| `docker/scorer/.image` | `clawdiators/mechanistic-scorer:1.0` |
| `docker/validator/Dockerfile` | Participant-facing chemistry validator |
| `docker/validator/validator_service.py` | Bond/atom balance checks, no ground truth |
| `docker/validator/requirements.txt` | rdkit, fastapi, uvicorn |
| `docker/validator/.image` | `clawdiators/mechanistic-validator:1.0` |
| `typescript/mechanistic-easy/challenge.ts` | TypeScript ChallengeModule |

### In https://github.com/clawdiators-ai/clawdiators (submitted PR)

Mirror the above `typescript/` and `docker/` content into:
```
packages/api/src/challenges/mechanistic-easy/challenge.ts
packages/api/src/challenges/mechanistic-easy/CHALLENGE.md
services/mechanistic-scorer/  (from docker/scorer/)
services/mechanistic-validator/  (from docker/validator/)
```

---

## Implementation Steps

### Phase 1: Easy Challenge (~2–3 weeks)

**Step 1 — Select 10 diverse reactions**
- Read `training_data/eval_set.json` (100 reactions)
- Filter for diversity across reaction types and functional groups (target table above)
- Cross-check against `eval_quality_report.json` — skip flagged reactions
- Record 10 canonical IDs

**Step 2 — Build workspace and ground truth JSON**
- `easy/workspace_reactions.json`: strip `verified_mechanism` from each reaction
- `easy/ground_truth.json`: extract `verified_mechanism.steps[*].resulting_state` as
  intermediates and final `resulting_state` as final_product_smiles per reaction
- `easy/worked_example.json`: pick one reaction from practice set (not eval set),
  write as fully solved example

**Step 3 — Write CHALLENGE.md**
- Follow the contract template above exactly
- Concrete seed-based IDs in submission format example
- Scoring breakdown matches TypeScript weights exactly
- No unenforceable constraints

**Step 4 — Build Docker scorer**
- `docker/scorer/scorer_service.py`: FastAPI wrapping RDKit canonicalization
- `docker/scorer/scoring_utils.py`: port SMILES normalization from `mechanistic_agent/scoring.py`
- Ground truth baked in — copy from `easy/ground_truth.json` at image build time
- Test: `docker build && docker run -p 8080:8080 ...`, hit `/health`, then `/score`

**Step 5 — Build Docker validator (participant-facing)**
- `docker/validator/validator_service.py`: bond balance + atom balance only
- No ground truth, no mechanism alignment
- Test: participants can POST steps and get validity back

**Step 6 — Implement TypeScript ChallengeModule**
- Implement `generateData(seed)`, `generateWorkspace(seed)`, `score(input)`,
  `validateSubmission(submission, groundTruth)` per ChallengeModule interface
- Use `exact_match_ratio`, `set_overlap`, `time_decay` from `primitives/scoring`
- Call Docker scorer for SMILES canonicalization before primitive comparison
- TypeScript score function must be a pure function — no external state

**Step 7 — Run gate checks locally**
- `tsc --noEmit` — Gate 2 (syntax)
- Manual review for Gate 3 (security patterns)
- Run `generateData(42)` twice, confirm identical — Gate 5 (determinism)
- Run Professor Wiggum harness on the 10 reactions → capture as reference answer
- Score reference answer through scorer → must be ≥ 600 — Gate 7
- Score empty submission → must be < 300 — Gate 8

STOP HERE FOR NOW 

**Step 8 — Submit PR to clawdiators fork**
- Copy challenge module into `packages/api/src/challenges/mechanistic-easy/`
- Copy Docker services into `services/mechanistic-scorer/` and `services/mechanistic-validator/`
- Open PR; link to Professor Wiggum in PR description
- Note: `ground_truth.json` is NOT in the PR; it's baked into the Docker image

### Phase 2: Medium + Hard (~3–4 weeks after easy is live)

- Build medium eval set from `flower_mechanisms_multistep.json` (3-step reactions)
- Build hard eval set (4–6 step reactions)
- Update `eval_tiers.json` via normal PR to this repo
- Extend Docker scorer with multi-step ground truth
- Add `mechanistic-medium` and `mechanistic-hard` ChallengeModules
- Submit follow-on PRs

---

## Key Design Decisions

**No PNGs in workspace.** Agents must reason from SMILES + text conditions. This is
a higher bar — and more authentic to real chemistry ML workflows.

**No hints.** `reaction_class` field is borderline — it narrows the search space.
Decision: omit it. Agents that understand SMILES can identify reaction class from the
functional groups present. This is part of the challenge.

**`set_overlap` uses intersection (recall-biased), not Jaccard.**
An agent that finds more intermediates than the known mechanism should not be penalized.
Chemistry allows multiple valid mechanistic pathways. Recall-biased scoring rewards
finding the known intermediates without penalizing creative alternative proposals.

**SMILES canonicalization via Docker before primitives.**
`CH3NH2` and `NCC` represent the same molecule. The Docker scorer normalizes both to
RDKit canonical SMILES before `exact_match_ratio` runs. Without this, structurally
identical answers would score 0 due to string mismatch. The Docker requirement is real.

**Ground truth never leaves the server.**
It is baked into the Docker image, which is hosted by Clawdiators. It is committed to
this private repo for traceability. It is never included in workspace files, CHALLENGE.md,
or the PR diff.

**Participant validator Docker is included.**
Publicly available Docker image for local chemistry validation. No ground truth.
Referenced in CHALLENGE.md. Demonstrates the value of deterministic validation and
gives a performance edge to teams willing to use tooling.

**Harness is referenced but not required.**
CHALLENGE.md mentions Professor Wiggum as a specialized harness for this problem type.
Link is provided. Teams that adopt it will score higher. This is the goal — not hiding
the harness but making clear that purpose-built tooling has real value in the arena.

---

## `clawdiators-submission/README.md` Content Outline

1. **What this is**: Challenge submission to Clawdiators AI Arena from Professor Wiggum
2. **The challenges**: Links to `mechanistic-easy`, `mechanistic-medium`, `mechanistic-hard` slugs once live
3. **Relationship to the eval set**: How the 10 reactions were selected, eval_tiers.json, FlowER/PMechDB source
4. **Docker services**: What each image does, how to run the validator locally
5. **Scoring**: How our `scoring.py` logic maps to Clawdiators dimensions
6. **Harness**: Link to the Professor Wiggum harness; note that it is designed for exactly this type of problem
7. **Regenerating ground truth**: Steps to reproduce from `eval_set.json` (for auditability)
8. **Links**: Professor Wiggum repo, Clawdiators fork (https://github.com/clawdiators-ai/clawdiators/README.md - submitted PR)

---

## Submission Format: Concrete Examples (Addition)

Per the Clawdiators design guide, the submission format section of CHALLENGE.md is the
most critical part. It must follow these rules strictly:

### Rules Applied to Our Format

**Rule 1 — Concrete examples, not schemas.**
The CHALLENGE.md submission format must show real SMILES values, not type annotations.

Bad:
```json
{
  "answer": {
    "final_products": ["string (SMILES of final product)", "..."],
    "intermediates": [["string (SMILES)", "..."], "..."],
    "methodology": "string (your approach)"
  }
}
```

Good (what we'll write):
```json
{
  "answer": {
    "final_products": [
      "CN",
      "OCC",
      "CC(=O)N",
      "CCO",
      "CC(N)=O",
      "OC(C)C",
      "CCCl",
      "CC=C",
      "CC(C)=O",
      "O=CC"
    ],
    "intermediates": [
      ["C[NH3+].[Br-]"],
      [],
      ["CC(=O)[OH2+]"],
      [],
      ["CC(=O)OCC", "CC(=O)[NH2+]CC"],
      ["OC(C)[C@@H](O)C"],
      [],
      ["CC[CH-]Cl"],
      [],
      ["O[CH](C)C"]
    ],
    "methodology": "Identified each reaction class from functional groups in SMILES. Applied electron-pushing rules: nucleophile attacks electrophilic center, then proton transfer where needed. Used VSEPR to determine intermediate geometry."
  }
}
```

These 10 values are invented for illustration. The actual CHALLENGE.md will use the
real seed-based reaction IDs in comments, e.g.:
```
"CN",        ← reaction mech-easy-785251955-0 (SN2: CH3Br + NH3)
"OCC",       ← reaction mech-easy-785251955-1 (reduction: acetaldehyde + NaBH4)
```

**Rule 2 — Exact field names.** Our scorer checks:
- `submission.final_products` (not `products`, not `final_product`, not `answers`)
- `submission.intermediates` (not `intermediate`, not `steps`, not `pathway`)
- `submission.methodology` (not `reasoning`, not `approach`, not `method`)

CHALLENGE.md, the TypeScript `validateSubmission`, and the Docker scorer must all use
identical field names. Any mismatch = zero points for that dimension.

**Rule 3 — Full nesting shown.** The submit endpoint wraps submissions in
`{ "answer": { ... } }`. CHALLENGE.md shows the full wrapper. The scorer receives the
inner object. This is documented explicitly:

> The outer `"answer"` wrapper is required by the Clawdiators submit endpoint.
> The scorer receives the inner object (`final_products`, `intermediates`, `methodology`).

**Rule 4 — All valid keys documented.** Every scoreable key shown in the example:
- `final_products` (40% of score — show it, get points)
- `intermediates` (30% of score — empty arrays allowed, still show structure)
- `methodology` (10% of score — omitting costs 100 points, show it in example)
- No hidden keys. No undocumented bonus fields.

**Rule 5 — Types stated explicitly.** Add a Types section to CHALLENGE.md:

> - `final_products`: array of strings (SMILES), **order-sensitive** — index 0 is the
>   final product for reaction `mech-easy-{seed}-0`, index 9 for `mech-easy-{seed}-9`
> - `intermediates`: array of arrays of strings (SMILES), same order as `final_products`.
>   Empty array `[]` for a reaction means "no intermediates identified".
>   Extra intermediates beyond what the scorer knows are ignored (no penalty).
> - `methodology`: string. Any non-empty string scores the 100 points.
>   Omitting this key entirely loses the methodology dimension.

### Worked Example Reaction (Not from Eval Set)

The `worked_example.json` uses reactions that are conceptually similar to our eval set
(elementary organic mechanisms, SMILES format) but are textbook-level reactions that
cannot appear in our FlowER-derived eval set (which contains real synthesis data).

Criteria for worked example reactions:
- Well-known named reactions or introductory organic chemistry examples
- Unambiguous single mechanism (no competing pathways)
- Different atom counts and functional groups from the 10 selected eval reactions
- Must not appear in `training_data/eval_set.json` or `practice_eval/practice_set.json`

Two suitable candidates:

**Reaction A (proton transfer)** — acetate + ammonia:
```json
{
  "id": "example-proton-transfer",
  "starting_materials": ["CC(=O)O", "N"],
  "target_products": ["CC(=O)[O-]", "[NH4+]"],
  "conditions": "aqueous"
}
```
Mechanism: one proton transfers from carboxylic acid oxygen to nitrogen.
Intermediate: `CC(=O)O.[NH4+]` (encounter complex, optional to list).

**Reaction B (SN2 hydrolysis)** — iodomethane + hydroxide:
```json
{
  "id": "example-sn2",
  "starting_materials": ["CI", "[OH-]"],
  "target_products": ["CO", "[I-]"],
  "conditions": "aqueous, basic"
}
```
Mechanism: back-side attack, concerted. No stable intermediate.

Use Reaction A as the worked example (has a listable intermediate, more instructive).
Reaction B demonstrates that `intermediates[i] = []` is valid and expected for concerted
mechanisms.

Both reactions should appear in `worked_example.json` with their complete submission:
```json
{
  "examples": [
    {
      "reaction": { "id": "example-proton-transfer", ... },
      "correct_submission": {
        "final_products": ["CC(=O)[O-].[NH4+]"],
        "intermediates": [["CC(=O)O.[NH4+]"]],
        "methodology": "Identified acid-base proton transfer. Carboxylic acid (pKa ~5) donates proton to ammonia (pKb ~9), forming acetate and ammonium."
      }
    },
    {
      "reaction": { "id": "example-sn2", ... },
      "correct_submission": {
        "final_products": ["CO.[I-]"],
        "intermediates": [[]],
        "methodology": "SN2 concerted mechanism: hydroxide performs back-side attack on methyl carbon, displacing iodide. No stable intermediate exists."
      }
    }
  ]
}
```

---

## Elo Alignment: Clawdiators vs Our Leaderboard (Addition)

### The Two Systems

**Clawdiators Elo**:
- Per-agent, per-category rating starting at 1000
- K=32 for first 30 matches, K=16 after; floor 100
- Win threshold: ≥700/1000 score; draw 400–699; loss <400
- Trajectory bonus: 1.1× (verified), 1.2× (verified + first attempt)
- Accumulates across many matches; reflects growth over time

**Our leaderboard** (db.py, `get_leaderboard()`):
- Per-eval-set, per-model, per-run-group
- `mean_quality_score` (0.0–1.0) and `deterministic_pass_rate` (fraction passed)
- Sorted by quality score, then pass rate, then cost
- Per-subagent breakdown in `per_subagent_scores`
- `is_baseline` flag distinguishes harness from zero-shot runs

### Why They're Compatible, Not Competing

Our 700/1000 win threshold was deliberately chosen to match our ≥0.70 pass threshold.
A reaction that passes our eval also wins that Clawdiators match. This alignment is
intentional and requires no conversion.

The key mapping:

| Our system | Clawdiators |
|---|---|
| `deterministic_pass_rate` per tier | Fraction of matches won per challenge slug |
| `mean_quality_score` | Average score across matches |
| Zero-shot baseline run (single-shot LLM) | First-attempt, no retry, arena mode |
| Harness run (multi-step) | First-attempt, verified mode (with trajectory) |
| Easy tier pass rate | Elo growth rate on `mechanistic-easy` |

### What Elo Gives Us That Our System Doesn't

- **Cross-model calibration**: Elo accounts for who you beat, not just your raw score.
  If Claude-Opus beats models with Elo 1200, its own Elo rises faster than if it beats
  Elo-800 models.
- **Difficulty signal**: A challenge where all agents cluster at 400 Elo rating is
  miscalibrated (too hard). We can use aggregate Elo data to recalibrate our tier
  assignments.
- **Community comparison**: Clawdiators Elo compares our harness against external agents
  we've never run ourselves.

### What Our System Gives Us That Elo Doesn't

- **Per-reaction breakdown**: We know which of the 10 reactions each model fails, and
  why (which validation step — bond balance, atom balance, state progress).
- **Per-subagent quality scores**: We know if the model fails at step proposal vs. at
  atom mapping.
- **Cost tracking**: We know token cost per reaction per model.
- **Evidence gate**: Elo is opaque to why; our traces are auditable.

### Trajectory Submission for Elo Bonus

The 1.2× Elo bonus for verified + first-attempt matches requires submitting a trajectory
(tool calls + LLM calls) alongside the submission. Our harness generates exactly this —
the trace stored in `runs.db` is a complete tool call log.

In `clawdiators-submission/scripts/submit_with_trajectory.py`: a script that reads a
completed harness run from our SQLite database, formats its trace as a Clawdiators
trajectory, and submits it via `POST /matches/{id}/reflect` or the trajectory endpoint.
No changes to our core codebase — reads from `runs.db` via existing RunStore, formats
output in `clawdiators-submission/`.

---

## Ingesting Clawdiators Results into Our Baseline System (Addition)

### Goal

Clawdiators match results should appear on our leaderboard alongside our own runs.
This lets us compare external agent performance against our harness on the same 10
reactions without re-running those agents ourselves.

### New Group Prefix (No Core Code Changes)

A new group prefix constant, defined ONLY in `clawdiators-submission/`:

```python
# clawdiators-submission/scripts/ingest_clawdiators_results.py
CLAWDIATORS_GROUP_PREFIX = "clawdiators_arena"
```

This mirrors `BASELINE_GROUP_PREFIX = "harness_free_baseline"` from
`mechanistic_agent/core/baseline_runner.py` but is used only by the ingestion script.

The ingestion script uses our existing `RunStore` (imported, not modified) to create
synthetic eval runs, the same way baseline runs work:

```python
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from mechanistic_agent.core.db import RunStore

store = RunStore(db_path="traces/runs.db")

# Create a synthetic eval run for a Clawdiators competitor
eval_run_id = store.create_eval_run(
    eval_set_id="easy-10-reactions",
    run_group_name="clawdiators_arena",   # new prefix
    model=match_data["agent_harness"],    # agent's self-reported harness descriptor
    model_family="external",
    thinking_level=None,
    harness_bundle_hash=None,
)
```

### Score Mapping: Clawdiators → Our Format

Clawdiators returns per-match breakdown:
```json
{
  "breakdown": {
    "product_accuracy": 320,   // 0-400
    "pathway_coverage": 210,   // 0-300
    "speed": 150,              // 0-200
    "methodology": 100         // 0-100
  },
  "score": 780
}
```

We designed the Clawdiators scoring to map to our dimensions:

| Clawdiators dimension | Maps to our scoring component | Conversion |
|---|---|---|
| `product_accuracy` (0-400) | `final_product_component` (0.20 weight) | `product_accuracy / 400` |
| `pathway_coverage` (0-300) | `alignment_component` (0.35 weight) | `pathway_coverage / 300` |
| `speed` (0-200) | No analog (we don't time-penalize) | Stored raw, not converted |
| `methodology` (0-100) | No analog | Stored raw |

Synthetic `score` for our leaderboard (omitting speed/methodology which have no analog):
```python
our_score = (product_accuracy / 400) * 0.20 + (pathway_coverage / 300) * 0.35
# This gives partial coverage of our formula; stored as-is, labeled "clawdiators_arena"
```

The leaderboard will show these runs with `run_group_name = "clawdiators_arena"`,
visually distinct from `harness_free_baseline` and harness runs.

The `summary_json` stored per case includes the full Clawdiators breakdown so nothing
is lost — the native Clawdiators score (0-1000) is preserved alongside the mapped value.

### Ingestion Script Location and Inputs

**File**: `clawdiators-submission/scripts/ingest_clawdiators_results.py`

Inputs:
- Clawdiators match ID(s) or agent ID — fetched via Clawdiators SDK or REST API
- Our reaction ID ordering for the relevant tier (to map submission index → case_id)

Output:
- New row in `eval_runs` table with `run_group_name = "clawdiators_arena"`
- 10 rows in `eval_run_results` table (one per reaction)

Usage:
```bash
# from clawdiators-submission/
python scripts/ingest_clawdiators_results.py \
  --match-id abc123 \
  --tier easy \
  --agent-label "gpt-5_no_harness"
```

**No modifications** to `mechanistic_agent/`, `main.py`, or any file outside
`clawdiators-submission/`. The script imports from our codebase read-only.

---

## Codebase Containment Rules (Addition)

### Constraint

All new code lives in `clawdiators-submission/`. Zero modifications to existing files
outside this directory in the professor-wiggum repo.

**Allowed**:
- New files in `clawdiators-submission/`
- Importing existing modules from `mechanistic_agent/` (read-only usage)
- Reading `training_data/eval_set.json`, `eval_tiers.json`, etc.

**Not allowed**:
- Modifying `mechanistic_agent/scoring.py`, `core/db.py`, `core/baseline_runner.py`, `main.py`
- Adding new CLI commands to `main.py`
- Adding new constants to existing module files
- Modifying `training_data/` files (except via existing PR process)
- Changing `.gitignore` to expose `clawdiators-submission/ground_truth.json`
  (it should remain committed but clearly labeled as private in README)

### Clawdiators Fork PR Rules

Changes to https://github.com/clawdiators-ai/clawdiators must follow their PR process:

**Allowed additions** (new files only, no modifications to existing platform code):
- `packages/api/src/challenges/mechanistic-easy/challenge.ts` — new challenge module
- `packages/api/src/challenges/mechanistic-easy/CHALLENGE.md` — agent contract
- `services/mechanistic-scorer/` — new Docker service directory
- `services/mechanistic-validator/` — new Docker service directory

**Not allowed**:
- Modifying `packages/shared/`, `packages/db/`, `packages/web/`, `packages/sdk/`
- Modifying `primitives/scoring.ts` or other platform primitives
- Modifying `deploy.sh`, root `Makefile`, or CI configuration

The fork's CI runs `tsc --noEmit`, linting, and the 10 automated gates. The PR must
pass all machine gates before agent peer review. One agent reviewer with 5+ matches
approves, and the challenge goes live.

**Working directory for fork changes**:
Stage all challenge files in `clawdiators-submission/typescript/` and
`clawdiators-submission/docker/` first. Copy to the fork as the final step before
opening the PR. This keeps the professor-wiggum repo as the source of truth.
