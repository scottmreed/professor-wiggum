# PRD: Jev-First Decision Layer and Persistent Atom Identity for Mechanistic

**Status:** Proposed  
**Project:** MechanisticWiggum / Mechanistic Agent  
**Primary scope:** Mechanistic prediction harness  
**Target implementation area:** `mechanistic_agent/core/`, `skills/mechanistic/`, `harness_versions/`, eval/trace infrastructure  
**Design principle:** Use Jev to avoid full LLM calls. Reserve full LLMs for genuinely generative chemistry, ambiguity escalation, and opportunistic review of upstream Jev decisions.

---

## 1. Summary

Mechanistic currently combines deterministic chemistry checks with several LLM-backed pre-loop analyses and one major generative LLM step inside the mechanism loop. This PRD proposes moving bounded judgment and bookkeeping tasks away from full LLMs and into a combination of:

1. deterministic RDKit logic,
2. Jev `Choice`, `Score`, and `Noul` decisions,
3. explicit uncertainty thresholds and fallback behavior,
4. full LLM calls only when new chemical content must be generated or when bounded methods remain ambiguous.

The primary opportunities are:

- replace pH recommendation with a Jev `Score`,
- replace reaction-condition assessment with Jev `Choice` + `Score`,
- replace most global atom mapping with deterministic candidate generation + Jev `Choice`,
- replace most step atom mapping with persistent atom identity or deterministic candidate generation + Jev `Choice`,
- replace reaction-type classification with Jev `Choice`,
- retain full LLM generation for mechanism proposals and missing chemistry,
- allow the major mechanism LLM call to flag suspicious upstream Jev decisions without adding a separate review call,
- experimentally compare Jev ranking and LLM ranking of mechanism candidates before allowing the harness to evolve toward Jev-only ranking.

A critical technical spike is required before implementing Jev-based step mapping: determine whether the deterministic mechanism transform can preserve persistent atom identities while applying graph edits. If it can, post-step atom mapping may largely disappear as an inference task.

---

## 2. Why this change

The target architecture is:

> **deterministic chemistry → bounded Jev decisions → full LLM only for open-ended chemistry**

Jev should not become an extra judge added after every LLM output. Its primary value is reducing or replacing full-model calls.

The current harness already supports this direction:

- deterministic balance and functional-group analysis,
- explicit pre-loop modules,
- a generative mechanism proposal step,
- deterministic mechanism execution/validation,
- branch points, retries, and backtracking,
- post-step mapping,
- eval-tier and leaderboard-based harness evolution.

The current coordinator also makes candidate rank operationally important: validated candidates are sorted by rank, the top candidate is applied, and lower-ranked validated candidates are stored as branch alternatives. This creates a clean experimental surface for comparing rankers without changing candidate generation.

---

## 3. Goals

### 3.1 Product goals

- Reduce unnecessary full-LLM calls.
- Reduce latency and inference cost without reducing mechanism quality.
- Make uncertainty explicit and actionable.
- Preserve deterministic chemistry validators as final arbiters.
- Make atom mapping more reproducible and less dependent on generative models.
- Turn Jev-vs-LLM decisions into measurable harness experiments.
- Allow successful Jev replacements to become defaults through normal eval-backed harness evolution.

### 3.2 Technical goals

- Add a Jev adapter supporting:
  - `Choice`
  - `Score`
  - `Noul`
  - probability distributions
  - model/version metadata
  - timeout/failure fallback
- Add deterministic mapping-candidate generation.
- Add persistent atom identity support through mechanism graph edits if technically feasible.
- Add configurable decision policies and thresholds at the harness level.
- Record Jev outputs and probability distributions in traces.
- Add shadow-mode ranker comparison.
- Add LLM context review fields without creating an additional LLM call.

---

## 4. Non-goals

This PRD does **not** propose:

- replacing deterministic chemistry validators with Jev,
- using Jev to generate new intermediate SMILES,
- using Jev to generate arbitrary reagents,
- using Jev to generate electron pushes or reaction SMIRKS,
- removing full LLM mechanism proposal,
- trusting Jev probabilities as proof of chemical correctness,
- immediately changing production candidate ordering to Jev,
- using raw RDKit atom indices as persistent atom identity.

---

## 5. Current architecture relevant to this PRD

### Pre-loop

Current functional sequence is approximately:

```text
Check atom balance
→ Identify functional groups
→ Recommend pH
→ Assess reaction conditions
→ Predict missing reagents
→ Global atom mapping
→ Reaction-type mapping
```

The coordinator currently groups pH recommendation and condition assessment under `ConditionsAgent`.

Initial atom mapping is dispatched through `MappingAgent`.

Reaction-type mapping is currently an LLM-backed classification except for deterministic example mappings/fallbacks.

### Mechanism loop

The current loop is approximately:

```text
LLM mechanism candidate proposal
→ deterministic mechanism synthesis / execution
→ deterministic validators
→ validate candidates
→ sort validated candidates by rank
→ choose top candidate
→ retain alternatives as branch points
→ apply candidate
→ post-step modules
→ continue / backtrack / terminate
```

### Post-step

`step_atom_mapping` currently calls:

```python
self.mapping_agent.run_step_mapping(
    state,
    current_state=mapping_current,
    resulting_state=mapping_resulting,
)
```

This means post-step atom identity is currently re-inferred after a successful step rather than necessarily propagated through the transformation.

---

# 6. Proposed target architecture

```text
DETERMINISTIC
  atom balance
  functional groups
       │
       ▼
JEV
  pH Score
  condition-environment Choice
  condition-compatibility Score
       │
       ▼
DETERMINISTIC
  generate global mapping candidates
       │
       ├── one candidate ─────────────→ accept
       │
       ├── small candidate set ──────→ JEV Choice
       │
       └── unresolved/large set ─────→ FULL LLM fallback
       │
       ▼
JEV
  reaction-type Choice
       │
       ▼
OPTIONAL FULL LLM
  generate missing chemistry only when required
       │
       ▼
════════════ MECHANISM LOOP ════════════
       │
       ▼
FULL LLM
  generate candidate elementary steps
  + review provisional upstream Jev context
       │
       ▼
RANKING EXPERIMENT
  Jev Choice rank
  AND
  LLM judge rank
  AND
  existing generator rank
       │
       ▼
production ordering policy
(initially unchanged / shadow mode)
       │
       ▼
DETERMINISTIC
  execute candidate
  validate
  preserve persistent atom identity if possible
       │
       ▼
STEP MAPPING
  identity preserved → no inference
  otherwise deterministic candidate generation
      → one candidate: accept
      → small set: Jev Choice
      → unresolved: full LLM fallback
       │
       ▼
DETERMINISTIC
  retry / branch / backtrack / completion
```

---

# 7. Jev integration surfaces

## 7.1 pH recommendation → Jev `Score`

Replace a single-point pH guess with an ordered distribution.

Recommended initial levels:

```text
0: strongly acidic      approximately pH 0–3
1: mildly acidic        approximately pH 3–6
2: near neutral         approximately pH 6–8
3: mildly basic         approximately pH 8–11
4: strongly basic       approximately pH 11–14
```

Input should include:

- starting materials,
- products,
- detected functional groups,
- supplied reagents/conditions if present,
- user-specified pH if present.

If the user explicitly supplied a pH, preserve it as authoritative input and use Jev only for compatibility assessment rather than overwriting it.

Store:

```json
{
  "selected_level": "mildly_basic",
  "probabilities": {
    "strongly_acidic": 0.01,
    "mildly_acidic": 0.05,
    "near_neutral": 0.16,
    "mildly_basic": 0.68,
    "strongly_basic": 0.10
  },
  "confidence": 0.68
}
```

If legacy downstream code requires a representative pH, derive one deterministically from level midpoints. The probability distribution remains the canonical result.

### Acceptance test

On the existing eval tiers, compare:

- current pH method,
- Jev Score method,
- downstream pathway completion,
- reaction-type accuracy,
- retries/backtracks.

The pH module should not be accepted merely because it matches a single expected numeric pH.

---

## 7.2 Assess reaction conditions → Jev `Choice` + `Score`

Use one Jev request with shared state and at least two questions.

### Choice: environment

```text
acidic
basic
neutral
mixed
unclear
```

### Score: compatibility with target transformation

```text
0: strongly incompatible
1: questionable
2: broadly plausible
3: strongly supportive
```

Optional future questions may include bounded concepts such as:

- protic/aprotic compatibility,
- oxidizing/reducing character,
- strong/weak nucleophilic environment,
- likely thermal requirement.

Do not ask Jev to generate arbitrary reagents or prose recommendations.

### Fallback policy

No full LLM call should occur merely because compatibility confidence is moderate.

Escalate only when:

- downstream logic requires a missing chemical object,
- condition ambiguity blocks reaction-type classification or mechanism proposal,
- configured uncertainty threshold is crossed and the harness policy explicitly permits escalation.

---

## 7.3 Reaction-type mapping → Jev `Choice`

This is a strong bounded-decision target.

Input:

- normalized starting materials/products,
- balance analysis,
- functional groups,
- pH/condition Jev outputs,
- missing reagent context,
- atom mapping or mapping summary.

Output:

```json
{
  "choice": "E2",
  "probabilities": {
    "E2": 0.74,
    "SN2": 0.17,
    "E1": 0.05,
    "other": 0.04
  }
}
```

### Candidate taxonomy control

Do not necessarily expose the entire reaction taxonomy on every call.

Prefer deterministic pre-filtering using:

- functional groups,
- bond-change signatures,
- reagent classes,
- mapped reaction center if available.

Target option count:

- ideal: 3–20 choices,
- include `no_match`,
- benchmark larger candidate sets rather than assuming a hard maximum.

Existing reaction-template confidence-gap behavior should be adapted to real choice probabilities rather than free-form LLM confidence.

---

# 8. Global atom mapping redesign

## 8.1 Principle

Do not ask Jev to invent an atom mapping.

Generate a bounded set of chemically reasonable mapping candidates deterministically, then use Jev only to select among them.

Full LLM atom mapping becomes a fallback for unresolved cases.

---

## 8.2 Deterministic candidate generator

Add:

```text
MappingCandidateGenerator
```

Suggested implementation stages:

1. Parse all reactants/products with RDKit.
2. Assign stable temporary atom identities.
3. Enforce hard elemental/isotopic compatibility.
4. Lock unique atom correspondences.
5. Compare local graph environments.
6. Use MCS/substructure correspondence where appropriate.
7. Preserve the maximum number of existing bonds.
8. Identify likely reaction-center changes.
9. Enumerate only unresolved chemically distinct correspondences.
10. Canonicalize equivalent mappings.
11. Score deterministic mapping cost.
12. retain top-K candidates.

Useful features:

- atomic number,
- isotope,
- formal charge,
- aromaticity,
- ring membership,
- degree,
- total valence,
- local neighbor signatures,
- Morgan-style local environment signatures,
- unchanged-bond count,
- changed-bond count,
- reaction-center compactness,
- stereochemical consistency where represented.

Element mismatch should be a hard rejection, not a soft score.

---

## 8.3 Collapse symmetry-equivalent mappings

Do not treat every atom-index permutation as a distinct Jev option.

Mappings should be considered equivalent if they differ only by exchange of chemically/mechanistically equivalent atoms and produce the same relevant:

- reaction center,
- bond-change set,
- atom identities participating in changed bonds,
- stereochemical outcome.

This is required to keep Jev option sets small.

---

## 8.4 Candidate-count policy

Initial experimental policy:

```text
0 candidates
    → full LLM mapping fallback

1 candidate
    → deterministic accept

2–6 chemically distinct candidates
    → Jev Choice

7–12
    → further deterministic pruning
    → Jev only if reduced to configured max

> configured maximum
    → full LLM fallback
```

The `6` and `12` values are starting hypotheses, not product requirements. Tune them empirically.

Primary benchmark:

> **Does the deterministic top-K set contain an acceptable benchmark mapping?**

Only after top-K recall is strong should Jev selection accuracy become the main optimization target.

---

## 8.5 Jev mapping Choice

Each option should describe the chemically meaningful mapping difference, not dump an enormous atom table without context.

Include:

- concise map pairs near the reaction center,
- preserved-bond count,
- changed bonds,
- relevant functional-group identities,
- ambiguity description.

Example:

```text
mapping_A:
  carbonyl C7 → product C7
  carbonyl O8 → product O8
  nucleophile N12 → product N12
  new bond: C7-N12

mapping_B:
  carbonyl C7 → product C7
  equivalent O9 → product O8
  nucleophile N12 → product N12
  new bond: C7-N12
```

The full machine-readable mapping remains in state; Jev receives the compact discrimination-relevant representation.

---

# 9. Persistent atom identity spike

## 9.1 Key question

Can the deterministic mechanism transform preserve atom identity while applying graph edits?

If yes, `step_atom_mapping` should become mostly unnecessary.

---

## 9.2 Identity representation

Do **not** use RDKit `Atom.GetIdx()` as persistent identity.

Atom indices are container positions and can change when atoms are deleted/reordered.

Preferred identity options:

### Option A — atom-map number as persistent identity

Use:

```python
atom.SetAtomMapNum(id)
atom.GetAtomMapNum()
```

Advantages:

- survives mapped-SMILES serialization,
- directly visible/debuggable,
- already chemically familiar.

### Option B — sidecar persistent-ID map

Maintain a sidecar structure:

```python
PersistentAtomId -> {
    molecule_id,
    current_atom_index,
    provenance,
    original_atom_map
}
```

Advantages:

- avoids overloading chemically meaningful atom-map numbering,
- can preserve identity even if output SMILES intentionally strips atom maps.

### Recommended implementation

Use a sidecar identity model as the canonical internal identity and atom-map numbers as a serialization/debug bridge where safe.

Arbitrary custom RDKit atom properties must not be assumed to survive SMILES round-trips.

---

# 10. Required atom-ID preservation tests

Create a dedicated spike test module before changing production mapping behavior.

Suggested path:

```text
tests/fast/test_persistent_atom_identity.py
```

and, if a new transform utility is introduced:

```text
mechanistic_agent/core/atom_identity.py
```

## 10.1 Baseline: no-op round trip

1. Parse molecule.
2. assign persistent atom IDs.
3. serialize.
4. reparse.
5. verify every surviving atom has the same persistent identity.

Test both:

- mapped SMILES,
- current actual state serialization path.

Expected:

- atom-map identity survives mapped-SMILES round trip,
- custom properties are explicitly tested and documented as preserved/not preserved,
- no code relies on `GetIdx()` remaining unchanged.

---

## 10.2 Bond-order change

Example class:

```text
C-C single → C=C
C=O → C-O
```

Procedure:

1. assign persistent IDs,
2. modify bond order with `RWMol`,
3. sanitize,
4. serialize/reparse,
5. verify all unchanged atoms retain identity,
6. verify changed-bond endpoints retain identity.

Pass criterion:

```text
100% identity retention for surviving atoms
```

---

## 10.3 Bond formation without atom creation

Example:

```text
nucleophile + electrophile → new bond
```

Requirements:

- separate components may merge,
- existing atoms must retain IDs,
- the new bond references persistent IDs,
- product serialization must not scramble identity.

---

## 10.4 Bond cleavage / component split

Example:

```text
C-LG → C+ / LG- or corresponding neutralized products
```

Requirements:

- one connected component may become two,
- all surviving atoms retain identities,
- molecule/component ordering may change without affecting identity.

This specifically tests that identity is not tied to molecule-array position.

---

## 10.5 Substitution

Example:

```text
SN2-like:
Nu + C-LG → C-Nu + LG
```

Test simultaneous:

- bond deletion,
- bond addition,
- component merge/split,
- charge changes if applicable.

All existing atoms must retain persistent identity.

---

## 10.6 Proton transfer

Run two versions:

### implicit-H representation

Determine whether the current mechanism engine can meaningfully preserve proton identity when the proton is implicit.

Expected result may be:

```text
heavy-atom identity preserved;
individual implicit-H identity is undefined.
```

### explicit-H representation

Assign a persistent identity to the proton and verify it transfers from donor to acceptor.

This test determines whether explicit hydrogens are required for mechanistically auditable proton tracking.

---

## 10.7 Atom addition/removal test

Mechanistic chemistry should not create/destroy atoms outside explicitly modeled species, but the graph-edit layer must still have defined behavior when adding/removing RDKit atoms.

Test:

- adding an explicit proton from an explicitly represented proton source,
- deleting an atom only as part of a controlled component operation if production code permits it.

Requirements:

- pre-existing IDs remain unchanged,
- newly introduced atom receives a new ID from a monotonic allocator,
- IDs are never silently reused.

---

## 10.8 Aromatic / Kekulé transformations

Test identity through:

- aromatic input,
- Kekulization,
- bond editing,
- sanitization,
- canonical SMILES output.

Identity must survive representation changes.

---

## 10.9 Stereochemical transformation

Test at least:

- tetrahedral stereocenter unaffected by remote edit,
- inversion/retention when the mechanism intentionally changes stereochemistry,
- E/Z bond context if represented.

Identity and stereochemical state must be independently testable.

---

## 10.10 Canonicalization stress test

Because canonical SMILES can reorder atoms:

1. assign persistent IDs,
2. perform graph edit,
3. canonicalize,
4. parse canonical representation,
5. compare by persistent ID rather than RDKit index.

This test should explicitly demonstrate that:

```text
atom index != atom identity
```

---

## 10.11 Multi-step persistence test

Run a complete 3–8 step known mechanism.

For every step:

- collect persistent IDs before transform,
- apply transform,
- collect IDs after transform,
- verify every surviving atom remains the same identity,
- verify newly introduced explicitly modeled atoms have provenance,
- verify no duplicate persistent IDs,
- verify no ID reuse,
- compare final mapping to benchmark mapping where available.

This is the decisive test for removing LLM-based step mapping.

---

## 10.12 Backtracking identity test

Because Mechanistic restores earlier branch snapshots:

1. apply step A,
2. apply step B,
3. backtrack to branch before A/B,
4. apply alternative C.

Verify:

- snapshot restores exact persistent IDs,
- discarded path IDs do not contaminate restored state,
- alternative path preserves identities from the branch state,
- trace remains auditable.

---

## 10.13 Serialization and persistence test

Persist a run state to SQLite/file artifacts, reload it, and resume.

Verify:

- persistent atom identities are restored,
- identity survives pause/resume,
- identity survives branch-point serialization,
- replay produces the same mapping state.

---

# 11. Step atom mapping target behavior

If persistent identity tests pass:

```text
mechanism transform
    ↓
identity propagation
    ↓
step mapping derived deterministically
```

`attempt_atom_mapping_for_step` becomes a fallback, not a routine LLM call.

Recommended hierarchy:

```text
1. derive mapping from persistent IDs
2. if incomplete:
      local deterministic mapping candidate generation
3. if exactly one:
      accept
4. if small ambiguous set:
      Jev Choice
5. if unresolved:
      full LLM mapping fallback
```

Expected end state:

> Most accepted mechanism steps should require **zero model calls for step mapping**.

---

# 12. Missing reagent behavior

Jev should not invent arbitrary missing reagents.

Use deterministic evidence and optionally a Jev `Noul` gate:

```text
Does the current reaction specification appear incomplete enough
that missing chemistry must be generated before a coherent mechanism
can be proposed?
```

Behavior:

```text
low probability
    → skip missing-reagent LLM

high probability
    → call full LLM missing-reagent generator

uncertain
    → policy-configurable;
      preferably allow the major mechanism LLM to discover the issue
      before adding another preparatory call
```

Candidate rescue can continue to use a full LLM where actual missing chemical species must be invented.

---

# 13. Full LLM responsibilities after this change

The full LLM should be concentrated on "big chemistry":

## Required generative roles

- propose next elementary mechanism candidates,
- generate intermediate structures,
- generate reaction SMIRKS/electron pushes where still required,
- generate missing reagents/byproducts when no bounded candidate set exists,
- resolve mapping only when deterministic + Jev mapping fails.

## Opportunistic review role

The mechanism-proposal LLM should also receive provisional upstream decisions:

```json
{
  "ph_assessment": {...},
  "conditions_assessment": {...},
  "atom_mapping": {...},
  "reaction_type": {...}
}
```

Add a compact output field:

```json
{
  "context_review": {
    "ph": "accept",
    "conditions": "accept",
    "atom_mapping": "question",
    "reaction_type": "accept"
  }
}
```

Instruction:

> Upstream Jev and deterministic decisions are provisional context. Use them when chemically consistent. If one materially conflicts with the mechanism you propose, flag it as `question`. Do not provide commentary for accepted context.

This obtains LLM review without adding an extra review call.

A questioned upstream decision should:

- be recorded in the trace,
- lower trust in that upstream decision,
- optionally trigger fallback/re-evaluation only if required for successful execution,
- become evaluation data for improving Jev thresholds/candidate generation.

---

# 14. Candidate-ranking experiment

## 14.1 Objective

Determine whether Jev can replace LLM-based candidate ranking without reducing mechanism quality.

Do **not** immediately use Jev to reorder production candidates.

---

## 14.2 Rankers to compare

Record three rankings:

```text
A. generator-provided rank
B. Jev Choice rank
C. independent LLM-judge rank
```

The LLM judge should receive the same reaction context and candidates as Jev but should not see:

- generator rank,
- Jev rank,
- candidate labels with ordinal hints.

Use neutral/randomized candidate IDs.

---

## 14.3 Jev ranking

One `Choice` over the candidate set plus `none`.

Example:

```text
candidate_H
candidate_Q
candidate_W
none
```

Store the complete probability distribution and derive the rank order from it.

---

## 14.4 Shadow mode

Initial behavior:

```text
production choice:
    unchanged existing ranking

shadow outputs:
    Jev ranking
    LLM judge ranking
```

No ranking experiment should change the mechanism path during the first comparison phase.

---

## 14.5 Evaluation

Compare each ranker using:

- expected next-intermediate agreement,
- expected bond-change agreement,
- successful pathway completion,
- retries,
- backtracks,
- path length,
- final benchmark score,
- cost,
- latency.

Also measure:

- Jev/LLM agreement,
- Jev/generator agreement,
- LLM/generator agreement,
- cases where Jev uniquely selects the successful candidate,
- cases where Jev would have caused a failure,
- calibration of Jev top-choice probability.

---

## 14.6 Harness variants after shadow evaluation

If shadow results justify active testing, add harness variants such as:

```text
rank_generator
rank_jev
rank_llm_judge
```

Then run normal easy/medium/hard evals.

Only move toward Jev-only ranking if it improves or meets the required harness contribution gate under the project's normal evidence process.

---

# 15. Probability and uncertainty policy

Jev probability should control **whether another model call is needed**, not override deterministic chemistry.

Examples:

### Mapping

```text
one deterministic candidate
    → no Jev, no LLM

small candidate set + strong Jev separation
    → Jev select, no LLM

small candidate set + weak Jev separation
    → LLM fallback

large/unbounded candidate set
    → LLM fallback
```

### Reaction type

```text
high top probability + sufficient margin
    → use template guidance

moderate probability
    → weak guidance

low probability / no_match
    → disable template guidance
```

### Candidate rank

Initially:

```text
probability is recorded only
```

Later it may influence production ordering if eval-backed.

All thresholds must be harness/config values and empirically calibrated. Avoid baking unvalidated numerical thresholds into core logic.

---

# 16. Proposed code changes

## 16.1 New modules

Possible structure:

```text
mechanistic_agent/
  decisions/
    jev.py
    policies.py
  core/
    atom_identity.py
    mapping_candidates.py
    ranking.py
```

Alternative: keep under existing `core/` if avoiding a new package.

---

## 16.2 Jev adapter

Responsibilities:

```python
class JevDecisionClient:
    def choice(...)
    def score(...)
    def noul(...)
    def decide_many(...)
```

Must capture:

- model identifier,
- provider,
- question type,
- selected output,
- full probability distribution,
- confidence if supplied,
- latency,
- usage/cost,
- request failure.

No chemistry-specific logic should live in the generic client.

---

## 16.3 Mapping agent refactor

Current:

```text
MappingAgent
  global LLM mapping
  step LLM mapping
```

Target:

```text
MappingAgent
  generate_global_candidates()
  choose_global_candidate()
  derive_step_mapping_from_identity()
  generate_step_candidates()
  choose_step_candidate()
  llm_fallback_global()
  llm_fallback_step()
```

This preserves a single public mapping surface while making the internal policy replaceable.

---

## 16.4 Conditions agent refactor

Current grouped dispatch can remain conceptually useful.

Target:

```text
ConditionsAgent
  pH → Jev Score
  environment → Jev Choice
  compatibility → Jev Score
```

Prefer one Jev request containing all condition questions for the same reaction state.

---

## 16.5 ReactionTypeAgent

Add a Jev implementation behind the same structured output expected by existing template-guidance code.

Preserve existing fields where practical:

```json
{
  "selected_label_exact": "...",
  "selected_type_id": "...",
  "confidence": 0.0,
  "top_candidates": [...]
}
```

Populate `top_candidates` directly from the Jev probability distribution.

This minimizes coordinator changes.

---

## 16.6 Mechanism proposal schema

Add optional:

```json
"context_review": {
  "ph": "accept|question",
  "conditions": "accept|question",
  "atom_mapping": "accept|question",
  "reaction_type": "accept|question"
}
```

Do not require prose rationale unless a value is `question`.

---

## 16.7 Ranking abstraction

Add:

```python
class CandidateRanker(Protocol):
    def rank(state, candidates) -> RankingResult: ...
```

Implement:

```text
GeneratorRanker
JevRanker
LLMJudgeRanker
```

Support:

```text
active ranker
shadow rankers[]
```

in harness configuration.

---

# 17. Harness configuration

Possible additions:

```json
{
  "decision_policy": {
    "conditions": "jev",
    "global_mapping": "rdkit_jev_llm_fallback",
    "step_mapping": "identity_rdkit_jev_llm_fallback",
    "reaction_type": "jev",
    "candidate_ranker": "generator",
    "shadow_rankers": ["jev", "llm_judge"]
  },
  "jev": {
    "model": "typesafe/jev-1.13",
    "mapping_max_options": 6,
    "mapping_hard_max_options": 12,
    "thresholds": {
      "mapping_accept_probability": null,
      "mapping_min_margin": null,
      "reaction_type_active_probability": null,
      "reaction_type_min_margin": null
    }
  }
}
```

Thresholds should initially remain unset or observational until calibration data exist.

---

# 18. Trace schema additions

Record every Jev decision:

```json
{
  "decision_engine": "jev",
  "model": "typesafe/jev-...",
  "decision_type": "choice|score|noul",
  "question_id": "...",
  "selected": "...",
  "probabilities": {...},
  "confidence": 0.0,
  "fallback_triggered": false,
  "fallback_reason": null,
  "latency_ms": 0,
  "cost": 0.0
}
```

Mapping-specific trace:

```json
{
  "candidate_count_raw": 18,
  "candidate_count_after_equivalence_collapse": 4,
  "candidate_count_after_pruning": 3,
  "benchmark_mapping_in_top_k": true,
  "selected_candidate_id": "map_2"
}
```

Atom identity trace:

```json
{
  "identity_source": "persistent",
  "preserved_atom_count": 14,
  "new_atom_count": 0,
  "lost_atom_ids": [],
  "duplicate_atom_ids": []
}
```

---

# 19. Test plan

## Phase A — infrastructure tests

### Jev client

- `Choice` parsing
- `Score` parsing
- `Noul` parsing
- multi-question request
- timeout
- malformed provider response
- missing probabilities
- model metadata capture
- deterministic fallback behavior

Use mocks in `tests/fast/`; no network in fast tests.

---

## Phase B — persistent atom identity spike

Must include every test in Section 10.

**Gate:** do not remove or downgrade current `step_atom_mapping` until the multi-step, backtracking, and persistence tests pass.

---

## Phase C — mapping candidate generator

Unit tests:

- identity reaction,
- simple substitution,
- elimination,
- addition,
- carbonyl reaction,
- aromatic substitution,
- symmetric molecules,
- rearrangement,
- multi-component reactions,
- explicit proton transfer,
- stereochemical cases.

Metrics:

```text
top-1 mapping recall
top-3 mapping recall
top-6 mapping recall
mean candidate count
p95 candidate count
fraction of cases with exactly one candidate
fraction requiring Jev
fraction requiring full LLM
```

Do not evaluate Jev selection before measuring candidate recall.

---

## Phase D — Jev mapping

For cases where the acceptable mapping is in the deterministic candidate set:

```text
Jev top-1 accuracy
accuracy vs option count
accuracy vs probability
accuracy vs margin
Brier score / calibration
fallback coverage
```

Track global and step mapping separately.

---

## Phase E — conditions and reaction type

Compare old and new modules on the eval tiers.

Evaluate downstream effects, not just isolated label agreement.

Metrics:

- mechanism completion,
- pathway score,
- retries,
- backtracks,
- template disablement rate,
- reaction-type accuracy when benchmark label exists,
- full-LLM calls avoided,
- cost,
- latency.

---

## Phase F — candidate rank shadow comparison

Run:

```text
generator rank
Jev rank
LLM judge rank
```

without changing production ordering.

Then create active harness variants only if shadow data support the experiment.

---

# 20. Acceptance criteria

## Persistent atom identity

Required before replacing routine step mapping:

- 100% identity retention for surviving atoms in supported deterministic edit types.
- No duplicate persistent IDs.
- No ID reuse.
- Correct behavior across component merge/split.
- Correct pause/resume restoration.
- Correct branch/backtrack restoration.
- Mapped-SMILES/canonicalization round-trip passes.
- Multi-step benchmark test passes.
- Explicitly documented policy for implicit hydrogens.

---

## Global mapping candidate generation

Before enabling Jev selection by default:

- strong top-K benchmark recall,
- option counts small enough for bounded decisions on most eval reactions,
- symmetry-equivalent mappings collapsed,
- safe full-LLM fallback for unresolved cases.

Suggested research target, not a merge requirement:

```text
>=90% of eval reactions reduced to <=6 meaningful mapping options
```

---

## Step mapping

Suggested research target:

```text
>=95% of accepted steps require either:
  - no mapping inference because persistent IDs suffice, or
  - <=4 residual mapping choices
```

Again, treat this as an experiment target, not an assumed fact.

---

## Candidate ranking

Jev-only ranking must not become the default until an eval-backed harness change demonstrates acceptable or improved performance under the project's contribution rules.

At minimum compare:

- medium tier,
- hard tier strongly preferred,
- completion,
- retries/backtracks,
- pathway score,
- cost and latency.

---

# 21. Rollout plan

## Milestone 1 — atom identity spike

Deliver:

- `atom_identity.py` prototype,
- persistent-ID test suite,
- report of which edit/serialization paths preserve identity,
- decision: `persistent_identity_viable = true|false`.

No production behavior change.

---

## Milestone 2 — mapping candidate generator

Deliver:

- deterministic global candidate generation,
- deterministic local step candidate generation,
- equivalence collapse,
- top-K recall benchmark.

No Jev requirement yet.

---

## Milestone 3 — Jev foundation

Deliver:

- Jev adapter,
- trace support,
- mocked fast tests,
- pH Score,
- condition Choice + Score,
- reaction-type Choice.

Run as harness variant.

---

## Milestone 4 — mapping Jev + LLM fallback

Deliver:

- Jev global mapping selection,
- Jev residual step mapping,
- full-LLM ambiguity fallback,
- mapping metrics dashboard/leaderboard fields.

---

## Milestone 5 — LLM context review

Deliver:

- `context_review` schema,
- proposal prompt update,
- trace disagreement statistics.

Use disagreement as evaluation signal, not automatic proof that Jev was wrong.

---

## Milestone 6 — ranking shadow experiment

Deliver:

- Jev ranker,
- independent LLM judge ranker,
- generator rank baseline,
- neutralized candidate labels,
- shadow traces,
- comparison report.

---

## Milestone 7 — harness evolution

If supported by eval evidence:

- make Jev reaction type default,
- make Jev conditions default,
- make deterministic/Jev mapping default,
- retire routine mapping LLM calls,
- optionally make Jev candidate ranking default.

All default changes go through existing harness/eval contribution gates.

---

# 22. Failure modes and guardrails

### Jev overconfidence

Mitigation:

- probabilities never override deterministic chemistry,
- thresholds calibrated empirically,
- ambiguous cases can fall back.

### Mapping candidate generator excludes the correct mapping

Mitigation:

- measure top-K recall before Jev accuracy,
- do not prune aggressively until benchmark evidence supports it,
- preserve LLM fallback.

### Atom-map numbers collide with chemistry-facing map numbers

Mitigation:

- use sidecar persistent IDs as canonical identity,
- use atom-map numbers only as transport/debug representation where appropriate.

### Canonicalization changes atom indices

Mitigation:

- never use index as persistent identity,
- test canonicalization explicitly.

### Explicit/implicit hydrogen mismatch

Mitigation:

- define heavy-atom and proton identity policies separately,
- require explicit H where proton provenance matters.

### LLM blindly trusts wrong Jev context

Mitigation:

- proposal prompt labels upstream context provisional,
- `context_review` can flag disagreement,
- deterministic validators remain final arbiters.

### LLM review causes prompt bloat

Mitigation:

- compact fixed enum only,
- no rationale unless `question`.

### Ranker experiment changes chemistry accidentally

Mitigation:

- shadow mode first,
- active ranker only via explicit harness variant.

---

# 23. Open research questions

1. What fraction of global mappings can RDKit reduce to <=6 meaningful candidates?
2. What fraction of step mappings can disappear entirely through persistent identity?
3. Is atom-map-number transport sufficient, or should all production identity use a sidecar?
4. How should implicit proton identity be represented?
5. How many mapping options can Jev reliably distinguish for chemistry-specific inputs?
6. Does reaction-type pre-filtering improve Jev accuracy enough to justify the extra deterministic logic?
7. Can pH/conditions Jev outputs improve downstream proposal quality, or is their main benefit call reduction?
8. How often does the full mechanism LLM question upstream Jev context?
9. When LLM and Jev disagree, which is more predictive of benchmark success?
10. Does Jev candidate ranking reduce retries/backtracks relative to generator rank and independent LLM rank?
11. What probability/margin thresholds minimize full-LLM calls while preserving pathway completion?
12. Does persistent identity simplify electron-push validation enough to enable additional deterministic checks?

---

# 24. Definition of done

This project is complete when Mechanistic supports an eval-backed harness in which:

```text
pH recommendation            = Jev Score
reaction condition assessment = Jev Choice + Score
reaction type                 = Jev Choice

global atom mapping           = deterministic candidates
                                + Jev Choice
                                + LLM fallback

step atom mapping             = persistent identity when possible
                                + deterministic residual candidates
                                + Jev Choice
                                + LLM fallback

mechanism generation          = full LLM

candidate ranking             = pluggable:
                                generator / Jev / LLM judge
                                with Jev-vs-LLM comparison evidence

hard chemistry validation     = deterministic
retry/backtracking            = deterministic
```

and the eval/trace system can quantify:

- full LLM calls avoided,
- mapping candidate recall,
- Jev selection quality,
- Jev calibration,
- LLM/Jev disagreements,
- mechanism completion,
- retries,
- backtracks,
- leaderboard score,
- cost,
- latency.

The final architecture should make a full LLM call because **new chemistry must be invented**, not because the harness needs a classifier, bookkeeping decision, or atom-identity guess.
