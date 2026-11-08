# Explicit Mechanism Move Notation

This repository now uses an explicit move block inside CXSMILES/SMIRKS metadata:

`|mech:v1;lp:4>2;sigma:2-3>3|`

The reaction SMIRKS stays standard. The mechanism encoding lives in CX metadata so it does not collide with base SMILES or SMIRKS syntax.

## Why this replaces `dbe`

The old `|dbe:...|` format encoded net bond deltas only. It was compact, but it threw away the actual electron source. That made three important tasks harder:

- LLM prompting: the model had to infer whether a move came from a lone pair, pi bond, or sigma bond.
- Deterministic validation: the validator could only check arithmetic balance, not whether the described move matched the reaction step.
- Learning/evals: approved traces were less reusable because the mechanistic intent was under-specified.

The new format makes the source explicit while remaining short enough for repeated LLM use.

## Syntax

- `lp:a>b`
  Lone-pair attack from atom `a` onto atom `b`.
- `pi:a-b>c`
  Pi-bond donation from bond `(a,b)` through atom `b` onto atom `c`.
- `sigma:a-b>c`
  Sigma-bond attack from bond `(a,b)` through atom `b` onto atom `c`.

Examples:

- SN2 displacement:
  `[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] |mech:v1;lp:4>2;sigma:2-3>3|`
- Carbonyl attack plus pi shift:
  `[CH:4](=[O:1])...[O-:2]>>... |mech:v1;lp:2>4;pi:4-1>1|`

## Structured `electron_pushes`

The runtime also normalizes `electron_pushes` into explicit objects:

```json
{
  "kind": "sigma_bond",
  "source_bond": ["2", "3"],
  "through_atom": "3",
  "target_atom": "3",
  "electrons": 2
}
```

Legacy `{start_atom, end_atom, electrons}` entries are still read as lone-pair attacks during the transition.

## LLM value

- Fewer latent decisions: the model chooses the electron source explicitly instead of hiding it in post-hoc bond deltas.
- Better repairability: malformed outputs can be normalized from structured move objects into `|mech:v1;...|`.
- Better validator signal: the runtime now compares implied bond changes against the reaction SMIRKS instead of only checking that deltas sum to zero.
- Better retrieval: few-shot examples become semantically sharper because the notation names the mechanistic source directly.

## Parsing and validation

- Shared parser: [mechanism_moves.py](../mechanistic_agent/core/mechanism_moves.py)
- Proposal schema/prompt: [tool_schemas.py](../mechanistic_agent/tool_schemas.py) and [SKILL.md](../skills/mechanistic/propose_mechanism_step/SKILL.md)
- Deterministic validation: [tools.py](../mechanistic_agent/tools.py) and [validator.py](../skills/mechanistic/bond_electron_validation/validator.py)
- Arrow-push annotation: [arrow_push.py](../mechanistic_agent/core/arrow_push.py)

## Migration rule

New runtime outputs should emit only `|mech:v1;...|`.

Legacy `dbe` strings are no longer the authoring target. They may still exist in historical traces, but current proposal repair and validation flow is centered on the explicit move notation.

## References

- [2512.05722v1.pdf](file:///Users/scottreed/Desktop/Mechanistic/Refs/2512.05722v1.pdf) - This variant uses explicit terms for the sake of the LLMs

## One-electron extension

Existing pair tokens are unchanged. Radical bond contributions use `rad:a>b`;
one-electron bond moves use `pi:a-b>c:e1` or `sigma:a-b>c:e1`. The second
source-bond map is the delivering endpoint. Atom-to-atom single-electron transfer
(with no bond formation) uses `et:a>b`. Structured kinds are `radical` and
`electron_transfer`; both require `electrons: 1`. Lone pairs require 2.

For example homolysis is `sigma:1-2>2:e1;sigma:2-1>1:e1`, recombination is
`rad:1>2;rad:2>1`. Radical addition to a double bond needs the radical contribution
plus **two** fishhooks from the pi bond: `rad:1>2;pi:3-2>1:e1;pi:2-3>3:e1`.

The validator checks the complete mapped before/after electron inventory, source
capacity, atom identities, and runtime species multiplicities. Transferred H
must be explicit and mapped. Unsupported atoms and incomplete mapped radical
states fail closed. This is main-group electron accounting, not a spin,
excitation-energy, or reaction-feasibility prediction. Transition-metal
photoredox cycles are outside this initial support.

ReactionFocus adds `changed_radical_electrons`. The pair-arrow-count preference
abstains on radical species rather than treating a fishhook pair as one arrow.
Runtime and quality scoring use the same radical validation contract.
