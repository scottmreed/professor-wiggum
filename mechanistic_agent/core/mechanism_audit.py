"""Whole-mechanism audit run once the target product is reached.

Per-step atom balance cannot see what the whole path makes obvious: a second
equivalent of acetic acid that is protonated in one step and regenerated two
steps later is a catalyst, not conjured atoms, and H3O+ + AcO- is the same
matter as H2O + AcOH. With ``balance_mode = "deferred"`` the loop accepts a step
whose only failed check is atom balance and flags it; this module resolves the
flags from the net equation of the chosen path and reports efficiency findings.

Pure and deterministic (RDKit only, no LLM, no store access):

``audit_mechanism(starting, targets, steps)`` where ``steps`` is the chosen path
in order, each ``{"step_index", "current_state", "resulting_state",
"rescue_additions": {"add_reactants", "add_products"}, "balance_flag"}``.

Grades: ``exact`` (net equation balances with nothing added or flagged),
``reconciled`` (balances once catalysts / conjugate pairs / recorded reagent
additions are accounted for — every flag resolved), ``approximate`` (atoms
still appear or vanish), ``invalid_species`` (a SMILES does not parse).
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - optional runtime dependency
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover
    Chem = None  # type: ignore[assignment]
    rdMolStandardize = None  # type: ignore[assignment]

AUDIT_SCHEMA = "mechanism_audit.v1"
RESOLVED = frozenset(
    {"resolved_catalyst", "resolved_conjugate_pair", "resolved_dropped_species", "resolved_by_path"}
)


class _InvalidSpecies(ValueError):
    pass


def _tokens(species: Iterable[str]) -> List[str]:
    out: List[str] = []
    for item in species or []:
        for token in str(item or "").split("."):
            token = token.strip()
            if token:
                out.append(token)
    return out


def _mol(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise _InvalidSpecies(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


def canonical(smiles: str) -> str:
    return Chem.MolToSmiles(_mol(smiles))


def neutral_parent(smiles: str) -> str:
    """Canonical SMILES after removing charge by adding/removing protons."""
    return Chem.MolToSmiles(rdMolStandardize.Uncharger().uncharge(_mol(smiles)))


def composition(smiles: str) -> Counter:
    """Element counts including hydrogens, plus net charge under key ``+``."""
    mol = Chem.AddHs(_mol(smiles))
    counts: Counter = Counter(atom.GetSymbol() for atom in mol.GetAtoms())
    counts["+"] = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    return counts


def _composition_of(species: Counter) -> Counter:
    total: Counter = Counter()
    for smiles, n in species.items():
        for key, value in composition(smiles).items():
            total[key] += value * n
    return total


def _delta(left: Counter, right: Counter) -> Dict[str, int]:
    keys = set(left) | set(right)
    return {k: int(right.get(k, 0) - left.get(k, 0)) for k in sorted(keys) if right.get(k, 0) != left.get(k, 0)}


def _species(values: Iterable[str]) -> Counter:
    return Counter(canonical(token) for token in _tokens(values))


def _is_proton_only(delta: Mapping[str, int]) -> bool:
    return bool(delta) and set(delta) <= {"H", "+"} and delta.get("H", 0) == delta.get("+", 0)


def _heavy_atoms(smiles: str) -> int:
    return _mol(smiles).GetNumHeavyAtoms()


def _dropped_carrier_candidates(steps: Sequence[Mapping[str, Any]]) -> List[Tuple[str, int]]:
    """(species, copies) for small species (<= 1 heavy atom) that a flagged step stopped carrying."""
    out: List[Tuple[str, int]] = []
    for step in steps:
        if not step.get("balance_flag"):
            continue
        for species in sorted(_species(step.get("current_state") or [])):
            if _heavy_atoms(species) <= 1:
                for copies in (1, 2, 3):
                    if (species, copies) not in out:
                        out.append((species, copies))
        # A carrier can also be lost as a leaving group that never appears (e.g. H2O).
        for species in ("O",):
            for copies in (1, 2):
                if (species, copies) not in out:
                    out.append((species, copies))
    return out


def _has_proton_donor_or_acceptor(pool: Counter) -> bool:
    """True when some species can give or take a proton (O/N with H, a lone pair, or a charge)."""
    for smiles in pool:
        mol = _mol(smiles)
        for atom in mol.GetAtoms():
            if atom.GetFormalCharge() != 0:
                return True
            if atom.GetSymbol() in {"O", "N", "S"} and (atom.GetTotalNumHs() > 0 or atom.GetFormalCharge() == 0):
                return True
    return False


def _divides(delta: Mapping[str, int], formula: Mapping[str, int]) -> bool:
    """True when ``delta`` is a nonzero integer multiple of ``formula`` (ignoring charge)."""
    d = {k: v for k, v in delta.items() if k != "+" and v}
    f = {k: v for k, v in formula.items() if k != "+" and v}
    if not d or not f or set(d) != set(f):
        return False
    ratios = {d[k] / f[k] for k in f}
    if len(ratios) != 1:
        return False
    ratio = ratios.pop()
    return ratio != 0 and float(ratio).is_integer()


def audit_mechanism(
    *,
    starting: Sequence[str],
    targets: Sequence[str],
    steps: Sequence[Mapping[str, Any]],
    final_state: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """``final_state`` defaults to the last step's resulting state (or ``starting``)."""
    if Chem is None:  # pragma: no cover
        return {"schema": AUDIT_SCHEMA, "grade": "unavailable", "error": "RDKit not available"}
    try:
        return _audit(starting, targets, steps, final_state)
    except _InvalidSpecies as exc:
        return {"schema": AUDIT_SCHEMA, "grade": "invalid_species", "invalid_species": [str(exc)], "balanced": False}


def _audit(
    starting: Sequence[str],
    targets: Sequence[str],
    steps: Sequence[Mapping[str, Any]],
    final_state_override: Optional[Sequence[str]],
) -> Dict[str, Any]:
    ordered = sorted(steps, key=lambda s: int(s.get("step_index") or 0))
    if final_state_override is not None:
        final_state = list(final_state_override)
    else:
        final_state = list((ordered[-1].get("resulting_state") if ordered else starting) or [])

    added_left: Counter = Counter()
    added_right: Counter = Counter()
    for step in ordered:
        additions = step.get("rescue_additions") if isinstance(step.get("rescue_additions"), Mapping) else {}
        added_left += _species(additions.get("add_reactants") or [])
        added_right += _species(additions.get("add_products") or [])

    left = _species(starting) + added_left
    right = _species(final_state) + added_right

    # Anything on both sides of the net equation is a catalyst (if it was added
    # mid-path) or a spectator (if it was there from the start).
    common = left & right
    residual_left = left - common
    residual_right = right - common
    catalysts = sorted(s for s in common if added_left.get(s))
    spectators = sorted(s for s in common if not added_left.get(s))

    left_comp = _composition_of(residual_left)
    right_comp = _composition_of(residual_right)
    net_delta = _delta(left_comp, right_comp)
    balanced = not net_delta
    # A residual of exactly n protons (H and charge move together) is proton
    # bookkeeping: an acid/base catalyst in the pool (e.g. AcOH -> AcO- + H+)
    # whose conjugate was not carried in the state. Reconcile it, but report it.
    proton_reconciled = False
    if not balanced and _is_proton_only(net_delta) and _has_proton_donor_or_acceptor(left):
        balanced = True
        proton_reconciled = True
    # A flagged step that simply stopped carrying a small species (a water, a
    # hydroxide, a halide: at most one heavy atom) leaves exactly that species
    # missing from the net equation, possibly plus proton bookkeeping.
    dropped_species: List[Dict[str, Any]] = []
    if not balanced:
        for species, copies in _dropped_carrier_candidates(ordered):
            remainder = dict(net_delta)
            for key, value in composition(species).items():
                remainder[key] = remainder.get(key, 0) + value * copies
            remainder = {k: v for k, v in remainder.items() if v}
            if not remainder or (_is_proton_only(remainder) and _has_proton_donor_or_acceptor(left)):
                balanced = True
                proton_reconciled = bool(remainder)
                dropped_species.append({"species": species, "count": copies})
                break

    # Conjugate acid/base pairs left in the residual (e.g. AcO- vs AcOH, H3O+ vs H2O).
    conjugate_pairs: List[Dict[str, str]] = []
    parents_left = {s: neutral_parent(s) for s in residual_left}
    for s_right in residual_right:
        parent = neutral_parent(s_right)
        for s_left, p_left in parents_left.items():
            if p_left == parent and s_left != s_right:
                conjugate_pairs.append({"left": s_left, "right": s_right, "neutral_parent": parent})

    target_parents = {neutral_parent(t) for t in _tokens(targets)}
    final_canonical = {canonical(t) for t in _tokens(final_state)}
    final_parents = {neutral_parent(t) for t in final_canonical}
    targets_exact = {canonical(t) for t in _tokens(targets)} <= final_canonical
    targets_as_conjugate = (not targets_exact) and target_parents <= final_parents

    reagents_added = sorted(s for s in added_left if s not in common)
    catalyst_formulas = [composition(s) for s in catalysts]

    flags: List[Dict[str, Any]] = []
    for step in ordered:
        flag = step.get("balance_flag")
        if not flag:
            continue
        current = _species(step.get("current_state") or [])
        additions = step.get("rescue_additions") if isinstance(step.get("rescue_additions"), Mapping) else {}
        current += _species(additions.get("add_reactants") or [])
        resulting = _species(step.get("resulting_state") or []) + _species(additions.get("add_products") or [])
        step_delta = _delta(_composition_of(current), _composition_of(resulting))
        if not balanced:
            resolution = "unresolved"
        elif dropped_species and _divides(step_delta, composition(dropped_species[0]["species"])):
            resolution = "resolved_dropped_species"
        elif _is_proton_only(step_delta):
            resolution = "resolved_conjugate_pair"
        elif any(_divides(step_delta, formula) for formula in catalyst_formulas):
            resolution = "resolved_catalyst"
        else:
            resolution = "resolved_by_path"
        flags.append({"step_index": int(step.get("step_index") or 0), "step_delta": step_delta, "resolution": resolution})

    findings = _efficiency_findings(ordered, added_left, final_state)
    for item in dropped_species:
        findings.append({"type": "dropped_species", **item})
    if proton_reconciled:
        findings.append({"type": "unaccounted_proton", "net_delta": dict(net_delta)})

    if not balanced:
        grade = "approximate"
    elif flags or added_left or added_right or proton_reconciled or dropped_species:
        grade = "reconciled"
    else:
        grade = "exact"

    return {
        "schema": AUDIT_SCHEMA,
        "grade": grade,
        "balanced": balanced,
        "net_left": dict(sorted(residual_left.items())),
        "net_right": dict(sorted(residual_right.items())),
        "net_delta": net_delta,
        "proton_reconciled": proton_reconciled,
        "dropped_species": dropped_species,
        "catalysts": catalysts,
        "spectators": spectators,
        "reagents_added": reagents_added,
        "conjugate_pairs": conjugate_pairs,
        "targets_reached": targets_exact or targets_as_conjugate,
        "targets_as_conjugate": targets_as_conjugate,
        "flags": flags,
        "unresolved_steps": [f["step_index"] for f in flags if f["resolution"] not in RESOLVED],
        "findings": findings,
    }


def _efficiency_findings(
    steps: Sequence[Mapping[str, Any]], added_left: Counter, final_state: Sequence[str]
) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    seen: Dict[Tuple[str, ...], int] = {}
    proton_only: List[int] = []
    appears: Counter = Counter()
    for step in steps:
        index = int(step.get("step_index") or 0)
        current = tuple(sorted(_species(step.get("current_state") or []).elements()))
        resulting = tuple(sorted(_species(step.get("resulting_state") or []).elements()))
        appears.update(set(current) | set(resulting))
        if resulting in seen:
            findings.append({"type": "repeated_state", "step_index": index, "first_seen_step": seen[resulting]})
        seen.setdefault(resulting, index)
        parents_current = Counter(neutral_parent(s) for s in current)
        parents_resulting = Counter(neutral_parent(s) for s in resulting)
        if current != resulting and parents_current == parents_resulting:
            proton_only.append(index)
    for prev, nxt in zip(steps, steps[1:]):
        if Counter(_species(nxt.get("resulting_state") or [])) == Counter(_species(prev.get("current_state") or [])):
            findings.append({"type": "undo_step", "step_index": int(nxt.get("step_index") or 0),
                             "undoes_step": int(prev.get("step_index") or 0)})
    for a, b in zip(proton_only, proton_only[1:]):
        if b == a + 1:
            findings.append({"type": "mergeable_proton_transfers", "step_indices": [a, b]})
    for species in sorted(added_left):
        if not appears.get(species) and species not in {canonical(t) for t in _tokens(final_state)}:
            findings.append({"type": "unused_addition", "species": species})
    return findings


def chosen_path_from_events(events: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """The final accepted path: the last acceptance per step index, dropping steps
    abandoned by a later backtrack (an acceptance at step k discards steps > k)."""
    path: Dict[int, Dict[str, Any]] = {}
    for event in sorted(events, key=lambda e: int(e.get("seq") or 0)):
        if str(event.get("event_type") or "") != "mechanism_step_accepted":
            continue
        payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
        index = int(payload.get("step_index") or 0)
        for later in [k for k in path if k > index]:
            del path[later]
        path[index] = dict(payload)
    return [path[k] for k in sorted(path)]


def step_for_audit(payload: Mapping[str, Any], *, fallback_additions: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    additions = payload.get("rescue_additions") if isinstance(payload.get("rescue_additions"), Mapping) else None
    return {
        "step_index": int(payload.get("step_index") or 0),
        "current_state": list(payload.get("current_state") or []),
        "resulting_state": list(payload.get("resulting_state") or []),
        "rescue_additions": dict(additions or fallback_additions or {}),
        "balance_flag": payload.get("balance_flag"),
    }
