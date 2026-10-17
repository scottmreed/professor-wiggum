"""Spectator bookkeeping for a proposed mechanism step.

A step's ``reaction_smirks`` names the species it transforms (the reaction
centre). Every other current-state species is a spectator and must reappear,
unchanged and with its multiplicity, in ``resulting_state``. Models drafting a
step on a large state (enzyme active sites, several equivalents of a base,
solvent copies) often drop spectators or collapse duplicates, and the step then
fails atom balance although its chemistry is right.

``analyse_step_species`` compares the three species multisets (canonical,
map-free, explicit H removed). ``carry_spectators`` repairs a candidate only
when that is unambiguous: every SMIRKS reactant is in the current state, every
SMIRKS product is in the stated resulting state, and adding back exactly the
dropped spectators makes the step heavy-atom balanced. It only ever adds
species that are already in the current state.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

try:  # pragma: no cover - RDKit is a hard dependency of the validators
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover
    Chem = None  # type: ignore[assignment]


def canonical_species(smiles: str) -> Optional[str]:
    """Canonical SMILES without atom maps or explicit (non-bare) H atoms; None if unparseable."""
    if Chem is None:
        return None
    text = str(smiles or "").strip()
    if not text:
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    try:
        mol = Chem.RemoveHs(mol)
    except Exception:
        pass
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def heavy_atom_counts(species: Iterable[str]) -> Optional[Counter]:
    """Heavy-atom element counts (the atom-balance validator's policy); None if unparseable."""
    if Chem is None:
        return None
    counts: Counter = Counter()
    for item in species:
        mol = Chem.MolFromSmiles(str(item or "").strip())
        if mol is None:
            return None
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() > 1:
                counts[atom.GetSymbol()] += 1
    return counts


def _canonical_multiset(species: Iterable[str]) -> Optional[Counter]:
    counts: Counter = Counter()
    for item in species:
        canonical = canonical_species(item)
        if canonical is None:
            return None
        counts[canonical] += 1
    return counts


def _smirks_sides(reaction_smirks: str) -> Optional[tuple[List[str], List[str]]]:
    text = str(reaction_smirks or "").strip()
    if not text or ">>" not in text:
        return None
    text = text.split(" |", 1)[0].strip()
    left, _, right = text.partition(">>")
    if ">" in right:
        return None
    reactants = [part for part in left.split(".") if part.strip()]
    products = [part for part in right.split(".") if part.strip()]
    if not reactants or not products:
        return None
    return reactants, products


def _expand(counter: Counter) -> List[str]:
    out: List[str] = []
    for key in sorted(counter):
        out.extend([key] * int(counter[key]))
    return out


def analyse_step_species(
    current_state: List[str],
    resulting_state: List[str],
    reaction_smirks: str,
) -> Optional[Dict[str, Any]]:
    """Species-level diff of a step against its SMIRKS; None when it cannot be assessed."""
    sides = _smirks_sides(reaction_smirks)
    if sides is None:
        return None
    current = _canonical_multiset(current_state)
    resulting = _canonical_multiset(resulting_state)
    reactants = _canonical_multiset(sides[0])
    products = _canonical_multiset(sides[1])
    if current is None or resulting is None or reactants is None or products is None:
        return None
    reactants_in_state = not (reactants - current)
    spectators = current - reactants if reactants_in_state else Counter()
    expected = spectators + products
    missing = expected - resulting
    extra = resulting - expected
    return {
        "reactants_in_current_state": reactants_in_state,
        "products_in_resulting_state": not (products - resulting),
        "missing_reactants": _expand(reactants - current),
        "dropped_spectators": _expand(missing) if reactants_in_state else [],
        "unexpected_species": _expand(extra) if reactants_in_state else [],
        "spectator_count": int(sum(spectators.values())),
    }


def carry_spectators(
    current_state: List[str],
    resulting_state: List[str],
    reaction_smirks: str,
) -> Optional[Dict[str, Any]]:
    """Return ``{"resulting_state", "carried"}`` when carrying dropped spectators balances the step.

    ``carried`` holds the original current-state strings that were appended.
    Returns None whenever the repair is not clearly safe.
    """
    analysis = analyse_step_species(current_state, resulting_state, reaction_smirks)
    if not analysis or not analysis["reactants_in_current_state"] or not analysis["products_in_resulting_state"]:
        return None
    dropped = Counter(analysis["dropped_spectators"])
    if not dropped:
        return None
    current_counts = heavy_atom_counts(current_state)
    before = heavy_atom_counts(resulting_state)
    if current_counts is None or before is None or before == current_counts:
        return None

    carried: List[str] = []
    for item in current_state:
        canonical = canonical_species(item)
        if canonical and dropped.get(canonical, 0) > 0:
            carried.append(str(item))
            dropped[canonical] -= 1
    if any(count > 0 for count in dropped.values()):
        return None
    repaired = list(resulting_state) + carried
    if heavy_atom_counts(repaired) != current_counts:
        return None
    return {"resulting_state": repaired, "carried": carried, "analysis": analysis}


def _fmt_counts(counts: Any) -> str:
    if not isinstance(counts, dict) or not counts:
        return ""
    return " ".join(f"{element}{int(amount)}" for element, amount in sorted(counts.items()))


def _short(items: List[str], limit: int = 4, width: int = 60) -> str:
    shown = [item if len(item) <= width else item[: width - 1] + "…" for item in items[:limit]]
    more = f" (+{len(items) - limit} more)" if len(items) > limit else ""
    return "[" + ", ".join(shown) + "]" + more


def rejection_note(candidate: Dict[str, Any], attempt: Dict[str, Any], *, max_chars: int = 420) -> str:
    """One concise, actionable line on why a candidate was rejected."""
    rank = candidate.get("rank")
    label = str(candidate.get("intermediate_smiles") or "?")
    if len(label) > 70:
        label = label[:69] + "…"
    failed = [str(item) for item in attempt.get("failed_checks") or []]
    status = str(attempt.get("status") or "")
    reason = str(attempt.get("reason") or "")
    parts: List[str] = []
    if status == "incomplete":
        parts.append(f"incomplete payload ({reason or 'missing SMIRKS or electron pushes'})")
    elif failed:
        parts.append("failed " + ", ".join(failed))
    elif reason:
        parts.append(reason)

    validation = attempt.get("last_validation") if isinstance(attempt.get("last_validation"), dict) else {}
    for check in validation.get("checks") or []:
        if not isinstance(check, dict) or check.get("passed"):
            continue
        details = check.get("details") if isinstance(check.get("details"), dict) else {}
        name = str(check.get("name") or "")
        if name == "atom_balance":
            invalid = [str(i.get("species") or "") for i in details.get("invalid_species") or [] if isinstance(i, dict)]
            if invalid:
                parts.append(f"unparseable species {_short(invalid, 2)}")
            deficit = _fmt_counts(details.get("deficit"))
            surplus = _fmt_counts(details.get("surplus"))
            if deficit:
                parts.append(f"resulting_state is short {deficit} vs current_state")
            if surplus:
                parts.append(f"resulting_state has extra {surplus}")
        else:
            text = str(details.get("error") or details.get("message") or details.get("error_code") or "").strip()
            if text:
                parts.append(f"{name}: {text[:120]}")

    output = attempt.get("mechanism_output") if isinstance(attempt.get("mechanism_output"), dict) else {}
    smirks = str(candidate.get("reaction_smirks") or output.get("reaction_smirks") or "")
    current = [str(x) for x in output.get("current_state") or []]
    resulting = [str(x) for x in output.get("resulting_state") or candidate.get("resulting_state") or []]
    if "atom_balance" in failed and smirks and current and resulting:
        analysis = analyse_step_species(current, resulting, smirks)
        if analysis:
            if analysis["missing_reactants"]:
                parts.append(f"SMIRKS reactants not in current_state {_short(analysis['missing_reactants'], 2)}")
            if analysis["dropped_spectators"]:
                parts.append(
                    "your resulting_state dropped spectators (keep every untouched species and every copy) "
                    + _short(analysis["dropped_spectators"])
                )
            if analysis["unexpected_species"]:
                parts.append(
                    "resulting_state has species that are neither carried over nor SMIRKS products "
                    + _short(analysis["unexpected_species"], 3)
                )
    rescue = attempt.get("rescue_validation") if isinstance(attempt.get("rescue_validation"), dict) else {}
    if rescue:
        parts.append(f"reagent rescue added {_short(rescue.get('added') or [], 3)} but still short "
                     f"{_fmt_counts(rescue.get('deficit')) or '-'} / extra {_fmt_counts(rescue.get('surplus')) or '-'}")
    note = f"rank {rank} ({label}): " + "; ".join(parts or ["rejected"])
    return note if len(note) <= max_chars else note[: max_chars - 1] + "…"
