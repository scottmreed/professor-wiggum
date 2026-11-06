"""Arrow-count tie-break over validated candidates (deterministic, AI-free).

Ported from ChemIllusion's ``mechanism_resonance_service`` (``be_distance``,
``arrows_for`` and the ``(arrows, charged_atoms)`` sort key of ``assess_step``).

Every two-electron arrow changes the bond-electron (BE) matrix by exactly 4 in
L1 distance (lone pair -> new bond: -2 on the donor's diagonal, +2 on the new
bond's off-diagonal cell; heterolysis and bond -> new bond likewise), so
``ceil(BE_L1 / 4)`` is the arrow count of an ordinary step. The BE matrices come
from ``core/bond_electron.py`` (``ugi_flower_kekule_v1``: off-diagonal =
2 x Kekulé bond order, diagonal = non-bonding electrons with H folded in), and
the L1 distance counts every diagonal cell plus each off-diagonal cell **once
per atom pair** (upper triangle), which equals ChemIllusion's
``Σ|Δfree| + 2·Σ|Δorder|``. Unmapped hydrogens are folded into their heavy atom,
so a proton transfer counts as one arrow; that is a constant offset for
candidates with the same outcome, which are the only ones compared.

The count is measured from the candidate's mapped ``reaction_smirks``, never
from ``len(electron_pushes)``: the pushes are what the model claimed, and that
is the thing being judged.

:func:`choose_arrow_count_preference` only **reorders** candidates that every
validator already passed. It never rejects, filters or changes one.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from mechanistic_agent.resonance import resonance_key
from mechanistic_agent.skeleton_alignment import _parse

from .bond_electron import build_bond_electron_view

OUTCOME_KEY_VERSION = "resonance_multiset_heavy_ge2.v1"

OutcomeKey = Tuple[str, ...]


def _fragments(species: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for item in species or []:
        text = str(item or "").strip()
        if text:
            out.extend(part for part in text.split(".") if part.strip())
    return out


def _heavy_atoms(mol: Any) -> int:
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)


def _folded_hydrogens(side: str) -> Dict[int, int]:
    """Per mapped heavy atom: hydrogens that are not BE rows (implicit, or unmapped ``[H]``).

    ``ugi_flower_kekule_v1`` folds these into the diagonal, so the H-X bond itself
    never appears as an off-diagonal cell; :func:`step_arrow_count` adds it back
    (2 electrons per H gained or lost) so a proton transfer is two arrows, as with
    explicit hydrogen atoms. Mapped ``[H:n]`` atoms are rows already and are skipped.
    """
    from rdkit import Chem

    from .mechanism_moves import mapped_smiles_parser_params

    counts: Dict[int, int] = {}
    for token in side.split("."):
        token = token.strip()
        if not token:
            continue
        mol = Chem.MolFromSmiles(token, mapped_smiles_parser_params(sanitize=True))
        if mol is None:
            raise ValueError(f"Invalid mapped SMILES: {token!r}")
        for atom in mol.GetAtoms():
            m = int(atom.GetAtomMapNum() or 0)
            if m <= 0 or atom.GetAtomicNum() == 1:
                continue
            unmapped_h = sum(
                1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 1 and not n.GetAtomMapNum()
            )
            counts[m] = int(atom.GetTotalNumHs()) + unmapped_h
    return counts


def step_arrow_count(reaction_smirks: Any) -> Optional[int]:
    """``ceil(BE_L1 / 4)`` for a mapped step; ``None`` when it cannot be measured.

    Unmeasurable: no SMIRKS, no atom maps, a BE ``projection_error``, or a step
    that changes no electrons (a real step always takes at least one arrow).
    """
    text = str(reaction_smirks or "").strip()
    if not text or ">>" not in text:
        return None
    view = build_bond_electron_view(text)
    if view.get("error") or not view.get("atom_ids"):
        return None
    delta = view.get("delta") or []
    total = 0.0
    for i, row in enumerate(delta):
        for j in range(i, len(row)):
            total += abs(float(row[j]))
    try:
        left_text, right_text = text.split(">>", 1)
        h_before, h_after = _folded_hydrogens(left_text), _folded_hydrogens(right_text)
    except Exception:  # noqa: BLE001 - keep the BE-only count
        h_before, h_after = {}, {}
    total += 2.0 * sum(abs(h_after[m] - h_before[m]) for m in set(h_before) & set(h_after))
    arrows = math.ceil(total / 4.0 - 1e-9)
    return arrows if arrows >= 1 else None


def outcome_key(resulting_state: Iterable[Any]) -> Optional[OutcomeKey]:
    """Multiset of resonance keys of the resulting species, or ``None`` if one is unparseable.

    Species with at most one heavy atom (H2O, H3O+, OH-, H+, halides, metal
    counter-ions) are left out, as the proton-transfer and skeleton keys do:
    they are the shuttles and leaving ions. Spectators carried unchanged appear
    in every candidate's state, so they compare equal without special handling.
    """
    keys: List[str] = []
    for fragment in _fragments(resulting_state):
        mol = _parse(fragment)
        if mol is None:
            return None
        if _heavy_atoms(mol) <= 1:
            continue
        key = resonance_key(fragment)
        if key is None:
            return None
        keys.append(key)
    return tuple(sorted(keys))


def candidate_arrow_count(reaction_smirks: Any, resulting_state: Iterable[Any]) -> Optional[int]:
    """:func:`step_arrow_count` when the SMIRKS describes the candidate's stated outcome.

    The arrows are measured on the SMIRKS while "same outcome" is judged on
    ``resulting_state``; the count only stands for the candidate when the SMIRKS
    product side describes that state. Whole species: every product with at least
    two heavy atoms is, up to resonance, in the state (as a sub-multiset). A
    reaction-centre SMIRKS (partial species): every product fragment is a
    substructure of the stated species, charges included — the product half of
    ``quality_scoring._smirks_matches_states``. The reactant side is not checked,
    because pushing from another contributor of the current state is exactly what
    this preference rewards. Otherwise the candidate is unmeasurable and keeps
    its rank.
    """
    text = str(reaction_smirks or "").strip()
    if ">>" not in text:
        return None
    resulting = list(resulting_state or [])
    state_keys = outcome_key(resulting)
    if state_keys is None:
        return None
    product_keys = outcome_key([text.split(">>", 1)[1]])
    if product_keys is None or Counter(product_keys) - Counter(state_keys):
        from mechanistic_agent.quality_scoring import _fragments_in_state, normalize_smirks

        raw_right = normalize_smirks(text).partition(">>")[2]
        if not _fragments_in_state(raw_right, resulting):
            return None
    return step_arrow_count(text)


def charged_atom_count(resulting_state: Iterable[Any]) -> int:
    """Atoms with a non-zero formal charge in the species :func:`outcome_key` counts."""
    total = 0
    for fragment in _fragments(resulting_state):
        mol = _parse(fragment)
        if mol is None or _heavy_atoms(mol) <= 1:
            continue
        total += sum(1 for atom in mol.GetAtoms() if atom.GetFormalCharge() != 0)
    return total


def choose_arrow_count_preference(
    resulting_states: Sequence[Sequence[Any]],
    arrow_counts: Sequence[Optional[int]],
) -> Optional[Dict[str, Any]]:
    """Pick the fewest-arrow candidate with the same outcome as the top one.

    ``resulting_states`` are validated candidates in rank order (index 0 would be
    chosen) and ``arrow_counts`` their :func:`step_arrow_count` values. Only
    candidates whose outcome (:func:`outcome_key`) equals the top's compete; the
    sort key is ``(arrows, charged_atoms, index)``. A candidate with no arrow
    count takes no part: an unmeasurable top keeps its place and an unmeasurable
    lower rank is never chosen. Returns ``None`` when the top stays first,
    otherwise ``{chosen_index, displaced_index, chosen_arrows, displaced_arrows,
    chosen_charged_atoms, displaced_charged_atoms, outcome_key, reason}``.
    """
    if len(resulting_states) < 2 or len(arrow_counts) != len(resulting_states):
        return None
    top_arrows = arrow_counts[0]
    top_outcome = outcome_key(resulting_states[0])
    if top_arrows is None or top_outcome is None:
        return None
    charged = {0: charged_atom_count(resulting_states[0])}
    best_index = 0
    best_key = (top_arrows, charged[0], 0)
    for idx in range(1, len(resulting_states)):
        arrows = arrow_counts[idx]
        if arrows is None or outcome_key(resulting_states[idx]) != top_outcome:
            continue
        charged[idx] = charged_atom_count(resulting_states[idx])
        key = (arrows, charged[idx], idx)
        if key < best_key:
            best_index, best_key = idx, key
    if best_index == 0:
        return None
    chosen_arrows = int(arrow_counts[best_index])  # type: ignore[arg-type]
    return {
        "chosen_index": best_index,
        "displaced_index": 0,
        "chosen_arrows": chosen_arrows,
        "displaced_arrows": int(top_arrows),
        "chosen_charged_atoms": charged[best_index],
        "displaced_charged_atoms": charged[0],
        "outcome_key": ".".join(top_outcome),
        "reason": (
            "fewer_arrows_same_outcome" if chosen_arrows < top_arrows else "fewer_charged_atoms_same_arrows"
        ),
    }


__all__ = [
    "OUTCOME_KEY_VERSION",
    "charged_atom_count",
    "choose_arrow_count_preference",
    "outcome_key",
    "step_arrow_count",
]
