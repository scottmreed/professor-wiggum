"""Catalyst tracking for transition-metal catalysed runs (PRD_catalysis_handling P3).

Deterministic and RDKit-only:

* :func:`identify_catalysts` finds d-block species in the starting pool and gives each a
  role. Pd, Pt, Ni, Rh, Ru, Ir, Au, Os, Mo, W and Re complexes are ``catalyst``;
  other d-block species (CuI, ZnCl2, FeCl3, organozinc reagents, Ti alkoxides) are
  ``unknown`` because they are as often stoichiometric reagents or Lewis acids.
* :func:`catalyst_regeneration` decides from the accepted path whether each catalyst
  metal comes back: the starting species itself, or a metal species that already
  occurred earlier on the path (the active form after precatalyst activation).
* :func:`catalysis_tolerated` says whether a leftover species in a final state is
  catalyst bookkeeping (the catalyst or a form of it, a released phosphine or CO, a
  boron/tin/zinc/silicon by-product of a reagent in the pool) rather than junk.
* :func:`project_catalytic_cycle` turns the accepted path into the house
  ``catalytic_cycle`` spec shape (``catalytic_cycle_projection.v1``).
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from organometallic_chem.notation import to_house_convention
from organometallic_chem.organometallic_electron_count import try_count_electrons
from organometallic_chem.organometallic_smiles import is_d_block

try:  # pragma: no cover - RDKit is a hard dependency of the runtime
    from rdkit import Chem, rdBase
except Exception:  # pragma: no cover
    Chem = None  # type: ignore[assignment]
    rdBase = None  # type: ignore[assignment]

CATALYTIC_METALS = frozenset({"Pd", "Pt", "Ni", "Rh", "Ru", "Ir", "Au", "Os", "Mo", "W", "Re"})
# By-product elements a coupling partner leaves behind (Suzuki, Stille, Negishi, Hiyama).
BYPRODUCT_ELEMENTS = frozenset({"B", "Sn", "Zn", "Si"})
LEAVING_IONS = frozenset({"F", "Cl", "Br", "I", "Li", "Na", "K", "Cs"})
PROJECTION_SCHEMA = "catalytic_cycle_projection.v1"
REGENERATION_SCHEMA = "catalyst_regeneration.v1"


def _parse(smiles: str):
    if Chem is None or not smiles:
        return None
    blocker = rdBase.BlockLogs()  # noqa: F841
    try:
        mol = Chem.MolFromSmiles(str(smiles), sanitize=False)
    except Exception:
        return None
    if mol is not None:
        mol.UpdatePropertyCache(strict=False)
    return mol


def _fragments(species: Iterable[str]) -> List[str]:
    out: List[str] = []
    for item in species or []:
        out.extend(part for part in str(item or "").split(".") if part.strip())
    return out


def metals_in(smiles: str) -> List[str]:
    mol = _parse(smiles)
    if mol is None:
        return []
    return sorted({a.GetSymbol() for a in mol.GetAtoms() if is_d_block(a.GetAtomicNum())})


def species_key(smiles: str) -> Optional[str]:
    """Map-free canonical key of a species as written (no valence model applied)."""
    mol = _parse(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    blocker = rdBase.BlockLogs()  # noqa: F841
    Chem.FastFindRings(mol)
    try:
        Chem.SetAromaticity(mol)
    except Exception:
        pass
    return Chem.MolToSmiles(mol)


def metal_species(species: Iterable[str]) -> List[str]:
    return [frag for frag in _fragments(species) if metals_in(frag)]


def identify_catalysts(starting: Sequence[str], declared: Sequence[str] = ()) -> List[Dict[str, Any]]:
    """Catalyst candidates in the starting pool, each with a ``role`` and ``role_source``.

    The role is inferred when it is clear, so no model is asked:

    * ``declared``: the species was given as a catalyst (written over the reaction arrow,
      ``CreateRunRequest.catalysts``, or confirmed by the pre-analysis call). Metal-free
      declared species (DMAP, proline) are organocatalysts and are included too.
    * ``metal_class``: a Pd, Pt, Ni, Rh, Ru, Ir, Au, Os, Mo, W or Re species.

    Any other d-block species (CuI, ZnCl2, FeCl3, an organozinc reagent) has role
    ``unknown``: it is as often a stoichiometric reagent or Lewis acid. Only those are
    worth asking the model about (:func:`unclear_metal_species`).
    """
    declared_keys = {species_key(s) for s in _fragments(declared)} - {None}
    found: List[Dict[str, Any]] = []
    seen = set()
    for frag in metal_species(starting):
        key = species_key(frag)
        if key is None or key in seen:
            continue
        seen.add(key)
        metals = metals_in(frag)
        if key in declared_keys:
            role, source = "catalyst", "declared"
        elif any(m in CATALYTIC_METALS for m in metals):
            role, source = "catalyst", "metal_class"
        else:
            role, source = "unknown", None
        found.append({"smiles": frag, "key": key, "metals": metals, "role": role, "role_source": source})
    for frag in _fragments(declared):
        key = species_key(frag)
        if key is None or key in seen or metals_in(frag):
            continue
        seen.add(key)
        found.append({"smiles": frag, "key": key, "metals": [], "role": "catalyst", "role_source": "declared"})
    return found


def unclear_metal_species(starting: Sequence[str], declared: Sequence[str] = ()) -> List[str]:
    """d-block species whose role cannot be inferred (see :func:`identify_catalysts`)."""
    return [item["smiles"] for item in identify_catalysts(starting, declared) if item["role"] == "unknown"]


def is_catalytic_run(starting: Sequence[str], declared: Sequence[str] = ()) -> bool:
    """A transition-metal catalyst is present: declared, or a metal-class catalyst."""
    return any(item["role"] == "catalyst" and item["metals"] for item in identify_catalysts(starting, declared))


def catalyst_regeneration(
    starting: Sequence[str],
    states: Sequence[Sequence[str]],
    declared: Sequence[str] = (),
) -> Dict[str, Any]:
    """Whether each catalyst metal is regenerated by the end of the accepted path.

    ``states`` are the resulting states of the chosen path's steps, in order. A metal
    counts as regenerated when the last state holds the starting catalyst species, or a
    species of that metal that already occurred in an earlier state (an active form
    produced by precatalyst activation, then closed around a cycle).
    """
    catalysts = identify_catalysts(starting, declared)
    if not catalysts:
        return {"schema": REGENERATION_SCHEMA, "catalysts": [], "applies": False}
    final_state = states[-1] if states else starting
    final_keys = {species_key(s) for s in metal_species(final_state)}
    # Every species in the final state: a declared metal-free organocatalyst (DMAP)
    # counts as regenerated only when it is there itself.
    final_all = {species_key(s) for s in _fragments(final_state)}
    earlier_keys = {species_key(s) for s in metal_species(starting)}
    for state in list(states)[:-1]:
        earlier_keys |= {species_key(s) for s in metal_species(state)}
    records = []
    for item in catalysts:
        metal_final = {k for k in final_keys if k and set(metals_in(k)) & set(item["metals"])}
        if item["key"] in final_keys or item["key"] in final_all:
            via = "starting_species"
        elif metal_final & earlier_keys:
            via = "active_form"
        elif _oxidation_state(item["smiles"]) is not None and any(
            _oxidation_state(k) == _oxidation_state(item["smiles"]) for k in metal_final
        ):
            # Pd(PPh3)4 in, Pd(PPh3)2 out: ligand loss drawn implicitly, same Pd(0).
            via = "same_oxidation_state"
        else:
            via = None
        records.append({
            "smiles": item["smiles"],
            "metals": item["metals"],
            "role": item["role"],
            "role_source": item.get("role_source"),
            "regenerated": via is not None,
            "via": via,
            "final_metal_species": sorted(metal_final),
        })
    applies = any(r["role"] == "catalyst" for r in records)
    return {
        "schema": REGENERATION_SCHEMA,
        "applies": applies,
        "catalysts": records,
        "unregenerated": [r["smiles"] for r in records if r["role"] == "catalyst" and not r["regenerated"]],
    }


def _oxidation_state(smiles: str) -> Optional[int]:
    notation = to_house_convention(smiles)
    if not notation.ok:
        return None
    count, _ = try_count_electrons(notation.smiles)
    return count.oxidation_state if count is not None else None


def _is_free_l_donor(mol) -> bool:
    """A free phosphine/arsine (PR3) or CO: what a metal releases on ligand dissociation."""
    heavy = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if not heavy or any(a.GetFormalCharge() for a in heavy):
        return False
    symbols = Counter(a.GetSymbol() for a in heavy)
    if set(symbols) == {"C", "O"} and len(heavy) == 2:
        return True  # CO
    donors = [a for a in heavy if a.GetSymbol() in {"P", "As"}]
    if len(donors) != 1:
        return False
    donor = donors[0]
    return sum(1 for n in donor.GetNeighbors() if n.GetAtomicNum() > 1) == 3 and all(
        n.GetSymbol() == "C" for n in donor.GetNeighbors() if n.GetAtomicNum() > 1
    )


def catalysis_tolerated(smiles: str, starting: Sequence[str]) -> Optional[str]:
    """Reason a leftover species is catalyst bookkeeping, or ``None``.

    Applies only when the starting pool holds a d-block species.
    """
    catalysts = identify_catalysts(starting)
    if not catalysts:
        return None
    mol = _parse(smiles)
    if mol is None:
        return None
    pool_metals = {m for item in catalysts for m in item["metals"]}
    species_metals = {a.GetSymbol() for a in mol.GetAtoms() if is_d_block(a.GetAtomicNum())}
    if species_metals & pool_metals:
        return "catalyst_form"
    if _is_free_l_donor(mol):
        return "released_ligand"
    heavy = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if len(heavy) == 1 and heavy[0].GetSymbol() in LEAVING_IONS and heavy[0].GetFormalCharge() != 0:
        return "leaving_ion"  # the halide released at oxidative addition, a K+/Cs+ base cation
    elements = {a.GetSymbol() for a in mol.GetAtoms()}
    pool_elements = set()
    for frag in _fragments(starting):
        frag_mol = _parse(frag)
        if frag_mol is not None:
            pool_elements |= {a.GetSymbol() for a in frag_mol.GetAtoms()}
    if elements & BYPRODUCT_ELEMENTS & pool_elements:
        return "coupling_byproduct"
    return None


def _count(smiles: str) -> Dict[str, Any]:
    notation = to_house_convention(smiles)
    entry: Dict[str, Any] = {"smiles": smiles, "house_smiles": notation.smiles}
    if notation.ok:
        count, error = try_count_electrons(notation.smiles)
        if count is not None:
            entry.update(
                ox_state=count.oxidation_state,
                electron_count=count.valence_electrons,
                label=f"{count.metal}({count.roman_oxidation_state})",
            )
        else:
            entry["count_error"] = error
    else:
        entry["count_error"] = "; ".join(notation.ambiguous)
    return entry


def project_catalytic_cycle(
    steps: Sequence[Mapping[str, Any]],
    starting: Sequence[str],
    *,
    provenance: Optional[Mapping[str, Any]] = None,
    declared: Sequence[str] = (),
) -> Optional[Dict[str, Any]]:
    """Project the chosen path onto the house ``catalytic_cycle`` spec shape.

    ``steps`` are chosen-path steps (``current_state``, ``resulting_state``, optional
    ``metal_center`` details and ``step_index``). Returns ``None`` when no catalyst
    metal is present. The metal-species sequence is split at the first species that
    recurs: steps before it are the ``preamble`` (precatalyst activation), the loop
    from that species back to itself is the ``cycle``. ``closed`` is false when no
    species recurs; ChemIllusion then shows the linear path.
    """
    catalysts = [c for c in identify_catalysts(starting, declared) if c["role"] == "catalyst" and c["metals"]]
    if not catalysts:
        return None
    metal = catalysts[0]["metals"][0]
    ordered = sorted(steps, key=lambda s: int(s.get("step_index") or 0))

    def _metal_species_in(state: Sequence[str]) -> List[str]:
        return [s for s in metal_species(state) if metal in metals_in(s)]

    sequence: List[str] = []  # species keys, in path order, each metal-state change
    drawings: Dict[str, str] = {}
    transitions: List[Dict[str, Any]] = []
    first = _metal_species_in(starting)
    if first:
        key = species_key(first[0])
        sequence.append(key)
        drawings[key] = first[0]
    for step in ordered:
        before = Counter(species_key(s) for s in _metal_species_in(step.get("current_state") or []))
        after_list = _metal_species_in(step.get("resulting_state") or [])
        after = Counter(species_key(s) for s in after_list)
        formed = list((after - before).elements())
        if len(formed) != 1:
            continue
        key = formed[0]
        drawings.setdefault(key, next(s for s in after_list if species_key(s) == key))
        consumed = Counter(species_key(s) for s in _fragments(step.get("current_state") or [])) - Counter(
            species_key(s) for s in _fragments(step.get("resulting_state") or [])
        )
        released = Counter(species_key(s) for s in _fragments(step.get("resulting_state") or [])) - Counter(
            species_key(s) for s in _fragments(step.get("current_state") or [])
        )
        metal_center = step.get("metal_center") if isinstance(step.get("metal_center"), Mapping) else {}
        kind = (metal_center or {}).get("declared_kind")
        if not kind:
            pairs = (metal_center or {}).get("pairs") or []
            kinds = (pairs[0].get("consistent_kinds") if pairs else None) or []
            kind = kinds[0] if len(kinds) == 1 else "other"
        transitions.append({
            "step_index": int(step.get("step_index") or 0),
            "from": sequence[-1] if sequence else None,
            "to": key,
            "kind": kind,
            "enters": sorted(k for k in consumed.elements() if k and not metals_in(k)),
            "leaves": sorted(k for k in released.elements() if k and not metals_in(k)),
        })
        sequence.append(key)

    start_at = None
    for i, key in enumerate(sequence):
        if key in sequence[i + 1:]:
            start_at = i
            break
    closed = start_at is not None
    if closed:
        end_at = sequence.index(sequence[start_at], start_at + 1)
        cycle_keys = sequence[start_at:end_at]
        cycle_transitions = transitions[start_at:end_at]
        preamble = transitions[:start_at]
    else:
        cycle_keys = sequence
        cycle_transitions = transitions
        preamble = []

    ids = {key: f"m{i + 1}" for i, key in enumerate(dict.fromkeys(cycle_keys))}
    species = []
    for key, sid in ids.items():
        entry = _count(drawings.get(key) or key)
        spec: Dict[str, Any] = {"id": sid, "label": entry.get("label") or metal}
        if entry.get("house_smiles") and "count_error" not in entry:
            spec["smiles"] = entry["house_smiles"]
        elif "ox_state" in entry:
            spec["ox_state"] = entry["ox_state"]
        species.append(spec)
    spec_steps = [
        {
            "from": ids.get(t["from"]) if t["from"] in ids else None,
            "to": ids.get(t["to"]),
            "kind": t["kind"] if t["kind"] else "other",
            "label": str(t["kind"] or "step").replace("_", " "),
            "enters": t["enters"],
            "leaves": t["leaves"],
        }
        for t in cycle_transitions
    ]
    return {
        "schema": PROJECTION_SCHEMA,
        "metal": metal,
        "closed": closed,
        "preamble": [
            {"step_index": t["step_index"], "kind": t["kind"], "enters": t["enters"], "leaves": t["leaves"]}
            for t in preamble
        ],
        "cycle": {
            "title": f"Predicted {metal}-catalysed cycle",
            "species": species,
            "steps": spec_steps,
            "legend": "Predicted by Professor Wiggum; not an authored mechanism.",
        },
        "step_indices": [t["step_index"] for t in cycle_transitions],
        "provenance": {"source": "predicted", **dict(provenance or {})},
    }


__all__ = [
    "BYPRODUCT_ELEMENTS",
    "CATALYTIC_METALS",
    "PROJECTION_SCHEMA",
    "catalysis_tolerated",
    "catalyst_regeneration",
    "identify_catalysts",
    "is_catalytic_run",
    "unclear_metal_species",
    "metal_species",
    "metals_in",
    "project_catalytic_cycle",
    "species_key",
]
