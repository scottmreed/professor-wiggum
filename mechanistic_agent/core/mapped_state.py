"""Mapped-state executor and persistent atom identity (PRD §9, §10, §16.8).

This module answers the Phase B spike question of
``docs/PRD_jev_atom_identity_mechanistic.md``: can the proposal LLM's mapped
``reaction_smirks`` (Route A) or its ``mech:`` / ``electron_pushes`` moves
(Route B) be *executed* against a persistently atom-mapped ``current_state`` to
derive ``resulting_state``, carry atom identity through the step, and check the
derived state against the LLM's stated ``resulting_state``?

Identity model (§9.3)
---------------------
* ``PersistentAtomId`` values are plain ``int`` handles issued by a
  :class:`PersistentAtomIdAllocator`. Issuance is monotonic and ids are never
  reused, including across snapshot/restore (backtracking) and resume.
* Atom-map numbers are the *serialization bridge*: a :class:`MappedState`
  stores mapped SMILES plus a sidecar ``map_number -> PersistentAtomId`` table
  and a ``PersistentAtomId -> AtomRecord`` table (component, current index,
  provenance, original map). ``Atom.GetIdx()`` is never used as identity; it is
  recorded only as ``current_index`` and changes with canonicalization.
* Custom RDKit atom properties (``SetIntProp``) do **not** survive a SMILES
  round trip; only the atom-map number does. This is asserted in
  ``tests/fast/test_persistent_atom_identity.py`` (10.1).

Hydrogen policy (§9.2, §10.6)
-----------------------------
* ``"heavy_atom"``: the state stores heavy atoms only (hydrogens folded into
  bracket H counts). Heavy-atom identity is required and preserved; proton
  identity is *undefined*. Template hydrogens (``[H:n]``) in a SMIRKS are matched
  by element only (any attached H is equivalent), and ``mech:`` moves that name
  a hydrogen map number are resolved through the SMIRKS left side to the heavy
  atom carrying it. A released proton (``[H+]``) gets a fresh id with provenance
  ``"proton:released"``; an absorbed free proton's id is reported as lost.
* ``"explicit"``: hydrogens are explicit, mapped atoms with ids of their own and
  are tracked exactly like heavy atoms (the benchmark representation). A proton
  transferred donor->acceptor keeps its id.
* ``"auto"`` (default): ``explicit`` when the input state contains explicit
  hydrogen atoms, else ``heavy_atom``.

Route A notes (RDKit ``rdChemReactions``)
-----------------------------------------
* Left and right sides are wrapped in component-level grouping ``(A.B)>>(C.D)``
  so a multi-component state is matched as one reactant.
* Matching is first *map-constrained*: each mapped template atom is AND-ed with
  a ``molAtomMapNumber`` equality query so the template can only bind the state
  atom carrying the same map number. If that fails (e.g. the LLM invented its
  own numbering on a stripped state) an *unconstrained* match is tried and the
  result is flagged ``match_mode="unconstrained"``. Either way identity is
  carried by RDKit's ``react_atom_idx`` product property, not by template maps.
* RDKit reaction-SMARTS semantics leave an atom's charge unchanged when the
  product template does not state one (``[O+:1]>>[O:1]`` keeps ``+1``). The
  benchmark and the LLM write SMIRKS in SMILES semantics (bracket atom without a
  charge means neutral), so after the reaction runs, formal charges and bracket H
  counts of template-mapped atoms are overwritten from a SMILES parse of the
  right side. ``ReactionFromSmarts(useSmiles=True)`` was tested and does not
  apply product charges at all, so it is not used.
* RDKit drops reactant fragments that no template atom touches; those
  components are passed through unchanged.
* Matching is tried on the Kekulé form as written first, then on the
  aromatic-perceived form, so Kekulé SMIRKS match aromatic states and vice
  versa.

Route B notes (``RWMol`` edits)
-------------------------------
Two-electron moves only. ``lp:a>b`` forms ``a-b`` (a +1, b -1); ``x-y>y``
heterolysis lowers ``x-y`` (x +1, y -1); ``x-y>z`` shifts the pair so ``x-y``
drops and ``y-z`` rises (x +1, z -1). Edits run on the Kekulé form as written,
then the result is fully sanitized. Moves carry no stereo information, so
stereocenters at reacting atoms are not re-specified (remote centers survive).
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

from rdkit import Chem, rdBase
from rdkit.Chem import rdChemReactions, rdqueries

from mechanistic_agent.core.mechanism_moves import (
    MechanismMove,
    extract_mechanism_moves,
    mapped_smiles_parser_params,
    normalize_electron_pushes,
    split_cxsmiles_metadata,
)
from mechanistic_agent.smiles_utils import species_match_signature

HydrogenPolicy = Literal["auto", "heavy_atom", "explicit"]
PersistentAtomId = int

_MAX_REACTION_PRODUCTS = 64

_BOND_TYPES = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
}


class MappedStateError(ValueError):
    """Raised when a state cannot be parsed or identity invariants are broken."""


# ---------------------------------------------------------------------------
# Identity bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class PersistentAtomIdAllocator:
    """Monotonic, never-reused issuer of persistent atom ids."""

    next_id: int = 1

    def allocate(self) -> PersistentAtomId:
        value = self.next_id
        self.next_id += 1
        return value

    def observe(self, pid: int) -> None:
        """Advance the high-water mark past an externally restored id."""
        if pid >= self.next_id:
            self.next_id = pid + 1

    def as_dict(self) -> Dict[str, int]:
        return {"next_id": self.next_id}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersistentAtomIdAllocator":
        return cls(next_id=max(1, int((data or {}).get("next_id") or 1)))


@dataclass
class AtomRecord:
    """Sidecar identity record for one atom (never keyed by ``GetIdx()``)."""

    pid: PersistentAtomId
    element: str
    map_number: int
    component: int
    current_index: int
    provenance: str
    original_map: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "pid": self.pid,
            "element": self.element,
            "map_number": self.map_number,
            "component": self.component,
            "current_index": self.current_index,
            "provenance": self.provenance,
            "original_map": self.original_map,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AtomRecord":
        return cls(
            pid=int(data["pid"]),
            element=str(data["element"]),
            map_number=int(data["map_number"]),
            component=int(data.get("component", 0)),
            current_index=int(data.get("current_index", -1)),
            provenance=str(data.get("provenance") or ""),
            original_map=int(data.get("original_map", 0)),
        )


def _parser_params(sanitize: bool = False) -> Chem.SmilesParserParams:
    return mapped_smiles_parser_params(sanitize=sanitize)


def _parse_raw(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(str(smiles or "").strip(), _parser_params(sanitize=False))
    if mol is None:
        raise MappedStateError(f"unparseable SMILES: {smiles!r}")
    return mol


def _sanitize_kekule_as_written(mol: Chem.Mol) -> None:
    """Sanitize without aromaticity perception so written Kekulé bonds survive."""
    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
    Chem.Kekulize(mol, clearAromaticFlags=True)


def _freeze_hydrogens(mol: Chem.Mol) -> None:
    """Pin implicit H counts so later bond edits cannot silently re-derive them."""
    for atom in mol.GetAtoms():
        atom.SetNumExplicitHs(int(atom.GetTotalNumHs()))
        atom.SetNoImplicit(True)


def _remove_hs_params() -> Chem.RemoveHsParameters:
    params = Chem.RemoveHsParameters()
    params.removeMapped = True
    params.removeDegreeZero = False
    params.removeHydrides = True
    params.removeIsotopes = False
    params.showWarnings = False
    return params


def _has_explicit_h(mol: Chem.Mol) -> bool:
    return any(a.GetAtomicNum() == 1 and a.GetDegree() > 0 for a in mol.GetAtoms())


def _sanitize_product(mol: Chem.Mol) -> None:
    """Validate valences/aromaticity, then store the Kekulé form as derived.

    Keeping the derived Kekulé bonds (instead of RDKit's re-kekulization of an
    aromatic ring) is what lets Route B apply the next step's ``pi:`` moves to
    the bonds the benchmark/LLM wrote.
    """
    probe = Chem.Mol(mol)
    Chem.SanitizeMol(probe)
    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
    Chem.Kekulize(mol, clearAromaticFlags=True)


def mapped_signature(species: Sequence[str]) -> List[str]:
    """Sorted canonical *mapped* SMILES with aromaticity perceived.

    Independent of Kekulé choice and atom order, sensitive to map numbers:
    use it to compare a derived mapping with a benchmark mapping.
    """
    out: List[str] = []
    for item in species or []:
        for token in str(item or "").split("."):
            if not token.strip():
                continue
            mol = Chem.MolFromSmiles(token, _parser_params(sanitize=True))
            out.append(Chem.MolToSmiles(mol) if mol is not None else token.strip())
    return sorted(out)


def _canonical_mapped(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol)


def _stripped_signature_of_mol(mol: Chem.Mol) -> str:
    work = Chem.Mol(mol)
    for atom in work.GetAtoms():
        atom.SetAtomMapNum(0)
    try:
        work = Chem.RemoveHs(work, _remove_hs_params())
    except Exception:
        pass
    return species_match_signature(Chem.MolToSmiles(work))


def species_signature(smiles: str) -> str:
    """Map-stripped, H-suppressed canonical signature for one species.

    Explicit-H mapped benchmark SMILES and heavy-atom unmapped LLM SMILES for
    the same species produce the same signature. Unparseable text falls back to
    :func:`species_match_signature` (trimmed text) so comparison stays total.
    """
    text = str(smiles or "").strip()
    if not text:
        return ""
    blocker = rdBase.BlockLogs()
    try:
        mol = Chem.MolFromSmiles(text, _parser_params(sanitize=True))
        if mol is None:
            return species_match_signature(text)
        return _stripped_signature_of_mol(mol)
    finally:
        del blocker


def state_signature(species: Sequence[str]) -> List[str]:
    """Sorted multiset of species signatures (fragments split on '.')."""
    out: List[str] = []
    for item in species or []:
        for token in str(item or "").split("."):
            if token.strip():
                out.append(species_signature(token))
    return sorted(out)


@dataclass
class MappedState:
    """A loop state with persistent atom identity.

    ``species`` holds canonical mapped SMILES (one entry per connected
    component). ``map_to_pid`` is the serialization bridge; ``records`` is the
    canonical identity table.
    """

    species: List[str]
    map_to_pid: Dict[int, PersistentAtomId]
    records: Dict[PersistentAtomId, AtomRecord]
    allocator: PersistentAtomIdAllocator
    hydrogen_policy: Literal["heavy_atom", "explicit"]
    next_map: int = 1
    retired_pids: List[PersistentAtomId] = field(default_factory=list)
    retired_maps: List[int] = field(default_factory=list)

    # -- construction -----------------------------------------------------

    @classmethod
    def from_smiles(
        cls,
        species: Sequence[str],
        *,
        allocator: Optional[PersistentAtomIdAllocator] = None,
        hydrogen_policy: HydrogenPolicy = "auto",
        provenance: str = "initial",
    ) -> "MappedState":
        """Parse a (possibly partly) mapped state and issue ids for every atom.

        Atoms without a map number get a fresh map number. Under
        ``heavy_atom`` policy explicit hydrogens are folded into H counts.
        """
        blocker = rdBase.BlockLogs()
        try:
            return cls._from_smiles(
                species, allocator=allocator, hydrogen_policy=hydrogen_policy, provenance=provenance
            )
        finally:
            del blocker

    @classmethod
    def _from_smiles(
        cls,
        species: Sequence[str],
        *,
        allocator: Optional[PersistentAtomIdAllocator],
        hydrogen_policy: HydrogenPolicy,
        provenance: str,
    ) -> "MappedState":
        allocator = allocator or PersistentAtomIdAllocator()
        tokens = [t for item in species or [] for t in str(item or "").split(".") if t.strip()]
        combined = _parse_raw(".".join(tokens)) if tokens else Chem.Mol()
        policy: Literal["heavy_atom", "explicit"]
        if hydrogen_policy == "auto":
            policy = "explicit" if _has_explicit_h(combined) else "heavy_atom"
        else:
            policy = hydrogen_policy  # type: ignore[assignment]
        if tokens:
            _sanitize_kekule_as_written(combined)
            if policy == "heavy_atom":
                combined = Chem.RemoveHs(combined, _remove_hs_params(), sanitize=False)
                combined.UpdatePropertyCache(strict=False)
            _freeze_hydrogens(combined)
        used = {a.GetAtomMapNum() for a in combined.GetAtoms() if a.GetAtomMapNum() > 0}
        next_map = (max(used) + 1) if used else 1
        seen: set[int] = set()
        for atom in combined.GetAtoms():
            num = atom.GetAtomMapNum()
            if num <= 0 or num in seen:
                atom.SetAtomMapNum(next_map)
                next_map += 1
            seen.add(atom.GetAtomMapNum())
        state = cls(
            species=[],
            map_to_pid={},
            records={},
            allocator=allocator,
            hydrogen_policy=policy,
            next_map=next_map,
        )
        original = {a.GetIdx(): a.GetAtomMapNum() for a in combined.GetAtoms()}
        for atom in combined.GetAtoms():
            pid = allocator.allocate()
            state.map_to_pid[atom.GetAtomMapNum()] = pid
            state.records[pid] = AtomRecord(
                pid=pid,
                element=atom.GetSymbol(),
                map_number=atom.GetAtomMapNum(),
                component=-1,
                current_index=-1,
                provenance=provenance,
                original_map=original[atom.GetIdx()],
            )
        state._set_species_from_mol(combined)
        return state

    def _set_species_from_mol(self, mol: Chem.Mol) -> None:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False) if mol.GetNumAtoms() else ()
        self.species = []
        for comp, frag in enumerate(frags):
            smiles = _canonical_mapped(frag)
            self.species.append(smiles)
        self._refresh_indices()

    def _refresh_indices(self) -> None:
        """Recompute component / current_index from the serialized species."""
        for comp, smiles in enumerate(self.species):
            mol = _parse_raw(smiles)
            for atom in mol.GetAtoms():
                pid = self.map_to_pid.get(atom.GetAtomMapNum())
                if pid is None:
                    raise MappedStateError(f"map {atom.GetAtomMapNum()} has no persistent id")
                rec = self.records[pid]
                rec.component = comp
                rec.current_index = atom.GetIdx()
                rec.map_number = atom.GetAtomMapNum()

    # -- views ------------------------------------------------------------

    def to_mol(self) -> Chem.Mol:
        mol = _parse_raw(".".join(self.species)) if self.species else Chem.Mol()
        if self.species:
            _sanitize_kekule_as_written(mol)
            _freeze_hydrogens(mol)
        return mol

    def mapped_smiles(self, *, include_hydrogens: bool = True) -> List[str]:
        """Mapped species. ``include_hydrogens=False`` hides explicit H atoms."""
        if include_hydrogens or self.hydrogen_policy == "heavy_atom":
            return list(self.species)
        out: List[str] = []
        for smiles in self.species:
            mol = Chem.MolFromSmiles(smiles, _parser_params(sanitize=True))
            params = Chem.RemoveHsParameters()
            params.removeMapped = True
            params.showWarnings = False
            out.append(Chem.MolToSmiles(Chem.RemoveHs(mol, params)))
        return out

    def stripped_smiles(self) -> List[str]:
        return [species_signature(s) for s in self.species]

    def signature(self) -> List[str]:
        return state_signature(self.species)

    def pids(self) -> List[PersistentAtomId]:
        return sorted(self.map_to_pid.values())

    def pid_by_map(self, map_number: int) -> Optional[PersistentAtomId]:
        return self.map_to_pid.get(int(map_number))

    def map_by_pid(self) -> Dict[PersistentAtomId, int]:
        return {pid: m for m, pid in self.map_to_pid.items()}

    # -- persistence ------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """JSON-serializable snapshot (branch points, SQLite, files)."""
        return {
            "schema": "mapped_state.v1",
            "species": list(self.species),
            "map_to_pid": {str(k): v for k, v in self.map_to_pid.items()},
            "records": [r.as_dict() for r in self.records.values()],
            "allocator": self.allocator.as_dict(),
            "hydrogen_policy": self.hydrogen_policy,
            "next_map": self.next_map,
            "retired_pids": list(self.retired_pids),
            "retired_maps": list(self.retired_maps),
        }

    @classmethod
    def restore(
        cls,
        snapshot: Dict[str, Any],
        *,
        allocator: Optional[PersistentAtomIdAllocator] = None,
    ) -> "MappedState":
        """Rebuild from :meth:`snapshot`.

        When a live ``allocator`` is passed (backtracking), it is kept and only
        advanced, so ids issued on a discarded path are never reissued.
        """
        restored_alloc = PersistentAtomIdAllocator.from_dict(snapshot.get("allocator") or {})
        if allocator is not None:
            allocator.observe(restored_alloc.next_id - 1)
            restored_alloc = allocator
        state = cls(
            species=list(snapshot.get("species") or []),
            map_to_pid={int(k): int(v) for k, v in (snapshot.get("map_to_pid") or {}).items()},
            records={int(r["pid"]): AtomRecord.from_dict(r) for r in snapshot.get("records") or []},
            allocator=restored_alloc,
            hydrogen_policy=snapshot.get("hydrogen_policy") or "heavy_atom",
            next_map=int(snapshot.get("next_map") or 1),
            retired_pids=list(snapshot.get("retired_pids") or []),
            retired_maps=list(snapshot.get("retired_maps") or []),
        )
        for pid in state.records:
            state.allocator.observe(pid)
        return state

    def to_json(self) -> str:
        return json.dumps(self.snapshot(), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "MappedState":
        return cls.restore(json.loads(text))

    def copy(self) -> "MappedState":
        clone = MappedState.restore(copy.deepcopy(self.snapshot()), allocator=self.allocator)
        return clone


MappedStateInput = Union[MappedState, Sequence[str]]


def _coerce_state(state: MappedStateInput, hydrogen_policy: HydrogenPolicy) -> MappedState:
    if isinstance(state, MappedState):
        return state
    return MappedState.from_smiles(list(state), hydrogen_policy=hydrogen_policy)


# ---------------------------------------------------------------------------
# Execution result
# ---------------------------------------------------------------------------


@dataclass
class ExecutionResult:
    """Outcome of executing one step against a mapped state."""

    ok: bool
    route: Literal["smirks", "moves"]
    resulting_state_mapped: List[str] = field(default_factory=list)
    resulting_state: List[str] = field(default_factory=list)
    identity_map: Dict[PersistentAtomId, int] = field(default_factory=dict)
    preserved_ids: List[PersistentAtomId] = field(default_factory=list)
    new_ids: List[PersistentAtomId] = field(default_factory=list)
    lost_ids: List[PersistentAtomId] = field(default_factory=list)
    duplicate_ids: List[PersistentAtomId] = field(default_factory=list)
    smirks_state_agreement: Optional[bool] = None
    agreement_detail: Dict[str, Any] = field(default_factory=dict)
    match_mode: Optional[str] = None
    distinct_outcomes: int = 0
    hydrogen_policy: str = ""
    error: Optional[str] = None
    failure_category: Optional[str] = None
    next_state: Optional[MappedState] = None

    def as_dict(self, *, include_state: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "ok": self.ok,
            "route": self.route,
            "resulting_state_mapped": list(self.resulting_state_mapped),
            "resulting_state": list(self.resulting_state),
            "preserved_id_count": len(self.preserved_ids),
            "new_ids": list(self.new_ids),
            "lost_ids": list(self.lost_ids),
            "duplicate_ids": list(self.duplicate_ids),
            "smirks_state_agreement": self.smirks_state_agreement,
            "agreement_detail": dict(self.agreement_detail),
            "match_mode": self.match_mode,
            "distinct_outcomes": self.distinct_outcomes,
            "hydrogen_policy": self.hydrogen_policy,
            "error": self.error,
            "failure_category": self.failure_category,
        }
        if include_state and self.next_state is not None:
            payload["next_state"] = self.next_state.snapshot()
        return payload


def compare_states(derived: Sequence[str], expected: Optional[Sequence[str]]) -> Tuple[Optional[bool], Dict[str, Any]]:
    """Multiset comparison of map-stripped canonical signatures."""
    if expected is None:
        return None, {}
    expected_list = [str(s) for s in expected if str(s or "").strip()]
    if not expected_list:
        return None, {}
    d_sig = state_signature(derived)
    e_sig = state_signature(expected_list)
    missing = list(e_sig)
    extra: List[str] = []
    for sig in d_sig:
        if sig in missing:
            missing.remove(sig)
        else:
            extra.append(sig)
    return d_sig == e_sig, {
        "derived_signature": d_sig,
        "expected_signature": e_sig,
        "missing_from_derived": missing,
        "extra_in_derived": extra,
        "set_agreement": set(d_sig) == set(e_sig),
    }


def _finalize(
    *,
    route: Literal["smirks", "moves"],
    state: MappedState,
    product: Chem.Mol,
    source_pid: Dict[int, Optional[PersistentAtomId]],
    new_provenance: Dict[int, str],
    template_map_hint: Dict[int, int],
    expected: Optional[Sequence[str]],
    match_mode: Optional[str],
    distinct_outcomes: int,
    step_label: str,
) -> ExecutionResult:
    """Assign map numbers/ids to a sanitized product and build the result.

    ``source_pid[idx]`` is the pid carried into product atom ``idx`` (None for
    a new atom); ``template_map_hint[idx]`` is a preferred map number for new
    atoms (the LLM's own number) when it is free.
    """
    next_state = MappedState(
        species=[],
        map_to_pid={},
        records={},
        allocator=state.allocator,
        hydrogen_policy=state.hydrogen_policy,
        next_map=state.next_map,
        retired_pids=list(state.retired_pids),
        retired_maps=list(state.retired_maps),
    )
    old_map_by_pid = state.map_by_pid()
    taken_maps = set(old_map_by_pid.values()) | set(state.retired_maps)
    seen: Dict[PersistentAtomId, int] = {}
    duplicates: List[PersistentAtomId] = []
    new_ids: List[PersistentAtomId] = []
    for atom in product.GetAtoms():
        idx = atom.GetIdx()
        pid = source_pid.get(idx)
        if pid is not None and pid in seen:
            duplicates.append(pid)
            pid = None
        if pid is None:
            pid = state.allocator.allocate()
            new_ids.append(pid)
            hint = template_map_hint.get(idx, 0)
            if hint > 0 and hint not in taken_maps:
                map_number = hint
            else:
                map_number = max(next_state.next_map, max(taken_maps, default=0) + 1)
            taken_maps.add(map_number)
            next_state.next_map = max(next_state.next_map, map_number + 1)
            next_state.records[pid] = AtomRecord(
                pid=pid,
                element=atom.GetSymbol(),
                map_number=map_number,
                component=-1,
                current_index=-1,
                provenance=new_provenance.get(idx)
                or ("proton:released" if atom.GetAtomicNum() == 1 and state.hydrogen_policy == "heavy_atom" else f"{step_label}:new"),
                original_map=hint,
            )
        else:
            map_number = old_map_by_pid[pid]
            rec = copy.copy(state.records[pid])
            next_state.records[pid] = rec
        seen[pid] = idx
        atom.SetAtomMapNum(map_number)
        next_state.map_to_pid[map_number] = pid
    preserved = sorted(pid for pid in seen if pid in state.records)
    lost = sorted(set(state.records) - set(seen))
    for pid in lost:
        next_state.retired_pids.append(pid)
        next_state.retired_maps.append(old_map_by_pid[pid])
    next_state._set_species_from_mol(product)
    agreement, detail = compare_states(next_state.species, expected)
    return ExecutionResult(
        ok=not duplicates,
        route=route,
        resulting_state_mapped=list(next_state.species),
        resulting_state=next_state.stripped_smiles(),
        identity_map={pid: next_state.map_by_pid()[pid] for pid in sorted(next_state.records)},
        preserved_ids=preserved,
        new_ids=sorted(new_ids),
        lost_ids=lost,
        duplicate_ids=sorted(set(duplicates)),
        smirks_state_agreement=agreement,
        agreement_detail=detail,
        match_mode=match_mode,
        distinct_outcomes=distinct_outcomes,
        hydrogen_policy=state.hydrogen_policy,
        error="duplicate persistent atom ids in product" if duplicates else None,
        failure_category="identity" if duplicates else None,
        next_state=next_state,
    )


def _failure(route: Literal["smirks", "moves"], state: MappedState, category: str, message: str) -> ExecutionResult:
    return ExecutionResult(
        ok=False,
        route=route,
        hydrogen_policy=state.hydrogen_policy,
        error=message,
        failure_category=category,
    )


# ---------------------------------------------------------------------------
# Route A: SMIRKS execution
# ---------------------------------------------------------------------------


def _split_reaction(reaction_smirks: str) -> Tuple[str, str]:
    core, _meta = split_cxsmiles_metadata(str(reaction_smirks or ""))
    if ">>" in core:
        lhs, rhs = core.split(">>", 1)
    else:
        parts = core.split(">")
        if len(parts) != 3:
            raise MappedStateError("reaction_smirks has no '>>' separator")
        lhs, rhs = parts[0], parts[2]
    lhs, rhs = lhs.strip(), rhs.strip()
    if not lhs or not rhs:
        raise MappedStateError("reaction_smirks has an empty side")
    return lhs, rhs


def _side_props(side: str) -> Dict[int, Dict[str, Any]]:
    """SMILES-semantics atom properties per map number for one reaction side."""
    mol = Chem.MolFromSmiles(side, _parser_params(sanitize=False))
    if mol is None:
        return {}
    mol.UpdatePropertyCache(strict=False)
    props: Dict[int, Dict[str, Any]] = {}
    for atom in mol.GetAtoms():
        num = atom.GetAtomMapNum()
        if num <= 0:
            continue
        h_nbrs = [n.GetAtomMapNum() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
        props[num] = {
            "atomic_num": atom.GetAtomicNum(),
            "charge": atom.GetFormalCharge(),
            "bracket_h": atom.GetNumExplicitHs(),
            "total_h": atom.GetNumExplicitHs() + len(h_nbrs),
            "h_neighbor_maps": [m for m in h_nbrs if m > 0],
            "radicals": atom.GetNumRadicalElectrons(),
        }
    return props


def _aromatized_side(side: str) -> Optional[str]:
    """Rewrite one reaction side with RDKit aromaticity (maps and H kept)."""
    mol = Chem.MolFromSmiles(side, _parser_params(sanitize=False))
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol)


def _build_reaction(
    lhs: str,
    rhs: str,
    *,
    constrain: bool,
    heavy_only_constraint: bool,
) -> Tuple[rdChemReactions.ChemicalReaction, set[int]]:
    """Build a single-template reaction with *ghost* products for deleted atoms.

    Every left-side atom that has no right-side counterpart (including unmapped
    left-side atoms, which RDKit deletes) is added to the product template as
    an isolated ``[*:n]`` ghost. RDKit then keeps it in the product with its
    ``react_atom_idx``, so the executor knows exactly which state atoms the
    template consumed; ghosts are removed afterwards and reported as lost.
    """
    lhs_t = Chem.MolFromSmarts(lhs)
    rhs_t = Chem.MolFromSmarts(rhs)
    if lhs_t is None or rhs_t is None:
        raise MappedStateError("RDKit could not parse reaction_smirks")
    rhs_maps = {a.GetAtomMapNum() for a in rhs_t.GetAtoms() if a.GetAtomMapNum() > 0}
    all_maps = rhs_maps | {a.GetAtomMapNum() for a in lhs_t.GetAtoms()}
    ghost_num = max(all_maps, default=0) + 1000
    lhs_rw = Chem.RWMol(lhs_t)
    ghosts: set[int] = set()
    for atom in lhs_rw.GetAtoms():
        num = atom.GetAtomMapNum()
        if num <= 0:
            atom.SetAtomMapNum(ghost_num)
            ghosts.add(ghost_num)
            ghost_num += 1
            continue
        if num not in rhs_maps:
            ghosts.add(num)
        if constrain and not (heavy_only_constraint and atom.GetAtomicNum() == 1):
            atom.ExpandQuery(rdqueries.HasIntPropWithValueQueryAtom("molAtomMapNumber", num))
    rhs_rw = Chem.RWMol(rhs_t)
    for num in sorted(ghosts):
        rhs_rw.InsertMol(Chem.MolFromSmarts(f"[*:{num}]"))
    rxn = rdChemReactions.ChemicalReaction()
    rxn.AddReactantTemplate(lhs_rw.GetMol())
    rxn.AddProductTemplate(rhs_rw.GetMol())
    # ReactionFromSmarts sets per-atom inversion flags at parse time; a
    # hand-assembled reaction needs this call for [C@]>>[C@@] to invert.
    rdChemReactions.UpdateProductsStereochemistry(rxn)
    rxn.Initialize()
    return rxn, ghosts


def _prepare_match_mol(state: MappedState, *, kekule: bool, add_hs: bool) -> Tuple[Chem.Mol, List[Optional[PersistentAtomId]]]:
    mol = _parse_raw(".".join(state.species))
    if kekule:
        _sanitize_kekule_as_written(mol)
    else:
        Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    _freeze_hydrogens(mol)
    if add_hs:
        mol = Chem.AddHs(mol)
    pids = [state.map_to_pid.get(a.GetAtomMapNum()) if a.GetAtomMapNum() > 0 else None for a in mol.GetAtoms()]
    return mol, pids


def _remove_atoms(mol: Chem.Mol, indices: Iterable[int]) -> Tuple[Chem.Mol, Dict[int, int]]:
    """Remove atoms; return the new mol and an old->new index map."""
    drop = set(indices)
    if not drop:
        return mol, {i: i for i in range(mol.GetNumAtoms())}
    rw = Chem.RWMol(mol)
    keep = [i for i in range(mol.GetNumAtoms()) if i not in drop]
    for idx in sorted(drop, reverse=True):
        rw.RemoveAtom(idx)
    return rw.GetMol(), {old: new for new, old in enumerate(keep)}


def _remap(mapping: Dict[int, Any], index_map: Dict[int, int]) -> Dict[int, Any]:
    return {index_map[k]: v for k, v in mapping.items() if k in index_map}


def execute_smirks(
    mapped_current_state: MappedStateInput,
    reaction_smirks: str,
    *,
    expected_resulting_state: Optional[Sequence[str]] = None,
    hydrogen_policy: HydrogenPolicy = "auto",
    step_label: str = "step",
) -> ExecutionResult:
    """Route A: run ``reaction_smirks`` on the mapped state with RDKit.

    Charges, radicals and H counts of atoms mapped on both sides change by the
    *difference* between the SMILES-semantics right and left sides, applied to
    the state atom (so ``[C:2]>>[C:2]`` leaves a CH3 a CH3, and the benchmark's
    full-state SMIRKS reproduce exactly). Atoms new on the right take the
    right side's values.
    """
    state = _coerce_state(mapped_current_state, hydrogen_policy)
    try:
        lhs, rhs = _split_reaction(reaction_smirks)
    except MappedStateError as exc:
        return _failure("smirks", state, "smirks_syntax", str(exc))
    heavy_policy = state.hydrogen_policy == "heavy_atom"
    lhs_props = _side_props(lhs)
    rhs_props = _side_props(rhs)
    template_has_h = any(p["atomic_num"] == 1 for p in lhs_props.values()) or "[H" in lhs or "#1" in lhs
    add_hs = heavy_policy and template_has_h

    blocker = rdBase.BlockLogs()
    try:
        last_error = "left side of reaction_smirks does not match current_state"
        last_category = "lhs_state_mismatch"
        # Attempt order: map-constrained before unconstrained; for each, the
        # template as written on the Kekulé state, then on the aromatic state,
        # then an aromatized copy of the left side on the aromatic state (the
        # LLM's Kekulé structure need not match ours).
        aromatic_lhs = _aromatized_side(lhs)
        attempts: List[Tuple[bool, str, bool]] = []
        for constrain in (True, False):
            attempts.append((constrain, lhs, True))
            attempts.append((constrain, lhs, False))
            if aromatic_lhs and aromatic_lhs != lhs:
                attempts.append((constrain, aromatic_lhs, False))
        reactions: Dict[Tuple[bool, str], Tuple[rdChemReactions.ChemicalReaction, set[int]]] = {}
        for constrain, template, kekule in attempts:
            if (constrain, template) not in reactions:
                try:
                    reactions[(constrain, template)] = _build_reaction(
                        template, rhs, constrain=constrain, heavy_only_constraint=heavy_policy
                    )
                except Exception as exc:  # RDKit raises ValueError on bad SMARTS
                    if template is lhs:
                        return _failure("smirks", state, "smirks_syntax", f"reaction parse failed: {exc}")
                    continue
            rxn, ghosts = reactions[(constrain, template)]
            try:
                mol, mol_pids = _prepare_match_mol(state, kekule=kekule, add_hs=add_hs)
            except Exception as exc:
                last_error, last_category = f"state preparation failed: {exc}", "aromaticity"
                continue
            try:
                outcomes = rxn.RunReactants((mol,), _MAX_REACTION_PRODUCTS)
            except Exception as exc:
                last_error, last_category = f"RunReactants failed: {exc}", "smirks_syntax"
                continue
            if not outcomes:
                continue
            built: List[Tuple[Chem.Mol, Dict[int, Optional[PersistentAtomId]], Dict[int, int]]] = []
            distinct: set[str] = set()
            for outcome in outcomes:
                candidate = _materialize_product(
                    outcome[0],
                    mol,
                    mol_pids,
                    lhs_props,
                    rhs_props,
                    ghosts,
                    heavy_policy=heavy_policy,
                )
                if isinstance(candidate, str):
                    last_error, last_category = candidate, "valence"
                    continue
                key = Chem.MolToSmiles(candidate[0])
                if key in distinct:
                    continue
                distinct.add(key)
                built.append(candidate)
            if not built:
                continue
            product, source_pid, hints = built[0]
            return _finalize(
                route="smirks",
                state=state,
                product=product,
                source_pid=source_pid,
                new_provenance={},
                template_map_hint=hints,
                expected=expected_resulting_state,
                match_mode="map_constrained" if constrain else "unconstrained",
                distinct_outcomes=len(built),
                step_label=step_label,
            )
        return _failure("smirks", state, last_category, last_error)
    finally:
        del blocker


_STEREO_BOND = {
    Chem.BondStereo.STEREOE,
    Chem.BondStereo.STEREOZ,
    Chem.BondStereo.STEREOCIS,
    Chem.BondStereo.STEREOTRANS,
}


def _materialize_product(
    raw_product: Chem.Mol,
    reactant: Chem.Mol,
    reactant_pids: List[Optional[PersistentAtomId]],
    lhs_props: Dict[int, Dict[str, Any]],
    rhs_props: Dict[int, Dict[str, Any]],
    ghosts: set[int],
    *,
    heavy_policy: bool,
) -> Union[str, Tuple[Chem.Mol, Dict[int, Optional[PersistentAtomId]], Dict[int, int]]]:
    """Turn one RunReactants product into a sanitized mol plus identity maps.

    Returns an error string on failure.
    """
    rw = Chem.RWMol(raw_product)
    touched: set[int] = set()
    source_pid: Dict[int, Optional[PersistentAtomId]] = {}
    source_ridx: Dict[int, int] = {}
    hints: Dict[int, int] = {}
    target_h: Dict[int, int] = {}
    ghost_idx: List[int] = []
    for atom in rw.GetAtoms():
        idx = atom.GetIdx()
        num = atom.GetIntProp("old_mapno") if atom.HasProp("old_mapno") else 0
        atom.SetAtomMapNum(0)
        ridx = atom.GetIntProp("react_atom_idx") if atom.HasProp("react_atom_idx") else None
        if ridx is not None:
            touched.add(ridx)
            source_pid[idx] = reactant_pids[ridx]
            source_ridx[idx] = ridx
        else:
            source_pid[idx] = None
            if num > 0:
                hints[idx] = num
        if num in ghosts:
            ghost_idx.append(idx)
            continue
        spec = rhs_props.get(num) if num > 0 else None
        if spec is None:
            continue
        before = lhs_props.get(num)
        if ridx is not None and before is not None:
            src = reactant.GetAtomWithIdx(ridx)
            atom.SetFormalCharge(src.GetFormalCharge() + int(spec["charge"]) - int(before["charge"]))
            atom.SetNumRadicalElectrons(
                max(0, src.GetNumRadicalElectrons() + int(spec["radicals"]) - int(before["radicals"]))
            )
            h0 = src.GetTotalNumHs(includeNeighbors=True)
            target_h[idx] = max(0, h0 + int(spec["total_h"]) - int(before["total_h"]))
        else:
            atom.SetFormalCharge(int(spec["charge"]))
            atom.SetNumRadicalElectrons(int(spec["radicals"]))
            target_h[idx] = int(spec["total_h"])
    product: Chem.Mol = rw.GetMol()
    product, index_map = _remove_atoms(product, ghost_idx)
    source_pid, source_ridx = _remap(source_pid, index_map), _remap(source_ridx, index_map)
    hints, target_h = _remap(hints, index_map), _remap(target_h, index_map)

    # RDKit drops reactant fragments untouched by the template: carry them over.
    frags = Chem.GetMolFrags(reactant, asMols=False, sanitizeFrags=False)
    carry = [frag for frag in frags if not (set(frag) & touched)]
    if carry:
        combo = Chem.RWMol(product)
        placed: Dict[int, int] = {}
        for frag in carry:
            for ridx in frag:
                new_atom = Chem.Atom(reactant.GetAtomWithIdx(ridx))
                new_atom.SetAtomMapNum(0)
                new_idx = combo.AddAtom(new_atom)
                placed[ridx] = new_idx
                source_pid[new_idx] = reactant_pids[ridx]
                source_ridx[new_idx] = ridx
        for bond in reactant.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a in placed and b in placed:
                combo.AddBond(placed[a], placed[b], bond.GetBondType())
        product = combo.GetMol()

    # Hydrogens: explicit policy keeps H atoms and sets bracket counts to the
    # remainder; heavy-atom policy folds bonded H back into counts.
    rw = Chem.RWMol(product)
    if heavy_policy:
        bonded_h = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 1 and a.GetDegree() > 0]
        for idx in bonded_h:
            heavy = rw.GetAtomWithIdx(idx).GetNeighbors()[0]
            if heavy.GetIdx() not in target_h:
                heavy.SetNumExplicitHs(heavy.GetNumExplicitHs() + 1)
        for idx, count in target_h.items():
            if rw.GetAtomWithIdx(idx).GetAtomicNum() != 1:
                rw.GetAtomWithIdx(idx).SetNumExplicitHs(count)
        product, index_map = _remove_atoms(rw.GetMol(), bonded_h)
        source_pid, source_ridx = _remap(source_pid, index_map), _remap(source_ridx, index_map)
        hints = _remap(hints, index_map)
    else:
        for idx, count in target_h.items():
            atom = rw.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() == 1:
                continue
            carried = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 1)
            atom.SetNumExplicitHs(max(0, count - carried))
        product = rw.GetMol()

    # Restore E/Z on untouched double bonds that RDKit's copy did not keep.
    rev = {r: p for p, r in source_ridx.items()}
    rw = Chem.RWMol(product)
    for bond in rw.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE or bond.GetStereo() != Chem.BondStereo.STEREONONE:
            continue
        pa, pb = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ra, rb = source_ridx.get(pa), source_ridx.get(pb)
        if ra is None or rb is None:
            continue
        rbond = reactant.GetBondBetweenAtoms(ra, rb)
        if rbond is None or rbond.GetBondType() != Chem.BondType.DOUBLE or rbond.GetStereo() not in _STEREO_BOND:
            continue
        s_atoms = list(rbond.GetStereoAtoms())
        if len(s_atoms) != 2 or s_atoms[0] not in rev or s_atoms[1] not in rev:
            continue
        sa, sb = rev[s_atoms[0]], rev[s_atoms[1]]
        if rbond.GetBeginAtomIdx() != ra:
            sa, sb = sb, sa
        if rw.GetBondBetweenAtoms(pa, sa) is None or rw.GetBondBetweenAtoms(pb, sb) is None:
            continue
        bond.SetStereoAtoms(sa, sb)
        bond.SetStereo(rbond.GetStereo())
    product = rw.GetMol()
    try:
        for atom in product.GetAtoms():
            atom.SetNoImplicit(True)
        _sanitize_product(product)
    except Exception as exc:
        return f"product sanitization failed: {exc}"
    return product, source_pid, hints


# ---------------------------------------------------------------------------
# Route B: move execution
# ---------------------------------------------------------------------------


def _coerce_moves(moves: Union[str, Iterable[Any]]) -> List[MechanismMove]:
    if isinstance(moves, str):
        _core, parsed, details = extract_mechanism_moves(moves)
        if not parsed:
            raise MappedStateError(str(details.get("error") or "no mech moves"))
        return parsed
    items = list(moves or [])
    if items and all(isinstance(m, MechanismMove) for m in items):
        return items  # type: ignore[return-value]
    parsed = normalize_electron_pushes(items)
    if not parsed:
        raise MappedStateError("no parseable electron pushes")
    return parsed


def _align_kekule_to_lhs(mol: Chem.Mol, reaction_smirks: str) -> Chem.Mol:
    """Re-kekulize aromatic bonds to the SMIRKS left side's Kekulé structure.

    ``pi:`` moves name specific double bonds; if our stored Kekulé structure
    differs from the one the SMIRKS author wrote, adopt theirs when it is a
    valid Kekulé structure of the same state. Otherwise keep ours.
    """
    try:
        lhs, _rhs = _split_reaction(reaction_smirks)
    except MappedStateError:
        return mol
    ref = Chem.MolFromSmiles(lhs, _parser_params(sanitize=False))
    if ref is None:
        return mol
    orders: Dict[Tuple[int, int], int] = {}
    for bond in ref.GetBonds():
        a, b = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()
        if a > 0 and b > 0 and bond.GetBondType() in (Chem.BondType.SINGLE, Chem.BondType.DOUBLE):
            orders[(min(a, b), max(a, b))] = int(bond.GetBondTypeAsDouble())
    perceived = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(perceived)
    except Exception:
        return mol
    trial = Chem.RWMol(mol)
    changed = False
    for bond in perceived.GetBonds():
        if not bond.GetIsAromatic():
            continue
        a, b = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()
        want = orders.get((min(a, b), max(a, b)))
        if want is None:
            continue
        target = trial.GetBondBetweenAtoms(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        if int(target.GetBondTypeAsDouble()) != want:
            target.SetBondType(_BOND_TYPES[want])
            changed = True
    if not changed:
        return mol
    candidate = trial.GetMol()
    try:
        probe = Chem.Mol(candidate)
        _sanitize_kekule_as_written(probe)
        if _stripped_signature_of_mol(probe) != _stripped_signature_of_mol(mol):
            return mol
    except Exception:
        return mol
    return candidate


def execute_moves(
    mapped_current_state: MappedStateInput,
    moves: Union[str, Iterable[Any]],
    *,
    expected_resulting_state: Optional[Sequence[str]] = None,
    hydrogen_policy: HydrogenPolicy = "auto",
    reaction_smirks: Optional[str] = None,
    step_label: str = "step",
) -> ExecutionResult:
    """Route B: apply two-electron arrow-pushing moves as ``RWMol`` edits.

    ``moves`` may be a SMIRKS string carrying a ``|mech:v1;...|`` block, a list
    of ``electron_pushes`` dicts, or parsed :class:`MechanismMove` objects.
    ``reaction_smirks`` is used only under ``heavy_atom`` policy to resolve
    hydrogen map numbers to the heavy atom carrying them.
    """
    state = _coerce_state(mapped_current_state, hydrogen_policy)
    try:
        parsed = _coerce_moves(moves)
    except MappedStateError as exc:
        return _failure("moves", state, "smirks_syntax", str(exc))
    if isinstance(moves, str) and reaction_smirks is None:
        reaction_smirks = moves
    for move in parsed:
        if int(move.electrons or 2) != 2:
            return _failure("moves", state, "unsupported_move", "only two-electron moves are supported")

    blocker = rdBase.BlockLogs()
    try:
        mol = _parse_raw(".".join(state.species))
        try:
            _sanitize_kekule_as_written(mol)
        except Exception as exc:
            return _failure("moves", state, "aromaticity", f"kekulization failed: {exc}")
        _freeze_hydrogens(mol)
        if reaction_smirks:
            mol = _align_kekule_to_lhs(mol, reaction_smirks)
        rw = Chem.RWMol(mol)
        idx_by_map = {a.GetAtomMapNum(): a.GetIdx() for a in rw.GetAtoms() if a.GetAtomMapNum() > 0}
        source_pid: Dict[int, Optional[PersistentAtomId]] = {
            a.GetIdx(): state.map_to_pid.get(a.GetAtomMapNum()) for a in rw.GetAtoms()
        }
        temp_h: set[int] = set()

        referenced: set[int] = set()
        for move in parsed:
            for ref in (move.source_atom, move.bond_start, move.bond_end, move.target_atom):
                if ref is not None:
                    referenced.add(int(ref))
        missing = sorted(r for r in referenced if r not in idx_by_map)
        if missing:
            if state.hydrogen_policy != "heavy_atom" or not reaction_smirks:
                return _failure("moves", state, "lhs_state_mismatch", f"moves reference unknown maps {missing}")
            try:
                lhs, _rhs = _split_reaction(reaction_smirks)
            except MappedStateError as exc:
                return _failure("moves", state, "smirks_syntax", str(exc))
            lhs_props = _side_props(lhs)
            carrier: Dict[int, int] = {}
            for heavy_map, spec in lhs_props.items():
                for h_map in spec["h_neighbor_maps"]:
                    carrier[h_map] = heavy_map
            for ref in missing:
                heavy_map = carrier.get(ref)
                if heavy_map is None or heavy_map not in idx_by_map:
                    return _failure("moves", state, "h_handling", f"cannot resolve hydrogen map {ref}")
                heavy = rw.GetAtomWithIdx(idx_by_map[heavy_map])
                if heavy.GetNumExplicitHs() < 1:
                    return _failure("moves", state, "h_handling", f"atom {heavy_map} has no H for map {ref}")
                heavy.SetNumExplicitHs(heavy.GetNumExplicitHs() - 1)
                h = Chem.Atom(1)
                h.SetAtomMapNum(ref)
                h.SetNoImplicit(True)
                new_idx = rw.AddAtom(h)
                rw.AddBond(heavy.GetIdx(), new_idx, Chem.BondType.SINGLE)
                idx_by_map[ref] = new_idx
                source_pid[new_idx] = None
                temp_h.add(new_idx)

        order_delta: Dict[Tuple[int, int], int] = {}
        charge_delta: Dict[int, int] = {}

        def bump(a: int, b: int, d: int) -> None:
            key = (min(a, b), max(a, b))
            order_delta[key] = order_delta.get(key, 0) + d

        for move in parsed:
            if move.kind == "lone_pair":
                src, tgt = int(move.source_atom), int(move.target_atom)  # type: ignore[arg-type]
                bump(src, tgt, +1)
                charge_delta[src] = charge_delta.get(src, 0) + 1
                charge_delta[tgt] = charge_delta.get(tgt, 0) - 1
                continue
            x, y = int(move.bond_start), int(move.bond_end)  # type: ignore[arg-type]
            tgt = int(move.target_atom)
            bump(x, y, -1)
            if tgt == y:
                charge_delta[x] = charge_delta.get(x, 0) + 1
                charge_delta[y] = charge_delta.get(y, 0) - 1
            elif tgt == x:
                charge_delta[y] = charge_delta.get(y, 0) + 1
                charge_delta[x] = charge_delta.get(x, 0) - 1
            else:
                bump(y, tgt, +1)
                charge_delta[x] = charge_delta.get(x, 0) + 1
                charge_delta[tgt] = charge_delta.get(tgt, 0) - 1

        for (a, b), delta in sorted(order_delta.items()):
            if delta == 0:
                continue
            ia, ib = idx_by_map[a], idx_by_map[b]
            bond = rw.GetBondBetweenAtoms(ia, ib)
            current = int(round(bond.GetBondTypeAsDouble())) if bond is not None else 0
            new_order = current + delta
            if new_order < 0 or new_order > 3:
                return _failure("moves", state, "valence", f"bond {a}-{b} order {current}{delta:+d} invalid")
            if bond is not None:
                if new_order == 0:
                    rw.RemoveBond(ia, ib)
                else:
                    bond.SetBondType(_BOND_TYPES[new_order])
                    bond.SetStereo(Chem.BondStereo.STEREONONE)
            elif new_order > 0:
                rw.AddBond(ia, ib, _BOND_TYPES[new_order])
        for num, delta in charge_delta.items():
            atom = rw.GetAtomWithIdx(idx_by_map[num])
            atom.SetFormalCharge(atom.GetFormalCharge() + delta)

        product: Chem.Mol = rw.GetMol()
        new_provenance: Dict[int, str] = {}
        if state.hydrogen_policy == "heavy_atom":
            # Fold every bonded H back into counts. Temporary H that ended up
            # free are released protons (fresh id); pre-existing free protons
            # that became bonded disappear and their ids are reported lost.
            fold = [
                a.GetIdx() for a in product.GetAtoms() if a.GetAtomicNum() == 1 and a.GetDegree() > 0
            ]
            if fold:
                rw2 = Chem.RWMol(product)
                for i in fold:
                    heavy = rw2.GetAtomWithIdx(i).GetNeighbors()[0]
                    heavy.SetNumExplicitHs(heavy.GetNumExplicitHs() + 1)
                fold_set = set(fold)
                keep = [a.GetIdx() for a in rw2.GetAtoms() if a.GetIdx() not in fold_set]
                new_index = {old: new for new, old in enumerate(keep)}
                for i in sorted(fold, reverse=True):
                    rw2.RemoveAtom(i)
                product = rw2.GetMol()
                source_pid = {new_index[o]: p for o, p in source_pid.items() if o in new_index}
                temp_h = {new_index[o] for o in temp_h if o in new_index}
            for idx in temp_h:
                new_provenance[idx] = "proton:released"
        try:
            for atom in product.GetAtoms():
                atom.SetNoImplicit(True)
            _sanitize_product(product)
        except Exception as exc:
            return _failure("moves", state, "valence", f"product sanitization failed: {exc}")
        hints = {i: 0 for i in range(product.GetNumAtoms())}
        for idx in range(product.GetNumAtoms()):
            if source_pid.get(idx) is None:
                hints[idx] = product.GetAtomWithIdx(idx).GetAtomMapNum()
        for atom in product.GetAtoms():
            atom.SetAtomMapNum(0)
        return _finalize(
            route="moves",
            state=state,
            product=product,
            source_pid=source_pid,
            new_provenance=new_provenance,
            template_map_hint=hints,
            expected=expected_resulting_state,
            match_mode="map_constrained",
            distinct_outcomes=1,
            step_label=step_label,
        )
    finally:
        del blocker


# ---------------------------------------------------------------------------
# Loop helpers (coordinator hook, loop_state_mapping="mapped")
# ---------------------------------------------------------------------------


def execute_candidate(
    mapped_current_state: MappedState,
    *,
    reaction_smirks: Optional[str],
    electron_pushes: Any = None,
    expected_resulting_state: Optional[Sequence[str]] = None,
    step_label: str = "step",
) -> ExecutionResult:
    """Route A first, Route B as fallback (§16.8 ``mapped_state.execute``)."""
    result: Optional[ExecutionResult] = None
    if reaction_smirks:
        result = execute_smirks(
            mapped_current_state,
            reaction_smirks,
            expected_resulting_state=expected_resulting_state,
            step_label=step_label,
        )
        if result.ok:
            return result
    move_source: Any = electron_pushes if electron_pushes else reaction_smirks
    if move_source:
        fallback = execute_moves(
            mapped_current_state,
            move_source,
            expected_resulting_state=expected_resulting_state,
            reaction_smirks=reaction_smirks,
            step_label=step_label,
        )
        if fallback.ok or result is None:
            return fallback
    if result is not None:
        return result
    return _failure("smirks", mapped_current_state, "smirks_syntax", "no reaction_smirks or electron_pushes")


LOOP_HYDROGEN_POLICY: Literal["heavy_atom"] = "heavy_atom"


def sync_mapped_loop_state(
    snapshot: Optional[Dict[str, Any]],
    *,
    current_state: Sequence[str],
    history: Optional[Dict[int, Dict[str, Any]]] = None,
    step_index: Optional[int] = None,
    seed_species: Optional[Sequence[str]] = None,
) -> Tuple[MappedState, str]:
    """Return a mapped state that matches ``current_state`` (map-stripped).

    Resolution order: the live snapshot; the history snapshot for
    ``step_index`` (backtracking restores exact ids); the mapped ``seed_species``
    (benchmark ingress maps); otherwise fresh ids. The loop runs under the
    ``heavy_atom`` hydrogen policy. The allocator only ever advances, so ids
    issued on a discarded branch are never reissued. Returns the state and how
    it was obtained (``live``/``history``/``seed``/``resync``).
    """
    target = state_signature(current_state)
    allocator = PersistentAtomIdAllocator()
    if snapshot:
        live = MappedState.restore(snapshot)
        allocator = live.allocator
        if live.signature() == target:
            return live, "live"
    if history and step_index is not None and step_index in history:
        restored = MappedState.restore(history[step_index], allocator=allocator)
        if restored.signature() == target:
            return restored, "history"
    if seed_species:
        try:
            seeded = MappedState.from_smiles(
                list(seed_species), allocator=allocator, hydrogen_policy=LOOP_HYDROGEN_POLICY, provenance="seed"
            )
            if seeded.signature() == target:
                return seeded, "seed"
        except Exception:
            pass
    fresh = MappedState.from_smiles(
        [s for s in current_state if str(s or "").strip()],
        allocator=allocator,
        hydrogen_policy=LOOP_HYDROGEN_POLICY,
        provenance="resync",
    )
    return fresh, "resync"


def advance_mapped_loop_state(
    snapshot: Optional[Dict[str, Any]],
    *,
    previous_state: Sequence[str],
    stated_resulting_state: Sequence[str],
    reaction_smirks: Optional[str],
    electron_pushes: Any = None,
    history: Optional[Dict[int, Dict[str, Any]]] = None,
    step_index: Optional[int] = None,
    seed_species: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Execute an accepted candidate on the mapped loop state (record only).

    Returns ``(next_snapshot, record)``. ``next_snapshot`` always matches the
    LLM's ``stated_resulting_state`` (the loop's source of truth is unchanged):
    it is the executor's derived state when the two agree, else a resync of
    the stated state with fresh ids. ``record`` is JSON-serializable for the
    step trace / validation JSON. If ``history`` is given, the pre-step
    snapshot is stored under ``step_index`` for backtracking.
    """
    mapped, origin = sync_mapped_loop_state(
        snapshot,
        current_state=previous_state,
        history=history,
        step_index=step_index,
        seed_species=seed_species,
    )
    if history is not None and step_index is not None:
        history[step_index] = mapped.snapshot()
    result = execute_candidate(
        mapped,
        reaction_smirks=reaction_smirks,
        electron_pushes=electron_pushes,
        expected_resulting_state=list(stated_resulting_state),
        step_label=f"step{(step_index or 0) + 1}",
    )
    agreed = bool(result.ok and result.smirks_state_agreement)
    if agreed and result.next_state is not None:
        next_state = result.next_state
        resynced = False
    else:
        next_state = MappedState.from_smiles(
            [s for s in stated_resulting_state if str(s or "").strip()],
            allocator=mapped.allocator,
            hydrogen_policy=LOOP_HYDROGEN_POLICY,
            provenance="resync",
        )
        resynced = True
    record = {
        "smirks_state_agreement": result.smirks_state_agreement if result.ok else None,
        "executed": result.ok,
        "route": result.route,
        "match_mode": result.match_mode,
        "failure_category": result.failure_category,
        "error": result.error,
        "input_state_origin": origin,
        "identity_resynced": resynced,
        "preserved_id_count": len(result.preserved_ids),
        "new_ids": list(result.new_ids),
        "lost_ids": list(result.lost_ids),
        "duplicate_ids": list(result.duplicate_ids),
        "distinct_outcomes": result.distinct_outcomes,
        "derived_resulting_state": list(result.resulting_state),
        "missing_from_derived": list(result.agreement_detail.get("missing_from_derived") or []),
        "extra_in_derived": list(result.agreement_detail.get("extra_in_derived") or []),
        "mapped_resulting_state": list(next_state.species),
        "blocking": False,
    }
    return next_state.snapshot(), record


__all__ = [
    "AtomRecord",
    "ExecutionResult",
    "HydrogenPolicy",
    "MappedState",
    "MappedStateError",
    "PersistentAtomId",
    "PersistentAtomIdAllocator",
    "advance_mapped_loop_state",
    "compare_states",
    "execute_candidate",
    "execute_moves",
    "execute_smirks",
    "mapped_signature",
    "species_signature",
    "state_signature",
    "sync_mapped_loop_state",
]
