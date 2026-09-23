"""Mapping recall against benchmark atom maps (PRD v2 §8.5, §11, §19 Phase 0).

Every benchmark record is fully atom-mapped with explicit hydrogens, but the
runtime strips maps at ingress and the LLM mapping tool answers in its own
index space (zero-based atom indices into the canonical, map-free SMILES the
prompt shows). This module puts both on one footing:

* :class:`SideFrame` is one side of a reaction (reactants or products) in the
  runtime's index space: component ``c`` is ``remove_mapping_and_canonicalize``
  of the input component (exactly what ``db.py`` / ``tool_executor.py`` pass to
  the prompt), and atom ``i`` is the ``i``-th atom of ``MolFromSmiles`` of that
  canonical string. Heavy atoms occupy ``0..n-1``; with the explicit-H policy
  on, hydrogens are appended in ``AddHs`` order (grouped by parent atom).
* :class:`AtomMapping` is a set of ``(reactant AtomRef) -> (product AtomRef)``
  pairs over a pair of frames.
* :func:`compare_mappings` scores a predicted mapping against a reference with
  symmetry awareness (``Chem.CanonicalRankAtoms(breakTies=False)``).

Converters build an :class:`AtomMapping` from mapped SMILES lists (benchmark
``starting_materials``/``products`` or step ``current_state``/``resulting_state``),
from a mapped ``reaction_smirks``, and from the LLM ``mapped_atoms`` payload
(full or ``compact_mapped_atoms``).

This module only measures. It never feeds back into the runtime.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except ImportError:  # pragma: no cover
    Chem = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Canonical representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class AtomRef:
    """One atom in a :class:`SideFrame`: (component index, atom index)."""

    component: int
    atom: int


@dataclass
class SideFrame:
    """One reaction side in the runtime's canonical index space."""

    smiles: List[str]
    include_hydrogens: bool
    elements: Dict[AtomRef, str]
    symmetry_class: Dict[AtomRef, int]
    heavy_counts: List[int]

    def is_hydrogen(self, ref: AtomRef) -> bool:
        return self.elements.get(ref) == "H"

    def same_space(self, other: "SideFrame") -> bool:
        return self.smiles == other.smiles and self.include_hydrogens == other.include_hydrogens

    def resolve_component(self, smiles: str, molecule_index: Optional[int] = None) -> Optional[int]:
        """Return the frame component matching ``smiles`` (canonical comparison)."""

        canonical = _canonical_unmapped(smiles)
        if canonical is None:
            return None
        if molecule_index is not None and 0 <= molecule_index < len(self.smiles):
            if self.smiles[molecule_index] == canonical:
                return molecule_index
        for idx, item in enumerate(self.smiles):
            if item == canonical:
                return idx
        return None


@dataclass
class AtomMapping:
    """Reactant-atom -> product-atom pairs over two frames."""

    reactants: SideFrame
    products: SideFrame
    pairs: Dict[AtomRef, AtomRef]
    source: str
    unresolved_entries: int = 0

    @property
    def include_hydrogens(self) -> bool:
        return self.reactants.include_hydrogens


@dataclass
class MappingAgreement:
    """Result of :func:`compare_mappings`."""

    agreement: float
    exact_match: bool
    agreed: int
    disagreed: int
    unmapped: int
    extra: int
    reference_atoms: int
    include_hydrogens: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "agreement": round(self.agreement, 6),
            "exact_match": self.exact_match,
            "agreed": self.agreed,
            "disagreed": self.disagreed,
            "unmapped": self.unmapped,
            "extra": self.extra,
            "reference_atoms": self.reference_atoms,
            "include_hydrogens": self.include_hydrogens,
            **self.details,
        }


class MappingMetricError(ValueError):
    """Raised when a mapping cannot be placed in a comparable frame."""


# ---------------------------------------------------------------------------
# Frame construction
# ---------------------------------------------------------------------------


def _require_rdkit() -> None:
    if Chem is None:  # pragma: no cover
        raise MappingMetricError("RDKit is required for mapping metrics")


def _parse(smiles: str):
    _require_rdkit()
    mol = Chem.MolFromSmiles(str(smiles or "").strip())
    if mol is None:
        raise MappingMetricError(f"unparseable SMILES: {smiles!r}")
    return mol


def _canonical_with_order(mol) -> Tuple[str, List[int]]:
    """Map-free canonical SMILES of ``mol`` and the output order of its atoms.

    ``order[i]`` is the index in ``mol`` of the ``i``-th atom of the returned
    string. This reproduces ``smiles_utils.remove_mapping_and_canonicalize``.
    """

    work = Chem.Mol(mol)
    for atom in work.GetAtoms():
        atom.SetAtomMapNum(0)
    canonical = Chem.MolToSmiles(work)
    order = list(work.GetPropsAsDict(True, True).get("_smilesAtomOutputOrder") or [])
    if len(order) != work.GetNumAtoms():  # pragma: no cover - defensive
        raise MappingMetricError("RDKit did not report an atom output order")
    return canonical, [int(i) for i in order]


def _canonical_unmapped(smiles: str) -> Optional[str]:
    try:
        return _canonical_with_order(_parse(smiles))[0]
    except MappingMetricError:
        return None


def _split_components(smiles_list: Sequence[str]) -> List[str]:
    out: List[str] = []
    for item in smiles_list:
        for part in str(item or "").split("."):
            part = part.strip()
            if part:
                out.append(part)
    return out


def _build_side(
    smiles_list: Sequence[str],
    *,
    include_hydrogens: bool,
    mapped: bool,
) -> Tuple[SideFrame, Dict[int, AtomRef]]:
    """Build a frame; when ``mapped`` also return map number -> AtomRef."""

    _require_rdkit()
    canon_smiles: List[str] = []
    elements: Dict[AtomRef, str] = {}
    heavy_counts: List[int] = []
    map_lookup: Dict[int, AtomRef] = {}
    frame_mols = []

    for comp_idx, raw in enumerate(smiles_list):
        mol = _parse(raw)  # default parse drops explicit (incl. mapped) H
        canonical, order = _canonical_with_order(mol)
        canon_mol = _parse(canonical)
        position_of = {orig: pos for pos, orig in enumerate(order)}
        canon_smiles.append(canonical)
        heavy_counts.append(canon_mol.GetNumAtoms())

        if mapped:
            for atom in mol.GetAtoms():
                num = atom.GetAtomMapNum()
                if num:
                    map_lookup[num] = AtomRef(comp_idx, position_of[atom.GetIdx()])

        frame_mol = Chem.AddHs(canon_mol) if include_hydrogens else canon_mol
        for atom in frame_mol.GetAtoms():
            elements[AtomRef(comp_idx, atom.GetIdx())] = atom.GetSymbol()
        frame_mols.append(frame_mol)

        if mapped and include_hydrogens:
            _assign_mapped_hydrogens(raw, comp_idx, mol, position_of, frame_mol, map_lookup)

    symmetry = _symmetry_classes(frame_mols)
    frame = SideFrame(
        smiles=canon_smiles,
        include_hydrogens=include_hydrogens,
        elements=elements,
        symmetry_class=symmetry,
        heavy_counts=heavy_counts,
    )
    return frame, map_lookup


def _assign_mapped_hydrogens(raw: str, comp_idx: int, mol, position_of: Dict[int, int], frame_mol, map_lookup) -> None:
    """Place mapped explicit H atoms onto the ``AddHs`` slots of their parent.

    ``mol`` is the default (H-removing) parse; heavy atoms are matched between
    the two parses through their map numbers, so the parent of every removed
    H resolves to its canonical position.
    """

    params = Chem.SmilesParserParams()
    params.removeHs = False
    explicit = Chem.MolFromSmiles(str(raw).strip(), params)
    if explicit is None:
        return
    kept_by_num = {a.GetAtomMapNum(): a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum()}

    slots: Dict[int, List[int]] = {}
    for atom in frame_mol.GetAtoms():
        if atom.GetAtomicNum() == 1 and atom.GetDegree() == 1 and atom.GetIdx() >= mol.GetNumAtoms():
            parent = atom.GetNeighbors()[0].GetIdx()
            slots.setdefault(parent, []).append(atom.GetIdx())

    by_parent: Dict[int, List[int]] = {}
    for atom in explicit.GetAtoms():
        num = atom.GetAtomMapNum()
        if atom.GetAtomicNum() != 1 or not num or num in kept_by_num:
            continue
        neighbors = atom.GetNeighbors()
        if len(neighbors) != 1:
            continue
        parent_default = kept_by_num.get(neighbors[0].GetAtomMapNum())
        if parent_default is None:
            continue
        by_parent.setdefault(position_of[parent_default], []).append(num)

    for parent_canon, nums in by_parent.items():
        for num, slot in zip(sorted(nums), slots.get(parent_canon, [])):
            map_lookup[num] = AtomRef(comp_idx, slot)


def _symmetry_classes(frame_mols: Sequence[Any]) -> Dict[AtomRef, int]:
    """Symmetry classes across a whole side (identical components collapse)."""

    if not frame_mols:
        return {}
    combined = frame_mols[0]
    for mol in frame_mols[1:]:
        combined = Chem.CombineMols(combined, mol)
    ranks = list(Chem.CanonicalRankAtoms(combined, breakTies=False))
    out: Dict[AtomRef, int] = {}
    offset = 0
    for comp_idx, mol in enumerate(frame_mols):
        for atom in mol.GetAtoms():
            out[AtomRef(comp_idx, atom.GetIdx())] = int(ranks[offset + atom.GetIdx()])
        offset += mol.GetNumAtoms()
    return out


def build_side_frame(smiles_list: Sequence[str], *, include_hydrogens: bool = False) -> SideFrame:
    """Frame for an (optionally mapped) species list, in runtime index space."""

    frame, _ = _build_side(_split_components(smiles_list), include_hydrogens=include_hydrogens, mapped=False)
    return frame


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------


def mapping_from_mapped_smiles(
    reactants: Sequence[str],
    products: Sequence[str],
    *,
    include_hydrogens: bool = False,
    source: str = "benchmark",
) -> AtomMapping:
    """Benchmark mapped species lists -> :class:`AtomMapping` (map numbers -> pairs)."""

    r_frame, r_lookup = _build_side(_split_components(reactants), include_hydrogens=include_hydrogens, mapped=True)
    p_frame, p_lookup = _build_side(_split_components(products), include_hydrogens=include_hydrogens, mapped=True)
    pairs: Dict[AtomRef, AtomRef] = {}
    for num, r_ref in r_lookup.items():
        p_ref = p_lookup.get(num)
        if p_ref is None:
            continue
        if not include_hydrogens and (r_frame.is_hydrogen(r_ref) or p_frame.is_hydrogen(p_ref)):
            continue
        pairs[r_ref] = p_ref
    return AtomMapping(reactants=r_frame, products=p_frame, pairs=pairs, source=source)


def split_reaction_smirks(smirks: str) -> Tuple[List[str], List[str]]:
    """Split a (CX)SMIRKS into reactant and product component lists."""

    text = str(smirks or "").strip()
    if " |" in text:
        text = text.split(" |", 1)[0].strip()
    parts = text.split(">")
    if len(parts) != 3:
        raise MappingMetricError(f"not a reaction SMIRKS: {smirks!r}")
    left, _agents, right = parts
    return _split_components([left]), _split_components([right])


def mapping_from_reaction_smirks(
    smirks: str,
    *,
    include_hydrogens: bool = False,
    source: str = "reaction_smirks",
) -> AtomMapping:
    """Mapped ``reaction_smirks`` -> :class:`AtomMapping`."""

    reactants, products = split_reaction_smirks(smirks)
    return mapping_from_mapped_smiles(reactants, products, include_hydrogens=include_hydrogens, source=source)


def _translate_index(frame: SideFrame, component: int, written_smiles: str, index: int) -> Optional[int]:
    """Translate an index into ``written_smiles`` to the frame's canonical index."""

    if index < 0:
        return None
    if written_smiles.strip() == frame.smiles[component]:
        return index if index < frame.heavy_counts[component] else None
    try:
        mol = _parse(written_smiles)
        canonical, order = _canonical_with_order(mol)
    except MappingMetricError:
        return None
    if canonical != frame.smiles[component] or index >= len(order):
        return None
    return order.index(index)


def _parse_product_atom(value: Any) -> Tuple[Optional[str], Optional[int]]:
    text = str(value or "").strip()
    if "#" not in text:
        return None, None
    smiles, _, idx = text.rpartition("#")
    try:
        return smiles, int(idx)
    except ValueError:
        return None, None


def mapping_from_llm_mapped_atoms(
    mapped_atoms: Optional[Iterable[Mapping[str, Any]]],
    *,
    reactants: SideFrame,
    products: SideFrame,
    source: str = "llm",
) -> AtomMapping:
    """LLM ``mapped_atoms`` (or ``compact_mapped_atoms``) -> :class:`AtomMapping`.

    Entries follow ``ATOM_MAPPING_TOOL``: ``product_atom = "<smiles>#<idx>"``
    and ``source = {molecule_index, smiles, atom_index}``; the compact form
    stored by ``attempt_atom_mapping_for_step`` uses ``source_smiles`` and
    ``source_atom_index``. Indices are zero-based into the canonical SMILES the
    prompt showed, so they are resolved against frames built from the same
    lists. Entries that cannot be placed are counted in ``unresolved_entries``.
    """

    pairs: Dict[AtomRef, AtomRef] = {}
    unresolved = 0
    for entry in mapped_atoms or []:
        if not isinstance(entry, Mapping):
            unresolved += 1
            continue
        p_smiles, p_idx = _parse_product_atom(entry.get("product_atom"))
        src = entry.get("source") if isinstance(entry.get("source"), Mapping) else {}
        s_smiles = src.get("smiles", entry.get("source_smiles"))
        s_idx = src.get("atom_index", entry.get("source_atom_index"))
        s_mol_idx = src.get("molecule_index")
        try:
            s_idx = int(s_idx)
            s_mol_idx = int(s_mol_idx) if s_mol_idx is not None else None
        except (TypeError, ValueError):
            unresolved += 1
            continue
        if p_smiles is None or p_idx is None or not s_smiles:
            unresolved += 1
            continue
        r_comp = reactants.resolve_component(str(s_smiles), s_mol_idx)
        p_comp = products.resolve_component(p_smiles)
        if r_comp is None or p_comp is None:
            unresolved += 1
            continue
        r_atom = _translate_index(reactants, r_comp, str(s_smiles), s_idx)
        p_atom = _translate_index(products, p_comp, p_smiles, p_idx)
        if r_atom is None or p_atom is None:
            unresolved += 1
            continue
        pairs[AtomRef(r_comp, r_atom)] = AtomRef(p_comp, p_atom)
    return AtomMapping(
        reactants=reactants,
        products=products,
        pairs=pairs,
        source=source,
        unresolved_entries=unresolved,
    )


def mapping_to_llm_mapped_atoms(mapping: AtomMapping) -> List[Dict[str, Any]]:
    """Render heavy-atom pairs in the ``ATOM_MAPPING_TOOL`` format (for tests and tooling)."""

    out: List[Dict[str, Any]] = []
    for r_ref, p_ref in sorted(mapping.pairs.items()):
        if mapping.reactants.is_hydrogen(r_ref) or mapping.products.is_hydrogen(p_ref):
            continue
        out.append(
            {
                "product_atom": f"{mapping.products.smiles[p_ref.component]}#{p_ref.atom}",
                "source": {
                    "molecule_index": r_ref.component,
                    "smiles": mapping.reactants.smiles[r_ref.component],
                    "atom_index": r_ref.atom,
                },
            }
        )
    return out


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_mappings(
    predicted: AtomMapping,
    reference: AtomMapping,
    *,
    reactant_symmetry: bool = True,
) -> MappingAgreement:
    """Symmetry-aware per-atom agreement of ``predicted`` against ``reference``.

    A reference reactant atom agrees when the prediction maps it to the same
    product atom or to one in the same product symmetry class. With
    ``reactant_symmetry`` (default) the comparison is also invariant to
    relabelling symmetry-equivalent *reactant* atoms: within each reactant
    class the multisets of product classes are intersected. For reactant
    classes of size one this is exactly the per-atom rule.

    ``agreement = agreed / reference_atoms``. ``extra`` counts predicted pairs
    on reactant atoms the reference does not map (spectators, H under the
    heavy-atom policy are excluded before counting). ``exact_match`` is true
    when every reference atom agrees.
    """

    if not predicted.reactants.same_space(reference.reactants) or not predicted.products.same_space(
        reference.products
    ):
        raise MappingMetricError("predicted and reference mappings use different frames")

    include_h = reference.include_hydrogens
    r_frame, p_frame = reference.reactants, reference.products

    def _keep(ref: AtomRef) -> bool:
        return include_h or not r_frame.is_hydrogen(ref)

    ref_pairs = {r: p for r, p in reference.pairs.items() if _keep(r)}
    pred_pairs = {r: p for r, p in predicted.pairs.items() if _keep(r)}

    groups: Dict[Any, List[AtomRef]] = {}
    for r_ref in ref_pairs:
        key = r_frame.symmetry_class[r_ref] if reactant_symmetry else r_ref
        groups.setdefault(key, []).append(r_ref)

    agreed = disagreed = unmapped = 0
    for members in groups.values():
        ref_classes: Dict[int, int] = {}
        pred_classes: Dict[int, int] = {}
        for r_ref in members:
            cls = p_frame.symmetry_class[ref_pairs[r_ref]]
            ref_classes[cls] = ref_classes.get(cls, 0) + 1
            if r_ref in pred_pairs:
                p_cls = p_frame.symmetry_class.get(pred_pairs[r_ref])
                if p_cls is not None:
                    pred_classes[p_cls] = pred_classes.get(p_cls, 0) + 1
            else:
                unmapped += 1
        hits = sum(min(count, pred_classes.get(cls, 0)) for cls, count in ref_classes.items())
        agreed += hits
        disagreed += sum(pred_classes.values()) - hits

    extra = sum(1 for r_ref in pred_pairs if r_ref not in ref_pairs)
    total = len(ref_pairs)
    agreement = (agreed / total) if total else 0.0
    return MappingAgreement(
        agreement=agreement,
        exact_match=bool(total) and agreed == total,
        agreed=agreed,
        disagreed=disagreed,
        unmapped=unmapped,
        extra=extra,
        reference_atoms=total,
        include_hydrogens=include_h,
        details={"unresolved_entries": predicted.unresolved_entries, "source": predicted.source},
    )


# ---------------------------------------------------------------------------
# Benchmark extraction and run-level evaluation (used by scoring.py)
# ---------------------------------------------------------------------------


def _has_maps(smiles_list: Sequence[str]) -> bool:
    return any(":" in str(item) for item in smiles_list)


def _signature_counts(smiles_list: Sequence[str]) -> Counter:
    out: Counter = Counter()
    for item in _split_components(smiles_list):
        canonical = _canonical_unmapped(item)
        if canonical:
            out[canonical] += 1
    return out


def _contains(outer: Counter, inner: Counter) -> bool:
    return all(outer.get(key, 0) >= count for key, count in inner.items())


def benchmark_steps(expected: Mapping[str, Any] | None) -> List[Dict[str, Any]]:
    """Mapped benchmark steps ``[{step_index, current_state, resulting_state}]``."""

    if not isinstance(expected, Mapping):
        return []
    verified = expected.get("verified_mechanism")
    steps = verified.get("steps") if isinstance(verified, Mapping) else None
    out: List[Dict[str, Any]] = []
    for step in steps or []:
        if not isinstance(step, Mapping):
            continue
        current = [str(x) for x in step.get("current_state") or []]
        resulting = [str(x) for x in step.get("resulting_state") or []]
        if (not current or not resulting) and step.get("reaction_smirks"):
            try:
                current, resulting = split_reaction_smirks(str(step["reaction_smirks"]))
            except MappingMetricError:
                continue
        if current and resulting and _has_maps(current) and _has_maps(resulting):
            out.append({"step_index": int(step.get("step_index") or 0), "current_state": current, "resulting_state": resulting})
    return out


def benchmark_global_species(
    expected: Mapping[str, Any] | None,
    snapshot_input: Mapping[str, Any] | None = None,
) -> Optional[Tuple[List[str], List[str]]]:
    """Mapped (starting_materials, products) for the whole reaction, if available."""

    starting: List[str] = []
    products: List[str] = []
    boundary = (snapshot_input or {}).get("input_boundary") if isinstance(snapshot_input, Mapping) else None
    if isinstance(boundary, Mapping):
        starting = [str(x) for x in boundary.get("original_starting_materials") or []]
        products = [str(x) for x in boundary.get("original_products") or []]
    if isinstance(expected, Mapping):
        if not _has_maps(starting):
            starting = [str(x) for x in expected.get("starting_materials") or []]
        if not _has_maps(starting):
            steps = benchmark_steps(expected)
            starting = list(steps[0]["current_state"]) if steps else []
        if not _has_maps(products):
            products = [str(x) for x in expected.get("products") or []]
    if _has_maps(starting) and _has_maps(products):
        return starting, products
    return None


def _llm_mapped_atoms(output: Mapping[str, Any]) -> Optional[List[Any]]:
    """Full ``mapped_atoms`` from a stored atom_mapping / step_atom_mapping output."""

    for container in (output.get("llm_response"), (output.get("raw") or {}).get("llm_response") if isinstance(output.get("raw"), Mapping) else None):
        if isinstance(container, Mapping) and isinstance(container.get("mapped_atoms"), list):
            return list(container["mapped_atoms"])
    compact = output.get("compact_mapped_atoms")
    if isinstance(compact, list) and compact:
        return list(compact)
    return None


def evaluate_output_against_reference(
    output: Mapping[str, Any],
    reference_reactants: Sequence[str],
    reference_products: Sequence[str],
    *,
    include_hydrogens: bool = False,
) -> Dict[str, Any]:
    """Score one stored mapping output against mapped reference species."""

    mapped_atoms = _llm_mapped_atoms(output)
    if not mapped_atoms:
        return {"status": "no_predicted_mapping"}
    reference = mapping_from_mapped_smiles(reference_reactants, reference_products, include_hydrogens=include_hydrogens)
    predicted = mapping_from_llm_mapped_atoms(
        mapped_atoms, reactants=reference.reactants, products=reference.products
    )
    result = compare_mappings(predicted, reference).as_dict()
    result["status"] = "scored"
    return result


def _parse_output(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        import json

        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _latest_output_by_attempt(step_outputs: Sequence[Mapping[str, Any]], step_name: str) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Tuple[int, Dict[str, Any]]] = {}
    for row in step_outputs:
        if str(row.get("step_name") or "") != step_name:
            continue
        attempt = int(row.get("attempt") or 0)
        retry = int(row.get("retry_index") or 0)
        if attempt in out and out[attempt][0] > retry:
            continue
        out[attempt] = (retry, _parse_output(row.get("output")))
    return {k: v for k, (_r, v) in out.items()}


def _match_benchmark_step(
    step_index: int,
    current_state: Sequence[str],
    resulting_state: Sequence[str],
    bench: Sequence[Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    """Benchmark step whose species (as a multiset) are contained in the predicted step.

    Spectators the prediction carries but the benchmark omits are allowed; a
    step with the same index is preferred when several benchmark steps fit.
    """

    cur = _signature_counts(current_state)
    res = _signature_counts(resulting_state)
    candidates = []
    for step in bench:
        if _contains(cur, _signature_counts(step["current_state"])) and _contains(
            res, _signature_counts(step["resulting_state"])
        ):
            candidates.append(step)
    if not candidates:
        return None
    for step in candidates:
        if int(step.get("step_index") or 0) == step_index:
            return step
    return candidates[0]


def compute_run_mapping_agreement(
    snapshot: Mapping[str, Any],
    expected: Mapping[str, Any] | None,
    *,
    accepted_steps: Optional[Sequence[Mapping[str, Any]]] = None,
    include_hydrogens: bool = False,
) -> Dict[str, Any]:
    """Global and per-step mapping agreement for one run against its benchmark.

    Returns ``{"available": False, ...}`` when the case has no benchmark
    mapping. Per-step entries carry a ``status``: ``scored``,
    ``no_predicted_mapping``, ``benchmark_step_unmatched`` or ``error``.
    """

    snapshot_input: Mapping[str, Any] = {}
    for key in ("input_payload", "input"):
        if isinstance(snapshot.get(key), Mapping):
            snapshot_input = snapshot[key]
            break
    bench_steps = benchmark_steps(expected)
    bench_global = benchmark_global_species(expected, snapshot_input)
    if not bench_steps and bench_global is None:
        return {"available": False, "reason": "no_benchmark_mapping"}

    step_outputs = list(snapshot.get("step_outputs") or [])
    result: Dict[str, Any] = {"available": True, "include_hydrogens": include_hydrogens}

    # Global mapping (pre-loop atom_mapping module).
    global_outputs = _latest_output_by_attempt(step_outputs, "atom_mapping")
    if bench_global is None:
        result["global"] = {"status": "no_benchmark_mapping"}
    elif not global_outputs:
        result["global"] = {"status": "no_predicted_mapping"}
    else:
        output = global_outputs[max(global_outputs)]
        try:
            result["global"] = evaluate_output_against_reference(
                output, bench_global[0], bench_global[1], include_hydrogens=include_hydrogens
            )
        except Exception as exc:  # pragma: no cover - defensive
            result["global"] = {"status": "error", "error": str(exc)}

    # Per accepted step (step_atom_mapping module).
    step_map_outputs = _latest_output_by_attempt(step_outputs, "step_atom_mapping")
    steps_out: List[Dict[str, Any]] = []
    for step in accepted_steps or []:
        step_index = int(step.get("step_index") or 0)
        entry: Dict[str, Any] = {"step_index": step_index}
        output = step_map_outputs.get(step_index)
        if not output:
            entry["status"] = "no_predicted_mapping"
            steps_out.append(entry)
            continue
        current = [str(x) for x in output.get("current_state") or step.get("current_state") or []]
        resulting = [str(x) for x in output.get("resulting_state") or step.get("resulting_state") or []]
        try:
            bench = _match_benchmark_step(step_index, current, resulting, bench_steps)
            if bench is None:
                entry["status"] = "benchmark_step_unmatched"
            else:
                # Score in the prediction's own frame (it may carry spectators the
                # benchmark step omits); the reference is restricted to benchmark atoms.
                entry.update(
                    _evaluate_step(output, current, resulting, bench, include_hydrogens=include_hydrogens)
                )
                entry["benchmark_step_index"] = int(bench.get("step_index") or 0)
        except Exception as exc:
            entry = {"step_index": step_index, "status": "error", "error": str(exc)}
        steps_out.append(entry)
    result["steps"] = steps_out
    scored = [item["agreement"] for item in steps_out if item.get("status") == "scored"]
    result["step_mean"] = round(sum(scored) / len(scored), 6) if scored else None
    result["steps_scored"] = len(scored)
    return result


def _evaluate_step(
    output: Mapping[str, Any],
    current: Sequence[str],
    resulting: Sequence[str],
    bench: Mapping[str, Any],
    *,
    include_hydrogens: bool,
) -> Dict[str, Any]:
    mapped_atoms = _llm_mapped_atoms(output)
    if not mapped_atoms:
        return {"status": "no_predicted_mapping"}
    bench_map = mapping_from_mapped_smiles(
        bench["current_state"], bench["resulting_state"], include_hydrogens=include_hydrogens
    )
    r_frame = build_side_frame(current, include_hydrogens=include_hydrogens)
    p_frame = build_side_frame(resulting, include_hydrogens=include_hydrogens)
    reference = _reframe(bench_map, r_frame, p_frame)
    predicted = mapping_from_llm_mapped_atoms(mapped_atoms, reactants=r_frame, products=p_frame)
    out = compare_mappings(predicted, reference).as_dict()
    out["status"] = "scored"
    return out


def _reframe(mapping: AtomMapping, reactants: SideFrame, products: SideFrame) -> AtomMapping:
    """Move ``mapping`` into larger frames containing its components (same canonical SMILES)."""

    def _component_map(src: SideFrame, dst: SideFrame) -> Dict[int, int]:
        used: set[int] = set()
        out: Dict[int, int] = {}
        for idx, smi in enumerate(src.smiles):
            for jdx, other in enumerate(dst.smiles):
                if jdx not in used and other == smi:
                    out[idx] = jdx
                    used.add(jdx)
                    break
            else:
                raise MappingMetricError(f"component {smi!r} not present in target frame")
        return out

    r_map = _component_map(mapping.reactants, reactants)
    p_map = _component_map(mapping.products, products)
    pairs = {
        AtomRef(r_map[r.component], r.atom): AtomRef(p_map[p.component], p.atom)
        for r, p in mapping.pairs.items()
    }
    return AtomMapping(reactants=reactants, products=products, pairs=pairs, source=mapping.source)


def summarize_mapping_agreement(values: Iterable[Optional[float]]) -> Optional[float]:
    kept = [float(v) for v in values if isinstance(v, (int, float))]
    return round(sum(kept) / len(kept), 6) if kept else None


__all__ = [
    "AtomMapping",
    "AtomRef",
    "MappingAgreement",
    "MappingMetricError",
    "SideFrame",
    "benchmark_global_species",
    "benchmark_steps",
    "build_side_frame",
    "compare_mappings",
    "compute_run_mapping_agreement",
    "evaluate_output_against_reference",
    "mapping_from_llm_mapped_atoms",
    "mapping_from_mapped_smiles",
    "mapping_from_reaction_smirks",
    "mapping_to_llm_mapped_atoms",
    "split_reaction_smirks",
    "summarize_mapping_agreement",
]
