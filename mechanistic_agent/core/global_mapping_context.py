"""Deterministic consumers of the global ``atom_mapping`` pre-loop output.

``attempt_atom_mapping`` returns its LLM answer under ``llm_response``:
``mapped_atoms`` pairs of the form ``{"product_atom": "<product_smiles>#<idx>",
"source": {"molecule_index": int, "smiles": str, "atom_index": int}}``,
plus ``confidence`` and ``unmapped_atoms``. Indices are zero-based atom
indices in the (map-free, RDKit-canonical) SMILES the mapping LLM was shown.

This module turns that output into the two things the mechanism-proposal
call can consume:

* :func:`summarize_global_mapping` -- the compact ``atom_mapping_summary``
  (confidence + unmapped atoms) placed in proposal guidance.
* :func:`render_global_mapping` -- atom-mapped SMILES for the starting
  materials and products, in the format ``propose_intermediates`` expects for
  ``mapped_starting_materials`` / ``mapped_products`` (one mapped SMILES per
  species, same order as the input lists).

Both functions are pure: no LLM calls, no store access. Pairs that cannot be
resolved unambiguously (unknown species, out-of-range index, element
mismatch, duplicate use of an atom) are dropped, never repaired.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from mechanistic_agent.smiles_utils import remove_mapping_and_canonicalize

try:  # pragma: no cover - rdkit is a hard dependency in practice
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None  # type: ignore[assignment]


_UNMAPPED_ATOMS_LIMIT = 12

# template_guidance key the coordinator uses to hand rendered mapped SMILES
# to IntermediateAgent; popped before guidance is JSON-rendered into the prompt.
MAPPED_SPECIES_CONTEXT_KEY = "_mapped_species_context"


_CONFIDENCE_WORDS = {
    "high": 0.9,
    "very_high": 0.9,
    "strong": 0.9,
    "medium": 0.6,
    "moderate": 0.6,
    "low": 0.3,
    "weak": 0.3,
}


def _coerce_confidence(value: Any) -> Optional[float]:
    """Numeric confidence in [0, 1], or ``None`` when absent/unparseable.

    Mirrors ``tools._normalise_mapping_confidence`` for present values but
    keeps "missing" distinguishable from 0.0.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
    elif isinstance(value, str):
        text = value.strip().lower()
        if text in _CONFIDENCE_WORDS:
            numeric = _CONFIDENCE_WORDS[text]
        else:
            try:
                numeric = float(text)
            except ValueError:
                return None
    else:
        return None
    return min(1.0, max(0.0, numeric))


def summarize_global_mapping(output: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return ``{"confidence", "unmapped_atoms"}`` for a global mapping output.

    Confidence is read from ``llm_response.confidence`` (where the tool puts
    it, and where ``MappingAgent`` clamps it after a failed atom-map check),
    falling back to a top-level ``confidence`` for legacy outputs. Returns
    ``{}`` when there is no mapping output (module disabled or not run).
    """
    if not isinstance(output, dict) or not output:
        return {}
    llm_response = output.get("llm_response")
    llm_response = llm_response if isinstance(llm_response, dict) else {}
    confidence = _coerce_confidence(llm_response.get("confidence"))
    if confidence is None:
        confidence = _coerce_confidence(output.get("confidence"))
    unmapped = llm_response.get("unmapped_atoms")
    unmapped_atoms = (
        [str(item) for item in unmapped if item is not None][:_UNMAPPED_ATOMS_LIMIT]
        if isinstance(unmapped, list)
        else []
    )
    return {"confidence": confidence, "unmapped_atoms": unmapped_atoms}


def _canonical(smiles: Any) -> str:
    text = str(smiles or "").strip()
    if not text:
        return ""
    return str(remove_mapping_and_canonicalize(text) or text).strip()


def _parse_product_ref(ref: Any) -> Optional[Tuple[str, int]]:
    text = str(ref or "").strip()
    if "#" not in text:
        return None
    # SMILES may contain '#' (triple bond); the index is after the last one.
    smiles, _, index_text = text.rpartition("#")
    try:
        index = int(index_text.strip())
    except ValueError:
        return None
    if not smiles.strip() or index < 0:
        return None
    return smiles.strip(), index


def _resolve_source(
    source: Any,
    starting_canonical: Sequence[str],
) -> Optional[Tuple[int, int]]:
    if not isinstance(source, dict):
        return None
    atom_index = source.get("atom_index")
    if isinstance(atom_index, bool) or not isinstance(atom_index, int) or atom_index < 0:
        return None
    mol_index = source.get("molecule_index")
    source_smiles = _canonical(source.get("smiles")) if source.get("smiles") else ""
    if isinstance(mol_index, int) and not isinstance(mol_index, bool) and 0 <= mol_index < len(starting_canonical):
        if not source_smiles or source_smiles == starting_canonical[mol_index]:
            return mol_index, atom_index
    if source_smiles and source_smiles in starting_canonical:
        return starting_canonical.index(source_smiles), atom_index
    return None


def render_global_mapping(
    starting_materials: Sequence[str],
    products: Sequence[str],
    mapped_atoms: Any,
) -> Dict[str, Any]:
    """Render LLM ``mapped_atoms`` pairs as atom-mapped SMILES.

    ``starting_materials`` / ``products`` should be the map-free canonical
    SMILES the mapping LLM saw (they are re-canonicalised here defensively).
    Each accepted pair gets a fresh map number (1, 2, ...) applied to both the
    source atom and the product atom; atoms without an accepted pair stay
    unmapped.

    Returns ``{"mapped_starting_materials", "mapped_products", "pairs_used",
    "pairs_dropped"}``. Both lists are empty when no pair is usable, so callers
    can pass them straight through (``propose_intermediates`` omits the mapped
    section when all lists are empty).
    """
    empty: Dict[str, Any] = {
        "mapped_starting_materials": [],
        "mapped_products": [],
        "pairs_used": 0,
        "pairs_dropped": 0,
    }
    if Chem is None or not isinstance(mapped_atoms, list) or not mapped_atoms:
        if isinstance(mapped_atoms, list):
            empty["pairs_dropped"] = len(mapped_atoms)
        return empty

    starting_canonical = [_canonical(item) for item in starting_materials]
    products_canonical = [_canonical(item) for item in products]
    start_mols = [Chem.MolFromSmiles(item) if item else None for item in starting_canonical]
    product_mols = [Chem.MolFromSmiles(item) if item else None for item in products_canonical]

    used_sources: set[Tuple[int, int]] = set()
    used_products: set[Tuple[int, int]] = set()
    next_map = 1
    dropped = 0
    for entry in mapped_atoms:
        if not isinstance(entry, dict):
            dropped += 1
            continue
        parsed = _parse_product_ref(entry.get("product_atom"))
        source = _resolve_source(entry.get("source"), starting_canonical)
        if parsed is None or source is None:
            dropped += 1
            continue
        product_smiles, product_atom_idx = parsed
        product_key = _canonical(product_smiles)
        if product_key not in products_canonical:
            dropped += 1
            continue
        product_mol_idx = products_canonical.index(product_key)
        source_mol_idx, source_atom_idx = source
        product_mol = product_mols[product_mol_idx]
        source_mol = start_mols[source_mol_idx]
        if product_mol is None or source_mol is None:
            dropped += 1
            continue
        if product_atom_idx >= product_mol.GetNumAtoms() or source_atom_idx >= source_mol.GetNumAtoms():
            dropped += 1
            continue
        product_atom = product_mol.GetAtomWithIdx(product_atom_idx)
        source_atom = source_mol.GetAtomWithIdx(source_atom_idx)
        if product_atom.GetAtomicNum() != source_atom.GetAtomicNum():
            dropped += 1
            continue
        product_key_pair = (product_mol_idx, product_atom_idx)
        source_key_pair = (source_mol_idx, source_atom_idx)
        if product_key_pair in used_products or source_key_pair in used_sources:
            dropped += 1
            continue
        used_products.add(product_key_pair)
        used_sources.add(source_key_pair)
        product_atom.SetAtomMapNum(next_map)
        source_atom.SetAtomMapNum(next_map)
        next_map += 1

    pairs_used = next_map - 1
    if pairs_used == 0:
        empty["pairs_dropped"] = dropped
        return empty

    def _render(mols: List[Any], fallback: List[str]) -> List[str]:
        rendered: List[str] = []
        for mol, text in zip(mols, fallback):
            rendered.append(Chem.MolToSmiles(mol) if mol is not None else text)
        return rendered

    return {
        "mapped_starting_materials": _render(start_mols, starting_canonical),
        "mapped_products": _render(product_mols, products_canonical),
        "pairs_used": pairs_used,
        "pairs_dropped": dropped,
    }


def mapped_current_state_for(
    current_state: Sequence[str],
    starting_materials: Sequence[str],
    mapped_starting_materials: Sequence[str],
) -> List[str]:
    """Return mapped current state only when it is still the starting state.

    The global mapping describes starting materials -> products; it says
    nothing about intermediates. It is reused for the current state only when
    the current state is (as a multiset) exactly the starting materials, i.e.
    before any step has been accepted. Loop-state mapping after step 0 is out
    of scope here.
    """
    if not mapped_starting_materials:
        return []
    current = sorted(_canonical(item) for item in current_state)
    starting = sorted(_canonical(item) for item in starting_materials)
    if current != starting:
        return []
    return list(mapped_starting_materials)
