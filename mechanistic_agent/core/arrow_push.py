"""Deterministic OrbChain-lite arrow push annotations for mechanism steps."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .mechanism_moves import normalize_electron_pushes, split_cxsmiles_metadata

try:  # pragma: no cover - optional dependency
    from rdkit import Chem
except Exception:  # pragma: no cover - defensive
    Chem = None  # type: ignore[assignment]


_TEMPLATE_PRIOR: Dict[str, float] = {
    "sn2_substitution": 0.9,
    "nucleophilic_addition": 0.82,
    "elimination_like": 0.74,
    "proton_transfer": 0.7,
}


def _clean_smiles_token(token: str) -> str:
    return str(token or "").strip()


def _strip_cxsmiles_metadata(expression: str) -> str:
    return split_cxsmiles_metadata(str(expression or ""))[0]


def _parse_molecules(side: str) -> List[Any]:
    if Chem is None:
        return []
    molecules: List[Any] = []
    for token in side.split("."):
        smiles = _clean_smiles_token(token)
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            molecules.append(mol)
    return molecules


def _collect_side_state(side: str) -> Tuple[Dict[Tuple[int, int], float], Dict[int, Dict[str, Any]], set[int]]:
    bonds: Dict[Tuple[int, int], float] = {}
    atom_state: Dict[int, Dict[str, Any]] = {}
    carbonyl_carbons: set[int] = set()
    for mol in _parse_molecules(side):
        for atom in mol.GetAtoms():
            amap = int(atom.GetAtomMapNum() or 0)
            if amap <= 0:
                continue
            atom_state[amap] = {
                "symbol": atom.GetSymbol(),
                "formal_charge": int(atom.GetFormalCharge()),
                "total_valence": int(atom.GetTotalValence()),
                "degree": int(atom.GetDegree()),
            }
            if atom.GetSymbol() == "C":
                for bond in atom.GetBonds():
                    nbr = bond.GetOtherAtom(atom)
                    if nbr.GetSymbol() == "O" and float(bond.GetBondTypeAsDouble()) >= 1.9:
                        carbonyl_carbons.add(amap)
                        break

        for bond in mol.GetBonds():
            begin = int(bond.GetBeginAtom().GetAtomMapNum() or 0)
            end = int(bond.GetEndAtom().GetAtomMapNum() or 0)
            if begin <= 0 or end <= 0:
                continue
            pair = tuple(sorted((begin, end)))
            bonds[pair] = float(bond.GetBondTypeAsDouble())

    return bonds, atom_state, carbonyl_carbons


def _extract_bond_changes(reaction_smirks: str) -> Dict[str, Any]:
    core = _strip_cxsmiles_metadata(reaction_smirks)
    if not core or ">>" not in core:
        return {
            "core": core,
            "changes": [],
            "atom_state": {},
            "carbonyl_carbons": set(),
        }

    left, right = core.split(">>", 1)
    left_bonds, left_atoms, left_carbonyl = _collect_side_state(left)
    right_bonds, right_atoms, _ = _collect_side_state(right)

    # Prefer reactant-side atom properties (source/sink sanity is interpreted
    # from the state before the electron push).
    atom_state: Dict[int, Dict[str, Any]] = dict(right_atoms)
    atom_state.update(left_atoms)

    changes: List[Dict[str, Any]] = []
    for pair in sorted(set(left_bonds) | set(right_bonds)):
        left_order = float(left_bonds.get(pair, 0.0))
        right_order = float(right_bonds.get(pair, 0.0))
        if abs(left_order - right_order) < 1e-6:
            continue
        kind = "order_changed"
        if left_order == 0.0 and right_order > 0.0:
            kind = "bond_formed"
        elif left_order > 0.0 and right_order == 0.0:
            kind = "bond_broken"
        changes.append(
            {
                "map_i": int(pair[0]),
                "map_j": int(pair[1]),
                "left_order": left_order,
                "right_order": right_order,
                "kind": kind,
                "delta": right_order - left_order,
            }
        )

    return {
        "core": core,
        "changes": changes,
        "atom_state": atom_state,
        "carbonyl_carbons": left_carbonyl,
    }


def _normalize_push_ref(value: Any) -> str:
    text = str(value or "").strip()
    if text.isdigit():
        return str(int(text))
    return text


def _electron_pushes(payload: Any) -> List[Dict[str, Any]]:
    pushes: List[Dict[str, Any]] = []
    for move in normalize_electron_pushes(payload):
        source_ref = ""
        if move.kind == "lone_pair" and move.source_atom is not None:
            source_ref = _normalize_push_ref(move.source_atom)
        elif move.through_atom is not None:
            source_ref = _normalize_push_ref(move.through_atom)
        sink_ref = _normalize_push_ref(move.target_atom)
        if source_ref and sink_ref:
            pushes.append({"start_atom": source_ref, "end_atom": sink_ref, "electrons": move.electrons})
    return pushes


def _map_ref(map_id: int) -> str:
    return str(int(map_id))


def _candidate_templates(
    *,
    src_map: int,
    snk_map: int,
    kind: str,
    atom_state: Dict[int, Dict[str, Any]],
    carbonyl_carbons: set[int],
) -> List[str]:
    templates: List[str] = []
    src_symbol = str((atom_state.get(src_map) or {}).get("symbol") or "")
    snk_symbol = str((atom_state.get(snk_map) or {}).get("symbol") or "")
    halogens = {"F", "Cl", "Br", "I"}

    if src_symbol == "H" or snk_symbol == "H":
        templates.append("proton_transfer")

    if kind == "bond_broken":
        templates.append("elimination_like")

    if snk_map in carbonyl_carbons:
        templates.append("nucleophilic_addition")

    if kind in {"bond_formed", "order_changed"} and (
        snk_symbol in {"C", "S", "P"}
        or (src_symbol == "C" and snk_symbol in halogens)
        or (snk_symbol == "C" and src_symbol in halogens)
    ):
        templates.append("sn2_substitution")

    if not templates:
        templates.append("nucleophilic_addition")

    deduped: List[str] = []
    for template in templates:
        if template not in deduped:
            deduped.append(template)
    return deduped


def _push_alignment_score(src_ref: str, snk_ref: str, pushes: List[Dict[str, Any]]) -> Tuple[float, int]:
    if not pushes:
        return 0.0, 2
    best = 0.0
    electrons = 2
    for push in pushes:
        start = str(push.get("start_atom") or "")
        end = str(push.get("end_atom") or "")
        if start == src_ref and end == snk_ref:
            return 1.0, int(push.get("electrons") or 2)
        if start == snk_ref and end == src_ref:
            if best < 0.45:
                best = 0.45
                electrons = int(push.get("electrons") or 2)
            continue
        if start == src_ref or end == snk_ref:
            if best < 0.2:
                best = 0.2
                electrons = int(push.get("electrons") or 2)
    return best, electrons


def _sanity_check(
    source_ref: str,
    sink_ref: str,
    atom_state: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    penalties: List[str] = []
    hard_rejected = False

    if not source_ref or not sink_ref or source_ref == sink_ref:
        hard_rejected = True
        penalties.append("invalid_or_self_referential_source_sink")
        return {"hard_rejected": hard_rejected, "penalties": penalties, "penalty": 1.0}

    if source_ref.isdigit():
        source_atom = atom_state.get(int(source_ref)) or {}
        if int(source_atom.get("formal_charge") or 0) > 0:
            penalties.append("positive_charge_source")
        if int(source_atom.get("total_valence") or 0) >= 5:
            penalties.append("high_valence_source")

    if sink_ref.isdigit():
        sink_atom = atom_state.get(int(sink_ref)) or {}
        if int(sink_atom.get("formal_charge") or 0) < 0:
            penalties.append("negative_charge_sink")

    return {
        "hard_rejected": False,
        "penalties": penalties,
        "penalty": min(0.12 * len(penalties), 0.45),
    }


def _molecule_fallback_refs(current_state: List[str]) -> Tuple[str, str, str]:
    if Chem is None:
        return "s0:a0", "s0:a1", "nucleophilic_addition"

    source_ref: Optional[str] = None
    sink_ref: Optional[str] = None
    source_symbol = ""
    sink_symbol = ""

    for species_index, smiles in enumerate(current_state):
        mol = Chem.MolFromSmiles(str(smiles or "").strip())
        if mol is None:
            continue
        for atom in mol.GetAtoms():
            atom_ref = f"s{species_index}:a{atom.GetIdx()}"
            charge = int(atom.GetFormalCharge())
            symbol = atom.GetSymbol()
            if source_ref is None and charge < 0:
                source_ref = atom_ref
                source_symbol = symbol
            if sink_ref is None and charge > 0:
                sink_ref = atom_ref
                sink_symbol = symbol

        for atom in mol.GetAtoms():
            atom_ref = f"s{species_index}:a{atom.GetIdx()}"
            symbol = atom.GetSymbol()
            if source_ref is None and symbol in {"N", "O", "S", "P", "F", "Cl", "Br", "I"}:
                source_ref = atom_ref
                source_symbol = symbol
            if sink_ref is None and symbol == "C":
                sink_ref = atom_ref
                sink_symbol = symbol

        if source_ref is None and mol.GetNumAtoms() > 0:
            source_ref = f"s{species_index}:a0"
            source_symbol = mol.GetAtomWithIdx(0).GetSymbol()
        if sink_ref is None and mol.GetNumAtoms() > 1:
            sink_ref = f"s{species_index}:a1"
            sink_symbol = mol.GetAtomWithIdx(1).GetSymbol()

        if source_ref and sink_ref:
            break

    if source_ref is None:
        source_ref = "s0:a0"
    if sink_ref is None:
        sink_ref = "s0:a1"

    template = "proton_transfer" if "H" in {source_symbol, sink_symbol} else "nucleophilic_addition"
    return source_ref, sink_ref, template


def _format_annotation_suffix(selected: Dict[str, Any]) -> str:
    src = str(selected.get("source_atom_ref") or "")
    snk = str(selected.get("sink_atom_ref") or "")
    electrons = int(selected.get("electrons") or 2)
    template = str(selected.get("template") or "nucleophilic_addition")
    score = float(selected.get("score") or 0.0)
    score_text = f"{max(0.0, min(score, 1.0)):.3f}"
    return f"|aps:v1;src={src};snk={snk};e={electrons};tpl={template};sc={score_text}|"


def _candidate_record(
    *,
    source_atom_ref: str,
    sink_atom_ref: str,
    template: str,
    bond_evidence: float,
    push_alignment: float,
    electrons: int,
    sanity: Dict[str, Any],
) -> Dict[str, Any]:
    template_prior = _TEMPLATE_PRIOR.get(template, 0.6)
    score = (0.6 * bond_evidence) + (0.25 * push_alignment) + (0.15 * template_prior)
    score -= float(sanity.get("penalty") or 0.0)
    if bool(sanity.get("hard_rejected")):
        score = -1.0
    score = max(-1.0, min(score, 1.0))
    return {
        "source_atom_ref": source_atom_ref,
        "sink_atom_ref": sink_atom_ref,
        "electrons": int(electrons),
        "template": template,
        "score": round(score, 6),
        "score_components": {
            "bond_evidence": round(bond_evidence, 4),
            "push_alignment": round(push_alignment, 4),
            "template_prior": round(template_prior, 4),
            "sanity_penalty": round(float(sanity.get("penalty") or 0.0), 4),
        },
        "sanity": sanity,
    }


def predict_arrow_push_annotation(
    current_state: List[str],
    resulting_state: List[str],
    reaction_smirks: Optional[str],
    raw_reaction_smirks: Optional[str],
    electron_pushes: Optional[List[Dict[str, object]]],
    step_index: int,
    candidate_rank: Optional[int] = None,
) -> Dict[str, Any]:
    """Predict an internal source/sink arrow annotation for a mechanism step."""

    del resulting_state  # retained for signature compatibility and future scoring features.

    base_reaction = str(raw_reaction_smirks or reaction_smirks or "").strip()
    parsed = _extract_bond_changes(base_reaction)
    changes = list(parsed.get("changes") or [])
    atom_state = dict(parsed.get("atom_state") or {})
    carbonyl_carbons = set(parsed.get("carbonyl_carbons") or set())
    pushes = _electron_pushes(electron_pushes)

    candidates: List[Dict[str, Any]] = []

    for change in changes:
        map_i = int(change.get("map_i") or 0)
        map_j = int(change.get("map_j") or 0)
        if map_i <= 0 or map_j <= 0:
            continue

        kind = str(change.get("kind") or "order_changed")
        delta = float(change.get("delta") or 0.0)
        direction_pairs: List[Tuple[int, int]] = []
        if delta > 0:
            direction_pairs.extend([(map_i, map_j), (map_j, map_i)])
        else:
            direction_pairs.extend([(map_j, map_i), (map_i, map_j)])

        for src_map, snk_map in direction_pairs:
            src_ref = _map_ref(src_map)
            snk_ref = _map_ref(snk_map)
            push_alignment, electrons = _push_alignment_score(src_ref, snk_ref, pushes)
            sanity = _sanity_check(src_ref, snk_ref, atom_state)
            templates = _candidate_templates(
                src_map=src_map,
                snk_map=snk_map,
                kind=kind,
                atom_state=atom_state,
                carbonyl_carbons=carbonyl_carbons,
            )
            for template in templates:
                candidates.append(
                    _candidate_record(
                        source_atom_ref=src_ref,
                        sink_atom_ref=snk_ref,
                        template=template,
                        bond_evidence=1.0,
                        push_alignment=push_alignment,
                        electrons=electrons,
                        sanity=sanity,
                    )
                )

    if not candidates:
        src_ref, snk_ref, template = _molecule_fallback_refs(current_state)
        push_alignment, electrons = _push_alignment_score(src_ref, snk_ref, pushes)
        sanity = _sanity_check(src_ref, snk_ref, atom_state={})
        candidates.append(
            _candidate_record(
                source_atom_ref=src_ref,
                sink_atom_ref=snk_ref,
                template=template,
                bond_evidence=0.35,
                push_alignment=push_alignment,
                electrons=electrons,
                sanity=sanity,
            )
        )

    valid_candidates = [
        item for item in candidates if not bool((item.get("sanity") or {}).get("hard_rejected"))
    ]
    if not valid_candidates:
        valid_candidates = list(candidates)

    valid_candidates.sort(key=lambda item: float(item.get("score") or -1.0), reverse=True)
    ranked: List[Dict[str, Any]] = []
    for idx, item in enumerate(valid_candidates, start=1):
        entry = dict(item)
        entry["rank"] = idx
        ranked.append(entry)

    selected = dict(ranked[0]) if ranked else {
        "rank": 1,
        "source_atom_ref": "s0:a0",
        "sink_atom_ref": "s0:a1",
        "electrons": 2,
        "template": "nucleophilic_addition",
        "score": 0.0,
        "score_components": {
            "bond_evidence": 0.0,
            "push_alignment": 0.0,
            "template_prior": _TEMPLATE_PRIOR["nucleophilic_addition"],
            "sanity_penalty": 0.0,
        },
        "sanity": {"hard_rejected": False, "penalties": [], "penalty": 0.0},
    }
    suffix = _format_annotation_suffix(selected)

    annotated_smirks = ""
    if base_reaction:
        annotated_smirks = f"{base_reaction} {suffix}" if not base_reaction.endswith(suffix) else base_reaction
    else:
        annotated_smirks = suffix

    return {
        "status": "success" if changes else "fallback",
        "step_index": int(step_index),
        "candidate_rank": candidate_rank,
        "source": "orbchain_lite_v1",
        "ranked_candidates": ranked,
        "selected_candidate": selected,
        "annotation_suffix": suffix,
        "annotated_reaction_smirks": annotated_smirks,
        "used_bond_change_priority": bool(changes),
        "used_electron_push_reconciliation": bool(pushes),
        "used_functional_group_fallback": not bool(changes),
    }


__all__ = ["predict_arrow_push_annotation"]
