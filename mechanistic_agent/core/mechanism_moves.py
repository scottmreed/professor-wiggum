"""Shared parser/serializer for explicit mechanism-move metadata."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

try:  # pragma: no cover - optional dependency
    from rdkit import Chem
except Exception:  # pragma: no cover - defensive
    Chem = None  # type: ignore[assignment]


MoveKind = Literal["lone_pair", "pi_bond", "sigma_bond"]


class MechanismMoveFormatError(ValueError):
    """Raised when explicit mechanism-move metadata cannot be parsed."""


@dataclass(frozen=True)
class MechanismMove:
    """One explicit arrow-pushing move."""

    kind: MoveKind
    target_atom: int
    source_atom: Optional[int] = None
    bond_start: Optional[int] = None
    bond_end: Optional[int] = None
    through_atom: Optional[int] = None
    electrons: int = 2

    def as_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "kind": self.kind,
            "target_atom": str(self.target_atom),
            "electrons": self.electrons,
        }
        if self.source_atom is not None:
            payload["source_atom"] = str(self.source_atom)
        if self.bond_start is not None and self.bond_end is not None:
            payload["source_bond"] = [str(self.bond_start), str(self.bond_end)]
        if self.through_atom is not None:
            payload["through_atom"] = str(self.through_atom)
        payload["notation"] = serialize_move_token(self)
        return payload


def split_cxsmiles_metadata(expression: str) -> Tuple[str, List[str]]:
    text = str(expression or "").strip()
    if "|" not in text:
        return text, []
    core_part, remainder = text.split("|", 1)
    metadata_chunks = remainder.split("|")
    metadata = [chunk.strip() for chunk in metadata_chunks if chunk.strip()]
    return core_part.strip(), metadata


def _normalize_kind(raw: str) -> Optional[MoveKind]:
    text = str(raw or "").strip().lower().replace("-", "_")
    aliases = {
        "lp": "lone_pair",
        "lone_pair": "lone_pair",
        "lonepair": "lone_pair",
        "pi": "pi_bond",
        "pi_bond": "pi_bond",
        "pibond": "pi_bond",
        "sigma": "sigma_bond",
        "sigma_bond": "sigma_bond",
        "sigmabond": "sigma_bond",
    }
    value = aliases.get(text)
    if value in {"lone_pair", "pi_bond", "sigma_bond"}:
        return value
    return None


def serialize_move_token(move: MechanismMove) -> str:
    if move.kind == "lone_pair":
        return f"lp:{move.source_atom}>{move.target_atom}"
    prefix = "pi" if move.kind == "pi_bond" else "sigma"
    return f"{prefix}:{move.bond_start}-{move.bond_end}>{move.target_atom}"


def serialize_mechanism_moves(moves: Iterable[MechanismMove]) -> str:
    rendered = [serialize_move_token(move) for move in moves]
    if not rendered:
        raise MechanismMoveFormatError("mechanism move block did not contain any entries")
    return ";".join(rendered)


def parse_move_token(token: str) -> MechanismMove:
    text = str(token or "").strip()
    if not text:
        raise MechanismMoveFormatError("mechanism move token is empty")
    if ":" not in text:
        raise MechanismMoveFormatError(f"Entry '{text}' is missing the ':' separator")
    kind_text, rest = text.split(":", 1)
    kind = _normalize_kind(kind_text)
    if kind is None:
        raise MechanismMoveFormatError(f"Entry '{text}' uses an unsupported move kind")
    if ">" not in rest:
        raise MechanismMoveFormatError(f"Entry '{text}' is missing the '>' separator")
    source_text, target_text = rest.split(">", 1)
    if not re.fullmatch(r"\d+", target_text.strip()):
        raise MechanismMoveFormatError(f"Entry '{text}' must reference a numeric target atom")
    target_atom = int(target_text.strip())
    if kind == "lone_pair":
        if not re.fullmatch(r"\d+", source_text.strip()):
            raise MechanismMoveFormatError(f"Entry '{text}' must reference a numeric source atom")
        source_atom = int(source_text.strip())
        return MechanismMove(kind=kind, source_atom=source_atom, target_atom=target_atom)
    source_text = source_text.strip()
    if "-" not in source_text:
        raise MechanismMoveFormatError(f"Entry '{text}' must reference a source bond as mapI-mapJ")
    left, right = source_text.split("-", 1)
    if not re.fullmatch(r"\d+", left) or not re.fullmatch(r"\d+", right):
        raise MechanismMoveFormatError(f"Entry '{text}' must reference numeric atom-map indices")
    bond_start = int(left)
    bond_end = int(right)
    return MechanismMove(
        kind=kind,
        bond_start=bond_start,
        bond_end=bond_end,
        through_atom=bond_end,
        target_atom=target_atom,
    )


def parse_mechanism_moves(entries: str) -> List[MechanismMove]:
    if not str(entries or "").strip():
        raise MechanismMoveFormatError("mechanism move block is empty")
    parsed: List[MechanismMove] = []
    for raw in str(entries).split(";"):
        token = raw.strip()
        if not token:
            continue
        parsed.append(parse_move_token(token))
    if not parsed:
        raise MechanismMoveFormatError("mechanism move block did not contain any entries")
    return parsed


def normalize_electron_pushes(pushes: Any) -> List[MechanismMove]:
    moves: List[MechanismMove] = []
    if not isinstance(pushes, list):
        return moves
    for item in pushes:
        if not isinstance(item, dict):
            continue
        notation = item.get("notation")
        if notation is not None:
            try:
                moves.append(parse_move_token(str(notation)))
            except MechanismMoveFormatError:
                continue
            continue
        kind = _normalize_kind(str(item.get("kind") or item.get("source_kind") or ""))
        if kind == "lone_pair":
            source_atom = item.get("source_atom", item.get("start_atom"))
            target_atom = item.get("target_atom", item.get("end_atom"))
            if source_atom is None or target_atom is None:
                continue
            try:
                moves.append(
                    MechanismMove(
                        kind="lone_pair",
                        source_atom=int(str(source_atom)),
                        target_atom=int(str(target_atom)),
                    )
                )
            except ValueError:
                continue
            continue
        if kind in {"pi_bond", "sigma_bond"}:
            source_bond = item.get("source_bond")
            through_atom = item.get("through_atom")
            target_atom = item.get("target_atom", item.get("end_atom"))
            if not isinstance(source_bond, list) or len(source_bond) != 2 or target_atom is None:
                continue
            # Auto-fill through_atom from source_bond[1] when absent (convention:
            # the second bond atom delivers the electrons, matching parse_move_token).
            if through_atom is None:
                through_atom = source_bond[1]
            try:
                moves.append(
                    MechanismMove(
                        kind=kind,
                        bond_start=int(str(source_bond[0])),
                        bond_end=int(str(source_bond[1])),
                        through_atom=int(str(through_atom)),
                        target_atom=int(str(target_atom)),
                    )
                )
            except ValueError:
                continue
            continue
        start_atom = item.get("start_atom")
        end_atom = item.get("end_atom")
        if start_atom is None or end_atom is None:
            continue
        try:
            moves.append(
                MechanismMove(
                    kind="lone_pair",
                    source_atom=int(str(start_atom)),
                    target_atom=int(str(end_atom)),
                )
            )
        except ValueError:
            continue
    return moves


def extract_mechanism_moves(expression: str) -> Tuple[str, List[MechanismMove], Dict[str, Any]]:
    core, metadata = split_cxsmiles_metadata(expression)
    details: Dict[str, Any] = {"raw": str(expression or "").strip(), "core": core, "metadata": metadata}
    mech_entry = None
    legacy_dbe_entry = None
    for item in metadata:
        lowered = item.lower()
        if lowered.startswith("mech:"):
            mech_entry = item
            break
        if lowered.startswith("dbe:"):
            legacy_dbe_entry = item
    if mech_entry is not None:
        value = mech_entry.split(":", 1)[1].strip() if ":" in mech_entry else ""
        if value.lower().startswith("v1;"):
            value = value[3:]
        details["mech"] = value
        try:
            moves = parse_mechanism_moves(value)
            details["moves"] = [move.as_dict() for move in moves]
            return core, moves, details
        except MechanismMoveFormatError as exc:
            details["error"] = str(exc)
            return core, [], details
    if legacy_dbe_entry is not None:
        details["error"] = "Legacy dbe metadata is deprecated; rewrite as |mech:v1;...|"
        details["legacy_dbe"] = legacy_dbe_entry.split(":", 1)[1].strip() if ":" in legacy_dbe_entry else ""
        return core, [], details
    details["error"] = "Missing mech metadata block"
    return core, [], details


def reaction_bond_deltas(reaction_smirks: str) -> List[Dict[str, Any]]:
    if Chem is None:
        return []
    core, _metadata = split_cxsmiles_metadata(reaction_smirks)
    if ">>" not in core:
        return []
    left, right = core.split(">>", 1)

    def collect(side: str) -> Dict[Tuple[int, int], float]:
        bonds: Dict[Tuple[int, int], float] = {}
        for token in side.split("."):
            smiles = str(token or "").strip()
            if not smiles:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            for bond in mol.GetBonds():
                begin = int(bond.GetBeginAtom().GetAtomMapNum() or 0)
                end = int(bond.GetEndAtom().GetAtomMapNum() or 0)
                if begin <= 0 or end <= 0:
                    continue
                bonds[tuple(sorted((begin, end)))] = float(bond.GetBondTypeAsDouble())
        return bonds

    left_bonds = collect(left)
    right_bonds = collect(right)
    changes: List[Dict[str, Any]] = []
    for pair in sorted(set(left_bonds) | set(right_bonds)):
        left_order = left_bonds.get(pair, 0.0)
        right_order = right_bonds.get(pair, 0.0)
        if abs(left_order - right_order) < 1e-6:
            continue
        changes.append({"pair": pair, "delta": right_order - left_order})
    return changes


def implied_bond_deltas(moves: Iterable[MechanismMove]) -> List[Dict[str, Any]]:
    totals: Dict[Tuple[int, int], float] = {}
    for move in moves:
        if move.kind == "lone_pair":
            assert move.source_atom is not None
            pair = tuple(sorted((move.source_atom, move.target_atom)))
            totals[pair] = totals.get(pair, 0.0) + 1.0
            continue
        assert move.bond_start is not None and move.bond_end is not None and move.through_atom is not None
        broken = tuple(sorted((move.bond_start, move.bond_end)))
        totals[broken] = totals.get(broken, 0.0) - 1.0
        if move.through_atom != move.target_atom:
            formed = tuple(sorted((move.through_atom, move.target_atom)))
            totals[formed] = totals.get(formed, 0.0) + 1.0
    return [{"pair": pair, "delta": delta} for pair, delta in sorted(totals.items()) if abs(delta) > 1e-6]


def synthesize_mechanism_metadata(electron_pushes: Any) -> Optional[str]:
    moves = normalize_electron_pushes(electron_pushes)
    if not moves:
        return None
    return serialize_mechanism_moves(moves)


def synthesize_dbe_entries(electron_pushes: Any) -> Optional[str]:
    moves = normalize_electron_pushes(electron_pushes)
    if not moves:
        return None
    pair_totals: Dict[Tuple[int, int], int] = {}
    for move in moves:
        if move.kind == "lone_pair":
            assert move.source_atom is not None
            pair_totals[(move.source_atom, move.target_atom)] = pair_totals.get((move.source_atom, move.target_atom), 0) + 2
            pair_totals[(move.source_atom, move.source_atom)] = pair_totals.get((move.source_atom, move.source_atom), 0) - 2
            continue
        assert move.bond_start is not None and move.bond_end is not None and move.through_atom is not None
        pair_totals[(move.bond_start, move.bond_end)] = pair_totals.get((move.bond_start, move.bond_end), 0) - 2
        if move.through_atom == move.target_atom:
            pair_totals[(move.target_atom, move.target_atom)] = pair_totals.get((move.target_atom, move.target_atom), 0) + 2
        else:
            pair_totals[(move.through_atom, move.target_atom)] = pair_totals.get((move.through_atom, move.target_atom), 0) + 2
    rows = [(pair, delta) for pair, delta in pair_totals.items() if delta != 0]
    if not rows:
        return None
    rows.sort(key=lambda item: (item[0][0], item[0][1]))
    return ";".join(f"{pair[0]}-{pair[1]}:{delta:+d}" for pair, delta in rows)


def repair_candidate_reaction_smirks(*, reaction_smirks: Any, electron_pushes: Any) -> Tuple[Optional[str], Optional[str]]:
    text = str(reaction_smirks or "").strip()
    if not text:
        return None, "missing_reaction_smirks"
    moves = normalize_electron_pushes(electron_pushes)
    if not moves:
        return None, "missing_electron_pushes"
    match = re.search(r"\|mech:([^|]*)\|", text, flags=re.IGNORECASE)
    if match:
        value = match.group(1).strip()
        if value.lower().startswith("v1;"):
            value = value[3:]
        try:
            normalized = serialize_mechanism_moves(parse_mechanism_moves(value))
        except MechanismMoveFormatError:
            return None, "reaction_smirks_invalid_mech_block"
        repaired = re.sub(
            r"\|mech:[^|]*\|",
            f"|mech:v1;{normalized}|",
            text,
            count=1,
            flags=re.IGNORECASE,
        )
        return repaired, "normalized_mech_entries"
    synthesized = synthesize_mechanism_metadata(electron_pushes)
    if not synthesized:
        return None, "reaction_smirks_missing_mech_block"
    text = re.sub(r"\|dbe:[^|]*\|", "", text, flags=re.IGNORECASE).strip()
    return f"{text} |mech:v1;{synthesized}|", "synthesized_mech_from_electron_pushes"


__all__ = [
    "MechanismMove",
    "MechanismMoveFormatError",
    "extract_mechanism_moves",
    "implied_bond_deltas",
    "normalize_electron_pushes",
    "parse_mechanism_moves",
    "reaction_bond_deltas",
    "repair_candidate_reaction_smirks",
    "serialize_mechanism_moves",
    "split_cxsmiles_metadata",
    "synthesize_dbe_entries",
    "synthesize_mechanism_metadata",
]
