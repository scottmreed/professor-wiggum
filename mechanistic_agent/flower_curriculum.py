"""Shared FlowER curriculum and conversion helpers.

This module centralises:
- FlowER train.txt parsing and lookup caching
- deterministic curriculum indexing and ranking
- conversion from FlowER elementary steps into Mechanistic rich-step cases
- curriculum progress inspection from eval history
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FLOWER_INPUT = PROJECT_ROOT.parent / "FlowER" / "data" / "flower_new_dataset" / "train.txt"
DEFAULT_LOOKUP_CACHE = PROJECT_ROOT / "data" / "flower_train_lookup.sqlite"
DEFAULT_INDEX_PATH = PROJECT_ROOT / "training_data" / "flower_mechanism_index.jsonl"
DEFAULT_INDEX_REPORT_PATH = PROJECT_ROOT / "training_data" / "flower_mechanism_index_report.json"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "training_data" / "flower_mechanisms_100.json"
DEFAULT_DATASET_REPORT_PATH = PROJECT_ROOT / "training_data" / "flower_mechanisms_100_report.json"

SOURCE_LABEL = "FlowER flower_new_dataset train.txt"
SOURCE_REF = "https://github.com/schwallergroup/ChRIMP"
ATOM_MAP_PATTERN = re.compile(r":\d+\]")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mechanistic_agent.core.mechanism_moves import (  # noqa: E402
    extract_mechanism_moves,
    normalize_electron_pushes,
    repair_candidate_reaction_smirks,
    serialize_mechanism_moves,
    synthesize_mechanism_metadata,
)


class ConversionError(ValueError):
    """Raised when a mapped FlowER step cannot be converted."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ParsedLine:
    """One parsed FlowER line."""

    mechanism_id: int
    mapped_reaction: str


@dataclass(frozen=True)
class ConvertedStep:
    """One converted elementary step."""

    reaction_smirks: str
    electron_pushes: List[Dict[str, Any]]
    raw_current_state: List[str]
    raw_resulting_state: List[str]
    predicted_intermediate: str


@dataclass(frozen=True)
class CurriculumIndexEntry:
    """One ranked mechanism summary from the FlowER corpus."""

    mechanism_id: int
    case_id: str
    step_count: int
    reagent_count: int
    total_heavy_atoms: int
    max_reagent_heavy_atoms: int
    total_hetero_atoms: int
    global_rank: int
    rank_within_step_count: int
    rank_key: List[int]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mechanism_id": self.mechanism_id,
            "case_id": self.case_id,
            "step_count": self.step_count,
            "reagent_count": self.reagent_count,
            "total_heavy_atoms": self.total_heavy_atoms,
            "max_reagent_heavy_atoms": self.max_reagent_heavy_atoms,
            "total_hetero_atoms": self.total_hetero_atoms,
            "global_rank": self.global_rank,
            "rank_within_step_count": self.rank_within_step_count,
            "rank_key": list(self.rank_key),
        }


def _json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _unique_preserving_order(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _tokenize_species(text: str) -> List[str]:
    return [token.strip() for token in str(text or "").split(".") if token.strip()]


def _parse_line(line: str) -> Optional[ParsedLine]:
    text = str(line or "").rstrip("\n")
    if not text or "|" not in text:
        return None
    mapped_reaction, raw_idx = text.rsplit("|", 1)
    idx = raw_idx.strip()
    if not idx.isdigit():
        return None
    return ParsedLine(mechanism_id=int(idx), mapped_reaction=mapped_reaction.strip())


def _split_reaction_sides(mapped_reaction: str) -> Tuple[str, str]:
    text = str(mapped_reaction or "").strip()
    if not text:
        raise ConversionError("empty_reaction")
    if ">>" in text:
        left, right = text.split(">>", 1)
        return left.strip(), right.strip()
    parts = text.split(">")
    if len(parts) == 3:
        left, _middle, right = parts
        return left.strip(), right.strip()
    raise ConversionError("invalid_reaction_shape")


def _strip_shared_species(mapped_reaction: str) -> Tuple[List[str], List[str], List[str]]:
    left, right = _split_reaction_sides(mapped_reaction)
    left_tokens = _tokenize_species(left)
    right_tokens = _tokenize_species(right)
    if not left_tokens or not right_tokens:
        raise ConversionError("missing_reaction_side")

    right_counts = Counter(right_tokens)
    shared: List[str] = []
    reactant_only: List[str] = []
    for token in left_tokens:
        if right_counts[token] > 0:
            shared.append(token)
            right_counts[token] -= 1
        else:
            reactant_only.append(token)

    left_counts = Counter(left_tokens)
    product_only: List[str] = []
    for token in right_tokens:
        if left_counts[token] > 0:
            left_counts[token] -= 1
        else:
            product_only.append(token)

    return reactant_only, shared, product_only


def _is_trivial_reaction(mapped_reaction: str) -> bool:
    try:
        left, right = _split_reaction_sides(mapped_reaction)
    except ConversionError:
        return False
    return _unique_preserving_order(_tokenize_species(left)) == _unique_preserving_order(_tokenize_species(right))


def _bond_orders_by_map(mol: Chem.Mol) -> Dict[Tuple[int, int], int]:
    bonds: Dict[Tuple[int, int], int] = {}
    for bond in mol.GetBonds():
        degree = float(bond.GetBondTypeAsDouble())
        if degree != int(degree):
            raise ConversionError("aromatic_bond_change")
        begin = int(bond.GetBeginAtom().GetAtomMapNum() or 0)
        end = int(bond.GetEndAtom().GetAtomMapNum() or 0)
        if begin <= 0 or end <= 0:
            raise ConversionError("missing_atom_map")
        bonds[tuple(sorted((begin, end)))] = int(degree)
    return bonds


def _charges_by_map(mol: Chem.Mol) -> Dict[int, int]:
    charges: Dict[int, int] = {}
    for atom in mol.GetAtoms():
        map_num = int(atom.GetAtomMapNum() or 0)
        if map_num <= 0:
            raise ConversionError("missing_atom_map")
        charges[map_num] = int(atom.GetFormalCharge())
    return charges


def _shortest_path(adjacency: Dict[int, set[int]], start: int, goal: int) -> Optional[List[int]]:
    queue: deque[int] = deque([start])
    parent: Dict[int, Optional[int]] = {start: None}
    while queue:
        node = queue.popleft()
        if node == goal:
            path: List[int] = []
            current: Optional[int] = node
            while current is not None:
                path.append(current)
                current = parent[current]
            return list(reversed(path))
        for neighbor in sorted(adjacency.get(node, set())):
            if neighbor in parent:
                continue
            parent[neighbor] = node
            queue.append(neighbor)
    return None


def _find_cycle(adjacency: Dict[int, set[int]]) -> Optional[List[int]]:
    visited: set[int] = set()
    in_stack: set[int] = set()
    parent: Dict[int, Optional[int]] = {}

    def dfs(node: int, prev: Optional[int]) -> Optional[List[int]]:
        visited.add(node)
        in_stack.add(node)
        parent[node] = prev
        for neighbor in sorted(adjacency.get(node, set())):
            if neighbor == prev:
                continue
            if neighbor not in visited:
                cycle = dfs(neighbor, node)
                if cycle:
                    return cycle
                continue
            if neighbor not in in_stack:
                continue
            chain = [node]
            while chain[-1] != neighbor:
                parent_node = parent.get(chain[-1])
                if parent_node is None:
                    return None
                chain.append(parent_node)
            chain.reverse()
            chain.append(chain[0])
            return chain
        in_stack.remove(node)
        return None

    for root in sorted(adjacency):
        if root in visited:
            continue
        cycle = dfs(root, None)
        if cycle:
            return cycle
    return None


def _move_dict_from_arrow(
    arrow: object,
    reactant_bonds: Dict[Tuple[int, int], int],
) -> Dict[str, Any]:
    if isinstance(arrow, tuple) and len(arrow) == 2 and all(isinstance(part, int) for part in arrow):
        source_atom, target_atom = arrow
        return {
            "kind": "lone_pair",
            "source_atom": str(source_atom),
            "target_atom": str(target_atom),
            "electrons": 2,
        }

    if (
        isinstance(arrow, tuple)
        and len(arrow) == 2
        and isinstance(arrow[0], tuple)
        and len(arrow[0]) == 2
        and all(isinstance(part, int) for part in arrow[0])
        and isinstance(arrow[1], int)
    ):
        bond_start, bond_end = arrow[0]
        target_atom = arrow[1]
        bond_order = reactant_bonds.get(tuple(sorted((bond_start, bond_end))))
        if bond_order is None:
            raise ConversionError("missing_source_bond")
        if bond_order == 2:
            kind = "pi_bond"
        elif bond_order == 1:
            kind = "sigma_bond"
        else:
            raise ConversionError("unsupported_bond_order")
        return {
            "kind": kind,
            "source_bond": [str(bond_start), str(bond_end)],
            "through_atom": str(bond_end),
            "target_atom": str(target_atom),
            "electrons": 2,
        }

    raise ConversionError("unsupported_arrow_shape")


def _infer_arrows_from_bond_and_charge_changes(mapped_reaction: str) -> Tuple[List[Dict[str, Any]], str]:
    from rdkit import Chem  # RDKit required for molecule parsing
    reactant_text, product_text = _split_reaction_sides(mapped_reaction)
    reactant = Chem.MolFromSmiles(reactant_text, sanitize=False)
    product = Chem.MolFromSmiles(product_text, sanitize=False)
    if reactant is None or product is None:
        raise ConversionError("rdkit_parse_failed")
    if reactant.GetNumAtoms() == 0 or product.GetNumAtoms() == 0:
        raise ConversionError("empty_molecule")
    if reactant.GetNumAtoms() != product.GetNumAtoms():
        raise ConversionError("atom_count_mismatch")

    reactant_bonds = _bond_orders_by_map(reactant)
    product_bonds = _bond_orders_by_map(product)
    reactant_charges = _charges_by_map(reactant)
    product_charges = _charges_by_map(product)

    if set(reactant_charges) != set(product_charges):
        raise ConversionError("atom_map_mismatch")

    charge_diff = {
        atom_map: int(product_charges.get(atom_map, 0) - reactant_charges.get(atom_map, 0))
        for atom_map in reactant_charges
    }

    bond_diff: Dict[Tuple[int, int], int] = {}
    for pair in sorted(set(reactant_bonds) | set(product_bonds)):
        delta = int(product_bonds.get(pair, 0) - reactant_bonds.get(pair, 0))
        if delta:
            bond_diff[pair] = delta

    if not bond_diff:
        raise ConversionError("no_bond_changes")

    adjacency: Dict[int, set[int]] = defaultdict(set)
    for begin, end in bond_diff:
        adjacency[begin].add(end)
        adjacency[end].add(begin)

    plus_nodes = sorted(atom for atom, delta in charge_diff.items() if delta == 1)
    minus_nodes = sorted(atom for atom, delta in charge_diff.items() if delta == -1)

    node_path: Optional[List[int]]
    if len(plus_nodes) == 1 and len(minus_nodes) == 1:
        node_path = _shortest_path(adjacency, plus_nodes[0], minus_nodes[0])
        if node_path is None:
            raise ConversionError("no_path_found")
    elif len(plus_nodes) == 0 and len(minus_nodes) == 0:
        node_path = _find_cycle(adjacency)
        if node_path is None:
            raise ConversionError("no_cycle_found")
    else:
        raise ConversionError(f"unsupported_charge_pattern_{len(plus_nodes)}_{len(minus_nodes)}")

    edge_deltas = [bond_diff[tuple(sorted((node_path[i], node_path[i + 1])))] for i in range(len(node_path) - 1)]
    if plus_nodes == [] and minus_nodes == [] and edge_deltas and edge_deltas[0] != -1:
        node_path = list(reversed(node_path))
        edge_deltas = list(reversed(edge_deltas))

    primitive_arrows: List[Tuple[str, int, int]] = []
    for idx, delta in enumerate(edge_deltas):
        primitive_arrows.append(("i" if delta < 0 else "a", node_path[idx], node_path[idx + 1]))

    arrows: List[object] = []
    cursor = 0
    while cursor < len(primitive_arrows):
        if (
            cursor + 1 < len(primitive_arrows)
            and primitive_arrows[cursor][0] == "i"
            and primitive_arrows[cursor + 1][0] == "a"
            and primitive_arrows[cursor][2] == primitive_arrows[cursor + 1][1]
        ):
            first = primitive_arrows[cursor]
            second = primitive_arrows[cursor + 1]
            arrows.append(((first[1], first[2]), second[2]))
            cursor += 2
            continue

        direction, start_atom, end_atom = primitive_arrows[cursor]
        if direction == "a":
            arrows.append((start_atom, end_atom))
        else:
            arrows.append(((start_atom, end_atom), end_atom))
        cursor += 1

    electron_pushes = [_move_dict_from_arrow(arrow, reactant_bonds) for arrow in arrows]
    normalized = normalize_electron_pushes(electron_pushes)
    if len(normalized) != len(electron_pushes):
        raise ConversionError("electron_push_normalization_failed")
    notation = serialize_mechanism_moves(normalized)
    return [move.as_dict() for move in normalized], notation


def convert_elementary_step(mapped_reaction: str) -> ConvertedStep:
    reactant_only, _carry_through, product_only = _strip_shared_species(mapped_reaction)
    left, right = _split_reaction_sides(mapped_reaction)
    left_state = _unique_preserving_order(_tokenize_species(left))
    right_state = _unique_preserving_order(_tokenize_species(right))
    if left_state == right_state:
        raise ConversionError("trivial_step")
    if not reactant_only and not product_only:
        raise ConversionError("trivial_step")

    electron_pushes, notation = _infer_arrows_from_bond_and_charge_changes(mapped_reaction)
    repaired, reason = repair_candidate_reaction_smirks(
        reaction_smirks=f"{left}>>{right}",
        electron_pushes=electron_pushes,
    )
    if not repaired:
        raise ConversionError(reason or "reaction_smirks_repair_failed")

    core, parsed_moves, details = extract_mechanism_moves(repaired)
    if details.get("error"):
        raise ConversionError("invalid_mech_metadata")
    if len(parsed_moves) != len(electron_pushes):
        raise ConversionError("mech_metadata_roundtrip_mismatch")
    synthesized = synthesize_mechanism_metadata(electron_pushes)
    if synthesized != notation:
        raise ConversionError("mech_metadata_synthesis_mismatch")

    predicted_intermediate = (
        product_only[0]
        if len(product_only) == 1
        else ".".join(_unique_preserving_order(product_only or right_state))
    )

    return ConvertedStep(
        reaction_smirks=f"{core} |mech:v1;{notation}|",
        electron_pushes=electron_pushes,
        raw_current_state=_unique_preserving_order(left_state),
        raw_resulting_state=_unique_preserving_order(right_state),
        predicted_intermediate=predicted_intermediate,
    )


def _convert_group(mechanism_id: int, mapped_reactions: Sequence[str]) -> Dict[str, Any]:
    converted_steps: List[ConvertedStep] = []
    prior_state: Optional[List[str]] = None
    for mapped_reaction in mapped_reactions:
        step = convert_elementary_step(mapped_reaction)
        if prior_state is not None:
            expected = set(prior_state)
            observed = set(step.raw_current_state)
            if expected != observed:
                raise ConversionError("state_discontinuity")
            step = ConvertedStep(
                reaction_smirks=step.reaction_smirks,
                electron_pushes=step.electron_pushes,
                raw_current_state=list(prior_state),
                raw_resulting_state=step.raw_resulting_state,
                predicted_intermediate=step.predicted_intermediate,
            )
        converted_steps.append(step)
        prior_state = list(step.raw_resulting_state)

    if not converted_steps:
        raise ConversionError("empty_mechanism")

    starting_materials = _unique_preserving_order(converted_steps[0].raw_current_state)
    products = _unique_preserving_order(converted_steps[-1].raw_resulting_state)
    target_products = list(products)

    case_steps: List[Dict[str, Any]] = []
    for index, step in enumerate(converted_steps, start=1):
        case_steps.append(
            {
                "step_index": index,
                "current_state": list(step.raw_current_state),
                "resulting_state": list(step.raw_resulting_state),
                "predicted_intermediate": step.predicted_intermediate,
                "target_products": list(target_products),
                "electron_pushes": list(step.electron_pushes),
                "reaction_smirks": step.reaction_smirks,
                "confidence": 1.0,
                "note": "Converted from FlowER elementary step; auto-generated.",
            }
        )

    return {
        "id": f"flower_{mechanism_id:06d}",
        "name": f"FlowER mechanism {mechanism_id}",
        "description": f"Converted from FlowER train.txt group {mechanism_id}.",
        "starting_materials": starting_materials,
        "products": products,
        "temperature_celsius": None,
        "ph": None,
        "source": SOURCE_LABEL,
        "tags": ["flower", "train", "multistep"],
        "n_mechanistic_steps": len(case_steps),
        "verified_mechanism": {
            "version": "1.0.0",
            "provisional": True,
            "source_refs": [SOURCE_REF],
            "steps": case_steps,
        },
    }


def _strip_atom_maps(text: str) -> str:
    return ATOM_MAP_PATTERN.sub("]", str(text or ""))


@lru_cache(maxsize=500_000)
def _species_metric_tuple(smiles: str) -> Tuple[str, int, int]:
    from rdkit import Chem  # RDKit required for canonicalization
    stripped = _strip_atom_maps(smiles)
    if not stripped:
        return ("", 0, 0)
    mol = Chem.MolFromSmiles(stripped)
    canonical = stripped
    if mol is None:
        mol = Chem.MolFromSmiles(stripped, sanitize=False)
    if mol is None:
        return (canonical, 0, 0)
    try:
        canonical = Chem.MolToSmiles(mol)
    except Exception:
        canonical = stripped
    heavy_atoms = int(mol.GetNumHeavyAtoms())
    hetero_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in {1, 6})
    return (canonical, heavy_atoms, hetero_atoms)


def metrics_for_species(species: Iterable[str]) -> Dict[str, int]:
    unique: List[str] = []
    seen: set[str] = set()
    for item in species:
        canonical, _heavy_atoms, _hetero_atoms = _species_metric_tuple(str(item or ""))
        key = canonical or _strip_atom_maps(str(item or ""))
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(key)

    total_heavy_atoms = 0
    max_reagent_heavy_atoms = 0
    total_hetero_atoms = 0
    for token in unique:
        _canonical, heavy_atoms, hetero_atoms = _species_metric_tuple(token)
        total_heavy_atoms += heavy_atoms
        max_reagent_heavy_atoms = max(max_reagent_heavy_atoms, heavy_atoms)
        total_hetero_atoms += hetero_atoms

    return {
        "reagent_count": len(unique),
        "total_heavy_atoms": total_heavy_atoms,
        "max_reagent_heavy_atoms": max_reagent_heavy_atoms,
        "total_hetero_atoms": total_hetero_atoms,
    }


def build_lookup_cache(
    *,
    input_path: Path = DEFAULT_FLOWER_INPUT,
    cache_path: Path = DEFAULT_LOOKUP_CACHE,
    force: bool = False,
) -> Path:
    """Create or refresh the local lookup cache for non-contiguous FlowER ids."""

    input_path = Path(input_path)
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    source_size = input_path.stat().st_size
    source_mtime = input_path.stat().st_mtime

    if not force and lookup_cache_is_current(cache_path=cache_path, input_path=input_path):
        return cache_path

    if cache_path.exists():
        cache_path.unlink()

    conn = sqlite3.connect(cache_path)
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(
            """
            CREATE TABLE cache_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE mechanism_rows (
                mechanism_id INTEGER NOT NULL,
                row_order INTEGER NOT NULL,
                offset INTEGER NOT NULL,
                length INTEGER NOT NULL,
                PRIMARY KEY(mechanism_id, row_order)
            );
            CREATE INDEX idx_mechanism_rows_lookup
                ON mechanism_rows(mechanism_id, row_order);
            """
        )

        inserts: List[Tuple[int, int, int, int]] = []
        row_order: Dict[int, int] = defaultdict(int)
        numeric_rows = 0
        trivial_rows = 0
        with input_path.open("rb") as handle:
            while True:
                offset = handle.tell()
                raw_line = handle.readline()
                if not raw_line:
                    break
                text = raw_line.decode("utf-8")
                parsed = _parse_line(text)
                if parsed is None:
                    continue
                numeric_rows += 1
                if _is_trivial_reaction(parsed.mapped_reaction):
                    trivial_rows += 1
                    continue
                order = row_order[parsed.mechanism_id]
                row_order[parsed.mechanism_id] = order + 1
                inserts.append((parsed.mechanism_id, order, offset, len(raw_line)))
                if len(inserts) >= 10_000:
                    conn.executemany(
                        """
                        INSERT INTO mechanism_rows(mechanism_id, row_order, offset, length)
                        VALUES (?, ?, ?, ?)
                        """,
                        inserts,
                    )
                    conn.commit()
                    inserts.clear()

        if inserts:
            conn.executemany(
                """
                INSERT INTO mechanism_rows(mechanism_id, row_order, offset, length)
                VALUES (?, ?, ?, ?)
                """,
                inserts,
            )

        conn.executemany(
            "INSERT INTO cache_meta(key, value) VALUES (?, ?)",
            [
                ("input_path", str(input_path)),
                ("input_size", str(source_size)),
                ("input_mtime", str(source_mtime)),
                ("generated_at", datetime.now(timezone.utc).isoformat()),
                ("numeric_rows", str(numeric_rows)),
                ("trivial_rows_skipped", str(trivial_rows)),
                ("mechanism_count", str(len(row_order))),
            ],
        )
        conn.commit()
    finally:
        conn.close()
    return cache_path


def lookup_cache_is_current(*, cache_path: Path = DEFAULT_LOOKUP_CACHE, input_path: Path = DEFAULT_FLOWER_INPUT) -> bool:
    cache_path = Path(cache_path)
    input_path = Path(input_path)
    if not cache_path.exists() or not input_path.exists():
        return False
    try:
        source_size = str(input_path.stat().st_size)
        source_mtime = str(input_path.stat().st_mtime)
    except FileNotFoundError:
        return False
    conn = sqlite3.connect(cache_path)
    try:
        rows = dict(conn.execute("SELECT key, value FROM cache_meta").fetchall())
    except sqlite3.DatabaseError:
        return False
    finally:
        conn.close()
    return rows.get("input_path") == str(input_path) and rows.get("input_size") == source_size and rows.get("input_mtime") == source_mtime


def ensure_lookup_cache(
    *,
    input_path: Path = DEFAULT_FLOWER_INPUT,
    cache_path: Path = DEFAULT_LOOKUP_CACHE,
) -> Path:
    if lookup_cache_is_current(cache_path=cache_path, input_path=input_path):
        return Path(cache_path)
    return build_lookup_cache(input_path=input_path, cache_path=cache_path, force=True)


def load_mechanism_reactions(
    mechanism_id: int,
    *,
    input_path: Path = DEFAULT_FLOWER_INPUT,
    cache_path: Path = DEFAULT_LOOKUP_CACHE,
) -> List[str]:
    """Load ordered non-trivial mapped reactions for one mechanism id."""

    cache_path = ensure_lookup_cache(input_path=input_path, cache_path=cache_path)
    conn = sqlite3.connect(cache_path)
    try:
        rows = conn.execute(
            """
            SELECT offset, length
            FROM mechanism_rows
            WHERE mechanism_id = ?
            ORDER BY row_order ASC
            """,
            (int(mechanism_id),),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return []

    mapped_reactions: List[str] = []
    with Path(input_path).open("rb") as handle:
        for offset, length in rows:
            handle.seek(int(offset))
            text = handle.read(int(length)).decode("utf-8")
            parsed = _parse_line(text)
            if parsed is None:
                continue
            mapped_reactions.append(parsed.mapped_reaction)
    return mapped_reactions


def convert_mechanism_id_to_case(
    mechanism_id: int,
    *,
    input_path: Path = DEFAULT_FLOWER_INPUT,
    cache_path: Path = DEFAULT_LOOKUP_CACHE,
) -> Dict[str, Any]:
    mapped_reactions = load_mechanism_reactions(mechanism_id, input_path=input_path, cache_path=cache_path)
    if not mapped_reactions:
        raise ConversionError("missing_candidate_group")
    return _convert_group(int(mechanism_id), mapped_reactions)


def build_curriculum_index(
    *,
    input_path: Path = DEFAULT_FLOWER_INPUT,
) -> Tuple[List[CurriculumIndexEntry], Dict[str, Any]]:
    """Scan FlowER train.txt and produce a deterministic ranked index."""

    input_path = Path(input_path)
    total_lines = 0
    numeric_rows = 0
    trivial_rows = 0
    invalid_rows = 0
    step_counts: Counter[int] = Counter()
    first_state_by_mechanism: Dict[int, List[str]] = {}

    with input_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            total_lines += 1
            parsed = _parse_line(raw_line)
            if parsed is None:
                continue
            numeric_rows += 1
            try:
                left, right = _split_reaction_sides(parsed.mapped_reaction)
            except ConversionError:
                invalid_rows += 1
                continue
            if _unique_preserving_order(_tokenize_species(left)) == _unique_preserving_order(_tokenize_species(right)):
                trivial_rows += 1
                continue
            step_counts[parsed.mechanism_id] += 1
            if parsed.mechanism_id not in first_state_by_mechanism:
                first_state_by_mechanism[parsed.mechanism_id] = _unique_preserving_order(_tokenize_species(left))

    ranked: List[Tuple[Tuple[int, int, int, int, int, int], CurriculumIndexEntry]] = []
    for mechanism_id in sorted(step_counts):
        metrics = metrics_for_species(first_state_by_mechanism.get(mechanism_id, []))
        rank_key = [
            int(step_counts[mechanism_id]),
            int(metrics["reagent_count"]),
            int(metrics["total_heavy_atoms"]),
            int(metrics["max_reagent_heavy_atoms"]),
            int(metrics["total_hetero_atoms"]),
            int(mechanism_id),
        ]
        ranked.append(
            (
                tuple(rank_key),
                CurriculumIndexEntry(
                    mechanism_id=int(mechanism_id),
                    case_id=f"flower_{mechanism_id:06d}",
                    step_count=int(step_counts[mechanism_id]),
                    reagent_count=int(metrics["reagent_count"]),
                    total_heavy_atoms=int(metrics["total_heavy_atoms"]),
                    max_reagent_heavy_atoms=int(metrics["max_reagent_heavy_atoms"]),
                    total_hetero_atoms=int(metrics["total_hetero_atoms"]),
                    global_rank=0,
                    rank_within_step_count=0,
                    rank_key=list(rank_key),
                ),
            )
        )

    ranked.sort(key=lambda item: item[0])
    per_step_rank: Counter[int] = Counter()
    entries: List[CurriculumIndexEntry] = []
    for global_rank, (_key, entry) in enumerate(ranked, start=1):
        per_step_rank[entry.step_count] += 1
        entries.append(
            CurriculumIndexEntry(
                mechanism_id=entry.mechanism_id,
                case_id=entry.case_id,
                step_count=entry.step_count,
                reagent_count=entry.reagent_count,
                total_heavy_atoms=entry.total_heavy_atoms,
                max_reagent_heavy_atoms=entry.max_reagent_heavy_atoms,
                total_hetero_atoms=entry.total_hetero_atoms,
                global_rank=global_rank,
                rank_within_step_count=int(per_step_rank[entry.step_count]),
                rank_key=list(entry.rank_key),
            )
        )

    step_distribution = Counter(entry.step_count for entry in entries)
    report = {
        "input_path": str(input_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_lines_read": total_lines,
        "numeric_id_rows": numeric_rows,
        "invalid_rows_skipped": invalid_rows,
        "trivial_rows_skipped": trivial_rows,
        "total_grouped_mechanisms_discovered": len(entries),
        "step_count_distribution": {str(key): step_distribution[key] for key in sorted(step_distribution)},
        "ranking_policy": {
            "rank_key": [
                "step_count",
                "reagent_count",
                "total_heavy_atoms",
                "max_reagent_heavy_atoms",
                "total_hetero_atoms",
                "mechanism_id",
            ],
            "complexity_inputs": [
                "unique starting-state species from the first non-trivial step",
                "atom maps stripped before uniqueness and atom counting",
            ],
        },
    }
    return entries, report


def write_curriculum_index(
    entries: Sequence[CurriculumIndexEntry],
    *,
    output_path: Path = DEFAULT_INDEX_PATH,
    report_path: Optional[Path] = None,
    report: Optional[Dict[str, Any]] = None,
) -> None:
    _write_jsonl((entry.as_dict() for entry in entries), Path(output_path))
    if report_path is not None and report is not None:
        _json_dump(report, Path(report_path))


def load_curriculum_index(index_path: Path = DEFAULT_INDEX_PATH) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(index_path).open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def build_ranked_dataset(
    *,
    input_path: Path = DEFAULT_FLOWER_INPUT,
    cache_path: Path = DEFAULT_LOOKUP_CACHE,
    index_entries: Sequence[Dict[str, Any]] | Sequence[CurriculumIndexEntry],
    sample_size: int = 100,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Convert the lowest-ranked mechanisms until sample_size successes are written."""

    dataset: List[Dict[str, Any]] = []
    conversion_failures = Counter()
    skipped_ids: List[str] = []
    selected_ids: List[str] = []
    attempted = 0

    for raw_entry in index_entries:
        entry = raw_entry.as_dict() if isinstance(raw_entry, CurriculumIndexEntry) else dict(raw_entry)
        attempted += 1
        mechanism_id = int(entry["mechanism_id"])
        case_id = str(entry["case_id"])
        try:
            case = convert_mechanism_id_to_case(mechanism_id, input_path=input_path, cache_path=cache_path)
        except ConversionError as exc:
            conversion_failures[exc.reason] += 1
            skipped_ids.append(case_id)
            continue
        dataset.append(case)
        selected_ids.append(case_id)
        if len(dataset) >= int(sample_size):
            break

    if len(dataset) < int(sample_size):
        raise SystemExit(
            f"Unable to build {sample_size} FlowER mechanisms; only {len(dataset)} converted successfully."
        )

    step_distribution = Counter(int(item.get("n_mechanistic_steps") or 0) for item in dataset)
    report = {
        "input_path": str(input_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_size_requested": int(sample_size),
        "sample_size_written": len(dataset),
        "candidate_attempt_count": attempted,
        "selected_case_ids": selected_ids,
        "skipped_case_ids_before_cutoff": skipped_ids,
        "conversion_failures_by_reason": dict(sorted(conversion_failures.items())),
        "step_count_distribution_sampled_set": {str(key): step_distribution[key] for key in sorted(step_distribution)},
        "selection_policy": "lowest-ranked successfully converted mechanisms from flower_mechanism_index.jsonl",
    }
    return dataset, report


def build_stratified_dataset(
    *,
    input_path: Path = DEFAULT_FLOWER_INPUT,
    cache_path: Path = DEFAULT_LOOKUP_CACHE,
    index_entries: Sequence[Dict[str, Any]] | Sequence[CurriculumIndexEntry],
    per_step: int = 20,
    max_step: int = 8,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Sample per_step mechanisms from each step-count tier 1..max_step.

    Unlike build_ranked_dataset(), which takes the lowest-ranked mechanisms
    across all step counts, this function samples from each tier independently
    so the resulting dataset spans the full complexity range.  Tiers with fewer
    than one successful conversion are silently skipped rather than failing.
    """
    # Group entries by step_count
    by_step: Dict[int, List[Dict[str, Any]]] = {}
    for raw_entry in index_entries:
        entry = raw_entry.as_dict() if isinstance(raw_entry, CurriculumIndexEntry) else dict(raw_entry)
        sc = int(entry.get("step_count") or 0)
        by_step.setdefault(sc, []).append(entry)

    dataset: List[Dict[str, Any]] = []
    conversion_failures: Counter[str] = Counter()
    selected_ids: List[str] = []
    skipped_ids: List[str] = []
    tier_summary: Dict[str, Any] = {}

    for step in range(1, int(max_step) + 1):
        tier_entries = by_step.get(step, [])
        tier_selected: List[str] = []
        tier_skipped: List[str] = []
        tier_attempted = 0
        for entry in tier_entries:
            if len(tier_selected) >= int(per_step):
                break
            tier_attempted += 1
            mechanism_id = int(entry["mechanism_id"])
            case_id = str(entry["case_id"])
            try:
                case = convert_mechanism_id_to_case(mechanism_id, input_path=input_path, cache_path=cache_path)
            except ConversionError as exc:
                conversion_failures[exc.reason] += 1
                tier_skipped.append(case_id)
                continue
            dataset.append(case)
            tier_selected.append(case_id)

        selected_ids.extend(tier_selected)
        skipped_ids.extend(tier_skipped)
        tier_summary[str(step)] = {
            "attempted": tier_attempted,
            "selected": len(tier_selected),
            "skipped": len(tier_skipped),
            "available_in_index": len(tier_entries),
        }

    step_distribution = Counter(int(item.get("n_mechanistic_steps") or 0) for item in dataset)
    report = {
        "input_path": str(input_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "per_step_requested": int(per_step),
        "max_step": int(max_step),
        "total_selected": len(dataset),
        "selected_case_ids": selected_ids,
        "skipped_case_ids": skipped_ids,
        "conversion_failures_by_reason": dict(sorted(conversion_failures.items())),
        "step_count_distribution_sampled_set": {str(key): step_distribution[key] for key in sorted(step_distribution)},
        "tier_summary": tier_summary,
        "selection_policy": "stratified: per_step lowest-ranked conversions per step-count tier",
    }
    return dataset, report


def known_mechanism_from_case(case: Dict[str, Any]) -> Dict[str, Any]:
    steps = list(((case.get("verified_mechanism") or {}).get("steps") or []))
    known_steps: List[Dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        resulting = _unique_preserving_order(step.get("resulting_state") or [])
        target = resulting[0] if resulting else str(step.get("predicted_intermediate") or "").strip()
        if not target:
            continue
        known_steps.append(
            {
                "step_index": int(step.get("step_index") or len(known_steps) + 1),
                "target_smiles": target,
            }
        )
    min_steps = int(case.get("n_mechanistic_steps") or len(known_steps) or 0)
    return {
        "source": SOURCE_LABEL,
        "min_steps": min_steps,
        "citation": "FlowER curriculum benchmark",
        "steps": known_steps,
    }


def eval_case_from_case(case: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(case)
    out["source"] = SOURCE_LABEL
    out["known_mechanism"] = known_mechanism_from_case(case)
    out["n_mechanistic_steps"] = int(
        out.get("n_mechanistic_steps")
        or len((((out.get("verified_mechanism") or {}).get("steps")) or []))
        or 0
    )
    return out


def ensure_index(
    *,
    input_path: Path = DEFAULT_FLOWER_INPUT,
    index_path: Path = DEFAULT_INDEX_PATH,
    report_path: Path = DEFAULT_INDEX_REPORT_PATH,
) -> Path:
    index_path = Path(index_path)
    report_path = Path(report_path)
    if index_path.exists() and report_path.exists():
        return index_path
    entries, report = build_curriculum_index(input_path=input_path)
    write_curriculum_index(entries, output_path=index_path, report_path=report_path, report=report)
    return index_path


def curriculum_history(
    store: Any,
    *,
    model_name: str,
    harness: str,
    curriculum_index_path: Path,
) -> Dict[str, Any]:
    """Summarise curriculum attempts for a model+harness+index scope."""

    attempted_case_ids: set[str] = set()
    passed_case_ids: set[str] = set()
    pass_count_by_step: Dict[int, int] = defaultdict(int)
    step_pass_target_by_step: Dict[int, int] = {}
    successful_batches: List[Dict[str, Any]] = []
    batch_history: List[Dict[str, Any]] = []
    index_path_text = str(Path(curriculum_index_path))

    for run in store.list_eval_runs():
        if str(run.get("model_name") or "") != str(model_name):
            continue
        run_group = str(run.get("run_group_name") or "")
        if not run_group.startswith("curriculum_"):
            continue
        results = store.list_eval_run_results(str(run.get("id") or ""))
        matched: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for result in results:
            summary = result.get("summary") or {}
            if not isinstance(summary, dict):
                continue
            if str(summary.get("curriculum_harness") or "") != str(harness):
                continue
            if str(summary.get("curriculum_index_path") or "") != index_path_text:
                continue
            case_id = str(summary.get("curriculum_case_id") or "")
            if not case_id:
                continue
            matched.append((result, summary))
            attempted_case_ids.add(case_id)
            step_count = int(summary.get("step_count") or 0)
            step_pass_target = int(summary.get("step_pass_target") or 50)
            if step_count > 0:
                step_pass_target_by_step[step_count] = max(step_pass_target_by_step.get(step_count, 0), step_pass_target)
            if result.get("pass_bool") is True:
                passed_case_ids.add(case_id)
                pass_count_by_step[step_count] += 1
        if not matched:
            continue

        first_summary = matched[0][1]
        step_count = int(first_summary.get("step_count") or 0)
        group_size = int(first_summary.get("group_size") or len(matched))
        promotion_threshold = int(first_summary.get("promotion_threshold") or max(1, group_size))
        batch_start_rank = int(first_summary.get("batch_start_rank") or first_summary.get("global_rank") or 0)
        pass_count = sum(1 for result, _summary in matched if result.get("pass_bool") is True)
        batch = {
            "eval_run_id": str(run.get("id") or ""),
            "run_group_name": run_group,
            "step_count": step_count,
            "group_size": group_size,
            "promotion_threshold": promotion_threshold,
            "batch_start_rank": batch_start_rank,
            "pass_count": pass_count,
            "attempted_case_ids": [str(summary.get("curriculum_case_id") or "") for _result, summary in matched],
            "successful": pass_count >= promotion_threshold,
        }
        batch_history.append(batch)
        if batch["successful"]:
            successful_batches.append(batch)

    completed_steps = [
        step_count
        for step_count, pass_count in pass_count_by_step.items()
        if pass_count >= max(1, int(step_pass_target_by_step.get(step_count, 50)))
    ]
    highest_successful_step_count = max(completed_steps, default=max((batch["step_count"] for batch in successful_batches), default=None))
    return {
        "attempted_case_ids": attempted_case_ids,
        "passed_case_ids": passed_case_ids,
        "pass_count_by_step": dict(sorted(pass_count_by_step.items())),
        "highest_successful_step_count": highest_successful_step_count,
        "successful_batches": successful_batches,
        "batch_history": batch_history,
    }


def current_curriculum_step_count(
    index_entries: Sequence[Dict[str, Any]] | Sequence[CurriculumIndexEntry],
    *,
    pass_count_by_step: Dict[int, int] | Dict[str, int] | None = None,
    required_passes_per_step: int = 50,
) -> Optional[int]:
    step_counts = sorted({int((entry.as_dict() if isinstance(entry, CurriculumIndexEntry) else entry)["step_count"]) for entry in index_entries})
    if not step_counts:
        return None
    normalized_pass_counts = {
        int(step_count): int(count)
        for step_count, count in dict(pass_count_by_step or {}).items()
    }
    threshold = max(1, int(required_passes_per_step))
    for step_count in step_counts:
        if normalized_pass_counts.get(step_count, 0) < threshold:
            return step_count
    return None


def next_curriculum_candidates(
    index_entries: Sequence[Dict[str, Any]] | Sequence[CurriculumIndexEntry],
    *,
    attempted_case_ids: Iterable[str],
    pass_count_by_step: Dict[int, int] | Dict[str, int] | None = None,
    required_passes_per_step: int = 50,
) -> Dict[str, Any]:
    attempted = set(str(item) for item in attempted_case_ids)
    normalized_pass_counts = {
        int(step_count): int(count)
        for step_count, count in dict(pass_count_by_step or {}).items()
    }
    current_step_count = current_curriculum_step_count(
        index_entries,
        pass_count_by_step=normalized_pass_counts,
        required_passes_per_step=required_passes_per_step,
    )
    candidates: List[Dict[str, Any]] = []
    for raw_entry in index_entries:
        entry = raw_entry.as_dict() if isinstance(raw_entry, CurriculumIndexEntry) else dict(raw_entry)
        if current_step_count is None or int(entry["step_count"]) != current_step_count:
            continue
        if str(entry["case_id"]) in attempted:
            continue
        candidates.append(entry)
    start_rank = int(candidates[0]["global_rank"]) if candidates else None
    return {
        "current_step_count": current_step_count,
        "candidates": candidates,
        "batch_start_rank": start_rank,
        "step_pass_target": max(1, int(required_passes_per_step)),
        "current_step_pass_count": int(normalized_pass_counts.get(current_step_count or 0, 0)) if current_step_count is not None else 0,
    }


__all__ = [
    "ConversionError",
    "CurriculumIndexEntry",
    "DEFAULT_DATASET_PATH",
    "DEFAULT_DATASET_REPORT_PATH",
    "DEFAULT_FLOWER_INPUT",
    "DEFAULT_INDEX_PATH",
    "DEFAULT_INDEX_REPORT_PATH",
    "DEFAULT_LOOKUP_CACHE",
    "SOURCE_LABEL",
    "build_curriculum_index",
    "build_lookup_cache",
    "build_ranked_dataset",
    "build_stratified_dataset",
    "convert_elementary_step",
    "convert_mechanism_id_to_case",
    "current_curriculum_step_count",
    "curriculum_history",
    "ensure_index",
    "ensure_lookup_cache",
    "eval_case_from_case",
    "known_mechanism_from_case",
    "load_curriculum_index",
    "load_mechanism_reactions",
    "lookup_cache_is_current",
    "metrics_for_species",
    "next_curriculum_candidates",
    "normalize_electron_pushes",
    "write_curriculum_index",
]
