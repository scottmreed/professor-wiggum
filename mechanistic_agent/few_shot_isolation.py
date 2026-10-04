"""Keep eval cases out of few-shot files.

A few-shot example that shows an eval case's mechanism hands the model the answer, so every
``skills/mechanistic/**/few_shot.jsonl`` line is checked against the eval cases. A line leaks a
case when it names the case id or contains the case's main product (its largest product, at least
``MIN_HEAVY_ATOMS`` heavy atoms, compared as canonical SMILES with atom maps stripped). Small
products such as water, HCl or a common salt are ignored because they appear in unrelated examples.

Dev tiers: the main products of every ``training_data/eval_tiers.json`` case are committed in
``training_data/eval_tier_main_products.json`` (case id -> SMILES, nothing else), because the
medium/hard structures live in an untracked file. Rebuild it after growing the tiers with
``python -m mechanistic_agent.few_shot_isolation --write-index``. The official holdout is checked
only when the data checkout is present; its structures are never committed.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from . import data_paths

MIN_HEAVY_ATOMS = 10
INDEX_FILE = "eval_tier_main_products.json"
TIER_CASE_FILES = ("eval_set.json", "flower_mechanisms_100.json", "flower_mechanisms_multistep.json")
_CASE_ID_RE = re.compile(r"\bflower_(?:test_)?\d+\b")
_SMILES_TOKEN_RE = re.compile(r"[A-Za-z0-9@+\-\[\]()=#\\/%.:]{" + str(MIN_HEAVY_ATOMS) + ",}")


@dataclass(frozen=True)
class FewShotLeak:
    path: str
    line: int
    case_ids: tuple
    reason: str


def _canonical(smiles: str) -> Optional[str]:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def _heavy_atoms(smiles: str) -> int:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumHeavyAtoms() if mol is not None else 0


def main_product(products: Iterable[str]) -> Optional[str]:
    canon = [c for c in (_canonical(str(p)) for p in products or []) if c]
    if not canon:
        return None
    best = max(canon, key=_heavy_atoms)
    return best if _heavy_atoms(best) >= MIN_HEAVY_ATOMS else None


def _iter_cases(path: Optional[Path]) -> List[Dict]:
    if path is None:
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(payload, dict):
        payload = payload.get("cases") or payload.get("items") or payload.get("examples") or []
    return [item for item in payload if isinstance(item, dict)]


def tier_case_ids(base_dir: Path) -> Set[str]:
    tiers = json.loads((base_dir / "training_data" / "eval_tiers.json").read_text(encoding="utf-8"))
    return {str(cid) for key, ids in tiers.items() if not key.startswith("_") and isinstance(ids, list) for cid in ids}


def load_tier_index(base_dir: Path) -> Dict[str, Optional[str]]:
    payload = json.loads((base_dir / "training_data" / INDEX_FILE).read_text(encoding="utf-8"))
    return dict(payload.get("main_products") or {})


def build_tier_index(base_dir: Path) -> Dict[str, Optional[str]]:
    """case id -> main product for every tier case, from the local FlowER case files."""
    wanted = tier_case_ids(base_dir)
    found: Dict[str, Optional[str]] = {}
    dirs = [data_paths.repo_training_dir(base_dir), data_paths.bulk_training_dir(base_dir)]
    for name in TIER_CASE_FILES:
        for directory in dirs:
            for case in _iter_cases(directory / name):
                cid = str(case.get("id") or case.get("case_id") or "")
                if cid in wanted and cid not in found:
                    found[cid] = main_product(case.get("products") or [])
    missing = sorted(wanted - set(found))
    if missing:
        raise RuntimeError(f"no structure for tier cases: {missing[:10]}")
    return dict(sorted(found.items()))


def write_tier_index(base_dir: Path) -> Path:
    path = base_dir / "training_data" / INDEX_FILE
    payload = {
        "_meta": {
            "purpose": "Main product (largest, >= %d heavy atoms, else null) of each eval_tiers.json case, used "
            "only to keep eval cases out of few-shot files. Regenerate with "
            "`python -m mechanistic_agent.few_shot_isolation --write-index`." % MIN_HEAVY_ATOMS,
        },
        "main_products": build_tier_index(base_dir),
    }
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return path


def _holdout_products(base_dir: Path) -> Dict[str, Set[str]]:
    try:
        path = data_paths.holdout_eval_set_path(base_dir)
    except Exception:  # noqa: BLE001 - no data checkout configured
        return {}
    index: Dict[str, Set[str]] = {}
    for case in _iter_cases(path if path.exists() else None):
        product = main_product(case.get("products") or [])
        if product:
            index.setdefault(product, set()).add(str(case.get("id") or case.get("case_id") or ""))
    return index


@lru_cache(maxsize=4)
def eval_case_products(base_dir: Path, include_holdout: bool = True) -> Dict[str, frozenset]:
    """Canonical main product -> eval case ids that produce it."""
    index: Dict[str, Set[str]] = {}
    for cid, product in load_tier_index(base_dir).items():
        if product:
            index.setdefault(product, set()).add(cid)
    if include_holdout:
        for product, ids in _holdout_products(base_dir).items():
            index.setdefault(product, set()).update(ids)
    return {smiles: frozenset(ids) for smiles, ids in index.items()}


def find_leaks(line: str, products: Dict[str, frozenset], case_ids: Set[str]) -> Dict[str, Set[str]]:
    """{'case_id': ids named in the line, 'main_product': ids whose main product appears}."""
    named = {cid for cid in _CASE_ID_RE.findall(line) if cid in case_ids}
    matched: Set[str] = set()
    for token in set(_SMILES_TOKEN_RE.findall(line)):
        for part in token.split("."):
            if len(part) >= MIN_HEAVY_ATOMS:
                canon = _canonical(part)
                if canon in products:
                    matched |= products[canon]
    return {"case_id": named, "main_product": matched}


def scan_few_shot_files(base_dir: Path, *, include_holdout: bool = True) -> List[FewShotLeak]:
    products = eval_case_products(base_dir, include_holdout)
    case_ids = tier_case_ids(base_dir) | {cid for ids in products.values() for cid in ids}
    leaks: List[FewShotLeak] = []
    for path in sorted((base_dir / "skills" / "mechanistic").glob("**/few_shot.jsonl")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            for reason, ids in find_leaks(line, products, case_ids).items():
                if ids:
                    leaks.append(FewShotLeak(str(path.relative_to(base_dir)), number, tuple(sorted(ids)), reason))
    return leaks


def strip_leaks(base_dir: Path, *, include_holdout: bool = True) -> Dict[str, int]:
    """Remove every leaking line from the few-shot files; returns lines removed per file."""
    by_file: Dict[str, Set[int]] = {}
    for leak in scan_few_shot_files(base_dir, include_holdout=include_holdout):
        by_file.setdefault(leak.path, set()).add(leak.line)
    for rel, numbers in by_file.items():
        path = base_dir / rel
        lines = path.read_text(encoding="utf-8").splitlines()
        kept = [line for number, line in enumerate(lines, start=1) if number not in numbers]
        path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return {rel: len(numbers) for rel, numbers in sorted(by_file.items())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write-index", action="store_true", help=f"rebuild training_data/{INDEX_FILE}")
    parser.add_argument("--strip", action="store_true", help="delete leaking few-shot lines")
    args = parser.parse_args()
    base = data_paths.repo_root()
    if args.write_index:
        print(f"wrote {write_tier_index(base)}")
    if args.strip:
        for rel, count in strip_leaks(base).items():
            print(f"removed {count} line(s) from {rel}")
    leaks = scan_few_shot_files(base)
    for leak in leaks:
        print(f"{leak.path}:{leak.line} {leak.reason} {', '.join(leak.case_ids)}")
    print(f"{len(leaks)} leak(s)")
    return 1 if leaks else 0


if __name__ == "__main__":
    raise SystemExit(main())
