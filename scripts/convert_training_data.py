#!/usr/bin/env python3
"""Convert reaction SMILES from an Excel/CSV file into train/test JSON sets.

Default input:  training_data/41597_2024_3709_MOESM4_ESM.xlsx
Default output: training_data/train_set.json
                training_data/test_set.json

Usage
-----
# Convert the bundled dataset (80/20 stratified split, seed 42)
    python scripts/convert_training_data.py

# Custom split fraction or random seed
    python scripts/convert_training_data.py --train-frac 0.9 --seed 123

# Point at a different Excel or CSV file
    python scripts/convert_training_data.py --excel my_reactions.xlsx

# Override column names (defaults shown)
    python scripts/convert_training_data.py \\
        --reaction-col updated_reaction \\
        --class-col mechanistic_class \\
        --steps-col n_mechanistic_steps \\
        --score-col mean

Input file requirements
-----------------------
Excel (.xlsx) or CSV (.csv) with at least one column containing reaction SMILES
strings in the format  ``reactants>>products``.  Atom-mapping labels such as
``[Cl:101]`` are automatically stripped and canonical SMILES are produced via
RDKit.  All other columns are optional.

Output format
-------------
Each entry in the output JSON files is compatible with the
``data/mechanism_examples.json`` schema consumed by ``GET /api/examples``:

    {
      "id":                 "sn2_001",
      "name":               "SN2 (1)",
      "description":        "SN2 reaction. Expert consensus: 0.95 | Steps: 2",
      "starting_materials": ["CCl", "CN"],
      "products":           ["CCN"],
      "temperature_celsius": null,
      "ph":                 null,
      "source":             "train_set",       # injected at load time by app.py
      "mechanistic_class":  "SN2",
      "n_mechanistic_steps": 2,
      "expert_consensus":   0.95
    }
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Attempt RDKit import for canonical SMILES; fall back to regex strip only
# ---------------------------------------------------------------------------
try:
    from rdkit import Chem as _Chem  # type: ignore[import-not-found]
    _RDKIT = True
except ImportError:
    _RDKIT = False


def _strip_atom_map_regex(smiles: str) -> str:
    """Remove atom-map numbers (:N) inside square brackets using regex."""
    return re.sub(r":([0-9]+)\]", "]", smiles)


def _canonical(smiles: str) -> str:
    """Return canonical SMILES.  Falls back to regex-stripped SMILES if RDKit unavailable."""
    stripped = _strip_atom_map_regex(smiles)
    if not _RDKIT:
        return stripped
    mol = _Chem.MolFromSmiles(stripped)
    if mol is None:
        return stripped  # keep regex result if RDKit can't parse
    return _Chem.MolToSmiles(mol)


def _parse_reaction(rxn_smiles: str) -> Tuple[List[str], List[str]]:
    """Split ``reactants>>products`` SMILES into canonical starting materials and products."""
    parts = rxn_smiles.split(">>")
    if len(parts) != 2:
        raise ValueError(f"Expected exactly one '>>' separator, got: {rxn_smiles!r}")
    reactants = [_canonical(s) for s in parts[0].split(".") if s.strip()]
    products  = [_canonical(s) for s in parts[1].split(".") if s.strip()]
    return reactants, products


# ---------------------------------------------------------------------------
# ID / name helpers
# ---------------------------------------------------------------------------
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _class_slug(cls: str) -> str:
    """Return a filesystem-safe slug from a mechanistic class string (≤ 28 chars)."""
    slug = _SLUG_RE.sub("_", str(cls).lower()).strip("_")
    return slug[:28]


def _build_entries(
    rows: List[Dict[str, Any]],
    reaction_col: str,
    class_col: Optional[str],
    steps_col: Optional[str],
    score_col: Optional[str],
) -> List[Dict[str, Any]]:
    """Convert raw DataFrame rows to the JSON entry schema."""
    counters: Dict[str, int] = defaultdict(int)
    entries: List[Dict[str, Any]] = []

    for row in rows:
        rxn_smiles = str(row.get(reaction_col, "") or "").strip()
        if not rxn_smiles or ">>" not in rxn_smiles:
            continue

        try:
            starting_materials, products = _parse_reaction(rxn_smiles)
        except ValueError as exc:
            print(f"  [warn] skipping row — {exc}", file=sys.stderr)
            continue

        # ---- class / name / id ----
        raw_class = str(row.get(class_col, "unknown") or "unknown").strip() if class_col else "unknown"
        slug = _class_slug(raw_class)
        counters[slug] += 1
        n = counters[slug]
        entry_id   = f"{slug}_{n:03d}"
        entry_name = f"{raw_class} ({n})"

        # ---- optional metadata ----
        steps: Any = row.get(steps_col) if steps_col else None
        score: Any = row.get(score_col)  if score_col else None

        desc_parts = [raw_class]
        if score is not None and score == score:  # NaN guard
            desc_parts.append(f"Expert consensus: {float(score):.2f}")
        if steps is not None and steps == steps:
            desc_parts.append(f"Steps: {int(steps)}")
        description = " | ".join(desc_parts) + "."

        entry: Dict[str, Any] = {
            "id":                  entry_id,
            "name":                entry_name,
            "description":         description,
            "starting_materials":  starting_materials,
            "products":            products,
            "temperature_celsius": None,
            "ph":                  None,
            "mechanistic_class":   raw_class,
        }
        if steps is not None and steps == steps:
            entry["n_mechanistic_steps"] = int(steps)
        if score is not None and score == score:
            entry["expert_consensus"] = round(float(score), 4)

        entries.append(entry)

    return entries


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def _stratified_split(
    entries: List[Dict[str, Any]],
    train_frac: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split entries into train/test by mechanistic_class, then shuffle each set."""
    rng = random.Random(seed)

    by_class: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in entries:
        by_class[e.get("mechanistic_class", "unknown")].append(e)

    train: List[Dict[str, Any]] = []
    test:  List[Dict[str, Any]] = []

    for cls, group in sorted(by_class.items()):
        rng.shuffle(group)
        n_train = max(1, round(len(group) * train_frac))
        if len(group) == 1:
            # Only one example → goes to train so class isn't absent from training
            train.extend(group)
        else:
            train.extend(group[:n_train])
            test.extend(group[n_train:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    default_excel = project_root / "training_data" / "41597_2024_3709_MOESM4_ESM.xlsx"
    default_out   = project_root / "training_data"

    parser = argparse.ArgumentParser(
        description="Convert reaction SMILES from Excel/CSV to train/test JSON sets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--excel",        default=str(default_excel), help="Path to input Excel or CSV file")
    parser.add_argument("--out-dir",      default=str(default_out),   help="Output directory for train/test JSON files")
    parser.add_argument("--train-frac",   type=float, default=0.8,    help="Fraction of data used for training (default: 0.8)")
    parser.add_argument("--seed",         type=int,   default=42,      help="Random seed for reproducible splits (default: 42)")
    parser.add_argument("--reaction-col", default="updated_reaction",       help="Column containing reaction SMILES (default: updated_reaction)")
    parser.add_argument("--class-col",    default="mechanistic_class",      help="Column for reaction class label (default: mechanistic_class)")
    parser.add_argument("--steps-col",    default="n_mechanistic_steps",    help="Column for mechanistic step count (default: n_mechanistic_steps)")
    parser.add_argument("--score-col",    default="mean",                   help="Column for expert consensus score (default: mean)")
    args = parser.parse_args()

    # ---- read input ----
    in_path = Path(args.excel)
    if not in_path.exists():
        sys.exit(f"Input file not found: {in_path}")

    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ImportError:
        sys.exit(
            "pandas is required.  Install it with:\n"
            "  pip install pandas openpyxl\n"
            "(activate your virtual environment first)"
        )

    suffix = in_path.suffix.lower()
    print(f"Reading {in_path} …")
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(in_path)
    elif suffix == ".csv":
        df = pd.read_csv(in_path)
    else:
        sys.exit(f"Unsupported file type '{suffix}'. Use .xlsx, .xls, or .csv.")

    print(f"  {len(df)} rows, columns: {df.columns.tolist()}")

    # ---- validate required column ----
    if args.reaction_col not in df.columns:
        sys.exit(
            f"Reaction column '{args.reaction_col}' not found.\n"
            f"Available columns: {df.columns.tolist()}\n"
            f"Use --reaction-col to specify the correct column name."
        )

    # Resolve optional columns (use None if absent so _build_entries skips them)
    class_col = args.class_col if args.class_col in df.columns else None
    steps_col = args.steps_col if args.steps_col in df.columns else None
    score_col = args.score_col if args.score_col in df.columns else None

    if class_col is None:
        print(f"  [info] class column '{args.class_col}' not found — IDs will use 'unknown'")
    if steps_col is None:
        print(f"  [info] steps column '{args.steps_col}' not found — step counts omitted")
    if score_col is None:
        print(f"  [info] score column '{args.score_col}' not found — consensus scores omitted")

    rows = df.to_dict(orient="records")

    # ---- build entries ----
    print("Parsing reactions …")
    entries = _build_entries(rows, args.reaction_col, class_col, steps_col, score_col)
    print(f"  {len(entries)} valid reactions parsed")

    if not entries:
        sys.exit("No valid reactions found.  Check --reaction-col and file contents.")

    # ---- split ----
    train, test = _stratified_split(entries, args.train_frac, args.seed)
    print(f"  train: {len(train)}  |  test: {len(test)}")

    # ---- write output ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train_set.json"
    test_path  = out_dir / "test_set.json"

    train_path.write_text(json.dumps(train, indent=2, ensure_ascii=False), encoding="utf-8")
    test_path.write_text( json.dumps(test,  indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n✅  Written:")
    print(f"   {train_path}  ({len(train)} reactions)")
    print(f"   {test_path}   ({len(test)} reactions)")
    if not _RDKIT:
        print(
            "\n⚠  RDKit not found — SMILES were atom-map-stripped only (no canonicalisation).\n"
            "   Install RDKit for fully canonical output:\n"
            "     conda install -c conda-forge rdkit\n"
            "   or: pip install rdkit"
        )


if __name__ == "__main__":
    main()
