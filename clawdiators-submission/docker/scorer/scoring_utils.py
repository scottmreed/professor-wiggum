"""
SMILES canonicalization and scoring utilities for the mechanistic-easy scorer.

Ported from mechanistic_agent/scoring.py — only the SMILES normalization and
set-comparison logic relevant to Clawdiators scoring dimensions.

Ground truth is baked into the Docker image at build time (see Dockerfile).
No network calls, no external state — deterministic given the same inputs.
"""
from __future__ import annotations

from typing import List, Optional, Tuple
from rdkit import Chem


def canonicalize_smiles(smi: str) -> Optional[str]:
    """Return RDKit-canonical SMILES, or None if invalid."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def canonicalize_list(smiles_list: List[str]) -> List[Optional[str]]:
    """Canonicalize a list of SMILES strings. Invalid SMILES → None."""
    return [canonicalize_smiles(s) for s in smiles_list]


def normalize_product_smiles(smi: str) -> Optional[str]:
    """
    Normalize a submitted product SMILES for comparison.

    Handles dot-joined multi-species (e.g. "CC[N+]1CCCC1.[Cl-]") by
    canonicalizing each fragment individually and sorting them, producing
    a canonical dot-joined string that matches ground truth regardless of
    species order.
    """
    if not smi or not isinstance(smi, str):
        return None
    parts = smi.strip().split(".")
    canonical_parts = []
    for part in parts:
        c = canonicalize_smiles(part)
        if c is None:
            return None  # Any invalid fragment invalidates the whole submission
        canonical_parts.append(c)
    return ".".join(sorted(canonical_parts))


def normalize_ground_truth_product(smiles_list: List[str]) -> Optional[str]:
    """
    Normalize a ground truth product list (list of strings) into a sorted
    dot-joined canonical SMILES string for comparison.
    """
    canonical_parts = []
    for smi in smiles_list:
        c = canonicalize_smiles(smi)
        if c is None:
            return None
        canonical_parts.append(c)
    return ".".join(sorted(canonical_parts))


def exact_match_ratio(
    submitted_products: List[str],
    ground_truth_reactions: list,
) -> Tuple[float, List[dict]]:
    """
    Compute product accuracy: fraction of reactions where submitted final product
    exactly matches ground truth after canonicalization.

    Args:
        submitted_products: list of 10 SMILES strings (one per reaction)
        ground_truth_reactions: list of 10 ground truth dicts with 'final_products' key

    Returns:
        (score 0-1, per-reaction breakdown list)
    """
    n = len(ground_truth_reactions)
    if n == 0:
        return 0.0, []

    breakdown = []
    hits = 0

    for i, gt in enumerate(ground_truth_reactions):
        if i >= len(submitted_products):
            breakdown.append({"index": i, "match": False, "reason": "missing"})
            continue

        submitted_smi = submitted_products[i]
        gt_products = gt.get("final_products", [])

        submitted_norm = normalize_product_smiles(submitted_smi)
        gt_norm = normalize_ground_truth_product(gt_products)

        if submitted_norm is None:
            breakdown.append({
                "index": i,
                "match": False,
                "reason": "invalid_smiles",
                "submitted": submitted_smi,
            })
        elif submitted_norm == gt_norm:
            hits += 1
            breakdown.append({
                "index": i,
                "match": True,
                "submitted_canonical": submitted_norm,
                "expected_canonical": gt_norm,
            })
        else:
            breakdown.append({
                "index": i,
                "match": False,
                "reason": "mismatch",
                "submitted_canonical": submitted_norm,
                "expected_canonical": gt_norm,
            })

    return hits / n, breakdown


def set_overlap_score(
    submitted_intermediates: List[List[str]],
    ground_truth_reactions: list,
) -> Tuple[float, List[dict]]:
    """
    Compute pathway coverage: average Jaccard set overlap between submitted
    intermediates and known intermediates across all reactions.

    For concerted single-step reactions (intermediates = []), correct answer
    is also [] — submitting empty arrays scores 1.0 for those reactions.
    Submitting non-empty arrays for reactions with no known intermediates
    scores 0.0 (Jaccard of set vs empty = 0).

    Args:
        submitted_intermediates: list of 10 lists of SMILES strings
        ground_truth_reactions: list of 10 ground truth dicts with 'intermediates' key

    Returns:
        (average score 0-1, per-reaction breakdown list)
    """
    n = len(ground_truth_reactions)
    if n == 0:
        return 0.0, []

    scores = []
    breakdown = []

    for i, gt in enumerate(ground_truth_reactions):
        if i >= len(submitted_intermediates):
            scores.append(0.0)
            breakdown.append({"index": i, "score": 0.0, "reason": "missing"})
            continue

        sub_list = submitted_intermediates[i]
        if not isinstance(sub_list, list):
            sub_list = []

        gt_list = gt.get("intermediates", [])

        # Canonicalize submitted intermediates (skip invalid)
        sub_canon = set()
        for smi in sub_list:
            c = canonicalize_smiles(smi)
            if c is not None:
                sub_canon.add(c)

        gt_canon = set()
        for smi in gt_list:
            c = canonicalize_smiles(smi)
            if c is not None:
                gt_canon.add(c)

        # Jaccard overlap
        if len(sub_canon) == 0 and len(gt_canon) == 0:
            score = 1.0
        else:
            intersection = len(sub_canon & gt_canon)
            union = len(sub_canon | gt_canon)
            score = intersection / union if union > 0 else 1.0

        scores.append(score)
        breakdown.append({
            "index": i,
            "score": round(score, 4),
            "submitted_count": len(sub_canon),
            "expected_count": len(gt_canon),
            "intersection": len(sub_canon & gt_canon) if sub_canon or gt_canon else 0,
        })

    avg = sum(scores) / n
    return avg, breakdown


def time_decay(elapsed_secs: float, limit_secs: float) -> float:
    """Linear time decay: 1.0 at t=0, 0.0 at t>=limit."""
    if elapsed_secs <= 0:
        return 1.0
    if elapsed_secs >= limit_secs:
        return 0.0
    return 1.0 - elapsed_secs / limit_secs
