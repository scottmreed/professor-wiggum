"""
Local scorer for the clawdiators-test harness (v2 — supports steps + electron_pushes).

Scoring dimensions (1000 pts total):
  - Product Accuracy  : 30%  (300 pts) — exact SMILES match after canonicalization
  - Pathway Coverage  : 30%  (300 pts) — step count + intermediate Jaccard
  - Electron Push Quality: 20% (200 pts) — push type (lp/sigma/pi) Jaccard, partial credit
  - Speed             : 10%  (100 pts) — always full points in local test mode
  - Methodology       : 10%  (100 pts) — non-empty methodology key

Win threshold : 700 / 1000
Draw          : 400–699
Loss          : < 400

Submission format (v2):
    {
      "answer": {
        "final_products": ["SMILES_0", ..., "SMILES_9"],
        "steps": [
          [{"resulting_state": ["SMILES"], "electron_pushes": ["lp:N>M", ...]}],
          ...10 entries...
        ],
        "methodology": "..."
      }
    }

Usage:
    python score_submission.py --submission <path/to/submission.json>
    python score_submission.py --submission results/my_run.json --ground-truth easy/test_ground_truth.json
    python score_submission.py --perfect   # score a perfect submission for calibration

Requires RDKit for SMILES canonicalization. Falls back to raw string comparison
if RDKit is not available (less accurate — install rdkit-pypi for precise results).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional RDKit import
# ---------------------------------------------------------------------------
try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# ---------------------------------------------------------------------------
# SMILES utilities
# ---------------------------------------------------------------------------

def canonicalize_smiles(smi: str) -> Optional[str]:
    """Return RDKit-canonical SMILES, or None if invalid."""
    if not smi or not isinstance(smi, str):
        return None
    if RDKIT_AVAILABLE:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol)
        except Exception:
            return None
    else:
        return smi.strip() or None


def normalize_product_smiles(smi: str) -> Optional[str]:
    """Normalize a submitted product SMILES (handles dot-joined multi-species)."""
    if not smi or not isinstance(smi, str):
        return None
    parts = smi.strip().split(".")
    canonical_parts = []
    for part in parts:
        c = canonicalize_smiles(part)
        if c is None:
            return None
        canonical_parts.append(c)
    return ".".join(sorted(canonical_parts))


def normalize_ground_truth_product(smiles_list: List[str]) -> Optional[str]:
    """Normalize a ground truth product list into sorted canonical form."""
    canonical_parts = []
    for smi in smiles_list:
        c = canonicalize_smiles(smi)
        if c is None:
            return None
        canonical_parts.append(c)
    return ".".join(sorted(canonical_parts))


# ---------------------------------------------------------------------------
# Electron push utilities
# ---------------------------------------------------------------------------

def extract_push_type(notation: str) -> Optional[str]:
    """
    Extract the push type from an electron push notation string.
    "lp:7>1" → "lp", "sigma:1-2>2" → "sigma", "pi:3-4>7" → "pi"
    Returns None if the notation doesn't match expected format.
    """
    if not isinstance(notation, str):
        return None
    m = re.match(r'^(lp|sigma|pi):', notation)
    if m:
        return m.group(1)
    # Also accept descriptive forms like "lp:N>C_electrophile"
    m2 = re.match(r'^(lp|sigma|pi)\b', notation)
    if m2:
        return m2.group(1)
    return None


def extract_push_types(pushes: List[str]) -> List[str]:
    """Extract sorted list of push types from a list of push notations."""
    types = []
    for p in pushes:
        t = extract_push_type(p)
        if t:
            types.append(t)
    return sorted(types)


def type_jaccard(a: List[str], b: List[str]) -> float:
    """
    Jaccard similarity of two type multisets (sorted lists).
    Uses multiset intersection/union.
    """
    if not a and not b:
        return 1.0
    # Count occurrences
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    all_keys = set(ca.keys()) | set(cb.keys())
    intersection = sum(min(ca.get(k, 0), cb.get(k, 0)) for k in all_keys)
    union = sum(max(ca.get(k, 0), cb.get(k, 0)) for k in all_keys)
    return intersection / union if union > 0 else 1.0


# ---------------------------------------------------------------------------
# Scoring dimensions
# ---------------------------------------------------------------------------

def score_product_accuracy(
    submitted_products: List[str],
    ground_truth_reactions: list,
) -> Tuple[float, List[dict]]:
    """Product Accuracy: fraction of reactions with exact SMILES match."""
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
                "index": i, "match": False,
                "reason": "invalid_smiles", "submitted": submitted_smi,
            })
        elif submitted_norm == gt_norm:
            hits += 1
            breakdown.append({
                "index": i, "match": True,
                "submitted_canonical": submitted_norm,
                "expected_canonical": gt_norm,
            })
        else:
            breakdown.append({
                "index": i, "match": False,
                "reason": "mismatch",
                "submitted_canonical": submitted_norm,
                "expected_canonical": gt_norm,
            })

    return hits / n, breakdown


def score_pathway_coverage(
    submitted_steps: List[List[dict]],
    ground_truth_reactions: list,
) -> Tuple[float, List[dict]]:
    """
    Pathway Coverage: step count accuracy + intermediate Jaccard.
    For each reaction:
      - Step count score: 1.0 if correct number of steps, 0.5 if off by 1, 0.0 if off by 2+
      - Intermediate Jaccard: Jaccard overlap of intermediate SMILES (states between first and last step)
    Reaction score = (step_count_score + intermediate_jaccard) / 2
    Returns (average score 0–1, per-reaction breakdown).
    """
    n = len(ground_truth_reactions)
    if n == 0:
        return 0.0, []

    scores = []
    breakdown = []

    for i, gt in enumerate(ground_truth_reactions):
        if i >= len(submitted_steps):
            scores.append(0.0)
            breakdown.append({"index": i, "score": 0.0, "reason": "missing"})
            continue

        sub_rxn_steps = submitted_steps[i]
        if not isinstance(sub_rxn_steps, list):
            sub_rxn_steps = []

        gt_steps = gt.get("steps", [])
        gt_n_steps = len(gt_steps)
        sub_n_steps = len(sub_rxn_steps)

        # Step count score
        step_diff = abs(sub_n_steps - gt_n_steps)
        if step_diff == 0:
            step_score = 1.0
        elif step_diff == 1:
            step_score = 0.5
        else:
            step_score = 0.0

        # Intermediate Jaccard: collect intermediate states (all steps except the last)
        gt_intermediates = set()
        for step in gt_steps[:-1]:  # all but last step = intermediates
            for smi in step.get("resulting_state", []):
                c = canonicalize_smiles(smi)
                if c:
                    gt_intermediates.add(c)

        sub_intermediates = set()
        for step in sub_rxn_steps[:-1]:  # all but last step
            state = step.get("resulting_state", []) if isinstance(step, dict) else []
            for smi in state:
                c = canonicalize_smiles(smi)
                if c:
                    sub_intermediates.add(c)

        if not gt_intermediates and not sub_intermediates:
            inter_jaccard = 1.0  # both correctly have no intermediates
        else:
            intersection = len(gt_intermediates & sub_intermediates)
            union = len(gt_intermediates | sub_intermediates)
            inter_jaccard = intersection / union if union > 0 else 1.0

        rxn_score = (step_score + inter_jaccard) / 2.0
        scores.append(rxn_score)
        breakdown.append({
            "index": i,
            "score": round(rxn_score, 4),
            "step_score": step_score,
            "intermediate_jaccard": round(inter_jaccard, 4),
            "gt_n_steps": gt_n_steps,
            "sub_n_steps": sub_n_steps,
            "gt_intermediates": len(gt_intermediates),
            "sub_intermediates": len(sub_intermediates),
        })

    avg = sum(scores) / n
    return avg, breakdown


def score_electron_push_quality(
    submitted_steps: List[List[dict]],
    ground_truth_reactions: list,
) -> Tuple[float, List[dict]]:
    """
    Electron Push Quality: type Jaccard (lp/sigma/pi) per step, averaged.
    Partial credit for getting push types right even if atom indices wrong.
    Returns (average score 0–1, per-reaction breakdown).
    """
    n = len(ground_truth_reactions)
    if n == 0:
        return 0.0, []

    scores = []
    breakdown = []

    for i, gt in enumerate(ground_truth_reactions):
        if i >= len(submitted_steps):
            scores.append(0.0)
            breakdown.append({"index": i, "score": 0.0, "reason": "missing"})
            continue

        sub_rxn_steps = submitted_steps[i]
        if not isinstance(sub_rxn_steps, list):
            sub_rxn_steps = []

        gt_steps = gt.get("steps", [])
        if not gt_steps:
            scores.append(1.0)
            breakdown.append({"index": i, "score": 1.0, "reason": "no_pushes_expected"})
            continue

        step_scores = []
        for j, gt_step in enumerate(gt_steps):
            gt_push_types = extract_push_types(gt_step.get("electronPushes", gt_step.get("electron_pushes", [])))
            if j < len(sub_rxn_steps):
                sub_step = sub_rxn_steps[j]
                sub_pushes = sub_step.get("electron_pushes", []) if isinstance(sub_step, dict) else []
                sub_push_types = extract_push_types(sub_pushes)
            else:
                sub_push_types = []

            step_score = type_jaccard(sub_push_types, gt_push_types)
            step_scores.append(step_score)

        rxn_score = sum(step_scores) / len(step_scores) if step_scores else 0.0
        scores.append(rxn_score)
        breakdown.append({
            "index": i,
            "score": round(rxn_score, 4),
            "per_step": [round(s, 4) for s in step_scores],
        })

    avg = sum(scores) / n
    return avg, breakdown


def score_methodology(methodology: str) -> float:
    """Methodology: 1.0 if non-empty string, else 0.0."""
    if isinstance(methodology, str) and methodology.strip():
        return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Anti-gaming gate
# ---------------------------------------------------------------------------

def apply_anti_gaming_gate(
    product_score: float,
    pathway_score: float,
    push_score: float,
) -> Tuple[float, float, float]:
    """
    If zero products are correct, pathway, push, and speed scores are zeroed out.
    Returns (gated_pathway, gated_push, speed_ratio).
    Speed is always 1.0 in local mode.
    """
    if product_score == 0.0:
        return 0.0, 0.0, 0.0
    return pathway_score, push_score, 1.0


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score(submission: dict, ground_truth: dict) -> dict:
    """
    Score a submission against ground truth.

    submission format (v2):
        {"answer": {"final_products": [...], "steps": [[...], ...], "methodology": "..."}}

    Returns a score report dict.
    """
    answer = submission.get("answer", submission)
    submitted_products = answer.get("final_products", [])
    submitted_steps = answer.get("steps", [[] for _ in range(10)])
    methodology_text = answer.get("methodology", "")

    # Backward compat: if old "intermediates" format, convert to steps
    if "intermediates" in answer and "steps" not in answer:
        old_intermediates = answer.get("intermediates", [])
        gt_reactions = ground_truth.get("reactions", [])
        submitted_steps = []
        for i in range(len(gt_reactions)):
            inter = old_intermediates[i] if i < len(old_intermediates) else []
            final_prod = submitted_products[i] if i < len(submitted_products) else ""
            if not inter:
                submitted_steps.append([{
                    "resulting_state": final_prod.split(".") if final_prod else [],
                    "electron_pushes": []
                }])
            else:
                submitted_steps.append([
                    {"resulting_state": inter, "electron_pushes": []},
                    {"resulting_state": final_prod.split(".") if final_prod else [], "electron_pushes": []}
                ])

    gt_reactions = ground_truth.get("reactions", [])
    n = len(gt_reactions)

    # Score each dimension
    product_ratio, product_breakdown = score_product_accuracy(submitted_products, gt_reactions)
    pathway_ratio, pathway_breakdown = score_pathway_coverage(submitted_steps, gt_reactions)
    push_ratio, push_breakdown = score_electron_push_quality(submitted_steps, gt_reactions)
    methodology_ratio = score_methodology(methodology_text)

    # Anti-gaming gate
    pathway_gated, push_gated, speed_ratio = apply_anti_gaming_gate(
        product_ratio, pathway_ratio, push_ratio
    )

    # Weighted points (out of 1000)
    product_pts = round(product_ratio * 300)
    pathway_pts = round(pathway_gated * 300)
    push_pts = round(push_gated * 200)
    speed_pts = round(speed_ratio * 100)
    methodology_pts = round(methodology_ratio * 100)
    total_pts = product_pts + pathway_pts + push_pts + speed_pts + methodology_pts

    if total_pts >= 700:
        outcome = "WIN"
    elif total_pts >= 400:
        outcome = "DRAW"
    else:
        outcome = "LOSS"

    return {
        "scores": {
            "product_accuracy": {"ratio": round(product_ratio, 4), "points": product_pts, "weight": "30%"},
            "pathway_coverage": {"ratio": round(pathway_gated, 4), "points": pathway_pts, "weight": "30%"},
            "electron_push_quality": {"ratio": round(push_gated, 4), "points": push_pts, "weight": "20%"},
            "speed": {"ratio": speed_ratio, "points": speed_pts, "weight": "10%", "note": "always full in local mode"},
            "methodology": {"ratio": methodology_ratio, "points": methodology_pts, "weight": "10%"},
        },
        "total": total_pts,
        "outcome": outcome,
        "n_reactions": n,
        "products_correct": sum(1 for r in product_breakdown if r.get("match")),
        "rdkit_available": RDKIT_AVAILABLE,
        "breakdown": {
            "products": product_breakdown,
            "pathway": pathway_breakdown,
            "electron_push": push_breakdown,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_report(result: dict) -> None:
    sep = "─" * 56
    print(sep)
    print(f"  clawdiators-test local scorer (v2)")
    if not result["rdkit_available"]:
        print("  ⚠ RDKit not found — using raw string comparison (less accurate)")
        print("    Install with: pip install rdkit-pypi")
    print(sep)

    s = result["scores"]
    print(f"  Product Accuracy     : {result['products_correct']:2d}/{result['n_reactions']} correct"
          f"  →  {s['product_accuracy']['points']:3d} pts  ({s['product_accuracy']['weight']})")
    print(f"  Pathway Coverage     : avg {s['pathway_coverage']['ratio']:.3f}"
          f"           →  {s['pathway_coverage']['points']:3d} pts  ({s['pathway_coverage']['weight']})")
    print(f"  Electron Push Quality: avg {s['electron_push_quality']['ratio']:.3f}"
          f"           →  {s['electron_push_quality']['points']:3d} pts  ({s['electron_push_quality']['weight']})")
    print(f"  Speed                : local mode (full)"
          f"    →  {s['speed']['points']:3d} pts  ({s['speed']['weight']})")
    print(f"  Methodology          : {'present' if s['methodology']['ratio'] == 1.0 else 'MISSING'}"
          f"              →  {s['methodology']['points']:3d} pts  ({s['methodology']['weight']})")
    print(sep)
    print(f"  TOTAL            : {result['total']:4d} / 1000")
    print(f"  PREDICTED OUTCOME: {result['outcome']}"
          f"  (win ≥700 / draw 400–699 / loss <400)")
    print(sep)

    print("\n  Per-reaction product accuracy:")
    for r in result["breakdown"]["products"]:
        mark = "✓" if r.get("match") else "✗"
        reason = "" if r.get("match") else f"  ({r.get('reason', '')})"
        print(f"    [{mark}] rxn {r['index']}{reason}")
        if not r.get("match") and "submitted_canonical" in r:
            print(f"          submitted : {r['submitted_canonical']}")
            print(f"          expected  : {r.get('expected_canonical', 'n/a')}")

    print("\n  Per-reaction electron push quality:")
    for r in result["breakdown"]["electron_push"]:
        print(f"    rxn {r['index']}: {r.get('score', 0.0):.3f}  {r.get('per_step', r.get('reason', ''))}")
    print()


def make_perfect_submission(ground_truth: dict) -> dict:
    """Build a perfect submission from ground truth for calibration."""
    reactions = ground_truth.get("reactions", [])
    products = []
    steps = []
    for rxn in reactions:
        gt_prods = rxn.get("final_products", [])
        products.append(".".join(gt_prods) if gt_prods else "")
        gt_steps = rxn.get("steps", [])
        rxn_steps = []
        for s in gt_steps:
            rxn_steps.append({
                "resulting_state": s.get("resulting_state", []),
                "electron_pushes": s.get("electron_pushes", []),
            })
        steps.append(rxn_steps)
    return {
        "answer": {
            "final_products": products,
            "steps": steps,
            "methodology": "Perfect calibration submission — ground truth answers.",
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a clawdiators-test submission against ground truth (v2)."
    )
    parser.add_argument("--submission", "-s", type=Path, default=None,
                        help="Path to submission JSON file")
    parser.add_argument("--ground-truth", "-g", type=Path,
                        default=Path(__file__).parent.parent / "easy" / "test_ground_truth.json",
                        help="Path to ground truth JSON (default: easy/test_ground_truth.json)")
    parser.add_argument("--perfect", action="store_true",
                        help="Score a perfect submission (calibration check)")
    parser.add_argument("--json", action="store_true",
                        help="Output full JSON report")
    args = parser.parse_args()

    if not args.ground_truth.exists():
        print(f"Error: ground truth file not found: {args.ground_truth}", file=sys.stderr)
        sys.exit(1)

    with open(args.ground_truth) as f:
        ground_truth = json.load(f)

    if args.perfect:
        submission = make_perfect_submission(ground_truth)
        print("(Scoring a perfect submission for calibration...)\n")
    elif args.submission is not None:
        if not args.submission.exists():
            print(f"Error: submission file not found: {args.submission}", file=sys.stderr)
            sys.exit(1)
        with open(args.submission) as f:
            submission = json.load(f)
    else:
        default_template = Path(__file__).parent / "submission_template.json"
        print(f"No --submission specified. Using template: {default_template}", file=sys.stderr)
        with open(default_template) as f:
            submission = json.load(f)

    result = score(submission, ground_truth)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_report(result)


if __name__ == "__main__":
    main()
