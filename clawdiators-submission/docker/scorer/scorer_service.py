"""
Mechanistic-Easy scorer service for Clawdiators.

Ground truth is baked into this image at build time (COPY ground_truth.json /app/).
No network calls, no runtime external dependencies — fully deterministic.

Endpoints:
  GET  /health          -> {"ok": true}
  POST /canonicalize    -> canonicalize a list of SMILES strings
  POST /score           -> full scoring with SMILES canonicalization + post-submission validation

The /score endpoint is the primary endpoint used by the TypeScript ChallengeModule.
It returns dimension scores (0-1000) matching the CHALLENGE.md scoring table.

Submission notes:
  - `steps` is required and must contain 10 reaction entries.
  - Each step entry should be objects with `resulting_state` and `electron_pushes`.
  - Invalid or missing step content degrades score but only shape errors return HTTP 400.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rdkit import Chem

from scoring_utils import (
    canonicalize_list,
    canonicalize_smiles,
    exact_match_ratio,
    time_decay,
)

# -- Load ground truth at startup (baked into image) --------------------------------
GROUND_TRUTH_PATH = os.environ.get("GROUND_TRUTH_PATH", "/app/ground_truth.json")

try:
    with open(GROUND_TRUTH_PATH) as f:
        _GT_DATA = json.load(f)
    GROUND_TRUTH_REACTIONS: List[Dict[str, Any]] = _GT_DATA.get("reactions", [])
    if len(GROUND_TRUTH_REACTIONS) != 10:
        raise ValueError(
            f"Expected 10 ground truth reactions, got {len(GROUND_TRUTH_REACTIONS)}"
        )
except FileNotFoundError as exc:
    raise RuntimeError(
        f"Ground truth file not found at {GROUND_TRUTH_PATH}. "
        "Was the Docker image built with 'COPY ground_truth.json /app/'?"
    ) from exc

TIME_LIMIT_SECS = 400.0
NUM_REACTIONS = 10

app = FastAPI(
    title="Mechanistic-Easy Scorer",
    description="Clawdiators scorer for organic mechanism prediction (easy tier)",
    version="1.1.0",
)


# -- Request/Response models ---------------------------------------------------------


class CanonicalizeRequest(BaseModel):
    smiles: List[str]


class CanonicalizeResponse(BaseModel):
    canonical: List[Optional[str]]


class ScoreRequest(BaseModel):
    submission: Dict[str, Any]
    started_at: Optional[str] = None
    submitted_at: Optional[str] = None


class ScoreResponse(BaseModel):
    breakdown: Dict[str, int]
    details: Dict[str, Any]


# -- Shared helpers -----------------------------------------------------------------


def _parse_smiles(smi: str) -> Optional[Chem.Mol]:
    try:
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None


def _atom_counts(smiles_list: List[str]) -> Counter:
    counts: Counter = Counter()
    for smi in smiles_list:
        mol = _parse_smiles(smi)
        if mol is None:
            continue
        mol_h = Chem.AddHs(mol)
        for atom in mol_h.GetAtoms():
            counts[atom.GetSymbol()] += 1
    return counts


def _total_charge(smiles_list: List[str]) -> int:
    total = 0
    for smi in smiles_list:
        mol = _parse_smiles(smi)
        if mol is not None:
            total += Chem.GetFormalCharge(mol)
    return total


def _validate_smiles_list(smiles_list: List[str]) -> Tuple[bool, List[str]]:
    invalid = [s for s in smiles_list if _parse_smiles(s) is None]
    return len(invalid) == 0, invalid


def _validate_step(from_smiles: List[str], to_smiles: List[str]) -> Dict[str, Any]:
    from_valid, from_invalid = _validate_smiles_list(from_smiles)
    to_valid, to_invalid = _validate_smiles_list(to_smiles)
    all_smiles_valid = from_valid and to_valid

    warnings: List[str] = []
    if from_invalid:
        warnings.append(f"Invalid from_smiles: {from_invalid}")
    if to_invalid:
        warnings.append(f"Invalid to_smiles: {to_invalid}")

    from_atoms = _atom_counts(from_smiles)
    to_atoms = _atom_counts(to_smiles)
    all_keys = set(from_atoms) | set(to_atoms)
    atom_imbalance = {
        atom: to_atoms.get(atom, 0) - from_atoms.get(atom, 0)
        for atom in all_keys
        if to_atoms.get(atom, 0) != from_atoms.get(atom, 0)
    }
    atom_balance = len(atom_imbalance) == 0
    if not atom_balance:
        warnings.append(f"Atom imbalance: {atom_imbalance}")

    from_charge = _total_charge(from_smiles)
    to_charge = _total_charge(to_smiles)
    charge_balance = from_charge == to_charge
    charge_imbalance = to_charge - from_charge
    if not charge_balance:
        warnings.append(f"Charge imbalance: from={from_charge}, to={to_charge}")

    return {
        "all_smiles_valid": all_smiles_valid,
        "atom_balance": atom_balance,
        "charge_balance": charge_balance,
        "valid": all_smiles_valid and atom_balance and charge_balance,
        "atom_imbalance": atom_imbalance or None,
        "charge_imbalance": charge_imbalance if not charge_balance else None,
        "warnings": warnings,
    }


def _normalize_product_smiles(smi: str) -> str:
    if not smi or not isinstance(smi, str):
        return ""
    return ".".join(sorted(part.strip() for part in smi.split(".") if part.strip()))


def _extract_push_type(push: str) -> Optional[str]:
    if not push or not isinstance(push, str):
        return None
    left = push.split(":", 1)[0].strip().lower()
    if left in {"lp", "sigma", "pi"}:
        return left
    return None


def _extract_push_types(pushes: Any) -> List[str]:
    if not isinstance(pushes, list):
        return []
    types: List[str] = []
    for push in pushes:
        push_type = _extract_push_type(push)
        if push_type is not None:
            types.append(push_type)
    return types


def _type_jaccard(a: List[str], b: List[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    ca = Counter(a)
    cb = Counter(b)
    keys = set(ca) | set(cb)

    intersection = sum(min(ca[k], cb[k]) for k in keys)
    union = sum(max(ca[k], cb[k]) for k in keys)
    return 1.0 if union == 0 else intersection / union


def _normalize_smiles_set(smiles_list: List[str]) -> set[str]:
    out: set[str] = set()
    for smi in smiles_list:
        can = canonicalize_smiles(smi)
        if can is not None:
            out.add(can)
    return out


def _set_jaccard(a: List[str], b: List[str]) -> float:
    sa = _normalize_smiles_set(a)
    sb = _normalize_smiles_set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


def _safe_steps_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [step for step in value if isinstance(step, dict)]


def _score_mechanism(
    submitted_steps: List[Any],
    ground_truth_reactions: List[Dict[str, Any]],
) -> Tuple[float, List[Dict[str, Any]]]:
    scores: List[float] = []
    breakdown: List[Dict[str, Any]] = []

    for i, gt in enumerate(ground_truth_reactions):
        gt_steps = _safe_steps_list(gt.get("steps", []))
        sub_steps = _safe_steps_list(submitted_steps[i] if i < len(submitted_steps) else [])

        gt_step_count = len(gt_steps)
        sub_step_count = len(sub_steps)

        if sub_step_count == gt_step_count:
            step_count_score = 1.0
        elif abs(sub_step_count - gt_step_count) == 1:
            step_count_score = 0.5
        else:
            step_count_score = 0.0

        gt_intermediates = [
            smi
            for step in gt_steps[:-1]
            for smi in (step.get("resulting_state") or [])
            if isinstance(smi, str)
        ]
        sub_intermediates = [
            smi
            for step in sub_steps[:-1]
            for smi in (step.get("resulting_state") or [])
            if isinstance(smi, str)
        ]

        intermediate_jaccard = _set_jaccard(sub_intermediates, gt_intermediates)
        reaction_score = (step_count_score + intermediate_jaccard) / 2.0

        scores.append(reaction_score)
        breakdown.append(
            {
                "index": i,
                "score": round(reaction_score, 4),
                "step_count_score": step_count_score,
                "intermediate_jaccard": round(intermediate_jaccard, 4),
                "submitted_step_count": sub_step_count,
                "expected_step_count": gt_step_count,
            }
        )

    return (sum(scores) / len(scores), breakdown) if scores else (0.0, [])


def _score_electron_pushes(
    submitted_steps: List[Any],
    ground_truth_reactions: List[Dict[str, Any]],
) -> Tuple[float, List[Dict[str, Any]]]:
    reaction_scores: List[float] = []
    breakdown: List[Dict[str, Any]] = []

    for i, gt in enumerate(ground_truth_reactions):
        gt_steps = _safe_steps_list(gt.get("steps", []))
        sub_steps = _safe_steps_list(submitted_steps[i] if i < len(submitted_steps) else [])

        compared_steps = max(len(gt_steps), len(sub_steps), 1)
        step_scores: List[float] = []

        for j in range(compared_steps):
            gt_step = gt_steps[j] if j < len(gt_steps) else None
            sub_step = sub_steps[j] if j < len(sub_steps) else None

            if gt_step is None or sub_step is None:
                step_scores.append(0.0)
                continue

            gt_types = _extract_push_types(gt_step.get("electron_pushes"))
            sub_types = _extract_push_types(sub_step.get("electron_pushes"))
            step_scores.append(_type_jaccard(sub_types, gt_types))

        rxn_score = sum(step_scores) / compared_steps
        reaction_scores.append(rxn_score)
        breakdown.append(
            {
                "index": i,
                "score": round(rxn_score, 4),
                "step_scores": [round(x, 4) for x in step_scores],
                "submitted_step_count": len(sub_steps),
                "expected_step_count": len(gt_steps),
            }
        )

    return (sum(reaction_scores) / len(reaction_scores), breakdown) if reaction_scores else (0.0, [])


def _post_submission_validation(
    final_products: List[str],
    steps: List[Any],
    ground_truth_reactions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Run chemistry validation across final_products and submitted mechanism steps.

    For each reaction:
      - Validate submitted final product SMILES parseability.
      - Validate each step.resulting_state SMILES parseability.
      - Validate atom/charge balance on available transitions.
    """
    per_reaction: List[Dict[str, Any]] = []
    overall_valid = True

    for i, gt in enumerate(ground_truth_reactions):
        starting_materials = gt.get("starting_materials") or []
        submitted_product = final_products[i] if i < len(final_products) else ""
        submitted_steps = _safe_steps_list(steps[i] if i < len(steps) else [])

        reaction_result: Dict[str, Any] = {
            "index": i,
            "final_product_smiles_valid": False,
            "final_product_canonical": None,
            "step_smiles_valid": True,
            "invalid_step_smiles": [],
            "step_validations": [],
            "warnings": [],
        }

        canonical = canonicalize_smiles(submitted_product)
        reaction_result["final_product_smiles_valid"] = canonical is not None
        reaction_result["final_product_canonical"] = canonical
        if canonical is None:
            reaction_result["warnings"].append(
                f"final_products[{i}] is not a valid SMILES string: {repr(submitted_product)}"
            )
            overall_valid = False

        invalid_step_smiles: List[str] = []
        for step in submitted_steps:
            resulting_state = step.get("resulting_state") if isinstance(step, dict) else []
            if not isinstance(resulting_state, list):
                continue
            for smi in resulting_state:
                if not isinstance(smi, str) or _parse_smiles(smi) is None:
                    invalid_step_smiles.append(str(smi))

        reaction_result["invalid_step_smiles"] = invalid_step_smiles
        reaction_result["step_smiles_valid"] = len(invalid_step_smiles) == 0
        if invalid_step_smiles:
            reaction_result["warnings"].append(
                f"steps[{i}] contains invalid resulting_state SMILES: {invalid_step_smiles}"
            )
            overall_valid = False

        step_validations: List[Dict[str, Any]] = []
        prev_state: Optional[List[str]] = starting_materials if starting_materials else None

        for j, step in enumerate(submitted_steps):
            to_state = step.get("resulting_state") if isinstance(step, dict) else None
            if not isinstance(to_state, list):
                continue
            if prev_state is None:
                prev_state = to_state
                continue
            sv = _validate_step(prev_state, to_state)
            step_validations.append({"step": j, "description": "transition", **sv})
            if not sv["valid"]:
                overall_valid = False
            prev_state = to_state

        if prev_state is not None and canonical is not None:
            final_as_list = [frag for frag in submitted_product.split(".") if frag]
            if final_as_list:
                sv_final = _validate_step(prev_state, final_as_list)
                step_validations.append({"step": "final", "description": "last state -> final_products", **sv_final})
                if not sv_final["valid"]:
                    overall_valid = False

        reaction_result["step_validations"] = step_validations
        per_reaction.append(reaction_result)

    return {
        "overall_valid": overall_valid,
        "per_reaction": per_reaction,
        "total_reactions": len(ground_truth_reactions),
        "valid_final_products": sum(1 for r in per_reaction if r["final_product_smiles_valid"]),
        "valid_steps": sum(1 for r in per_reaction if r["step_smiles_valid"]),
    }


# -- Endpoints ----------------------------------------------------------------------


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "reactions_loaded": len(GROUND_TRUTH_REACTIONS),
        "time_limit_secs": TIME_LIMIT_SECS,
    }


@app.post("/canonicalize", response_model=CanonicalizeResponse)
def canonicalize(request: CanonicalizeRequest) -> CanonicalizeResponse:
    return CanonicalizeResponse(canonical=canonicalize_list(request.smiles))


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    """
    Score a submission against ground truth.

    Expected submission format (inner object, without the 'answer' wrapper):
    {
      "final_products": ["SMILES0", ..., "SMILES9"],
      "steps": [
        [{"resulting_state": ["..."], "electron_pushes": ["lp:...", ...]}],
        ... 10 total entries ...
      ],
      "methodology": "string"
    }
    """
    submission = request.submission

    final_products = submission.get("final_products")
    steps = submission.get("steps")
    methodology = submission.get("methodology")

    if not isinstance(final_products, list):
        raise HTTPException(
            status_code=400,
            detail="submission.final_products must be an array of 10 SMILES strings",
        )
    if len(final_products) != NUM_REACTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"submission.final_products must have exactly 10 items, got {len(final_products)}",
        )

    if steps is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "submission.steps is required. "
                "Submit an array of 10 arrays (one per reaction)."
            ),
        )
    if not isinstance(steps, list):
        raise HTTPException(
            status_code=400,
            detail="submission.steps must be an array of 10 arrays (one per reaction)",
        )
    if len(steps) != NUM_REACTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"submission.steps must have exactly 10 items, got {len(steps)}",
        )

    product_score_raw, product_breakdown = exact_match_ratio(
        final_products, GROUND_TRUTH_REACTIONS
    )
    product_score = round(product_score_raw * 300)

    has_correct_product = product_score_raw > 0

    if has_correct_product:
        pathway_score_raw, pathway_breakdown = _score_mechanism(
            steps, GROUND_TRUTH_REACTIONS
        )
        electron_push_raw, electron_push_breakdown = _score_electron_pushes(
            steps, GROUND_TRUTH_REACTIONS
        )
    else:
        pathway_score_raw, pathway_breakdown = 0.0, []
        electron_push_raw, electron_push_breakdown = 0.0, []

    pathway_score = round(pathway_score_raw * 300)
    electron_push_score = round(electron_push_raw * 200)

    speed_score = 0
    elapsed_secs: Optional[float] = None
    if has_correct_product and request.started_at and request.submitted_at:
        try:
            t_start = datetime.fromisoformat(request.started_at.replace("Z", "+00:00"))
            t_submit = datetime.fromisoformat(request.submitted_at.replace("Z", "+00:00"))
            elapsed_secs = (t_submit - t_start).total_seconds()
            speed_score = round(time_decay(elapsed_secs, TIME_LIMIT_SECS) * 100)
        except Exception:
            speed_score = 0

    methodology_score = 100 if isinstance(methodology, str) and methodology.strip() else 0

    total = (
        product_score
        + pathway_score
        + electron_push_score
        + speed_score
        + methodology_score
    )

    post_validation = _post_submission_validation(
        final_products, steps, GROUND_TRUTH_REACTIONS
    )

    return ScoreResponse(
        breakdown={
            "product_accuracy": product_score,
            "pathway_coverage": pathway_score,
            "electron_push_quality": electron_push_score,
            "speed": speed_score,
            "methodology": methodology_score,
            "total": total,
        },
        details={
            "product_breakdown": product_breakdown,
            "pathway_breakdown": pathway_breakdown,
            "electron_push_breakdown": electron_push_breakdown,
            "elapsed_secs": elapsed_secs,
            "product_score_raw": round(product_score_raw, 4),
            "pathway_score_raw": round(pathway_score_raw, 4),
            "electron_push_score_raw": round(electron_push_raw, 4),
            "post_submission_validation": post_validation,
        },
    )
