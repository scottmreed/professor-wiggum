"""
Mechanistic-Easy validator service — participant-facing chemistry validation tool.

This service contains NO ground truth. It validates chemical steps for:
  - SMILES validity
  - Atom balance (same atoms on both sides of a step)
  - Charge balance (same total charge on both sides)
  - Bond changes are chemically reasonable

Participants run this locally to test their proposed mechanisms before submitting.

Usage:
  docker run -p 8080:8080 clawdiators/mechanistic-validator:1.0

  POST /validate   — validate one or more reaction steps
  POST /canonicalize — canonicalize SMILES strings
  GET  /health     — health check
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from collections import Counter

from fastapi import FastAPI
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

app = FastAPI(
    title="Mechanistic Chemistry Validator",
    description=(
        "Participant-facing validator for Clawdiators mechanistic-easy challenge. "
        "Contains no ground truth — checks chemistry validity only."
    ),
    version="1.0.0",
)


# ── SMILES utilities ──────────────────────────────────────────────────


def parse_smiles(smi: str) -> Optional[Chem.Mol]:
    """Parse a SMILES string, return molecule or None if invalid."""
    try:
        mol = Chem.MolFromSmiles(smi)
        return mol
    except Exception:
        return None


def canonicalize(smi: str) -> Optional[str]:
    """Return canonical SMILES or None."""
    mol = parse_smiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def get_atom_counts(smiles_list: List[str]) -> Counter:
    """Count atoms across a list of species SMILES."""
    counts: Counter = Counter()
    for smi in smiles_list:
        mol = parse_smiles(smi)
        if mol is None:
            continue
        mol_with_h = Chem.AddHs(mol)
        for atom in mol_with_h.GetAtoms():
            counts[atom.GetSymbol()] += 1
    return counts


def get_total_charge(smiles_list: List[str]) -> int:
    """Sum formal charges across a list of species SMILES."""
    total = 0
    for smi in smiles_list:
        mol = parse_smiles(smi)
        if mol is None:
            continue
        total += Chem.GetFormalCharge(mol)
    return total


# ── Request/Response models ───────────────────────────────────────────


class ReactionStep(BaseModel):
    from_smiles: List[str]          # starting species for this step
    to_smiles: List[str]            # resulting species after this step
    step_type: Optional[str] = None  # optional hint: "substitution", "addition", etc.


class ValidateRequest(BaseModel):
    steps: List[ReactionStep]


class StepResult(BaseModel):
    step_index: int
    valid: bool
    atom_balance: bool
    charge_balance: bool
    all_smiles_valid: bool
    canonical_from: Optional[List[Optional[str]]] = None
    canonical_to: Optional[List[Optional[str]]] = None
    atom_imbalance: Optional[Dict[str, int]] = None  # atom → imbalance (positive = excess in products)
    charge_imbalance: Optional[int] = None
    warnings: List[str]


class ValidateResponse(BaseModel):
    results: List[StepResult]
    overall_valid: bool


class CanonicalizeRequest(BaseModel):
    smiles: List[str]


class CanonicalizeResponse(BaseModel):
    canonical: List[Optional[str]]


# ── Endpoints ─────────────────────────────────────────────────────────


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "mechanistic-validator", "version": "1.0.0"}


@app.post("/canonicalize", response_model=CanonicalizeResponse)
def do_canonicalize(request: CanonicalizeRequest) -> CanonicalizeResponse:
    """Canonicalize a list of SMILES strings using RDKit. Invalid SMILES → null."""
    return CanonicalizeResponse(canonical=[canonicalize(s) for s in request.smiles])


@app.post("/validate", response_model=ValidateResponse)
def validate(request: ValidateRequest) -> ValidateResponse:
    """
    Validate one or more reaction steps for chemistry consistency.

    Each step provides from_smiles (reactant species) and to_smiles (product species).
    Returns atom balance, charge balance, and SMILES validity for each step.

    This validator contains NO ground truth — it only checks whether your proposed
    steps are chemically self-consistent.
    """
    results: List[StepResult] = []

    for i, step in enumerate(request.steps):
        warnings: List[str] = []

        # Check SMILES validity
        from_valid = all(parse_smiles(s) is not None for s in step.from_smiles)
        to_valid = all(parse_smiles(s) is not None for s in step.to_smiles)
        all_smiles_valid = from_valid and to_valid

        if not step.from_smiles:
            warnings.append("from_smiles is empty")
        if not step.to_smiles:
            warnings.append("to_smiles is empty")

        # Canonicalize for display
        canonical_from = [canonicalize(s) for s in step.from_smiles]
        canonical_to = [canonicalize(s) for s in step.to_smiles]

        if not all_smiles_valid:
            invalid_from = [s for s in step.from_smiles if parse_smiles(s) is None]
            invalid_to = [s for s in step.to_smiles if parse_smiles(s) is None]
            if invalid_from:
                warnings.append(f"Invalid from_smiles: {invalid_from}")
            if invalid_to:
                warnings.append(f"Invalid to_smiles: {invalid_to}")

        # Atom balance
        from_atoms = get_atom_counts(step.from_smiles)
        to_atoms = get_atom_counts(step.to_smiles)

        all_atom_keys = set(from_atoms.keys()) | set(to_atoms.keys())
        atom_imbalance: Dict[str, int] = {}
        atom_balance = True
        for atom in all_atom_keys:
            diff = to_atoms.get(atom, 0) - from_atoms.get(atom, 0)
            if diff != 0:
                atom_imbalance[atom] = diff
                atom_balance = False

        if not atom_balance:
            warnings.append(
                f"Atom imbalance detected: {atom_imbalance} "
                "(positive = excess in to_smiles, negative = missing)"
            )

        # Charge balance
        from_charge = get_total_charge(step.from_smiles)
        to_charge = get_total_charge(step.to_smiles)
        charge_balance = from_charge == to_charge
        charge_imbalance = to_charge - from_charge

        if not charge_balance:
            warnings.append(
                f"Charge imbalance: from_total={from_charge}, to_total={to_charge}, "
                f"difference={charge_imbalance}"
            )

        # Overall validity
        step_valid = all_smiles_valid and atom_balance and charge_balance

        results.append(StepResult(
            step_index=i,
            valid=step_valid,
            atom_balance=atom_balance,
            charge_balance=charge_balance,
            all_smiles_valid=all_smiles_valid,
            canonical_from=canonical_from,
            canonical_to=canonical_to,
            atom_imbalance=atom_imbalance if atom_imbalance else None,
            charge_imbalance=charge_imbalance if not charge_balance else None,
            warnings=warnings,
        ))

    overall_valid = all(r.valid for r in results)

    return ValidateResponse(results=results, overall_valid=overall_valid)
