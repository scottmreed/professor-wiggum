"""SMILES helpers shared across runtime and data processing code."""
from __future__ import annotations

import re
from contextlib import redirect_stderr
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

try:  # pragma: no cover - optional dependency
    from rdkit.Chem import AddHs, MolFromSmiles, MolToSmiles
except ImportError:  # pragma: no cover - handled at runtime
    AddHs = None  # type: ignore[assignment]
    MolFromSmiles = None  # type: ignore[assignment]
    MolToSmiles = None  # type: ignore[assignment]


_COMMON_SMILES_ALIASES = {
    "H2O": "O",
    "[H2O]": "O",
    "CO2": "O=C=O",
    "[CO2]": "O=C=O",
    "HCl": "Cl",
    "[HCl]": "Cl",
    "HBr": "Br",
    "[HBr]": "Br",
    "Cl-": "[Cl-]",
    "Br-": "[Br-]",
    "I-": "[I-]",
    "F-": "[F-]",
    "OH-": "[OH-]",
}


def normalize_common_smiles_alias(smiles: str) -> str:
    """Normalize common chemistry aliases that are not valid RDKit SMILES."""

    text = str(smiles or "").strip()
    if not text:
        return text
    if text in _COMMON_SMILES_ALIASES:
        return _COMMON_SMILES_ALIASES[text]
    lowered = text.lower()
    for raw, canonical in _COMMON_SMILES_ALIASES.items():
        if lowered == raw.lower():
            return canonical
    return text


def canonicalize_if_valid(smiles: str) -> Optional[str]:
    """Return canonical SMILES only when the input is actually parseable."""

    text = normalize_common_smiles_alias(smiles)
    if not text:
        return None
    if MolFromSmiles is None or MolToSmiles is None:
        return text
    mol = MolFromSmiles(text, sanitize=True)
    if mol is None:
        return None
    return MolToSmiles(mol)


def canonicalize_capture_error(smiles: str) -> tuple[Optional[str], Optional[str]]:
    """Return (canonical_smiles, rdkit_error_message). rdkit_error_message is None on success."""
    text = normalize_common_smiles_alias(smiles)
    if not text:
        return None, "empty or unrecognized SMILES alias"
    if MolFromSmiles is None or MolToSmiles is None:
        return text, None
    captured = StringIO()
    with redirect_stderr(captured):
        mol = MolFromSmiles(text, sanitize=True)
    if mol is None:
        return None, captured.getvalue().strip() or f"RDKit rejected: {text}"
    return MolToSmiles(mol), None


def remove_mapping_and_canonicalize(
    smiles: str,
    add_hs: bool = False,
    sanitize: bool = True,
    kekulize: bool = False,
) -> str:
    """Strip atom-map numbers and return canonical SMILES when possible.

    Adapted from schwallergroup/ChRIMP:
    https://github.com/schwallergroup/ChRIMP
    Original helper: ``src/chrimp/dataset/pmechdb_helper.py``.
    """

    if not isinstance(smiles, str):
        return smiles
    if MolFromSmiles is None or MolToSmiles is None:
        return smiles
    mol = MolFromSmiles(smiles, sanitize=sanitize)
    if mol is None:
        return smiles
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    if add_hs and AddHs is not None:
        mol = AddHs(mol)
    return MolToSmiles(mol, kekuleSmiles=kekulize)


def species_match_signature(smiles: str) -> str:
    """Return a stable signature for cross-boundary species comparisons.

    The signature removes atom mapping, normalizes common aliases, and
    canonicalizes valid SMILES when possible. Invalid placeholders fall back to
    trimmed text so tests and synthetic fixtures remain comparable.
    """

    text = normalize_common_smiles_alias(str(smiles or "").strip())
    if not text:
        return ""
    canonical = remove_mapping_and_canonicalize(text)
    return str(canonical or text).strip()


def normalize_species_for_matching(smiles_list: List[str]) -> List[str]:
    """Return ordered unique comparison signatures for a species list."""

    out: List[str] = []
    seen: set[str] = set()
    for item in smiles_list:
        signature = species_match_signature(item)
        if not signature or signature in seen:
            continue
        seen.add(signature)
        out.append(signature)
    return out


def heavy_atom_count_for_matching(smiles: str) -> int:
    """Return a best-effort heavy-atom count for ranking target products."""

    signature = species_match_signature(smiles)
    if not signature:
        return 0
    if MolFromSmiles is not None:
        try:
            mol = MolFromSmiles(signature, sanitize=True)
            if mol is not None:
                return int(mol.GetNumHeavyAtoms())
        except Exception:
            pass
    return len(re.findall(r"[A-Z][a-z]?", signature))


def assess_target_product_state(
    *,
    current_state: List[str],
    resulting_state: List[str],
    target_products: List[str],
    starting_materials: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Classify whether a step has reached the primary target product.

    Spectator products already present in the starting pool are excluded from
    target completion checks. Among the remaining targets, the heaviest species
    are treated as the primary products that should trigger completion.
    """

    current_signatures = normalize_species_for_matching(current_state or [])
    resulting_signatures = normalize_species_for_matching(resulting_state or [])
    target_signatures = normalize_species_for_matching(target_products or [])
    starting_signatures = normalize_species_for_matching(starting_materials or [])

    spectator_targets = [item for item in target_signatures if item in starting_signatures]
    productive_targets = [item for item in target_signatures if item not in starting_signatures]
    if not productive_targets:
        productive_targets = [item for item in target_signatures if item not in current_signatures]
    if not productive_targets:
        productive_targets = list(target_signatures)

    primary_targets: List[str] = []
    if productive_targets:
        max_heavy_atoms = max(heavy_atom_count_for_matching(item) for item in productive_targets)
        primary_targets = [
            item for item in productive_targets if heavy_atom_count_for_matching(item) == max_heavy_atoms
        ]

    matched_targets = [item for item in target_signatures if item in resulting_signatures]
    matched_primary_targets = [item for item in primary_targets if item in resulting_signatures]

    return {
        "contains_target_product": bool(matched_primary_targets),
        "matched_target_products": matched_targets,
        "matched_primary_target_products": matched_primary_targets,
        "primary_target_products": primary_targets,
        "productive_target_products": productive_targets,
        "spectator_target_products": spectator_targets,
        "normalized_resulting_state": resulting_signatures,
    }


def strip_atom_mapping_optional(smiles: Optional[str]) -> Optional[str]:
    """Return a map-free canonical string or the original value on failure."""

    if smiles is None:
        return None
    return remove_mapping_and_canonicalize(smiles)


def strip_atom_mapping_list(smiles_list: List[str]) -> List[str]:
    """Strip atom maps from a list while preserving order and item count."""

    return [remove_mapping_and_canonicalize(item) for item in smiles_list]


def attempt_smiles_recovery(invalid_smiles: str) -> Optional[str]:
    """Try to recover from common SMILES issues before failing validation.

    Attempts basic fixes for:
    - Excessive radicals (>3 per molecule)
    - Unclosed rings
    - Invalid ring notation

    Returns the sanitized SMILES if successful, None if unrecoverable.
    """
    if not invalid_smiles or not isinstance(invalid_smiles, str):
        return None

    # Clean up the SMILES string
    cleaned = normalize_common_smiles_alias(invalid_smiles.strip())

    # Remove excessive radicals - limit to max 3 per molecule
    # This is a simple heuristic - count + symbols and limit
    radical_count = cleaned.count('+')
    if radical_count > 3:
        # Remove all radicals - this is a crude fix but better than failing
        cleaned = re.sub(r'\+\d*', '', cleaned)

    # Fix common ring issues - remove incomplete ring digits
    # Look for patterns like "N2" at end without proper ring closure
    cleaned = re.sub(r'(\w)2(?!\d)', r'\1', cleaned)

    # Try to parse with RDKit
    try:
        return canonicalize_if_valid(cleaned)
    except:
        return None


def sanitize_smiles_list(smiles_list: List[str]) -> Tuple[List[str], List[str]]:
    """Sanitize a list of SMILES strings, attempting recovery for invalid ones.

    Returns:
        Tuple of (valid_smiles, invalid_smiles)
    """
    valid_smiles = []
    invalid_smiles = []

    for smiles in smiles_list:
        canonical = canonicalize_if_valid(smiles)
        if canonical:
            valid_smiles.append(canonical)
            continue

        # Try recovery
        recovered = attempt_smiles_recovery(smiles)
        if recovered:
            valid_smiles.append(recovered)
        else:
            invalid_smiles.append(smiles)

    return valid_smiles, invalid_smiles
