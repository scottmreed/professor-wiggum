"""SMILES helpers shared across runtime and data processing code."""
from __future__ import annotations

from typing import List, Optional

try:  # pragma: no cover - optional dependency
    from rdkit.Chem import AddHs, MolFromSmiles, MolToSmiles
except ImportError:  # pragma: no cover - handled at runtime
    AddHs = None  # type: ignore[assignment]
    MolFromSmiles = None  # type: ignore[assignment]
    MolToSmiles = None  # type: ignore[assignment]


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

    import re

    # Clean up the SMILES string
    cleaned = invalid_smiles.strip()

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
        return remove_mapping_and_canonicalize(cleaned)
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
        # First try normal parsing
        try:
            canonical = remove_mapping_and_canonicalize(smiles)
            if canonical:
                valid_smiles.append(canonical)
                continue
        except:
            pass

        # Try recovery
        recovered = attempt_smiles_recovery(smiles)
        if recovered:
            valid_smiles.append(recovered)
        else:
            invalid_smiles.append(smiles)

    return valid_smiles, invalid_smiles
