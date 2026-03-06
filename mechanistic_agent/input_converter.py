"""Chemical input format conversion utilities.

Supported conversions:
  - Common name -> SMILES (PubChem PUG REST, optional/graceful)
  - InChI -> SMILES (RDKit)
  - MOL / SDF block -> SMILES (RDKit)
  - SMILES passthrough with canonical validation (RDKit)
"""
from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional

try:
    from rdkit import Chem
    from rdkit.Chem.inchi import MolFromInchi as _rdkit_mol_from_inchi  # type: ignore[import-untyped]
except ImportError:
    Chem = None  # type: ignore[assignment,misc]
    _rdkit_mol_from_inchi = None

logger = logging.getLogger(__name__)

PUBCHEM_TIMEOUT = 6  # seconds


@dataclass
class ConversionResult:
    canonical_smiles: Optional[str]
    input_format: str  # "smiles" | "inchi" | "mol_block" | "name"
    raw_input: str
    success: bool
    error: Optional[str] = None
    pubchem_cid: Optional[int] = None


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def detect_format(raw: str) -> str:
    """Heuristically detect the input format.

    Returns one of: ``"smiles"``, ``"inchi"``, ``"mol_block"``, ``"name"``.
    """
    stripped = raw.strip()
    if stripped.startswith("InChI="):
        return "inchi"
    if "\n" in stripped and ("V2000" in stripped or "V3000" in stripped or "M  END" in stripped):
        return "mol_block"
    # Names typically contain spaces but lack SMILES-specific punctuation.
    if " " in stripped and not any(ch in stripped for ch in ("(", ")", "=", "#", "@", "\\", "/", "[", "]")):
        return "name"
    # Single-word strings that are purely alphabetic (no digits, no SMILES
    # punctuation) are likely common names (e.g. "ethanol", "aspirin").
    if stripped.isalpha() and len(stripped) > 1:
        return "name"
    return "smiles"


# ---------------------------------------------------------------------------
# Individual converters
# ---------------------------------------------------------------------------

def convert_smiles(raw: str) -> ConversionResult:
    """Validate and canonicalise an existing SMILES string."""
    cleaned = raw.strip()
    if Chem is None:
        return ConversionResult(
            canonical_smiles=cleaned,
            input_format="smiles",
            raw_input=raw,
            success=True,
            error="RDKit not available; SMILES returned as-is",
        )
    mol = Chem.MolFromSmiles(cleaned)
    if mol is None:
        return ConversionResult(
            canonical_smiles=None,
            input_format="smiles",
            raw_input=raw,
            success=False,
            error=f"RDKit could not parse SMILES: {raw!r}",
        )
    return ConversionResult(
        canonical_smiles=Chem.MolToSmiles(mol),
        input_format="smiles",
        raw_input=raw,
        success=True,
    )


def convert_inchi(raw: str) -> ConversionResult:
    """Convert an InChI string to canonical SMILES."""
    if Chem is None or _rdkit_mol_from_inchi is None:
        return ConversionResult(
            canonical_smiles=None,
            input_format="inchi",
            raw_input=raw,
            success=False,
            error="RDKit InChI support not available",
        )
    mol = _rdkit_mol_from_inchi(raw.strip())
    if mol is None:
        return ConversionResult(
            canonical_smiles=None,
            input_format="inchi",
            raw_input=raw,
            success=False,
            error=f"RDKit could not parse InChI: {raw!r}",
        )
    return ConversionResult(
        canonical_smiles=Chem.MolToSmiles(mol),
        input_format="inchi",
        raw_input=raw,
        success=True,
    )


def convert_mol_block(raw: str) -> ConversionResult:
    """Convert a MOL/SDF block to canonical SMILES."""
    if Chem is None:
        return ConversionResult(
            canonical_smiles=None,
            input_format="mol_block",
            raw_input=raw[:60] + "...",
            success=False,
            error="RDKit not available",
        )
    mol = Chem.MolFromMolBlock(raw, sanitize=True, removeHs=True)
    if mol is None:
        return ConversionResult(
            canonical_smiles=None,
            input_format="mol_block",
            raw_input=raw[:60] + "...",
            success=False,
            error="RDKit could not parse MOL/SDF block",
        )
    return ConversionResult(
        canonical_smiles=Chem.MolToSmiles(mol),
        input_format="mol_block",
        raw_input=raw[:60] + "...",
        success=True,
    )


def convert_common_name(name: str) -> ConversionResult:
    """Resolve a common chemical name to SMILES via PubChem PUG REST.

    Network-dependent. Returns ``success=False`` without raising on failure.
    """
    encoded = urllib.parse.quote(name.strip())
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{encoded}/property/IsomericSMILES,CID/JSON"
    )
    try:
        with urllib.request.urlopen(url, timeout=PUBCHEM_TIMEOUT) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return ConversionResult(
            canonical_smiles=None,
            input_format="name",
            raw_input=name,
            success=False,
            error=f"PubChem lookup failed: {exc}",
        )

    props = payload.get("PropertyTable", {}).get("Properties", [])
    if not props:
        return ConversionResult(
            canonical_smiles=None,
            input_format="name",
            raw_input=name,
            success=False,
            error=f"PubChem returned no results for: {name!r}",
        )

    entry = props[0]
    smiles = str(entry.get("IsomericSMILES") or "").strip()
    cid = entry.get("CID")
    if not smiles:
        return ConversionResult(
            canonical_smiles=None,
            input_format="name",
            raw_input=name,
            success=False,
            error="PubChem returned entry but SMILES field was empty",
        )

    # Re-canonicalise with RDKit when available.
    if Chem is not None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            smiles = Chem.MolToSmiles(mol)

    return ConversionResult(
        canonical_smiles=smiles,
        input_format="name",
        raw_input=name,
        success=True,
        pubchem_cid=int(cid) if cid is not None else None,
    )


# ---------------------------------------------------------------------------
# Auto-dispatch
# ---------------------------------------------------------------------------

def auto_convert(raw: str) -> ConversionResult:
    """Detect format and dispatch to the appropriate converter."""
    fmt = detect_format(raw)
    if fmt == "inchi":
        return convert_inchi(raw)
    if fmt == "mol_block":
        return convert_mol_block(raw)
    if fmt == "name":
        return convert_common_name(raw)
    return convert_smiles(raw)


def convert_many(inputs: List[str]) -> List[ConversionResult]:
    """Convert a list of mixed-format chemical strings."""
    return [auto_convert(raw) for raw in inputs]


__all__ = [
    "ConversionResult",
    "detect_format",
    "auto_convert",
    "convert_many",
    "convert_smiles",
    "convert_inchi",
    "convert_mol_block",
    "convert_common_name",
]
