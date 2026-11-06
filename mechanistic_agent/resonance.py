"""Species identity up to resonance (deterministic, AI-free).

Vendored from ChemIllusion's ``backend/app/services/mechanism_resonance_service.py``
(``resonance_key``, ``same_up_to_resonance``, ``match_species``); this module never
imports ChemIllusion.

:func:`resonance_key` names a species independently of which contributor was
drawn: heavy-atom connectivity, **hydrogens per atom**, net charge and radical
count, with bond orders, per-atom formal charges, aromatic flags, stereo and atom
maps erased. Format: ``"<canonical skeleton>|q=<net>|r=<radicals>"``. Two
resonance forms share it (acetate ``CC(=O)[O-]`` / ``CC([O-])=O``; enolate
``C=C(C)[O-]`` / ``[CH2-]C(C)=O``). Tautomers (an H moved) and constitutional
isomers do not.

How it differs from ``skeleton_alignment``'s key: that key
(``heavy_atom_connectivity_charge_minus_h.v1``) keeps only ``charge - n_H`` per
fragment and no H per atom, so it is deliberately invariant under proton transfer
and tautomerism for path alignment. The resonance key is stricter: a tautomer or a
protonation state is a different key. Both keys stay; neither replaces the other.

Use it only to **add** matches: compare exact canonical SMILES first and fall back
to the resonance key when the exact match fails (:func:`match_species`).
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

try:  # pragma: no cover - RDKit is a hard dependency in practice
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover
    Chem = None  # type: ignore[assignment]

from mechanistic_agent.skeleton_alignment import _parse

RESONANCE_KEY_VERSION = "connectivity_h_per_atom_net_charge_radicals.v1"

_KEY_CACHE: Dict[str, Optional[str]] = {}


def _compute_key(text: str) -> Optional[str]:
    mol = _parse(text)
    if mol is None:
        return None
    try:
        mol = Chem.RemoveHs(mol, sanitize=False)
        mol.UpdatePropertyCache(strict=False)
        net = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        radicals = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
        rw = Chem.RWMol(mol)
        for atom in rw.GetAtoms():
            hydrogens = atom.GetTotalNumHs()
            atom.SetAtomMapNum(0)
            atom.SetFormalCharge(0)
            atom.SetNumRadicalElectrons(0)
            atom.SetIsAromatic(False)
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(hydrogens)
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        for bond in rw.GetBonds():
            bond.SetBondType(Chem.BondType.SINGLE)
            bond.SetIsAromatic(False)
            bond.SetStereo(Chem.BondStereo.STEREONONE)
        skeleton = rw.GetMol()
        skeleton.UpdatePropertyCache(strict=False)
        # allHsExplicit: without it the writer prints an organic-subset atom bare
        # whenever it has *fewer* H than its default valence implies, which drops
        # the H count (acetone's carbonyl C and an alkoxide O both print bare).
        text_key = Chem.MolToSmiles(skeleton, canonical=True, allHsExplicit=True)
    except Exception:  # noqa: BLE001 - an unnameable species has no key
        return None
    return f"{text_key}|q={net}|r={radicals}"


def resonance_key(smiles: Any) -> Optional[str]:
    """A name shared by every resonance form of a species, or ``None`` if unparseable."""
    if Chem is None:
        return None
    text = str(smiles or "").strip()
    if not text:
        return None
    if text in _KEY_CACHE:
        return _KEY_CACHE[text]
    key = _compute_key(text)
    _KEY_CACHE[text] = key
    return key


def same_up_to_resonance(a: Any, b: Any) -> bool:
    """True when ``a`` and ``b`` are the same species, possibly drawn as different contributors."""
    ka = resonance_key(a)
    return ka is not None and ka == resonance_key(b)


def _canonical(smiles: Any) -> Optional[str]:
    if Chem is None:
        return None
    mol = Chem.MolFromSmiles(str(smiles or "").strip())
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def match_species(expected: Iterable[Any], produced: Iterable[Any]) -> Dict[str, Any]:
    """Which expected species appear among ``produced``, exactly or as a resonance form.

    Returns ``{"exact": [...], "resonance": [...], "missing": [...]}`` of the
    expected SMILES. "Exact" compares canonical SMILES (stereo-free, map-free);
    "resonance" is the same species drawn as another contributor.
    """
    produced = list(produced)
    exact_set = {_canonical(p) for p in produced} - {None}
    resonance_set = {resonance_key(p) for p in produced} - {None}
    out: Dict[str, Any] = {"exact": [], "resonance": [], "missing": []}
    for smiles in expected:
        canonical = _canonical(smiles)
        if canonical is not None and canonical in exact_set:
            out["exact"].append(smiles)
        elif resonance_key(smiles) in resonance_set:
            out["resonance"].append(smiles)
        else:
            out["missing"].append(smiles)
    return out


__all__ = ["RESONANCE_KEY_VERSION", "match_species", "resonance_key", "same_up_to_resonance"]
