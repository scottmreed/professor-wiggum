"""Covalent <-> dative notation bridge for transition-metal species (PRD R1b).

FlowER and most LLM output draw every metal-ligand bond as an ordinary bond:
``[Cl][Pd]([Cl])([P](C)(C)C)[P](C)(C)C``. The house conventions in
:mod:`organometallic_chem.organometallic_smiles` require a dative arrow for a
neutral 2-electron donor (``CP(C)(C)->[Pd]``) and the charge-separated
``[O+]#[C-]->[M]`` form for CO; written covalently, RDKit reads the donor as
over-valent (a radical or phantom-H phosphorane) and the counter would treat an
L donor as an X ligand.

:func:`to_house_convention` rewrites a covalent drawing into house convention so
it can be counted and rendered; :func:`to_covalent` is the inverse used for
runtime states and SMIRKS. Only bonds are rewritten (plus the CO charges), so
atom maps survive. Forms whose L/X reading is genuinely ambiguous are reported
in ``ambiguous`` and callers must abstain instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from rdkit import Chem, rdBase

from .organometallic_smiles import (
    DONOR_NORMAL_VALENCE,
    is_d_block,
    is_transition_metal,
    lint_organometallic_smiles,
)

NOTATION_VERSION = "covalent_dative_bridge.v1"


@dataclass(frozen=True)
class NotationResult:
    smiles: Optional[str]
    #: Human-readable list of rewrites, e.g. ``"P(map 45)->Pd(map 43): L donor"``.
    changes: Tuple[str, ...] = field(default_factory=tuple)
    #: Non-empty when the species cannot be put in house convention unambiguously.
    ambiguous: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.smiles is not None and not self.ambiguous


def _parse(smiles: str) -> Optional[Chem.Mol]:
    blocker = rdBase.BlockLogs()  # noqa: F841
    try:
        mol = Chem.MolFromSmiles(smiles.strip(), sanitize=False)
    except Exception:
        return None
    if mol is None:
        return None
    mol.UpdatePropertyCache(strict=False)
    return mol


def _label(atom: Chem.Atom) -> str:
    tag = atom.GetAtomMapNum()
    return f"{atom.GetSymbol()}({'map ' + str(tag) if tag else 'idx ' + str(atom.GetIdx())})"


def _bond_valence(bond: Chem.Bond, atom: Chem.Atom) -> float:
    if bond.GetBondType() == Chem.BondType.DATIVE:
        return 0.0 if bond.GetBeginAtomIdx() == atom.GetIdx() else 1.0
    if bond.GetBondType() == Chem.BondType.AROMATIC or bond.GetIsAromatic():
        return 1.5
    return float(bond.GetBondTypeAsDouble())


def _written_valence(atom: Chem.Atom) -> float:
    return sum(_bond_valence(b, atom) for b in atom.GetBonds()) + atom.GetTotalNumHs()


def _replace_with_dative(rw: Chem.RWMol, donor_idx: int, metal_idx: int) -> None:
    rw.RemoveBond(donor_idx, metal_idx)
    rw.AddBond(donor_idx, metal_idx, Chem.BondType.DATIVE)


def _to_smiles(mol: Chem.Mol) -> Tuple[Optional[str], Optional[str]]:
    blocker = rdBase.BlockLogs()  # noqa: F841
    copy = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(copy)
    except Exception as exc:  # RDKit raises several sanitize exception types
        return None, str(exc).strip() or type(exc).__name__
    return Chem.MolToSmiles(copy), None


def to_house_convention(smiles: str) -> NotationResult:
    """Rewrite a covalently drawn TM species into house convention v1.

    * a neutral N/P/As/O/S/Se donor whose written valence exceeds its normal
      valence on an ordinary bond to a d-block metal -> ``D->[M]`` (L donor);
    * ``[M]C#O`` -> ``[O+]#[C-]->[M]``;
    * amido, alkoxide, halide, hydride, alkyl and aryl bonds stay X.

    Ambiguous (reported, never guessed): more than one metal, a donor bonded to
    two metals (bridging), two mutually bonded carbons both sigma-bonded to the
    metal (an eta2-alkene drawn as a metallacyclopropane: L vs X2 differ by two
    in the oxidation state), a 3-connected neutral carbene carbon on the metal,
    and anything that still fails the house lint after the rewrite.
    Species without a transition metal are returned unchanged.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return NotationResult(None, (), ("empty SMILES",))
    mol = _parse(smiles)
    if mol is None:
        return NotationResult(None, (), ("RDKit cannot parse it",))
    metals = [a for a in mol.GetAtoms() if is_transition_metal(a)]
    if not metals:
        out, _ = _to_smiles(mol)
        return NotationResult(out or smiles.strip())

    ambiguous: List[str] = []
    changes: List[str] = []
    if len(metals) > 1:
        ambiguous.append(f"{len(metals)} metal atoms (multinuclear species are out of scope)")
    if any(not is_d_block(a.GetAtomicNum()) for a in metals):
        ambiguous.append("f-block metal")

    rw = Chem.RWMol(mol)
    for metal in metals:
        m_idx = metal.GetIdx()
        sigma_carbons = []
        for bond in list(metal.GetBonds()):
            if bond.GetBondType() == Chem.BondType.DATIVE:
                continue
            donor = bond.GetOtherAtom(metal)
            symbol = donor.GetSymbol()
            if sum(1 for n in donor.GetNeighbors() if is_transition_metal(n)) > 1:
                ambiguous.append(f"{_label(donor)} bridges two metals")
                continue
            if symbol in DONOR_NORMAL_VALENCE and bond.GetBondType() == Chem.BondType.SINGLE:
                expected = DONOR_NORMAL_VALENCE[symbol] + donor.GetFormalCharge()
                if _written_valence(donor) > expected:
                    _replace_with_dative(rw, donor.GetIdx(), m_idx)
                    changes.append(f"{_label(donor)}->{_label(metal)}: L donor")
                continue
            if symbol == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
                triple_o = [
                    b.GetOtherAtom(donor)
                    for b in donor.GetBonds()
                    if b.GetBondType() == Chem.BondType.TRIPLE and b.GetOtherAtom(donor).GetSymbol() == "O"
                ]
                if triple_o and donor.GetFormalCharge() == 0 and triple_o[0].GetFormalCharge() == 0:
                    rw.GetAtomWithIdx(donor.GetIdx()).SetFormalCharge(-1)
                    rw.GetAtomWithIdx(triple_o[0].GetIdx()).SetFormalCharge(1)
                    _replace_with_dative(rw, donor.GetIdx(), m_idx)
                    changes.append(f"{_label(donor)}->{_label(metal)}: CO, charge-separated")
                    continue
                heavy_degree = sum(1 for n in donor.GetNeighbors() if n.GetAtomicNum() > 1)
                if (
                    donor.GetFormalCharge() == 0
                    and donor.GetTotalNumHs() == 0
                    and heavy_degree == 3
                    and _written_valence(donor) == 3
                ):
                    ambiguous.append(f"{_label(donor)} is a carbene carbon (L vs X2 reading)")
                sigma_carbons.append(donor.GetIdx())
        for i, a in enumerate(sigma_carbons):
            for b in sigma_carbons[i + 1:]:
                if mol.GetBondBetweenAtoms(a, b) is not None:
                    ambiguous.append(
                        f"{_label(mol.GetAtomWithIdx(a))} and {_label(mol.GetAtomWithIdx(b))} are both "
                        f"sigma-bonded to {_label(metal)} (eta2-alkene vs metallacyclopropane)"
                    )

    converted = rw.GetMol()
    out, error = _to_smiles(converted)
    if out is None:
        return NotationResult(None, tuple(changes), tuple(ambiguous) + (f"cannot sanitize after rewrite: {error}",))
    if not ambiguous:
        ambiguous.extend(lint_organometallic_smiles(out))
    return NotationResult(out, tuple(changes), tuple(ambiguous))


def to_covalent(smiles: str) -> NotationResult:
    """Inverse of :func:`to_house_convention`: dative M-L bonds become ordinary bonds.

    The charge-separated CO form is kept (``[O+]#[C-]-[M]`` does not round-trip
    to ``[M]C#O``, which RDKit cannot sanitize). The result is written without
    sanitization so the over-valent donors survive exactly as Wiggum states and
    FlowER draw them.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return NotationResult(None, (), ("empty SMILES",))
    mol = _parse(smiles)
    if mol is None:
        return NotationResult(None, (), ("RDKit cannot parse it",))
    rw = Chem.RWMol(mol)
    changes: List[str] = []
    for bond in list(mol.GetBonds()):
        if bond.GetBondType() != Chem.BondType.DATIVE:
            continue
        begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
        if not (is_transition_metal(begin) or is_transition_metal(end)):
            continue
        rw.RemoveBond(begin.GetIdx(), end.GetIdx())
        rw.AddBond(begin.GetIdx(), end.GetIdx(), Chem.BondType.SINGLE)
        changes.append(f"{_label(begin)}-{_label(end)}: dative -> single")
    return NotationResult(canonical_unsanitized(rw.GetMol()), tuple(changes))


def canonical_unsanitized(mol_or_smiles) -> Optional[str]:
    """Canonical SMILES of a species as written (no valence model applied).

    Used to compare covalent drawings, which RDKit cannot sanitize cleanly
    (over-valent donors), across a round trip.
    """
    mol = _parse(mol_or_smiles) if isinstance(mol_or_smiles, str) else Chem.Mol(mol_or_smiles)
    if mol is None:
        return None
    blocker = rdBase.BlockLogs()  # noqa: F841
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    try:
        Chem.SetAromaticity(mol)
    except Exception:
        pass
    return Chem.MolToSmiles(mol)


__all__ = [
    "NOTATION_VERSION",
    "NotationResult",
    "canonical_unsanitized",
    "to_covalent",
    "to_house_convention",
]
