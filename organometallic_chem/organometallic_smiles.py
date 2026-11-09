"""Organometallic SMILES conventions and lint.

RDKit (2023.09) parses transition-metal complexes, but it does not know which
ligand bonds are *dative*. Written with ordinary bonds, triphenylphosphine on
palladium (``[Pd]P(c1ccccc1)(c1ccccc1)c1ccccc1``) is silently re-read as a
valence-expanded phosphorane with a phantom hydrogen -- canonical output becomes
``[Pd][PH](c1ccccc1)...`` -- and a neutral amine or thioether either fails to
sanitize or is "cleaned up" behind the author's back. The figure then shows an
H that is not there, and the electron count treats a 2-electron L donor as a
1-electron X ligand.

House conventions (v1) for SMILES that the book renders or counts:

* **Neutral 2-electron donors (L-type) use a dative arrow** pointing from donor
  to metal: ``CP(C)(C)->[Pd]<-P(C)(C)C`` (Pd(PMe3)2), ``CN(C)(C)->[Pd]``,
  ``n1ccccc1->[Pd]``. RDKit counts a dative bond as zero valence on the donor,
  so P keeps its three substituents and no hydrogen appears.
* **Carbon monoxide** is the charge-separated form with a dative arrow from
  carbon: ``[O+]#[C-]->[Cr]``. ``[Cr]C#O`` does not parse (O valence 3) and
  ``O=C=[Cr]`` reads as a ketenylidene (X2), so neither is accepted.
* **X-type ligands** (halide, hydride, alkyl, aryl, amido, alkoxide) use an
  ordinary single bond: ``Br[Pd]c1ccccc1``. An alkylidene ``[M]=C`` is X2.
* **Haptic (eta-n, n >= 3) ligands cannot be represented** as RDKit bonds:
  drawing eta5-Cp as five sigma bonds makes five pentavalent carbons. Metallocenes
  and other Cp complexes are written *ionically*, as a metal cation plus
  cyclopentadienide anions: ferrocene is ``[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1``.
  :mod:`organometallic_chem.organometallic_electron_count` reads each Cp- fragment as an
  eta5, L2X ligand. eta3-allyl and eta6-arene complexes are out of scope for v1.
  CXSMILES coordination extensions (``m:``/``C:``) and dummy-atom centroids
  are not drawn by our pipeline and are flagged too.

This module is RDKit-only (no app imports) so the offline topic-package
validator in ``tools/topic_packages`` can use it without importing the backend's
service layer.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from rdkit import Chem, rdBase

# d-block (groups 3-12) plus the lanthanides and actinides. The f-block is
# included for the lint (the dative-bond rules are the same) but the electron
# counter refuses it: the 18-electron rule does not apply there.
_D_BLOCK_RANGES = ((21, 30), (39, 48), (72, 80), (104, 112))
_F_BLOCK = frozenset(range(57, 72)) | frozenset(range(89, 104))
TRANSITION_METAL_ATOMIC_NUMBERS = frozenset(
    z for low, high in _D_BLOCK_RANGES for z in range(low, high + 1)
) | _F_BLOCK

# Normal neutral valence of the main-group donor atoms the lint polices. A
# formal charge shifts it one-for-one (N+ 4, O+ 3, O- 1, P- 2).
DONOR_NORMAL_VALENCE: Dict[str, int] = {
    "N": 3,
    "P": 3,
    "As": 3,
    "O": 2,
    "S": 2,
    "Se": 2,
}

_CXSMILES_COORDINATION = re.compile(r"(?:^|[|,])\s*(?:m|C):")


def is_transition_metal(atom: Chem.Atom) -> bool:
    return atom.GetAtomicNum() in TRANSITION_METAL_ATOMIC_NUMBERS


def is_d_block(atomic_number: int) -> bool:
    return any(low <= atomic_number <= high for low, high in _D_BLOCK_RANGES)


def group_number(atomic_number: int) -> Optional[int]:
    """IUPAC group (3-12) of a d-block metal; ``None`` for anything else.

    Works for every period, unlike the Aufbau-based electron_configuration
    service, which stops at Xe.
    """
    # Sc-Zn and Y-Cd start at group 3; Hf-Hg and Rf-Cn start at group 4
    # because La/Ac (group 3) sit in the f-block ranges above.
    for (low, high), first_group in zip(_D_BLOCK_RANGES, (3, 3, 4, 4)):
        if low <= atomic_number <= high:
            return atomic_number - low + first_group
    return None


def contains_transition_metal(smiles: str | None) -> bool:
    """Cheap check (no sanitization) for a transition-metal atom."""
    mol = _parse_unsanitized(smiles)
    return mol is not None and any(is_transition_metal(atom) for atom in mol.GetAtoms())


def _parse_unsanitized(smiles: str | None) -> Optional[Chem.Mol]:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    blocker = rdBase.BlockLogs()  # noqa: F841 - silences RDKit parse chatter
    try:
        mol = Chem.MolFromSmiles(smiles.strip(), sanitize=False)
    except Exception:
        return None
    if mol is None:
        return None
    # Valences as literally written: with strict=False an over-valent N is
    # recorded rather than rejected, and RDKit's organometallic "cleanup"
    # (which silently turns some sigma bonds into dative ones) never runs.
    mol.UpdatePropertyCache(strict=False)
    return mol


def _sanitize_error(smiles: str) -> Optional[str]:
    blocker = rdBase.BlockLogs()  # noqa: F841
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return "RDKit cannot parse it"
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:  # RDKit raises several sanitize exception types
        return str(exc).strip() or type(exc).__name__
    return None


def _bond_valence(bond: Chem.Bond, atom: Chem.Atom) -> float:
    if bond.GetBondType() == Chem.BondType.DATIVE:
        # RDKit's own rule: a dative bond adds nothing to its donor's valence.
        return 0.0 if bond.GetBeginAtomIdx() == atom.GetIdx() else 1.0
    if bond.GetBondType() == Chem.BondType.AROMATIC or bond.GetIsAromatic():
        return 1.5
    return float(bond.GetBondTypeAsDouble())


def _written_valence(atom: Chem.Atom) -> float:
    return (
        sum(_bond_valence(bond, atom) for bond in atom.GetBonds())
        + atom.GetNumExplicitHs()
        + atom.GetNumImplicitHs()
    )


def _metal_neighbors(atom: Chem.Atom, *, dative: bool) -> List[Chem.Atom]:
    found = []
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        if not is_transition_metal(other):
            continue
        if (bond.GetBondType() == Chem.BondType.DATIVE) == dative:
            found.append(other)
    return found


def lint_organometallic_smiles(smiles: str | None) -> List[str]:
    """Problems with how a transition-metal complex is written; ``[]`` if clean.

    Molecules without a transition metal always lint clean, so this is safe to
    run over every molecule asset.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return []
    raw = smiles.strip()
    core, _, extensions = raw.partition(" ")
    mol = _parse_unsanitized(core)
    if mol is None:
        return []  # not ours to report; the ordinary SMILES checks will
    if not any(is_transition_metal(atom) for atom in mol.GetAtoms()):
        return []

    problems: List[str] = []

    if extensions.strip().startswith("|") and _CXSMILES_COORDINATION.search(extensions):
        problems.append(
            "CXSMILES coordination/multicenter extensions (m:, C:) are not drawn "
            "by the ChemIllusion renderer; write eta5-Cp complexes ionically "
            "(ferrocene: [Fe+2].c1cc[cH-]c1.c1cc[cH-]c1)"
        )

    for atom in mol.GetAtoms():
        if not is_transition_metal(atom):
            continue
        metal = atom.GetSymbol()
        sigma_partners = [
            bond.GetOtherAtom(atom)
            for bond in atom.GetBonds()
            if bond.GetBondType() != Chem.BondType.DATIVE
        ]
        if any(partner.GetAtomicNum() == 0 for partner in sigma_partners):
            problems.append(
                f"{metal} is bonded to a dummy atom (*); centroid/haptic "
                "attachment points are not supported -- write eta5-Cp complexes "
                "ionically ([Fe+2].c1cc[cH-]c1.c1cc[cH-]c1)"
            )
        # eta-n drawn as n sigma bonds: three or more metal-bonded atoms that are
        # bonded to each other in a chain. (Two adjacent atoms is the
        # metallacyclopropane limit of an eta2-alkene and is allowed.)
        partner_ids = {partner.GetIdx() for partner in sigma_partners if partner.GetAtomicNum() > 1}
        largest = _largest_connected_subset(mol, partner_ids)
        if largest >= 3:
            problems.append(
                f"{metal} is sigma-bonded to {largest} mutually bonded atoms, "
                f"i.e. an eta{largest} ligand drawn as {largest} single bonds; RDKit "
                "cannot represent haptic bonding (the ring carbons become "
                "over-valent). Write Cp complexes ionically, e.g. ferrocene as "
                "[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1"
            )

    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in DONOR_NORMAL_VALENCE:
            continue
        metals = _metal_neighbors(atom, dative=False)
        if not metals:
            continue
        expected = DONOR_NORMAL_VALENCE[symbol] + atom.GetFormalCharge()
        written = _written_valence(atom)
        if written <= expected:
            continue  # an X-type amido/phosphido/alkoxide/thiolate bond: fine
        metal = metals[0].GetSymbol()
        phantom = atom.GetNumImplicitHs()
        detail = (
            f"RDKit fills it to valence {int(written)} with {phantom} phantom H "
            f"(canonical output shows [{symbol}H])"
            if phantom
            else f"that makes {symbol} {int(written)}-valent"
        )
        problems.append(
            f"neutral {symbol} donor (atom {atom.GetIdx()}) is bonded to {metal} "
            f"by an ordinary bond; {detail}. Write the L-type donor bond as a "
            f"dative arrow from {symbol} to {metal} ({symbol}->[{metal}]), e.g. "
            f"Pd(PMe3)2 is CP(C)(C)->[Pd]<-P(C)(C)C"
        )

    for atom in mol.GetAtoms():
        # [M]C#O: the carbonyl O ends up 3-valent and the SMILES will not
        # sanitize; say what to write instead of surfacing RDKit's error.
        if atom.GetSymbol() != "C" or not _metal_neighbors(atom, dative=False):
            continue
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if other.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.TRIPLE:
                metal = _metal_neighbors(atom, dative=False)[0].GetSymbol()
                problems.append(
                    f"carbonyl ligand written as {metal}-C#O; write CO as the "
                    f"charge-separated L donor [O+]#[C-]->[{metal}]"
                )

    if not problems:
        error = _sanitize_error(core)
        if error:
            problems.append(f"RDKit cannot sanitize this complex: {error}")
    # A dative arrow starts at the atom written immediately before "->". Writing
    # Ph3P as "P(c1ccccc1)(c1ccccc1)c1ccccc1->[Pd]" therefore makes a ring
    # carbon the donor, and the complex still counts correctly. A neutral carbon
    # can only donate through a C=C or C#C pi bond -- either the one-arrow
    # shorthand "<-C=C" or an eta2 pair of dative carbons. CO and carbanions
    # carry a negative charge on carbon. Aromatic and saturated carbons cannot.
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DATIVE:
            continue
        donor, acceptor = bond.GetBeginAtom(), bond.GetEndAtom()
        if not is_transition_metal(acceptor) or donor.GetSymbol() != "C" or donor.GetFormalCharge() < 0:
            continue
        pi_donor = any(
            b.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)
            and b.GetOtherAtom(donor).GetSymbol() == "C"
            for b in donor.GetBonds()
        )
        eta2_partner = any(
            other.GetSymbol() == "C"
            and any(
                b.GetBondType() == Chem.BondType.DATIVE and is_transition_metal(b.GetOtherAtom(other))
                for b in other.GetBonds()
            )
            for other in donor.GetNeighbors()
        )
        if not (pi_donor or eta2_partner):
            problems.append(
                f"dative bond to {acceptor.GetSymbol()} starts at a neutral carbon (atom "
                f"{donor.GetIdx()}) with no lone pair; the donor is the atom written "
                "immediately before '->', so write the ligand with its donor last, e.g. "
                "PPh3 as c1ccc(cc1)P(c1ccccc1)(c1ccccc1)->[Pd]"
            )

    # A radical on a non-metal atom of a complex is almost always a SMILES slip
    # that the counts above cannot see: an eta2 alkene closed with a single bond
    # (both carbons still read as dative donors, so the complex counts fine) or
    # a missing hydrogen. Radicals on the metal itself are RDKit's valence-model
    # artefact and are cleared at depiction time instead.
    sanitized = Chem.MolFromSmiles(core)
    if sanitized is not None:
        radical_atoms = [
            f"{atom.GetSymbol()} (atom {atom.GetIdx()})"
            for atom in sanitized.GetAtoms()
            if not is_transition_metal(atom) and atom.GetNumRadicalElectrons()
        ]
        if radical_atoms:
            problems.append(
                f"{', '.join(radical_atoms)} carry an unpaired electron; usually a "
                "missing double bond (an eta2 alkene is written C=C with both "
                "carbons dative to the metal, e.g. C(=C->1)->[Pd]1) or a missing H"
            )

    return problems


def _largest_connected_subset(mol: Chem.Mol, atom_ids: set[int]) -> int:
    """Size of the largest group of ``atom_ids`` connected by bonds among themselves."""
    remaining = set(atom_ids)
    best = 0
    while remaining:
        stack = [remaining.pop()]
        size = 0
        while stack:
            idx = stack.pop()
            size += 1
            for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                n_idx = neighbor.GetIdx()
                if n_idx in remaining:
                    remaining.discard(n_idx)
                    stack.append(n_idx)
        best = max(best, size)
    return best
