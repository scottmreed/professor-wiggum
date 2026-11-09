"""Electron counting for single-metal transition-metal complexes.

Counting model: the **covalent (neutral-ligand) method**, in Green's MLX
formulation (M. L. H. Green, *J. Organomet. Chem.* 1995, 500, 127; the
convention used by Crabtree, *The Organometallic Chemistry of the Transition
Metals*, and Hartwig, *Organotransition Metal Chemistry*). Every ligand is
counted as a neutral fragment:

* **L** -- a neutral 2-electron donor (PR3, CO, NR3, alkene as a donor);
* **X** -- a 1-electron, one-bond ligand (halide, H, alkyl, aryl, amido);
* **Z** -- a 2-electron acceptor (BR3), contributing no electrons.

With ``m`` the metal's group number (3-12) and ``q`` the overall charge of the
complex ion::

    total valence electrons  VE   = m + 2*nL + nX - q
    oxidation state          OS   = nX + 2*nZ + q
    d-electron count         d^n  = m - OS
    coordination number      CN   = coordination sites (see below)

(The ionic model gives the same VE and OS; only the bookkeeping differs.)

How each SMILES form maps onto L/X/Z:

* ``D->[M]`` dative bond from a *neutral* ligand fragment: **L** per bond.
* ``D->[M]`` from an *anionic* ligand fragment (xyz2mol_tm output, which writes
  every metal-donor bond dative and puts the oxidation state on the metal as
  its formal charge, e.g. ``[Cl-]->[Pd+2]``): one dative bond per unit of
  negative fragment charge is **X** (the anion's lone pair is the ionic-model
  picture of an X ligand); any further dative bonds of that fragment are L.
  acac- bonded through both oxygens is therefore LX.
* Several ``->`` donors that are bonded to *each other* are one haptic
  ligand, counted by hapticity in the neutral model: eta2-alkene L, eta3-allyl
  LX, eta5-Cp L2X, eta6-arene L3 (xyz2mol_tm writes Cp as five ``->`` bonds;
  Zeise's anion is ``Cl[Pt-]1(Cl)(Cl)<-C=C->1``). The fragment's charge stays
  in ``q``. The same atoms on *ordinary* bonds are rejected by the lint.
* ``[M]->A`` (dative bond *from* the metal): **Z**.
* ``[M]-Y`` ordinary bond: **X** per bond order (``[M]=CR2`` alkylidene and
  ``[M]=O`` oxo are X2). Under this covalent convention a Grubbs-type
  ruthenium alkylidene ``[Ru]=CHPh`` is Ru(IV); textbooks that treat it as a
  Fischer-type L carbene call it Ru(II), with the same 16-electron total.
* ``[M]-[NH3+]`` charge-separated dative (positively charged ligand fragment on
  an ordinary bond): **L**, one bond per unit of positive fragment charge.
* A **separate anionic fragment** with a single metal cation is an inner-sphere
  ligand (ionic convention): a cyclopentadienide ring (any 5-membered,
  all-carbon aromatic ring carrying the charge, including Cp*) is eta5 **L2X**;
  any other anion of charge -k is X_k (``[Pd+2].[Cl-].[Cl-]`` is PdCl2).
  Outer-sphere anions must therefore be left out of the SMILES; give the
  complex ion with its charge instead (``Cl[Pt-2](Cl)(Cl)Cl`` for [PtCl4]2-).
* A separate cation that is not a transition metal (K+, Na+, NR4+) is a
  counter-ion and is excluded from ``q``.

Because the overall charge ``q`` is a sum over every fragment, the counts are
the same whichever way the charges are distributed: ``Cl[Pt-2](Cl)(Cl)Cl``,
``[Pt+2].[Cl-].[Cl-].[Cl-].[Cl-]`` and ``[Cl-]->[Pt+2](<-[Cl-])(<-[Cl-])<-[Cl-]``
all give Pt(II), d8, 16 e.

Coordination number counts coordination *sites*: one per metal-bonded donor
atom (a chelate counts each donor), and three for an eta5-Cp (the L2X sites),
so ferrocene is 6-coordinate.

Unlike ``electron_configuration_service`` (Aufbau, stops at Xe), this needs only
the group number, so it works for Pt, Ir, Au, W, Rh and every other d-block
metal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from rdkit import Chem

from .organometallic_smiles import (
    group_number,
    is_d_block,
    is_transition_metal,
    lint_organometallic_smiles,
)


class ElectronCountError(ValueError):
    """Raised when a SMILES cannot be counted under the house conventions."""


@dataclass(frozen=True)
class Ligand:
    smiles: str
    #: Green MLX class of the whole ligand, e.g. "L", "X", "X2", "LX", "L2X".
    ligand_class: str
    n_l: int
    n_x: int
    n_z: int = 0
    hapticity: int = 1
    #: "bonded" (bonds to the metal in the SMILES) or "ionic" (separate fragment).
    attachment: str = "bonded"
    sites: int = 1


@dataclass(frozen=True)
class ElectronCount:
    smiles: str
    metal: str
    group: int
    ligands: Tuple[Ligand, ...]
    #: Overall charge of the complex ion (counter-ions excluded).
    charge: int
    oxidation_state: int
    d_electrons: int
    valence_electrons: int
    coordination_number: int
    counterions: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_l(self) -> int:
        return sum(ligand.n_l for ligand in self.ligands)

    @property
    def n_x(self) -> int:
        return sum(ligand.n_x for ligand in self.ligands)

    @property
    def n_z(self) -> int:
        return sum(ligand.n_z for ligand in self.ligands)

    @property
    def roman_oxidation_state(self) -> str:
        return roman_numeral(self.oxidation_state)

    @property
    def summary(self) -> str:
        """e.g. ``Pd(II), d8, 16 e``."""
        return (
            f"{self.metal}({self.roman_oxidation_state}), d{self.d_electrons}, "
            f"{self.valence_electrons} e"
        )


_ROMAN = ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"))


def roman_numeral(value: int) -> str:
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    remaining = abs(value)
    out = ""
    for amount, numeral in _ROMAN:
        while remaining >= amount:
            out += numeral
            remaining -= amount
    return sign + out


def mlx_class(n_l: int, n_x: int, n_z: int = 0) -> str:
    def part(symbol: str, count: int) -> str:
        if count <= 0:
            return ""
        return symbol if count == 1 else f"{symbol}{count}"

    return part("L", n_l) + part("X", n_x) + part("Z", n_z) or "-"


def _is_cyclopentadienide(fragment: Chem.Mol) -> bool:
    if Chem.GetFormalCharge(fragment) != -1:
        return False
    for ring in fragment.GetRingInfo().AtomRings():
        if len(ring) != 5:
            continue
        atoms = [fragment.GetAtomWithIdx(idx) for idx in ring]
        if all(atom.GetSymbol() == "C" and atom.GetIsAromatic() for atom in atoms) and any(
            atom.GetFormalCharge() == -1 for atom in atoms
        ):
            return True
    return False


def count_electrons(smiles: str) -> ElectronCount:
    """Count a single-metal complex; raises :class:`ElectronCountError`."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ElectronCountError("SMILES is required")
    smiles = smiles.strip()
    problems = lint_organometallic_smiles(smiles)
    if problems:
        raise ElectronCountError("; ".join(problems))
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ElectronCountError(f"RDKit cannot parse {smiles!r}")

    metals = [atom for atom in mol.GetAtoms() if is_transition_metal(atom)]
    if len(metals) != 1:
        raise ElectronCountError(
            f"expected exactly one transition-metal atom, found {len(metals)}"
        )
    metal = metals[0]
    if not is_d_block(metal.GetAtomicNum()):
        raise ElectronCountError(
            f"{metal.GetSymbol()} is an f-block metal; the 18-electron count does not apply"
        )
    group = group_number(metal.GetAtomicNum())
    assert group is not None
    metal_idx = metal.GetIdx()

    # Ligand fragments are what is left when the metal is deleted; each carries
    # the original atom indices so the bonds to the metal can be classified.
    fragments = Chem.GetMolFrags(_without_atom(mol, metal_idx), asMols=False)
    remap = [idx for idx in range(mol.GetNumAtoms()) if idx != metal_idx]
    metal_fragment_ids = set(Chem.GetMolFrags(mol, asMols=False)[_fragment_of(mol, metal_idx)])

    ligands: List[Ligand] = []
    counterions: List[str] = []
    charge = metal.GetFormalCharge()
    # Hydrides: RDKit folds both ``[PdH]`` and ``[Pd]([H])`` into an H count on
    # the metal rather than keeping a separate atom, so they never show up as a
    # fragment below.
    for _ in range(metal.GetTotalNumHs()):
        ligands.append(Ligand("[H]", "X", 0, 1))

    for fragment in fragments:
        atom_ids = [remap[idx] for idx in fragment]
        fragment_smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=atom_ids)
        fragment_charge = sum(mol.GetAtomWithIdx(idx).GetFormalCharge() for idx in atom_ids)
        bonded = atom_ids[0] in metal_fragment_ids

        if not bonded:
            if fragment_charge > 0:
                counterions.append(fragment_smiles)
                continue
            if fragment_charge == 0:
                raise ElectronCountError(
                    f"neutral fragment {fragment_smiles!r} is not bonded to the metal; "
                    "bond a neutral ligand with a dative arrow (D->[M]) or leave it out"
                )
            charge += fragment_charge
            sub = Chem.MolFromSmiles(fragment_smiles)
            if sub is not None and _is_cyclopentadienide(sub):
                ligands.append(
                    Ligand(fragment_smiles, "L2X", 2, 1, hapticity=5, attachment="ionic", sites=3)
                )
            else:
                n_x = -fragment_charge
                ligands.append(
                    Ligand(fragment_smiles, mlx_class(0, n_x), 0, n_x, attachment="ionic")
                )
            continue

        charge += fragment_charge
        dative_donors: List[int] = []
        dative_out = 0
        covalent_orders = 0
        covalent_bonds = 0
        donor_atoms = set()
        for idx in atom_ids:
            bond = mol.GetBondBetweenAtoms(idx, metal_idx)
            if bond is None:
                continue
            donor_atoms.add(idx)
            if bond.GetBondType() == Chem.BondType.DATIVE:
                if bond.GetEndAtomIdx() == metal_idx:
                    dative_donors.append(idx)
                else:
                    dative_out += 1
            else:
                covalent_bonds += 1
                covalent_orders += int(round(bond.GetBondTypeAsDouble()))
        # Dative donors bonded to one another are one haptic ligand: an
        # eta-n unit donates n electrons in the neutral model (eta2-alkene L,
        # eta3-allyl LX, eta5-Cp L2X, eta6-arene L3) and fills (n+1)//2 sites.
        # This is how xyz2mol_tm writes a Cp ring (five "->" bonds).
        sigma_l = 0
        haptic_l = haptic_x = 0
        hapticity = 1
        sites = len(donor_atoms) - len(dative_donors)
        for component in _connected_components(mol, dative_donors):
            size = len(component)
            if size == 1:
                sigma_l += 1
                sites += 1
                continue
            haptic_l += size // 2
            haptic_x += size % 2
            hapticity = max(hapticity, size)
            sites += (size + 1) // 2
        n_l, n_x, n_z = sigma_l, covalent_orders, dative_out
        if fragment_charge < 0:
            # xyz2mol_tm / ionic-style: an anion's sigma lone pair is an X bond.
            # (Haptic units are already counted as neutral ligands; their
            # charge simply stays in q.)
            shift = min(-fragment_charge, n_l)
            n_l -= shift
            n_x += shift
        elif fragment_charge > 0:
            # [M]-[NH3+]: a charge-separated dative bond is an L, not an X.
            shift = min(fragment_charge, covalent_bonds, n_x)
            n_x -= shift
            n_l += shift
        n_l += haptic_l
        n_x += haptic_x
        ligands.append(
            Ligand(
                fragment_smiles,
                mlx_class(n_l, n_x, n_z),
                n_l,
                n_x,
                n_z,
                hapticity=hapticity,
                attachment="bonded",
                sites=sites,
            )
        )

    n_l = sum(ligand.n_l for ligand in ligands)
    n_x = sum(ligand.n_x for ligand in ligands)
    n_z = sum(ligand.n_z for ligand in ligands)
    oxidation_state = n_x + 2 * n_z + charge
    return ElectronCount(
        smiles=smiles,
        metal=metal.GetSymbol(),
        group=group,
        ligands=tuple(ligands),
        charge=charge,
        oxidation_state=oxidation_state,
        d_electrons=group - oxidation_state,
        valence_electrons=group + 2 * n_l + n_x - charge,
        coordination_number=sum(ligand.sites for ligand in ligands),
        counterions=tuple(counterions),
    )


def try_count_electrons(smiles: Optional[str]) -> Tuple[Optional[ElectronCount], Optional[str]]:
    """``(count, None)`` on success, ``(None, reason)`` otherwise."""
    try:
        return count_electrons(smiles or ""), None
    except ElectronCountError as exc:
        return None, str(exc)


def _connected_components(mol: Chem.Mol, atom_ids: List[int]) -> List[List[int]]:
    remaining = set(atom_ids)
    components: List[List[int]] = []
    while remaining:
        stack = [remaining.pop()]
        component: List[int] = []
        while stack:
            idx = stack.pop()
            component.append(idx)
            for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                if neighbor.GetIdx() in remaining:
                    remaining.discard(neighbor.GetIdx())
                    stack.append(neighbor.GetIdx())
        components.append(component)
    return components


def _without_atom(mol: Chem.Mol, atom_idx: int) -> Chem.Mol:
    editable = Chem.RWMol(mol)
    editable.RemoveAtom(atom_idx)
    return editable.GetMol()


def _fragment_of(mol: Chem.Mol, atom_idx: int) -> int:
    for position, fragment in enumerate(Chem.GetMolFrags(mol, asMols=False)):
        if atom_idx in fragment:
            return position
    raise ElectronCountError("metal atom not found")  # pragma: no cover


__all__: List[str] = [
    "ElectronCount",
    "ElectronCountError",
    "Ligand",
    "STEP_DELTAS",
    "count_electrons",
    "mlx_class",
    "roman_numeral",
    "try_count_electrons",
]

#: Expected changes (to - from) per elementary step under the covalent model,
#: shared with the catalytic-cycle validator. ``None`` means "not checked".
STEP_DELTAS: Dict[str, Tuple[Optional[int], Optional[int]]] = {
    # kind: (oxidation-state delta, electron-count delta)
    "oxidative_addition": (2, 2),
    "reductive_elimination": (-2, -2),
    "transmetalation": (0, 0),
    "migratory_insertion": (0, -2),
    "beta_hydride_elimination": (0, 2),
    "ligand_association": (0, 2),
    "ligand_dissociation": (0, -2),
    "other": (None, None),
}
