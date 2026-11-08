"""Strict mapped electron-inventory checks for steps involving radicals.

Bond entries count both bonding electrons once; atom entries are nonbonding
valence electrons. Hydrogens folded into bracket atoms must remain unchanged:
transferred hydrogens must be explicit mapped atoms. This is an electron
bookkeeping check, not a spin, kinetics, or excitation-state model.
"""
from collections import Counter
from typing import Iterable

from rdkit import Chem

from .mechanism_moves import MechanismMove, mapped_smiles_parser_params, split_cxsmiles_metadata


def _inventory(side):
    mol = Chem.MolFromSmiles(side, mapped_smiles_parser_params(sanitize=True))
    if mol is None:
        raise ValueError("Cannot parse mapped radical state")
    Chem.Kekulize(mol, clearAromaticFlags=True)
    atoms, bonds, nonbonding = {}, {}, {}
    for atom in mol.GetAtoms():
        idx = atom.GetAtomMapNum()
        if idx <= 0 or idx in atoms:
            raise ValueError("Radical validation requires unique positive maps for every atom")
        if atom.GetAtomicNum() not in {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}:
            raise ValueError("Radical electron inventory currently supports main-group organic atoms only")
        hydrogens = atom.GetTotalNumHs()
        valence = Chem.GetPeriodicTable().GetNOuterElecs(atom.GetAtomicNum())
        bonded = sum(b.GetBondTypeAsDouble() for b in atom.GetBonds()) + hydrogens
        nb = valence - atom.GetFormalCharge() - bonded
        if nb < 0 or nb != int(nb) or nb < atom.GetNumRadicalElectrons():
            raise ValueError("Invalid nonbonding electron inventory")
        atoms[idx] = (atom.GetAtomicNum(), atom.GetIsotope(), hydrogens, atom.GetNumRadicalElectrons())
        nonbonding[idx] = int(nb)
    for bond in mol.GetBonds():
        pair = tuple(sorted((bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum())))
        bonds[pair] = int(2 * bond.GetBondTypeAsDouble())
    return atoms, bonds, nonbonding


# Atoms whose radical electrons mark a genuinely radical state. Hydrogen and
# metal atoms are excluded: ``[H]``, ``[Na]``, ``[Cu]`` in a pair-arrow step
# are almost always a proton or a counter-ion drawn without charge, and are
# not open-shell organic intermediates.
_RADICAL_STATE_ELEMENTS = {5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53}


def state_has_radicals(species: Iterable[str]) -> bool:
    """Whether runtime state SMILES contain an open-shell main-group atom.

    Runtime states are ordinary SMILES with their real hydrogen counts, unlike
    core-only SMIRKS fragments where an H-less bracket atom (``[C:1]``) only
    looks like a radical. Only an odd radical-electron count counts, so
    closed-shell singlet carbenes/nitrenes in pair-arrow chemistry keep the
    historical validation.
    """
    for text in species or []:
        mol = Chem.MolFromSmiles(str(text or ""))
        if mol is None:
            continue
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() in _RADICAL_STATE_ELEMENTS and atom.GetNumRadicalElectrons() % 2 == 1:
                return True
    return False


def requires_radical_validation(moves: Iterable[MechanismMove], states: Iterable[Iterable[str]] = ()) -> bool:
    """Choose strict radical checking from explicit fishhooks or radical states.

    Never inferred from SMIRKS atoms: RDKit reads H-less bracket atoms in
    core-only SMIRKS (``[C:1]``) as radicals although they are not.
    """
    if any(m.electrons == 1 or m.kind in {"radical", "electron_transfer"} for m in moves):
        return True
    return any(state_has_radicals(species) for species in states or ())


def validate_radical_moves(reaction_smirks: str, moves: Iterable[MechanismMove]) -> dict:
    """Compare complete arrow-implied and observed bond/nonbonding changes."""
    try:
        core, _ = split_cxsmiles_metadata(reaction_smirks)
        if core.count(">>") != 1:
            raise ValueError("Expected mapped reactants>>products")
        left, right = (_inventory(side) for side in core.split(">>"))
        la, lb, ln = left
        ra, rb, rn = right
        if set(la) != set(ra) or any(la[k][:3] != ra[k][:3] for k in la):
            raise ValueError("Atom identities and folded hydrogens must be conserved; map transferred H explicitly")
        expected_bonds, expected_nb = Counter(), Counter()
        spent_atoms, spent_radicals, spent_pairs, spent_bonds, spent_pi, spent_sigma = (Counter() for _ in range(6))
        for move in moves:
            n = move.electrons
            if type(n) is not int or n not in {1, 2}:
                raise ValueError("Electron count must be integer 1 or 2")
            if move.target_atom not in la:
                raise ValueError("Unknown target atom map")
            if move.kind in {"lone_pair", "radical", "electron_transfer"}:
                src = move.source_atom
                if src not in la or src == move.target_atom:
                    raise ValueError("Unknown or self-referencing atom source")
                if move.kind == "lone_pair" and n != 2 or move.kind != "lone_pair" and n != 1:
                    raise ValueError("Electron count does not match source kind")
                spent_atoms[src] += n
                if move.kind == "radical":
                    spent_radicals[src] += n
                elif move.kind == "lone_pair":
                    spent_pairs[src] += n
                expected_nb[src] -= n
                if move.kind == "electron_transfer":
                    expected_nb[move.target_atom] += n
                else:
                    expected_bonds[tuple(sorted((src, move.target_atom)))] += n
            elif move.kind in {"sigma_bond", "pi_bond"}:
                pair = tuple(sorted((move.bond_start, move.bond_end)))
                if pair not in lb or move.through_atom not in pair:
                    raise ValueError("Missing source bond or invalid delivering endpoint")
                if move.target_atom in pair and move.target_atom != move.through_atom:
                    # Breaking a bond back onto its own other end changes nothing,
                    # so it would let invalid arrows cancel.
                    raise ValueError("Bond-source move has no net electron change")
                spent_bonds[pair] += n
                (spent_pi if move.kind == "pi_bond" else spent_sigma)[pair] += n
                expected_bonds[pair] -= n
                if move.through_atom == move.target_atom:
                    expected_nb[move.target_atom] += n
                else:
                    expected_bonds[tuple(sorted((move.through_atom, move.target_atom)))] += n
            else:
                raise ValueError("Unknown move kind")
        if any(n > ln[k] for k, n in spent_atoms.items()):
            raise ValueError("Nonbonding source electron capacity exceeded")
        if any(n > la[k][3] for k, n in spent_radicals.items()):
            raise ValueError("Radical source electron capacity exceeded")
        if any(n > ln[k] - la[k][3] for k, n in spent_pairs.items()):
            raise ValueError("Lone-pair source electron capacity exceeded")
        if any(n > lb[k] for k, n in spent_bonds.items()) or any(n > lb[k] - 2 for k, n in spent_pi.items()) or any(n > 2 for n in spent_sigma.values()):
            raise ValueError("Bond source electron capacity exceeded")
        observed_bonds = {k: rb.get(k, 0) - lb.get(k, 0) for k in set(lb) | set(rb)}
        observed_nb = {k: rn[k] - ln[k] for k in la}
        clean = lambda values: {k: v for k, v in values.items() if v}
        if clean(expected_bonds) != clean(observed_bonds):
            raise ValueError("Arrow-implied bond electron changes do not match mapped states")
        if clean(expected_nb) != clean(observed_nb):
            raise ValueError("Arrow-implied nonbonding electrons do not match mapped charges/radicals")
        return {"valid": True, "method": "mapped_radical_electron_inventory.v1", "message": "Mapped bond and nonbonding electron inventories match"}
    except (ValueError, TypeError, RuntimeError) as exc:
        return {"valid": False, "method": "mapped_radical_electron_inventory.v1", "error": str(exc)}


def radical_states_match(reaction_smirks: str, current_state, resulting_state) -> bool:
    """Ensure the checked mapped reaction actually represents the runtime states.

    Each SMIRKS side must be a sub-multiset of its runtime state (species and
    multiplicities), and the untouched remainder (spectators) must be the same
    on both sides.
    """
    def species(items):
        counts = Counter()
        for text in items:
            mol = Chem.MolFromSmiles(text, mapped_smiles_parser_params(sanitize=True))
            if mol is None:
                raise ValueError("Invalid state")
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            # Remove explicit H only for comparison, preserving charge/radicals.
            mol = Chem.RemoveHs(mol)
            for fragment in Chem.GetMolFrags(mol, asMols=True):
                counts[Chem.MolToSmiles(fragment)] += 1
        return counts
    try:
        core, _ = split_cxsmiles_metadata(reaction_smirks)
        left, right = core.split('>>')
        lhs, rhs = species([left]), species([right])
        current, resulting = species(current_state), species(resulting_state)
        if lhs - current or rhs - resulting:
            return False
        return (current - lhs) == (resulting - rhs)
    except (ValueError, TypeError, RuntimeError):
        return False
