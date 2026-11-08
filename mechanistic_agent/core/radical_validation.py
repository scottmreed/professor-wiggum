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


def _fragment_is_radical(mol) -> bool:
    return any(
        atom.GetAtomicNum() in _RADICAL_STATE_ELEMENTS and atom.GetNumRadicalElectrons() % 2 == 1
        for atom in mol.GetAtoms()
    )


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
        if mol is not None and _fragment_is_radical(mol):
            return True
    return False


def _species_counts(items: Iterable[str], *, skip_invalid: bool = False) -> Counter:
    """Canonical (maps stripped, H removed) fragment SMILES with multiplicities."""
    counts: Counter = Counter()
    for text in items or []:
        mol = Chem.MolFromSmiles(str(text or ""), mapped_smiles_parser_params(sanitize=True))
        if mol is None:
            if skip_invalid:
                continue
            raise ValueError("Invalid state")
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        # Remove explicit H only for comparison, preserving charge/radicals.
        mol = Chem.RemoveHs(mol)
        for fragment in Chem.GetMolFrags(mol, asMols=True):
            counts[Chem.MolToSmiles(fragment)] += 1
    return counts


def _canonical_is_radical(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None and _fragment_is_radical(mol)


def radical_species_changed(states: Iterable[Iterable[str]] = (), reaction_smirks: str = "") -> bool:
    """Whether a radical-bearing species takes part in the step.

    ``states`` is ``[current, resulting]``. A radical species counts when its
    multiplicity differs between the two states (created, consumed or split),
    or when a whole fragment of the mapped SMIRKS is exactly a radical species
    present in the states. An untouched radical spectator (TEMPO, O2) carried
    unchanged next to a pair-arrow step does not count.
    """
    states = [list(side or []) for side in (states or ())]
    if not states:
        return False
    current = _species_counts(states[0], skip_invalid=True)
    resulting = _species_counts(states[1] if len(states) > 1 else [], skip_invalid=True)
    changed = (current - resulting) + (resulting - current)
    if any(_canonical_is_radical(smiles) for smiles in changed):
        return True
    radicals = {smiles for smiles in set(current) | set(resulting) if _canonical_is_radical(smiles)}
    if not radicals or not reaction_smirks:
        return False
    try:
        core, _ = split_cxsmiles_metadata(reaction_smirks)
        if core.count(">>") != 1:
            return False
        touched = set()
        for side in core.split(">>"):
            if side.strip():
                touched |= set(_species_counts([side], skip_invalid=True))
    except (ValueError, TypeError, RuntimeError):
        return False
    return bool(touched & radicals)


def requires_radical_validation(
    moves: Iterable[MechanismMove],
    states: Iterable[Iterable[str]] = (),
    reaction_smirks: str = "",
) -> bool:
    """Choose strict radical checking from explicit fishhooks or a radical species change.

    Never inferred from SMIRKS atoms: RDKit reads H-less bracket atoms in
    core-only SMIRKS (``[C:1]``) as radicals although they are not. A radical
    spectator that the step leaves untouched keeps the pair-arrow path.
    """
    if any(m.electrons == 1 or m.kind in {"radical", "electron_transfer"} for m in moves):
        return True
    return radical_species_changed(states, reaction_smirks)


def move_key(move: MechanismMove) -> tuple:
    """Order-insensitive identity of a move for metadata/push agreement.

    Bond-source moves compare their bond as an unordered pair plus the
    delivering endpoint, so ``source_bond: [4, 3]`` agrees with ``sigma:3-4>4``.
    """
    bond = frozenset((move.bond_start, move.bond_end)) if move.bond_start is not None else None
    return (move.kind, move.source_atom, bond, move.through_atom, move.target_atom, move.electrons)


def moves_agree(first: Iterable[MechanismMove], second: Iterable[MechanismMove]) -> bool:
    return Counter(map(move_key, first)) == Counter(map(move_key, second))


def _move_effect(move: MechanismMove) -> Counter:
    effect: Counter = Counter()
    n = move.electrons
    if move.kind in {"lone_pair", "radical", "electron_transfer"}:
        effect[("n", move.source_atom)] -= n
        if move.kind == "electron_transfer":
            effect[("n", move.target_atom)] += n
        else:
            effect[("b", tuple(sorted((move.source_atom, move.target_atom))))] += n
    else:
        effect[("b", tuple(sorted((move.bond_start, move.bond_end))))] -= n
        if move.through_atom == move.target_atom:
            effect[("n", move.target_atom)] += n
        else:
            effect[("b", tuple(sorted((move.through_atom, move.target_atom))))] += n
    return effect


_MAX_CANCEL_SUBSET_MOVES = 12


def _has_cancelling_moves(moves: list) -> bool:
    """Whether some arrows have no net effect together (et:1>2 with et:2>1)."""
    effects = [_move_effect(m) for m in moves]

    def zero(indices) -> bool:
        total: Counter = Counter()
        for i in indices:
            for key, value in effects[i].items():
                total[key] += value
        return not any(total.values())

    n = len(effects)
    if n > _MAX_CANCEL_SUBSET_MOVES:
        return any(zero((i, j)) for i in range(n) for j in range(i + 1, n))
    return any(zero([i for i in range(n) if mask >> i & 1]) for mask in range(1, 1 << n))


def validate_radical_moves(reaction_smirks: str, moves: Iterable[MechanismMove]) -> dict:
    """Compare complete arrow-implied and observed bond/nonbonding changes."""
    try:
        core, _ = split_cxsmiles_metadata(reaction_smirks)
        if core.count(">>") != 1:
            raise ValueError("Expected mapped reactants>>products")
        if any(not side.strip() for side in core.split(">>")):
            raise ValueError("Both sides of the mapped reaction must be non-empty")
        moves = list(moves)
        if not moves:
            raise ValueError("At least one electron move is required")
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
        if _has_cancelling_moves(moves):
            raise ValueError("Electron moves cancel each other; remove arrows with no net change")
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
    try:
        core, _ = split_cxsmiles_metadata(reaction_smirks)
        if core.count(">>") != 1:
            return False
        left, right = core.split('>>')
        if not left.strip() or not right.strip():
            return False
        lhs, rhs = _species_counts([left]), _species_counts([right])
        current, resulting = _species_counts(current_state), _species_counts(resulting_state)
        if lhs - current or rhs - resulting:
            return False
        return (current - lhs) == (resulting - rhs)
    except (ValueError, TypeError, RuntimeError):
        return False
