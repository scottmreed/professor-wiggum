"""Persistent atom identity and mapped-state execution (PRD §9, §10, §16.8).

Covers §10.1-10.14 of docs/PRD_jev_atom_identity_mechanistic.md with small
hand-built cases plus a ground-truth sweep over the benchmark (§10.14).

Scope notes
-----------
* §10.12 (backtracking) and §10.13 (resume) are tested at the ``mapped_state``
  level: snapshot/restore of the identity map, allocator monotonicity across
  restore, and JSON persistence through SQLite and a file. Coordinator-level
  integration is a follow-up, not an xfail: branch alternatives are not
  persisted across resume today (``coordinator.py`` restores branch points
  from events without their alternatives), and ``RunState.mapped_loop_state``
  is not persisted either, so a resumed run re-seeds identity.
* The benchmark sweep runs Route A (SMIRKS) and Route B (moves) over every
  step of ``training_data/eval_set.json`` and
  ``training_data/practice_eval/practice_set.json`` (tracked) and
  ``training_data/flower_mechanisms_multistep.json`` (untracked; skipped on a
  fresh clone). Route A must reproduce 100%. Route B's measured rate is
  asserted as a floor and every Route B failure must be a step whose ``mech:``
  block implies fewer bond changes than its SMIRKS (benchmark data, not the
  executor). The heavy-atom-policy sweep uses a deterministic 1-in-4 sample of
  the tracked sets to keep the fast suite quick.
"""
from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

pytest.importorskip("rdkit")

from rdkit import Chem  # noqa: E402
from rdkit.Chem import rdCIPLabeler  # noqa: E402

from mechanistic_agent.core.mapped_state import (  # noqa: E402
    MappedState,
    PersistentAtomIdAllocator,
    advance_mapped_loop_state,
    execute_moves,
    execute_smirks,
    mapped_signature,
    species_signature,
    state_signature,
    sync_mapped_loop_state,
)
from mechanistic_agent.core.mechanism_moves import (  # noqa: E402
    extract_mechanism_moves,
    implied_bond_deltas,
    reaction_bond_deltas,
)

from _optional_assets import PROJECT_ROOT, require_repo_asset  # noqa: E402


def _cip(species: List[str], map_number: int) -> str | None:
    for smiles in species:
        mol = Chem.MolFromSmiles(smiles)
        rdCIPLabeler.AssignCIPLabels(mol)
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() == map_number:
                return atom.GetPropsAsDict().get("_CIPCode")
    return None


def _assert_all_preserved(before: MappedState, result) -> None:
    assert result.ok, result.error
    assert result.duplicate_ids == []
    assert result.new_ids == []
    assert result.lost_ids == []
    assert set(result.preserved_ids) == set(before.pids())
    after = result.next_state
    assert after is not None
    for map_number, pid in before.map_to_pid.items():
        assert after.pid_by_map(map_number) == pid


# ---------------------------------------------------------------------------
# 10.1 No-op round trip
# ---------------------------------------------------------------------------


def test_10_1_ids_survive_mapped_smiles_and_snapshot_round_trip() -> None:
    state = MappedState.from_smiles(["[CH3:5][CH2:9][OH:2]", "CC"])
    # Unmapped atoms get fresh, non-colliding map numbers; every atom gets an id.
    assert sorted(state.map_to_pid) == [2, 5, 9, 10, 11]
    assert state.pids() == [1, 2, 3, 4, 5]
    assert state.allocator.next_id == 6

    # Mapped-SMILES round trip: reparse the serialized species.
    reparsed = MappedState.restore(
        {**state.snapshot(), "species": [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in state.species]}
    )
    assert reparsed.map_to_pid == state.map_to_pid

    # The actual state serialization path is JSON (events / step outputs).
    restored = MappedState.from_json(json.loads(json.dumps(state.to_json())))
    assert restored.species == state.species
    assert restored.map_to_pid == state.map_to_pid
    assert {p: r.as_dict() for p, r in restored.records.items()} == {
        p: r.as_dict() for p, r in state.records.items()
    }


def test_10_1_custom_rdkit_props_do_not_survive_smiles_but_map_numbers_do() -> None:
    mol = Chem.MolFromSmiles("[CH3:5][CH2:9][OH:2]")
    for atom in mol.GetAtoms():
        atom.SetIntProp("pid", 100 + atom.GetAtomMapNum())
        atom.SetProp("note", "x")
    back = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    assert all(not a.HasProp("pid") and not a.HasProp("note") for a in back.GetAtoms())
    assert sorted(a.GetAtomMapNum() for a in back.GetAtoms()) == [2, 5, 9]


def test_10_1_ingress_keeps_original_mapping_for_seeding() -> None:
    from mechanistic_agent.core.db import _normalize_run_input_payload

    payload = _normalize_run_input_payload(
        {"starting_materials": ["[CH3:1][OH:2]"], "products": ["[CH2:1]=[O:2]"]}
    )
    assert payload["starting_materials"] == ["CO"]
    seed = payload["input_boundary"]["original_starting_materials"]
    mapped, origin = sync_mapped_loop_state(None, current_state=payload["starting_materials"], seed_species=seed)
    assert origin == "seed"
    assert sorted(mapped.map_to_pid) == [1, 2]


# ---------------------------------------------------------------------------
# 10.2 Bond-order change
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "species,smirks,expected",
    [
        # C=C -> C-C (allylic shift)
        (["[CH2:1]=[CH:2][CH2+:3]"], "[CH2:1]=[CH:2][CH2+:3]>>[CH2+:1][CH:2]=[CH2:3] |mech:v1;pi:1-2>3|", ["C=C[CH2+]"]),
        # C=O -> C-O
        (
            ["[CH3:1][C:2](=[O:3])[CH3:4]"],
            "[CH3:1][C:2](=[O:3])[CH3:4]>>[CH3:1][C+:2]([O-:3])[CH3:4] |mech:v1;pi:2-3>3|",
            ["C[C+](C)[O-]"],
        ),
    ],
)
@pytest.mark.parametrize("route", ["smirks", "moves"])
def test_10_2_bond_order_change_retains_all_ids(species, smirks, expected, route) -> None:
    state = MappedState.from_smiles(species)
    run = execute_smirks if route == "smirks" else execute_moves
    result = run(state, smirks, expected_resulting_state=expected)
    _assert_all_preserved(state, result)
    assert result.smirks_state_agreement is True


# ---------------------------------------------------------------------------
# 10.3 Bond formation / 10.4 cleavage and component split
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route", ["smirks", "moves"])
def test_10_3_bond_formation_merges_components(route) -> None:
    state = MappedState.from_smiles(["[CH3:1][CH+:2][CH3:3]", "[OH2:4]"])
    smirks = "[CH3:1][CH+:2][CH3:3].[OH2:4]>>[CH3:1][CH:2]([CH3:3])[OH2+:4] |mech:v1;lp:4>2|"
    run = execute_smirks if route == "smirks" else execute_moves
    result = run(state, smirks, expected_resulting_state=["CC(C)[OH2+]"])
    _assert_all_preserved(state, result)
    assert result.smirks_state_agreement is True
    after = result.next_state
    assert len(after.species) == 1
    comps = {after.records[after.pid_by_map(m)].component for m in (1, 2, 3, 4)}
    assert comps == {0}


@pytest.mark.parametrize("route", ["smirks", "moves"])
def test_10_4_cleavage_identity_independent_of_array_position(route) -> None:
    smirks = "[CH3:1][CH:2]([CH3:3])[OH2+:4]>>[CH3:1][CH+:2][CH3:3].[OH2:4] |mech:v1;sigma:2-4>4|"
    run = execute_smirks if route == "smirks" else execute_moves
    allocator = PersistentAtomIdAllocator()
    a = MappedState.from_smiles(["[Na+:9]", "[CH3:1][CH:2]([CH3:3])[OH2+:4]"], allocator=allocator)
    b = MappedState.restore({**a.snapshot(), "species": list(reversed(a.species))})
    ra, rb = run(a, smirks), run(b, smirks)
    for result, before in ((ra, a), (rb, b)):
        _assert_all_preserved(before, result)
        assert len(result.next_state.species) == 3
    assert ra.identity_map == rb.identity_map
    assert state_signature(ra.resulting_state) == state_signature(["C[CH+]C", "O", "[Na+]"])


# ---------------------------------------------------------------------------
# 10.5 Substitution (SN2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route", ["smirks", "moves"])
def test_10_5_sn2_merge_split_and_charge_change(route) -> None:
    state = MappedState.from_smiles(["[OH-:1]", "[CH3:2][Br:3]"])
    smirks = "[OH-:1].[CH3:2][Br:3]>>[OH:1][CH3:2].[Br-:3] |mech:v1;lp:1>2;sigma:2-3>3|"
    run = execute_smirks if route == "smirks" else execute_moves
    result = run(state, smirks, expected_resulting_state=["CO", "[Br-]"])
    _assert_all_preserved(state, result)
    assert result.smirks_state_agreement is True
    after = result.next_state
    o, c, br = (after.records[after.pid_by_map(m)] for m in (1, 2, 3))
    assert o.component == c.component != br.component


def test_10_5_llm_renumbered_smirks_on_stripped_state_is_unconstrained() -> None:
    state = MappedState.from_smiles(["CBr", "[OH-]"])
    result = execute_smirks(state, "[OH-:7].[CH3:8][Br:9]>>[OH:7][CH3:8].[Br-:9]", expected_resulting_state=["CO", "[Br-]"])
    assert result.ok and result.smirks_state_agreement is True
    assert result.match_mode == "unconstrained"
    assert result.new_ids == [] and result.lost_ids == []


def test_lhs_not_matching_state_is_reported_not_raised() -> None:
    state = MappedState.from_smiles(["[CH3:1][OH:2]"])
    result = execute_smirks(state, "[CH3:1][Cl:2]>>[CH3+:1].[Cl-:2]")
    assert not result.ok
    assert result.failure_category == "lhs_state_mismatch"


def test_disagreement_with_stated_state_is_flagged() -> None:
    state = MappedState.from_smiles(["[OH-:1]", "[CH3:2][Br:3]"])
    result = execute_smirks(
        state, "[OH-:1].[CH3:2][Br:3]>>[OH:1][CH3:2].[Br-:3]", expected_resulting_state=["CCO", "[Br-]"]
    )
    assert result.ok
    assert result.smirks_state_agreement is False
    assert result.agreement_detail["missing_from_derived"] == ["CCO"]
    assert result.agreement_detail["extra_in_derived"] == ["CO"]


# ---------------------------------------------------------------------------
# 10.6 Proton transfer (heavy-atom and explicit-H policies)
# ---------------------------------------------------------------------------

_PT_SMIRKS = (
    "[O-:1][H:7].[C:3](=[O:4])([O:5][H:6])[H:8]>>[O:1]([H:7])[H:6].[C:3](=[O:4])([O-:5])[H:8]"
    " |mech:v1;lp:1>6;sigma:6-5>5|"
)


@pytest.mark.parametrize("route", ["smirks", "moves"])
def test_10_6_proton_transfer_heavy_atom_policy(route) -> None:
    state = MappedState.from_smiles(["[OH-:1]", "[CH:3](=[O:4])[OH:5]"])
    assert state.hydrogen_policy == "heavy_atom"
    run = execute_smirks if route == "smirks" else execute_moves
    result = run(state, _PT_SMIRKS, expected_resulting_state=["O", "O=C[O-]"])
    _assert_all_preserved(state, result)
    assert result.smirks_state_agreement is True
    # Proton identity is undefined under this policy: no H atoms carry ids.
    assert all(r.element != "H" for r in result.next_state.records.values())


@pytest.mark.parametrize("route", ["smirks", "moves"])
def test_10_6_proton_transfer_explicit_policy_moves_proton_id(route) -> None:
    state = MappedState.from_smiles(["[O-:1][H:7]", "[C:3](=[O:4])([O:5][H:6])[H:8]"])
    assert state.hydrogen_policy == "explicit"
    proton = state.pid_by_map(6)
    run = execute_smirks if route == "smirks" else execute_moves
    result = run(state, _PT_SMIRKS, expected_resulting_state=["O", "O=C[O-]"])
    _assert_all_preserved(state, result)
    after = result.next_state
    donor, acceptor = after.records[after.pid_by_map(5)], after.records[after.pid_by_map(1)]
    assert after.records[proton].element == "H"
    assert after.records[proton].component == acceptor.component != donor.component


@pytest.mark.parametrize("route", ["smirks", "moves"])
def test_10_6_free_proton_absorbed_and_released_heavy_policy(route) -> None:
    run = execute_smirks if route == "smirks" else execute_moves
    water = MappedState.from_smiles(["[OH2:1]", "[H+:2]"])
    absorbed = run(
        water,
        "[OH2:1].[H+:2]>>[OH2+:1][H:2] |mech:v1;lp:1>2|",
        expected_resulting_state=["[OH3+]"],
    )
    assert absorbed.ok and absorbed.smirks_state_agreement is True
    assert absorbed.lost_ids == [water.pid_by_map(2)]
    released = run(
        absorbed.next_state,
        "[O+:1]([H:2])([H:3])[H:4]>>[O:1]([H:3])[H:4].[H+:2] |mech:v1;sigma:2-1>1|",
        expected_resulting_state=["O", "[H+]"],
    )
    assert released.ok and released.smirks_state_agreement is True
    (new_pid,) = released.new_ids
    assert new_pid not in water.pids()
    assert released.next_state.records[new_pid].provenance == "proton:released"


# ---------------------------------------------------------------------------
# 10.7 Atom addition / removal: fresh monotonic ids, no reuse
# ---------------------------------------------------------------------------


def test_10_7_removed_ids_retire_and_new_atoms_get_fresh_ids() -> None:
    state = MappedState.from_smiles(["[CH3:1][OH2+:2]"])
    removed = execute_smirks(state, "[CH3:1][OH2+:2]>>[CH3+:1]")
    assert removed.ok
    assert removed.lost_ids == [state.pid_by_map(2)]
    assert removed.next_state.retired_pids == [state.pid_by_map(2)]
    assert 2 in removed.next_state.retired_maps

    added = execute_smirks(removed.next_state, "[CH3+:1]>>[CH3:1][Cl:2]")
    assert added.ok
    (new_pid,) = added.new_ids
    assert new_pid > max(state.pids())
    assert new_pid not in removed.lost_ids
    # The retired map number is not reused either; the LLM's [Cl:2] is renumbered.
    assert added.next_state.records[new_pid].map_number != 2
    assert added.next_state.records[new_pid].original_map == 2
    assert added.next_state.pid_by_map(1) == state.pid_by_map(1)


def test_10_7_allocator_is_monotonic() -> None:
    alloc = PersistentAtomIdAllocator()
    issued = [alloc.allocate() for _ in range(5)]
    alloc.observe(3)
    assert alloc.allocate() == 6
    alloc.observe(20)
    assert alloc.allocate() == 21
    assert issued == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# 10.8 Aromatic / Kekulé
# ---------------------------------------------------------------------------

_ARENIUM_SMIRKS = (
    "[CH:1]1=[CH:2][CH:3]=[CH:4][CH:5]=[CH:6]1.[Br+:7]>>[CH:1]1([Br:7])[CH+:2][CH:3]=[CH:4][CH:5]=[CH:6]1"
    " |mech:v1;pi:2-1>7|"
)
_REAROMATIZE_SMIRKS = (
    "[C:1]1([H:9])([Br:7])[CH+:2][CH:3]=[CH:4][CH:5]=[CH:6]1.[OH2:8]"
    ">>[C:1]1([Br:7])=[CH:2][CH:3]=[CH:4][CH:5]=[CH:6]1.[OH2+:8][H:9]"
    " |mech:v1;lp:8>9;sigma:9-1>2|"
)


@pytest.mark.parametrize("route", ["smirks", "moves"])
@pytest.mark.parametrize(
    "benzene",
    ["[cH:1]1[cH:2][cH:3][cH:4][cH:5][cH:6]1", "[CH:1]1=[CH:2][CH:3]=[CH:4][CH:5]=[CH:6]1"],
    ids=["aromatic_input", "kekule_input"],
)
def test_10_8_identity_survives_kekulization_edit_and_rearomatization(route, benzene) -> None:
    run = execute_smirks if route == "smirks" else execute_moves
    state = MappedState.from_smiles([benzene, "[Br+:7]", "[OH2:8]"])
    step1 = run(state, _ARENIUM_SMIRKS, expected_resulting_state=["Br[CH]1[CH+]C=CC=C1", "O"])
    _assert_all_preserved(state, step1)
    assert step1.smirks_state_agreement is True
    step2 = run(step1.next_state, _REAROMATIZE_SMIRKS, expected_resulting_state=["Brc1ccccc1", "[OH3+]"])
    _assert_all_preserved(step1.next_state, step2)
    assert step2.smirks_state_agreement is True
    assert "Brc1ccccc1" in step2.resulting_state  # canonical, aromatic output


# ---------------------------------------------------------------------------
# 10.9 Stereochemistry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "route,smirks",
    [("smirks", "[C:5]=[O:6].[H+:8]>>[C:5]=[OH+:6]"), ("moves", "x>>y |mech:v1;lp:6>8|")],
)
def test_10_9_remote_edit_leaves_stereocenter_alone(route, smirks) -> None:
    state = MappedState.from_smiles(["[CH3:1][C@H:2]([OH:3])[CH2:4][C:5](=[O:6])[CH3:7]", "[H+:8]"])
    before = _cip(state.species, 2)
    run = execute_smirks if route == "smirks" else execute_moves
    result = run(state, smirks)
    assert result.ok, result.error
    assert before is not None and _cip(result.resulting_state_mapped, 2) == before
    assert result.next_state.pid_by_map(2) == state.pid_by_map(2)


@pytest.mark.parametrize(
    "product_center,expected_cip",
    [("[C@@H:2]", "S"), ("[C@H:2]", "R")],
    ids=["retention", "inversion"],
)
def test_10_9_intentional_retention_or_inversion_via_smirks(product_center, expected_cip) -> None:
    state = MappedState.from_smiles(["[Br:1][C@@H:2]([CH3:3])[CH2:4][CH3:5]", "[OH-:6]"])
    assert _cip(state.species, 2) == "S"
    smirks = (
        "[Br:1][C@@H:2]([CH3:3])[CH2:4][CH3:5].[OH-:6]>>"
        f"[Br-:1].[OH:6]{product_center}([CH3:3])[CH2:4][CH3:5]"
    )
    result = execute_smirks(state, smirks)
    assert result.ok, result.error
    # Identity and stereo are independent: same ids, chosen configuration.
    _assert_all_preserved(state, result)
    assert _cip(result.resulting_state_mapped, 2) == expected_cip


@pytest.mark.parametrize(
    "route,smirks",
    [("smirks", "[OH:5].[H+:6]>>[OH2+:5]"), ("moves", "x>>y |mech:v1;lp:5>6|")],
)
def test_10_9_remote_edit_preserves_double_bond_geometry(route, smirks) -> None:
    state = MappedState.from_smiles(["[CH3:1]/[CH:2]=[CH:3]/[CH2:4][OH:5]", "[H+:6]"])
    run = execute_smirks if route == "smirks" else execute_moves
    result = run(state, smirks, expected_resulting_state=["C/C=C/C[OH2+]"])
    assert result.ok and result.smirks_state_agreement is True
    assert "C/C=C/C[OH2+]" in result.resulting_state


# ---------------------------------------------------------------------------
# 10.10 Canonicalization stress: index != identity
# ---------------------------------------------------------------------------


def test_10_10_shuffled_atom_order_keeps_identity() -> None:
    state = MappedState.from_smiles(["[OH-:1]", "[CH3:2][CH2:4][CH:5]([CH3:6])[Br:3]"])
    smirks = "[OH-:1].[CH:5][Br:3]>>[OH:1][CH:5].[Br-:3]"
    reference = execute_smirks(state, smirks)
    assert reference.ok
    rng = random.Random(1234)
    index_by_pid_seen: Dict[int, set] = {}
    for _ in range(8):
        shuffled: List[str] = []
        for smiles in state.species:
            mol = Chem.MolFromSmiles(smiles)
            order = list(range(mol.GetNumAtoms()))
            rng.shuffle(order)
            shuffled.append(Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False))
        rng.shuffle(shuffled)
        twin = MappedState.restore({**state.snapshot(), "species": shuffled})
        twin._refresh_indices()
        assert twin.map_to_pid == state.map_to_pid
        for pid, rec in twin.records.items():
            index_by_pid_seen.setdefault(pid, set()).add((rec.component, rec.current_index))
        result = execute_smirks(twin, smirks)
        assert result.ok
        assert result.identity_map == reference.identity_map
        assert mapped_signature(result.resulting_state_mapped) == mapped_signature(reference.resulting_state_mapped)
    # At least one atom was seen at several (component, index) positions.
    assert any(len(v) > 1 for v in index_by_pid_seen.values())


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

_TRACKED_SETS = ("training_data/eval_set.json", "training_data/practice_eval/practice_set.json")
_MULTISTEP = "training_data/flower_mechanisms_multistep.json"


def _load_cases(relpath: str) -> List[Dict[str, Any]]:
    return json.loads((PROJECT_ROOT / relpath).read_text())


def _steps(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list((case.get("verified_mechanism") or {}).get("steps") or [])


def _explicit_h_bond_deltas(reaction_smirks: str) -> Dict[Tuple[int, int], float]:
    # Independent oracle for reaction_bond_deltas(): keeps mapped H atoms so
    # the benchmark's proton moves are counted.
    params = Chem.SmilesParserParams()
    params.removeHs = False
    core = reaction_smirks.split("|")[0].strip()
    sides = []
    for side in core.split(">>"):
        mol = Chem.MolFromSmiles(side, params)
        bonds = {}
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()
            bonds[(min(a, b), max(a, b))] = bond.GetBondTypeAsDouble()
        sides.append(bonds)
    left, right = sides
    return {
        pair: right.get(pair, 0.0) - left.get(pair, 0.0)
        for pair in set(left) | set(right)
        if abs(right.get(pair, 0.0) - left.get(pair, 0.0)) > 1e-6
    }


def _moves_under_specified(step: Dict[str, Any]) -> bool:
    """True when the mech: block implies different bond changes than the SMIRKS."""
    _core, moves, _details = extract_mechanism_moves(step["reaction_smirks"])
    implied = {tuple(d["pair"]): d["delta"] for d in implied_bond_deltas(moves)}
    return implied != _explicit_h_bond_deltas(step["reaction_smirks"])


def test_reaction_bond_deltas_keeps_hydrogen_bonds_on_benchmark_smirks() -> None:
    """The spike found reaction_bond_deltas dropped mapped H; now fixed
    (see tests/fast/test_reaction_bond_deltas_hydrogens.py)."""
    smirks = "[O-:1][H:7].[C:3](=[O:4])([O:5][H:6])[H:8]>>[O:1]([H:7])[H:6].[C:3](=[O:4])([O-:5])[H:8]"
    observed = {tuple(d["pair"]): d["delta"] for d in reaction_bond_deltas(smirks)}
    assert observed == _explicit_h_bond_deltas(smirks)
    assert observed[(5, 6)] == -1.0 and observed[(1, 6)] == 1.0


def _sweep(relpath: str, route: str, *, policy: str = "auto", every: int = 1) -> Tuple[int, int, List[Dict[str, Any]]]:
    total = passed = 0
    failures: List[Dict[str, Any]] = []
    k = 0
    for case in _load_cases(relpath):
        for step in _steps(case):
            k += 1
            if (k - 1) % every:
                continue
            total += 1
            state = MappedState.from_smiles(step["current_state"], hydrogen_policy=policy)  # type: ignore[arg-type]
            if route == "smirks":
                res = execute_smirks(state, step["reaction_smirks"], expected_resulting_state=step["resulting_state"])
            else:
                res = execute_moves(
                    state,
                    step["electron_pushes"],
                    expected_resulting_state=step["resulting_state"],
                    reaction_smirks=step["reaction_smirks"],
                )
            ok = res.ok and res.smirks_state_agreement is True
            if ok and policy != "heavy_atom":
                ok = mapped_signature(res.resulting_state_mapped) == mapped_signature(step["resulting_state"])
            if ok:
                passed += 1
            else:
                failures.append(
                    {
                        "case": case["id"],
                        "step": step.get("step_index"),
                        "category": res.failure_category or "state_disagreement",
                        "moves_under_specified": _moves_under_specified(step),
                        "error": res.error,
                    }
                )
    return total, passed, failures


# ---------------------------------------------------------------------------
# 10.14 SMIRKS-vs-state agreement on ground truth
# ---------------------------------------------------------------------------

# Measured on 2026-09-23 (see PRD §9 "Spike result"). Route B floors are the
# measured pass rates; every Route B miss is an under-specified mech block.
_ROUTE_B_FLOOR = {
    "training_data/eval_set.json": 0.92,
    "training_data/practice_eval/practice_set.json": 58 / 60,
    _MULTISTEP: 419 / 420,
}


def _benchmark_sets() -> List[str]:
    return list(_TRACKED_SETS) + [_MULTISTEP]


@pytest.mark.parametrize("relpath", _benchmark_sets())
def test_10_14_route_a_reproduces_every_benchmark_step(relpath: str) -> None:
    require_repo_asset(relpath)
    total, passed, failures = _sweep(relpath, "smirks")
    print(f"\n[10.14 route A] {relpath}: {passed}/{total} = {passed / total:.3f}")
    assert total > 0
    assert failures == [], failures[:5]


@pytest.mark.parametrize("relpath", _benchmark_sets())
def test_10_14_route_b_floor_and_failure_categories(relpath: str) -> None:
    require_repo_asset(relpath)
    total, passed, failures = _sweep(relpath, "moves")
    rate = passed / total
    print(f"\n[10.14 route B] {relpath}: {passed}/{total} = {rate:.3f}; failures={failures}")
    assert rate >= _ROUTE_B_FLOOR[relpath] - 1e-9
    assert all(f["moves_under_specified"] for f in failures), failures


@pytest.mark.parametrize("relpath", _TRACKED_SETS)
def test_10_14_route_a_heavy_atom_policy_sample(relpath: str) -> None:
    """Runtime representation: heavy-atom state, benchmark explicit-H SMIRKS."""
    total, passed, failures = _sweep(relpath, "smirks", policy="heavy_atom", every=4)
    print(f"\n[10.14 route A heavy-atom, 1-in-4] {relpath}: {passed}/{total}")
    assert failures == [], failures[:5]


# ---------------------------------------------------------------------------
# 10.11 Multi-step persistence (decisive test for retiring LLM step mapping)
# ---------------------------------------------------------------------------


def _chain(case: Dict[str, Any], route: str, policy: str) -> MappedState:
    steps = _steps(case)
    state = MappedState.from_smiles(steps[0]["current_state"], hydrogen_policy=policy)  # type: ignore[arg-type]
    initial = dict(state.map_to_pid)
    issued_high_water = state.allocator.next_id
    for step in steps:
        if route == "smirks":
            res = execute_smirks(state, step["reaction_smirks"], expected_resulting_state=step["resulting_state"])
        else:
            res = execute_moves(
                state, step["electron_pushes"], expected_resulting_state=step["resulting_state"],
                reaction_smirks=step["reaction_smirks"],
            )
        assert res.ok, (case["id"], step["step_index"], res.error)
        assert res.smirks_state_agreement is True, (case["id"], step["step_index"], res.agreement_detail)
        assert res.duplicate_ids == []
        # Benchmark steps neither create nor destroy atoms.
        assert res.new_ids == [] and res.lost_ids == []
        assert state.allocator.next_id == issued_high_water  # no ids issued, none reused
        if policy == "explicit":
            assert mapped_signature(res.resulting_state_mapped) == mapped_signature(step["resulting_state"])
        state = res.next_state
    # Final mapping equals the benchmark mapping: every map keeps its first id.
    for map_number, pid in state.map_to_pid.items():
        assert initial[map_number] == pid
    return state


def _chain_cases(relpath: str, min_steps: int = 3) -> List[Dict[str, Any]]:
    return [c for c in _load_cases(relpath) if 3 <= len(_steps(c)) <= 8 and len(_steps(c)) >= min_steps]


@pytest.mark.parametrize("policy", ["explicit", "heavy_atom"])
def test_10_11_multistep_route_a_carries_identity_practice_set(policy: str) -> None:
    cases = _chain_cases("training_data/practice_eval/practice_set.json")
    assert len(cases) >= 10
    for case in cases:
        _chain(case, "smirks", policy)


def test_10_11_multistep_route_b_carries_identity_practice_set() -> None:
    cases = [
        c
        for c in _chain_cases("training_data/practice_eval/practice_set.json")
        if not any(_moves_under_specified(s) for s in _steps(c))
    ]
    assert len(cases) >= 8
    for case in cases:
        _chain(case, "moves", "explicit")


def test_10_11_multistep_route_a_carries_identity_flower_multistep() -> None:
    require_repo_asset(_MULTISTEP)
    cases = _chain_cases(_MULTISTEP)
    assert cases
    for case in cases:
        _chain(case, "smirks", "explicit")


# ---------------------------------------------------------------------------
# 10.12 Backtracking identity (mapped_state level)
# ---------------------------------------------------------------------------


def test_10_12_backtrack_restores_exact_ids_and_discarded_ids_do_not_leak() -> None:
    s0 = MappedState.from_smiles(["[CH3:1][OH2+:2]", "[Br-:5]"])
    a = execute_smirks(s0, "[CH3:1][OH2+:2]>>[CH3+:1].[OH2:2]")
    s1 = a.next_state
    snapshot_s1 = s1.snapshot()
    history = {1: snapshot_s1}

    b = execute_smirks(s1, "[CH3+:1]>>[CH3:1][Cl:9]")  # path B introduces a new atom
    (discarded_pid,) = b.new_ids
    s2_snapshot = b.next_state.snapshot()

    # Backtrack to the branch point after A (coordinator-style: the live
    # snapshot no longer matches current_state, history does).
    restored, origin = sync_mapped_loop_state(
        s2_snapshot, current_state=s1.stripped_smiles(), history=history, step_index=1
    )
    assert origin == "history"
    assert restored.map_to_pid == s1.map_to_pid
    assert discarded_pid not in restored.pids()

    c = execute_smirks(restored, "[CH3+:1].[Br-:5]>>[CH3:1][Br:5]")
    assert c.ok
    assert discarded_pid not in c.next_state.pids()
    assert c.next_state.pid_by_map(1) == s0.pid_by_map(1)
    # A new atom after backtracking never reuses the discarded path's id.
    d = execute_smirks(c.next_state, "[CH3:1][Br:5]>>[CH3:1][Br:5].[Cl-:9]")
    (fresh,) = d.new_ids
    assert fresh > discarded_pid


# ---------------------------------------------------------------------------
# 10.13 Serialization / resume (mapped_state level)
# ---------------------------------------------------------------------------


def test_10_13_snapshot_persists_through_sqlite_and_file_and_replays(tmp_path: Path) -> None:
    cases = _chain_cases("training_data/practice_eval/practice_set.json")
    case = next(c for c in cases if len(_steps(c)) >= 4)
    steps = _steps(case)
    uninterrupted = _chain(case, "smirks", "explicit")

    state = MappedState.from_smiles(steps[0]["current_state"])
    for step in steps[:2]:
        state = execute_smirks(state, step["reaction_smirks"]).next_state

    db = sqlite3.connect(tmp_path / "state.db")
    db.execute("create table mapped_state (run_id text, step integer, payload text)")
    db.execute("insert into mapped_state values (?, ?, ?)", ("r1", 2, state.to_json()))
    db.commit()
    (payload,) = db.execute("select payload from mapped_state where run_id='r1' and step=2").fetchone()
    db.close()
    (tmp_path / "state.json").write_text(payload)

    for text in (payload, (tmp_path / "state.json").read_text()):
        resumed = MappedState.from_json(text)
        assert resumed.map_to_pid == state.map_to_pid
        assert resumed.allocator.next_id == state.allocator.next_id
        for step in steps[2:]:
            res = execute_smirks(resumed, step["reaction_smirks"], expected_resulting_state=step["resulting_state"])
            assert res.ok and res.smirks_state_agreement is True
            resumed = res.next_state
        assert resumed.map_to_pid == uninterrupted.map_to_pid
        assert mapped_signature(resumed.species) == mapped_signature(uninterrupted.species)


# ---------------------------------------------------------------------------
# Loop helpers, coordinator hook, harness flag
# ---------------------------------------------------------------------------


def test_advance_mapped_loop_state_records_and_resyncs_on_disagreement() -> None:
    history: Dict[int, Dict[str, Any]] = {}
    snap, record = advance_mapped_loop_state(
        None,
        previous_state=["[OH-]", "CBr"],
        stated_resulting_state=["CO", "[Br-]"],
        reaction_smirks="[OH-:1].[CH3:2][Br:3]>>[OH:1][CH3:2].[Br-:3]",
        history=history,
        step_index=0,
    )
    assert record["smirks_state_agreement"] is True
    assert record["identity_resynced"] is False
    assert record["blocking"] is False
    assert 0 in history
    state = MappedState.restore(snap)
    assert state.signature() == state_signature(["CO", "[Br-]"])

    snap2, record2 = advance_mapped_loop_state(
        snap,
        previous_state=["CO", "[Br-]"],
        stated_resulting_state=["C=O", "[Br-]"],  # LLM claims something else
        reaction_smirks="[CH3:9][OH:8]>>[CH3+:9].[OH-:8]",
        step_index=1,
    )
    assert record2["smirks_state_agreement"] is False
    assert record2["identity_resynced"] is True
    assert MappedState.restore(snap2).signature() == state_signature(["C=O", "[Br-]"])


def _coordinator_state(tmp_path: Path):
    from mechanistic_agent.core.coordinator import RunCoordinator
    from mechanistic_agent.core.db import RunStore

    store = RunStore(tmp_path / "data" / "mechanistic.db")
    run_id = store.create_run(
        mode="unverified",
        input_payload={
            "starting_materials": ["[OH-:1]", "[CH3:2][Br:3]"],
            "products": ["[OH:1][CH3:2]", "[Br-:3]"],
            "temperature_celsius": 25.0,
            "ph": 7.0,
        },
        config={"model": "gpt-4o-mini", "model_family": "openai", "max_steps": 2},
        prompt_bundle_hash="p",
        skill_bundle_hash="s",
        memory_bundle_hash="m",
    )
    coordinator = RunCoordinator(store)
    state = coordinator._build_state(store.get_run_row(run_id))
    return coordinator, store, state


@pytest.mark.parametrize(
    "stated,expected_flag",
    [(["CO", "[Br-]"], True), (["CCO", "[Br-]"], False)],
    ids=["agree", "disagree"],
)
def test_coordinator_records_agreement_without_changing_acceptance(tmp_path: Path, stated, expected_flag) -> None:
    from mechanistic_agent.core.types import BranchCandidate

    coordinator, store, state = _coordinator_state(tmp_path)
    assert state.current_state == ["[OH-]", "CBr"]
    assert state.mapped_seed_species == ["[OH-:1]", "[CH3:2][Br:3]"]
    coordinator._configure_loop_state_mapping(state, None)
    assert state.record_smirks_state_agreement is True
    assert state.loop_state_mapping == "stripped"

    candidate = BranchCandidate(
        rank=1,
        intermediate_smiles=stated[0],
        mechanism_output={
            "reaction_smirks": "[OH-:1].[CH3:2][Br:3]>>[OH:1][CH3:2].[Br-:3] |mech:v1;lp:1>2;sigma:2-3>3|"
        },
        resulting_state=list(stated),
        validation_summary={"passed": True},
    )
    coordinator._apply_candidate(state, candidate)

    assert state.current_state == list(stated)  # acceptance unchanged either way
    assert state.step_index == 1
    events = [e for e in store.list_events(state.run_id) if e["event_type"] == "mechanism_step_accepted"]
    record = events[-1]["payload"]["validation_summary"]["smirks_state_agreement"]
    assert record["smirks_state_agreement"] is expected_flag
    assert record["blocking"] is False
    assert record["input_state_origin"] == "seed"
    assert events[-1]["payload"]["validation_summary"]["passed"] is True


def test_coordinator_hook_can_be_disabled(tmp_path: Path) -> None:
    from mechanistic_agent.core.types import BranchCandidate, HarnessConfig

    coordinator, _store, state = _coordinator_state(tmp_path)
    coordinator._configure_loop_state_mapping(state, HarnessConfig(record_smirks_state_agreement=False))
    candidate = BranchCandidate(rank=1, intermediate_smiles="CO", resulting_state=["CO", "[Br-]"])
    coordinator._apply_candidate(state, candidate)
    assert "smirks_state_agreement" not in candidate.validation_summary
    assert state.mapped_loop_state is None


def test_mapped_mode_sends_mapped_current_state_to_proposal(tmp_path: Path, monkeypatch) -> None:
    import mechanistic_agent.core.tool_executor as tool_executor_module
    from mechanistic_agent.core.subagents import IntermediateAgent
    from mechanistic_agent.core.tool_executor import ToolExecutor
    from mechanistic_agent.core.types import HarnessConfig

    captured: Dict[str, Any] = {}

    def fake_intermediates(**kwargs):
        captured.update(kwargs)
        return "{}"

    monkeypatch.setattr(tool_executor_module, "propose_intermediates", fake_intermediates)
    coordinator, _store, state = _coordinator_state(tmp_path)
    agent = IntermediateAgent(ToolExecutor())

    coordinator._configure_loop_state_mapping(state, HarnessConfig())
    agent.run(state)
    assert captured["current_state"] == ["[OH-]", "CBr"]  # default: stripped
    assert captured["starting_materials"] == ["[OH-]", "CBr"]

    coordinator._configure_loop_state_mapping(state, HarnessConfig(loop_state_mapping="mapped"))
    agent.run(state)
    assert sorted(captured["current_state"]) == sorted(["[OH-:1]", "[CH3:2][Br:3]"])
    assert captured["starting_materials"] == ["[OH-]", "CBr"]  # pre-loop inputs stay stripped
    assert state.mapped_loop_state is not None


def test_harness_config_loop_state_mapping_round_trip() -> None:
    from mechanistic_agent.core.types import HarnessConfig

    default = HarnessConfig.from_dict({"name": "x"})
    assert default.loop_state_mapping == "stripped"
    assert default.record_smirks_state_agreement is True
    assert "loop_state_mapping" not in default.as_dict()

    cfg = HarnessConfig.from_dict({"name": "x", "loop_state_mapping": "mapped", "record_smirks_state_agreement": False})
    again = HarnessConfig.from_dict(cfg.as_dict())
    assert again.loop_state_mapping == "mapped"
    assert again.record_smirks_state_agreement is False
    assert HarnessConfig.from_dict({"loop_state_mapping": "bogus"}).loop_state_mapping == "stripped"


def test_default_harness_files_keep_stripped_loop_state() -> None:
    from mechanistic_agent.core.registries import HarnessRegistry

    registry = HarnessRegistry(PROJECT_ROOT / "harness_versions")
    for name in ("default", "adaptive_default", "permissive_default"):
        cfg = registry.load(name)
        assert cfg.loop_state_mapping == "stripped"
        assert cfg.record_smirks_state_agreement is True


def test_species_signature_matches_explicit_h_mapped_and_plain_forms() -> None:
    assert species_signature("[C:1]([H:2])([H:3])([H:4])[O:5][H:6]") == species_signature("CO")
