"""``reaction_bond_deltas`` must keep mapped explicit hydrogens.

Before the fix it parsed each species with RDKit defaults, which remove
``[H:n]`` atoms, so every proton move vanished from the recorded
``observed_bond_deltas`` metadata while ``implied_bond_deltas`` (from the
``mech:`` block) still listed it.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Set, Tuple

import pytest

pytest.importorskip("rdkit")

from rdkit import Chem  # noqa: E402

from mechanistic_agent.core.mechanism_moves import (  # noqa: E402
    extract_mechanism_moves,
    implied_bond_deltas,
    reaction_bond_deltas,
)
from mechanistic_agent.tools import predict_mechanistic_step  # noqa: E402
from mechanistic_agent.core.validators import ALL_VALIDATOR_IDS, _run_python_validators  # noqa: E402

from _optional_assets import PROJECT_ROOT  # noqa: E402

_BENCHMARK_SETS = ("training_data/eval_set.json", "training_data/practice_eval/practice_set.json")

# practice_set flower_254799 step 3: fluoride deprotonates an oxonium.
_PROTON_TRANSFER = (
    "[C:1](=[O:3])([F:4])[O+:10]([C:9]([C:6]([F:5])([F:7])[F:8])([H:11])[H:12])[H:13].[F-:2]"
    ">>[C:1](=[O:3])([F:4])[O:10][C:9]([C:6]([F:5])([F:7])[F:8])([H:11])[H:12].[F:2][H:13]"
    " |mech:v1;lp:2>13;sigma:13-10>10|"
)


def _as_map(deltas: Iterable[Dict[str, Any]]) -> Dict[Tuple[int, int], float]:
    return {tuple(d["pair"]): d["delta"] for d in deltas}


def _default_parse_deltas(reaction_smirks: str) -> Dict[Tuple[int, int], float]:
    """The pre-fix behaviour: RDKit default parse (mapped H removed)."""
    core = reaction_smirks.split("|")[0].strip()
    sides = []
    for side in core.split(">>"):
        bonds: Dict[Tuple[int, int], float] = {}
        for token in side.split("."):
            mol = Chem.MolFromSmiles(token)
            if mol is None:
                continue
            for bond in mol.GetBonds():
                a, b = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()
                if a > 0 and b > 0:
                    bonds[(min(a, b), max(a, b))] = bond.GetBondTypeAsDouble()
        sides.append(bonds)
    left, right = sides
    return {
        p: right.get(p, 0.0) - left.get(p, 0.0)
        for p in set(left) | set(right)
        if abs(right.get(p, 0.0) - left.get(p, 0.0)) > 1e-6
    }


def _hydrogen_maps(reaction_smirks: str) -> Set[int]:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.sanitize = False
    maps: Set[int] = set()
    for side in reaction_smirks.split("|")[0].strip().split(">>"):
        mol = Chem.MolFromSmiles(side, params)
        maps |= {a.GetAtomMapNum() for a in mol.GetAtoms() if a.GetAtomicNum() == 1}
    return maps


def _benchmark_steps() -> List[Tuple[str, Dict[str, Any]]]:
    steps = []
    for relpath in _BENCHMARK_SETS:
        for case in json.loads((PROJECT_ROOT / relpath).read_text()):
            for step in (case.get("verified_mechanism") or {}).get("steps") or []:
                steps.append((f"{case['id']}:{step.get('step_index')}", {**step, "_set": relpath}))
    return steps


def test_reaction_bond_deltas_reports_proton_transfer_on_benchmark_step() -> None:
    """Before the fix this returned ``[]``: both changed bonds involve H13."""
    assert _default_parse_deltas(_PROTON_TRANSFER) == {}  # the old, wrong answer
    observed = _as_map(reaction_bond_deltas(_PROTON_TRANSFER))
    assert observed == {(2, 13): 1.0, (10, 13): -1.0}
    _core, moves, _details = extract_mechanism_moves(_PROTON_TRANSFER)
    assert observed == _as_map(implied_bond_deltas(moves))


def test_reaction_bond_deltas_heavy_atom_input_unchanged() -> None:
    smirks = "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] |mech:v1;lp:4>2;sigma:2-3>3|"
    observed = _as_map(reaction_bond_deltas(smirks))
    assert observed == _default_parse_deltas(smirks) == {(2, 3): -1.0, (2, 4): 1.0}
    aromatic = "[cH:1]1[cH:2][cH:3][cH:4][cH:5][cH:6]1.[Br:7][Br:8]>>[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[Br:7].[Br:8]"
    assert _as_map(reaction_bond_deltas(aromatic)) == _default_parse_deltas(aromatic)


def test_reaction_bond_deltas_only_adds_hydrogen_pairs_across_benchmark() -> None:
    """Every benchmark step: heavy-atom pairs match the old output exactly;
    the only additions are bonds to mapped H."""
    added_h_steps = 0
    for label, step in _benchmark_steps():
        smirks = step["reaction_smirks"]
        h_maps = _hydrogen_maps(smirks)
        new = _as_map(reaction_bond_deltas(smirks))
        old = _default_parse_deltas(smirks)
        heavy_new = {p: v for p, v in new.items() if not (set(p) & h_maps)}
        assert heavy_new == old, label
        if new != old:
            added_h_steps += 1
    assert added_h_steps >= 30  # 36 of 160 steps carry an H bond change


def test_reaction_bond_deltas_agrees_with_mech_moves_on_benchmark() -> None:
    """With H kept, observed and implied deltas agree wherever the mech block
    is complete; the residue is steps whose mech block omits moves (the
    SMIRKS changes a superset of the pairs the moves name)."""
    agree = 0
    for label, step in _benchmark_steps():
        _core, moves, _details = extract_mechanism_moves(step["reaction_smirks"])
        implied = _as_map(implied_bond_deltas(moves))
        observed = _as_map(reaction_bond_deltas(step["reaction_smirks"]))
        if observed == implied:
            agree += 1
        else:
            assert set(implied.items()) <= set(observed.items()), label
    assert agree >= 145  # 149 of 160 (was 123 before the fix)


def _validator_sample() -> List[Tuple[str, Dict[str, Any]]]:
    """Deterministic sample: every 4th eval_set step plus every benchmark step
    whose SMIRKS changes a bond to mapped H (proton/hydride moves), capped."""
    eval_steps = [item for item in _benchmark_steps() if item[1]["_set"] == _BENCHMARK_SETS[0]]
    chosen = {label: step for label, step in eval_steps[::4]}
    for label, step in _benchmark_steps():
        smirks = step["reaction_smirks"]
        if _as_map(reaction_bond_deltas(smirks)) != _default_parse_deltas(smirks):
            chosen.setdefault(label, step)
    return sorted(chosen.items())[:150]


def _run_step(step: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(
        predict_mechanistic_step(
            step_index=0,
            current_state=step["current_state"],
            target_products=step.get("target_products") or [],
            electron_pushes=step["electron_pushes"],
            reaction_smirks=step["reaction_smirks"],
            predicted_intermediate=step.get("predicted_intermediate"),
            resulting_state=step.get("resulting_state"),
        )
    )


def _old_reaction_bond_deltas(smirks: str) -> List[Dict[str, Any]]:
    return [{"pair": p, "delta": d} for p, d in sorted(_default_parse_deltas(smirks).items())]


def test_validator_outcome_unchanged_before_after_on_benchmark_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each sampled step goes through predict_mechanistic_step and the step
    validators twice: with the pre-fix (H-dropping) parser patched in, and
    with the fix. ``observed_bond_deltas`` is metadata only, so every other
    payload field and every validator check must be identical."""
    import mechanistic_agent.tools as tools_mod

    monkeypatch.setenv("MECHANISTIC_CHEMISTRY_BACKEND", "python")  # no rdkit-agent subprocesses
    sample = _validator_sample()
    assert len(sample) >= 50
    changed_metadata = 0
    for label, step in sample:
        after = _run_step(step)
        with monkeypatch.context() as m:
            m.setattr(tools_mod, "reaction_bond_deltas", _old_reaction_bond_deltas)
            before = _run_step(step)
        bev_before = before["bond_electron_validation"]
        bev_after = after["bond_electron_validation"]
        if bev_before.get("observed_bond_deltas") != bev_after.get("observed_bond_deltas"):
            changed_metadata += 1
        bev_before.pop("observed_bond_deltas", None)
        bev_after.pop("observed_bond_deltas", None)
        assert before == after, label
        assert after["status"] == "accepted" and bev_after["valid"] is True, label
        for policy in ("strict", "soft"):
            checks_before = _run_python_validators(before, dbe_policy=policy, active=ALL_VALIDATOR_IDS).checks
            checks_after = _run_python_validators(after, dbe_policy=policy, active=ALL_VALIDATOR_IDS).checks
            assert [(c.name, c.passed) for c in checks_before] == [(c.name, c.passed) for c in checks_after], label
    assert changed_metadata >= 30
