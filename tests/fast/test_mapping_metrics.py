"""Benchmark mapping-recall metric (PRD v2 §19 Phase 0)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("rdkit")

from rdkit import Chem  # noqa: E402

from mechanistic_agent.core.mapping_metrics import (  # noqa: E402
    AtomMapping,
    AtomRef,
    build_side_frame,
    compare_mappings,
    compute_run_mapping_agreement,
    mapping_from_llm_mapped_atoms,
    mapping_from_mapped_smiles,
    mapping_from_reaction_smirks,
    mapping_to_llm_mapped_atoms,
)
from mechanistic_agent.scoring import score_snapshot_against_known  # noqa: E402
from mechanistic_agent.smiles_utils import strip_atom_mapping_list  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def _eval_record(index: int = 0) -> dict:
    records = json.loads((REPO_ROOT / "training_data" / "eval_set.json").read_text())
    return records[index]


def _with_pairs(mapping: AtomMapping, pairs: dict) -> AtomMapping:
    return AtomMapping(reactants=mapping.reactants, products=mapping.products, pairs=pairs, source="test")


# ---------------------------------------------------------------------------
# Self-agreement and exact counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("include_hydrogens", [False, True])
def test_benchmark_vs_itself_is_one(include_hydrogens: bool) -> None:
    record = _eval_record()
    ref = mapping_from_mapped_smiles(
        record["starting_materials"], record["products"], include_hydrogens=include_hydrogens
    )
    result = compare_mappings(ref, ref)
    assert result.agreement == 1.0
    assert result.exact_match is True
    assert (result.disagreed, result.unmapped, result.extra) == (0, 0, 0)
    assert result.reference_atoms == len(ref.pairs) > 0


def test_every_committed_eval_record_self_agrees() -> None:
    for record in json.loads((REPO_ROOT / "training_data" / "eval_set.json").read_text()):
        ref = mapping_from_mapped_smiles(record["starting_materials"], record["products"])
        assert compare_mappings(ref, ref).exact_match, record["id"]


def test_perturbed_mapping_has_exact_counts() -> None:
    # Spectator water in the reactants has no product partner: not in the reference.
    ref = mapping_from_mapped_smiles(
        ["[CH3:1][CH2:2][Br:3]", "[Cl-:4]", "[OH2:5]"],
        ["[CH3:1][CH2:2][Cl:4]", "[Br-:3]"],
    )
    assert ref.reactants.smiles == ["CCBr", "[Cl-]", "O"]
    assert ref.products.smiles == ["CCCl", "[Br-]"]
    c_me, c_ch2, br = AtomRef(0, 0), AtomRef(0, 1), AtomRef(0, 2)
    cl, water = AtomRef(1, 0), AtomRef(2, 0)
    assert ref.pairs == {c_me: AtomRef(0, 0), c_ch2: AtomRef(0, 1), br: AtomRef(1, 0), cl: AtomRef(0, 2)}

    perturbed = dict(ref.pairs)
    perturbed[c_me], perturbed[c_ch2] = ref.pairs[c_ch2], ref.pairs[c_me]  # swap CH3/CH2 (not equivalent)
    del perturbed[cl]  # leave Cl unmapped
    perturbed[water] = AtomRef(0, 2)  # map the spectator: extra
    result = compare_mappings(_with_pairs(ref, perturbed), ref)

    assert (result.agreed, result.disagreed, result.unmapped, result.extra) == (1, 2, 1, 1)
    assert result.reference_atoms == 4
    assert result.agreement == pytest.approx(0.25)
    assert result.exact_match is False


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------


def test_equivalent_carbonyl_oxygens_count_as_agreement() -> None:
    # Bicarbonate -> CO2 + hydroxide: the two CO2 oxygens are equivalent.
    ref = mapping_from_mapped_smiles(["[O:1]=[C:2]([OH:3])[O-:4]"], ["[O:1]=[C:2]=[O:3]", "[OH-:4]"])
    swapped = dict(ref.pairs)
    targets = {r: p for r, p in ref.pairs.items() if p.component == 0 and ref.products.elements[p] == "O"}
    (ra, pa), (rb, pb) = sorted(targets.items())
    swapped[ra], swapped[rb] = pb, pa
    assert swapped != ref.pairs
    result = compare_mappings(_with_pairs(ref, swapped), ref)
    assert result.agreement == 1.0 and result.exact_match


def test_para_substituted_ring_symmetry_counts_as_agreement() -> None:
    ref = mapping_from_mapped_smiles(
        ["[CH3:1][c:2]1[cH:3][cH:4][c:5]([OH:8])[cH:6][cH:7]1"],
        ["[CH3:1][c:2]1[cH:3][cH:4][c:5]([O-:8])[cH:6][cH:7]1"],
    )
    classes = ref.products.symmetry_class
    # Mirror the ring: every ortho/meta carbon maps onto its equivalent partner.
    mirrored = {}
    for r_ref, p_ref in ref.pairs.items():
        partners = [q for q in classes if classes[q] == classes[p_ref] and q != p_ref]
        mirrored[r_ref] = partners[0] if partners else p_ref
    assert mirrored != ref.pairs
    for reactant_symmetry in (True, False):
        result = compare_mappings(_with_pairs(ref, mirrored), ref, reactant_symmetry=reactant_symmetry)
        assert result.exact_match, reactant_symmetry


# ---------------------------------------------------------------------------
# LLM format and SMIRKS converters
# ---------------------------------------------------------------------------


def test_llm_format_round_trip_on_real_eval_record() -> None:
    record = _eval_record()
    ref = mapping_from_mapped_smiles(record["starting_materials"], record["products"])
    # Frames built from exactly what the runtime shows the LLM after ingress stripping.
    runtime_reactants = build_side_frame(strip_atom_mapping_list(record["starting_materials"]))
    runtime_products = build_side_frame(strip_atom_mapping_list(record["products"]))
    assert runtime_reactants.smiles == ref.reactants.smiles
    assert runtime_products.smiles == ref.products.smiles

    llm_atoms = mapping_to_llm_mapped_atoms(ref)
    assert llm_atoms[0]["product_atom"].rsplit("#", 1)[0] in runtime_products.smiles
    predicted = mapping_from_llm_mapped_atoms(llm_atoms, reactants=runtime_reactants, products=runtime_products)
    result = compare_mappings(predicted, ref)
    assert result.exact_match and predicted.unresolved_entries == 0

    # The compact form stored by attempt_atom_mapping_for_step resolves identically.
    compact = [
        {
            "product_atom": item["product_atom"],
            "source_smiles": item["source"]["smiles"],
            "source_atom_index": item["source"]["atom_index"],
        }
        for item in llm_atoms
    ]
    compact_pred = mapping_from_llm_mapped_atoms(compact, reactants=runtime_reactants, products=runtime_products)
    assert compact_pred.pairs == predicted.pairs


def test_llm_indices_into_non_canonical_smiles_are_translated() -> None:
    record = _eval_record()
    ref = mapping_from_mapped_smiles(record["starting_materials"], record["products"])
    llm_atoms = mapping_to_llm_mapped_atoms(ref)

    # Rewrite each source SMILES in a different (non-canonical) atom order and re-index.
    rewritten = {}
    for comp, smiles in enumerate(ref.reactants.smiles):
        mol = Chem.MolFromSmiles(smiles)
        new_order = list(reversed(range(mol.GetNumAtoms())))
        renumbered = Chem.RenumberAtoms(mol, new_order)
        written = Chem.MolToSmiles(renumbered, canonical=False)
        written_mol = Chem.MolFromSmiles(written)
        out_order = list(renumbered.GetPropsAsDict(True, True)["_smilesAtomOutputOrder"])
        # canonical index i -> renumbered index new_order.index(i) -> written position
        rewritten[comp] = (written, {i: out_order.index(new_order.index(i)) for i in range(written_mol.GetNumAtoms())})
    for item in llm_atoms:
        comp = item["source"]["molecule_index"]
        written, index_map = rewritten[comp]
        item["source"] = {"molecule_index": comp, "smiles": written, "atom_index": index_map[item["source"]["atom_index"]]}
    assert any(rewritten[c][0] != ref.reactants.smiles[c] for c in rewritten)

    predicted = mapping_from_llm_mapped_atoms(llm_atoms, reactants=ref.reactants, products=ref.products)
    assert compare_mappings(predicted, ref).exact_match


def test_reaction_smirks_converter_matches_mapped_states() -> None:
    step = _eval_record()["verified_mechanism"]["steps"][0]
    for include_hydrogens in (False, True):
        from_smirks = mapping_from_reaction_smirks(step["reaction_smirks"], include_hydrogens=include_hydrogens)
        from_states = mapping_from_mapped_smiles(
            step["current_state"], step["resulting_state"], include_hydrogens=include_hydrogens
        )
        assert from_smirks.pairs == from_states.pairs
        assert compare_mappings(from_smirks, from_states).exact_match


def test_explicit_hydrogen_policy_toggle() -> None:
    record = _eval_record()
    heavy = mapping_from_mapped_smiles(record["starting_materials"], record["products"])
    with_h = mapping_from_mapped_smiles(record["starting_materials"], record["products"], include_hydrogens=True)
    h_pairs = [r for r in with_h.pairs if with_h.reactants.is_hydrogen(r)]
    assert len(h_pairs) > 0
    assert len(with_h.pairs) == len(heavy.pairs) + len(h_pairs)
    assert not any(heavy.reactants.is_hydrogen(r) for r in heavy.pairs)

    # An LLM mapping never names hydrogens: perfect under heavy-atom policy,
    # hydrogens count as unmapped under the explicit-H policy.
    llm_atoms = mapping_to_llm_mapped_atoms(heavy)
    pred_heavy = mapping_from_llm_mapped_atoms(llm_atoms, reactants=heavy.reactants, products=heavy.products)
    pred_h = mapping_from_llm_mapped_atoms(llm_atoms, reactants=with_h.reactants, products=with_h.products)
    assert compare_mappings(pred_heavy, heavy).agreement == 1.0
    result_h = compare_mappings(pred_h, with_h)
    assert result_h.unmapped == len(h_pairs)
    assert result_h.agreement == pytest.approx(len(heavy.pairs) / len(with_h.pairs))
    # Hydrogen ids are symmetry-aware too: permuting H on the same parent still agrees.
    parent_groups: dict = {}
    for r_ref, p_ref in with_h.pairs.items():
        if with_h.reactants.is_hydrogen(r_ref):
            parent_groups.setdefault(with_h.products.symmetry_class[p_ref], []).append(r_ref)
    permuted = dict(with_h.pairs)
    for members in parent_groups.values():
        if len(members) > 1:
            targets = [with_h.pairs[m] for m in members]
            for member, target in zip(members, targets[1:] + targets[:1]):
                permuted[member] = target
    assert compare_mappings(_with_pairs(with_h, permuted), with_h).exact_match


# ---------------------------------------------------------------------------
# Scoring hook (recorded only)
# ---------------------------------------------------------------------------


def _snapshot_for_record(record: dict, llm_atoms: list, *, confidence: float = 0.9) -> dict:
    stripped_sm = strip_atom_mapping_list(record["starting_materials"])
    stripped_pr = strip_atom_mapping_list(record["products"])
    return {
        "input_payload": {
            "starting_materials": stripped_sm,
            "products": stripped_pr,
            "input_boundary": {
                "original_starting_materials": record["starting_materials"],
                "original_products": record["products"],
            },
        },
        "events": [
            {
                "seq": 1,
                "event_type": "mechanism_step_accepted",
                "payload": {
                    "step_index": 1,
                    "current_state": stripped_sm,
                    "resulting_state": stripped_pr,
                    "validation_summary": {
                        "checks": [
                            {"name": "dbe_metadata", "passed": True},
                            {"name": "atom_balance", "passed": True},
                            {"name": "state_progress", "passed": True},
                        ]
                    },
                },
            }
        ],
        "step_outputs": [
            {
                "step_name": "atom_mapping",
                "attempt": 0,
                "output": {"llm_response": {"mapped_atoms": llm_atoms, "confidence": confidence}},
            },
            {
                "step_name": "step_atom_mapping",
                "attempt": 1,
                "output": {
                    "confidence": confidence,
                    "current_state": stripped_sm,
                    "resulting_state": stripped_pr,
                    "raw": {"llm_response": {"mapped_atoms": llm_atoms, "confidence": confidence}},
                },
            },
        ],
    }


def _expected_for_record(record: dict) -> dict:
    return {
        "products": record["products"],
        "known_mechanism": record["known_mechanism"],
        "verified_mechanism": record["verified_mechanism"],
    }


def test_scoring_records_mapping_agreement_without_changing_score() -> None:
    record = _eval_record()
    ref = mapping_from_mapped_smiles(record["starting_materials"], record["products"])
    good_atoms = mapping_to_llm_mapped_atoms(ref)
    bad_atoms = [dict(item) for item in good_atoms]
    bad_atoms[0], bad_atoms[1] = (
        {**bad_atoms[0], "product_atom": good_atoms[1]["product_atom"]},
        {**bad_atoms[1], "product_atom": good_atoms[0]["product_atom"]},
    )
    expected = _expected_for_record(record)

    good = score_snapshot_against_known(_snapshot_for_record(record, good_atoms), expected)
    bad = score_snapshot_against_known(_snapshot_for_record(record, bad_atoms), expected)

    assert good["mapping_agreement"]["available"] is True
    assert good["mapping_agreement"]["global"]["agreement"] == 1.0
    assert good["mapping_agreement"]["steps"][0]["status"] == "scored"
    assert good["step_breakdown"][0]["mapping_agreement"] == 1.0
    assert bad["mapping_agreement"]["global"]["agreement"] < 1.0
    assert bad["step_breakdown"][0]["mapping_agreement"] < 1.0
    # Recorded only: the composite score and validity are untouched.
    assert good["score"] == bad["score"]
    assert good["step_breakdown"][0]["validity_score"] == bad["step_breakdown"][0]["validity_score"]


def test_mapping_agreement_unavailable_without_benchmark() -> None:
    result = compute_run_mapping_agreement({"step_outputs": []}, {"known_mechanism": {"steps": []}})
    assert result == {"available": False, "reason": "no_benchmark_mapping"}


# ---------------------------------------------------------------------------
# Backfill tooling (read-only, degrades gracefully)
# ---------------------------------------------------------------------------


def test_backfill_degrades_gracefully_when_db_missing(tmp_path: Path, capsys) -> None:
    from mechanistic_agent.mapping_audit import audit_mapping_agreement
    import importlib.util

    report = audit_mapping_agreement(tmp_path / "missing.db")
    assert report.available is False
    assert "not found" in report.message

    spec = importlib.util.spec_from_file_location(
        "backfill_mapping_agreement", REPO_ROOT / "scripts" / "backfill_mapping_agreement.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.main(["--db-path", str(tmp_path / "missing.db")]) == 0
    assert "not found" in capsys.readouterr().out
    assert not (tmp_path / "missing.db").exists()


def test_readonly_store_never_writes(tmp_path: Path) -> None:
    from mechanistic_agent.core.db import ReadOnlyRunStore, RunStore
    from mechanistic_agent.mapping_audit import audit_mapping_agreement

    db = tmp_path / "mechanistic.db"
    RunStore(db)  # create schema
    before = db.stat().st_mtime_ns
    store = ReadOnlyRunStore(db)
    assert store.list_eval_runs() == []
    with pytest.raises(sqlite3.OperationalError):
        store.create_eval_run(
            eval_set_id="x", run_group_name="g", model="m", model_name="m", model_family="f",
            thinking_level=None, harness_bundle_hash="", metadata=None, status="running",
        )
    report = audit_mapping_agreement(db)
    assert report.available is True and report.rows == []
    assert db.stat().st_mtime_ns == before
