"""Whole-mechanism audit: catalysts, conjugate pairs, flag resolution, efficiency findings."""

from __future__ import annotations

from mechanistic_agent.core.mechanism_audit import (
    audit_mechanism,
    chosen_path_from_events,
    excess_reagent_equivalents,
)

ACID = "CC(=O)O"
HYDRAZINE = "NNc1ccc([N+](=O)[O-])cc1"
PRODUCT = "CC(=O)NNc1ccc([N+](=O)[O-])cc1"
TETRA = "CC(O)(O)NNc1ccc([N+](=O)[O-])cc1"
OXONIUM = "CC(O)([OH2+])NNc1ccc([N+](=O)[O-])cc1"


def _step(i, current, resulting, *, adds=None, flag=False):
    return {
        "step_index": i,
        "current_state": current,
        "resulting_state": resulting,
        "rescue_additions": {"add_reactants": adds or [], "add_products": []},
        "balance_flag": {"reason": "balance_pending"} if flag else None,
    }


def test_acid_shuttle_is_reconciled_as_catalyst() -> None:
    # Mapped starting SMILES (as runs store them) must compare equal to unmapped path SMILES.
    audit = audit_mechanism(
        starting=["[CH3:1][C:2](=[O:3])[OH:4]", HYDRAZINE],
        targets=[PRODUCT],
        steps=[
            _step(1, [ACID, HYDRAZINE], [TETRA]),
            _step(2, [TETRA], [OXONIUM, "CC(=O)[O-]"], adds=[ACID]),
            _step(3, [OXONIUM, "CC(=O)[O-]"], [PRODUCT, "O", ACID]),
        ],
    )
    assert audit["grade"] == "reconciled"
    assert audit["balanced"] is True
    assert audit["catalysts"] == [ACID]
    assert audit["reagents_added"] == []
    assert audit["targets_reached"] is True


def test_flagged_proton_transfer_resolves_as_conjugate_pair() -> None:
    audit = audit_mechanism(
        starting=[ACID, "O"],
        targets=["CC(=O)[O-]"],
        steps=[
            _step(1, [ACID], ["CC(=O)[O-]"], flag=True),  # proton dropped from the state
            _step(2, ["CC(=O)[O-]", "O"], ["CC(=O)[O-]", "[OH3+]"]),
        ],
    )
    assert audit["balanced"] is True
    assert audit["grade"] == "reconciled"
    assert audit["flags"][0]["resolution"] == "resolved_conjugate_pair"
    assert {"left": ACID, "right": "CC(=O)[O-]", "neutral_parent": ACID} in audit["conjugate_pairs"]
    assert audit["unresolved_steps"] == []


def test_conjured_carbon_stays_unresolved() -> None:
    # Easy run 1, flower_135501: the Michael adduct gained a CH2 from nowhere.
    audit = audit_mechanism(
        starting=["C=CC#N", "[cH-]1cccc1"],
        targets=["N#CCCC1C=CC=C1"],
        steps=[_step(1, ["C=CC#N", "[cH-]1cccc1"], ["N#C[CH-]CCC1C=CC=C1"], flag=True)],
    )
    assert audit["grade"] == "approximate"
    assert audit["net_delta"]["C"] == 1
    assert audit["unresolved_steps"] == [1]


def test_clean_mechanism_is_exact() -> None:
    audit = audit_mechanism(
        starting=["CCBr", "[Cl-]"],
        targets=["CCCl"],
        steps=[_step(1, ["CCBr", "[Cl-]"], ["CCCl", "[Br-]"])],
    )
    assert audit["grade"] == "exact"
    assert audit["flags"] == [] and audit["findings"] == []


def test_efficiency_findings_undo_and_mergeable_proton_transfers() -> None:
    audit = audit_mechanism(
        starting=["CCN", ACID],
        targets=["CC[NH3+]", "CC(=O)[O-]"],
        steps=[
            _step(1, ["CCN", ACID], ["CC[NH3+]", "CC(=O)[O-]"]),
            _step(2, ["CC[NH3+]", "CC(=O)[O-]"], ["CCN", ACID]),
            _step(3, ["CCN", ACID], ["CC[NH3+]", "CC(=O)[O-]"]),
        ],
    )
    types = [f["type"] for f in audit["findings"]]
    assert "undo_step" in types
    assert "repeated_state" in types
    assert "mergeable_proton_transfers" in types
    assert audit["grade"] == "exact"


def test_invalid_species_grade() -> None:
    audit = audit_mechanism(starting=["C(("], targets=["C"], steps=[])
    assert audit["grade"] == "invalid_species"


def test_chosen_path_drops_steps_abandoned_by_backtrack() -> None:
    def ev(seq, step, tag):
        return {"seq": seq, "event_type": "mechanism_step_accepted", "payload": {"step_index": step, "tag": tag}}

    events = [ev(1, 1, "a"), ev(2, 2, "a"), ev(3, 3, "a"), ev(4, 2, "b"), ev(5, 3, "b"), ev(6, 4, "b")]
    path = chosen_path_from_events(events)
    assert [(p["step_index"], p["tag"]) for p in path] == [(1, "a"), (2, "b"), (3, "b"), (4, "b")]


def test_unaccounted_catalytic_proton_is_reconciled_and_reported() -> None:
    # Deferred hard rerun of flower_025913: acid catalysis drawn without the acid, so the
    # product arrives with a stray H3O+ (net residual = exactly one proton).
    audit = audit_mechanism(
        starting=[ACID, HYDRAZINE],
        targets=[PRODUCT],
        steps=[
            _step(1, [ACID, HYDRAZINE], [TETRA]),
            _step(2, [TETRA], ["CC(=[OH+])NNc1ccc([N+](=O)[O-])cc1", "O"]),
            _step(3, ["CC(=[OH+])NNc1ccc([N+](=O)[O-])cc1", "O"], [PRODUCT, "[OH3+]"]),
        ],
    )
    assert audit["grade"] == "reconciled"
    assert audit["proton_reconciled"] is True
    assert audit["net_delta"] == {"+": 1, "H": 1}
    assert any(f["type"] == "unaccounted_proton" for f in audit["findings"])


def test_heavy_atom_residual_is_never_proton_reconciled() -> None:
    audit = audit_mechanism(
        starting=["CCO"],
        targets=["CCOC"],
        steps=[_step(1, ["CCO"], ["CCOC"])],
    )
    assert audit["grade"] == "approximate"
    assert audit["proton_reconciled"] is False


# TFA Boc deprotection (hard tier flower_064575): TFA is the solvent, so the model carries
# a second CF3COOH the harness pool holds only one equivalent of.
TFA = "O=C(O)C(F)(F)F"
TFA_ANION = "O=C([O-])C(F)(F)F"
BOC = "CC(C)(C)OC(=O)NC1CCC(F)(F)CC1"
BOC_H = "CC(C)(C)OC(=[OH+])NC1CCC(F)(F)CC1"
AMINE = "NC1CCC(F)(F)CC1"


def test_excess_equivalent_of_pool_reagent_is_detected() -> None:
    found = excess_reagent_equivalents(
        [TFA, BOC], [BOC_H, TFA_ANION, TFA], {TFA: "starting_material"}
    )
    assert found == {"species": TFA, "count": 1, "source": "starting_material", "proton_residual": {}}


def test_dropped_excess_equivalent_is_the_mirror_image() -> None:
    found = excess_reagent_equivalents(
        [AMINE, "O=C=O", "C[C+](C)C", TFA_ANION, TFA],
        [AMINE, "O=C=O", "C=C(C)C", TFA],
        {TFA: "starting_material"},
    )
    assert found is not None
    assert (found["species"], found["count"]) == (TFA, -1)


def test_excess_reagent_tolerates_proton_bookkeeping_on_the_extra_equivalent() -> None:
    # The extra equivalent drawn as the trifluoroacetate: residual = TFA minus one proton.
    found = excess_reagent_equivalents([TFA, BOC], [BOC_H, TFA_ANION, TFA_ANION], {TFA: "starting_material"})
    assert found is not None
    assert (found["species"], found["count"]) == (TFA, 1)
    assert found["proton_residual"] == {"+": -1, "H": -1}


def test_heavy_residual_that_is_not_whole_equivalents_is_not_excess_reagent() -> None:
    # CF3 conjured without the carboxylate: not a whole TFA.
    assert excess_reagent_equivalents([TFA, BOC], [BOC_H, TFA_ANION, "FC(F)(F)"], {TFA: "x"}) is None
    # Whole equivalent of a species that is not in the pool.
    assert excess_reagent_equivalents([BOC], [BOC, "CC(=O)O"], {TFA: "x"}) is None
    # Exactly one methanol's atoms, but conjured into the ester rather than carried intact.
    assert excess_reagent_equivalents(["CC(=O)O"], ["CC(=O)OC", "O"], {"CO": "starting_material"}) is None
    # Deleting the only equivalent of a reactant is not dropping an excess one.
    assert excess_reagent_equivalents(["CCBr", "[Cl-]"], ["CCBr"], {"[Cl-]": "starting_material"}) is None
    # Balanced steps have nothing to reconcile.
    assert excess_reagent_equivalents([TFA, BOC], [BOC_H, TFA_ANION], {TFA: "x"}) is None


def test_final_state_carrying_extra_solvent_equivalent_is_excess_reconciled() -> None:
    audit = audit_mechanism(
        starting=[TFA, BOC],
        targets=[AMINE, "O=C=O", "C=C(C)C"],
        steps=[
            _step(1, [TFA, BOC], [BOC_H, TFA_ANION, TFA]),
            _step(2, [BOC_H, TFA_ANION, TFA], [AMINE, "O=C=O", "C=C(C)C", TFA, TFA]),
        ],
    )
    assert audit["balanced"] is True
    assert audit["grade"] == "reconciled"
    assert audit["excess_reagent_reconciled"] == {
        "species": TFA, "count": 1, "source": "starting_material", "proton_residual": {},
    }
    assert any(f["type"] == "excess_reagent" and f["species"] == TFA for f in audit["findings"])


def test_steps_reconciled_as_excess_reagent_grade_reconciled_and_are_listed() -> None:
    step1 = _step(1, [TFA, BOC], [BOC_H, TFA_ANION, TFA])
    step1["excess_reagent_reconciled"] = {"species": TFA, "count": 1, "source": "starting_material"}
    step2 = _step(2, [BOC_H, TFA_ANION, TFA], [AMINE, "O=C=O", "C=C(C)C", TFA])
    step2["excess_reagent_reconciled"] = {"species": TFA, "count": -1, "source": "starting_material"}
    audit = audit_mechanism(starting=[TFA, BOC], targets=[AMINE], steps=[step1, step2])
    assert audit["grade"] == "reconciled"
    assert audit["excess_reagent_reconciled"] is None
    assert audit["excess_reagent_steps"] == [
        {"step_index": 1, "species": TFA, "count": 1, "source": "starting_material"},
        {"step_index": 2, "species": TFA, "count": -1, "source": "starting_material"},
    ]


def test_conjured_carbon_is_never_excess_reconciled() -> None:
    audit = audit_mechanism(starting=["CCO"], targets=["CCOC"], steps=[_step(1, ["CCO"], ["CCOC"])])
    assert audit["grade"] == "approximate"
    assert audit["excess_reagent_reconciled"] is None


def test_vanished_starting_material_is_a_deficit_not_excess() -> None:
    # One listed water never appears again: the net equation is short a reagent, which
    # the audit must not wave through as an "excess" equivalent.
    audit = audit_mechanism(starting=["N#CO", "O"], targets=["N#CO"], steps=[_step(1, ["N#CO", "O"], ["N#CO"])])
    assert audit["grade"] == "approximate"
    assert audit["excess_reagent_reconciled"] is None
