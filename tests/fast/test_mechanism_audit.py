"""Whole-mechanism audit: catalysts, conjugate pairs, flag resolution, efficiency findings."""

from __future__ import annotations

from mechanistic_agent.core.mechanism_audit import audit_mechanism, chosen_path_from_events

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


def test_flagged_step_that_drops_a_leaving_water_is_resolved() -> None:
    # Deferred rerun of flower_002647: step 4 eliminated water but never carried it,
    # and the product then shows up with H3O+ instead of the water target.
    hydrazide = "CC(=O)NN"
    acid = "O=C(O)c1ccc2occc2c1"
    product = "CC(=O)NNC(=O)c1ccc2occc2c1"
    tetra = "CC(=O)NNC(O)(O)c1ccc2occc2c1"
    oxonium = "CC(=O)NNC(O)([OH2+])c1ccc2occc2c1"
    protonated = "CC(=O)NNC(=[OH+])c1ccc2occc2c1"
    audit = audit_mechanism(
        starting=[hydrazide, acid],
        targets=[product, "O"],
        steps=[
            _step(1, [hydrazide, acid], [tetra]),
            _step(2, [tetra], [oxonium, "O"], adds=["[OH3+]"]),  # acid catalyst added by rescue
            _step(3, [oxonium, "O"], [protonated, "O"], flag=True),  # leaving water not carried
            _step(4, [protonated, "O"], [product, "[OH3+]"]),
        ],
    )
    assert audit["grade"] == "reconciled"
    assert audit["dropped_species"] == [{"species": "O", "count": 1}]
    assert audit["flags"][0]["resolution"] == "resolved_dropped_species"


def test_dropped_organic_fragment_is_not_excused() -> None:
    audit = audit_mechanism(
        starting=["CCOC(C)=O", "N"],
        targets=["CC(N)=O"],
        steps=[_step(1, ["CCOC(C)=O", "N"], ["CC(N)=O"], flag=True)],  # ethanol vanished
    )
    assert audit["grade"] == "approximate"
    assert audit["unresolved_steps"] == [1]
