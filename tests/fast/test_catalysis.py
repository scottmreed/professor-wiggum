"""Catalyst tracking, completion tolerance, step cap and cycle projection (PRD_catalysis_handling P3/P4/P6)."""
from mechanistic_agent.core.catalysis import (
    catalysis_tolerated,
    catalyst_regeneration,
    identify_catalysts,
    is_catalytic_run,
    project_catalytic_cycle,
)
from mechanistic_agent.core.coordinator import RunCoordinator
from mechanistic_agent.core.mechanism_audit import audit_mechanism
from mechanistic_agent.core.types import CATALYTIC_MAX_STEPS, DEFAULT_MAX_STEPS
from mechanistic_agent.quality_scoring import QualityStep, _targets_reached
from mechanistic_agent.skeleton_alignment import collapse_skeleton_path
from mechanistic_agent.smiles_utils import assess_target_product_state
from organometallic_chem.catalytic_cycle import validate_catalytic_cycle
from skills.mechanistic.metal_center_validation.validator import validate_metal_center

L = "[P](C)(C)C"  # PMe3 drawn covalently, as FlowER and models draw it
PD0 = f"[Pd]({L}){L}"
PD0_L4 = f"[Pd]({L})({L})({L}){L}"
PHBR = "Brc1ccccc1"
TOLB = "Cc1ccc(B(O)O)cc1"
OH = "[OH-]"
OA = f"c1ccccc1[Pd](Br)({L}){L}"
TM = f"c1ccccc1[Pd](c1ccc(C)cc1)({L}){L}"
BIARYL = "Cc1ccc(-c2ccccc2)cc1"
BORATE = "OB(O)O"
BR = "[Br-]"

STARTING = [PD0, PHBR, TOLB, OH]
STEPS = [
    {"step_index": 1, "current_state": [PD0, PHBR, TOLB, OH], "resulting_state": [OA, TOLB, OH]},
    {"step_index": 2, "current_state": [OA, TOLB, OH], "resulting_state": [TM, BORATE, BR]},
    {"step_index": 3, "current_state": [TM, BORATE, BR], "resulting_state": [BIARYL, PD0, BORATE, BR]},
]


def _with_metal_center(steps):
    out = []
    for step in steps:
        check = validate_metal_center(step["current_state"], step["resulting_state"])
        out.append({**step, "metal_center": check.details if check else None})
    return out


def test_identify_catalysts_roles():
    assert [c["role"] for c in identify_catalysts(STARTING)] == ["catalyst"]
    assert identify_catalysts(["CCO", "Cl[Zn]Cl"])[0]["role"] == "unknown"
    assert is_catalytic_run(STARTING)
    assert not is_catalytic_run(["CCO", "Cl[Zn]Cl"])
    assert not is_catalytic_run(["CCO", "CC(=O)O"])


def test_regeneration_variants():
    states = [s["resulting_state"] for s in STEPS]
    assert catalyst_regeneration(STARTING, states)["catalysts"][0]["via"] == "starting_species"
    # Pd(PMe3)4 precatalyst, ligand loss drawn implicitly: ends as Pd(0)L2, same Pd(0).
    pre = [PD0_L4, PHBR, TOLB, OH]
    states_pre = [[OA, L, L, TOLB, OH], [TM, L, L, BORATE, BR], [BIARYL, PD0, L, L, BORATE, BR]]
    record = catalyst_regeneration(pre, states_pre)["catalysts"][0]
    assert record["regenerated"] and record["via"] == "same_oxidation_state"
    # Stuck mid-cycle: Pd(II) at the end.
    stuck = catalyst_regeneration(STARTING, [[OA, TOLB, OH], [TM, BORATE, BR]])
    assert stuck["unregenerated"] == [PD0]


def test_audit_grades_unregenerated_catalyst_approximate():
    audit = audit_mechanism(starting=STARTING, targets=[BIARYL], steps=STEPS[:2])
    assert audit["grade"] == "approximate"
    assert {"type": "unregenerated_catalyst", "species": [PD0]} in audit["findings"]
    full = audit_mechanism(starting=STARTING, targets=[BIARYL], steps=STEPS)
    assert full["catalyst_regeneration"]["applies"] and not full["catalyst_regeneration"]["unregenerated"]
    assert not any(f["type"] == "unregenerated_catalyst" for f in full["findings"])


def test_completion_tolerates_catalyst_forms_ligands_and_byproducts():
    assert catalysis_tolerated(PD0, STARTING) == "catalyst_form"
    assert catalysis_tolerated("CP(C)C", STARTING) == "released_ligand"
    assert catalysis_tolerated(BORATE, STARTING) == "coupling_byproduct"
    assert catalysis_tolerated("CCCl", STARTING) is None
    assert catalysis_tolerated(PD0, ["CCO"]) is None
    pre = [PD0_L4, PHBR, TOLB, OH]
    state = assess_target_product_state(
        current_state=[TM, "CP(C)C", BORATE, BR],
        resulting_state=[BIARYL, PD0, "CP(C)C", "CP(C)C", BORATE, BR],
        target_products=[BIARYL],
        starting_materials=pre,
    )
    assert state["contains_target_product"], state


def test_catalyst_form_in_products_is_not_the_main_target():
    pre = [PD0_L4, PHBR, TOLB, OH]
    path = [QualityStep(1, pre, [BIARYL, PD0, BORATE, BR, "CP(C)C", "CP(C)C"])]
    targets = _targets_reached(path, [BIARYL, PD0, BORATE], pre)
    assert targets["main_product"] == "Cc1ccc(-c2ccccc2)cc1"
    assert targets["all_reached"]


def test_cycle_projection_is_closed_and_validates():
    projection = project_catalytic_cycle(_with_metal_center(STEPS), STARTING, provenance={"run_id": "r1"})
    assert projection["closed"]
    assert [s["kind"] for s in projection["cycle"]["steps"]] == [
        "oxidative_addition", "transmetalation", "reductive_elimination",
    ]
    assert validate_catalytic_cycle(projection["cycle"]) == []
    assert projection["provenance"] == {"source": "predicted", "run_id": "r1"}
    assert project_catalytic_cycle(STEPS, ["CCO"]) is None
    open_cycle = project_catalytic_cycle(_with_metal_center(STEPS[:2]), STARTING)
    assert not open_cycle["closed"]


def test_ligand_dissociation_collapses_in_skeleton_path():
    path = [[PD0_L4, PHBR], [f"[Pd]({L})({L}){L}", "CP(C)C", PHBR], [PD0, "CP(C)C", "CP(C)C", PHBR], [OA, "CP(C)C", "CP(C)C"]]
    assert len(collapse_skeleton_path(path)) == 2


def test_step_cap_resolution():
    resolve = RunCoordinator._resolve_max_steps
    assert resolve({}, STARTING) == (CATALYTIC_MAX_STEPS, "catalytic")
    assert resolve({}, ["CCO"]) == (DEFAULT_MAX_STEPS, "default")
    assert resolve({"max_steps": 9}, STARTING) == (9, "explicit")
    assert resolve({"max_steps": None}, ["CCO"]) == (DEFAULT_MAX_STEPS, "default")


def test_observatory_projects_metal_center_and_catalytic_cycle():
    from mechanistic_agent.core.observatory import build_observatory

    events = []

    def add(kind, payload):
        events.append({"seq": len(events) + 1, "event_type": kind, "step_name": None, "ts": 0.0, "payload": payload})

    for step in STEPS:
        idx = step["step_index"]
        check = validate_metal_center(step["current_state"], step["resulting_state"])
        cid = f"c{idx}-r1-x"
        add("candidate_validation_result", {
            "step_index": idx, "candidate_id": cid, "candidate_rank": 1, "accepted": True,
            "failed_checks": [], "resulting_state": step["resulting_state"],
            "metal_center": check.details,
        })
        add("mechanism_step_accepted", {
            "step_index": idx, "candidate_id": cid, "candidate_rank": 1, "acceptance_kind": "validated",
            "validation_summary": {"passed": True},
            "current_state": step["current_state"], "resulting_state": step["resulting_state"],
        })
    view = build_observatory(events, run_id="r1", run_input={"starting_materials": STARTING, "products": [BIARYL]})
    assert view["accepted_path"][0]["metal_center"]["pairs"][0]["consistent_kinds"] == ["oxidative_addition"]
    cycle = view["catalytic_cycle"]
    assert cycle["renderable"] and cycle["validation_errors"] == []
    assert cycle["provenance"]["source"] == "predicted"
    organic = build_observatory([], run_id="r2", run_input={"starting_materials": ["CCO"], "products": ["CC=O"]})
    assert organic["catalytic_cycle"] is None
