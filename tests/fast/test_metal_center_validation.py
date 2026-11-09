"""Metal-centre validator (PRD_catalysis_handling P2) and its dispatcher wiring."""
from types import SimpleNamespace

from mechanistic_agent.core.validators import _append_metal_center_check
from mechanistic_agent.core.types import StepValidationResult
from skills.mechanistic.metal_center_validation.validator import infer_step_kinds, validate_metal_center

PD_L2_CL2 = "[Cl:1][Pd:2]([Cl:3])([P:4]([CH3:5])([CH3:6])[CH3:7])[P:8]([CH3:9])([CH3:10])[CH3:11]"
PD_L_CL2 = "[Cl:1][Pd:2]([Cl:3])[P:4]([CH3:5])([CH3:6])[CH3:7]"
PME3 = "[P:8]([CH3:9])([CH3:10])[CH3:11]"

PD0_L2 = "[Pd]([P](C)(C)C)[P](C)(C)C"
PHBR = "Brc1ccccc1"
OA_PRODUCT = "c1ccccc1[Pd](Br)([P](C)(C)C)[P](C)(C)C"


def test_no_metal_change_adds_no_check():
    assert validate_metal_center(["CCO"], ["CC=O"]) is None
    assert validate_metal_center([PD0_L2, "CCO"], [PD0_L2, "CC=O"]) is None


def test_oxidative_addition_is_consistent():
    check = validate_metal_center([PD0_L2, PHBR], [OA_PRODUCT], declared_kind="oxidative_addition")
    assert check.passed
    pair = check.details["pairs"][0]
    assert (pair["delta_oxidation_state"], pair["delta_valence_electrons"]) == (2, 2)
    assert pair["consistent_kinds"] == ["oxidative_addition"]
    assert check.details["findings"] == []


def test_declared_kind_mismatch_warns_when_soft_and_fails_when_strict():
    soft = validate_metal_center([PD_L2_CL2], [PD_L_CL2, PME3], declared_kind="oxidative_addition")
    assert soft.passed
    assert soft.details["findings"][0]["code"] == "declared_kind_mismatch"
    assert "labelled oxidative addition" in soft.details["warnings"][0]
    strict = validate_metal_center(
        [PD_L2_CL2], [PD_L_CL2, PME3], declared_kind="oxidative_addition", policy="strict"
    )
    assert not strict.passed
    assert "Pd(II), d8, 16 e to Pd(II), d8, 14 e" in strict.details["error"]


def test_impossible_oxidation_state_fails_even_when_soft():
    # Pd bearing six X ligands would be Pd(VI).
    pd6 = "Cl[Pd](Cl)(Cl)(Cl)(Cl)Cl"
    check = validate_metal_center(["Cl[Pd](Cl)(Cl)Cl", "ClCl"], [pd6])
    assert not check.passed
    assert check.details["findings"][-1]["code"] == "impossible_oxidation_state"


def test_ambiguous_notation_abstains():
    check = validate_metal_center([PD0_L2, "C=C"], ["C1C[Pd]1([P](C)(C)C)[P](C)(C)C"])
    assert check.passed
    assert "metallacyclopropane" in check.details["abstained"]


def test_unpaired_metal_species_abstain():
    check = validate_metal_center([PD0_L2, PD0_L2, PHBR, PHBR], [OA_PRODUCT, OA_PRODUCT, "C"])
    assert check is None or check.passed


def test_infer_step_kinds():
    assert infer_step_kinds(2, 2) == ["oxidative_addition"]
    assert set(infer_step_kinds(0, 2)) == {"beta_hydride_elimination", "ligand_association"}
    assert infer_step_kinds(4, 0) == []


def test_dispatcher_appends_check_by_policy():
    payload = {"current_state": [PD0_L2, PHBR], "resulting_state": [OA_PRODUCT],
               "metal_step_kind": "transmetalation"}
    result = _append_metal_center_check(
        StepValidationResult(checks=[]), payload=payload, run_config=SimpleNamespace(metal_center_policy="strict")
    )
    assert [c.name for c in result.checks] == ["metal_center"]
    assert not result.passed
    off = _append_metal_center_check(
        StepValidationResult(checks=[]), payload=payload, run_config=SimpleNamespace(metal_center_policy="off")
    )
    assert off.checks == []
    organic = _append_metal_center_check(
        StepValidationResult(checks=[]),
        payload={"current_state": ["CCO"], "resulting_state": ["CC=O"]},
        run_config=None,
    )
    assert organic.checks == []
