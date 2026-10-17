"""Atom balance counts hydrogen the same way on both sides, in both backends.

Regression fixtures come from the 2026-10 threaded_unified 92-case run
(Opus 5.5, keyless bridge): mechuspto31k-17799 and mechuspto31k-02453 paused with
``mechanism_retry_exhausted`` because a correct protonation by a bare ``[H+]``
failed as ``H: 1->0`` (the bare proton was an explicit H atom, the bonded one an
implicit H) while the candidate rescue, counting implicit H, reported
``balanced``. flower-test-016681 paused because a valid tetrahedral adduct was
declared an invalid species with ``state_progress_unchanged_starting_materials``.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from mechanistic_agent.core import chemistry_backend as backend_module
from mechanistic_agent.core.chemistry_backend import ChemistryBackendConfig, resolve_rdkit_cli_command
from mechanistic_agent.core.validators import validate_mechanism_step_output
from mechanistic_agent.tools import _canonicalise_candidate_smiles, predict_missing_reagents_for_candidate
from skills.mechanistic.atom_balance_validation.validator import validate_atom_balance

# mechuspto31k-17799, step 1, rank 1: ketone protonation by a bare proton.
CASE_17799_CURRENT = [
    "CN1CCN(S(=O)(=O)c2cccc(C(=O)NN)c2)CC1",
    "CC(=O)c1cc(Cl)ccc1O",
    "[H+]",
]
CASE_17799_RESULTING = [
    "CN1CCN(S(=O)(=O)c2cccc(C(=O)NN)c2)CC1",
    "CC(=[OH+])c1cc(Cl)ccc1O",
]
CASE_17799_SMIRKS = (
    "[CH3:1][C:2](=[O:3])[c:4]1[cH:5][c:6]([Cl:7])[cH:8][cH:9][c:10]1[OH:11].[H+:12]>>"
    "[CH3:1][C:2](=[O+:3][H:12])[c:4]1[cH:5][c:6]([Cl:7])[cH:8][cH:9][c:10]1[OH:11] |mech:v1;lp:3>12|"
)
# mechuspto31k-17799, step 1, rank 2: amine addition with O-protonation.
CASE_17799_R2_RESULTING = ["CN1CCN(S(=O)(=O)c2cccc(C(=O)N[NH2+]C(C)(O)c3cc(Cl)ccc3O)c2)CC1"]

# mechuspto31k-02453, step 2: alkoxide protonation by a bare proton.
CASE_02453_CURRENT = [
    "O=C(CCc1cccc(CCc2ccccc2)c1)N1CCc2ccoc2C1",
    "CC[NH+](CC)C[O-]",
    "[H+]",
]
CASE_02453_RESULTING = [
    "O=C(CCc1cccc(CCc2ccccc2)c1)N1CCc2ccoc2C1",
    "CC[NH+](CC)CO",
]

# flower-test-016681, step 1: aniline adds to the acyl chloride.
CASE_016681_CURRENT = [
    "O=C(Cl)c1ccc(C(=O)N2Cc3ccccc3Cc3ccccc32)cc1",
    "CCN(CC)CC",
    "Cc1ccccc1N",
    "ClCCl",
]
CASE_016681_ADDUCT = "Cc1ccccc1[NH2+]C([O-])(Cl)c1ccc(C(=O)N2Cc3ccccc3Cc3ccccc32)cc1"
CASE_016681_RESULTING = [CASE_016681_ADDUCT, "CCN(CC)CC", "ClCCl"]
CASE_016681_SMIRKS = (
    "[O:1]=[C:2]([Cl:3])[c:4]1[cH:5][cH:6][c:7]([C:8](=[O:9])[N:10]2[CH2:11][c:12]3[cH:13][cH:14][cH:15]"
    "[cH:16][c:17]3[CH2:18][c:19]3[cH:20][cH:21][cH:22][cH:23][c:24]32)[cH:25][cH:26]1."
    "[CH3:27][c:28]1[cH:31][cH:32][cH:33][cH:34][c:29]1[NH2:30]>>"
    "[O-:1][C:2]([Cl:3])([NH2+:30][c:29]1[cH:34][cH:33][cH:32][cH:31][c:28]1[CH3:27])[c:4]1[cH:5][cH:6]"
    "[c:7]([C:8](=[O:9])[N:10]2[CH2:11][c:12]3[cH:13][cH:14][cH:15][cH:16][c:17]3[CH2:18][c:19]3[cH:20]"
    "[cH:21][cH:22][cH:23][c:24]32)[cH:25][cH:26]1 |mech:v1;lp:30>2;pi:2-1>1|"
)

# rdkit-agent 0.1.1's answer to `check {"smiles": ...}`: routed to state-progress mode.
MISROUTED_SMILES_CHECK = {
    "overall_pass": False,
    "summary": "Validation failed: state_progress",
    "mode": "state_progress",
    "backend": "rdkit_cli",
    "method": "check.state_progress",
    "checks": [
        {
            "name": "state_progress",
            "pass": False,
            "error_code": "state_progress_unchanged_starting_materials",
            "message": "Unchanged starting materials detected or resulting state did not progress",
            "details": {"current_state": [], "resulting_state": []},
        }
    ],
    "failed_checks": [
        {
            "name": "state_progress",
            "error_code": "state_progress_unchanged_starting_materials",
            "message": "Unchanged starting materials detected or resulting state did not progress",
        }
    ],
    "failed_check_names": ["state_progress"],
    "fix_suggestions": [],
}


def _cli_available() -> bool:
    return bool(resolve_rdkit_cli_command(ChemistryBackendConfig.from_config({"chemistry_backend": "rdkit_cli"})).command_parts)


def _step_payload(current: List[str], resulting: List[str], smirks: str) -> Dict[str, Any]:
    return {
        "current_state": list(current),
        "resulting_state": list(resulting),
        "reaction_smirks": smirks,
        "unchanged_starting_materials_detected": False,
        "resulting_state_changed": True,
        "bond_electron_validation": {"valid": True, "total_delta": 0},
    }


@pytest.fixture
def python_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MECHANISTIC_CHEMISTRY_BACKEND", "python")


@pytest.mark.parametrize(
    "current,resulting",
    [
        (CASE_17799_CURRENT, CASE_17799_RESULTING),
        (CASE_17799_CURRENT[:2] + ["[H+]"], CASE_17799_R2_RESULTING),
        (CASE_02453_CURRENT, CASE_02453_RESULTING),
        (["C=C", "[H][H]"], ["CC"]),  # explicit H2 consumed into implicit H
        (["CC(=O)C", "[H-]"], ["CC([O-])C"]),  # bare hydride
    ],
)
def test_bare_hydrogen_species_do_not_unbalance_a_step(python_backend: None, current: List[str], resulting: List[str]) -> None:
    check = validate_atom_balance(current, resulting)
    assert check.passed, check.details
    assert check.details["hydrogen_policy"] == "heavy_atoms_only"
    assert "H" not in check.details.get("current_counts", {})


def test_heavy_atom_loss_still_fails(python_backend: None) -> None:
    check = validate_atom_balance(CASE_17799_CURRENT, CASE_17799_RESULTING[:1])
    assert not check.passed
    assert check.details["deficit"] == {"C": 8, "Cl": 1, "O": 2}
    assert "H" not in check.details["deficit"]


def test_step_validation_python_backend_passes_bare_proton_step(python_backend: None) -> None:
    result = validate_mechanism_step_output(
        _step_payload(CASE_17799_CURRENT, CASE_17799_RESULTING, CASE_17799_SMIRKS),
        dbe_policy="soft",
        run_config={"chemistry_backend": "python"},
    )
    atom = next(c for c in result.checks if c.name == "atom_balance")
    assert atom.passed, atom.details


@pytest.mark.skipif(not _cli_available(), reason="rdkit-agent CLI not installed")
def test_step_validation_rdkit_cli_backend_agrees(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MECHANISTIC_CHEMISTRY_BACKEND", "rdkit_cli")
    for current, resulting in ((CASE_17799_CURRENT, CASE_17799_RESULTING), (CASE_02453_CURRENT, CASE_02453_RESULTING)):
        result = validate_mechanism_step_output(
            _step_payload(current, resulting, CASE_17799_SMIRKS),
            dbe_policy="soft",
            run_config={"chemistry_backend": "rdkit_cli"},
        )
        atom = next(c for c in result.checks if c.name == "atom_balance")
        assert atom.details["chemistry_backend"]["backend_used"] == "rdkit_cli"
        assert atom.passed, atom.details
        # The CLI's own (heavy-atom) verdict is kept next to the Python one.
        assert atom.details["backend_atom_balance_passed"] is True


def _fake_missing_reagents(suggested_reactants: List[str], suggested_products: List[str], status: str = "success"):
    def _fake(*, starting_materials: List[str], products: List[str], conditions_guidance: Optional[str] = None) -> str:
        return json.dumps(
            {"status": status, "suggested_reactants": suggested_reactants, "suggested_products": suggested_products}
        )

    return _fake


def test_rescue_counts_hydrogen_like_the_validator(python_backend: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mechanistic_agent.tools.predict_missing_reagents", _fake_missing_reagents([], [], "balanced"))
    payload = json.loads(
        predict_missing_reagents_for_candidate(
            current_state=CASE_17799_CURRENT,
            resulting_state=CASE_17799_RESULTING,
            failed_checks=["atom_balance"],
        )
    )
    validator = validate_atom_balance(CASE_17799_CURRENT, CASE_17799_RESULTING)
    assert payload["hydrogen_policy"] == validator.details["hydrogen_policy"]
    assert payload["balance_diagnostics"]["balanced"] == validator.passed is True
    assert payload["balance_diagnostics"]["include_hydrogens"] is False


def test_rescue_status_reflects_the_additions_it_kept(python_backend: None, monkeypatch: pytest.MonkeyPatch) -> None:
    current = ["CC(=O)Cl", "O"]
    resulting = ["CC(=O)O"]
    # CC(=O)Cl + H2O -> CC(=O)OH is short one Cl; adding HCl closes it.
    monkeypatch.setattr("mechanistic_agent.tools.predict_missing_reagents", _fake_missing_reagents([], ["Cl"]))
    ok = json.loads(predict_missing_reagents_for_candidate(current_state=current, resulting_state=resulting, failed_checks=["atom_balance"]))
    assert ok["status"] == "success"
    assert ok["post_additions_balance"]["balanced"] is True

    # A Cl-free product does not close the deficit -> partial, whatever the model said.
    monkeypatch.setattr("mechanistic_agent.tools.predict_missing_reagents", _fake_missing_reagents([], ["CO"]))
    partial = json.loads(
        predict_missing_reagents_for_candidate(current_state=current, resulting_state=resulting, failed_checks=["atom_balance"])
    )
    assert partial["status"] == "partial"
    assert partial["model_status"] == "success"
    assert partial["post_additions_balance"]["balanced"] is False


def _force_fake_cli(monkeypatch: pytest.MonkeyPatch) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []

    def _fake_resolve(_cfg: Any) -> backend_module.RdkitCliResolution:
        return backend_module.RdkitCliResolution(command_parts=["rdkit-agent"], source="test")

    def _fake_invoke(*, command_parts: List[str], payload: Dict[str, Any], timeout_seconds: float) -> Dict[str, Any]:
        calls.append(dict(payload))
        if payload.get("mechanism-step"):
            return {
                "overall_pass": True,
                "mode": "mechanism_step",
                "checks": [
                    {"name": "atom_balance", "pass": True, "details": {}},
                    {"name": "dbe_metadata", "pass": True, "details": {}},
                    {"name": "state_progress", "pass": True, "details": {}},
                ],
                "failed_check_names": [],
            }
        return dict(MISROUTED_SMILES_CHECK)

    def _no_repair(smiles: str, *, backend_config: Any = None):  # noqa: ANN001
        return None, {"error": "repair-smiles timed out"}

    monkeypatch.setattr(backend_module, "resolve_rdkit_cli_command", _fake_resolve)
    monkeypatch.setattr(backend_module, "_invoke_rdkit_cli", _fake_invoke)
    monkeypatch.setattr("mechanistic_agent.tools._attempt_repair_smiles_via_rdkit_cli", _no_repair)
    monkeypatch.setenv("MECHANISTIC_CHEMISTRY_BACKEND", "rdkit_cli")
    return calls


def test_misrouted_cli_smiles_check_falls_back_to_rdkit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _force_fake_cli(monkeypatch)
    canonical, details = _canonicalise_candidate_smiles(CASE_016681_ADDUCT, backend_config={"chemistry_backend": "rdkit_cli"})
    assert calls, "the CLI path must have been exercised"
    assert canonical, details
    assert details["rdkit_cli_mode_mismatch"]["got"] == "state_progress"
    assert "error_code" not in details


def test_tetrahedral_adduct_is_a_valid_new_species(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_fake_cli(monkeypatch)
    result = validate_mechanism_step_output(
        _step_payload(CASE_016681_CURRENT, CASE_016681_RESULTING, CASE_016681_SMIRKS),
        dbe_policy="soft",
        run_config={"chemistry_backend": "rdkit_cli"},
    )
    by_name = {c.name: c for c in result.checks}
    atom = by_name["atom_balance"]
    assert atom.passed, atom.details
    assert atom.details["invalid_species"] == []
    assert by_name["state_progress"].passed
    assert result.passed


@pytest.mark.skipif(not _cli_available(), reason="rdkit-agent CLI not installed")
def test_tetrahedral_adduct_with_real_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MECHANISTIC_CHEMISTRY_BACKEND", "rdkit_cli")
    canonical, details = _canonicalise_candidate_smiles(CASE_016681_ADDUCT, backend_config={"chemistry_backend": "rdkit_cli"})
    assert canonical, details
    check = validate_atom_balance(CASE_016681_CURRENT, CASE_016681_RESULTING)
    assert check.passed, check.details
