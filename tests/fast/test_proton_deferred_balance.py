"""balance_mode = proton_deferred: only a residual of whole protons may be deferred."""

from __future__ import annotations

from mechanistic_agent.core.coordinator import RunCoordinator


def _attempt(current, resulting):
    output = {"current_state": current, "resulting_state": resulting, "contains_target_product": False}
    validation = {"passed": False, "checks": [{"name": "atom_balance", "passed": False},
                                              {"name": "state_progress", "passed": True}]}
    return ({"rank": 1, "intermediate_smiles": resulting[0]},
            {"last_validation": validation, "failed_checks": ["atom_balance"], "mechanism_output": output})


def test_coerce_balance_mode() -> None:
    assert RunCoordinator._coerce_balance_mode("proton_deferred") == "proton_deferred"
    assert RunCoordinator._coerce_balance_mode("Deferred") == "deferred"
    assert RunCoordinator._coerce_balance_mode("anything") == "strict"


def test_proton_only_residual() -> None:
    # carbonyl protonated by an acid the state did not carry: +H+
    assert RunCoordinator._proton_only_residual({"current_state": ["CC(C)=O"], "resulting_state": ["CC(C)=[OH+]"]})
    # acetic acid conjured from nothing: heavy-atom residual
    assert not RunCoordinator._proton_only_residual(
        {"current_state": ["CC(C)=O"], "resulting_state": ["CC(C)=O", "CC(=O)O"]}
    )
    # hydroxide from nothing: H and charge move but O too
    assert not RunCoordinator._proton_only_residual({"current_state": ["CC(C)=O"], "resulting_state": ["CC(C)=O", "[OH-]"]})


def test_proton_deferred_skips_heavy_atom_residuals(tmp_path) -> None:
    from mechanistic_agent.core.db import RunStore

    coordinator = RunCoordinator(RunStore(tmp_path / "db.sqlite"))
    coordinator._validation_check_passed = lambda validation, name: True  # type: ignore[method-assign]
    proton = _attempt(["CC(C)=O"], ["CC(C)=[OH+]"])
    conjured = _attempt(["CC(C)=O"], ["CC(C)=O", "CC(=O)O"])
    assert coordinator._best_balance_pending_candidate(candidate_attempts=[conjured], proton_only=True) is None
    assert coordinator._best_balance_pending_candidate(candidate_attempts=[conjured, proton], proton_only=True) is not None
    # plain deferred still accepts the heavy-atom residual
    assert coordinator._best_balance_pending_candidate(candidate_attempts=[conjured], proton_only=False) is not None
