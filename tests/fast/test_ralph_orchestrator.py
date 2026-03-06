from __future__ import annotations

from pathlib import Path

from mechanistic_agent.core.coordinator import RunCoordinator
from mechanistic_agent.core.db import RunStore
from mechanistic_agent.core.ralph_orchestrator import RalphOrchestrator
from mechanistic_agent.core.types import HarnessConfig, ModuleSpec


def _orchestrator(tmp_path: Path) -> RalphOrchestrator:
    store = RunStore(tmp_path / "mechanistic.db")
    coordinator = RunCoordinator(store)
    return RalphOrchestrator(store=store, coordinator=coordinator)


def test_completion_promise_builtins_pass(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path)
    snapshot = {
        "status": "completed",
        "events": [
            {"event_type": "run_completed"},
            {"event_type": "target_products_detected"},
        ],
        "step_outputs": [
            {
                "step_name": "mechanism_synthesis",
                "output": {"contains_target_product": True},
                "validation": {"passed": True},
            }
        ],
    }
    ok, details = orchestrator._evaluate_completion_promise(
        completion_promise="target_products_reached && flow_node:run_complete",
        snapshot=snapshot,
    )
    assert ok is True
    assert details["target_products_reached"] is True
    assert details["flow_node:run_complete"] is True


def test_mutation_invalid_smiles_tunes_loop_module(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path)
    harness = HarnessConfig(
        name="default",
        loop_module={"id": "mechanism_step_proposal", "max_candidates": 3, "max_retries_per_candidate": 3},
    )
    snapshot = {
        "events": [
            {"event_type": "run_failed", "payload": {"reason": "proposal_invalid_smiles_loop"}},
        ]
    }
    mutated, actions, diff = orchestrator._mutate_harness(
        harness=harness,
        latest_snapshot=snapshot,
        allow_validator_mutation=False,
    )
    assert actions
    assert int((mutated.loop_module or {}).get("max_candidates") or 0) >= 4
    assert int((mutated.loop_module or {}).get("max_retries_per_candidate") or 0) >= 4
    assert diff["action_count"] >= 1


def test_mutation_can_disable_failed_validator_when_allowed(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path)
    harness = HarnessConfig(
        name="default",
        post_step_modules=[
            ModuleSpec(
                id="bond_electron_validation",
                label="Bond/Electron",
                kind="deterministic",
                phase="post_step",
                enabled=True,
            ),
            ModuleSpec(
                id="atom_balance_validation",
                label="Atom balance",
                kind="deterministic",
                phase="post_step",
                enabled=True,
            ),
        ],
    )
    snapshot = {
        "events": [
            {
                "event_type": "mechanism_retry_failed",
                "payload": {"failed_checks": ["bond_electron_validation", "bond_electron_validation"]},
            }
        ]
    }
    mutated, actions, _ = orchestrator._mutate_harness(
        harness=harness,
        latest_snapshot=snapshot,
        allow_validator_mutation=True,
    )
    target = next((m for m in mutated.post_step_modules if m.id == "bond_electron_validation"), None)
    assert target is not None
    assert target.enabled is False
    assert any(action.get("type") == "validator_toggle" for action in actions)


def test_mutation_uses_babysit_votes_as_signal(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path)
    harness = HarnessConfig(
        name="default",
        loop_module={"id": "mechanism_step_proposal", "max_candidates": 3},
    )
    mutated, actions, _ = orchestrator._mutate_harness(
        harness=harness,
        latest_snapshot={"events": []},
        allow_validator_mutation=False,
        votes=[{"vote": "B"}, {"vote": "B"}, {"vote": "A"}],
    )
    assert int((mutated.loop_module or {}).get("max_candidates") or 0) >= 4
    assert any(action.get("type") == "babysit_bias" for action in actions)
