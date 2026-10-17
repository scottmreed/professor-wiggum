"""Harness-level RunConfig defaults are honoured and strict by default."""
from __future__ import annotations

from pathlib import Path

from mechanistic_agent.core.coordinator import RunCoordinator
from mechanistic_agent.core.registries import HarnessRegistry
from mechanistic_agent.core.types import HarnessConfig, RunConfig, RunInput, RunState

_ROOT = Path(__file__).resolve().parents[2]


class _EventStore:
    def __init__(self) -> None:
        self.events = []

    def append_event(self, run_id, event_type, payload, step_name=None):  # noqa: ANN001
        self.events.append({"run_id": run_id, "event_type": event_type, "payload": payload, "step_name": step_name})


def _state(**config_overrides) -> RunState:
    return RunState(
        run_id="run-1",
        mode="unverified",
        run_input=RunInput(starting_materials=["CCBr", "[Cl-]"], products=["CCCl", "[Br-]"]),
        run_config=RunConfig(model="gpt-5", **config_overrides),
    )


def test_run_config_dataclass_defaults_to_strict_validation() -> None:
    assert RunConfig(model="gpt-5").proceed_on_validation_failure is False


def test_default_harness_is_strict_and_permissive_variant_exists() -> None:
    registry = HarnessRegistry(_ROOT / "harness_versions")
    default = registry.load("default")
    assert default.run_config_defaults["proceed_on_validation_failure"] is False
    permissive = registry.load("permissive_default")
    assert permissive.name == "permissive_default"
    assert permissive.run_config_defaults["proceed_on_validation_failure"] is True
    # Same module graph as default.
    assert [m.id for m in permissive.pre_loop_modules] == [m.id for m in default.pre_loop_modules]
    assert [m.id for m in permissive.post_step_modules] == [m.id for m in default.post_step_modules]


def test_harness_config_round_trips_run_config_defaults() -> None:
    cfg = HarnessConfig.from_dict({"name": "x", "run_config_defaults": {"proceed_on_validation_failure": True}})
    assert cfg.run_config_defaults == {"proceed_on_validation_failure": True}
    assert cfg.as_dict()["run_config_defaults"] == {"proceed_on_validation_failure": True}


def test_coordinator_applies_harness_defaults_only_for_unset_keys() -> None:
    store = _EventStore()
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]
    harness = HarnessConfig.from_dict(
        {
            "name": "permissive_default",
            "run_config_defaults": {
                "proceed_on_validation_failure": True,
                "candidate_rescue_enabled": False,
                "not_a_run_config_key": 123,
            },
        }
    )

    state = _state()
    applied = coordinator._apply_harness_run_config_defaults(state, harness, raw_config={})
    assert applied == {"proceed_on_validation_failure": True, "candidate_rescue_enabled": False}
    assert state.run_config.proceed_on_validation_failure is True
    assert state.run_config.candidate_rescue_enabled is False
    assert any(ev["event_type"] == "harness_run_config_defaults_applied" for ev in store.events)

    # An explicit per-run value wins over the harness default.
    state2 = _state(proceed_on_validation_failure=False)
    applied2 = coordinator._apply_harness_run_config_defaults(
        state2, harness, raw_config={"proceed_on_validation_failure": False}
    )
    assert "proceed_on_validation_failure" not in applied2
    assert state2.run_config.proceed_on_validation_failure is False


def test_balance_pending_soft_advance_is_gated_by_flag() -> None:
    """With both flags off (strict balance), a candidate failing only atom_balance is never soft-accepted."""
    store = _EventStore()
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]
    attempts = [
        (
            {"rank": 1, "intermediate_smiles": "CCCl"},
            {
                "last_validation": {"checks": [{"name": "atom_balance", "passed": False}, {"name": "state_progress", "passed": True}]},
                "failed_checks": ["atom_balance"],
                "mechanism_output": {"resulting_state": ["CCCl", "[Br-]", "X"], "current_state": ["CCBr", "[Cl-]"]},
            },
        )
    ]
    # The selector itself still finds the candidate...
    assert coordinator._best_balance_pending_candidate(candidate_attempts=attempts) is not None
    # ...but the loop only consults it when the flag is on (source-level guard).
    import inspect

    src = inspect.getsource(RunCoordinator._run_mechanism_loop)
    assert "state.run_config.proceed_on_validation_failure" in src
    assert 'or state.run_config.balance_mode in {"deferred", "proton_deferred"}' in src


def test_threaded_unified_run_resolves_harness_retry_budget() -> None:
    """loop_module.max_retries_per_candidate (3) reaches the coordinator's retry loop."""
    registry = HarnessRegistry(_ROOT / "harness_versions")
    harness = registry.load("threaded_unified")
    assert harness.loop_module["max_retries_per_candidate"] == 3
    store = _EventStore()
    coordinator = RunCoordinator(store=store)  # type: ignore[arg-type]

    state = _state()
    assert state.run_config.retry_same_candidate_max == 1  # dataclass default
    applied = coordinator._apply_harness_run_config_defaults(state, harness, raw_config={"harness_name": "threaded_unified"})
    assert state.run_config.retry_same_candidate_max == 3
    assert applied["retry_same_candidate_max"] == 3
    event = next(ev for ev in store.events if ev["event_type"] == "harness_run_config_defaults_applied")
    assert event["payload"]["sources"]["retry_same_candidate_max"] == "loop_module.max_retries_per_candidate"


def test_explicit_retry_budget_beats_harness_values() -> None:
    harness = HarnessConfig.from_dict(
        {
            "name": "x",
            "loop_module": {"max_retries_per_candidate": 3},
            "run_config_defaults": {"retry_same_candidate_max": 2},
        }
    )
    coordinator = RunCoordinator(store=_EventStore())  # type: ignore[arg-type]

    # run_config_defaults beats loop_module.
    state = _state()
    coordinator._apply_harness_run_config_defaults(state, harness, raw_config={})
    assert state.run_config.retry_same_candidate_max == 2

    # An explicit run value (CLI flag / API field / stored config key) beats both.
    explicit = _state(retry_same_candidate_max=1)
    applied = coordinator._apply_harness_run_config_defaults(explicit, harness, raw_config={"retry_same_candidate_max": 1})
    assert explicit.run_config.retry_same_candidate_max == 1
    assert "retry_same_candidate_max" not in applied


def test_api_request_leaves_retry_budget_to_the_harness() -> None:
    from mechanistic_agent.api.schemas import CreateRunRequest

    request = CreateRunRequest(starting_materials=["CCO"], products=["CC=O"])
    assert request.retry_same_candidate_max is None
    assert CreateRunRequest(starting_materials=["CCO"], products=["CC=O"], retry_same_candidate_max=2).retry_same_candidate_max == 2
