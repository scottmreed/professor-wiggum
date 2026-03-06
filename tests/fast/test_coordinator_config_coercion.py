from __future__ import annotations

from pathlib import Path

from mechanistic_agent.core.coordinator import RunCoordinator
from mechanistic_agent.core.db import RunStore


def test_build_state_coerces_config_types_safely(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "mechanistic.db")
    run_id = store.create_run(
        mode="unverified",
        input_payload={"starting_materials": ["CCO"], "products": ["CC=O"]},
        config={
            "model": "gpt-5",
            "functional_groups_enabled": "false",
            "intermediate_prediction_enabled": "true",
            "reproposal_on_repeat_failure": "0",
            "candidate_rescue_enabled": "yes",
            "max_steps": "not-an-int",
            "max_runtime_seconds": None,
            "dbe_policy": "STRICT",
            "reaction_template_policy": "UNKNOWN",
        },
        prompt_bundle_hash="a",
        skill_bundle_hash="b",
        memory_bundle_hash="c",
    )
    row = store.get_run_row(run_id)
    assert row is not None

    coordinator = RunCoordinator(store)
    state = coordinator._build_state(row)

    assert state.run_config.functional_groups_enabled is False
    assert state.run_config.intermediate_prediction_enabled is True
    assert state.run_config.reproposal_on_repeat_failure is False
    assert state.run_config.candidate_rescue_enabled is True
    assert state.run_config.max_steps == 10
    assert state.run_config.max_runtime_seconds == 240.0
    assert state.run_config.dbe_policy == "strict"
    assert state.run_config.reaction_template_policy == "auto"
