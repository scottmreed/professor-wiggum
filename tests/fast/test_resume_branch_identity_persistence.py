"""Run-level resume persistence for branch alternatives and atom identity.

PRD docs/PRD_jev_atom_identity_mechanistic.md §9.4 blocker 3, §10.12, §10.13.

The coordinator snapshots its branch points (with the full untried
alternatives) and the mapped loop state (mapped species, sidecar id map,
allocator) into ``run_resume_state`` on every applied candidate and branch
point. A resumed run hydrates them, so backtracking after resume explores the
same alternatives with the same persistent atom ids as an uninterrupted run.

A "pause" here is modelled the way a real pause/resume works: the SQLite file
is the only thing that survives, and a fresh ``RunStore`` + ``RunCoordinator``
rebuilds the run state from it (``_build_state`` -> ``_hydrate_state_from_outputs``).
"""
from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

pytest.importorskip("rdkit")

from mechanistic_agent.core.coordinator import RunCoordinator  # noqa: E402
from mechanistic_agent.core.db import RunStore  # noqa: E402
from mechanistic_agent.core.mapped_state import MappedState  # noqa: E402
from mechanistic_agent.core.types import BranchCandidate, HarnessConfig, RunState  # noqa: E402

DB_NAME = "mechanistic.db"

# A mocked 3-step run over a small SN1-like system. Step 2 has a branch point:
# B (rank 1, chosen) creates a new atom, C and C2 are validated alternatives.
STEP_A = ("[CH3:1][OH2+:2]>>[CH3+:1].[OH2:2]", ["[CH3+]", "O", "[Br-]"])
STEP_B = ("[CH3+:1]>>[CH3:1][Cl:9]", ["CCl", "O", "[Br-]"])  # issues a fresh id
STEP_C = ("[CH3+:1].[Br-:5]>>[CH3:1][Br:5]", ["CBr", "O"])
STEP_C2 = ("[CH3+:1].[OH2:2]>>[CH3:1][OH2+:2]", ["C[OH2+]", "[Br-]"])
STEP_D = ("[CH3:1][Br:5]>>[CH3:1][Br:5].[Cl-:9]", ["CBr", "O", "[Cl-]"])  # issues a fresh id


def _candidate(step: Tuple[str, List[str]], rank: int = 1) -> BranchCandidate:
    smirks, resulting = step
    return BranchCandidate(
        rank=rank,
        intermediate_smiles=resulting[0],
        intermediate_output={"template_alignment": "unknown", "reaction_smirks": smirks},
        mechanism_output={"reaction_smirks": smirks, "contains_target_product": False},
        resulting_state=list(resulting),
        validation_summary={"passed": True, "rank": rank},
    )


def _create_run(db_path: Path) -> str:
    store = RunStore(db_path)
    return store.create_run(
        mode="unverified",
        input_payload={
            "starting_materials": ["[CH3:1][OH2+:2]", "[Br-:5]"],
            "products": ["[CH3:1][Br:5]", "[OH2:2]"],
            "temperature_celsius": 25.0,
            "ph": 7.0,
        },
        config={"model": "gpt-4o-mini", "model_family": "openai", "max_steps": 5},
        prompt_bundle_hash="p",
        skill_bundle_hash="s",
        memory_bundle_hash="m",
    )


def _open(db_path: Path, run_id: str, harness: Optional[HarnessConfig] = None) -> Tuple[RunCoordinator, RunStore, RunState]:
    """Fresh store + coordinator + hydrated state, as a resumed process sees it."""
    store = RunStore(db_path)
    coordinator = RunCoordinator(store)
    state = coordinator._build_state(store.get_run_row(run_id))
    coordinator._configure_loop_state_mapping(state, harness)
    return coordinator, store, state


def _pause_and_copy(store: RunStore, run_id: str, src: Path, dst: Path) -> Path:
    store.set_run_status(run_id, "paused")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    return dst


def _record_passed(store: RunStore, state: RunState, candidates: List[BranchCandidate]) -> None:
    """Write the passed mechanism_synthesis rows the live loop records per attempt."""
    for retry_index, cand in enumerate(candidates):
        store.record_step_output(
            run_id=state.run_id,
            step_name="mechanism_synthesis",
            attempt=state.step_index + 1,
            retry_index=retry_index,
            model="gpt-4o-mini",
            reasoning_level=None,
            tool_name="propose_mechanism_step",
            output={
                **cand.mechanism_output,
                "resulting_state": list(cand.resulting_state),
                "predicted_intermediate": cand.intermediate_smiles,
            },
            validation={"passed": True},
        )


def _step(coordinator: RunCoordinator, state: RunState, chosen: BranchCandidate, alternatives=()) -> None:
    """One mocked loop step: passed outputs, optional branch point, apply."""
    _record_passed(coordinator.store, state, [chosen, *alternatives])
    if alternatives:
        coordinator._record_branch_point(state, chosen, list(alternatives))
    coordinator._apply_candidate(state, chosen)


def _run_two_steps_with_branch(coordinator: RunCoordinator, state: RunState) -> None:
    _step(coordinator, state, _candidate(STEP_A))
    _step(coordinator, state, _candidate(STEP_B, rank=1), [_candidate(STEP_C, rank=2), _candidate(STEP_C2, rank=3)])


def _agreement(candidate: BranchCandidate) -> Dict[str, Any]:
    return dict((candidate.validation_summary or {}).get("smirks_state_agreement") or {})


def _cursor(state: RunState) -> Dict[str, Any]:
    return {
        "step_index": state.step_index,
        "current_state": list(state.current_state),
        "previous_intermediates": list(state.previous_intermediates),
        "branch_points": [bp.to_persisted_dict() for bp in state.branch_points],
        "mapped_loop_state": state.mapped_loop_state,
        "mapped_state_history": dict(state.mapped_state_history),
    }


# ---------------------------------------------------------------------------
# (a) Backtracking after resume == backtracking before pause
# ---------------------------------------------------------------------------


def test_backtrack_after_resume_applies_same_alternative_as_before_pause(tmp_path: Path) -> None:
    live_db = tmp_path / "live" / DB_NAME
    run_id = _create_run(live_db)
    coordinator, store, state = _open(live_db, run_id)
    _run_two_steps_with_branch(coordinator, state)  # paused after step 2 of 3
    pre_pause = _cursor(state)
    resumed_db = _pause_and_copy(store, run_id, live_db, tmp_path / "resumed" / DB_NAME)

    # Reference: the uninterrupted process backtracks (step 3 exhausted retries).
    assert coordinator._backtrack(state) is True

    # Resume in a fresh store/coordinator from the persisted file only.
    r_coordinator, _r_store, resumed = _open(resumed_db, run_id)
    assert _cursor(resumed) == pre_pause
    (bp,) = resumed.branch_points
    assert [a.rank for a in bp.alternatives] == [2, 3]
    assert bp.alternatives[0].mechanism_output["reaction_smirks"] == STEP_C[0]
    assert r_coordinator._backtrack(resumed) is True

    applied_live = state.branch_points[0].chosen_candidate
    applied_resumed = resumed.branch_points[0].chosen_candidate
    assert applied_resumed.to_persisted_dict() == applied_live.to_persisted_dict()
    assert applied_resumed.rank == 2 and applied_resumed.resulting_state == STEP_C[1]
    assert _cursor(resumed) == _cursor(state)

    # The remaining alternative is still reachable after resume, as before pause.
    assert coordinator._backtrack(state) is True
    assert r_coordinator._backtrack(resumed) is True
    assert resumed.branch_points[0].chosen_candidate.rank == 3
    assert _cursor(resumed) == _cursor(state)
    assert coordinator._backtrack(state) is False
    assert r_coordinator._backtrack(resumed) is False


def test_legacy_run_without_resume_state_falls_back_to_event_replay(tmp_path: Path) -> None:
    db = tmp_path / DB_NAME
    run_id = _create_run(db)
    coordinator, store, state = _open(db, run_id)
    _run_two_steps_with_branch(coordinator, state)
    with sqlite3.connect(db) as conn:  # a run recorded before run_resume_state existed
        conn.execute("DELETE FROM run_resume_state WHERE run_id = ?", (run_id,))
        conn.commit()

    _c, _s, resumed = _open(db, run_id)
    (bp,) = resumed.branch_points
    assert bp.step_index == 1 and bp.alternatives == []  # old behaviour: alternatives lost
    # Legacy cursor heuristic: the last passed row of the latest attempt, which
    # is not necessarily the chosen candidate when several passed.
    assert resumed.step_index == 2 and resumed.current_state == STEP_C2[1]
    assert resumed.mapped_loop_state is None


def test_last_chance_resume_consumes_the_persisted_alternative(tmp_path: Path) -> None:
    db = tmp_path / DB_NAME
    run_id = _create_run(db)
    coordinator, store, state = _open(db, run_id)
    _run_two_steps_with_branch(coordinator, state)

    r_coordinator, _s, resumed = _open(db, run_id)
    bp, alt = r_coordinator._peek_next_alternative(resumed)
    # What the last_chance_backtrack resume path rebuilds from pause details.
    resumed.current_state = list(bp.current_state)
    resumed.previous_intermediates = list(bp.previous_intermediates)
    resumed.step_index = bp.step_index
    from_pause = BranchCandidate(
        rank=alt.rank,
        intermediate_smiles=alt.intermediate_smiles,
        intermediate_output=dict(alt.intermediate_output),
        mechanism_output=dict(alt.mechanism_output),
        resulting_state=list(alt.resulting_state),
    )
    taken = r_coordinator._consume_resumed_alternative(resumed, from_pause)
    assert taken.to_persisted_dict() == _candidate(STEP_C, rank=2).to_persisted_dict()
    assert [a.rank for a in resumed.branch_points[0].alternatives] == [3]
    assert resumed.branch_points[0].chosen_candidate is taken
    r_coordinator._apply_candidate(resumed, taken, resume_state_kind="backtrack")

    # A second resume sees the alternative as taken, not offered again.
    _c2, _s2, again = _open(db, run_id)
    assert [a.rank for a in again.branch_points[0].alternatives] == [3]
    assert again.current_state == STEP_C[1]


# ---------------------------------------------------------------------------
# (b) §10.12 at run level: apply A, B; backtrack; apply C
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("resume_before_backtrack", [False, True], ids=["in_process", "after_resume"])
def test_10_12_run_level_backtrack_restores_exact_ids_without_leakage(
    tmp_path: Path, resume_before_backtrack: bool
) -> None:
    db = tmp_path / DB_NAME
    run_id = _create_run(db)
    coordinator, store, state = _open(db, run_id)
    coordinator._apply_candidate(state, _candidate(STEP_A))
    after_a = MappedState.restore(state.mapped_loop_state)
    b = _candidate(STEP_B)
    coordinator._record_branch_point(state, b, [_candidate(STEP_C, rank=2)])
    coordinator._apply_candidate(state, b)
    (discarded_pid,) = _agreement(b)["new_ids"]
    assert discarded_pid in MappedState.restore(state.mapped_loop_state).pids()

    if resume_before_backtrack:
        store.set_run_status(run_id, "paused")
        coordinator, store, state = _open(db, run_id)

    assert coordinator._backtrack(state) is True
    c = state.branch_points[0].chosen_candidate
    record = _agreement(c)
    assert record["smirks_state_agreement"] is True
    assert record["input_state_origin"] == "history"  # snapshot restored, not re-seeded
    assert record["new_ids"] == [] and record["lost_ids"] == []
    assert MappedState.restore(state.mapped_state_history[1]).map_to_pid == after_a.map_to_pid

    after_c = MappedState.restore(state.mapped_loop_state)
    assert discarded_pid not in after_c.pids()
    for map_number in (1, 2, 5):  # every surviving atom keeps its pre-branch id
        assert after_c.pid_by_map(map_number) == after_a.pid_by_map(map_number)

    d = _candidate(STEP_D)
    coordinator._apply_candidate(state, d)
    (fresh_pid,) = _agreement(d)["new_ids"]
    assert fresh_pid > discarded_pid  # the discarded path's id is never reissued


# ---------------------------------------------------------------------------
# (c) §10.13: persist, reload, resume; ids and mapping state identical
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "harness",
    [HarnessConfig(), HarnessConfig(loop_state_mapping="mapped", record_smirks_state_agreement=False)],
    ids=["record_smirks_state_agreement", "loop_state_mapping_mapped"],
)
def test_10_13_run_level_resume_restores_ids_and_replays_mapping_state(
    tmp_path: Path, harness: HarnessConfig
) -> None:
    steps = [STEP_A, STEP_C, STEP_D]

    # Uninterrupted reference run.
    ref_db = tmp_path / "ref" / DB_NAME
    ref_run = _create_run(ref_db)
    ref_coord, _ref_store, ref_state = _open(ref_db, ref_run, harness)
    ref_records = []
    for step in steps:
        cand = _candidate(step)
        ref_coord._apply_candidate(ref_state, cand)
        ref_records.append(_agreement(cand))

    # Interrupted run: two steps, pause, persist, reload in a fresh process.
    run_db = tmp_path / "run" / DB_NAME
    run_id = _create_run(run_db)
    coord, store, state = _open(run_db, run_id, harness)
    records = []
    for step in steps[:2]:
        cand = _candidate(step)
        coord._apply_candidate(state, cand)
        records.append(_agreement(cand))
    pre_pause = _cursor(state)
    store.set_run_status(run_id, "paused")

    # SQLite row -> file -> MappedState: the snapshot survives both media.
    row = store.get_latest_run_resume_state(run_id)
    assert row["kind"] == "accepted_step" and row["step_index"] == 2
    snapshot_file = tmp_path / "resume_state.json"
    snapshot_file.write_text(json.dumps(row["payload"], sort_keys=True))
    from_file = json.loads(snapshot_file.read_text())
    assert MappedState.restore(from_file["mapped_loop_state"]).map_to_pid == MappedState.restore(
        state.mapped_loop_state
    ).map_to_pid

    coord, store, resumed = _open(run_db, run_id, harness)
    assert _cursor(resumed) == pre_pause  # ids, allocator, history identical
    live = MappedState.restore(resumed.mapped_loop_state)
    assert live.allocator.next_id == MappedState.restore(pre_pause["mapped_loop_state"]).allocator.next_id

    cand = _candidate(steps[2])
    coord._apply_candidate(resumed, cand)
    records.append(_agreement(cand))

    assert records == ref_records  # smirks_state_agreement recomputed identically
    assert records[2]["input_state_origin"] == "live"  # not re-seeded on resume
    assert resumed.mapped_loop_state == ref_state.mapped_loop_state
    assert resumed.mapped_state_history == ref_state.mapped_state_history
    assert resumed.current_state == ref_state.current_state


def test_resume_allocator_never_reissues_ids_below_run_high_water(tmp_path: Path) -> None:
    db = tmp_path / DB_NAME
    run_id = _create_run(db)
    coordinator, store, state = _open(db, run_id)
    coordinator._apply_candidate(state, _candidate(STEP_A))
    coordinator._apply_candidate(state, _candidate(STEP_C))
    # Simulate ids issued on some path the live snapshot no longer reflects.
    state.mapped_id_high_water = 50
    coordinator._persist_resume_state(state, "accepted_step")

    coordinator, _store, resumed = _open(db, run_id)
    assert MappedState.restore(resumed.mapped_loop_state).allocator.next_id == 50
    d = _candidate(STEP_D)
    coordinator._apply_candidate(resumed, d)
    assert _agreement(d)["new_ids"] == [50]


def test_resume_state_rows_recorded_per_step_branch_and_backtrack(tmp_path: Path) -> None:
    db = tmp_path / DB_NAME
    run_id = _create_run(db)
    coordinator, store, state = _open(db, run_id)
    _run_two_steps_with_branch(coordinator, state)
    coordinator._backtrack(state)
    kinds = [(row["kind"], row["step_index"]) for row in store.list_run_resume_states(run_id)]
    assert kinds == [("accepted_step", 1), ("branch_point", 1), ("accepted_step", 2), ("backtrack", 2)]


def test_disabled_identity_flags_persist_branch_points_without_mapped_state(tmp_path: Path) -> None:
    harness = HarnessConfig(record_smirks_state_agreement=False)
    db = tmp_path / DB_NAME
    run_id = _create_run(db)
    coordinator, _store, state = _open(db, run_id, harness)
    _run_two_steps_with_branch(coordinator, state)
    _c, _s, resumed = _open(db, run_id, harness)
    assert resumed.mapped_loop_state is None and resumed.mapped_state_history == {}
    assert [a.rank for a in resumed.branch_points[0].alternatives] == [2, 3]


def test_run_store_migration_creates_resume_state_table_idempotently(tmp_path: Path) -> None:
    db = tmp_path / DB_NAME
    RunStore(db)
    RunStore(db)  # re-running init_db is a no-op
    with sqlite3.connect(db) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(run_resume_state)")}
        versions = {row[0] for row in conn.execute("SELECT version FROM db_migrations")}
    assert {"run_id", "seq", "kind", "step_index", "payload_json"} <= columns
    assert "2026_09_run_resume_state_v1" in versions

    run_id = _create_run(db)
    store = RunStore(db)
    store.record_run_resume_state(run_id, kind="accepted_step", step_index=1, payload={"x": 1})
    store.record_run_resume_state(run_id, kind="branch_point", step_index=1, payload={"x": 2})
    assert store.get_latest_run_resume_state(run_id)["payload"] == {"x": 2}
    assert store.delete_run(run_id) is True
    assert store.list_run_resume_states(run_id) == []
