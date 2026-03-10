from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from mechanistic_agent.core.db import RunStore
from mechanistic_agent.core.experiment_ledger import ExperimentLedger
from mechanistic_agent.core.lane_mutator import TopologyLaneMutator
from mechanistic_agent.core.micro_eval_runner import MicroEvalRunner
from mechanistic_agent.core.overnight_ralph import OvernightRalphOrchestrator
from mechanistic_agent.core.types import ExperimentRecord, MicroEvalResult, OvernightRalphConfig


def test_topology_lane_mutator_changes_profile_field(tmp_path: Path) -> None:
    harness = {
        "name": "default",
        "topology_profiles": {
            "centralized_mas": {
                "agent_count": 1,
                "max_candidates_per_agent": 3,
                "peer_rounds": 0,
            }
        },
    }
    parent = tmp_path / "harness.json"
    parent.write_text(json.dumps(harness), encoding="utf-8")

    mutated = TopologyLaneMutator().propose(parent)
    assert mutated.lane == "topology"
    payload = json.loads(mutated.asset_path.read_text(encoding="utf-8"))
    new_profile = payload["topology_profiles"]["centralized_mas"]
    assert new_profile != harness["topology_profiles"]["centralized_mas"]


class _DummyCoordinator:
    def execute_run(self, run_id: str, _stop_event) -> None:
        return None


class _DummyStore:
    def __init__(self, snapshots: dict[str, dict]):
        self.snapshots = snapshots

    def get_run_snapshot(self, run_id: str):
        return self.snapshots.get(run_id)


class _StubMicroEvalRunner(MicroEvalRunner):
    def __init__(self, *, base_dir: Path, snapshots: dict[str, dict], run_ids: list[str]) -> None:
        super().__init__(base_dir=base_dir, store=_DummyStore(snapshots), coordinator=_DummyCoordinator())
        self._run_ids = list(run_ids)

    def _create_case_run(self, *, case, base_config, harness_config_path):
        return self._run_ids.pop(0)


def test_micro_eval_runner_aggregates_metrics(tmp_path: Path) -> None:
    snapshots = {
        "r1": {
            "status": "completed",
            "events": [{"event_type": "mechanism_retry_started"}],
            "step_outputs": [{"step_name": "mechanism_synthesis", "validation": {"passed": True}}],
            "cost_summary": {"total_cost": {"total_cost": 0.5}},
        },
        "r2": {
            "status": "failed",
            "events": [{"event_type": "backtrack"}, {"event_type": "mechanism_retry_started"}],
            "step_outputs": [{"step_name": "mechanism_synthesis", "validation": {"passed": False}}],
            "cost_summary": {"total_cost": {"total_cost": 0.25}},
        },
    }
    runner = _StubMicroEvalRunner(base_dir=tmp_path, snapshots=snapshots, run_ids=["r1", "r2"])
    result = runner.run_slice(
        eval_slice_id="slice_a",
        cases=[{"starting_materials": ["A"], "products": ["B"]}, {"starting_materials": ["C"], "products": ["D"]}],
        base_config={"model": "x"},
    )

    assert result.case_count == 2
    assert result.completion_pct == 0.5
    assert result.validator_pass_pct == 0.5
    assert result.avg_retries == 1.0
    assert result.avg_backtracks == 0.5
    assert result.token_cost_usd == 0.75


def test_experiment_ledger_dual_writes(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "mechanistic.db")
    ledger = ExperimentLedger(base_dir=tmp_path, store=store)
    record = ExperimentRecord(
        parent_checkpoint_sha="abc",
        mutated_asset_path="/tmp/mut.json",
        mutation_lane="topology",
        mutation_summary="changed field",
        eval_slice_id="slice_x",
        completion_pct=0.8,
        validator_pass_pct=0.9,
        avg_retries=0.2,
        avg_backtracks=0.1,
        token_cost_usd=1.2,
        keep=True,
    )
    row_id = ledger.append(record)
    rows = ledger.list(eval_slice_id="slice_x")

    assert row_id
    assert rows
    assert rows[0]["mutation_lane"] == "topology"
    jsonl_files = list((tmp_path / "traces" / "overnight_ralph").glob("*_ledger.jsonl"))
    assert jsonl_files


def test_overnight_orchestrator_keep_discard_loop(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "mechanistic.db")
    orchestrator = OvernightRalphOrchestrator(base_dir=tmp_path, store=store)

    harness_dir = tmp_path / "harness_versions" / "default"
    harness_dir.mkdir(parents=True)
    parent_harness = harness_dir / "harness.json"
    parent_harness.write_text(
        json.dumps(
            {
                "name": "default",
                "topology_profiles": {"centralized_mas": {"agent_count": 1, "max_candidates_per_agent": 3, "peer_rounds": 0}},
            }
        ),
        encoding="utf-8",
    )

    mut1 = tmp_path / "mut1.json"
    mut1.write_text(parent_harness.read_text(encoding="utf-8"), encoding="utf-8")
    mut2 = tmp_path / "mut2.json"
    mut2.write_text(parent_harness.read_text(encoding="utf-8"), encoding="utf-8")

    orchestrator._load_eval_slice = lambda _config: [{"starting_materials": ["A"], "products": ["B"]}]  # type: ignore[method-assign]
    orchestrator._resolve_parent_harness_asset = lambda _run_config: parent_harness  # type: ignore[method-assign]

    muts = [
        SimpleNamespace(asset_path=mut1, summary="m1", lane="topology"),
        SimpleNamespace(asset_path=mut2, summary="m2", lane="topology"),
    ]
    orchestrator._propose_mutation = lambda lane, parent_asset: muts.pop(0)  # type: ignore[method-assign]

    results = [
        MicroEvalResult("slice", 1, 0.4, 0.4, 0.0, 0.0, 0.2, 1.0, 0, 0),
        MicroEvalResult("slice", 1, 0.5, 0.5, 0.0, 0.0, 0.2, 1.0, 0, 0),
        MicroEvalResult("slice", 1, 0.5, 0.5, 0.0, 0.0, 2.0, 1.0, 0, 0),
    ]

    def _fake_run_slice(**_kwargs):
        return results.pop(0)

    orchestrator.micro_eval.run_slice = _fake_run_slice  # type: ignore[method-assign]

    summary = orchestrator.run(
        config=OvernightRalphConfig(
            eval_slice_id="slice",
            eval_slice_size=1,
            max_experiments=2,
            max_cost_usd=10.0,
            acceptance_threshold_pct=0.05,
            allowed_lanes=["topology"],
        ),
        run_config={"harness_name": "default"},
    )

    assert summary["keep_count"] == 1
    assert summary["discard_count"] == 1
    rows = orchestrator.ledger.list(eval_slice_id="slice")
    assert len(rows) == 2
