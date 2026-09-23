"""The mutated variant must be the one an evolution eval actually runs.

Island mode (``scripts/evolve_harness.py::evolve_islands``) and overnight Ralph
(``OvernightRalphOrchestrator.run``) both mutate a sibling asset (a harness JSON
variant, a ``prompt_variant_*.SKILL.md`` or a ``few_shot_variant_*.jsonl``) and
then score an evaluation batch. These tests drive the real loops with the model
calls mocked out and inspect, at the moment each evaluation run executes, the
harness and prompt assets that the run resolves. A mutation that is not
observable there means the archive / keep-discard rule is scoring the parent.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

from mechanistic_agent.core.archive import EvolutionArchive
from mechanistic_agent.core.coordinator import RunCoordinator
from mechanistic_agent.core.db import RunStore
from mechanistic_agent.core.llm_mutator import LLMLaneMutator
from mechanistic_agent.core.overnight_ralph import OvernightRalphOrchestrator
from mechanistic_agent.core.types import (
    ArchiveEntry,
    IslandEvolutionConfig,
    MicroEvalResult,
    OvernightRalphConfig,
)
from mechanistic_agent.prompt_assets import compose_system_prompt, load_call_few_shot_examples

_ROOT = Path(__file__).resolve().parents[2]

APPENDED = "Always number every mapped heavy atom before listing bond changes."
ATOM_MAPPING_FEW_SHOTS = [
    {"input": "few-shot input ZERO", "output": "few-shot output ZERO"},
    {"input": "few-shot input ONE", "output": "few-shot output ONE"},
]


def _load_evolve_module():
    name = "evolve_harness_mod_mutation_applied"
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / "evolve_harness.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _write_skill(repo: Path, call_name: str, body: str, few_shots: List[Dict[str, str]] | None = None) -> None:
    skill_dir = repo / "skills" / "mechanistic" / call_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {call_name}\ncall_name: {call_name}\n---\n<!-- PROMPT_START -->\n{body}\n<!-- PROMPT_END -->\n",
        encoding="utf-8",
    )
    if few_shots is not None:
        (skill_dir / "few_shot.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in few_shots), encoding="utf-8"
        )


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    harness_dir = repo / "harness_versions" / "default"
    harness_dir.mkdir(parents=True)
    shutil.copyfile(_ROOT / "harness_versions" / "default" / "harness.json", harness_dir / "harness.json")
    _write_skill(repo, "base_system", "You are a careful mechanistic chemist.")
    _write_skill(repo, "attempt_atom_mapping", "Map atoms from reactants to products.", ATOM_MAPPING_FEW_SHOTS)
    _write_skill(repo, "select_reaction_type", "Pick the best reaction type.", [])
    _write_skill(
        repo,
        "propose_mechanism_step",
        "Propose the next elementary step.",
        [{"input": "pms input ZERO", "output": "pms output ZERO"}, {"input": "pms input ONE", "output": "pms output ONE"}],
    )
    return repo


def _observe_run(coordinator: RunCoordinator, run_id: str) -> Dict[str, Any]:
    """Resolve harness + prompt assets exactly as ``execute_run`` would."""
    state = coordinator._build_state(coordinator.store.get_run_row(run_id))
    harness = coordinator._resolve_harness(state)
    return {
        "harness_config_path": state.run_config.harness_config_path,
        "modules": {m.id: bool(m.enabled) for m in harness.all_modules()},
        "centralized_mas": harness.get_topology_profile("centralized_mas").as_dict(),
        "atom_mapping_prompt": compose_system_prompt(call_name="attempt_atom_mapping", dynamic_system_prompt=""),
        "atom_mapping_few_shot_inputs": [
            row["input"] for row in load_call_few_shot_examples("attempt_atom_mapping")
        ],
    }


def _run_one_island_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    island_id: str,
    proposal: Dict[str, Any],
) -> List[Dict[str, Any]]:
    mod = _load_evolve_module()
    repo = _make_repo(tmp_path)
    monkeypatch.chdir(repo)  # runtime prompt assets resolve relative to cwd
    monkeypatch.setattr(mod, "_PROJECT_ROOT", repo)
    monkeypatch.setattr(mod, "resolve_db_path", lambda base: Path(base) / "data" / "mechanistic.db")

    store = RunStore(repo / "data" / "mechanistic.db")
    eval_set_id = store.add_eval_set(
        name="practice_set",
        version="v1",
        source_path=None,
        sha256=None,
        cases=[
            {
                "case_id": "case_a",
                "input": {"starting_materials": ["CCBr", "[Cl-]"], "products": ["CCCl", "[Br-]"]},
                "expected": {
                    "products": ["CCCl", "[Br-]"],
                    "known_mechanism": {"steps": [{"step_index": 1, "target_smiles": "CCCl"}]},
                },
            }
        ],
    )
    EvolutionArchive(store=store).insert(
        ArchiveEntry(
            id="seed",
            generation=0,
            island_id=island_id,
            parent_id=None,
            archive_inspiration_ids=[],
            harness_name="default",
            harness_config_json="{}",
            prompt_bundle_hash="",
            skill_bundle_hash="",
            few_shot_snapshot_json="{}",
            topology_profile_json="{}",
            mutation_type="seed",
            mutation_summary="seed",
            mean_quality_score=0.1,
            weighted_pass_rate=0.0,
            per_subagent_scores_json="{}",
            total_cost=0.0,
            eval_run_id=None,
            eval_tier="mixed",
            case_count=0,
            children_count=0,
            score_delta=0.0,
            migration_history_json="[]",
            created_at=time.time(),
        )
    )

    monkeypatch.setattr(LLMLaneMutator, "_ask_model", lambda self, messages: dict(proposal))
    observed: List[Dict[str, Any]] = []

    def _capture(self: RunCoordinator, run_id: str, _stop_event: Any) -> None:
        observed.append(_observe_run(self, run_id))

    monkeypatch.setattr(RunCoordinator, "execute_run", _capture)

    mod.evolve_islands(
        mod.EvolutionConfig(model_name="gpt-5", eval_set_id=eval_set_id, group_size=1),
        IslandEvolutionConfig(
            islands=[island_id],
            migration_interval=1000,
            max_generations=1,
            mutation_proposer="llm",
            mutation_model="stub-model",
        ),
    )
    assert observed, "the island loop never executed an evaluation run"
    return observed


def _proposal(lane: str, target: str, operation: str, value: Any) -> Dict[str, Any]:
    return {
        "lane": lane,
        "target": target,
        "operation": operation,
        "value": value,
        "rationale": "test",
        "expected_effect": "test",
    }


def test_island_harness_lane_set_enabled_false_disables_module_in_eval_run(tmp_path, monkeypatch) -> None:
    observed = _run_one_island_generation(
        tmp_path, monkeypatch, island_id="topology",
        proposal=_proposal("harness", "reflection", "set_enabled", False),
    )
    for run in observed:
        assert run["harness_config_path"], "eval run was not pointed at the mutated harness"
        assert run["modules"]["reflection"] is False


def test_island_topology_lane_set_field_reaches_eval_run(tmp_path, monkeypatch) -> None:
    observed = _run_one_island_generation(
        tmp_path, monkeypatch, island_id="topology",
        proposal=_proposal("topology", "centralized_mas.max_candidates_per_agent", "set_field", 5),
    )
    for run in observed:
        assert run["centralized_mas"]["max_candidates_per_agent"] == 5


def test_island_prompt_lane_append_instruction_reaches_eval_run(tmp_path, monkeypatch) -> None:
    observed = _run_one_island_generation(
        tmp_path, monkeypatch, island_id="mapping",
        proposal=_proposal("prompt", "attempt_atom_mapping", "append_instruction", APPENDED),
    )
    for run in observed:
        assert APPENDED in run["atom_mapping_prompt"]
    # The override is scoped to the evaluation: the committed prompt is untouched afterwards.
    assert APPENDED not in compose_system_prompt(call_name="attempt_atom_mapping", dynamic_system_prompt="")


def test_island_few_shot_lane_remove_example_reaches_eval_run(tmp_path, monkeypatch) -> None:
    observed = _run_one_island_generation(
        tmp_path, monkeypatch, island_id="mapping",
        proposal=_proposal("few_shot", "attempt_atom_mapping", "remove_few_shot", 0),
    )
    for run in observed:
        assert "few-shot input ZERO" not in run["atom_mapping_few_shot_inputs"]
        assert "few-shot input ONE" in run["atom_mapping_few_shot_inputs"]
    remaining = [row["input"] for row in load_call_few_shot_examples("attempt_atom_mapping")]
    assert "few-shot input ZERO" in remaining


# ---------------------------------------------------------------------------
# Overnight Ralph: prompt / few_shot lanes must be applied during run_slice too
# ---------------------------------------------------------------------------


def _run_overnight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lane: str) -> List[Dict[str, Any]]:
    repo = _make_repo(tmp_path)
    monkeypatch.chdir(repo)
    store = RunStore(repo / "data" / "mechanistic.db")
    orchestrator = OvernightRalphOrchestrator(base_dir=repo, store=store)
    orchestrator._load_eval_slice = lambda _config: [{"starting_materials": ["A"], "products": ["B"]}]  # type: ignore[method-assign]

    observed: List[Dict[str, Any]] = []

    def _fake_run_slice(**kwargs: Any) -> MicroEvalResult:
        observed.append(
            {
                "harness_config_path": kwargs.get("harness_config_path"),
                "pms_prompt": compose_system_prompt(call_name="propose_mechanism_step", dynamic_system_prompt=""),
                "pms_few_shot_inputs": [
                    row["input"] for row in load_call_few_shot_examples("propose_mechanism_step")
                ],
            }
        )
        return MicroEvalResult("slice", 1, 0.4, 0.4, 0.0, 0.0, 0.0, 1.0, 0, 0)

    orchestrator.micro_eval.run_slice = _fake_run_slice  # type: ignore[method-assign]
    orchestrator.run(
        config=OvernightRalphConfig(
            eval_slice_id="slice",
            eval_slice_size=1,
            max_experiments=1,
            max_cost_usd=10.0,
            allowed_lanes=[lane],  # type: ignore[list-item]
        ),
        run_config={"harness_name": "default"},
    )
    assert len(observed) == 2  # baseline + one experiment
    return observed


def test_overnight_prompt_lane_variant_is_applied_during_experiment(tmp_path, monkeypatch) -> None:
    baseline, experiment = _run_overnight(tmp_path, monkeypatch, "prompt")
    note = "Mutation note: prefer concise mechanism-step proposals."
    assert note not in baseline["pms_prompt"]
    assert note in experiment["pms_prompt"]
    assert note not in compose_system_prompt(call_name="propose_mechanism_step", dynamic_system_prompt="")


def test_overnight_few_shot_lane_variant_is_applied_during_experiment(tmp_path, monkeypatch) -> None:
    baseline, experiment = _run_overnight(tmp_path, monkeypatch, "few_shot")
    assert "pms input ONE" in baseline["pms_few_shot_inputs"]
    assert "pms input ONE" not in experiment["pms_few_shot_inputs"]  # blind mutator drops the last line
    assert "pms input ZERO" in experiment["pms_few_shot_inputs"]
