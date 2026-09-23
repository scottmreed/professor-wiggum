"""The mutated variant must be the one an evolution eval actually runs.

Island mode (``scripts/evolve_harness.py::evolve_islands``) and overnight Ralph
(``OvernightRalphOrchestrator.run``) both mutate a sibling asset (a harness JSON
variant, a ``prompt_variant_*.SKILL.md`` or a ``few_shot_variant_*.jsonl``) and
then score an evaluation batch. These tests drive the real loops with the model
calls mocked out and inspect, at the moment each evaluation run executes, the
harness and prompt assets that the run resolves. A mutation that is not
observable there means the archive / keep-discard rule is scoring the parent.

Follow-ups covered further down:

* a kept variant is the parent of later mutations (island archive lineage and
  overnight Ralph keeps), and Ralph compares experiments against a baseline
  evaluated under the same base assets;
* prompt / few-shot variants are derived from the asset resolved for the run's
  model (a ``models/<slug>/`` override when present) and applied at that scope;
* a dry island run leaves the checkout clean.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
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
from mechanistic_agent.prompt_assets import (
    call_asset_overrides,
    compose_system_prompt,
    load_call_few_shot_examples,
)

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


def _write_model_lane(
    repo: Path, call_name: str, model_name: str, body: str | None = None, few_shots: List[Dict[str, str]] | None = None
) -> None:
    lane_dir = repo / "skills" / "mechanistic" / call_name / "models" / model_name.replace("/", "__")
    lane_dir.mkdir(parents=True, exist_ok=True)
    if body is not None:
        (lane_dir / "SKILL.md").write_text(
            f"---\nname: {call_name}\ncall_name: {call_name}\n---\n<!-- PROMPT_START -->\n{body}\n<!-- PROMPT_END -->\n",
            encoding="utf-8",
        )
    if few_shots is not None:
        (lane_dir / "few_shot.jsonl").write_text("".join(json.dumps(row) + "\n" for row in few_shots), encoding="utf-8")


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
        "atom_mapping_prompt_gpt5": compose_system_prompt(
            call_name="attempt_atom_mapping", dynamic_system_prompt="", model_name="gpt-5"
        ),
    }


def _seed_entry(island_id: str, harness_config_json: str = "{}") -> ArchiveEntry:
    return ArchiveEntry(
        id=f"seed_{island_id}",
        generation=0,
        island_id=island_id,
        parent_id=None,
        archive_inspiration_ids=[],
        harness_name="default",
        harness_config_json=harness_config_json,
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


def _run_island(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    island_ids: List[str],
    proposals: List[Dict[str, Any]],
    generations: int = 1,
    dry_run: bool = False,
    newest_parent: bool = False,
    before_run: Any = None,
    seed_config: Any = None,
) -> Dict[str, Any]:
    """Drive the real island loop; return the observed eval runs, the repo and its store."""
    mod = _load_evolve_module()
    repo = _make_repo(tmp_path)
    if before_run is not None:
        before_run(repo)
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
    for island_id in island_ids:
        EvolutionArchive(store=store).insert(
            _seed_entry(island_id, json.dumps(seed_config(repo)) if seed_config is not None else "{}")
        )

    if newest_parent:
        # Deterministic lineage: always extend the most recent entry of the island.
        monkeypatch.setattr(
            EvolutionArchive,
            "select_parent",
            lambda self, island_id, rng=None: max(self.list_island(island_id), key=lambda e: (e.generation, e.created_at)),
        )
    queue = [dict(p) for p in proposals]
    monkeypatch.setattr(LLMLaneMutator, "_ask_model", lambda self, messages: queue.pop(0))
    observed: List[Dict[str, Any]] = []

    def _capture(self: RunCoordinator, run_id: str, _stop_event: Any) -> None:
        observed.append(_observe_run(self, run_id))

    monkeypatch.setattr(RunCoordinator, "execute_run", _capture)

    mod.evolve_islands(
        mod.EvolutionConfig(model_name="gpt-5", eval_set_id=eval_set_id, group_size=1, dry_run=dry_run),
        IslandEvolutionConfig(
            islands=list(island_ids),
            migration_interval=1000,
            max_generations=generations,
            mutation_proposer="llm",
            mutation_model="stub-model",
        ),
    )
    assert observed, "the island loop never executed an evaluation run"
    return {"observed": observed, "repo": repo, "store": store}


def _children(store: RunStore, island_id: str) -> List[ArchiveEntry]:
    entries = [e for e in EvolutionArchive(store=store).list_island(island_id) if e.parent_id]
    return sorted(entries, key=lambda e: (e.generation, e.created_at))


def _run_one_island_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    island_id: str,
    proposal: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return _run_island(tmp_path, monkeypatch, island_ids=[island_id], proposals=[proposal])["observed"]


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


def _run_overnight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lane: str,
    *,
    lanes: List[str] | None = None,
    scores: List[float] | None = None,
    run_config: Dict[str, Any] | None = None,
    before_run: Any = None,
) -> List[Dict[str, Any]]:
    """Drive ``OvernightRalphOrchestrator.run``; ``scores`` are the slice results (baseline first)."""
    repo = _make_repo(tmp_path)
    if before_run is not None:
        before_run(repo)
    monkeypatch.chdir(repo)
    store = RunStore(repo / "data" / "mechanistic.db")
    orchestrator = OvernightRalphOrchestrator(base_dir=repo, store=store)
    orchestrator._load_eval_slice = lambda _config: [{"starting_materials": ["A"], "products": ["B"]}]  # type: ignore[method-assign]

    observed: List[Dict[str, Any]] = []
    queue = list(scores or [0.4, 0.4])

    def _fake_run_slice(**kwargs: Any) -> MicroEvalResult:
        harness_path = kwargs.get("harness_config_path")
        observed.append(
            {
                "harness_config_path": harness_path,
                "harness_payload": json.loads(Path(harness_path).read_text(encoding="utf-8")) if harness_path else None,
                "pms_prompt": compose_system_prompt(call_name="propose_mechanism_step", dynamic_system_prompt=""),
                "pms_few_shot_inputs": [
                    row["input"] for row in load_call_few_shot_examples("propose_mechanism_step")
                ],
                "pms_prompt_gpt5": compose_system_prompt(
                    call_name="propose_mechanism_step", dynamic_system_prompt="", model_name="gpt-5"
                ),
                "pms_few_shot_inputs_gpt5": [
                    row["input"] for row in load_call_few_shot_examples("propose_mechanism_step", model_name="gpt-5")
                ],
            }
        )
        score = queue.pop(0)
        return MicroEvalResult("slice", 1, score, score, 0.0, 0.0, 0.0, 1.0, 0, 0)

    orchestrator.micro_eval.run_slice = _fake_run_slice  # type: ignore[method-assign]
    experiments = len(scores) - 1 if scores else 1
    summary = orchestrator.run(
        config=OvernightRalphConfig(
            eval_slice_id="slice",
            eval_slice_size=1,
            max_experiments=experiments,
            max_cost_usd=10.0,
            acceptance_threshold_pct=0.02,
            allowed_lanes=list(lanes or [lane]),  # type: ignore[arg-type]
        ),
        run_config=dict(run_config or {"harness_name": "default"}),
    )
    assert len(observed) == 1 + experiments  # baseline + experiments
    observed[0]["summary"] = summary
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


# ---------------------------------------------------------------------------
# Follow-up 1: a kept variant is the parent of later mutations
# ---------------------------------------------------------------------------

APPENDED_TWO = "Then check every leaving group keeps its electron pair."
PMS_NOTE = "Mutation note: prefer concise mechanism-step proposals."


def test_island_prompt_child_builds_on_parent_prompt_variant(tmp_path, monkeypatch) -> None:
    result = _run_island(
        tmp_path, monkeypatch, island_ids=["mapping"], generations=2, newest_parent=True,
        proposals=[
            _proposal("prompt", "attempt_atom_mapping", "append_instruction", APPENDED),
            _proposal("prompt", "attempt_atom_mapping", "append_instruction", APPENDED_TWO),
        ],
    )
    gen1, gen2 = result["observed"]
    assert APPENDED in gen1["atom_mapping_prompt"] and APPENDED_TWO not in gen1["atom_mapping_prompt"]
    # The second generation's parent is the first child: its prompt carries both edits.
    assert APPENDED in gen2["atom_mapping_prompt"]
    assert APPENDED_TWO in gen2["atom_mapping_prompt"]

    child1, child2 = _children(result["store"], "mapping")
    assert child2.parent_id == child1.id
    state1 = json.loads(child1.harness_config_json)["asset_state"]
    state2 = json.loads(child2.harness_config_json)["asset_state"]
    [variant1] = state1["call_variants"]
    [variant2] = state2["call_variants"]
    assert variant1["call_name"] == variant2["call_name"] == "attempt_atom_mapping"
    # Each lineage node keeps its own snapshot: the child did not overwrite the parent's file.
    assert variant1["path"] != variant2["path"]
    assert APPENDED_TWO not in Path(variant1["path"]).read_text(encoding="utf-8")
    assert APPENDED in Path(variant2["path"]).read_text(encoding="utf-8")


def test_island_few_shot_child_builds_on_parent_few_shot_variant(tmp_path, monkeypatch) -> None:
    result = _run_island(
        tmp_path, monkeypatch, island_ids=["mapping"], generations=2, newest_parent=True,
        proposals=[
            _proposal("few_shot", "attempt_atom_mapping", "remove_few_shot", 0),
            _proposal("few_shot", "attempt_atom_mapping", "remove_few_shot", 0),
        ],
    )
    gen1, gen2 = result["observed"]
    assert gen1["atom_mapping_few_shot_inputs"] == ["few-shot input ONE"]
    # Removing index 0 of the parent's variant drops the remaining example.
    assert gen2["atom_mapping_few_shot_inputs"] == []
    assert [r["input"] for r in load_call_few_shot_examples("attempt_atom_mapping")] == [
        "few-shot input ZERO",
        "few-shot input ONE",
    ]


def test_island_harness_child_builds_on_parent_topology_variant(tmp_path, monkeypatch) -> None:
    result = _run_island(
        tmp_path, monkeypatch, island_ids=["topology"], generations=2, newest_parent=True,
        proposals=[
            _proposal("topology", "centralized_mas.max_candidates_per_agent", "set_field", 5),
            _proposal("harness", "reflection", "set_enabled", False),
        ],
    )
    gen1, gen2 = result["observed"]
    assert gen1["centralized_mas"]["max_candidates_per_agent"] == 5
    assert gen1["modules"]["reflection"] is True
    assert gen2["centralized_mas"]["max_candidates_per_agent"] == 5
    assert gen2["modules"]["reflection"] is False


def test_island_prompt_child_keeps_parent_harness_variant(tmp_path, monkeypatch) -> None:
    def _parent_with_topology_variant(repo: Path) -> Dict[str, Any]:
        base = repo / "harness_versions" / "default" / "harness.json"
        payload = json.loads(base.read_text(encoding="utf-8"))
        payload["topology_profiles"]["centralized_mas"]["max_candidates_per_agent"] = 5
        variant = tmp_path / "parent_variant" / "harness.json"
        variant.parent.mkdir()
        variant.write_text(json.dumps(payload), encoding="utf-8")
        return {"asset_state": {"harness_path": str(variant), "call_variants": []}}

    result = _run_island(
        tmp_path, monkeypatch, island_ids=["mapping"], seed_config=_parent_with_topology_variant,
        proposals=[_proposal("prompt", "attempt_atom_mapping", "append_instruction", APPENDED)],
    )
    [run] = result["observed"]
    assert run["harness_config_path"], "the child dropped the parent's harness variant"
    assert run["centralized_mas"]["max_candidates_per_agent"] == 5
    assert APPENDED in run["atom_mapping_prompt"]


def test_island_skips_parent_whose_variant_files_are_gone(tmp_path, monkeypatch) -> None:
    missing = {"asset_state": {"harness_path": str(tmp_path / "gone" / "harness.json"), "call_variants": []}}
    with pytest.raises(AssertionError, match="never executed"):
        _run_island(
            tmp_path, monkeypatch, island_ids=["mapping"], seed_config=lambda _repo: missing,
            proposals=[_proposal("prompt", "attempt_atom_mapping", "append_instruction", APPENDED)],
        )


def test_overnight_kept_prompt_variant_stays_applied_in_later_experiments(tmp_path, monkeypatch) -> None:
    baseline, exp1, exp2 = _run_overnight(
        tmp_path, monkeypatch, "prompt", lanes=["prompt", "few_shot"], scores=[0.4, 0.6, 0.6]
    )
    assert PMS_NOTE not in baseline["pms_prompt"]
    assert PMS_NOTE in exp1["pms_prompt"]
    # exp2 is compared against exp1's result, so it must run with exp1's kept prompt too.
    assert PMS_NOTE in exp2["pms_prompt"]
    assert "pms input ONE" not in exp2["pms_few_shot_inputs"]
    assert baseline["summary"]["keep_count"] == 1
    assert PMS_NOTE not in compose_system_prompt(call_name="propose_mechanism_step", dynamic_system_prompt="")


def test_overnight_prompt_mutation_derives_from_kept_prompt_variant(tmp_path, monkeypatch) -> None:
    _baseline, exp1, exp2 = _run_overnight(tmp_path, monkeypatch, "prompt", scores=[0.4, 0.6, 0.6])
    assert exp1["pms_prompt"].count("Mutation note:") == 1
    assert exp2["pms_prompt"].count("Mutation note:") == 2


def test_overnight_discarded_prompt_variant_is_not_carried_forward(tmp_path, monkeypatch) -> None:
    baseline, _exp1, exp2 = _run_overnight(tmp_path, monkeypatch, "prompt", scores=[0.4, 0.4, 0.4])
    assert exp2["pms_prompt"].count("Mutation note:") == 1
    assert baseline["summary"]["keep_count"] == 0


def test_overnight_kept_topology_variant_stays_applied_in_later_prompt_experiment(tmp_path, monkeypatch) -> None:
    baseline, exp1, exp2 = _run_overnight(
        tmp_path, monkeypatch, "topology", lanes=["topology", "prompt"], scores=[0.4, 0.6, 0.6]
    )
    assert baseline["harness_config_path"] is None
    mutated_profiles = exp1["harness_payload"]["topology_profiles"]
    assert exp2["harness_config_path"], "the prompt experiment ran without the kept topology variant"
    assert exp2["harness_payload"]["topology_profiles"] == mutated_profiles
    assert PMS_NOTE in exp2["pms_prompt"]
    final_assets = baseline["summary"]["final_parent_assets"]
    assert final_assets["harness_path"] == exp2["harness_config_path"]


# ---------------------------------------------------------------------------
# Follow-up 2: variants derive from, and replace, the asset resolved for the model
# ---------------------------------------------------------------------------

GPT5_ATOM_MAPPING = "GPT-5 lane: map atoms using explicit element labels."
GPT5_PMS = "GPT-5 lane: propose exactly one elementary step."


def test_call_asset_override_is_scoped_to_its_model(tmp_path, monkeypatch) -> None:
    repo = _make_repo(tmp_path)
    _write_model_lane(repo, "propose_mechanism_step", "gpt-5", body=GPT5_PMS)
    monkeypatch.chdir(repo)
    variant = tmp_path / "variant.SKILL.md"
    variant.write_text("<!-- PROMPT_START -->\nVARIANT BODY\n<!-- PROMPT_END -->\n", encoding="utf-8")

    with call_asset_overrides(prompts={"propose_mechanism_step": variant}, model_name="gpt-5"):
        assert "VARIANT BODY" in compose_system_prompt(call_name="propose_mechanism_step", dynamic_system_prompt="", model_name="gpt-5")
        assert "VARIANT BODY" not in compose_system_prompt(call_name="propose_mechanism_step", dynamic_system_prompt="")
    # A shared-scope variant replaces the base prompt, not a model's own override.
    with call_asset_overrides(prompts={"propose_mechanism_step": variant}):
        assert "VARIANT BODY" in compose_system_prompt(call_name="propose_mechanism_step", dynamic_system_prompt="")
        gpt5 = compose_system_prompt(call_name="propose_mechanism_step", dynamic_system_prompt="", model_name="gpt-5")
        assert GPT5_PMS in gpt5 and "VARIANT BODY" not in gpt5


def test_island_prompt_variant_derives_from_model_override(tmp_path, monkeypatch) -> None:
    result = _run_island(
        tmp_path, monkeypatch, island_ids=["mapping"],
        proposals=[_proposal("prompt", "attempt_atom_mapping", "append_instruction", APPENDED)],
        before_run=lambda repo: _write_model_lane(repo, "attempt_atom_mapping", "gpt-5", body=GPT5_ATOM_MAPPING),
    )
    [run] = result["observed"]
    # The run's model (gpt-5) sees its own lane plus the edit, not the base prompt plus the edit.
    assert GPT5_ATOM_MAPPING in run["atom_mapping_prompt_gpt5"]
    assert APPENDED in run["atom_mapping_prompt_gpt5"]
    assert "Map atoms from reactants to products." not in run["atom_mapping_prompt_gpt5"]
    # The shared prompt (other models) is untouched: the variant is scoped to gpt-5.
    assert APPENDED not in run["atom_mapping_prompt"]
    [child] = _children(result["store"], "mapping")
    [variant] = json.loads(child.harness_config_json)["asset_state"]["call_variants"]
    assert variant["scope_model"] == "gpt-5"


def test_overnight_prompt_variant_derives_from_model_override(tmp_path, monkeypatch) -> None:
    baseline, experiment = _run_overnight(
        tmp_path, monkeypatch, "prompt",
        run_config={"harness_name": "default", "model_name": "gpt-5"},
        before_run=lambda repo: _write_model_lane(repo, "propose_mechanism_step", "gpt-5", body=GPT5_PMS),
    )
    assert GPT5_PMS in baseline["pms_prompt_gpt5"]
    assert GPT5_PMS in experiment["pms_prompt_gpt5"]
    assert PMS_NOTE in experiment["pms_prompt_gpt5"]
    assert PMS_NOTE not in experiment["pms_prompt"]


def test_overnight_few_shot_variant_derives_from_model_override(tmp_path, monkeypatch) -> None:
    lane_rows = [{"input": "gpt5 input A", "output": "a"}, {"input": "gpt5 input B", "output": "b"}]
    baseline, experiment = _run_overnight(
        tmp_path, monkeypatch, "few_shot",
        run_config={"harness_name": "default", "model_name": "gpt-5"},
        before_run=lambda repo: _write_model_lane(repo, "propose_mechanism_step", "gpt-5", few_shots=lane_rows),
    )
    assert baseline["pms_few_shot_inputs_gpt5"] == ["gpt5 input A", "gpt5 input B", "pms input ZERO", "pms input ONE"]
    # The blind mutator dropped the last example of the gpt-5 lane; the shared base is still merged.
    assert experiment["pms_few_shot_inputs_gpt5"] == ["gpt5 input A", "pms input ZERO", "pms input ONE"]
    assert experiment["pms_few_shot_inputs"] == ["pms input ZERO", "pms input ONE"]


# ---------------------------------------------------------------------------
# Follow-up 3: a dry island run leaves the checkout clean
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false", *args],
        cwd=repo, check=True, capture_output=True, text=True,
    ).stdout


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required")
@pytest.mark.parametrize(
    ("island_id", "proposal"),
    [
        ("mapping", _proposal("prompt", "attempt_atom_mapping", "append_instruction", APPENDED)),
        ("mapping", _proposal("few_shot", "attempt_atom_mapping", "remove_few_shot", 0)),
        ("topology", _proposal("topology", "centralized_mas.max_candidates_per_agent", "set_field", 5)),
    ],
)
def test_island_dry_run_leaves_repo_checkout_clean(tmp_path, monkeypatch, island_id, proposal) -> None:
    def _commit_checkout(repo: Path) -> None:
        shutil.copyfile(_ROOT / ".gitignore", repo / ".gitignore")
        _git(repo, "init", "-q")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "checkout")

    result = _run_island(
        tmp_path, monkeypatch, island_ids=[island_id], proposals=[proposal], dry_run=True,
        before_run=_commit_checkout,
    )
    repo = result["repo"]
    assert _git(repo, "status", "--porcelain", "--untracked-files=all") == ""
    assert sorted(p.name for p in (repo / "harness_versions" / "default").iterdir()) == ["harness.json"]
    [run] = result["observed"]
    if run["harness_config_path"]:
        assert Path(run["harness_config_path"]).resolve().is_relative_to((repo / "evolution_workspace").resolve())
