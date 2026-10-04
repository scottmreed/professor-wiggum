"""The mutated variant must be the one an overnight Ralph eval actually runs.

Overnight Ralph (``OvernightRalphOrchestrator.run``) mutates a sibling asset (a
harness JSON variant, a ``prompt_variant_*.SKILL.md`` or a
``few_shot_variant_*.jsonl``) and then scores an evaluation slice. These tests
drive the real loop with the model calls mocked out and inspect, at the moment
each slice runs, the harness and prompt assets it resolves. A mutation that is
not observable there means the keep/discard rule is scoring the parent.

Follow-ups covered further down:

* a kept variant is the parent of later mutations, and Ralph compares
  experiments against a baseline evaluated under the same base assets;
* prompt / few-shot variants are derived from the asset resolved for the run's
  model (a ``models/<slug>/`` override when present) and applied at that scope.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import pytest

from mechanistic_agent.core.db import RunStore
from mechanistic_agent.core.overnight_ralph import OvernightRalphOrchestrator
from mechanistic_agent.core.types import MicroEvalResult, OvernightRalphConfig
from mechanistic_agent.prompt_assets import (
    call_asset_overrides,
    compose_system_prompt,
    load_call_few_shot_examples,
)

_ROOT = Path(__file__).resolve().parents[2]

ATOM_MAPPING_FEW_SHOTS = [
    {"input": "few-shot input ZERO", "output": "few-shot output ZERO"},
    {"input": "few-shot input ONE", "output": "few-shot output ONE"},
]


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

PMS_NOTE = "Mutation note: prefer concise mechanism-step proposals."


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
