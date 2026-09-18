"""Trace-conditioned LLM mutation proposals: digest, application, fallback, wiring."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from mechanistic_agent.core.llm_mutator import (
    LLMLaneMutator,
    MutationProposal,
    ProposalRejected,
    build_failure_digest,
    digest_from_case_results,
)
from mechanistic_agent.core.overnight_ralph import load_overnight_program
from mechanistic_agent.tool_schemas import HARNESS_MUTATION_TOOL


class _StubResponse:
    def __init__(self, arguments: dict) -> None:
        self.tool_calls = [{"name": "harness_mutation_proposal", "arguments": json.dumps(arguments)}]
        self.content = ""


class _StubLLM:
    def __init__(self, arguments: dict) -> None:
        self.arguments = arguments
        self.calls: list[dict] = []

    def invoke(self, messages, tools=None, tool_choice=None):  # noqa: ANN001
        self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
        return _StubResponse(self.arguments)


def _harness(tmp_path: Path) -> Path:
    path = tmp_path / "harness.json"
    path.write_text(
        json.dumps(
            {
                "name": "default",
                "pre_loop_modules": [
                    {"id": "missing_reagents", "kind": "llm", "enabled": True, "removable": True},
                    {"id": "balance_analysis", "kind": "deterministic", "enabled": True, "removable": False},
                ],
                "post_step_modules": [],
                "run_config_defaults": {"proceed_on_validation_failure": False},
                "topology_profiles": {"centralized_mas": {"agent_count": 1, "max_candidates_per_agent": 3, "peer_rounds": 0}},
                "metadata": {"changelog": []},
            }
        ),
        encoding="utf-8",
    )
    return path


def _skills(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "mechanistic" / "propose_mechanism_step"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\ncall_name: propose_mechanism_step\n---\n<!-- PROMPT_START -->\nPropose the next step.\n- Keep SMILES valid.\n<!-- PROMPT_END -->\n",
        encoding="utf-8",
    )
    (skill_dir / "few_shot.jsonl").write_text(
        json.dumps({"input": "a", "output": "b"}) + "\n" + json.dumps({"input": "c", "output": "d"}) + "\n",
        encoding="utf-8",
    )


def _snapshot() -> dict:
    return {
        "id": "run-1",
        "status": "failed",
        "input": {"starting_materials": ["O=C(O)C=Cc1cncc(Br)c1", "O=S(Cl)Cl"], "products": ["O=C(Cl)C=Cc1cncc(Br)c1", "Cl", "O=S=O"]},
        "step_outputs": [
            {
                "step_name": "mechanism_synthesis",
                "validation": {
                    "passed": False,
                    "checks": [
                        {"name": "atom_balance", "passed": False, "details": {"error": "Atom imbalance detected: Cl: 2->4 (+2)"}},
                        {"name": "state_progress", "passed": True},
                    ],
                },
                "output": {"current_state": ["A", "O=S(Cl)Cl"], "resulting_state": ["B", "O=S(Cl)Cl"], "reaction_smirks": "A>>B |mech:v1;lp:1>2|"},
            },
            {"step_name": "mechanism_synthesis", "validation": {"passed": True}, "output": {}},
        ],
        "events": [
            {"event_type": "mechanism_reproposal_requested", "payload": {"reason": "atom_balance_dead_end"}},
            {"event_type": "candidate_rescue_completed", "payload": {"status": "success"}},
            {"event_type": "run_failed", "payload": {"reason": "proposal_incomplete_loop"}},
        ],
    }


def test_failure_digest_is_compact_and_model_visible_only() -> None:
    digest = build_failure_digest([_snapshot()])
    assert len(digest) == 1
    entry = digest[0]
    assert entry["failed_checks"] == {"atom_balance": 1}
    assert entry["accepted_steps"] == 1
    assert entry["reproposal_reasons"] == {"atom_balance_dead_end": 1}
    assert entry["rescue_outcomes"] == {"success": 1}
    assert entry["terminal_reason"] == "run_failed:proposal_incomplete_loop"
    assert "Cl: 2->4" in entry["failure_examples"][0]["error"]
    serialized = json.dumps(digest)
    assert "verified_mechanism" not in serialized and "known_mechanism" not in serialized


def test_digest_from_case_results_adapts_evolve_harness_shape() -> None:
    results = [{"case_id": "c1", "passed": False, "run_status": "failed", "error": "boom", "input_payload": {"starting_materials": ["A"], "products": ["B"]}, "step_outputs": []}]
    digest = digest_from_case_results(results)
    assert digest[0]["terminal_reason"] == "run_failed:boom"


def test_llm_mutator_applies_harness_proposal_and_records_changelog(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    _skills(tmp_path)
    stub = _StubLLM({"lane": "harness", "operation": "set_enabled", "target": "missing_reagents", "value": False, "rationale": "reagent injection double-counts SOCl2"})
    mutator = LLMLaneMutator(base_dir=tmp_path, model_name="stub-model", chat_model_factory=lambda _name: stub)
    result = mutator.propose(harness, failure_digest=build_failure_digest([_snapshot()]), allowed_lanes=["harness", "prompt"])
    assert result.lane == "harness"
    payload = json.loads(result.asset_path.read_text())
    module = next(m for m in payload["pre_loop_modules"] if m["id"] == "missing_reagents")
    assert module["enabled"] is False
    assert "LLM-proposed" in payload["metadata"]["changelog"][-1]["description"]
    # The model was asked with the forced tool and saw the digest.
    call = stub.calls[0]
    assert call["tools"] == [HARNESS_MUTATION_TOOL]
    assert call["tool_choice"]["function"]["name"] == "harness_mutation_proposal"
    assert "atom_balance" in call["messages"][1]["content"]


def test_llm_mutator_prompt_and_few_shot_lanes(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    _skills(tmp_path)
    prompt_stub = _StubLLM({"lane": "prompt", "operation": "append_instruction", "target": "propose_mechanism_step", "value": "List every unchanged spectator in resulting_state.", "rationale": "r"})
    result = LLMLaneMutator(base_dir=tmp_path, model_name="m", chat_model_factory=lambda _n: prompt_stub).propose(harness, allowed_lanes=["prompt"])
    text = result.asset_path.read_text()
    assert "List every unchanged spectator" in text and text.index("List every") < text.index("<!-- PROMPT_END -->")

    fs_stub = _StubLLM({"lane": "few_shot", "operation": "remove_few_shot", "target": "propose_mechanism_step", "value": 0, "rationale": "r"})
    result_fs = LLMLaneMutator(base_dir=tmp_path, model_name="m", chat_model_factory=lambda _n: fs_stub).propose(harness, allowed_lanes=["few_shot"])
    lines = [json.loads(line) for line in result_fs.asset_path.read_text().splitlines() if line.strip()]
    assert lines == [{"input": "c", "output": "d"}]


def test_llm_mutator_rejects_frozen_or_invalid_targets_and_falls_back(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    _skills(tmp_path)
    # Load-bearing module cannot be disabled -> proposal rejected -> blind fallback in the preferred lane.
    stub = _StubLLM({"lane": "harness", "operation": "set_enabled", "target": "balance_analysis", "value": False, "rationale": "r"})
    mutator = LLMLaneMutator(base_dir=tmp_path, model_name="m", chat_model_factory=lambda _n: stub)
    result = mutator.propose(harness, allowed_lanes=["topology", "harness"], preferred_lane="topology")
    assert result.lane == "topology"
    assert result.summary.startswith("[llm proposal rejected")
    assert "load-bearing" in result.metadata["llm_error"]

    # Disallowed lane is rejected even when the edit itself would be valid.
    stub2 = _StubLLM({"lane": "prompt", "operation": "append_instruction", "target": "propose_mechanism_step", "value": "x", "rationale": "r"})
    result2 = LLMLaneMutator(base_dir=tmp_path, model_name="m", chat_model_factory=lambda _n: stub2).propose(harness, allowed_lanes=["topology"])
    assert result2.lane == "topology" and "not allowed" in result2.metadata["llm_error"]

    # Direct apply of a bad topology field raises.
    try:
        mutator.apply(MutationProposal(lane="topology", target="centralized_mas.not_a_field", operation="set_field", value=2, rationale="r"), harness)
    except ProposalRejected as exc:
        assert "not editable" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ProposalRejected")


def test_program_file_parses_mutation_proposer(tmp_path: Path) -> None:
    program = tmp_path / "ralph_program.md"
    program.write_text("allowed_lanes:\n- topology\nmutation_proposer: llm\nmutation_model: agent-bridge\n", encoding="utf-8")
    cfg = load_overnight_program(program)
    assert cfg.mutation_proposer == "llm"
    assert cfg.mutation_model == "agent-bridge"
    default_cfg = load_overnight_program(Path(__file__).resolve().parents[2] / "ralph_program.md")
    assert default_cfg.mutation_proposer == "random"


def test_overnight_orchestrator_routes_to_llm_mutator(tmp_path: Path) -> None:
    from mechanistic_agent.core.db import RunStore
    from mechanistic_agent.core.overnight_ralph import OvernightRalphOrchestrator

    harness = _harness(tmp_path)
    _skills(tmp_path)
    stub = _StubLLM({"lane": "topology", "operation": "set_field", "target": "centralized_mas.max_candidates_per_agent", "value": 2, "rationale": "fewer alternates"})
    orchestrator = OvernightRalphOrchestrator(base_dir=tmp_path, store=RunStore(tmp_path / "mechanistic.db"), chat_model_factory=lambda _n: stub)
    orchestrator._mutation_proposer = "llm"
    orchestrator._mutation_model = "stub"
    orchestrator._allowed_lanes = ["topology", "harness"]
    orchestrator._latest_failure_digest = build_failure_digest([_snapshot()])
    result = orchestrator._propose_mutation(lane="topology", parent_asset=harness)
    assert result.summary == "centralized_mas.max_candidates_per_agent: 3 -> 2"
    assert json.loads(result.asset_path.read_text())["topology_profiles"]["centralized_mas"]["max_candidates_per_agent"] == 2
