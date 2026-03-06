"""Tests for the coordination topology selector feature."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from mechanistic_agent.core.types import (
    CoordinationTopology,
    HarnessConfig,
    RunConfig,
    RunInput,
    RunState,
    StepResult,
    TopologyProfile,
)


# ---------------------------------------------------------------------------
# TopologyProfile unit tests
# ---------------------------------------------------------------------------

class TestTopologyProfile:
    def test_defaults(self):
        p = TopologyProfile()
        assert p.agent_count == 1
        assert p.max_candidates_per_agent == 3
        assert p.peer_rounds == 0
        assert p.aggregation_mode == "orchestrator_select"

    def test_from_dict_round_trip(self):
        data = {
            "agent_count": 3,
            "max_candidates_per_agent": 2,
            "peer_rounds": 2,
            "aggregation_mode": "consensus",
            "consensus_key": "reaction_smirks",
            "consensus_fallback_key": "intermediate_smiles",
            "description": "test",
        }
        p = TopologyProfile.from_dict(data)
        assert p.agent_count == 3
        assert p.peer_rounds == 2
        assert p.as_dict() == data

    def test_from_dict_missing_keys(self):
        p = TopologyProfile.from_dict({})
        assert p.agent_count == 1
        assert p.aggregation_mode == "orchestrator_select"

    def test_from_dict_none(self):
        p = TopologyProfile.from_dict(None)
        assert p.agent_count == 1


# ---------------------------------------------------------------------------
# HarnessConfig topology profile tests
# ---------------------------------------------------------------------------

class TestHarnessConfigTopology:
    def test_from_dict_with_profiles(self):
        data = {
            "name": "test",
            "topology_profiles": {
                "sas": {"agent_count": 1, "max_candidates_per_agent": 1},
                "centralized_mas": {"agent_count": 1, "max_candidates_per_agent": 3},
            },
        }
        config = HarnessConfig.from_dict(data)
        assert "sas" in config.topology_profiles
        assert config.topology_profiles["sas"].max_candidates_per_agent == 1

    def test_from_dict_without_profiles(self):
        config = HarnessConfig.from_dict({"name": "test"})
        assert config.topology_profiles == {}

    def test_get_topology_profile_hit(self):
        config = HarnessConfig.from_dict({
            "topology_profiles": {
                "sas": {"agent_count": 1, "max_candidates_per_agent": 1},
            },
        })
        profile = config.get_topology_profile("sas")
        assert profile.max_candidates_per_agent == 1

    def test_get_topology_profile_fallback(self):
        config = HarnessConfig.from_dict({
            "topology_profiles": {
                "centralized_mas": {"agent_count": 1, "max_candidates_per_agent": 3},
            },
        })
        profile = config.get_topology_profile("sas")
        assert profile.max_candidates_per_agent == 3  # fell back to centralized_mas

    def test_get_topology_profile_empty(self):
        config = HarnessConfig.from_dict({})
        profile = config.get_topology_profile("sas")
        assert profile.agent_count == 1  # default TopologyProfile

    def test_as_dict_includes_profiles(self):
        config = HarnessConfig.from_dict({
            "topology_profiles": {
                "sas": {"agent_count": 1, "max_candidates_per_agent": 1},
            },
        })
        d = config.as_dict()
        assert "topology_profiles" in d
        assert d["topology_profiles"]["sas"]["agent_count"] == 1

    def test_as_dict_omits_empty_profiles(self):
        config = HarnessConfig.from_dict({})
        d = config.as_dict()
        assert "topology_profiles" not in d


# ---------------------------------------------------------------------------
# RunConfig coordination_topology field
# ---------------------------------------------------------------------------

class TestRunConfigTopology:
    def test_default_topology(self):
        rc = RunConfig(model="test-model")
        assert rc.coordination_topology == "centralized_mas"

    def test_custom_topology(self):
        rc = RunConfig(model="test-model", coordination_topology="sas")
        assert rc.coordination_topology == "sas"


# ---------------------------------------------------------------------------
# Harness JSON loading test
# ---------------------------------------------------------------------------

class TestHarnessJsonProfiles:
    def test_default_harness_has_profiles(self):
        harness_path = Path(__file__).resolve().parents[2] / "harness_versions" / "default" / "harness.json"
        if not harness_path.exists():
            pytest.skip("default harness.json not found")
        data = json.loads(harness_path.read_text())
        config = HarnessConfig.from_dict(data)
        assert "sas" in config.topology_profiles
        assert "centralized_mas" in config.topology_profiles
        assert "independent_mas" in config.topology_profiles
        assert "decentralized_mas" in config.topology_profiles
        assert config.topology_profiles["sas"].max_candidates_per_agent == 1
        assert config.topology_profiles["independent_mas"].agent_count == 3


# ---------------------------------------------------------------------------
# Coordinator routing tests (mocked)
# ---------------------------------------------------------------------------

def _make_state(topology: str = "centralized_mas") -> RunState:
    return RunState(
        run_id="test-run",
        mode="unverified",
        run_input=RunInput(
            starting_materials=["CCO"],
            products=["CC=O"],
        ),
        run_config=RunConfig(
            model="test-model",
            coordination_topology=topology,  # type: ignore[arg-type]
        ),
    )


def _mock_proposal_output(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"candidates": candidates}


def _make_candidate(rank: int, smiles: str = "C=O", smirks: str = "CCO>>CC=O") -> Dict[str, Any]:
    return {
        "rank": rank,
        "intermediate_smiles": smiles,
        "reaction_smirks": smirks,
        "reaction_description": f"step {rank}",
        "resulting_state": [smiles],
    }


class TestCoordinatorRouting:
    """Test that _propose_for_topology routes correctly."""

    def _make_coordinator(self, mock_run_fn=None):
        from mechanistic_agent.core.coordinator import RunCoordinator
        store = MagicMock()
        store.append_event = MagicMock()
        coord = RunCoordinator(store)
        # Replace slotted agent with a plain mock
        mock_agent = MagicMock()
        if mock_run_fn:
            mock_agent.run = mock_run_fn
        coord.intermediate_agent = mock_agent
        return coord

    def test_sas_returns_single_candidate(self):
        candidates = [_make_candidate(1), _make_candidate(2), _make_candidate(3)]
        mock_result = MagicMock()
        mock_result.output = _mock_proposal_output(candidates)
        mock_result.attempt = 1

        coord = self._make_coordinator()
        coord.intermediate_agent.run = MagicMock(return_value=mock_result)
        state = _make_state("sas")
        harness = HarnessConfig.from_dict({
            "topology_profiles": {
                "sas": {"agent_count": 1, "max_candidates_per_agent": 1},
            },
        })

        output, result_candidates = coord._propose_for_topology(state, harness)
        assert len(result_candidates) == 1
        assert result_candidates[0]["rank"] == 1

    def test_centralized_returns_all_candidates(self):
        candidates = [_make_candidate(1), _make_candidate(2), _make_candidate(3)]
        mock_result = MagicMock()
        mock_result.output = _mock_proposal_output(candidates)
        mock_result.attempt = 1

        coord = self._make_coordinator()
        coord.intermediate_agent.run = MagicMock(return_value=mock_result)
        state = _make_state("centralized_mas")
        harness = HarnessConfig.from_dict({
            "topology_profiles": {
                "centralized_mas": {"agent_count": 1, "max_candidates_per_agent": 3},
            },
        })

        output, result_candidates = coord._propose_for_topology(state, harness)
        assert len(result_candidates) == 3

    def test_independent_fan_out(self):
        call_count = 0

        def mock_run(s, *, template_guidance=None):
            nonlocal call_count
            call_count += 1
            idx = call_count
            result = MagicMock()
            result.output = _mock_proposal_output([
                _make_candidate(1, smiles=f"C{idx}=O"),
                _make_candidate(2, smiles=f"C{idx}C=O"),
            ])
            return result

        coord = self._make_coordinator(mock_run_fn=mock_run)
        state = _make_state("independent_mas")
        harness = HarnessConfig.from_dict({
            "topology_profiles": {
                "independent_mas": {"agent_count": 3, "max_candidates_per_agent": 2},
            },
        })

        output, result_candidates = coord._propose_for_topology(state, harness)
        assert call_count == 3
        assert len(result_candidates) == 6  # 3 agents x 2 candidates
        # All candidates should have source_agent metadata
        agents = {c.get("source_agent") for c in result_candidates}
        assert len(agents) == 3

    def test_decentralized_peer_rounds(self):
        call_count = 0
        peer_proposals_seen: List = []

        def mock_run(s, *, template_guidance=None):
            nonlocal call_count
            call_count += 1
            tg = template_guidance or {}
            if "peer_proposals" in tg:
                peer_proposals_seen.append(tg["peer_proposals"])
            result = MagicMock()
            result.output = _mock_proposal_output([
                _make_candidate(1, smiles=f"C{call_count}=O", smirks=f"CCO>>C{call_count}=O"),
            ])
            return result

        coord = self._make_coordinator(mock_run_fn=mock_run)
        state = _make_state("decentralized_mas")
        harness = HarnessConfig.from_dict({
            "topology_profiles": {
                "decentralized_mas": {
                    "agent_count": 2,
                    "max_candidates_per_agent": 1,
                    "peer_rounds": 2,
                    "aggregation_mode": "consensus",
                },
            },
        })

        output, result_candidates = coord._propose_for_topology(state, harness)
        # 2 agents x 2 rounds = 4 calls
        assert call_count == 4
        # Peer proposals should have been injected in round 2
        assert len(peer_proposals_seen) == 2  # both agents in round 2

    def test_independent_records_aggregated_usage_and_cost(self):
        call_count = 0

        def mock_run(_state, *, template_guidance=None):
            nonlocal call_count
            call_count += 1
            return StepResult(
                step_name="mechanism_step_proposal",
                tool_name="propose_mechanism_step",
                output=_mock_proposal_output([_make_candidate(1, smiles=f"C{call_count}=O")]),
                source="llm",
                token_usage={
                    "input_tokens": 100,
                    "cached_input_tokens": 20,
                    "output_tokens": 30,
                    "total_tokens": 150,
                },
                cost={
                    "input_cost": 0.001,
                    "cached_input_cost": 0.0002,
                    "output_cost": 0.002,
                    "total_cost": 0.0032,
                },
            )

        captured: List[StepResult] = []
        coord = self._make_coordinator(mock_run_fn=mock_run)
        coord._record_step = lambda _state, result: captured.append(result)  # type: ignore[method-assign]
        state = _make_state("independent_mas")
        harness = HarnessConfig.from_dict(
            {"topology_profiles": {"independent_mas": {"agent_count": 3, "max_candidates_per_agent": 2}}}
        )

        output, _ = coord._propose_for_topology(state, harness)
        assert call_count == 3
        assert len(captured) == 1
        assert captured[0].token_usage is not None
        assert captured[0].cost is not None
        assert captured[0].token_usage["total_tokens"] == 450
        assert round(captured[0].cost["total_cost"], 6) == 0.0096
        assert len(output["agent_usage_cost"]) == 3
        assert output["aggregated_usage_cost"]["cost"]["total_cost"] > 0

    def test_decentralized_records_aggregated_usage_and_cost(self):
        call_count = 0

        def mock_run(_state, *, template_guidance=None):
            nonlocal call_count
            call_count += 1
            return StepResult(
                step_name="mechanism_step_proposal",
                tool_name="propose_mechanism_step",
                output=_mock_proposal_output([_make_candidate(1, smiles=f"C{call_count}=O", smirks="A>>B")]),
                source="llm",
                token_usage={
                    "input_tokens": 50,
                    "cached_input_tokens": 10,
                    "output_tokens": 20,
                    "total_tokens": 80,
                },
                cost={
                    "input_cost": 0.0005,
                    "cached_input_cost": 0.0001,
                    "output_cost": 0.001,
                    "total_cost": 0.0016,
                },
            )

        captured: List[StepResult] = []
        coord = self._make_coordinator(mock_run_fn=mock_run)
        coord._record_step = lambda _state, result: captured.append(result)  # type: ignore[method-assign]
        state = _make_state("decentralized_mas")
        harness = HarnessConfig.from_dict(
            {
                "topology_profiles": {
                    "decentralized_mas": {
                        "agent_count": 2,
                        "max_candidates_per_agent": 1,
                        "peer_rounds": 2,
                        "aggregation_mode": "consensus",
                    }
                }
            }
        )

        output, _ = coord._propose_for_topology(state, harness)
        assert call_count == 4
        assert len(captured) == 1
        assert captured[0].token_usage is not None
        assert captured[0].cost is not None
        assert captured[0].token_usage["total_tokens"] == 320
        assert round(captured[0].cost["total_cost"], 6) == 0.0064
        assert len(output["agent_usage_cost"]) == 4
        assert all(isinstance(item.get("round"), int) for item in output["agent_usage_cost"])

    def test_default_topology_is_centralized(self):
        mock_result = MagicMock()
        mock_result.output = _mock_proposal_output([_make_candidate(1)])
        mock_result.attempt = 1

        coord = self._make_coordinator()
        coord.intermediate_agent.run = MagicMock(return_value=mock_result)
        state = _make_state()  # default = centralized_mas
        harness = HarnessConfig.from_dict({})

        output, candidates = coord._propose_for_topology(state, harness)
        assert len(candidates) == 1


# ---------------------------------------------------------------------------
# Consensus merge unit tests
# ---------------------------------------------------------------------------

class TestConsensusMerge:
    def test_consensus_bonus(self):
        from mechanistic_agent.core.coordinator import RunCoordinator

        profile = TopologyProfile(consensus_key="reaction_smirks")
        round_outputs = [
            [_make_candidate(1, smiles="C=O", smirks="CCO>>CC=O")],
            [_make_candidate(1, smiles="C=O", smirks="CCO>>CC=O")],
            [_make_candidate(1, smiles="CC", smirks="CCO>>CC")],
        ]
        merged = RunCoordinator._consensus_merge(round_outputs, profile)
        # CCO>>CC=O has 2-agent support, should rank first
        assert merged[0]["reaction_smirks"] == "CCO>>CC=O"
        assert merged[1]["reaction_smirks"] == "CCO>>CC"

    def test_dedup_by_consensus_key(self):
        from mechanistic_agent.core.coordinator import RunCoordinator

        profile = TopologyProfile(consensus_key="reaction_smirks")
        round_outputs = [
            [_make_candidate(1, smirks="A>>B"), _make_candidate(2, smirks="C>>D")],
            [_make_candidate(1, smirks="A>>B")],
        ]
        merged = RunCoordinator._consensus_merge(round_outputs, profile)
        smirks_list = [c["reaction_smirks"] for c in merged]
        assert smirks_list.count("A>>B") == 1  # deduped
        assert "C>>D" in smirks_list

    def test_empty_outputs(self):
        from mechanistic_agent.core.coordinator import RunCoordinator

        profile = TopologyProfile()
        merged = RunCoordinator._consensus_merge([[], [], []], profile)
        assert merged == []

    def test_deterministic_tiebreak(self):
        from mechanistic_agent.core.coordinator import RunCoordinator

        profile = TopologyProfile(consensus_key="reaction_smirks")
        round_outputs = [
            [_make_candidate(2, smirks="A>>B")],
            [_make_candidate(1, smirks="C>>D")],
        ]
        merged = RunCoordinator._consensus_merge(round_outputs, profile)
        # Both have support=1; C>>D has lower original rank (1) so ranks first
        assert merged[0]["reaction_smirks"] == "C>>D"


# ---------------------------------------------------------------------------
# Build peer summaries tests
# ---------------------------------------------------------------------------

class TestBuildPeerSummaries:
    def test_excludes_self(self):
        from mechanistic_agent.core.coordinator import RunCoordinator

        round_outputs = [
            [_make_candidate(1, smiles="A")],
            [_make_candidate(1, smiles="B")],
            [_make_candidate(1, smiles="C")],
        ]
        summaries = RunCoordinator._build_peer_summaries(round_outputs, current_round=1)
        assert len(summaries) == 3
        # Agent 0 should see B and C but not A
        agent0_smiles = {p["smiles"] for p in summaries[0]}
        assert "A" not in agent0_smiles
        assert "B" in agent0_smiles
        assert "C" in agent0_smiles


# ---------------------------------------------------------------------------
# Config roundtrip (coordination_topology survives serialization)
# ---------------------------------------------------------------------------

class TestConfigRoundtrip:
    def test_build_state_parses_topology(self):
        from mechanistic_agent.core.coordinator import RunCoordinator

        store = MagicMock()
        store.list_step_outputs = MagicMock(return_value=[])
        coord = RunCoordinator(store)

        run_row = {
            "id": "test-run",
            "mode": "unverified",
            "input_payload": {
                "starting_materials": ["CCO"],
                "products": ["CC=O"],
            },
            "config": {
                "model": "test-model",
                "coordination_topology": "independent_mas",
            },
        }
        state = coord._build_state(run_row)
        assert state.run_config.coordination_topology == "independent_mas"

    def test_build_state_defaults_missing_topology(self):
        from mechanistic_agent.core.coordinator import RunCoordinator

        store = MagicMock()
        store.list_step_outputs = MagicMock(return_value=[])
        coord = RunCoordinator(store)

        run_row = {
            "id": "test-run",
            "mode": "unverified",
            "input_payload": {
                "starting_materials": ["CCO"],
                "products": ["CC=O"],
            },
            "config": {
                "model": "test-model",
                # no coordination_topology key
            },
        }
        state = coord._build_state(run_row)
        assert state.run_config.coordination_topology == "centralized_mas"

    def test_build_state_invalid_topology_defaults(self):
        from mechanistic_agent.core.coordinator import RunCoordinator

        store = MagicMock()
        store.list_step_outputs = MagicMock(return_value=[])
        coord = RunCoordinator(store)

        run_row = {
            "id": "test-run",
            "mode": "unverified",
            "input_payload": {
                "starting_materials": ["CCO"],
                "products": ["CC=O"],
            },
            "config": {
                "model": "test-model",
                "coordination_topology": "invalid_value",
            },
        }
        state = coord._build_state(run_row)
        assert state.run_config.coordination_topology == "centralized_mas"


# ---------------------------------------------------------------------------
# API schema tests
# ---------------------------------------------------------------------------

class TestAPISchema:
    @pytest.fixture(autouse=True)
    def _skip_without_fastapi(self):
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("fastapi not installed")

    def test_create_run_request_default_topology(self):
        from mechanistic_agent.api.schemas import CreateRunRequest

        req = CreateRunRequest(
            starting_materials=["CCO"],
            products=["CC=O"],
        )
        assert req.coordination_topology == "centralized_mas"

    def test_create_run_request_custom_topology(self):
        from mechanistic_agent.api.schemas import CreateRunRequest

        req = CreateRunRequest(
            starting_materials=["CCO"],
            products=["CC=O"],
            coordination_topology="decentralized_mas",
        )
        assert req.coordination_topology == "decentralized_mas"
