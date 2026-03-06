from __future__ import annotations

from mechanistic_agent.scoring import score_snapshot_against_known, score_subagents_from_step_outputs


def _expected(min_steps: int = 2):
    return {
        "known_mechanism": {
            "source": "FlowER flower_new_dataset train.txt",
            "min_steps": min_steps,
            "steps": [
                {"step_index": 1, "target_smiles": "INT1"},
                {"step_index": 2, "target_smiles": "P"},
            ],
        }
    }


def test_scoring_near_perfect_for_exact_path() -> None:
    snapshot = {
        "events": [
            {
                "seq": 1,
                "event_type": "mechanism_step_accepted",
                "payload": {
                    "step_index": 1,
                    "candidate_rank": 1,
                    "current_state": ["A"],
                    "resulting_state": ["INT1"],
                    "contains_target_product": False,
                    "validation_summary": {
                        "passed": True,
                        "checks": [
                            {"name": "dbe_metadata", "passed": True},
                            {"name": "atom_balance", "passed": True},
                            {"name": "state_progress", "passed": True},
                        ],
                    },
                },
            },
            {
                "seq": 2,
                "event_type": "mechanism_step_accepted",
                "payload": {
                    "step_index": 2,
                    "candidate_rank": 1,
                    "current_state": ["INT1"],
                    "resulting_state": ["P"],
                    "contains_target_product": True,
                    "validation_summary": {
                        "passed": True,
                        "checks": [
                            {"name": "dbe_metadata", "passed": True},
                            {"name": "atom_balance", "passed": True},
                            {"name": "state_progress", "passed": True},
                        ],
                    },
                },
            },
        ],
        "step_outputs": [],
    }
    scored = score_snapshot_against_known(snapshot, _expected())
    assert scored["score"] > 0.9
    assert scored["passed"] is True


def test_scoring_high_but_not_perfect_for_reasonable_non_identical_path() -> None:
    snapshot = {
        "events": [
            {
                "seq": 1,
                "event_type": "mechanism_step_accepted",
                "payload": {
                    "step_index": 1,
                    "resulting_state": ["ALT1"],
                    "validation_summary": {
                        "checks": [
                            {"name": "dbe_metadata", "passed": True},
                            {"name": "atom_balance", "passed": True},
                            {"name": "state_progress", "passed": True},
                        ],
                    },
                },
            },
            {
                "seq": 2,
                "event_type": "mechanism_step_accepted",
                "payload": {
                    "step_index": 2,
                    "resulting_state": ["P"],
                    "contains_target_product": True,
                    "validation_summary": {
                        "checks": [
                            {"name": "dbe_metadata", "passed": True},
                            {"name": "atom_balance", "passed": True},
                            {"name": "state_progress", "passed": True},
                        ],
                    },
                },
            },
        ],
        "step_outputs": [],
    }
    scored = score_snapshot_against_known(snapshot, _expected())
    assert 0.7 <= scored["score"] < 1.0


def test_scoring_caps_when_final_product_not_reached() -> None:
    snapshot = {
        "events": [
            {
                "seq": 1,
                "event_type": "mechanism_step_accepted",
                "payload": {
                    "step_index": 1,
                    "resulting_state": ["INT1"],
                    "validation_summary": {
                        "checks": [
                            {"name": "dbe_metadata", "passed": True},
                            {"name": "atom_balance", "passed": True},
                            {"name": "state_progress", "passed": True},
                        ],
                    },
                },
            },
            {
                "seq": 2,
                "event_type": "mechanism_step_accepted",
                "payload": {
                    "step_index": 2,
                    "resulting_state": ["ALT2"],
                    "contains_target_product": False,
                    "validation_summary": {
                        "checks": [
                            {"name": "dbe_metadata", "passed": True},
                            {"name": "atom_balance", "passed": True},
                            {"name": "state_progress", "passed": True},
                        ],
                    },
                },
            },
        ],
        "step_outputs": [],
    }
    scored = score_snapshot_against_known(snapshot, _expected())
    assert scored["score"] <= 0.55
    assert scored["passed"] is False


def test_scoring_penalizes_repeated_or_circular_steps() -> None:
    snapshot = {
        "events": [
            {
                "seq": 1,
                "event_type": "mechanism_step_accepted",
                "payload": {
                    "step_index": 1,
                    "resulting_state": ["INT1"],
                    "validation_summary": {"checks": [{"name": "dbe_metadata", "passed": True}]},
                },
            },
            {
                "seq": 2,
                "event_type": "mechanism_step_accepted",
                "payload": {
                    "step_index": 2,
                    "resulting_state": ["INT1"],
                    "validation_summary": {"checks": [{"name": "dbe_metadata", "passed": True}]},
                },
            },
            {
                "seq": 3,
                "event_type": "mechanism_step_accepted",
                "payload": {
                    "step_index": 3,
                    "resulting_state": ["P"],
                    "contains_target_product": True,
                    "validation_summary": {"checks": [{"name": "dbe_metadata", "passed": True}]},
                },
            },
        ],
        "step_outputs": [],
    }
    scored = score_snapshot_against_known(snapshot, _expected(min_steps=2))
    assert scored["efficiency_penalty_total"] > 0.0


def test_subagent_scoring_coerces_legacy_step_mapping_confidence() -> None:
    step_outputs = [
        {
            "step_name": "step_atom_mapping",
            "attempt": 1,
            "retry_index": 0,
            "output": {"confidence": "high"},
        }
    ]
    scored = score_subagents_from_step_outputs(step_outputs)
    assert scored["step_atom_mapping"]["quality_score"] == 0.9
    assert scored["step_atom_mapping"]["pass_rate"] == 1.0


def test_subagent_validation_scoring_prefers_latest_outcome_per_attempt_retry() -> None:
    step_outputs = [
        {
            "step_name": "atom_balance_validation",
            "attempt": 1,
            "retry_index": 0,
            "output": {"check": "atom_balance", "passed": False, "details": {"balanced": False}},
        },
        {
            "step_name": "atom_balance_validation",
            "attempt": 1,
            "retry_index": 0,
            "output": {"check": "atom_balance", "passed": True, "details": {"balanced": True}},
        },
    ]
    scored = score_subagents_from_step_outputs(step_outputs)
    assert scored["atom_balance_validation"]["calls"] == 1
    assert scored["atom_balance_validation"]["quality_score"] == 1.0
    assert scored["atom_balance_validation"]["pass_rate"] == 1.0


def test_subagent_mechanism_proposal_quality_reflects_downstream_validation() -> None:
    step_outputs = [
        {
            "step_name": "mechanism_step_proposal",
            "attempt": 1,
            "retry_index": 0,
            "output": {"candidates": [{"rank": 1, "intermediate_smiles": "CCCl"}]},
        },
        {
            "step_name": "mechanism_synthesis",
            "attempt": 1,
            "retry_index": 0,
            "validation": {"passed": False, "checks": [{"name": "atom_balance", "passed": False}]},
            "output": {"resulting_state": ["CCCl"]},
        },
    ]
    scored = score_subagents_from_step_outputs(step_outputs)
    assert scored["mechanism_step_proposal"]["pass_rate"] == 1.0
    assert scored["mechanism_step_proposal"]["quality_score"] < 0.5
