"""Tests for delta bond-electron metadata handling in predict_mechanistic_step."""

from __future__ import annotations

import json

import pytest

from mechanistic_agent.tools import predict_mechanistic_step


def _base_args(**overrides):
    args = {
        "step_index": 0,
        "current_state": ["[CH3:1][CH2:2][Br:3]", "[Cl-:4]"],
        "target_products": ["[CH3:1][CH2:2][Cl:4]", "[Br-:3]"],
        "predicted_intermediate": "[CH3:1][CH2:2][Cl:4]",
        "resulting_state": ["[CH3:1][CH2:2][Cl:4]", "[Br-:3]"],
        "electron_pushes": [
            {"kind": "lone_pair", "source_atom": "4", "target_atom": "2", "electrons": 2},
            {"kind": "sigma_bond", "source_bond": ["2", "3"], "through_atom": "3", "target_atom": "3", "electrons": 2},
        ],
    }
    args.update(overrides)
    return args


def test_predict_mechanistic_step_accepts_valid_inferred_dbe_block() -> None:
    reaction = (
        "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] "
        "|mech:v1;lp:4>2;sigma:2-3>3|"
    )

    payload = json.loads(predict_mechanistic_step(**_base_args(reaction_smirks=reaction)))

    assert payload["status"] == "accepted"
    assert payload["bond_electron_validation"]["valid"] is True
    assert payload["reaction_smirks"] == "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3]"
    assert len(payload["bond_electron_deltas"]) == 4
    assert payload["bond_electron_validation"]["dbe_source"] == "inferred_from_electron_pushes"


def test_predict_mechanistic_step_infers_dbe_when_missing() -> None:
    reaction = "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3]"

    payload = json.loads(predict_mechanistic_step(**_base_args(reaction_smirks=reaction)))

    assert payload["status"] == "accepted"
    assert payload["bond_electron_validation"]["valid"] is True
    assert payload["bond_electron_validation"]["dbe_source"] == "inferred_from_electron_pushes"
    assert payload["bond_electron_deltas"]


def test_predict_mechanistic_step_prefers_inferred_dbe_over_incomplete_mech_block() -> None:
    reaction = (
        "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] "
        "|mech:v1;lp:4>2|"
    )

    payload = json.loads(predict_mechanistic_step(**_base_args(reaction_smirks=reaction)))

    assert payload["status"] == "accepted"
    assert payload["bond_electron_validation"]["valid"] is True
    assert payload["bond_electron_validation"]["dbe_source"] == "inferred_from_electron_pushes"
    assert payload["bond_electron_validation"]["mech"] == "lp:4>2"
