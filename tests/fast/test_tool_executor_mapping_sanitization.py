from __future__ import annotations

import pytest

pytest.importorskip("rdkit")
import mechanistic_agent.core.tool_executor as tool_executor_module

from mechanistic_agent.core.tool_executor import ToolExecutor


def test_run_conditions_strips_atom_maps(monkeypatch) -> None:
    captured = {}

    def fake_conditions(starting, products, ph):
        captured["starting"] = starting
        captured["products"] = products
        captured["ph"] = ph
        return '{"ok": true}'

    monkeypatch.setattr(tool_executor_module, "assess_initial_conditions", fake_conditions)
    executor = ToolExecutor()

    executor.run_conditions(["[CH3:1][OH:2]"], ["[CH2:1]=[O:2]"], 7.0)

    assert captured["starting"] == ["CO"]
    assert captured["products"] == ["C=O"]
    assert captured["ph"] == 7.0


def test_run_intermediates_strips_mapping_from_llm_inputs(monkeypatch) -> None:
    captured = {}

    def fake_intermediates(**kwargs):
        captured.update(kwargs)
        return '{"ok": true}'

    monkeypatch.setattr(tool_executor_module, "propose_intermediates", fake_intermediates)
    executor = ToolExecutor()

    executor.run_intermediates(
        starting=["[CH3:1][OH:2]"],
        products=["[CH2:1]=[O:2]"],
        current_state=["[CH3:1][OH:2]"],
        previous_intermediates=["[CH3:1][OH:2]"],
        ph=7.0,
        temperature=25.0,
        step_index=1,
    )

    assert captured["starting_materials"] == ["CO"]
    assert captured["products"] == ["C=O"]
    assert captured["current_state"] == ["CO"]
    assert captured["previous_intermediates"] == ["CO"]
    assert captured["mapped_starting_materials"] == ["[CH3:1][OH:2]"]
    assert captured["mapped_products"] == ["[CH2:1]=[O:2]"]
    assert captured["mapped_current_state"] == ["[CH3:1][OH:2]"]


def test_run_candidate_rescue_strips_mapping(monkeypatch) -> None:
    captured = {}

    def fake_rescue(**kwargs):
        captured.update(kwargs)
        return '{"ok": true}'

    monkeypatch.setattr(tool_executor_module, "predict_missing_reagents_for_candidate", fake_rescue)
    executor = ToolExecutor()

    executor.run_candidate_rescue(
        current_state=["[CH3:1][OH:2]"],
        resulting_state=["[CH2:1]=[O:2]"],
        failed_checks=["balance"],
    )

    assert captured["current_state"] == ["CO"]
    assert captured["resulting_state"] == ["C=O"]


def test_run_mechanism_step_preserves_reaction_smirks(monkeypatch) -> None:
    captured = {}

    def fake_mechanism_step(**kwargs):
        captured.update(kwargs)
        return '{"ok": true}'

    monkeypatch.setattr(tool_executor_module, "predict_mechanistic_step", fake_mechanism_step)
    executor = ToolExecutor()
    mapped_smirks = "[CH3:1][OH:2]>>[CH2:1]=[O:2] |mech:v1;lp:2>1|"

    executor.run_mechanism_step(
        step_index=1,
        current_state=["[CH3:1][OH:2]"],
        target_products=["[CH2:1]=[O:2]"],
        predicted_intermediate="[CH2:1]=[O:2]",
        resulting_state=["[CH2:1]=[O:2]"],
        electron_pushes=[{"kind": "lone_pair", "source_atom": "2", "target_atom": "1", "electrons": 2}],
        reaction_smirks=mapped_smirks,
        previous_intermediates=[],
        starting_materials=["[CH3:1][OH:2]"],
        note="fixture",
    )

    assert captured["reaction_smirks"] == mapped_smirks
