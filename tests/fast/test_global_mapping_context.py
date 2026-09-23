"""Global atom_mapping output -> mechanism-proposal consumers."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

pytest.importorskip("rdkit")
from rdkit import Chem

from mechanistic_agent.core.coordinator import RunCoordinator
from mechanistic_agent.core.global_mapping_context import (
    MAPPED_SPECIES_CONTEXT_KEY,
    mapped_current_state_for,
    render_global_mapping,
    summarize_global_mapping,
)
from mechanistic_agent.core.subagents import IntermediateAgent
from mechanistic_agent.core.types import RunConfig, RunInput, RunState


def _sn2_mapping_output(*, confidence: Any = 0.82) -> Dict[str, Any]:
    """Shape of a real ``attempt_atom_mapping`` payload (tools.py) for CCBr + Cl- -> CCCl + Br-."""
    return {
        "stoichiometry": {
            "reactants": {"Br": 1, "C": 2, "Cl": 1},
            "products": {"Br": 1, "C": 2, "Cl": 1},
        },
        "deficit": {},
        "surplus": {},
        "mapping_model": "gpt-5",
        "tool_calling_used": True,
        "guidance": "Mappings use zero-based atom indices; treat spectator regions as conserved.",
        "functional_groups": {"starting_materials": {}, "products": {}},
        "functional_group_transformation": {"label": "alkyl_halide_exchange"},
        "llm_response": {
            "mapped_atoms": [
                {"product_atom": "CCCl#0", "source": {"molecule_index": 0, "smiles": "CCBr", "atom_index": 0}},
                {"product_atom": "CCCl#1", "source": {"molecule_index": 0, "smiles": "CCBr", "atom_index": 1}},
                {"product_atom": "CCCl#2", "source": {"molecule_index": 1, "smiles": "[Cl-]", "atom_index": 0}},
                {"product_atom": "[Br-]#0", "source": {"molecule_index": 0, "smiles": "CCBr", "atom_index": 2}},
            ],
            "unmapped_atoms": ["implicit H on C1 (spectator)"],
            "confidence": confidence,
            "reasoning": "SN2 at the primary carbon; chloride displaces bromide.",
        },
        "schema_validation": {"status": "ok", "validator": "AtomMappingPayload"},
        "bond_electron_guidance": {"format": "...", "example": "..."},
    }


def _maps_by_number(smiles_list: List[str]) -> Dict[int, str]:
    found: Dict[int, str] = {}
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, smiles
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum():
                assert atom.GetAtomMapNum() not in found
                found[atom.GetAtomMapNum()] = atom.GetSymbol()
    return found


def _strip(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


# ---------------------------------------------------------------------------
# summary (confidence path)
# ---------------------------------------------------------------------------


def test_summary_reads_confidence_from_llm_response() -> None:
    summary = summarize_global_mapping(_sn2_mapping_output(confidence=0.82))
    assert summary == {"confidence": 0.82, "unmapped_atoms": ["implicit H on C1 (spectator)"]}


def test_summary_normalises_word_confidence_and_falls_back_to_top_level() -> None:
    assert summarize_global_mapping(_sn2_mapping_output(confidence="medium"))["confidence"] == 0.6
    legacy = {"confidence": 0.4, "llm_response": {"reasoning": "x"}}
    assert summarize_global_mapping(legacy)["confidence"] == 0.4
    assert summarize_global_mapping({"llm_response": {}})["confidence"] is None


def test_summary_empty_when_module_did_not_run() -> None:
    assert summarize_global_mapping(None) == {}
    assert summarize_global_mapping({}) == {}


def test_proposal_guidance_carries_mapping_confidence_and_unmapped_atoms() -> None:
    coordinator = RunCoordinator(store=_Store())  # type: ignore[arg-type]
    state = _state()
    output = _sn2_mapping_output(confidence=0.82)
    coordinator._latest_output_by_step = (  # type: ignore[method-assign]
        lambda _run_id, step_name: output if step_name == "atom_mapping" else None
    )

    guidance = coordinator._build_proposal_constraint_guidance(state)

    assert guidance is not None
    assert guidance["atom_mapping_summary"]["confidence"] == 0.82
    assert guidance["atom_mapping_summary"]["unmapped_atoms"] == ["implicit H on C1 (spectator)"]


# ---------------------------------------------------------------------------
# rendering mapped SMILES
# ---------------------------------------------------------------------------


def test_render_sn2_mapping_round_trips() -> None:
    llm = _sn2_mapping_output()["llm_response"]
    rendered = render_global_mapping(["CCBr", "[Cl-]"], ["CCCl", "[Br-]"], llm["mapped_atoms"])

    assert rendered["pairs_used"] == 4
    assert rendered["pairs_dropped"] == 0
    start = rendered["mapped_starting_materials"]
    prods = rendered["mapped_products"]
    assert [_strip(s) for s in start] == ["CCBr", "[Cl-]"]
    assert [_strip(s) for s in prods] == ["CCCl", "[Br-]"]
    start_maps = _maps_by_number(start)
    prod_maps = _maps_by_number(prods)
    assert start_maps == prod_maps
    assert sorted(start_maps.values()) == ["Br", "C", "C", "Cl"]


def test_render_esterification_keeps_species_order_and_elements() -> None:
    starting = ["CC(=O)O", "CCO"]
    products = ["CCOC(C)=O", "O"]
    pairs = [
        ("CCOC(C)=O#0", 1, 0),
        ("CCOC(C)=O#1", 1, 1),
        ("CCOC(C)=O#2", 1, 2),
        ("CCOC(C)=O#3", 0, 1),
        ("CCOC(C)=O#4", 0, 0),
        ("CCOC(C)=O#5", 0, 2),
        ("O#0", 0, 3),
    ]
    mapped_atoms = [
        {"product_atom": ref, "source": {"molecule_index": mol, "smiles": starting[mol], "atom_index": idx}}
        for ref, mol, idx in pairs
    ]
    rendered = render_global_mapping(starting, products, mapped_atoms)

    assert rendered["pairs_used"] == 7
    assert [_strip(s) for s in rendered["mapped_starting_materials"]] == starting
    assert [_strip(s) for s in rendered["mapped_products"]] == ["CCOC(C)=O", "O"]
    assert _maps_by_number(rendered["mapped_starting_materials"]) == _maps_by_number(rendered["mapped_products"])
    # Water oxygen comes from the acid hydroxyl (acid index 3), not ethanol.
    water_map = next(
        a.GetAtomMapNum() for a in Chem.MolFromSmiles(rendered["mapped_products"][1]).GetAtoms()
    )
    acid = Chem.MolFromSmiles(rendered["mapped_starting_materials"][0])
    hydroxyl = [a for a in acid.GetAtoms() if a.GetAtomMapNum() == water_map]
    assert hydroxyl and hydroxyl[0].GetSymbol() == "O" and hydroxyl[0].GetDegree() == 1


def test_render_handles_triple_bond_in_product_reference() -> None:
    mapped_atoms = [
        {"product_atom": "CC#N#0", "source": {"molecule_index": 0, "smiles": "CBr", "atom_index": 0}},
        {"product_atom": "CC#N#1", "source": {"molecule_index": 1, "smiles": "[C-]#N", "atom_index": 0}},
        {"product_atom": "CC#N#2", "source": {"molecule_index": 1, "smiles": "[C-]#N", "atom_index": 1}},
    ]
    rendered = render_global_mapping(["CBr", "[C-]#N"], ["CC#N", "[Br-]"], mapped_atoms)
    assert rendered["pairs_used"] == 3
    assert _maps_by_number(rendered["mapped_products"]) == {1: "C", 2: "C", 3: "N"}


def test_render_drops_inconsistent_pairs() -> None:
    mapped_atoms = [
        {"product_atom": "CCCl#0", "source": {"molecule_index": 0, "atom_index": 0}},  # ok
        {"product_atom": "CCCl#2", "source": {"molecule_index": 0, "atom_index": 2}},  # Cl <- Br mismatch
        {"product_atom": "CCCl#9", "source": {"molecule_index": 0, "atom_index": 1}},  # out of range
        {"product_atom": "CCO#0", "source": {"molecule_index": 0, "atom_index": 1}},  # unknown product
        {"product_atom": "CCCl#1", "source": {"molecule_index": 0, "atom_index": 0}},  # reused source atom
        {"product_atom": "CCCl#1", "source": {"molecule_index": 5, "smiles": "CCO", "atom_index": 1}},  # unknown source
        {"product_atom": "CCCl", "source": {"molecule_index": 0, "atom_index": 1}},  # no index
        "not-a-dict",
    ]
    rendered = render_global_mapping(["CCBr", "[Cl-]"], ["CCCl", "[Br-]"], mapped_atoms)
    assert rendered["pairs_used"] == 1
    assert rendered["pairs_dropped"] == 7
    assert _maps_by_number(rendered["mapped_products"]) == {1: "C"}


def test_render_empty_when_nothing_usable() -> None:
    for mapped_atoms in (None, [], [{"product_atom": "CCCl#2", "source": {"molecule_index": 0, "atom_index": 2}}]):
        rendered = render_global_mapping(["CCBr", "[Cl-]"], ["CCCl", "[Br-]"], mapped_atoms)
        assert rendered["mapped_starting_materials"] == []
        assert rendered["mapped_products"] == []
        assert rendered["pairs_used"] == 0


def test_mapped_current_state_only_before_first_step() -> None:
    mapped = ["[CH3:1][CH2:2][Br:3]", "[Cl-:4]"]
    assert mapped_current_state_for(["[Cl-]", "CCBr"], ["CCBr", "[Cl-]"], mapped) == mapped
    assert mapped_current_state_for(["CCCl", "[Br-]"], ["CCBr", "[Cl-]"], mapped) == []
    assert mapped_current_state_for(["CCBr", "[Cl-]"], ["CCBr", "[Cl-]"], []) == []


# ---------------------------------------------------------------------------
# wiring: coordinator -> IntermediateAgent -> run_intermediates
# ---------------------------------------------------------------------------


def test_coordinator_builds_mapped_species_context_from_global_mapping() -> None:
    coordinator = RunCoordinator(store=_Store())  # type: ignore[arg-type]
    state = _state()
    output = _sn2_mapping_output()
    coordinator._latest_output_by_step = (  # type: ignore[method-assign]
        lambda _run_id, step_name: output if step_name == "atom_mapping" else None
    )

    context = coordinator._build_global_mapping_species_context(state)

    assert context is not None
    assert [_strip(s) for s in context["mapped_starting_materials"]] == ["CCBr", "[Cl-]"]
    assert [_strip(s) for s in context["mapped_products"]] == ["CCCl", "[Br-]"]
    assert context["mapped_current_state"] == context["mapped_starting_materials"]

    state.current_state = ["CCCl", "[Br-]"]
    later = coordinator._build_global_mapping_species_context(state)
    assert later is not None and later["mapped_current_state"] == []


def test_coordinator_mapped_species_context_absent_when_mapping_disabled() -> None:
    coordinator = RunCoordinator(store=_Store())  # type: ignore[arg-type]
    coordinator._latest_output_by_step = lambda _run_id, _step: None  # type: ignore[method-assign]
    assert coordinator._build_global_mapping_species_context(_state()) is None


def test_intermediate_agent_passes_mapped_context_and_strips_it_from_guidance() -> None:
    captured: Dict[str, Any] = {}

    class _Executor:
        def run_intermediates(self, **kwargs: Any) -> Dict[str, Any]:
            captured.update(kwargs)
            return {"candidates": []}

    agent = IntermediateAgent(_Executor())  # type: ignore[arg-type]
    guidance = {
        "guidance_strength": "weak",
        MAPPED_SPECIES_CONTEXT_KEY: {
            "mapped_starting_materials": ["[CH3:1][CH2:2][Br:3]", "[Cl-:4]"],
            "mapped_products": ["[CH3:1][CH2:2][Cl:4]", "[Br-:3]"],
            "mapped_current_state": ["[CH3:1][CH2:2][Br:3]", "[Cl-:4]"],
        },
    }

    agent.run(_state(), template_guidance=guidance)

    assert captured["mapped_starting_materials"] == ["[CH3:1][CH2:2][Br:3]", "[Cl-:4]"]
    assert captured["mapped_products"] == ["[CH3:1][CH2:2][Cl:4]", "[Br-:3]"]
    assert captured["mapped_current_state"] == ["[CH3:1][CH2:2][Br:3]", "[Cl-:4]"]
    assert MAPPED_SPECIES_CONTEXT_KEY not in captured["template_guidance"]
    assert captured["template_guidance"]["guidance_strength"] == "weak"
    assert MAPPED_SPECIES_CONTEXT_KEY in guidance  # caller's dict not mutated


def test_intermediate_agent_defaults_to_empty_mapped_args() -> None:
    captured: Dict[str, Any] = {}

    class _Executor:
        def run_intermediates(self, **kwargs: Any) -> Dict[str, Any]:
            captured.update(kwargs)
            return {}

    IntermediateAgent(_Executor()).run(_state())  # type: ignore[arg-type]
    assert captured["mapped_starting_materials"] == []
    assert captured["mapped_products"] == []
    assert captured["mapped_current_state"] == []


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _Store:
    def append_event(self, *args: Any, **kwargs: Any) -> None:
        return

    def list_step_outputs(self, run_id: str) -> List[Dict[str, Any]]:
        return []


def _state() -> RunState:
    run_input = RunInput(
        starting_materials=["CCBr", "[Cl-]"],
        products=["CCCl", "[Br-]"],
        ph=7.0,
        temperature_celsius=25.0,
    )
    run_config = RunConfig(model="gpt-4", model_family="openai", intermediate_prediction_enabled=True)
    state = RunState(run_id="run-test-id", mode="unverified", run_input=run_input, run_config=run_config)
    state.initialise()
    return state
