"""Metal roles are inferred when clear and asked about only when unclear; metal_step_kind is
requested only for states that hold a metal (PRD_catalysis_handling, decisions of 2026-11-10)."""
import json

from mechanistic_agent.core.catalysis import (
    catalyst_regeneration,
    identify_catalysts,
    is_catalytic_run,
    unclear_metal_species,
)
from mechanistic_agent.core.coordinator import RunCoordinator
from mechanistic_agent.core.db import RunStore
from mechanistic_agent.core.types import CATALYTIC_MAX_STEPS, DEFAULT_MAX_STEPS
from mechanistic_agent.tool_schemas import (
    MECHANISM_STEP_PROPOSAL_TOOL,
    build_mechanism_step_proposal_tool,
    build_unified_preanalysis_tool,
)
from mechanistic_agent.tools import _d_block_metals_in
from mechanistic_agent.unified_preanalysis import build_user_prompt

PD0 = "[Pd]([P](C)(C)C)[P](C)(C)C"
CUI = "I[Cu]"
ZNCL2 = "Cl[Zn]Cl"
DMAP = "CN(C)c1ccncc1"
ALKYNE = "C#Cc1ccccc1"
ARI = "Ic1ccccc1"


def _candidate_props(tool):
    return tool["function"]["parameters"]["properties"]["candidates"]["items"]["properties"]


# ---- role inference --------------------------------------------------------


def test_role_is_inferred_for_metal_class_and_declared_species():
    roles = {c["smiles"]: (c["role"], c["role_source"]) for c in identify_catalysts([PD0, CUI, ARI])}
    assert roles[PD0] == ("catalyst", "metal_class")
    assert roles[CUI] == ("unknown", None)
    # Written over the arrow: CuI is a catalyst without asking.
    declared = {c["smiles"]: (c["role"], c["role_source"]) for c in identify_catalysts([CUI, ARI], [CUI])}
    assert declared[CUI] == ("catalyst", "declared")


def test_only_unclear_metals_are_asked_about():
    assert unclear_metal_species([PD0, CUI, ZNCL2, ARI]) == [CUI, ZNCL2]
    assert unclear_metal_species([PD0, CUI, ARI], [CUI]) == []
    assert unclear_metal_species([PD0, ARI]) == []
    assert unclear_metal_species(["CCO", "CC(=O)O"]) == []


def test_declared_metal_makes_the_run_catalytic():
    assert not is_catalytic_run([CUI, ALKYNE, ARI])
    assert is_catalytic_run([CUI, ALKYNE, ARI], [CUI])
    # A declared organocatalyst has no metal: no 22-step cap from it.
    assert not is_catalytic_run([DMAP, "CC(=O)OC(C)=O", "CCO"], [DMAP])


def test_declared_organocatalyst_must_be_regenerated():
    starting = [DMAP, "CC(=O)OC(C)=O", "CCO"]
    ok = catalyst_regeneration(starting, [["CCOC(C)=O", "CC(=O)O", DMAP]], [DMAP])
    assert ok["applies"] and not ok["unregenerated"]
    lost = catalyst_regeneration(starting, [["CCOC(C)=O", "CC(=O)[n+]1ccc(N(C)C)cc1"]], [DMAP])
    assert lost["unregenerated"] == [DMAP]


def test_step_cap_uses_declared_catalysts():
    resolve = RunCoordinator._resolve_max_steps
    assert resolve({}, [CUI, ALKYNE, ARI]) == (DEFAULT_MAX_STEPS, "default")
    assert resolve({}, [CUI, ALKYNE, ARI], [CUI]) == (CATALYTIC_MAX_STEPS, "catalytic")


# ---- asking only when unclear -----------------------------------------------


def test_preanalysis_asks_about_metal_roles_only_when_unclear():
    plain = build_unified_preanalysis_tool()
    asking = build_unified_preanalysis_tool(ask_metal_roles=True)
    assert "metal_roles" not in plain["function"]["parameters"]["properties"]
    assert "metal_roles" in asking["function"]["parameters"]["properties"]
    kwargs = dict(starting_materials=[CUI, ALKYNE, ARI], products=["C(#Cc1ccccc1)c1ccccc1"],
                  products_hidden=False, context={})
    assert "metal_roles" not in build_user_prompt(**kwargs)
    assert "metal_roles" in build_user_prompt(**kwargs, unclear_metals=[CUI])


def test_apply_metal_roles_promotes_a_confirmed_catalyst(tmp_path):
    store = RunStore(tmp_path / "data" / "mechanistic.db")
    run_id = store.create_run(
        mode="unverified",
        input_payload={"starting_materials": [CUI, ALKYNE, ARI], "products": ["C(#Cc1ccccc1)c1ccccc1"]},
        config={"model": "gpt-5"},
        prompt_bundle_hash="p",
        skill_bundle_hash="s",
        memory_bundle_hash="m",
    )
    coordinator = RunCoordinator(store)
    state = coordinator._build_state(store.get_run_row(run_id))
    assert (state.run_config.max_steps, state.run_config.max_steps_source) == (DEFAULT_MAX_STEPS, "default")
    resolved = coordinator._apply_metal_roles(
        state, [CUI], [{"smiles": "[Cu]I", "role": "catalyst", "justification": "Sonogashira co-catalyst"}]
    )
    assert resolved == [CUI]
    assert state.run_input.catalysts == [CUI]
    assert (state.run_config.max_steps, state.run_config.max_steps_source) == (CATALYTIC_MAX_STEPS, "catalytic")
    # Persisted: a resumed run keeps the role and the cap.
    reloaded = coordinator._build_state(store.get_run_row(run_id))
    assert reloaded.run_input.catalysts == [CUI]
    assert reloaded.run_config.max_steps == CATALYTIC_MAX_STEPS
    event = [e for e in store.list_events(run_id) if e["event_type"] == "metal_roles_resolved"][-1]["payload"]
    assert event["resolved_catalysts"] == [CUI] and event["answered"][0]["role"] == "catalyst"


def test_apply_metal_roles_leaves_a_reagent_alone(tmp_path):
    store = RunStore(tmp_path / "data" / "mechanistic.db")
    run_id = store.create_run(
        mode="unverified",
        input_payload={"starting_materials": [ZNCL2, "CCO"], "products": ["CC=O"]},
        config={"model": "gpt-5", "max_steps": 9},
        prompt_bundle_hash="p",
        skill_bundle_hash="s",
        memory_bundle_hash="m",
    )
    coordinator = RunCoordinator(store)
    state = coordinator._build_state(store.get_run_row(run_id))
    assert coordinator._apply_metal_roles(state, [ZNCL2], [{"smiles": ZNCL2, "role": "lewis_acid"}]) == []
    assert state.run_input.catalysts == [] and state.run_config.max_steps == 9


# ---- metal_step_kind only with a metal ----------------------------------------


def test_metal_step_kind_is_requested_only_for_metal_states():
    assert "metal_step_kind" not in _candidate_props(MECHANISM_STEP_PROPOSAL_TOOL)
    assert build_mechanism_step_proposal_tool() is MECHANISM_STEP_PROPOSAL_TOOL
    assert "metal_step_kind" in _candidate_props(build_mechanism_step_proposal_tool(include_metal_step_kind=True))
    # The base tool is never mutated by building the metal variant.
    assert "metal_step_kind" not in json.dumps(MECHANISM_STEP_PROPOSAL_TOOL)
    assert _d_block_metals_in([PD0, ARI]) == ["Pd"]
    assert _d_block_metals_in(["CCO", "[Na+].[OH-]"]) == []
