"""Persistent-species rules for the proposal constraint registry.

A species may only be carried forward as *persistent* (catalyst / spectator /
counterion) when it appears unchanged on BOTH sides of the overall reaction.
Stoichiometric reagents that the conditions step happens to list as an acid or
base (SOCl2, HCl, Et3N, ...) are consumed and must not be re-appended to later
resulting states — doing so double-counts atoms and rejects correct steps.
"""
from __future__ import annotations

from mechanistic_agent.tools import _build_species_registry_and_constraints


def _roles(payload, species):
    for entry in payload["species_registry"]:
        if entry["species"] == species:
            return set(entry["roles"]), set(entry["tags"])
    raise AssertionError(f"{species} not in registry")


def test_stoichiometric_reagent_listed_as_acid_is_not_persistent() -> None:
    payload = _build_species_registry_and_constraints(
        starting_materials=["O=C(O)C=Cc1cncc(Br)c1", "O=S(Cl)Cl"],
        products=["O=C(Cl)C=Cc1cncc(Br)c1", "Cl", "O=S=O"],
        conditions_context={"environment": "acidic", "acid_candidates": ["Cl", "O=S(Cl)Cl"]},
        missing_reactants=[],
        missing_products=[],
    )
    constraints = payload["proposal_constraints"]
    assert "O=S(Cl)Cl" not in constraints["persistent_species"]
    assert "Cl" not in constraints["persistent_species"]
    roles, tags = _roles(payload, "O=S(Cl)Cl")
    assert "acid" in roles
    assert "catalyst" not in roles
    assert "consumed_on_use" in tags and "persistent" not in tags
    # Still eligible so the model may use it.
    assert "O=S(Cl)Cl" in constraints["eligible_reactants"]


def test_condition_additive_base_is_eligible_but_not_persistent() -> None:
    payload = _build_species_registry_and_constraints(
        starting_materials=["CC(C)(C)OC(=O)OC(=O)OC(C)(C)C", "CCOC(=O)C1CCCNC1"],
        products=["CCOC(=O)C1CCCN(C(=O)OC(C)(C)C)C1", "CC(C)(C)O", "O=C=O"],
        conditions_context={"environment": "basic", "base_candidates": ["CCN(CC)CC"]},
        missing_reactants=[],
        missing_products=[],
    )
    constraints = payload["proposal_constraints"]
    assert "CCN(CC)CC" in constraints["eligible_reactants"]
    assert "CCN(CC)CC" not in constraints["persistent_species"]
    roles, tags = _roles(payload, "CCN(CC)CC")
    assert "base" in roles and "catalyst" not in roles
    assert "condition_additive" in tags


def test_true_spectator_present_on_both_sides_is_persistent() -> None:
    payload = _build_species_registry_and_constraints(
        starting_materials=["CC(C)(C)OC(=O)OC(=O)OC(C)(C)C", "C1CCOC1", "CCOC(=O)C1CCCNC1"],
        products=["CCOC(=O)C1CCCN(C(=O)OC(C)(C)C)C1", "C1CCOC1", "CC(C)(C)O", "O=C=O"],
        conditions_context={"environment": "basic"},
        missing_reactants=[],
        missing_products=[],
    )
    constraints = payload["proposal_constraints"]
    assert "C1CCOC1" in constraints["persistent_species"]
    assert "C1CCOC1" in constraints["spectator_species"]


def test_catalytic_acid_present_on_both_sides_stays_persistent() -> None:
    payload = _build_species_registry_and_constraints(
        starting_materials=["CC(=O)O", "CCO", "[OH3+]"],
        products=["CC(=O)OCC", "O", "[OH3+]"],
        conditions_context={"environment": "acidic", "acid_candidates": ["[OH3+]"]},
        missing_reactants=[],
        missing_products=[],
    )
    constraints = payload["proposal_constraints"]
    assert "[OH3+]" in constraints["persistent_species"]
    roles, _tags = _roles(payload, "[OH3+]")
    assert "catalyst" in roles


def test_consumed_counterion_is_not_persistent_but_unchanged_one_is() -> None:
    payload = _build_species_registry_and_constraints(
        starting_materials=["CCBr", "[Cl-]", "[Na+]"],
        products=["CCCl", "[Br-]", "[Na+]"],
        conditions_context={"environment": "neutral"},
        missing_reactants=[],
        missing_products=[],
    )
    constraints = payload["proposal_constraints"]
    assert "[Na+]" in constraints["persistent_species"]
    assert "[Cl-]" not in constraints["persistent_species"]
