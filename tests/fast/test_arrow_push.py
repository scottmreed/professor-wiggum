from __future__ import annotations

import re

from mechanistic_agent.core.arrow_push import predict_arrow_push_annotation


def test_mapped_sn2_prioritizes_bond_change_and_push_alignment() -> None:
    prediction = predict_arrow_push_annotation(
        current_state=["[CH3:1][CH2:2][Br:3]", "[Cl-:4]"],
        resulting_state=["[CH3:1][CH2:2][Cl:4]", "[Br-:3]"],
        reaction_smirks="[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3]",
        raw_reaction_smirks="[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] |mech:v1;lp:4>2;sigma:2-3>3|",
        electron_pushes=[{"kind": "lone_pair", "source_atom": "4", "target_atom": "2", "electrons": 2}],
        step_index=1,
        candidate_rank=1,
    )

    assert prediction["used_bond_change_priority"] is True
    selected = prediction["selected_candidate"]
    assert selected["source_atom_ref"] == "4"
    assert selected["sink_atom_ref"] == "2"
    assert selected["template"] == "sn2_substitution"
    assert re.match(r"^\|aps:v1;src=[^;]+;snk=[^;]+;e=[12];tpl=[a-z0-9_]+;sc=\d\.\d{3}\|$", prediction["annotation_suffix"])


def test_missing_mapping_uses_functional_group_fallback_refs() -> None:
    prediction = predict_arrow_push_annotation(
        current_state=["CCO"],
        resulting_state=["CC=O"],
        reaction_smirks="CCO>>CC=O",
        raw_reaction_smirks="CCO>>CC=O",
        electron_pushes=[],
        step_index=2,
    )

    assert prediction["used_functional_group_fallback"] is True
    selected = prediction["selected_candidate"]
    assert selected["source_atom_ref"].startswith("s")
    assert selected["sink_atom_ref"].startswith("s")


def test_positive_charge_source_is_penalized() -> None:
    prediction = predict_arrow_push_annotation(
        current_state=["[CH3+:1]", "[Cl-:2]"],
        resulting_state=["[CH3:1][Cl:2]"],
        reaction_smirks="[CH3+:1].[Cl-:2]>>[CH3:1][Cl:2]",
        raw_reaction_smirks="[CH3+:1].[Cl-:2]>>[CH3:1][Cl:2]",
        electron_pushes=[{"kind": "lone_pair", "source_atom": "2", "target_atom": "1", "electrons": 2}],
        step_index=1,
    )

    penalties = [
        penalty
        for candidate in prediction["ranked_candidates"]
        for penalty in (candidate.get("sanity") or {}).get("penalties", [])
    ]
    assert "positive_charge_source" in penalties


def test_annotation_suffix_uses_aps_v1_nomenclature() -> None:
    prediction = predict_arrow_push_annotation(
        current_state=["C=O", "O"],
        resulting_state=["CO"],
        reaction_smirks="C=O.O>>CO",
        raw_reaction_smirks="C=O.O>>CO",
        electron_pushes=[],
        step_index=3,
    )

    assert prediction["annotation_suffix"].startswith("|aps:v1;")
    assert "src=" in prediction["annotation_suffix"]
    assert "snk=" in prediction["annotation_suffix"]
