"""Scoring v3: proton-transfer / shuttle agnostic alignment against FlowER paths."""
from __future__ import annotations

import pytest

from mechanistic_agent.scoring import (
    DEFAULT_SCORING_VERSION,
    score_snapshot_against_known,
)
from mechanistic_agent.skeleton_alignment import (
    collapse_skeleton_path,
    proton_agnostic_alignment,
    skeleton_species_keys,
    skeleton_state_key,
)

pytest.importorskip("rdkit")

_OK = {
    "checks": [
        {"name": "dbe_metadata", "passed": True},
        {"name": "atom_balance", "passed": True},
        {"name": "state_progress", "passed": True},
    ]
}

# flower_038130: methyl imidazolylacetate + water -> acid + MeOH (unmapped).
SM_038130 = ["O", "COC(=O)Cn1ccnc1"]
REF_038130 = [
    ["COC([O-])(Cn1ccnc1)[OH2+]"],
    ["COC([O-])(O)Cn1cc[nH+]c1"],
    ["COC(O)(O)Cn1ccnc1"],
    ["OC(=[OH+])Cn1ccnc1", "C[O-]"],
    ["O=C(O)Cn1ccnc1", "CO"],
]
HYDRONIUM_038130 = [
    ["O", "COC(=[OH+])Cn1ccnc1"],
    ["COC(O)([OH2+])Cn1ccnc1"],
    ["COC(O)(O)Cn1ccnc1", "[OH3+]"],
    ["C[OH+]C(O)(O)Cn1ccnc1", "O"],
    ["OC(=[OH+])Cn1ccnc1", "CO", "O"],
    ["O=C(O)Cn1ccnc1", "CO", "[OH3+]"],
]
INTRAMOLECULAR_038130 = [
    ["COC([O-])([OH2+])Cn1ccnc1"],
    ["C[OH+]C([O-])(O)Cn1ccnc1"],
    ["O=C(O)Cn1ccnc1", "CO"],
]
# A genuinely different heavy-atom path: transesterification-like detour through
# an anhydride-ish dimer, never passing the tetrahedral intermediate.
DIFFERENT_038130 = [
    ["COC(=O)Cn1ccnc1", "O"],
    ["COC(=O)C[n+]1ccn(C(=O)OC)c1", "O"],
    ["O=C(O)Cn1ccnc1", "CO"],
]


# ---------------------------------------------------------------------------
# Skeleton keys
# ---------------------------------------------------------------------------


def test_skeleton_key_ignores_protonation_and_charge() -> None:
    assert skeleton_species_keys("COC(=[OH+])Cn1ccnc1") == skeleton_species_keys("COC(=O)Cn1ccnc1")
    assert skeleton_species_keys("C[O-]") == skeleton_species_keys("CO")
    assert skeleton_species_keys("OC(=[OH+])Cn1ccnc1") == skeleton_species_keys("O=C(O)Cn1ccnc1")
    assert skeleton_species_keys("CC(=O)[O-]") == skeleton_species_keys("CC(=O)O")
    # zwitterion vs neutral tetrahedral intermediate vs imidazolium tautomer
    ti = skeleton_species_keys("COC(O)(O)Cn1ccnc1")
    assert skeleton_species_keys("COC([O-])(Cn1ccnc1)[OH2+]") == ti
    assert skeleton_species_keys("COC([O-])(O)Cn1cc[nH+]c1") == ti
    assert skeleton_species_keys("C[OH+]C(O)(O)Cn1ccnc1") == ti


def test_skeleton_key_handles_atom_maps_explicit_h_and_kekule() -> None:
    mapped = (
        "[O:1]([C:2]([O-:3])([C:4]([N:5]1[C:6]([H:14])=[C:7]([H:15])[N+:8]([H:21])=[C:9]1[H:16])"
        "([H:12])[H:13])[O:11][H:20])[C:10]([H:17])([H:18])[H:19]"
    )
    assert skeleton_species_keys(mapped) == skeleton_species_keys("COC(O)(O)Cn1ccnc1")


def test_skeleton_key_distinguishes_hydride_transfer_and_connectivity() -> None:
    # ketone vs alkoxide differ by a hydride, not a proton
    assert skeleton_species_keys("CC(C)=O") != skeleton_species_keys("CC(C)[O-]")
    # ester vs tetrahedral intermediate: new C-O bond
    assert skeleton_species_keys("COC(=O)C") != skeleton_species_keys("COC(O)(O)C")


def test_small_species_and_shuttle_equivalents_drop_out_of_state_key() -> None:
    for small in ("O", "[OH3+]", "[OH-]", "[H+]", "[Cl-]", "Cl", "[Na+]"):
        assert skeleton_species_keys(small) == []
    # TFA vs trifluoroacetate: same key, set semantics
    assert skeleton_state_key(["CC(=O)O", "OC(=O)C(F)(F)F"]) == skeleton_state_key(
        ["CC(=O)O", "[O-]C(=O)C(F)(F)F", "OC(=O)C(F)(F)F", "O"]
    )


def test_collapse_removes_proton_transfer_only_steps() -> None:
    collapsed = collapse_skeleton_path([SM_038130] + REF_038130)
    assert len(collapsed) == 3  # ester -> tetrahedral intermediate -> acid + MeOH
    collapsed_h = collapse_skeleton_path([SM_038130] + HYDRONIUM_038130)
    assert collapsed_h == collapsed


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("predicted", [HYDRONIUM_038130, INTRAMOLECULAR_038130, REF_038130])
def test_proton_variants_of_flower_038130_align_fully(predicted) -> None:
    result = proton_agnostic_alignment([SM_038130] + predicted, [SM_038130] + REF_038130)
    assert result["score"] == pytest.approx(1.0)
    assert result["reference_skeleton_steps"] == 2
    assert result["predicted_skeleton_steps"] == 2


def test_different_heavy_atom_path_does_not_align_fully() -> None:
    result = proton_agnostic_alignment([SM_038130] + DIFFERENT_038130, [SM_038130] + REF_038130)
    assert result["score"] < 1.0


def test_persistent_catalyst_is_a_spectator() -> None:
    tfa = "OC(=O)C(F)(F)F"
    tfa_anion = "[O-]C(=O)C(F)(F)F"
    predicted = [SM_038130 + [tfa]] + [
        state + ([tfa_anion] if i % 2 else [tfa]) for i, state in enumerate(HYDRONIUM_038130)
    ]
    result = proton_agnostic_alignment(predicted, [SM_038130] + REF_038130)
    assert result["score"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# score_snapshot_against_known integration
# ---------------------------------------------------------------------------


def _expected_038130() -> dict:
    steps = []
    current = list(SM_038130)
    for idx, resulting in enumerate(REF_038130, start=1):
        steps.append({"step_index": idx, "current_state": current, "resulting_state": resulting})
        current = resulting
    return {
        "products": ["O=C(O)Cn1ccnc1", "CO"],
        "verified_mechanism": {"steps": steps},
    }


def _snapshot(path: list) -> dict:
    events = []
    current = list(SM_038130)
    for idx, resulting in enumerate(path, start=1):
        events.append(
            {
                "seq": idx,
                "event_type": "mechanism_step_accepted",
                "payload": {
                    "step_index": idx,
                    "current_state": current,
                    "resulting_state": resulting,
                    "validation_summary": _OK,
                },
            }
        )
        current = resulting
    return {
        "input": {"starting_materials": list(SM_038130)},
        "events": events,
        "step_outputs": [],
        "overall_balance": {"grade": "exact", "balanced": True},
    }


def test_default_scoring_version_is_v3() -> None:
    assert DEFAULT_SCORING_VERSION == "v3"


def test_hydronium_path_gets_full_alignment_and_no_step_count_penalty() -> None:
    expected = _expected_038130()
    v2 = score_snapshot_against_known(_snapshot(HYDRONIUM_038130), expected, scoring_version="v2")
    v3 = score_snapshot_against_known(_snapshot(HYDRONIUM_038130), expected, scoring_version="v3")
    assert v3["scoring_version"] == "v3"
    assert v3["proton_agnostic_alignment_component"] == pytest.approx(1.0)
    assert v3["exact_alignment_component"] == pytest.approx(v2["known_alignment_component"])
    assert v3["exact_alignment_component"] < 1.0
    assert v3["alignment_basis"] == "proton_agnostic"
    assert v3["known_alignment_component"] == pytest.approx(1.0)
    assert not any(p["type"] in {"extra_steps", "unexpected_final_species"} for p in v3["penalties"])
    assert v3["tolerated_final_species"] == ["[OH3+]"]
    assert v3["score"] > v2["score"]
    assert v3["passed"] is True
    labels = [step["skeleton_label"] for step in v3["step_breakdown"]]
    assert labels.count("proton_shuttle_step") == 4
    assert labels.count("skeleton_match") == 2


def test_intramolecular_path_matches_v3_full_alignment() -> None:
    v3 = score_snapshot_against_known(_snapshot(INTRAMOLECULAR_038130), _expected_038130())
    assert v3["known_alignment_component"] == pytest.approx(1.0)
    assert v3["proton_agnostic_alignment_component"] == pytest.approx(1.0)


def test_exact_replay_scores_as_before() -> None:
    expected = _expected_038130()
    v2 = score_snapshot_against_known(_snapshot(REF_038130), expected, scoring_version="v2")
    v3 = score_snapshot_against_known(_snapshot(REF_038130), expected, scoring_version="v3")
    assert v3["alignment_basis"] == "exact"
    assert v3["score"] == pytest.approx(v2["score"])
    assert v3["known_alignment_component"] == pytest.approx(v2["known_alignment_component"])
    assert v3["penalties"] == v2["penalties"]


def test_v2_output_is_unchanged_by_v3_fields() -> None:
    v2 = score_snapshot_against_known(_snapshot(HYDRONIUM_038130), _expected_038130(), scoring_version="v2")
    assert "alignment_basis" not in v2
    assert any(p["type"] == "extra_steps" for p in v2["penalties"])
    assert v2["unexpected_final_species"] == ["[OH3+]"]


def test_v3_still_penalizes_unexpected_heavy_leftovers() -> None:
    path = [list(state) for state in HYDRONIUM_038130]
    path[-1] = path[-1] + ["COC(O)(O)Cn1ccnc1"]  # leftover tetrahedral intermediate
    v3 = score_snapshot_against_known(_snapshot(path), _expected_038130())
    assert v3["unexpected_final_species"] == ["COC(O)(O)Cn1ccnc1"]
    assert any(p["type"] == "unexpected_final_species" for p in v3["penalties"])


def test_without_reference_states_v3_falls_back_to_exact() -> None:
    expected = {
        "known_mechanism": {
            "min_steps": 2,
            "steps": [{"step_index": 1, "target_smiles": "INT1"}, {"step_index": 2, "target_smiles": "P"}],
        }
    }
    snapshot = {
        "events": [
            {
                "seq": i,
                "event_type": "mechanism_step_accepted",
                "payload": {"step_index": i, "resulting_state": [t], "validation_summary": _OK},
            }
            for i, t in ((1, "INT1"), (2, "P"))
        ],
        "step_outputs": [],
    }
    v2 = score_snapshot_against_known(snapshot, expected, scoring_version="v2")
    v3 = score_snapshot_against_known(snapshot, expected, scoring_version="v3")
    assert v3["proton_agnostic_alignment_component"] is None
    assert v3["alignment_basis"] == "exact"
    assert v3["score"] == pytest.approx(v2["score"])
