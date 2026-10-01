"""Parity and behaviour tests for ``mechanistic_agent.reaction_signatures``.

The 22 parity vectors were generated from ChemIllusion's
``backend/app/services/mechanism_predictor/submission_normalizer.py`` at
``6c45cc8b`` with RDKit 2023.09.1. Every expected field must match exactly:
ChemIllusion deletes its own copy only once these hashes are identical.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

pytest.importorskip("rdkit")

from mechanistic_agent import reaction_signatures as rs  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "reaction_signature_vectors.json"
HASH_FIELDS = (
    "exact_reaction_hash",
    "stoich_hash",
    "endpoint_hash",
    "core_hash",
    "major_key_hash",
    "inchikey_reaction_hash",
    "conditions_hash",
    "similarity_cluster_id",
    "executable",
)


def _vectors() -> List[Dict[str, Any]]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


def _roles_from_payload(payload: Dict[str, Any]) -> Dict[str, List[Tuple[str, Optional[float]]]]:
    """Mirror ChemIllusion's ``_participant_lists`` (and the schema validator
    that splits ``raw_input.reaction_smiles`` when no participants are given)."""
    participants = payload.get("participants") or {}
    roles = {
        name: [
            (str(p["input"]).strip(), float(p["coefficient"]) if p.get("coefficient") is not None else None)
            for p in participants.get(name) or []
        ]
        for name, _ in rs.ROLE_ORDER
    }
    reaction_smiles = ((payload.get("raw_input") or {}).get("reaction_smiles") or "").strip()
    if reaction_smiles and not any(roles.values()):
        for name, values in rs.parse_reaction_smiles(reaction_smiles).items():
            roles[name] = [(value, None) for value in values]
    return roles


def test_fixture_metadata() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["schema"] == rs.RECIPE_VERSION == "mechanism_submission.v2"
    assert len(data["cases"]) == 22


# Hashes keyed on InChIKeys or conditions alone, which survive RDKit SMILES-writer changes.
WRITER_INDEPENDENT_FIELDS = ("inchikey_reaction_hash", "conditions_hash", "executable")


def _writer_divergence(participants: List[Dict[str, Any]]) -> List[str]:
    """Expected canonical SMILES that the running RDKit itself writes differently.

    The SMILES-derived hashes are only reproducible where RDKit's writer still
    emits the vector's canonical string. RDKit 2025.03 brackets atoms bonded to
    a transition metal (``O=[Cr](=O)=O`` -> ``[O]=[Cr](=[O])=[O]``), which
    2023.09 and 2024.03 do not. Round-tripping the expected string through bare
    RDKit isolates that from a regression in ``normalize_species``.
    """
    from rdkit import Chem, rdBase

    changed = []
    with rdBase.BlockLogs():
        for p in participants:
            if not p["valid"]:
                continue
            mol = Chem.MolFromSmiles(p["canonical_smiles"])
            if mol is None or Chem.MolToSmiles(mol, isomericSmiles=True) != p["canonical_smiles"]:
                changed.append(p["canonical_smiles"])
    return changed


@pytest.mark.parametrize("case", _vectors(), ids=lambda c: json.dumps(c["payload"], sort_keys=True)[:80])
def test_v2_hash_parity(case: Dict[str, Any]) -> None:
    expected = case["expected"]
    result = rs.reaction_hashes(_roles_from_payload(case["payload"]), case["payload"].get("conditions") or {})
    divergent = _writer_divergence(expected["participants"])
    if divergent:
        for name in WRITER_INDEPENDENT_FIELDS:
            assert getattr(result, name) == expected[name], name
        observed_keys = [(p.role, p.index, p.species.inchikey, p.species.valid) for p in result.participants]
        assert observed_keys == [(p["role"], p["index"], p["inchikey"], p["valid"]) for p in expected["participants"]]
        pytest.skip(
            f"RDKit {rs.RDKIT_VERSION} writes {divergent} differently from the vectors' RDKit; "
            "SMILES-derived hashes are not comparable (InChIKey-based fields checked)"
        )
    for name in HASH_FIELDS:
        assert getattr(result, name) == expected[name], name
    observed = [
        {
            "role": p.role,
            "index": p.index,
            "canonical_smiles": p.species.canonical_smiles,
            "inchikey": p.species.inchikey,
            "valid": p.species.valid,
        }
        for p in result.participants
    ]
    assert observed == expected["participants"]


def test_rdkit_version_recorded() -> None:
    assert rs.RDKIT_VERSION
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if rs.RDKIT_VERSION != data["rdkit_version"]:  # pragma: no cover - informational
        pytest.skip(f"vectors generated with RDKit {data['rdkit_version']}, running {rs.RDKIT_VERSION}")


def test_normalize_species_clears_maps_and_reports_fragments() -> None:
    mapped = rs.normalize_species("[CH3:1][CH2:2][O-:3].[Na+:4]")
    plain = rs.normalize_species("[Na+].CC[O-]")
    assert mapped.valid and plain.valid
    assert mapped.canonical_smiles == plain.canonical_smiles
    assert sorted(mapped.fragments) == sorted(plain.fragments)
    assert mapped.formal_charge == 0
    assert mapped.atom_counts == {"C": 2, "O": 1, "Na": 1, "H": 5}
    assert sorted(mapped.fragment_heavy_atoms) == [1, 3]
    assert all(mapped.fragment_inchikeys)

    bad = rs.normalize_species("not_a_smiles")
    assert not bad.valid and bad.error == "Could not parse SMILES"
    assert rs.normalize_species("  ").error == "Enter a structure (SMILES)."


def test_parse_reaction_smiles_shapes_and_errors() -> None:
    assert rs.parse_reaction_smiles("CCO.O=[Cr](=O)=O>ClCCl>CC=O |f:1|") == {
        "reactants": ["CCO", "O=[Cr](=O)=O"],
        "reagents": ["ClCCl"],
        "products": ["CC=O"],
    }
    with pytest.raises(ValueError):
        rs.parse_reaction_smiles("CC>CC")
    with pytest.raises(ValueError):
        rs.parse_reaction_smiles(">>")
    with pytest.raises(ValueError):
        rs.parse_reaction_smiles("")


def test_conditions_int_and_float_hash_identically() -> None:
    roles = {"reactants": ["CCBr"], "products": ["CCO"]}
    a = rs.reaction_hashes(roles, {"ph": 7, "temperature_celsius": 25})
    b = rs.reaction_hashes(roles, {"ph": 7.0, "temperature_celsius": 25.0})
    assert a.conditions_hash == b.conditions_hash
    assert a.conditions_hash != rs.reaction_hashes(roles, {}).conditions_hash


# ---------------------------------------------------------------------------
# corpus keys
# ---------------------------------------------------------------------------

# A FlowER-style mapped SN2 step with explicit mapped hydrogens, and the same
# chemistry as a user would type it (unmapped, implicit H, reordered).
FLOWER_LEFT = ["[Cl-:1]", "[C:2]([H:4])([H:5])([H:6])[Br:3]"]
FLOWER_RIGHT = ["[Br-:3]", "[C:2]([H:4])([H:5])([H:6])[Cl:1]"]
USER_LEFT = ["BrC", "[Cl-]"]
USER_RIGHT = ["ClC", "[Br-]"]


def test_corpus_keys_flower_mapped_equals_user_unmapped() -> None:
    flower = rs.corpus_keys(FLOWER_LEFT, FLOWER_RIGHT)
    user = rs.corpus_keys(USER_LEFT, USER_RIGHT)
    assert flower == user
    submission = rs.corpus_keys_from_submission({"reactants": USER_LEFT[::-1], "products": USER_RIGHT[::-1]})
    assert submission == flower
    for value in flower.as_dict().values():
        assert 0 <= value < 2**64


def test_corpus_keys_direction_matters() -> None:
    forward = rs.corpus_keys(USER_LEFT, USER_RIGHT)
    reverse = rs.corpus_keys(USER_RIGHT, USER_LEFT)
    for kind in ("exact", "core", "endpoint", "family"):
        assert getattr(forward, kind) != getattr(reverse, kind), kind


def test_corpus_kinds_are_separately_prefixed() -> None:
    keys = rs.corpus_keys(USER_LEFT, USER_RIGHT)
    assert len({keys.exact, keys.core, keys.endpoint}) == 3


def test_submission_auxiliaries_only_move_endpoint() -> None:
    base = rs.corpus_keys_from_submission({"reactants": ["CC(=O)OC", "O"], "products": ["CC(=O)O", "CO"]})
    with_solvent = rs.corpus_keys_from_submission(
        {"reactants": ["CC(=O)OC", "O"], "products": ["CC(=O)O", "CO"], "solvents": [("C1CCOC1", None)]}
    )
    assert base.exact == with_solvent.exact
    assert base.core == with_solvent.core
    assert base.family == with_solvent.family
    assert base.endpoint != with_solvent.endpoint
    # FlowER puts everything on the left: its endpoint matches the submission's
    # endpoint when the user lists the same auxiliaries.
    flower = rs.corpus_keys(["CC(=O)OC", "O", "C1CCOC1"], ["CC(=O)O", "CO"])
    assert flower.endpoint == with_solvent.endpoint


def test_family_ignores_spectators_and_counterions() -> None:
    sodium = rs.corpus_keys(["Cc1ccc(cc1)C(=O)OC", "[OH-]", "[Na+]"], ["Cc1ccc(cc1)C(=O)[O-]", "[Na+]", "CO"])
    potassium = rs.corpus_keys(["Cc1ccc(cc1)C(=O)OC", "[OH-]", "[K+]"], ["Cc1ccc(cc1)C(=O)[O-]", "[K+]", "CO"])
    assert sodium.family == potassium.family
    assert sodium.exact != potassium.exact
    # A heavy species carried through unchanged (e.g. DMF) does not become the family.
    with_dmf = rs.corpus_keys(
        ["Cc1ccc(cc1)C(=O)OC", "[OH-]", "CN(C)C=O"], ["Cc1ccc(cc1)C(=O)[O-]", "CO", "CN(C)C=O"]
    )
    without = rs.corpus_keys(["Cc1ccc(cc1)C(=O)OC", "[OH-]"], ["Cc1ccc(cc1)C(=O)[O-]", "CO"])
    assert with_dmf.family == without.family


# Acid-catalysed dehydration of tert-butanol as FlowER carries it (mapped,
# catalytic H+ on both sides, water released) and as a user types it.
FLOWER_DEHYDRATION_LEFT = ["[CH3:1][C:2]([CH3:3])([CH3:4])[OH:5]", "[H+:6]"]
FLOWER_DEHYDRATION_RIGHT = ["[CH3:1][C:2]([CH3:3])=[CH2:4]", "[OH2:5]", "[H+:6]"]


def test_core_drops_catalysts_spectators_and_small_byproducts() -> None:
    flower = rs.corpus_keys(FLOWER_DEHYDRATION_LEFT, FLOWER_DEHYDRATION_RIGHT)
    user = rs.corpus_keys_from_submission({"reactants": ["CC(C)(C)O"], "products": ["C=C(C)C"]})
    assert flower.core == user.core
    assert flower.family == user.family
    assert flower.exact != user.exact
    assert rs.corpus_keys(["CC(C)(C)O"], ["C=C(C)C"]).core == flower.core


def test_core_distinguishes_a_different_transformation_of_the_same_substrate() -> None:
    dehydration = rs.corpus_keys(FLOWER_DEHYDRATION_LEFT, FLOWER_DEHYDRATION_RIGHT)
    substitution = rs.corpus_keys(["CC(C)(C)O", "[H+]", "[Cl-]"], ["CC(C)(C)Cl", "O", "[H+]"])
    assert substitution.core != dehydration.core
    assert substitution.core == rs.corpus_keys(["CC(C)(C)O"], ["CC(C)(C)Cl"]).core


def test_core_side_falls_back_when_filters_would_empty_it() -> None:
    # Every product fragment is small: the side keeps its shared-filtered set.
    proton_transfer = rs.corpus_keys(["[OH-]", "[H+]"], ["O"])
    assert proton_transfer.core == rs.corpus_keys(["[H+]", "[OH-]"], ["O"]).core
    assert proton_transfer.core != rs.corpus_keys(["[OH-]"], ["O"]).core


def test_corpus_keys_reject_invalid_input() -> None:
    with pytest.raises(rs.CorpusKeyError):
        rs.corpus_keys(["not_a_smiles"], ["CC"])
    with pytest.raises(rs.CorpusKeyError):
        rs.corpus_keys([], ["CC"])
    with pytest.raises(rs.CorpusKeyError):
        rs.corpus_keys_from_submission({"reactants": ["CC"]})


# ---------------------------------------------------------------------------
# fingerprints
# ---------------------------------------------------------------------------


def test_fingerprints_shape_determinism_and_map_invariance() -> None:
    diff, prod = rs.reaction_fingerprints(FLOWER_LEFT, FLOWER_RIGHT)
    assert len(diff) == len(prod) == rs.FINGERPRINT_BYTES == 32
    assert rs.reaction_fingerprints(FLOWER_LEFT, FLOWER_RIGHT) == (diff, prod)
    assert rs.reaction_fingerprints(USER_LEFT, USER_RIGHT) == (diff, prod)
    assert any(diff) and any(prod)


def test_fingerprint_spectators_cancel() -> None:
    plain = rs.reaction_fingerprints(USER_LEFT, USER_RIGHT)
    with_water = rs.reaction_fingerprints(USER_LEFT + ["O"], USER_RIGHT + ["O"])
    assert plain == with_water


def test_tanimoto() -> None:
    a, _ = rs.reaction_fingerprints(["CCBr", "[Cl-]"], ["CCCl", "[Br-]"])
    b, _ = rs.reaction_fingerprints(["CCCBr", "[Cl-]"], ["CCCCl", "[Br-]"])
    c, _ = rs.reaction_fingerprints(["C=CC=C", "C=C"], ["C1=CCCCC1"])
    assert rs.tanimoto(a, a) == 1.0
    assert rs.tanimoto(a, b) > rs.tanimoto(a, c)
    assert rs.tanimoto(bytes(32), bytes(32)) == 0.0
    with pytest.raises(ValueError):
        rs.tanimoto(bytes(32), bytes(31))
