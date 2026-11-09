"""catalytic_cycle spec validation and accessible long description."""

import copy
import json
from pathlib import Path

import pytest

from organometallic_chem.catalytic_cycle import (
    describe_catalytic_cycle,
    species_values,
    validate_catalytic_cycle,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "catalytic_cycles" / "suzuki_miyaura.json"
PME3 = "P(C)(C)C"


def suzuki():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_suzuki_fixture_validates():
    assert validate_catalytic_cycle(suzuki()) == []


def test_suzuki_species_values_are_computed_from_smiles():
    values = species_values(suzuki())
    assert values["pd0"] == {"ox_state": 0, "electron_count": 14}
    assert values["oa"] == {"ox_state": 2, "electron_count": 16}
    assert values["tm"] == {"ox_state": 2, "electron_count": 16}


def test_open_cycle_is_rejected():
    spec = suzuki()
    spec["steps"][-1]["to"] = "oa"
    errors = validate_catalytic_cycle(spec)
    assert any("not closed" in e for e in errors)


def test_out_of_order_steps_are_rejected():
    spec = suzuki()
    spec["steps"][0], spec["steps"][1] = spec["steps"][1], spec["steps"][0]
    assert any("list steps in cycle order" in e for e in validate_catalytic_cycle(spec))


def test_wrong_kind_for_the_oxidation_state_change_is_rejected():
    spec = suzuki()
    spec["steps"][0]["kind"] = "ligand_association"  # Pd(0) -> Pd(II) is not that
    errors = validate_catalytic_cycle(spec)
    assert any("'pd0' -> 'oa': oxidation state changes by +2" in e and "expected +0" in e for e in errors)


def test_authored_ox_state_disagreeing_with_smiles_is_rejected():
    spec = suzuki()
    spec["species"][1]["ox_state"] = 0
    assert any("authored ox_state 0 disagrees with 2" in e for e in validate_catalytic_cycle(spec))


def test_smiles_less_species_use_authored_values():
    spec = {
        "title": "Toy",
        "species": [
            {"id": "a", "label": "M(0)", "ox_state": 0, "electron_count": 14},
            {"id": "b", "label": "M(I)", "ox_state": 1, "electron_count": 16},
        ],
        "steps": [
            {"from": "a", "to": "b", "kind": "oxidative_addition"},
            {"from": "b", "to": "a", "kind": "reductive_elimination"},
        ],
    }
    errors = validate_catalytic_cycle(spec)
    assert any("oxidation state changes by +1" in e for e in errors)
    assert any("oxidation state changes by -1" in e for e in errors)


def test_species_smiles_must_pass_the_lint():
    spec = suzuki()
    spec["species"][0]["smiles"] = "[Pd](P(C)(C)C)P(C)(C)C"
    assert any("species 'pd0'" in e and "dative" in e for e in validate_catalytic_cycle(spec))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda s: s["species"].append(copy.deepcopy(s["species"][0])), "duplicate species id"),
        (lambda s: s["steps"][0].update(kind="insertion"), "kind 'insertion'"),
        (lambda s: s["steps"][0].update(to="nowhere"), "'nowhere' is not a species id"),
        (lambda s: s.update(colour="red"), "unknown catalytic_cycle key"),
        (lambda s: s["steps"][1]["enters"].append({"smiles": "C1CC"}), "does not parse"),
        (lambda s: s.pop("title"), "title is required"),
        (lambda s: s["species"].append({"id": "extra", "label": "X"}), "not on the cycle"),
    ],
)
def test_malformed_specs_are_rejected(mutation, message):
    spec = suzuki()
    mutation(spec)
    assert any(message in e for e in validate_catalytic_cycle(spec)), validate_catalytic_cycle(spec)


def test_heck_style_steps_check_insertion_and_beta_h_deltas():
    spec = {
        "title": "Heck fragment",
        "species": [
            {"id": "pi", "label": "PhPd(Br)L(alkene)", "smiles": f"Br[Pd](<-{PME3})(c1ccccc1)<-C=C"},
            {"id": "alkyl", "label": "alkyl-Pd(Br)L", "smiles": f"Br[Pd](<-{PME3})CCc1ccccc1"},
            {"id": "hyd", "label": "HPd(Br)L(styrene)", "smiles": f"Br[PdH](<-{PME3})<-C=Cc1ccccc1"},
        ],
        "steps": [
            {"from": "pi", "to": "alkyl", "kind": "migratory_insertion"},
            {"from": "alkyl", "to": "hyd", "kind": "beta_hydride_elimination"},
            {"from": "hyd", "to": "pi", "kind": "other", "label": "alkene exchange"},
        ],
    }
    assert validate_catalytic_cycle(spec) == []
    spec["steps"][0]["kind"] = "ligand_dissociation"  # same e delta, also 0 ox delta
    assert validate_catalytic_cycle(spec) == []
    spec["steps"][1]["kind"] = "ligand_dissociation"
    assert any("electron count changes by +2" in e for e in validate_catalytic_cycle(spec))


def test_long_description_reads_the_cycle_in_order():
    text = describe_catalytic_cycle(suzuki())
    assert text.startswith("Suzuki–Miyaura cross-coupling: a catalytic cycle of 3 steps")
    order = [
        "Pd(0)L2 undergoes oxidative addition with Ar–Br to give trans-ArPd(Br)L2 (Pd(II), 16-electron)",
        "undergoes transmetalation with Ar′B(OH)2 and base, releasing B(OH)3 and Br−",
        "undergoes reductive elimination, releasing Ar–Ar′ and regenerating Pd(0)L2.",
    ]
    positions = [text.index(fragment) for fragment in order]
    assert positions == sorted(positions)
    assert "Key: L = PPh3" in text
