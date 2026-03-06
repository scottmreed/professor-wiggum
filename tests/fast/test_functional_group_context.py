import json

import pytest

from mechanistic_agent import tools

pytest.importorskip("rdkit")


@pytest.fixture(autouse=True)
def clear_functional_group_history():
    history = getattr(tools, "_FUNCTIONAL_GROUP_HISTORY", None)
    if history is not None:
        history.clear()
    yield
    if history is not None:
        history.clear()


def test_atom_mapping_uses_cached_functional_groups(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("MECHANISTIC_FUNCTIONAL_GROUPS_ENABLED", "1")

    starting = ["C=O"]
    products = ["CO"]

    fingerprint_json = tools.fingerprint_functional_groups(starting + products)
    fingerprint_data = json.loads(fingerprint_json)["functional_groups"]

    result_json = tools.attempt_atom_mapping(starting, products)
    result = json.loads(result_json)

    starting_groups = result["functional_groups"]["starting_materials"]
    product_groups = result["functional_groups"]["products"]
    assert starting_groups == {smiles: fingerprint_data[smiles] for smiles in starting}
    assert product_groups == {smiles: fingerprint_data[smiles] for smiles in products}

    start_source = result["functional_group_context"]["starting_materials"]["source"]
    product_source = result["functional_group_context"]["products"]["source"]
    assert "fingerprint_functional_groups" in start_source
    assert "fingerprint_functional_groups" in product_source
    assert "error" in result


def test_atom_mapping_computes_functional_groups_when_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("MECHANISTIC_FUNCTIONAL_GROUPS_ENABLED", "1")

    starting = ["C=O"]
    products = ["CO"]

    result_json = tools.attempt_atom_mapping(starting, products)
    result = json.loads(result_json)

    start_source = result["functional_group_context"]["starting_materials"]["source"]
    product_source = result["functional_group_context"]["products"]["source"]
    assert "computed" in start_source
    assert "computed" in product_source


def test_checkmol_functional_group_matches():
    fingerprint_json = tools.fingerprint_functional_groups(["CC=O", "CC(=O)O"])
    fingerprint_data = json.loads(fingerprint_json)["functional_groups"]

    aldehyde_groups = fingerprint_data.get("CC=O", {})
    assert "aldehyde" in aldehyde_groups
    assert aldehyde_groups["aldehyde"] >= 1

    acid_groups = fingerprint_data.get("CC(=O)O", {})
    assert "carboxylic_acid" in acid_groups
    assert acid_groups["carboxylic_acid"] >= 1


def test_fingerprint_functional_groups_handles_atom_mapped_smiles():
    fingerprint_json = tools.fingerprint_functional_groups(["[CH3:1][CH:2]=[O:3]"])
    fingerprint_data = json.loads(fingerprint_json)["functional_groups"]

    mapped_groups = fingerprint_data.get("[CH3:1][CH:2]=[O:3]", {})
    assert "aldehyde" in mapped_groups
    assert mapped_groups["aldehyde"] >= 1


def test_classify_functional_group_transformation_reports_uncertainty():
    result = tools.classify_functional_group_transformation(
        ["C=C"],
        ["CC(O)Br"],
    )

    assert result["label"] == "alkene -> alcohol"
    assert result["uncertain"] is True
    assert "alkene -> alkyl halide" in result["label_candidates"]
    assert "best-effort" in result["llm_summary"].lower()
