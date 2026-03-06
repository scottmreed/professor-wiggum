import pytest

pytest.importorskip("rdkit")
pytest.importorskip("fastapi")

from mechanistic_agent.api.app import _parse_smirks_text  # noqa: E402


def test_parse_smirks_strips_metadata():
    result = _parse_smirks_text("C.O>>CO |10=20|")
    assert result["reactants"] == ["C", "O"]
    assert result["products"] == ["CO"]
    assert result["smirks"] == "C.O>>CO"


def test_parse_smirks_detects_atom_mapping():
    result = _parse_smirks_text("[C:1]O>>[C:1]O")
    assert result["mapping_detected"] is True


def test_parse_smirks_requires_arrow():
    with pytest.raises(ValueError):
        _parse_smirks_text("CO CO")
