"""Tests for mechanistic_agent.input_converter."""
from __future__ import annotations

import pytest

from mechanistic_agent.input_converter import (
    ConversionResult,
    auto_convert,
    convert_common_name,
    convert_inchi,
    convert_many,
    convert_mol_block,
    convert_smiles,
    detect_format,
)

try:
    from rdkit import Chem as _Chem

    HAS_RDKIT = _Chem is not None
except ImportError:
    HAS_RDKIT = False


# ---------------------------------------------------------------------------
# detect_format
# ---------------------------------------------------------------------------

class TestDetectFormat:
    def test_smiles(self):
        assert detect_format("C=O") == "smiles"
        assert detect_format("CC(=O)O") == "smiles"
        assert detect_format("[Na+].[Cl-]") == "smiles"

    def test_inchi(self):
        assert detect_format("InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3") == "inchi"

    def test_mol_block(self):
        block = "\n  Mrv  01272508372D\n\n  2  1  0  0  0  0  999 V2000\n"
        assert detect_format(block) == "mol_block"
        block_m_end = "header\n M  END\n"
        assert detect_format(block_m_end) == "mol_block"

    def test_common_name(self):
        assert detect_format("ethanol") == "name"
        assert detect_format("acetic acid") == "name"
        assert detect_format("sodium chloride") == "name"

    def test_smiles_with_brackets_not_name(self):
        # Contains brackets → not detected as name even if it has spaces
        assert detect_format("[Na+] [Cl-]") != "name"


# ---------------------------------------------------------------------------
# convert_smiles
# ---------------------------------------------------------------------------

class TestConvertSmiles:
    def test_valid(self):
        result = convert_smiles("C=O")
        assert result.success
        assert result.canonical_smiles is not None
        assert result.input_format == "smiles"

    @pytest.mark.skipif(not HAS_RDKIT, reason="RDKit not installed")
    def test_invalid(self):
        result = convert_smiles("not_a_smiles!!!")
        assert not result.success
        assert result.canonical_smiles is None
        assert result.error is not None

    @pytest.mark.skipif(not HAS_RDKIT, reason="RDKit not installed")
    def test_canonicalisation(self):
        result = convert_smiles("OCC")
        assert result.success
        assert result.canonical_smiles == "CCO"


# ---------------------------------------------------------------------------
# convert_inchi
# ---------------------------------------------------------------------------

class TestConvertInchi:
    def test_valid_ethanol(self):
        inchi = "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
        result = convert_inchi(inchi)
        assert result.input_format == "inchi"
        if result.success:
            assert result.canonical_smiles is not None

    def test_invalid(self):
        result = convert_inchi("InChI=garbage")
        assert result.input_format == "inchi"
        # Either fails gracefully or produces a result
        assert isinstance(result.success, bool)


# ---------------------------------------------------------------------------
# convert_mol_block
# ---------------------------------------------------------------------------

class TestConvertMolBlock:
    def test_invalid_block(self):
        result = convert_mol_block("not a mol block but has\nV2000\nM  END\n")
        assert result.input_format == "mol_block"
        # Graceful failure expected
        assert isinstance(result.success, bool)


# ---------------------------------------------------------------------------
# convert_common_name
# ---------------------------------------------------------------------------

class TestConvertCommonName:
    def test_graceful_on_any_outcome(self):
        """PubChem lookup must not raise regardless of network availability."""
        result = convert_common_name("ethanol")
        assert isinstance(result.success, bool)
        assert result.input_format == "name"
        if not result.success:
            assert result.error is not None
        else:
            assert result.canonical_smiles is not None


# ---------------------------------------------------------------------------
# auto_convert
# ---------------------------------------------------------------------------

class TestAutoConvert:
    def test_dispatches_smiles(self):
        result = auto_convert("C=O")
        assert result.input_format == "smiles"
        assert result.success

    def test_dispatches_inchi(self):
        inchi = "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
        result = auto_convert(inchi)
        assert result.input_format == "inchi"

    def test_dispatches_name(self):
        result = auto_convert("acetic acid")
        assert result.input_format == "name"


# ---------------------------------------------------------------------------
# convert_many
# ---------------------------------------------------------------------------

class TestConvertMany:
    def test_batch(self):
        results = convert_many(["C=O", "CC(=O)O"])
        assert len(results) == 2
        for r in results:
            assert r.success

    def test_empty(self):
        assert convert_many([]) == []


# ---------------------------------------------------------------------------
# compare_with_known_answers (from run_evaluator)
# ---------------------------------------------------------------------------

class TestCompareWithKnownAnswers:
    def test_no_verified_mechanism(self):
        from mechanistic_agent.run_evaluator import compare_with_known_answers

        result = compare_with_known_answers([], None)
        assert result == {"available": False}

    def test_empty_steps(self):
        from mechanistic_agent.run_evaluator import compare_with_known_answers

        result = compare_with_known_answers([], {"steps": []})
        assert result["available"] is True
        assert result["expected_step_count"] == 0

    def test_step_count_match(self):
        from mechanistic_agent.run_evaluator import compare_with_known_answers

        verified = {
            "steps": [
                {"step_index": 1, "resulting_state": ["C1OCOC1"]},
            ],
            "provisional": False,
            "source_refs": ["https://example.com"],
        }
        mechanism_rows = [
            {"step_name": "mechanism_synthesis", "output": '{"resulting_state": ["C1OCOC1"]}'},
        ]
        result = compare_with_known_answers(mechanism_rows, verified)
        assert result["available"] is True
        assert result["expected_step_count"] == 1
        assert result["actual_step_count"] == 1
        assert result["step_count_match"] is True
        assert result["product_match"] is True
        assert result["provisional"] is False
        assert result["source_refs"] == ["https://example.com"]

    def test_product_mismatch(self):
        from mechanistic_agent.run_evaluator import compare_with_known_answers

        verified = {
            "steps": [{"step_index": 1, "resulting_state": ["C1OCOC1"]}],
        }
        mechanism_rows = [
            {"step_name": "mechanism_synthesis", "output": '{"resulting_state": ["CCO"]}'},
        ]
        result = compare_with_known_answers(mechanism_rows, verified)
        assert result["product_match"] is False

    def test_known_mechanism_target_smiles_format(self):
        from mechanistic_agent.run_evaluator import compare_with_known_answers

        known = {
            "steps": [{"step_index": 1, "target_smiles": "C1OCOC1"}],
        }
        mechanism_rows = [
            {"step_name": "mechanism_synthesis", "output": '{"resulting_state": ["C1OCOC1"]}'},
        ]
        result = compare_with_known_answers(mechanism_rows, known)
        assert result["available"] is True
        assert result["product_match"] is True
