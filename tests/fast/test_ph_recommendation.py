"""Tests pinning which branch of ``recommend_ph`` runs.

``recommend_ph`` has three branches: a user-supplied pH short-circuit, a
Dimorphite-DL branch, and a SMARTS heuristic fallback. Before this fix the
Dimorphite branch was dead code: the module imported ``DimorphiteDL``, a
class dropped from ``dimorphite_dl`` 2.0+, so the import always failed and
every no-pH call silently fell through to the heuristic. These tests pin
which branch executes for each of the three input combinations so a future
regression (e.g. reverting to the old class-based import) is caught.
"""

import json

import pytest

from mechanistic_agent import tools

pytest.importorskip("rdkit")


def test_user_supplied_ph_short_circuits_both_other_branches():
    """A user-supplied pH always wins, regardless of Dimorphite availability."""

    result = json.loads(tools.recommend_ph(["CC(=O)O"], ["CC(=O)[O-]"], 7.4))

    assert result["source"] == "user"
    assert result["recommended"] == 7.4
    assert result["provided_ph"] == 7.4


def test_no_ph_uses_dimorphite_branch_when_available(monkeypatch):
    """When dimorphite_dl's protonate_smiles is importable, it is used (not the heuristic)."""

    def fake_protonate_smiles(smiles, ph_min=0.0, ph_max=14.0, **kwargs):
        return [smiles]

    monkeypatch.setattr(tools, "_dimorphite_protonate_smiles", fake_protonate_smiles)

    result = json.loads(tools.recommend_ph(["CC(=O)O"], ["CC(=O)[O-]"], None))

    assert result["source"] == "dimorphite_dl"
    assert result["recommended"] == "see_profiles"
    assert "profiles" in result
    assert result["profiles"]["CC(=O)O"] == ["CC(=O)O"]
    assert result["profiles"]["CC(=O)[O-]"] == ["CC(=O)[O-]"]
    # The heuristic-only fields must not be present when the Dimorphite branch ran.
    assert "recommended_range" not in result


def test_no_ph_falls_back_to_heuristic_when_dimorphite_unavailable(monkeypatch):
    """When dimorphite_dl is not importable, the SMARTS heuristic runs instead."""

    monkeypatch.setattr(tools, "_dimorphite_protonate_smiles", None)

    result = json.loads(tools.recommend_ph(["CC(=O)O"], ["CC(=O)[O-]"], None))

    assert result["source"] == "heuristic"
    assert "recommended_range" in result
    assert "acidic_score" in result
    assert "basic_score" in result
    # The Dimorphite-only fields must not be present when the heuristic ran.
    assert "profiles" not in result


def test_dimorphite_branch_survives_per_smiles_errors(monkeypatch):
    """A per-SMILES failure inside the Dimorphite branch stays in that branch."""

    def flaky_protonate_smiles(smiles, ph_min=0.0, ph_max=14.0, **kwargs):
        if smiles == "bad":
            raise ValueError("boom")
        return [smiles]

    monkeypatch.setattr(tools, "_dimorphite_protonate_smiles", flaky_protonate_smiles)

    result = json.loads(tools.recommend_ph(["bad"], ["CC(=O)[O-]"], None))

    assert result["source"] == "dimorphite_dl"
    assert result["profiles"]["bad"] == ["error: boom"]
    assert result["profiles"]["CC(=O)[O-]"] == ["CC(=O)[O-]"]
