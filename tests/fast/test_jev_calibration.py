"""Phase D calibration metrics (mechanistic_agent/decisions/calibration.py)."""
from __future__ import annotations

import math

import pytest

from mechanistic_agent.decisions.calibration import (
    accuracy,
    accuracy_by_band,
    brier_multiclass,
    brier_top_label,
    expected_calibration_error,
    summarize,
)


def test_brier_top_label() -> None:
    assert brier_top_label([1.0, 0.0], [True, False]) == 0.0
    assert brier_top_label([1.0], [False]) == 1.0
    assert math.isclose(brier_top_label([0.8, 0.6], [True, False]), (0.04 + 0.36) / 2)
    assert brier_top_label([], []) is None


def test_brier_multiclass() -> None:
    assert brier_multiclass([{"a": 1.0, "b": 0.0}], ["a"]) == 0.0
    assert math.isclose(brier_multiclass([{"a": 0.7, "b": 0.3}], ["b"]), 0.49 + 0.49)
    # a label missing from the distribution counts as probability 0
    assert math.isclose(brier_multiclass([{"a": 1.0}], ["z"]), 2.0)
    with pytest.raises(ValueError):
        brier_multiclass([{"a": 1.0}], [])


def test_ece_perfectly_calibrated_and_overconfident() -> None:
    # 10 cases at p=0.7, 7 correct -> calibrated
    assert math.isclose(expected_calibration_error([0.7] * 10, [True] * 7 + [False] * 3), 0.0, abs_tol=1e-12)
    # always 0.9 confident, half right -> ECE 0.4
    assert math.isclose(expected_calibration_error([0.9] * 4, [True, False, True, False]), 0.4)
    # confidence exactly 1.0 falls in the last bin
    assert expected_calibration_error([1.0], [True]) == 0.0
    # two bins weighted by size
    ece = expected_calibration_error([0.15, 0.15, 0.95, 0.95], [False, False, True, False], n_bins=10)
    assert math.isclose(ece, 0.5 * 0.15 + 0.5 * abs(0.5 - 0.95))


def test_accuracy_by_band() -> None:
    rows = accuracy_by_band([0.3, 0.55, 0.7, 0.85, 0.95, 1.0], [False, True, True, False, True, True])
    by_band = {row["band"]: row for row in rows}
    assert by_band["[0.00, 0.50)"]["n"] == 1 and by_band["[0.00, 0.50)"]["accuracy"] == 0.0
    assert by_band["[0.90, 1.00]"]["n"] == 2 and by_band["[0.90, 1.00]"]["accuracy"] == 1.0
    assert math.isclose(by_band["[0.90, 1.00]"]["mean_confidence"], 0.975)
    assert sum(row["n"] for row in rows) == 6
    empty = accuracy_by_band([], [])
    assert all(row["n"] == 0 and row["accuracy"] is None for row in empty)
    with pytest.raises(ValueError):
        accuracy_by_band([0.5], [True], bands=[0.0, 0.5, 0.5])


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        brier_top_label([0.5], [True, False])
    with pytest.raises(ValueError):
        expected_calibration_error([1.2], [True])


def test_summarize() -> None:
    report = summarize([0.9, 0.6], [True, False], distributions=[{"a": 0.9, "b": 0.1}, {"a": 0.6, "b": 0.4}],
                       labels=["a", "b"])
    assert report["n"] == 2 and report["accuracy"] == 0.5
    assert set(report) >= {"brier_top_label", "ece", "accuracy_by_band", "brier_multiclass"}
    assert accuracy([]) is None
