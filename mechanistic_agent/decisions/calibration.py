"""Calibration metrics for decision questions (PRD §19 Phase D).

Pure functions, no model calls. ``confidence`` is the probability the engine
assigned to its selected option; ``correct`` says whether that option matched
the label.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

DEFAULT_BANDS: Sequence[float] = (0.0, 0.5, 0.65, 0.8, 0.9, 1.0)


def _check(confidences: Sequence[float], correct: Sequence[bool]) -> None:
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have the same length")
    for value in confidences:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"confidence {value} outside [0, 1]")


def accuracy(correct: Sequence[bool]) -> Optional[float]:
    return (sum(1 for c in correct if c) / len(correct)) if correct else None


def brier_top_label(confidences: Sequence[float], correct: Sequence[bool]) -> Optional[float]:
    """Mean squared error of the top-label probability against 0/1 correctness."""
    _check(confidences, correct)
    if not confidences:
        return None
    return sum((float(p) - (1.0 if y else 0.0)) ** 2 for p, y in zip(confidences, correct)) / len(confidences)


def brier_multiclass(distributions: Sequence[Mapping[str, float]], labels: Sequence[str]) -> Optional[float]:
    """Mean over cases of sum_k (p_k - 1[k == label])^2 (range 0..2).

    A label absent from a distribution counts as probability 0 for it.
    """
    if len(distributions) != len(labels):
        raise ValueError("distributions and labels must have the same length")
    if not distributions:
        return None
    total = 0.0
    for dist, label in zip(distributions, labels):
        keys = set(dist) | {label}
        total += sum((float(dist.get(k, 0.0)) - (1.0 if k == label else 0.0)) ** 2 for k in keys)
    return total / len(distributions)


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    n_bins: int = 10,
) -> Optional[float]:
    """Top-label ECE with ``n_bins`` equal-width bins over [0, 1].

    ``sum_b (n_b / N) * |accuracy_b - mean_confidence_b|``; confidence 1.0
    falls in the last bin.
    """
    _check(confidences, correct)
    if not confidences:
        return None
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    bins: List[List[int]] = [[] for _ in range(n_bins)]
    for index, p in enumerate(confidences):
        b = min(int(float(p) * n_bins), n_bins - 1)
        bins[b].append(index)
    n = len(confidences)
    ece = 0.0
    for members in bins:
        if not members:
            continue
        acc = sum(1 for i in members if correct[i]) / len(members)
        conf = sum(float(confidences[i]) for i in members) / len(members)
        ece += (len(members) / n) * abs(acc - conf)
    return ece


def accuracy_by_band(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    bands: Sequence[float] = DEFAULT_BANDS,
) -> List[Dict[str, Any]]:
    """Accuracy and mean confidence per confidence band ``[lo, hi)`` (last band closed)."""
    _check(confidences, correct)
    edges = list(bands)
    if len(edges) < 2 or any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError("bands must be strictly increasing with at least two edges")
    rows: List[Dict[str, Any]] = []
    for position, (lo, hi) in enumerate(zip(edges, edges[1:])):
        last = position == len(edges) - 2
        members = [
            i for i, p in enumerate(confidences)
            if lo <= float(p) < hi or (last and float(p) == hi)
        ]
        rows.append(
            {
                "band": f"[{lo:.2f}, {hi:.2f}{']' if last else ')'}",
                "n": len(members),
                "accuracy": (sum(1 for i in members if correct[i]) / len(members)) if members else None,
                "mean_confidence": (
                    sum(float(confidences[i]) for i in members) / len(members) if members else None
                ),
            }
        )
    return rows


def summarize(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    distributions: Optional[Sequence[Mapping[str, float]]] = None,
    labels: Optional[Sequence[str]] = None,
    n_bins: int = 10,
    bands: Sequence[float] = DEFAULT_BANDS,
) -> Dict[str, Any]:
    """All Phase D metrics for one question and one engine."""
    report: Dict[str, Any] = {
        "n": len(confidences),
        "accuracy": accuracy(correct),
        "brier_top_label": brier_top_label(confidences, correct),
        "ece": expected_calibration_error(confidences, correct, n_bins=n_bins),
        "accuracy_by_band": accuracy_by_band(confidences, correct, bands=bands),
    }
    if distributions is not None and labels is not None:
        report["brier_multiclass"] = brier_multiclass(distributions, labels)
    return report


__all__ = [
    "DEFAULT_BANDS",
    "accuracy",
    "accuracy_by_band",
    "brier_multiclass",
    "brier_top_label",
    "expected_calibration_error",
    "summarize",
]
