#!/usr/bin/env python3
"""Generate docs/leaderboard_snapshot.png from the official holdout leaderboard data.

Usage:
    python scripts/update_leaderboard_snapshot.py

Reads directly from data/mechanistic.db (no server required).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mechanistic_agent.core.db import RunStore  # noqa: E402
from mechanistic_agent.data_paths import db_path as resolve_db_path  # noqa: E402


def main() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        sys.exit("matplotlib is required: pip install matplotlib")

    import matplotlib.pyplot as plt
    import numpy as np

    db_path = resolve_db_path(ROOT)
    if not db_path.exists():
        sys.exit(f"Database not found at {db_path}")

    store = RunStore(db_path)
    holdouts = store.list_eval_sets(purpose="leaderboard_holdout")
    if not holdouts:
        sys.exit("No leaderboard_holdout eval set found in the database.")

    eval_set_id = str(holdouts[0].get("id") or "")
    items = store.leaderboard(eval_set_id=eval_set_id, limit=20)
    if not items:
        sys.exit("No leaderboard rows found for the official holdout eval set.")

    # Group by model (best harness / best baseline per model)
    by_model: dict[str, dict] = {}
    for row in items:
        model = row.get("model_name") or row.get("model") or "unknown"
        thinking = row.get("thinking_level") or ""
        key = f"{model} ({thinking})" if thinking else model
        if key not in by_model:
            by_model[key] = {"harness": None, "baseline": None}
        score = float(row.get("mean_quality_score") or 0)
        if row.get("is_baseline"):
            prev = by_model[key]["baseline"]
            if prev is None or score > float(prev.get("mean_quality_score") or 0):
                by_model[key]["baseline"] = row
        else:
            prev = by_model[key]["harness"]
            if prev is None or score > float(prev.get("mean_quality_score") or 0):
                by_model[key]["harness"] = row

    # Sort by best harness score descending
    sorted_models = sorted(
        by_model.items(),
        key=lambda kv: float(
            (kv[1]["harness"] or kv[1]["baseline"] or {}).get("mean_quality_score", 0)
        ),
        reverse=True,
    )

    labels = [k for k, _ in sorted_models]
    harness_scores = [
        float((v["harness"] or {}).get("mean_quality_score", 0)) * 1000
        if v["harness"]
        else 0
        for _, v in sorted_models
    ]
    baseline_scores = [
        float((v["baseline"] or {}).get("mean_quality_score", 0)) * 1000
        if v["baseline"]
        else 0
        for _, v in sorted_models
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2), 5))
    bars_h = ax.bar(x - width / 2, harness_scores, width, label="Harness", color="#276621", alpha=0.8)
    bars_b = ax.bar(x + width / 2, baseline_scores, width, label="Baseline (no harness)", color="#1a6fa8", alpha=0.7)

    ax.set_ylabel("Clawdiators pts (quality × 1000)")
    ax.set_title("Official Holdout Leaderboard")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1000)
    ax.legend()

    # Value labels on bars
    for bar in bars_h:
        h = bar.get_height()
        if h > 0:
            ax.annotate(f"{h:.0f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for bar in bars_b:
        h = bar.get_height()
        if h > 0:
            ax.annotate(f"{h:.0f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out_path = ROOT / "docs" / "leaderboard_snapshot.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved leaderboard snapshot to {out_path}")


if __name__ == "__main__":
    main()
