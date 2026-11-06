#!/usr/bin/env python3
"""Offline counterfactual for the arrow-count tie-break (read-only).

For every stored step with two or more validated candidates (a
``branch_point_created`` event), rebuild the rank-ordered validated list from the
``mechanism_synthesis`` step outputs of that attempt whose validation passed, and
compute what ``Coordinator._apply_arrow_count_preference`` would have chosen.

Reports: steps examined, steps that could be reconstructed, steps where the
preference would fire (and where the proton-transfer preference would have
pre-empted it), agreement with the original top candidate, and how the chosen
candidate's per-step quality axes (``quality_scoring.score_step``: balance,
notation, electrons; the axes quality_v2 averages per step) compare with the
displaced top's. Also times ``resonance_key`` and the preference on the real
candidate sets (p50 / p95).

The database is opened read-only (``mode=ro``); nothing is written to it.

    python scripts/arrow_count_counterfactual.py [--db PATH] [--json-out PATH]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mechanistic_agent.core import arrow_count  # noqa: E402
from mechanistic_agent.core.proton_transfer import (  # noqa: E402
    available_shuttles,
    choose_proton_transfer_preference,
)
from mechanistic_agent.quality_scoring import QualityStep, score_step  # noqa: E402
from mechanistic_agent.resonance import _KEY_CACHE, resonance_key  # noqa: E402


def _connect(db: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _json(text: Any) -> Any:
    try:
        return json.loads(text) if text else None
    except (TypeError, ValueError):
        return None


def _passed(validation: Any) -> bool:
    return isinstance(validation, dict) and validation.get("passed") is True


def _validated_rows(conn: sqlite3.Connection, run_id: str, attempt: int) -> List[Dict[str, Any]]:
    """Distinct passed mechanism_synthesis outputs for one attempt, in row order."""
    seen = set()
    out: List[Dict[str, Any]] = []
    for row in conn.execute(
        "SELECT output_json, validation_json FROM step_outputs "
        "WHERE run_id = ? AND step_name = 'mechanism_synthesis' AND attempt = ? ORDER BY rowid",
        (run_id, attempt),
    ):
        output = _json(row["output_json"]) or {}
        if not isinstance(output, dict) or not _passed(_json(row["validation_json"])):
            continue
        if output.get("soft_advance"):
            continue
        key = (str(output.get("reaction_smirks") or ""), tuple(output.get("resulting_state") or []))
        if key in seen:
            continue
        seen.add(key)
        out.append(output)
    return out


def _step_quality(output: Dict[str, Any], current: List[str], run_input: Dict[str, Any]) -> Dict[str, float]:
    step = QualityStep(
        step_index=1,
        current_state=list(output.get("current_state") or current),
        resulting_state=list(output.get("resulting_state") or []),
        reaction_smirks=str(output.get("reaction_smirks") or ""),
        electron_pushes=output.get("electron_pushes"),
    )
    scored = score_step(step, run_input.get("starting_materials") or [], run_input.get("products") or [])
    checks = scored["checks"]
    axes = {
        "balance": float(checks["atom_balance"] and checks["state_progress"]),
        "notation": float(checks["arrows"] and checks["smirks_matches_states"]),
        "electrons": float(bool(scored.get("electron_conserved"))),
    }
    axes["step_quality"] = sum(axes.values()) / 3.0
    return axes


def _pct(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return round(ordered[idx], 3)


def analyse(db: Path, created_before: Optional[float] = None) -> Dict[str, Any]:
    conn = _connect(db)
    inputs = {
        row["id"]: (_json(row["input_payload_json"]) or {})
        for row in conn.execute("SELECT id, input_payload_json FROM runs")
    }
    branch_events = conn.execute(
        "SELECT e.run_id, e.payload_json FROM run_events e JOIN runs r ON r.id = e.run_id "
        "WHERE e.event_type = 'branch_point_created' AND (? IS NULL OR r.created_at < ?) ORDER BY e.run_id, e.seq",
        (created_before, created_before),
    ).fetchall()

    counts = {
        "steps_examined": 0,
        "reconstructed": 0,
        "not_reconstructable": 0,
        "proton_preference_would_fire": 0,
        "arrow_preference_would_fire": 0,
        "agrees_with_rank1": 0,
        "unmeasurable_top": 0,
        "same_outcome_alternative": 0,
        "same_outcome_both_measurable": 0,
    }
    reasons: Dict[str, int] = {}
    quality_cmp = {"better": 0, "worse": 0, "same": 0}
    fired: List[Dict[str, Any]] = []
    key_ms: List[float] = []
    pref_ms: List[float] = []
    seen_pairs = set()

    for event in branch_events:
        payload = _json(event["payload_json"]) or {}
        run_id = event["run_id"]
        attempt = int(payload.get("step_index") or 0) + 1
        if (run_id, attempt) in seen_pairs:
            continue  # a revisit of the same step after a backtrack
        seen_pairs.add((run_id, attempt))
        counts["steps_examined"] += 1
        expected = 1 + int(payload.get("alternative_count") or len(payload.get("alternative_ranks") or []))
        rows = _validated_rows(conn, run_id, attempt)
        if len(rows) != expected:
            counts["not_reconstructable"] += 1
            continue
        counts["reconstructed"] += 1
        current = list(payload.get("current_state") or [])
        run_input = inputs.get(run_id) or {}
        resulting = [list(r.get("resulting_state") or []) for r in rows]
        smirks = [r.get("reaction_smirks") for r in rows]

        for species in {s for state in resulting for s in state}:
            _KEY_CACHE.pop(species.strip(), None)
            t0 = time.perf_counter()
            resonance_key(species)
            key_ms.append((time.perf_counter() - t0) * 1000.0)

        shuttles = available_shuttles(current, run_input.get("starting_materials") or [])
        if choose_proton_transfer_preference(current, resulting, shuttles=shuttles, reaction_smirks=smirks):
            counts["proton_preference_would_fire"] += 1
            continue

        _KEY_CACHE.clear()
        t0 = time.perf_counter()
        arrows = [arrow_count.candidate_arrow_count(s, r, current) for s, r in zip(smirks, resulting)]
        decision = arrow_count.choose_arrow_count_preference(resulting, arrows)
        pref_ms.append((time.perf_counter() - t0) * 1000.0)
        top_outcome = arrow_count.outcome_key(resulting[0])
        same = [i for i in range(1, len(rows)) if top_outcome is not None
                and arrow_count.outcome_key(resulting[i]) == top_outcome]
        if same:
            counts["same_outcome_alternative"] += 1
            if arrows[0] is not None and any(arrows[i] is not None for i in same):
                counts["same_outcome_both_measurable"] += 1
        if arrows[0] is None:
            counts["unmeasurable_top"] += 1
        if decision is None:
            counts["agrees_with_rank1"] += 1
            continue
        counts["arrow_preference_would_fire"] += 1
        reasons[decision["reason"]] = reasons.get(decision["reason"], 0) + 1
        chosen = rows[decision["chosen_index"]]
        top = rows[0]
        q_chosen = _step_quality(chosen, current, run_input)
        q_top = _step_quality(top, current, run_input)
        delta = q_chosen["step_quality"] - q_top["step_quality"]
        quality_cmp["better" if delta > 1e-9 else "worse" if delta < -1e-9 else "same"] += 1
        fired.append({
            "run_id": run_id,
            "step": attempt,
            "arrows": arrows,
            "decision": {k: decision[k] for k in decision if k != "outcome_key"},
            "top_smirks": top.get("reaction_smirks"),
            "chosen_smirks": chosen.get("reaction_smirks"),
            "step_quality_top": q_top,
            "step_quality_chosen": q_chosen,
        })

    return {
        "db": str(db),
        "created_before": created_before,
        "counts": counts,
        "reasons": reasons,
        "displaced_vs_chosen_step_quality": quality_cmp,
        "latency_ms": {
            "resonance_key": {"n": len(key_ms), "p50": _pct(key_ms, 0.5), "p95": _pct(key_ms, 0.95)},
            "preference_per_step": {"n": len(pref_ms), "p50": _pct(pref_ms, 0.5), "p95": _pct(pref_ms, 0.95),
                                    "mean": round(statistics.fmean(pref_ms), 3) if pref_ms else None},
        },
        "fired": fired,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", type=Path, default=None, help="SQLite store (default: data_paths.db_path())")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--created-before", type=float, default=None,
                        help="only runs created before this epoch time (pins the sample)")
    args = parser.parse_args(argv)
    db = args.db
    if db is None:
        from mechanistic_agent.data_paths import db_path

        db = db_path()
    if not Path(db).is_file():
        print(f"no database at {db}", file=sys.stderr)
        return 2
    report = analyse(Path(db), args.created_before)
    text = json.dumps(report, indent=2, default=str)
    if args.json_out:
        args.json_out.write_text(text + "\n")
    summary = {k: report[k] for k in ("counts", "reasons", "displaced_vs_chosen_step_quality", "latency_ms")}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
