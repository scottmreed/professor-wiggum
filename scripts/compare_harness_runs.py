"""Compare harness eval runs case by case: quality_v2, LLM calls, wall time, thread events.

Each arm is ``label=<eval_run_id>[,<eval_run_id>...]`` (later runs win per case, as
``publish-results --combine``) or ``label=results/runs/<record>.json`` (a published record:
scores only, no call counts).

    python scripts/compare_harness_runs.py \
        --arm jev=db71026865a34e1283b6ed8d3c712d83,6d403f9219524729a8c6109c63f0c3ea \
        --arm baseline=results/runs/2026-10-05_harness_free_baseline_no_product_claude-opus-5-5_064f00ba.json \
        --arm threaded=<eval_run_id> [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CALL_EVENTS = {"inference_call_completed", "inference_call_failed"}
THREAD_EVENTS = ("thread_forked", "thread_compacted", "thread_fallback", "thread_root_reset")


def _all_events(store: Any, run_id: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    after = 0
    while True:
        page = store.list_events(run_id, after_seq=after, limit=2000)
        if not page:
            return events
        events.extend(page)
        after = int(page[-1]["seq"])


def _run_metrics(store: Any, run_id: str) -> Dict[str, Any]:
    events = _all_events(store, run_id)
    calls = [e for e in events if e["event_type"] in CALL_EVENTS]
    by_step = Counter(str(e.get("step_name") or (e.get("payload") or {}).get("step_name") or "?") for e in calls)
    kinds = Counter(e["event_type"] for e in events)
    prompt_chars = [int((e.get("payload") or {}).get("prompt_chars") or 0) for e in events if e["event_type"] == "thread_turn_appended"]
    ts = [float(e["ts"]) for e in events if e.get("ts") is not None]
    accepted = [e for e in events if e["event_type"] == "mechanism_step_accepted"]
    return {
        "llm_calls": len(calls),
        "failed_calls": kinds.get("inference_call_failed", 0),
        "calls_by_step": dict(by_step),
        "wall_s": round(max(ts) - min(ts), 1) if ts else None,
        "accepted_steps": len(accepted),
        "backtracks": kinds.get("backtrack", 0),
        "thread": {name: kinds.get(name, 0) for name in THREAD_EVENTS},
        "max_prompt_chars": max(prompt_chars) if prompt_chars else None,
        "run_status": (store.get_run_row(run_id) or {}).get("status"),
    }


def _eval_arm(store: Any, eval_run_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    from mechanistic_agent.rescoring import default_expected_resolver
    from mechanistic_agent.results_publish import _case_quality

    resolver = default_expected_resolver(store)
    cases: Dict[str, Dict[str, Any]] = {}
    for eval_run_id in eval_run_ids:
        run = store.get_eval_run(eval_run_id) or {}
        for result in store.list_eval_run_results(eval_run_id):
            run_id = result.get("run_id")
            expected = resolver(result, run)
            snapshot = store.get_run_snapshot(run_id) if run_id else None
            quality = _case_quality(dict(result.get("summary") or {}), snapshot, expected) or {}
            v2 = quality.get("quality_v2") or {}
            row: Dict[str, Any] = {
                "points": v2.get("points"),
                "passed": v2.get("passed"),
                "product_correct": (v2.get("axes") or {}).get("product"),
                "axes": v2.get("axes"),
                "steps": v2.get("step_count"),
                "latency_s": round(float(result.get("latency_ms") or 0) / 1000.0, 1) or None,
                "eval_run_id": eval_run_id,
            }
            if run_id:
                row.update(_run_metrics(store, run_id))
            cases[str(result["case_id"])] = row
    return cases


def _record_arm(path: Path) -> Dict[str, Dict[str, Any]]:
    record = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(c["case_id"]): {
            "points": c.get("quality_points"),
            "passed": c.get("quality_passed"),
            "product_correct": (c.get("axes") or {}).get("product"),
            "axes": c.get("axes"),
            "steps": c.get("predicted_steps"),
            "latency_s": c.get("latency_s"),
        }
        for c in record.get("cases") or []
    }


def _mean(values: List[Optional[float]], digits: int = 1) -> Optional[float]:
    nums = [float(v) for v in values if v is not None]
    return round(sum(nums) / len(nums), digits) if nums else None


def _summary(cases: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(cases.values())
    axes_keys = sorted({k for r in rows for k in (r.get("axes") or {})})
    out: Dict[str, Any] = {
        "cases": len(rows),
        "points": _mean([r.get("points") for r in rows]),
        "passed": sum(1 for r in rows if r.get("passed")),
        "product_correct": sum(1 for r in rows if r.get("product_correct")),
        "axes": {k: _mean([(r.get("axes") or {}).get(k) for r in rows], 3) for k in axes_keys},
        "latency_s": _mean([r.get("latency_s") for r in rows]),
    }
    if any("llm_calls" in r for r in rows):
        out["llm_calls"] = _mean([r.get("llm_calls") for r in rows])
        steps = sum(int(r.get("accepted_steps") or 0) for r in rows)
        out["llm_calls_per_accepted_step"] = round(sum(int(r.get("llm_calls") or 0) for r in rows) / max(1, steps), 2)
        out["wall_s"] = _mean([r.get("wall_s") for r in rows])
        totals: Counter = Counter()
        for r in rows:
            totals.update(r.get("calls_by_step") or {})
        out["calls_by_step"] = dict(totals.most_common())
        out["thread"] = {name: sum(int((r.get("thread") or {}).get(name, 0)) for r in rows) for name in THREAD_EVENTS}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", required=True, help="label=<eval_run_id>[,...] or label=<record.json>")
    parser.add_argument("--db", default=None, help="SQLite path (default: the configured data dir)")
    parser.add_argument("--json", dest="json_out", default=None)
    args = parser.parse_args()

    from mechanistic_agent.core.db import RunStore

    store = None
    arms: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for spec in args.arm:
        label, _, value = spec.partition("=")
        if value.endswith(".json"):
            arms[label] = _record_arm(Path(value))
            continue
        if store is None:
            from mechanistic_agent.data_paths import db_path

            store = RunStore(Path(args.db) if args.db else db_path(ROOT))
        arms[label] = _eval_arm(store, [v for v in value.split(",") if v])

    labels = list(arms)
    case_ids = sorted({cid for cases in arms.values() for cid in cases})
    header = f"{'case':<16}" + "".join(f"{label[:22]:>24}" for label in labels)
    print(header)
    for cid in case_ids:
        cells = []
        for label in labels:
            r = arms[label].get(cid)
            if not r:
                cells.append(f"{'-':>24}")
                continue
            calls = f" {r['llm_calls']}c" if "llm_calls" in r else ""
            mark = "P" if r.get("product_correct") else "-"
            cells.append(f"{str(r.get('points')) + ' ' + mark + calls:>24}")
        print(f"{cid:<16}" + "".join(cells))
    summaries = {label: _summary(cases) for label, cases in arms.items()}
    print()
    print(json.dumps(summaries, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"summaries": summaries, "cases": arms}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
