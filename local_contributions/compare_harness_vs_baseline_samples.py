#!/usr/bin/env python
"""Compare the full harness against the one-shot baseline on holdout samples.

Default behavior:
- select one official holdout case for each mechanism step bucket 1..6
- run the standard harness and the harness-free baseline on the same cases
- print a side-by-side terminal summary for quick manual comparison

Examples:
    source .venv/bin/activate
    python local_contributions/compare_harness_vs_baseline_samples.py --model-name gpt-4o-mini
    python local_contributions/compare_harness_vs_baseline_samples.py --model-name claude-sonnet-4.6
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mechanistic_agent.config import LLM_STEP_KEYS
from mechanistic_agent.core import RegistrySet, RunCoordinator, RunStore
from mechanistic_agent.core.baseline_runner import BaselineRunner, score_baseline_result
from mechanistic_agent.model_registry import (
    calculate_cost,
    get_model_family,
    normalise_token_usage,
    resolve_model_key,
    to_internal_reasoning_level,
)
from mechanistic_agent.scoring import score_snapshot_against_known

try:  # pragma: no cover - optional helper
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - fallback when python-dotenv is absent
    def load_dotenv(_: Path) -> bool:
        return False


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    step_count: int
    rank_within_step_count: int
    input_payload: Dict[str, Any]
    expected: Dict[str, Any]


@dataclass(frozen=True)
class ComparisonResult:
    mode: str
    score: float
    passed: bool
    final_product_reached: bool
    accepted_path_step_count: int
    known_step_count: int
    latency_ms: float
    total_cost: float
    error: Optional[str]
    run_id: Optional[str] = None
    mechanism_type: Optional[str] = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the full harness against the one-shot baseline on holdout samples."
    )
    parser.add_argument(
        "--model-name",
        "--model",
        dest="model_name",
        required=True,
        help="Model identifier, for example gpt-4o-mini or claude-sonnet-4.6.",
    )
    parser.add_argument(
        "--thinking-level",
        choices=("low", "high"),
        default=None,
        help="Optional reasoning level for models that support it.",
    )
    parser.add_argument(
        "--harness",
        default="default",
        help="Harness name from harness_versions/ for the full-pipeline run.",
    )
    parser.add_argument(
        "--min-step",
        type=int,
        default=1,
        help="Lowest mechanism step bucket to include.",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=6,
        help="Highest mechanism step bucket to include.",
    )
    parser.add_argument(
        "--bucket-rank",
        type=int,
        default=1,
        help="Use the Nth case inside each step bucket. Default is the first case.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Harness max_steps setting for each run.",
    )
    parser.add_argument(
        "--max-runtime",
        type=float,
        default=300.0,
        help="Harness per-case timeout in seconds.",
    )
    parser.add_argument(
        "--baseline-timeout",
        type=float,
        default=180.0,
        help="Per-case timeout in seconds for the one-shot baseline.",
    )
    parser.add_argument(
        "--show-rdkit-warnings",
        action="store_true",
        help="Leave RDKit logging enabled. Default behavior suppresses noisy parser warnings.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only print which cases would run. Do not call any model.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the final comparison payload as JSON after the text summary.",
    )
    return parser.parse_args()


def _configure_rdkit_logging(show_warnings: bool) -> None:
    if show_warnings:
        return
    try:
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except Exception:
        return


def _load_cases(base: Path, min_step: int, max_step: int, bucket_rank: int) -> List[EvalCase]:
    from mechanistic_agent.data_paths import holdout_dir

    holdout_root = holdout_dir(base)
    holdout_path = holdout_root / "eval_set_holdout.json"
    bucket_path = holdout_root / "eval_step_buckets_holdout.json"

    raw_cases = json.loads(holdout_path.read_text(encoding="utf-8"))
    raw_buckets = json.loads(bucket_path.read_text(encoding="utf-8"))
    case_by_id = {str(item.get("id") or ""): item for item in raw_cases if isinstance(item, Mapping)}
    step_buckets = raw_buckets.get("step_buckets") or {}

    selected: List[EvalCase] = []
    for step_count in range(min_step, max_step + 1):
        bucket = step_buckets.get(str(step_count))
        if not isinstance(bucket, Mapping):
            raise SystemExit(f"No holdout bucket found for step count {step_count}")

        case_ids = list(bucket.get("case_ids") or [])
        if bucket_rank < 1 or bucket_rank > len(case_ids):
            raise SystemExit(
                f"Bucket rank {bucket_rank} is out of range for step count {step_count} "
                f"(available: 1..{len(case_ids)})"
            )

        case_id = str(case_ids[bucket_rank - 1])
        entry = case_by_id.get(case_id)
        if not isinstance(entry, Mapping):
            raise SystemExit(f"Selected case '{case_id}' was not found in {holdout_path}")

        selected.append(_convert_case(entry, step_count, bucket_rank))
    return selected


def _convert_case(entry: Mapping[str, Any], step_count: int, bucket_rank: int) -> EvalCase:
    products = list(entry.get("products") or [])
    verified = entry.get("verified_mechanism")
    known = entry.get("known_mechanism")
    raw_temperature = entry.get("temperature_celsius")
    temperature_celsius = float(raw_temperature) if isinstance(raw_temperature, (int, float)) else 25.0
    input_payload = {
        "starting_materials": list(entry.get("starting_materials") or []),
        "products": products,
        "temperature_celsius": temperature_celsius,
        "ph": entry.get("ph"),
        "n_mechanistic_steps": step_count,
    }
    expected: Dict[str, Any] = {
        "products": products,
        "n_mechanistic_steps": step_count,
    }
    if isinstance(known, Mapping):
        expected["known_mechanism"] = dict(known)
    if isinstance(verified, Mapping):
        expected["verified_mechanism"] = dict(verified)
    return EvalCase(
        case_id=str(entry.get("id") or ""),
        step_count=step_count,
        rank_within_step_count=bucket_rank,
        input_payload=input_payload,
        expected=expected,
    )


def _extract_harness_error(snapshot: Mapping[str, Any]) -> Optional[str]:
    events = list(snapshot.get("events") or [])
    llm_error: Optional[str] = None
    run_failure: Optional[str] = None

    for event in reversed(events):
        event_type = str(event.get("event_type") or "")
        payload = event.get("payload") or {}
        if event_type == "run_failed" and not run_failure:
            reason = str(payload.get("reason") or "").strip()
            invalid_detail = str(
                ((payload.get("proposal_quality_summary") or {}).get("first_invalid_detail")) or ""
            ).strip()
            parts = [part for part in (reason, invalid_detail) if part]
            run_failure = "; ".join(parts) if parts else "run_failed"
        if event_type == "step_output" and str(event.get("step_name") or "") == "mechanism_step_proposal":
            output = payload.get("output") or {}
            message = str(output.get("error") or output.get("message") or "").strip()
            if message:
                llm_error = message
                break

    if run_failure and llm_error:
        return f"{run_failure}; {llm_error}"
    return run_failure or llm_error


def _run_harness_case(
    *,
    store: RunStore,
    registry: RegistrySet,
    coordinator: RunCoordinator,
    case: EvalCase,
    model_name: str,
    thinking_level: Optional[str],
    harness: str,
    max_steps: int,
    max_runtime: float,
) -> ComparisonResult:
    model_family = get_model_family(model_name) or "unknown"
    internal_reasoning = to_internal_reasoning_level(thinking_level)
    hashes = registry.bundle_hashes(model_name=model_name)
    step_models = {step_name: model_name for step_name in sorted(LLM_STEP_KEYS)}
    step_models["mechanism_synthesis"] = model_name
    step_reasoning = (
        {step_name: internal_reasoning for step_name in step_models}
        if internal_reasoning
        else {}
    )

    run_id = store.create_run(
        mode="unverified",
        input_payload=dict(case.input_payload),
        config={
            "model": model_name,
            "model_name": model_name,
            "model_family": model_family,
            "thinking_level": thinking_level,
            "reasoning_level": internal_reasoning,
            "step_models": step_models,
            "step_reasoning": step_reasoning,
            "optional_llm_tools": ["attempt_atom_mapping", "predict_missing_reagents"],
            "functional_groups_enabled": True,
            "intermediate_prediction_enabled": True,
            "max_steps": max_steps,
            "max_runtime_seconds": max_runtime,
            "harness_name": harness,
        },
        **hashes,
    )

    started_at = time.perf_counter()
    try:
        coordinator.execute_run(run_id, threading.Event())
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        snapshot = store.get_run_snapshot(run_id) or {}
        graded = score_snapshot_against_known(snapshot, case.expected)
        total_cost = float(
            (((snapshot.get("cost_summary") or {}).get("total_cost") or {}).get("total_cost") or 0.0)
        )
        error_text = None
        if str(snapshot.get("status") or "") != "completed":
            error_text = _extract_harness_error(snapshot)
        return ComparisonResult(
            mode="harness",
            score=float(graded.get("score") or 0.0),
            passed=bool(graded.get("passed")),
            final_product_reached=bool(graded.get("final_product_reached")),
            accepted_path_step_count=int(graded.get("accepted_path_step_count") or 0),
            known_step_count=int(graded.get("known_step_count") or 0),
            latency_ms=latency_ms,
            total_cost=total_cost,
            error=error_text,
            run_id=run_id,
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        return ComparisonResult(
            mode="harness",
            score=0.0,
            passed=False,
            final_product_reached=False,
            accepted_path_step_count=0,
            known_step_count=int(case.step_count),
            latency_ms=latency_ms,
            total_cost=0.0,
            error=str(exc),
            run_id=run_id,
        )


def _run_baseline_case(
    *,
    case: EvalCase,
    model_name: str,
    thinking_level: Optional[str],
    baseline_timeout: float,
) -> ComparisonResult:
    runner = BaselineRunner()
    result = runner.run_case(
        starting_materials=list(case.input_payload.get("starting_materials") or []),
        products=list(case.input_payload.get("products") or []),
        model=model_name,
        thinking_level=thinking_level,
        temperature_celsius=float(case.input_payload.get("temperature_celsius") or 25.0),
        ph=case.input_payload.get("ph"),
        timeout=baseline_timeout,
    )
    graded = score_baseline_result(result, case.expected)

    usage = result.get("token_usage")
    total_cost = 0.0
    if isinstance(usage, Mapping):
        try:
            normalised = normalise_token_usage(usage)
            total_cost = float(calculate_cost(model_name, normalised).get("total_cost") or 0.0)
        except Exception:
            total_cost = 0.0

    scoring_breakdown = graded.get("scoring_breakdown") or {}
    return ComparisonResult(
        mode="baseline",
        score=float(graded.get("score") or 0.0),
        passed=bool(graded.get("passed")),
        final_product_reached=bool(scoring_breakdown.get("final_product_reached")),
        accepted_path_step_count=int(scoring_breakdown.get("accepted_path_step_count") or 0),
        known_step_count=int(scoring_breakdown.get("known_step_count") or 0),
        latency_ms=float(result.get("latency_ms") or 0.0),
        total_cost=total_cost,
        error=graded.get("error"),
        mechanism_type=str(graded.get("mechanism_type") or "") or None,
    )


def _format_bool(value: bool) -> str:
    return "yes" if value else "no"


def _format_seconds(latency_ms: float) -> str:
    return f"{latency_ms / 1000.0:.1f}s"


def _winner_label(harness: ComparisonResult, baseline: ComparisonResult) -> str:
    delta = baseline.score - harness.score
    if abs(delta) < 1e-9:
        return "tie"
    return "baseline" if delta > 0 else "harness"


def _print_case_selection(cases: Iterable[EvalCase]) -> None:
    print("Selected holdout samples:")
    for case in cases:
        print(f"  step={case.step_count} rank={case.rank_within_step_count} case_id={case.case_id}")


def _print_case_result(case: EvalCase, harness: ComparisonResult, baseline: ComparisonResult) -> None:
    print("")
    print(f"[step {case.step_count}] {case.case_id}")
    print(
        "  Harness : "
        f"score={harness.score:.3f} pass={_format_bool(harness.passed)} "
        f"final={_format_bool(harness.final_product_reached)} "
        f"path={harness.accepted_path_step_count}/{harness.known_step_count} "
        f"cost=${harness.total_cost:.3f} latency={_format_seconds(harness.latency_ms)} "
        f"run_id={harness.run_id or 'n/a'}"
    )
    if harness.error:
        print(f"             error={harness.error}")
    print(
        "  Baseline: "
        f"score={baseline.score:.3f} pass={_format_bool(baseline.passed)} "
        f"final={_format_bool(baseline.final_product_reached)} "
        f"path={baseline.accepted_path_step_count}/{baseline.known_step_count} "
        f"cost=${baseline.total_cost:.3f} latency={_format_seconds(baseline.latency_ms)}"
    )
    if baseline.mechanism_type:
        print(f"             mechanism_type={baseline.mechanism_type}")
    if baseline.error:
        print(f"             error={baseline.error}")

    score_delta = baseline.score - harness.score
    cost_delta = baseline.total_cost - harness.total_cost
    latency_delta_s = (baseline.latency_ms - harness.latency_ms) / 1000.0
    print(
        "  Delta   : "
        f"score={score_delta:+.3f} cost=${cost_delta:+.3f} latency={latency_delta_s:+.1f}s "
        f"winner={_winner_label(harness, baseline)}"
    )


def _aggregate(results: List[ComparisonResult]) -> Dict[str, float]:
    count = float(len(results) or 1)
    return {
        "avg_score": sum(item.score for item in results) / count,
        "passes": float(sum(1 for item in results if item.passed)),
        "final_hits": float(sum(1 for item in results if item.final_product_reached)),
        "total_cost": sum(item.total_cost for item in results),
        "total_latency_ms": sum(item.latency_ms for item in results),
    }


def _build_json_payload(
    *,
    model_name: str,
    thinking_level: Optional[str],
    harness_name: str,
    cases: List[EvalCase],
    rows: List[Dict[str, Any]],
    harness_summary: Dict[str, float],
    baseline_summary: Dict[str, float],
) -> Dict[str, Any]:
    return {
        "model_name": model_name,
        "thinking_level": thinking_level,
        "harness": harness_name,
        "selected_cases": [
            {
                "case_id": case.case_id,
                "step_count": case.step_count,
                "rank_within_step_count": case.rank_within_step_count,
            }
            for case in cases
        ],
        "rows": rows,
        "summary": {
            "harness": harness_summary,
            "baseline": baseline_summary,
            "baseline_better_cases": sum(
                1 for row in rows if float(row["baseline"]["score"]) > float(row["harness"]["score"])
            ),
            "harness_better_cases": sum(
                1 for row in rows if float(row["harness"]["score"]) > float(row["baseline"]["score"])
            ),
            "tied_cases": sum(
                1
                for row in rows
                if abs(float(row["baseline"]["score"]) - float(row["harness"]["score"])) < 1e-9
            ),
        },
    }


def main() -> None:
    args = _parse_args()
    if args.min_step < 1 or args.max_step < args.min_step:
        raise SystemExit("Step range must satisfy 1 <= min-step <= max-step")
    if args.bucket_rank < 1:
        raise SystemExit("--bucket-rank must be at least 1")

    base = REPO_ROOT
    load_dotenv(base / ".env")
    _configure_rdkit_logging(args.show_rdkit_warnings)

    requested_model_name = args.model_name
    resolved_model_name = resolve_model_key(requested_model_name)
    model_name = requested_model_name
    cases = _load_cases(base, args.min_step, args.max_step, args.bucket_rank)

    if requested_model_name != resolved_model_name:
        print(
            f"Warning: requested model '{requested_model_name}' resolves to catalog model '{resolved_model_name}'."
        )
        print(
            "         This script will still call the exact requested model name; "
            "pricing totals may be incomplete if that model is not in model_pricing.json."
        )
    print(
        f"Comparing harness vs one-shot baseline on official holdout samples "
        f"for model={model_name} harness={args.harness}"
    )
    _print_case_selection(cases)

    if args.list_only:
        return

    store = RunStore(base / "data" / "mechanistic.db")
    registry = RegistrySet(base)
    coordinator = RunCoordinator(store)

    harness_results: List[ComparisonResult] = []
    baseline_results: List[ComparisonResult] = []
    rows: List[Dict[str, Any]] = []

    for case in cases:
        harness_result = _run_harness_case(
            store=store,
            registry=registry,
            coordinator=coordinator,
            case=case,
            model_name=model_name,
            thinking_level=args.thinking_level,
            harness=args.harness,
            max_steps=args.max_steps,
            max_runtime=args.max_runtime,
        )
        baseline_result = _run_baseline_case(
            case=case,
            model_name=model_name,
            thinking_level=args.thinking_level,
            baseline_timeout=args.baseline_timeout,
        )

        harness_results.append(harness_result)
        baseline_results.append(baseline_result)
        rows.append(
            {
                "case_id": case.case_id,
                "step_count": case.step_count,
                "rank_within_step_count": case.rank_within_step_count,
                "harness": harness_result.__dict__,
                "baseline": baseline_result.__dict__,
            }
        )
        _print_case_result(case, harness_result, baseline_result)

    harness_summary = _aggregate(harness_results)
    baseline_summary = _aggregate(baseline_results)

    print("")
    print("Aggregate summary:")
    print(
        "  Harness : "
        f"avg_score={harness_summary['avg_score']:.3f} "
        f"passes={int(harness_summary['passes'])}/{len(harness_results)} "
        f"final_hits={int(harness_summary['final_hits'])}/{len(harness_results)} "
        f"cost=${harness_summary['total_cost']:.3f} "
        f"latency={harness_summary['total_latency_ms'] / 1000.0:.1f}s"
    )
    print(
        "  Baseline: "
        f"avg_score={baseline_summary['avg_score']:.3f} "
        f"passes={int(baseline_summary['passes'])}/{len(baseline_results)} "
        f"final_hits={int(baseline_summary['final_hits'])}/{len(baseline_results)} "
        f"cost=${baseline_summary['total_cost']:.3f} "
        f"latency={baseline_summary['total_latency_ms'] / 1000.0:.1f}s"
    )
    print(
        "  Outcome : "
        f"score_delta={baseline_summary['avg_score'] - harness_summary['avg_score']:+.3f} "
        f"cost_delta=${baseline_summary['total_cost'] - harness_summary['total_cost']:+.3f} "
        f"latency_delta={(baseline_summary['total_latency_ms'] - harness_summary['total_latency_ms']) / 1000.0:+.1f}s"
    )

    if args.json:
        payload = _build_json_payload(
            model_name=model_name,
            thinking_level=args.thinking_level,
            harness_name=args.harness,
            cases=cases,
            rows=rows,
            harness_summary=harness_summary,
            baseline_summary=baseline_summary,
        )
        print("")
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
