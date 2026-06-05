#!/usr/bin/env python3
"""Harness evolution against the deterministic FlowER curriculum."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(_: Path) -> bool:  # type: ignore[override]
        return False

load_dotenv(Path.cwd() / ".env")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mechanistic_agent.core import RegistrySet, RunCoordinator, RunStore, select_step_models
from mechanistic_agent.data_paths import db_path as resolve_db_path, flower_curriculum_pngs_dir, flower_train_lookup_path
from mechanistic_agent.core.archive import (
    DEFAULT_ISLANDS,
    ISLAND_CALL_NAMES,
    EvolutionArchive,
)
from mechanistic_agent.core.lane_mutator import (
    FewShotLaneMutator,
    HarnessLaneMutator,
    PromptLaneMutator,
    TopologyLaneMutator,
)
from mechanistic_agent.core.types import ArchiveEntry, IslandConfig, IslandEvolutionConfig
from mechanistic_agent.flower_curriculum import (
    DEFAULT_FLOWER_INPUT,
    DEFAULT_INDEX_PATH,
    DEFAULT_INDEX_REPORT_PATH,
    DEFAULT_LOOKUP_CACHE,
    SOURCE_LABEL,
    build_lookup_cache,
    curriculum_history,
    ensure_index,
    eval_case_from_case,
    known_mechanism_from_case,
    load_curriculum_index,
    next_curriculum_candidates,
    convert_mechanism_id_to_case,
)
from mechanistic_agent.flower_rendering import render_curriculum_pngs
from mechanistic_agent.model_registry import (
    get_model_family,
    get_reasoning_levels,
    to_internal_reasoning_level,
    to_public_reasoning_level,
)
from mechanistic_agent.prompt_assets import (
    STEP_TO_CALL_NAME,
    append_call_few_shot_example,
    best_few_shot_score,
    load_call_few_shot_examples,
    score_few_shot_example,
)
from mechanistic_agent.scoring import score_snapshot_against_known, score_subagents_from_step_outputs
from mechanistic_agent.smiles_utils import strip_atom_mapping_list


MINEABLE_SUBAGENTS: Dict[str, str] = {
    "initial_conditions": "assess_initial_conditions",
    "missing_reagents": "predict_missing_reagents",
    "atom_mapping": "attempt_atom_mapping",
    "reaction_type_mapping": "select_reaction_type",
    "mechanism_step_proposal": "propose_mechanism_step",
    "mechanism_synthesis": "propose_mechanism_step",
}


def _resolve_thinking_level(model_name: str, user_level: Optional[str]) -> Optional[str]:
    """Return thinking level for model; if user_level is None/auto, use highest available."""
    if user_level and str(user_level).strip().lower() not in ("auto", "none", ""):
        level = str(user_level).strip().lower()
        return level if level in ("low", "high", "max") else level
    levels = get_reasoning_levels(model_name)
    if not levels:
        return None
    for key in ("highest", "high", "max", "xhigh"):
        if key in levels:
            return to_public_reasoning_level(key) or "high"
    last = levels[-1] if levels else None
    return to_public_reasoning_level(last) if last else None


@dataclass
class EvolutionConfig:
    model_name: str
    harness: str = "default"
    eval_set_id: str = "eval_set"
    group_size: int = 4
    min_score_threshold: float = 0.2
    mining_score_threshold: float = 0.5
    max_few_shots_per_step: int = 3
    dry_run: bool = False
    max_steps: int = 10
    max_runtime: float = 600.0
    retry_same_candidate_max: int = 1
    repeat_failure_signature_limit: int = 2
    max_reproposals_per_step: int = 4
    seed: int = 42
    step_pass_target: int = 20
    coordination_topology: str = "centralized_mas"
    thinking_level: Optional[str] = None
    step_count_filter: Optional[int] = None
    step_count_filter_list: List[int] = field(default_factory=list)
    loop_when_done: bool = False
    train_input: Path = DEFAULT_FLOWER_INPUT
    curriculum_index_path: Path = DEFAULT_INDEX_PATH
    curriculum_index_report_path: Path = DEFAULT_INDEX_REPORT_PATH
    lookup_cache_path: Path = DEFAULT_LOOKUP_CACHE


def _get_api_keys_from_env() -> Dict[str, str]:
    keys: Dict[str, str] = {}
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        keys["openai"] = openai_key
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        keys["openrouter"] = openrouter_key
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        keys["anthropic"] = anthropic_key
    for gemini_var in ["GOOGLE_API_KEY", "GEMINI_API_KEY", "VERTEX_API_KEY"]:
        gemini_key = os.getenv(gemini_var)
        if gemini_key:
            keys["gemini"] = gemini_key
            break
    return keys


def resolve_eval_set_id(store: RunStore, identifier: str) -> str:
    all_sets = store.list_eval_sets()
    for eval_set in all_sets:
        if eval_set.get("id") == identifier:
            return identifier

    identifier_lower = str(identifier or "").lower()
    for eval_set in all_sets:
        name = str(eval_set.get("name", "")).lower()
        if name == identifier_lower:
            return str(eval_set.get("id", ""))

    alias_map = {
        "eval_set": "flower_100_default",
        "default": "flower_100_default",
        "flower": "flower_100_default",
        "flower_100": "flower_100_default",
        "flower_100_default": "flower_100_default",
    }
    alias_target = alias_map.get(identifier_lower)
    if alias_target:
        for eval_set in all_sets:
            if str(eval_set.get("name", "")).lower() == alias_target:
                return str(eval_set.get("id", ""))

    available = [f"  - {s.get('name')} (id: {s.get('id')})" for s in all_sets]
    raise ValueError(
        f"Eval set '{identifier}' not found.\n"
        f"Available eval sets:\n" + "\n".join(available) if available else "  (none)"
    )


def ensure_default_flower_eval_set(
    store: RunStore,
    *,
    base_dir: Path,
    version: str = "flower100_v1",
) -> Optional[str]:
    for item in store.list_eval_sets():
        if str(item.get("name") or "") == "flower_100_default":
            eval_set_id = str(item.get("id") or "")
            if eval_set_id:
                return eval_set_id

    eval_path = base_dir / "training_data" / "eval_set.json"
    if not eval_path.exists():
        return None

    raw = json.loads(eval_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return None

    cases: List[Dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        case_id = str(entry.get("id") or "")
        starting = entry.get("starting_materials") or []
        products = entry.get("products") or []
        if not case_id or not isinstance(starting, list) or not isinstance(products, list):
            continue
        cases.append(
            {
                "case_id": case_id,
                "input": {
                    "starting_materials": starting,
                    "products": products,
                    "temperature_celsius": entry.get("temperature_celsius", 25.0),
                    "ph": entry.get("ph"),
                },
                "expected": {
                    "products": products,
                    **({"known_mechanism": entry["known_mechanism"]} if isinstance(entry.get("known_mechanism"), dict) else {}),
                    **({"verified_mechanism": entry["verified_mechanism"]} if isinstance(entry.get("verified_mechanism"), dict) else {}),
                },
                "tags": ["flower_100", "default_eval"],
            }
        )

    if not cases:
        return None

    return store.add_eval_set(
        name="flower_100_default",
        version=version,
        source_path=str(eval_path),
        sha256=None,
        cases=cases,
        active=True,
    )


def setup_workspace(base_dir: Path, dry_run: bool) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    workspace = base_dir / "evolution_workspace" / timestamp
    workspace.mkdir(parents=True, exist_ok=True)

    if dry_run:
        for root_name in ["skills", "training_data", "harness_versions"]:
            source = base_dir / root_name
            if source.exists():
                shutil.copytree(source, workspace / root_name, dirs_exist_ok=True)
        source_db = resolve_db_path(base_dir)
        workspace_db = workspace / "data" / "mechanistic.db"
        workspace_db.parent.mkdir(parents=True, exist_ok=True)
        if source_db.exists():
            shutil.copy2(source_db, workspace_db)
        source_lookup = flower_train_lookup_path(base_dir)
        if source_lookup.exists():
            shutil.copy2(source_lookup, workspace / "data" / "flower_train_lookup.sqlite")
        (workspace / "traces" / "runs").mkdir(parents=True, exist_ok=True)
        (workspace / "traces" / "evidence").mkdir(parents=True, exist_ok=True)

    return workspace


@contextlib.contextmanager
def pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def get_run_failure_error(store: RunStore, run_id: str) -> str | None:
    try:
        events = store.list_events(run_id)
        for event in events:
            if event.get("event_type") == "run_failed":
                payload = event.get("payload", {})
                return payload.get("error") or payload.get("reason")
    except Exception:
        pass
    return None


def extract_validation_errors(step_outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    errors: List[Dict[str, Any]] = []
    for step in step_outputs:
        validation = step.get("validation", {})
        if not (isinstance(validation, dict) and validation.get("passed") is False):
            continue
        attempt = step.get("attempt")
        retry_index = step.get("retry_index")
        derived_step_index = step.get("step_index")
        if derived_step_index is None and isinstance(attempt, int):
            derived_step_index = attempt
        checks = validation.get("checks")
        if isinstance(checks, list) and checks:
            for check in checks:
                if not isinstance(check, dict):
                    continue
                if check.get("passed") is True:
                    continue
                details = check.get("details")
                if not isinstance(details, dict):
                    details = {}
                errors.append({
                    "step_name": step.get("step_name"),
                    "step_index": derived_step_index,
                    "attempt": attempt,
                    "retry_index": retry_index,
                    "check_name": check.get("name"),
                    "validation_error": details.get("error") or details.get("message"),
                    "validation_details": details,
                })
            continue
        errors.append({
            "step_name": step.get("step_name"),
            "step_index": derived_step_index,
            "attempt": attempt,
            "retry_index": retry_index,
            "check_name": None,
            "validation_error": validation.get("error"),
            "validation_details": validation.get("details"),
        })
    return errors


def _diagnostic_step_output(
    step_out: Dict[str, Any],
    input_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a sanitized step entry for run diagnostics (input summary, output, validation)."""
    step_name = step_out.get("step_name", "")
    output = step_out.get("output") or {}
    if not isinstance(output, dict):
        output = {}
    input_text = _reconstruct_input_text(
        step_name,
        input_payload,
        output,
        step_out,
    )
    sanitized_output = dict(output)
    for key in ("token_usage", "usage"):
        if key in sanitized_output and isinstance(sanitized_output[key], dict):
            sanitized_output[key] = {k: v for k, v in sanitized_output[key].items() if k in ("total_tokens", "prompt_tokens", "completion_tokens", "input_tokens", "output_tokens")}
    return {
        "step_name": step_name,
        "input": input_text,
        "output": sanitized_output,
        "validation": step_out.get("validation"),
        "accepted_bool": step_out.get("accepted_bool"),
    }


def save_run_diagnostics(
    workspace: Path,
    batch_name: str,
    case_results: List[Dict[str, Any]],
    dry_run: bool = True,
    batch_start_rank: Optional[int] = None,
) -> Path:
    diagnostics: Dict[str, Any] = {
        "batch": batch_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dry_run": dry_run,
        "attempted": len(case_results),
        "pass_count": sum(1 for r in case_results if r.get("passed")),
        "cases": [],
    }
    if batch_start_rank is not None:
        diagnostics["batch_start_rank"] = batch_start_rank

    input_payload_default: Dict[str, Any] = {}
    for result in case_results:
        input_payload = result.get("input_payload") or input_payload_default
        step_outputs_raw = result.get("step_outputs", [])
        step_outputs_diag = [
            _diagnostic_step_output(step_out, input_payload)
            for step_out in step_outputs_raw
        ]
        case_diag = {
            "case_id": result.get("case_id"),
            "passed": result.get("passed"),
            "score": result.get("score"),
            "run_status": result.get("run_status"),
            "run_error": result.get("run_error"),
            "validation_errors": extract_validation_errors(step_outputs_raw),
            "current_state": result.get("current_state"),
            "step_outputs": step_outputs_diag,
        }
        diagnostics["cases"].append(case_diag)

    diag_path = workspace / f"run_diagnostics_{batch_name}_{time.strftime('%H%M%S')}.json"
    diag_path.write_text(json.dumps(diagnostics, indent=2, default=str))
    return diag_path


def analyze_subagent_performance(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    from collections import defaultdict

    aggregated: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"quality_scores": [], "pass_rates": [], "calls": 0}
    )

    for result in results:
        for sa_name, sa_data in result.get("subagent_scores", {}).items():
            agg = aggregated[sa_name]
            agg["quality_scores"].append(sa_data.get("quality_score", 0.0))
            agg["pass_rates"].append(sa_data.get("pass_rate", 0.0))
            agg["calls"] += sa_data.get("calls", 0)

    summary: Dict[str, Any] = {}
    failing: List[str] = []
    for sa_name, agg in aggregated.items():
        qs = agg["quality_scores"]
        pr = agg["pass_rates"]
        mean_q = sum(qs) / len(qs) if qs else 0.0
        mean_p = sum(pr) / len(pr) if pr else 0.0
        summary[sa_name] = {
            "mean_quality_score": round(mean_q, 4),
            "mean_pass_rate": round(mean_p, 4),
            "total_calls": agg["calls"],
            "case_count": len(qs),
        }
        if mean_q < 0.3 and agg["calls"] > 0:
            failing.append(sa_name)

    return {"subagents": summary, "consistently_failing": failing}


def _reconstruct_input_text(
    step_name: str,
    input_payload: Dict[str, Any],
    output: Dict[str, Any],
    step_out: Dict[str, Any],
) -> str:
    sm = strip_atom_mapping_list(input_payload.get("starting_materials", []))
    prods = strip_atom_mapping_list(input_payload.get("products", []))
    if step_name == "initial_conditions":
        lines = [f"Starting materials: {sm}", f"Products: {prods}"]
        temp = input_payload.get("temperature_celsius")
        if temp is not None:
            lines.append(f"Temperature: {temp}C")
        ph = input_payload.get("ph")
        if ph is not None:
            lines.append(f"pH: {ph}")
        return "\n".join(lines)
    if step_name in ("missing_reagents", "atom_mapping", "reaction_type_mapping"):
        return f"Starting materials: {sm}\nProducts: {prods}"
    if step_name in ("mechanism_step_proposal", "mechanism_synthesis"):
        current_state = output.get("current_state", sm)
        attempt = step_out.get("attempt", 0)
        return f"Current state: {current_state}\nTarget products: {prods}\nStep index: {attempt}"
    return f"Starting materials: {sm}\nProducts: {prods}"


def mine_few_shots(
    results: List[Dict[str, Any]],
    config: EvolutionConfig,
    existing_hashes: Dict[str, set],
    best_scores_by_call: Dict[str, float],
) -> Dict[str, List[Dict[str, Any]]]:
    mined: Dict[str, List[Dict[str, Any]]] = {}
    for result in results:
        if result.get("passed") is not True:
            continue
        if result.get("run_status") != "completed":
            continue
        graded_details = result.get("graded_details") or {}
        if not bool(graded_details.get("final_product_reached")):
            continue
        if result.get("score", 0.0) < config.mining_score_threshold:
            continue
        if result.get("error"):
            continue
        step_outputs = result.get("step_outputs", [])
        input_payload = result.get("input_payload", {})
        for step_out in step_outputs:
            step_name = step_out.get("step_name", "")
            if step_name not in MINEABLE_SUBAGENTS:
                continue
            call_name = MINEABLE_SUBAGENTS[step_name]
            validation = step_out.get("validation")
            if isinstance(validation, dict) and validation.get("passed") is False:
                continue
            if step_out.get("accepted_bool") is False:
                continue
            output = step_out.get("output", {})
            if not isinstance(output, dict) or not output:
                continue
            if output.get("status") == "failed" or output.get("error"):
                continue
            schema_validation = output.get("schema_validation")
            if not isinstance(schema_validation, dict) or schema_validation.get("status") != "ok":
                continue
            if schema_validation.get("source") not in {"tool_call", "text_json"}:
                continue

            input_text = _reconstruct_input_text(step_name, input_payload, output, step_out)
            output_text = json.dumps(output, indent=2, sort_keys=True)
            example_score = score_few_shot_example(call_name, input_text=input_text, output_text=output_text)
            output_hash = hashlib.sha256(output_text.encode()).hexdigest()[:16]
            # No filter by duplicate hash, best score, or max_few_shots_per_step — mine all eligible steps
            mined.setdefault(call_name, []).append({
                "input": input_text,
                "output": output_text,
                "score": round(example_score, 4),
                "example_key": output_hash,
            })
            existing_hashes.setdefault(call_name, set()).add(output_hash)
            best_scores_by_call[call_name] = max(best_scores_by_call.get(call_name, 0.0), example_score)
    return mined


def apply_mined_examples(
    mined: Dict[str, List[Dict[str, Any]]],
    store: RunStore,
    base_dir: Path,
    workspace: Path,
    dry_run: bool,
    batch_name: str,
    case_scores: Dict[str, float],
    model_name: str | None = None,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    target_base = workspace if dry_run else base_dir
    for call_name, examples in mined.items():
        for ex in examples:
            append_call_few_shot_example(
                call_name,
                input_text=ex["input"],
                output_text=ex["output"],
                score=float(ex["score"]),
                example_key=str(ex["example_key"]),
                base_dir=target_base,
                model_name=model_name,
            )
            step_name = next((sn for sn, cn in STEP_TO_CALL_NAME.items() if cn == call_name), None)
            if step_name:
                example_key = str(ex.get("example_key") or f"{batch_name}_{uuid.uuid4().hex[:8]}")
                try:
                    store.add_few_shot_example(
                        step_name=step_name,
                        example_key=example_key,
                        input_text=ex["input"],
                        output_text=ex["output"],
                        approved=not dry_run,
                        source_trace_id=None,
                        score=float(ex["score"]) if ex.get("score") is not None else (max(case_scores.values()) if case_scores else None),
                    )
                except Exception:
                    pass
        counts[call_name] = len(examples)
    return counts


def _curriculum_summary(
    entry: Dict[str, Any],
    *,
    config: EvolutionConfig,
    batch_start_rank: int,
    current_step_count: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    summary = {
        "curriculum_case_id": str(entry["case_id"]),
        "step_count": int(entry["step_count"]),
        "global_rank": int(entry["global_rank"]),
        "row": int(entry["global_rank"]),
        "rank_within_step_count": int(entry["rank_within_step_count"]),
        "group_size": int(config.group_size),
        "step_pass_target": int(config.step_pass_target),
        "curriculum_index_path": str(config.curriculum_index_path),
        "curriculum_harness": str(config.harness),
        "batch_start_rank": int(batch_start_rank),
        "current_step_count": int(current_step_count),
    }
    if extra:
        summary.update(extra)
    return summary


def prepare_curriculum_batch(
    *,
    config: EvolutionConfig,
    candidate_entries: Sequence[Dict[str, Any]],
    current_step_count: int,
) -> Dict[str, Any]:
    runnable: List[Dict[str, Any]] = []
    conversion_failures: List[Dict[str, Any]] = []
    batch_start_rank = int(candidate_entries[0]["global_rank"]) if candidate_entries else 0
    for entry in candidate_entries:
        entry_dict = dict(entry)
        try:
            case = convert_mechanism_id_to_case(
                int(entry_dict["mechanism_id"]),
                input_path=config.train_input,
                cache_path=config.lookup_cache_path,
            )
            eval_case = {
                "case_id": str(case["id"]),
                "input": {
                    "starting_materials": list(case.get("starting_materials") or []),
                    "products": list(case.get("products") or []),
                    "temperature_celsius": case.get("temperature_celsius", 25.0),
                    "ph": case.get("ph"),
                },
                "expected": {
                    "products": list(case.get("products") or []),
                    "known_mechanism": known_mechanism_from_case(case),
                    "verified_mechanism": case.get("verified_mechanism"),
                },
                "tags": list(case.get("tags") or []) + ["flower_curriculum"],
                "entry": entry_dict,
                "case": case,
            }
            runnable.append(eval_case)
            if len(runnable) >= config.group_size:
                break
        except Exception as exc:
            conversion_failures.append(
                {
                    "entry": entry_dict,
                    "error": str(exc),
                    "summary": _curriculum_summary(
                        entry_dict,
                        config=config,
                        batch_start_rank=batch_start_rank,
                        current_step_count=current_step_count,
                        extra={"error": str(exc), "run_status": "conversion_failed"},
                    ),
                }
            )

    return {
        "runnable_cases": runnable,
        "conversion_failures": conversion_failures,
        "batch_start_rank": batch_start_rank,
        "current_step_count": int(current_step_count),
    }


def run_curriculum_batch(
    config: EvolutionConfig,
    *,
    prepared_batch: Dict[str, Any],
    store: RunStore,
    base_dir: Path,
    existing_hashes: Optional[Dict[str, set]] = None,
    best_scores_by_call: Optional[Dict[str, float]] = None,
    workspace: Optional[Path] = None,
    evolution_log_path: Optional[Path] = None,
    selection_meta: Optional[Dict[str, Any]] = None,
    progress: Optional[Dict[str, Any]] = None,
    accumulated_examples_mined: Optional[Dict[str, int]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    registry = RegistrySet(base_dir)
    model_family = get_model_family(config.model_name) or "unknown"
    internal_reasoning = to_internal_reasoning_level(config.thinking_level)
    hashes = registry.bundle_hashes()
    runnable_cases = list(prepared_batch["runnable_cases"])
    conversion_failures = list(prepared_batch["conversion_failures"])
    batch_start_rank = int(prepared_batch["batch_start_rank"])
    current_step_count = int(prepared_batch["current_step_count"])
    step_count = int(runnable_cases[0]["entry"]["step_count"]) if runnable_cases else int(conversion_failures[0]["entry"]["step_count"])
    run_group_name = f"curriculum_{config.harness}_s{step_count}_r{batch_start_rank}_n{config.group_size}"

    eval_run_id = store.create_eval_run(
        eval_set_id=config.eval_set_id,
        run_group_name=run_group_name,
        model=config.model_name,
        model_name=config.model_name,
        model_family=model_family,
        thinking_level=config.thinking_level,
        harness_bundle_hash=hashes.get("prompt_bundle_hash", ""),
        status="running",
    )

    results: List[Dict[str, Any]] = []
    for failed in conversion_failures:
        entry = failed["entry"]
        summary = dict(failed["summary"])
        store.record_eval_run_result(
            eval_run_id=eval_run_id,
            case_id=str(entry["case_id"]),
            run_id=None,
            score=0.0,
            passed=False,
            cost={"total_cost": 0.0},
            latency_ms=0.0,
            summary=summary,
        )
        results.append({
            "case_id": str(entry["case_id"]),
            "run_id": None,
            "score": 0.0,
            "passed": False,
            "error": failed["error"],
            "run_status": "conversion_failed",
            "run_error": failed["error"],
            "step_outputs": [],
            "subagent_scores": {},
            "input_payload": {},
            "entry": entry,
        })

    coordinator = RunCoordinator(store)
    for eval_case in runnable_cases:
        entry = dict(eval_case["entry"])
        case_id = str(eval_case["case_id"])
        input_payload = eval_case["input"]
        expected = eval_case["expected"]
        sm = strip_atom_mapping_list([str(s) for s in input_payload.get("starting_materials", [])])
        prods = strip_atom_mapping_list([str(p) for p in input_payload.get("products", [])])
        try:
            model_plan = select_step_models(
                model_name=config.model_name,
                thinking_level=config.thinking_level,
                functional_groups_enabled=True,
                intermediate_prediction_enabled=True,
                optional_llm_tools=["attempt_atom_mapping", "predict_missing_reagents"],
            )
            run_id = store.create_run(
                mode="unverified",
                input_payload={
                    "starting_materials": sm,
                    "products": prods,
                    "temperature_celsius": float(input_payload.get("temperature_celsius") or 25.0),
                    "ph": input_payload.get("ph"),
                },
                config={
                    "model": model_plan.step_models.get("mechanism_synthesis", config.model_name),
                    "model_name": model_plan.model_name,
                    "model_family": model_family,
                    "thinking_level": model_plan.thinking_level,
                    "reasoning_level": internal_reasoning,
                    "step_models": model_plan.step_models,
                    "step_reasoning": dict(model_plan.step_reasoning),
                    "api_keys": _get_api_keys_from_env(),
                    "optional_llm_tools": ["attempt_atom_mapping", "predict_missing_reagents"],
                    "functional_groups_enabled": True,
                    "intermediate_prediction_enabled": True,
                    "max_steps": config.max_steps,
                    "max_runtime_seconds": config.max_runtime,
                    "retry_same_candidate_max": max(1, int(config.retry_same_candidate_max)),
                    "repeat_failure_signature_limit": max(2, int(config.repeat_failure_signature_limit)),
                    "max_reproposals_per_step": max(1, int(config.max_reproposals_per_step)),
                    "harness_name": config.harness,
                    "coordination_topology": config.coordination_topology,
                },
                prompt_bundle_hash=hashes.get("prompt_bundle_hash", ""),
                skill_bundle_hash=hashes.get("skill_bundle_hash", ""),
                memory_bundle_hash=hashes.get("memory_bundle_hash", ""),
            )
            coordinator.execute_run(run_id, threading.Event())

            snapshot = store.get_run_snapshot(run_id) or {}
            step_outputs = snapshot.get("step_outputs", [])
            graded = score_snapshot_against_known(snapshot, expected) if expected else {"score": 0.0, "passed": False}
            score = float(graded.get("score", 0.0))
            passed = bool(graded.get("passed", False))
            subagent_scores: Dict[str, Any] = {}
            try:
                subagent_scores = score_subagents_from_step_outputs(step_outputs)
            except Exception:
                pass
            current_state = snapshot.get("current_state", [])
            run_status = snapshot.get("status", "unknown")
            run_error = snapshot.get("error")
            cost_summary = snapshot.get("cost_summary") or {}
            run_cost = cost_summary.get("total_cost") or {}
            if run_status == "failed" and not run_error:
                run_error = get_run_failure_error(store, run_id)

            summary = _curriculum_summary(
                entry,
                config=config,
                batch_start_rank=batch_start_rank,
                current_step_count=current_step_count,
                extra={
                    "score": score,
                    "passed": passed,
                    "error": graded.get("error") or run_error,
                    "eval_mode": "harness",
                    "subagent_scores": subagent_scores,
                    "run_status": run_status,
                    "current_state": current_state,
                },
            )
            store.record_eval_run_result(
                eval_run_id=eval_run_id,
                case_id=case_id,
                run_id=run_id,
                score=score,
                passed=passed,
                cost=run_cost,
                latency_ms=0.0,
                summary=summary,
            )
            results.append({
                "case_id": case_id,
                "run_id": run_id,
                "score": score,
                "passed": passed,
                "subagent_scores": subagent_scores,
                "step_outputs": step_outputs,
                "input_payload": input_payload,
                "run_status": run_status,
                "run_error": run_error,
                "current_state": current_state,
                "graded_details": graded,
                "entry": entry,
            })
            total_cost = run_cost.get("total_cost", 0.0)
            if run_status == "failed":
                print(
                    f"  [{len(results)}] {case_id}: FAILED ({run_error or 'run_failed'}) "
                    f"score={score:.3f} cost=${total_cost:.3f}"
                )
            else:
                print(f"  [{len(results)}] {case_id}: score={score:.3f} passed={passed} cost=${total_cost:.3f}")
            if (
                passed
                and run_status == "completed"
                and score >= config.mining_score_threshold
                and existing_hashes is not None
                and best_scores_by_call is not None
                and workspace is not None
                and evolution_log_path is not None
                and selection_meta is not None
                and progress is not None
                and accumulated_examples_mined is not None
            ):
                result_item = results[-1]
                mined = mine_few_shots([result_item], config, existing_hashes, best_scores_by_call)
                if mined:
                    case_scores = {case_id: score}
                    counts = apply_mined_examples(
                        mined,
                        store,
                        base_dir,
                        workspace,
                        config.dry_run,
                        "curriculum",
                        case_scores,
                        model_name=config.model_name,
                    )
                    for call_name, count in counts.items():
                        accumulated_examples_mined[call_name] = accumulated_examples_mined.get(call_name, 0) + count
                step_count_entry = int(entry.get("step_count", 0))
                if step_count_entry:
                    progress["pass_count_by_step"][step_count_entry] = progress["pass_count_by_step"].get(step_count_entry, 0) + 1
                scores_so_far = [r.get("score") for r in results if r.get("score") is not None]
                mean_so_far = sum(scores_so_far) / len(scores_so_far) if scores_so_far else 0.0
                pass_so_far = sum(1 for r in results if r.get("passed"))
                log = {
                    "config": {
                        "model_name": config.model_name,
                        "harness": config.harness,
                        "eval_set_id": config.eval_set_id,
                        "group_size": config.group_size,
                        "step_pass_target": config.step_pass_target,
                        "min_score_threshold": config.min_score_threshold,
                        "mining_score_threshold": config.mining_score_threshold,
                        "dry_run": config.dry_run,
                        "coordination_topology": config.coordination_topology,
                        "thinking_level": config.thinking_level,
                        "curriculum_index_path": str(config.curriculum_index_path),
                        "train_input": str(config.train_input),
                    },
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "current_step_count": selection_meta.get("current_step_count"),
                    "step_pass_target": selection_meta.get("step_pass_target"),
                    "current_step_pass_count_after_run": progress["pass_count_by_step"].get(step_count_entry, 0),
                    "batch_start_rank": selection_meta.get("batch_start_rank"),
                    "eval_run_id": eval_run_id,
                    "mean_score": round(mean_so_far, 4),
                    "pass_rate": round(pass_so_far / len(results), 4) if results else 0,
                    "pass_count": pass_so_far,
                    "attempted_cases": len(results),
                    "examples_mined": dict(accumulated_examples_mined),
                    "case_results": [{k: v for k, v in r.items() if k != "step_outputs"} for r in results],
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                evolution_log_path.write_text(json.dumps(log, indent=2, default=str))
        except Exception as exc:
            summary = _curriculum_summary(
                entry,
                config=config,
                batch_start_rank=batch_start_rank,
                current_step_count=current_step_count,
                extra={"error": str(exc), "eval_mode": "harness", "run_status": "failed"},
            )
            store.record_eval_run_result(
                eval_run_id=eval_run_id,
                case_id=case_id,
                run_id=None,
                score=0.0,
                passed=False,
                cost={"total_cost": 0.0},
                latency_ms=0.0,
                summary=summary,
            )
            results.append({
                "case_id": case_id,
                "run_id": None,
                "score": 0.0,
                "passed": False,
                "error": str(exc),
                "entry": entry,
                "run_status": "failed",
                "run_error": str(exc),
            })
            print(f"  [{len(results)}] {case_id}: FAILED ({exc})")

    store.set_eval_run_status(eval_run_id, "completed")
    return eval_run_id, results


def evolve(config: EvolutionConfig) -> None:
    base_dir = _PROJECT_ROOT
    source_store = RunStore(resolve_db_path(base_dir))

    requested_eval_identifier = str(config.eval_set_id or "").strip().lower()
    if requested_eval_identifier in {"eval_set", "default", "flower", "flower_100", "flower_100_default"}:
        imported_id = ensure_default_flower_eval_set(source_store, base_dir=base_dir)
        if imported_id:
            config.eval_set_id = imported_id

    try:
        resolved_id = resolve_eval_set_id(source_store, config.eval_set_id)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    resolved_eval_set = source_store.get_eval_set(resolved_id)
    if resolved_eval_set is None:
        print(f"Error: resolved eval set not found: {resolved_id}")
        sys.exit(1)
    if str(resolved_eval_set.get("purpose") or "general") == "leaderboard_holdout":
        print(
            "Error: leaderboard_holdout eval sets are reserved for official ranking only "
            "and cannot be used by scripts/evolve_harness.py."
        )
        sys.exit(1)
    config.eval_set_id = resolved_id

    workspace = setup_workspace(base_dir, config.dry_run)
    runtime_base = workspace if config.dry_run else base_dir
    store = RunStore(runtime_base / "data" / "mechanistic.db")
    mode_label = "DRY RUN" if config.dry_run else "FULL RUN"
    print(f"\n{mode_label} — workspace: {workspace}")

    if config.dry_run and config.curriculum_index_path.is_relative_to(base_dir):
        config.curriculum_index_path = runtime_base / config.curriculum_index_path.relative_to(base_dir)
    if config.dry_run and config.curriculum_index_report_path.is_relative_to(base_dir):
        config.curriculum_index_report_path = runtime_base / config.curriculum_index_report_path.relative_to(base_dir)
    if config.dry_run and config.lookup_cache_path.is_relative_to(base_dir):
        config.lookup_cache_path = runtime_base / config.lookup_cache_path.relative_to(base_dir)

    config.thinking_level = _resolve_thinking_level(config.model_name, config.thinking_level)
    if config.thinking_level:
        print(f"Thinking level: {config.thinking_level}")

    ensure_index(
        input_path=config.train_input,
        index_path=config.curriculum_index_path,
        report_path=config.curriculum_index_report_path,
    )
    build_lookup_cache(input_path=config.train_input, cache_path=config.lookup_cache_path)
    index_entries = load_curriculum_index(config.curriculum_index_path)

    _step_filter_list = list(config.step_count_filter_list)
    _step_filter_idx = 0
    _current_step_filter = config.step_count_filter
    run_total_examples_mined: Dict[str, int] = {}

    while True:
        progress = curriculum_history(
            store,
            model_name=config.model_name,
            harness=config.harness,
            curriculum_index_path=config.curriculum_index_path,
        )
        selection = next_curriculum_candidates(
            index_entries,
            attempted_case_ids=progress["attempted_case_ids"],
            pass_count_by_step=progress["pass_count_by_step"],
            required_passes_per_step=config.step_pass_target,
            step_count_override=_current_step_filter,
            allow_repeats=False,
        )
        candidates = list(selection["candidates"])
        if not candidates and config.loop_when_done:
            selection = next_curriculum_candidates(
                index_entries,
                attempted_case_ids=progress["attempted_case_ids"],
                pass_count_by_step=progress["pass_count_by_step"],
                required_passes_per_step=config.step_pass_target,
                step_count_override=_current_step_filter,
                allow_repeats=True,
            )
            candidates = list(selection["candidates"])
        if not candidates:
            if _step_filter_list and _step_filter_idx < len(_step_filter_list) - 1:
                exhausted_step = _step_filter_list[_step_filter_idx]
                _step_filter_idx += 1
                _current_step_filter = _step_filter_list[_step_filter_idx]
                print(f"Step {exhausted_step} exhausted — advancing to step {_current_step_filter}")
                continue
            if run_total_examples_mined:
                print(f"Curriculum exhausted. Total few-shots mined this run: {run_total_examples_mined}")
            else:
                print("No new eligible curriculum groups remain for this model+harness+index scope.")
            return

        print(
            "Current curriculum step: "
            f"{selection['current_step_count']} "
            f"({selection['current_step_pass_count']}/{selection['step_pass_target']} passed)"
        )
        print(f"Highest successful step count so far: {progress['highest_successful_step_count']}")
        print(f"Next batch starts at row: {selection['batch_start_rank']}")

        prepared_batch = prepare_curriculum_batch(
            config=config,
            candidate_entries=candidates,
            current_step_count=int(selection["current_step_count"]),
        )
        if not prepared_batch["runnable_cases"] and not prepared_batch["conversion_failures"]:
            print("No runnable curriculum cases found at the current step count.")
            continue

        try:
            render_curriculum_pngs(
                input_path=config.train_input,
                index_path=config.curriculum_index_path,
                cache_path=config.lookup_cache_path,
                output_dir=flower_curriculum_pngs_dir(runtime_base),
                entries=[dict(item["entry"]) for item in prepared_batch["runnable_cases"]],
                only_missing=True,
            )
        except Exception as e:
            print(f"Warning: PNG rendering failed ({e}), continuing without visualization...")

        existing_hashes = {}
        best_scores_by_call = {}
        for call_name in set(MINEABLE_SUBAGENTS.values()):
            examples = load_call_few_shot_examples(call_name, runtime_base)
            existing_hashes[call_name] = {hashlib.sha256(ex.get("output", "").encode()).hexdigest()[:16] for ex in examples}
            best_scores_by_call[call_name] = best_few_shot_score(call_name, runtime_base)
        evolution_log_path = workspace / "evolution_log.json"
        accumulated_examples_mined: Dict[str, int] = {}

        if config.dry_run:
            with pushd(runtime_base):
                eval_run_id, case_results = run_curriculum_batch(
                    config,
                    prepared_batch=prepared_batch,
                    store=store,
                    base_dir=runtime_base,
                    existing_hashes=existing_hashes,
                    best_scores_by_call=best_scores_by_call,
                    workspace=workspace,
                    evolution_log_path=evolution_log_path,
                    selection_meta=selection,
                    progress=progress,
                    accumulated_examples_mined=accumulated_examples_mined,
                )
        else:
            eval_run_id, case_results = run_curriculum_batch(
                config,
                prepared_batch=prepared_batch,
                store=store,
                base_dir=runtime_base,
                existing_hashes=existing_hashes,
                best_scores_by_call=best_scores_by_call,
                workspace=workspace,
                evolution_log_path=evolution_log_path,
                selection_meta=selection,
                progress=progress,
                accumulated_examples_mined=accumulated_examples_mined,
            )

        scores = [result["score"] for result in case_results if "score" in result]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        pass_count = sum(1 for result in case_results if result.get("passed"))
        pass_rate = pass_count / len(case_results) if case_results else 0.0
        current_step_before = int(selection["current_step_count"])
        current_step_passes_before = int(progress["pass_count_by_step"].get(current_step_before, 0))
        current_step_passes_after = current_step_passes_before + sum(
            1
            for result in case_results
            if result.get("passed") and int((result.get("entry") or {}).get("step_count") or 0) == current_step_before
        )
        step_advance_ready = current_step_passes_after >= config.step_pass_target
        sa_analysis = analyze_subagent_performance(case_results)

        print(f"\nBatch Summary: mean_score={mean_score:.3f}, pass_rate={pass_rate:.1%}, attempted={len(case_results)}")
        print(
            "Step progression: "
            f"{current_step_passes_after}/{config.step_pass_target} passes at step {current_step_before}; "
            f"advance_ready={step_advance_ready}"
        )
        for _k, _v in accumulated_examples_mined.items():
            run_total_examples_mined[_k] = run_total_examples_mined.get(_k, 0) + _v
        if accumulated_examples_mined:
            print(f"Mined few-shots (this batch): {accumulated_examples_mined}")

        stopped_early = mean_score < config.min_score_threshold
        diag_path = save_run_diagnostics(
            workspace,
            "curriculum",
            case_results,
            config.dry_run,
            batch_start_rank=selection.get("batch_start_rank"),
        )
        print(f"Run diagnostics saved to: {diag_path}")

        log = {
            "config": {
                "model_name": config.model_name,
                "harness": config.harness,
                "eval_set_id": config.eval_set_id,
                "group_size": config.group_size,
                "step_pass_target": config.step_pass_target,
                "min_score_threshold": config.min_score_threshold,
                "mining_score_threshold": config.mining_score_threshold,
                "dry_run": config.dry_run,
                "coordination_topology": config.coordination_topology,
                "thinking_level": config.thinking_level,
                "curriculum_index_path": str(config.curriculum_index_path),
                "train_input": str(config.train_input),
            },
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "current_step_count": selection["current_step_count"],
            "step_pass_target": selection["step_pass_target"],
            "current_step_pass_count_before_run": selection["current_step_pass_count"],
            "current_step_pass_count_after_run": current_step_passes_after,
            "highest_successful_step_count_before_run": progress["highest_successful_step_count"],
            "batch_start_rank": selection["batch_start_rank"],
            "eval_run_id": eval_run_id,
            "mean_score": round(mean_score, 4),
            "pass_rate": round(pass_rate, 4),
            "pass_count": pass_count,
            "attempted_cases": len(case_results),
            "step_advance_ready": step_advance_ready,
            "examples_mined": dict(accumulated_examples_mined),
            "failing_subagents": sa_analysis["consistently_failing"],
            "subagent_scores": sa_analysis["subagents"],
            "case_results": [{k: v for k, v in result.items() if k != "step_outputs"} for result in case_results],
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        evolution_log_path.write_text(json.dumps(log, indent=2, default=str))

        print(f"\nEvolution log: {evolution_log_path}")
        if config.dry_run:
            print(f"Dry run — no permanent changes to skills/. Shadow dir: {workspace}")
        else:
            print("Full run — few-shot files in skills/mechanistic/ have been updated.")


# ---------------------------------------------------------------------------
# Island-based archive evolution
# ---------------------------------------------------------------------------


def apply_island_mutation(
    island: IslandConfig,
    parent: ArchiveEntry,
    base_dir: Path,
    harness_path: Path,
    rng: random.Random,
) -> Tuple[str, str, Path]:
    """Dispatch to the appropriate lane mutator for this island.

    Returns (mutation_type, mutation_summary, mutated_asset_path).
    """
    lane = rng.choice(island.allowed_lanes)
    call_names = ISLAND_CALL_NAMES.get(island.mutation_target, [])

    if lane in ("prompt",) and call_names:
        call_name = rng.choice(call_names)
        mutator = PromptLaneMutator(base_dir=base_dir, call_name=call_name)
        result = mutator.propose(harness_path)
        return "prompt_edit", result.summary, result.asset_path

    if lane in ("few_shot",) and call_names:
        call_name = rng.choice(call_names)
        mutator_fs = FewShotLaneMutator(base_dir=base_dir, call_name=call_name)
        result = mutator_fs.propose(harness_path)
        return "few_shot_mine", result.summary, result.asset_path

    if lane == "topology":
        mutator_topo = TopologyLaneMutator(rng=rng)
        result = mutator_topo.propose(harness_path)
        return "topology_mutate", result.summary, result.asset_path

    if lane == "harness":
        mutator_harness = HarnessLaneMutator()
        result = mutator_harness.propose(harness_path)
        return "harness_toggle", result.summary, result.asset_path

    # Fallback: few-shot on default call_name
    fallback_cn = call_names[0] if call_names else "propose_mechanism_step"
    mutator_fb = FewShotLaneMutator(base_dir=base_dir, call_name=fallback_cn)
    result = mutator_fb.propose(harness_path)
    return "few_shot_mine", result.summary, result.asset_path


def evolve_islands(config: EvolutionConfig, island_config: IslandEvolutionConfig) -> None:
    """Archive-based island evolution loop."""
    base_dir = _PROJECT_ROOT
    store = RunStore(resolve_db_path(base_dir))

    requested_eval_identifier = str(config.eval_set_id or "").strip().lower()
    if requested_eval_identifier in {"eval_set", "default", "flower", "flower_100", "flower_100_default"}:
        imported_id = ensure_default_flower_eval_set(store, base_dir=base_dir)
        if imported_id:
            config.eval_set_id = imported_id

    try:
        resolved_id = resolve_eval_set_id(store, config.eval_set_id)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    resolved_eval_set = store.get_eval_set(resolved_id)
    if resolved_eval_set is None:
        print(f"Error: resolved eval set not found: {resolved_id}")
        sys.exit(1)
    if str(resolved_eval_set.get("purpose") or "general") == "leaderboard_holdout":
        print("Error: leaderboard_holdout eval sets cannot be used for evolution.")
        sys.exit(1)
    config.eval_set_id = resolved_id

    workspace = setup_workspace(base_dir, config.dry_run)
    runtime_base = workspace if config.dry_run else base_dir
    runtime_store = RunStore(runtime_base / "data" / "mechanistic.db")

    # Set up archive
    active_islands = [isl for isl in DEFAULT_ISLANDS if isl.id in island_config.islands]
    archive = EvolutionArchive(store=runtime_store, islands=active_islands)

    # Seed from leaderboard if empty
    if archive.is_empty():
        lb_dir = base_dir / "curriculum" / "generated"
        count = archive.seed_from_leaderboard(lb_dir)
        print(f"Seeded archive with {count} entries from leaderboard")

    config.thinking_level = _resolve_thinking_level(config.model_name, config.thinking_level)
    ensure_index(
        input_path=config.train_input,
        index_path=config.curriculum_index_path,
        report_path=config.curriculum_index_report_path,
    )
    build_lookup_cache(input_path=config.train_input, cache_path=config.lookup_cache_path)
    index_entries = load_curriculum_index(config.curriculum_index_path)

    rng = random.Random(island_config.seed)
    generation = archive.max_generation() + 1
    harness_path = base_dir / "harness_versions" / config.harness / "harness.json"

    mode_label = "DRY RUN" if config.dry_run else "FULL RUN"
    print(f"\nIsland Evolution {mode_label} — generation {generation}")
    print(f"  Active islands: {[isl.id for isl in active_islands]}")
    print(f"  Migration interval: {island_config.migration_interval}")
    print()

    while True:
        for island in active_islands:
            print(f"\n--- Generation {generation} | Island: {island.id} ({island.label}) ---")

            # 1. Select parent
            try:
                parent = archive.select_parent(island.id, rng=rng)
                archive.increment_children(parent.id)
                print(f"  Parent: {parent.id[:8]}... (score={parent.mean_quality_score:.4f}, children={parent.children_count})")
            except ValueError:
                print(f"  No parents available on island {island.id}, skipping")
                continue

            # 2. Apply mutation
            try:
                mutation_type, mutation_summary, mutated_path = apply_island_mutation(
                    island, parent, runtime_base, harness_path, rng,
                )
                print(f"  Mutation: {mutation_type} — {mutation_summary}")
            except (FileNotFoundError, ValueError) as exc:
                print(f"  Mutation failed: {exc}, skipping island")
                continue

            # 3. Run eval batch
            progress = curriculum_history(
                runtime_store,
                model_name=config.model_name,
                harness=config.harness,
                curriculum_index_path=config.curriculum_index_path,
            )

            # Apply tier filter for hard_multistep island
            step_override = config.step_count_filter
            if island.eval_tier_filter == "hard":
                step_override = 9  # 9+ step reactions

            selection = next_curriculum_candidates(
                index_entries,
                attempted_case_ids=progress["attempted_case_ids"],
                pass_count_by_step=progress["pass_count_by_step"],
                required_passes_per_step=config.step_pass_target,
                step_count_override=step_override,
                allow_repeats=True,
            )
            candidates = list(selection["candidates"])
            if not candidates:
                print(f"  No curriculum candidates for island {island.id}")
                continue

            prepared_batch = prepare_curriculum_batch(
                config=config,
                candidate_entries=candidates,
                current_step_count=int(selection["current_step_count"]),
            )
            if not prepared_batch["runnable_cases"]:
                print(f"  No runnable cases for island {island.id}")
                continue

            registry = RegistrySet(runtime_base)
            hashes = registry.bundle_hashes()

            eval_run_id, case_results = run_curriculum_batch(
                config,
                prepared_batch=prepared_batch,
                store=runtime_store,
                base_dir=runtime_base,
                workspace=workspace,
            )

            # 4. Score and create archive entry
            scores = [r["score"] for r in case_results if "score" in r]
            mean_score = sum(scores) / len(scores) if scores else 0.0
            pass_count = sum(1 for r in case_results if r.get("passed"))
            pass_rate = pass_count / len(case_results) if case_results else 0.0
            sa_analysis = analyze_subagent_performance(case_results)

            score_delta = mean_score - parent.mean_quality_score

            entry = ArchiveEntry(
                id=uuid.uuid4().hex,
                generation=generation,
                island_id=island.id,
                parent_id=parent.id,
                archive_inspiration_ids=[],
                harness_name=config.harness,
                harness_config_json=json.dumps(hashes),
                prompt_bundle_hash=hashes.get("prompt_bundle_hash", ""),
                skill_bundle_hash=hashes.get("skill_bundle_hash", ""),
                few_shot_snapshot_json="{}",
                topology_profile_json="{}",
                mutation_type=mutation_type,
                mutation_summary=mutation_summary,
                mean_quality_score=mean_score,
                weighted_pass_rate=pass_rate,
                per_subagent_scores_json=json.dumps(sa_analysis.get("subagents", {})),
                total_cost=sum(
                    float((r.get("step_outputs") or {}).get("total_cost", 0))
                    for r in case_results
                    if isinstance(r.get("step_outputs"), dict)
                ),
                eval_run_id=eval_run_id or None,
                eval_tier=island.eval_tier_filter or "mixed",
                case_count=len(case_results),
                children_count=0,
                score_delta=score_delta,
                migration_history_json="[]",
                created_at=time.time(),
            )
            archive.insert(entry)

            print(f"  Result: score={mean_score:.4f} (delta={score_delta:+.4f}), pass_rate={pass_rate:.1%}")

            # 5. Stagnation check
            if archive.stagnation_check(island.id, window=island.stagnation_threshold):
                print(f"  WARNING: Island {island.id} stagnating ({island.stagnation_threshold} gens without improvement)")

        # 6. Migration check
        if generation % island_config.migration_interval == 0:
            print(f"\n--- Migration check (generation {generation}) ---")
            for source in active_islands:
                source_best = archive.island_best(source.id)
                if source_best is None:
                    continue
                for target in active_islands:
                    if target.id == source.id:
                        continue
                    if archive.attempt_migration(source.id, target.id, source_best):
                        print(f"  Migrated {source.id} -> {target.id} (score={source_best.mean_quality_score:.4f})")

        generation += 1
        print(f"\n=== Completed generation {generation - 1} ===\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evolve the harness against the ranked FlowER curriculum.")
    parser.add_argument("--model-name", required=True, help="Model identifier")
    parser.add_argument("--harness", default="default", help="Harness name from harness_versions/")
    parser.add_argument("--eval-set-id", default="eval_set", help="Eval set ID, name, or alias.")
    parser.add_argument("--group-size", type=int, default=4, help="Target number of runnable curriculum cases per invocation.")
    parser.add_argument("--cases-per-tier", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--step-pass-target", type=int, default=20, help="Stay on the current step count until this many cases at that step have passed.")
    parser.add_argument("--min-score-threshold", type=float, default=0.2, help="Stop mining if the batch mean score drops below this.")
    parser.add_argument("--mining-score-threshold", type=float, default=0.5, help="Only mine few-shots from sufficiently strong batches.")
    parser.add_argument("--max-few-shots-per-step", type=int, default=3, help="Max new examples per subagent per batch.")
    parser.add_argument("--max-steps", type=int, default=10, help="Max mechanism steps per case.")
    parser.add_argument("--max-runtime", type=float, default=600.0, help="Per-case timeout in seconds.")
    parser.add_argument("--retry-same-candidate-max", type=int, default=1, help="Retries per candidate before moving on.")
    parser.add_argument("--repeat-failure-signature-limit", type=int, default=2, help="Repeat count of same validation signature before forced reproposal.")
    parser.add_argument("--max-reproposals-per-step", type=int, default=4, help="Maximum reproposals allowed per mechanism step.")
    parser.add_argument("--seed", type=int, default=42, help="Reserved for deterministic future extensions.")
    parser.add_argument("--coordination-topology", default="centralized_mas", choices=["sas", "centralized_mas", "independent_mas", "decentralized_mas"], help="Coordination topology strategy (default: centralized_mas).")
    parser.add_argument("--thinking-level", default=None, help="Reasoning level: low, high, max, or auto (default: highest available for the model).")
    parser.add_argument("--step-count", default="mixed", help="Step count filter: 1, 2, 3, ... 8, mixed (default: curriculum-driven), or a range like 2-3 to run step 2 then step 3.")
    parser.add_argument("--loop", action="store_true", help="When no candidates remain, allow repeats and continue until none.")
    parser.add_argument("--dry-run", action="store_true", help="Store updates in a shadow workspace.")
    parser.add_argument("--island-mode", action="store_true", help="Enable island-based archive evolution (ShinkaEvolve-inspired).")
    parser.add_argument("--islands", default=None, help="Comma-separated island IDs (default: mapping,reagent_conditions,topology,hard_multistep).")
    parser.add_argument("--migration-interval", type=int, default=5, help="Generations between migration attempts (default: 5).")
    parser.add_argument("--train-input", default=str(DEFAULT_FLOWER_INPUT), help="Path to FlowER train.txt.")
    parser.add_argument("--curriculum-index-path", default=str(DEFAULT_INDEX_PATH), help="Curriculum index JSONL path.")
    parser.add_argument("--lookup-cache-path", default=str(DEFAULT_LOOKUP_CACHE), help="Lookup cache SQLite path.")
    args = parser.parse_args()

    group_size = int(args.group_size)
    if args.cases_per_tier is not None:
        print("Warning: --cases-per-tier is deprecated; use --group-size instead.")
        if group_size != parser.get_default("group_size"):
            print("Warning: both --group-size and --cases-per-tier provided; using --group-size.")
        else:
            group_size = int(args.cases_per_tier)

    step_count_filter: Optional[int] = None
    step_count_filter_list: List[int] = []
    raw_sc = (args.step_count or "mixed").strip().lower()
    if raw_sc != "mixed":
        if "-" in raw_sc:
            parts = raw_sc.split("-", 1)
            try:
                lo, hi = int(parts[0]), int(parts[1])
                step_count_filter_list = [s for s in range(lo, hi + 1) if 1 <= s <= 20]
                if step_count_filter_list:
                    step_count_filter = step_count_filter_list[0]
            except (TypeError, ValueError):
                pass
        else:
            try:
                step_count_filter = int(raw_sc)
                if step_count_filter < 1 or step_count_filter > 20:
                    step_count_filter = None
            except (TypeError, ValueError):
                step_count_filter = None

    config = EvolutionConfig(
        model_name=args.model_name,
        harness=args.harness,
        eval_set_id=args.eval_set_id,
        group_size=max(1, group_size),
        step_pass_target=max(1, int(args.step_pass_target)),
        min_score_threshold=args.min_score_threshold,
        mining_score_threshold=args.mining_score_threshold,
        max_few_shots_per_step=args.max_few_shots_per_step,
        dry_run=args.dry_run,
        max_steps=args.max_steps,
        max_runtime=args.max_runtime,
        retry_same_candidate_max=max(1, int(args.retry_same_candidate_max)),
        repeat_failure_signature_limit=max(2, int(args.repeat_failure_signature_limit)),
        max_reproposals_per_step=max(1, int(args.max_reproposals_per_step)),
        seed=args.seed,
        coordination_topology=args.coordination_topology,
        thinking_level=args.thinking_level,
        step_count_filter=step_count_filter,
        step_count_filter_list=step_count_filter_list,
        loop_when_done=args.loop,
        train_input=Path(args.train_input),
        curriculum_index_path=Path(args.curriculum_index_path),
        lookup_cache_path=Path(args.lookup_cache_path),
    )

    print(f"Harness Evolution — {config.model_name}")
    print(f"  Harness: {config.harness}")
    print(f"  Mode: {'DRY RUN' if config.dry_run else 'FULL RUN'}")
    print(f"  Group size: {config.group_size}")
    print(f"  Step pass target: {config.step_pass_target}")
    print(f"  Coordination topology: {config.coordination_topology}")
    print(f"  Thinking level: {config.thinking_level or 'auto'}")
    if config.step_count_filter_list:
        sc_display = "-".join(str(s) for s in config.step_count_filter_list)
    else:
        sc_display = str(config.step_count_filter) if config.step_count_filter else "mixed"
    print(f"  Step count filter: {sc_display}")
    print(f"  Loop when done: {config.loop_when_done}")
    print(f"  Island mode: {args.island_mode}")
    print(f"  Curriculum index: {config.curriculum_index_path}")
    print()
    try:
        if args.island_mode:
            island_ids = (
                [s.strip() for s in args.islands.split(",") if s.strip()]
                if args.islands
                else ["mapping", "reagent_conditions", "topology", "hard_multistep"]
            )
            island_cfg = IslandEvolutionConfig(
                islands=island_ids,
                migration_interval=max(1, int(args.migration_interval)),
                seed=config.seed,
            )
            evolve_islands(config, island_cfg)
        else:
            evolve(config)
    except KeyboardInterrupt:
        print("\nRun interrupted by user. Few-shot examples from completed cases have already been saved to skills/.")


if __name__ == "__main__":
    main()
