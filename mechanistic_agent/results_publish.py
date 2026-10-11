"""Publish local eval results as committed, public artifacts.

Eval runs live in the local SQLite store (gitignored). This module turns one
eval run into a self-contained record under ``results/runs/`` plus a rendered
image of the hardest mechanism it solved under ``results/mechanisms/``, and
regenerates the public ``LEADERBOARD.md`` and the README leaderboard block from
every committed record. Git, not the local database, is the source of truth for
the public board.

Harness-free baseline eval runs (run group ``harness_free_baseline[_<tier>]``)
are exported from their stored case summaries as ``kind: "baseline"`` records.
They have no run snapshots and no mechanism image, are kept out of the
best-by-tier table and per-run sections, and render in their own table.

Guards:
* runs whose responder declares it saw the ground truth are refused;
* leaderboard-holdout eval sets are exported as aggregates only (no per-case
  rows, no mechanisms), so holdout chemistry never lands in the repo.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from mechanistic_agent.agent_bridge import origin_integrity_contaminated
from mechanistic_agent.quality_scoring import (
    QUALITY_VERSION,
    WEIGHTS as QUALITY_WEIGHTS,  # noqa: F401 - re-exported for callers and tests
    _baseline_snapshot,
    quality_or_error,
    summarize as summarize_quality,
)
from mechanistic_agent.quality_v2 import (
    AXES as V2_AXES,
    AXIS_LABELS as V2_AXIS_LABELS,
    QUALITY_V2_VERSION,
    summarize as summarize_v2,
)
from mechanistic_agent.rescoring import default_expected_resolver
from mechanistic_agent.scoring import (
    DEFAULT_SCORING_VERSION,
    extract_accepted_path,
    graded_to_points,
    score_snapshot_against_known,
)
from mechanistic_agent.smiles_utils import strip_atom_mapping_list

RECORD_SCHEMA = "wiggum.published_eval_run@1"  # @1 records without ``scoring`` are legacy-scored
RESULTS_DIR = Path("results")
RUNS_DIR = RESULTS_DIR / "runs"
MECHANISMS_DIR = RESULTS_DIR / "mechanisms"
RADAR_DIR = RESULTS_DIR / "radar"
LEADERBOARD_PATH = Path("LEADERBOARD.md")
README_PATH = Path("README.md")
README_LEADERBOARD_START = "<!-- leaderboard:start -->"
README_LEADERBOARD_END = "<!-- leaderboard:end -->"
TIER_ORDER = ("easy", "medium", "hard")
# Mirrors core.baseline_runner.BASELINE_GROUP_PREFIX (not imported: that module pulls in the LLM stack).
BASELINE_GROUP_PREFIX = "harness_free_baseline"
BASELINE_TIER_ORDER = (*TIER_ORDER, "holdout")
BASELINE_ERROR_MAX_CHARS = 200


class PublishError(RuntimeError):
    """Raised when an eval run cannot be published."""


def _truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"true", "1", "yes"}


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "")).strip("_") or "run"


def _git_commit(base_dir: Path) -> Optional[str]:
    proc = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=base_dir, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _synthesis_details(snapshot: Dict[str, Any], step: Dict[str, Any]) -> Dict[str, Any]:
    """Find the validated synthesis row behind an accepted step (SMIRKS + pushes)."""
    target = sorted(step.get("resulting_state") or [])
    for row in snapshot.get("step_outputs") or []:
        if row.get("step_name") != "mechanism_synthesis":
            continue
        if int(row.get("attempt") or 0) != int(step.get("step_index") or 0):
            continue
        output = row.get("output") if isinstance(row.get("output"), dict) else {}
        if sorted(str(item) for item in output.get("resulting_state") or []) == target:
            return output
    return {}


def _accepted_path_record(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    events = {
        int((e.get("payload") or {}).get("step_index") or 0): e.get("payload") or {}
        for e in snapshot.get("events") or []
        if e.get("event_type") == "mechanism_step_accepted"
    }
    steps: List[Dict[str, Any]] = []
    for step in extract_accepted_path(snapshot):
        payload = events.get(int(step.get("step_index") or 0), {})
        details = _synthesis_details(snapshot, step)
        steps.append(
            {
                "step_index": int(step.get("step_index") or 0),
                "current_state": list(step.get("current_state") or []),
                "resulting_state": list(step.get("resulting_state") or []),
                "reaction_smirks": payload.get("reaction_smirks") or details.get("reaction_smirks"),
                "electron_pushes": payload.get("electron_pushes") or details.get("electron_pushes") or [],
                "acceptance_kind": payload.get("acceptance_kind"),
            }
        )
    return steps


def _pick_hardest(cases: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    solved = [c for c in cases if c.get("passed") and c.get("accepted_steps")]
    if not solved:
        return None
    return max(
        solved,
        key=lambda c: (
            int(c.get("known_steps") or 0),
            float(c.get("quality_points") if c.get("quality_points") is not None else (c.get("score") or 0.0) * 1000),
            int(c.get("accepted_steps") or 0),
        ),
    )


def _record_date(run: Dict[str, Any]) -> str:
    created = run.get("created_at")
    if isinstance(created, (int, float)):
        return datetime.fromtimestamp(float(created)).strftime("%Y-%m-%d")
    return time.strftime("%Y-%m-%d")


def _refuse_ground_truth(origin: Optional[Dict[str, Any]], eval_run_id: str, where: str) -> None:
    if origin and _truthy(origin.get("responder_saw_ground_truth")):
        raise PublishError(f"eval run {eval_run_id} is a ground-truth replay ({where}); refusing to publish")
    _refuse_contaminated(origin, eval_run_id, where)


def _refuse_contaminated(block: Optional[Dict[str, Any]], eval_run_id: str, where: str) -> None:
    """Refuse runs a responder-integrity audit marked contaminated (origin or eval-run metadata)."""
    if origin_integrity_contaminated(block):
        raise PublishError(
            f"eval run {eval_run_id} is contaminated per the responder-integrity audit ({where}); refusing to publish"
        )


def _case_quality(summary: Dict[str, Any], snapshot: Optional[Dict[str, Any]], expected: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The case's quality_v1 result: the stored one, else computed from the snapshot (harness) or
    the saved baseline steps. None when there is nothing to re-check (a legacy baseline)."""
    stored = summary.get("quality")
    stored_ok = isinstance(stored, dict) and stored.get("version") == QUALITY_VERSION and "error" not in stored
    if stored_ok and isinstance(stored.get("quality_v2"), dict):
        return stored
    snapshot = (snapshot or _baseline_snapshot(summary, expected)) if expected else None
    if not snapshot:
        return stored if stored_ok else None
    return quality_or_error(snapshot, expected)


def _v2_case_fields(quality: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-case quality_v2 columns of a record (the v1 points are kept alongside)."""
    v2 = (quality or {}).get("quality_v2")
    if not isinstance(v2, dict):
        return {}
    return {
        "quality_v1_points": quality.get("points"),
        "quality_points": v2.get("points"),
        "axes": v2.get("axes"),
        "failed_start": bool(v2.get("failed_start")),
        "connected_steps": f"{sum(1 for link in v2.get('links') or [] if link.get('connected'))}/{v2.get('step_count')}",
        "broken_links": [link for link in v2.get("links") or [] if not link.get("connected")],
    }


def _quality_fields(qualities: List[Optional[Dict[str, Any]]], legacy_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Record-level scoring fields: quality_v1 when every case has a quality result, else legacy."""
    if not qualities or any(q is None for q in qualities):
        return {"scoring": "legacy", "summary": legacy_summary}
    quality = summarize_quality([q for q in qualities if q is not None])
    v1_summary = {
        "points": int(round(quality["points"])),
        "mechanism_points": int(round(quality.get("mechanism_points", quality["points"]))),
        "product_correct": quality.get("product_correct", quality["targets_reached"]),
        "products_hidden": bool(quality.get("products_hidden")),
        "outcome": "pass" if quality["passed"] == quality["cases"] else "partial",
        "components": quality["components"],
        "cases": quality["cases"],
        "targets_reached": quality["targets_reached"],
        "passed": quality["passed"],
        "valid_step_fraction": quality["valid_step_fraction"],
        "avg_latency_s": legacy_summary.get("avg_latency_s"),
    }
    v2_cases = [q.get("quality_v2") for q in qualities if q is not None]
    if not all(isinstance(v, dict) for v in v2_cases):
        return {"scoring": QUALITY_VERSION, "summary": v1_summary, "legacy": legacy_summary}
    v2 = summarize_v2(v2_cases)
    return {
        "scoring": QUALITY_V2_VERSION,
        "summary": {
            **v1_summary,
            "points": int(round(v2["points"])),
            "outcome": "pass" if v2["passed"] == v2["cases"] else "partial",
            "passed": v2["passed"],
            "axes": v2["axes"],
            "step_quality": v2["step_quality"],
            "polygon_area": v2["polygon_area"],
            "failed_starts": v2["failed_starts"],
            "failed_start_fraction": v2["failed_start_fraction"],
            "fully_connected": v2["fully_connected"],
        },
        "quality_v1": v1_summary,
        "legacy": legacy_summary,
    }


def is_baseline_eval_run(run: Dict[str, Any], results: Sequence[Dict[str, Any]]) -> bool:
    """Harness-free baseline: baseline run group, or every case summary says ``eval_mode: baseline``."""
    if str(run.get("run_group_name") or "").startswith(BASELINE_GROUP_PREFIX):
        return True
    return bool(results) and all(
        isinstance(r.get("summary"), dict) and r["summary"].get("eval_mode") == "baseline" for r in results
    )


def _baseline_tier(run: Dict[str, Any], metadata: Dict[str, Any], holdout: bool) -> Optional[str]:
    group = str(run.get("run_group_name") or "")
    if holdout or group == BASELINE_GROUP_PREFIX:
        return "holdout"
    suffix = group[len(BASELINE_GROUP_PREFIX):].lstrip("_") if group.startswith(BASELINE_GROUP_PREFIX) else ""
    if suffix in TIER_ORDER:
        return suffix
    return metadata.get("tier_name")


def _export_baseline_run(
    store: Any,
    run: Dict[str, Any],
    results: List[Dict[str, Any]],
    *,
    holdout: bool,
    base_dir: Optional[Path],
) -> Dict[str, Any]:
    """Record for a harness-free baseline run, built from case summaries alone (no snapshots).

    Only scores, counts and labels are copied: the summaries' SMILES (known product,
    step states) never reach the record.
    """
    eval_run_id = str(run.get("id") or "")
    metadata = run.get("metadata") if isinstance(run.get("metadata"), dict) else {}
    origin = metadata.get("origin") if isinstance(metadata.get("origin"), dict) else None
    _refuse_ground_truth(origin, eval_run_id, "eval run origin")
    thinking = run.get("thinking_level")
    case_ids_hash = metadata.get("selected_case_ids_hash")
    versions = set()
    cases: List[Dict[str, Any]] = []
    qualities: List[Optional[Dict[str, Any]]] = []
    resolver = default_expected_resolver(store) if store is not None else None
    for result in results:
        summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
        breakdown = summary.get("scoring_breakdown") if isinstance(summary.get("scoring_breakdown"), dict) else {}
        run_metadata = summary.get("run_metadata") if isinstance(summary.get("run_metadata"), dict) else {}
        case_origin = run_metadata.get("origin") if isinstance(run_metadata.get("origin"), dict) else None
        _refuse_ground_truth(case_origin, eval_run_id, f"case {result.get('case_id')}")
        origin = origin or case_origin
        thinking = thinking or run_metadata.get("thinking_level")
        case_ids_hash = case_ids_hash or run_metadata.get("eval_case_ids_hash")
        if summary.get("scoring_version"):
            versions.add(str(summary["scoring_version"]))
        error = summary.get("error")
        score = result.get("score") if result.get("score") is not None else summary.get("score")
        quality = _case_quality(summary, None, resolver(result, run)) if resolver else None
        qualities.append(quality)
        cases.append(
            {
                "case_id": str(result.get("case_id") or ""),
                "quality_points": quality.get("points") if quality else None,
                "quality_passed": bool(((quality or {}).get("quality_v2") or quality).get("passed")) if quality else None,
                "valid_steps": f"{quality.get('valid_steps')}/{quality.get('step_count')}" if quality else None,
                **_v2_case_fields(quality),
                "score": round(float(score or 0.0), 4),
                "target_reached": bool(breakdown.get("final_product_reached")),
                "alignment": round(float(breakdown.get("known_alignment_component") or 0.0), 4),
                "known_steps": int(breakdown.get("known_step_count") or 0),
                "predicted_steps": int(breakdown.get("accepted_path_step_count") or summary.get("step_count") or 0),
                "latency_s": round(float(result.get("latency_ms") or 0.0) / 1000.0, 1),
                "mechanism_type": summary.get("mechanism_type"),
                "error": str(error)[:BASELINE_ERROR_MAX_CHARS] if error else None,
            }
        )

    n = len(cases)
    mean_score = round(sum(c["score"] for c in cases) / n, 4) if n else 0.0
    record: Dict[str, Any] = {
        "schema": RECORD_SCHEMA,
        "kind": "baseline",
        "eval_run_id": eval_run_id,
        "run_group": run.get("run_group_name"),
        "date": _record_date(run),
        "model": run.get("model_name") or run.get("model"),
        "thinking_level": thinking,
        "origin": origin,
        "harness": None,
        "tier": _baseline_tier(run, metadata, holdout),
        "eval_set_id": run.get("eval_set_id"),
        "case_ids_hash": case_ids_hash,
        "holdout": holdout,
        "scoring_version": versions.pop() if len(versions) == 1 else ("mixed" if versions else None),
        "git_commit": _git_commit(base_dir or Path.cwd()),
        "legacy_summary": {
            # Legacy: ``points`` is the mean case score x 1000, not a rubric.
            "points": int(round(mean_score * 1000)),
            "outcome": "baseline",
            "cases": n,
            "targets_reached": sum(1 for c in cases if c["target_reached"]),
            # Baseline steps are not validator-checked, so no case passes.
            "passed": 0,
            "errors": sum(1 for c in cases if c["error"]),
            "mean_score": mean_score,
            "avg_latency_s": round(sum(c["latency_s"] for c in cases) / n, 1) if n else 0.0,
        },
    }
    record.update(_quality_fields(qualities, record.pop("legacy_summary")))
    if not holdout:
        record["cases"] = cases
    return record


def _combined_results(store: Any, eval_run_ids: Sequence[str]) -> tuple:
    """Merge case results from several eval runs of one tier; later runs win per case.

    Used to publish a tier that was resumed (e.g. after a responder outage): the
    record lists every source run and which cases came from it.
    """
    runs: List[Dict[str, Any]] = []
    by_case: Dict[str, tuple] = {}
    for eval_run_id in eval_run_ids:
        run = store.get_eval_run(eval_run_id)
        if not run:
            raise PublishError(f"eval run {eval_run_id} not found")
        if runs:
            first = runs[0]
            for key in ("eval_set_id", "model_name", "thinking_level"):
                if run.get(key) != first.get(key):
                    raise PublishError(f"cannot combine eval runs with different {key}: {eval_run_ids}")
        runs.append(run)
        for result in store.list_eval_run_results(eval_run_id):
            by_case[str(result.get("case_id") or "")] = (run, result)
    if not by_case:
        raise PublishError(f"eval run(s) {', '.join(eval_run_ids)} have no case results")
    sources = [
        {
            "eval_run_id": run.get("id"),
            "run_group": run.get("run_group_name"),
            "cases": sorted(case for case, (src, _) in by_case.items() if src is run),
        }
        for run in runs
    ]
    return runs, [by_case[case] for case in sorted(by_case)], sources


def export_eval_run(
    store: Any,
    eval_run_id: Any,
    *,
    base_dir: Optional[Path] = None,
    expected_resolver: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Optional[Dict[str, Any]]]] = None,
    scoring_version: str = DEFAULT_SCORING_VERSION,
) -> Dict[str, Any]:
    """Build the public record for one eval run, or a combined record when given a
    list of eval run ids for the same tier (does not write anything)."""
    ids = [eval_run_id] if isinstance(eval_run_id, str) else [str(i) for i in eval_run_id]
    runs, pairs, sources = _combined_results(store, ids)
    run = runs[0]
    eval_run_id = ids[-1] if len(ids) > 1 else ids[0]
    for source_run in runs:
        _refuse_contaminated(source_run.get("metadata"), str(source_run.get("id") or eval_run_id), "eval run metadata")
    results = [result for _, result in pairs]
    run_for_result = {id(result): src for src, result in pairs}
    eval_set = store.get_eval_set(str(run.get("eval_set_id") or "")) or {}
    holdout = str(eval_set.get("purpose") or "") == "leaderboard_holdout"
    if is_baseline_eval_run(run, results):
        record = _export_baseline_run(store, {**run, "id": run.get("id") or eval_run_id}, results, holdout=holdout,
                                      base_dir=base_dir)
        record["eval_set_name"] = eval_set.get("name")
        return record
    resolver = expected_resolver or default_expected_resolver(store)
    metadata = run.get("metadata") if isinstance(run.get("metadata"), dict) else {}

    origin: Optional[Dict[str, Any]] = None
    harness_name: Optional[str] = None
    graded_all: List[Dict[str, Any]] = []
    latencies: List[float] = []
    cases: List[Dict[str, Any]] = []
    qualities: List[Optional[Dict[str, Any]]] = []
    snapshots: Dict[str, Dict[str, Any]] = {}

    for result in results:
        run_id = str(result.get("run_id") or "")
        snapshot = store.get_run_snapshot(run_id) if run_id else None
        config = (snapshot or {}).get("config") if isinstance((snapshot or {}).get("config"), dict) else {}
        case_origin = config.get("origin") if isinstance(config.get("origin"), dict) else None
        _refuse_ground_truth(case_origin, eval_run_id, f"case {result.get('case_id')}")
        origin = origin or case_origin
        harness_name = harness_name or config.get("harness_name")
        latency = float(result.get("latency_ms") or 0.0)
        latencies.append(latency)
        expected = resolver(result, run_for_result.get(id(result), run)) if snapshot else None
        graded = score_snapshot_against_known(snapshot, expected, scoring_version=scoring_version) if (snapshot and expected) else {}
        graded_all.append(graded)
        summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
        quality = _case_quality(summary, snapshot, expected) if snapshot else None
        qualities.append(quality)
        known_steps = int((expected or {}).get("n_mechanistic_steps") or len(((expected or {}).get("verified_mechanism") or {}).get("steps") or []) or 0)
        case = {
            "case_id": str(result.get("case_id") or ""),
            "known_steps": known_steps,
            "accepted_steps": int(graded.get("accepted_path_step_count") or 0),
            "target_reached": bool(graded.get("final_product_reached")),
            "passed": bool(((quality.get("quality_v2") or quality)).get("passed")) if quality else bool(graded.get("passed", result.get("pass_bool"))),
            "quality_points": quality.get("points") if quality else None,
            "valid_steps": f"{quality.get('valid_steps')}/{quality.get('step_count')}" if quality else None,
            **_v2_case_fields(quality),
            "score": round(float(graded.get("score", result.get("score") or 0.0)), 4),
            "latency_s": round(latency / 1000.0, 1),
            "run_status": (snapshot or {}).get("status"),
        }
        cases.append(case)
        if snapshot:
            snapshots[case["case_id"]] = snapshot

    points = graded_to_points(graded_all, latencies)
    n = len(cases)
    date = _record_date(run)
    record: Dict[str, Any] = {
        "schema": RECORD_SCHEMA,
        "eval_run_id": eval_run_id,
        "run_group": run.get("run_group_name") if len(ids) == 1 else f"{run.get('run_group_name')}+resumed",
        "date": date,
        "model": run.get("model_name") or run.get("model"),
        "thinking_level": run.get("thinking_level"),
        "origin": origin,
        "harness": harness_name,
        "harness_bundle_hash": run.get("harness_bundle_hash"),
        "tier": metadata.get("tier_name"),
        "eval_set_id": run.get("eval_set_id"),
        "eval_set_name": eval_set.get("name"),
        "case_ids_hash": metadata.get("selected_case_ids_hash"),
        "holdout": holdout,
        "scoring_version": scoring_version,
        "git_commit": _git_commit(base_dir or Path.cwd()),
        "sources": sources if len(ids) > 1 else None,
        "legacy_summary": {
            "points": points["total"],
            "outcome": points["outcome"],
            "breakdown": {k: points[k] for k in ("product", "pathway", "push", "speed", "methodology")},
            "cases": n,
            "targets_reached": sum(1 for c in cases if c["target_reached"]),
            "passed": sum(1 for c in cases if c["passed"]),
            "mean_score": round(sum(c["score"] for c in cases) / n, 4) if n else 0.0,
            "avg_latency_s": round(points["avg_latency_ms"] / 1000.0, 1),
        },
    }
    record.update(_quality_fields(qualities, record.pop("legacy_summary")))
    if holdout:
        return record
    record["cases"] = cases
    hardest = _pick_hardest(cases)
    if hardest is not None:
        snapshot = snapshots.get(hardest["case_id"]) or {}
        payload = snapshot.get("input_payload") if isinstance(snapshot.get("input_payload"), dict) else {}
        record["hardest_solved"] = {
            "case_id": hardest["case_id"],
            "known_steps": hardest["known_steps"],
            "accepted_steps": hardest["accepted_steps"],
            "score": hardest["score"],
            "starting_materials": strip_atom_mapping_list([str(s) for s in payload.get("starting_materials") or []]),
            "products": strip_atom_mapping_list([str(s) for s in payload.get("products") or []]),
            "steps": _accepted_path_record(snapshot),
        }
    return record


def _is_baseline_record(record: Dict[str, Any]) -> bool:
    return record.get("kind") == "baseline"


def record_filename(record: Dict[str, Any]) -> str:
    stem = f"{record.get('date')}_{_slug(record.get('run_group') or record.get('eval_run_id'))}"
    if _is_baseline_record(record):
        # Baseline run groups are shared across models (harness_free_baseline_<tier>), so add the model
        # and the eval run id to keep one file per run; re-publishing the same run still overwrites it.
        stem += f"_{_slug(_model_key(_effective_model(record)))}_{str(record.get('eval_run_id') or '')[:8]}"
    return f"{stem}.json"


def mechanism_image_path(record: Dict[str, Any]) -> Optional[Path]:
    hardest = record.get("hardest_solved")
    if not hardest:
        return None
    return MECHANISMS_DIR / f"{_slug(record.get('run_group') or record.get('eval_run_id'))}__{_slug(hardest['case_id'])}.png"


def write_record(record: Dict[str, Any], base_dir: Path, *, render_image: bool = True) -> List[Path]:
    """Write the record JSON (and mechanism PNG); return the paths written."""
    written: List[Path] = []
    image_rel = mechanism_image_path(record)
    if image_rel is not None and render_image:
        from mechanistic_agent.flower_rendering import render_mechanism_png

        hardest = record["hardest_solved"]
        case = {
            "id": hardest["case_id"],
            "name": f"{_model_plain(record)} · {record.get('tier') or ''} · {hardest['accepted_steps']} steps",
            "starting_materials": hardest["starting_materials"],
            "products": hardest["products"],
            "verified_mechanism": {"steps": hardest["steps"]},
        }
        render_mechanism_png(case, base_dir / image_rel)
        record["hardest_solved"]["image"] = image_rel.as_posix()
        written.append(image_rel)
    path = RUNS_DIR / record_filename(record)
    (base_dir / path).parent.mkdir(parents=True, exist_ok=True)
    (base_dir / path).write_text(json.dumps(record, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    written.insert(0, path)
    return written


def load_records(base_dir: Path) -> List[Dict[str, Any]]:
    records = []
    for path in sorted((base_dir / RUNS_DIR).glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if data.get("schema") == RECORD_SCHEMA:
            data["_path"] = path.relative_to(base_dir).as_posix()
            records.append(data)
    return records


def _catalog_labels() -> Dict[str, str]:
    path = Path(__file__).with_name("model_pricing.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    models = data.get("models", data) if isinstance(data, dict) else {}
    labels: Dict[str, str] = {}
    for key, entry in models.items():
        if isinstance(entry, dict) and entry.get("label"):
            labels[_model_key(key)] = str(entry["label"])
    return labels


def _model_key(name: str) -> str:
    """Normalise ids so `anthropic/claude-opus-5.5` and `claude-opus-5-5` compare equal."""
    return str(name or "").split("/")[-1].split(" (")[0].strip().lower().replace(".", "-")


def _is_bridge(record: Dict[str, Any]) -> bool:
    origin = record.get("origin") or {}
    return origin.get("responder") == "agent-bridge" and bool(origin.get("declared_underlying_model"))


def _effective_model(record: Dict[str, Any]) -> str:
    """Model id that answered: the declared model for bridge runs, else the run's model."""
    name = (record.get("origin") or {}).get("declared_underlying_model") if _is_bridge(record) else record.get("model")
    return str(name)


def _model_plain(record: Dict[str, Any]) -> str:
    """Human label for the model that answered: catalog label when known, else the id."""
    name = _effective_model(record)
    return _catalog_labels().get(_model_key(str(name)), str(name).split(" (")[0])


def _model_label(record: Dict[str, Any]) -> str:
    return f"**{_model_plain(record)}**" + (" †" if _is_bridge(record) else "")


def _anchor(record: Dict[str, Any]) -> str:
    return _slug(f"{record.get('date')}-{record.get('run_group')}").lower().replace("_", "-").replace(".", "")


def is_quality_record(record: Dict[str, Any]) -> bool:
    """Scored by the current rubric (``quality_v2``). ``quality_v1``-only records are listed with
    the legacy scores until they are re-published."""
    return record.get("scoring") == QUALITY_V2_VERSION


def best_by_tier(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Best quality_v2 harness record per tier (baselines, holdout and legacy-scored records excluded)."""
    best: Dict[str, Dict[str, Any]] = {}
    for record in records:
        tier = str(record.get("tier") or "")
        if tier not in TIER_ORDER or record.get("holdout") or _is_baseline_record(record) or not is_quality_record(record):
            continue
        key = (int(record["summary"]["points"]), str(record.get("date") or ""))
        current = best.get(tier)
        if current is None or key > (int(current["summary"]["points"]), str(current.get("date") or "")):
            best[tier] = record
    return best


def _best_table(records: Sequence[Dict[str, Any]], *, link_prefix: str) -> List[str]:
    best = best_by_tier(records)
    lines = [
        "| Tier | Best model | Thinking | Quality | Valid steps | Passed | Harness | Date | Details |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for tier in TIER_ORDER:
        record = best.get(tier)
        if record is None:
            lines.append(f"| {tier} | — | — | — | — | — | — | — | — |")
            continue
        s = record["summary"]
        lines.append(
            f"| {tier} | {_model_label(record)} | {record.get('thinking_level') or 'default'} | **{s['points']}**/1000 | "
            f"{s['valid_step_fraction']:.0%} | {s['passed']}/{s['cases']} | `{record.get('harness')}` | "
            f"{record.get('date')} | [results]({link_prefix}#{_anchor(record)}) |"
        )
    return lines


def _hardest_overall(records: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    candidates = [r for r in records if (r.get("hardest_solved") or {}).get("image")]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda r: (int(r["hardest_solved"]["known_steps"]), int(r["hardest_solved"]["accepted_steps"]), str(r.get("date"))),
    )


BASELINES_ANCHOR = "harness-free-baselines"
COMPARISON_ANCHOR = "harness-vs-baseline"
BEST_BY_TIER_ANCHOR = "best-model-by-tier"
LEGACY_ANCHOR = "legacy-scores"
COMPONENT_LABELS = {
    "step_validity": "Step validity",
    "sequence": "Sequence",
    "electron_conservation": "Electron conservation",
    "proton_bookkeeping": "Proton sources/sinks",
    "protonation_states": "Protonation states",
    "reagents_and_solvent": "Reagents & solvent",
    "efficiency": "Efficiency",
    "intermolecular": "Intermolecular",
}
RUBRIC_TEXT = [
    "Every mechanism, harness or harness-free baseline, is scored by the same `quality_v2` rubric "
    "(`mechanistic_agent/quality_v2.py`, [docs/scoring_quality_v2.md](docs/scoring_quality_v2.md)). Each step is "
    "re-checked with the same deterministic code whatever produced it, and every measure is one axis from 0 to 1:",
    "",
    "| Axis | Measures |",
    "|---|---|",
    "| Product correct | all targets reached (product given) or the main product in any protonation state (product hidden) |",
    "| Step connectivity | each step starts where the path left off: no intermediate vanishes, no product or intermediate appears from nowhere (small species and solvents ignored) |",
    "| Balanced, progressing | per step: atom and charge balance, and the state changes |",
    "| Arrows + SMIRKS | per step: arrows parse into a bond/electron change and the SMIRKS matches the stated species |",
    "| Electron conservation | per step: the mapped SMIRKS conserves electrons in the bond-electron matrix |",
    "| Matches reference | heavy-atom events in the FlowER reference order (proton-shuttle agnostic) |",
    "",
    "**Score** = 1000 × (0.2 · product + 0.3 · connectivity + 0.5 · step quality), where step quality is the mean of "
    "the four step axes over the steps the mechanism made. Stopping early costs the product, not step quality; "
    "reaching the product through disconnected steps costs more than the product earns. A mechanism with no step "
    "(a **failed start**) scores 0 on every axis. **Area** is the radar polygon area (1 = perfect on every axis). "
    "A case passes when it also passes `quality_v1` (product, every step valid, mass and charge close, no repeated "
    "state), every step is connected, and it scores at least 700.",
]
BASELINE_NOTE = (
    "Baselines make one full-mechanism call with no harness. Their steps are scored exactly like harness steps, "
    "so an unbalanced, unparseable or disconnected step costs the same. `quality_v1` points stay in each record "
    "(`quality_v1`) for reference."
)


def _pct(value: Any) -> str:
    return f"{float(value or 0.0):.0%}"


def _comparison_table(records: Sequence[Dict[str, Any]]) -> List[str]:
    """Every quality_v2 record (harness and baseline), by tier then score."""
    rows = sorted(
        (r for r in records if is_quality_record(r)),
        key=lambda r: (
            BASELINE_TIER_ORDER.index(r["tier"]) if r.get("tier") in BASELINE_TIER_ORDER else 99,
            -int(r["summary"]["points"]),
            str(r.get("date") or ""),
        ),
    )
    if not rows:
        return ["No runs scored with `quality_v2` yet."]
    header = "| Tier | Model | Mode | Thinking | Cases | Score | Area | Failed starts | " + " | ".join(
        V2_AXIS_LABELS[k] for k in V2_AXES
    ) + " | Passed | Date | Record |"
    lines = [header, "|" + "---|" * (11 + len(V2_AXES))]
    for record in rows:
        s = record["summary"]
        mode = "baseline" if _is_baseline_record(record) else f"harness `{record.get('harness')}`"
        if s.get("products_hidden"):
            mode += " · no product"
        axes = " | ".join(f"{float(s['axes'].get(k, 0.0)):.2f}" for k in V2_AXES)
        link = f"[json]({record['_path']})" if record.get("_path") else "—"
        lines.append(
            f"| {record.get('tier') or '—'} | {_model_label(record)} | {mode} | {record.get('thinking_level') or 'default'} | "
            f"{s['cases']} | **{s['points']}** | {float(s['polygon_area']):.2f} | {_pct(s['failed_start_fraction'])} | "
            f"{axes} | {s['passed']}/{s['cases']} | {record.get('date')} | {link} |"
        )
    return lines


def _legacy_table(records: Sequence[Dict[str, Any]]) -> List[str]:
    rows = sorted(
        (r for r in records if not is_quality_record(r)),
        key=lambda r: (str(r.get("tier") or ""), str(r.get("date") or "")),
    )
    if not rows:
        return ["None."]
    lines = [
        "| Tier | Model | Mode | Thinking | Cases | Legacy points | Scoring | Date | Record |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for record in rows:
        s = record["summary"]
        mode = "baseline" if _is_baseline_record(record) else f"harness `{record.get('harness')}`"
        basis = "mean case score × 1000" if _is_baseline_record(record) else "Clawdiators-style rubric (incl. speed)"
        link = f"[json]({record['_path']})" if record.get("_path") else "—"
        lines.append(
            f"| {record.get('tier') or '—'} | {_model_label(record)} | {mode} | {record.get('thinking_level') or 'default'} | "
            f"{s.get('cases')} | {s.get('points')} | {basis}, case scoring `{record.get('scoring_version') or 'v2'}` | "
            f"{record.get('date')} | {link} |"
        )
    return lines


PROVENANCE_NOTE = (
    "† Answered through the [agent bridge](docs/agent_bridge.md): each model call went to the declared model in a "
    "fresh session that saw only the prompt (`responder_saw_ground_truth: false`). Cost is opaque, so these "
    "rows make no cost claim."
)


def render_leaderboard_markdown(records: Sequence[Dict[str, Any]]) -> str:
    ordered = sorted(
        (r for r in records if not _is_baseline_record(r) and is_quality_record(r)),
        key=lambda r: (TIER_ORDER.index(r["tier"]) if r.get("tier") in TIER_ORDER else 99, str(r.get("date"))),
    )
    lines = [
        "# Leaderboard",
        "",
        "Blind mechanism prediction on FlowER-derived development tiers (easy = 1–2 steps, medium = 3, hard = 4–10), "
        "10 cases per tier.",
        "",
        *RUBRIC_TEXT,
        "",
        "Generated by `python main.py publish-results` from the records in [`results/runs/`](results/runs/). "
        "Do not edit by hand.",
        "",
        f'<a id="{BEST_BY_TIER_ANCHOR}"></a>',
        "## Best model by tier",
        "",
        *_best_table(records, link_prefix=""),
        "",
        f'<a id="{COMPARISON_ANCHOR}"></a>',
        f'<a id="{BASELINES_ANCHOR}"></a>',
        "## Harness vs harness-free baseline",
        "",
        "Paired runs (same cases, model and thinking level):",
        "",
        *_paired_table(records, link_prefix=""),
        "",
        PAIRED_NOTE,
        "",
        "All `quality_v2` runs:",
        "",
        *_comparison_table(records),
        "",
        BASELINE_NOTE,
        "",
        PROVENANCE_NOTE,
        "",
        "## Published runs",
        "",
    ]
    for record in ordered:
        s = record["summary"]
        parts = ", ".join(f"{V2_AXIS_LABELS[k].lower()} {float(s['axes'].get(k, 0.0)):.2f}" for k in V2_AXES)
        lines += [
            f'<a id="{_anchor(record)}"></a>',
            f"### {record.get('tier') or 'eval'} · {_model_label(record)} · {record.get('date')}",
            "",
            f"- **{s['points']}/1000** (area {float(s['polygon_area']):.2f}, failed starts "
            f"{_pct(s['failed_start_fraction'])}) — {parts}",
            f"- Targets reached {s['targets_reached']}/{s['cases']}, passed {s['passed']}/{s['cases']}, "
            f"fully connected {s.get('fully_connected', 0)}/{s['cases']}, `quality_v1` "
            f"{(record.get('quality_v1') or {}).get('points', '—')}, "
            f"valid steps {s['valid_step_fraction']:.0%}, thinking `{record.get('thinking_level') or 'default'}`",
            f"- Model id `{record.get('model')}`, harness `{record.get('harness')}`, run group `{record.get('run_group')}`, eval run "
            f"`{record.get('eval_run_id')}`, commit `{record.get('git_commit')}` — [record]({record.get('_path')})",
        ]
        if record.get("sources"):
            srcs = ", ".join(f"`{src['eval_run_id'][:8]}` ({len(src['cases'])} cases)" for src in record["sources"])
            lines.append(f"- Combined from resumed eval runs: {srcs}")
        lines.append("")
        if record.get("cases"):
            lines += ["| Case | Known steps | Accepted steps | Valid steps | Connected | Target | Passed | Score |",
                      "|---|---|---|---|---|---|---|---|"]
            for case in record["cases"]:
                q = case.get("quality_points")
                lines.append(
                    f"| `{case['case_id']}` | {case['known_steps']} | {case['accepted_steps']} | {case.get('valid_steps') or '—'} | "
                    f"{case.get('connected_steps') or '—'} | "
                    f"{'✓' if case['target_reached'] else '✗'} | {'✓' if case['passed'] else '✗'} | "
                    f"{f'{q:.0f}' if q is not None else '—'} |"
                )
            lines.append("")
        hardest = record.get("hardest_solved") or {}
        if hardest.get("image"):
            lines += [
                f"Hardest mechanism solved: `{hardest['case_id']}` — reference {hardest['known_steps']} steps, "
                f"predicted {hardest['accepted_steps']} steps",
                "",
                f"![{hardest['case_id']}]({hardest['image']})",
                "",
            ]
    lines += [
        f'<a id="{LEGACY_ANCHOR}"></a>',
        "## Legacy scores",
        "",
        "Records published before `quality_v2`, or baselines run before their steps were saved, keep their old "
        "numbers here. They are not comparable with the tables above: harness rows used a Clawdiators-style "
        "1000-point rubric with product and speed points, and baseline rows used the mean case score × 1000.",
        "",
        *_legacy_table(records),
        "",
        "---",
        "",
        "Official holdout results and the historical Clawdiators arena material are in "
        "[docs/legacy/clawdiators_leaderboard.md](docs/legacy/clawdiators_leaderboard.md).",
    ]
    return "\n".join(lines) + "\n"


README_PAIRED_START = "<!-- harness-vs-baseline:start -->"
README_PAIRED_END = "<!-- harness-vs-baseline:end -->"


def paired_comparisons(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Harness vs harness-free baseline on the same cases: quality_v2 records that share an eval set,
    product mode and thinking level, latest of each kind per group."""
    groups: Dict[tuple, Dict[str, Dict[str, Any]]] = {}
    for record in records:
        if not is_quality_record(record) or record.get("holdout"):
            continue
        s = record["summary"]
        key = (str(record.get("eval_set_id") or ""), bool(s.get("products_hidden")),
               str(record.get("thinking_level") or "default"), _model_key(_effective_model(record)))
        kind = "baseline" if _is_baseline_record(record) else "harness"
        current = groups.setdefault(key, {}).get(kind)
        if current is None or str(record.get("date") or "") >= str(current.get("date") or ""):
            groups[key][kind] = record
    pairs = [
        {"eval_set_id": key[0], "products_hidden": key[1], "thinking_level": key[2], **kinds}
        for key, kinds in groups.items()
        if "baseline" in kinds and "harness" in kinds
    ]
    def family(pair: Dict[str, Any]) -> tuple:
        key = _model_key(_effective_model(pair["harness"]))
        return (next((i for i, f in enumerate(("opus", "sonnet", "haiku")) if f in key), 3), key)

    return sorted(pairs, key=lambda p: (p["harness"].get("tier") or "", p["products_hidden"], p["eval_set_id"], family(p)))


def radar_path(pair: Dict[str, Any]) -> Path:
    h = pair["harness"]
    name = (f"{_model_key(_effective_model(h))}_{(h.get('tier') or 'eval')}_{pair['eval_set_id'][:8]}_"
            f"{'hidden' if pair['products_hidden'] else 'given'}_{pair['thinking_level']}")
    return RADAR_DIR / f"{_slug(name)}.svg"


def _radar_note(record: Dict[str, Any]) -> str:
    s = record["summary"]
    return (f"score {s['points']} · area {float(s['polygon_area']):.2f}\n"
            f"failed starts {_pct(s['failed_start_fraction'])}")


def write_radars(base_dir: Path, records: Sequence[Dict[str, Any]]) -> List[Path]:
    """One radar SVG per harness/baseline pair (baseline orange, harness blue)."""
    from mechanistic_agent.quality_radar import BASELINE_COLOR, HARNESS_COLOR, radar_svg

    written: List[Path] = []
    for pair in paired_comparisons(records):
        b, h = pair["baseline"], pair["harness"]
        title = (f"{_model_plain(h)} · {h['summary']['cases']} {h.get('tier') or ''} · product "
                 f"{'hidden' if pair['products_hidden'] else 'given'} · {pair['thinking_level']} thinking")
        svg = radar_svg(
            [("One-shot baseline", BASELINE_COLOR, b["summary"]["axes"], _radar_note(b)),
             ("Harness", HARNESS_COLOR, h["summary"]["axes"], _radar_note(h))],
            title=title,
        )
        path = radar_path(pair)
        (base_dir / path).parent.mkdir(parents=True, exist_ok=True)
        (base_dir / path).write_text(svg, encoding="utf-8")
        written.append(path)
    return written


def _paired_table(records: Sequence[Dict[str, Any]], *, link_prefix: str) -> List[str]:
    pairs = paired_comparisons(records)
    if not pairs:
        return ["No paired harness / baseline runs published yet."]
    lines = [
        "| Cases | Product | Thinking | Model | Radar | One-shot baseline | Harness | Score Δ |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def cell(record: Dict[str, Any]) -> str:
        s = record["summary"]
        product = f"{s.get('product_correct', s['targets_reached'])}/{s['cases']}"
        return (f"**{s['points']}**<br>area {float(s['polygon_area']):.2f}<br>"
                f"failed starts {_pct(s['failed_start_fraction'])}<br>product {product}, "
                f"pass {s['passed']}/{s['cases']}")

    for pair in pairs:
        b, h = pair["baseline"], pair["harness"]
        delta = int(h["summary"]["points"]) - int(b["summary"]["points"])
        label = f"{h['summary']['cases']} {h.get('tier') or ''} ({h.get('eval_set_name') or pair['eval_set_id'][:8]})".strip()
        image = radar_path(pair).as_posix()
        lines.append(
            f"| {label} | {'hidden' if pair['products_hidden'] else 'given'} | {pair['thinking_level']} | "
            f"{_model_label(h)} | <img src=\"{image}\" width=\"260\" alt=\"radar\"> | {cell(b)} | {cell(h)} | "
            f"{delta:+d} |"
        )
    return lines


PAIRED_NOTE = (
    "Same cases, same model, same thinking level. Radar: one-shot baseline orange, harness blue, on the six "
    "`quality_v2` axes. **Score** is on 1000 (0.2 product + 0.3 step connectivity + 0.5 mean step quality); "
    "**area** is the radar polygon area (1 = perfect); **failed starts** is the share of cases where the "
    "mechanism made no step. With the product **hidden** the model also had to predict it."
)


def render_paired_block(records: Sequence[Dict[str, Any]]) -> str:
    return "\n".join([*_paired_table(records, link_prefix="LEADERBOARD.md"), "", PAIRED_NOTE])


def render_readme_block(records: Sequence[Dict[str, Any]]) -> str:
    lines = _best_table(records, link_prefix="LEADERBOARD.md")
    hardest = _hardest_overall(records)
    if hardest:
        h = hardest["hardest_solved"]
        lines += [
            "",
            f"Hardest mechanism solved so far: [`{h['case_id']}`, a {h['known_steps']}-step mechanism "
            f"({hardest.get('tier')} tier, {_model_plain(hardest)})]({h['image']}). "
            "Scored with the `quality_v2` rubric; harness vs baseline and legacy scores: [LEADERBOARD.md](LEADERBOARD.md).",
        ]
    lines += ["", PROVENANCE_NOTE]
    return "\n".join(lines)


def splice_block(text: str, start: str, end: str, block: str) -> str:
    if start not in text or end not in text:
        raise PublishError(f"markers {start} / {end} not found")
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{start}\n{block}\n{end}{tail}"


def regenerate_boards(base_dir: Path) -> List[Path]:
    """Rewrite LEADERBOARD.md and the README leaderboard block from committed records."""
    records = load_records(base_dir)
    written = write_radars(base_dir, records)
    (base_dir / LEADERBOARD_PATH).write_text(render_leaderboard_markdown(records), encoding="utf-8")
    written.append(LEADERBOARD_PATH)
    readme = base_dir / README_PATH
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        updated = text
        if README_LEADERBOARD_START in updated:
            updated = splice_block(updated, README_LEADERBOARD_START, README_LEADERBOARD_END, render_readme_block(records))
        if README_PAIRED_START in updated:
            updated = splice_block(updated, README_PAIRED_START, README_PAIRED_END, render_paired_block(records))
        if updated != text:
            readme.write_text(updated, encoding="utf-8")
            written.append(README_PATH)
    return written


def _run(cmd: List[str], cwd: Path) -> str:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PublishError(f"{' '.join(cmd)} failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return proc.stdout.strip()


PUBLISH_PATHS = [RESULTS_DIR.as_posix(), LEADERBOARD_PATH.as_posix(), README_PATH.as_posix()]


def pr_body(records: Sequence[Dict[str, Any]]) -> str:
    lines = ["## Published eval results", ""]
    for record in records:
        s = record["summary"]
        origin = record.get("origin") or {}
        if _is_baseline_record(record):
            lines.append(
                f"- **{record.get('tier')}** · `{record.get('model')}` · harness-free baseline — mean score "
                f"{s['points']}/1000, products {s['targets_reached']}/{s['cases']} · eval run "
                f"`{record.get('eval_run_id')}`"
                + (f" · declared model `{origin.get('declared_underlying_model')}`" if origin else "")
            )
            continue
        lines.append(
            f"- **{record.get('tier')}** · `{record.get('model')}` · harness `{record.get('harness')}` — "
            f"{s['points']}/1000 ({s['outcome']}), targets {s['targets_reached']}/{s['cases']}, "
            f"passed {s['passed']}/{s['cases']} · eval run `{record.get('eval_run_id')}`"
            + (f" · declared model `{origin.get('declared_underlying_model')}`, "
               f"saw ground truth: `{origin.get('responder_saw_ground_truth')}`" if origin else "")
        )
    lines += [
        "",
        "Generated by `python main.py publish-results --open-pr`. Only `results/`, `LEADERBOARD.md`, and the README "
        "leaderboard block change. Results only — no prompt, harness, or validator changes.",
    ]
    return "\n".join(lines)


def open_results_pr(
    base_dir: Path,
    records_writer: Callable[[], List[Dict[str, Any]]],
    *,
    branch: str,
    title: str,
    run: Callable[[List[str], Path], str] = _run,
) -> str:
    """Branch from origin/main, write results, commit, push, open a PR; return its URL."""
    dirty = run(["git", "status", "--porcelain", "--", *PUBLISH_PATHS], base_dir)
    if dirty:
        raise PublishError(f"uncommitted changes in {', '.join(PUBLISH_PATHS)}; commit or stash them first")
    run(["git", "fetch", "origin", "main"], base_dir)
    run(["git", "switch", "-c", branch, "origin/main"], base_dir)
    records = records_writer()
    run(["git", "add", "--", *PUBLISH_PATHS], base_dir)
    run(["git", "commit", "-m", title], base_dir)
    run(["git", "push", "-u", "origin", branch], base_dir)
    return run(["gh", "pr", "create", "--base", "main", "--title", title, "--body", pr_body(records)], base_dir)
