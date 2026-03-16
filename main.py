"""CLI entrypoint for the local-first mechanistic runtime."""
from __future__ import annotations

import json
from datetime import datetime
import sqlite3
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import typer

from mechanistic_agent import OPTIONAL_LLM_TOOL_NAMES, ReactionInputs
from mechanistic_agent.core import RegistrySet, RunCoordinator, RunStore, select_step_models
from mechanistic_agent.core.overnight_ralph import OvernightRalphOrchestrator, load_overnight_program
from mechanistic_agent.curriculum import (
    OPUS_MODEL,
    build_curriculum_status,
    curriculum_history,
    publish_curriculum_release,
    publish_due_curriculum_releases,
    render_curriculum_readme,
    render_launchd_plist,
    submit_curriculum_release,
)
from mechanistic_agent.model_registry import (
    get_default_model,
    get_model_family,
    resolve_model_key,
    to_internal_reasoning_level,
)
from mechanistic_agent.eval_set_resolution import (
    EvalSetResolutionError,
    case_ids_hash,
    resolve_eval_set,
    select_eval_cases,
)

try:  # pragma: no cover - optional helper
    from dotenv import load_dotenv
    import os
except ImportError:  # pragma: no cover - fallback when python-dotenv is absent
    def load_dotenv(_: Path) -> bool:  # type: ignore[override]
        return False
    import os  # fallback import


_CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
app = typer.Typer(add_completion=False, no_args_is_help=True, context_settings=_CONTEXT_SETTINGS)
curriculum_app = typer.Typer(add_completion=False, no_args_is_help=True, context_settings=_CONTEXT_SETTINGS)
app.add_typer(curriculum_app, name="curriculum")
load_dotenv(Path.cwd() / ".env")


def _load_api_keys() -> Dict[str, str]:
    """Load API keys from environment, preferring dotenv values."""
    keys = {}

    # Load from environment variables, which may be set by dotenv
    env_mappings = {
        'OPENAI_API_KEY': 'openai_api_key',
        'OPENROUTER_API_KEY': 'openrouter_api_key',
        'ANTHROPIC_API_KEY': 'anthropic_api_key',
        'GOOGLE_API_KEY': 'google_api_key',
        'OPENAI_ADMIN_KEY': 'openai_admin_key',
    }

    for env_var, key_name in env_mappings.items():
        value = os.environ.get(env_var)
        if value:
            keys[key_name] = value

    # Special handling: if OPENROUTER_API_KEY is not set but ANTHROPIC_API_KEY is,
    # use ANTHROPIC_API_KEY as fallback for OPENROUTER_API_KEY
    if 'openrouter_api_key' not in keys and 'anthropic_api_key' in keys:
        keys['openrouter_api_key'] = keys['anthropic_api_key']

    return keys


def _parse_materials(raw: Optional[str], fallback: List[str]) -> List[str]:
    if raw is None:
        return list(fallback)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _canonicalize_model_name_or_raise(model_name: str) -> str:
    try:
        return resolve_model_key(model_name)
    except ValueError as exc:
        raise typer.BadParameter(f"Unsupported model '{model_name}'") from exc


def _first_text_value(*values: Any) -> Optional[str]:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _validation_error_text(validation: Any) -> Optional[str]:
    if not isinstance(validation, dict):
        return None
    direct = _first_text_value(validation.get("error"), validation.get("message"))
    if direct:
        return direct
    for check in validation.get("checks") or []:
        if not isinstance(check, dict):
            continue
        details = check.get("details")
        if not isinstance(details, dict):
            continue
        detail_text = _first_text_value(details.get("error"), details.get("message"))
        if detail_text:
            return detail_text
    return None


def _extract_eval_run_diagnostics(snapshot: Dict[str, Any]) -> Dict[str, Optional[str]]:
    run_status = _first_text_value(snapshot.get("status"))
    failure_reason: Optional[str] = None
    first_step_error: Optional[str] = None

    events = list(snapshot.get("events") or [])
    for event in reversed(events):
        if str(event.get("event_type") or "") != "run_failed":
            continue
        payload = event.get("payload") or {}
        if isinstance(payload, dict):
            failure_reason = _first_text_value(
                payload.get("reason"),
                payload.get("error"),
                payload.get("message"),
            )
        break

    for row in snapshot.get("step_outputs") or []:
        if str(row.get("source") or "") != "llm":
            continue
        step_name = _first_text_value(row.get("step_name")) or "unknown_step"
        output = row.get("output")
        if not isinstance(output, dict):
            output = {}
        error_text = _first_text_value(output.get("error"))
        if error_text is None and str(output.get("status") or "").lower() in {"failed", "no_response"}:
            error_text = _first_text_value(output.get("message"), output.get("note"), output.get("rationale"))
        if error_text is None:
            error_text = _validation_error_text(row.get("validation"))
        if error_text:
            first_step_error = f"{step_name}: {error_text}"
            break

    if first_step_error is None:
        for event in events:
            payload = event.get("payload") or {}
            if not isinstance(payload, dict):
                continue
            error_text = _first_text_value(payload.get("error"), payload.get("message"))
            if error_text is None:
                error_text = _validation_error_text(payload.get("validation"))
            if error_text:
                step_name = _first_text_value(event.get("step_name"), payload.get("step_name"))
                first_step_error = f"{step_name}: {error_text}" if step_name else error_text
                break

    return {
        "run_status": run_status,
        "failure_reason": failure_reason,
        "first_step_error": first_step_error,
    }


def _extract_chemistry_backend_diagnostics(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    events = list(snapshot.get("events") or [])
    for event in reversed(events):
        if str(event.get("event_type") or "") != "chemistry_backend_summary":
            continue
        payload = event.get("payload")
        if isinstance(payload, dict):
            return {
                "chemistry_backend": payload,
                "chemistry_backend_source": "event",
            }

    backend_used_counts: Dict[str, int] = {}
    backend_requested_counts: Dict[str, int] = {}
    fallback_count = 0
    fallback_reasons: Dict[str, int] = {}
    rdkit_cli_error_counts: Dict[str, int] = {}
    first_rdkit_cli_error: Optional[Dict[str, Any]] = None
    calls = 0

    for row in snapshot.get("step_outputs") or []:
        output = row.get("output") if isinstance(row.get("output"), dict) else {}
        details = output.get("details") if isinstance(output.get("details"), dict) else {}
        backend_meta = details.get("chemistry_backend") if isinstance(details.get("chemistry_backend"), dict) else None
        if not isinstance(backend_meta, dict):
            continue
        calls += 1
        backend_used = str(backend_meta.get("backend_used") or "python")
        backend_requested = str(backend_meta.get("backend_requested") or "auto")
        backend_used_counts[backend_used] = backend_used_counts.get(backend_used, 0) + 1
        backend_requested_counts[backend_requested] = backend_requested_counts.get(backend_requested, 0) + 1

        if bool(backend_meta.get("fallback_used")):
            fallback_count += 1
            reason = str(backend_meta.get("fallback_reason") or "unknown")
            fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1

        error_code = str(backend_meta.get("rdkit_cli_error_code") or "").strip()
        error_text = str(backend_meta.get("rdkit_cli_error") or "").strip()
        key = error_code or (error_text[:120] if error_text else "")
        if key:
            rdkit_cli_error_counts[key] = rdkit_cli_error_counts.get(key, 0) + 1
            if first_rdkit_cli_error is None:
                first_rdkit_cli_error = {
                    "code": error_code or None,
                    "message": error_text or None,
                    "step_name": row.get("step_name"),
                }

    if calls == 0:
        return {"chemistry_backend": {}, "chemistry_backend_source": "none"}
    return {
        "chemistry_backend": {
            "calls": calls,
            "backend_used_counts": backend_used_counts,
            "backend_requested_counts": backend_requested_counts,
            "fallback_count": fallback_count,
            "fallback_reasons": fallback_reasons,
            "rdkit_cli_error_counts": rdkit_cli_error_counts,
            "first_rdkit_cli_error": first_rdkit_cli_error,
        },
        "chemistry_backend_source": "step_outputs",
    }


def _build_eval_case_summary(
    *,
    snapshot: Dict[str, Any],
    score: float,
    passed: bool,
    step_outputs: List[Dict[str, Any]],
    case_step_count: Optional[int],
    subagent_scores: Dict[str, Any],
    scored_error: Optional[str] = None,
) -> Dict[str, Any]:
    diagnostics = _extract_eval_run_diagnostics(snapshot)
    summary: Dict[str, Any] = {
        "score": score,
        "passed": passed,
        "step_count": len([s for s in step_outputs if s.get("step_name") == "mechanism_synthesis"]),
        "n_mechanistic_steps": case_step_count,
        "error": scored_error or diagnostics.get("first_step_error") or diagnostics.get("failure_reason"),
        "eval_mode": "harness",
        "subagent_scores": subagent_scores,
    }
    summary.update(diagnostics)
    summary.update(_extract_chemistry_backend_diagnostics(snapshot))
    return summary


def _format_eval_case_result_line(
    *,
    index: int,
    case_id: str,
    score: float,
    passed: bool,
    total_cost: float,
    latency_ms: Optional[float],
    summary: Dict[str, Any],
) -> str:
    latency_text = ""
    if latency_ms is not None:
        latency_text = f" latency={latency_ms/1000:.1f}s"
    line = f"  [{index}] {case_id}: score={score:.3f} passed={passed} cost=${total_cost:.3f}{latency_text}"
    chemistry = summary.get("chemistry_backend") if isinstance(summary.get("chemistry_backend"), dict) else {}
    if chemistry:
        backend_used_counts = chemistry.get("backend_used_counts")
        if isinstance(backend_used_counts, dict) and backend_used_counts:
            primary = max(backend_used_counts.items(), key=lambda item: item[1])[0]
            line += f" backend={primary}"
        if int(chemistry.get("fallback_count") or 0) > 0:
            line += f" fallback={int(chemistry.get('fallback_count') or 0)}"
        error_counts = chemistry.get("rdkit_cli_error_counts")
        if isinstance(error_counts, dict) and error_counts:
            top_error = max(error_counts.items(), key=lambda item: item[1])[0]
            line += f" rdkit_cli_error={top_error}"
    if str(summary.get("run_status") or "") != "failed":
        return line
    detail = _first_text_value(summary.get("failure_reason"), summary.get("first_step_error"))
    if detail:
        return f"{line} status=failed reason={detail}"
    return f"{line} status=failed"


def _historical_cost_stats(db_path: Path) -> Dict[str, float]:
    if not db_path.exists():
        return {"avg_nonzero": 0.0, "p90_nonzero": 0.0, "max_nonzero": 0.0}
    query = """
    WITH totals AS (
      SELECT run_id, SUM(COALESCE(json_extract(cost_json, '$.total_cost'), 0)) AS cost
      FROM step_outputs
      GROUP BY run_id
      HAVING cost > 0
    ),
    ordered AS (
      SELECT cost, ROW_NUMBER() OVER (ORDER BY cost) AS rn, COUNT(*) OVER() AS cnt
      FROM totals
    )
    SELECT
      COALESCE((SELECT AVG(cost) FROM totals), 0.0) AS avg_nonzero,
      COALESCE((SELECT MAX(cost) FROM totals), 0.0) AS max_nonzero,
      COALESCE(
        (
          SELECT cost
          FROM ordered
          WHERE rn = CAST(cnt * 0.9 AS INT)
          LIMIT 1
        ),
        0.0
      ) AS p90_nonzero
    """
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute(query).fetchone()
        finally:
            conn.close()
    except Exception:
        return {"avg_nonzero": 0.0, "p90_nonzero": 0.0, "max_nonzero": 0.0}
    if not row:
        return {"avg_nonzero": 0.0, "p90_nonzero": 0.0, "max_nonzero": 0.0}
    return {
        "avg_nonzero": float(row[0] or 0.0),
        "max_nonzero": float(row[1] or 0.0),
        "p90_nonzero": float(row[2] or 0.0),
    }


def _filter_leaderboard_rows(items: List[Dict[str, object]], *, completed_only: bool) -> List[Dict[str, object]]:
    if not completed_only:
        return items
    return [item for item in items if str(item.get("status") or "").lower() == "completed"]


# ── Clawdiators 1000-pt speed calibration ──────────────────────────────────
# Set HARNESS_SPEED_CALIBRATION_MS to the per-case average latency (in ms)
# observed during the benchmark dry-run. That run defines 75 pts.
# Formula: T_max = 4 × HARNESS_SPEED_CALIBRATION_MS
#   At calibration latency → speed_pts = 75  (benchmark)
#   At 0 ms               → speed_pts = 100
#   At T_max ms           → speed_pts = 0
# Calibrated to 100s/case benchmark (400s max).
HARNESS_SPEED_CALIBRATION_MS: int = 100_000


def _latency_to_speed_pts(avg_latency_ms: float) -> int:
    """Convert average per-case latency to a 0-100 speed score.

    Uses HARNESS_SPEED_CALIBRATION_MS as the anchor: opus-4.6 benchmark latency → 75 pts.
    Returns 100 when uncalibrated (HARNESS_SPEED_CALIBRATION_MS == 0).
    """
    if HARNESS_SPEED_CALIBRATION_MS <= 0:
        return 100  # uncalibrated — full credit until benchmark is set
    t_max = HARNESS_SPEED_CALIBRATION_MS * 4.0
    return round(max(0.0, 1.0 - avg_latency_ms / t_max) * 100)


def _graded_to_clawdiators_pts(
    all_graded: List[Dict[str, Any]],
    all_latencies_ms: List[float],
) -> Dict[str, Any]:
    """Convert harness graded dicts + per-case latencies to a clawdiators 1000-pt breakdown.

    Rubric mapping (harness proxies):
      Product Accuracy (30%)  ← final_product_reached count / total
      Pathway Coverage (30%)  ← known_alignment_component avg
      Electron Push Quality (20%) ← step_validity_component avg (validation+mapping proxy)
      Speed (10%)             ← per-case wall-clock latency via _latency_to_speed_pts()
      Methodology (10%)       ← always 100 pts in harness mode (methodology always present)
    Anti-gaming gate: if product_ratio == 0, pathway/push/speed are all zeroed.
    """
    n = len(all_graded)
    n_hit = sum(1 for g in all_graded if g.get("final_product_reached", False))
    product_ratio = n_hit / n
    pathway_ratio = sum(g.get("known_alignment_component", 0.0) for g in all_graded) / n
    push_ratio    = sum(g.get("step_validity_component", 0.0) for g in all_graded) / n
    avg_lat       = sum(all_latencies_ms) / n if all_latencies_ms else 0.0

    if product_ratio == 0.0:
        pathway_ratio = push_ratio = 0.0
        speed_pts = 0
    else:
        speed_pts = _latency_to_speed_pts(avg_lat)

    pts: Dict[str, Any] = {
        "product":        round(product_ratio * 300),
        "pathway":        round(pathway_ratio * 300),
        "push":           round(push_ratio * 200),
        "speed":          speed_pts,
        "methodology":    100,
        "avg_latency_ms": round(avg_lat, 1),
        "n_total":        n,
        "n_product_hit":  n_hit,
    }
    pts["total"]   = pts["product"] + pts["pathway"] + pts["push"] + pts["speed"] + pts["methodology"]
    pts["outcome"] = "WIN" if pts["total"] >= 700 else ("DRAW" if pts["total"] >= 400 else "LOSS")
    return pts


def _leaderboard_row_to_pts(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a leaderboard DB row (0.0-1.0 metrics) to a clawdiators 1000-pt display dict.

    This is an approximation: the DB stores mean_quality_score and deterministic_pass_rate
    but not the full component breakdown. Use _graded_to_clawdiators_pts() during eval for
    the authoritative per-component breakdown.
    """
    quality   = float(row.get("weighted_quality_score") or row.get("mean_quality_score") or 0.0)
    pass_rate = float(row.get("weighted_pass_rate") or row.get("deterministic_pass_rate") or 0.0)
    avg_lat   = float(row.get("avg_latency_ms") or 0.0)
    product_pts     = round(pass_rate * 300)
    pathway_pts     = round(quality * 300)
    push_pts        = round(quality * 200)
    speed_pts       = _latency_to_speed_pts(avg_lat) if pass_rate > 0 else 0
    methodology_pts = 100
    if pass_rate == 0.0:
        pathway_pts = push_pts = speed_pts = 0
    total   = product_pts + pathway_pts + push_pts + speed_pts + methodology_pts
    outcome = "WIN" if total >= 700 else ("DRAW" if total >= 400 else "LOSS")
    return {"total": total, "outcome": outcome}


def _render_leaderboard_markdown(
    eval_set_id: str,
    items: List[Dict[str, object]],
    *,
    generated_at: Optional[str] = None,
) -> str:
    timestamp = generated_at or time.strftime("%Y-%m-%d %H:%M:%S")
    uses_weighted = any(str(item.get("aggregate_weighting") or "") for item in items)
    includes_cost = any("total_cost" in item for item in items)
    ranking_text = (
        "- Ranking order: weighted quality score, then weighted pass rate, then lower total cost."
        if uses_weighted
        else "- Ranking order: mean quality score, then deterministic pass rate, then lower total cost."
    )
    lines = [
        "# Mechanistic Agent Leaderboard",
        "",
        f"- Eval set ID: `{eval_set_id}`",
        f"- Generated at: `{timestamp}`",
        ranking_text,
        "- Current SOTA: the rank 1 completed row in the table below for the eval scope you care about.",
        "",
        "## PR Acceptance Rule",
        "",
        "A PR is only mergeable if it improves the relevant leaderboard gate for its contribution track.",
        "Single-reaction submissions are explicitly excluded from merge gates; they are review inputs only.",
        "",
    ]
    if items:
        top = items[0]
        top_model = str(top.get("model_name") or top.get("model") or "unknown")
        top_thinking = str(top.get("thinking_level") or "none")
        top_pts = _leaderboard_row_to_pts(top)
        top_pass_rate = float(top.get("weighted_pass_rate") or top.get("deterministic_pass_rate") or 0.0) * 100.0
        top_group = str(top.get("run_group_name") or "n/a")
        lines.extend(
            [
                "## Current SOTA",
                "",
                f"- Model: `{top_model}`",
                f"- Thinking: `{top_thinking}`",
                f"- Score: `{top_pts['total']}/1000` ({top_pts['outcome']})",
                f"- Deterministic pass rate: `{top_pass_rate:.1f}%`",
            ]
        )
        if includes_cost:
            top_cost = float(top.get("total_cost") or 0.0)
            lines.append(f"- Total cost: `${top_cost:.3f}`")
        lines.extend(
            [
                f"- Run group: `{top_group}`",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Current SOTA",
                "",
                "No completed leaderboard rows exist yet for this eval set.",
                "",
            ]
        )
    lines.extend(
        [
            "## Completed Runs",
            "",
        ]
    )
    if includes_cost:
        lines.extend(
            [
                "| Rank | Model | Thinking | Type | Score | Outcome | Pass | Cases | Cost | Group |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
    else:
        lines.extend(
            [
                "| Rank | Model | Thinking | Type | Score | Outcome | Pass | Cases | Group |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
    if not items:
        if includes_cost:
            lines.append("| - | - | - | - | - | - | - | - | - | No completed rows |")
        else:
            lines.append("| - | - | - | - | - | - | - | - | No completed rows |")
        return "\n".join(lines)

    for index, row in enumerate(items, 1):
        model = str(row.get("model_name") or row.get("model") or "unknown")
        thinking = str(row.get("thinking_level") or "none")
        run_type = "Baseline" if row.get("is_baseline") else "Harness"
        pts = _leaderboard_row_to_pts(row)
        score_display = f"{pts['total']}/1000"
        outcome = pts["outcome"]
        pass_rate = f"{float(row.get('weighted_pass_rate') or row.get('deterministic_pass_rate') or 0.0) * 100.0:.1f}%"
        case_count = str(row.get("case_count") or 0)
        group = str(row.get("run_group_name") or "n/a")
        if includes_cost:
            total_cost = float(row.get("total_cost") or 0.0)
            cost_display = f"${total_cost:.3f}"
            lines.append(
                f"| {index} | `{model}` | `{thinking}` | {run_type} | {score_display} | {outcome} | {pass_rate} | {case_count} | {cost_display} | `{group}` |"
            )
        else:
            lines.append(
                f"| {index} | `{model}` | `{thinking}` | {run_type} | {score_display} | {outcome} | {pass_rate} | {case_count} | `{group}` |"
            )
    return "\n".join(lines)


def _eval_case_step_count(case: Dict[str, Any]) -> Optional[int]:
    expected = case.get("expected") or {}
    if isinstance(expected, dict):
        direct = expected.get("n_mechanistic_steps")
        if isinstance(direct, int):
            return int(direct)
        if isinstance(direct, float):
            return int(direct)
        known = expected.get("known_mechanism") or expected.get("verified_mechanism")
        if isinstance(known, dict):
            min_steps = known.get("min_steps")
            if isinstance(min_steps, int):
                return int(min_steps)
            if isinstance(min_steps, float):
                return int(min_steps)
            steps = known.get("steps")
            if isinstance(steps, list):
                return len(steps)
    return None


BASELINE_TIER_NAMES: tuple[str, str, str] = ("easy", "medium", "hard")
BASELINE_TIER_MAP_DEFAULT_PATH: Path = Path("training_data") / "baseline_tier_eval_set_map.json"
BASELINE_TIER_DEFINITIONS_DEFAULT_PATH: Path = Path("training_data") / "baseline_tiers_clawdiator.json"
EVAL_TIERS_DEFAULT_PATH: Path = Path("training_data") / "eval_tiers.json"
DEVELOPMENT_LEADERBOARD_POLICY_DEFAULT_PATH: Path = (
    Path("training_data") / "development_leaderboard_policy.json"
)


def _load_eval_tier_ids(tier_file: Path) -> Dict[str, List[str]]:
    """Load and validate tier IDs from a tier-definition JSON file."""
    if not tier_file.exists():
        raise typer.BadParameter(f"Tier file not found: {tier_file}")
    try:
        raw = json.loads(tier_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"Invalid JSON in {tier_file}: {exc}") from exc
    if not isinstance(raw, dict):
        raise typer.BadParameter(f"{tier_file} must be a JSON object")

    resolved: Dict[str, List[str]] = {}
    for tier_name in BASELINE_TIER_NAMES:
        raw_ids = raw.get(tier_name)
        if not isinstance(raw_ids, list):
            raise typer.BadParameter(f"{tier_file} is missing tier list '{tier_name}'")
        resolved[tier_name] = [str(item) for item in raw_ids if str(item)]
    return resolved


def _resolve_baseline_tier_definitions_path(
    *,
    base: Path,
    override_path: Optional[str],
) -> Path:
    """Resolve the tier case-id source file for baseline tier mode."""
    if override_path:
        return Path(override_path).expanduser().resolve()
    clawdiator_path = (base / BASELINE_TIER_DEFINITIONS_DEFAULT_PATH).resolve()
    if clawdiator_path.exists():
        return clawdiator_path
    return (base / EVAL_TIERS_DEFAULT_PATH).resolve()


def _normalize_requested_baseline_tiers(
    requested_tiers: Optional[Sequence[str]],
    *,
    all_tiers: bool,
) -> List[str]:
    """Normalize --tier / --all-tiers input while preserving user order."""
    if all_tiers:
        return list(BASELINE_TIER_NAMES)

    normalized: List[str] = []
    for item in requested_tiers or []:
        value = str(item or "").strip().lower()
        if not value:
            continue
        if value not in BASELINE_TIER_NAMES:
            raise typer.BadParameter("tier must be one of: easy, medium, hard")
        if value not in normalized:
            normalized.append(value)
    return normalized


def _load_baseline_tier_eval_set_map(path: Path) -> Dict[str, str]:
    """Load tier -> eval_set_id mapping from JSON."""
    if not path.exists():
        raise typer.BadParameter(
            f"Tier map file not found: {path}. "
            "Create it or pass --tier-map-path to a valid JSON mapping file."
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"Invalid JSON in tier map file {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise typer.BadParameter(f"Tier map file must be a JSON object: {path}")

    missing_tiers: List[str] = []
    resolved: Dict[str, str] = {}
    for tier_name in BASELINE_TIER_NAMES:
        tier_payload = raw.get(tier_name)
        if not isinstance(tier_payload, dict):
            missing_tiers.append(tier_name)
            continue
        eval_set_id = str(tier_payload.get("eval_set_id") or "").strip()
        if not eval_set_id:
            raise typer.BadParameter(
                f"Tier map file {path} must define a non-empty {tier_name}.eval_set_id"
            )
        resolved[tier_name] = eval_set_id
    if missing_tiers:
        raise typer.BadParameter(
            f"Tier map file {path} is missing required tiers: {', '.join(missing_tiers)}"
        )
    return resolved


def _load_development_leaderboard_policy(base: Path) -> Dict[str, Any]:
    path = (base / DEVELOPMENT_LEADERBOARD_POLICY_DEFAULT_PATH).resolve()
    if not path.exists():
        raise typer.BadParameter(
            f"Development leaderboard policy file not found: {path}"
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"Invalid JSON in development leaderboard policy {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise typer.BadParameter(f"Development leaderboard policy must be a JSON object: {path}")

    tier_order = list(raw.get("tier_order") or BASELINE_TIER_NAMES)
    if tier_order != list(BASELINE_TIER_NAMES):
        raise typer.BadParameter(
            "Development leaderboard policy tier_order must be easy, medium, hard"
        )
    initial = raw.get("initial_qualifying") or {}
    if str(initial.get("tier") or "").strip().lower() not in BASELINE_TIER_NAMES:
        raise typer.BadParameter("Development leaderboard policy initial_qualifying.tier is invalid")
    try:
        initial_case_count = int(initial.get("case_count") or 0)
    except Exception as exc:
        raise typer.BadParameter(
            "Development leaderboard policy initial_qualifying.case_count must be an integer"
        ) from exc
    if initial_case_count < 1:
        raise typer.BadParameter("Development leaderboard policy initial_qualifying.case_count must be >= 1")

    next_case_count = int((raw.get("next_tier") or {}).get("case_count") or 0)
    extend_increment = int((raw.get("extend") or {}).get("case_increment") or 0)
    if next_case_count < 1 or extend_increment < 1:
        raise typer.BadParameter("Development leaderboard policy next_tier.case_count and extend.case_increment must be >= 1")

    source_defs = raw.get("tier_definition_sources") or {}
    if not isinstance(source_defs, dict):
        raise typer.BadParameter("Development leaderboard policy tier_definition_sources must be an object")
    active_sources = raw.get("active_tier_sources") or {}
    if not isinstance(active_sources, dict):
        raise typer.BadParameter("Development leaderboard policy active_tier_sources must be an object")

    normalized_source_defs: Dict[str, Dict[str, Any]] = {}
    for source_name, payload in source_defs.items():
        if not isinstance(payload, dict):
            raise typer.BadParameter(
                f"Development leaderboard policy source '{source_name}' must be an object"
            )
        rel_path = str(payload.get("path") or "").strip()
        if not rel_path:
            raise typer.BadParameter(
                f"Development leaderboard policy source '{source_name}' must define path"
            )
        normalized_source_defs[str(source_name)] = {
            "path": rel_path,
        }

    for tier_name in BASELINE_TIER_NAMES:
        source_name = str(active_sources.get(tier_name) or "").strip()
        if source_name not in normalized_source_defs:
            raise typer.BadParameter(
                f"Development leaderboard policy tier '{tier_name}' references unknown source '{source_name}'"
            )

    return {
        "path": str(path),
        "version": int(raw.get("version") or 1),
        "description": str(raw.get("description") or "").strip(),
        "tier_order": tier_order,
        "initial_qualifying": {
            "tier": str(initial.get("tier") or "").strip().lower(),
            "case_count": initial_case_count,
        },
        "extend": {"case_increment": extend_increment},
        "next_tier": {"case_count": next_case_count},
        "tier_definition_sources": normalized_source_defs,
        "active_tier_sources": {
            tier_name: str(active_sources.get(tier_name) or "").strip()
            for tier_name in BASELINE_TIER_NAMES
        },
        "comparison_scope": str(raw.get("comparison_scope") or "model_name+thinking_level"),
    }


def _resolve_development_tier_contexts(
    *,
    base: Path,
    store: RunStore,
    tier_map_path: Optional[str],
    tier_definitions_path: Optional[str],
    allow_holdout: bool,
) -> Dict[str, Dict[str, Any]]:
    policy = _load_development_leaderboard_policy(base)
    resolved_tier_map_path = (
        Path(tier_map_path).expanduser().resolve()
        if tier_map_path
        else (base / BASELINE_TIER_MAP_DEFAULT_PATH).resolve()
    )
    tier_eval_set_ids = _load_baseline_tier_eval_set_map(resolved_tier_map_path)

    loaded_sources: Dict[str, Dict[str, Any]] = {}
    for source_name, payload in policy["tier_definition_sources"].items():
        if tier_definitions_path:
            path = Path(tier_definitions_path).expanduser().resolve()
        else:
            path = (base / str(payload.get("path") or "")).resolve()
        loaded_sources[source_name] = {
            "path": str(path),
            "tiers": _load_eval_tier_ids(path),
        }

    contexts: Dict[str, Dict[str, Any]] = {}
    for tier_name in policy["tier_order"]:
        eval_set_id = str(tier_eval_set_ids.get(tier_name) or "").strip()
        if not eval_set_id:
            raise typer.BadParameter(f"Tier '{tier_name}' has no configured eval_set_id in tier map")
        try:
            resolved_eval_set = resolve_eval_set(
                store=store,
                requested_eval_set_id=eval_set_id,
            )
        except EvalSetResolutionError as exc:
            raise typer.BadParameter(
                f"Tier '{tier_name}' references unknown eval_set_id '{eval_set_id}': {exc}"
            ) from exc
        if resolved_eval_set.purpose == "leaderboard_holdout" and not allow_holdout:
            raise typer.BadParameter(
                f"Tier '{tier_name}' references holdout eval set '{eval_set_id}', "
                "which is not allowed for development eval planning."
            )

        source_name = str(policy["active_tier_sources"].get(tier_name) or "")
        source_payload = loaded_sources[source_name]
        tier_case_ids = list((source_payload.get("tiers") or {}).get(tier_name) or [])
        by_id = {
            str(case.get("case_id") or ""): case
            for case in resolved_eval_set.cases
            if str(case.get("case_id") or "")
        }
        selected_case_ids = [case_id for case_id in tier_case_ids if case_id in by_id]
        contexts[tier_name] = {
            "tier": tier_name,
            "eval_set_id": eval_set_id,
            "resolved_eval_set": resolved_eval_set,
            "source_name": source_name,
            "source_path": str(source_payload["path"]),
            "case_ids": selected_case_ids,
            "case_count": len(selected_case_ids),
        }

    return {
        "policy": policy,
        "tiers": contexts,
        "tier_map_path": str(resolved_tier_map_path),
    }


def _summarize_eval_run_results(
    *,
    run: Dict[str, Any],
    results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    scores = [float(item["score"]) for item in results if isinstance(item.get("score"), (int, float))]
    if not scores:
        return {}
    passes = [item.get("pass_bool") for item in results if item.get("pass_bool") is not None]
    pass_rate = (sum(1 for item in passes if item) / len(passes)) if passes else 0.0
    total_cost = 0.0
    for item in results:
        cost = item.get("cost")
        if not isinstance(cost, dict):
            continue
        value = cost.get("total_cost")
        if isinstance(value, (int, float)):
            total_cost += float(value)
    case_ids = [
        str(item.get("case_id") or "").strip()
        for item in results
        if str(item.get("case_id") or "").strip()
    ]
    return {
        "eval_run_id": str(run.get("id") or ""),
        "run_group_name": str(run.get("run_group_name") or ""),
        "status": str(run.get("status") or ""),
        "created_at": float(run.get("created_at") or 0.0),
        "mean_quality_score": sum(scores) / len(scores),
        "deterministic_pass_rate": pass_rate,
        "total_cost": total_cost,
        "case_count": len(case_ids),
        "case_ids": case_ids,
        "metadata": dict(run.get("metadata") or {}),
    }


def _planner_sort_key(row: Dict[str, Any]) -> tuple[float, float, float]:
    return (
        -float(row.get("mean_quality_score") or 0.0),
        -float(row.get("deterministic_pass_rate") or 0.0),
        float(row.get("total_cost") or 0.0),
    )


def _planner_case_ids_match_canonical_prefix(
    *,
    run_summary: Dict[str, Any],
    tier_name: str,
    canonical_case_ids: Sequence[str],
) -> bool:
    metadata = run_summary.get("metadata") or {}
    selected_count = int(metadata.get("selected_case_count") or 0) if metadata else 0
    selected_hash = str(metadata.get("selected_case_ids_hash") or "") if metadata else ""
    metadata_tier = str(metadata.get("tier_name") or "") if metadata else ""
    if selected_count > 0 and selected_hash and metadata_tier == tier_name:
        expected_case_ids = list(canonical_case_ids[:selected_count])
        return (
            len(expected_case_ids) == selected_count
            and bool(metadata.get("is_policy_canonical_slice"))
            and selected_hash == case_ids_hash(expected_case_ids)
        )

    run_case_ids = {
        str(case_id).strip()
        for case_id in (run_summary.get("case_ids") or [])
        if str(case_id).strip()
    }
    if not run_case_ids:
        return False
    expected_case_ids = list(canonical_case_ids[: len(run_case_ids)])
    return len(expected_case_ids) == len(run_case_ids) and set(expected_case_ids) == run_case_ids


def _collect_development_canonical_rows(
    *,
    store: RunStore,
    tier_contexts: Mapping[str, Dict[str, Any]],
    model_name: str,
    thinking_level: Optional[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    normalized_model_name = str(model_name or "").strip()
    normalized_thinking = str(thinking_level or "").strip().lower()

    for tier_name, context in tier_contexts.items():
        canonical_case_ids = list(context.get("case_ids") or [])
        if not canonical_case_ids:
            continue
        eval_set_id = str(context.get("eval_set_id") or "")
        runs = store.list_eval_runs(eval_set_id=eval_set_id)
        run_ids = [
            str(run.get("id") or "")
            for run in runs
            if (
                str(run.get("id") or "")
                and str(run.get("status") or "").lower() == "completed"
                and str(run.get("model_name") or run.get("model") or "").strip() == normalized_model_name
                and str(run.get("thinking_level") or "").strip().lower() == normalized_thinking
            )
        ]
        if not run_ids:
            continue
        results_by_run = store.list_eval_run_results_many(run_ids)
        for run in runs:
            run_id = str(run.get("id") or "")
            if run_id not in run_ids:
                continue
            summary = _summarize_eval_run_results(
                run=run,
                results=results_by_run.get(run_id, []),
            )
            if not summary:
                continue
            if not _planner_case_ids_match_canonical_prefix(
                run_summary=summary,
                tier_name=tier_name,
                canonical_case_ids=canonical_case_ids,
            ):
                continue
            summary.update(
                {
                    "tier_name": tier_name,
                    "eval_set_id": eval_set_id,
                    "source_name": str(context.get("source_name") or ""),
                }
            )
            rows.append(summary)
    return rows


def _build_development_leaderboard_status(
    *,
    base: Path,
    store: RunStore,
    model_name: str,
    thinking_level: Optional[str],
    requested_tier: str,
    tier_map_path: Optional[str],
    tier_definitions_path: Optional[str],
    allow_holdout: bool,
) -> Dict[str, Any]:
    context_bundle = _resolve_development_tier_contexts(
        base=base,
        store=store,
        tier_map_path=tier_map_path,
        tier_definitions_path=tier_definitions_path,
        allow_holdout=allow_holdout,
    )
    policy = dict(context_bundle["policy"])
    tier_contexts = dict(context_bundle["tiers"])
    tier_order = list(policy["tier_order"])
    rows = _collect_development_canonical_rows(
        store=store,
        tier_contexts=tier_contexts,
        model_name=model_name,
        thinking_level=thinking_level,
    )

    rows_by_tier: Dict[str, List[Dict[str, Any]]] = {tier_name: [] for tier_name in tier_order}
    for row in rows:
        rows_by_tier.setdefault(str(row.get("tier_name") or ""), []).append(row)

    current_winner: Optional[Dict[str, Any]] = None
    current_winner_tier: Optional[str] = None
    for tier_name in reversed(tier_order):
        tier_rows = rows_by_tier.get(tier_name) or []
        if not tier_rows:
            continue
        current_winner = sorted(tier_rows, key=_planner_sort_key)[0]
        current_winner_tier = tier_name
        break

    policy_snapshot = {
        "version": policy["version"],
        "path": policy["path"],
        "comparison_scope": policy["comparison_scope"],
        "initial_qualifying": dict(policy["initial_qualifying"]),
        "extend": dict(policy["extend"]),
        "next_tier": dict(policy["next_tier"]),
        "active_tier_sources": dict(policy["active_tier_sources"]),
    }

    routes: List[Dict[str, Any]] = []
    if current_winner is None:
        seed_tier = str(policy["initial_qualifying"]["tier"])
        seed_count = int(policy["initial_qualifying"]["case_count"])
        seed_context = tier_contexts[seed_tier]
        seed_case_ids = list(seed_context.get("case_ids") or [])[:seed_count]
        if not seed_case_ids:
            raise typer.BadParameter(
                f"Development leaderboard seed route has 0 available cases for tier '{seed_tier}'"
            )
        routes.append(
            {
                "key": "seed",
                "label": "Seed",
                "tier_name": seed_tier,
                "eval_set_id": str(seed_context.get("eval_set_id") or ""),
                "case_ids": seed_case_ids,
                "case_count": len(seed_case_ids),
                "description": (
                    f"Run the policy seed slice on {seed_tier} "
                    f"({len(seed_case_ids)} case(s)); this establishes the first qualifying row."
                ),
                "requires_better_score": False,
                "source_eval_run_id": None,
                "source_case_ids_hash": None,
                "planner_metadata": {
                    "planner_version": policy["version"],
                    "route_kind": "seed",
                    "tier_name": seed_tier,
                    "selected_case_count": len(seed_case_ids),
                    "selected_case_ids_hash": case_ids_hash(seed_case_ids),
                    "policy_snapshot": policy_snapshot,
                    "source_eval_run_id": None,
                    "source_case_ids_hash": None,
                    "is_policy_canonical_slice": True,
                },
            }
        )
    else:
        winner_tier = str(current_winner_tier or current_winner.get("tier_name") or "")
        winner_context = tier_contexts[winner_tier]
        winner_case_count = int(current_winner.get("case_count") or 0)
        winner_case_ids = list(winner_context.get("case_ids") or [])[:winner_case_count]
        winner_case_hash = case_ids_hash(winner_case_ids)
        routes.append(
            {
                "key": "same",
                "label": "Same",
                "tier_name": winner_tier,
                "eval_set_id": str(winner_context.get("eval_set_id") or ""),
                "case_ids": winner_case_ids,
                "case_count": len(winner_case_ids),
                "description": (
                    f"Rerun the current canonical {winner_tier} slice "
                    f"({len(winner_case_ids)} case(s)); must beat the current winner."
                ),
                "requires_better_score": True,
                "source_eval_run_id": str(current_winner.get("eval_run_id") or ""),
                "source_case_ids_hash": winner_case_hash,
                "planner_metadata": {
                    "planner_version": policy["version"],
                    "route_kind": "same",
                    "tier_name": winner_tier,
                    "selected_case_count": len(winner_case_ids),
                    "selected_case_ids_hash": winner_case_hash,
                    "policy_snapshot": policy_snapshot,
                    "source_eval_run_id": str(current_winner.get("eval_run_id") or ""),
                    "source_case_ids_hash": winner_case_hash,
                    "is_policy_canonical_slice": True,
                },
            }
        )

        extend_increment = int(policy["extend"]["case_increment"])
        tier_case_ids = list(winner_context.get("case_ids") or [])
        if winner_case_count < len(tier_case_ids):
            extend_case_ids = tier_case_ids[: min(len(tier_case_ids), winner_case_count + extend_increment)]
            routes.append(
                {
                    "key": "extend",
                    "label": "Extend",
                    "tier_name": winner_tier,
                    "eval_set_id": str(winner_context.get("eval_set_id") or ""),
                    "case_ids": extend_case_ids,
                    "case_count": len(extend_case_ids),
                    "description": (
                        f"Extend the current {winner_tier} canonical slice from {winner_case_count} "
                        f"to {len(extend_case_ids)} case(s); must beat the current winner."
                    ),
                    "requires_better_score": True,
                    "source_eval_run_id": str(current_winner.get("eval_run_id") or ""),
                    "source_case_ids_hash": winner_case_hash,
                    "planner_metadata": {
                        "planner_version": policy["version"],
                        "route_kind": "extend",
                        "tier_name": winner_tier,
                        "selected_case_count": len(extend_case_ids),
                        "selected_case_ids_hash": case_ids_hash(extend_case_ids),
                        "policy_snapshot": policy_snapshot,
                        "source_eval_run_id": str(current_winner.get("eval_run_id") or ""),
                        "source_case_ids_hash": winner_case_hash,
                        "is_policy_canonical_slice": True,
                    },
                }
            )

        winner_index = tier_order.index(winner_tier)
        if winner_index + 1 < len(tier_order):
            next_tier = tier_order[winner_index + 1]
            next_context = tier_contexts[next_tier]
            next_case_count = int(policy["next_tier"]["case_count"])
            next_case_ids = list(next_context.get("case_ids") or [])[:next_case_count]
            if next_case_ids:
                routes.append(
                    {
                        "key": "next",
                        "label": "Next tier",
                        "tier_name": next_tier,
                        "eval_set_id": str(next_context.get("eval_set_id") or ""),
                        "case_ids": next_case_ids,
                        "case_count": len(next_case_ids),
                        "description": (
                            f"Move up to {next_tier} and run the policy seed slice "
                            f"({len(next_case_ids)} case(s)); any completed scored row qualifies."
                        ),
                        "requires_better_score": False,
                        "source_eval_run_id": str(current_winner.get("eval_run_id") or ""),
                        "source_case_ids_hash": winner_case_hash,
                        "planner_metadata": {
                            "planner_version": policy["version"],
                            "route_kind": "next",
                            "tier_name": next_tier,
                            "selected_case_count": len(next_case_ids),
                            "selected_case_ids_hash": case_ids_hash(next_case_ids),
                            "policy_snapshot": policy_snapshot,
                            "source_eval_run_id": str(current_winner.get("eval_run_id") or ""),
                            "source_case_ids_hash": winner_case_hash,
                            "is_policy_canonical_slice": True,
                        },
                    }
                )

    requested_tier_has_cases = bool((tier_contexts.get(requested_tier) or {}).get("case_ids"))
    return {
        "policy": policy,
        "tier_contexts": tier_contexts,
        "requested_tier": requested_tier,
        "requested_tier_has_cases": requested_tier_has_cases,
        "current_winner": current_winner,
        "current_winner_tier": current_winner_tier,
        "routes": routes,
        "recommended_route": routes[0] if routes else None,
    }


def _print_development_leaderboard_status(
    *,
    status: Dict[str, Any],
    model_name: str,
    thinking_level: Optional[str],
    include_custom_hint: bool = True,
) -> None:
    typer.echo(
        "Development leaderboard planner: "
        f"model={model_name} thinking={thinking_level or 'none'} "
        f"scope={status['policy']['comparison_scope']}"
    )
    active_sources = ", ".join(
        f"{tier}={status['tier_contexts'][tier]['source_name']}"
        for tier in status["policy"]["tier_order"]
    )
    typer.echo(f"  Active tier sources: {active_sources}")

    winner = status.get("current_winner")
    if winner is None:
        seed = status["policy"]["initial_qualifying"]
        typer.echo(
            "  Current winner: none. "
            f"The policy seed route starts at {seed['tier']} with {seed['case_count']} case(s)."
        )
    else:
        typer.echo(
            "  Current winner: "
            f"tier={winner['tier_name']} cases={winner['case_count']} "
            f"score={winner['mean_quality_score']:.4f} pass_rate={winner['deterministic_pass_rate']:.4f} "
            f"run={winner['eval_run_id']}"
        )
        if winner.get("run_group_name"):
            typer.echo(f"    run_group={winner['run_group_name']}")

    if status["requested_tier"] != (status.get("current_winner_tier") or status["policy"]["initial_qualifying"]["tier"]):
        typer.echo(
            f"  Requested tier '{status['requested_tier']}' is not the current planner tier; "
            "the planner is following the highest qualifying tier for this model + thinking scope."
        )

    typer.echo("  Available routes:")
    for index, route in enumerate(status.get("routes") or [], start=1):
        better_text = " must beat current winner" if route.get("requires_better_score") else ""
        typer.echo(
            f"    {index}. {route['key']} -> tier={route['tier_name']} "
            f"cases={route['case_count']} eval_set_id={route['eval_set_id']}{better_text}"
        )
        typer.echo(f"       {route['description']}")

    if include_custom_hint:
        typer.echo(
            "    custom -> preserve manual tier selection. "
            "Use --leaderboard-route custom or pass --case-id to bypass planner case selection."
        )


def _is_interactive_tty() -> bool:
    stdin = getattr(sys, "stdin", None)
    stdout = getattr(sys, "stdout", None)
    return bool(stdin and stdout and stdin.isatty() and stdout.isatty())


def _select_development_route(
    *,
    status: Dict[str, Any],
    requested_route: str,
    auto_confirm: bool,
    json_output: bool,
) -> Dict[str, Any]:
    routes = list(status.get("routes") or [])
    if not routes:
        raise typer.BadParameter("No development leaderboard routes are available for this scope")

    route_by_key = {str(route["key"]): route for route in routes}
    if requested_route != "auto":
        selected = route_by_key.get(requested_route)
        if selected is None:
            raise typer.BadParameter(
                f"Requested leaderboard route '{requested_route}' is not available for this scope"
            )
        return selected

    recommended = dict(status.get("recommended_route") or routes[0])
    if json_output or auto_confirm or not _is_interactive_tty():
        return recommended

    selection_map = {str(index): route for index, route in enumerate(routes, start=1)}
    prompt = "Select route"
    typed = input(f"{prompt} [{'/'.join(selection_map.keys())}/custom] (default 1): ").strip().lower()
    if not typed:
        return selection_map["1"]
    if typed == "custom":
        raise typer.BadParameter(
            "Interactive custom selection is only available via --leaderboard-route custom or --case-id."
        )
    selected = selection_map.get(typed)
    if selected is None:
        raise typer.BadParameter(f"Unknown route selection: {typed}")
    return selected


def _build_baseline_tier_execution_plan(
    *,
    base: Path,
    store: RunStore,
    requested_tiers: Sequence[str],
    tier_eval_set_ids: Mapping[str, str],
    tier_definitions_path: Path,
    allow_holdout: bool,
) -> List[Dict[str, Any]]:
    """Resolve tier configuration and ensure each tier has runnable cases."""
    tier_case_ids_by_name = _load_eval_tier_ids(tier_definitions_path)
    plan: List[Dict[str, Any]] = []

    for tier_name in requested_tiers:
        eval_set_id = str(tier_eval_set_ids.get(tier_name) or "").strip()
        if not eval_set_id:
            raise typer.BadParameter(f"Tier '{tier_name}' has no configured eval_set_id in tier map")
        try:
            resolved_eval_set = resolve_eval_set(
                store=store,
                requested_eval_set_id=eval_set_id,
            )
        except EvalSetResolutionError as exc:
            raise typer.BadParameter(
                f"Tier '{tier_name}' references unknown eval_set_id '{eval_set_id}': {exc}"
            ) from exc
        if resolved_eval_set.purpose == "leaderboard_holdout" and not allow_holdout:
            raise typer.BadParameter(
                f"Tier '{tier_name}' references holdout eval set '{eval_set_id}', "
                "which is not allowed for baseline tier runs."
            )

        by_id = {
            str(case.get("case_id") or ""): case
            for case in resolved_eval_set.cases
            if str(case.get("case_id") or "")
        }
        tier_case_ids = tier_case_ids_by_name.get(tier_name, [])
        selected_case_ids = [case_id for case_id in tier_case_ids if case_id in by_id]
        if not selected_case_ids:
            raise typer.BadParameter(
                f"Tier '{tier_name}' has 0 cases; sync eval_tiers.json or tier mapping"
            )

        plan.append(
            {
                "tier": tier_name,
                "eval_set_id": eval_set_id,
                "resolved_eval_set": resolved_eval_set,
                "case_ids": selected_case_ids,
            }
        )

    return plan


def _filter_unrun_case_ids_for_model(
    *,
    store: RunStore,
    case_ids: Sequence[str],
    model_name: str,
    thinking_level: Optional[str],
    model_family: Optional[str],
) -> List[str]:
    """Return case IDs that have not yet been attempted for model+thinking."""
    ordered_case_ids = [
        str(case_id).strip()
        for case_id in case_ids
        if str(case_id).strip()
    ]
    if not ordered_case_ids:
        return []
    history = store.list_case_attempt_history(
        model_name=model_name,
        thinking_level=thinking_level,
        model_family=model_family,
        case_ids=ordered_case_ids,
    )
    attempted = set(history)
    return [case_id for case_id in ordered_case_ids if case_id not in attempted]


OFFICIAL_TIER_NAMES: tuple[str, str, str] = ("easy", "medium", "hard")


def _official_case_matches_tier(case: Dict[str, Any], tier_name: str) -> bool:
    step_count = _eval_case_step_count(case)
    if step_count is None:
        return False
    if tier_name == "easy":
        return 1 <= step_count <= 2
    if tier_name == "medium":
        return step_count == 3
    return step_count >= 4


def _official_case_ids_for_tier(
    *,
    cases: Sequence[Dict[str, Any]],
    tier_name: str,
) -> List[str]:
    ordered_case_ids: List[str] = []
    for case in cases:
        case_id = str(case.get("case_id") or "").strip()
        if not case_id:
            continue
        if not _official_case_matches_tier(case, tier_name):
            continue
        ordered_case_ids.append(case_id)
    return ordered_case_ids


def _list_attempted_eval_case_ids_for_scope(
    *,
    store: RunStore,
    eval_set_id: str,
    model_name: str,
    thinking_level: Optional[str],
    run_group_name: Optional[str],
) -> List[str]:
    normalized_model_name = str(model_name or "").strip()
    normalized_thinking = str(thinking_level or "").strip().lower()
    normalized_run_group = str(run_group_name or "").strip()

    run_ids: List[str] = []
    for row in store.list_eval_runs(eval_set_id=eval_set_id):
        row_model_name = str(row.get("model_name") or row.get("model") or "").strip()
        row_thinking = str(row.get("thinking_level") or "").strip().lower()
        row_run_group = str(row.get("run_group_name") or "").strip()
        if row_model_name != normalized_model_name:
            continue
        if row_thinking != normalized_thinking:
            continue
        if normalized_run_group and row_run_group != normalized_run_group:
            continue
        run_id = str(row.get("id") or "").strip()
        if run_id:
            run_ids.append(run_id)

    if not run_ids:
        return []

    results_by_run = store.list_eval_run_results_many(run_ids)
    seen: set[str] = set()
    attempted_case_ids: List[str] = []
    for run_id in run_ids:
        for result in results_by_run.get(run_id, []):
            case_id = str(result.get("case_id") or "").strip()
            if not case_id or case_id in seen:
                continue
            seen.add(case_id)
            attempted_case_ids.append(case_id)
    return attempted_case_ids


def _select_case_ids_resume_then_cycle(
    *,
    candidate_case_ids: Sequence[str],
    attempted_case_ids: Sequence[str],
    max_cases: int,
) -> tuple[List[str], Dict[str, Any]]:
    ordered_candidate_case_ids: List[str] = []
    seen_candidates: set[str] = set()
    for case_id in candidate_case_ids:
        normalized = str(case_id).strip()
        if not normalized or normalized in seen_candidates:
            continue
        seen_candidates.add(normalized)
        ordered_candidate_case_ids.append(normalized)

    if max_cases <= 0 or not ordered_candidate_case_ids:
        return [], {
            "candidate_count": len(ordered_candidate_case_ids),
            "attempted_count": 0,
            "unrun_count": 0,
            "target_count": 0,
            "wrapped": False,
        }

    attempted_set = {
        str(case_id).strip()
        for case_id in attempted_case_ids
        if str(case_id).strip()
    }
    unrun_case_ids = [
        case_id
        for case_id in ordered_candidate_case_ids
        if case_id not in attempted_set
    ]

    target_count = min(max_cases, len(ordered_candidate_case_ids))
    selected_case_ids = list(unrun_case_ids[:target_count])
    wrapped = False

    if len(selected_case_ids) < target_count:
        wrapped = True
        selected_set = set(selected_case_ids)
        for case_id in ordered_candidate_case_ids:
            if case_id in selected_set:
                continue
            selected_case_ids.append(case_id)
            selected_set.add(case_id)
            if len(selected_case_ids) >= target_count:
                break

    return selected_case_ids, {
        "candidate_count": len(ordered_candidate_case_ids),
        "attempted_count": len(attempted_set.intersection(seen_candidates)),
        "unrun_count": len(unrun_case_ids),
        "target_count": target_count,
        "wrapped": wrapped,
    }


def _run_baseline_eval_set(
    *,
    runner: Any,
    score_baseline_result_fn: Any,
    store: RunStore,
    run_group_name: str,
    resolved_eval_set: Any,
    model_name: str,
    model_family: str,
    thinking_level: Optional[str],
    temperature: float,
    ph: Optional[float],
    max_cases: int,
    timeout: float,
    llm_seed: int,
    llm_temperature: float,
    sampling_policy: str,
    harness_hash: str,
    case_ids: Optional[Sequence[str]] = None,
    api_keys: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Execute one baseline eval run and persist leaderboard results."""
    eval_run_id = store.create_eval_run(
        eval_set_id=str(resolved_eval_set.eval_set_id),
        run_group_name=run_group_name,
        model=model_name,
        model_name=model_name,
        model_family=model_family,
        thinking_level=thinking_level,
        harness_bundle_hash=harness_hash,
        status="running",
    )

    selected_cases = select_eval_cases(
        cases=resolved_eval_set.cases,
        case_ids=(list(case_ids) if case_ids else None),
        max_cases=max_cases,
    )
    resolved_case_ids = [
        str(item.get("case_id") or "")
        for item in selected_cases
        if str(item.get("case_id") or "")
    ]
    resolved_case_ids_digest = case_ids_hash(resolved_case_ids)

    completed = 0
    passed_count = 0
    failed = 0
    errored = 0
    prompt_hashes: List[str] = []
    for case in selected_cases:
        case_id = str(case.get("case_id") or "")
        input_payload = case.get("input") or {}
        sm = [str(s) for s in input_payload.get("starting_materials", [])]
        prods = [str(p) for p in input_payload.get("products", [])]
        if not sm or not prods:
            continue

        expected = case.get("expected") or {}
        if not isinstance(expected, dict):
            expected = {}

        try:
            result = runner.run_case(
                starting_materials=sm,
                products=prods,
                model=model_name,
                thinking_level=thinking_level,
                temperature_celsius=temperature,
                ph=ph,
                timeout=timeout,
                llm_seed=llm_seed,
                llm_temperature=(llm_temperature if sampling_policy == "fixed" else None),
                sampling_policy=sampling_policy,
                api_keys=api_keys or None,
            )
            graded = score_baseline_result_fn(result, expected if expected else None)
            score = float(graded["score"])
            case_passed = bool(graded["passed"])
            latency_ms = float(result.get("latency_ms") or 0.0)
            prompt_hash = str(result.get("prompt_hash") or "")
            if prompt_hash:
                prompt_hashes.append(prompt_hash)
            summary: Dict[str, Any] = {
                "score": score,
                "passed": case_passed,
                "step_count": graded.get("step_count"),
                "mechanism_type": graded.get("mechanism_type"),
                "scoring_breakdown": graded.get("scoring_breakdown", {}),
                "error": graded.get("error"),
                "eval_mode": "baseline",
                "run_metadata": {
                    "eval_set_id": resolved_eval_set.eval_set_id,
                    "eval_set_purpose": resolved_eval_set.purpose,
                    "eval_case_ids_hash": resolved_case_ids_digest,
                    "case_id": case_id,
                    "model": model_name,
                    "thinking_level": thinking_level,
                    "llm_seed": llm_seed,
                    "llm_temperature": (llm_temperature if sampling_policy == "fixed" else None),
                    "sampling_policy": sampling_policy,
                    "prompt_hash": result.get("prompt_hash"),
                    "prompt_system_hash": result.get("prompt_system_hash"),
                    "prompt_user_hash": result.get("prompt_user_hash"),
                },
                "subagent_scores": {
                    "full_mechanism_baseline": {
                        "quality_score": score,
                        "pass_rate": 1.0 if case_passed else 0.0,
                        "case_count": 1,
                    }
                },
            }
            store.record_eval_run_result(
                eval_run_id=eval_run_id,
                case_id=case_id or uuid.uuid4().hex,
                run_id=None,
                score=score,
                passed=case_passed,
                cost={},
                latency_ms=latency_ms,
                summary=summary,
            )
            completed += 1
            if case_passed:
                passed_count += 1
            else:
                failed += 1
            error_text = str(graded.get("error") or "").strip()
            if error_text:
                typer.echo(
                    f"  [{completed}] {case_id}: score={score:.3f} "
                    f"passed={case_passed} error={error_text}"
                )
            else:
                typer.echo(f"  [{completed}] {case_id}: score={score:.3f} passed={case_passed}")
        except Exception as exc:
            store.record_eval_run_result(
                eval_run_id=eval_run_id,
                case_id=case_id or uuid.uuid4().hex,
                run_id=None,
                score=0.0,
                passed=False,
                cost={},
                latency_ms=0.0,
                summary={"error": str(exc), "eval_mode": "baseline"},
            )
            completed += 1
            failed += 1
            errored += 1
            typer.echo(f"  [{completed}] {case_id}: FAILED ({exc})")

    store.set_eval_run_status(eval_run_id, "completed")
    return {
        "eval_run_id": eval_run_id,
        "model": model_name,
        "thinking_level": thinking_level,
        "completed": completed,
        "passed": passed_count,
        "failed": failed,
        "errored": errored,
        "eval_set_id": resolved_eval_set.eval_set_id,
        "eval_set_purpose": resolved_eval_set.purpose,
        "eval_case_ids_hash": resolved_case_ids_digest,
        "llm_seed": llm_seed,
        "llm_temperature": (llm_temperature if sampling_policy == "fixed" else None),
        "sampling_policy": sampling_policy,
        "prompt_hashes": sorted(set(prompt_hashes))[:20],
        "run_group_name": run_group_name,
    }
@app.command()
def run(
    starting: Optional[str] = typer.Option(
        None, "--starting", help="Comma-separated SMILES for starting materials"
    ),
    products: Optional[str] = typer.Option(
        None, "--products", help="Comma-separated SMILES for products"
    ),
    temperature: float = typer.Option(25.0, "--temperature", help="Reaction temperature in Celsius"),
    ph: Optional[float] = typer.Option(None, "--ph", help="Observed reaction pH (optional)"),
    mode: str = typer.Option("unverified", "--mode", help="Run mode: verified or unverified"),
    model_name: str = typer.Option(
        get_default_model(),
        "--model-name",
        "--model",
        help="Exact model identifier used for all LLM-backed subagents",
    ),
    max_steps: int = typer.Option(10, "--max-steps", help="Maximum mechanism loop steps"),
    max_runtime: float = typer.Option(600.0, "--max-runtime", help="Maximum runtime in seconds"),
    orchestration_mode: str = typer.Option(
        "standard",
        "--orchestration-mode",
        help="Orchestration mode: standard or ralph",
    ),
    harness: str = typer.Option("default", "--harness", help="Harness name from harness_versions/"),
    harness_strategy: str = typer.Option(
        "latest",
        "--harness-strategy",
        help="Ralph harness strategy: latest, portfolio, or mutate",
    ),
    harness_list: Optional[List[str]] = typer.Option(
        None,
        "--harness-list",
        help="Repeatable harness names for portfolio strategy",
    ),
    max_iterations: int = typer.Option(
        0,
        "--max-iterations",
        help="Ralph outer-loop max iterations (0 = unlimited)",
    ),
    ralph_max_runtime: float = typer.Option(
        900.0,
        "--ralph-max-runtime",
        help="Ralph outer-loop runtime cap (seconds)",
    ),
    max_cost_usd: Optional[float] = typer.Option(
        2.0,
        "--max-cost-usd",
        help="Ralph cumulative run budget cap in USD",
    ),
    repeat_failure_signature_limit: int = typer.Option(
        2,
        "--repeat-failure-signature-limit",
        help="Stop Ralph after the same failure signature repeats N times",
    ),
    babysit_mode: str = typer.Option(
        "off",
        "--babysit",
        help="Babysit mode: off or advisory",
    ),
    mutation_lane: Optional[str] = typer.Option(
        None,
        "--mutation-lane",
        help="Optional Ralph mutation lane guard: topology, harness, prompt, few_shot",
    ),
    allow_validator_mutation: bool = typer.Option(
        True,
        "--allow-validator-mutation/--no-allow-validator-mutation",
        help="Allow Ralph to mutate validator modules between attempts",
    ),
    functional_groups: bool = typer.Option(
        True,
        "--functional-groups/--no-functional-groups",
        help="Enable functional group analysis (default: enabled)",
    ),
    intermediates: bool = typer.Option(
        True,
        "--intermediates/--no-intermediates",
        help="Enable intermediate prediction (default: enabled)",
    ),
    llm_tools: Optional[List[str]] = typer.Option(
        None,
        "--llm-tool",
        "-T",
        help=(
            "Repeatable optional LLM tools. "
            f"Allowed: {', '.join(OPTIONAL_LLM_TOOL_NAMES)}"
        ),
    ),
    thinking_level: Optional[str] = typer.Option(
        None,
        "--thinking-level",
        "--reasoning",
        help="Optional thinking level: low, high, or max (model-dependent)",
    ),
    show_events: bool = typer.Option(False, "--show-events", help="Print recorded run events"),
    json_output: bool = typer.Option(False, "--json", help="Emit final summary as JSON"),
) -> None:
    """Execute one mechanistic run through the core local runtime."""
    mode = mode.strip().lower()
    if mode not in {"verified", "unverified"}:
        raise typer.BadParameter("mode must be 'verified' or 'unverified'")
    orchestration_mode = orchestration_mode.strip().lower()
    if orchestration_mode not in {"standard", "ralph"}:
        raise typer.BadParameter("orchestration-mode must be 'standard' or 'ralph'")
    harness_strategy = harness_strategy.strip().lower()
    if harness_strategy not in {"latest", "portfolio", "mutate"}:
        raise typer.BadParameter("harness-strategy must be one of: latest, portfolio, mutate")
    babysit_mode = babysit_mode.strip().lower()
    if babysit_mode not in {"off", "advisory"}:
        raise typer.BadParameter("babysit must be one of: off, advisory")
    if mutation_lane is not None:
        mutation_lane = mutation_lane.strip().lower()
        if mutation_lane not in {"topology", "harness", "prompt", "few_shot"}:
            raise typer.BadParameter("mutation-lane must be one of: topology, harness, prompt, few_shot")
    if thinking_level is not None:
        thinking_level = thinking_level.strip().lower()
        if thinking_level not in {"low", "high", "max"}:
            raise typer.BadParameter("thinking-level must be one of: low, high, max")
    model_name = _canonicalize_model_name_or_raise(model_name)

    base = Path.cwd()
    registry = RegistrySet(base)
    store = RunStore(base / "data" / "mechanistic.db")
    store.record_assets(
        [
            {
                "asset_type": record.asset_type,
                "path": record.path,
                "sha256": record.sha256,
                "metadata": record.metadata,
            }
            for record in registry.all_assets()
        ]
    )
    if orchestration_mode == "ralph" and not json_output:
        stats = _historical_cost_stats(base / "data" / "mechanistic.db")
        p90 = float(stats.get("p90_nonzero", 0.0))
        finite_iters = max(0, int(max_iterations))
        worst_case = (p90 * finite_iters) if finite_iters > 0 else None
        worst_case_text = f"${worst_case:.4f}" if worst_case is not None else "unbounded"
        typer.echo(
            "Ralph cost warning: "
            f"avg_nonzero=${stats.get('avg_nonzero', 0.0):.4f}, "
            f"p90=${p90:.4f}, max=${stats.get('max_nonzero', 0.0):.4f}, "
            f"max_iterations={finite_iters}, estimated_p90_worst_case={worst_case_text}, "
            f"budget_cap={f'${max_cost_usd:.4f}' if max_cost_usd is not None else 'none'}"
        )

    payload = {
        "starting_materials": _parse_materials(starting, ReactionInputs().starting_materials),
        "products": _parse_materials(products, ReactionInputs().products),
        "temperature_celsius": temperature,
        "ph": ph,
        "model": model_name,
        "optional_llm_tools": list(llm_tools) if llm_tools is not None else list(OPTIONAL_LLM_TOOL_NAMES),
        "functional_groups_enabled": functional_groups,
        "intermediate_prediction_enabled": intermediates,
    }
    reaction = ReactionInputs(**payload)

    model_plan = select_step_models(
        model_name=reaction.model,
        thinking_level=thinking_level,
        functional_groups_enabled=reaction.functional_groups_enabled,
        intermediate_prediction_enabled=reaction.intermediate_prediction_enabled,
        optional_llm_tools=reaction.optional_llm_tools,
    )
    step_reasoning: Dict[str, str] = dict(model_plan.step_reasoning)
    internal_reasoning = to_internal_reasoning_level(thinking_level)

    hashes = registry.bundle_hashes(model_name=model_plan.model_name)
    run_id = store.create_run(
        mode=mode,
        input_payload={
            "starting_materials": reaction.starting_materials,
            "products": reaction.products,
            "temperature_celsius": reaction.temperature_celsius,
            "ph": reaction.ph,
        },
        config={
            "model": model_plan.step_models.get("mechanism_synthesis", reaction.model),
            "model_name": model_plan.model_name,
            "model_family": get_model_family(model_plan.model_name),
            "thinking_level": model_plan.thinking_level,
            "reasoning_level": internal_reasoning,
            "step_models": model_plan.step_models,
            "step_reasoning": step_reasoning,
            "optional_llm_tools": reaction.optional_llm_tools,
            "functional_groups_enabled": reaction.functional_groups_enabled,
            "intermediate_prediction_enabled": reaction.intermediate_prediction_enabled,
            "model_plan_notes": model_plan.notes,
            "max_steps": max_steps,
            "max_runtime_seconds": max_runtime,
            "orchestration_mode": orchestration_mode,
            "harness_name": harness,
            "harness_strategy": harness_strategy,
            "harness_list": list(harness_list or []),
            "max_iterations": max(0, int(max_iterations)),
            "completion_promise": "target_products_reached && flow_node:run_complete",
            "ralph_max_runtime_seconds": max(1.0, float(ralph_max_runtime)),
            "max_cost_usd": max_cost_usd,
            "repeat_failure_signature_limit": max(1, int(repeat_failure_signature_limit)),
            "babysit_mode": babysit_mode,
            "allow_validator_mutation": allow_validator_mutation,
            "mutation_lane": mutation_lane,
        },
        **hashes,
    )
    prompt_records = registry.prompt_step_map(model_name=model_plan.model_name)
    prompt_ids_by_step = store.upsert_prompt_versions(
        [
            {
                "name": value.get("name"),
                "call_name": value.get("call_name"),
                "step": step,
                "version": value.get("version"),
                "path": value.get("path"),
                "sha256": value.get("sha256"),
                "shared_base_sha256": value.get("shared_base_sha256"),
                "call_base_sha256": value.get("call_base_sha256"),
                "few_shot_sha256": value.get("few_shot_sha256"),
                "prompt_bundle_sha256": value.get("prompt_bundle_sha256"),
                "template": value.get("template"),
                "model_name": value.get("model_name"),
                "resolved_shared_base_path": value.get("resolved_shared_base_path"),
                "resolved_call_base_path": value.get("resolved_call_base_path"),
                "resolved_few_shot_path": value.get("resolved_few_shot_path"),
                "asset_scope": value.get("asset_scope"),
            }
            for step, value in prompt_records.items()
        ]
    )
    bound_steps = set(model_plan.step_models)
    if "intermediates" in bound_steps and "mechanism_step_proposal" in prompt_ids_by_step:
        bound_steps.add("mechanism_step_proposal")
    for step_name in sorted(bound_steps):
        prompt_id = prompt_ids_by_step.get(step_name)
        if prompt_id:
            store.bind_run_step_prompt(
                run_id=run_id,
                step_name=step_name,
                prompt_version_id=prompt_id,
                attempt=0,
            )
    store.append_event(
        run_id,
        "run_created",
        {
            "mode": mode,
            "starting_materials": reaction.starting_materials,
            "products": reaction.products,
            "model_name": model_plan.model_name,
            "thinking_level": model_plan.thinking_level,
            "step_models": model_plan.step_models,
            "model_plan_notes": model_plan.notes,
            "prompt_versions_by_step": prompt_ids_by_step,
            **hashes,
        },
    )

    coordinator = RunCoordinator(store)
    start_ts = time.monotonic()
    coordinator.execute_run(run_id, threading.Event())
    elapsed = time.monotonic() - start_ts
    snapshot = store.get_run_snapshot(run_id)
    if snapshot is None:
        raise RuntimeError(f"Failed to load run snapshot for run_id={run_id}")

    mechanism_steps = [
        row for row in snapshot.get("step_outputs", []) if row.get("step_name") == "mechanism_synthesis"
    ]
    failed_steps = [
        row
        for row in mechanism_steps
        if isinstance(row.get("validation"), dict) and row["validation"].get("passed") is False
    ]
    cost_summary = snapshot.get("cost_summary") or {}
    total_cost = cost_summary.get("total_cost", {}).get("total_cost", 0.0)
    ralph_attempts = list(snapshot.get("ralph_attempts") or [])
    ralph_total_cost = sum(float(item.get("cost_usd") or 0.0) for item in ralph_attempts)
    latest_child_status = snapshot.get("ralph_latest_child_status")

    summary = {
        "run_id": run_id,
        "status": snapshot.get("status"),
        "mode": snapshot.get("mode"),
        "elapsed_seconds": round(elapsed, 3),
        "mechanism_step_count": len(mechanism_steps),
        "failed_validation_steps": len(failed_steps),
        "pending_verification": len(snapshot.get("pending_verification", [])),
        "total_cost": (
            ralph_total_cost if orchestration_mode == "ralph" and ralph_attempts else total_cost
        ),
        "model_name": model_plan.model_name,
        "thinking_level": model_plan.thinking_level,
        "step_models": model_plan.step_models,
        "orchestration_mode": orchestration_mode,
        "ralph_attempt_count": len(ralph_attempts),
        "ralph_latest_child_status": latest_child_status,
    }

    if json_output:
        typer.echo(json.dumps(summary, indent=2, sort_keys=True))
    else:
        typer.echo(f"Run ID: {summary['run_id']}")
        typer.echo(f"Status: {summary['status']} ({summary['mode']})")
        typer.echo(f"Elapsed: {summary['elapsed_seconds']}s")
        typer.echo(f"Total cost: ${summary['total_cost']:.3f}")
        typer.echo(f"Mechanism steps: {summary['mechanism_step_count']}")
        typer.echo(f"Failed validation steps: {summary['failed_validation_steps']}")
        typer.echo(f"Pending verification: {summary['pending_verification']}")
        typer.echo(f"Model: {summary['model_name']}")
        typer.echo(f"Thinking level: {summary['thinking_level'] or 'none'}")
        typer.echo(f"Orchestration mode: {summary['orchestration_mode']}")
        if summary["orchestration_mode"] == "ralph":
            typer.echo(f"Ralph attempts: {summary['ralph_attempt_count']}")
            typer.echo(f"Latest child status: {summary['ralph_latest_child_status'] or 'n/a'}")
        typer.echo(f"Step models: {json.dumps(summary['step_models'], sort_keys=True)}")

    if show_events:
        events = snapshot.get("events", [])
        typer.echo("\nEvents:")
        for event in events:
            event_type = event.get("event_type", "unknown")
            step_name = event.get("step_name")
            payload_preview = json.dumps(event.get("payload", {}), sort_keys=True)
            if step_name:
                typer.echo(f"- {event_type} [{step_name}] {payload_preview}")
            else:
                typer.echo(f"- {event_type} {payload_preview}")


@app.command()
def vote(
    run_id: str = typer.Option(..., "--run-id", help="Parent Ralph run id"),
    attempt: int = typer.Option(..., "--attempt", help="Ralph attempt index"),
    step: int = typer.Option(..., "--step", help="Mechanism step index"),
    candidate_a: str = typer.Option("{}", "--a", help="Candidate A JSON"),
    candidate_b: str = typer.Option("{}", "--b", help="Candidate B JSON"),
    choice: str = typer.Option(..., "--vote", help="Vote choice: A or B"),
    confidence: Optional[float] = typer.Option(None, "--confidence", help="Optional confidence [0,1]"),
    source: str = typer.Option("cli", "--source", help="Vote source label"),
) -> None:
    """Submit a non-blocking advisory A/B vote for a Ralph run."""
    choice = choice.strip().upper()
    if choice not in {"A", "B"}:
        raise typer.BadParameter("--vote must be A or B")
    try:
        parsed_a = json.loads(candidate_a) if candidate_a.strip() else {}
        parsed_b = json.loads(candidate_b) if candidate_b.strip() else {}
    except Exception as exc:
        raise typer.BadParameter(f"Candidate payload must be valid JSON: {exc}") from exc
    if not isinstance(parsed_a, dict) or not isinstance(parsed_b, dict):
        raise typer.BadParameter("Candidate payloads must be JSON objects")

    base = Path.cwd()
    store = RunStore(base / "data" / "mechanistic.db")
    row = store.get_run_row(run_id)
    if row is None:
        raise typer.BadParameter(f"Run not found: {run_id}")
    config = row.get("config") if isinstance(row.get("config"), dict) else {}
    if str(config.get("orchestration_mode") or "standard") != "ralph":
        raise typer.BadParameter("Run is not in Ralph orchestration mode")

    vote_id = store.record_ralph_vote(
        run_id=run_id,
        attempt_index=attempt,
        step_index=step,
        candidate_a=parsed_a,
        candidate_b=parsed_b,
        vote=choice,
        confidence=confidence,
        source=source,
    )
    store.append_event(
        run_id,
        "ralph_vote_recorded",
        {
            "vote_id": vote_id,
            "attempt_index": attempt,
            "step_index": step,
            "vote": choice,
            "confidence": confidence,
            "source": source,
        },
    )
    typer.echo(f"Vote recorded: {vote_id}")


@app.command(name="overnight-ralph")
def overnight_ralph(
    eval_slice_id: str = typer.Option("default", "--eval-slice-id", help="Frozen eval slice id"),
    lanes: str = typer.Option(
        "topology,harness",
        "--lanes",
        help="Comma-separated lane list: topology,harness,prompt,few_shot",
    ),
    max_experiments: int = typer.Option(20, "--max-experiments", help="Maximum experiment count"),
    max_cost_usd: float = typer.Option(15.0, "--max-cost-usd", help="Max overnight budget in USD"),
    acceptance_threshold: float = typer.Option(
        0.02,
        "--acceptance-threshold",
        help="Required absolute improvement for keep decisions",
    ),
    program: str = typer.Option("ralph_program.md", "--program", help="Program markdown path"),
    model_name: str = typer.Option(
        get_default_model(),
        "--model-name",
        "--model",
        help="Model identifier used for child micro-eval runs",
    ),
    harness: str = typer.Option("default", "--harness", help="Harness name from harness_versions/"),
    json_output: bool = typer.Option(False, "--json", help="Emit summary as JSON"),
) -> None:
    """Run lane-scoped overnight Ralph hill-climbing on a frozen eval slice."""
    base = Path.cwd()
    program_path = Path(program)
    if not program_path.is_absolute():
        program_path = (base / program_path).resolve()
    if not program_path.exists():
        raise typer.BadParameter(f"Program file not found: {program_path}")

    model_name = _canonicalize_model_name_or_raise(model_name)
    parsed_lanes = [item.strip() for item in lanes.split(",") if item.strip()]
    for lane in parsed_lanes:
        if lane not in {"topology", "harness", "prompt", "few_shot"}:
            raise typer.BadParameter(f"Invalid lane '{lane}'")

    program_config = load_overnight_program(program_path)
    if eval_slice_id:
        program_config.eval_slice_id = eval_slice_id
    if parsed_lanes:
        program_config.allowed_lanes = parsed_lanes  # type: ignore[assignment]
    program_config.max_experiments = max(1, int(max_experiments))
    program_config.max_cost_usd = float(max_cost_usd)
    program_config.acceptance_threshold_pct = max(0.0, float(acceptance_threshold))

    store = RunStore(base / "data" / "mechanistic.db")
    orchestrator = OvernightRalphOrchestrator(base_dir=base, store=store)
    plan = select_step_models(model_name=model_name)
    run_config = {
        "model": model_name,
        "model_name": model_name,
        "model_family": get_model_family(model_name),
        "thinking_level": None,
        "reasoning_level": None,
        "step_models": plan.step_models,
        "step_reasoning": plan.step_reasoning,
        "optional_llm_tools": list(OPTIONAL_LLM_TOOL_NAMES),
        "functional_groups_enabled": True,
        "intermediate_prediction_enabled": True,
        "max_steps": 6,
        "max_runtime_seconds": 180.0,
        "orchestration_mode": "standard",
        "harness_name": harness,
    }
    summary = orchestrator.run(config=program_config, run_config=run_config)
    if json_output:
        typer.echo(json.dumps(summary, indent=2, sort_keys=True))
        return
    typer.echo(f"Status: {summary.get('status')}")
    typer.echo(f"Eval slice: {summary.get('eval_slice_id')}")
    typer.echo(f"Experiments: {summary.get('experiments_attempted')}/{summary.get('max_experiments')}")
    typer.echo(f"Keeps: {summary.get('keep_count')}  Discards: {summary.get('discard_count')}")
    typer.echo(f"Spent cost: ${float(summary.get('spent_cost_usd') or 0.0):.4f}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Host for the FastAPI runtime server"),
    port: int = typer.Option(8010, help="Port for the FastAPI runtime server"),
    reload: bool = typer.Option(False, help="Enable auto-reload for local development"),
) -> None:
    """Launch the local-first FastAPI runtime and browser UI."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "uvicorn is required to run the API server. Install with `pip install uvicorn`."
        ) from exc

    from mechanistic_agent.api import create_app

    uvicorn.run(
        create_app(Path.cwd()),
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def baseline(
    starting: Optional[str] = typer.Option(
        None, "--starting", help="Comma-separated SMILES for starting materials"
    ),
    products: Optional[str] = typer.Option(
        None, "--products", help="Comma-separated SMILES for products"
    ),
    eval_set_id: Optional[str] = typer.Option(
        None, "--eval-set-id", help="Run baseline against all cases in an eval set"
    ),
    tier: Optional[List[str]] = typer.Option(
        None,
        "--tier",
        help="Repeatable tier name: easy, medium, or hard (uses tier map eval_set_id)",
    ),
    all_tiers: bool = typer.Option(
        False,
        "--all-tiers",
        help="Run easy, medium, and hard baseline tiers in one command",
    ),
    tier_map_path: Optional[str] = typer.Option(
        None,
        "--tier-map-path",
        help="Path to tier eval-set mapping JSON (default: training_data/baseline_tier_eval_set_map.json)",
    ),
    tier_definitions_path: Optional[str] = typer.Option(
        None,
        "--tier-definitions-path",
        help=(
            "Path to tier case-id definitions JSON "
            "(default: training_data/baseline_tiers_clawdiator.json, fallback: training_data/eval_tiers.json)"
        ),
    ),
    run_group_prefix: str = typer.Option(
        "harness_free_baseline",
        "--run-group-prefix",
        help="Run-group prefix for tier mode; final name is <prefix>_<tier>",
    ),
    model_name: str = typer.Option(
        get_default_model(),
        "--model-name",
        "--model",
        help="Model identifier (e.g. gpt-5.4, claude-opus-4.6)",
    ),
    thinking_level: Optional[str] = typer.Option(
        None, "--thinking-level", "--reasoning", help="Thinking level: low, high, or max (model-dependent)"
    ),
    temperature: float = typer.Option(25.0, "--temperature", help="Reaction temperature in Celsius"),
    ph: Optional[float] = typer.Option(None, "--ph", help="Observed reaction pH (optional)"),
    max_cases: int = typer.Option(25, "--max-cases", help="Max cases when running an eval set"),
    timeout: float = typer.Option(180.0, "--timeout", help="Per-case timeout in seconds"),
    llm_seed: int = typer.Option(42, "--llm-seed", help="Deterministic seed hint for providers that support it"),
    llm_temperature: float = typer.Option(0.0, "--llm-temperature", help="Sampling temperature when using fixed policy"),
    sampling_policy: str = typer.Option(
        "fixed",
        "--sampling-policy",
        help="LLM sampling policy: fixed or provider_default",
    ),
    allow_repeats: bool = typer.Option(
        False,
        "--allow-repeats",
        help="Allow rerunning cases already attempted for this model + thinking level",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit results as JSON"),
    allow_holdout: bool = typer.Option(False, "--allow-holdout", hidden=True),
) -> None:
    """Run harness-free single-shot baseline mechanism prediction.

    Either provide --starting/--products for a single case, or --eval-set-id
    to run against an eval set and record results on the leaderboard. Tier mode
    is available via --tier/--all-tiers and runs without starting the API server.
    """
    from mechanistic_agent.core.baseline_runner import (
        BASELINE_GROUP_PREFIX,
        BaselineRunner,
        score_baseline_result,
    )

    # Load API keys from environment (dotenv + environment variables)
    api_keys = _load_api_keys()

    if thinking_level is not None:
        thinking_level = thinking_level.strip().lower()
        if thinking_level not in {"low", "high", "max"}:
            raise typer.BadParameter("thinking-level must be one of: low, high, max")
    sampling_policy = sampling_policy.strip().lower()
    if sampling_policy not in {"fixed", "provider_default"}:
        raise typer.BadParameter("sampling-policy must be one of: fixed, provider_default")
    model_name = _canonicalize_model_name_or_raise(model_name)
    run_group_prefix = run_group_prefix.strip()
    if not run_group_prefix:
        raise typer.BadParameter("run-group-prefix must not be empty")
    requested_tiers = _normalize_requested_baseline_tiers(tier, all_tiers=all_tiers)

    runner = BaselineRunner()

    if requested_tiers:
        # ---- Tier mode: run easy/medium/hard in sequence via local CLI ----
        base = Path.cwd()
        store = RunStore(base / "data" / "mechanistic.db")
        registry = RegistrySet(base)
        model_family = get_model_family(model_name) or "unknown"
        harness_hash = registry.bundle_hashes(model_name=model_name).get("prompt_bundle_hash", "")
        if eval_set_id and not json_output:
            typer.echo("Ignoring --eval-set-id because --tier/--all-tiers was provided.")

        resolved_tier_map_path = (
            Path(tier_map_path).expanduser().resolve()
            if tier_map_path
            else (base / BASELINE_TIER_MAP_DEFAULT_PATH).resolve()
        )
        resolved_tier_definitions_path = _resolve_baseline_tier_definitions_path(
            base=base,
            override_path=tier_definitions_path,
        )
        tier_eval_set_ids = _load_baseline_tier_eval_set_map(resolved_tier_map_path)
        execution_plan = _build_baseline_tier_execution_plan(
            base=base,
            store=store,
            requested_tiers=requested_tiers,
            tier_eval_set_ids=tier_eval_set_ids,
            tier_definitions_path=resolved_tier_definitions_path,
            allow_holdout=allow_holdout,
        )

        tier_results: List[Dict[str, Any]] = []
        for item in execution_plan:
            tier_name = str(item.get("tier") or "")
            resolved_eval_set = item["resolved_eval_set"]
            case_ids = list(item.get("case_ids") or [])
            if not allow_repeats:
                case_ids = _filter_unrun_case_ids_for_model(
                    store=store,
                    case_ids=case_ids,
                    model_name=model_name,
                    thinking_level=thinking_level,
                    model_family=model_family,
                )
                if not case_ids:
                    raise typer.BadParameter(
                        f"Tier '{tier_name}' has 0 unrun cases for model={model_name} "
                        f"thinking={thinking_level or 'none'}; pass --allow-repeats to rerun."
                    )
            run_group_name = f"{run_group_prefix}_{tier_name}"
            if not json_output:
                typer.echo(
                    f"\nRunning baseline tier '{tier_name}' "
                    f"(eval_set_id={resolved_eval_set.eval_set_id}, cases={len(case_ids)})..."
                )
            result_obj = _run_baseline_eval_set(
                runner=runner,
                score_baseline_result_fn=score_baseline_result,
                store=store,
                run_group_name=run_group_name,
                resolved_eval_set=resolved_eval_set,
                model_name=model_name,
                model_family=model_family,
                thinking_level=thinking_level,
                temperature=temperature,
                ph=ph,
                max_cases=max_cases,
                timeout=timeout,
                llm_seed=llm_seed,
                llm_temperature=llm_temperature,
                sampling_policy=sampling_policy,
                harness_hash=harness_hash,
                case_ids=case_ids,
                api_keys=api_keys,
            )
            result_obj["tier"] = tier_name
            tier_results.append(result_obj)
            if not json_output:
                typer.echo(
                    f"Tier '{tier_name}' complete: eval_run_id={result_obj['eval_run_id']} "
                    f"completed={result_obj['completed']} passed={result_obj.get('passed', 0)} "
                    f"failed={result_obj['failed']} "
                    f"run_group={run_group_name}"
                )

        if json_output:
            typer.echo(
                json.dumps(
                    {
                        "mode": "baseline_tiers",
                        "model": model_name,
                        "thinking_level": thinking_level,
                        "tier_map_path": str(resolved_tier_map_path),
                        "tier_definitions_path": str(resolved_tier_definitions_path),
                        "results": tier_results,
                    },
                    indent=2,
                )
            )
        else:
            typer.echo("\nBaseline tier runs complete.")
            unique_eval_set_ids = sorted(
                {
                    str(item.get("eval_set_id") or "")
                    for item in tier_results
                    if str(item.get("eval_set_id") or "")
                }
            )
            for eval_id in unique_eval_set_ids:
                typer.echo(f"View leaderboard: python main.py leaderboard --eval-set-id {eval_id}")
        return

    if eval_set_id:
        # ---- Eval-set mode: run all cases and record to leaderboard ----
        base = Path.cwd()
        store = RunStore(base / "data" / "mechanistic.db")
        registry = RegistrySet(base)
        model_family = get_model_family(model_name) or "unknown"
        harness_hash = registry.bundle_hashes(model_name=model_name).get("prompt_bundle_hash", "")
        try:
            resolved_eval_set = resolve_eval_set(
                store=store,
                requested_eval_set_id=eval_set_id,
            )
        except EvalSetResolutionError as exc:
            raise typer.BadParameter(str(exc)) from exc
        if resolved_eval_set.purpose == "leaderboard_holdout" and not allow_holdout:
            raise typer.BadParameter(
                "leaderboard_holdout eval sets are restricted to 'baseline-runset-official'"
            )
        selected_case_ids: Optional[List[str]] = None
        if not allow_repeats and resolved_eval_set.purpose != "leaderboard_holdout":
            all_case_ids = [
                str(item.get("case_id") or "")
                for item in resolved_eval_set.cases
                if str(item.get("case_id") or "")
            ]
            selected_case_ids = _filter_unrun_case_ids_for_model(
                store=store,
                case_ids=all_case_ids,
                model_name=model_name,
                thinking_level=thinking_level,
                model_family=model_family,
            )
            if not selected_case_ids:
                typer.echo(
                    "No unrun cases remain for the selected eval set/model/thinking. "
                    "Pass --allow-repeats to rerun cases."
                )
                raise typer.Exit(code=1)
        result_obj = _run_baseline_eval_set(
            runner=runner,
            score_baseline_result_fn=score_baseline_result,
            store=store,
            run_group_name=BASELINE_GROUP_PREFIX,
            resolved_eval_set=resolved_eval_set,
            model_name=model_name,
            model_family=model_family,
            thinking_level=thinking_level,
            temperature=temperature,
            ph=ph,
            max_cases=max_cases,
            timeout=timeout,
            llm_seed=llm_seed,
            llm_temperature=llm_temperature,
            sampling_policy=sampling_policy,
            harness_hash=harness_hash,
            case_ids=selected_case_ids,
            api_keys=api_keys,
        )
        if json_output:
            typer.echo(json.dumps(result_obj, indent=2))
        else:
            typer.echo(
                f"\nBaseline eval complete: completed={result_obj['completed']} "
                f"passed={result_obj.get('passed', 0)} failed={result_obj['failed']} "
                f"errored={result_obj.get('errored', 0)}"
            )
            typer.echo(f"Eval run ID: {result_obj['eval_run_id']}")
            typer.echo(
                f"Resolved eval set: {resolved_eval_set.eval_set_id} ({resolved_eval_set.purpose}) "
                f"case_ids_hash={result_obj['eval_case_ids_hash']}"
            )

    else:
        # ---- Single-case mode ----
        sm = _parse_materials(starting, ReactionInputs().starting_materials)
        prods = _parse_materials(products, ReactionInputs().products)
        result = runner.run_case(
            starting_materials=sm,
            products=prods,
            model=model_name,
            thinking_level=thinking_level,
            temperature_celsius=temperature,
            ph=ph,
            timeout=timeout,
            llm_seed=llm_seed,
            llm_temperature=(llm_temperature if sampling_policy == "fixed" else None),
            sampling_policy=sampling_policy,
            api_keys=api_keys or None,
        )
        graded = score_baseline_result(result, None)
        output = {
            "model": model_name,
            "thinking_level": thinking_level,
            "starting_materials": sm,
            "products": prods,
            "step_count": graded.get("step_count"),
            "mechanism_type": result.get("mechanism_type"),
            "score": graded["score"],
            "passed": graded["passed"],
            "error": result.get("error"),
            "latency_ms": round(result.get("latency_ms") or 0.0, 1),
            "run_metadata": {
                "llm_seed": llm_seed,
                "llm_temperature": (llm_temperature if sampling_policy == "fixed" else None),
                "sampling_policy": sampling_policy,
                "prompt_hash": result.get("prompt_hash"),
                "prompt_system_hash": result.get("prompt_system_hash"),
                "prompt_user_hash": result.get("prompt_user_hash"),
            },
        }
        if json_output:
            typer.echo(json.dumps(output, indent=2))
        else:
            typer.echo(f"Model: {model_name}")
            typer.echo(f"Thinking: {thinking_level or 'none'}")
            typer.echo(f"Steps: {graded.get('step_count')}")
            typer.echo(f"Mechanism: {result.get('mechanism_type') or 'unknown'}")
            typer.echo(f"Score: {graded['score']:.3f}")
            typer.echo(f"Passed: {graded['passed']}")
            typer.echo(f"Latency: {round(result.get('latency_ms') or 0, 1)} ms")
            typer.echo(
                f"Run metadata: seed={llm_seed} sampling_policy={sampling_policy} "
                f"prompt_hash={str(result.get('prompt_hash') or '')[:12]}"
            )
            if result.get("error"):
                typer.echo(f"Error: {result['error']}")


@app.command(name="baseline-runset-official")
def baseline_runset_official_cmd(
    eval_set_id: Optional[str] = typer.Option(
        None,
        "--eval-set-id",
        help="Optional holdout eval set id (defaults to latest purpose=leaderboard_holdout).",
    ),
    allow_non_holdout: bool = typer.Option(
        False,
        "--allow-non-holdout",
        help="Explicit opt-in to run non-holdout eval sets from this command.",
    ),
    model_name: str = typer.Option(
        get_default_model(),
        "--model-name",
        "--model",
        help="Model identifier (e.g. gpt-5.4, claude-opus-4.6)",
    ),
    thinking_level: Optional[str] = typer.Option(
        None, "--thinking-level", "--reasoning", help="Thinking level: low, high, or max (model-dependent)"
    ),
    temperature: float = typer.Option(25.0, "--temperature", help="Reaction temperature in Celsius"),
    ph: Optional[float] = typer.Option(None, "--ph", help="Observed reaction pH (optional)"),
    max_cases: int = typer.Option(20, "--max-cases", help="Max cases when running an eval set"),
    timeout: float = typer.Option(300.0, "--timeout", help="Per-case timeout in seconds"),
    llm_seed: int = typer.Option(42, "--llm-seed", help="Deterministic seed hint for providers that support it"),
    llm_temperature: float = typer.Option(0.0, "--llm-temperature", help="Sampling temperature when using fixed policy"),
    sampling_policy: str = typer.Option(
        "fixed",
        "--sampling-policy",
        help="LLM sampling policy: fixed or provider_default",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit results as JSON"),
) -> None:
    """Run the official holdout-only baseline eval (one-shot mode)."""
    base = Path.cwd()
    store = RunStore(base / "data" / "mechanistic.db")
    try:
        resolved_eval_set = resolve_eval_set(
            store=store,
            requested_eval_set_id=eval_set_id,
            require_purpose=(None if allow_non_holdout else "leaderboard_holdout"),
            default_purpose=("leaderboard_holdout" if not eval_set_id else None),
        )
    except EvalSetResolutionError as exc:
        raise typer.BadParameter(str(exc)) from exc

    baseline(
        starting=None,
        products=None,
        eval_set_id=str(resolved_eval_set.eval_set_id),
        tier=None,
        all_tiers=False,
        tier_map_path=None,
        tier_definitions_path=None,
        run_group_prefix="harness_free_baseline",
        model_name=model_name,
        thinking_level=thinking_level,
        temperature=temperature,
        ph=ph,
        max_cases=max_cases,
        timeout=timeout,
        llm_seed=llm_seed,
        llm_temperature=llm_temperature,
        sampling_policy=sampling_policy,
        allow_repeats=False,
        json_output=json_output,
        allow_holdout=True,
    )


@app.command(name="seed-simulated")
def seed_simulated(
    eval_set_id: str = typer.Option(..., "--eval-set-id", help="Eval set to seed simulated data for"),
    case_count: int = typer.Option(5, "--case-count", help="Simulated cases per config (1-50)"),
    delete: bool = typer.Option(False, "--delete", help="Delete simulated rows instead of seeding"),
) -> None:
    """Seed or delete simulated placeholder leaderboard rows.

    Inserts clearly-labelled [SIMULATED] rows for each model family × thinking
    level × mode (harness/baseline). Delete them with --delete once real data
    is available.
    """
    base = Path.cwd()
    store = RunStore(base / "data" / "mechanistic.db")

    if delete:
        result = store.delete_simulated_leaderboard_rows(eval_set_id=eval_set_id)
        typer.echo(f"Deleted {result.get('deleted_count', 0)} simulated eval runs")
    else:
        result = store.seed_simulated_leaderboard(
            eval_set_id=eval_set_id,
            case_count=max(1, min(case_count, 50)),
        )
        typer.echo(f"Inserted {result.get('inserted_eval_run_count', 0)} simulated eval runs")
        typer.echo(result.get("note", ""))


@app.command(name="compare-eval-runs")
def compare_eval_runs(
    run_a: str = typer.Option(..., "--run-a", help="First eval_run id"),
    run_b: str = typer.Option(..., "--run-b", help="Second eval_run id"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON comparison"),
) -> None:
    """Compare reproducibility metadata between two eval runs."""
    base = Path.cwd()
    store = RunStore(base / "data" / "mechanistic.db")

    def _run_summary(eval_run_id: str) -> Dict[str, Any]:
        row = store.get_eval_run(eval_run_id)
        if row is None:
            raise typer.BadParameter(f"Eval run not found: {eval_run_id}")
        results = store.list_eval_run_results(eval_run_id)
        ordered_case_ids = [str(item.get("case_id") or "") for item in results if str(item.get("case_id") or "")]
        summary_meta = []
        prompt_hashes: set[str] = set()
        for result in results:
            summary = result.get("summary") or {}
            if not isinstance(summary, dict):
                continue
            run_meta = summary.get("run_metadata") or {}
            if isinstance(run_meta, dict):
                summary_meta.append(run_meta)
                prompt_hash = str(run_meta.get("prompt_hash") or "")
                if prompt_hash:
                    prompt_hashes.add(prompt_hash)
        first_meta = summary_meta[0] if summary_meta else {}
        return {
            "eval_run_id": eval_run_id,
            "eval_set_id": str(row.get("eval_set_id") or ""),
            "model": str(row.get("model_name") or row.get("model") or ""),
            "thinking_level": str(row.get("thinking_level") or ""),
            "case_count": len(ordered_case_ids),
            "case_ids_hash": case_ids_hash(ordered_case_ids),
            "metadata_case_ids_hash": str(first_meta.get("eval_case_ids_hash") or ""),
            "prompt_hashes": sorted(prompt_hashes),
            "sampling_policy": first_meta.get("sampling_policy"),
            "llm_seed": first_meta.get("llm_seed"),
            "llm_temperature": first_meta.get("llm_temperature"),
        }

    a = _run_summary(run_a)
    b = _run_summary(run_b)
    comparison = {
        "run_a": a,
        "run_b": b,
        "same_eval_set_id": a["eval_set_id"] == b["eval_set_id"],
        "same_case_ids_hash": a["case_ids_hash"] == b["case_ids_hash"],
        "same_metadata_case_ids_hash": a["metadata_case_ids_hash"] == b["metadata_case_ids_hash"],
        "same_model": a["model"] == b["model"],
        "same_thinking_level": a["thinking_level"] == b["thinking_level"],
    }

    if json_output:
        typer.echo(json.dumps(comparison, indent=2))
        return

    typer.echo(f"Run A: {a['eval_run_id']}")
    typer.echo(
        f"  eval_set_id={a['eval_set_id']} case_ids_hash={a['case_ids_hash']} "
        f"model={a['model']} thinking={a['thinking_level'] or 'none'}"
    )
    typer.echo(f"  sampling_policy={a['sampling_policy']} llm_seed={a['llm_seed']} llm_temperature={a['llm_temperature']}")
    typer.echo(f"  prompt_hashes={', '.join(a['prompt_hashes'][:3]) or 'none'}")
    typer.echo(f"Run B: {b['eval_run_id']}")
    typer.echo(
        f"  eval_set_id={b['eval_set_id']} case_ids_hash={b['case_ids_hash']} "
        f"model={b['model']} thinking={b['thinking_level'] or 'none'}"
    )
    typer.echo(f"  sampling_policy={b['sampling_policy']} llm_seed={b['llm_seed']} llm_temperature={b['llm_temperature']}")
    typer.echo(f"  prompt_hashes={', '.join(b['prompt_hashes'][:3]) or 'none'}")
    typer.echo(
        "Comparison: "
        f"same_eval_set_id={comparison['same_eval_set_id']} "
        f"same_case_ids_hash={comparison['same_case_ids_hash']} "
        f"same_model={comparison['same_model']} "
        f"same_thinking_level={comparison['same_thinking_level']}"
    )


@app.command(name="import-eval-set")
def import_eval_set(
    path: Optional[str] = typer.Option(
        None, "--path", help="Path to eval_set.json (default: training_data/eval_set.json)"
    ),
    version: str = typer.Option("flower100_v1", "--version", help="Version label for this eval set"),
    json_output: bool = typer.Option(False, "--json", help="Emit result as JSON"),
) -> None:
    """Import the default FlowER eval set into the local DB from training_data/eval_set.json."""
    base = Path.cwd()
    store = RunStore(base / "data" / "mechanistic.db")
    eval_path = Path(path) if path else base / "training_data" / "eval_set.json"
    if not eval_path.exists():
        raise typer.BadParameter(f"Eval set file not found: {eval_path}")

    raw = json.loads(eval_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise typer.BadParameter("eval_set.json must contain a JSON list")

    cases: List[Dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        case_id = str(entry.get("id") or "")
        starting = entry.get("starting_materials") or []
        products = entry.get("products") or []
        if not case_id or not isinstance(starting, list) or not isinstance(products, list):
            continue
        cases.append({
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
        })

    if not cases:
        raise typer.BadParameter("No valid cases found in eval set file")

    # Deduplication: skip import if identical eval set already exists.
    expected_has_multistep = any(
        isinstance(((c.get("expected") or {}).get("known_mechanism")), dict)
        and len((((c.get("expected") or {}).get("known_mechanism") or {}).get("steps") or [])) >= 2
        for c in cases
    )
    for item in store.list_eval_sets():
        if item.get("name") != "flower_100_default" or item.get("version") != version:
            continue
        existing_cases = store.list_eval_set_cases(str(item.get("id") or ""))
        existing_has_multistep = any(
            isinstance((case.get("expected") or {}).get("known_mechanism"), dict)
            and len(((case.get("expected") or {}).get("known_mechanism") or {}).get("steps") or []) >= 2
            for case in existing_cases
        )
        if len(existing_cases) == len(cases) and (not expected_has_multistep or existing_has_multistep):
            result = {"eval_set_id": item["id"], "name": item["name"], "version": item["version"], "existing": True}
            if json_output:
                typer.echo(json.dumps(result, indent=2))
            else:
                typer.echo(f"Eval set already exists: {item['name']} ({item['version']}), id={item['id']}")
            return

    eval_set_id = store.add_eval_set(
        name="flower_100_default",
        version=version,
        source_path=str(eval_path),
        sha256=None,
        cases=cases,
        active=True,
        purpose="general",
        exposed_in_ui=True,
    )
    result = {"eval_set_id": eval_set_id, "name": "flower_100_default", "version": version, "case_count": len(cases)}
    if json_output:
        typer.echo(json.dumps(result, indent=2))
    else:
        typer.echo(f"Imported {len(cases)} cases as eval set '{result['name']}' ({result['version']})")
        typer.echo(f"Eval set ID: {eval_set_id}")


@app.command(name="import-holdout-eval-set")
def import_holdout_eval_set(
    path: Optional[str] = typer.Option(
        None,
        "--path",
        help="Path to holdout eval set JSON (default: training_data/leaderboard_holdout/eval_set_holdout.json)",
    ),
    version: str = typer.Option("flower_test_holdout_v1", "--version", help="Version label for this holdout eval set"),
    json_output: bool = typer.Option(False, "--json", help="Emit result as JSON"),
) -> None:
    """Import the isolated leaderboard holdout eval set into the local DB."""
    base = Path.cwd()
    store = RunStore(base / "data" / "mechanistic.db")
    eval_path = Path(path) if path else base / "training_data" / "leaderboard_holdout" / "eval_set_holdout.json"
    if not eval_path.exists():
        raise typer.BadParameter(f"Holdout eval set file not found: {eval_path}")

    raw = json.loads(eval_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise typer.BadParameter("Holdout eval set must contain a JSON list")

    cases: List[Dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        case_id = str(entry.get("id") or "")
        starting = entry.get("starting_materials") or []
        products = entry.get("products") or []
        if not case_id or not isinstance(starting, list) or not isinstance(products, list):
            continue
        n_steps = int(
            entry.get("n_mechanistic_steps")
            or len((((entry.get("verified_mechanism") or {}).get("steps")) or []))
            or 0
        )
        cases.append(
            {
                "case_id": case_id,
                "input": {
                    "starting_materials": starting,
                    "products": products,
                    "temperature_celsius": entry.get("temperature_celsius", 25.0),
                    "ph": entry.get("ph"),
                    "n_mechanistic_steps": n_steps,
                },
                "expected": {
                    "products": products,
                    "n_mechanistic_steps": n_steps,
                    **({"known_mechanism": entry["known_mechanism"]} if isinstance(entry.get("known_mechanism"), dict) else {}),
                    **({"verified_mechanism": entry["verified_mechanism"]} if isinstance(entry.get("verified_mechanism"), dict) else {}),
                },
                "tags": ["leaderboard_holdout", "official_holdout"],
            }
        )

    if not cases:
        raise typer.BadParameter("No valid cases found in holdout eval set file")

    for item in store.list_eval_sets(purpose="leaderboard_holdout"):
        if item.get("name") != "flower_test_holdout_official" or item.get("version") != version:
            continue
        existing_cases = store.list_eval_set_cases(str(item.get("id") or ""))
        if len(existing_cases) == len(cases):
            result = {"eval_set_id": item["id"], "name": item["name"], "version": item["version"], "existing": True}
            if json_output:
                typer.echo(json.dumps(result, indent=2))
            else:
                typer.echo(
                    f"Holdout eval set already exists: {item['name']} ({item['version']}), id={item['id']}"
                )
            return

    eval_set_id = store.add_eval_set(
        name="flower_test_holdout_official",
        version=version,
        source_path=str(eval_path),
        sha256=None,
        cases=cases,
        active=True,
        purpose="leaderboard_holdout",
        exposed_in_ui=False,
    )
    result = {
        "eval_set_id": eval_set_id,
        "name": "flower_test_holdout_official",
        "version": version,
        "case_count": len(cases),
    }
    if json_output:
        typer.echo(json.dumps(result, indent=2))
    else:
        typer.echo(
            f"Imported {len(cases)} holdout cases as eval set '{result['name']}' ({result['version']})"
        )
        typer.echo(f"Eval set ID: {eval_set_id}")


@app.command(name="leaderboard")
def leaderboard(
    eval_set_id: str = typer.Option(..., "--eval-set-id", help="Eval set to show leaderboard for"),
    limit: int = typer.Option(20, "--limit", help="Max rows to display"),
    json_output: bool = typer.Option(False, "--json", help="Emit results as JSON"),
    markdown_output: bool = typer.Option(False, "--markdown", help="Emit results as Markdown"),
    output_path: Optional[Path] = typer.Option(None, "--output", help="Write output to a file"),
    completed_only: bool = typer.Option(True, "--completed-only/--include-running", help="Show only completed eval runs by default"),
) -> None:
    """Print the eval leaderboard for a given eval set."""
    if json_output and markdown_output:
        raise typer.BadParameter("choose at most one of --json or --markdown")

    base = Path.cwd()
    store = RunStore(base / "data" / "mechanistic.db")
    items = store.leaderboard(eval_set_id=eval_set_id, limit=max(1, min(limit, 100)))
    items = _filter_leaderboard_rows(items, completed_only=completed_only)

    content: str
    if json_output:
        content = json.dumps(items, indent=2)
    elif markdown_output:
        content = _render_leaderboard_markdown(eval_set_id, items)
    elif not items:
        content = "No leaderboard results yet for this eval set."
    else:
        lines = []
        includes_cost = any("total_cost" in row for row in items)
        header = (
            f"{'Rank':<5} {'Model':<25} {'Thinking':<8} {'Type':<8} {'Score':<10} {'Outcome':<7} {'Pass':<7} {'Cases':<6} {'Cost':<8} {'Group'}"
            if includes_cost
            else f"{'Rank':<5} {'Model':<25} {'Thinking':<8} {'Type':<8} {'Score':<10} {'Outcome':<7} {'Pass':<7} {'Cases':<6} {'Group'}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for i, row in enumerate(items, 1):
            model = row.get("model_name") or row.get("model") or "unknown"
            # Truncate model name if too long
            if len(model) > 25:
                model = model[:22] + "..."
            thinking = row.get("thinking_level") or "none"
            run_type = "Baseline" if row.get("is_baseline") else "Harness"
            pts = _leaderboard_row_to_pts(row)
            score_display = f"{pts['total']}/1000"
            outcome = pts["outcome"]
            pass_rate = f"{float(row.get('weighted_pass_rate') or row.get('deterministic_pass_rate') or 0) * 100:.1f}%"
            case_count = str(row.get("case_count") or 0)
            group = row.get("run_group_name") or "n/a"
            if includes_cost:
                total_cost = float(row.get("total_cost") or 0)
                cost_display = f"${total_cost:.3f}" if total_cost > 0 else "$0.000"
                lines.append(
                    f"{i:<5} {model:<25} {thinking:<8} {run_type:<8} {score_display:<10} {outcome:<7} {pass_rate:<7} {case_count:<6} {cost_display:<8} {group}"
                )
            else:
                lines.append(
                    f"{i:<5} {model:<25} {thinking:<8} {run_type:<8} {score_display:<10} {outcome:<7} {pass_rate:<7} {case_count:<6} {group}"
                )
        if HARNESS_SPEED_CALIBRATION_MS <= 0:
            lines.append("")
            lines.append("  [!] Speed uncalibrated — HARNESS_SPEED_CALIBRATION_MS=0 in main.py (speed shown as 100 pts)")
        content = "\n".join(lines)

    if output_path is not None:
        output_path.write_text(content + ("\n" if not content.endswith("\n") else ""), encoding="utf-8")
        typer.echo(f"Wrote leaderboard output to {output_path}")
        return

    typer.echo(content)


@app.command(name="leaderboard-official")
def leaderboard_official(
    eval_set_id: Optional[str] = typer.Option(
        None,
        "--eval-set-id",
        help="Optional holdout eval set id (defaults to latest purpose=leaderboard_holdout).",
    ),
    limit: int = typer.Option(20, "--limit", help="Max rows to display"),
    json_output: bool = typer.Option(False, "--json", help="Emit results as JSON"),
    markdown_output: bool = typer.Option(False, "--markdown", help="Emit results as Markdown"),
    output_path: Optional[Path] = typer.Option(None, "--output", help="Write output to a file"),
    completed_only: bool = typer.Option(
        True, "--completed-only/--include-running", help="Show only completed eval runs by default"
    ),
) -> None:
    """Print leaderboard rows for the official holdout suite."""
    base = Path.cwd()
    store = RunStore(base / "data" / "mechanistic.db")
    try:
        resolved = resolve_eval_set(
            store=store,
            requested_eval_set_id=eval_set_id,
            require_purpose="leaderboard_holdout",
            default_purpose=("leaderboard_holdout" if not eval_set_id else None),
        )
    except EvalSetResolutionError as exc:
        raise typer.BadParameter(str(exc)) from exc

    leaderboard(
        eval_set_id=str(resolved.eval_set_id),
        limit=limit,
        json_output=json_output,
        markdown_output=markdown_output,
        output_path=output_path,
        completed_only=completed_only,
    )


def _arena_table_from_leaderboard_items(items: List[Dict[str, Any]]) -> str:
    """Build Arena Submissions table rows for LEADERBOARD.md."""
    lines = [
        "| Date | Model | Score | Outcome | Pass Rate | Avg Latency | Run Group |",
        "|---|---|---|---|---|---|---|",
    ]
    if not items:
        lines.append("| — | *(pending first submission)* | — | — | — | — | — |")
        return "\n".join(lines)
    for row in items:
        created = row.get("created_at")
        if created is None:
            date_str = "—"
        elif isinstance(created, (int, float)):
            date_str = datetime.fromtimestamp(float(created)).strftime("%Y-%m-%d")
        else:
            date_str = str(created)[:10] if created else "—"
        model = str(row.get("model_name") or row.get("model") or "unknown")
        pts = _leaderboard_row_to_pts(row)
        score_display = f"{pts['total']}/1000"
        outcome = pts["outcome"]
        pass_rate = f"{float(row.get('weighted_pass_rate') or row.get('deterministic_pass_rate') or 0.0) * 100.0:.1f}%"
        avg_ms = float(row.get("avg_latency_ms") or 0.0)
        avg_s = f"{avg_ms / 1000:.1f}s" if avg_ms > 0 else "—"
        group = str(row.get("run_group_name") or "n/a")
        lines.append(f"| {date_str} | `{model}` | {score_display} | {outcome} | {pass_rate} | {avg_s} | `{group}` |")
    return "\n".join(lines)


@app.command(name="update-leaderboard-artifacts")
def update_leaderboard_artifacts_cmd(
    refresh_curriculum: bool = typer.Option(
        True,
        "--refresh-curriculum/--no-refresh-curriculum",
        help="Also run curriculum render-readme to refresh curriculum/generated/leaderboard_*.json",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print changes without writing files"),
) -> None:
    """Regenerate LEADERBOARD.md Arena table and curriculum/generated/ from live leaderboard data.

    Run after eval-runset-official to sync LEADERBOARD.md and curriculum artifacts.
    """
    base = Path.cwd()
    store = RunStore(base / "data" / "mechanistic.db")
    try:
        resolved = resolve_eval_set(
            store=store,
            requested_eval_set_id=None,
            require_purpose="leaderboard_holdout",
            default_purpose="leaderboard_holdout",
        )
    except EvalSetResolutionError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    items = store.leaderboard(eval_set_id=str(resolved.eval_set_id), limit=50)
    items = _filter_leaderboard_rows(items, completed_only=True)

    arena_table = _arena_table_from_leaderboard_items(items)

    leaderboard_md = base / "LEADERBOARD.md"
    if not leaderboard_md.exists():
        typer.echo(f"LEADERBOARD.md not found at {leaderboard_md}", err=True)
        raise typer.Exit(1)

    content = leaderboard_md.read_text(encoding="utf-8")
    # Replace Arena Submissions table (from | Date | header through last row before ### Speed)
    import re
    pattern = r"\| Date \| Model \| Score \| Outcome \| Pass Rate \| Avg Latency \| Run Group \|\n\|[-|]+\|\n(?:\|[^\n]+\n)*(\n### Speed Calibration)"
    match = re.search(pattern, content)
    if match:
        new_content = content[: match.start()] + arena_table + "\n" + content[match.start(1) :]
        if dry_run:
            typer.echo("LEADERBOARD.md Arena table (dry-run):")
            typer.echo(arena_table)
        else:
            leaderboard_md.write_text(new_content, encoding="utf-8")
            typer.echo(f"Updated {leaderboard_md}")
    else:
        typer.echo("Could not find Arena Submissions table in LEADERBOARD.md", err=True)
        raise typer.Exit(1)

    if refresh_curriculum and not dry_run:
        typer.echo("Refreshing curriculum/generated/ (leaderboard_*.json, readme_context.json)...")
        render_curriculum_readme(base, store)
        gen_dir = base / "curriculum" / "generated"
        if gen_dir.is_dir():
            typer.echo(f"Updated {gen_dir}/")


@curriculum_app.command("status")
def curriculum_status_cmd(
    model_name: str = typer.Option(OPUS_MODEL, "--model-name", help="Exact model lane to inspect"),
    json_output: bool = typer.Option(False, "--json", help="Emit curriculum status as JSON"),
) -> None:
    base = Path.cwd()
    store = RunStore(base / "data" / "mechanistic.db")
    payload = build_curriculum_status(base, store, model_name=model_name)
    if json_output:
        typer.echo(json.dumps(payload, indent=2, default=str))
        return
    current_module = payload.get("current_module") or {}
    today_slot = payload.get("today_slot") or {}
    queued = payload.get("queued_release") or {}
    typer.echo(f"Model lane: {model_name}")
    typer.echo(f"Current module: Module {current_module.get('number', 1)} - {current_module.get('label', 'n/a')}")
    if today_slot:
        typer.echo(f"Today's slot: {today_slot.get('release_date')} {today_slot.get('label')} @ {today_slot.get('scheduled_publish_at_iso')}")
    else:
        typer.echo("Today's slot: none")
    if queued:
        typer.echo(f"Queued release: {queued.get('id')} ({queued.get('status')})")


@curriculum_app.command("submit")
def curriculum_submit_cmd(
    model_name: str = typer.Option(OPUS_MODEL, "--model-name", help="Exact model lane to submit"),
    json_output: bool = typer.Option(False, "--json", help="Emit queue payload as JSON"),
) -> None:
    base = Path.cwd()
    store = RunStore(base / "data" / "mechanistic.db")
    payload = submit_curriculum_release(base, store, model_name=model_name)
    if json_output:
        typer.echo(json.dumps(payload, indent=2, default=str))
        return
    typer.echo(f"Queued curriculum release: {payload.get('id')}")
    typer.echo(f"Date: {payload.get('release_date')}  Kind: {payload.get('release_kind')}")


@curriculum_app.command("publish-due")
def curriculum_publish_due_cmd(
    json_output: bool = typer.Option(False, "--json", help="Emit published checkpoints as JSON"),
) -> None:
    base = Path.cwd()
    store = RunStore(base / "data" / "mechanistic.db")
    payload = publish_due_curriculum_releases(base, store)
    if json_output:
        typer.echo(json.dumps(payload, indent=2, default=str))
        return
    typer.echo(f"Published {len(payload)} curriculum checkpoint(s)")
    for item in payload:
        typer.echo(f"- {item.get('release_date')} {item.get('release_kind')} -> {item.get('id')}")


@curriculum_app.command("publish")
def curriculum_publish_cmd(
    checkpoint_id: str = typer.Option(..., "--checkpoint-id", help="Queued curriculum release id"),
    force: bool = typer.Option(False, "--force", help="Publish even if release is not yet due"),
    json_output: bool = typer.Option(False, "--json", help="Emit checkpoint as JSON"),
) -> None:
    base = Path.cwd()
    store = RunStore(base / "data" / "mechanistic.db")
    payload = publish_curriculum_release(base, store, queue_id=checkpoint_id, force=force)
    if json_output:
        typer.echo(json.dumps(payload, indent=2, default=str))
        return
    typer.echo(f"Published checkpoint: {payload.get('id')}")
    typer.echo(f"Manifest: {payload.get('manifest_path')}")


@curriculum_app.command("render-readme")
def curriculum_render_readme_cmd() -> None:
    base = Path.cwd()
    store = RunStore(base / "data" / "mechanistic.db")
    render_curriculum_readme(base, store)
    typer.echo("Rendered curriculum README")


@curriculum_app.command("history")
def curriculum_history_cmd(
    model_name: str = typer.Option(OPUS_MODEL, "--model-name", help="Exact model lane to inspect"),
    json_output: bool = typer.Option(False, "--json", help="Emit checkpoint history as JSON"),
) -> None:
    base = Path.cwd()
    store = RunStore(base / "data" / "mechanistic.db")
    payload = curriculum_history(store, model_name=model_name)
    if json_output:
        typer.echo(json.dumps(payload, indent=2, default=str))
        return
    if not payload:
        typer.echo("No curriculum checkpoints yet.")
        return
    for item in payload:
        typer.echo(f"{item.get('release_date')} {item.get('release_kind')} {item.get('git_tag') or 'no-tag'} {item.get('commit_sha') or 'no-commit'}")


@curriculum_app.command("tag-history")
def curriculum_tag_history_cmd() -> None:
    base = Path.cwd()
    store = RunStore(base / "data" / "mechanistic.db")
    checkpoints = curriculum_history(store, model_name=OPUS_MODEL)
    for item in checkpoints:
        typer.echo(f"{item.get('release_date')} {item.get('git_tag') or 'n/a'}")


@curriculum_app.command("install-launchd")
def curriculum_install_launchd_cmd(
    output: Path = typer.Option(Path("/tmp/mechanistic_curriculum_publish.plist"), "--output", help="Where to write the plist example"),
) -> None:
    base = Path.cwd()
    output.write_text(render_launchd_plist(base), encoding="utf-8")
    typer.echo(f"Wrote launchd plist example to {output}")


@curriculum_app.command("build-lookup")
def curriculum_build_lookup_cmd() -> None:
    """Build the flower lookup SQLite cache from the committed .jsonl index.

    Required for curriculum operations (submit, status, etc.).
    Reads from training_data/flower_mechanism_index.jsonl and writes
    data/flower_train_lookup.sqlite. Takes several minutes to build.
    """
    from mechanistic_agent.flower_curriculum import DEFAULT_LOOKUP_CACHE, build_lookup_cache

    typer.echo(f"Building lookup cache at {DEFAULT_LOOKUP_CACHE} ...")
    build_lookup_cache(cache_path=DEFAULT_LOOKUP_CACHE, force=True)
    typer.echo("Done.")


def _execute_harness_eval_run(
    *,
    store: RunStore,
    registry: RegistrySet,
    resolved_eval_set: Any,
    model_name: str,
    thinking_level: Optional[str],
    harness: str,
    run_group: Optional[str],
    max_cases: int,
    max_steps: int,
    max_runtime: float,
    chemistry_backend: str,
    rdkit_cli_command: Optional[str],
    chemistry_backend_parity: bool,
    json_output: bool,
    trace_runtime: bool,
    selected_case_ids: Optional[Sequence[str]] = None,
    planner_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from mechanistic_agent.scoring import score_snapshot_against_known, score_subagents_from_step_outputs

    resolved_eval_set_id = str(resolved_eval_set.eval_set_id)
    model_family = get_model_family(model_name) or "unknown"
    internal_reasoning = to_internal_reasoning_level(thinking_level)
    hashes = registry.bundle_hashes(model_name=model_name)
    eval_run_id = store.create_eval_run(
        eval_set_id=resolved_eval_set_id,
        run_group_name=run_group or f"cli_eval_{harness}",
        model=model_name,
        model_name=model_name,
        model_family=model_family,
        thinking_level=thinking_level,
        harness_bundle_hash=hashes.get("prompt_bundle_hash", ""),
        metadata=planner_metadata,
        status="running",
    )

    if selected_case_ids:
        selected = select_eval_cases(
            cases=resolved_eval_set.cases,
            case_ids=selected_case_ids,
            max_cases=max_cases,
        )
    else:
        selected = select_eval_cases(
            cases=resolved_eval_set.cases,
            max_cases=max_cases,
        )

    if not selected:
        typer.echo("No cases matched the selection criteria")
        store.set_eval_run_status(eval_run_id, "failed")
        raise typer.Exit(code=1)

    selected_case_ids_list = [str(item.get("case_id") or "") for item in selected if str(item.get("case_id") or "")]
    selected_case_ids_digest = case_ids_hash(selected_case_ids_list)
    typer.echo(
        f"Running {len(selected)} cases with model={model_name} harness={harness} "
        f"eval_set_id={resolved_eval_set_id} case_ids_hash={selected_case_ids_digest}"
    )

    coordinator = RunCoordinator(store)
    completed = 0
    failed = 0
    all_graded: List[Dict[str, Any]] = []
    all_latencies: List[float] = []
    backend_usage_counts: Dict[str, int] = {}
    backend_fallback_cases = 0
    backend_error_counts: Dict[str, int] = {}

    for case in selected:
        case_id = str(case.get("case_id") or "")
        input_payload = case.get("input") or {}
        sm = [str(s) for s in input_payload.get("starting_materials", [])]
        prods = [str(p) for p in input_payload.get("products", [])]
        expected = case.get("expected") or {}
        if not isinstance(expected, dict):
            expected = {}
        case_step_count = _eval_case_step_count(case)

        if not sm or not prods:
            continue

        try:
            model_plan = select_step_models(
                model_name=model_name,
                thinking_level=thinking_level,
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
                    "model": model_plan.step_models.get("mechanism_synthesis", model_name),
                    "model_name": model_plan.model_name,
                    "model_family": model_family,
                    "thinking_level": model_plan.thinking_level,
                    "reasoning_level": internal_reasoning,
                    "step_models": model_plan.step_models,
                    "step_reasoning": dict(model_plan.step_reasoning),
                    "optional_llm_tools": ["attempt_atom_mapping", "predict_missing_reagents"],
                    "functional_groups_enabled": True,
                    "intermediate_prediction_enabled": True,
                    "max_steps": max_steps,
                    "max_runtime_seconds": max_runtime,
                    "harness_name": harness,
                    "chemistry_backend": chemistry_backend,
                    "chemistry_backend_parity": chemistry_backend_parity,
                    "rdkit_cli_command": str(rdkit_cli_command or "rdkit_cli"),
                    "runtime_trace_enabled": trace_runtime,
                    "runtime_trace_label": case_id,
                },
                **hashes,
            )

            _t0 = time.monotonic()
            if trace_runtime and not json_output:
                typer.echo(
                    f"    trace start case={case_id} starting={sm} products={prods}"
                )
            coordinator.execute_run(run_id, threading.Event())
            case_latency_ms = (time.monotonic() - _t0) * 1000.0
            all_latencies.append(case_latency_ms)

            snapshot = store.get_run_snapshot(run_id) or {}
            step_outputs = snapshot.get("step_outputs", [])

            graded = score_snapshot_against_known(snapshot, expected) if expected else {"score": 0.0, "passed": False}
            score = float(graded.get("score", 0.0))
            passed = bool(graded.get("passed", False))
            all_graded.append(graded)

            subagent_scores: Dict[str, Any] = {}
            try:
                subagent_scores = score_subagents_from_step_outputs(step_outputs)
            except Exception:
                pass

            cost_summary = snapshot.get("cost_summary") or {}
            run_cost = cost_summary.get("total_cost") or {}

            summary = _build_eval_case_summary(
                snapshot=snapshot,
                score=score,
                passed=passed,
                step_outputs=step_outputs,
                case_step_count=case_step_count,
                subagent_scores=subagent_scores,
                scored_error=graded.get("error"),
            )
            chemistry = summary.get("chemistry_backend") if isinstance(summary.get("chemistry_backend"), dict) else {}
            if chemistry:
                used_counts = chemistry.get("backend_used_counts")
                if isinstance(used_counts, dict):
                    for key, value in used_counts.items():
                        if not isinstance(key, str):
                            continue
                        try:
                            backend_usage_counts[key] = backend_usage_counts.get(key, 0) + int(value)
                        except Exception:
                            continue
                if int(chemistry.get("fallback_count") or 0) > 0:
                    backend_fallback_cases += 1
                err_counts = chemistry.get("rdkit_cli_error_counts")
                if isinstance(err_counts, dict):
                    for key, value in err_counts.items():
                        if not isinstance(key, str):
                            continue
                        try:
                            backend_error_counts[key] = backend_error_counts.get(key, 0) + int(value)
                        except Exception:
                            continue
            store.record_eval_run_result(
                eval_run_id=eval_run_id,
                case_id=case_id,
                run_id=run_id,
                score=score,
                passed=passed,
                cost=run_cost,
                latency_ms=case_latency_ms,
                summary=summary,
            )
            completed += 1
            total_cost = run_cost.get("total_cost", 0.0)
            typer.echo(
                _format_eval_case_result_line(
                    index=completed + failed,
                    case_id=case_id,
                    score=score,
                    passed=passed,
                    total_cost=total_cost,
                    latency_ms=case_latency_ms,
                    summary=summary,
                )
            )
        except Exception as exc:
            all_latencies.append(0.0)
            store.record_eval_run_result(
                eval_run_id=eval_run_id,
                case_id=case_id or uuid.uuid4().hex,
                run_id=None,
                score=0.0,
                passed=False,
                cost={},
                latency_ms=0.0,
                summary={"error": str(exc), "eval_mode": "harness"},
            )
            failed += 1
            typer.echo(f"  [{completed + failed}] {case_id}: FAILED ({exc})")
            all_graded.append(
                {
                    "score": 0.0,
                    "passed": False,
                    "final_product_reached": False,
                    "known_alignment_component": 0.0,
                    "step_validity_component": 0.0,
                }
            )

    store.set_eval_run_status(eval_run_id, "completed")

    if all_graded and not json_output:
        pts = _graded_to_clawdiators_pts(all_graded, all_latencies)
        sep = "=" * 60
        typer.echo("")
        typer.echo(sep)
        typer.echo(f"  Score Summary (harness eval) — {model_name}")
        typer.echo(sep)
        typer.echo(f"  Product Accuracy     : {pts['n_product_hit']:2}/{pts['n_total']} reached   → {pts['product']:3} pts  (30%)")
        typer.echo(f"  Pathway Coverage     : avg {pts['pathway']/300:.3f}           → {pts['pathway']:3} pts  (30%)")
        typer.echo(f"  Electron Push Quality: avg {pts['push']/200:.3f}           → {pts['push']:3} pts  (20%)")
        typer.echo(f"  Speed                : avg {pts['avg_latency_ms']/1000:.1f}s/case       → {pts['speed']:3} pts  (10%)")
        typer.echo(f"  Methodology          : present              → {pts['methodology']:3} pts  (10%)")
        typer.echo(sep)
        typer.echo(f"  TOTAL            : {pts['total']:4} / 1000")
        typer.echo(f"  PREDICTED OUTCOME: {pts['outcome']}  (win ≥700 / draw 400-699 / loss <400)")
        typer.echo(sep)
        typer.echo("")
        if HARNESS_SPEED_CALIBRATION_MS <= 0:
            typer.echo("  NOTE: HARNESS_SPEED_CALIBRATION_MS not set — speed is uncalibrated (100 pts).")
            typer.echo("        Run opus-4.6 dry-run and set constant in main.py to enable live speed scoring.")
        else:
            typer.echo(f"  Speed calibration: {HARNESS_SPEED_CALIBRATION_MS}ms/case (opus-4.6 benchmark = 75 pts)")
        typer.echo("  Harness proxy mapping:")
        typer.echo("    Product Accuracy  ← final_product_reached (harness path score)")
        typer.echo("    Pathway Coverage  ← known_alignment_component (step alignment avg)")
        typer.echo("    Push Quality      ← step_validity_component (validation+mapping avg)")
        typer.echo(sep)

    result_obj = {
        "eval_run_id": eval_run_id,
        "model": model_name,
        "thinking_level": thinking_level,
        "harness": harness,
        "chemistry_backend": chemistry_backend,
        "chemistry_backend_parity": chemistry_backend_parity,
        "rdkit_cli_command": str(rdkit_cli_command or "rdkit_cli"),
        "completed": completed,
        "failed": failed,
        "eval_set_id": resolved_eval_set_id,
        "eval_case_ids_hash": selected_case_ids_digest,
        "avg_latency_ms": round(sum(all_latencies) / len(all_latencies), 1) if all_latencies else 0.0,
        "backend_usage_counts": backend_usage_counts,
        "backend_fallback_cases": backend_fallback_cases,
        "backend_error_counts": backend_error_counts,
        "planner_metadata": planner_metadata or {},
    }
    if json_output:
        typer.echo(json.dumps(result_obj, indent=2))
    return result_obj


@app.command(name="eval")
def eval_cmd(
    eval_set_id: Optional[str] = typer.Option(None, "--eval-set-id", help="Eval set to run against (ignored if --tier/--all-tiers)"),
    model_name: str = typer.Option(
        get_default_model(), "--model-name", "--model",
        help="Model identifier (e.g. gpt-5.4, claude-opus-4.6)",
    ),
    thinking_level: Optional[str] = typer.Option(
        None, "--thinking-level", "--reasoning", help="Thinking level: low, high, or max (model-dependent)",
    ),
    tier: Optional[str] = typer.Option(None, "--tier", help="Tier name: easy, medium, or hard"),
    all_tiers: bool = typer.Option(
        False,
        "--all-tiers",
        help="Run easy, medium, and hard eval tiers in one command",
    ),
    tier_map_path: Optional[str] = typer.Option(
        None,
        "--tier-map-path",
        help="Path to tier eval-set mapping JSON (default: training_data/baseline_tier_eval_set_map.json)",
    ),
    tier_definitions_path: Optional[str] = typer.Option(
        None,
        "--tier-definitions-path",
        help=(
            "Path to tier case-id definitions JSON "
            "(default: training_data/baseline_tiers_clawdiator.json, fallback: training_data/eval_tiers.json)"
        ),
    ),
    run_group_prefix: str = typer.Option(
        "cli_eval_tier",
        "--run-group-prefix",
        help="Run-group prefix for --all-tiers mode; final name is <prefix>_<tier>",
    ),
    case_ids: Optional[List[str]] = typer.Option(None, "--case-id", help="Specific case IDs to run (repeatable)"),
    harness: str = typer.Option("default", "--harness", help="Harness name from harness_versions/"),
    run_group: Optional[str] = typer.Option(None, "--run-group", help="Run group name for leaderboard"),
    max_cases: int = typer.Option(
        25, "--max-cases",
        help="Max cases per run (per tier when using --tier/--all-tiers)",
    ),
    max_per_tier: Optional[int] = typer.Option(
        None, "--max-per-tier",
        help="Max cases per tier (only with --tier/--all-tiers). Overrides --max-cases for each tier when set.",
    ),
    max_steps: int = typer.Option(10, "--max-steps", help="Max mechanism steps per case"),
    max_runtime: float = typer.Option(600.0, "--max-runtime", help="Per-case timeout in seconds"),
    chemistry_backend: str = typer.Option(
        "auto",
        "--chemistry-backend",
        help="Chemistry backend: python (default, no CLI), rdkit_cli, or auto",
    ),
    rdkit_cli_command: Optional[str] = typer.Option(
        None,
        "--rdkit-cli-command",
        help="Optional rdkit_cli executable command/path (used when chemistry backend is rdkit_cli/auto)",
    ),
    chemistry_backend_parity: bool = typer.Option(
        False,
        "--chemistry-backend-parity",
        help="Enable dual-run parity compare (Python authoritative on mismatch)",
    ),
    allow_repeats: bool = typer.Option(
        False,
        "--allow-repeats",
        help="Allow rerunning cases already attempted for this model + thinking level",
    ),
    leaderboard_route: str = typer.Option(
        "auto",
        "--leaderboard-route",
        help="Development leaderboard route: auto, same, extend, next, or custom",
    ),
    leaderboard_status_only: bool = typer.Option(
        False,
        "--leaderboard-status-only",
        help="Show development leaderboard route status for a tier flow and exit",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Auto-confirm the recommended development leaderboard route in TTY mode",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit results as JSON"),
    trace_runtime: bool = typer.Option(
        True,
        "--trace-runtime/--no-trace-runtime",
        help="Emit compact per-step runtime trace lines during eval runs.",
    ),
    allow_holdout: bool = typer.Option(False, "--allow-holdout", hidden=True),
) -> None:
    """Run a full harness eval set and record results on the leaderboard.

    Uses the harness pipeline (not baseline single-shot) to evaluate each case.
    Results are stored in the DB and appear on the leaderboard.

    Use --tier or --all-tiers to run tiered evals (easy/medium/hard). Use --max-cases
    or --max-per-tier to limit how many examples run per tier. Use -h/--help to list
    all options.
    """
    from typer.models import OptionInfo

    def _unwrap_option(value, fallback=None):  # type: ignore[no-untyped-def]
        if isinstance(value, OptionInfo):
            return value.default if value.default is not None else fallback
        return value if value is not None else fallback

    # When eval_cmd is called programmatically (e.g. from eval_runset_official_cmd),
    # parameters may still carry Typer's OptionInfo defaults. Normalize them here so
    # the rest of the function can assume plain Python types.
    eval_set_id = _unwrap_option(eval_set_id)
    model_name = _unwrap_option(model_name)
    thinking_level = _unwrap_option(thinking_level)
    tier = _unwrap_option(tier)
    all_tiers = bool(_unwrap_option(all_tiers, False))
    tier_map_path = _unwrap_option(tier_map_path)
    tier_definitions_path = _unwrap_option(tier_definitions_path)
    run_group_prefix = _unwrap_option(run_group_prefix, "")
    case_ids = _unwrap_option(case_ids)
    harness = _unwrap_option(harness, "default")
    run_group = _unwrap_option(run_group)
    max_cases = int(_unwrap_option(max_cases, 25))
    max_per_tier = _unwrap_option(max_per_tier)
    max_steps = int(_unwrap_option(max_steps, 10))
    max_runtime = float(_unwrap_option(max_runtime, 300.0))
    chemistry_backend = str(_unwrap_option(chemistry_backend, "auto") or "auto").strip().lower()
    rdkit_cli_command = _unwrap_option(rdkit_cli_command)
    chemistry_backend_parity = bool(_unwrap_option(chemistry_backend_parity, False))
    allow_repeats = bool(_unwrap_option(allow_repeats, False))
    leaderboard_route = str(_unwrap_option(leaderboard_route, "auto") or "auto").strip().lower()
    leaderboard_status_only = bool(_unwrap_option(leaderboard_status_only, False))
    yes = bool(_unwrap_option(yes, False))
    json_output = bool(_unwrap_option(json_output, False))
    trace_runtime = bool(_unwrap_option(trace_runtime, True))
    allow_holdout = bool(_unwrap_option(allow_holdout, False))

    if thinking_level is not None:
        thinking_level = thinking_level.strip().lower()
        if thinking_level not in {"low", "high", "max"}:
            raise typer.BadParameter("thinking-level must be one of: low, high, max")
    model_name = _canonicalize_model_name_or_raise(model_name)

    if tier and tier not in {"easy", "medium", "hard"}:
        raise typer.BadParameter("tier must be one of: easy, medium, hard")
    if leaderboard_route not in {"auto", "same", "extend", "next", "custom"}:
        raise typer.BadParameter("leaderboard-route must be one of: auto, same, extend, next, custom")
    if chemistry_backend not in {"python", "rdkit_cli", "auto"}:
        raise typer.BadParameter("chemistry-backend must be one of: python, rdkit_cli, auto")
    requested_tiers = _normalize_requested_baseline_tiers(
        ([tier] if tier else None),
        all_tiers=all_tiers,
    )
    run_group_prefix = run_group_prefix.strip()
    if not run_group_prefix:
        raise typer.BadParameter("run-group-prefix must not be empty")
    _max_per_tier: Optional[int] = max_per_tier if isinstance(max_per_tier, int) else None
    if _max_per_tier is not None and _max_per_tier < 1:
        raise typer.BadParameter("--max-per-tier must be at least 1 when set")

    base = Path.cwd()
    store = RunStore(base / "data" / "mechanistic.db")
    registry = RegistrySet(base)
    model_family = get_model_family(model_name) or "unknown"
    if requested_tiers:
        if eval_set_id and not json_output:
            typer.echo("Ignoring --eval-set-id because --tier/--all-tiers was provided.")
        if all_tiers and not json_output:
            typer.echo(
                "Development leaderboard route planner is bypassed for --all-tiers; "
                "running the explicit sweep instead."
            )
        resolved_tier_map_path = (
            Path(tier_map_path).expanduser().resolve()
            if tier_map_path
            else (base / BASELINE_TIER_MAP_DEFAULT_PATH).resolve()
        )
        resolved_tier_definitions_path = _resolve_baseline_tier_definitions_path(
            base=base,
            override_path=tier_definitions_path,
        )
        tier_eval_set_ids = _load_baseline_tier_eval_set_map(resolved_tier_map_path)
        execution_plan = _build_baseline_tier_execution_plan(
            base=base,
            store=store,
            requested_tiers=requested_tiers,
            tier_eval_set_ids=tier_eval_set_ids,
            tier_definitions_path=resolved_tier_definitions_path,
            allow_holdout=allow_holdout,
        )

        if not all_tiers:
            requested_tier = requested_tiers[0]
            status = _build_development_leaderboard_status(
                base=base,
                store=store,
                model_name=model_name,
                thinking_level=thinking_level,
                requested_tier=requested_tier,
                tier_map_path=tier_map_path,
                tier_definitions_path=tier_definitions_path,
                allow_holdout=allow_holdout,
            )
            if not json_output:
                _print_development_leaderboard_status(
                    status=status,
                    model_name=model_name,
                    thinking_level=thinking_level,
                )
            if leaderboard_status_only:
                if json_output:
                    typer.echo(json.dumps(status, indent=2, default=str))
                return

            requested_context = status["tier_contexts"][requested_tier]
            planner_bypassed = bool(case_ids) or leaderboard_route == "custom"
            selected_tier = requested_tier
            resolved_eval_set = requested_context["resolved_eval_set"]
            planner_metadata: Optional[Dict[str, Any]] = None

            if planner_bypassed:
                if not json_output:
                    if case_ids:
                        typer.echo("Development leaderboard planner bypassed because --case-id was provided.")
                    else:
                        typer.echo("Development leaderboard planner bypassed by --leaderboard-route custom.")
                selected_case_ids = [
                    str(case_id).strip()
                    for case_id in (case_ids or [])
                    if str(case_id).strip()
                ]
                if not selected_case_ids:
                    selected_case_ids = list(requested_context.get("case_ids") or [])
                    if not allow_repeats:
                        selected_case_ids = _filter_unrun_case_ids_for_model(
                            store=store,
                            case_ids=selected_case_ids,
                            model_name=model_name,
                            thinking_level=thinking_level,
                            model_family=model_family,
                        )
                        if not selected_case_ids:
                            raise typer.BadParameter(
                                f"Tier '{requested_tier}' has 0 unrun cases for model={model_name} "
                                f"thinking={thinking_level or 'none'}; pass --allow-repeats to rerun."
                            )
                    if max_cases > 0:
                        selected_case_ids = selected_case_ids[:max_cases]
                planner_metadata = {
                    "planner_version": status["policy"]["version"],
                    "route_kind": "custom",
                    "tier_name": requested_tier,
                    "selected_case_count": len(selected_case_ids),
                    "selected_case_ids_hash": case_ids_hash(selected_case_ids),
                    "policy_snapshot": {
                        "version": status["policy"]["version"],
                        "path": status["policy"]["path"],
                        "comparison_scope": status["policy"]["comparison_scope"],
                        "active_tier_sources": dict(status["policy"]["active_tier_sources"]),
                    },
                    "source_eval_run_id": None,
                    "source_case_ids_hash": None,
                    "is_policy_canonical_slice": False,
                }
            else:
                selected_route = _select_development_route(
                    status=status,
                    requested_route=leaderboard_route,
                    auto_confirm=yes,
                    json_output=json_output,
                )
                selected_tier = str(selected_route["tier_name"])
                resolved_eval_set = status["tier_contexts"][selected_tier]["resolved_eval_set"]
                selected_case_ids = list(selected_route["case_ids"])
                planner_metadata = dict(selected_route["planner_metadata"])
                if not json_output:
                    typer.echo(
                        f"Selected development leaderboard route '{selected_route['key']}' "
                        f"for tier '{selected_tier}'."
                    )

            tier_run_group = run_group or f"{run_group_prefix}_{selected_tier}"
            result_obj = _execute_harness_eval_run(
                store=store,
                registry=registry,
                resolved_eval_set=resolved_eval_set,
                model_name=model_name,
                thinking_level=thinking_level,
                harness=harness,
                run_group=tier_run_group,
                max_cases=max_cases,
                max_steps=max_steps,
                max_runtime=max_runtime,
                chemistry_backend=chemistry_backend,
                rdkit_cli_command=rdkit_cli_command,
                chemistry_backend_parity=chemistry_backend_parity,
                json_output=json_output,
                trace_runtime=trace_runtime,
                selected_case_ids=selected_case_ids,
                planner_metadata=planner_metadata,
            )
            if not json_output:
                typer.echo(
                    f"\nHarness eval complete: completed={result_obj['completed']} failed={result_obj['failed']} "
                    f"eval_run_id={result_obj['eval_run_id']} case_ids_hash={result_obj['eval_case_ids_hash']}"
                )
            return

        aggregated: List[Dict[str, Any]] = []
        for item in execution_plan:
            tier_name = str(item.get("tier") or "")
            tier_eval_set_id = str(item.get("eval_set_id") or "")
            tier_case_ids = list(item.get("case_ids") or [])
            if not allow_repeats:
                tier_case_ids = _filter_unrun_case_ids_for_model(
                    store=store,
                    case_ids=tier_case_ids,
                    model_name=model_name,
                    thinking_level=thinking_level,
                    model_family=model_family,
                )
                if not tier_case_ids:
                    raise typer.BadParameter(
                        f"Tier '{tier_name}' has 0 unrun cases for model={model_name} "
                        f"thinking={thinking_level or 'none'}; pass --allow-repeats to rerun."
                    )
            tier_run_group = f"{run_group_prefix}_{tier_name}"
            tier_limit = _max_per_tier if _max_per_tier is not None else max_cases
            if not json_output:
                typer.echo(
                    f"\nRunning harness eval tier '{tier_name}' "
                    f"(eval_set_id={tier_eval_set_id}, cases={len(tier_case_ids)}, max={tier_limit})..."
                )

            result_obj = _execute_harness_eval_run(
                store=store,
                registry=registry,
                resolved_eval_set=item["resolved_eval_set"],
                model_name=model_name,
                thinking_level=thinking_level,
                harness=harness,
                run_group=tier_run_group,
                max_cases=tier_limit,
                max_steps=max_steps,
                max_runtime=max_runtime,
                chemistry_backend=chemistry_backend,
                rdkit_cli_command=rdkit_cli_command,
                chemistry_backend_parity=chemistry_backend_parity,
                json_output=False,
                trace_runtime=trace_runtime,
                selected_case_ids=tier_case_ids,
                planner_metadata=None,
            )
            aggregated.append(
                {
                    "tier": tier_name,
                    "eval_set_id": tier_eval_set_id,
                    "eval_run_id": str(result_obj.get("eval_run_id") or ""),
                    "run_group_name": tier_run_group,
                    "status": "completed",
                }
            )

        if json_output:
            typer.echo(
                json.dumps(
                    {
                        "mode": "eval_tiers",
                        "model": model_name,
                        "thinking_level": thinking_level,
                        "harness": harness,
                        "tier_map_path": str(resolved_tier_map_path),
                        "tier_definitions_path": str(resolved_tier_definitions_path),
                        "results": aggregated,
                    },
                    indent=2,
                )
            )
        else:
            typer.echo("\nAll requested harness eval tiers complete.")
        return

    if leaderboard_status_only:
        raise typer.BadParameter("--leaderboard-status-only requires --tier")
    if leaderboard_route != "auto":
        raise typer.BadParameter("--leaderboard-route is only supported with --tier")
    if yes:
        raise typer.BadParameter("--yes is only supported with --tier")
    if not eval_set_id:
        raise typer.BadParameter("--eval-set-id is required unless using --tier or --all-tiers")

    try:
        resolved_eval_set = resolve_eval_set(
            store=store,
            requested_eval_set_id=eval_set_id,
        )
    except EvalSetResolutionError as exc:
        raise typer.BadParameter(str(exc)) from exc
    resolved_eval_set_id = str(resolved_eval_set.eval_set_id)
    is_holdout = resolved_eval_set.purpose == "leaderboard_holdout"
    if is_holdout and not allow_holdout:
        raise typer.BadParameter(
            "leaderboard_holdout eval sets are restricted to 'eval-runset-official'"
        )

    tier_case_ids: Optional[List[str]] = None
    if tier:
        resolved_tier_definitions_path = _resolve_baseline_tier_definitions_path(
            base=base,
            override_path=tier_definitions_path,
        )
        try:
            tier_data = _load_eval_tier_ids(resolved_tier_definitions_path)
        except typer.BadParameter as exc:
            typer.echo(f"Warning: {exc}; ignoring --tier")
        else:
            tier_case_ids = tier_data.get(tier, [])

    all_cases = list(resolved_eval_set.cases)
    if not all_cases:
        typer.echo(f"No cases found for eval set {resolved_eval_set_id}")
        raise typer.Exit(code=1)

    if case_ids:
        selected_case_ids = [
            str(case_id).strip()
            for case_id in case_ids
            if str(case_id).strip()
        ]
    else:
        candidate_cases = select_eval_cases(
            cases=all_cases,
            case_ids=None,
            tier_case_ids=tier_case_ids,
            max_cases=None,
        )
        candidate_case_ids = [
            str(item.get("case_id") or "")
            for item in candidate_cases
            if str(item.get("case_id") or "")
        ]
        selected_case_ids = list(candidate_case_ids)
        if not allow_repeats and not is_holdout:
            selected_case_ids = _filter_unrun_case_ids_for_model(
                store=store,
                case_ids=candidate_case_ids,
                model_name=model_name,
                thinking_level=thinking_level,
                model_family=model_family,
            )
            if not selected_case_ids:
                typer.echo(
                    "No unrun cases remain for the selected scope/model/thinking. "
                    "Pass --allow-repeats to rerun cases."
                )
                raise typer.Exit(code=1)
        if max_cases > 0:
            selected_case_ids = selected_case_ids[:max_cases]

    result_obj = _execute_harness_eval_run(
        store=store,
        registry=registry,
        resolved_eval_set=resolved_eval_set,
        model_name=model_name,
        thinking_level=thinking_level,
        harness=harness,
        run_group=run_group,
        max_cases=max_cases,
        max_steps=max_steps,
        max_runtime=max_runtime,
        chemistry_backend=chemistry_backend,
        rdkit_cli_command=rdkit_cli_command,
        chemistry_backend_parity=chemistry_backend_parity,
        json_output=json_output,
        trace_runtime=trace_runtime,
        selected_case_ids=selected_case_ids,
        planner_metadata=None,
    )
    if not json_output:
        typer.echo(
            f"\nHarness eval complete: completed={result_obj['completed']} failed={result_obj['failed']} "
            f"eval_run_id={result_obj['eval_run_id']} case_ids_hash={result_obj['eval_case_ids_hash']}"
        )


@app.command(name="eval-runset-official")
def eval_runset_official_cmd(
    eval_set_id: Optional[str] = typer.Option(
        None,
        "--eval-set-id",
        help="Optional holdout eval set id (defaults to latest purpose=leaderboard_holdout).",
    ),
    allow_non_holdout: bool = typer.Option(
        False,
        "--allow-non-holdout",
        help="Explicit opt-in to run a non-holdout eval set from this command.",
    ),
    model_name: str = typer.Option(
        get_default_model(), "--model-name", "--model",
        help="Model identifier (e.g. gpt-5.4, claude-opus-4.6)",
    ),
    thinking_level: Optional[str] = typer.Option(
        None, "--thinking-level", "--reasoning", help="Thinking level: low, high, or max (model-dependent)"
    ),
    tier: Optional[str] = typer.Option(
        None,
        "--tier",
        help="Optional official tier filter: easy (1-2 steps), medium (3 steps), hard (4+ steps).",
    ),
    case_ids: Optional[List[str]] = typer.Option(None, "--case-id", help="Specific case IDs to run (repeatable)"),
    harness: str = typer.Option("default", "--harness", help="Harness name from harness_versions/"),
    run_group: Optional[str] = typer.Option(
        None, "--run-group", help="Run group name for leaderboard (default: official_holdout_harness)"
    ),
    max_cases: Optional[int] = typer.Option(
        None,
        "--max-cases",
        "--num-examples",
        help="Max cases/examples to run (default: 20).",
    ),
    max_steps: int = typer.Option(10, "--max-steps", help="Max mechanism steps per case"),
    max_runtime: float = typer.Option(300.0, "--max-runtime", help="Per-case timeout in seconds"),
    chemistry_backend: str = typer.Option(
        "auto",
        "--chemistry-backend",
        help="Chemistry backend: python (default, no CLI), rdkit_cli, or auto",
    ),
    rdkit_cli_command: Optional[str] = typer.Option(
        None,
        "--rdkit-cli-command",
        help="Optional rdkit_cli executable command/path (used when chemistry backend is rdkit_cli/auto)",
    ),
    chemistry_backend_parity: bool = typer.Option(
        False,
        "--chemistry-backend-parity",
        help="Enable dual-run parity compare (Python authoritative on mismatch)",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit results as JSON"),
    trace_runtime: bool = typer.Option(
        True,
        "--trace-runtime/--no-trace-runtime",
        help="Emit compact per-step runtime trace lines during official eval runs.",
    ),
) -> None:
    """Run the official holdout-only leaderboard eval.

    Defaults to 20 cases.
    The holdout set is sourced from FlowER test.txt (separate from the Clawdiators
    arena reactions, which come from training_data/eval_set.json) and is never used
    for prompt tuning or harness development.
    """
    base = Path.cwd()
    store = RunStore(base / "data" / "mechanistic.db")
    model_name = _canonicalize_model_name_or_raise(model_name)
    resolved_max_cases = int(max_cases) if max_cases is not None else 20
    if resolved_max_cases < 1:
        raise typer.BadParameter("--max-cases/--num-examples must be at least 1")
    if thinking_level is not None:
        thinking_level = thinking_level.strip().lower()
        if thinking_level not in {"low", "high", "max"}:
            raise typer.BadParameter("thinking-level must be one of: low, high, max")
    if tier is not None:
        tier = tier.strip().lower()
        if tier not in OFFICIAL_TIER_NAMES:
            raise typer.BadParameter("tier must be one of: easy, medium, hard")

    try:
        resolved = resolve_eval_set(
            store=store,
            requested_eval_set_id=eval_set_id,
            require_purpose=(None if allow_non_holdout else "leaderboard_holdout"),
            default_purpose=("leaderboard_holdout" if not eval_set_id else None),
        )
    except EvalSetResolutionError as exc:
        raise typer.BadParameter(str(exc)) from exc

    run_group_name = run_group or (
        "official_holdout_harness" if resolved.purpose == "leaderboard_holdout" else "official_compare_harness"
    )
    selected_case_ids = list(case_ids or [])
    if not selected_case_ids:
        candidate_cases = list(resolved.cases)
        if tier:
            tier_case_id_set = set(_official_case_ids_for_tier(cases=candidate_cases, tier_name=tier))
            candidate_cases = [
                case for case in candidate_cases
                if str(case.get("case_id") or "").strip() in tier_case_id_set
            ]

        candidate_case_ids = [
            str(case.get("case_id") or "").strip()
            for case in candidate_cases
            if str(case.get("case_id") or "").strip()
        ]
        attempted_case_ids = _list_attempted_eval_case_ids_for_scope(
            store=store,
            eval_set_id=str(resolved.eval_set_id or ""),
            model_name=model_name,
            thinking_level=thinking_level,
            run_group_name=run_group_name,
        )
        selected_case_ids, selection_meta = _select_case_ids_resume_then_cycle(
            candidate_case_ids=candidate_case_ids,
            attempted_case_ids=attempted_case_ids,
            max_cases=resolved_max_cases,
        )
        if not selected_case_ids:
            raise typer.BadParameter("No official eval cases available for the requested scope")
        if not json_output:
            if tier:
                typer.echo(f"Official tier filter '{tier}': {selection_meta['candidate_count']} candidate cases")
            typer.echo(
                "Official resume selection: "
                f"{selection_meta['unrun_count']} unrun of {selection_meta['candidate_count']} candidates; "
                f"running {selection_meta['target_count']} case(s)."
            )
            if selection_meta["wrapped"]:
                if int(selection_meta["unrun_count"]) == 0:
                    typer.echo("All candidate cases were already attempted; cycling back to the beginning.")
                else:
                    wrapped_count = int(selection_meta["target_count"]) - int(selection_meta["unrun_count"])
                    typer.echo(
                        f"Only {selection_meta['unrun_count']} unrun case(s) remained; "
                        f"cycled the remaining {max(0, wrapped_count)} case(s) from the start."
                    )
    elif tier:
        tier_case_id_set = set(_official_case_ids_for_tier(cases=resolved.cases, tier_name=tier))
        selected_case_ids = [
            str(case_id).strip()
            for case_id in selected_case_ids
            if str(case_id).strip() in tier_case_id_set
        ]
        if not selected_case_ids:
            raise typer.BadParameter(
                "No provided --case-id values matched the requested --tier filter"
            )

    if not json_output:
        typer.echo(
            "Prompt/few-shot update status: eval-runset-official is read-only for "
            "skills/mechanistic prompts and few-shot files (no updates in this command)."
        )

    eval_cmd(
        eval_set_id=str(resolved.eval_set_id),
        model_name=model_name,
        thinking_level=thinking_level,
        tier=None,
        case_ids=selected_case_ids,
        harness=harness,
        run_group=run_group_name,
        max_cases=resolved_max_cases,
        max_steps=max_steps,
        max_runtime=max_runtime,
        chemistry_backend=chemistry_backend,
        rdkit_cli_command=rdkit_cli_command,
        chemistry_backend_parity=chemistry_backend_parity,
        json_output=json_output,
        trace_runtime=trace_runtime,
        allow_holdout=True,
    )


if __name__ == "__main__":
    app()
