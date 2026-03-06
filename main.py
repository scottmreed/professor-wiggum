"""CLI entrypoint for the local-first mechanistic runtime."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from mechanistic_agent import OPTIONAL_LLM_TOOL_NAMES, ReactionInputs
from mechanistic_agent.core import RegistrySet, RunCoordinator, RunStore, select_step_models
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
from mechanistic_agent.model_registry import get_default_model, get_model_family, to_internal_reasoning_level

try:  # pragma: no cover - optional helper
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - fallback when python-dotenv is absent
    def load_dotenv(_: Path) -> bool:  # type: ignore[override]
        return False


app = typer.Typer(add_completion=False, no_args_is_help=True)
curriculum_app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(curriculum_app, name="curriculum")
load_dotenv(Path.cwd() / ".env")


def _parse_materials(raw: Optional[str], fallback: List[str]) -> List[str]:
    if raw is None:
        return list(fallback)
    return [item.strip() for item in raw.split(",") if item.strip()]


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
        top_quality = float(top.get("weighted_quality_score") or top.get("mean_quality_score") or 0.0)
        top_pass_rate = float(top.get("weighted_pass_rate") or top.get("deterministic_pass_rate") or 0.0) * 100.0
        top_group = str(top.get("run_group_name") or "n/a")
        lines.extend(
            [
                "## Current SOTA",
                "",
                f"- Model: `{top_model}`",
                f"- Thinking: `{top_thinking}`",
                f"- Mean quality: `{top_quality:.3f}`",
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
                "| Rank | Model | Thinking | Type | Quality | Pass | Cases | Cost | Group |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
    else:
        lines.extend(
            [
                "| Rank | Model | Thinking | Type | Quality | Pass | Cases | Group |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
    if not items:
        if includes_cost:
            lines.append("| - | - | - | - | - | - | - | - | No completed rows |")
        else:
            lines.append("| - | - | - | - | - | - | - | No completed rows |")
        return "\n".join(lines)

    for index, row in enumerate(items, 1):
        model = str(row.get("model_name") or row.get("model") or "unknown")
        thinking = str(row.get("thinking_level") or "none")
        run_type = "Baseline" if row.get("is_baseline") else "Harness"
        quality = f"{float(row.get('weighted_quality_score') or row.get('mean_quality_score') or 0.0):.3f}"
        pass_rate = f"{float(row.get('weighted_pass_rate') or row.get('deterministic_pass_rate') or 0.0) * 100.0:.1f}%"
        case_count = str(row.get("case_count") or 0)
        group = str(row.get("run_group_name") or "n/a")
        if includes_cost:
            total_cost = float(row.get("total_cost") or 0.0)
            cost_display = f"${total_cost:.3f}"
            lines.append(
                f"| {index} | `{model}` | `{thinking}` | {run_type} | {quality} | {pass_rate} | {case_count} | {cost_display} | `{group}` |"
            )
        else:
            lines.append(
                f"| {index} | `{model}` | `{thinking}` | {run_type} | {quality} | {pass_rate} | {case_count} | `{group}` |"
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
    max_runtime: float = typer.Option(240.0, "--max-runtime", help="Maximum runtime in seconds"),
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
        help="Optional thinking level: low or high",
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
    if thinking_level is not None:
        thinking_level = thinking_level.strip().lower()
        if thinking_level not in {"low", "high"}:
            raise typer.BadParameter("thinking-level must be one of: low, high")

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
    model_name: str = typer.Option(
        get_default_model(),
        "--model-name",
        "--model",
        help="Model identifier (e.g. gpt-5.4, claude-opus-4.6)",
    ),
    thinking_level: Optional[str] = typer.Option(
        None, "--thinking-level", "--reasoning", help="Thinking level: low or high"
    ),
    temperature: float = typer.Option(25.0, "--temperature", help="Reaction temperature in Celsius"),
    ph: Optional[float] = typer.Option(None, "--ph", help="Observed reaction pH (optional)"),
    max_cases: int = typer.Option(25, "--max-cases", help="Max cases when running an eval set"),
    timeout: float = typer.Option(180.0, "--timeout", help="Per-case timeout in seconds"),
    json_output: bool = typer.Option(False, "--json", help="Emit results as JSON"),
) -> None:
    """Run harness-free single-shot baseline mechanism prediction.

    Either provide --starting/--products for a single case, or --eval-set-id
    to run against an eval set and record results on the leaderboard.
    """
    from mechanistic_agent.core.baseline_runner import (
        BASELINE_GROUP_PREFIX,
        BaselineRunner,
        score_baseline_result,
    )

    if thinking_level is not None:
        thinking_level = thinking_level.strip().lower()
        if thinking_level not in {"low", "high"}:
            raise typer.BadParameter("thinking-level must be one of: low, high")

    runner = BaselineRunner()

    if eval_set_id:
        # ---- Eval-set mode: run all cases and record to leaderboard ----
        base = Path.cwd()
        store = RunStore(base / "data" / "mechanistic.db")
        registry = RegistrySet(base)
        model_family = get_model_family(model_name) or "unknown"
        harness_hash = registry.bundle_hashes().get("prompt_bundle_hash", "")

        eval_run_id = store.create_eval_run(
            eval_set_id=eval_set_id,
            run_group_name=BASELINE_GROUP_PREFIX,
            model=model_name,
            model_name=model_name,
            model_family=model_family,
            thinking_level=thinking_level,
            harness_bundle_hash=harness_hash,
            status="running",
        )

        cases = store.list_eval_set_cases(eval_set_id)
        if len(cases) > max_cases:
            cases = cases[:max_cases]

        completed = 0
        failed = 0
        for case in cases:
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
                )
                graded = score_baseline_result(result, expected if expected else None)
                score = float(graded["score"])
                passed = bool(graded["passed"])
                latency_ms = float(result.get("latency_ms") or 0.0)
                summary: Dict[str, Any] = {
                    "score": score,
                    "passed": passed,
                    "step_count": graded.get("step_count"),
                    "mechanism_type": graded.get("mechanism_type"),
                    "scoring_breakdown": graded.get("scoring_breakdown", {}),
                    "error": graded.get("error"),
                    "eval_mode": "baseline",
                    "subagent_scores": {
                        "full_mechanism_baseline": {
                            "quality_score": score,
                            "pass_rate": 1.0 if passed else 0.0,
                            "case_count": 1,
                        }
                    },
                }
                store.record_eval_run_result(
                    eval_run_id=eval_run_id,
                    case_id=case_id or uuid.uuid4().hex,
                    run_id=None,
                    score=score,
                    passed=passed,
                    cost={},
                    latency_ms=latency_ms,
                    summary=summary,
                )
                completed += 1
                typer.echo(f"  [{completed + failed}] {case_id}: score={score:.3f} passed={passed}")
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
                failed += 1
                typer.echo(f"  [{completed + failed}] {case_id}: FAILED ({exc})")

        store.set_eval_run_status(eval_run_id, "completed")
        result_obj = {
            "eval_run_id": eval_run_id,
            "model": model_name,
            "thinking_level": thinking_level,
            "completed": completed,
            "failed": failed,
        }
        if json_output:
            typer.echo(json.dumps(result_obj, indent=2))
        else:
            typer.echo(f"\nBaseline eval complete: {completed} passed, {failed} failed")
            typer.echo(f"Eval run ID: {eval_run_id}")

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
            if result.get("error"):
                typer.echo(f"Error: {result['error']}")


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
            f"{'Rank':<5} {'Model':<25} {'Thinking':<8} {'Type':<8} {'Quality':<8} {'Pass':<7} {'Cases':<6} {'Cost':<8} {'Group'}"
            if includes_cost
            else f"{'Rank':<5} {'Model':<25} {'Thinking':<8} {'Type':<8} {'Quality':<8} {'Pass':<7} {'Cases':<6} {'Group'}"
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
            quality = f"{float(row.get('weighted_quality_score') or row.get('mean_quality_score') or 0):.3f}"
            pass_rate = f"{float(row.get('weighted_pass_rate') or row.get('deterministic_pass_rate') or 0) * 100:.1f}%"
            case_count = str(row.get("case_count") or 0)
            group = row.get("run_group_name") or "n/a"
            if includes_cost:
                total_cost = float(row.get("total_cost") or 0)
                cost_display = f"${total_cost:.3f}" if total_cost > 0 else "$0.000"
                lines.append(
                    f"{i:<5} {model:<25} {thinking:<8} {run_type:<8} {quality:<8} {pass_rate:<7} {case_count:<6} {cost_display:<8} {group}"
                )
            else:
                lines.append(
                    f"{i:<5} {model:<25} {thinking:<8} {run_type:<8} {quality:<8} {pass_rate:<7} {case_count:<6} {group}"
                )
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
    resolved_eval_set_id = eval_set_id
    if resolved_eval_set_id:
        row = store.get_eval_set(resolved_eval_set_id)
        if row is None:
            raise typer.BadParameter(f"Eval set not found: {resolved_eval_set_id}")
        if str(row.get("purpose") or "general") != "leaderboard_holdout":
            raise typer.BadParameter("eval-set-id must point to purpose=leaderboard_holdout")
    else:
        holdouts = store.list_eval_sets(purpose="leaderboard_holdout")
        if not holdouts:
            raise typer.BadParameter("No purpose=leaderboard_holdout eval set found")
        resolved_eval_set_id = str(holdouts[0].get("id") or "")
        if not resolved_eval_set_id:
            raise typer.BadParameter("Unable to resolve holdout eval set id")

    leaderboard(
        eval_set_id=resolved_eval_set_id,
        limit=limit,
        json_output=json_output,
        markdown_output=markdown_output,
        output_path=output_path,
        completed_only=completed_only,
    )


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


@app.command(name="eval")
def eval_cmd(
    eval_set_id: str = typer.Option(..., "--eval-set-id", help="Eval set to run against"),
    model_name: str = typer.Option(
        get_default_model(), "--model-name", "--model",
        help="Model identifier (e.g. gpt-5.4, claude-opus-4.6)",
    ),
    thinking_level: Optional[str] = typer.Option(
        None, "--thinking-level", "--reasoning", help="Thinking level: low or high"
    ),
    tier: Optional[str] = typer.Option(None, "--tier", help="Tier name: easy, medium, or hard"),
    case_ids: Optional[List[str]] = typer.Option(None, "--case-id", help="Specific case IDs to run (repeatable)"),
    harness: str = typer.Option("default", "--harness", help="Harness name from harness_versions/"),
    run_group: Optional[str] = typer.Option(None, "--run-group", help="Run group name for leaderboard"),
    max_cases: int = typer.Option(25, "--max-cases", help="Max cases to run"),
    max_steps: int = typer.Option(10, "--max-steps", help="Max mechanism steps per case"),
    max_runtime: float = typer.Option(300.0, "--max-runtime", help="Per-case timeout in seconds"),
    json_output: bool = typer.Option(False, "--json", help="Emit results as JSON"),
    allow_holdout: bool = typer.Option(False, "--allow-holdout", hidden=True),
) -> None:
    """Run a full harness eval set and record results on the leaderboard.

    Uses the harness pipeline (not baseline single-shot) to evaluate each case.
    Results are stored in the DB and appear on the leaderboard.
    """
    from mechanistic_agent.core import RunCoordinator
    from mechanistic_agent.scoring import score_snapshot_against_known, score_subagents_from_step_outputs

    if thinking_level is not None:
        thinking_level = thinking_level.strip().lower()
        if thinking_level not in {"low", "high"}:
            raise typer.BadParameter("thinking-level must be one of: low, high")

    if tier and tier not in {"easy", "medium", "hard"}:
        raise typer.BadParameter("tier must be one of: easy, medium, hard")

    base = Path.cwd()
    store = RunStore(base / "data" / "mechanistic.db")
    registry = RegistrySet(base)
    model_family = get_model_family(model_name) or "unknown"
    internal_reasoning = to_internal_reasoning_level(thinking_level)
    eval_set = store.get_eval_set(eval_set_id)
    if eval_set is None:
        raise typer.BadParameter(f"Eval set not found: {eval_set_id}")
    is_holdout = str(eval_set.get("purpose") or "general") == "leaderboard_holdout"
    if is_holdout and not allow_holdout:
        raise typer.BadParameter(
            "leaderboard_holdout eval sets are restricted to 'eval-runset-official'"
        )

    # Load eval tier data if needed
    tier_case_ids: Optional[List[str]] = None
    if tier:
        tier_file = base / "training_data" / "eval_tiers.json"
        if tier_file.exists():
            tier_data = json.loads(tier_file.read_text(encoding="utf-8"))
            tier_case_ids = tier_data.get(tier, [])
        else:
            typer.echo("Warning: eval_tiers.json not found, ignoring --tier")

    hashes = registry.bundle_hashes(model_name=model_name)
    eval_run_id = store.create_eval_run(
        eval_set_id=eval_set_id,
        run_group_name=run_group or f"cli_eval_{harness}",
        model=model_name,
        model_name=model_name,
        model_family=model_family,
        thinking_level=thinking_level,
        harness_bundle_hash=hashes.get("prompt_bundle_hash", ""),
        status="running",
    )

    all_cases = store.list_eval_set_cases(eval_set_id)
    if not all_cases:
        typer.echo(f"No cases found for eval set {eval_set_id}")
        store.set_eval_run_status(eval_run_id, "failed")
        raise typer.Exit(code=1)

    # Case selection
    if case_ids:
        id_set = set(case_ids)
        selected = [c for c in all_cases if str(c.get("case_id", "")) in id_set]
    elif tier_case_ids is not None:
        id_set = {str(cid) for cid in tier_case_ids}
        selected = [c for c in all_cases if str(c.get("case_id", "")) in id_set]
    else:
        selected = all_cases[:max_cases]

    if not selected:
        typer.echo("No cases matched the selection criteria")
        store.set_eval_run_status(eval_run_id, "failed")
        raise typer.Exit(code=1)

    typer.echo(f"Running {len(selected)} cases with model={model_name} harness={harness}")

    coordinator = RunCoordinator(store)
    completed = 0
    failed = 0
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
                    "temperature_celsius": float(input_payload.get("temperature_celsius", 25.0)),
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
                },
                **hashes,
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

            # Get cost from run snapshot
            cost_summary = snapshot.get("cost_summary") or {}
            run_cost = cost_summary.get("total_cost") or {}

            summary: Dict[str, Any] = {
                "score": score,
                "passed": passed,
                "step_count": len([s for s in step_outputs if s.get("step_name") == "mechanism_synthesis"]),
                "n_mechanistic_steps": case_step_count,
                "error": graded.get("error"),
                "eval_mode": "harness",
                "subagent_scores": subagent_scores,
            }
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
            completed += 1
            total_cost = run_cost.get("total_cost", 0.0)
            typer.echo(f"  [{completed + failed}] {case_id}: score={score:.3f} passed={passed} cost=${total_cost:.3f}")
        except Exception as exc:
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

    store.set_eval_run_status(eval_run_id, "completed")
    result_obj = {
        "eval_run_id": eval_run_id,
        "model": model_name,
        "thinking_level": thinking_level,
        "harness": harness,
        "completed": completed,
        "failed": failed,
    }
    if json_output:
        typer.echo(json.dumps(result_obj, indent=2))
    else:
        typer.echo(f"\nEval complete: {completed} completed, {failed} failed")
        typer.echo(f"Eval run ID: {eval_run_id}")
        typer.echo(f"View results: python main.py leaderboard --eval-set-id {eval_set_id}")


@app.command(name="eval-runset-official")
def eval_runset_official_cmd(
    eval_set_id: Optional[str] = typer.Option(
        None,
        "--eval-set-id",
        help="Optional holdout eval set id (defaults to latest purpose=leaderboard_holdout).",
    ),
    model_name: str = typer.Option(
        get_default_model(), "--model-name", "--model",
        help="Model identifier (e.g. gpt-5.4, claude-opus-4.6)",
    ),
    thinking_level: Optional[str] = typer.Option(
        None, "--thinking-level", "--reasoning", help="Thinking level: low or high"
    ),
    case_ids: Optional[List[str]] = typer.Option(None, "--case-id", help="Specific case IDs to run (repeatable)"),
    harness: str = typer.Option("default", "--harness", help="Harness name from harness_versions/"),
    run_group: Optional[str] = typer.Option(
        None, "--run-group", help="Run group name for leaderboard (default: official_holdout_harness)"
    ),
    max_cases: int = typer.Option(200, "--max-cases", help="Max cases to run"),
    max_steps: int = typer.Option(10, "--max-steps", help="Max mechanism steps per case"),
    max_runtime: float = typer.Option(300.0, "--max-runtime", help="Per-case timeout in seconds"),
    json_output: bool = typer.Option(False, "--json", help="Emit results as JSON"),
) -> None:
    """Run the official holdout-only leaderboard eval."""
    base = Path.cwd()
    store = RunStore(base / "data" / "mechanistic.db")
    resolved_eval_set_id = eval_set_id
    if resolved_eval_set_id:
        row = store.get_eval_set(resolved_eval_set_id)
        if row is None:
            raise typer.BadParameter(f"Eval set not found: {resolved_eval_set_id}")
        if str(row.get("purpose") or "general") != "leaderboard_holdout":
            raise typer.BadParameter("eval-set-id must point to purpose=leaderboard_holdout")
    else:
        holdouts = store.list_eval_sets(purpose="leaderboard_holdout")
        if not holdouts:
            raise typer.BadParameter("No purpose=leaderboard_holdout eval set found")
        resolved_eval_set_id = str(holdouts[0].get("id") or "")
        if not resolved_eval_set_id:
            raise typer.BadParameter("Unable to resolve holdout eval set id")

    eval_cmd(
        eval_set_id=resolved_eval_set_id,
        model_name=model_name,
        thinking_level=thinking_level,
        tier=None,
        case_ids=case_ids,
        harness=harness,
        run_group=run_group or "official_holdout_harness",
        max_cases=max_cases,
        max_steps=max_steps,
        max_runtime=max_runtime,
        json_output=json_output,
        allow_holdout=True,
    )


if __name__ == "__main__":
    app()
