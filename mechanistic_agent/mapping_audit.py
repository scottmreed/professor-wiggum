"""Read-only audit of benchmark mapping agreement over stored eval traces.

Backs ``scripts/backfill_mapping_agreement.py``. Opens the runtime SQLite DB
with ``mode=ro`` (never writes), joins every ``eval_run_results`` row that has a
stored run trace to its eval-set case, and computes the Part-1 mapping metric
(:mod:`mechanistic_agent.core.mapping_metrics`) for the global ``atom_mapping``
output and each accepted step's ``step_atom_mapping`` output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EvalRunMappingSummary:
    eval_run_id: str
    eval_set_id: str
    eval_set_name: str
    run_group_name: str
    model_name: str
    results: int = 0
    with_trace: int = 0
    with_benchmark: int = 0
    global_scored: int = 0
    global_agreement_sum: float = 0.0
    global_exact: int = 0
    steps_scored: int = 0
    step_agreement_sum: float = 0.0
    step_status_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def global_mean(self) -> Optional[float]:
        return (self.global_agreement_sum / self.global_scored) if self.global_scored else None

    @property
    def step_mean(self) -> Optional[float]:
        return (self.step_agreement_sum / self.steps_scored) if self.steps_scored else None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "eval_run_id": self.eval_run_id,
            "eval_set_id": self.eval_set_id,
            "eval_set_name": self.eval_set_name,
            "run_group_name": self.run_group_name,
            "model_name": self.model_name,
            "results": self.results,
            "with_trace": self.with_trace,
            "with_benchmark": self.with_benchmark,
            "global_scored": self.global_scored,
            "global_mean_agreement": None if self.global_mean is None else round(self.global_mean, 4),
            "global_exact_matches": self.global_exact,
            "steps_scored": self.steps_scored,
            "step_mean_agreement": None if self.step_mean is None else round(self.step_mean, 4),
            "step_status_counts": dict(sorted(self.step_status_counts.items())),
        }


@dataclass
class MappingAuditReport:
    available: bool
    db_path: Optional[Path]
    message: str = ""
    rows: List[EvalRunMappingSummary] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "db_path": str(self.db_path) if self.db_path else None,
            "message": self.message,
            "rows": [row.as_dict() for row in self.rows],
        }


def resolve_db_path(db_path: Optional[Path] = None) -> Path:
    if db_path is not None:
        return Path(db_path).expanduser()
    from mechanistic_agent.data_paths import db_path as default_db_path

    return default_db_path()


def open_readonly_store(db_path: Optional[Path] = None):
    """Return ``(store, path, message)``; ``store`` is None when unavailable."""

    path = resolve_db_path(db_path)
    if not path.is_file():
        return None, path, (
            f"Runtime DB not found at {path}. Set MECHANISTIC_DATA_DIR (or place the sibling "
            "../wiggum-data checkout) or pass --db-path; nothing to backfill."
        )
    from mechanistic_agent.core.db import ReadOnlyRunStore

    try:
        return ReadOnlyRunStore(path), path, ""
    except Exception as exc:  # pragma: no cover - defensive
        return None, path, f"Could not open {path} read-only: {exc}"


def audit_mapping_agreement(
    db_path: Optional[Path] = None,
    *,
    eval_set_id: Optional[str] = None,
    eval_run_ids: Optional[List[str]] = None,
    include_hydrogens: bool = False,
) -> MappingAuditReport:
    from mechanistic_agent.core.mapping_metrics import compute_run_mapping_agreement
    from mechanistic_agent.scoring import extract_accepted_path

    store, path, message = open_readonly_store(db_path)
    if store is None:
        return MappingAuditReport(available=False, db_path=path, message=message)

    eval_sets = {}
    cases_by_set: Dict[str, Dict[str, Dict[str, Any]]] = {}
    rows: List[EvalRunMappingSummary] = []
    wanted = set(eval_run_ids or [])
    for run in store.list_eval_runs(eval_set_id):
        run_id = str(run.get("id") or "")
        if wanted and run_id not in wanted:
            continue
        set_id = str(run.get("eval_set_id") or "")
        if set_id not in eval_sets:
            eval_sets[set_id] = store.get_eval_set(set_id) or {}
            cases_by_set[set_id] = {str(c.get("case_id")): c for c in store.list_eval_set_cases(set_id)}
        summary = EvalRunMappingSummary(
            eval_run_id=run_id,
            eval_set_id=set_id,
            eval_set_name=str(eval_sets[set_id].get("name") or ""),
            run_group_name=str(run.get("run_group_name") or ""),
            model_name=str(run.get("model_name") or run.get("model") or ""),
        )
        for result in store.list_eval_run_results(run_id):
            summary.results += 1
            trace_id = result.get("run_id")
            case = cases_by_set[set_id].get(str(result.get("case_id")))
            if not trace_id:
                continue
            snapshot = store.get_run_snapshot(str(trace_id))
            if not snapshot:
                continue
            summary.with_trace += 1
            expected = (case or {}).get("expected")
            metric = compute_run_mapping_agreement(
                snapshot,
                expected if isinstance(expected, dict) else None,
                accepted_steps=extract_accepted_path(snapshot),
                include_hydrogens=include_hydrogens,
            )
            if not metric.get("available"):
                continue
            summary.with_benchmark += 1
            glob = metric.get("global") or {}
            if glob.get("status") == "scored":
                summary.global_scored += 1
                summary.global_agreement_sum += float(glob.get("agreement") or 0.0)
                summary.global_exact += int(bool(glob.get("exact_match")))
            for step in metric.get("steps") or []:
                status = str(step.get("status") or "unknown")
                summary.step_status_counts[status] = summary.step_status_counts.get(status, 0) + 1
                if status == "scored":
                    summary.steps_scored += 1
                    summary.step_agreement_sum += float(step.get("agreement") or 0.0)
        rows.append(summary)
    return MappingAuditReport(available=True, db_path=path, rows=rows)


def _fmt(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.3f}"


def format_audit_table(report: MappingAuditReport, *, include_empty: bool = False) -> str:
    if not report.available:
        return report.message
    header = (
        "| eval_run | eval_set | run_group | model | results | traced | benchmark | "
        "global n | global mean | global exact | steps n | step mean |"
    )
    lines = [header, "|" + "---|" * 12]
    total = EvalRunMappingSummary("TOTAL", "", "", "", "")
    shown = 0
    for row in report.rows:
        for attr in ("results", "with_trace", "with_benchmark", "global_scored", "global_exact", "steps_scored"):
            setattr(total, attr, getattr(total, attr) + getattr(row, attr))
        total.global_agreement_sum += row.global_agreement_sum
        total.step_agreement_sum += row.step_agreement_sum
        if not include_empty and not row.with_benchmark:
            continue
        shown += 1
        lines.append(
            f"| {row.eval_run_id[:8]} | {row.eval_set_name} | {row.run_group_name[:40]} | {row.model_name} | "
            f"{row.results} | {row.with_trace} | {row.with_benchmark} | {row.global_scored} | "
            f"{_fmt(row.global_mean)} | {row.global_exact} | {row.steps_scored} | {_fmt(row.step_mean)} |"
        )
    lines.append(
        f"| **total** | | | | {total.results} | {total.with_trace} | {total.with_benchmark} | "
        f"{total.global_scored} | {_fmt(total.global_mean)} | {total.global_exact} | "
        f"{total.steps_scored} | {_fmt(total.step_mean)} |"
    )
    hidden = len(report.rows) - shown
    if hidden:
        lines.append(f"\n{hidden} eval run(s) with no benchmark-mapped traced results omitted (use --all to show).")
    return "\n".join(lines)
