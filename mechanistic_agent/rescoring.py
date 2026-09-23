"""Recompute stored eval results from stored traces under a scoring version.

Backs ``python main.py rescore-eval-results``. For every ``eval_run_results``
row it re-grades the stored run trace with
:func:`mechanistic_agent.scoring.score_snapshot_against_known` and
:func:`~mechanistic_agent.scoring.score_subagents_from_step_outputs` under the
requested ``scoring_version`` and, when writing, replaces ``score``,
``pass_bool`` and the summary fields that scoring owns. The prior values are
kept under ``summary.scoring_history.<old version>``.

Rows that cannot be re-graded keep their stored values:

* ``no_trace``: no ``run_id``, or the trace is gone. When the stored score is
  0 and the summary records an error (a case that crashed before a trace
  existed), the score is version-independent and the row is stamped with the
  new version (``status = "version_independent"``).
* ``no_expected``: no eval-set case and no resolvable curriculum case.

Leaderboard deltas are computed by running ``RunStore.leaderboard`` before and
after on the same database, so holdout weighting and ground-truth filtering are
exactly what the leaderboard applies.
"""
from __future__ import annotations

import sqlite3
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from mechanistic_agent.scoring import (
    LEGACY_SCORING_VERSION,
    normalize_scoring_version,
    score_snapshot_against_known,
    score_subagents_from_step_outputs,
)

ExpectedResolver = Callable[[Dict[str, Any], Dict[str, Any]], Optional[Dict[str, Any]]]


@dataclass
class ResultOutcome:
    result_id: str
    eval_run_id: str
    case_id: str
    status: str  # rescored | version_independent | no_trace | no_expected | error
    old_score: Optional[float]
    old_version: str
    v1_score: Optional[float] = None
    new_score: Optional[float] = None
    error: Optional[str] = None


@dataclass
class RescoreReport:
    scoring_version: str
    outcomes: List[ResultOutcome] = field(default_factory=list)
    written: bool = False

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for item in self.outcomes:
            out[item.status] = out.get(item.status, 0) + 1
        return dict(sorted(out.items()))

    def by_eval_run(self) -> Dict[str, List[ResultOutcome]]:
        grouped: Dict[str, List[ResultOutcome]] = {}
        for item in self.outcomes:
            grouped.setdefault(item.eval_run_id, []).append(item)
        return grouped


# ---------------------------------------------------------------------------
# Expected-answer resolution
# ---------------------------------------------------------------------------


def curriculum_expected_resolver() -> Optional[Callable[[str], Optional[Dict[str, Any]]]]:
    """Resolve ``flower_<id>`` curriculum cases from the FlowER lookup cache, read-only.

    Mirrors ``flower_curriculum.convert_mechanism_id_to_case`` but opens the
    lookup cache with SQLite ``mode=ro`` and never rebuilds it. Returns None
    when the cache or the FlowER ``train.txt`` it indexes is unavailable.
    """
    try:
        from mechanistic_agent import flower_curriculum as fc
    except Exception:  # pragma: no cover - optional data
        return None
    cache_path = Path(fc.DEFAULT_LOOKUP_CACHE)
    input_path = Path(fc.DEFAULT_FLOWER_INPUT)
    if not cache_path.is_file() or not input_path.is_file():
        return None
    # The cache is a WAL database; immutable=1 stops a read-only connection
    # from leaving -wal/-shm files next to it (it is a static index).
    uri = f"file:{cache_path.resolve()}?mode=ro&immutable=1"
    try:
        conn = sqlite3.connect(uri, uri=True)
        try:
            meta = dict(conn.execute("SELECT key, value FROM cache_meta").fetchall())
        finally:
            conn.close()
        stat = input_path.stat()
        if (
            meta.get("input_path") != str(input_path)
            or meta.get("input_size") != str(stat.st_size)
            or meta.get("input_mtime") != str(stat.st_mtime)
        ):
            return None
    except Exception:
        return None

    cache: Dict[str, Optional[Dict[str, Any]]] = {}

    def _load(mechanism_id: int) -> List[str]:
        conn = sqlite3.connect(uri, uri=True)
        try:
            rows = conn.execute(
                "SELECT offset, length FROM mechanism_rows WHERE mechanism_id = ? ORDER BY row_order ASC",
                (int(mechanism_id),),
            ).fetchall()
        finally:
            conn.close()
        reactions: List[str] = []
        with input_path.open("rb") as handle:
            for offset, length in rows:
                handle.seek(int(offset))
                parsed = fc._parse_line(handle.read(int(length)).decode("utf-8"))
                if parsed is not None:
                    reactions.append(parsed.mapped_reaction)
        return reactions

    def _resolve(case_id: str) -> Optional[Dict[str, Any]]:
        if case_id in cache:
            return cache[case_id]
        expected: Optional[Dict[str, Any]] = None
        text = str(case_id or "")
        if text.startswith("flower_") and text[len("flower_"):].isdigit():
            try:
                mechanism_id = int(text[len("flower_"):])
                reactions = _load(mechanism_id)
                if reactions:
                    case = fc._convert_group(mechanism_id, reactions)
                    expected = {
                        "products": list(case.get("products") or []),
                        "known_mechanism": fc.known_mechanism_from_case(case),
                        "verified_mechanism": case.get("verified_mechanism"),
                    }
            except Exception:
                expected = None
        cache[case_id] = expected
        return expected

    return _resolve


def default_expected_resolver(store: Any, *, use_curriculum: bool = True) -> ExpectedResolver:
    """Eval-set case first, then (optionally) the curriculum lookup."""
    cases_by_set: Dict[str, Dict[str, Dict[str, Any]]] = {}
    curriculum = curriculum_expected_resolver() if use_curriculum else None

    def _resolve(result: Dict[str, Any], run: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        set_id = str(run.get("eval_set_id") or "")
        if set_id not in cases_by_set:
            cases_by_set[set_id] = {str(c.get("case_id")): c for c in store.list_eval_set_cases(set_id)}
        case = cases_by_set[set_id].get(str(result.get("case_id")))
        expected = (case or {}).get("expected")
        if isinstance(expected, dict) and expected:
            return expected
        if curriculum is not None:
            summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
            case_id = str(summary.get("curriculum_case_id") or result.get("case_id") or "")
            return curriculum(case_id)
        return None

    return _resolve


# ---------------------------------------------------------------------------
# Rescoring
# ---------------------------------------------------------------------------


def _rescored_summary(
    summary: Dict[str, Any],
    *,
    old_score: Optional[float],
    old_passed: Optional[bool],
    old_version: str,
    graded: Dict[str, Any],
    subagent_scores: Dict[str, Any],
    scoring_version: str,
) -> Dict[str, Any]:
    out = dict(summary)
    history = dict(out.get("scoring_history") or {})
    if old_version != scoring_version and old_version not in history:
        history[old_version] = {"score": old_score, "passed": old_passed}
    out["scoring_history"] = history
    out["scoring_version"] = scoring_version
    out["rescored_at"] = time.time()
    out["score"] = graded.get("score")
    out["passed"] = graded.get("passed")
    out["subagent_scores"] = subagent_scores
    out["mapping_agreement"] = graded.get("mapping_agreement")
    if isinstance(out.get("scoring_breakdown"), dict) and out["scoring_breakdown"]:
        out["scoring_breakdown"] = graded
    return out


def _version_independent_without_trace(summary: Dict[str, Any], old_score: Optional[float]) -> bool:
    """True when a trace-less result provably scores the same under v1 and v2.

    * A case that failed before producing a trace is recorded with score 0.
    * A harness-free baseline has no step-mapping module: every accepted step
      (index >= 1) got the neutral 0.5 mapping component under v1, which is
      also what v2 assigns without a predicted mapping.
    """
    if (old_score or 0.0) == 0.0 and summary.get("error"):
        return True
    if str(summary.get("eval_mode") or "") != "baseline":
        return False
    breakdown = summary.get("scoring_breakdown") if isinstance(summary.get("scoring_breakdown"), dict) else {}
    steps = breakdown.get("step_breakdown")
    if not isinstance(steps, list):
        return False
    for step in steps:
        if not isinstance(step, dict):
            return False
        if int(step.get("step_index") or 0) <= 0 or float(step.get("mapping_confidence", -1)) != 0.5:
            return False
    return True


def rescore_eval_results(
    store: Any,
    *,
    scoring_version: str,
    eval_set_id: Optional[str] = None,
    eval_run_ids: Optional[List[str]] = None,
    write: bool = False,
    expected_resolver: Optional[ExpectedResolver] = None,
) -> RescoreReport:
    """Re-grade stored eval results; write them back only when ``write``."""
    version = normalize_scoring_version(scoring_version)
    resolver = expected_resolver or default_expected_resolver(store)
    report = RescoreReport(scoring_version=version, written=write)
    wanted = set(eval_run_ids or [])

    for run in store.list_eval_runs(eval_set_id):
        eval_run_id = str(run.get("id") or "")
        if wanted and eval_run_id not in wanted:
            continue
        for result in store.list_eval_run_results(eval_run_id):
            summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
            old_score = result.get("score") if isinstance(result.get("score"), (int, float)) else None
            old_passed = result.get("pass_bool")
            old_version = str(summary.get("scoring_version") or LEGACY_SCORING_VERSION)
            outcome = ResultOutcome(
                result_id=str(result.get("id")),
                eval_run_id=eval_run_id,
                case_id=str(result.get("case_id") or ""),
                status="no_trace",
                old_score=old_score,
                old_version=old_version,
            )
            report.outcomes.append(outcome)

            run_id = str(result.get("run_id") or "")
            snapshot = store.get_run_snapshot(run_id) if run_id else None
            if not snapshot:
                if _version_independent_without_trace(summary, old_score):
                    outcome.status = "version_independent"
                    outcome.v1_score = outcome.new_score = old_score
                    if write and old_version != version:
                        new_summary = dict(summary)
                        new_summary["scoring_version"] = version
                        store.update_eval_run_result(
                            outcome.result_id, score=old_score, passed=old_passed, summary=new_summary
                        )
                continue

            expected = resolver(result, run)
            if not expected:
                outcome.status = "no_expected"
                continue
            try:
                graded = score_snapshot_against_known(snapshot, expected, scoring_version=version)
                v1 = (
                    graded
                    if version == "v1"
                    else score_snapshot_against_known(snapshot, expected, scoring_version="v1")
                )
                subagents = score_subagents_from_step_outputs(
                    list(snapshot.get("step_outputs") or []),
                    scoring_version=version,
                    mapping_agreement=graded.get("mapping_agreement"),
                )
            except Exception as exc:
                outcome.status = "error"
                outcome.error = str(exc)
                continue
            outcome.status = "rescored"
            outcome.v1_score = float(v1.get("score") or 0.0)
            outcome.new_score = float(graded.get("score") or 0.0)
            if write:
                store.update_eval_run_result(
                    outcome.result_id,
                    score=outcome.new_score,
                    passed=bool(graded.get("passed")),
                    summary=_rescored_summary(
                        summary,
                        old_score=old_score,
                        old_passed=old_passed,
                        old_version=old_version,
                        graded=graded,
                        subagent_scores=subagents,
                        scoring_version=version,
                    ),
                )
    return report


# ---------------------------------------------------------------------------
# Leaderboard delta
# ---------------------------------------------------------------------------


def _row_score(row: Dict[str, Any]) -> float:
    if row.get("aggregate_weighting"):
        return float(row.get("weighted_quality_score") or 0.0)
    return float(row.get("mean_quality_score") or 0.0)


def leaderboard_snapshot(
    store: Any,
    *,
    eval_set_ids: Optional[List[str]] = None,
    row_filter: Optional[Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    ids = eval_set_ids or [str(item.get("id")) for item in store.list_eval_sets()]
    out: Dict[str, List[Dict[str, Any]]] = {}
    for set_id in ids:
        rows = store.leaderboard(eval_set_id=set_id, limit=100)
        if row_filter is not None:
            rows = row_filter(rows)
        if rows:
            out[set_id] = rows
    return out


def leaderboard_delta_rows(
    before: Dict[str, List[Dict[str, Any]]],
    after: Dict[str, List[Dict[str, Any]]],
    report: RescoreReport,
    *,
    eval_set_names: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    by_run = report.by_eval_run()
    names = eval_set_names or {}
    rows: List[Dict[str, Any]] = []
    for set_id in sorted(set(before) | set(after), key=lambda k: names.get(k, k)):
        before_rows = {str(r.get("eval_run_id")): (rank, r) for rank, r in enumerate(before.get(set_id, []), 1)}
        after_rows = {str(r.get("eval_run_id")): (rank, r) for rank, r in enumerate(after.get(set_id, []), 1)}
        for run_id in sorted(set(before_rows) | set(after_rows), key=lambda k: after_rows.get(k, before_rows.get(k))[0]):
            rank_before, row_before = before_rows.get(run_id, (None, {}))
            rank_after, row_after = after_rows.get(run_id, (None, {}))
            row = row_after or row_before
            outcomes = by_run.get(run_id, [])
            rescored = [o for o in outcomes if o.status == "rescored"]
            # Unweighted means over every case of the row; cases that were not
            # re-graded contribute their stored score to both sides.
            scored_outcomes = [o for o in outcomes if o.old_score is not None or o.status == "rescored"]
            v1_recomputed = v2_rescored = None
            if rescored and scored_outcomes:
                v1_recomputed = sum(
                    (o.v1_score if o.status == "rescored" else o.old_score) or 0.0 for o in scored_outcomes
                ) / len(scored_outcomes)
                v2_rescored = sum(
                    (o.new_score if o.status == "rescored" else o.old_score) or 0.0 for o in scored_outcomes
                ) / len(scored_outcomes)
            old = _row_score(row_before) if row_before else None
            new = _row_score(row_after) if row_after else None
            rows.append(
                {
                    "eval_set_id": set_id,
                    "eval_set_name": names.get(set_id, set_id),
                    "eval_run_id": run_id,
                    "model_name": row.get("model_name") or row.get("model"),
                    "thinking_level": row.get("thinking_level"),
                    "run_group_name": row.get("run_group_name"),
                    "case_count": row.get("case_count"),
                    "rank_before": rank_before,
                    "rank_after": rank_after,
                    "v1_score": old,
                    "v2_score": new,
                    "delta": (new - old) if (old is not None and new is not None) else None,
                    "v1_recomputed_mean": v1_recomputed,
                    "version_delta": (v2_rescored - v1_recomputed) if rescored else None,
                    "rescored_cases": len(rescored),
                    "kept_cases": len(outcomes) - len(rescored),
                    "scoring_version_after": row_after.get("scoring_version") if row_after else None,
                }
            )
    return rows


def _f(value: Optional[float], digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def format_delta_markdown(rows: List[Dict[str, Any]], report: RescoreReport) -> str:
    lines = [
        f"Scoring {LEGACY_SCORING_VERSION} (stored) -> {report.scoring_version}. "
        f"Result outcomes: {report.counts()}",
        "",
        "v1 score / v2 score = the leaderboard row score before / after (holdout rows use the "
        "leaderboard's step-weighted score). Delta = v2 - v1. v1 recomp = unweighted case mean with "
        "re-graded cases scored by today's v1 scorer (differs from the stored mean when the scorer or "
        "case data changed after the run). Delta_ver = unweighted v2 mean - v1 recomp: the pure "
        "v1->v2 effect with scorer drift removed.",
        "",
        "| Eval set | Rank v1->v2 | Model | Thinking | Run group | Cases (re-graded/kept) | v1 score | v1 recomp | v2 score | Delta | Delta_ver | Version after |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        delta = row.get("delta")
        delta_text = "-" if delta is None else f"{delta:+.4f}"
        ver = row.get("version_delta")
        ver_text = "-" if ver is None else f"{ver:+.4f}"
        lines.append(
            f"| {row['eval_set_name']} | {row['rank_before'] or '-'}->{row['rank_after'] or '-'} | "
            f"`{row['model_name']}` | {row['thinking_level'] or 'none'} | `{row['run_group_name']}` | "
            f"{row['case_count']} ({row['rescored_cases']}/{row['kept_cases']}) | {_f(row['v1_score'])} | "
            f"{_f(row['v1_recomputed_mean'])} | {_f(row['v2_score'])} | {delta_text} | {ver_text} | "
            f"{row['scoring_version_after'] or '-'} |"
        )
    return "\n".join(lines)


def _sqlite_copy(source: Path, dest: Path) -> None:
    """Consistent copy via the SQLite backup API (source opened read-only)."""
    src = sqlite3.connect(f"file:{Path(source).resolve()}?mode=ro", uri=True)
    try:
        dst = sqlite3.connect(str(dest))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()


@dataclass
class RescoreRun:
    report: RescoreReport
    delta_rows: List[Dict[str, Any]]
    db_path: Path
    applied: bool
    backup_path: Optional[Path] = None


def run_rescore(
    db_path: Path,
    *,
    scoring_version: str,
    apply: bool = False,
    eval_set_id: Optional[str] = None,
    eval_run_ids: Optional[List[str]] = None,
    backup: bool = True,
    row_filter: Optional[Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]] = None,
    work_dir: Optional[Path] = None,
    use_curriculum: bool = True,
) -> RescoreRun:
    """Rescore and diff the leaderboard.

    Dry run (default): copy the DB to a temporary directory, rescore the copy,
    diff leaderboards there, delete the copy. The original is never opened for
    writing. ``apply``: back up the DB next to itself (unless ``backup=False``)
    and rescore it in place.
    """
    from mechanistic_agent.core.db import RunStore

    source = Path(db_path)
    if not source.is_file():
        raise FileNotFoundError(str(source))
    backup_path: Optional[Path] = None
    temp_dir: Optional[tempfile.TemporaryDirectory] = None
    if apply:
        target = source
        if backup:
            backup_path = source.with_name(f"{source.name}.pre-rescore-{time.strftime('%Y%m%d-%H%M%S')}.bak")
            _sqlite_copy(source, backup_path)
    else:
        temp_dir = tempfile.TemporaryDirectory(dir=str(work_dir) if work_dir else None)
        target = Path(temp_dir.name) / source.name
        _sqlite_copy(source, target)
    try:
        store = RunStore(target)
        set_ids = [eval_set_id] if eval_set_id else None
        names = {
            str(item.get("id")): f"{item.get('name') or ''}@{item.get('version') or ''}"
            for item in store.list_eval_sets()
        }
        before = leaderboard_snapshot(store, eval_set_ids=set_ids, row_filter=row_filter)
        report = rescore_eval_results(
            store,
            scoring_version=scoring_version,
            eval_set_id=eval_set_id,
            eval_run_ids=eval_run_ids,
            write=True,
            expected_resolver=default_expected_resolver(store, use_curriculum=use_curriculum),
        )
        report.written = apply
        after = leaderboard_snapshot(store, eval_set_ids=set_ids, row_filter=row_filter)
        rows = leaderboard_delta_rows(before, after, report, eval_set_names=names)
        return RescoreRun(report=report, delta_rows=rows, db_path=source, applied=apply, backup_path=backup_path)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
