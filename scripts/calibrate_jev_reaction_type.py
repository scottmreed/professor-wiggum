#!/usr/bin/env python3
"""Phase D calibration scaffold for the Jev reaction-type Choice (PRD §19).

Builds the labeled set (cases whose reaction type has a curated label in
``training_data/reaction_type_templates.json`` ``example_mappings``), then
either prints the Jev request payloads (default, dry run, no network) or,
with ``--live``, runs Jev in shadow mode on every case and reports accuracy by
confidence band, Brier and ECE. Nothing is written to the runtime DB.

Label ids are ``rxn_NNNN``; they correspond to HumanBenchmark rows
``hb350_NNN`` (e.g. ``rxn_0002`` = ``hb350_002``, Finkelstein). Cases are
resolved from benchmark JSON files (``--cases``; defaults: the legacy
HumanBenchmark eval set under the data root, plus the repo eval set) and from
DB runs whose ``example_id`` matches (``--db-path``). A DB run supplies its
recorded pre-loop outputs as Jev state and its LLM ``reaction_type_mapping``
output as a baseline; a file case gets the deterministic pre-loop analyses
(balance, functional groups, pH) computed locally.

Examples::

    # dry run: print the request payloads for the first 2 labeled cases
    PYTHONPATH=. python scripts/calibrate_jev_reaction_type.py --limit 2

    # live shadow run (real Jev calls; needs OPENROUTER_API_KEY)
    PYTHONPATH=. python scripts/calibrate_jev_reaction_type.py --live --limit 0 \\
        --output local_contributions/runs/jev_reaction_type_calibration.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mechanistic_agent.core.reaction_type_jev import (  # noqa: E402
    NO_MATCH,
    QUESTION_ID,
    build_reaction_type_question,
    build_reaction_type_state,
)
from mechanistic_agent.core.reaction_type_templates import load_reaction_type_catalog  # noqa: E402
from mechanistic_agent.decisions.calibration import summarize  # noqa: E402

_CONTEXT_STEPS = (
    "balance_analysis",
    "functional_groups",
    "ph_recommendation",
    "initial_conditions",
    "missing_reagents",
    "atom_mapping",
)


# ---------------------------------------------------------------------------
# Labeled set
# ---------------------------------------------------------------------------
def reaction_id_aliases(reaction_id: str) -> List[str]:
    """``rxn_NNNN`` <-> ``hb350_NNN`` aliases (the label id first)."""
    rid = str(reaction_id or "").strip()
    aliases = [rid]
    match = re.fullmatch(r"rxn_(\d+)", rid)
    if match:
        aliases.append(f"hb350_{int(match.group(1)):03d}")
    match = re.fullmatch(r"hb350_(\d+)", rid)
    if match:
        aliases.append(f"rxn_{int(match.group(1)):04d}")
    return aliases


def load_labels(catalog: Dict[str, Any], *, min_label_confidence: float = 0.0) -> Dict[str, Dict[str, Any]]:
    """Curated reaction-type labels keyed by reaction id."""
    by_id = dict(catalog.get("by_id") or {})
    labels: Dict[str, Dict[str, Any]] = {}
    for row in catalog.get("example_mappings") or []:
        rid = str(row.get("reaction_id") or "").strip()
        type_id = str(row.get("mechanism_type_id") or row.get("selected_type_id") or "").strip()
        if not rid or type_id not in by_id:
            continue
        confidence = row.get("confidence")
        confidence = float(confidence) if isinstance(confidence, (int, float)) else None
        if confidence is not None and confidence < min_label_confidence:
            continue
        labels[rid] = {
            "type_id": type_id,
            "label_exact": str(by_id[type_id].get("label_exact") or ""),
            "label_confidence": confidence,
        }
    return labels


def default_case_files() -> List[Path]:
    from mechanistic_agent.data_paths import bulk_training_dir, repo_training_dir

    candidates = [
        bulk_training_dir() / "local_legacy" / "humanbenchmark" / "eval_set.json",
        repo_training_dir() / "local_legacy" / "humanbenchmark" / "eval_set.json",
        repo_training_dir() / "eval_set.json",
    ]
    seen: List[Path] = []
    for path in candidates:
        if path.exists() and path.resolve() not in [p.resolve() for p in seen]:
            seen.append(path)
    return seen


def load_case_files(paths: Iterable[Path]) -> Dict[str, Dict[str, Any]]:
    cases: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        items = payload if isinstance(payload, list) else (payload.get("items") or payload.get("cases") or [])
        for item in items:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("id") or "").strip()
            if cid and cid not in cases and item.get("starting_materials") and item.get("products"):
                cases[cid] = {
                    "starting_materials": list(item["starting_materials"]),
                    "products": list(item["products"]),
                    "source": str(path),
                }
    return cases


def load_db_cases(db_path: Optional[Path], wanted_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """Earliest run per example_id with its recorded pre-loop outputs (read-only)."""
    wanted = set(wanted_ids)
    if db_path is None or not Path(db_path).exists() or not wanted:
        return {}
    conn = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT id, input_payload_json FROM runs ORDER BY created_at ASC"
        ).fetchall()
        cases: Dict[str, Dict[str, Any]] = {}
        for run_id, payload_json in rows:
            try:
                payload = json.loads(payload_json or "{}")
            except ValueError:
                continue
            example_id = str(payload.get("example_id") or "").strip()
            if example_id not in wanted or example_id in cases:
                continue
            context: Dict[str, Any] = {}
            llm_selection: Optional[Dict[str, Any]] = None
            for step_name, output_json in conn.execute(
                "SELECT step_name, output_json FROM step_outputs WHERE run_id = ? ORDER BY rowid ASC",
                (run_id,),
            ):
                try:
                    output = json.loads(output_json or "{}")
                except ValueError:
                    continue
                if step_name in _CONTEXT_STEPS:
                    context[step_name] = output
                elif step_name == "reaction_type_mapping" and str(output.get("model_used") or "") != "deterministic_example_mapping":
                    llm_selection = output
            cases[example_id] = {
                "starting_materials": list(payload.get("starting_materials") or []),
                "products": list(payload.get("products") or []),
                "context": context,
                "llm_selection": llm_selection,
                "run_id": run_id,
                "source": f"db:{run_id}",
            }
        return cases
    finally:
        conn.close()


def deterministic_context(starting_materials: List[str], products: List[str]) -> Dict[str, Any]:
    """Balance, functional groups and pH heuristics (no model calls)."""
    from mechanistic_agent.core.tool_executor import ToolExecutor

    executor = ToolExecutor()
    context: Dict[str, Any] = {}
    for key, call in (
        ("balance_analysis", lambda: executor.run_balance(starting_materials, products)),
        ("functional_groups", lambda: executor.run_functional_groups(starting_materials + products)),
        ("ph_recommendation", lambda: executor.run_ph_recommendation(starting_materials, products, None)),
    ):
        try:
            context[key] = call()
        except Exception as exc:  # RDKit parse errors on odd SMILES
            context[key] = {"error": type(exc).__name__}
    return context


def build_labeled_set(
    catalog: Dict[str, Any],
    *,
    case_files: Sequence[Path],
    db_path: Optional[Path],
    min_label_confidence: float = 0.0,
    compute_context: bool = True,
) -> List[Dict[str, Any]]:
    labels = load_labels(catalog, min_label_confidence=min_label_confidence)
    alias_to_label = {alias: rid for rid in labels for alias in reaction_id_aliases(rid)}
    file_cases = load_case_files(case_files)
    db_cases = load_db_cases(db_path, alias_to_label.keys())
    labeled: List[Dict[str, Any]] = []
    for rid, label in sorted(labels.items()):
        case = None
        for alias in reaction_id_aliases(rid):
            case = db_cases.get(alias) or file_cases.get(alias)
            if case is not None:
                break
        if case is None:
            continue
        context = dict(case.get("context") or {})
        if compute_context and not context:
            context = deterministic_context(case["starting_materials"], case["products"])
        labeled.append(
            {
                "case_id": rid,
                "label_type_id": label["type_id"],
                "label_exact": label["label_exact"],
                "label_confidence": label["label_confidence"],
                "starting_materials": case["starting_materials"],
                "products": case["products"],
                "context": context,
                "llm_selection": case.get("llm_selection"),
                "source": case.get("source"),
            }
        )
    return labeled


# ---------------------------------------------------------------------------
# Requests and evaluation
# ---------------------------------------------------------------------------
def state_for_case(case: Dict[str, Any]) -> Dict[str, Any]:
    from mechanistic_agent.smiles_utils import strip_atom_mapping_list

    context = case.get("context") or {}
    return build_reaction_type_state(
        starting_materials=strip_atom_mapping_list(case["starting_materials"]),
        products=strip_atom_mapping_list(case["products"]),
        **{key: context.get(key) for key in _CONTEXT_STEPS},
    )


def evaluate_records(cases: List[Dict[str, Any]], records: List[Any]) -> Dict[str, Any]:
    """Metrics for Jev (and the LLM baseline where the DB has one)."""
    conf: List[float] = []
    correct: List[bool] = []
    dists: List[Dict[str, float]] = []
    labels: List[str] = []
    failures: Dict[str, int] = {}
    rows: List[Dict[str, Any]] = []
    for case, record in zip(cases, records):
        row = {"case_id": case["case_id"], "label": case["label_type_id"]}
        if not record.ok:
            failures[record.failure] = failures.get(record.failure, 0) + 1
            row["failure"] = record.failure
            rows.append(row)
            continue
        selected = str(record.selected)
        p_sel = float(record.probabilities.get(selected, 0.0))
        conf.append(p_sel)
        correct.append(selected == case["label_type_id"])
        dists.append(dict(record.probabilities))
        labels.append(case["label_type_id"])
        row.update({
            "selected": selected,
            "p_selected": p_sel,
            "p_label": float(record.probabilities.get(case["label_type_id"], 0.0)),
            "p_no_match": float(record.probabilities.get(NO_MATCH, 0.0)),
            "jev_confidence": record.confidence,
            "latency_ms": record.latency_ms,
            "cost": (record.cost or {}).get("total_cost"),
        })
        rows.append(row)

    llm_conf: List[float] = []
    llm_correct: List[bool] = []
    for case in cases:
        sel = case.get("llm_selection")
        if not isinstance(sel, dict) or not isinstance(sel.get("confidence"), (int, float)):
            continue
        llm_conf.append(max(0.0, min(1.0, float(sel["confidence"]))))
        llm_correct.append(str(sel.get("selected_type_id") or "") == case["label_type_id"])

    return {
        "jev": summarize(conf, correct, distributions=dists, labels=labels) if conf else {"n": 0},
        "llm_baseline": summarize(llm_conf, llm_correct) if llm_conf else {"n": 0},
        "failures": failures,
        "cases": rows,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    from mechanistic_agent.data_paths import db_path as default_db_path

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", action="append", type=Path, default=None,
                        help="benchmark JSON file(s) with id/starting_materials/products (repeatable)")
    parser.add_argument("--db-path", type=Path, default=None, help="runtime DB (read-only); default: data root DB")
    parser.add_argument("--no-db", action="store_true", help="do not read the runtime DB")
    parser.add_argument("--limit", type=int, default=2,
                        help="cases to print (dry run) or run (live); 0 = all")
    parser.add_argument("--min-label-confidence", type=float, default=0.0)
    parser.add_argument("--live", action="store_true",
                        help="make real Jev calls (needs OPENROUTER_API_KEY); default is a dry run")
    parser.add_argument("--model", default=None, help="catalog decision-model id (default: catalog default)")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=None, help="write the JSON report here (live mode)")
    parser.add_argument("--summary-only", action="store_true", help="dry run: print the labeled-set summary only")
    args = parser.parse_args(argv)
    try:  # RDKit parse warnings on legacy SMILES are noise here
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except Exception:
        pass

    catalog = load_reaction_type_catalog(REPO_ROOT)
    case_files = args.cases if args.cases else default_case_files()
    db = None if args.no_db else (args.db_path or default_db_path())
    labeled = build_labeled_set(
        catalog,
        case_files=case_files,
        db_path=db,
        min_label_confidence=args.min_label_confidence,
    )
    selected = labeled if args.limit <= 0 else labeled[: args.limit]
    question = build_reaction_type_question(catalog)
    print(
        f"labeled set: {len(labeled)} cases with curated labels "
        f"(of {len(load_labels(catalog, min_label_confidence=args.min_label_confidence))} labels); "
        f"case files: {[str(p) for p in case_files]}; db: {db if db and Path(db).exists() else 'none'}",
        file=sys.stderr,
    )

    if not args.live:
        if args.summary_only:
            return 0
        from mechanistic_agent.decisions.jev import JevDecisionClient

        client = JevDecisionClient(args.model, api_key=None)
        for case in selected:
            payload = client.build_payload(state_for_case(case), [question])
            print(json.dumps({"case_id": case["case_id"], "label": case["label_type_id"],
                              "url": client.url, "request": payload}, indent=1))
        print(f"dry run: printed {len(selected)} request payload(s); no network calls made. "
              "Pass --live to call Jev.", file=sys.stderr)
        return 0

    from mechanistic_agent.llm import get_decision_model

    if not os.getenv("OPENROUTER_API_KEY"):
        print("--live needs OPENROUTER_API_KEY in the environment.", file=sys.stderr)
        return 2
    client = get_decision_model(args.model, timeout=args.timeout)
    records = []
    for index, case in enumerate(selected, start=1):
        record = client.decide_many(state_for_case(case), [question])[QUESTION_ID]
        records.append(record)
        status = record.failure or f"{record.selected} p={record.probabilities.get(str(record.selected), 0):.3f}"
        print(f"[{index}/{len(selected)}] {case['case_id']} label={case['label_type_id']} -> {status}",
              file=sys.stderr)
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": client.model,
        "api_model_id": client.api_model_id,
        "model_versions": sorted({r.model_version for r in records if r.model_version}),
        "question_id": QUESTION_ID,
        "option_count": len(question.options),
        "case_count": len(selected),
        **evaluate_records(selected, records),
    }
    text = json.dumps(report, indent=1, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    summary = {k: report[k] for k in ("model_versions", "case_count", "jev", "llm_baseline", "failures")}
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
