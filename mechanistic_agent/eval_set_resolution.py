"""Shared eval-set resolution helpers for CLI, API, and local scripts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from mechanistic_agent.core.db import RunStore


class EvalSetResolutionError(ValueError):
    """Raised when an eval set cannot be resolved or violates policy."""


def case_ids_hash(case_ids: Sequence[str]) -> str:
    payload = json.dumps([str(item) for item in case_ids], separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalise_json_eval_case(index: int, entry: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    # Native DB-export style: {case_id, input, expected, tags}
    case_id = str(entry.get("case_id") or "").strip()
    input_payload = entry.get("input")
    if case_id and isinstance(input_payload, Mapping):
        return {
            "case_id": case_id,
            "input": dict(input_payload),
            "expected": dict(entry.get("expected") or {}) if isinstance(entry.get("expected"), Mapping) else {},
            "tags": list(entry.get("tags") or []),
        }

    # training_data/eval_set.json style: {id, starting_materials, products, ...}
    training_case_id = str(entry.get("id") or "").strip()
    starting = entry.get("starting_materials")
    products = entry.get("products")
    if not training_case_id or not isinstance(starting, list) or not isinstance(products, list):
        return None

    expected: Dict[str, Any] = {"products": [str(item) for item in products]}
    known = entry.get("known_mechanism")
    verified = entry.get("verified_mechanism")
    if isinstance(known, Mapping):
        expected["known_mechanism"] = dict(known)
    if isinstance(verified, Mapping):
        expected["verified_mechanism"] = dict(verified)

    return {
        "case_id": training_case_id,
        "input": {
            "starting_materials": [str(item) for item in starting],
            "products": [str(item) for item in products],
            "temperature_celsius": entry.get("temperature_celsius", 25.0),
            "ph": entry.get("ph"),
        },
        "expected": expected,
        "tags": list(entry.get("tags") or []),
        "_source_index": index,
    }


def load_eval_cases_from_json(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise EvalSetResolutionError(f"Eval set JSON not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise EvalSetResolutionError("Eval set JSON must be a list")

    cases: List[Dict[str, Any]] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            continue
        item = _normalise_json_eval_case(index, entry)
        if item is not None:
            cases.append(item)

    if not cases:
        raise EvalSetResolutionError(f"No valid cases found in {path}")
    return cases


@dataclass(frozen=True)
class ResolvedEvalSet:
    eval_set_id: Optional[str]
    purpose: str
    source: str
    cases: List[Dict[str, Any]]
    case_ids: List[str]
    case_ids_hash: str
    name: Optional[str] = None
    version: Optional[str] = None


def resolve_db_eval_set(
    store: RunStore,
    *,
    requested_eval_set_id: Optional[str] = None,
    require_purpose: Optional[str] = None,
    default_purpose: Optional[str] = None,
) -> ResolvedEvalSet:
    row: Optional[Dict[str, Any]] = None
    if requested_eval_set_id:
        row = store.get_eval_set(requested_eval_set_id)
        if row is None:
            raise EvalSetResolutionError(f"Eval set not found: {requested_eval_set_id}")
    elif default_purpose:
        candidates = store.list_eval_sets(purpose=default_purpose)
        if not candidates:
            raise EvalSetResolutionError(f"No purpose={default_purpose} eval set found")
        row = dict(candidates[0])
    else:
        candidates = store.list_eval_sets(exposed_in_ui=True)
        if not candidates:
            raise EvalSetResolutionError("No eval sets found")
        row = dict(candidates[0])

    assert row is not None
    eval_set_id = str(row.get("id") or "")
    if not eval_set_id:
        raise EvalSetResolutionError("Resolved eval set is missing id")

    purpose = str(row.get("purpose") or "general")
    if require_purpose and purpose != require_purpose:
        raise EvalSetResolutionError(f"eval-set-id must point to purpose={require_purpose}")

    cases = store.list_eval_set_cases(eval_set_id)
    case_ids = [str(item.get("case_id") or "") for item in cases if str(item.get("case_id") or "")]
    return ResolvedEvalSet(
        eval_set_id=eval_set_id,
        purpose=purpose,
        source=f"db:{eval_set_id}",
        cases=cases,
        case_ids=case_ids,
        case_ids_hash=case_ids_hash(case_ids),
        name=str(row.get("name") or "") or None,
        version=str(row.get("version") or "") or None,
    )


def resolve_eval_set(
    *,
    store: Optional[RunStore],
    requested_eval_set_id: Optional[str] = None,
    eval_set_path: Optional[Path] = None,
    require_purpose: Optional[str] = None,
    default_purpose: Optional[str] = None,
) -> ResolvedEvalSet:
    if eval_set_path is not None:
        path = eval_set_path.expanduser().resolve()
        cases = load_eval_cases_from_json(path)
        case_ids = [str(item.get("case_id") or "") for item in cases if str(item.get("case_id") or "")]
        return ResolvedEvalSet(
            eval_set_id=None,
            purpose="json_path",
            source=f"path:{path}",
            cases=cases,
            case_ids=case_ids,
            case_ids_hash=case_ids_hash(case_ids),
            name=path.name,
            version=None,
        )

    if store is None:
        raise EvalSetResolutionError("store is required when eval_set_path is not provided")
    return resolve_db_eval_set(
        store,
        requested_eval_set_id=requested_eval_set_id,
        require_purpose=require_purpose,
        default_purpose=default_purpose,
    )


def select_eval_cases(
    *,
    cases: Sequence[Dict[str, Any]],
    case_ids: Optional[Sequence[str]] = None,
    tier_case_ids: Optional[Sequence[str]] = None,
    max_cases: Optional[int] = None,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]]
    by_id = {str(case.get("case_id") or ""): dict(case) for case in cases}
    if case_ids:
        selected = [
            by_id[str(case_id)]
            for case_id in case_ids
            if str(case_id) in by_id
        ]
    elif tier_case_ids:
        selected = [
            by_id[str(case_id)]
            for case_id in tier_case_ids
            if str(case_id) in by_id
        ]
    else:
        selected = [dict(case) for case in cases]

    if max_cases is not None and max_cases > 0 and len(selected) > max_cases:
        selected = selected[:max_cases]
    return selected
