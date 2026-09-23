"""Observatory replay projection (Observatory PRD §17, §22).

``build_observatory(events)`` reconstructs the mechanism search from the
persisted event log alone — accepted path, candidate sets with a status per
candidate, validation results with their ReactionFocus / bond-electron
payloads, branch points, backtracks, failed paths, provenance — so a page
reload, a historical replay and the live view all show the same thing.

State identity: ``s0`` is the initial state; every candidate's resulting state
is ``st:<candidate_id>``. A backtrack truncates the accepted path and marks the
truncated candidates (and their states) ``abandoned``; the alternative applied
after it is a normal accepted step with ``acceptance_kind =
backtrack_alternative``. Soft-advanced steps keep ``validation_passed = false``
and are never counted as validated.

Event index conventions (from ``RunCoordinator``): ``mechanism_step_accepted.step_index``
and ``mechanism_candidates_proposed.step_index`` are the **1-based step number**
(``_apply_candidate`` increments before emitting); ``branch_point_created.step_index``
and ``backtrack.reverted_to_step`` are the **0-based index** of the step being
decided, i.e. the number of accepted steps that remain valid.

Candidate status vocabulary (in order of precedence, latest evidence wins):

    proposed → incomplete | constraint_rejected | rejected | validated
             → accepted | branch_alternative → abandoned

Legacy runs recorded before candidate ids existed get synthetic ids
(``legacy-<step>``, ``soft-<step>``) so the projection never fails.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .provenance import build_run_provenance

OBSERVATORY_SCHEMA = "mechanism_observatory.v1"
INITIAL_STATE_ID = "s0"


def state_id_for(candidate_id: str) -> str:
    return f"st:{candidate_id}"


def _sorted(events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted((e for e in events if isinstance(e, dict)), key=lambda e: int(e.get("seq") or 0))


def _species(value: Any) -> List[str]:
    return [str(s) for s in (value or []) if s is not None]


def build_observatory(
    events: Iterable[Dict[str, Any]],
    *,
    run_id: Optional[str] = None,
    run_input: Optional[Dict[str, Any]] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    ordered = _sorted(events)
    run_input = run_input if isinstance(run_input, dict) else {}
    starting = _species(run_input.get("starting_materials"))
    products = _species(run_input.get("products"))

    states: Dict[str, Dict[str, Any]] = {
        INITIAL_STATE_ID: {"state_id": INITIAL_STATE_ID, "species": list(starting), "kind": "initial", "step_index": 0, "candidate_id": None}
    }
    candidate_sets: List[Dict[str, Any]] = []
    candidates: Dict[str, Dict[str, Any]] = {}  # candidate_id -> candidate entry (shared with candidate_sets)
    accepted_path: List[Dict[str, Any]] = []
    abandoned: List[str] = []
    branch_points: List[Dict[str, Any]] = []
    backtracks: List[Dict[str, Any]] = []
    failed_paths: List[Dict[str, Any]] = []
    latest_proposal_provenance: Dict[int, Dict[str, Any]] = {}  # attempt -> provenance
    open_steps: Dict[str, Dict[str, Any]] = {}
    current_state_id = INITIAL_STATE_ID
    completed = False
    last_seq = 0

    def _ensure_candidate(cid: str, *, step_index: int, rank: Any = None, smiles: Any = None, resulting: Any = None) -> Dict[str, Any]:
        entry = candidates.get(cid)
        if entry is None:
            entry = {
                "candidate_id": cid,
                "rank": rank,
                "intermediate_smiles": smiles,
                "reaction_smirks": None,
                "reaction_description": None,
                "resulting_state": _species(resulting),
                "state_id": state_id_for(cid),
                "status": "proposed",
                "failed_checks": [],
                "validation_attempts": 0,
                "reaction_focus": None,
                "bond_electron_view": None,
                "smirks_state_agreement": None,
                "step_index": step_index,
            }
            candidates[cid] = entry
            # a candidate seen outside a proposal event (legacy / soft) gets its own set
            candidate_sets.append({
                "step_index": step_index, "proposal_round": None, "candidate_set_id": f"implicit-{cid}",
                "current_state_id": current_state_id, "coordination_topology": None, "rejected_candidate_count": 0,
                "candidates": [entry],
            })
        states.setdefault(entry["state_id"], {
            "state_id": entry["state_id"], "species": list(entry["resulting_state"]), "kind": "candidate",
            "step_index": step_index, "candidate_id": cid,
        })
        return entry

    for event in ordered:
        kind = str(event.get("event_type") or "")
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        seq = int(event.get("seq") or 0)
        last_seq = max(last_seq, seq)
        step_name = str(event.get("step_name") or payload.get("step_name") or "")

        if kind == "step_started":
            if step_name:
                open_steps[step_name] = {
                    "step_name": step_name,
                    "planned_engine": payload.get("planned_engine"),
                    "planned_model": payload.get("planned_model"),
                }
        elif kind in {"step_output", "step_completed", "step_failed"}:
            open_steps.pop(step_name, None)
            if kind == "step_output" and step_name == "mechanism_step_proposal" and isinstance(payload.get("provenance"), dict):
                latest_proposal_provenance[int(payload.get("attempt") or 0)] = dict(payload["provenance"])
        elif kind == "mechanism_candidates_proposed":
            step_index = int(payload.get("step_index") or 0)
            entries: List[Dict[str, Any]] = []
            for raw in payload.get("candidates") or []:
                if not isinstance(raw, dict):
                    continue
                cid = str(raw.get("candidate_id") or f"legacy-{step_index}-r{raw.get('rank')}")
                entry = candidates.get(cid)
                if entry is None:
                    entry = {
                        "candidate_id": cid,
                        "rank": raw.get("rank"),
                        "intermediate_smiles": raw.get("intermediate_smiles"),
                        "reaction_smirks": raw.get("reaction_smirks"),
                        "reaction_description": raw.get("reaction_description"),
                        "resulting_state": _species(raw.get("resulting_state")),
                        "state_id": state_id_for(cid),
                        "status": "proposed",
                        "failed_checks": [],
                        "validation_attempts": 0,
                        "reaction_focus": None,
                        "bond_electron_view": None,
                        "smirks_state_agreement": None,
                        "step_index": step_index,
                    }
                    candidates[cid] = entry
                states.setdefault(entry["state_id"], {
                    "state_id": entry["state_id"], "species": list(entry["resulting_state"]), "kind": "candidate",
                    "step_index": step_index, "candidate_id": cid,
                })
                entries.append(entry)
            candidate_sets.append({
                "step_index": step_index,
                "proposal_round": payload.get("proposal_round"),
                "candidate_set_id": payload.get("candidate_set_id") or f"cs{step_index}-{seq}",
                "current_state_id": current_state_id,
                "coordination_topology": payload.get("coordination_topology"),
                "rejected_candidate_count": int(payload.get("rejected_candidate_count") or 0),
                "candidates": entries,
            })
        elif kind in {"mechanism_candidate_incomplete", "mechanism_candidate_constraint_rejected"}:
            cid = payload.get("candidate_id")
            if cid:
                entry = _ensure_candidate(str(cid), step_index=int(payload.get("attempt") or 0), rank=payload.get("candidate_rank"), smiles=payload.get("candidate_smiles"))
                entry["status"] = "incomplete" if kind.endswith("incomplete") else "constraint_rejected"
                states[entry["state_id"]]["kind"] = entry["status"]
        elif kind == "candidate_validation_result":
            cid = payload.get("candidate_id")
            if cid:
                entry = _ensure_candidate(str(cid), step_index=int(payload.get("step_index") or 0), rank=payload.get("candidate_rank"),
                                          smiles=payload.get("predicted_intermediate"), resulting=payload.get("resulting_state"))
                entry["validation_attempts"] += 1
                if payload.get("resulting_state"):
                    entry["resulting_state"] = _species(payload.get("resulting_state"))
                    states[entry["state_id"]]["species"] = list(entry["resulting_state"])
                if payload.get("reaction_smirks"):
                    entry["reaction_smirks"] = payload.get("reaction_smirks")
                entry["reaction_focus"] = payload.get("reaction_focus")
                entry["bond_electron_view"] = payload.get("bond_electron_view")
                entry["smirks_state_agreement"] = payload.get("smirks_state_agreement")
                if payload.get("accepted"):
                    entry["status"] = "validated"
                    entry["failed_checks"] = []
                    states[entry["state_id"]]["kind"] = "validated"
                elif entry["status"] not in {"validated", "accepted"}:
                    entry["status"] = "rejected"
                    entry["failed_checks"] = [str(c) for c in (payload.get("failed_checks") or [])]
                    states[entry["state_id"]]["kind"] = "rejected"
        elif kind == "branch_point_created":
            alt_ids = [str(a) for a in (payload.get("alternative_candidate_ids") or []) if a]
            branch_points.append({
                "seq": seq,
                "step_index": int(payload.get("step_index") or 0),
                "chosen_candidate_id": payload.get("chosen_candidate_id"),
                "alternative_candidate_ids": alt_ids,
            })
            for alt in alt_ids:
                entry = candidates.get(alt)
                if entry is not None and entry["status"] in {"proposed", "validated"}:
                    entry["status"] = "branch_alternative"
                    states[entry["state_id"]]["kind"] = "branch_alternative"
        elif kind == "mechanism_step_accepted":
            step_number = int(payload.get("step_index") or 0)  # 1-based (see module docstring)
            acceptance_kind = str(payload.get("acceptance_kind") or "validated")
            summary = payload.get("validation_summary") if isinstance(payload.get("validation_summary"), dict) else {}
            passed = bool(summary.get("passed")) and acceptance_kind != "soft_advance"
            cid = payload.get("candidate_id")
            if not cid:
                cid = f"soft-{step_number}" if acceptance_kind == "soft_advance" else f"legacy-{step_number}"
            cid = str(cid)
            entry = _ensure_candidate(cid, step_index=step_number, rank=payload.get("candidate_rank"),
                                      smiles=payload.get("predicted_intermediate"), resulting=payload.get("resulting_state"))
            entry["status"] = "accepted"
            entry["resulting_state"] = _species(payload.get("resulting_state")) or entry["resulting_state"]
            state = states[entry["state_id"]]
            state["species"] = list(entry["resulting_state"])
            state["kind"] = "soft_advance" if acceptance_kind == "soft_advance" else "accepted"
            state["step_index"] = step_number
            identity = payload.get("atom_identity") if isinstance(payload.get("atom_identity"), dict) else None
            if identity is not None:
                state["mapped_species"] = _species(identity.get("mapped_species"))
                state["atoms"] = [dict(a) for a in (identity.get("atoms") or []) if isinstance(a, dict)]
            # A re-acceptance at an earlier index (backtrack alternative) truncates the path.
            while accepted_path and accepted_path[-1]["step_index"] >= step_number:
                dropped = accepted_path.pop()
                _mark_abandoned(candidates, states, abandoned, dropped["candidate_id"])
            from_state_id = accepted_path[-1]["to_state_id"] if accepted_path else INITIAL_STATE_ID
            accepted_path.append({
                "seq": seq,
                "step_index": step_number,
                "from_state_id": from_state_id,
                "to_state_id": entry["state_id"],
                "candidate_id": cid,
                "candidate_rank": payload.get("candidate_rank"),
                "acceptance_kind": acceptance_kind,
                "predicted_intermediate": payload.get("predicted_intermediate"),
                "current_state": _species(payload.get("current_state")),
                "resulting_state": _species(payload.get("resulting_state")),
                "contains_target_product": bool(payload.get("contains_target_product")),
                "validation_passed": passed,
                "proposal_provenance": latest_proposal_provenance.get(step_number),
                "identity": (
                    {
                        "identity_source": identity.get("identity_source"),
                        "preserved_id_count": identity.get("preserved_id_count"),
                        "new_ids": list(identity.get("new_ids") or []),
                        "lost_ids": list(identity.get("lost_ids") or []),
                        "identity_resynced": identity.get("identity_resynced"),
                        "smirks_state_agreement": identity.get("smirks_state_agreement"),
                    }
                    if identity is not None
                    else None
                ),
            })
            current_state_id = entry["state_id"]
        elif kind == "failed_path_recorded":
            failed_paths.append({
                "seq": seq,
                "branch_step_index": int(payload.get("branch_step_index") or 0),
                "candidate_id": payload.get("candidate_id"),
                "candidate_rank": payload.get("candidate_rank"),
                "steps_in_path": int(payload.get("steps_in_path") or 0),
            })
        elif kind == "backtrack":
            reverted = int(payload.get("reverted_to_step") or 0)
            backtracks.append({
                "seq": seq,
                "reverted_to_step": reverted,
                "candidate_id": payload.get("candidate_id"),
                "alternative_rank": payload.get("alternative_rank"),
                "remaining_alternatives": payload.get("remaining_alternatives"),
            })
            while accepted_path and accepted_path[-1]["step_index"] > reverted:
                dropped = accepted_path.pop()
                _mark_abandoned(candidates, states, abandoned, dropped["candidate_id"])
            current_state_id = accepted_path[-1]["to_state_id"] if accepted_path else INITIAL_STATE_ID
        elif kind in {"run_completed", "target_products_detected"}:
            completed = True

    active_step: Optional[Dict[str, Any]] = None
    if open_steps:
        active_step = list(open_steps.values())[-1]

    return {
        "schema_version": OBSERVATORY_SCHEMA,
        "run_id": run_id,
        "status": status,
        "reaction": {"starting_materials": starting, "products": products},
        "states": states,
        "accepted_path": accepted_path,
        "candidate_sets": candidate_sets,
        "branch_points": branch_points,
        "backtracks": backtracks,
        "failed_paths": failed_paths,
        "abandoned_candidate_ids": abandoned,
        "unvalidated_step_count": sum(1 for s in accepted_path if not s["validation_passed"]),
        "completed": completed,
        "active_step": active_step,
        "atom_lineage": _build_atom_lineage(accepted_path, states),
        "provenance": build_run_provenance(ordered),
        "event_count": len(ordered),
        "last_seq": last_seq,
    }


ATOM_LINEAGE_SCHEMA = "atom_lineage.v1"


def _build_atom_lineage(accepted_path: List[Dict[str, Any]], states: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Per-atom lineage across the accepted path (PRD §11.4).

    Rows are persistent ids; columns are the accepted states in order. An atom
    is ``present`` in a state when its pid has a record there; ``new_at_step``
    / ``lost_at_step`` come from the identity counters of the step that
    introduced or dropped it (falling back to first/last presence).
    """
    steps_with_identity = [s for s in accepted_path if s.get("identity") is not None and "atoms" in states.get(s["to_state_id"], {})]
    if not steps_with_identity:
        return {"schema_version": ATOM_LINEAGE_SCHEMA, "state_ids": [], "atoms": [], "changed_pids": [], "identity_source": None}
    state_ids = [s["to_state_id"] for s in steps_with_identity]
    per_state: Dict[str, Dict[int, Dict[str, Any]]] = {}
    elements: Dict[int, str] = {}
    for sid in state_ids:
        table: Dict[int, Dict[str, Any]] = {}
        for atom in states[sid].get("atoms") or []:
            try:
                pid = int(atom.get("pid"))
            except (TypeError, ValueError):
                continue
            table[pid] = atom
            if atom.get("element") and pid not in elements:
                elements[pid] = str(atom["element"])
        per_state[sid] = table
    new_at: Dict[int, int] = {}
    lost_at: Dict[int, int] = {}
    for step in steps_with_identity:
        for pid in step["identity"].get("new_ids") or []:
            new_at.setdefault(int(pid), int(step["step_index"]))
        for pid in step["identity"].get("lost_ids") or []:
            lost_at.setdefault(int(pid), int(step["step_index"]))
    rows: List[Dict[str, Any]] = []
    for pid in sorted(elements):
        path: List[Dict[str, Any]] = []
        seen = False
        for step, sid in zip(steps_with_identity, state_ids):
            atom = per_state[sid].get(pid)
            present = atom is not None
            if present and not seen and step is not steps_with_identity[0]:
                new_at.setdefault(pid, int(step["step_index"]))
            if seen and not present:
                lost_at.setdefault(pid, int(step["step_index"]))
            seen = seen or present
            path.append({
                "state_id": sid,
                "step_index": int(step["step_index"]),
                "map_number": int(atom["map_number"]) if present and atom.get("map_number") is not None else None,
                "present": present,
            })
        rows.append({
            "pid": pid,
            "element": elements[pid],
            "path": path,
            "new_at_step": new_at.get(pid),
            "lost_at_step": lost_at.get(pid),
        })
    changed = sorted({pid for pid in elements if pid in new_at or pid in lost_at})
    return {
        "schema_version": ATOM_LINEAGE_SCHEMA,
        "state_ids": state_ids,
        "atoms": rows,
        "changed_pids": changed,
        "identity_source": steps_with_identity[-1]["identity"].get("identity_source"),
    }


def _mark_abandoned(candidates: Dict[str, Dict[str, Any]], states: Dict[str, Dict[str, Any]], abandoned: List[str], cid: str) -> None:
    if cid not in abandoned:
        abandoned.append(cid)
    entry = candidates.get(cid)
    if entry is not None:
        entry["status"] = "abandoned"
        states[entry["state_id"]]["kind"] = "abandoned"


__all__ = ["ATOM_LINEAGE_SCHEMA", "INITIAL_STATE_ID", "OBSERVATORY_SCHEMA", "build_observatory", "state_id_for"]
