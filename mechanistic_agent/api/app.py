"""FastAPI application for the local-first mechanistic runtime."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

try:  # pragma: no cover - optional dependency
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors, rdMolDescriptors
    from rdkit.Chem.Draw import rdMolDraw2D
except ImportError:  # pragma: no cover - fallback when RDKit absent
    Chem = None  # type: ignore[assignment]
    Draw = None  # type: ignore[assignment]
    Descriptors = None  # type: ignore[assignment]
    rdMolDescriptors = None  # type: ignore[assignment]
    rdMolDraw2D = None  # type: ignore[assignment]

from mechanistic_agent.core import (
    HarnessConfig,
    RegistrySet,
    RunCoordinator,
    RunManager,
    RunStore,
    select_step_models,
)
from mechanistic_agent.core.job_executor import ThreadJobExecutor
from mechanistic_agent.core.storage_interfaces import (
    LocalArtifactStore,
    RunStateStore,
    SQLiteRunStore,
)
from mechanistic_agent.core.arrow_push import predict_arrow_push_annotation
from mechanistic_agent.curriculum import (
    OPUS_MODEL,
    build_curriculum_status,
    curriculum_history,
    publish_curriculum_release,
    publish_due_curriculum_releases,
    render_curriculum_readme,
    submit_curriculum_release,
)
from mechanistic_agent.model_registry import (
    get_all_families,
    get_default_model,
    get_model_family,
    get_model_options,
    to_internal_reasoning_level,
    to_public_reasoning_level,
)
from mechanistic_agent.core.model_selection import preview_step_models
from mechanistic_agent.prompt_assets import (
    append_call_few_shot_example,
    get_call_prompt_version,
    list_call_prompt_versions,
    normalize_call_name,
    replace_prompt_in_skill_md,
    resolve_call_name_from_step,
    traces_root,
    unified_prompt_diff,
)
from mechanistic_agent.prompt_trace_validator import validate_evidence_for_calls
from mechanistic_agent.smiles_utils import strip_atom_mapping_list, strip_atom_mapping_optional
from mechanistic_agent.tools import classify_functional_group_transformation, predict_mechanistic_step
from mechanistic_agent.core.validators import validate_mechanism_step_output

from .schemas import (
    ApproveTraceRequest,
    BaselineEvalRunSetRequest,
    ConvertInputsRequest,
    CurriculumPublishRequest,
    CurriculumSubmitRequest,
    CurationExportRequest,
    CreateRunRequest,
    CreateRunResponse,
    EvalRunSetRequest,
    EvaluateRunRequest,
    FeedbackRequest,
    HarnessApplyRequest,
    HarnessPRRequest,
    ImportTemplateRequest,
    MechanismStepSubmitRequest,
    MemoryItemRequest,
    MemoryQueryRequest,
    OfficialEvalRunSetRequest,
    ResumeRunRequest,
    SaveEvaluationRequest,
    SeedSimulatedLeaderboardRequest,
    StartVerificationRequest,
    TraceEvidenceExportRequest,
    TraceToFewShotRequest,
    VerifyStepRequest,
    RalphVoteRequest,
)


TERMINAL_STATUSES = {"completed", "failed", "stopped"}
WORKFLOW_ORDER = [
    "balance_analysis",
    "functional_groups",
    "ph_recommendation",
    "initial_conditions",
    "missing_reagents",
    "atom_mapping",
    "reaction_type_mapping",
    "mechanism_step_proposal",
    "mechanism_synthesis",
    "bond_electron_validation",
    "atom_balance_validation",
    "state_progress_validation",
    "retry_gate",
    "backtrack_gate",
    "completion_check",
    "reflection",
    "step_atom_mapping",
    "run_failed",
    "run_complete",
]

FLOW_NODE_SPECS: List[Dict[str, Any]] = [
    {"id": "balance_analysis", "label": "Check Atom Balance", "kind": "deterministic", "step_name": "balance_analysis"},
    {"id": "functional_groups", "label": "Identify Functional Groups", "kind": "deterministic", "step_name": "functional_groups"},
    {"id": "ph_recommendation", "label": "Recommend pH", "kind": "deterministic", "step_name": "ph_recommendation"},
    {"id": "initial_conditions", "label": "Assess Reaction Conditions", "kind": "llm", "step_name": "initial_conditions"},
    {"id": "missing_reagents", "label": "Predict Missing Reagents", "kind": "llm", "step_name": "missing_reagents"},
    {"id": "atom_mapping", "label": "Map Atoms Between Reactants/Products", "kind": "llm", "step_name": "atom_mapping"},
    {"id": "reaction_type_mapping", "label": "Map To Reaction Type", "kind": "llm", "step_name": "reaction_type_mapping"},
    {"id": "mechanism_step_proposal", "label": "Propose Mechanism Step", "kind": "llm", "step_name": "mechanism_step_proposal"},
    {"id": "mechanism_synthesis", "label": "Validate Mechanism Step", "kind": "deterministic", "step_name": "mechanism_synthesis"},
    {"id": "bond_electron_validation", "label": "Check Bond/Electron Conservation", "kind": "deterministic", "step_name": "bond_electron_validation"},
    {"id": "atom_balance_validation", "label": "Check Step Atom Balance", "kind": "deterministic", "step_name": "atom_balance_validation"},
    {"id": "state_progress_validation", "label": "Check State Changed", "kind": "deterministic", "step_name": "state_progress_validation"},
    {"id": "retry_gate", "label": "Retry or Continue?", "kind": "decision", "step_name": "retry_gate"},
    {"id": "backtrack_gate", "label": "Backtrack?", "kind": "decision", "step_name": "backtrack_gate"},
    {"id": "completion_check", "label": "Target Products Reached?", "kind": "decision", "step_name": "mechanism_synthesis"},
    {"id": "reflection", "label": "Collect Validation Warnings", "kind": "deterministic", "step_name": "reflection"},
    {"id": "step_atom_mapping", "label": "Map Atoms for Step", "kind": "llm", "step_name": "step_atom_mapping"},
    {"id": "run_failed", "label": "Run Failed", "kind": "decision", "step_name": "run_failed"},
    {"id": "run_complete", "label": "Run Complete", "kind": "deterministic", "step_name": "run_complete"},
]

FLOW_EDGES: List[Dict[str, Any]] = [
    {"source": "balance_analysis", "target": "functional_groups"},
    {"source": "functional_groups", "target": "ph_recommendation"},
    {"source": "ph_recommendation", "target": "initial_conditions"},
    {"source": "initial_conditions", "target": "missing_reagents"},
    {"source": "missing_reagents", "target": "atom_mapping"},
    {"source": "atom_mapping", "target": "reaction_type_mapping"},
    {"source": "reaction_type_mapping", "target": "mechanism_step_proposal"},
    {"source": "mechanism_step_proposal", "target": "mechanism_synthesis"},
    {"source": "mechanism_synthesis", "target": "bond_electron_validation"},
    {"source": "bond_electron_validation", "target": "atom_balance_validation"},
    {"source": "atom_balance_validation", "target": "state_progress_validation"},
    {"source": "state_progress_validation", "target": "retry_gate"},
    {"source": "retry_gate", "target": "completion_check", "label": "valid"},
    {"source": "retry_gate", "target": "mechanism_synthesis", "is_cycle": True, "label": "retry"},
    {"source": "retry_gate", "target": "mechanism_step_proposal", "is_cycle": True, "label": "repropose"},
    {"source": "retry_gate", "target": "backtrack_gate", "label": "all_failed"},
    {"source": "backtrack_gate", "target": "mechanism_step_proposal", "is_cycle": True, "label": "revert_to_branch"},
    {"source": "backtrack_gate", "target": "run_failed", "label": "no_branches"},
    {"source": "completion_check", "target": "run_complete", "label": "yes"},
    {"source": "completion_check", "target": "reflection", "label": "no"},
    {"source": "reflection", "target": "step_atom_mapping"},
    {"source": "step_atom_mapping", "target": "mechanism_step_proposal", "is_cycle": True, "label": "loop"},
]


# Fixed backbone nodes that exist regardless of harness config.
FIXED_BACKBONE_NODES: List[Dict[str, Any]] = [
    {
        "id": "mechanism_step_proposal",
        "label": "Propose Mechanism Step",
        "kind": "llm",
        "step_name": "mechanism_step_proposal",
        "description": (
            "Topology-aware: dispatches to SAS, centralized MAS, independent MAS, "
            "or decentralized MAS based on coordination_topology."
        ),
    },
    {"id": "mechanism_synthesis", "label": "Validate Mechanism Step", "kind": "deterministic", "step_name": "mechanism_synthesis"},
    {"id": "retry_gate", "label": "Retry or Continue?", "kind": "decision", "step_name": "retry_gate"},
    {"id": "backtrack_gate", "label": "Backtrack?", "kind": "decision", "step_name": "backtrack_gate"},
    {"id": "completion_check", "label": "Target Products Reached?", "kind": "decision", "step_name": "mechanism_synthesis"},
    {"id": "run_failed", "label": "Run Failed", "kind": "decision", "step_name": "run_failed"},
    {"id": "run_complete", "label": "Run Complete", "kind": "deterministic", "step_name": "run_complete"},
]

FIXED_BACKBONE_EDGES: List[Dict[str, Any]] = [
    {"source": "retry_gate", "target": "completion_check", "label": "valid"},
    {"source": "retry_gate", "target": "mechanism_synthesis", "is_cycle": True, "label": "retry"},
    {"source": "retry_gate", "target": "mechanism_step_proposal", "is_cycle": True, "label": "repropose"},
    {"source": "retry_gate", "target": "backtrack_gate", "label": "all_failed"},
    {"source": "backtrack_gate", "target": "mechanism_step_proposal", "is_cycle": True, "label": "revert_to_branch"},
    {"source": "backtrack_gate", "target": "run_failed", "label": "no_branches"},
    {"source": "completion_check", "target": "run_complete", "label": "yes"},
]

# IDs of backbone nodes for dedup when building from harness
_BACKBONE_IDS = {n["id"] for n in FIXED_BACKBONE_NODES}


def _post_step_module_node(m: Any) -> Dict[str, Any]:
    return {
        "id": m.id,
        "label": m.label,
        "kind": m.kind,
        "step_name": m.step_name or m.id,
        "description": m.description,
        "io_schema": m.io_schema,
        "phase": "post_step",
    }


def build_flow_node_specs(harness: HarnessConfig) -> List[Dict[str, Any]]:
    """Generate flow node specs from a harness config + fixed backbone.

    Node order determines Mermaid top-down layout:
      pre-loop → proposal → synthesis → validators → decision gates →
      non-validators (reflection, step mapping) → terminals
    """
    nodes: List[Dict[str, Any]] = []

    for m in harness.enabled_pre_loop():
        nodes.append({
            "id": m.id,
            "label": m.label,
            "kind": m.kind,
            "step_name": m.step_name or m.id,
            "description": m.description,
            "io_schema": m.io_schema,
            "phase": "pre_loop",
        })

    # Backbone: mechanism_step_proposal, mechanism_synthesis
    nodes.append(FIXED_BACKBONE_NODES[0])  # mechanism_step_proposal
    nodes.append(FIXED_BACKBONE_NODES[1])  # mechanism_synthesis

    all_post = [m for m in harness.enabled_post_step() if m.id not in _BACKBONE_IDS]
    validators = [m for m in all_post if m.group_key == "validators"]
    non_validators = [m for m in all_post if m.group_key != "validators"]

    for m in validators:
        nodes.append(_post_step_module_node(m))

    # Decision gates: retry_gate, backtrack_gate, completion_check
    for n in FIXED_BACKBONE_NODES[2:5]:
        nodes.append(n)

    for m in non_validators:
        nodes.append(_post_step_module_node(m))

    # Terminal nodes: run_failed, run_complete
    for n in FIXED_BACKBONE_NODES[5:]:
        nodes.append(n)

    return nodes


def build_flow_edges(harness: HarnessConfig) -> List[Dict[str, Any]]:
    """Generate flow edges from a harness config + fixed backbone.

    The topology of the graph mirrors the coordinator's actual execution:

    1. Pre-loop modules chain sequentially → mechanism_step_proposal.
    2. mechanism_step_proposal → mechanism_synthesis → validators → retry_gate.
       Validators run during ``_try_candidate_with_retries`` *before* a
       candidate is accepted, so they sit between synthesis and the retry gate.
    3. retry_gate fans out: valid → completion_check, retry → synthesis,
       repropose → proposal, all_failed → backtrack_gate.
    4. completion_check: yes → run_complete, no → non-validator post-step
       modules (reflection, step_atom_mapping) → mechanism_step_proposal (loop).
    5. backtrack_gate: revert_to_branch → proposal, no_branches → run_failed.

    Coordination topology (SAS / centralized / independent / decentralized MAS)
    only affects Step A (proposal) — the rest of the graph is invariant.
    """
    edges: List[Dict[str, Any]] = []

    # --- pre-loop chain ---
    pre_loop = harness.enabled_pre_loop()
    for i in range(len(pre_loop) - 1):
        edges.append({"source": pre_loop[i].id, "target": pre_loop[i + 1].id})
    if pre_loop:
        edges.append({"source": pre_loop[-1].id, "target": "mechanism_step_proposal"})

    # --- proposal → synthesis ---
    edges.append({"source": "mechanism_step_proposal", "target": "mechanism_synthesis"})

    # --- split post-step into validators and non-validators ---
    all_post = [m for m in harness.enabled_post_step() if m.id not in _BACKBONE_IDS]
    validators = [m for m in all_post if m.group_key == "validators"]
    non_validators = [m for m in all_post if m.group_key != "validators"]

    # validators: synthesis → validators → retry_gate
    if validators:
        edges.append({"source": "mechanism_synthesis", "target": validators[0].id})
        for i in range(len(validators) - 1):
            edges.append({"source": validators[i].id, "target": validators[i + 1].id})
        edges.append({"source": validators[-1].id, "target": "retry_gate"})
    else:
        edges.append({"source": "mechanism_synthesis", "target": "retry_gate"})

    # non-validators: completion_check → (no) → non-validators → proposal (loop)
    if non_validators:
        edges.append({"source": "completion_check", "target": non_validators[0].id, "label": "no"})
        for i in range(len(non_validators) - 1):
            edges.append({"source": non_validators[i].id, "target": non_validators[i + 1].id})
        edges.append({
            "source": non_validators[-1].id,
            "target": "mechanism_step_proposal",
            "is_cycle": True,
            "label": "loop",
        })
    else:
        edges.append({
            "source": "completion_check",
            "target": "mechanism_step_proposal",
            "is_cycle": True,
            "label": "no",
        })

    # --- fixed backbone edges (retry_gate, backtrack_gate, terminals) ---
    edges.extend(FIXED_BACKBONE_EDGES)

    return edges


def build_workflow_order(harness: HarnessConfig) -> List[str]:
    """Generate ordered step names for progress tracking.

    Mirrors the node ordering in ``build_flow_node_specs``:
      pre-loop → proposal → synthesis → validators →
      decision gates → non-validators → terminals
    """
    order: List[str] = []
    for m in harness.enabled_pre_loop():
        order.append(m.step_name or m.id)
    order.append("mechanism_step_proposal")
    order.append("mechanism_synthesis")

    all_post = [m for m in harness.enabled_post_step() if m.id not in _BACKBONE_IDS]
    validators = [m for m in all_post if m.group_key == "validators"]
    non_validators = [m for m in all_post if m.group_key != "validators"]

    for m in validators:
        order.append(m.step_name or m.id)
    order.extend(["retry_gate", "backtrack_gate", "completion_check"])
    for m in non_validators:
        order.append(m.step_name or m.id)
    order.extend(["run_failed", "run_complete"])
    return order


def _match_tags(item_tags: List[str], required_tags: List[str]) -> bool:
    if not required_tags:
        return True
    item_set = {tag.lower() for tag in item_tags}
    return all(tag.lower() in item_set for tag in required_tags)


def _compute_progress(
    snapshot: Dict[str, Any],
    harness: Optional[HarnessConfig] = None,
) -> Dict[str, Any]:
    step_outputs = list(snapshot.get("step_outputs") or [])

    if harness is not None:
        enabled_steps = build_workflow_order(harness)
    else:
        enabled_steps = list(WORKFLOW_ORDER)
        cfg = snapshot.get("config") or {}
        if not cfg.get("functional_groups_enabled"):
            enabled_steps = [step for step in enabled_steps if step != "functional_groups"]
        if not cfg.get("intermediate_prediction_enabled"):
            enabled_steps = [step for step in enabled_steps if step != "mechanism_step_proposal"]
        optional_tools = set(cfg.get("optional_llm_tools") or [])
        if "predict_missing_reagents" not in optional_tools:
            enabled_steps = [step for step in enabled_steps if step != "missing_reagents"]
        if "attempt_atom_mapping" not in optional_tools:
            enabled_steps = [step for step in enabled_steps if step != "atom_mapping"]

    by_step: Dict[str, List[Dict[str, Any]]] = {}
    for row in step_outputs:
        name = str(row.get("step_name") or "")
        by_step.setdefault(name, []).append(row)

    active_steps = _active_steps_from_events(snapshot)
    retry_state = _latest_retry_state(snapshot)
    active_step_name = _latest_active_step(snapshot)

    steps: List[Dict[str, Any]] = []
    completed = 0
    failed = 0
    for step_name in enabled_steps:
        entries = by_step.get(step_name, [])
        if not entries:
            status = "pending"
            last_attempt = None
            validation = None
            model_used = (cfg.get("step_models") or {}).get(step_name)
        else:
            latest = sorted(entries, key=lambda item: int(item.get("attempt") or 0))[-1]
            validation = latest.get("validation")
            model_used = latest.get("model") or (cfg.get("step_models") or {}).get(step_name)
            if isinstance(validation, dict) and validation.get("passed") is False:
                status = "failed"
                failed += 1
            else:
                status = "completed"
                completed += 1
            last_attempt = latest.get("attempt")
        if step_name in active_steps:
            status = "active"
        if (
            step_name == "reaction_type_mapping"
            and str(cfg.get("reaction_template_policy") or "auto") == "off"
            and not entries
        ):
            status = "completed"
        if step_name == "retry_gate":
            if retry_state.get("exhausted"):
                status = "failed"
            elif retry_state.get("active"):
                status = "retrying"
            elif retry_state.get("last_failed"):
                status = "completed"
        if step_name == "run_failed":
            run_status = str(snapshot.get("status") or "")
            if run_status == "failed":
                status = "failed"
            elif run_status == "paused":
                status = "paused"
        if step_name == "completion_check":
            if snapshot.get("status") == "completed":
                status = "completed"
        steps.append(
            {
                "name": step_name,
                "status": status,
                "attempt": last_attempt,
                "validation": validation,
                "model": model_used,
            }
        )

    total = len(steps)
    percent = int((completed / total) * 100) if total else 0
    return {
        "steps": steps,
        "completed_count": completed,
        "failed_count": failed,
        "total_count": total,
        "progress_percentage": percent,
        "active_step_name": active_step_name,
        "retry_state": retry_state,
    }


def _enabled_steps_from_config(
    config: Dict[str, Any],
    harness: Optional[HarnessConfig] = None,
) -> List[str]:
    if harness is not None:
        return build_workflow_order(harness)
    enabled_steps = list(WORKFLOW_ORDER)
    if not config.get("functional_groups_enabled"):
        enabled_steps = [step for step in enabled_steps if step != "functional_groups"]
    if not config.get("intermediate_prediction_enabled"):
        enabled_steps = [step for step in enabled_steps if step != "mechanism_step_proposal"]
    optional_tools = set(config.get("optional_llm_tools") or [])
    if "predict_missing_reagents" not in optional_tools:
        enabled_steps = [step for step in enabled_steps if step != "missing_reagents"]
    if "attempt_atom_mapping" not in optional_tools:
        enabled_steps = [step for step in enabled_steps if step != "atom_mapping"]
    return enabled_steps


def _active_steps_from_events(snapshot: Dict[str, Any]) -> set[str]:
    active: set[str] = set()
    events = sorted(snapshot.get("events") or [], key=lambda item: int(item.get("seq") or 0))
    for event in events:
        step_name = str(event.get("step_name") or "")
        if not step_name:
            continue
        event_type = str(event.get("event_type") or "")
        if event_type == "step_started":
            active.add(step_name)
        elif event_type in {"step_completed", "step_failed", "step_output"} and step_name in active:
            active.remove(step_name)
        elif event_type == "mechanism_retry_started":
            active.add("retry_gate")
        elif event_type in {"mechanism_retry_failed", "mechanism_retry_exhausted"} and "retry_gate" in active:
            active.remove("retry_gate")
        elif event_type == "run_paused":
            active.add("run_failed")
        elif event_type == "run_resumed" and "run_failed" in active:
            active.remove("run_failed")
        elif event_type == "run_failed":
            active.add("run_failed")
    return active


def _latest_active_step(snapshot: Dict[str, Any]) -> str | None:
    active_set = _active_steps_from_events(snapshot)
    if not active_set:
        return None
    events = sorted(snapshot.get("events") or [], key=lambda item: int(item.get("seq") or 0), reverse=True)
    for event in events:
        event_type = str(event.get("event_type") or "")
        if event_type == "step_started":
            step_name = str(event.get("step_name") or "")
            if step_name and step_name in active_set:
                return step_name
        if event_type == "mechanism_retry_started" and "retry_gate" in active_set:
            return "retry_gate"
    return None


def _latest_retry_state(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "active": False,
        "last_failed": False,
        "exhausted": False,
        "attempt": None,
        "retry_index": None,
    }
    events = sorted(snapshot.get("events") or [], key=lambda item: int(item.get("seq") or 0))
    for event in events:
        event_type = str(event.get("event_type") or "")
        payload = event.get("payload") or {}
        if event_type == "mechanism_retry_started":
            state["active"] = True
            state["last_failed"] = False
            state["attempt"] = payload.get("attempt")
            state["retry_index"] = payload.get("retry_index")
        elif event_type == "mechanism_retry_failed":
            state["active"] = False
            state["last_failed"] = True
            state["attempt"] = payload.get("attempt")
            state["retry_index"] = payload.get("retry_index")
        elif event_type == "mechanism_retry_exhausted":
            state["active"] = False
            state["last_failed"] = True
            state["exhausted"] = True
            state["attempt"] = payload.get("attempt")
            state["retry_index"] = payload.get("retry_index")
    return state


def _node_status_history(snapshot: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    history: Dict[str, Dict[str, int]] = {}
    events = sorted(snapshot.get("events") or [], key=lambda item: int(item.get("seq") or 0))
    for event in events:
        step_name = str(event.get("step_name") or "")
        event_type = str(event.get("event_type") or "")
        if not step_name:
            continue
        row = history.setdefault(step_name, {"started": 0, "completed": 0, "failed": 0})
        if event_type == "step_started":
            row["started"] += 1
        elif event_type == "step_completed":
            row["completed"] += 1
        elif event_type == "step_failed":
            row["failed"] += 1
    return history


def _build_flow_state(
    snapshot: Dict[str, Any],
    *,
    prompt_step_map: Dict[str, Dict[str, Any]],
    harness: Optional[HarnessConfig] = None,
) -> Dict[str, Any]:
    config = snapshot.get("config") or {}

    if harness is not None:
        flow_node_specs = build_flow_node_specs(harness)
        flow_edges = build_flow_edges(harness)
    else:
        flow_node_specs = FLOW_NODE_SPECS
        flow_edges = FLOW_EDGES

    enabled_steps = set(_enabled_steps_from_config(config, harness))
    progress = _compute_progress(snapshot, harness)
    progress_by_step = {str(item.get("name") or ""): item for item in progress.get("steps", [])}
    active_steps = _active_steps_from_events(snapshot)

    nodes: List[Dict[str, Any]] = []
    for node in flow_node_specs:
        step_name = str(node.get("step_name") or "")
        node_id = str(node.get("id") or "")
        backbone_ids = {
            "completion_check", "reflection", "step_atom_mapping",
            "retry_gate", "backtrack_gate", "run_failed", "run_complete",
        }
        if step_name in backbone_ids:
            pass
        elif step_name not in enabled_steps:
            continue

        state = "pending"
        if node_id in active_steps or step_name in active_steps:
            state = "active"
        elif node_id == "completion_check":
            mech_status = progress_by_step.get("mechanism_synthesis", {}).get("status")
            run_status = str(snapshot.get("status") or "")
            if run_status == "completed":
                state = "completed"
            else:
                state = "completed" if mech_status in {"completed", "failed"} else "pending"
        elif node_id == "retry_gate":
            retry_state = progress.get("retry_state", {})
            if retry_state.get("exhausted"):
                state = "failed"
            elif retry_state.get("active"):
                state = "retrying"
            elif retry_state.get("last_failed"):
                state = "completed"
        elif node_id == "backtrack_gate":
            events = snapshot.get("events") or []
            has_backtrack = any(
                (ev.get("event_type") or "") == "backtrack" for ev in events
            )
            has_failed_path = any(
                (ev.get("event_type") or "") == "failed_path_recorded" for ev in events
            )
            if has_backtrack:
                state = "completed"
            elif has_failed_path:
                state = "failed"
        elif node_id == "run_failed":
            run_status = str(snapshot.get("status") or "")
            if run_status == "failed":
                state = "failed"
            elif run_status == "paused":
                state = "paused"
        elif node_id == "run_complete":
            run_status = str(snapshot.get("status") or "")
            state = "completed" if run_status == "completed" else "pending"
        else:
            step_progress = progress_by_step.get(step_name)
            if isinstance(step_progress, dict):
                status = str(step_progress.get("status") or "pending")
                if status in {"completed", "failed", "retrying", "paused"}:
                    state = status

        prompt_ref = prompt_step_map.get(step_name)
        nodes.append(
            {
                **node,
                "state": state,
                "prompt_ref": prompt_ref,
            }
        )

    node_ids = {str(node["id"]) for node in nodes}
    edges = [
        edge
        for edge in flow_edges
        if str(edge.get("source")) in node_ids and str(edge.get("target")) in node_ids
    ]
    return {
        "nodes": nodes,
        "edges": edges,
        "legend": {
            "kinds": {
                "llm": "LLM-driven step",
                "deterministic": "deterministic/tool check",
                "decision": "decision/check node",
            },
            "states": {
                "pending": "not started",
                "active": "in progress",
                "retrying": "retry in progress",
                "paused": "paused for user decision",
                "completed": "completed",
                "failed": "failed",
            },
        },
        "active_node_id": progress.get("active_step_name"),
        "retry_state": progress.get("retry_state", {}),
        "node_status_history": _node_status_history(snapshot),
    }


def _extract_failed_paths(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract failed path records from events for UI display."""
    events = snapshot.get("events") or []
    paths: List[Dict[str, Any]] = []
    for ev in events:
        if (ev.get("event_type") or "") == "failed_path_recorded":
            payload = ev.get("payload") or {}
            paths.append({
                "branch_step_index": payload.get("branch_step_index"),
                "candidate_rank": payload.get("candidate_rank"),
                "steps_in_path": payload.get("steps_in_path", 0),
                "failure_reason": payload.get("failure_reason", "validation_retry_exhausted"),
            })
    return paths


def _build_mechanism_summary(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for row in snapshot.get("step_outputs", []):
        if row.get("step_name") != "mechanism_synthesis":
            continue
        output = row.get("output") or {}
        if not isinstance(output, dict):
            continue
        summary.append(
            {
                "attempt": row.get("attempt"),
                "step_index": output.get("step_index") or row.get("attempt"),
                "current_state": output.get("current_state", []),
                "predicted_intermediate": output.get("predicted_intermediate"),
                "resulting_state": output.get("resulting_state", []),
                "contains_target_product": output.get("contains_target_product"),
                "validation": row.get("validation"),
            }
        )
    summary.sort(key=lambda item: int(item.get("attempt") or 0))
    return summary


def _latest_step_mapping_summary(snapshot: Dict[str, Any]) -> Dict[str, Any] | None:
    rows = [
        row for row in (snapshot.get("step_outputs") or [])
        if row.get("step_name") == "step_atom_mapping"
    ]
    if not rows:
        return None
    rows.sort(key=lambda row: (int(row.get("attempt") or 0), int(row.get("retry_index") or 0)))
    latest = rows[-1]
    output = latest.get("output") or {}
    if not isinstance(output, dict):
        return None
    mapped = list(output.get("compact_mapped_atoms") or [])
    return {
        "attempt": int(latest.get("attempt") or 0),
        "mapped_atom_count": len(mapped),
        "confidence": output.get("confidence"),
        "unmapped_atoms": list(output.get("unmapped_atoms") or [])[:12],
    }


def _latest_reaction_type_selection(snapshot: Dict[str, Any]) -> Dict[str, Any] | None:
    rows = [
        row for row in (snapshot.get("step_outputs") or [])
        if row.get("step_name") == "reaction_type_mapping"
    ]
    if not rows:
        return None
    rows.sort(key=lambda row: (int(row.get("attempt") or 0), int(row.get("retry_index") or 0)))
    latest = rows[-1]
    output = latest.get("output") or {}
    if not isinstance(output, dict):
        return None
    return {
        "attempt": int(latest.get("attempt") or 0),
        "selected_label_exact": output.get("selected_label_exact"),
        "selected_type_id": output.get("selected_type_id"),
        "confidence": output.get("confidence"),
        "rationale": output.get("rationale"),
        "top_candidates": list(output.get("top_candidates") or []),
    }


def _latest_template_guidance_state(snapshot: Dict[str, Any]) -> Dict[str, Any] | None:
    events = sorted(snapshot.get("events") or [], key=lambda item: int(item.get("seq") or 0))
    latest_payload = None
    for event in events:
        if str(event.get("event_type") or "") != "template_guidance_state_updated":
            continue
        payload = event.get("payload") or {}
        if isinstance(payload, dict):
            latest_payload = dict(payload)
    return latest_payload


def _parse_smirks_text(raw: str) -> Dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("SMIRKS text is required")
    text = text.split("|", 1)[0].strip()
    if ">>" not in text:
        raise ValueError("SMIRKS must include '>>' separating reactants and products")
    left, right = text.split(">>", 1)
    reactants = [part.strip() for part in left.split(".") if part.strip()]
    products = [part.strip() for part in right.split(".") if part.strip()]
    if not reactants or not products:
        raise ValueError("SMIRKS must include at least one reactant and one product")

    # Validate that all molecules are valid SMILES
    if Chem is not None:
        for smiles in reactants + products:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES in SMIRKS: {smiles}")

    mapping_detected = ":" in text
    return {
        "smirks": text,
        "reactants": reactants,
        "products": products,
        "mapping_detected": mapping_detected,
    }


_MOLECULE_IMAGE_CACHE: Dict[tuple[str, bool], str | None] = {}


def _render_molecule(smiles: str, show_atom_numbers: bool = False) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "smiles": smiles,
        "formula": None,
        "molecular_weight": None,
        "image_data": None,
    }
    import re as _re
    if not smiles or Chem is None or Draw is None or rdMolDescriptors is None or Descriptors is None:
        return entry
    # Reject natural-language tokens early to avoid noisy RDKit parse errors.
    _core = smiles.strip().strip('.,;:!?')
    if not _core or " " in _core:
        return entry
    if _re.fullmatch(r'[A-Za-z-]+', _core) and _re.search(r'[a-z]{2,}', _core):
        return entry

    cache_key = (smiles, show_atom_numbers)
    cached = _MOLECULE_IMAGE_CACHE.get(cache_key)
    if cached is not None:
        entry["image_data"] = cached
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                entry["formula"] = rdMolDescriptors.CalcMolFormula(mol)
                entry["molecular_weight"] = round(float(Descriptors.MolWt(mol)), 4)
        except Exception:
            pass
        return entry

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            _MOLECULE_IMAGE_CACHE[cache_key] = None
            return entry
        entry["formula"] = rdMolDescriptors.CalcMolFormula(mol)
        entry["molecular_weight"] = round(float(Descriptors.MolWt(mol)), 4)

        if show_atom_numbers and rdMolDraw2D is not None:
            drawer = rdMolDraw2D.MolDraw2DCairo(220, 160)
            options = drawer.drawOptions()
            options.addAtomIndices = True
            rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
            drawer.FinishDrawing()
            encoded = base64.b64encode(drawer.GetDrawingText()).decode("ascii")
        else:
            image = Draw.MolToImage(mol, size=(220, 160))
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        entry["image_data"] = encoded
        _MOLECULE_IMAGE_CACHE[cache_key] = encoded
        return entry
    except Exception:
        _MOLECULE_IMAGE_CACHE[cache_key] = None
        return entry


def _build_reaction_visuals(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    payload = snapshot.get("input_payload", {})
    starting = [str(item) for item in payload.get("starting_materials", [])]
    products = [str(item) for item in payload.get("products", [])]
    return {
        "starting_materials": [_render_molecule(smiles) for smiles in starting],
        "products": [_render_molecule(smiles) for smiles in products],
    }


def _apply_summary_images(summary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for item in summary:
        current = [str(s) for s in item.get("current_state", [])]
        resulting = [str(s) for s in item.get("resulting_state", [])]
        intermediate = item.get("predicted_intermediate")
        item["current_state_cards"] = [_render_molecule(smiles) for smiles in current]
        item["resulting_state_cards"] = [_render_molecule(smiles) for smiles in resulting]
        if isinstance(intermediate, str) and intermediate.strip():
            item["intermediate_card"] = _render_molecule(intermediate)
        else:
            item["intermediate_card"] = None
    return summary


def _build_evaluation_progress(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    steps = []
    for step in snapshot.get("progress", {}).get("steps", []):
        if not isinstance(step, dict):
            continue
        steps.append(
            {
                "name": step.get("name"),
                "status": step.get("status"),
                "description": None,
                "tool_output": None,
                "logs": [],
            }
        )
    return {
        "steps": steps,
        "mechanism_summary": snapshot.get("mechanism_summary", []),
    }


def _grade_run_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    mechanism_rows = [
        row
        for row in (snapshot.get("step_outputs") or [])
        if row.get("step_name") == "mechanism_synthesis"
    ]
    if not mechanism_rows:
        return {
            "score": 0.0,
            "passed": False,
            "summary": {
                "reason": "no_mechanism_steps",
                "run_status": snapshot.get("status"),
            },
        }

    latest = sorted(mechanism_rows, key=lambda item: int(item.get("attempt") or 0))[-1]
    output = latest.get("output") or {}
    validation = latest.get("validation") or {}
    contains_target = bool(output.get("contains_target_product"))
    validation_passed = bool(validation.get("passed")) if isinstance(validation, dict) else False
    run_status = str(snapshot.get("status") or "")

    score = 0.0
    if mechanism_rows:
        score += 0.2
    if validation_passed:
        score += 0.4
    if contains_target:
        score += 0.4
    score = min(score, 1.0)
    passed = contains_target and validation_passed and run_status in {"completed", "running"}
    return {
        "score": score,
        "passed": passed,
        "summary": {
            "contains_target_product": contains_target,
            "validation_passed": validation_passed,
            "attempt": latest.get("attempt"),
            "run_status": run_status,
        },
    }


def _load_eval_tier_ids(base: Path) -> Dict[str, List[str]]:
    tiers_path = base / "training_data" / "eval_tiers.json"
    if not tiers_path.exists():
        return {"easy": [], "medium": [], "hard": []}
    try:
        payload = json.loads(tiers_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"easy": [], "medium": [], "hard": []}
    out: Dict[str, List[str]] = {}
    for tier in ("easy", "medium", "hard"):
        raw = payload.get(tier) if isinstance(payload, dict) else None
        if isinstance(raw, list):
            out[tier] = [str(item) for item in raw]
        else:
            out[tier] = []
    return out


def _is_leaderboard_holdout_eval_set(eval_set_row: Dict[str, Any] | None) -> bool:
    if not isinstance(eval_set_row, dict):
        return False
    return str(eval_set_row.get("purpose") or "general") == "leaderboard_holdout"


def _normalize_expected_known(expected: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(expected, dict):
        return None
    known = expected.get("known_mechanism")
    if isinstance(known, dict):
        return known
    verified = expected.get("verified_mechanism")
    if isinstance(verified, dict):
        return verified
    return None


def _eval_case_step_count(case: Dict[str, Any]) -> Optional[int]:
    expected = case.get("expected") or {}
    if isinstance(expected, dict):
        direct = expected.get("n_mechanistic_steps")
        if isinstance(direct, int):
            return int(direct)
        if isinstance(direct, float):
            return int(direct)
        known = _normalize_expected_known(expected)
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


def _grade_eval_snapshot(snapshot: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, Any]:
    known = _normalize_expected_known(expected)
    if not known:
        graded = _grade_run_snapshot(snapshot)
        return {
            "score": float(graded["score"]),
            "passed": bool(graded["passed"]),
            "summary": dict(graded["summary"]),
            "scoring_breakdown": {},
        }

    from mechanistic_agent.scoring import score_snapshot_against_known

    scored = score_snapshot_against_known(snapshot, expected)
    return {
        "score": float(scored.get("score") or 0.0),
        "passed": bool(scored.get("passed")),
        "summary": {
            "run_status": snapshot.get("status"),
            "final_product_reached": bool(scored.get("final_product_reached")),
            "accepted_path_step_count": int(scored.get("accepted_path_step_count") or 0),
            "known_step_count": int(scored.get("known_step_count") or 0),
        },
        "scoring_breakdown": scored,
    }


def _find_known_expected_for_snapshot(store: RunStore, snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    payload = snapshot.get("input_payload") or {}
    starting = sorted([str(x).strip() for x in payload.get("starting_materials") or [] if str(x).strip()])
    products = sorted([str(x).strip() for x in payload.get("products") or [] if str(x).strip()])
    if not starting or not products:
        return None

    for eval_set in store.list_eval_sets(exposed_in_ui=True):
        eval_set_id = str(eval_set.get("id") or "")
        if not eval_set_id:
            continue
        for case in store.list_eval_set_cases(eval_set_id):
            input_payload = case.get("input") or {}
            case_starting = sorted([str(x).strip() for x in input_payload.get("starting_materials") or [] if str(x).strip()])
            case_products = sorted([str(x).strip() for x in input_payload.get("products") or [] if str(x).strip()])
            if case_starting == starting and case_products == products:
                expected = case.get("expected")
                if isinstance(expected, dict):
                    return expected
    return None


def _capability_enabled(env_var: str) -> bool:
    """Return True unless the named env var is explicitly set to 'false' or '0'."""
    val = os.environ.get(env_var, "true").strip().lower()
    return val not in ("false", "0", "no", "off")


def _run_git_command(args: List[str], cwd: Path) -> tuple[int, str, str]:
    if not _capability_enabled("MECH_ENABLE_SUBPROCESS_OPS"):
        raise HTTPException(
            status_code=403,
            detail="Subprocess operations are disabled (MECH_ENABLE_SUBPROCESS_OPS=false).",
        )
    completed = subprocess.run(
        args,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


_ATOM_MAP_TEXT_PATTERN = re.compile(r":\d+\]")
_GENERIC_EXAMPLE_NAME_PREFIXES = (
    "flower mechanism",
    "flower example",
    "humanbenchmark reaction",
    "derived from flower mechanism",
)


def _strip_atom_maps_from_text(text: Any) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    return _ATOM_MAP_TEXT_PATTERN.sub("]", raw)


def _is_generic_example_name(name: str) -> bool:
    normalized = str(name or "").strip().lower()
    if not normalized:
        return True
    return any(normalized.startswith(prefix) for prefix in _GENERIC_EXAMPLE_NAME_PREFIXES)


def _sanitize_known_mechanism_step(step: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = dict(step)
    if "target_smiles" in cleaned:
        cleaned["target_smiles"] = strip_atom_mapping_optional(cleaned.get("target_smiles")) or str(cleaned.get("target_smiles") or "")
    if "current_state" in cleaned and isinstance(cleaned.get("current_state"), list):
        cleaned["current_state"] = strip_atom_mapping_list([str(item) for item in cleaned.get("current_state") or []])
    if "resulting_state" in cleaned and isinstance(cleaned.get("resulting_state"), list):
        cleaned["resulting_state"] = strip_atom_mapping_list([str(item) for item in cleaned.get("resulting_state") or []])
    if "predicted_intermediate" in cleaned:
        cleaned["predicted_intermediate"] = (
            strip_atom_mapping_optional(cleaned.get("predicted_intermediate"))
            or str(cleaned.get("predicted_intermediate") or "")
        )
    if "target_products" in cleaned and isinstance(cleaned.get("target_products"), list):
        cleaned["target_products"] = strip_atom_mapping_list([str(item) for item in cleaned.get("target_products") or []])
    if "reaction_smirks" in cleaned:
        cleaned["reaction_smirks"] = _strip_atom_maps_from_text(cleaned.get("reaction_smirks"))
    return cleaned


def _build_known_mechanism_payload(example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    starting_materials = strip_atom_mapping_list([str(item) for item in example.get("starting_materials") or []])
    products = strip_atom_mapping_list([str(item) for item in example.get("products") or []])
    transformation = classify_functional_group_transformation(starting_materials, products)

    known = example.get("known_mechanism")
    known_base = dict(known) if isinstance(known, dict) else {}
    existing_steps = list(known_base.get("steps") or []) if isinstance(known_base.get("steps"), list) else []
    existing_by_index = {
        int(step.get("step_index") or idx + 1): _sanitize_known_mechanism_step(step)
        for idx, step in enumerate(existing_steps)
        if isinstance(step, dict)
    }

    verified = example.get("verified_mechanism")
    verified_steps = list((verified or {}).get("steps") or []) if isinstance(verified, dict) else []

    merged_steps: List[Dict[str, Any]] = []
    seen_indices: set[int] = set()
    for idx, step in enumerate(verified_steps, start=1):
        if not isinstance(step, dict):
            continue
        step_index = int(step.get("step_index") or idx)
        seen_indices.add(step_index)
        current_state = strip_atom_mapping_list([str(item) for item in step.get("current_state") or []])
        resulting_state = strip_atom_mapping_list([str(item) for item in step.get("resulting_state") or []])
        predicted_intermediate = strip_atom_mapping_optional(step.get("predicted_intermediate"))
        target_products = strip_atom_mapping_list([str(item) for item in step.get("target_products") or []])
        target_smiles = (
            strip_atom_mapping_optional(existing_by_index.get(step_index, {}).get("target_smiles"))
            or (resulting_state[0] if resulting_state else None)
            or predicted_intermediate
            or ""
        )
        step_transformation = classify_functional_group_transformation(current_state, resulting_state)
        merged = dict(existing_by_index.get(step_index, {}))
        merged.update({
            "step_index": step_index,
            "current_state": current_state,
            "resulting_state": resulting_state,
            "predicted_intermediate": predicted_intermediate or "",
            "target_products": target_products,
            "target_smiles": target_smiles,
            "reaction_smirks": _strip_atom_maps_from_text(step.get("reaction_smirks")),
            "reaction_label": step_transformation.get("label"),
            "reaction_label_candidates": list(step_transformation.get("label_candidates") or []),
            "uncertain": bool(step_transformation.get("uncertain")),
            "uncertainty_note": str(step_transformation.get("uncertainty_note") or ""),
        })
        merged_steps.append(merged)

    for idx in sorted(existing_by_index):
        if idx not in seen_indices:
            merged_steps.append(existing_by_index[idx])

    merged_steps.sort(key=lambda step: int(step.get("step_index") or 0))
    min_steps = int(
        known_base.get("min_steps")
        or example.get("n_mechanistic_steps")
        or len(verified_steps)
        or len(merged_steps)
        or 0
    )
    if not merged_steps and min_steps <= 0:
        return None

    raw_source = str(known_base.get("source") or example.get("source") or "benchmark")
    clean_source = re.sub(r"\b\d+_HumanBenchmark\.\w+", "benchmark", raw_source)
    clean_source = re.sub(r"FlowER\s+flower_new_dataset\s+train\.txt", "FlowER 100", clean_source)
    clean_source = re.sub(r"\.(csv|json|txt|xlsx)$", "", clean_source).strip() or "benchmark"

    return {
        "source": clean_source,
        "citation": str(
            known_base.get("citation")
            or ("Derived from verified mechanism steps" if verified_steps else "")
        ),
        "min_steps": min_steps,
        "steps": merged_steps,
        "final_products": products,
        "reaction_label": transformation.get("label"),
        "reaction_label_candidates": list(transformation.get("label_candidates") or []),
        "uncertain": bool(transformation.get("uncertain")),
        "uncertainty_note": str(transformation.get("uncertainty_note") or ""),
    }


def _prepare_example_record(item: Dict[str, Any], source_label: str) -> Optional[Dict[str, Any]]:
    item_id = str(item.get("id") or item.get("case_id") or "").strip()
    starting_materials_raw = item.get("starting_materials") or []
    products_raw = item.get("products") or []
    if not item_id or not isinstance(starting_materials_raw, list) or not isinstance(products_raw, list):
        return None

    starting_materials = strip_atom_mapping_list([str(entry) for entry in starting_materials_raw])
    products = strip_atom_mapping_list([str(entry) for entry in products_raw])
    transformation = classify_functional_group_transformation(starting_materials, products)
    original_name = str(item.get("name") or "").strip()
    display_name = original_name
    if _is_generic_example_name(display_name) or display_name == item_id:
        display_name = str(transformation.get("label") or "").strip() or item_id

    prepared = dict(item)
    prepared["id"] = item_id
    prepared["name"] = display_name
    prepared["source_name"] = original_name
    prepared["starting_materials"] = starting_materials
    prepared["products"] = products
    prepared["source"] = str(item.get("source") or source_label)
    prepared["functional_group_transformation"] = transformation
    prepared["derived_reaction_label"] = transformation.get("label")
    prepared["reaction_label_uncertain"] = bool(transformation.get("uncertain"))
    prepared["reaction_label_candidates"] = list(transformation.get("label_candidates") or [])
    prepared["known_mechanism"] = _build_known_mechanism_payload(prepared)
    prepared["n_mechanistic_steps"] = int(
        prepared.get("n_mechanistic_steps")
        or len((((prepared.get("verified_mechanism") or {}).get("steps")) or []))
        or len((((prepared.get("known_mechanism") or {}).get("steps")) or []))
        or 0
    )
    return prepared


def create_app(base_dir: Path | None = None) -> FastAPI:
    base = (base_dir or Path.cwd()).resolve()
    ui_dir = base / "mechanistic_agent" / "ui"
    db_path = base / "data" / "mechanistic.db"

    registry = RegistrySet(base)
    store: RunStateStore = SQLiteRunStore(db_path)
    artifact_store = LocalArtifactStore(base)
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

    coordinator = RunCoordinator(store)
    manager = RunManager(coordinator)
    eval_executor = ThreadJobExecutor()

    app = FastAPI(title="Mechanistic Local Runtime", version="0.2.0")

    # Strict CORS — only enabled when MECH_CORS_ORIGINS is set.
    # Set to a comma-separated list of exact origins, e.g.:
    #   MECH_CORS_ORIGINS=https://app.example.com,https://api.example.com
    # When unset, no cross-origin headers are added (same-origin local usage only).
    _cors_origins_raw = os.environ.get("MECH_CORS_ORIGINS", "").strip()
    if _cors_origins_raw:
        _cors_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=_cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type", "Authorization"],
        )

    if ui_dir.exists():
        app.mount("/ui", StaticFiles(directory=ui_dir), name="ui")

    def _resolve_model_name(*candidates: Any) -> str:
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return get_default_model()

    def _resolve_public_thinking_level(*candidates: Any) -> str | None:
        for candidate in candidates:
            if not isinstance(candidate, str):
                continue
            text = candidate.strip().lower()
            if not text:
                continue
            mapped = to_public_reasoning_level(text)
            if mapped in {"low", "high"}:
                return mapped
        return None

    def _create_run_internal(
        *,
        mode: str,
        starting_materials: List[str],
        products: List[str],
        example_id: str | None,
        temperature_celsius: float,
        ph: float | None,
        model_name: str,
        thinking_level: str | None,
        api_keys: Dict[str, str],
        optional_llm_tools: List[str],
        functional_groups_enabled: bool,
        intermediate_prediction_enabled: bool,
        max_steps: int,
        max_runtime_seconds: float,
        retry_same_candidate_max: int,
        max_reproposals_per_step: int,
        reproposal_on_repeat_failure: bool,
        candidate_rescue_enabled: bool,
        step_mapping_enabled: bool,
        arrow_push_annotation_enabled: bool,
        dbe_policy: str,
        reaction_template_policy: str,
        reaction_template_confidence_threshold: float,
        reaction_template_margin_threshold: float,
        orchestration_mode: str = "standard",
        harness_name: str = "default",
        harness_config_path: str | None = None,
        coordination_topology: str = "centralized_mas",
        ralph: Dict[str, Any] | None = None,
        dry_run: bool = False,
    ) -> tuple[str, Dict[str, Any], Dict[str, str]]:
        public_thinking = _resolve_public_thinking_level(thinking_level)
        model_plan = select_step_models(
            model_name=model_name,
            thinking_level=public_thinking,
            functional_groups_enabled=functional_groups_enabled,
            intermediate_prediction_enabled=intermediate_prediction_enabled,
            optional_llm_tools=optional_llm_tools,
        )
        model_family = model_plan.family
        internal_reasoning = to_internal_reasoning_level(public_thinking)

        selected_primary_model = model_plan.step_models.get("mechanism_synthesis") or model_name

        ralph = dict(ralph or {})
        hashes = registry.bundle_hashes(model_name=selected_primary_model)
        run_id = store.create_run(
            mode=mode,
            input_payload={
                "starting_materials": starting_materials,
                "products": products,
                "example_id": example_id,
                "temperature_celsius": temperature_celsius,
                "ph": ph,
            },
            prompt_bundle_hash=hashes.get("prompt_bundle_hash", ""),
            skill_bundle_hash=hashes.get("skill_bundle_hash", ""),
            memory_bundle_hash=hashes.get("memory_bundle_hash", ""),
            config={
                "model": selected_primary_model,
                "model_name": selected_primary_model,
                "model_family": model_family,
                "thinking_level": public_thinking,
                "reasoning_level": internal_reasoning,
                "step_models": model_plan.step_models,
                "step_reasoning": model_plan.step_reasoning,
                "api_keys": api_keys,
                "optional_llm_tools": optional_llm_tools,
                "functional_groups_enabled": functional_groups_enabled,
                "intermediate_prediction_enabled": intermediate_prediction_enabled,
                "model_plan_notes": model_plan.notes,
                "max_steps": max_steps,
                "max_runtime_seconds": max_runtime_seconds,
                "retry_same_candidate_max": retry_same_candidate_max,
                "max_reproposals_per_step": max_reproposals_per_step,
                "reproposal_on_repeat_failure": reproposal_on_repeat_failure,
                "candidate_rescue_enabled": candidate_rescue_enabled,
                "step_mapping_enabled": step_mapping_enabled,
                "arrow_push_annotation_enabled": arrow_push_annotation_enabled,
                "dbe_policy": dbe_policy,
                "reaction_template_policy": reaction_template_policy,
                "reaction_template_confidence_threshold": reaction_template_confidence_threshold,
                "reaction_template_margin_threshold": reaction_template_margin_threshold,
                "orchestration_mode": (
                    "ralph" if str(orchestration_mode).strip().lower() == "ralph" else "standard"
                ),
                "harness_name": str(harness_name or "default"),
                "harness_config_path": harness_config_path,
                "coordination_topology": str(coordination_topology or "centralized_mas"),
                "harness_strategy": str(ralph.get("harness_strategy") or "latest"),
                "harness_list": list(ralph.get("harness_list") or []),
                "max_iterations": int(ralph.get("max_iterations") or 0),
                "completion_promise": "target_products_reached && flow_node:run_complete",
                "ralph_max_runtime_seconds": float(ralph.get("max_runtime_seconds") or 6000.0),
                "max_cost_usd": ralph.get("max_cost_usd", 2.0),
                "repeat_failure_signature_limit": int(
                    ralph.get("repeat_failure_signature_limit") or 2
                ),
                "babysit_mode": str(ralph.get("babysit_mode") or "off"),
                "allow_validator_mutation": bool(ralph.get("allow_validator_mutation", False)),
                "example_id": example_id,
                "dry_run": dry_run,
            },
        )
        prompt_records = registry.prompt_step_map(model_name=selected_primary_model)
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
                "starting_materials": starting_materials,
                "products": products,
                "model_family": model_family,
                "model_name": selected_primary_model,
                "thinking_level": public_thinking,
                "step_models": model_plan.step_models,
                "model_plan_notes": model_plan.notes,
                    "prompt_versions_by_step": prompt_ids_by_step,
                    "note": "Run created in pending state; call /api/runs/{id}/start to begin execution.",
                    **hashes,
                },
            )
        return run_id, hashes, model_plan.step_models

    def _wait_for_run(run_id: str, *, timeout_seconds: float) -> Dict[str, Any]:
        start = time.monotonic()
        while True:
            snapshot = store.get_run_snapshot(run_id)
            if snapshot is None:
                return {"id": run_id, "status": "failed", "error": "run_not_found"}
            status = str(snapshot.get("status") or "")
            if status in TERMINAL_STATUSES:
                return snapshot
            if time.monotonic() - start > timeout_seconds:
                store.set_run_status(run_id, "stopped")
                store.append_event(run_id, "run_stopped", {"reason": "eval_timeout"})
                snapshot = store.get_run_snapshot(run_id) or {"id": run_id, "status": "stopped"}
                return snapshot
            time.sleep(0.25)

    def _record_validation_nodes(
        *,
        run_id: str,
        attempt: int,
        retry_index: int,
        validation: Dict[str, Any],
    ) -> None:
        checks = validation.get("checks")
        if not isinstance(checks, list):
            return
        check_to_node = {
            "dbe_metadata": "bond_electron_validation",
            "atom_balance": "atom_balance_validation",
            "state_progress": "state_progress_validation",
        }
        for check in checks:
            if not isinstance(check, dict):
                continue
            check_name = str(check.get("name") or "")
            node = check_to_node.get(check_name)
            if not node:
                continue
            payload = {
                "passed": bool(check.get("passed")),
                "checks": [check],
            }
            store.append_event(
                run_id,
                "step_started",
                {"step_name": node, "tool_name": node, "attempt": attempt, "retry_index": retry_index},
                step_name=node,
            )
            store.record_step_output(
                run_id=run_id,
                step_name=node,
                attempt=attempt,
                retry_index=retry_index,
                source="deterministic",
                model=None,
                reasoning_level=None,
                tool_name=node,
                output={"check": check_name, "details": check.get("details", {})},
                validation=payload,
                accepted_bool=True if bool(check.get("passed")) else None,
            )
            store.append_event(
                run_id,
                "step_completed" if bool(check.get("passed")) else "step_failed",
                {
                    "step_name": node,
                    "attempt": attempt,
                    "retry_index": retry_index,
                    "validation": payload,
                },
                step_name=node,
            )

    def _execute_eval_runset(eval_run_id: str, payload: EvalRunSetRequest) -> None:
        try:
            cases = store.list_eval_set_cases(payload.eval_set_id)
            max_cases = max(1, int(payload.max_cases))
            by_case_id = {str(case.get("case_id") or ""): case for case in cases}

            if payload.case_ids:
                selected_cases = [
                    by_case_id[case_id]
                    for case_id in [str(item) for item in payload.case_ids]
                    if case_id in by_case_id
                ]
            elif payload.step_count is not None:
                requested_steps = int(payload.step_count)
                selected_cases = [
                    case
                    for case in cases
                    if int(_eval_case_step_count(case) or 0) == requested_steps
                ]
            elif payload.tier_name:
                tiers = _load_eval_tier_ids(base)
                tier_ids = tiers.get(payload.tier_name, [])
                selected_cases = [by_case_id[case_id] for case_id in tier_ids if case_id in by_case_id]
            else:
                selected_cases = cases[:max_cases]

            if not selected_cases:
                store.set_eval_run_status(eval_run_id, "failed")
                return

            for case in selected_cases:
                case_id = str(case.get("case_id") or "")
                input_payload = case.get("input") or {}
                starting = [str(item) for item in input_payload.get("starting_materials", [])]
                products = [str(item) for item in input_payload.get("products", [])]
                if not starting or not products:
                    store.record_eval_run_result(
                        eval_run_id=eval_run_id,
                        case_id=case_id or "unknown_case",
                        run_id=None,
                        score=0.0,
                        passed=False,
                        cost={},
                        latency_ms=None,
                        summary={"reason": "invalid_case_payload"},
                    )
                    continue

                # Honour per-case run_config overrides from templates.
                run_config_override = input_payload.get("run_config") or {}
                case_model_name = _resolve_model_name(
                    run_config_override.get("model_name"),
                    run_config_override.get("model"),
                    payload.model_name,
                    payload.model,
                )
                case_thinking = _resolve_public_thinking_level(
                    run_config_override.get("thinking_level"),
                    run_config_override.get("reasoning_level"),
                    payload.thinking_level,
                    payload.reasoning_level,
                )

                created_at = time.time()
                run_id, _, step_models = _create_run_internal(
                    mode=payload.mode,
                    starting_materials=starting,
                    products=products,
                    example_id=None,
                    temperature_celsius=float(input_payload.get("temperature_celsius", 25.0)),
                    ph=input_payload.get("ph"),
                    model_name=case_model_name,
                    thinking_level=case_thinking,
                    api_keys={},
                    optional_llm_tools=["attempt_atom_mapping", "predict_missing_reagents"],
                    functional_groups_enabled=True,
                    intermediate_prediction_enabled=True,
                    max_steps=payload.max_steps,
                    max_runtime_seconds=payload.max_runtime_seconds,
                    retry_same_candidate_max=1,
                    max_reproposals_per_step=4,
                    reproposal_on_repeat_failure=True,
                    candidate_rescue_enabled=True,
                    step_mapping_enabled=True,
                    arrow_push_annotation_enabled=True,
                    dbe_policy="soft",
                    reaction_template_policy="auto",
                    reaction_template_confidence_threshold=0.65,
                    reaction_template_margin_threshold=0.10,
                )
                store.append_event(
                    run_id,
                    "eval_case_started",
                    {"eval_run_id": eval_run_id, "case_id": case_id},
                )
                manager.start(run_id)
                snapshot = _wait_for_run(run_id, timeout_seconds=payload.max_runtime_seconds + 30.0)

                # Compare with known answers when verified_mechanism is present.
                expected = case.get("expected") or {}
                graded = _grade_eval_snapshot(snapshot, expected if isinstance(expected, dict) else {})
                known_mech = _normalize_expected_known(expected if isinstance(expected, dict) else {})
                case_step_count = _eval_case_step_count(case)
                if known_mech:
                    from mechanistic_agent.run_evaluator import compare_with_known_answers

                    mechanism_rows = [
                        row
                        for row in (snapshot.get("step_outputs") or [])
                        if row.get("step_name") == "mechanism_synthesis"
                    ]
                    known_answer_comparison = compare_with_known_answers(mechanism_rows, known_mech)
                else:
                    known_answer_comparison = {"available": False}

                from mechanistic_agent.scoring import score_subagents_from_step_outputs

                subagent_scores = score_subagents_from_step_outputs(
                    list(snapshot.get("step_outputs") or [])
                )
                store.record_eval_run_result(
                    eval_run_id=eval_run_id,
                    case_id=case_id or run_id,
                    run_id=run_id,
                    score=float(graded["score"]),
                    passed=bool(graded["passed"]),
                    cost={"total_cost": 0.0},
                    latency_ms=max((time.time() - created_at) * 1000.0, 0.0),
                    summary={
                        **graded["summary"],
                        "n_mechanistic_steps": case_step_count,
                        "selected_step_models": step_models,
                        "known_answer_comparison": known_answer_comparison,
                        "scoring_breakdown": graded.get("scoring_breakdown", {}),
                        "subagent_scores": subagent_scores,
                    },
                )
            store.set_eval_run_status(eval_run_id, "completed")
        except Exception as exc:  # pragma: no cover - defensive
            store.set_eval_run_status(eval_run_id, "failed")
            failed_run = store.get_eval_run(eval_run_id)
            if failed_run is not None:
                store.record_eval_run_result(
                    eval_run_id=eval_run_id,
                    case_id="__error__",
                    run_id=None,
                    score=0.0,
                    passed=False,
                    cost={},
                    latency_ms=None,
                    summary={"error": str(exc)},
                )

    @app.get("/")
    def root() -> FileResponse:
        index_path = ui_dir / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="UI not found")
        return FileResponse(index_path)

    @app.post("/api/runs", response_model=CreateRunResponse)
    def create_run(payload: CreateRunRequest) -> CreateRunResponse:
        if not payload.starting_materials:
            raise HTTPException(status_code=400, detail="starting_materials cannot be empty")
        if not payload.products:
            raise HTTPException(status_code=400, detail="products cannot be empty")

        run_id, _, _ = _create_run_internal(
            mode=payload.mode,
            starting_materials=payload.starting_materials,
            products=payload.products,
            example_id=payload.example_id,
            temperature_celsius=payload.temperature_celsius,
            ph=payload.ph,
            model_name=_resolve_model_name(payload.model_name, payload.model),
            thinking_level=_resolve_public_thinking_level(payload.thinking_level, payload.reasoning_level),
            api_keys=payload.api_keys,
            optional_llm_tools=payload.optional_llm_tools,
            functional_groups_enabled=payload.functional_groups_enabled,
            intermediate_prediction_enabled=payload.intermediate_prediction_enabled,
            max_steps=payload.max_steps,
            max_runtime_seconds=payload.max_runtime_seconds,
            retry_same_candidate_max=payload.retry_same_candidate_max,
            max_reproposals_per_step=payload.max_reproposals_per_step,
            reproposal_on_repeat_failure=payload.reproposal_on_repeat_failure,
            candidate_rescue_enabled=payload.candidate_rescue_enabled,
            step_mapping_enabled=payload.step_mapping_enabled,
            arrow_push_annotation_enabled=payload.arrow_push_annotation_enabled,
            dbe_policy=payload.dbe_policy,
            reaction_template_policy=payload.reaction_template_policy,
            reaction_template_confidence_threshold=payload.reaction_template_confidence_threshold,
            reaction_template_margin_threshold=payload.reaction_template_margin_threshold,
            orchestration_mode=payload.orchestration_mode,
            harness_name=payload.harness_name,
            harness_config_path=payload.harness_config_path,
            coordination_topology=payload.coordination_topology,
            ralph=payload.ralph.model_dump() if payload.ralph is not None else None,
            dry_run=payload.dry_run,
        )
        return CreateRunResponse(
            run_id=run_id,
            status="pending",
            note="Run created and pending; call /api/runs/{run_id}/start to execute.",
        )

    @app.post("/api/runs/{run_id}/start")
    def start_run(run_id: str) -> Dict[str, Any]:
        row = store.get_run_row(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if row["status"] in TERMINAL_STATUSES:
            raise HTTPException(status_code=400, detail=f"Run is already {row['status']}")
        if row["status"] == "paused":
            raise HTTPException(status_code=400, detail="Run is paused; use /api/runs/{run_id}/resume")

        store.append_event(run_id, "run_start_requested", {})
        manager.start(run_id)
        return {
            "run_id": run_id,
            "status": "running",
            "note": "Run start was requested; execution is now in progress.",
        }

    @app.post("/api/runs/{run_id}/resume")
    def resume_run(run_id: str, payload: ResumeRunRequest) -> Dict[str, Any]:
        row = store.get_run_row(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if row.get("status") != "paused":
            raise HTTPException(status_code=400, detail="Run is not paused")

        pause = store.get_latest_run_pause(run_id)
        if pause and pause.get("decision") is None:
            store.resolve_run_pause(
                pause_id=str(pause["id"]),
                decision=payload.decision,
                decided_by=payload.decided_by,
                rationale=payload.rationale,
            )

        if payload.decision == "stop":
            store.set_run_status(run_id, "stopped")
            store.append_event(
                run_id,
                "run_stopped",
                {"reason": "user_stopped_from_pause", "rationale": payload.rationale},
            )
            return {"run_id": run_id, "status": "stopped"}

        store.append_event(
            run_id,
            "run_resume_requested",
            {"decision": payload.decision, "rationale": payload.rationale, "decided_by": payload.decided_by},
        )
        manager.start(run_id)
        return {"run_id": run_id, "status": "running"}

    @app.post("/api/runs/{run_id}/stop")
    def stop_run(run_id: str) -> Dict[str, Any]:
        row = store.get_run_row(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Run not found")
        issued = manager.stop(run_id)
        store.append_event(run_id, "run_stop_requested", {"issued": issued})
        if issued:
            return {"run_id": run_id, "status": "stopping"}
        return {"run_id": run_id, "status": row["status"]}

    @app.post("/api/runs/{run_id}/discard")
    def discard_run(run_id: str) -> Dict[str, Any]:
        """Discard a completed dry-run: delete all DB records and filesystem traces."""
        row = store.get_run_row(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if row["status"] not in TERMINAL_STATUSES:
            raise HTTPException(status_code=400, detail="Run must be in a terminal state before discarding")
        deleted = store.delete_run(run_id)
        return {"ok": deleted, "run_id": run_id, "deleted": deleted}

    @app.post("/api/runs/{run_id}/mechanism_steps")
    def submit_verified_mechanism_step(run_id: str, payload: MechanismStepSubmitRequest) -> Dict[str, Any]:
        row = store.get_run_row(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if row.get("mode") != "verified":
            raise HTTPException(status_code=400, detail="Mechanism step submission is only valid in verified mode")
        if row.get("status") in TERMINAL_STATUSES:
            raise HTTPException(status_code=400, detail=f"Run already {row.get('status')}")

        if payload.step_index < 1:
            raise HTTPException(status_code=400, detail="step_index must start at 1")

        try:
            raw = predict_mechanistic_step(
                step_index=payload.step_index,
                current_state=payload.current_state,
                target_products=payload.target_products,
                electron_pushes=payload.electron_pushes,
                reaction_smirks=payload.reaction_smirks,
                predicted_intermediate=payload.predicted_intermediate,
                resulting_state=payload.resulting_state,
                previous_intermediates=[],
                note=payload.note,
                starting_materials=payload.current_state,
            )
            output = json.loads(raw)
            if not isinstance(output, dict):
                raise ValueError("Invalid mechanism step output")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid mechanistic step payload: {exc}") from exc

        cfg = row.get("config") or {}
        validation_obj = validate_mechanism_step_output(
            output,
            dbe_policy=str(cfg.get("dbe_policy") or "soft"),
        )
        validation = validation_obj.as_dict()
        attempt = int(payload.step_index)

        store.append_event(
            run_id,
            "step_started",
            {
                "step_name": "mechanism_synthesis",
                "tool_name": "human_submitted_mechanistic_step",
                "attempt": attempt,
                "retry_index": 0,
            },
            step_name="mechanism_synthesis",
        )
        store.record_step_output(
            run_id=run_id,
            step_name="mechanism_synthesis",
            attempt=attempt,
            retry_index=0,
            source="human",
            model="human_input",
            reasoning_level=None,
            tool_name="human_submitted_mechanistic_step",
            output=output,
            validation=validation,
            accepted_bool=True if validation.get("passed") else None,
        )
        store.add_trace_record(
            run_id=run_id,
            step_name="mechanism_synthesis",
            model="human_input",
            reasoning_level=None,
            score=1.0 if validation.get("passed") else 0.0,
            source="run",
            approved=bool(validation.get("passed")),
            approval_label="verified_submission" if validation.get("passed") else None,
            trace={
                "tool_name": "human_submitted_mechanistic_step",
                "attempt": attempt,
                "retry_index": 0,
                "output": output,
                "validation": validation,
                "captured_at": time.time(),
            },
        )
        store.append_event(
            run_id,
            "step_completed" if validation.get("passed") else "step_failed",
            {
                "step_name": "mechanism_synthesis",
                "attempt": attempt,
                "validation": validation,
            },
            step_name="mechanism_synthesis",
        )
        _record_validation_nodes(run_id=run_id, attempt=attempt, retry_index=0, validation=validation)

        if not validation.get("passed"):
            store.append_event(
                run_id,
                "awaiting_manual_steps",
                {
                    "next_step_index": attempt,
                    "reason": "validation_failed",
                },
            )
            return {
                "run_id": run_id,
                "status": "running",
                "step_status": "needs_revision",
                "validation": validation,
            }

        if bool(cfg.get("arrow_push_annotation_enabled", True)):
            try:
                annotation = predict_arrow_push_annotation(
                    current_state=[str(item) for item in output.get("current_state") or []],
                    resulting_state=[str(item) for item in output.get("resulting_state") or []],
                    reaction_smirks=str(output.get("reaction_smirks") or ""),
                    raw_reaction_smirks=str(output.get("raw_reaction_smirks") or ""),
                    electron_pushes=output.get("electron_pushes"),
                    step_index=int(output.get("step_index") or attempt),
                    candidate_rank=None,
                )
                store.record_arrow_push_annotation(
                    run_id=run_id,
                    step_index=int(annotation.get("step_index") or attempt),
                    attempt=attempt,
                    retry_index=0,
                    candidate_rank=None,
                    source="verified_submission",
                    prediction=annotation,
                )
            except Exception:
                pass

        contains_target = bool(output.get("contains_target_product"))
        store.append_event(
            run_id,
            "completion_check",
            {
                "step_index": attempt,
                "contains_target_product": contains_target,
                "validation_passed": True,
            },
            step_name="completion_check",
        )
        if contains_target:
            store.set_run_status(run_id, "completed")
            store.append_event(
                run_id,
                "run_completed",
                {"reason": "verified_steps_completed", "step_index": attempt},
            )
            return {
                "run_id": run_id,
                "status": "completed",
                "step_status": "accepted",
                "validation": validation,
            }

        store.set_run_status(run_id, "running")
        store.append_event(
            run_id,
            "awaiting_manual_steps",
            {"next_step_index": attempt + 1},
        )
        return {
            "run_id": run_id,
            "status": "running",
            "step_status": "accepted",
            "next_step_index": attempt + 1,
            "validation": validation,
        }

    @app.get("/api/runs/{run_id}")
    def run_snapshot(
        run_id: str,
        verbose: bool = Query(default=False),
    ) -> Dict[str, Any]:
        snapshot = store.get_run_snapshot(run_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="Run not found")
        display_snapshot = snapshot
        parent_config = snapshot.get("config") if isinstance(snapshot.get("config"), dict) else {}
        if (
            str(parent_config.get("orchestration_mode") or "standard") == "ralph"
            and snapshot.get("ralph_latest_child_run_id")
        ):
            child_run_id = str(snapshot.get("ralph_latest_child_run_id") or "")
            if child_run_id:
                child_snapshot = store.get_run_snapshot(child_run_id)
                if isinstance(child_snapshot, dict):
                    display_snapshot = child_snapshot
                    snapshot["ralph_latest_child_snapshot"] = {
                        "id": child_snapshot.get("id"),
                        "status": child_snapshot.get("status"),
                        "step_outputs": child_snapshot.get("step_outputs", []),
                        "events": child_snapshot.get("events", []),
                    }
        snapshot["is_active"] = manager.is_running(run_id)
        snapshot["progress"] = _compute_progress(display_snapshot)
        snapshot["mechanism_summary"] = _apply_summary_images(_build_mechanism_summary(display_snapshot))
        snapshot["reaction_visuals"] = _build_reaction_visuals(display_snapshot)
        snapshot["failed_paths"] = _extract_failed_paths(display_snapshot)
        snapshot["latest_evaluation"] = store.get_latest_evaluation(run_id)
        snapshot["step_prompts"] = store.list_run_step_prompts(run_id)
        snapshot["flow"] = _build_flow_state(display_snapshot, prompt_step_map=registry.prompt_step_map())
        snapshot["latest_step_mapping"] = _latest_step_mapping_summary(display_snapshot)
        snapshot["reaction_type_selection"] = _latest_reaction_type_selection(display_snapshot)
        snapshot["template_guidance_state"] = _latest_template_guidance_state(display_snapshot)

        if verbose:
            return snapshot

        compact_config = snapshot.get("config", {})
        return {
            "id": snapshot.get("id"),
            "created_at": snapshot.get("created_at"),
            "status": snapshot.get("status"),
            "mode": snapshot.get("mode"),
            "input_payload": snapshot.get("input_payload"),
            "config": {
                "model": compact_config.get("model"),
                "model_name": compact_config.get("model_name") or compact_config.get("model"),
                "model_family": compact_config.get("model_family"),
                "thinking_level": compact_config.get("thinking_level"),
                "reasoning_level": compact_config.get("reasoning_level"),
                "step_models": compact_config.get("step_models", {}),
                "step_reasoning": compact_config.get("step_reasoning", {}),
                "functional_groups_enabled": compact_config.get("functional_groups_enabled"),
                "intermediate_prediction_enabled": compact_config.get("intermediate_prediction_enabled"),
                "optional_llm_tools": compact_config.get("optional_llm_tools", []),
                "max_steps": compact_config.get("max_steps"),
                "retry_same_candidate_max": compact_config.get("retry_same_candidate_max"),
                "reproposal_on_repeat_failure": compact_config.get("reproposal_on_repeat_failure"),
                "candidate_rescue_enabled": compact_config.get("candidate_rescue_enabled"),
                "step_mapping_enabled": compact_config.get("step_mapping_enabled"),
                "arrow_push_annotation_enabled": compact_config.get("arrow_push_annotation_enabled"),
                "dbe_policy": compact_config.get("dbe_policy"),
                "reaction_template_policy": compact_config.get("reaction_template_policy"),
                "reaction_template_confidence_threshold": compact_config.get("reaction_template_confidence_threshold"),
                "reaction_template_margin_threshold": compact_config.get("reaction_template_margin_threshold"),
                "orchestration_mode": compact_config.get("orchestration_mode", "standard"),
                "harness_name": compact_config.get("harness_name", "default"),
                "harness_config_path": compact_config.get("harness_config_path"),
                "harness_strategy": compact_config.get("harness_strategy", "latest"),
                "harness_list": compact_config.get("harness_list", []),
                "max_iterations": compact_config.get("max_iterations"),
                "ralph_max_runtime_seconds": compact_config.get("ralph_max_runtime_seconds"),
                "max_cost_usd": compact_config.get("max_cost_usd"),
                "repeat_failure_signature_limit": compact_config.get("repeat_failure_signature_limit"),
                "babysit_mode": compact_config.get("babysit_mode", "off"),
                "allow_validator_mutation": compact_config.get("allow_validator_mutation", False),
            },
            "is_active": snapshot.get("is_active"),
            "progress": snapshot.get("progress"),
            "flow": snapshot.get("flow"),
            "mechanism_summary": snapshot.get("mechanism_summary"),
            "reaction_visuals": snapshot.get("reaction_visuals"),
            "latest_evaluation": snapshot.get("latest_evaluation"),
            "latest_step_mapping": snapshot.get("latest_step_mapping"),
            "reaction_type_selection": snapshot.get("reaction_type_selection"),
            "template_guidance_state": snapshot.get("template_guidance_state"),
            "pending_verification": snapshot.get("pending_verification", []),
            "latest_pause": snapshot.get("latest_pause"),
            "ralph_attempts": snapshot.get("ralph_attempts", []),
            "ralph_votes": snapshot.get("ralph_votes", []),
            "ralph_latest_child_run_id": snapshot.get("ralph_latest_child_run_id"),
            "ralph_latest_child_status": snapshot.get("ralph_latest_child_status"),
            "step_outputs": display_snapshot.get("step_outputs", []),
            "step_prompts": snapshot.get("step_prompts", []),
        }

    @app.get("/api/runs/{run_id}/flow")
    def run_flow(run_id: str) -> Dict[str, Any]:
        snapshot = store.get_run_snapshot(run_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="Run not found")
        display_snapshot = snapshot
        config = snapshot.get("config") if isinstance(snapshot.get("config"), dict) else {}
        if (
            str(config.get("orchestration_mode") or "standard") == "ralph"
            and snapshot.get("ralph_latest_child_run_id")
        ):
            child_run_id = str(snapshot.get("ralph_latest_child_run_id") or "")
            if child_run_id:
                child_snapshot = store.get_run_snapshot(child_run_id)
                if child_snapshot is not None:
                    display_snapshot = child_snapshot
        return _build_flow_state(display_snapshot, prompt_step_map=registry.prompt_step_map())

    @app.get("/api/runs/{run_id}/events")
    async def run_events_stream(
        run_id: str,
        after_seq: int = Query(default=0, ge=0),
    ) -> StreamingResponse:
        if store.get_run_row(run_id) is None:
            raise HTTPException(status_code=404, detail="Run not found")

        async def event_generator():
            cursor = after_seq
            while True:
                events = store.list_events(run_id, after_seq=cursor, limit=250)
                if events:
                    for event in events:
                        cursor = int(event["seq"])
                        payload = {
                            "seq": event["seq"],
                            "event_type": event["event_type"],
                            "step_name": event.get("step_name"),
                            "ts": event["ts"],
                            "payload": event.get("payload", {}),
                        }
                        yield (
                            f"id: {event['seq']}\n"
                            f"event: {event['event_type']}\n"
                            f"data: {json.dumps(payload)}\n\n"
                        )
                else:
                    yield ": keep-alive\n\n"

                run_row = store.get_run_row(run_id)
                if run_row and run_row.get("status") in TERMINAL_STATUSES and not events:
                    yield "event: stream_end\ndata: {}\n\n"
                    break
                await asyncio.sleep(0.5)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    @app.post("/api/runs/{run_id}/steps/{step_name}/verify")
    def verify_step(run_id: str, step_name: str, payload: VerifyStepRequest) -> Dict[str, Any]:
        row = store.get_run_row(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Run not found")

        store.record_verification_decision(
            run_id=run_id,
            step_name=step_name,
            decision=payload.decision,
            rationale=payload.rationale,
            decided_by=payload.decided_by,
        )
        store.update_step_acceptance(
            run_id=run_id,
            step_name=step_name,
            attempt=payload.attempt,
            accepted=(payload.decision == "accept"),
        )
        store.append_event(
            run_id,
            "verification_decision",
            {
                "step_name": step_name,
                "decision": payload.decision,
                "attempt": payload.attempt,
                "rationale": payload.rationale,
                "decided_by": payload.decided_by,
            },
            step_name=step_name,
        )

        if payload.decision == "reject":
            store.set_run_status(run_id, "failed")
            store.append_event(
                run_id,
                "run_failed",
                {"reason": "verification_rejected", "step_name": step_name},
                step_name=step_name,
            )
            return {"run_id": run_id, "status": "failed"}

        pending = store.unaccepted_verified_steps(run_id)
        if row.get("mode") == "verified" and row.get("status") == "running" and not pending:
            store.set_run_status(run_id, "completed")
            store.append_event(run_id, "run_completed", {"reason": "all_steps_verified"})
            return {"run_id": run_id, "status": "completed"}

        return {"run_id": run_id, "status": store.get_run_row(run_id).get("status"), "pending": len(pending)}

    @app.post("/api/runs/{run_id}/feedback")
    def record_run_feedback(run_id: str, payload: FeedbackRequest) -> Dict[str, Any]:
        if store.get_run_row(run_id) is None:
            raise HTTPException(status_code=404, detail="Run not found")
        feedback_id = store.record_feedback(
            run_id=run_id,
            step_name=None,
            rating=payload.rating,
            label=payload.label,
            comment=payload.comment,
            payload=payload.payload,
        )
        store.append_event(
            run_id,
            "feedback_recorded",
            {"feedback_id": feedback_id, "step_name": None},
        )
        return {"feedback_id": feedback_id}

    @app.post("/api/runs/{run_id}/votes")
    def record_ralph_vote(run_id: str, payload: RalphVoteRequest) -> Dict[str, Any]:
        row = store.get_run_row(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Run not found")
        config = row.get("config") if isinstance(row.get("config"), dict) else {}
        if str(config.get("orchestration_mode") or "standard") != "ralph":
            raise HTTPException(status_code=400, detail="Run is not in Ralph orchestration mode")

        vote_id = store.record_ralph_vote(
            run_id=run_id,
            attempt_index=payload.attempt_index,
            step_index=payload.step_index,
            candidate_a=payload.candidate_a,
            candidate_b=payload.candidate_b,
            vote=payload.vote,
            confidence=payload.confidence,
            source=payload.source,
        )
        store.append_event(
            run_id,
            "ralph_vote_recorded",
            {
                "vote_id": vote_id,
                "attempt_index": payload.attempt_index,
                "step_index": payload.step_index,
                "vote": payload.vote,
                "confidence": payload.confidence,
                "source": payload.source,
            },
        )
        return {"vote_id": vote_id}

    @app.get("/api/runs/{run_id}/votes")
    def list_ralph_votes(run_id: str) -> Dict[str, Any]:
        if store.get_run_row(run_id) is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return {"items": store.list_ralph_votes(run_id)}

    @app.get("/api/catalog/models")
    def catalog_models() -> List[Dict[str, Any]]:
        return list(get_model_options())

    @app.get("/api/catalog/families")
    def catalog_families() -> List[Dict[str, Any]]:
        return get_all_families()

    @app.get("/api/catalog/preview_step_models")
    def catalog_preview_step_models(
        model_name: str = get_default_model(),
        thinking_level: str | None = None,
    ) -> Dict[str, Any]:
        step_map = preview_step_models(
            model_name=model_name,
            thinking_level=thinking_level,
        )
        return {"step_models": step_map}

    @app.get("/api/catalog/flow_template")
    def catalog_flow_template(harness_name: str | None = None) -> Dict[str, Any]:
        try:
            h = registry.harness.load(harness_name or "default")
            node_specs = build_flow_node_specs(h)
            edge_specs = build_flow_edges(h)
        except (FileNotFoundError, Exception):
            node_specs = FLOW_NODE_SPECS
            edge_specs = FLOW_EDGES
        return {
            "nodes": [
                {**node, "state": "pending", "prompt_ref": None}
                for node in node_specs
            ],
            "edges": edge_specs,
        }

    @app.get("/api/catalog/skills")
    def catalog_skills() -> List[Dict[str, Any]]:
        return [
            {
                "name": item.metadata.get("name"),
                "summary": item.metadata.get("summary"),
                "path": item.path,
                "sha256": item.sha256,
            }
            for item in registry.skills.list()
        ]

    @app.get("/api/catalog/prompts")
    def catalog_prompts(model_name: str | None = None) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for item in list_call_prompt_versions(base, model_name=model_name):
            payload.append(
                {
                    "call_name": item.get("call_name"),
                    "steps": item.get("steps"),
                    "shared_base_path": item.get("shared_base_path"),
                    "call_base_path": item.get("call_base_path"),
                    "few_shot_path": item.get("few_shot_path"),
                    "resolved_shared_base_path": item.get("resolved_shared_base_path"),
                    "resolved_call_base_path": item.get("resolved_call_base_path"),
                    "resolved_few_shot_path": item.get("resolved_few_shot_path"),
                    "asset_scope": item.get("asset_scope"),
                    "model_name": item.get("model_name"),
                    "shared_base_sha256": item.get("shared_base_sha256"),
                    "call_base_sha256": item.get("call_base_sha256"),
                    "few_shot_sha256": item.get("few_shot_sha256"),
                    "prompt_bundle_sha256": item.get("prompt_bundle_sha256"),
                    "few_shot_count": len(item.get("few_shot_examples") or []),
                }
            )
        return payload

    @app.get("/api/catalog/prompt_versions")
    def catalog_prompt_versions(
        step_name: str | None = None,
        call_name: str | None = None,
        sha256: str | None = None,
        limit: int = Query(default=100, ge=1, le=500),
    ) -> Dict[str, Any]:
        items = store.list_prompt_versions(
            step_name=step_name,
            call_name=call_name,
            sha256=sha256,
            limit=limit,
        )
        return {"items": items}

    @app.get("/api/harness/config")
    def get_harness_config(name: str = "default") -> Dict[str, Any]:
        """Return the harness config by name as JSON."""
        try:
            config = registry.harness.load(name)
            return config.as_dict()
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Harness config '{name}' not found")

    @app.post("/api/harness/config")
    def save_harness_config(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Save a modified harness config. Returns the new version SHA."""
        from mechanistic_agent.core.types import HarnessConfig as HC
        try:
            config = HC.from_dict(payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid harness config: {exc}")
        name = config.name or payload.get("name") or "custom"
        version_sha = registry.harness.save(config, name=name)
        return {"name": name, "version": version_sha}

    @app.get("/api/harness/configs")
    def list_harness_configs() -> Dict[str, Any]:
        """List all available harness config files."""
        return {"items": registry.harness.list_versions()}

    @app.get("/api/harness/versions")
    def harness_versions(model_family: str | None = None, limit: int = 100) -> Dict[str, Any]:
        family = model_family or "openai"
        history = store.list_verification_history(model_family=family, limit=max(1, min(limit, 500)))
        grouped: Dict[str, Dict[str, Any]] = {}
        for row in history:
            version = str(row.get("harness_version") or "")
            if not version:
                continue
            item = grouped.setdefault(
                version,
                {
                    "harness_version": version,
                    "model_family": family,
                    "step_models": {},
                    "step_reasoning": {},
                    "mean_step_score": 0.0,
                    "step_count": 0,
                },
            )
            step_name = str(row.get("step_name") or "")
            if step_name:
                item["step_models"][step_name] = row.get("verified_model")
                item["step_reasoning"][step_name] = row.get("verified_reasoning")
            if isinstance(row.get("step_score"), (int, float)):
                item["mean_step_score"] += float(row["step_score"])
                item["step_count"] += 1
        items: List[Dict[str, Any]] = []
        for version, item in grouped.items():
            step_count = int(item.pop("step_count") or 0)
            if step_count:
                item["mean_step_score"] = item["mean_step_score"] / step_count
            else:
                item["mean_step_score"] = 0.0
            item["run_defaults"] = {
                "mode": "unverified",
                "model_name": get_default_model(family),
                "model_family": family,
                "thinking_level": None,
            }
            items.append(item)
        items.sort(key=lambda x: str(x.get("harness_version") or ""), reverse=True)
        return {"items": items}

    @app.post("/api/eval_sets/import_examples")
    def import_examples_eval_set(version: str = "flower100_v1") -> Dict[str, Any]:
        examples_path = base / "training_data" / "eval_set.json"
        if not examples_path.exists():
            raise HTTPException(status_code=404, detail="training_data/eval_set.json not found")
        try:
            raw = json.loads(examples_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid eval_set.json: {exc}") from exc
        if not isinstance(raw, list):
            raise HTTPException(status_code=400, detail="eval_set.json must contain a list")

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
                        **{"products": products},
                        **({"known_mechanism": entry.get("known_mechanism")} if isinstance(entry.get("known_mechanism"), dict) else {}),
                        **(
                            {"verified_mechanism": entry.get("verified_mechanism")}
                            if isinstance(entry.get("verified_mechanism"), dict)
                            else {}
                        ),
                    },
                    "tags": ["flower_100", "default_eval"],
                }
            )

        expected_case_count = len(cases)
        expected_has_multistep = any(
            isinstance(((case.get("expected") or {}).get("known_mechanism")), dict)
            and len((((case.get("expected") or {}).get("known_mechanism") or {}).get("steps") or [])) >= 2
            for case in cases
        )

        existing = store.list_eval_sets()
        for item in existing:
            if item.get("name") != "flower_100_default" or item.get("version") != version:
                continue
            existing_cases = store.list_eval_set_cases(str(item.get("id") or ""))
            existing_has_multistep = False
            for case in existing_cases:
                expected = case.get("expected") or {}
                known = expected.get("known_mechanism") if isinstance(expected, dict) else None
                if isinstance(known, dict) and len(known.get("steps") or []) >= 2:
                    existing_has_multistep = True
                    break
            if len(existing_cases) == expected_case_count and (not expected_has_multistep or existing_has_multistep):
                return {"eval_set_id": item["id"], "name": item["name"], "version": item["version"], "existing": True}

        eval_set_id = store.add_eval_set(
            name="flower_100_default",
            version=version,
            source_path=str(examples_path),
            sha256=None,
            cases=cases,
            active=True,
            purpose="general",
            exposed_in_ui=True,
        )
        return {"eval_set_id": eval_set_id, "name": "flower_100_default", "version": version, "case_count": len(cases)}

    @app.post("/api/convert_inputs")
    def convert_inputs(payload: ConvertInputsRequest) -> Dict[str, Any]:
        """Convert starting_materials and products from any supported format to canonical SMILES."""
        from mechanistic_agent.input_converter import ConversionResult, convert_many

        def _result_to_dict(r: ConversionResult) -> Dict[str, Any]:
            return {
                "raw_input": r.raw_input,
                "input_format": r.input_format,
                "canonical_smiles": r.canonical_smiles,
                "success": r.success,
                "error": r.error,
                "pubchem_cid": r.pubchem_cid,
            }

        sm_results = convert_many(payload.starting_materials)
        pr_results = convert_many(payload.products)
        all_success = all(r.success for r in sm_results + pr_results)
        return {
            "starting_materials": [_result_to_dict(r) for r in sm_results],
            "products": [_result_to_dict(r) for r in pr_results],
            "all_converted": all_success,
            "canonical_starting_materials": [r.canonical_smiles for r in sm_results if r.canonical_smiles],
            "canonical_products": [r.canonical_smiles for r in pr_results if r.canonical_smiles],
        }

    @app.post("/api/eval_sets/import_template")
    def import_template_eval_set(payload: ImportTemplateRequest) -> Dict[str, Any]:
        """Import a filled-in test case template as a named eval set.

        Optionally auto-converts non-SMILES input formats before persisting.
        Stores verified_mechanism steps in expected_json for comparison during
        eval runs.
        """
        from mechanistic_agent.input_converter import auto_convert

        raw_cases = list(payload.cases)

        # Support YAML text as an alternative to the cases list.
        if payload.yaml_text and not raw_cases:
            try:
                import yaml

                parsed = yaml.safe_load(payload.yaml_text)
            except ImportError:
                raise HTTPException(status_code=400, detail="PyYAML is not installed; submit cases as JSON instead")
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Failed to parse YAML: {exc}")
            if isinstance(parsed, list):
                raw_cases = parsed
            else:
                raise HTTPException(status_code=400, detail="YAML must contain a list of test cases")

        cases: List[Dict[str, Any]] = []
        conversion_warnings: List[Dict[str, Any]] = []

        for entry in raw_cases:
            if not isinstance(entry, dict):
                continue
            case_id = str(entry.get("id") or "")
            if not case_id:
                continue

            raw_starting = list(entry.get("starting_materials") or [])
            raw_products = list(entry.get("products") or [])

            if payload.auto_convert:
                starting_materials: List[str] = []
                for raw in raw_starting:
                    result = auto_convert(str(raw))
                    if result.success and result.canonical_smiles:
                        starting_materials.append(result.canonical_smiles)
                    else:
                        starting_materials.append(str(raw))
                        conversion_warnings.append({
                            "case_id": case_id,
                            "field": "starting_materials",
                            "raw": str(raw),
                            "error": result.error,
                        })
                products: List[str] = []
                for raw in raw_products:
                    result = auto_convert(str(raw))
                    if result.success and result.canonical_smiles:
                        products.append(result.canonical_smiles)
                    else:
                        products.append(str(raw))
                        conversion_warnings.append({
                            "case_id": case_id,
                            "field": "products",
                            "raw": str(raw),
                            "error": result.error,
                        })
            else:
                starting_materials = [str(s) for s in raw_starting]
                products = [str(p) for p in raw_products]

            if not starting_materials or not products:
                continue

            verified_mechanism = entry.get("verified_mechanism")
            known_mechanism = entry.get("known_mechanism")
            run_config = entry.get("run_config") or {}
            # Strip comment fields from template.
            if isinstance(run_config, dict):
                run_config = {k: v for k, v in run_config.items() if not k.startswith("_")}

            expected: Dict[str, Any] = {"products": products}
            if verified_mechanism and isinstance(verified_mechanism, dict):
                expected["verified_mechanism"] = verified_mechanism
            if known_mechanism and isinstance(known_mechanism, dict):
                expected["known_mechanism"] = known_mechanism

            cases.append({
                "case_id": case_id,
                "input": {
                    "starting_materials": starting_materials,
                    "products": products,
                    "temperature_celsius": entry.get("temperature_celsius", 25.0),
                    "ph": entry.get("ph"),
                    "run_config": run_config,
                },
                "expected": expected,
                "tags": list(entry.get("tags") or []) + [payload.name],
            })

        if not cases:
            raise HTTPException(status_code=400, detail="No valid cases found in template payload")

        eval_set_id = store.add_eval_set(
            name=payload.name,
            version=payload.version,
            source_path=None,
            sha256=None,
            cases=cases,
            active=True,
            purpose="general",
            exposed_in_ui=True,
        )
        return {
            "eval_set_id": eval_set_id,
            "name": payload.name,
            "version": payload.version,
            "case_count": len(cases),
            "conversion_warnings": conversion_warnings,
        }

    @app.get("/api/eval_sets")
    def list_eval_sets(include_hidden: bool = False) -> Dict[str, Any]:
        if include_hidden:
            return {"items": store.list_eval_sets()}
        return {"items": store.list_eval_sets(exposed_in_ui=True)}

    @app.get("/api/eval_sets/{eval_set_id}/cases")
    def list_eval_set_cases(eval_set_id: str) -> Dict[str, Any]:
        return {"items": store.list_eval_set_cases(eval_set_id)}

    @app.get("/api/traces")
    def list_traces(
        step_name: str | None = None,
        source: str | None = None,
        run_id: str | None = None,
        approved_only: bool = False,
        limit: int = 200,
    ) -> Dict[str, Any]:
        return {
            "items": store.list_trace_records(
                step_name=step_name,
                source=source,
                run_id=run_id,
                approved_only=approved_only,
                limit=limit,
            )
        }

    @app.post("/api/traces/{trace_id}/approve")
    def approve_trace(trace_id: str, payload: ApproveTraceRequest) -> Dict[str, Any]:
        ok = store.approve_trace_record(
            trace_id=trace_id,
            approved=payload.approved,
            label=payload.label,
            notes=payload.notes,
            approved_by=payload.approved_by,
        )
        if not ok:
            raise HTTPException(status_code=404, detail="Trace not found")
        item = store.get_trace_record(trace_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Trace not found after approval")
        return {"item": item}

    def _export_trace_evidence_item(trace_row: Dict[str, Any], *, created_by: str | None = None) -> Dict[str, Any]:
        if not bool(trace_row.get("approved_bool")):
            raise HTTPException(status_code=400, detail=f"Trace {trace_row.get('id')} is not approved")
        trace_id = str(trace_row.get("id") or "")
        if not trace_id:
            raise HTTPException(status_code=400, detail="Trace record is missing id")

        step_name = str(trace_row.get("step_name") or "")
        call_name = resolve_call_name_from_step(step_name)
        if not call_name:
            raise HTTPException(status_code=400, detail=f"Trace {trace_id} step '{step_name}' is not LLM-call mapped")

        prompt_version_id = trace_row.get("prompt_version_id")
        model_version_id = trace_row.get("model_version_id")
        if not prompt_version_id or not model_version_id:
            raise HTTPException(
                status_code=400,
                detail=f"Trace {trace_id} must include prompt_version_id and model_version_id",
            )

        prompt_version = store.get_prompt_version(str(prompt_version_id))
        model_version = store.get_model_version(str(model_version_id))
        if prompt_version is None:
            raise HTTPException(status_code=400, detail=f"Trace {trace_id} prompt version not found")
        if model_version is None:
            raise HTTPException(status_code=400, detail=f"Trace {trace_id} model version not found")

        required_model_keys = {
            "id",
            "resolved_model_key",
            "provider",
            "family",
            "pricing_sha256",
        }
        missing_model_keys = sorted(
            key for key in required_model_keys if not model_version.get(key)
        )
        if missing_model_keys:
            raise HTTPException(
                status_code=400,
                detail=f"Trace {trace_id} model metadata missing keys: {', '.join(missing_model_keys)}",
            )

        prompt_bundle_sha = str(
            prompt_version.get("prompt_bundle_sha256")
            or prompt_version.get("sha256")
            or ""
        )
        if not prompt_bundle_sha:
            raise HTTPException(status_code=400, detail=f"Trace {trace_id} prompt bundle hash missing")

        evidence_dir = traces_root(base) / "evidence" / call_name / prompt_bundle_sha
        evidence_dir.mkdir(parents=True, exist_ok=True)
        evidence_path = evidence_dir / f"{trace_id}.json"
        payload = {
            "trace_id": trace_id,
            "run_id": trace_row.get("run_id"),
            "step_name": step_name,
            "call_name": call_name,
            "source": trace_row.get("source"),
            "approved_bool": bool(trace_row.get("approved_bool")),
            "approval_label": trace_row.get("approval_label"),
            "approval_notes": trace_row.get("approval_notes"),
            "approved_by": trace_row.get("approved_by"),
            "approved_at": trace_row.get("approved_at"),
            "prompt_version_id": prompt_version_id,
            "model_version_id": model_version_id,
            "prompt_version": {
                "id": prompt_version.get("id"),
                "call_name": prompt_version.get("call_name"),
                "step_name": prompt_version.get("step_name"),
                "prompt_bundle_sha256": prompt_bundle_sha,
                "shared_base_sha256": prompt_version.get("shared_base_sha256"),
                "call_base_sha256": prompt_version.get("call_base_sha256"),
                "few_shot_sha256": prompt_version.get("few_shot_sha256"),
                "model_name": prompt_version.get("model_name"),
                "resolved_shared_base_path": prompt_version.get("resolved_shared_base_path"),
                "resolved_call_base_path": prompt_version.get("resolved_call_base_path"),
                "resolved_few_shot_path": prompt_version.get("resolved_few_shot_path"),
                "asset_scope": prompt_version.get("asset_scope"),
            },
            "model_version": model_version,
            "trace": trace_row.get("trace") or {},
            "exported_at": time.time(),
            "exported_by": created_by,
        }
        evidence_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return {
            "trace_id": trace_id,
            "call_name": call_name,
            "prompt_bundle_sha256": prompt_bundle_sha,
            "path": str(evidence_path.resolve().relative_to(base)),
        }

    @app.post("/api/traces/export_evidence")
    def export_trace_evidence(payload: TraceEvidenceExportRequest) -> Dict[str, Any]:
        if not payload.trace_ids:
            raise HTTPException(status_code=400, detail="trace_ids cannot be empty")
        exported: List[Dict[str, Any]] = []
        for trace_id in payload.trace_ids:
            row = store.get_trace_record(trace_id)
            if row is None:
                raise HTTPException(status_code=404, detail=f"Trace not found: {trace_id}")
            exported.append(_export_trace_evidence_item(row, created_by=payload.created_by))
        return {"items": exported}

    @app.post("/api/traces/import_baseline")
    def import_baseline_traces(payload: Dict[str, Any]) -> Dict[str, Any]:
        traces = payload.get("traces")
        if not isinstance(traces, list):
            raise HTTPException(status_code=400, detail="Payload must include traces[]")
        imported = 0
        for item in traces:
            if not isinstance(item, dict):
                continue
            step_name = str(item.get("step_name") or "")
            trace = item.get("trace")
            if not step_name or not isinstance(trace, dict):
                continue
            store.add_trace_record(
                step_name=step_name,
                trace=trace,
                source=str(item.get("source") or "baseline"),
                model=item.get("model"),
                reasoning_level=item.get("reasoning_level"),
                model_version_id=store.upsert_model_version(
                    model_name=str(item.get("model") or ""),
                    reasoning_level=str(item.get("reasoning_level") or "") or None,
                )
                if item.get("model")
                else None,
                score=float(item["score"]) if isinstance(item.get("score"), (int, float)) else None,
                approved=bool(item.get("approved", True)),
                approval_label=str(item.get("approval_label") or "baseline"),
                approval_notes=str(item.get("approval_notes") or "imported_baseline"),
                approved_by=item.get("approved_by"),
                approved_at=float(item["approved_at"]) if isinstance(item.get("approved_at"), (int, float)) else None,
                actor_id=item.get("actor_id"),
            )
            imported += 1
        return {"imported": imported}

    @app.post("/api/curation/export")
    def export_curation(payload: CurationExportRequest) -> Dict[str, Any]:
        export_root = base / "data" / "curation_exports"
        export_root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = export_root / f"export_{stamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        files: List[str] = []
        hashed_files: List[Dict[str, Any]] = []
        manifest: Dict[str, Any] = {
            "created_at": time.time(),
            "eval_set_id": payload.eval_set_id,
            "files": [],
        }

        def _add_file(path: Path) -> None:
            rel = str(path.resolve().relative_to(base))
            files.append(rel)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hashed_files.append({"path": rel, "sha256": digest})

        if payload.include_few_shot:
            few_shot_rows = store.list_few_shot_examples(approved_only=True, limit=5000)
            grouped: Dict[str, List[Dict[str, Any]]] = {}
            for row in few_shot_rows:
                step_name = str(row.get("step_name") or "").strip()
                call_name = resolve_call_name_from_step(step_name)
                if not call_name:
                    continue
                grouped.setdefault(call_name, []).append(row)
            examples_dir = run_dir / "prompt_versions_calls"
            examples_dir.mkdir(parents=True, exist_ok=True)
            for call_name, rows in grouped.items():
                path = examples_dir / call_name / "few_shot.jsonl"
                repo_path = base / "prompt_versions" / "calls" / call_name / "few_shot.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                repo_path.parent.mkdir(parents=True, exist_ok=True)
                lines = []
                for row in rows:
                    lines.append(
                        json.dumps(
                            {
                                "input": row.get("input_text"),
                                "output": row.get("output_text"),
                            },
                            sort_keys=True,
                        )
                    )
                content = "\n".join(lines) + ("\n" if lines else "")
                path.write_text(content, encoding="utf-8")
                repo_path.write_text(content, encoding="utf-8")
                _add_file(path)
                _add_file(repo_path)

        if payload.include_baselines:
            baselines = store.list_trace_records(source="baseline", approved_only=True, limit=5000)
            baseline_dir = run_dir / "baselines"
            baseline_dir.mkdir(parents=True, exist_ok=True)
            baseline_path = baseline_dir / "baseline_traces.json"
            baseline_payload = {"items": baselines}
            baseline_path.write_text(json.dumps(baseline_payload, indent=2, sort_keys=True), encoding="utf-8")
            repo_baseline_dir = base / "data" / "baselines" / (payload.eval_set_id or "default")
            repo_baseline_dir.mkdir(parents=True, exist_ok=True)
            repo_baseline_path = repo_baseline_dir / f"{stamp}.json"
            repo_baseline_path.write_text(json.dumps(baseline_payload, indent=2, sort_keys=True), encoding="utf-8")
            _add_file(baseline_path)
            _add_file(repo_baseline_path)

        if payload.include_leaderboard and payload.eval_set_id:
            leaderboard_rows = store.leaderboard(payload.eval_set_id, limit=100)
            lb_path = run_dir / "leaderboard.json"
            lb_path.write_text(json.dumps({"items": leaderboard_rows}, indent=2, sort_keys=True), encoding="utf-8")
            _add_file(lb_path)

        manifest["files"] = hashed_files
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        _add_file(manifest_path)

        export_id = store.record_curation_export(
            export_type="phase6_bundle",
            path=str(run_dir.resolve()),
            manifest=manifest,
            created_by=payload.created_by,
        )
        return {
            "id": export_id,
            "path": str(run_dir.resolve()),
            "files": files,
            "manifest": manifest,
        }

    @app.get("/api/curation/exports")
    def list_curation_exports(export_type: str | None = None, limit: int = 50) -> Dict[str, Any]:
        return {"items": store.list_curation_exports(export_type=export_type, limit=limit)}

    @app.get("/api/few_shot")
    def list_few_shot(step_name: str | None = None, approved_only: bool = True, limit: int = 200) -> Dict[str, Any]:
        return {
            "items": store.list_few_shot_examples(
                step_name=step_name,
                approved_only=approved_only,
                limit=limit,
            )
        }

    @app.post("/api/few_shot/from_trace")
    def create_few_shot_from_trace(payload: TraceToFewShotRequest) -> Dict[str, Any]:
        selected = store.get_trace_record(payload.trace_id)
        if selected is None:
            raise HTTPException(status_code=404, detail="Trace record not found")
        step_name = str(selected.get("step_name") or "").strip()
        if not step_name:
            raise HTTPException(status_code=400, detail="Trace record has no step_name")
        call_name = resolve_call_name_from_step(step_name)
        if not call_name:
            raise HTTPException(status_code=400, detail=f"No call mapping for step '{step_name}'")

        trace = selected.get("trace") or {}
        input_text = json.dumps(
            {
                "step_name": step_name,
                "tool_name": trace.get("tool_name"),
                "attempt": trace.get("attempt"),
            },
            sort_keys=True,
        )
        output_text = json.dumps(
            {
                "output": trace.get("output"),
                "validation": trace.get("validation"),
            },
            sort_keys=True,
        )
        example_key = payload.example_key or f"{selected.get('id')}"
        example_id = store.add_few_shot_example(
            step_name=step_name,
            example_key=example_key,
            input_text=input_text,
            output_text=output_text,
            approved=payload.approved,
            source_trace_id=str(selected.get("id")),
            score=float(selected["score"]) if isinstance(selected.get("score"), (int, float)) else None,
            prompt_version_id=selected.get("prompt_version_id"),
        )
        target_path = append_call_few_shot_example(
            call_name,
            input_text=input_text,
            output_text=output_text,
            base_dir=base,
        )
        return {
            "id": example_id,
            "step_name": step_name,
            "call_name": call_name,
            "example_key": example_key,
            "path": str(target_path.resolve().relative_to(base)),
        }

    @app.post("/api/evals/runset")
    def run_eval_set(payload: EvalRunSetRequest) -> Dict[str, Any]:
        eval_set = store.get_eval_set(payload.eval_set_id)
        if eval_set is None:
            raise HTTPException(status_code=404, detail="Eval set not found")
        if _is_leaderboard_holdout_eval_set(eval_set):
            raise HTTPException(
                status_code=403,
                detail="leaderboard_holdout eval sets are restricted to /api/evals/official-runset",
            )
        harness_bundle_hash = registry.bundle_hashes().get("prompt_bundle_hash")
        model_name = _resolve_model_name(payload.model_name, payload.model)
        thinking_level = _resolve_public_thinking_level(payload.thinking_level, payload.reasoning_level)
        eval_run_id = store.create_eval_run(
            eval_set_id=payload.eval_set_id,
            run_group_name=payload.run_group_name,
            model=model_name,
            model_name=model_name,
            model_family=get_model_family(model_name),
            thinking_level=thinking_level,
            harness_bundle_hash=harness_bundle_hash,
            status="running",
        )
        if payload.async_mode:
            eval_executor.start(eval_run_id, _execute_eval_runset, eval_run_id, payload)
            return {"eval_run_id": eval_run_id, "status": "running", "async_mode": True}

        _execute_eval_runset(eval_run_id, payload)
        row = store.get_eval_run(eval_run_id) or {}
        return {"eval_run_id": eval_run_id, "status": row.get("status", "unknown"), "async_mode": False}

    def _resolve_official_holdout_eval_set_id(requested_eval_set_id: Optional[str]) -> str:
        if requested_eval_set_id:
            row = store.get_eval_set(requested_eval_set_id)
            if row is None:
                raise HTTPException(status_code=404, detail="Eval set not found")
            if not _is_leaderboard_holdout_eval_set(row):
                raise HTTPException(status_code=403, detail="Eval set is not a leaderboard_holdout suite")
            return str(row.get("id") or "")

        holdouts = store.list_eval_sets(purpose="leaderboard_holdout")
        if not holdouts:
            raise HTTPException(status_code=404, detail="No leaderboard_holdout eval set found")
        return str(holdouts[0].get("id") or "")

    @app.post("/api/evals/official-runset")
    def run_official_eval_set(payload: OfficialEvalRunSetRequest) -> Dict[str, Any]:
        eval_set_id = _resolve_official_holdout_eval_set_id(payload.eval_set_id)
        translated = EvalRunSetRequest(
            eval_set_id=eval_set_id,
            run_group_name=payload.run_group_name or "official_holdout_harness",
            case_ids=list(payload.case_ids or []),
            model_name=payload.model_name,
            model=payload.model,
            thinking_level=payload.thinking_level,
            reasoning_level=payload.reasoning_level,
            mode=payload.mode,
            max_cases=payload.max_cases,
            max_steps=payload.max_steps,
            max_runtime_seconds=payload.max_runtime_seconds,
            async_mode=payload.async_mode,
        )
        harness_bundle_hash = registry.bundle_hashes().get("prompt_bundle_hash")
        model_name = _resolve_model_name(translated.model_name, translated.model)
        thinking_level = _resolve_public_thinking_level(translated.thinking_level, translated.reasoning_level)
        eval_run_id = store.create_eval_run(
            eval_set_id=eval_set_id,
            run_group_name=translated.run_group_name,
            model=model_name,
            model_name=model_name,
            model_family=get_model_family(model_name),
            thinking_level=thinking_level,
            harness_bundle_hash=harness_bundle_hash,
            status="running",
        )
        if translated.async_mode:
            eval_executor.start(eval_run_id, _execute_eval_runset, eval_run_id, translated)
            return {"eval_run_id": eval_run_id, "status": "running", "async_mode": True}

        _execute_eval_runset(eval_run_id, translated)
        row = store.get_eval_run(eval_run_id) or {}
        return {"eval_run_id": eval_run_id, "status": row.get("status", "unknown"), "async_mode": False}

    @app.get("/api/evals/runs")
    def list_eval_runs(eval_set_id: str | None = None) -> Dict[str, Any]:
        return {"items": store.list_eval_runs(eval_set_id=eval_set_id)}

    @app.get("/api/evals/runs/{eval_run_id}")
    def get_eval_run(eval_run_id: str) -> Dict[str, Any]:
        row = store.get_eval_run(eval_run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Eval run not found")
        results = store.list_eval_run_results(eval_run_id)
        return {
            **row,
            "results_count": len(results),
            "results": results,
        }

    @app.get("/api/evals/runs/{eval_run_id}/results")
    def get_eval_run_results(eval_run_id: str) -> Dict[str, Any]:
        row = store.get_eval_run(eval_run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Eval run not found")
        return {"items": store.list_eval_run_results(eval_run_id)}

    @app.get("/api/evals/leaderboard")
    def eval_leaderboard(eval_set_id: str, limit: int = 20) -> Dict[str, Any]:
        items = store.leaderboard(eval_set_id=eval_set_id, limit=max(1, min(limit, 100)))
        exports = store.list_curation_exports(export_type="phase6_bundle", limit=20)
        return {"items": items, "evidence_exports": exports}

    @app.get("/api/evals/leaderboard/official")
    def official_eval_leaderboard(limit: int = 20) -> Dict[str, Any]:
        holdouts = store.list_eval_sets(purpose="leaderboard_holdout")
        if not holdouts:
            return {"eval_set_id": None, "items": [], "evidence_exports": []}
        eval_set_id = str(holdouts[0].get("id") or "")
        items = store.leaderboard(eval_set_id=eval_set_id, limit=max(1, min(limit, 100)))
        exports = store.list_curation_exports(export_type="phase6_bundle", limit=20)
        return {"eval_set_id": eval_set_id, "items": items, "evidence_exports": exports}

    # ---- Baseline (harness-free) evaluation ----

    def _run_baseline_eval_set(payload: BaselineEvalRunSetRequest) -> Dict[str, Any]:
        """Execute harness-free single-shot evaluation for an eval set."""
        from mechanistic_agent.core.baseline_runner import (
            BASELINE_GROUP_PREFIX,
            BaselineRunner,
            score_baseline_result,
        )

        eval_set = store.get_eval_set(payload.eval_set_id)
        if eval_set is None:
            raise HTTPException(status_code=404, detail="Eval set not found")
        if _is_leaderboard_holdout_eval_set(eval_set):
            raise HTTPException(
                status_code=403,
                detail="leaderboard_holdout eval sets are restricted to /api/evals/official-runset",
            )

        model = payload.model or payload.model_name or get_default_model()
        model_family = get_model_family(model) or "openai"
        thinking_level = payload.thinking_level
        run_group = payload.run_group_name or BASELINE_GROUP_PREFIX

        harness_bundle_hash = registry.bundle_hashes().get("prompt_bundle_hash", "")
        eval_run_id = store.create_eval_run(
            eval_set_id=payload.eval_set_id,
            run_group_name=run_group,
            model=model,
            model_name=model,
            model_family=model_family,
            thinking_level=thinking_level,
            harness_bundle_hash=harness_bundle_hash,
            status="running",
        )

        cases = store.list_eval_set_cases(payload.eval_set_id)
        if payload.case_ids:
            cases = [c for c in cases if str(c.get("case_id") or "") in set(payload.case_ids)]
        if payload.step_count is not None and not payload.case_ids:
            requested_steps = int(payload.step_count)
            cases = [c for c in cases if int(_eval_case_step_count(c) or 0) == requested_steps]
        if payload.tier_name and not payload.case_ids and payload.step_count is None:
            tier_ids = _load_eval_tier_ids(base).get(payload.tier_name, [])
            if tier_ids:
                cases = [c for c in cases if str(c.get("case_id") or "") in set(tier_ids)]
        if len(cases) > payload.max_cases:
            cases = cases[: payload.max_cases]

        runner = BaselineRunner()
        completed = 0
        failed = 0

        for case in cases:
            case_id = str(case.get("case_id") or "")
            input_payload = case.get("input") or {}
            starting = [str(s) for s in input_payload.get("starting_materials", [])]
            products = [str(p) for p in input_payload.get("products", [])]
            if not starting or not products:
                continue

            expected = case.get("expected") or {}
            if not isinstance(expected, dict):
                expected = {}
            case_step_count = _eval_case_step_count(case)

            try:
                result = runner.run_case(
                    starting_materials=starting,
                    products=products,
                    model=model,
                    thinking_level=thinking_level,
                    timeout=payload.timeout_seconds,
                )
                graded = score_baseline_result(result, expected if expected else None)
                score = float(graded["score"])
                passed = bool(graded["passed"])
                latency_ms = float(result.get("latency_ms") or 0.0)
                summary: Dict[str, Any] = {
                    "score": score,
                    "passed": passed,
                    "step_count": graded.get("step_count"),
                    "n_mechanistic_steps": case_step_count,
                    "mechanism_type": graded.get("mechanism_type"),
                    "scoring_breakdown": graded.get("scoring_breakdown", {}),
                    "error": graded.get("error"),
                    "eval_mode": "baseline",
                    # Baseline has no per-subagent breakdown; expose single entry.
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
                    case_id=case_id or str(uuid.uuid4().hex),
                    run_id=None,
                    score=score,
                    passed=passed,
                    cost={},
                    latency_ms=latency_ms,
                    summary=summary,
                )
                completed += 1
            except Exception as exc:
                store.record_eval_run_result(
                    eval_run_id=eval_run_id,
                    case_id=case_id or str(uuid.uuid4().hex),
                    run_id=None,
                    score=0.0,
                    passed=False,
                    cost={},
                    latency_ms=0.0,
                    summary={"error": str(exc), "eval_mode": "baseline"},
                )
                failed += 1

        store.set_eval_run_status(eval_run_id, "completed")
        return {
            "eval_run_id": eval_run_id,
            "status": "completed",
            "completed": completed,
            "failed": failed,
        }

    @app.post("/api/evals/baseline-runset")
    def run_baseline_eval_set(payload: BaselineEvalRunSetRequest) -> Dict[str, Any]:
        """Run harness-free single-shot mechanism evaluation on an eval set.

        Results are stored and visible on the leaderboard alongside harness runs,
        clearly labelled as baseline.
        """
        eval_set = store.get_eval_set(payload.eval_set_id)
        if eval_set is None:
            raise HTTPException(status_code=404, detail="Eval set not found")
        if _is_leaderboard_holdout_eval_set(eval_set):
            raise HTTPException(
                status_code=403,
                detail="leaderboard_holdout eval sets are restricted to /api/evals/official-runset",
            )

        if payload.async_mode:
            def _bg():
                try:
                    _run_baseline_eval_set(payload)
                except Exception:
                    pass

            eval_executor.start(str(uuid.uuid4()), _bg)
            return {"status": "started", "note": "Baseline eval running in background."}
        return _run_baseline_eval_set(payload)

    @app.post("/api/evals/seed-simulated-leaderboard")
    def seed_simulated_leaderboard(payload: SeedSimulatedLeaderboardRequest) -> Dict[str, Any]:
        """Insert clearly-labelled SIMULATED placeholder rows into the leaderboard.

        Use this to populate the UI for screenshots and design review before real
        eval runs complete.  All inserted rows have ``[SIMULATED]`` in their
        run_group_name and should be deleted once real data is available.
        """
        return store.seed_simulated_leaderboard(
            eval_set_id=payload.eval_set_id,
            case_count=payload.case_count,
        )

    @app.delete("/api/evals/seed-simulated-leaderboard")
    def delete_simulated_leaderboard(eval_set_id: str) -> Dict[str, Any]:
        """Delete all SIMULATED placeholder leaderboard rows for an eval set."""
        return store.delete_simulated_leaderboard_rows(eval_set_id=eval_set_id)

    @app.get("/api/evals/tiers")
    def eval_tiers(eval_set_id: str | None = None) -> Dict[str, Any]:
        eval_sets = store.list_eval_sets(exposed_in_ui=True)
        selected_eval_set_id = eval_set_id or (str(eval_sets[0].get("id")) if eval_sets else "")
        tiers = _load_eval_tier_ids(base)
        if not selected_eval_set_id:
            return {"eval_set_id": None, "tiers": tiers, "step_buckets": {}, "cases": []}
        cases = store.list_eval_set_cases(selected_eval_set_id)
        case_payload: List[Dict[str, Any]] = []
        step_buckets: Dict[str, List[str]] = {}
        for case in cases:
            expected = case.get("expected") or {}
            known = _normalize_expected_known(expected if isinstance(expected, dict) else {})
            min_steps = None
            if isinstance(known, dict):
                if isinstance(known.get("min_steps"), int):
                    min_steps = int(known.get("min_steps"))
                elif isinstance(known.get("steps"), list):
                    min_steps = len(known.get("steps"))
            if isinstance(min_steps, int) and min_steps > 0:
                step_buckets.setdefault(str(min_steps), []).append(str(case.get("case_id") or ""))
            case_payload.append(
                {
                    "case_id": case.get("case_id"),
                    "min_steps": min_steps,
                    "starting_materials": list((case.get("input") or {}).get("starting_materials") or []),
                    "products": list((case.get("input") or {}).get("products") or []),
                    "known_mechanism": known if isinstance(known, dict) else None,
                }
            )
        return {
            "eval_set_id": selected_eval_set_id,
            "tiers": tiers,
            "step_buckets": step_buckets,
            "cases": case_payload,
        }

    # ---- Verification endpoints ----

    verification_executor = ThreadJobExecutor()

    def _run_eval_set_score(
        eval_set_id: str,
        step_models: Dict[str, str],
        step_reasoning: Dict[str, str],
    ) -> float:
        """Run an eval set with specific model overrides and return aggregate score."""
        from mechanistic_agent.api.schemas import EvalRunSetRequest

        selected_model = next(iter(step_models.values()), get_default_model())
        model_family = get_model_family(selected_model) or "openai"
        selected_thinking = to_public_reasoning_level(next(iter(step_reasoning.values()), None))

        harness_bundle_hash = registry.bundle_hashes().get("prompt_bundle_hash")
        eval_run_id = store.create_eval_run(
            eval_set_id=eval_set_id,
            run_group_name=f"verification_{model_family}",
            model=selected_model,
            model_name=selected_model,
            model_family=model_family,
            thinking_level=selected_thinking,
            harness_bundle_hash=harness_bundle_hash,
            status="running",
        )

        cases = store.list_eval_set_cases(eval_set_id)
        if not cases:
            store.set_eval_run_status(eval_run_id, "failed")
            return 0.0

        total_score = 0.0
        count = 0

        for case in cases:
            case_id = str(case.get("case_id") or "")
            input_payload = case.get("input") or {}
            starting = [str(s) for s in input_payload.get("starting_materials", [])]
            products = [str(p) for p in input_payload.get("products", [])]
            if not starting or not products:
                continue

            created_at = time.time()
            # Verification uses a uniform exact model, derived from the provided step map.
            step_model_name = next(iter(step_models.values()), selected_model)
            run_id, _, _ = _create_run_internal(
                mode="unverified",
                starting_materials=starting,
                products=products,
                example_id=None,
                temperature_celsius=float(input_payload.get("temperature_celsius", 25.0)),
                ph=input_payload.get("ph"),
                model_name=step_model_name,
                thinking_level=selected_thinking,
                api_keys={},
                optional_llm_tools=["attempt_atom_mapping", "predict_missing_reagents"],
                functional_groups_enabled=True,
                intermediate_prediction_enabled=True,
                max_steps=6,
                max_runtime_seconds=180.0,
                retry_same_candidate_max=1,
                max_reproposals_per_step=4,
                reproposal_on_repeat_failure=True,
                candidate_rescue_enabled=True,
                step_mapping_enabled=True,
                arrow_push_annotation_enabled=True,
                dbe_policy="soft",
                reaction_template_policy="auto",
                reaction_template_confidence_threshold=0.65,
                reaction_template_margin_threshold=0.10,
            )
            manager.start(run_id)
            snapshot = _wait_for_run(run_id, timeout_seconds=210.0)
            expected = case.get("expected") or {}
            graded = _grade_eval_snapshot(snapshot, expected if isinstance(expected, dict) else {})

            score = float(graded["score"])
            total_score += score
            count += 1

            from mechanistic_agent.scoring import score_subagents_from_step_outputs

            subagent_scores = score_subagents_from_step_outputs(
                list(snapshot.get("step_outputs") or [])
            )
            store.record_eval_run_result(
                eval_run_id=eval_run_id,
                case_id=case_id or run_id,
                run_id=run_id,
                score=score,
                passed=bool(graded["passed"]),
                cost={"total_cost": 0.0},
                latency_ms=max((time.time() - created_at) * 1000.0, 0.0),
                summary={
                    **graded["summary"],
                    "selected_step_models": step_models,
                    "scoring_breakdown": graded.get("scoring_breakdown", {}),
                    "subagent_scores": subagent_scores,
                },
            )

        store.set_eval_run_status(eval_run_id, "completed")
        return total_score / count if count > 0 else 0.0

    def _execute_verification_job(job_id: str, eval_set_id: str, model_family: str) -> None:
        from mechanistic_agent.core.verification_runner import VerificationRunner

        try:
            store.update_verification_job_progress(
                job_id, status="running", progress={"phase": "starting"}
            )

            harness_ver = registry.harness_version()
            runner = VerificationRunner(
                run_eval_fn=_run_eval_set_score,
                store=store,
                harness_version=harness_ver,
            )

            def on_progress(info: Dict[str, Any]) -> None:
                store.update_verification_job_progress(
                    job_id, status="running", progress=info
                )

            result = runner.run_verification(
                eval_set_id=eval_set_id,
                model_family=model_family,
                progress_callback=on_progress,
            )

            store.complete_verification_job(
                job_id,
                status="completed",
                result={
                    "baseline_score": result.baseline_score,
                    "step_results": [
                        {
                            "step_name": sr.step_name,
                            "verified_model": sr.verified_model,
                            "verified_reasoning": sr.verified_reasoning,
                            "score": sr.score,
                        }
                        for sr in result.step_results
                    ],
                },
            )
        except Exception as exc:
            store.complete_verification_job(
                job_id, status="failed", result={"error": str(exc)}
            )

    @app.post("/api/verification/start")
    def start_verification(payload: StartVerificationRequest) -> Dict[str, Any]:
        eval_set = next(
            (row for row in store.list_eval_sets() if row.get("id") == payload.eval_set_id),
            None,
        )
        if eval_set is None:
            raise HTTPException(status_code=404, detail="Eval set not found")

        harness_ver = registry.harness_version()
        job_id = store.create_verification_job(
            model_family=payload.model_family,
            eval_set_id=payload.eval_set_id,
            harness_version=harness_ver,
        )
        verification_executor.start(
            job_id, _execute_verification_job, job_id, payload.eval_set_id, payload.model_family
        )
        return {"job_id": job_id, "status": "pending", "harness_version": harness_ver}

    @app.get("/api/verification/jobs")
    def list_verification_jobs() -> Dict[str, Any]:
        return {"items": store.list_verification_jobs()}

    @app.get("/api/verification/jobs/{job_id}")
    def get_verification_job(job_id: str) -> Dict[str, Any]:
        job = store.get_verification_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Verification job not found")
        return job

    @app.get("/api/verification/results")
    def get_verification_results(
        harness_version: str | None = None,
        model_family: str | None = None,
    ) -> Dict[str, Any]:
        version = harness_version or registry.harness_version()
        family = model_family or "openai"
        results = store.get_verified_step_models(
            harness_version=version, model_family=family
        )
        return {
            "harness_version": version,
            "model_family": family,
            "step_models": results,
        }

    @app.get("/api/verification/history")
    def get_verification_history(
        model_family: str | None = None, limit: int = 100
    ) -> Dict[str, Any]:
        items = store.list_verification_history(
            model_family=model_family, limit=max(1, min(limit, 500))
        )
        return {"items": items}

    @app.get("/api/examples")
    def list_examples() -> List[Dict[str, Any]]:
        results_by_id: Dict[str, Dict[str, Any]] = {}
        training_dir = base / "training_data"
        if training_dir.exists():
            preferred_names = {"flower_mechanisms_100.json": 0, "eval_set.json": 1}
            candidate_files = sorted(
                training_dir.glob("*.json"),
                key=lambda path: (preferred_names.get(path.name, 10), path.name),
            )
            for json_file in candidate_files:
                try:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                    if not isinstance(data, list):
                        continue
                    source_label = "FlowER 100" if json_file.name in preferred_names else json_file.stem
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        prepared = _prepare_example_record(item, source_label)
                        if not prepared:
                            continue
                        item_id = str(prepared.get("id") or "").strip()
                        if item_id in results_by_id:
                            continue
                        results_by_id[item_id] = prepared
                except json.JSONDecodeError:
                    pass

        for eval_set in store.list_eval_sets(exposed_in_ui=True):
            eval_set_id = str(eval_set.get("id") or "")
            if not eval_set_id:
                continue
            source_label = f"Eval set: {eval_set.get('name') or eval_set_id}"
            for case in store.list_eval_set_cases(eval_set_id):
                input_payload = case.get("input") or {}
                expected = case.get("expected") or {}
                item = {
                    "id": str(case.get("case_id") or ""),
                    "name": str(case.get("case_id") or ""),
                    "starting_materials": list(input_payload.get("starting_materials") or []),
                    "products": list(input_payload.get("products") or expected.get("products") or []),
                    "temperature_celsius": input_payload.get("temperature_celsius"),
                    "ph": input_payload.get("ph"),
                    "source": source_label,
                    "known_mechanism": expected.get("known_mechanism") if isinstance(expected.get("known_mechanism"), dict) else None,
                    "verified_mechanism": expected.get("verified_mechanism") if isinstance(expected.get("verified_mechanism"), dict) else None,
                }
                prepared = _prepare_example_record(item, source_label)
                if not prepared:
                    continue
                item_id = str(prepared.get("id") or "")
                if item_id in results_by_id:
                    continue
                results_by_id[item_id] = prepared

        results = list(results_by_id.values())
        results.sort(
            key=lambda row: (
                int(row.get("n_mechanistic_steps") or len((((row.get("verified_mechanism") or {}).get("steps")) or [])) or 0),
                str(row.get("name") or ""),
            )
        )
        return results

    @app.post("/api/parse_smirks")
    def parse_smirks(payload: Dict[str, Any]) -> Dict[str, Any]:
        raw = payload.get("smirks")
        if not isinstance(raw, str):
            raise HTTPException(status_code=400, detail="smirks must be a string")
        try:
            parsed = _parse_smirks_text(raw)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return parsed

    @app.post("/api/molecules/render")
    def render_molecules(payload: Dict[str, Any]) -> Dict[str, Any]:
        smiles = payload.get("smiles") or []
        if not isinstance(smiles, list):
            raise HTTPException(status_code=400, detail="smiles must be a list")
        show_atom_numbers = bool(payload.get("show_atom_numbers", False))
        cards = [_render_molecule(str(item), show_atom_numbers=show_atom_numbers) for item in smiles]
        return {"items": cards}

    @app.post("/api/runs/{run_id}/evaluate")
    def evaluate_run_endpoint(run_id: str, payload: EvaluateRunRequest) -> Dict[str, Any]:
        snapshot = store.get_run_snapshot(run_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if snapshot.get("status") not in TERMINAL_STATUSES:
            raise HTTPException(status_code=400, detail="Run must finish before evaluation")
        expected = _find_known_expected_for_snapshot(store, snapshot) or {}
        graded = _grade_eval_snapshot(snapshot, expected)
        breakdown = graded.get("scoring_breakdown", {})
        evaluation = {
            "type": "deterministic_known_mechanism",
            "judge_model": payload.judge_model,
            "overall_feedback": "Deterministic scoring against established benchmark mechanism.",
            "steps": breakdown.get("step_breakdown", []),
            "save_record": {
                "score": graded.get("score", 0.0),
                "step_scores": {
                    "validity": breakdown.get("step_validity_component", 0.0),
                    "alignment": breakdown.get("known_alignment_component", 0.0),
                    "final_product": breakdown.get("final_product_component", 0.0),
                    "penalty": breakdown.get("efficiency_penalty_total", 0.0),
                },
                "summary": graded.get("summary", {}),
                "known_answer_available": bool(_normalize_expected_known(expected)),
            },
            "harness_recommendations": [],
            "known_answer_comparison": breakdown,
        }
        store.record_evaluation(
            run_id=run_id,
            judge_model=payload.judge_model,
            score=float(evaluation.get("save_record", {}).get("score"))
            if isinstance(evaluation.get("save_record"), dict)
            and isinstance(evaluation.get("save_record", {}).get("score"), (int, float))
            else None,
            summary=evaluation,
        )
        store.append_event(
            run_id,
            "evaluation_completed",
            {"judge_model": payload.judge_model, "score": evaluation.get("save_record", {}).get("score")},
        )
        return {"run_id": run_id, "evaluation": evaluation}

    @app.post("/api/runs/{run_id}/evaluation/save")
    def save_evaluation_endpoint(run_id: str, payload: SaveEvaluationRequest) -> Dict[str, Any]:
        snapshot = store.get_run_snapshot(run_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="Run not found")

        evaluation = payload.evaluation or store.get_latest_evaluation(run_id)
        if not evaluation:
            raise HTTPException(status_code=400, detail="No evaluation payload available")
        if isinstance(evaluation, dict) and isinstance(evaluation.get("summary"), dict):
            evaluation = evaluation.get("summary", {})

        config = snapshot.get("config", {})
        input_payload = snapshot.get("input_payload", {})
        score_value = None
        if isinstance(evaluation, dict):
            save_record = evaluation.get("save_record")
            if isinstance(save_record, dict) and isinstance(save_record.get("score"), (int, float)):
                score_value = float(save_record.get("score"))

        record = {
            "recorded_at": time.time(),
            "run_id": run_id,
            "judge_model": payload.judge_model,
            "score": score_value,
            "model": config.get("model"),
            "model_family": config.get("model_family"),
            "starting_materials": list(input_payload.get("starting_materials") or []),
            "products": list(input_payload.get("products") or []),
            "evaluation": evaluation,
        }
        record_path = base / "data" / "reaction_test_results.json"
        container = {"entries": []}
        if record_path.exists():
            try:
                existing = json.loads(record_path.read_text(encoding="utf-8"))
                if isinstance(existing, dict) and isinstance(existing.get("entries"), list):
                    container = existing
            except Exception:
                container = {"entries": []}
        container["entries"].append(record)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(container, indent=2, sort_keys=True), encoding="utf-8")
        store.append_event(run_id, "evaluation_saved", {"judge_model": payload.judge_model})
        return {"run_id": run_id, "record": record}

    @app.post("/api/runs/{run_id}/harness/apply")
    def apply_harness_update(run_id: str, payload: HarnessApplyRequest) -> Dict[str, Any]:
        snapshot = store.get_run_snapshot(run_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="Run not found")
        run_config = snapshot.get("config") if isinstance(snapshot.get("config"), dict) else {}
        is_dry_run = bool(run_config.get("dry_run"))

        recommendation = (payload.recommendation or "").strip()
        if not recommendation:
            latest_eval = store.get_latest_evaluation(run_id)
            if isinstance(latest_eval, dict):
                summary = latest_eval.get("summary") if isinstance(latest_eval.get("summary"), dict) else latest_eval
                candidate = summary.get("harness_recommendations") if isinstance(summary, dict) else None
                if isinstance(candidate, list):
                    recommendation = "\n".join(str(item) for item in candidate if str(item).strip())
                elif isinstance(candidate, str):
                    recommendation = candidate.strip()
        if not recommendation:
            raise HTTPException(status_code=400, detail="No harness recommendation provided or found")

        try:
            call_name = normalize_call_name(payload.call_name)
            prompt_info = get_call_prompt_version(call_name, base)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if payload.component == "base":
            prompt_path = Path(str(prompt_info.get("call_base_path") or ""))
            existing = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""
            update_block = (
                "## Local Harness Update\n\n"
                f"- Source run: `{run_id}`\n"
                f"- Applied at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"- Recommendation:\n\n{recommendation.strip()}\n"
            )
            try:
                new_content = replace_prompt_in_skill_md(
                    existing,
                    prompt_text=update_block if payload.append_mode else recommendation,
                    append_mode=payload.append_mode,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            diff_text = unified_prompt_diff(
                existing,
                new_content,
                path=str(prompt_path.resolve().relative_to(base)),
            )
            if is_dry_run:
                preview_dir = traces_root(base) / "dry_run_prompt_updates" / run_id
                preview_dir.mkdir(parents=True, exist_ok=True)
                preview_path = preview_dir / f"{call_name}_SKILL.md"
                diff_path = preview_dir / f"{call_name}.diff"
                preview_path.write_text(new_content, encoding="utf-8")
                diff_path.write_text(diff_text, encoding="utf-8")
                prompt_path_for_response = preview_path
            else:
                prompt_path.parent.mkdir(parents=True, exist_ok=True)
                prompt_path.write_text(new_content, encoding="utf-8")
                prompt_path_for_response = prompt_path
        else:
            prompt_path = Path(str(prompt_info.get("few_shot_path") or ""))
            try:
                parsed = json.loads(recommendation)
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=400,
                    detail="few_shot component requires recommendation as JSON with input/output fields",
                ) from exc
            if not isinstance(parsed, dict) or not isinstance(parsed.get("input"), str) or not isinstance(parsed.get("output"), str):
                raise HTTPException(
                    status_code=400,
                    detail="few_shot recommendation JSON must contain string fields: input and output",
                )
            append_call_few_shot_example(
                call_name,
                input_text=parsed["input"],
                output_text=parsed["output"],
                base_dir=base,
            )
        store.append_event(
            run_id,
            "harness_updated",
            {
                "call_name": call_name,
                "component": payload.component,
                "append_mode": payload.append_mode,
            },
        )
        return {
            "run_id": run_id,
            "call_name": call_name,
            "component": payload.component,
            "path": str(prompt_path_for_response.resolve().relative_to(base)) if payload.component == "base" else str(prompt_path.resolve().relative_to(base)),
            "message": "Dry-run prompt update preview saved." if is_dry_run and payload.component == "base" else "Harness prompt updated locally.",
            "recommendation": recommendation,
            **(
                {
                    "diff": diff_text,
                    "diff_path": str(diff_path.resolve().relative_to(base)),
                    "dry_run": True,
                }
                if is_dry_run and payload.component == "base"
                else {}
            ),
        }

    def _generate_pr_timing_summary(evidence_trace_ids: Dict[str, List[str]], call_names: List[str]) -> str:
        """Generate a timing summary for PR description from evidence traces."""
        if not evidence_trace_ids:
            return ""

        timing_data = []
        for call_name in call_names:
            trace_ids = evidence_trace_ids.get(call_name, [])
            for trace_id in trace_ids:
                trace_row = store.get_trace_record(trace_id)
                if trace_row and trace_row.get("trace"):
                    trace = trace_row["trace"]
                    duration_seconds = trace.get("duration_seconds")
                    duration_human = trace.get("duration_human")
                    if duration_seconds is not None and duration_human:
                        timing_data.append({
                            "call_name": call_name,
                            "trace_id": trace_id,
                            "duration_seconds": duration_seconds,
                            "duration_human": duration_human,
                            "step_name": trace_row.get("step_name"),
                        })

        if not timing_data:
            return ""

        # Group by call_name and calculate averages
        call_summaries = []
        for call_name in call_names:
            call_times = [d for d in timing_data if d["call_name"] == call_name]
            if call_times:
                avg_duration = sum(t["duration_seconds"] for t in call_times) / len(call_times)
                if avg_duration < 60:
                    avg_human = ".1fs"
                elif avg_duration < 3600:
                    minutes = int(avg_duration // 60)
                    seconds = avg_duration % 60
                    avg_human = f"{minutes}m {seconds:.1f}s"
                else:
                    hours = int(avg_duration // 3600)
                    minutes = int((avg_duration % 3600) // 60)
                    avg_human = f"{hours}h {minutes}m"

                call_summaries.append(f"- **{call_name}**: {avg_human} average ({len(call_times)} traces)")

        summary = "### Average Step Execution Times\n" + "\n".join(call_summaries)

        # Add individual trace details
        if len(timing_data) <= 10:  # Only show details for small numbers
            summary += "\n\n### Individual Trace Times\n"
            for data in sorted(timing_data, key=lambda x: x["duration_seconds"]):
                summary += f"- {data['step_name']} ({data['trace_id'][:8]}): {data['duration_human']}\n"

        return summary

    @app.post("/api/harness/pr")
    def create_harness_pr(payload: HarnessPRRequest) -> Dict[str, Any]:
        if not _capability_enabled("MECH_ENABLE_GIT_OPS"):
            raise HTTPException(
                status_code=403,
                detail="Git operations are disabled (MECH_ENABLE_GIT_OPS=false).",
            )
        if shutil.which("git") is None:
            raise HTTPException(status_code=500, detail="git CLI is not available")

        track = str(payload.pr_track or "existing_harness")
        if track not in {"existing_harness", "new_harness"}:
            raise HTTPException(status_code=400, detail="pr_track must be existing_harness or new_harness")
        branch_prefix = "codex/existing-harness" if track == "existing_harness" else "codex/new-harness"
        branch = payload.branch_name or f"{branch_prefix}-{int(time.time())}"
        call_names = sorted({str(name).strip() for name in payload.call_names if str(name).strip()})
        if not call_names:
            raise HTTPException(status_code=400, detail="call_names cannot be empty")
        try:
            call_names = sorted({normalize_call_name(name) for name in call_names})
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        prompt_paths: List[str] = []
        for call_name in call_names:
            prompt_info = get_call_prompt_version(call_name, base)
            base_path = Path(str(prompt_info.get("call_base_path") or ""))
            fs_path = Path(str(prompt_info.get("few_shot_path") or ""))
            if not base_path.exists():
                raise HTTPException(status_code=404, detail=f"Call base prompt not found: {call_name}")
            if not fs_path.exists():
                raise HTTPException(status_code=404, detail=f"Call few-shot file not found: {call_name}")
            prompt_paths.extend([str(base_path.resolve()), str(fs_path.resolve())])

        exported_evidence_files: List[str] = []
        if payload.include_evidence:
            for call_name, trace_ids in payload.evidence_trace_ids.items():
                if call_name not in call_names:
                    continue
                for trace_id in trace_ids:
                    trace_row = store.get_trace_record(trace_id)
                    if trace_row is None:
                        raise HTTPException(status_code=404, detail=f"Trace not found for evidence: {trace_id}")
                    exported = _export_trace_evidence_item(trace_row, created_by="api_harness_pr")
                    exported_evidence_files.append(str(exported.get("path") or ""))

        gate_result = None
        if track == "existing_harness":
            gate_result = validate_evidence_for_calls(changed_calls=call_names, base_dir=base)
            if not gate_result.ok:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "prompt_trace_evidence_gate_failed",
                        "calls": gate_result.changed_calls,
                        "messages": gate_result.errors,
                    },
                )

        steps: List[Dict[str, Any]] = []

        def _run(args: List[str]) -> None:
            code, out, err = _run_git_command(args, base)
            steps.append({"cmd": " ".join(args), "code": code, "stdout": out, "stderr": err})
            if code != 0:
                raise RuntimeError(err.strip() or out.strip() or f"Command failed: {' '.join(args)}")

        try:
            _run(["git", "rev-parse", "--is-inside-work-tree"])
            code, _, _ = _run_git_command(["git", "checkout", branch], base)
            if code != 0:
                _run(["git", "checkout", "-b", branch])

            relative_files = [str(Path(path).resolve().relative_to(base)) for path in prompt_paths]
            if gate_result is not None:
                for files in gate_result.valid_evidence_by_call.values():
                    relative_files.extend(files)
            relative_files.extend([path for path in exported_evidence_files if path])
            relative_files = sorted({path for path in relative_files if path})
            _run(["git", "add", *relative_files])
            _run(["git", "commit", "-m", payload.commit_message])

            # Generate timing summary for PR body
            timing_summary = _generate_pr_timing_summary(payload.evidence_trace_ids, call_names)
            enhanced_pr_body = payload.pr_body
            if timing_summary:
                enhanced_pr_body += "\n\n## Performance Timing Summary\n\n" + timing_summary

            pr_url = None
            if payload.push:
                _run(["git", "push", "-u", "origin", branch])
            if payload.open_pr and shutil.which("gh"):
                code, out, err = _run_git_command(
                    ["gh", "pr", "create", "--title", payload.pr_title, "--body", enhanced_pr_body, "--head", branch],
                    base,
                )
                steps.append({"cmd": "gh pr create ...", "code": code, "stdout": out, "stderr": err})
                if code == 0:
                    pr_url = out.strip().splitlines()[-1] if out.strip() else None

            return {
                "ok": True,
                "pr_track": track,
                "branch": branch,
                "files": relative_files,
                "steps": steps,
                "pr_url": pr_url,
                "required_tests": (
                    [
                        "make test",
                        "PYTHONPATH=. pytest tests/llm/test_eval_tiers.py -k medium",
                    ]
                    if track == "existing_harness"
                    else [
                        "make test",
                        "PYTHONPATH=. pytest tests/llm/test_eval_tiers.py -k easy",
                    ]
                ),
            }
        except Exception as exc:
            return {"ok": False, "branch": branch, "steps": steps, "error": str(exc)}

    @app.get("/api/memory")
    def list_memory(scope: str | None = None) -> Dict[str, Any]:
        local_items = store.list_memory_items(scope=scope, active_only=True)
        curated = registry.curated_memory_items()
        if scope:
            curated = [item for item in curated if item.get("scope") == scope]
        return {"curated": curated, "local": local_items}

    @app.post("/api/memory/query")
    def query_memory(payload: MemoryQueryRequest) -> Dict[str, Any]:
        curated = registry.curated_memory_items()
        local_items = store.list_memory_items(scope=payload.scope, active_only=True)

        def _passes(item: Dict[str, Any]) -> bool:
            key = str(item.get("key") or "")
            tags = [str(tag) for tag in (item.get("tags") or [])]
            if payload.key_contains and payload.key_contains.lower() not in key.lower():
                return False
            return _match_tags(tags, payload.tags)

        curated_filtered = [item for item in curated if _passes(item)]
        local_filtered = [
            {
                "id": item["id"],
                "scope": item["scope"],
                "key": item["key"],
                "source": item["source"],
                "confidence": item.get("confidence"),
                "tags": item.get("tags", []),
                "value": item.get("value", {}),
            }
            for item in local_items
            if _passes(item)
        ]

        merged = curated_filtered + local_filtered
        return {
            "items": merged,
            "counts": {
                "curated": len(curated_filtered),
                "local": len(local_filtered),
                "total": len(merged),
            },
        }

    @app.post("/api/memory/items")
    def add_memory_item(payload: MemoryItemRequest) -> Dict[str, Any]:
        memory_id = store.add_memory_item(
            scope=payload.scope,
            key=payload.key,
            value=payload.value,
            source=payload.source,
            confidence=payload.confidence,
            tags=payload.tags,
            active=payload.active,
        )
        return {"id": memory_id}

    @app.get("/api/curriculum/status")
    def curriculum_status(model_name: str = OPUS_MODEL) -> Dict[str, Any]:
        return build_curriculum_status(base, store, model_name=model_name)

    @app.get("/api/curriculum/history")
    def curriculum_history_view(model_name: str = OPUS_MODEL, limit: int = 100) -> Dict[str, Any]:
        return {"items": curriculum_history(store, model_name=model_name, limit=max(1, min(limit, 500)))}

    @app.get("/api/curriculum/checkpoints/{checkpoint_id}")
    def curriculum_checkpoint_detail(checkpoint_id: str) -> Dict[str, Any]:
        item = store.get_curriculum_checkpoint(checkpoint_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Curriculum checkpoint not found")
        return item

    @app.post("/api/curriculum/submit")
    def curriculum_submit(payload: CurriculumSubmitRequest) -> Dict[str, Any]:
        try:
            item = submit_curriculum_release(base, store, model_name=payload.model_name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"item": item}

    @app.post("/api/curriculum/publish-due")
    def curriculum_publish_due() -> Dict[str, Any]:
        try:
            items = publish_due_curriculum_releases(base, store)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"items": items}

    @app.post("/api/curriculum/publish/{checkpoint_id}")
    def curriculum_publish(checkpoint_id: str, payload: CurriculumPublishRequest) -> Dict[str, Any]:
        try:
            item = publish_curriculum_release(base, store, queue_id=checkpoint_id, force=payload.force)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"item": item}

    @app.get("/api/curriculum/readme-preview")
    def curriculum_readme_preview(model_name: str = OPUS_MODEL) -> Dict[str, Any]:
        content = render_curriculum_readme(base, store, model_name=model_name)
        return {"content": content}

    @app.get("/api/health")
    def health() -> JSONResponse:
        return JSONResponse({"ok": True, "db_path": str(db_path)})

    return app


app = create_app()
