"""Pydantic schemas for the FastAPI mechanistic runtime endpoints."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class RalphRunConfig(BaseModel):
    max_iterations: int = 0
    max_runtime_seconds: float = 6000.0
    max_cost_usd: Optional[float] = 2.0
    repeat_failure_signature_limit: int = 2
    harness_strategy: Literal["latest", "portfolio", "mutate"] = "latest"
    harness_list: List[str] = Field(default_factory=list)
    babysit_mode: Literal["off", "advisory"] = "off"
    allow_validator_mutation: bool = False


class CreateRunRequest(BaseModel):
    mode: Literal["unverified"] = "unverified"
    orchestration_mode: Literal["standard", "ralph"] = "standard"
    starting_materials: List[str] = Field(default_factory=list)
    products: List[str] = Field(default_factory=list)
    example_id: Optional[str] = None
    temperature_celsius: float = 25.0
    ph: Optional[float] = None
    model_name: Optional[str] = None
    model: Optional[str] = None
    thinking_level: Optional[Literal["low", "high"]] = None
    reasoning_level: Optional[Literal["lowest", "highest"]] = None
    api_keys: Dict[str, str] = Field(default_factory=dict)
    optional_llm_tools: List[str] = Field(default_factory=lambda: ["attempt_atom_mapping", "predict_missing_reagents"])
    functional_groups_enabled: bool = True
    intermediate_prediction_enabled: bool = True
    max_steps: int = 10
    max_runtime_seconds: float = 600.0
    retry_same_candidate_max: int = 1
    max_reproposals_per_step: int = 4
    reproposal_on_repeat_failure: bool = True
    candidate_rescue_enabled: bool = True
    step_mapping_enabled: bool = True
    arrow_push_annotation_enabled: bool = True
    dbe_policy: Literal["strict", "soft"] = "soft"
    reaction_template_policy: Literal["off", "auto"] = "auto"
    reaction_template_confidence_threshold: float = 0.65
    reaction_template_margin_threshold: float = 0.10
    harness_name: str = "default"
    harness_config_path: Optional[str] = None
    coordination_topology: Literal[
        "sas", "centralized_mas", "independent_mas", "decentralized_mas"
    ] = "centralized_mas"
    ralph: Optional[RalphRunConfig] = None
    dry_run: bool = False


class CreateRunResponse(BaseModel):
    run_id: str
    status: str
    note: Optional[str] = None


class VerifyStepRequest(BaseModel):
    decision: Literal["accept", "reject"]
    rationale: Optional[str] = None
    decided_by: Optional[str] = None
    attempt: Optional[int] = None


class FeedbackRequest(BaseModel):
    rating: Optional[int] = None
    label: Optional[str] = None
    comment: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


class RalphVoteRequest(BaseModel):
    attempt_index: int
    step_index: int
    candidate_a: Dict[str, Any] = Field(default_factory=dict)
    candidate_b: Dict[str, Any] = Field(default_factory=dict)
    vote: Literal["A", "B"]
    confidence: Optional[float] = None
    source: str = "ui"


class MemoryItemRequest(BaseModel):
    scope: str
    key: str
    value: Dict[str, Any]
    source: str = "user"
    confidence: Optional[float] = None
    tags: List[str] = Field(default_factory=list)
    active: bool = True


class MemoryQueryRequest(BaseModel):
    scope: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    key_contains: Optional[str] = None


class EvaluateRunRequest(BaseModel):
    judge_model: str = "gpt-5.4"


class SaveEvaluationRequest(BaseModel):
    evaluation: Dict[str, Any] = Field(default_factory=dict)
    judge_model: str = "gpt-5.4"


class HarnessApplyRequest(BaseModel):
    call_name: str = "propose_mechanism_step"
    component: Literal["base", "few_shot"] = "base"
    recommendation: Optional[str] = None
    append_mode: bool = True


class HarnessPRRequest(BaseModel):
    pr_track: Literal["existing_harness", "new_harness"] = "existing_harness"
    call_names: List[str] = Field(default_factory=lambda: ["propose_mechanism_step"])
    evidence_trace_ids: Dict[str, List[str]] = Field(default_factory=dict)
    branch_name: Optional[str] = None
    commit_message: str = "Update harness prompts from evaluation feedback"
    pr_title: str = "Harness update from local evaluation"
    pr_body: str = "Applies local evaluation recommendations to harness prompts."
    push: bool = True
    open_pr: bool = True
    run_id: Optional[str] = None
    include_evidence: bool = True


class EvalRunSetRequest(BaseModel):
    eval_set_id: str
    run_group_name: Optional[str] = None
    tier_name: Optional[Literal["easy", "medium", "hard"]] = None
    step_count: Optional[int] = None
    case_ids: List[str] = Field(default_factory=list)
    model_name: Optional[str] = None
    model: Optional[str] = None
    thinking_level: Optional[Literal["low", "high"]] = None
    reasoning_level: Optional[Literal["lowest", "highest"]] = None
    mode: Literal["unverified"] = "unverified"
    max_cases: int = 25
    max_steps: int = 6
    max_runtime_seconds: float = 180.0
    async_mode: bool = True


class TraceToFewShotRequest(BaseModel):
    trace_id: str
    example_key: Optional[str] = None
    approved: bool = True


class ResumeRunRequest(BaseModel):
    decision: Literal["continue", "stop"] = "continue"
    rationale: Optional[str] = None
    decided_by: Optional[str] = None


class MechanismStepSubmitRequest(BaseModel):
    step_index: int
    current_state: List[str]
    resulting_state: List[str]
    predicted_intermediate: Optional[str] = None
    target_products: List[str]
    electron_pushes: List[Dict[str, Any]]
    reaction_smirks: str
    note: Optional[str] = None


class ApproveTraceRequest(BaseModel):
    approved: bool = True
    label: Optional[str] = None
    notes: Optional[str] = None
    approved_by: Optional[str] = None


class CurationExportRequest(BaseModel):
    eval_set_id: Optional[str] = None
    include_few_shot: bool = True
    include_baselines: bool = True
    include_leaderboard: bool = True
    created_by: Optional[str] = None


class TraceEvidenceExportRequest(BaseModel):
    trace_ids: List[str]
    created_by: Optional[str] = None


class ConvertInputsRequest(BaseModel):
    """Request body for POST /api/convert_inputs."""

    starting_materials: List[str] = Field(default_factory=list)
    products: List[str] = Field(default_factory=list)


class ImportTemplateRequest(BaseModel):
    """Request body for POST /api/eval_sets/import_template."""

    name: str = "user_template"
    version: str = "v1"
    cases: List[Dict[str, Any]] = Field(default_factory=list)
    yaml_text: Optional[str] = None
    auto_convert: bool = True


class StartVerificationRequest(BaseModel):
    """Request body for POST /api/verification/start."""

    eval_set_id: str
    model_family: Literal["openai", "claude", "gemini"] = "openai"


class BaselineEvalRunSetRequest(BaseModel):
    """Request body for POST /api/evals/baseline-runset.

    Runs harness-free single-shot mechanism prediction for comparison against
    the harness pipeline.  Results appear on the leaderboard labelled as baseline.
    """

    eval_set_id: str
    run_group_name: Optional[str] = None
    tier_name: Optional[Literal["easy", "medium", "hard"]] = None
    step_count: Optional[int] = None
    case_ids: List[str] = Field(default_factory=list)
    model_name: Optional[str] = None
    model: Optional[str] = None
    thinking_level: Optional[Literal["low", "high"]] = None
    max_cases: int = 25
    timeout_seconds: float = 180.0
    async_mode: bool = True


class OfficialEvalRunSetRequest(BaseModel):
    """Request body for POST /api/evals/official-runset.

    Runs the official leaderboard holdout suite only.
    """

    eval_set_id: Optional[str] = None
    run_group_name: Optional[str] = None
    case_ids: List[str] = Field(default_factory=list)
    model_name: Optional[str] = None
    model: Optional[str] = None
    thinking_level: Optional[Literal["low", "high"]] = None
    reasoning_level: Optional[Literal["lowest", "highest"]] = None
    mode: Literal["unverified"] = "unverified"
    max_cases: int = 200
    max_steps: int = 10
    max_runtime_seconds: float = 300.0
    async_mode: bool = True


class SeedSimulatedLeaderboardRequest(BaseModel):
    """Request body for POST /api/evals/seed-simulated-leaderboard.

    Inserts clearly-labelled placeholder rows into the leaderboard so the UI
    can be evaluated before real eval runs complete.  All rows are prefixed
    with ``[SIMULATED]`` in run_group_name and should be deleted once real
    data is available.
    """

    eval_set_id: str
    case_count: int = Field(default=5, ge=1, le=50)


class CurriculumSubmitRequest(BaseModel):
    model_name: str = "anthropic/claude-opus-4.6"


class CurriculumPublishRequest(BaseModel):
    force: bool = False
