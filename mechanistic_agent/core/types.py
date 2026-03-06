"""Typed contracts for the local-first mechanistic runtime."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


RunMode = Literal["verified", "unverified"]
RunStatus = Literal["pending", "running", "paused", "completed", "failed", "stopped"]
TemplateGuidanceMode = Literal["active", "weak", "disabled", "no_match"]
ModuleKind = Literal["llm", "deterministic", "decision", "text_completion"]
ModulePhase = Literal["pre_loop", "post_step", "loop"]
FewShotSelectionStrategy = Literal["top_score", "most_recent", "first"]
OrchestrationMode = Literal["standard", "ralph"]
CoordinationTopology = Literal["sas", "centralized_mas", "independent_mas", "decentralized_mas"]
HarnessStrategy = Literal["latest", "portfolio", "mutate"]
BabysitMode = Literal["off", "advisory"]
RalphStopReason = Literal[
    "completed",
    "max_iterations",
    "max_runtime_seconds",
    "max_cost_usd",
    "repeat_failure_signature_limit",
    "no_mutation_actions_remaining",
    "manual_stop",
    "unsafe_validator_state",
    "run_failed",
]


@dataclass(slots=True)
class StepValidationCheck:
    """A single deterministic check outcome."""

    name: str
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StepValidationResult:
    """Validation result for a step output."""

    checks: List[StepValidationCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [
                {"name": check.name, "passed": check.passed, "details": check.details}
                for check in self.checks
            ],
        }


@dataclass(slots=True)
class StepResult:
    """Normalized step output recorded for a run."""

    step_name: str
    tool_name: str
    output: Dict[str, Any]
    model: Optional[str] = None
    reasoning_level: Optional[str] = None
    attempt: int = 1
    retry_index: int = 0
    source: Literal["llm", "human", "deterministic"] = "llm"
    validation: Optional[StepValidationResult] = None
    token_usage: Optional[Dict[str, int]] = None
    cost: Optional[Dict[str, float]] = None


@dataclass(slots=True)
class VerificationDecision:
    """Human decision for a step in verified mode."""

    run_id: str
    step_name: str
    decision: Literal["accept", "reject"]
    rationale: Optional[str] = None
    decided_by: Optional[str] = None
    attempt: Optional[int] = None


@dataclass(slots=True)
class FeedbackRecord:
    """Feedback payload for step/run outcomes."""

    run_id: str
    step_name: Optional[str]
    rating: Optional[int]
    label: Optional[str]
    comment: Optional[str]
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BranchCandidate:
    """A validated candidate at a branch point."""

    rank: int
    intermediate_smiles: str
    intermediate_output: Dict[str, Any] = field(default_factory=dict)
    mechanism_output: Dict[str, Any] = field(default_factory=dict)
    resulting_state: List[str] = field(default_factory=list)
    validation_summary: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "intermediate_smiles": self.intermediate_smiles,
            "resulting_state": self.resulting_state,
        }


@dataclass(slots=True)
class BranchPoint:
    """Records a step where multiple validated candidates existed."""

    step_index: int
    current_state: List[str] = field(default_factory=list)
    previous_intermediates: List[str] = field(default_factory=list)
    template_guidance_snapshot: Optional[Dict[str, Any]] = None
    chosen_candidate: Optional[BranchCandidate] = None
    alternatives: List[BranchCandidate] = field(default_factory=list)
    exhausted: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "current_state": self.current_state,
            "chosen_rank": self.chosen_candidate.rank if self.chosen_candidate else None,
            "alternative_count": len(self.alternatives),
            "has_template_guidance_snapshot": bool(self.template_guidance_snapshot),
            "exhausted": self.exhausted,
        }


@dataclass(slots=True)
class FailedPath:
    """Records a failed exploration path for UI display."""

    branch_step_index: int
    candidate_rank: int
    steps_taken: List[Dict[str, Any]] = field(default_factory=list)
    failure_reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "branch_step_index": self.branch_step_index,
            "candidate_rank": self.candidate_rank,
            "steps_taken": self.steps_taken,
            "failure_reason": self.failure_reason,
        }


@dataclass(slots=True)
class FewShotSelectionConfig:
    """Harness-controlled policy for selecting active few-shot examples."""

    enabled: bool = True
    max_examples: int = 4
    selection_strategy: FewShotSelectionStrategy = "top_score"
    min_score: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "enabled": self.enabled,
            "max_examples": self.max_examples,
            "selection_strategy": self.selection_strategy,
        }
        if self.min_score is not None:
            payload["min_score"] = self.min_score
        return payload

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "FewShotSelectionConfig":
        payload = dict(data or {})
        raw_strategy = str(payload.get("selection_strategy") or "top_score").strip().lower()
        strategy: FewShotSelectionStrategy = "top_score"
        if raw_strategy in {"top_score", "most_recent", "first"}:
            strategy = raw_strategy  # type: ignore[assignment]

        raw_max_examples = payload.get("max_examples", 4)
        try:
            max_examples = max(0, int(raw_max_examples))
        except (TypeError, ValueError):
            max_examples = 4

        min_score = payload.get("min_score")
        try:
            min_score_value = float(min_score) if min_score is not None else None
        except (TypeError, ValueError):
            min_score_value = None

        return cls(
            enabled=bool(payload.get("enabled", True)),
            max_examples=max_examples,
            selection_strategy=strategy,
            min_score=min_score_value,
        )


@dataclass(slots=True)
class TopologyProfile:
    """Configuration for a coordination topology strategy."""

    agent_count: int = 1
    max_candidates_per_agent: int = 3
    peer_rounds: int = 0
    aggregation_mode: str = "orchestrator_select"
    consensus_key: str = "reaction_smirks"
    consensus_fallback_key: str = "intermediate_smiles"
    description: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "agent_count": self.agent_count,
            "max_candidates_per_agent": self.max_candidates_per_agent,
            "peer_rounds": self.peer_rounds,
            "aggregation_mode": self.aggregation_mode,
            "consensus_key": self.consensus_key,
            "consensus_fallback_key": self.consensus_fallback_key,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "TopologyProfile":
        d = dict(data or {})
        return cls(
            agent_count=max(1, int(d.get("agent_count") or 1)),
            max_candidates_per_agent=max(1, int(d.get("max_candidates_per_agent") or 3)),
            peer_rounds=max(0, int(d.get("peer_rounds") or 0)),
            aggregation_mode=str(d.get("aggregation_mode") or "orchestrator_select"),
            consensus_key=str(d.get("consensus_key") or "reaction_smirks"),
            consensus_fallback_key=str(d.get("consensus_fallback_key") or "intermediate_smiles"),
            description=str(d.get("description") or ""),
        )


@dataclass(slots=True)
class ModuleSpec:
    """Specification for a single pipeline module in the harness."""

    id: str
    label: str
    kind: ModuleKind
    phase: ModulePhase
    enabled: bool = True
    step: int = 0
    agent_class: str = ""
    tool_function: str = ""
    step_name: str = ""
    tool_name: str = ""
    validator_skill: Optional[str] = None
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    movable: bool = True
    removable: bool = True
    config_gate: Optional[str] = None
    prompt_call_name: Optional[str] = None
    group_key: Optional[str] = None
    description: str = ""
    io_schema: Optional[Dict[str, Any]] = None
    custom: bool = False
    prompt_text: Optional[str] = None
    code_text: Optional[str] = None
    few_shot: Optional[FewShotSelectionConfig] = None

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "step": self.step,
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "phase": self.phase,
            "enabled": self.enabled,
            "agent_class": self.agent_class,
            "tool_function": self.tool_function,
            "step_name": self.step_name,
            "tool_name": self.tool_name,
            "validator_skill": self.validator_skill,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "movable": self.movable,
            "removable": self.removable,
            "config_gate": self.config_gate,
            "prompt_call_name": self.prompt_call_name,
            "group_key": self.group_key,
            "description": self.description,
        }
        if self.io_schema is not None:
            d["io_schema"] = self.io_schema
        if self.custom:
            d["custom"] = True
        if self.prompt_text is not None:
            d["prompt_text"] = self.prompt_text
        if self.code_text is not None:
            d["code_text"] = self.code_text
        if self.few_shot is not None:
            d["few_shot"] = self.few_shot.as_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModuleSpec":
        return cls(
            id=str(data.get("id") or ""),
            label=str(data.get("label") or ""),
            kind=str(data.get("kind") or "deterministic"),  # type: ignore[arg-type]
            phase=str(data.get("phase") or "pre_loop"),  # type: ignore[arg-type]
            enabled=bool(data.get("enabled", True)),
            step=int(data.get("step") or 0),
            agent_class=str(data.get("agent_class") or ""),
            tool_function=str(data.get("tool_function") or ""),
            step_name=str(data.get("step_name") or data.get("id") or ""),
            tool_name=str(data.get("tool_name") or ""),
            validator_skill=data.get("validator_skill"),
            inputs=list(data.get("inputs") or []),
            outputs=list(data.get("outputs") or []),
            movable=bool(data.get("movable", True)),
            removable=bool(data.get("removable", True)),
            config_gate=data.get("config_gate"),
            prompt_call_name=data.get("prompt_call_name"),
            group_key=data.get("group_key"),
            description=str(data.get("description") or ""),
            io_schema=data.get("io_schema"),
            custom=bool(data.get("custom", False)),
            prompt_text=data.get("prompt_text"),
            code_text=data.get("code_text"),
            few_shot=FewShotSelectionConfig.from_dict(data.get("few_shot")) if data.get("few_shot") is not None else None,
        )


@dataclass(slots=True)
class HarnessConfig:
    """Complete harness pipeline definition.

    schema_version "2.0" adds explicit step ordering, loop_module, validator_skill
    references per module, tool_calling_mode, and a human-readable execution_note.
    """

    version: str = ""
    name: str = "default"
    schema_version: str = "2.0"
    description: str = ""
    tool_calling_mode: str = "forced"
    execution_note: str = ""
    pre_loop_modules: List[ModuleSpec] = field(default_factory=list)
    loop_module: Optional[Dict[str, Any]] = None
    post_step_modules: List[ModuleSpec] = field(default_factory=list)
    few_shot_defaults: FewShotSelectionConfig = field(default_factory=FewShotSelectionConfig)
    topology_profiles: Dict[str, TopologyProfile] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "schema_version": self.schema_version,
            "description": self.description,
            "tool_calling_mode": self.tool_calling_mode,
            "execution_note": self.execution_note,
            "pre_loop_modules": [m.as_dict() for m in self.pre_loop_modules],
            "loop_module": self.loop_module,
            "post_step_modules": [m.as_dict() for m in self.post_step_modules],
            "few_shot_defaults": self.few_shot_defaults.as_dict(),
            "metadata": dict(self.metadata),
        }
        if self.topology_profiles:
            d["topology_profiles"] = {k: v.as_dict() for k, v in self.topology_profiles.items()}
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HarnessConfig":
        raw_profiles = data.get("topology_profiles") or {}
        profiles: Dict[str, TopologyProfile] = {}
        if isinstance(raw_profiles, dict):
            for key, val in raw_profiles.items():
                if isinstance(val, dict):
                    profiles[str(key)] = TopologyProfile.from_dict(val)
        return cls(
            version=str(data.get("version") or ""),
            name=str(data.get("name") or "default"),
            schema_version=str(data.get("schema_version") or "1.0"),
            description=str(data.get("description") or ""),
            tool_calling_mode=str(data.get("tool_calling_mode") or "forced"),
            execution_note=str(data.get("execution_note") or ""),
            pre_loop_modules=[
                ModuleSpec.from_dict(m) for m in (data.get("pre_loop_modules") or [])
            ],
            loop_module=data.get("loop_module"),
            post_step_modules=[
                ModuleSpec.from_dict(m) for m in (data.get("post_step_modules") or [])
            ],
            few_shot_defaults=FewShotSelectionConfig.from_dict(data.get("few_shot_defaults")),
            topology_profiles=profiles,
            metadata=dict(data.get("metadata") or {}),
        )

    def get_topology_profile(self, topology: str) -> TopologyProfile:
        """Return the profile for a topology, falling back to centralized_mas defaults."""
        if topology in self.topology_profiles:
            return self.topology_profiles[topology]
        if "centralized_mas" in self.topology_profiles:
            return self.topology_profiles["centralized_mas"]
        return TopologyProfile()

    def all_modules(self) -> List[ModuleSpec]:
        """Return all modules across both phases."""
        return list(self.pre_loop_modules) + list(self.post_step_modules)

    def enabled_pre_loop(self) -> List[ModuleSpec]:
        return [m for m in self.pre_loop_modules if m.enabled]

    def enabled_post_step(self) -> List[ModuleSpec]:
        return [m for m in self.post_step_modules if m.enabled]

    def few_shot_policy_for_call(self, call_name: str) -> FewShotSelectionConfig:
        """Resolve the active few-shot policy for a prompt call in this harness."""
        normalized_call_name = str(call_name or "").strip()
        if not normalized_call_name:
            return self.few_shot_defaults

        for module in self.pre_loop_modules + self.post_step_modules:
            if module.prompt_call_name == normalized_call_name:
                return module.few_shot or self.few_shot_defaults

        loop_module = self.loop_module or {}
        if str(loop_module.get("prompt_call_name") or "").strip() == normalized_call_name:
            if isinstance(loop_module.get("few_shot"), dict):
                return FewShotSelectionConfig.from_dict(loop_module.get("few_shot"))
            return self.few_shot_defaults

        return self.few_shot_defaults

    def few_shot_policies_by_call(self) -> Dict[str, Dict[str, Any]]:
        """Return a serializable call_name -> policy map for runtime prompt selection."""
        policies: Dict[str, Dict[str, Any]] = {}
        for module in self.pre_loop_modules + self.post_step_modules:
            if module.prompt_call_name:
                policies[module.prompt_call_name] = (module.few_shot or self.few_shot_defaults).as_dict()
        loop_module = self.loop_module or {}
        loop_call_name = str(loop_module.get("prompt_call_name") or "").strip()
        if loop_call_name:
            if isinstance(loop_module.get("few_shot"), dict):
                policies[loop_call_name] = FewShotSelectionConfig.from_dict(loop_module.get("few_shot")).as_dict()
            else:
                policies[loop_call_name] = self.few_shot_defaults.as_dict()
        return policies


@dataclass(slots=True)
class RunConfig:
    """Per-run configuration persisted in storage."""

    model: str
    model_name: Optional[str] = None
    model_family: str = "openai"
    step_models: Dict[str, str] = field(default_factory=dict)
    step_reasoning: Dict[str, str] = field(default_factory=dict)
    thinking_level: Optional[str] = None
    reasoning_level: Optional[str] = None
    optional_llm_tools: List[str] = field(default_factory=list)
    functional_groups_enabled: bool = True
    intermediate_prediction_enabled: bool = True
    max_steps: int = 10
    max_runtime_seconds: float = 600.0
    api_keys: Dict[str, str] = field(default_factory=dict)
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
    reaction_template_disable_step_window: int = 3
    reaction_template_disable_consecutive_mismatch: int = 2
    orchestration_mode: OrchestrationMode = "standard"
    coordination_topology: CoordinationTopology = "centralized_mas"
    harness_name: str = "default"
    harness_config_path: Optional[str] = None
    harness_strategy: HarnessStrategy = "latest"
    harness_list: List[str] = field(default_factory=list)
    max_iterations: int = 0
    completion_promise: str = "target_products_reached && flow_node:run_complete"
    ralph_max_runtime_seconds: float = 6000.0
    max_cost_usd: Optional[float] = 2.0
    repeat_failure_signature_limit: int = 2
    babysit_mode: BabysitMode = "off"
    allow_validator_mutation: bool = False
    ralph_parent_run_id: Optional[str] = None


@dataclass(slots=True)
class RalphConfig:
    """Outer-loop orchestration controls for Ralph mode."""

    max_iterations: int = 0
    completion_promise: str = "target_products_reached && flow_node:run_complete"
    max_runtime_seconds: float = 6000.0
    max_cost_usd: Optional[float] = 2.0
    repeat_failure_signature_limit: int = 2
    harness_strategy: HarnessStrategy = "latest"
    harness_list: List[str] = field(default_factory=list)
    babysit_mode: BabysitMode = "off"
    allow_validator_mutation: bool = False


@dataclass(slots=True)
class RalphAttemptState:
    """One completed Ralph outer-loop attempt."""

    attempt_index: int
    child_run_id: str
    harness_name: str
    harness_sha: str
    parent_harness_sha: str
    mutation_actions: List[Dict[str, Any]] = field(default_factory=list)
    diff_summary: Dict[str, Any] = field(default_factory=dict)
    stop_reason: Optional[str] = None
    completion_promise_met: bool = False
    cost_usd: float = 0.0


@dataclass(slots=True)
class RunInput:
    """User-provided reaction input payload."""

    starting_materials: List[str]
    products: List[str]
    temperature_celsius: float = 25.0
    ph: Optional[float] = None
    example_id: Optional[str] = None


@dataclass(slots=True)
class TemplateGuidanceState:
    """Runtime state for optional reaction-template guidance."""

    mode: TemplateGuidanceMode = "no_match"
    selected_type_id: Optional[str] = None
    selected_label_exact: Optional[str] = None
    selection_confidence: float = 0.0
    selection_confidence_gap: Optional[float] = None
    selection_confidence_threshold: Optional[float] = None
    selection_margin_threshold: Optional[float] = None
    suitable_step_count: Optional[int] = None
    current_template_step_index: int = 1
    completed_steps_count: int = 0
    alignment_history: List[Dict[str, Any]] = field(default_factory=list)
    disable_reason: Optional[str] = None
    selection_decision_reason: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "selected_type_id": self.selected_type_id,
            "selected_label_exact": self.selected_label_exact,
            "selection_confidence": self.selection_confidence,
            "selection_confidence_gap": self.selection_confidence_gap,
            "selection_confidence_threshold": self.selection_confidence_threshold,
            "selection_margin_threshold": self.selection_margin_threshold,
            "suitable_step_count": self.suitable_step_count,
            "current_template_step_index": self.current_template_step_index,
            "completed_steps_count": self.completed_steps_count,
            "alignment_history": [dict(item) for item in self.alignment_history],
            "disable_reason": self.disable_reason,
            "selection_decision_reason": self.selection_decision_reason,
        }

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "TemplateGuidanceState":
        data = dict(payload or {})
        history = data.get("alignment_history")
        return cls(
            mode=str(data.get("mode") or "no_match"),  # type: ignore[arg-type]
            selected_type_id=data.get("selected_type_id"),
            selected_label_exact=data.get("selected_label_exact"),
            selection_confidence=float(data.get("selection_confidence") or 0.0),
            selection_confidence_gap=(
                float(data["selection_confidence_gap"])
                if data.get("selection_confidence_gap") is not None
                else None
            ),
            selection_confidence_threshold=(
                float(data["selection_confidence_threshold"])
                if data.get("selection_confidence_threshold") is not None
                else None
            ),
            selection_margin_threshold=(
                float(data["selection_margin_threshold"])
                if data.get("selection_margin_threshold") is not None
                else None
            ),
            suitable_step_count=(
                int(data["suitable_step_count"])
                if data.get("suitable_step_count") is not None
                else None
            ),
            current_template_step_index=max(1, int(data.get("current_template_step_index") or 1)),
            completed_steps_count=max(0, int(data.get("completed_steps_count") or 0)),
            alignment_history=(
                [dict(item) for item in history if isinstance(item, dict)]
                if isinstance(history, list)
                else []
            ),
            disable_reason=data.get("disable_reason"),
            selection_decision_reason=data.get("selection_decision_reason"),
        )


@dataclass(slots=True)
class RunState:
    """Mutable in-memory state for active run execution."""

    run_id: str
    mode: RunMode
    run_input: RunInput
    run_config: RunConfig
    current_state: List[str] = field(default_factory=list)
    previous_intermediates: List[str] = field(default_factory=list)
    step_index: int = 0
    stop_requested: bool = False
    awaiting_verification: bool = False
    paused: bool = False
    branch_points: List[BranchPoint] = field(default_factory=list)
    failed_paths: List[FailedPath] = field(default_factory=list)
    pending_resume_candidate: Optional["BranchCandidate"] = None
    latest_step_mapping: Optional[Dict[str, Any]] = None
    reaction_type_selection: Optional[Dict[str, Any]] = None
    template_guidance_state: Optional[TemplateGuidanceState] = None
    selected_reaction_template: Optional[Dict[str, Any]] = None
    step_start_times: Dict[str, float] = field(default_factory=dict)

    def initialise(self) -> None:
        if not self.current_state:
            self.current_state = list(self.run_input.starting_materials)
