"""LLM-proposed, trace-conditioned harness mutations.

The blind lane mutators in :mod:`lane_mutator` nudge a topology integer, toggle
a module, append a fixed sentence, or drop the last few-shot line. They cannot
express the fixes that failed-step traces actually point at. This module asks a
model to read a compact *failure digest* of the last micro-eval slice and
propose ONE targeted, machine-applicable edit to a prompt, few-shot lane,
topology profile field, or harness module flag. The proposal is applied to a
variant artifact exactly like the blind mutators produce, and it is still gated
by the existing keep/discard acceptance rule — the model proposes, the eval
decides.

Frozen surfaces (validators, scoring, eval data, model catalog) are never
reachable through the tool schema.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from .lane_mutator import (
    FewShotLaneMutator,
    HarnessLaneMutator,
    MutatedAsset,
    PromptLaneMutator,
    TopologyLaneMutator,
    resolve_call_source,
)

LANES = ("topology", "harness", "prompt", "few_shot")
_TOPOLOGY_FIELDS = {"agent_count", "max_candidates_per_agent", "peer_rounds"}
_RUN_CONFIG_DEFAULT_KEYS = {
    "proceed_on_validation_failure",
    "proceed_only_on_arrow_push_failure",
    "candidate_rescue_enabled",
    "retry_same_candidate_max",
    "max_reproposals_per_step",
    "repeat_failure_signature_limit",
}
_PROMPT_START = "<!-- PROMPT_START -->"
_PROMPT_END = "<!-- PROMPT_END -->"


# ---------------------------------------------------------------------------
# Failure digest
# ---------------------------------------------------------------------------
def _short(text: Any, limit: int = 200) -> str:
    value = str(text or "")
    return value if len(value) <= limit else value[: limit - 3] + "..."


def build_failure_digest(snapshots: Iterable[Dict[str, Any]], *, max_runs: int = 8) -> List[Dict[str, Any]]:
    """Summarise what went wrong in each run snapshot (deterministic, compact).

    Each entry records the run status, the target products, the final state,
    every failed validator check with its error text, reproposal reasons,
    rescue outcomes and soft-advances. Only model-visible chemistry (SMILES,
    check names, error strings) is included — never eval ground truth.
    """
    digest: List[Dict[str, Any]] = []
    for snapshot in list(snapshots)[:max_runs]:
        if not isinstance(snapshot, dict):
            continue
        events = [row for row in (snapshot.get("events") or []) if isinstance(row, dict)]
        step_outputs = [row for row in (snapshot.get("step_outputs") or []) if isinstance(row, dict)]
        run_input = snapshot.get("input") if isinstance(snapshot.get("input"), dict) else {}

        failed_checks: Dict[str, int] = {}
        failure_examples: List[Dict[str, Any]] = []
        accepted_steps = 0
        for row in step_outputs:
            if str(row.get("step_name") or "") != "mechanism_synthesis":
                continue
            validation = row.get("validation") if isinstance(row.get("validation"), dict) else {}
            if validation.get("passed") is True:
                accepted_steps += 1
                continue
            for check in validation.get("checks") or []:
                if not isinstance(check, dict) or check.get("passed") is not False:
                    continue
                name = str(check.get("name") or "unknown")
                failed_checks[name] = failed_checks.get(name, 0) + 1
                details = check.get("details") if isinstance(check.get("details"), dict) else {}
                error = details.get("error") or details.get("message") or details.get("reason") or ""
                if len(failure_examples) < 6:
                    output = row.get("output") if isinstance(row.get("output"), dict) else {}
                    failure_examples.append(
                        {
                            "check": name,
                            "error": _short(error),
                            "current_state": list(output.get("current_state") or [])[:6],
                            "resulting_state": list(output.get("resulting_state") or [])[:6],
                            "reaction_smirks": _short(output.get("reaction_smirks"), 160),
                        }
                    )

        reproposal_reasons: Dict[str, int] = {}
        rescue_outcomes: Dict[str, int] = {}
        soft_advances: Dict[str, int] = {}
        terminal_reason: Optional[str] = None
        for row in events:
            et = str(row.get("event_type") or "")
            payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
            if et == "mechanism_reproposal_requested":
                reason = str(payload.get("reason") or "unknown")
                reproposal_reasons[reason] = reproposal_reasons.get(reason, 0) + 1
            elif et == "candidate_rescue_completed":
                status = str(payload.get("status") or "unknown")
                rescue_outcomes[status] = rescue_outcomes.get(status, 0) + 1
            elif et == "mechanism_step_soft_advance":
                reason = str(payload.get("reason") or "unknown")
                soft_advances[reason] = soft_advances.get(reason, 0) + 1
            elif et in {"run_failed", "run_paused"}:
                terminal_reason = f"{et}:{payload.get('reason') or ''}"

        digest.append(
            {
                "run_id": str(snapshot.get("id") or snapshot.get("run_id") or ""),
                "status": str(snapshot.get("status") or ""),
                "terminal_reason": terminal_reason,
                "starting_materials": list(run_input.get("starting_materials") or [])[:6],
                "target_products": list(run_input.get("products") or [])[:6],
                "accepted_steps": accepted_steps,
                "failed_checks": failed_checks,
                "failure_examples": failure_examples,
                "reproposal_reasons": reproposal_reasons,
                "rescue_outcomes": rescue_outcomes,
                "soft_advances": soft_advances,
            }
        )
    return digest


def digest_from_case_results(case_results: Iterable[Dict[str, Any]], *, max_runs: int = 8) -> List[Dict[str, Any]]:
    """Adapt evolve_harness case-result dicts (step_outputs + error) into a digest."""
    snapshots: List[Dict[str, Any]] = []
    for result in list(case_results)[:max_runs]:
        if not isinstance(result, dict):
            continue
        snapshots.append(
            {
                "id": result.get("run_id") or result.get("case_id"),
                "status": result.get("run_status") or ("completed" if result.get("passed") else "failed"),
                "input": result.get("input_payload") or {},
                "step_outputs": result.get("step_outputs") or [],
                "events": (
                    [{"event_type": "run_failed", "payload": {"reason": result.get("error")}}]
                    if result.get("error")
                    else []
                ),
            }
        )
    return build_failure_digest(snapshots, max_runs=max_runs)


# ---------------------------------------------------------------------------
# Proposal application
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class MutationProposal:
    lane: str
    target: str
    operation: str
    value: Any
    rationale: str
    expected_effect: str = ""
    old_text: str = ""

    @classmethod
    def from_arguments(cls, args: Dict[str, Any]) -> "MutationProposal":
        return cls(
            lane=str(args.get("lane") or "").strip(),
            target=str(args.get("target") or "").strip(),
            operation=str(args.get("operation") or "").strip(),
            value=args.get("value"),
            rationale=str(args.get("rationale") or "").strip(),
            expected_effect=str(args.get("expected_effect") or "").strip(),
            old_text=str(args.get("old_text") or ""),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "lane": self.lane,
            "target": self.target,
            "operation": self.operation,
            "value": self.value,
            "rationale": self.rationale,
            "expected_effect": self.expected_effect,
            "old_text": self.old_text,
        }


class ProposalRejected(ValueError):
    """The model's proposal is not applicable (bad lane/target/operation)."""


def _find_module(payload: Dict[str, Any], module_id: str) -> Optional[Dict[str, Any]]:
    for section in ("pre_loop_modules", "post_step_modules", "post_loop_modules"):
        for module in payload.get(section) or []:
            if isinstance(module, dict) and str(module.get("id") or "") == module_id:
                return module
    return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "on", "enable", "enabled"}


class LLMLaneMutator:
    """Propose one trace-conditioned mutation via a forced tool call and apply it.

    ``chat_model_factory`` defaults to :func:`mechanistic_agent.llm.get_chat_model`
    so the proposer can be any catalog model, including the keyless agent bridge.
    When the model's proposal cannot be applied, the blind mutator for the
    requested lane is used instead and the summary records the fallback.
    """

    def __init__(
        self,
        *,
        base_dir: Path,
        model_name: str,
        chat_model_factory: Optional[Callable[..., Any]] = None,
        call_names: Optional[List[str]] = None,
        target_model_name: Optional[str] = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.model_name = str(model_name)
        # The model the evaluated runs use: prompt / few-shot edits are derived
        # from (and scoped to) the asset a run for this model resolves.
        self.target_model_name = str(target_model_name or "").strip() or None
        self._factory = chat_model_factory
        self.call_names = list(call_names or ["propose_mechanism_step", "attempt_atom_mapping", "assess_initial_conditions", "predict_missing_reagents", "select_reaction_type"])
        self.last_proposal: Optional[MutationProposal] = None
        self.last_error: Optional[str] = None

    # -- prompt -------------------------------------------------------------
    def _asset_summary(self, parent_asset_path: Path) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        try:
            payload = json.loads(parent_asset_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            summary["harness_name"] = payload.get("name")
            summary["topology_profiles"] = payload.get("topology_profiles") or {}
            summary["run_config_defaults"] = payload.get("run_config_defaults") or {}
            modules = []
            for section in ("pre_loop_modules", "post_step_modules", "post_loop_modules"):
                for module in payload.get(section) or []:
                    if isinstance(module, dict):
                        modules.append(
                            {
                                "id": module.get("id"),
                                "section": section,
                                "kind": module.get("kind"),
                                "enabled": module.get("enabled"),
                                "removable": module.get("removable"),
                            }
                        )
            summary["modules"] = modules
        prompts: Dict[str, str] = {}
        few_shot_counts: Dict[str, int] = {}
        for call_name in self.call_names:
            skill, _scope = resolve_call_source(self.base_dir, call_name, "prompt", self.target_model_name)
            if skill.exists():
                text = skill.read_text(encoding="utf-8")
                if _PROMPT_START in text and _PROMPT_END in text:
                    body = text.split(_PROMPT_START, 1)[1].split(_PROMPT_END, 1)[0]
                    prompts[call_name] = _short(body.strip(), 1800)
            few, _scope = resolve_call_source(self.base_dir, call_name, "few_shot", self.target_model_name)
            if few.exists():
                few_shot_counts[call_name] = sum(1 for line in few.read_text(encoding="utf-8").splitlines() if line.strip())
        summary["prompts"] = prompts
        summary["few_shot_counts"] = few_shot_counts
        return summary

    def _messages(self, *, allowed_lanes: List[str], failure_digest: List[Dict[str, Any]], asset_summary: Dict[str, Any]) -> List[Dict[str, str]]:
        system = (
            "You improve an organic-reaction-mechanism prediction harness by proposing exactly ONE "
            "targeted, machine-applicable change. You are given a failure digest from the last "
            "evaluation slice (failed deterministic validator checks with error text, reproposal "
            "reasons, rescue outcomes, soft-advances) and a summary of the current editable assets. "
            "Diagnose the most frequent or most damaging failure pattern and propose the single edit "
            "most likely to remove it. You may only edit: a topology profile integer field, a harness "
            "module enabled flag or run_config_defaults key, a call prompt (append or replace an "
            "instruction between the prompt markers), or a few-shot lane (remove an example by index or "
            "add one as a JSON object with input/output). Never propose changes to validators, scoring, "
            "eval data or the model catalog — they are not reachable. Prefer prompt/few-shot edits when "
            "the model produced wrong chemistry or malformed SMILES; prefer harness/topology edits when the "
            "chemistry was right but the harness bookkeeping rejected it. Keep prompt edits short and "
            "specific; do not restate existing instructions."
        )
        user = json.dumps(
            {
                "allowed_lanes": allowed_lanes,
                "failure_digest": failure_digest,
                "current_assets": asset_summary,
                "operations": {
                    "topology": "set_field: target='<profile>.<field>' (agent_count|max_candidates_per_agent|peer_rounds), value=int",
                    "harness": "set_enabled: target='<module_id>', value=true|false  OR  set_run_config_default: target='<key>', value",
                    "prompt": "append_instruction: target='<call_name>', value='<one instruction line>'  OR  replace_instruction: target='<call_name>', old_text='<exact text>', value='<replacement>'",
                    "few_shot": "remove_few_shot: target='<call_name>', value=<0-based index>  OR  add_few_shot: target='<call_name>', value={\"input\": ..., \"output\": ...}",
                },
            },
            indent=1,
            default=str,
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    # -- model call ---------------------------------------------------------
    def _ask_model(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        from mechanistic_agent.tool_schemas import HARNESS_MUTATION_TOOL, build_tool_choice

        if self._factory is not None:
            llm = self._factory(self.model_name)
        else:
            from mechanistic_agent.llm import get_chat_model

            llm = get_chat_model(self.model_name)
        response = llm.invoke(
            messages,
            tools=[HARNESS_MUTATION_TOOL],
            tool_choice=build_tool_choice("harness_mutation_proposal"),
        )
        tool_calls = getattr(response, "tool_calls", None) or []
        if tool_calls:
            raw = tool_calls[0].get("arguments") if isinstance(tool_calls[0], dict) else None
            if isinstance(raw, dict):
                return raw
            if isinstance(raw, str):
                return json.loads(raw)
        content = getattr(response, "content", None)
        if isinstance(content, str) and content.strip():
            text = content.strip()
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
        raise ProposalRejected("model returned no tool call and no JSON content")

    # -- application --------------------------------------------------------
    def apply(self, proposal: MutationProposal, parent_asset_path: Path) -> MutatedAsset:
        lane = proposal.lane
        stamp = int(time.time())
        if lane not in LANES:
            raise ProposalRejected(f"unknown lane {lane!r}")

        if lane in {"topology", "harness"}:
            payload = json.loads(parent_asset_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ProposalRejected("harness asset is not a JSON object")
            if lane == "topology":
                if proposal.operation != "set_field" or "." not in proposal.target:
                    raise ProposalRejected("topology proposals must be set_field on '<profile>.<field>'")
                profile_key, field = proposal.target.rsplit(".", 1)
                if field not in _TOPOLOGY_FIELDS:
                    raise ProposalRejected(f"topology field {field!r} not editable")
                profiles = payload.get("topology_profiles") or {}
                if profile_key not in profiles:
                    raise ProposalRejected(f"unknown topology profile {profile_key!r}")
                try:
                    new_value = int(proposal.value)
                except (TypeError, ValueError) as exc:
                    raise ProposalRejected(f"topology value must be an integer: {exc}") from exc
                lower = 0 if field == "peer_rounds" else 1
                new_value = max(lower, min(new_value, 6))
                old_value = profiles[profile_key].get(field)
                if new_value == old_value:
                    raise ProposalRejected("topology proposal is a no-op")
                profiles[profile_key][field] = new_value
                payload["topology_profiles"] = profiles
                summary = f"{profile_key}.{field}: {old_value} -> {new_value}"
                out_path = parent_asset_path.with_name(f"{parent_asset_path.stem}.llm_topology_mutated.json")
            else:
                if proposal.operation == "set_enabled":
                    module = _find_module(payload, proposal.target)
                    if module is None:
                        raise ProposalRejected(f"unknown module {proposal.target!r}")
                    if module.get("removable") is False and not _as_bool(proposal.value):
                        raise ProposalRejected(f"module {proposal.target!r} is load-bearing")
                    new_flag = _as_bool(proposal.value)
                    if bool(module.get("enabled", True)) == new_flag:
                        raise ProposalRejected("harness proposal is a no-op")
                    module["enabled"] = new_flag
                    summary = f"module {proposal.target}: enabled -> {new_flag}"
                elif proposal.operation == "set_run_config_default":
                    if proposal.target not in _RUN_CONFIG_DEFAULT_KEYS:
                        raise ProposalRejected(f"run_config_defaults key {proposal.target!r} not editable")
                    defaults = dict(payload.get("run_config_defaults") or {})
                    value: Any = proposal.value
                    if proposal.target in {"proceed_on_validation_failure", "proceed_only_on_arrow_push_failure", "candidate_rescue_enabled"}:
                        value = _as_bool(value)
                    else:
                        value = int(value)
                    if defaults.get(proposal.target) == value:
                        raise ProposalRejected("harness proposal is a no-op")
                    defaults[proposal.target] = value
                    payload["run_config_defaults"] = defaults
                    summary = f"run_config_defaults.{proposal.target} -> {value}"
                else:
                    raise ProposalRejected(f"unsupported harness operation {proposal.operation!r}")
                out_path = parent_asset_path.with_name(f"{parent_asset_path.stem}.llm_harness_mutated.json")
            changelog = payload.setdefault("metadata", {}).setdefault("changelog", [])
            changelog.append(
                {
                    "version": len(changelog) + 1,
                    "date": time.strftime("%Y-%m-%d"),
                    "description": f"LLM-proposed ({self.model_name}) {lane} mutation: {summary}. Rationale: {_short(proposal.rationale, 300)}",
                }
            )
            out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return MutatedAsset(lane=lane, asset_path=out_path, summary=summary, metadata={"proposal": proposal.as_dict(), "proposer": self.model_name})

        call_name = proposal.target
        if not call_name or "/" in call_name or ".." in call_name:
            raise ProposalRejected(f"invalid call name {call_name!r}")
        if lane == "prompt":
            src, scope_model = resolve_call_source(self.base_dir, call_name, "prompt", self.target_model_name)
            if not src.exists():
                raise ProposalRejected(f"prompt source not found: {src}")
            text = src.read_text(encoding="utf-8")
            if _PROMPT_START not in text or _PROMPT_END not in text:
                raise ProposalRejected("prompt markers not found")
            instruction = str(proposal.value or "").strip()
            if not instruction:
                raise ProposalRejected("empty prompt instruction")
            if proposal.operation == "append_instruction":
                mutated = text.replace(_PROMPT_END, f"\n- {instruction}\n{_PROMPT_END}")
                summary = f"appended instruction to {call_name}: {_short(instruction, 120)}"
            elif proposal.operation == "replace_instruction":
                head, rest = text.split(_PROMPT_START, 1)
                body, tail = rest.split(_PROMPT_END, 1)
                if not proposal.old_text or body.count(proposal.old_text) != 1:
                    raise ProposalRejected("old_text must occur exactly once inside the prompt block")
                mutated = head + _PROMPT_START + body.replace(proposal.old_text, instruction) + _PROMPT_END + tail
                summary = f"replaced instruction in {call_name}: {_short(proposal.old_text, 60)} -> {_short(instruction, 60)}"
            else:
                raise ProposalRejected(f"unsupported prompt operation {proposal.operation!r}")
            out_path = parent_asset_path.with_name(f"prompt_variant_{call_name}_{stamp}.SKILL.md")
            out_path.write_text(mutated, encoding="utf-8")
            return MutatedAsset(lane=lane, asset_path=out_path, summary=summary, metadata={"proposal": proposal.as_dict(), "call_name": call_name, "source": str(src), "scope_model": scope_model, "proposer": self.model_name})

        # few_shot
        src, scope_model = resolve_call_source(self.base_dir, call_name, "few_shot", self.target_model_name)
        lines = [line for line in src.read_text(encoding="utf-8").splitlines() if line.strip()] if src.exists() else []
        if proposal.operation == "remove_few_shot":
            try:
                index = int(proposal.value)
            except (TypeError, ValueError) as exc:
                raise ProposalRejected(f"remove_few_shot needs an integer index: {exc}") from exc
            if not (0 <= index < len(lines)):
                raise ProposalRejected(f"few-shot index {index} out of range ({len(lines)} examples)")
            del lines[index]
            summary = f"removed few-shot example {index} from {call_name}"
        elif proposal.operation == "add_few_shot":
            example = proposal.value
            if isinstance(example, str):
                example = json.loads(example)
            if not isinstance(example, dict) or "input" not in example or "output" not in example:
                raise ProposalRejected("add_few_shot value must be an object with input and output")
            payload_line = {
                "input": example["input"] if isinstance(example["input"], str) else json.dumps(example["input"], sort_keys=True),
                "output": example["output"] if isinstance(example["output"], str) else json.dumps(example["output"], sort_keys=True),
                "source_run_quality": "clean",
            }
            lines.append(json.dumps(payload_line, sort_keys=True))
            summary = f"added few-shot example to {call_name}"
        else:
            raise ProposalRejected(f"unsupported few_shot operation {proposal.operation!r}")
        out_path = parent_asset_path.with_name(f"few_shot_variant_{call_name}_{stamp}.jsonl")
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return MutatedAsset(lane=lane, asset_path=out_path, summary=summary, metadata={"proposal": proposal.as_dict(), "call_name": call_name, "source": str(src), "scope_model": scope_model, "proposer": self.model_name})

    # -- entry point --------------------------------------------------------
    def propose(
        self,
        parent_asset_path: Path,
        *,
        failure_digest: Optional[List[Dict[str, Any]]] = None,
        allowed_lanes: Optional[List[str]] = None,
        preferred_lane: Optional[str] = None,
    ) -> MutatedAsset:
        lanes = [lane for lane in (allowed_lanes or list(LANES)) if lane in LANES] or list(LANES)
        if preferred_lane in lanes:
            lanes = [preferred_lane] + [lane for lane in lanes if lane != preferred_lane]
        self.last_proposal = None
        self.last_error = None
        try:
            messages = self._messages(
                allowed_lanes=lanes,
                failure_digest=list(failure_digest or []),
                asset_summary=self._asset_summary(parent_asset_path),
            )
            proposal = MutationProposal.from_arguments(self._ask_model(messages))
            self.last_proposal = proposal
            if proposal.lane not in lanes:
                raise ProposalRejected(f"lane {proposal.lane!r} not allowed (allowed: {lanes})")
            return self.apply(proposal, parent_asset_path)
        except (ProposalRejected, json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
            self.last_error = str(exc)
            fallback_lane = preferred_lane if preferred_lane in lanes else lanes[0]
            fallback = self._blind_fallback(fallback_lane, parent_asset_path)
            fallback.summary = f"[llm proposal rejected: {_short(exc, 120)}] {fallback.summary}"
            fallback.metadata = {**(fallback.metadata or {}), "llm_error": str(exc), "llm_proposal": self.last_proposal.as_dict() if self.last_proposal else None}
            return fallback

    def _blind_fallback(self, lane: str, parent_asset_path: Path) -> MutatedAsset:
        if lane == "topology":
            return TopologyLaneMutator().propose(parent_asset_path)
        if lane == "harness":
            return HarnessLaneMutator().propose(parent_asset_path)
        if lane == "prompt":
            return PromptLaneMutator(base_dir=self.base_dir, model_name=self.target_model_name).propose(parent_asset_path)
        return FewShotLaneMutator(base_dir=self.base_dir, model_name=self.target_model_name).propose(parent_asset_path)
