"""Dynamic system prompt builder for the mechanistic agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .model_registry import get_model_spec


@dataclass(frozen=True)
class _StepPlan:
    tool_name: str
    step_key: str | None
    directive: str


_STEP_SEQUENCE: tuple[_StepPlan, ...] = (
    _StepPlan(
        "analyse_balance",
        None,
        "Call `analyse_balance` to confirm stoichiometry before using other tools",
    ),
    _StepPlan(
        "fingerprint_functional_groups",
        "functional_groups",
        "Run `fingerprint_functional_groups` on all provided SMILES (combine reactants and products when practical) to map reactive sites",
    ),
    _StepPlan(
        "assess_initial_conditions",
        "initial_conditions",
        "Call `assess_initial_conditions` to gauge acidic/basic bias, recommend compatible acids/bases, and keep its JSON handy for `predict_missing_reagents`",
    ),
    _StepPlan(
        "predict_missing_reagents",
        "missing_reagents",
        "When imbalances remain, invoke `predict_missing_reagents` and pass the prior conditions JSON via the `conditions_guidance` argument",
    ),
    _StepPlan(
        "attempt_atom_mapping",
        "atom_mapping",
        "Request `attempt_atom_mapping` and weave in functional-group and reagent insights so numbering aligns with candidates",
    ),
    _StepPlan(
        "select_reaction_type",
        "reaction_type_mapping",
        "Call `select_reaction_type` with prior analysis outputs and allow 'no_match' when taxonomy confidence is low",
    ),
)


def _normalise_tool_name(tool: object) -> str:
    name = getattr(tool, "name", None)
    if isinstance(name, str) and name:
        return name
    return str(tool)


def _format_model_hint(
    step_key: str | None,
    step_models: Mapping[str, str],
    primary_model: str,
) -> str:
    if step_key is None:
        return ""

    model_name = step_models.get(step_key, primary_model)
    try:
        spec = get_model_spec(model_name)
        label = spec.get("label", model_name)
    except Exception:  # pragma: no cover - defensive
        return f" (model: {model_name})"

    if model_name == primary_model:
        return f" (model: {label})"

    try:
        primary_spec = get_model_spec(primary_model)
        primary_cost = float(primary_spec.get("pricing_per_million", {}).get("input", 0.0))
    except Exception:  # pragma: no cover - defensive
        primary_cost = 0.0

    step_cost = float(spec.get("pricing_per_million", {}).get("input", 0.0))
    if primary_cost and step_cost:
        if step_cost < primary_cost:
            return f" (model: {label} — budget tier; keep prompts concise)"
        if step_cost > primary_cost:
            return f" (model: {label} — premium tier; justify extra detail)"

    return f" (model: {label} — alternate selection)"


def build_system_prompt(
    tools: Sequence[object],
    step_models: Mapping[str, str],
    *,
    primary_model: str,
    include_recommend_ph: bool,
) -> str:
    """Generate a system prompt tailored to the available tooling and models."""

    active_tools = {_normalise_tool_name(tool) for tool in tools}

    intro = [
        "You are Mechanistic Loop, a chemistry assistant guiding stepwise reaction reasoning.",
        "Coordinate with tools in the order below; skip any tool that is unavailable.",
    ]

    numbered: list[str] = []
    index = 1
    for plan in _STEP_SEQUENCE:
        if plan.tool_name not in active_tools:
            continue
        hint = _format_model_hint(plan.step_key, step_models, primary_model)
        numbered.append(f"{index}. {plan.directive}{hint}.")
        index += 1

    if include_recommend_ph and "recommend_ph" in active_tools:
        numbered.append(
            "Use `recommend_ph` only when no pH was supplied to suggest a feasible range."
        )
        if "assess_initial_conditions" in active_tools:
            numbered.append(
                "Do not forward the `assess_initial_conditions` payload to `recommend_ph`; evaluate that step independently."
            )

    mechanism_loop: list[str] = []
    if "propose_intermediates" in active_tools or "predict_mechanistic_step" in active_tools:
        loop_hint = _format_model_hint("intermediates", step_models, primary_model)
        synth_hint = _format_model_hint("mechanism_synthesis", step_models, primary_model)
        mechanism_loop.extend(
            [
                "After gathering context, initialise a mechanism loop with `current_state` equal to the starting materials plus any essential reagents you introduced.",
                "For up to ten iterations:",
            ]
        )
        if "propose_intermediates" in active_tools:
            mechanism_loop.append(
                f"  a. Call `propose_intermediates` to request the next forward-moving intermediate or confirm the products{loop_hint}."
            )
        if "predict_mechanistic_step" in active_tools:
            mechanism_loop.append(
                f"  b. Call `predict_mechanistic_step` to narrate electron pushing for that transition{synth_hint}. REQUIRED parameters: electron_pushes (explicit move objects with kind/source/target fields), step_index, current_state, target_products, and reaction_smirks encoded as CXSMILES or SMIRKS with a '|mech:v1;...|' block."
            )
            mechanism_loop.append(
                "  CRITICAL: You MUST call `predict_mechanistic_step` for EACH mechanistic step. "
                "Text-only responses describing mechanisms are NOT ACCEPTED during the mechanism loop. "
                "If you provide text without calling the tool, your response will be rejected and you will be asked to retry with a proper tool call."
            )

    safeguards = [
        "Always cite tool outputs, propagate atom mappings and reagent suggestions, and keep reasoning concise.",
        "Each mechanistic step must include at least one explicit electron move using `kind`, `target_atom`, and either `source_atom` or `source_bond` plus `through_atom`.",
        "When calling `predict_mechanistic_step`, ALWAYS provide the electron_pushes parameter as explicit move objects.",
        "MANDATORY TOOL USAGE: During mechanism synthesis (after initial analysis), you MUST call `predict_mechanistic_step` for each step. "
        "Text-only responses will be REJECTED and you will be forced to retry with a proper tool call. "
        "The only exception is the final summary after all target products are reached or 10 steps are completed.",
        "Never return an unchanged `resulting_state`; every accepted step must advance the chemistry.",
        "Stop once all target products are present or after ten steps, then summarise the accepted pathway succinctly.",
        "CRITICAL: When proposing intermediates, provide ONLY pure SMILES strings without any commentary, descriptors, or parenthetical text. Do not include phrases like '(hemiacetal-like adduct)' or any explanatory text within the SMILES field.",
    ]

    if "assess_initial_conditions" in active_tools:
        safeguards.insert(
            4,
            "Treat the `assess_initial_conditions` representative pH as guidance for every mechanistic step; call out deliberate deviations.",
        )

    safeguards.extend(
        [
            "Each reaction SMIRKS supplied to `predict_mechanistic_step` must append '|mech:v1;...|' using atom-map indices with tokens such as lp:4>2, pi:1-2>2, or sigma:2-3>3.",
            "Example CXSMILES format: [CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] |mech:v1;lp:4>2;sigma:2-3>3|.",
            "If the tool flags mechanism-move metadata warnings, repair the format in the next loop iteration while continuing forward progress.",
        ]
    )

    sections = [
        "\n".join(intro),
        "\n".join(numbered) if numbered else "",
        "\n".join(mechanism_loop) if mechanism_loop else "",
        "\n".join(safeguards),
    ]

    content = "\n\n".join(filter(None, sections))
    return content.strip()


__all__ = ["build_system_prompt"]
