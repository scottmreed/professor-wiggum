"""Single pre-analysis call for the ``threaded_unified`` harness.

One forced tool call returns conditions, missing reagents, atom mapping and
the reaction-type label. ``section_presets`` turns the answer into per-tool
arguments so the coordinator can replay each section through the old tool
(``llm.preset_tool_answers``) and keep every downstream contract unchanged.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from mechanistic_agent.core.reaction_type_templates import (
    list_reaction_type_choices,
    load_reaction_type_catalog_for_runtime,
)
from mechanistic_agent.llm import adapter_supports_forced_tools, extract_text_content, get_chat_model
from mechanistic_agent.prompt_assets import compose_system_prompt
from mechanistic_agent.tool_schemas import (
    UNIFIED_PREANALYSIS_SECTIONS,
    UNIFIED_PREANALYSIS_TOOL_NAME,
    build_tool_choice,
    build_unified_preanalysis_tool,
)

CALL_NAME = "unified_preanalysis"
STEP_NAME = "unified_preanalysis"

# Section -> name of the old tool its arguments are replayed through.
SECTION_TOOL_NAMES: Dict[str, str] = {
    name: spec["tool"]["function"]["name"] for name, spec in UNIFIED_PREANALYSIS_SECTIONS.items()
}

_CONTEXT_CHAR_CAP = 4000

_SYSTEM_PROMPT = (
    "You are an expert mechanistic organic chemist. Answer only by calling the "
    f"`{UNIFIED_PREANALYSIS_TOOL_NAME}` tool with every requested section filled in."
)


def sections_for(products_hidden: bool) -> List[str]:
    return ["conditions", "reaction_type"] if products_hidden else list(UNIFIED_PREANALYSIS_SECTIONS)


def _compact_json(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, default=str)
    return text if len(text) <= _CONTEXT_CHAR_CAP else text[: _CONTEXT_CHAR_CAP - 1] + "…"


def _taxonomy_lines() -> List[str]:
    catalog = load_reaction_type_catalog_for_runtime()
    by_id = dict(catalog.get("by_id") or {})
    lines: List[str] = []
    for item in list_reaction_type_choices(catalog):
        type_id = item.get("type_id") or ""
        template = by_id.get(type_id) or {}
        step_hint = int(template.get("suitable_step_count") or 0)
        lines.append(
            f"- {type_id}: {item.get('label_exact')} | group={item.get('canonical_group')} | steps≈{step_hint}"
        )
    return lines


def build_user_prompt(
    *,
    starting_materials: List[str],
    products: List[str],
    products_hidden: bool,
    context: Dict[str, Any],
    user_ph: Optional[float] = None,
    temperature: Optional[float] = None,
) -> str:
    sections = sections_for(products_hidden)
    lines: List[str] = ["Pre-analysis for a mechanism prediction.", f"Starting materials: {starting_materials}"]
    if products_hidden:
        lines.append(
            "Products: not given. The mechanism loop must find them; decide conditions and the "
            "mechanism type from the starting materials alone."
        )
    else:
        lines.append(f"Products: {products}")
    if user_ph is not None:
        lines.append(f"User-supplied pH: {user_ph}")
    if temperature is not None:
        lines.append(f"Temperature (°C): {temperature}")
    lines.append("")
    lines.append("Deterministic context:")
    for key in ("balance_analysis", "functional_groups", "ph_recommendation"):
        value = context.get(key)
        if value:
            lines.append(f"- {key}: {_compact_json(value)}")
    lines.append("")
    lines.append(f"Fill these sections of `{UNIFIED_PREANALYSIS_TOOL_NAME}`: {', '.join(sections)}.")
    lines.append(
        "conditions: environment is 'acidic', 'basic' or 'neutral'; always give a numeric "
        "representative_ph (0-14). Give acid_candidates only when acidic and base_candidates only "
        "when basic (≤3 each, each with name, SMILES, role and a ≤10-word justification). Avoid "
        "reagents that would destroy key functional groups."
    )
    if "missing_reagents" in sections:
        lines.append(
            "missing_reagents: the species to add to each side so the overall reaction balances "
            "under your conditions (empty lists when it already balances). Use the atom counts "
            "above; no speculative additions. SMILES only: water 'O', HCl 'Cl', NaOH "
            "'[Na+].[OH-]', CO2 'O=C=O'; never bracketed formulas such as '[H2O]'."
        )
    if "atom_mapping" in sections:
        lines.append(
            "atom_mapping: mapped_atoms is a list of {\"product_atom\": \"<product_smiles>#<index>\", "
            "\"source\": {\"molecule_index\": int, \"smiles\": str, \"atom_index\": int}, \"notes\": str}. "
            "Use zero-based atom indices in the order implied by the canonical SMILES of each "
            "molecule. List atoms you cannot assign in unmapped_atoms (they may come from the "
            "reagents you added). Give a 0-1 confidence and a short reasoning."
        )
    lines.append(
        "reaction_type: one exact label and its type id from the taxonomy below, or "
        "'no_match' (type id null) when none fits. Add up to 3 alternatives in top_candidates."
    )
    lines.append("")
    lines.append("Allowed taxonomy labels (exact text):")
    lines.extend(_taxonomy_lines())
    return "\n".join(lines)


def build_system_prompt() -> str:
    return compose_system_prompt(call_name=CALL_NAME, dynamic_system_prompt=_SYSTEM_PROMPT)


def assistant_turn_text(answer: Dict[str, Any]) -> str:
    return f"Called `{UNIFIED_PREANALYSIS_TOOL_NAME}` with:\n{json.dumps(answer, sort_keys=True)}"


def _resolve_model() -> str:
    from mechanistic_agent.tools import _resolve_step_model

    return _resolve_step_model(STEP_NAME, "MECHANISTIC_CONDITION_MODEL")


def call_model(
    messages: List[Dict[str, str]],
    *,
    products_hidden: bool,
) -> Dict[str, Any]:
    """One forced tool call. Returns answer, usage, model and any error."""
    from mechanistic_agent.tools import _apply_output_token_cap, _apply_reasoning_kwargs

    model = _resolve_model()
    tool = build_unified_preanalysis_tool(products_hidden=products_hidden)
    result: Dict[str, Any] = {"model_used": model, "answer": None, "usage": None, "error": None}
    try:
        llm_kwargs: Dict[str, Any] = {"model": model}
        _apply_reasoning_kwargs(llm_kwargs, model, os.getenv("MECHANISTIC_CONDITION_REASONING"))
        _apply_output_token_cap(llm_kwargs, model)
        llm = get_chat_model(model, temperature=None, model_kwargs=llm_kwargs.get("model_kwargs"))
        if adapter_supports_forced_tools(model):
            response = llm.invoke(
                messages,
                tools=[tool],
                tool_choice=build_tool_choice(UNIFIED_PREANALYSIS_TOOL_NAME),
            )
        else:
            response = llm.invoke(messages)
    except Exception as exc:
        result["error"] = f"LLM call failed: {exc}"
        return result
    result["usage"] = getattr(response, "usage", None)
    answer: Optional[Dict[str, Any]] = None
    for call in getattr(response, "tool_calls", None) or []:
        if isinstance(call, dict) and call.get("name") == UNIFIED_PREANALYSIS_TOOL_NAME:
            try:
                parsed = json.loads(call.get("arguments") or "{}")
            except (TypeError, json.JSONDecodeError):
                parsed = None
            if isinstance(parsed, dict):
                answer = parsed
    if answer is None:
        text = extract_text_content(response) or ""
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            answer = parsed
        else:
            result["error"] = "No unified_preanalysis_result tool call in the response."
            result["raw_response"] = text[:2000]
    if answer is not None:
        answer.pop("text", None)
        result["answer"] = answer
    return result


def section_presets(answer: Dict[str, Any], sections: List[str]) -> Dict[str, Dict[str, Any]]:
    """Old-tool arguments per section; missing or malformed sections are left out."""
    presets: Dict[str, Dict[str, Any]] = {}
    for section in sections:
        value = answer.get(section) if isinstance(answer, dict) else None
        if isinstance(value, dict):
            presets[SECTION_TOOL_NAMES[section]] = dict(value)
    return presets


def repair_prompt(failures: Dict[str, List[str]]) -> str:
    lines = [
        "Deterministic checks rejected parts of your pre-analysis. Call "
        f"`{UNIFIED_PREANALYSIS_TOOL_NAME}` again with every section; fix the failed sections "
        "and keep the others unless the fix requires changing them.",
    ]
    for section, errors in failures.items():
        lines.append(f"- {section}: " + "; ".join(errors[:4]))
    return "\n".join(lines)
