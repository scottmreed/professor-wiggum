"""Reaction-type selection as one Jev ``Choice`` (PRD §7.3, §16.5).

Behind ``decision_policy.reaction_type == "jev"``. Produces the same output
contract as :func:`mechanistic_agent.tools.select_reaction_type`
(``selected_type_id``, ``selected_label_exact``, ``confidence``,
``top_candidates``), so ``RunCoordinator._apply_reaction_type_selection`` and
its confidence/margin gates work unchanged. ``confidence`` is the Jev
probability of the selected option and ``top_candidates[].confidence`` are the
Jev probabilities, so the margin gate compares probabilities. Jev's own
``confidence`` field is kept as ``jev_confidence``.

The Jev ``state`` is machine-built only: canonical SMILES, element counts,
functional-group counts, numeric pH ranges, enum environments and mapping
summary numbers. Free-text fields from earlier LLM calls (justifications,
rationales, warnings, commentary) are never copied into it (PRD §7.0,
injection via state).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional

from mechanistic_agent.decisions.jev import DecisionQuestion, DecisionRecord
from mechanistic_agent.decisions.policies import choice_margin, top_candidates
from mechanistic_agent.smiles_utils import strip_atom_mapping_optional

from .reaction_type_templates import (
    compact_template_for_prompt,
    list_reaction_type_choices,
    load_reaction_type_catalog_for_runtime,
)
from .types import JevConfig

QUESTION_ID = "reaction_type"
NO_MATCH = "no_match"
STATE_SCHEMA = "mechanistic.jev.reaction_type_state@1"

_ENVIRONMENTS = {"acidic", "basic", "neutral", "mixed", "unclear"}
_MAX_LIST = 12


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
def _smiles(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    return strip_atom_mapping_optional(value.strip()) or value.strip()


def _smiles_list(values: Any) -> List[str]:
    out: List[str] = []
    if isinstance(values, list):
        for item in values[:_MAX_LIST]:
            if isinstance(item, dict):
                item = item.get("smiles")
            text = _smiles(item)
            if text:
                out.append(text)
    return out


def _counts(value: Any) -> Dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    out: Dict[str, int] = {}
    for key, raw in value.items():
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        out[str(key)] = int(raw)
    return out


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _range(value: Any) -> Optional[List[float]]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        low, high = _number(value[0]), _number(value[1])
        if low is not None and high is not None:
            return [low, high]
    return None


def _balance_state(balance: Any) -> Dict[str, Any]:
    if not isinstance(balance, Mapping):
        return {}
    inner = balance.get("rdkit") if isinstance(balance.get("rdkit"), Mapping) else balance
    out: Dict[str, Any] = {}
    if isinstance(inner.get("balanced"), bool):
        out["balanced"] = inner["balanced"]
    for key in ("reactant_counts", "product_counts", "deficit", "surplus"):
        counts = _counts(inner.get(key))
        if counts or key in inner:
            out[key] = counts
    return out


def _functional_group_state(functional_groups: Any) -> Dict[str, Dict[str, int]]:
    if not isinstance(functional_groups, Mapping):
        return {}
    groups = functional_groups.get("functional_groups", functional_groups)
    if not isinstance(groups, Mapping):
        return {}
    out: Dict[str, Dict[str, int]] = {}
    for smiles, counts in list(groups.items())[:_MAX_LIST]:
        key = _smiles(smiles)
        parsed = _counts(counts)
        if key and parsed:
            out[key] = parsed
    return out


def _group_labels(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value[:_MAX_LIST] if isinstance(item, str) and len(item) <= 40]


def _conditions_state(ph_recommendation: Any, initial_conditions: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(ph_recommendation, Mapping):
        ph: Dict[str, Any] = {}
        rng = _range(ph_recommendation.get("recommended_range"))
        if rng:
            ph["recommended_range"] = rng
        for key, out_key in (
            ("acidic_score", "acidic_score"),
            ("basic_score", "basic_score"),
            ("recommended", "recommended_ph"),
            ("recommended_ph", "recommended_ph"),
            ("provided_ph", "user_ph"),
        ):
            number = _number(ph_recommendation.get(key))
            if number is not None:
                ph[out_key] = number
        profiles = ph_recommendation.get("profiles")
        if isinstance(profiles, Mapping):
            parsed_profiles: Dict[str, List[str]] = {}
            for smiles, states in list(profiles.items())[:_MAX_LIST]:
                key = _smiles(smiles)
                values = _smiles_list(states)
                if key and values:
                    parsed_profiles[key] = values
            if parsed_profiles:
                ph["protonation_profiles"] = parsed_profiles
        source = ph_recommendation.get("source")
        if isinstance(source, str) and len(source) <= 32:
            ph["source"] = source
        if ph:
            out["ph_recommendation"] = ph
    if isinstance(initial_conditions, Mapping):
        cond: Dict[str, Any] = {}
        env = str(initial_conditions.get("environment") or "").strip().lower()
        if env in _ENVIRONMENTS:
            cond["environment"] = env
        rep = _number(initial_conditions.get("representative_ph"))
        if rep is not None:
            cond["representative_ph"] = rep
        rng = _range(initial_conditions.get("ph_range"))
        if rng:
            cond["ph_range"] = rng
        acids = _smiles_list(initial_conditions.get("acid_candidates"))
        bases = _smiles_list(initial_conditions.get("base_candidates"))
        if acids:
            cond["acid_candidate_smiles"] = acids
        if bases:
            cond["base_candidate_smiles"] = bases
        transformation = initial_conditions.get("functional_group_transformation")
        if isinstance(transformation, Mapping):
            consumed = _group_labels(transformation.get("consumed_groups"))
            formed = _group_labels(transformation.get("formed_groups"))
            if consumed or formed:
                cond["functional_group_change"] = {"consumed": consumed, "formed": formed}
        if cond:
            out["initial_conditions"] = cond
    return out


def _missing_reagents_state(missing: Any) -> Dict[str, Any]:
    if not isinstance(missing, Mapping):
        return {}
    out: Dict[str, Any] = {}
    status = missing.get("status")
    if isinstance(status, str) and len(status) <= 32:
        out["status"] = status
    for key in ("missing_reactants", "missing_products"):
        values = _smiles_list(missing.get(key))
        if values or key in missing:
            out[key] = values
    return out


def _mapping_state(atom_mapping: Any) -> Dict[str, Any]:
    if not isinstance(atom_mapping, Mapping) or not atom_mapping:
        return {}
    from .global_mapping_context import summarize_global_mapping

    summary = summarize_global_mapping(dict(atom_mapping))
    out: Dict[str, Any] = {}
    if summary.get("confidence") is not None:
        out["confidence"] = summary["confidence"]
    response = atom_mapping.get("llm_response")
    pairs = response.get("mapped_atoms") if isinstance(response, Mapping) else None
    if isinstance(pairs, list):
        out["mapped_pair_count"] = len(pairs)
    unmapped = summary.get("unmapped_atoms")
    if isinstance(unmapped, list):
        out["unmapped_atom_count"] = len(unmapped)
    validation = atom_mapping.get("atom_map_validation")
    if isinstance(validation, Mapping):
        for key in ("passed", "skipped"):
            if isinstance(validation.get(key), bool):
                out[f"atom_map_check_{key}"] = validation[key]
    return out


def build_reaction_type_state(
    *,
    starting_materials: List[str],
    products: List[str],
    balance_analysis: Optional[Dict[str, Any]] = None,
    functional_groups: Optional[Dict[str, Any]] = None,
    ph_recommendation: Optional[Dict[str, Any]] = None,
    initial_conditions: Optional[Dict[str, Any]] = None,
    missing_reagents: Optional[Dict[str, Any]] = None,
    atom_mapping: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Machine-only JSON state for the reaction-type Choice.

    Reuses the context the LLM selector receives, reduced to whitelisted
    numeric / SMILES / enum fields. Empty sections are omitted.
    """
    state: Dict[str, Any] = {
        "schema": STATE_SCHEMA,
        "reaction": {
            "starting_materials": _smiles_list(list(starting_materials or [])),
            "products": _smiles_list(list(products or [])),
        },
    }
    sections = {
        "balance": _balance_state(balance_analysis),
        "functional_groups": _functional_group_state(functional_groups),
        "conditions": _conditions_state(ph_recommendation, initial_conditions),
        "missing_reagents": _missing_reagents_state(missing_reagents),
        "atom_mapping": _mapping_state(atom_mapping),
    }
    for key, value in sections.items():
        if value:
            state[key] = value
    return state


# ---------------------------------------------------------------------------
# Question
# ---------------------------------------------------------------------------
QUESTION_INSTRUCTIONS = (
    "The state is machine-built JSON describing one chemical reaction: canonical SMILES of the "
    "starting materials and products, element balance, functional-group counts, pH/conditions "
    "analysis and a mapping summary. Choose the mechanism type from the options that best "
    "describes how the starting materials are converted into the products. Choose no_match when "
    "none of the listed mechanism types fits."
)


def _option_description(template: Mapping[str, Any], label: str, group: str) -> str:
    current = [str(x) for x in (template.get("current_state_generic") or []) if isinstance(x, str)]
    resulting = [str(x) for x in (template.get("resulting_state_generic") or []) if isinstance(x, str)]
    steps = int(template.get("suitable_step_count") or 0)
    parts = [label]
    if group:
        parts.append(f"group: {group}")
    if steps:
        parts.append(f"typical steps: {steps}")
    if current and resulting:
        parts.append(f"generic: {' + '.join(current)} -> {' + '.join(resulting)}")
    return "; ".join(parts)


def build_reaction_type_question(catalog: Optional[Dict[str, Any]] = None) -> DecisionQuestion:
    """One Choice over every taxonomy ``type_id`` plus ``no_match``."""
    catalog = catalog if catalog is not None else load_reaction_type_catalog_for_runtime()
    by_id = dict(catalog.get("by_id") or {})
    options: Dict[str, str] = {}
    for item in list_reaction_type_choices(catalog):
        type_id = str(item.get("type_id") or "").strip()
        if not type_id:
            continue
        options[type_id] = _option_description(
            by_id.get(type_id) or {},
            str(item.get("label_exact") or ""),
            str(item.get("canonical_group") or ""),
        )
    options[NO_MATCH] = "None of the listed mechanism types fits this reaction."
    return DecisionQuestion.choice(QUESTION_ID, QUESTION_INSTRUCTIONS, options)


# ---------------------------------------------------------------------------
# Record -> selector output contract
# ---------------------------------------------------------------------------
def selection_from_record(
    record: DecisionRecord,
    catalog: Dict[str, Any],
    *,
    top_n: int = 5,
) -> Dict[str, Any]:
    """Map a successful Choice record onto the select_reaction_type contract."""
    if not record.ok or not record.probabilities:
        raise ValueError(f"cannot build a selection from a failed record: {record.failure}")
    by_id = dict(catalog.get("by_id") or {})
    probabilities = dict(record.probabilities)
    selected_key = str(record.selected)

    def _label(key: str) -> str:
        if key == NO_MATCH:
            return NO_MATCH
        return str((by_id.get(key) or {}).get("label_exact") or key)

    candidates = []
    for key, prob in top_candidates(probabilities, top_n):
        candidates.append(
            {
                "label_exact": _label(key),
                "type_id": None if key == NO_MATCH else key,
                "confidence": float(prob),
            }
        )
    template = by_id.get(selected_key) if selected_key != NO_MATCH else None
    selected_prob = float(probabilities.get(selected_key, 0.0))
    margin = choice_margin(probabilities)
    margin_text = "n/a" if margin is None else f"{margin:.3f}"
    return {
        "status": "success",
        "selected_label_exact": _label(selected_key) if template is not None else NO_MATCH,
        "selected_type_id": selected_key if template is not None else None,
        "confidence": selected_prob,
        "jev_confidence": record.confidence,
        "rationale": (
            f"Jev choice over {len(probabilities)} options; p(selected)={selected_prob:.3f}; "
            f"top-2 margin={margin_text}."
        ),
        "top_candidates": candidates,
        "selected_template": compact_template_for_prompt(template) if template is not None else None,
        "available_reaction_type_count": len(list(catalog.get("templates") or [])),
        "taxonomy_labels": [str(t.get("label_exact") or "") for t in list(catalog.get("templates") or [])],
        "model_used": record.model,
        "model_version": record.model_version,
        "decision_engine": "jev",
        "tool_calling_used": False,
    }


def _default_client(jev_config: JevConfig) -> Any:
    from mechanistic_agent.llm import get_decision_model

    user_key = None
    try:
        from mechanistic_agent.core.model_context import get_api_key

        user_key = get_api_key("openrouter")
    except Exception:
        user_key = None
    return get_decision_model(
        jev_config.model,
        timeout=jev_config.timeout_seconds,
        user_api_key=user_key,
    )


def select_reaction_type_jev(
    *,
    starting_materials: List[str],
    products: List[str],
    balance_analysis: Optional[Dict[str, Any]] = None,
    functional_groups: Optional[Dict[str, Any]] = None,
    ph_recommendation: Optional[Dict[str, Any]] = None,
    initial_conditions: Optional[Dict[str, Any]] = None,
    missing_reagents: Optional[Dict[str, Any]] = None,
    atom_mapping: Optional[Dict[str, Any]] = None,
    jev_config: Optional[JevConfig] = None,
    client: Any = None,
    catalog: Optional[Dict[str, Any]] = None,
    llm_fallback: Optional[Callable[[], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Select the reaction type with one Jev Choice.

    Returns the selector output dict plus ``decision_trace`` (PRD §18 entries)
    and ``decision_engine``. Private keys ``_decision_usage`` /
    ``_decision_cost`` carry the Jev request usage for the step record. On a
    Jev failure, ``jev_config.fallback == "llm"`` calls ``llm_fallback`` (the
    existing LLM selector) and returns its output with the failed Jev decision
    in ``decision_trace``; ``"no_match"`` returns a ``no_match`` fallback.
    """
    cfg = jev_config if isinstance(jev_config, JevConfig) else JevConfig()
    catalog = catalog if catalog is not None else load_reaction_type_catalog_for_runtime()
    state = build_reaction_type_state(
        starting_materials=starting_materials,
        products=products,
        balance_analysis=balance_analysis,
        functional_groups=functional_groups,
        ph_recommendation=ph_recommendation,
        initial_conditions=initial_conditions,
        missing_reagents=missing_reagents,
        atom_mapping=atom_mapping,
    )
    question = build_reaction_type_question(catalog)

    record: Optional[DecisionRecord] = None
    try:
        jev_client = client if client is not None else _default_client(cfg)
        record = jev_client.decide_many(state, [question])[QUESTION_ID]
    except Exception as exc:  # routing/config errors become a failed decision
        record = DecisionRecord(
            question_id=QUESTION_ID,
            decision_type="choice",
            model=str(cfg.model or "unresolved"),
            failure="client_error",
            failure_detail=f"{type(exc).__name__}: {str(exc)[:300]}",
        )

    if record.ok:
        output = selection_from_record(record, catalog, top_n=cfg.reaction_type_top_n)
        output["decision_trace"] = [record.to_trace()]
        output["_decision_usage"] = record.usage
        output["_decision_cost"] = record.cost
        if record.notes:
            output["decision_notes"] = list(record.notes)
        return output

    reason = str(record.failure or "unknown_failure")
    fallback_mode = cfg.fallback if llm_fallback is not None else "no_match"
    trace = record.to_trace(fallback_triggered=True, fallback_reason=reason)
    if fallback_mode == "llm" and llm_fallback is not None:
        output = dict(llm_fallback() or {})
        output["decision_engine"] = "llm"
        output["decision_trace"] = [trace]
        output["jev_fallback"] = {"mode": "llm", "reason": reason, "detail": record.failure_detail}
        output["_decision_usage"] = record.usage
        output["_decision_cost"] = record.cost
        return output
    return {
        "status": "fallback",
        "selected_label_exact": NO_MATCH,
        "selected_type_id": None,
        "confidence": 0.0,
        "rationale": f"Jev reaction-type decision failed ({reason}); template guidance disabled.",
        "top_candidates": [],
        "selected_template": None,
        "available_reaction_type_count": len(list(catalog.get("templates") or [])),
        "model_used": record.model,
        "decision_engine": "jev",
        "tool_calling_used": False,
        "decision_trace": [trace],
        "jev_fallback": {"mode": "no_match", "reason": reason, "detail": record.failure_detail},
        "_decision_usage": record.usage,
        "_decision_cost": record.cost,
    }


__all__ = [
    "NO_MATCH",
    "QUESTION_ID",
    "build_reaction_type_question",
    "build_reaction_type_state",
    "select_reaction_type_jev",
    "selection_from_record",
]
