"""Reaction-type template catalog helpers."""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional


def _repo_base(base_dir: Optional[Path] = None) -> Path:
    if base_dir is not None:
        return base_dir.resolve()
    env_root = str(os.getenv("MECHANISTIC_PROJECT_ROOT") or "").strip()
    if env_root:
        return Path(env_root).resolve()
    # Fall back to repository root relative to this module, not process CWD.
    return Path(__file__).resolve().parents[2]


def _reaction_type_path(base_dir: Optional[Path] = None) -> Path:
    return _repo_base(base_dir) / "training_data" / "reaction_type_templates.json"


def _normalise_template(template: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(template)
    data["type_id"] = str(data.get("type_id") or "").strip()
    data["label_exact"] = str(data.get("label_exact") or "").strip()
    data["slug"] = str(data.get("slug") or "").strip()
    data["canonical_group"] = str(data.get("canonical_group") or "").strip()
    data["suitable_step_count"] = int(data.get("suitable_step_count") or 0)
    data["generic_mechanism_steps"] = list(data.get("generic_mechanism_steps") or [])
    return data


def _type_sort_key(type_id: str) -> tuple[int, str]:
    match = re.fullmatch(r"mt_(\d+)", str(type_id))
    if not match:
        return (10_000_000, str(type_id))
    return (int(match.group(1)), str(type_id))


def _normalise_example_mapping(
    raw: Dict[str, Any],
    *,
    by_id: Dict[str, Dict[str, Any]],
    by_label: Dict[str, Dict[str, Any]],
    reaction_id_hint: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    reaction_id = str(raw.get("reaction_id") or reaction_id_hint or "").strip()
    if not reaction_id:
        return None

    label = str(raw.get("mechanism_type_label") or raw.get("selected_label_exact") or "").strip()
    type_id = str(raw.get("mechanism_type_id") or raw.get("selected_type_id") or "").strip()

    template: Optional[Dict[str, Any]] = None
    if type_id and type_id in by_id:
        template = by_id[type_id]
    elif label and label in by_label:
        template = by_label[label]
    else:
        return None

    confidence_raw = raw.get("confidence")
    confidence: Optional[float]
    if isinstance(confidence_raw, (int, float)):
        confidence = max(0.0, min(1.0, float(confidence_raw)))
    else:
        confidence = None

    return {
        "reaction_id": reaction_id,
        "mechanism_type_label": str(template.get("label_exact") or ""),
        "mechanism_type_id": str(template.get("type_id") or ""),
        "confidence": confidence,
        "rationale": str(raw.get("rationale") or "").strip(),
    }


@lru_cache(maxsize=8)
def load_reaction_type_catalog_cached(base_dir_text: str) -> Dict[str, Any]:
    return load_reaction_type_catalog(Path(base_dir_text))


def load_reaction_type_catalog(base_dir: Optional[Path] = None) -> Dict[str, Any]:
    path = _reaction_type_path(base_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing reaction type template file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("reaction_type_templates.json must be a JSON object")

    templates = payload.get("templates")
    if not isinstance(templates, list):
        raise ValueError("reaction_type_templates.json must contain a 'templates' array")

    normalised = [_normalise_template(item) for item in templates if isinstance(item, dict)]
    if not normalised:
        raise ValueError("reaction_type_templates.json contains no templates")

    by_id: Dict[str, Dict[str, Any]] = {}
    by_label: Dict[str, Dict[str, Any]] = {}
    for item in normalised:
        type_id = str(item.get("type_id") or "")
        label = str(item.get("label_exact") or "")
        if not type_id or not label:
            raise ValueError(f"Template missing required keys type_id/label_exact: {item}")
        if type_id in by_id:
            raise ValueError(f"Duplicate reaction-type id in catalog: {type_id}")
        if label in by_label:
            raise ValueError(f"Duplicate reaction-type label in catalog: {label}")
        by_id[type_id] = item
        by_label[label] = item

    ordered_templates = sorted(normalised, key=lambda item: _type_sort_key(str(item.get("type_id") or "")))
    taxonomy_labels = [str(item.get("label_exact") or "") for item in ordered_templates]

    raw_example_mappings = payload.get("example_mappings")
    example_rows: List[Dict[str, Any]] = []
    if isinstance(raw_example_mappings, list):
        for item in raw_example_mappings:
            if not isinstance(item, dict):
                continue
            normalised_row = _normalise_example_mapping(item, by_id=by_id, by_label=by_label)
            if normalised_row is not None:
                example_rows.append(normalised_row)
    elif isinstance(raw_example_mappings, dict):
        for rid, item in raw_example_mappings.items():
            if not isinstance(item, dict):
                continue
            normalised_row = _normalise_example_mapping(
                item,
                by_id=by_id,
                by_label=by_label,
                reaction_id_hint=str(rid),
            )
            if normalised_row is not None:
                example_rows.append(normalised_row)

    example_mapping_by_id: Dict[str, Dict[str, Any]] = {}
    for row in example_rows:
        rid = str(row.get("reaction_id") or "").strip()
        if not rid:
            continue
        example_mapping_by_id[rid] = dict(row)

    return {
        "meta": dict(payload.get("meta") or {}),
        "source_path": str(path),
        "taxonomy_labels": taxonomy_labels,
        "templates": ordered_templates,
        "by_id": by_id,
        "by_label": by_label,
        "example_mappings": list(example_mapping_by_id.values()),
        "example_mapping_by_id": example_mapping_by_id,
    }


def load_reaction_type_catalog_for_runtime(base_dir: Optional[Path] = None) -> Dict[str, Any]:
    base = str(_repo_base(base_dir))
    return load_reaction_type_catalog_cached(base)


def list_reaction_type_choices(catalog: Dict[str, Any]) -> List[Dict[str, str]]:
    choices: List[Dict[str, str]] = []
    for item in list(catalog.get("templates") or []):
        if not isinstance(item, dict):
            continue
        choices.append(
            {
                "type_id": str(item.get("type_id") or ""),
                "label_exact": str(item.get("label_exact") or ""),
                "canonical_group": str(item.get("canonical_group") or ""),
            }
        )
    return choices


def compact_template_for_prompt(template: Dict[str, Any], *, step_limit: int = 8) -> Dict[str, Any]:
    steps: List[Dict[str, Any]] = []
    for step in list(template.get("generic_mechanism_steps") or [])[: max(1, int(step_limit))]:
        if not isinstance(step, dict):
            continue
        steps.append(
            {
                "step_index": int(step.get("step_index") or 0),
                "reaction_generic": str(step.get("reaction_generic") or ""),
                "note": str(step.get("note") or ""),
            }
        )
    return {
        "type_id": str(template.get("type_id") or ""),
        "label_exact": str(template.get("label_exact") or ""),
        "canonical_group": str(template.get("canonical_group") or ""),
        "suitable_step_count": int(template.get("suitable_step_count") or 0),
        "generic_mechanism_steps": steps,
    }


def example_mapping_for_reaction_id(catalog: Dict[str, Any], reaction_id: Optional[str]) -> Optional[Dict[str, Any]]:
    rid = str(reaction_id or "").strip()
    if not rid:
        return None
    mapping = (catalog.get("example_mapping_by_id") or {}).get(rid)
    if isinstance(mapping, dict):
        return dict(mapping)
    return None


def suggest_reaction_type_for_example(
    catalog: Dict[str, Any],
    *,
    starting_materials: List[str],
    products: List[str],
) -> Optional[Dict[str, Any]]:
    """Heuristic template suggestion when example mapping/LLM selection is unavailable."""
    by_label = dict(catalog.get("by_label") or {})
    if not by_label:
        return None

    reactants = ".".join(starting_materials).lower()
    products_blob = ".".join(products).lower()

    def _pick(label: str, confidence: float, rationale: str) -> Optional[Dict[str, Any]]:
        template = by_label.get(label)
        if not isinstance(template, dict):
            return None
        return {
            "selected_label_exact": label,
            "selected_type_id": str(template.get("type_id") or ""),
            "confidence": max(0.0, min(1.0, float(confidence))),
            "rationale": rationale,
            "selected_template": compact_template_for_prompt(template),
        }

    if "n+]=[n-" in reactants:
        picked = _pick("Methyl ester synthesis", 0.68, "Example fallback: diazomethane-like methylation pattern.")
        if picked:
            return picked
    if "[i-]" in reactants and ("cl" in reactants or "br" in reactants):
        picked = _pick("SN2 reaction", 0.60, "Example fallback: halide exchange substitution motif.")
        if picked:
            return picked
    if "c(=o)cl" in reactants and ("o" in reactants or "oc" in reactants):
        picked = _pick(
            "Nucleophilic attack to (thio)carbonyl",
            0.62,
            "Example fallback: acyl chloride + nucleophile addition/substitution motif.",
        )
        if picked:
            return picked
    if "cc#n" in reactants and ("n=" in products_blob or "n(" in products_blob):
        picked = _pick("Imine formation", 0.52, "Example fallback: nitrile/nitrogen pattern suggests imine manifold.")
        if picked:
            return picked

    # Safe default keeps examples mapped to at least one suggested template.
    return _pick(
        "Nucleophilic attack to (thio)carbonyl",
        0.35,
        "Example fallback: default mechanistic-template suggestion.",
    )
