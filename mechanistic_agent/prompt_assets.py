"""Prompt-version assets and helpers for call-scoped prompt composition.

Mechanistic skill prompts live in skills/mechanistic/<call_name>/SKILL.md.
The prompt text is extracted from between <!-- PROMPT_START --> and <!-- PROMPT_END -->
markers within each SKILL.md. Few-shot examples remain in few_shot.jsonl.

The shared base system prompt lives in skills/mechanistic/base_system/SKILL.md.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from mechanistic_agent.core.types import FEWSHOT_QUALITY_WEIGHT, FewShotSelectionConfig

_PROMPT_START_MARKER = "<!-- PROMPT_START -->"
_PROMPT_END_MARKER = "<!-- PROMPT_END -->"

# Deterministic skill kinds that have no prompt text to extract.
_DETERMINISTIC_KINDS = {"deterministic", "shared_base"}

STEP_TO_CALL_NAME: Dict[str, str] = {
    "initial_conditions": "assess_initial_conditions",
    "missing_reagents": "predict_missing_reagents",
    "atom_mapping": "attempt_atom_mapping",
    "reaction_type_mapping": "select_reaction_type",
    "mechanism_step_proposal": "propose_mechanism_step",
    "mechanism_synthesis": "propose_mechanism_step",
    "intermediates": "propose_mechanism_step",
    "evaluation_judge": "evaluate_run_judge",
}

CALL_TO_STEPS: Dict[str, List[str]] = {
    "assess_initial_conditions": ["initial_conditions"],
    "predict_missing_reagents": ["missing_reagents"],
    "attempt_atom_mapping": ["atom_mapping"],
    "select_reaction_type": ["reaction_type_mapping"],
    "propose_mechanism_step": ["intermediates", "mechanism_step_proposal"],
    "evaluate_run_judge": ["evaluation_judge"],
}

_VALID_CALL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def normalize_call_name(call_name: str) -> str:
    """Normalize and validate prompt call names used in filesystem paths."""
    value = str(call_name or "").strip()
    if not value:
        raise ValueError("call_name is required")
    if not _VALID_CALL_NAME_RE.fullmatch(value):
        raise ValueError("call_name contains invalid characters")
    return value


def mechanistic_skills_root(base_dir: Path | None = None) -> Path:
    """Return the root directory for mechanistic skills."""
    base = (base_dir or Path.cwd()).resolve()
    return base / "skills" / "mechanistic"


def prompt_versions_root(base_dir: Path | None = None) -> Path:
    """Deprecated alias for mechanistic_skills_root. Use mechanistic_skills_root() instead."""
    return mechanistic_skills_root(base_dir)


def traces_root(base_dir: Path | None = None) -> Path:
    base = (base_dir or Path.cwd()).resolve()
    return base / "traces"


def model_asset_slug(model_name: str) -> str:
    value = str(model_name or "").strip()
    if not value:
        raise ValueError("model_name is required")
    return value.replace("/", "__")


def _active_model_name(default: str | None = None) -> str | None:
    try:
        from mechanistic_agent.core.model_context import get_active_model

        return get_active_model(default)
    except Exception:
        return default


def resolve_call_name_from_step(step_name: str) -> str | None:
    return STEP_TO_CALL_NAME.get(str(step_name or "").strip())


def steps_for_call(call_name: str) -> List[str]:
    return list(CALL_TO_STEPS.get(call_name, []))


def _normalise_text(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(lines).strip() + "\n"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_text(path: Path, *, default: str = "") -> str:
    if not path.exists():
        return default
    return path.read_text(encoding="utf-8")


def extract_prompt_from_skill_md(text: str) -> str:
    """Extract the prompt text from between PROMPT_START/PROMPT_END markers in a SKILL.md."""
    start_idx = text.find(_PROMPT_START_MARKER)
    end_idx = text.find(_PROMPT_END_MARKER)
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        return ""
    extracted = text[start_idx + len(_PROMPT_START_MARKER):end_idx]
    return extracted.strip()


def replace_prompt_in_skill_md(
    text: str,
    *,
    prompt_text: str,
    append_mode: bool = True,
) -> str:
    """Replace only the prompt block inside a skill markdown document."""

    start_idx = text.find(_PROMPT_START_MARKER)
    end_idx = text.find(_PROMPT_END_MARKER)
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        raise ValueError("SKILL.md is missing PROMPT_START/PROMPT_END markers")

    prefix = text[: start_idx + len(_PROMPT_START_MARKER)]
    suffix = text[end_idx:]
    current_prompt = extract_prompt_from_skill_md(text)
    next_prompt = str(prompt_text or "").strip()

    if append_mode and current_prompt.strip():
        next_prompt = f"{current_prompt.rstrip()}\n\n{next_prompt}" if next_prompt else current_prompt.rstrip()

    replacement = f"\n{next_prompt}\n" if next_prompt else "\n"
    return f"{prefix}{replacement}{suffix}"


def unified_prompt_diff(
    before_text: str,
    after_text: str,
    *,
    path: str = "SKILL.md",
) -> str:
    """Return a unified diff between two prompt files."""

    return "".join(
        difflib.unified_diff(
            before_text.splitlines(keepends=True),
            after_text.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )


def _parse_skill_md_frontmatter(text: str) -> Dict[str, str]:
    """Parse YAML-like frontmatter from a SKILL.md file."""
    text = text.lstrip()
    if not text.startswith("---\n"):
        return {}
    lines = text.splitlines()
    metadata: Dict[str, str] = {}
    for idx in range(1, len(lines)):
        line = lines[idx]
        if line.strip() == "---":
            break
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                metadata[key] = value
    return metadata


def shared_base_path(base_dir: Path | None = None) -> Path:
    return mechanistic_skills_root(base_dir) / "base_system" / "SKILL.md"


def call_dir(call_name: str, base_dir: Path | None = None) -> Path:
    return mechanistic_skills_root(base_dir) / normalize_call_name(call_name)


def call_base_path(call_name: str, base_dir: Path | None = None) -> Path:
    return call_dir(call_name, base_dir) / "SKILL.md"


def call_few_shot_path(call_name: str, base_dir: Path | None = None) -> Path:
    return call_dir(call_name, base_dir) / "few_shot.jsonl"


def _shared_base_override_path(model_name: str, base_dir: Path | None = None) -> Path:
    return mechanistic_skills_root(base_dir) / "base_system" / "models" / model_asset_slug(model_name) / "SKILL.md"


def _call_override_dir(call_name: str, model_name: str, base_dir: Path | None = None) -> Path:
    return call_dir(call_name, base_dir) / "models" / model_asset_slug(model_name)


def resolved_shared_base_path(base_dir: Path | None = None, model_name: str | None = None) -> Path:
    selected_model = str(model_name or _active_model_name() or "").strip()
    if selected_model:
        candidate = _shared_base_override_path(selected_model, base_dir)
        if candidate.exists():
            return candidate
    return shared_base_path(base_dir)


def resolved_call_base_path(call_name: str, base_dir: Path | None = None, model_name: str | None = None) -> Path:
    normalized = normalize_call_name(call_name)
    selected_model = str(model_name or _active_model_name() or "").strip()
    if selected_model:
        candidate = _call_override_dir(normalized, selected_model, base_dir) / "SKILL.md"
        if candidate.exists():
            return candidate
    return call_base_path(normalized, base_dir)


def resolved_call_few_shot_path(call_name: str, base_dir: Path | None = None, model_name: str | None = None) -> Path:
    normalized = normalize_call_name(call_name)
    selected_model = str(model_name or _active_model_name() or "").strip()
    if selected_model:
        candidate = _call_override_dir(normalized, selected_model, base_dir) / "few_shot.jsonl"
        if candidate.exists():
            return candidate
    return call_few_shot_path(normalized, base_dir)


def load_shared_base_prompt(base_dir: Path | None = None, *, model_name: str | None = None) -> str:
    text = _read_text(resolved_shared_base_path(base_dir, model_name))
    return extract_prompt_from_skill_md(text)


def load_call_base_prompt(call_name: str, base_dir: Path | None = None, *, model_name: str | None = None) -> str:
    text = _read_text(resolved_call_base_path(call_name, base_dir, model_name))
    return extract_prompt_from_skill_md(text)


def load_call_few_shot_examples(
    call_name: str,
    base_dir: Path | None = None,
    *,
    model_name: str | None = None,
) -> List[Dict[str, Any]]:
    selected_model = str(model_name or _active_model_name() or "").strip() or None
    paths: List[Path] = [call_few_shot_path(call_name, base_dir)]
    if selected_model:
        override_path = _call_override_dir(normalize_call_name(call_name), selected_model, base_dir) / "few_shot.jsonl"
        paths = [override_path, call_few_shot_path(call_name, base_dir)]

    rows: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()
    running_index = 0
    for path in paths:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            input_text = payload.get("input")
            output_text = payload.get("output")
            if isinstance(input_text, str) and isinstance(output_text, str):
                explicit_score = payload.get("score")
                try:
                    score = float(explicit_score) if explicit_score is not None else None
                except (TypeError, ValueError):
                    score = None
                example_key = payload.get("example_key")
                if not isinstance(example_key, str) or not example_key.strip():
                    example_key = hashlib.sha256(
                        f"{input_text}\n\0{output_text}".encode("utf-8")
                    ).hexdigest()[:16]
                if example_key in seen_keys:
                    continue
                seen_keys.add(example_key)
                raw_quality = payload.get("source_run_quality")
                quality = (
                    raw_quality
                    if raw_quality in {"clean", "soft", "failed"}
                    else None
                )
                rows.append(
                    {
                        "input": input_text,
                        "output": output_text,
                        "score": score,
                        "index": running_index,
                        "example_key": example_key,
                        "source_run_quality": quality,
                    }
                )
                running_index += 1
    return rows


def append_call_few_shot_example(
    call_name: str,
    *,
    input_text: str,
    output_text: str,
    score: float | None = None,
    example_key: str | None = None,
    source_run_quality: str | None = None,
    base_dir: Path | None = None,
    model_name: str | None = None,
) -> Path:
    selected_model = str(model_name or _active_model_name() or "").strip() or None
    path = (
        (_call_override_dir(normalize_call_name(call_name), selected_model, base_dir) / "few_shot.jsonl")
        if selected_model
        else call_few_shot_path(call_name, base_dir)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"input": input_text, "output": output_text}
    if score is not None:
        payload["score"] = round(float(score), 6)
    if example_key:
        payload["example_key"] = str(example_key)
    if source_run_quality and source_run_quality in {"clean", "soft", "failed"}:
        payload["source_run_quality"] = source_run_quality
    line = json.dumps(payload, sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return path


def write_call_few_shot_examples(
    call_name: str,
    examples: List[Dict[str, Any]],
    *,
    base_dir: Path | None = None,
    model_name: str | None = None,
) -> Path:
    """Merge-write examples into the appropriate few_shot.jsonl file.

    Deduplicates by example_key. A new example replaces an existing one only
    if its score is >= the stored score (or the stored entry has no score).
    Net effect: the file always holds the best-known version of each example.

    Each dict in `examples` must have: input, output.
    Optional: score, example_key, source_run_quality.
    """
    selected_model = str(model_name or _active_model_name() or "").strip() or None
    path = (
        (_call_override_dir(normalize_call_name(call_name), selected_model, base_dir) / "few_shot.jsonl")
        if selected_model
        else call_few_shot_path(call_name, base_dir)
    )

    # Load existing entries keyed by example_key.
    existing: Dict[str, Dict[str, Any]] = {}
    if path.exists():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            key = obj.get("example_key") or hashlib.sha256(
                f"{obj.get('input', '')}\n\0{obj.get('output', '')}".encode("utf-8")
            ).hexdigest()[:16]
            existing[key] = obj

    # Merge: new example wins only when its score is better (or existing has none).
    for ex in examples:
        key = ex.get("example_key") or hashlib.sha256(
            f"{ex.get('input', '')}\n\0{ex.get('output', '')}".encode("utf-8")
        ).hexdigest()[:16]
        new_score = ex.get("score")
        old_score = existing.get(key, {}).get("score")
        if key not in existing or (
            new_score is not None and (old_score is None or float(new_score) >= float(old_score))
        ):
            payload: Dict[str, Any] = {"input": ex["input"], "output": ex["output"]}
            if new_score is not None:
                payload["score"] = round(float(new_score), 6)
            payload["example_key"] = key
            if ex.get("source_run_quality") in {"clean", "soft", "failed"}:
                payload["source_run_quality"] = ex["source_run_quality"]
            existing[key] = payload

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(v, sort_keys=True) for v in existing.values()) + "\n",
        encoding="utf-8",
    )
    return path


def _tool_schema_for_call_name(call_name: str) -> Dict[str, Any] | None:
    from mechanistic_agent.tool_schemas import (
        ASSESS_CONDITIONS_TOOL,
        ATOM_MAPPING_TOOL,
        INTERMEDIATES_TOOL,
        MECHANISM_STEP_PROPOSAL_TOOL,
        MISSING_REAGENTS_TOOL,
        REACTION_TYPE_SELECTION_TOOL,
    )

    return {
        "assess_initial_conditions": ASSESS_CONDITIONS_TOOL,
        "predict_missing_reagents": MISSING_REAGENTS_TOOL,
        "attempt_atom_mapping": ATOM_MAPPING_TOOL,
        "select_reaction_type": REACTION_TYPE_SELECTION_TOOL,
        "propose_mechanism_step": MECHANISM_STEP_PROPOSAL_TOOL,
        "evaluate_run_judge": INTERMEDIATES_TOOL,
    }.get(call_name)


def _value_quality(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0
    if isinstance(value, (int, float)):
        return 1.0
    if isinstance(value, str):
        return 1.0 if value.strip() else 0.0
    if isinstance(value, list):
        if not value:
            return 0.6
        child_scores = [_value_quality(item) for item in value]
        return min(1.0, 0.7 + (sum(child_scores) / len(child_scores)) * 0.3)
    if isinstance(value, dict):
        if not value:
            return 0.5
        child_scores = [_value_quality(item) for item in value.values()]
        return min(1.0, 0.65 + (sum(child_scores) / len(child_scores)) * 0.35)
    return 0.0


def score_few_shot_example(
    call_name: str,
    *,
    input_text: str,
    output_text: str,
    explicit_score: float | None = None,
) -> float:
    if explicit_score is not None:
        return max(0.0, min(float(explicit_score), 1.0))
    if not str(input_text or "").strip() or not str(output_text or "").strip():
        return 0.0

    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError:
        return 0.35

    if not isinstance(parsed, dict):
        return 0.25

    schema = _tool_schema_for_call_name(call_name)
    if not schema:
        return 0.6

    parameters = (
        schema.get("function", {}).get("parameters", {})
        if isinstance(schema, dict)
        else {}
    )
    properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
    required = [
        key for key in (parameters.get("required") or [])
        if key in properties and key != "text"
    ]
    optional = [
        key for key in properties.keys()
        if key not in set(required) and key != "text"
    ]

    required_score = (
        sum(_value_quality(parsed.get(key)) for key in required) / len(required)
        if required
        else 1.0
    )
    optional_present = [_value_quality(parsed.get(key)) for key in optional if parsed.get(key) is not None]
    optional_score = (
        sum(optional_present) / len(optional_present)
        if optional_present
        else 0.0
    )
    structure_bonus = 0.0
    if call_name == "propose_mechanism_step":
        classification = str(parsed.get("classification") or "").strip()
        candidates = parsed.get("candidates")
        if classification in {"intermediate_step", "final_step"}:
            structure_bonus += 0.05
        if isinstance(candidates, list) and candidates:
            structure_bonus += 0.05
    elif call_name == "select_reaction_type":
        confidence = parsed.get("confidence")
        if isinstance(confidence, (int, float)):
            structure_bonus += 0.05

    score = 0.7 * required_score + 0.2 * optional_score + 0.1 * min(1.0, 0.5 + structure_bonus)
    return round(max(0.0, min(score, 1.0)), 4)


def best_few_shot_score(call_name: str, base_dir: Path | None = None) -> float:
    examples = load_call_few_shot_examples(call_name, base_dir)
    if not examples:
        return 0.0
    return max(
        score_few_shot_example(
            call_name,
            input_text=str(example.get("input") or ""),
            output_text=str(example.get("output") or ""),
            explicit_score=example.get("score") if isinstance(example.get("score"), (int, float)) else None,
        )
        for example in examples
    )


def select_few_shot_examples(
    call_name: str,
    base_dir: Path | None = None,
    *,
    policy: FewShotSelectionConfig | Dict[str, Any] | None = None,
    model_name: str | None = None,
) -> List[Dict[str, Any]]:
    config = policy if isinstance(policy, FewShotSelectionConfig) else FewShotSelectionConfig.from_dict(policy)
    if not config.enabled or config.max_examples <= 0:
        return []

    examples = load_call_few_shot_examples(call_name, base_dir, model_name=model_name)
    scored_examples: List[Dict[str, Any]] = []
    for example in examples:
        score = score_few_shot_example(
            call_name,
            input_text=str(example.get("input") or ""),
            output_text=str(example.get("output") or ""),
            explicit_score=example.get("score") if isinstance(example.get("score"), (int, float)) else None,
        )
        enriched = dict(example)
        enriched["score"] = score
        if config.min_score is not None and score < config.min_score:
            continue
        scored_examples.append(enriched)

    def _sort_key(row: Dict[str, Any]) -> tuple:
        base_score = float(row.get("score") or 0.0)
        idx = int(row.get("index") or 0)
        if config.selection_strategy == "most_recent":
            return (-idx,)
        if config.selection_strategy == "first":
            return (idx,)
        # top_score: incorporate quality penalty if preference is active
        quality_penalty = 0.0
        if config.prefer_clean_traces:
            q = row.get("source_run_quality")
            quality_penalty = float(FEWSHOT_QUALITY_WEIGHT.get(q or "clean", 0.0))
        return (-base_score + quality_penalty, -idx)

    # When prefer_clean_traces is on, check whether we have enough clean
    # examples to enforce the preference.  If not, fall back to treating
    # all examples equally so we don't starve the prompt early in the
    # system's life.
    if config.prefer_clean_traces:
        clean_count = sum(
            1 for row in scored_examples if row.get("source_run_quality") in {None, "clean"}
        )
        if clean_count < config.min_clean_for_preference:
            # Not enough clean examples yet — ignore quality signal entirely.
            scored_examples.sort(
                key=lambda row: (
                    -float(row.get("score") or 0.0),
                    -int(row.get("index") or 0),
                )
            )
            return scored_examples[: config.max_examples]

    scored_examples.sort(key=_sort_key)
    return scored_examples[: config.max_examples]


def format_few_shot_block(
    call_name: str,
    base_dir: Path | None = None,
    *,
    max_examples: int = 4,
    policy: FewShotSelectionConfig | Dict[str, Any] | None = None,
    model_name: str | None = None,
) -> str:
    resolved_policy = policy
    if resolved_policy is None:
        try:
            from mechanistic_agent.core.model_context import get_few_shot_policy

            resolved_policy = get_few_shot_policy(call_name)
        except Exception:
            resolved_policy = None
    if resolved_policy is None:
        resolved_policy = FewShotSelectionConfig(max_examples=max(0, int(max_examples)))
    examples = select_few_shot_examples(call_name, base_dir, policy=resolved_policy, model_name=model_name)
    if not examples:
        return ""
    lines: List[str] = ["Few-shot examples:"]
    for index, row in enumerate(examples, start=1):
        lines.append(f"Example {index} input:")
        lines.append(row["input"])
        lines.append(f"Example {index} output:")
        lines.append(row["output"])
    return "\n".join(lines)


def _build_harness_context_preamble(
    enabled_upstream: list | None = None,
    enabled_downstream: list | None = None,
) -> str:
    """Build a context preamble telling the LLM what upstream/downstream modules exist."""
    lines: list[str] = []
    if enabled_upstream:
        lines.append("You are receiving the following upstream analysis:")
        for mod in enabled_upstream:
            label = mod.get("label") or mod.get("id", "")
            mod_id = mod.get("id", "")
            desc = mod.get("description", "")
            line = f"- {label} ({mod_id})"
            if desc:
                line += f": {desc}"
            lines.append(line)
    if enabled_downstream:
        names = [m.get("label") or m.get("id", "") for m in enabled_downstream]
        lines.append(f"\nYour output will be used by: {', '.join(names)}")
    if lines:
        lines.append("")  # trailing newline
    return "\n".join(lines)


def compose_system_prompt(
    *,
    call_name: str,
    dynamic_system_prompt: str,
    base_dir: Path | None = None,
    enabled_upstream_modules: list | None = None,
    enabled_downstream_modules: list | None = None,
    model_name: str | None = None,
) -> str:
    shared = load_shared_base_prompt(base_dir, model_name=model_name)
    call_base = load_call_base_prompt(call_name, base_dir, model_name=model_name)
    context_preamble = _build_harness_context_preamble(enabled_upstream_modules, enabled_downstream_modules)
    parts = [
        part.strip()
        for part in (shared, call_base, context_preamble, dynamic_system_prompt or "")
        if part and part.strip()
    ]
    return "\n\n".join(parts).strip()


def get_call_prompt_version(
    call_name: str,
    base_dir: Path | None = None,
    model_name: str | None = None,
) -> Dict[str, Any]:
    normalized_call_name = normalize_call_name(call_name)
    selected_model = str(model_name or _active_model_name() or "").strip() or None
    shared_path = resolved_shared_base_path(base_dir, selected_model)
    base_path = resolved_call_base_path(normalized_call_name, base_dir, selected_model)
    few_shot_path = resolved_call_few_shot_path(normalized_call_name, base_dir, selected_model)

    shared_text = _read_text(shared_path)
    base_text = _read_text(base_path)
    if selected_model:
        merged_examples = load_call_few_shot_examples(normalized_call_name, base_dir, model_name=selected_model)
        few_shot_text = "\n".join(
            json.dumps(
                {
                    "input": example.get("input"),
                    "output": example.get("output"),
                    **({"score": example.get("score")} if example.get("score") is not None else {}),
                    **({"example_key": example.get("example_key")} if example.get("example_key") else {}),
                },
                sort_keys=True,
            )
            for example in merged_examples
        )
    else:
        few_shot_text = _read_text(few_shot_path)

    shared_prompt = extract_prompt_from_skill_md(shared_text)
    call_prompt = extract_prompt_from_skill_md(base_text)

    shared_norm = _normalise_text(shared_prompt)
    base_norm = _normalise_text(call_prompt)
    few_shot_norm = _normalise_text(few_shot_text)
    bundle_seed = "\n".join(
        [
            f"model:{selected_model or 'shared'}",
            "shared_base",
            str(shared_path),
            shared_norm,
            "call_base",
            str(base_path),
            base_norm,
            "few_shot",
            str(few_shot_path),
            few_shot_norm,
        ]
    )

    default_shared_path = shared_base_path(base_dir)
    default_call_path = call_base_path(normalized_call_name, base_dir)
    default_few_shot_path = call_few_shot_path(normalized_call_name, base_dir)
    used_override = (
        shared_path != default_shared_path
        or base_path != default_call_path
        or few_shot_path != default_few_shot_path
    )
    return {
        "call_name": normalized_call_name,
        "steps": steps_for_call(normalized_call_name),
        "model_name": selected_model,
        "shared_base_path": str(default_shared_path),
        "call_base_path": str(default_call_path),
        "few_shot_path": str(default_few_shot_path),
        "resolved_shared_base_path": str(shared_path),
        "resolved_call_base_path": str(base_path),
        "resolved_few_shot_path": str(few_shot_path),
        "asset_scope": "exact_model" if used_override else "shared",
        "shared_base_sha256": _sha256_text(shared_norm),
        "call_base_sha256": _sha256_text(base_norm),
        "few_shot_sha256": _sha256_text(few_shot_norm),
        "prompt_bundle_sha256": _sha256_text(bundle_seed),
        "template": call_prompt.strip(),
        "shared_template": shared_prompt.strip(),
        "few_shot_examples": load_call_few_shot_examples(call_name, base_dir, model_name=selected_model),
    }


def list_call_prompt_versions(base_dir: Path | None = None, model_name: str | None = None) -> List[Dict[str, Any]]:
    """List all LLM call-type mechanistic skills (excludes base_system and deterministic skills)."""
    skills_dir = mechanistic_skills_root(base_dir)
    if not skills_dir.exists():
        return []
    items: List[Dict[str, Any]] = []
    for path in sorted(skills_dir.iterdir()):
        if not path.is_dir():
            continue
        skill_md = path / "SKILL.md"
        if not skill_md.exists():
            continue
        # Skip base_system and deterministic skills (no LLM prompt to version)
        frontmatter = _parse_skill_md_frontmatter(_read_text(skill_md))
        kind = frontmatter.get("kind", "")
        if kind in _DETERMINISTIC_KINDS:
            continue
        call_name = frontmatter.get("call_name") or path.name
        try:
            items.append(get_call_prompt_version(call_name, base_dir, model_name=model_name))
        except ValueError:
            continue
    return items
