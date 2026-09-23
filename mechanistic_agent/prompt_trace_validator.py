"""Validation helpers for prompt-change evidence gates.

The gate answers one question for a pull request: *for every LLM prompt asset
this PR changes, is there at least one approved evidence trace that was produced
with exactly the prompt bundle now in the tree?*

Prompt assets live under ``skills/mechanistic/<call>/`` (``SKILL.md``,
``few_shot.jsonl``) with optional per-model lanes under ``models/<slug>/``. The
shared base prompt in ``skills/mechanistic/base_system/SKILL.md`` feeds every
call, so a change there is treated as a change to every gated call.

This module must stay importable with the standard library only: the CI
workflow runs it without installing project dependencies.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Sequence, Set, Tuple, Union

from mechanistic_agent.data_paths import evidence_root
from mechanistic_agent.prompt_assets import (
    gated_call_names,
    get_call_prompt_version,
    model_name_from_asset_slug,
)


_SKILL_PATH_RE = re.compile(
    r"^skills/mechanistic/(?P<call>[A-Za-z0-9_-]+)/(?:models/(?P<slug>[^/]+)/)?(?P<file>SKILL\.md|few_shot\.jsonl)$"
)
_SHARED_BASE_CALL = "base_system"
_REQUIRED_MODEL_KEYS = {"model_version_id", "resolved_model_key", "provider", "family", "pricing_sha256"}

COMPONENT_SHARED_BASE = "shared_base"
COMPONENT_CALL_BASE = "call_base"
COMPONENT_FEW_SHOT = "few_shot"


@dataclass(frozen=True)
class PromptChange:
    """One gated prompt asset change: a call, optionally scoped to a model lane."""

    call_name: str
    model_name: str | None = None
    components: FrozenSet[str] = frozenset()

    @property
    def label(self) -> str:
        if self.model_name is None:
            return self.call_name
        return f"{self.call_name}@{self.model_name}"


ChangeLike = Union[str, PromptChange]


@dataclass
class PromptTraceValidationResult:
    changed_calls: List[str]
    changes: List[PromptChange] = field(default_factory=list)
    valid_evidence_by_call: Dict[str, List[str]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def calls_from_changed_paths(paths: Iterable[str], *, gated_calls: Sequence[str] | None = None) -> List[PromptChange]:
    """Map changed repository paths to the prompt changes they represent.

    Pure function (no git, no filesystem) so it can be tested directly.
    """
    gated: Tuple[str, ...] = tuple(gated_calls if gated_calls is not None else gated_call_names())
    gated_set = set(gated)
    merged: Dict[Tuple[str, str | None], Set[str]] = {}

    def _add(call: str, model: str | None, component: str) -> None:
        merged.setdefault((call, model), set()).add(component)

    for raw in paths:
        path = str(raw or "").strip()
        match = _SKILL_PATH_RE.match(path)
        if not match:
            continue
        call = match.group("call")
        slug = match.group("slug")
        file_name = match.group("file")
        model = model_name_from_asset_slug(slug) if slug else None

        if call == _SHARED_BASE_CALL:
            if file_name != "SKILL.md":
                continue
            for gated_call in gated:
                _add(gated_call, model, COMPONENT_SHARED_BASE)
            continue

        if call not in gated_set:
            continue
        component = COMPONENT_CALL_BASE if file_name == "SKILL.md" else COMPONENT_FEW_SHOT
        _add(call, model, component)

    return [
        PromptChange(call_name=call, model_name=model, components=frozenset(components))
        for (call, model), components in sorted(merged.items(), key=lambda item: (item[0][0], item[0][1] or ""))
    ]


def discover_changed_calls(*, base_ref: str, head_ref: str, cwd: Path | None = None) -> List[PromptChange]:
    workdir = (cwd or Path.cwd()).resolve()
    completed = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...{head_ref}"],
        cwd=str(workdir),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "Failed to detect changed files")
    return calls_from_changed_paths(completed.stdout.splitlines())


def _coerce_changes(changed_calls: Iterable[ChangeLike]) -> List[PromptChange]:
    changes: Dict[Tuple[str, str | None], PromptChange] = {}
    for item in changed_calls:
        if isinstance(item, PromptChange):
            change = item
        else:
            call = str(item or "").strip()
            if not call:
                continue
            change = PromptChange(call_name=call)
        key = (change.call_name, change.model_name)
        existing = changes.get(key)
        if existing is None:
            changes[key] = change
        else:
            changes[key] = PromptChange(
                call_name=change.call_name,
                model_name=change.model_name,
                components=existing.components | change.components,
            )
    return [changes[key] for key in sorted(changes, key=lambda k: (k[0], k[1] or ""))]


def _evidence_files_for_call(call_name: str, *, base_dir: Path) -> List[Path]:
    root = evidence_root(base_dir) / call_name
    if not root.exists():
        return []
    return sorted(root.glob("*/*.json"))


def _evidence_model_name(prompt_block: Dict[str, object]) -> str | None:
    value = str(prompt_block.get("model_name") or "").strip()
    return value or None


def validate_evidence_for_calls(
    *,
    changed_calls: Iterable[ChangeLike],
    base_dir: Path | None = None,
) -> PromptTraceValidationResult:
    base = (base_dir or Path.cwd()).resolve()
    changes = _coerce_changes(changed_calls)
    result = PromptTraceValidationResult(changed_calls=[change.label for change in changes], changes=changes)

    # Prompt versions are model-scoped at run time, so the bundle hash to match
    # depends on which model the evidence was produced with. Cache per scope.
    version_cache: Dict[Tuple[str, str | None], Dict[str, object]] = {}

    def _version(call_name: str, model_name: str | None) -> Dict[str, object]:
        key = (call_name, model_name)
        if key not in version_cache:
            version_cache[key] = get_call_prompt_version(call_name, base, model_name=model_name)
        return version_cache[key]

    for change in changes:
        call_name = change.call_name
        label = change.label
        try:
            _version(call_name, None)
        except Exception as exc:
            result.errors.append(f"{label}: unable to load prompt version ({exc})")
            continue

        candidates = _evidence_files_for_call(call_name, base_dir=base)
        if not candidates:
            result.errors.append(f"{label}: no evidence files found under traces/evidence/{call_name}/")
            continue

        valid_files: List[str] = []
        rejections: List[str] = []
        for path in candidates:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("approved_bool") is not True:
                continue
            exposure = evidence_ground_truth_exposure(payload)
            if exposure is not False:
                # Evidence must state, explicitly, that the responder did NOT see
                # the verified mechanism. Replays of ground truth are not evidence
                # of model capability and undeclared exposure is not accepted.
                rejections.append(f"{path.name}: responder_saw_ground_truth={exposure!r} (must be false)")
                continue
            prompt_block = payload.get("prompt_version")
            if not isinstance(prompt_block, dict):
                continue
            evidence_model = _evidence_model_name(prompt_block)
            if change.model_name is not None and evidence_model != change.model_name:
                rejections.append(
                    f"{path.name}: evidence model {evidence_model!r} does not match changed lane {change.model_name!r}"
                )
                continue
            try:
                version = _version(call_name, evidence_model)
            except Exception as exc:
                rejections.append(f"{path.name}: unable to load prompt version for model {evidence_model!r} ({exc})")
                continue
            current_bundle = str(version.get("prompt_bundle_sha256") or "")
            if not current_bundle:
                rejections.append(f"{path.name}: prompt bundle hash missing for model {evidence_model!r}")
                continue
            evidence_bundle = str(prompt_block.get("prompt_bundle_sha256") or "")
            if evidence_bundle != current_bundle:
                rejections.append(
                    f"{path.name}: bundle {evidence_bundle[:12] or '<missing>'} != current {current_bundle[:12]}"
                    f" (scope {evidence_model or 'shared'})"
                )
                continue
            # A model lane that overrides the changed shared file never exercised
            # that file, so its evidence proves nothing about the change.
            if COMPONENT_CALL_BASE in change.components and change.model_name is None and (
                str(version.get("resolved_call_base_path")) != str(version.get("call_base_path"))
            ):
                rejections.append(f"{path.name}: model {evidence_model!r} overrides SKILL.md; does not exercise the changed shared prompt")
                continue
            if COMPONENT_SHARED_BASE in change.components and change.model_name is None and (
                str(version.get("resolved_shared_base_path")) != str(version.get("shared_base_path"))
            ):
                rejections.append(f"{path.name}: model {evidence_model!r} overrides base_system; does not exercise the changed shared base")
                continue
            model_block = payload.get("model_version")
            if not isinstance(model_block, dict):
                rejections.append(f"{path.name}: model_version block missing")
                continue
            missing = sorted(key for key in _REQUIRED_MODEL_KEYS if not model_block.get(key))
            if missing:
                rejections.append(f"{path.name}: model_version missing {', '.join(missing)}")
                continue
            valid_files.append(str(path.resolve().relative_to(base)))

        if not valid_files:
            detail = " (" + "; ".join(rejections) + ")" if rejections else ""
            shared_bundle = str(_version(call_name, None).get("prompt_bundle_sha256") or "")[:12]
            result.errors.append(
                f"{label}: evidence exists but no approved+linked trace matches the current prompt bundle "
                f"(shared scope {shared_bundle}){detail}"
            )
            continue
        result.valid_evidence_by_call[label] = valid_files

    return result


def evidence_ground_truth_exposure(payload: Dict[str, object]) -> object:
    """Return the evidence file's declared ground-truth exposure.

    Accepted locations: top-level ``responder_saw_ground_truth`` or
    ``origin.responder_saw_ground_truth``. Returns ``True``/``False`` when
    declared, ``"undeclared"`` for unparseable values, and ``None`` when absent.
    """
    if not isinstance(payload, dict):
        return None
    value: object = payload.get("responder_saw_ground_truth")
    if value is None:
        origin = payload.get("origin")
        if isinstance(origin, dict):
            value = origin.get("responder_saw_ground_truth")
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return "undeclared"
