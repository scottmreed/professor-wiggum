"""Validation helpers for prompt-change evidence gates."""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Set

from .prompt_assets import get_call_prompt_version, traces_root


_PROMPT_CHANGE_RE = re.compile(r"^prompt_versions/calls/([^/]+)/(base\.md|few_shot\.jsonl)$")
_REQUIRED_MODEL_KEYS = {"model_version_id", "resolved_model_key", "provider", "family", "pricing_sha256"}


@dataclass
class PromptTraceValidationResult:
    changed_calls: List[str]
    valid_evidence_by_call: Dict[str, List[str]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def discover_changed_calls(*, base_ref: str, head_ref: str, cwd: Path | None = None) -> List[str]:
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
    calls: Set[str] = set()
    for line in completed.stdout.splitlines():
        match = _PROMPT_CHANGE_RE.match(line.strip())
        if not match:
            continue
        calls.add(match.group(1))
    return sorted(calls)


def _evidence_files_for_call(call_name: str, *, base_dir: Path) -> List[Path]:
    root = traces_root(base_dir) / "evidence" / call_name
    if not root.exists():
        return []
    return sorted(root.glob("*/*.json"))


def validate_evidence_for_calls(
    *,
    changed_calls: Iterable[str],
    base_dir: Path | None = None,
) -> PromptTraceValidationResult:
    base = (base_dir or Path.cwd()).resolve()
    normalized_calls = sorted({str(call).strip() for call in changed_calls if str(call).strip()})
    result = PromptTraceValidationResult(changed_calls=normalized_calls)

    for call_name in normalized_calls:
        try:
            prompt_version = get_call_prompt_version(call_name, base)
        except Exception as exc:
            result.errors.append(f"{call_name}: unable to load prompt version ({exc})")
            continue
        current_bundle = str(prompt_version.get("prompt_bundle_sha256") or "")
        if not current_bundle:
            result.errors.append(f"{call_name}: prompt bundle hash missing")
            continue

        candidates = _evidence_files_for_call(call_name, base_dir=base)
        if not candidates:
            result.errors.append(f"{call_name}: no evidence files found under traces/evidence/{call_name}/")
            continue

        valid_files: List[str] = []
        for path in candidates:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("approved_bool") is not True:
                continue
            prompt_block = payload.get("prompt_version")
            if not isinstance(prompt_block, dict):
                continue
            evidence_bundle = str(prompt_block.get("prompt_bundle_sha256") or "")
            if evidence_bundle != current_bundle:
                continue
            model_block = payload.get("model_version")
            if not isinstance(model_block, dict):
                continue
            if any(not model_block.get(key) for key in _REQUIRED_MODEL_KEYS):
                continue
            valid_files.append(str(path.resolve().relative_to(base)))

        if not valid_files:
            result.errors.append(
                f"{call_name}: evidence exists but no approved+linked trace matches current prompt bundle {current_bundle[:12]}"
            )
            continue
        result.valid_evidence_by_call[call_name] = valid_files

    return result
