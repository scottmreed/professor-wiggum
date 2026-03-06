"""Registries for prompts, skills, curated memory packs, and harness configs."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from mechanistic_agent.core.types import HarnessConfig, ModuleSpec, RunConfig
from mechanistic_agent.prompt_assets import (
    get_call_prompt_version,
    list_call_prompt_versions,
    mechanistic_skills_root,
    resolve_call_name_from_step,
    shared_base_path,
)


@dataclass(frozen=True)
class AssetRecord:
    asset_type: str
    path: str
    sha256: str
    metadata: Dict[str, Any]


def _parse_markdown_frontmatter(content: str) -> Tuple[Dict[str, str], str]:
    text = content.lstrip()
    if not text.startswith("---\n"):
        return {}, content
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, content
    metadata: Dict[str, str] = {}
    end_idx = None
    for idx in range(1, len(lines)):
        line = lines[idx]
        if line.strip() == "---":
            end_idx = idx
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            metadata[key] = value
    if end_idx is None:
        return {}, content
    body = "\n".join(lines[end_idx + 1 :]).lstrip("\n")
    return metadata, body


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 64), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_hash(parts: Iterable[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for path, file_hash in sorted(parts, key=lambda item: item[0]):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


class PromptRegistry:
    """Loads prompt assets from skills/mechanistic/."""

    def __init__(self, mechanistic_skills_dir: Path) -> None:
        self.mechanistic_skills_dir = mechanistic_skills_dir
        # Compute base_dir as the parent of skills/
        self._base_dir = mechanistic_skills_dir.parent.parent if mechanistic_skills_dir.exists() else None

    def list(self, *, model_name: str | None = None) -> List[AssetRecord]:
        records: List[AssetRecord] = []
        if not self.mechanistic_skills_dir.exists():
            return records
        shared_path = shared_base_path(self._base_dir)
        if shared_path.exists():
            records.append(
                AssetRecord(
                    asset_type="prompt_shared",
                    path=str(shared_path),
                    sha256=_file_sha256(shared_path),
                    metadata={
                        "name": "base_system",
                        "kind": "shared_base",
                    },
                )
            )

        for item in list_call_prompt_versions(self._base_dir, model_name=model_name):
            records.append(
                AssetRecord(
                    asset_type="prompt_call",
                    path=str(item.get("resolved_call_base_path") or item.get("call_base_path") or ""),
                    sha256=str(item.get("prompt_bundle_sha256") or ""),
                    metadata={
                        "name": str(item.get("call_name") or ""),
                        "call_name": str(item.get("call_name") or ""),
                        "steps": list(item.get("steps") or []),
                        "version": str(item.get("prompt_bundle_sha256") or "")[:12],
                        "template": item.get("template"),
                        "shared_template": item.get("shared_template"),
                        "shared_base_sha256": item.get("shared_base_sha256"),
                        "call_base_sha256": item.get("call_base_sha256"),
                        "few_shot_sha256": item.get("few_shot_sha256"),
                        "prompt_bundle_sha256": item.get("prompt_bundle_sha256"),
                        "few_shot_path": item.get("few_shot_path"),
                        "resolved_shared_base_path": item.get("resolved_shared_base_path"),
                        "resolved_call_base_path": item.get("resolved_call_base_path"),
                        "resolved_few_shot_path": item.get("resolved_few_shot_path"),
                        "asset_scope": item.get("asset_scope"),
                        "model_name": item.get("model_name"),
                    },
                )
            )
        return records

    def by_step(self, step_name: str, *, model_name: str | None = None) -> AssetRecord | None:
        call_name = resolve_call_name_from_step(step_name)
        if not call_name:
            return None
        payload = get_call_prompt_version(call_name, self._base_dir, model_name=model_name)
        return AssetRecord(
            asset_type="prompt_call",
            path=str(payload.get("resolved_call_base_path") or payload.get("call_base_path") or ""),
            sha256=str(payload.get("prompt_bundle_sha256") or ""),
            metadata={
                "name": call_name,
                "call_name": call_name,
                "steps": list(payload.get("steps") or []),
                "version": str(payload.get("prompt_bundle_sha256") or "")[:12],
                "template": payload.get("template"),
                "shared_template": payload.get("shared_template"),
                "shared_base_sha256": payload.get("shared_base_sha256"),
                "call_base_sha256": payload.get("call_base_sha256"),
                "few_shot_sha256": payload.get("few_shot_sha256"),
                "prompt_bundle_sha256": payload.get("prompt_bundle_sha256"),
                "few_shot_path": payload.get("few_shot_path"),
                "resolved_shared_base_path": payload.get("resolved_shared_base_path"),
                "resolved_call_base_path": payload.get("resolved_call_base_path"),
                "resolved_few_shot_path": payload.get("resolved_few_shot_path"),
                "asset_scope": payload.get("asset_scope"),
                "model_name": payload.get("model_name"),
            },
        )

    def bundle_hash(self, *, model_name: str | None = None) -> str:
        records = self.list(model_name=model_name)
        return _bundle_hash((record.path, record.sha256) for record in records)


class SkillRegistry:
    """Loads skill markdown assets from skills/**/SKILL.md.

    Scans both skills/project/ and skills/mechanistic/ subdirectories.
    Returns skill_type (project | mechanistic) and kind (llm | deterministic | shared_base)
    from the SKILL.md frontmatter.
    """

    def __init__(self, skills_dir: Path) -> None:
        self.skills_dir = skills_dir

    def list(self) -> List[AssetRecord]:
        records: List[AssetRecord] = []
        if not self.skills_dir.exists():
            return records
        for path in sorted(self.skills_dir.rglob("SKILL.md")):
            text = path.read_text(encoding="utf-8")
            frontmatter, body = _parse_markdown_frontmatter(text)
            summary = ""
            for line in body.splitlines():
                if line.strip():
                    summary = line.strip().lstrip("# ").strip()
                    break
            records.append(
                AssetRecord(
                    asset_type="skill",
                    path=str(path),
                    sha256=_file_sha256(path),
                    metadata={
                        "name": path.parent.name,
                        "summary": summary,
                        "skill_type": frontmatter.get("skill_type", "project"),
                        "kind": frontmatter.get("kind", ""),
                        "call_name": frontmatter.get("call_name", ""),
                    },
                )
            )
        return records

    def bundle_hash(self) -> str:
        records = self.list()
        return _bundle_hash((record.path, record.sha256) for record in records)


class MemoryPackRegistry:
    """Loads curated memory packs from memory_packs/*.md|*.json."""

    def __init__(self, memory_dir: Path) -> None:
        self.memory_dir = memory_dir

    def list(self) -> List[AssetRecord]:
        records: List[AssetRecord] = []
        if not self.memory_dir.exists():
            return records
        candidates = list(self.memory_dir.glob("*.md")) + list(self.memory_dir.glob("*.json"))
        for path in sorted(candidates):
            sha = _file_sha256(path)
            metadata: Dict[str, Any] = {"name": path.stem, "suffix": path.suffix}
            if path.suffix == ".json":
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    data = None
                if isinstance(data, dict):
                    metadata["keys"] = sorted(data.keys())
                elif isinstance(data, list):
                    metadata["entries"] = len(data)
            records.append(
                AssetRecord(
                    asset_type="memory_pack",
                    path=str(path),
                    sha256=sha,
                    metadata=metadata,
                )
            )
        return records

    def bundle_hash(self) -> str:
        records = self.list()
        return _bundle_hash((record.path, record.sha256) for record in records)


class HarnessRegistry:
    """Loads and validates harness configuration files from harness_versions/<name>/."""

    def __init__(self, harness_dir: Path) -> None:
        self.harness_dir = harness_dir

    def load(self, name: str = "default") -> HarnessConfig:
        """Load a harness config by name from its subdirectory, compute version SHA."""
        path = self.harness_dir / name / "harness.json"
        if not path.exists():
            raise FileNotFoundError(f"Harness config not found: {path}")
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        config = HarnessConfig.from_dict(data)
        config.version = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        config.name = name
        return config

    def list_versions(self) -> List[Dict[str, Any]]:
        """List all available harness configs (one per subdirectory) with their SHAs."""
        results: List[Dict[str, Any]] = []
        if not self.harness_dir.exists():
            return results
        for path in sorted(self.harness_dir.glob("*/harness.json")):
            raw = path.read_text(encoding="utf-8")
            sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            results.append({
                "name": path.parent.name,
                "path": str(path),
                "version": sha,
            })
        return results

    def save(self, config: HarnessConfig, name: Optional[str] = None) -> str:
        """Save a harness config to its subdirectory. Returns the version SHA."""
        target_name = name or config.name or "custom"
        harness_subdir = self.harness_dir / target_name
        harness_subdir.mkdir(parents=True, exist_ok=True)
        path = harness_subdir / "harness.json"
        data = config.as_dict()
        data.pop("version", None)
        raw = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        path.write_text(raw, encoding="utf-8")
        sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return sha

    def load_validator(
        self,
        skill_name: str,
        harness_name: str = "default",
        base_dir: Path | None = None,
    ):
        """Load a validator function, applying any harness-specific patch.

        Checks harness_versions/<harness>/patches/<skill_name>.py for a Python
        override module. Falls back to skills/mechanistic/<skill_name>/validator.py.

        Returns the validator module (not the function — callers pick the function).
        """
        # Check for harness-specific patch first
        patch_path = self.harness_dir / harness_name / "patches" / f"{skill_name}.py"
        if patch_path.exists():
            spec = importlib.util.spec_from_file_location(
                f"harness_patch.{harness_name}.{skill_name}", patch_path
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)  # type: ignore[union-attr]
                return module

        # Fall back to ground truth in skills/mechanistic/
        root = base_dir or Path.cwd()
        ground_truth_path = root / "skills" / "mechanistic" / skill_name / "validator.py"
        if ground_truth_path.exists():
            spec = importlib.util.spec_from_file_location(
                f"skills.mechanistic.{skill_name}.validator", ground_truth_path
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)  # type: ignore[union-attr]
                return module

        return None

    def resolve_from_run_config(self, run_config: RunConfig) -> HarnessConfig:
        """Load harness from run_config, applying legacy flag overrides."""
        if run_config.harness_config_path:
            path = Path(run_config.harness_config_path)
            if path.exists():
                raw = path.read_text(encoding="utf-8")
                data = json.loads(raw)
                config = HarnessConfig.from_dict(data)
                config.version = hashlib.sha256(raw.encode("utf-8")).hexdigest()
                return config
            # Try as harness name (subdirectory)
            name = path.stem if path.suffix == ".json" else path.name
            return self.load(name)

        harness_name = str(getattr(run_config, "harness_name", "") or "default").strip() or "default"
        config = self.load(harness_name)
        self._apply_legacy_overrides(config, run_config)
        return config

    @staticmethod
    def _apply_legacy_overrides(config: HarnessConfig, run_config: RunConfig) -> None:
        """Apply backward-compatible RunConfig flags to harness modules."""
        optional_tools = set(run_config.optional_llm_tools)
        for module in config.pre_loop_modules + config.post_step_modules:
            gate = module.config_gate
            if gate is None:
                continue
            if gate == "functional_groups_enabled":
                module.enabled = bool(run_config.functional_groups_enabled)
            elif gate == "step_mapping_enabled":
                module.enabled = bool(run_config.step_mapping_enabled)
            elif gate.startswith("optional_llm_tools:"):
                tool_name = gate.split(":", 1)[1]
                module.enabled = tool_name in optional_tools
            elif gate.startswith("reaction_template_policy:"):
                expected_value = gate.split(":", 1)[1]
                module.enabled = run_config.reaction_template_policy == expected_value


class RegistrySet:
    """Collection wrapper for all curated registries."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.prompts = PromptRegistry(mechanistic_skills_root(base_dir))
        self.skills = SkillRegistry(base_dir / "skills")
        self.memory_packs = MemoryPackRegistry(base_dir / "memory_packs")
        self.harness = HarnessRegistry(base_dir / "harness_versions")

    def bundle_hashes(self, *, model_name: str | None = None) -> Dict[str, str]:
        hashes: Dict[str, str] = {
            "prompt_bundle_hash": self.prompts.bundle_hash(model_name=model_name),
            "skill_bundle_hash": self.skills.bundle_hash(),
            "memory_bundle_hash": self.memory_packs.bundle_hash(),
        }
        try:
            harness = self.harness.load("default")
            hashes["harness_bundle_hash"] = harness.version
        except (FileNotFoundError, json.JSONDecodeError):
            hashes["harness_bundle_hash"] = ""
        return hashes

    def all_assets(self, *, model_name: str | None = None) -> List[AssetRecord]:
        return self.prompts.list(model_name=model_name) + self.skills.list() + self.memory_packs.list()

    def curated_memory_items(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for record in self.memory_packs.list():
            tags = ["curated", "memory_pack", record.metadata.get("name", "")]
            items.append(
                {
                    "scope": "curated",
                    "key": record.metadata.get("name") or Path(record.path).stem,
                    "source": "curated",
                    "confidence": 1.0,
                    "tags": [tag for tag in tags if tag],
                    "value": {
                        "path": record.path,
                        "sha256": record.sha256,
                        "metadata": record.metadata,
                    },
                }
            )
        return items

    def harness_version(self) -> str:
        """Compute a single harness version string from all bundle hashes."""
        hashes = self.bundle_hashes()
        combined = "|".join([
            hashes.get("prompt_bundle_hash", ""),
            hashes.get("skill_bundle_hash", ""),
            hashes.get("memory_bundle_hash", ""),
            hashes.get("harness_bundle_hash", ""),
        ])
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def prompt_step_map(self, *, model_name: str | None = None) -> Dict[str, Dict[str, Any]]:
        mapping: Dict[str, Dict[str, Any]] = {}
        for item in list_call_prompt_versions(self.base_dir, model_name=model_name):
            call_name = str(item.get("call_name") or "")
            if not call_name:
                continue
            for step in list(item.get("steps") or []):
                step_name = str(step or "").strip()
                if not step_name:
                    continue
                mapping[step_name] = {
                    "name": call_name,
                    "call_name": call_name,
                    "version": str(item.get("prompt_bundle_sha256") or "")[:12],
                    "path": item.get("call_base_path"),
                    "sha256": item.get("prompt_bundle_sha256"),
                    "template": item.get("template"),
                    "shared_template": item.get("shared_template"),
                    "shared_base_sha256": item.get("shared_base_sha256"),
                    "call_base_sha256": item.get("call_base_sha256"),
                    "few_shot_sha256": item.get("few_shot_sha256"),
                    "prompt_bundle_sha256": item.get("prompt_bundle_sha256"),
                    "few_shot_path": item.get("few_shot_path"),
                    "resolved_shared_base_path": item.get("resolved_shared_base_path"),
                    "resolved_call_base_path": item.get("resolved_call_base_path"),
                    "resolved_few_shot_path": item.get("resolved_few_shot_path"),
                    "asset_scope": item.get("asset_scope"),
                    "model_name": item.get("model_name"),
                }
        return mapping
