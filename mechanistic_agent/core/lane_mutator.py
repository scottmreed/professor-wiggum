"""Lane-scoped mutators for overnight Ralph experiments."""
from __future__ import annotations

import json
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class MutatedAsset:
    """Result of one lane mutation proposal."""

    lane: str
    asset_path: Path
    summary: str
    metadata: Dict[str, Any]


class TopologyLaneMutator:
    """Mutates one topology profile field in a harness JSON file."""

    _NUMERIC_FIELDS = ("agent_count", "max_candidates_per_agent", "peer_rounds")

    def __init__(self, *, rng: Optional[random.Random] = None) -> None:
        self._rng = rng or random.Random()

    def propose(self, parent_asset_path: Path) -> MutatedAsset:
        payload = json.loads(parent_asset_path.read_text(encoding="utf-8"))
        profiles = payload.get("topology_profiles") or {}
        if not isinstance(profiles, dict) or not profiles:
            raise ValueError("No topology_profiles found in harness asset")

        profile_key = sorted(str(k) for k in profiles.keys())[0]
        profile = dict(profiles.get(profile_key) or {})

        field = self._NUMERIC_FIELDS[self._rng.randrange(0, len(self._NUMERIC_FIELDS))]
        old_value = int(profile.get(field, 1))
        delta = -1 if self._rng.random() < 0.5 else 1
        lower_bound = 0 if field == "peer_rounds" else 1
        new_value = max(lower_bound, old_value + delta)
        if new_value == old_value:
            new_value = old_value + 1
        profile[field] = new_value
        profiles[profile_key] = profile
        payload["topology_profiles"] = profiles

        out_path = parent_asset_path.with_name(f"{parent_asset_path.stem}.topology_mutated.json")
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        return MutatedAsset(
            lane="topology",
            asset_path=out_path,
            summary=f"{profile_key}.{field}: {old_value} -> {new_value}",
            metadata={"profile": profile_key, "field": field, "from": old_value, "to": new_value},
        )


class HarnessLaneMutator:
    """Toggles one module enabled flag in a harness JSON file."""

    def propose(self, parent_asset_path: Path) -> MutatedAsset:
        payload = json.loads(parent_asset_path.read_text(encoding="utf-8"))
        module_lists: List[str] = ["pre_loop_modules", "post_step_modules", "post_loop_modules"]

        target_list_name = ""
        target_idx = -1
        old_enabled = True
        for list_name in module_lists:
            modules = payload.get(list_name) or []
            if not isinstance(modules, list):
                continue
            for idx, module in enumerate(modules):
                if not isinstance(module, dict):
                    continue
                if not bool(module.get("removable", True)):
                    continue
                target_list_name = list_name
                target_idx = idx
                old_enabled = bool(module.get("enabled", True))
                break
            if target_idx >= 0:
                break

        if target_idx < 0:
            raise ValueError("No mutable harness module found")

        modules = list(payload.get(target_list_name) or [])
        module = dict(modules[target_idx])
        module_id = str(module.get("id") or f"{target_list_name}[{target_idx}]")
        module["enabled"] = not old_enabled
        modules[target_idx] = module
        payload[target_list_name] = modules

        out_path = parent_asset_path.with_name(f"{parent_asset_path.stem}.harness_mutated.json")
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return MutatedAsset(
            lane="harness",
            asset_path=out_path,
            summary=f"toggle {module_id}: {old_enabled} -> {not old_enabled}",
            metadata={"module_id": module_id, "from": old_enabled, "to": (not old_enabled)},
        )


class PromptLaneMutator:
    """Creates a minimally-edited SKILL.md prompt variant for experiment tracking."""

    def __init__(self, *, base_dir: Path, call_name: str = "propose_mechanism_step") -> None:
        self.base_dir = base_dir
        self.call_name = call_name

    def propose(self, parent_asset_path: Path) -> MutatedAsset:
        src = self.base_dir / "skills" / "mechanistic" / self.call_name / "SKILL.md"
        if not src.exists():
            raise FileNotFoundError(f"Prompt source not found: {src}")
        text = src.read_text(encoding="utf-8")
        marker = "<!-- PROMPT_END -->"
        if marker not in text:
            raise ValueError("Prompt markers not found in SKILL.md")
        stamp = int(time.time())
        insertion = f"\n\n- Mutation note: prefer concise mechanism-step proposals. ({stamp})\n"
        mutated_text = text.replace(marker, f"{insertion}{marker}")

        out_path = parent_asset_path.with_name(f"prompt_variant_{stamp}.SKILL.md")
        out_path.write_text(mutated_text, encoding="utf-8")
        return MutatedAsset(
            lane="prompt",
            asset_path=out_path,
            summary=f"appended one targeted instruction to {self.call_name} prompt",
            metadata={"call_name": self.call_name, "source": str(src)},
        )


class FewShotLaneMutator:
    """Creates a small few-shot variant by dropping one example from JSONL."""

    def __init__(self, *, base_dir: Path, call_name: str = "propose_mechanism_step") -> None:
        self.base_dir = base_dir
        self.call_name = call_name

    def propose(self, parent_asset_path: Path) -> MutatedAsset:
        src = self.base_dir / "skills" / "mechanistic" / self.call_name / "few_shot.jsonl"
        if not src.exists():
            raise FileNotFoundError(f"Few-shot source not found: {src}")
        lines = [line for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) <= 1:
            out_path = parent_asset_path.with_name(f"few_shot_variant_{int(time.time())}.jsonl")
            shutil.copyfile(src, out_path)
            return MutatedAsset(
                lane="few_shot",
                asset_path=out_path,
                summary="few-shot unchanged (<=1 example)",
                metadata={"removed_index": None, "source": str(src)},
            )

        removed_index = len(lines) - 1
        kept = lines[:-1]
        out_path = parent_asset_path.with_name(f"few_shot_variant_{int(time.time())}.jsonl")
        out_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        return MutatedAsset(
            lane="few_shot",
            asset_path=out_path,
            summary=f"removed few-shot example at index {removed_index}",
            metadata={"removed_index": removed_index, "source": str(src)},
        )
