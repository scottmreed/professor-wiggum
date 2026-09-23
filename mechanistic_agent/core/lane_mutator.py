"""Lane-scoped mutators for overnight Ralph experiments."""
from __future__ import annotations

import contextlib
import hashlib
import json
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


@dataclass(slots=True)
class MutatedAsset:
    """Result of one lane mutation proposal."""

    lane: str
    asset_path: Path
    summary: str
    metadata: Dict[str, Any]


HARNESS_ASSET_LANES = frozenset({"topology", "harness"})
CALL_ASSET_LANES = frozenset({"prompt", "few_shot"})


def mutated_call_name(asset: MutatedAsset) -> Optional[str]:
    """Return the skill call_name a prompt / few-shot variant was derived from."""
    metadata = asset.metadata or {}
    call_name = str(metadata.get("call_name") or "").strip()
    if call_name:
        return call_name
    source = str(metadata.get("source") or "").strip()
    return Path(source).parent.name if source else None


def mutated_scope_model(asset: MutatedAsset) -> Optional[str]:
    """Model whose ``models/<slug>/`` lane a prompt / few-shot variant replaces (``None`` = shared)."""
    value = str((asset.metadata or {}).get("scope_model") or "").strip()
    return value or None


@dataclass(frozen=True, slots=True)
class CallAssetVariant:
    """One prompt / few-shot variant file and the asset it replaces."""

    call_name: str
    kind: str  # "prompt" | "few_shot"
    path: Path
    scope_model: Optional[str] = None  # None = shared call asset

    @property
    def key(self) -> Tuple[str, str, Optional[str]]:
        return (self.call_name, self.kind, self.scope_model)


@dataclass(frozen=True, slots=True)
class AssetState:
    """The assets an evolution lineage node is evaluated under.

    ``harness_path`` is the harness JSON to run (``None`` = the run's normal
    harness); ``call_variants`` are the prompt / few-shot variants installed on
    top of the committed skills. A kept mutation becomes the parent of later
    mutations by folding it in with :meth:`with_mutation`.
    """

    harness_path: Optional[Path] = None
    call_variants: Tuple[CallAssetVariant, ...] = ()

    def with_mutation(self, asset: MutatedAsset) -> "AssetState":
        lane = str(asset.lane or "")
        if lane in HARNESS_ASSET_LANES:
            return AssetState(harness_path=Path(asset.asset_path), call_variants=self.call_variants)
        if lane not in CALL_ASSET_LANES:
            raise ValueError(f"Unsupported mutation lane: {lane!r}")
        call_name = mutated_call_name(asset)
        if not call_name:
            raise ValueError(f"Cannot tell which call the {lane} variant {asset.asset_path} belongs to")
        variant = CallAssetVariant(
            call_name=call_name,
            kind="prompt" if lane == "prompt" else "few_shot",
            path=Path(asset.asset_path),
            scope_model=mutated_scope_model(asset),
        )
        kept = tuple(v for v in self.call_variants if v.key != variant.key)
        return AssetState(harness_path=self.harness_path, call_variants=kept + (variant,))

    def paths(self) -> List[Path]:
        out = [self.harness_path] if self.harness_path is not None else []
        return out + [v.path for v in self.call_variants]

    def missing_paths(self) -> List[Path]:
        return [path for path in self.paths() if not path.exists()]

    def fingerprint(self, default_harness: Path) -> str:
        """sha256 of the harness text, extended with each call variant when there are any."""
        harness = self.harness_path or default_harness
        digest = hashlib.sha256(harness.read_text(encoding="utf-8").encode("utf-8"))
        if not self.call_variants:
            return digest.hexdigest()
        for variant in sorted(self.call_variants, key=lambda v: (v.call_name, v.kind, v.scope_model or "")):
            digest.update(f"|{variant.call_name}:{variant.kind}:{variant.scope_model or ''}|".encode("utf-8"))
            digest.update(variant.path.read_bytes())
        return digest.hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "harness_path": str(self.harness_path) if self.harness_path is not None else None,
            "call_variants": [
                {"call_name": v.call_name, "kind": v.kind, "path": str(v.path), "scope_model": v.scope_model}
                for v in self.call_variants
            ],
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AssetState":
        data = data if isinstance(data, dict) else {}
        harness = str(data.get("harness_path") or "").strip()
        variants = []
        for row in data.get("call_variants") or []:
            if not isinstance(row, dict):
                continue
            kind = str(row.get("kind") or "")
            if kind not in {"prompt", "few_shot"}:
                raise ValueError(f"Unknown call variant kind: {kind!r}")
            variants.append(
                CallAssetVariant(
                    call_name=str(row["call_name"]),
                    kind=kind,
                    path=Path(str(row["path"])),
                    scope_model=(str(row.get("scope_model") or "").strip() or None),
                )
            )
        return cls(harness_path=Path(harness) if harness else None, call_variants=tuple(variants))


@contextlib.contextmanager
def applied_assets(state: AssetState) -> Iterator[Optional[str]]:
    """Make ``state`` the assets that an in-process evaluation (or mutation) resolves.

    Yields the harness path to pass as the run's ``harness_config_path`` (``None``
    keeps the run's normal harness). Each call variant is installed via
    :func:`mechanistic_agent.prompt_assets.call_asset_overrides` at the scope it
    was derived from.
    """
    from mechanistic_agent.prompt_assets import call_asset_overrides

    with contextlib.ExitStack() as stack:
        for variant in state.call_variants:
            key = "prompts" if variant.kind == "prompt" else "few_shots"
            stack.enter_context(
                call_asset_overrides(**{key: {variant.call_name: variant.path}}, model_name=variant.scope_model)
            )
        yield str(state.harness_path) if state.harness_path is not None else None


@contextlib.contextmanager
def applied_mutation(asset: MutatedAsset, parent: Optional[AssetState] = None) -> Iterator[Optional[str]]:
    """Make ``asset`` (on top of ``parent``) the variant that an in-process evaluation resolves.

    Mutators write sibling files and leave the committed assets untouched, so
    an evaluation only sees a mutation if it is pointed at the variant:

    * ``topology`` / ``harness`` variants are harness JSON files: the context
      yields their path, to be passed as the run's ``harness_config_path``.
    * ``prompt`` / ``few_shot`` variants are installed for the duration of the
      context via :func:`mechanistic_agent.prompt_assets.call_asset_overrides`
      at the scope they were derived from (yields the parent's harness path, or
      ``None``: the run keeps its normal harness).
    """
    with applied_assets((parent or AssetState()).with_mutation(asset)) as harness_override:
        yield harness_override


def snapshot_mutated_asset(asset: MutatedAsset, dest_dir: Path, *, prefix: str = "") -> MutatedAsset:
    """Copy ``asset``'s variant file into ``dest_dir`` and return the asset pointing at the copy.

    Mutators name variants after their parent (``<stem>.topology_mutated.json``)
    or a one-second timestamp, so a later mutation can overwrite the file a kept
    lineage node points at. Loops that carry a variant forward snapshot it first.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    source = Path(asset.asset_path)
    target = dest_dir / f"{prefix}{source.name}"
    shutil.copyfile(source, target)
    return MutatedAsset(
        lane=asset.lane,
        asset_path=target,
        summary=asset.summary,
        metadata={**dict(asset.metadata or {}), "scratch_asset_path": str(source)},
    )


def resolve_call_source(
    base_dir: Path, call_name: str, kind: str, model_name: Optional[str]
) -> Tuple[Path, Optional[str]]:
    """``(path, scope_model)`` of the asset a run for ``model_name`` resolves (honours active variants)."""
    from mechanistic_agent.prompt_assets import resolve_call_asset_source

    return resolve_call_asset_source(call_name, kind, base_dir, model_name)


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

    def __init__(
        self,
        *,
        base_dir: Path,
        call_name: str = "propose_mechanism_step",
        model_name: Optional[str] = None,
    ) -> None:
        self.base_dir = base_dir
        self.call_name = call_name
        self.model_name = model_name

    def propose(self, parent_asset_path: Path) -> MutatedAsset:
        # Derive from the prompt a run for ``model_name`` resolves: its per-model
        # override, or an active (kept) variant, not always the base SKILL.md.
        src, scope_model = resolve_call_source(self.base_dir, self.call_name, "prompt", self.model_name)
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
            metadata={"call_name": self.call_name, "source": str(src), "scope_model": scope_model},
        )


class FewShotLaneMutator:
    """Creates a small few-shot variant by dropping one example from JSONL."""

    def __init__(
        self,
        *,
        base_dir: Path,
        call_name: str = "propose_mechanism_step",
        model_name: Optional[str] = None,
    ) -> None:
        self.base_dir = base_dir
        self.call_name = call_name
        self.model_name = model_name

    def propose(self, parent_asset_path: Path) -> MutatedAsset:
        src, scope_model = resolve_call_source(self.base_dir, self.call_name, "few_shot", self.model_name)
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
                metadata={"removed_index": None, "call_name": self.call_name, "source": str(src), "scope_model": scope_model},
            )

        removed_index = len(lines) - 1
        kept = lines[:-1]
        out_path = parent_asset_path.with_name(f"few_shot_variant_{int(time.time())}.jsonl")
        out_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        return MutatedAsset(
            lane="few_shot",
            asset_path=out_path,
            summary=f"removed few-shot example at index {removed_index}",
            metadata={"removed_index": removed_index, "call_name": self.call_name, "source": str(src), "scope_model": scope_model},
        )
