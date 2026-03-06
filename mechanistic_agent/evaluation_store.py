"""Local storage for LLM evaluation traces."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


def _default_store_path() -> Path:
    base_dir = Path.cwd() / "traces" / "baselines"
    return base_dir / "evaluation_traces.json"


class EvaluationTraceStore:
    """Persist evaluation traces keyed by step/model/reasoning level."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or _default_store_path()

    @staticmethod
    def build_key(step_name: str, model: str, reasoning: Optional[str]) -> str:
        level = (reasoning or "default").strip() or "default"
        return f"{step_name}::{model}::{level}"

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"entries": {}}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"entries": {}}

    def save(self, payload: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def get_entry(self, key: str) -> Optional[Dict[str, Any]]:
        data = self.load()
        entries = data.get("entries", {})
        if isinstance(entries, dict):
            entry = entries.get(key)
            if isinstance(entry, dict):
                return entry
        return None

    def upsert_entry(
        self,
        *,
        key: str,
        step_name: str,
        model: str,
        reasoning: Optional[str],
        trace: Dict[str, Any],
    ) -> Dict[str, Any]:
        data = self.load()
        entries = data.setdefault("entries", {})
        if not isinstance(entries, dict):
            entries = {}
            data["entries"] = entries
        entry = {
            "step": step_name,
            "model": model,
            "reasoning": reasoning,
            "stored_at": time.time(),
            "trace": trace,
        }
        entries[key] = entry
        self.save(data)
        return entry
