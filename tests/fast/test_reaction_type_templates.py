from __future__ import annotations

import json
import shutil
from pathlib import Path

from mechanistic_agent.core.reaction_type_templates import (
    load_reaction_type_catalog,
    load_reaction_type_catalog_for_runtime,
)


def _taxonomy_labels() -> list[str]:
    path = Path("training_data/reaction_type_templates.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    templates = list(payload.get("templates") or [])
    def _sort_key(row: dict) -> int:
        type_id = str(row.get("type_id") or "")
        if type_id.startswith("mt_"):
            try:
                return int(type_id.split("_", 1)[1])
            except ValueError:
                return 999
        return 999
    templates.sort(key=_sort_key)
    return [str(row.get("label_exact") or "") for row in templates]


def test_reaction_type_catalog_matches_taxonomy_labels_exactly() -> None:
    catalog = load_reaction_type_catalog_for_runtime()
    labels = list(catalog.get("taxonomy_labels") or [])
    assert labels == _taxonomy_labels()
    assert len(labels) == 51


def test_reaction_type_catalog_has_stable_ids() -> None:
    catalog = load_reaction_type_catalog_for_runtime()
    templates = list(catalog.get("templates") or [])
    ids = [str(item.get("type_id") or "") for item in templates if isinstance(item, dict)]
    assert len(ids) == 51
    assert ids[0] == "mt_001"
    assert ids[-1] == "mt_051"


def test_runtime_catalog_loads_from_reaction_type_templates_only(tmp_path: Path) -> None:
    training_dir = tmp_path / "training_data"
    training_dir.mkdir(parents=True)
    shutil.copy(
        Path("training_data/reaction_type_templates.json"),
        training_dir / "reaction_type_templates.json",
    )
    # Ensure no runtime dependency on rxn_map.txt for loading.
    assert not (training_dir / "rxn_map.txt").exists()

    catalog = load_reaction_type_catalog(tmp_path)
    assert len(list(catalog.get("templates") or [])) == 51
    assert list(catalog.get("taxonomy_labels") or [])[0]


def test_catalog_contains_example_mappings() -> None:
    catalog = load_reaction_type_catalog_for_runtime()
    mappings = list(catalog.get("example_mappings") or [])
    assert mappings
    assert any(str(row.get("reaction_id", "")).startswith("hb350_") for row in mappings)
