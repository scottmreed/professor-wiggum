from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("rdkit")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUILDER_PATH = PROJECT_ROOT / "scripts" / "build_flower_mechanism_dataset.py"
RENDERER_PATH = PROJECT_ROOT / "scripts" / "render_flower_mechanism_pngs.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_module(BUILDER_PATH, "flower_builder_png")
renderer = _load_module(RENDERER_PATH, "flower_renderer_png")


def _fixture_input(tmp_path: Path) -> Path:
    path = tmp_path / "flower_fixture.txt"
    path.write_text(
        "".join(
            [
                "[O-:1].[H+:2]>>[O:1][H:2]|1\n",
                "[Cl-:1].[CH3:2][Br:3]>>[CH3:2][Cl:1].[Br-:3]|2\n",
                "[O-:1].[CH2:2]=[O:3]>>[O:1][CH2:2][O-:3]|3\n",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_render_flower_mechanism_pngs_smoke(tmp_path: Path) -> None:
    input_path = _fixture_input(tmp_path)
    dataset, _report = builder.build_dataset(input_path=input_path, sample_size=2, cache_path=tmp_path / "lookup.sqlite")
    dataset_path = tmp_path / "flower_cases.json"
    dataset_path.write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")

    output_dir = tmp_path / "pngs"
    index = renderer.render_pngs(input_path=dataset_path, output_dir=output_dir)

    assert index["rendered_count"] == 2
    assert (output_dir / "index.json").exists()
    pngs = sorted(output_dir.glob("*.png"))
    assert len(pngs) == 2


def test_render_curriculum_pngs_writes_ranked_index_and_respects_only_missing(tmp_path: Path) -> None:
    input_path = _fixture_input(tmp_path)
    dataset, _report = builder.build_dataset(
        input_path=input_path,
        sample_size=2,
        index_output=tmp_path / "index.jsonl",
        index_report=tmp_path / "index_report.json",
        cache_path=tmp_path / "lookup.sqlite",
    )

    output_dir = tmp_path / "curriculum_pngs"
    index_a = renderer.render_curriculum_pngs(
        input_path=input_path,
        index_path=tmp_path / "index.jsonl",
        cache_path=tmp_path / "lookup.sqlite",
        output_dir=output_dir,
        top_n=2,
    )
    index_b = renderer.render_curriculum_pngs(
        input_path=input_path,
        index_path=tmp_path / "index.jsonl",
        cache_path=tmp_path / "lookup.sqlite",
        output_dir=output_dir,
        top_n=2,
        only_missing=True,
    )

    assert index_a["rendered_count"] == 2
    assert index_b["rendered_count"] == 0
    manifest = json.loads((output_dir / "index.json").read_text(encoding="utf-8"))
    assert len(manifest["items"]) == 2
    assert "global_rank" in manifest["items"][0]
    assert manifest["items"][0]["png"] == "10001.png"
    assert manifest["items"][1]["png"] == "10002.png"
