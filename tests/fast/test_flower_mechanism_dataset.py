from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("rdkit")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUILDER_PATH = PROJECT_ROOT / "scripts" / "build_flower_mechanism_dataset.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_module(BUILDER_PATH, "flower_builder")


def _fixture_lines() -> list[str]:
    return [
        "[O-:1].[H+:2]>>[O:1][H:2]|10\n",
        "[Cl-:1].[CH3:2][Br:3]>>[CH3:2][Cl:1].[Br-:3]|2\n",
        "[O-:1].[CH2:2]=[O:3]>>[O:1][CH2:2][O-:3]|11\n",
        "[O-:1].[H+:2]>>[O-:1].[H+:2]|5\n",
        "[O-:1].[H+:2]>>[O:1][H:2]|abc\n",
    ]


def _write_fixture_input(tmp_path: Path) -> Path:
    path = tmp_path / "flower_fixture.txt"
    path.write_text("".join(_fixture_lines()), encoding="utf-8")
    return path


def test_convert_elementary_step_supports_lp_sigma_and_pi() -> None:
    lp = builder.convert_elementary_step("[O-:1].[H+:2]>>[O:1][H:2]")
    assert lp.electron_pushes[0]["notation"] == "lp:1>2"

    sigma = builder.convert_elementary_step("[Cl-:1].[CH3:2][Br:3]>>[CH3:2][Cl:1].[Br-:3]")
    assert [move["notation"] for move in sigma.electron_pushes] == ["lp:1>2", "sigma:2-3>3"]

    pi = builder.convert_elementary_step("[O-:1].[CH2:2]=[O:3]>>[O:1][CH2:2][O-:3]")
    assert [move["notation"] for move in pi.electron_pushes] == ["lp:1>2", "pi:2-3>3"]


def test_build_dataset_produces_ranked_verified_mechanism_cases(tmp_path: Path) -> None:
    input_path = _write_fixture_input(tmp_path)
    index_output = tmp_path / "index.jsonl"
    index_report = tmp_path / "index_report.json"
    dataset, report = builder.build_dataset(
        input_path=input_path,
        sample_size=3,
        index_output=index_output,
        index_report=index_report,
        cache_path=tmp_path / "lookup.sqlite",
    )

    assert len(dataset) == 3
    assert report["sample_size_written"] == 3
    assert report["candidate_attempt_count"] >= 3
    assert index_output.exists()
    assert index_report.exists()

    for case in dataset:
        assert {"id", "name", "starting_materials", "products", "verified_mechanism"} <= set(case)
        steps = list((case["verified_mechanism"] or {}).get("steps") or [])
        assert steps
        assert case["n_mechanistic_steps"] == len(steps)
        for step in steps:
            assert step["step_index"] >= 1
            assert step["current_state"]
            assert step["resulting_state"]
            assert "|mech:v1;" in step["reaction_smirks"]
            assert step["electron_pushes"]
            normalized = builder.normalize_electron_pushes(step["electron_pushes"])
            assert len(normalized) == len(step["electron_pushes"])

    index_rows = [json.loads(line) for line in index_output.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [row["case_id"] for row in index_rows] == ["flower_000010", "flower_000002", "flower_000011"]
    assert [row["global_rank"] for row in index_rows] == [1, 2, 3]


def test_build_dataset_is_deterministic_without_random_sampling(tmp_path: Path) -> None:
    input_path = _write_fixture_input(tmp_path)
    dataset_a, report_a = builder.build_dataset(
        input_path=input_path,
        sample_size=3,
        cache_path=tmp_path / "lookup_a.sqlite",
    )
    dataset_b, report_b = builder.build_dataset(
        input_path=input_path,
        sample_size=3,
        cache_path=tmp_path / "lookup_b.sqlite",
    )

    assert json.dumps(dataset_a, sort_keys=True) == json.dumps(dataset_b, sort_keys=True)
    assert report_a["selected_case_ids"] == report_b["selected_case_ids"]
