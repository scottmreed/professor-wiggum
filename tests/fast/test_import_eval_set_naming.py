"""`import-eval-set` names imports after their file instead of a hard-coded default."""
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import main as cli
from mechanistic_agent.core.db import RunStore
from mechanistic_agent.data_paths import db_path


def _write_cases(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "id": "case_a",
                    "starting_materials": ["CCBr", "[Cl-]"],
                    "products": ["CCCl", "[Br-]"],
                    "known_mechanism": {"steps": [{"step_index": 1, "target_smiles": "CCCl"}]},
                }
            ]
        ),
        encoding="utf-8",
    )


def test_resolve_import_name_prefers_stem_and_keeps_default_alias(tmp_path: Path) -> None:
    practice = tmp_path / "training_data" / "practice_eval" / "practice_set.json"
    practice.parent.mkdir(parents=True)
    practice.write_text("[]", encoding="utf-8")
    default = tmp_path / "training_data" / "eval_set.json"
    default.write_text("[]", encoding="utf-8")

    assert cli._resolve_import_eval_set_name(practice, base=tmp_path) == "practice_set"
    assert cli._resolve_import_eval_set_name(default, base=tmp_path) == "flower_100_default"
    assert cli._resolve_import_eval_set_name(practice, explicit_name="my_set", base=tmp_path) == "my_set"


def test_import_eval_set_cli_uses_file_stem(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MECHANISTIC_DATA_DIR", str(tmp_path / "wiggum-data"))
    (tmp_path / "training_data").mkdir()
    cases_path = tmp_path / "training_data" / "practice_set.json"
    _write_cases(cases_path)

    result = CliRunner().invoke(
        cli.app,
        ["import-eval-set", "--path", str(cases_path), "--version", "practice_v1", "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.index("{"):])
    assert payload["name"] == "practice_set"

    store = RunStore(db_path(tmp_path))
    names = {item["name"] for item in store.list_eval_sets()}
    assert "practice_set" in names
    assert "flower_100_default" not in names

    # Re-importing the same file/version is a no-op (dedup keyed on the real name).
    again = CliRunner().invoke(
        cli.app,
        ["import-eval-set", "--path", str(cases_path), "--version", "practice_v1", "--json"],
    )
    assert again.exit_code == 0, again.output
    assert json.loads(again.output[again.output.index("{"):]).get("existing") is True
