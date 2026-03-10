from __future__ import annotations

import json

import pytest

from mechanistic_agent.core.db import RunStore
from mechanistic_agent.eval_set_resolution import (
    EvalSetResolutionError,
    case_ids_hash,
    load_eval_cases_from_json,
    resolve_eval_set,
    select_eval_cases,
)


def _seed_eval_set(store: RunStore, *, name: str, purpose: str, case_ids: list[str]) -> str:
    cases = []
    for case_id in case_ids:
        cases.append(
            {
                "case_id": case_id,
                "input": {
                    "starting_materials": ["CCBr", "[Cl-]"],
                    "products": ["CCCl", "[Br-]"],
                    "temperature_celsius": 25.0,
                    "ph": 7.0,
                },
                "expected": {"products": ["CCCl", "[Br-]"]},
            }
        )
    return store.add_eval_set(
        name=name,
        version="v1",
        source_path=f"{name}.json",
        sha256=None,
        cases=cases,
        active=True,
        purpose=purpose,
        exposed_in_ui=(purpose != "leaderboard_holdout"),
    )


def test_resolve_eval_set_defaults_to_latest_holdout_when_requested(tmp_path) -> None:
    store = RunStore(tmp_path / "data" / "mechanistic.db")
    _seed_eval_set(store, name="general_set", purpose="general", case_ids=["g1"])
    holdout_id = _seed_eval_set(
        store,
        name="official_holdout",
        purpose="leaderboard_holdout",
        case_ids=["h1", "h2"],
    )

    resolved = resolve_eval_set(
        store=store,
        requested_eval_set_id=None,
        require_purpose="leaderboard_holdout",
        default_purpose="leaderboard_holdout",
    )

    assert resolved.eval_set_id == holdout_id
    assert resolved.purpose == "leaderboard_holdout"
    assert resolved.case_ids == ["h1", "h2"]
    assert resolved.case_ids_hash == case_ids_hash(["h1", "h2"])


def test_resolve_eval_set_rejects_purpose_mismatch(tmp_path) -> None:
    store = RunStore(tmp_path / "data" / "mechanistic.db")
    general_id = _seed_eval_set(store, name="general_set", purpose="general", case_ids=["g1"])

    with pytest.raises(EvalSetResolutionError):
        resolve_eval_set(
            store=store,
            requested_eval_set_id=general_id,
            require_purpose="leaderboard_holdout",
        )


def test_select_eval_cases_preserves_requested_order() -> None:
    cases = [{"case_id": "a"}, {"case_id": "b"}, {"case_id": "c"}]

    selected = select_eval_cases(cases=cases, case_ids=["c", "a"])

    assert [row["case_id"] for row in selected] == ["c", "a"]


def test_load_eval_cases_from_training_style_json(tmp_path) -> None:
    data = [
        {
            "id": "flower_000001",
            "starting_materials": ["CCBr", "[Cl-]"],
            "products": ["CCCl", "[Br-]"],
            "temperature_celsius": 30.0,
            "ph": 8.0,
            "known_mechanism": {"min_steps": 1},
        }
    ]
    eval_path = tmp_path / "eval_set.json"
    eval_path.write_text(json.dumps(data), encoding="utf-8")

    cases = load_eval_cases_from_json(eval_path)

    assert len(cases) == 1
    assert cases[0]["case_id"] == "flower_000001"
    assert cases[0]["input"]["temperature_celsius"] == 30.0
    assert "known_mechanism" in cases[0]["expected"]
