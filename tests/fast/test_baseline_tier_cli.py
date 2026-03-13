from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer

import main as main_module
from main import (
    _build_baseline_tier_execution_plan,
    _list_attempted_eval_case_ids_for_scope,
    _load_baseline_tier_eval_set_map,
    _normalize_requested_baseline_tiers,
    _official_case_ids_for_tier,
    _run_baseline_eval_set,
    _select_case_ids_resume_then_cycle,
    baseline,
    eval_cmd,
)
from mechanistic_agent.core.db import RunStore
from mechanistic_agent.core.baseline_runner import BASELINE_GROUP_PREFIX


def _seed_eval_set(store: RunStore, *, name: str, case_id: str, purpose: str = "general") -> str:
    return store.add_eval_set(
        name=name,
        version="v1",
        source_path=None,
        sha256=None,
        cases=[
            {
                "case_id": case_id,
                "input": {
                    "starting_materials": ["CCBr", "[Cl-]"],
                    "products": ["CCCl", "[Br-]"],
                    "temperature_celsius": 25.0,
                    "ph": None,
                },
                "expected": {"products": ["CCCl", "[Br-]"]},
            }
        ],
        active=True,
        purpose=purpose,
        exposed_in_ui=(purpose != "leaderboard_holdout"),
    )


def _write_eval_tiers(base: Path, *, easy: list[str], medium: list[str], hard: list[str]) -> None:
    training = base / "training_data"
    training.mkdir(parents=True, exist_ok=True)
    (training / "eval_tiers.json").write_text(
        json.dumps(
            {
                "_meta": {"source": "pytest"},
                "easy": easy,
                "medium": medium,
                "hard": hard,
            }
        ),
        encoding="utf-8",
    )


def test_normalize_requested_baseline_tiers_order_and_dedup() -> None:
    assert _normalize_requested_baseline_tiers(None, all_tiers=True) == ["easy", "medium", "hard"]
    assert _normalize_requested_baseline_tiers(["hard", "easy", "hard"], all_tiers=False) == ["hard", "easy"]
    with pytest.raises(typer.BadParameter):
        _normalize_requested_baseline_tiers(["legendary"], all_tiers=False)


def test_official_case_ids_for_tier_uses_step_bands() -> None:
    cases = [
        {"case_id": "case_easy_1", "expected": {"n_mechanistic_steps": 1}},
        {"case_id": "case_easy_2", "expected": {"n_mechanistic_steps": 2}},
        {"case_id": "case_medium", "expected": {"n_mechanistic_steps": 3}},
        {"case_id": "case_hard_1", "expected": {"n_mechanistic_steps": 4}},
        {"case_id": "case_hard_2", "expected": {"n_mechanistic_steps": 9}},
        {"case_id": "case_unknown", "expected": {}},
    ]

    assert _official_case_ids_for_tier(cases=cases, tier_name="easy") == ["case_easy_1", "case_easy_2"]
    assert _official_case_ids_for_tier(cases=cases, tier_name="medium") == ["case_medium"]
    assert _official_case_ids_for_tier(cases=cases, tier_name="hard") == ["case_hard_1", "case_hard_2"]


def test_select_case_ids_resume_then_cycle_uses_unrun_then_wrap() -> None:
    selected, meta = _select_case_ids_resume_then_cycle(
        candidate_case_ids=["case_a", "case_b", "case_c", "case_d"],
        attempted_case_ids=["case_a", "case_b", "case_d"],
        max_cases=3,
    )

    assert selected == ["case_c", "case_a", "case_b"]
    assert meta["candidate_count"] == 4
    assert meta["unrun_count"] == 1
    assert meta["wrapped"] is True


def test_select_case_ids_resume_then_cycle_cycles_from_start_when_exhausted() -> None:
    selected, meta = _select_case_ids_resume_then_cycle(
        candidate_case_ids=["case_a", "case_b", "case_c"],
        attempted_case_ids=["case_a", "case_b", "case_c"],
        max_cases=2,
    )

    assert selected == ["case_a", "case_b"]
    assert meta["unrun_count"] == 0
    assert meta["wrapped"] is True


def test_list_attempted_eval_case_ids_for_scope_filters_by_run_group_and_thinking(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "data" / "mechanistic.db")
    eval_set_id = _seed_eval_set(store, name="official", case_id="case_a", purpose="leaderboard_holdout")

    run_1 = store.create_eval_run(
        eval_set_id=eval_set_id,
        run_group_name="official_a",
        model="anthropic/claude-opus-4.6",
        model_name="anthropic/claude-opus-4.6",
        model_family="claude",
        thinking_level="high",
        harness_bundle_hash="hash",
        status="completed",
    )
    store.record_eval_run_result(
        eval_run_id=run_1,
        case_id="case_a",
        run_id=None,
        score=1.0,
        passed=True,
        cost={},
        latency_ms=1.0,
        summary={},
    )

    run_2 = store.create_eval_run(
        eval_set_id=eval_set_id,
        run_group_name="official_b",
        model="anthropic/claude-opus-4.6",
        model_name="anthropic/claude-opus-4.6",
        model_family="claude",
        thinking_level="high",
        harness_bundle_hash="hash",
        status="completed",
    )
    store.record_eval_run_result(
        eval_run_id=run_2,
        case_id="case_b",
        run_id=None,
        score=0.0,
        passed=False,
        cost={},
        latency_ms=1.0,
        summary={},
    )

    run_3 = store.create_eval_run(
        eval_set_id=eval_set_id,
        run_group_name="official_a",
        model="anthropic/claude-opus-4.6",
        model_name="anthropic/claude-opus-4.6",
        model_family="claude",
        thinking_level="low",
        harness_bundle_hash="hash",
        status="completed",
    )
    store.record_eval_run_result(
        eval_run_id=run_3,
        case_id="case_c",
        run_id=None,
        score=0.0,
        passed=False,
        cost={},
        latency_ms=1.0,
        summary={},
    )

    attempted = _list_attempted_eval_case_ids_for_scope(
        store=store,
        eval_set_id=eval_set_id,
        model_name="anthropic/claude-opus-4.6",
        thinking_level="high",
        run_group_name="official_a",
    )
    assert attempted == ["case_a"]


def test_load_baseline_tier_eval_set_map_requires_all_tiers(tmp_path: Path) -> None:
    mapping_path = tmp_path / "tier_map.json"
    mapping_path.write_text(
        json.dumps(
            {
                "easy": {"eval_set_id": "abc"},
                "hard": {"eval_set_id": "xyz"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(typer.BadParameter):
        _load_baseline_tier_eval_set_map(mapping_path)


def test_run_baseline_eval_set_tracks_completed_passed_failed(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "data" / "mechanistic.db")
    eval_set_id = store.add_eval_set(
        name="baseline_eval",
        version="v1",
        source_path=None,
        sha256=None,
        cases=[],
        active=True,
        purpose="general",
        exposed_in_ui=True,
    )

    resolved_eval_set = SimpleNamespace(
        eval_set_id=eval_set_id,
        purpose="general",
        cases=[
            {
                "case_id": "case_pass",
                "input": {"starting_materials": ["CCBr"], "products": ["CCCl"]},
                "expected": {},
            },
            {
                "case_id": "case_fail",
                "input": {"starting_materials": ["CCO"], "products": ["CC=O"]},
                "expected": {},
            },
        ],
    )

    class _Runner:
        def __init__(self) -> None:
            self.calls = 0

        def run_case(self, **_kwargs):  # noqa: ANN003
            self.calls += 1
            passed = self.calls == 1
            return {
                "latency_ms": 12.5,
                "prompt_hash": f"hash-{self.calls}",
                "prompt_system_hash": f"sys-{self.calls}",
                "prompt_user_hash": f"user-{self.calls}",
                "passed": passed,
            }

    def _score(result, _expected):  # noqa: ANN001
        passed = bool(result.get("passed"))
        return {
            "score": 1.0 if passed else 0.0,
            "passed": passed,
            "step_count": 1,
            "mechanism_type": "test",
            "scoring_breakdown": {},
            "error": None,
        }

    result = _run_baseline_eval_set(
        runner=_Runner(),
        score_baseline_result_fn=_score,
        store=store,
        run_group_name="baseline_test_group",
        resolved_eval_set=resolved_eval_set,
        model_name="anthropic/claude-opus-4.6",
        model_family="claude",
        thinking_level="high",
        temperature=25.0,
        ph=None,
        max_cases=10,
        timeout=10.0,
        llm_seed=42,
        llm_temperature=0.0,
        sampling_policy="fixed",
        harness_hash="bundle-hash",
        case_ids=None,
        api_keys=None,
    )

    assert result["completed"] == 2
    assert result["passed"] == 1
    assert result["failed"] == 1
    assert result["errored"] == 0


def test_build_baseline_tier_execution_plan_rejects_unknown_eval_set_id(tmp_path: Path) -> None:
    base = tmp_path
    _write_eval_tiers(base, easy=["case_easy"], medium=[], hard=[])
    tier_defs_path = base / "training_data" / "eval_tiers.json"
    store = RunStore(base / "data" / "mechanistic.db")
    with pytest.raises(typer.BadParameter, match="unknown eval_set_id"):
        _build_baseline_tier_execution_plan(
            base=base,
            store=store,
            requested_tiers=["easy"],
            tier_eval_set_ids={"easy": "missing"},
            tier_definitions_path=tier_defs_path,
            allow_holdout=False,
        )


def test_build_baseline_tier_execution_plan_rejects_holdout_eval_set(tmp_path: Path) -> None:
    base = tmp_path
    _write_eval_tiers(base, easy=["case_easy"], medium=[], hard=[])
    tier_defs_path = base / "training_data" / "eval_tiers.json"
    store = RunStore(base / "data" / "mechanistic.db")
    holdout_id = _seed_eval_set(
        store,
        name="holdout",
        case_id="case_easy",
        purpose="leaderboard_holdout",
    )
    with pytest.raises(typer.BadParameter, match="holdout eval set"):
        _build_baseline_tier_execution_plan(
            base=base,
            store=store,
            requested_tiers=["easy"],
            tier_eval_set_ids={"easy": holdout_id},
            tier_definitions_path=tier_defs_path,
            allow_holdout=False,
        )


def test_build_baseline_tier_execution_plan_fails_for_empty_tier_intersection(tmp_path: Path) -> None:
    base = tmp_path
    _write_eval_tiers(base, easy=["case_not_in_eval_set"], medium=[], hard=[])
    tier_defs_path = base / "training_data" / "eval_tiers.json"
    store = RunStore(base / "data" / "mechanistic.db")
    eval_set_id = _seed_eval_set(store, name="general", case_id="case_easy")
    with pytest.raises(typer.BadParameter, match="Tier 'easy' has 0 cases"):
        _build_baseline_tier_execution_plan(
            base=base,
            store=store,
            requested_tiers=["easy"],
            tier_eval_set_ids={"easy": eval_set_id},
            tier_definitions_path=tier_defs_path,
            allow_holdout=False,
        )


def test_baseline_all_tiers_runs_in_easy_medium_hard_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base = tmp_path
    _write_eval_tiers(base, easy=["case_easy"], medium=["case_medium"], hard=["case_hard"])
    store = RunStore(base / "data" / "mechanistic.db")
    easy_id = _seed_eval_set(store, name="easy_set", case_id="case_easy")
    medium_id = _seed_eval_set(store, name="medium_set", case_id="case_medium")
    hard_id = _seed_eval_set(store, name="hard_set", case_id="case_hard")
    tier_map_path = base / "training_data" / "tier_map.json"
    tier_map_path.write_text(
        json.dumps(
            {
                "easy": {"eval_set_id": easy_id},
                "medium": {"eval_set_id": medium_id},
                "hard": {"eval_set_id": hard_id},
            }
        ),
        encoding="utf-8",
    )

    run_groups: list[str] = []
    seen_api_keys: list[dict[str, str] | None] = []

    def _fake_run_baseline_eval_set(**kwargs):  # noqa: ANN003
        run_group_name = str(kwargs["run_group_name"])
        resolved_eval_set = kwargs["resolved_eval_set"]
        run_groups.append(run_group_name)
        seen_api_keys.append(kwargs.get("api_keys"))
        return {
            "eval_run_id": f"run_{run_group_name}",
            "model": kwargs["model_name"],
            "thinking_level": kwargs["thinking_level"],
            "completed": 1,
            "failed": 0,
            "eval_set_id": resolved_eval_set.eval_set_id,
            "eval_set_purpose": resolved_eval_set.purpose,
            "eval_case_ids_hash": "hash",
            "llm_seed": kwargs["llm_seed"],
            "llm_temperature": kwargs["llm_temperature"],
            "sampling_policy": kwargs["sampling_policy"],
            "prompt_hashes": [],
            "run_group_name": run_group_name,
        }

    monkeypatch.chdir(base)
    monkeypatch.setattr("main._run_baseline_eval_set", _fake_run_baseline_eval_set)
    monkeypatch.setattr(
        "main._load_api_keys",
        lambda: {"openrouter_api_key": "or-test-key"},
    )

    baseline(
        starting=None,
        products=None,
        eval_set_id=None,
        tier=None,
        all_tiers=True,
        tier_map_path=str(tier_map_path),
        tier_definitions_path=None,
        run_group_prefix="rg",
        model_name="anthropic/claude-opus-4.6",
        thinking_level="high",
        temperature=25.0,
        ph=None,
        max_cases=25,
        timeout=1.0,
        llm_seed=42,
        llm_temperature=0.0,
        sampling_policy="fixed",
        allow_repeats=False,
        json_output=True,
        allow_holdout=False,
    )

    assert run_groups == ["rg_easy", "rg_medium", "rg_hard"]
    assert seen_api_keys == [
        {"openrouter_api_key": "or-test-key"},
        {"openrouter_api_key": "or-test-key"},
        {"openrouter_api_key": "or-test-key"},
    ]


def test_baseline_eval_set_mode_keeps_legacy_run_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base = tmp_path
    store = RunStore(base / "data" / "mechanistic.db")
    eval_set_id = _seed_eval_set(store, name="general", case_id="case_easy")
    seen_run_groups: list[str] = []

    def _fake_run_baseline_eval_set(**kwargs):  # noqa: ANN003
        seen_run_groups.append(str(kwargs["run_group_name"]))
        resolved_eval_set = kwargs["resolved_eval_set"]
        return {
            "eval_run_id": "run1",
            "model": kwargs["model_name"],
            "thinking_level": kwargs["thinking_level"],
            "completed": 1,
            "failed": 0,
            "eval_set_id": resolved_eval_set.eval_set_id,
            "eval_set_purpose": resolved_eval_set.purpose,
            "eval_case_ids_hash": "hash",
            "llm_seed": kwargs["llm_seed"],
            "llm_temperature": kwargs["llm_temperature"],
            "sampling_policy": kwargs["sampling_policy"],
            "prompt_hashes": [],
            "run_group_name": str(kwargs["run_group_name"]),
        }

    monkeypatch.chdir(base)
    monkeypatch.setattr("main._run_baseline_eval_set", _fake_run_baseline_eval_set)

    baseline(
        starting=None,
        products=None,
        eval_set_id=eval_set_id,
        tier=None,
        all_tiers=False,
        tier_map_path=None,
        tier_definitions_path=None,
        run_group_prefix="ignored",
        model_name="anthropic/claude-opus-4.6",
        thinking_level="high",
        temperature=25.0,
        ph=None,
        max_cases=25,
        timeout=1.0,
        llm_seed=42,
        llm_temperature=0.0,
        sampling_policy="fixed",
        allow_repeats=False,
        json_output=True,
        allow_holdout=False,
    )

    assert seen_run_groups == [BASELINE_GROUP_PREFIX]


def test_eval_all_tiers_uses_tier_map_and_suffixes_run_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base = tmp_path
    _write_eval_tiers(base, easy=["case_easy"], medium=["case_medium"], hard=["case_hard"])
    store = RunStore(base / "data" / "mechanistic.db")
    easy_id = _seed_eval_set(store, name="easy_set", case_id="case_easy")
    medium_id = _seed_eval_set(store, name="medium_set", case_id="case_medium")
    hard_id = _seed_eval_set(store, name="hard_set", case_id="case_hard")
    tier_map_path = base / "training_data" / "tier_map.json"
    tier_map_path.write_text(
        json.dumps(
            {
                "easy": {"eval_set_id": easy_id},
                "medium": {"eval_set_id": medium_id},
                "hard": {"eval_set_id": hard_id},
            }
        ),
        encoding="utf-8",
    )

    calls: list[dict] = []

    def _fake_eval_cmd(**kwargs):  # noqa: ANN003
        calls.append(dict(kwargs))

    monkeypatch.chdir(base)
    monkeypatch.setattr(main_module, "eval_cmd", _fake_eval_cmd)

    eval_cmd(
        eval_set_id="ignored",
        model_name="anthropic/claude-opus-4.6",
        thinking_level="low",
        tier=None,
        all_tiers=True,
        tier_map_path=str(tier_map_path),
        tier_definitions_path=str(base / "training_data" / "eval_tiers.json"),
        run_group_prefix="grp",
        case_ids=None,
        harness="default",
        run_group=None,
        max_cases=25,
        max_steps=10,
        max_runtime=60.0,
        allow_repeats=False,
        json_output=True,
        allow_holdout=False,
    )

    assert [c["eval_set_id"] for c in calls] == [easy_id, medium_id, hard_id]
    assert [c["run_group"] for c in calls] == ["grp_easy", "grp_medium", "grp_hard"]
    assert [len(c["case_ids"]) for c in calls] == [1, 1, 1]


def test_baseline_all_tiers_fail_fast_when_any_tier_has_no_unrun_cases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base = tmp_path
    _write_eval_tiers(base, easy=["case_easy"], medium=["case_medium"], hard=["case_hard"])
    store = RunStore(base / "data" / "mechanistic.db")
    easy_id = _seed_eval_set(store, name="easy_set", case_id="case_easy")
    medium_id = _seed_eval_set(store, name="medium_set", case_id="case_medium")
    hard_id = _seed_eval_set(store, name="hard_set", case_id="case_hard")
    tier_map_path = base / "training_data" / "tier_map.json"
    tier_map_path.write_text(
        json.dumps(
            {
                "easy": {"eval_set_id": easy_id},
                "medium": {"eval_set_id": medium_id},
                "hard": {"eval_set_id": hard_id},
            }
        ),
        encoding="utf-8",
    )

    prior_run = store.create_eval_run(
        eval_set_id=easy_id,
        run_group_name="prior_easy",
        model="anthropic/claude-opus-4.6",
        model_name="anthropic/claude-opus-4.6",
        model_family="claude",
        thinking_level="high",
        harness_bundle_hash="h",
        status="completed",
    )
    store.record_eval_run_result(
        eval_run_id=prior_run,
        case_id="case_easy",
        run_id=None,
        score=1.0,
        passed=True,
        cost={},
        latency_ms=1.0,
        summary={},
    )

    monkeypatch.chdir(base)
    monkeypatch.setattr("main._run_baseline_eval_set", lambda **_kwargs: {})

    with pytest.raises(typer.BadParameter, match="0 unrun cases"):
        baseline(
            starting=None,
            products=None,
            eval_set_id=None,
            tier=None,
            all_tiers=True,
            tier_map_path=str(tier_map_path),
            tier_definitions_path=None,
            run_group_prefix="rg",
            model_name="anthropic/claude-opus-4.6",
            thinking_level="high",
            temperature=25.0,
            ph=None,
            max_cases=25,
            timeout=1.0,
            llm_seed=42,
            llm_temperature=0.0,
            sampling_policy="fixed",
            allow_repeats=False,
            json_output=True,
            allow_holdout=False,
        )


def test_baseline_all_tiers_allow_repeats_bypasses_unrun_filter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base = tmp_path
    _write_eval_tiers(base, easy=["case_easy"], medium=["case_medium"], hard=["case_hard"])
    store = RunStore(base / "data" / "mechanistic.db")
    easy_id = _seed_eval_set(store, name="easy_set", case_id="case_easy")
    medium_id = _seed_eval_set(store, name="medium_set", case_id="case_medium")
    hard_id = _seed_eval_set(store, name="hard_set", case_id="case_hard")
    tier_map_path = base / "training_data" / "tier_map.json"
    tier_map_path.write_text(
        json.dumps(
            {
                "easy": {"eval_set_id": easy_id},
                "medium": {"eval_set_id": medium_id},
                "hard": {"eval_set_id": hard_id},
            }
        ),
        encoding="utf-8",
    )

    prior_run = store.create_eval_run(
        eval_set_id=easy_id,
        run_group_name="prior_easy",
        model="anthropic/claude-opus-4.6",
        model_name="anthropic/claude-opus-4.6",
        model_family="claude",
        thinking_level="high",
        harness_bundle_hash="h",
        status="completed",
    )
    store.record_eval_run_result(
        eval_run_id=prior_run,
        case_id="case_easy",
        run_id=None,
        score=1.0,
        passed=True,
        cost={},
        latency_ms=1.0,
        summary={},
    )

    run_groups: list[str] = []

    def _fake_run_baseline_eval_set(**kwargs):  # noqa: ANN003
        run_groups.append(str(kwargs["run_group_name"]))
        resolved_eval_set = kwargs["resolved_eval_set"]
        return {
            "eval_run_id": f"run_{kwargs['run_group_name']}",
            "model": kwargs["model_name"],
            "thinking_level": kwargs["thinking_level"],
            "completed": 1,
            "failed": 0,
            "eval_set_id": resolved_eval_set.eval_set_id,
            "eval_set_purpose": resolved_eval_set.purpose,
            "eval_case_ids_hash": "hash",
            "llm_seed": kwargs["llm_seed"],
            "llm_temperature": kwargs["llm_temperature"],
            "sampling_policy": kwargs["sampling_policy"],
            "prompt_hashes": [],
            "run_group_name": str(kwargs["run_group_name"]),
        }

    monkeypatch.chdir(base)
    monkeypatch.setattr("main._run_baseline_eval_set", _fake_run_baseline_eval_set)

    baseline(
        starting=None,
        products=None,
        eval_set_id=None,
        tier=None,
        all_tiers=True,
        tier_map_path=str(tier_map_path),
        tier_definitions_path=None,
        run_group_prefix="rg",
        model_name="anthropic/claude-opus-4.6",
        thinking_level="high",
        temperature=25.0,
        ph=None,
        max_cases=25,
        timeout=1.0,
        llm_seed=42,
        llm_temperature=0.0,
        sampling_policy="fixed",
        allow_repeats=True,
        json_output=True,
        allow_holdout=False,
    )

    assert run_groups == ["rg_easy", "rg_medium", "rg_hard"]


def test_eval_all_tiers_fail_fast_when_any_tier_has_no_unrun_cases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base = tmp_path
    _write_eval_tiers(base, easy=["case_easy"], medium=["case_medium"], hard=["case_hard"])
    store = RunStore(base / "data" / "mechanistic.db")
    easy_id = _seed_eval_set(store, name="easy_set", case_id="case_easy")
    medium_id = _seed_eval_set(store, name="medium_set", case_id="case_medium")
    hard_id = _seed_eval_set(store, name="hard_set", case_id="case_hard")
    tier_map_path = base / "training_data" / "tier_map.json"
    tier_map_path.write_text(
        json.dumps(
            {
                "easy": {"eval_set_id": easy_id},
                "medium": {"eval_set_id": medium_id},
                "hard": {"eval_set_id": hard_id},
            }
        ),
        encoding="utf-8",
    )

    prior_run = store.create_eval_run(
        eval_set_id=easy_id,
        run_group_name="prior_easy",
        model="anthropic/claude-opus-4.6",
        model_name="anthropic/claude-opus-4.6",
        model_family="claude",
        thinking_level="low",
        harness_bundle_hash="h",
        status="completed",
    )
    store.record_eval_run_result(
        eval_run_id=prior_run,
        case_id="case_easy",
        run_id=None,
        score=1.0,
        passed=True,
        cost={},
        latency_ms=1.0,
        summary={},
    )

    monkeypatch.chdir(base)
    monkeypatch.setattr(main_module, "eval_cmd", lambda **_kwargs: None)

    with pytest.raises(typer.BadParameter, match="0 unrun cases"):
        eval_cmd(
            eval_set_id="ignored",
            model_name="anthropic/claude-opus-4.6",
            thinking_level="low",
            tier=None,
            all_tiers=True,
            tier_map_path=str(tier_map_path),
            tier_definitions_path=str(base / "training_data" / "eval_tiers.json"),
            run_group_prefix="grp",
            case_ids=None,
            harness="default",
            run_group=None,
            max_cases=25,
            max_steps=10,
            max_runtime=60.0,
            allow_repeats=False,
            json_output=True,
            allow_holdout=False,
        )


def test_eval_all_tiers_allow_repeats_bypasses_unrun_filter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base = tmp_path
    _write_eval_tiers(base, easy=["case_easy"], medium=["case_medium"], hard=["case_hard"])
    store = RunStore(base / "data" / "mechanistic.db")
    easy_id = _seed_eval_set(store, name="easy_set", case_id="case_easy")
    medium_id = _seed_eval_set(store, name="medium_set", case_id="case_medium")
    hard_id = _seed_eval_set(store, name="hard_set", case_id="case_hard")
    tier_map_path = base / "training_data" / "tier_map.json"
    tier_map_path.write_text(
        json.dumps(
            {
                "easy": {"eval_set_id": easy_id},
                "medium": {"eval_set_id": medium_id},
                "hard": {"eval_set_id": hard_id},
            }
        ),
        encoding="utf-8",
    )

    prior_run = store.create_eval_run(
        eval_set_id=easy_id,
        run_group_name="prior_easy",
        model="anthropic/claude-opus-4.6",
        model_name="anthropic/claude-opus-4.6",
        model_family="claude",
        thinking_level="low",
        harness_bundle_hash="h",
        status="completed",
    )
    store.record_eval_run_result(
        eval_run_id=prior_run,
        case_id="case_easy",
        run_id=None,
        score=1.0,
        passed=True,
        cost={},
        latency_ms=1.0,
        summary={},
    )

    calls: list[dict] = []

    def _fake_eval_cmd(**kwargs):  # noqa: ANN003
        calls.append(dict(kwargs))

    monkeypatch.chdir(base)
    monkeypatch.setattr(main_module, "eval_cmd", _fake_eval_cmd)

    eval_cmd(
        eval_set_id="ignored",
        model_name="anthropic/claude-opus-4.6",
        thinking_level="low",
        tier=None,
        all_tiers=True,
        tier_map_path=str(tier_map_path),
        tier_definitions_path=str(base / "training_data" / "eval_tiers.json"),
        run_group_prefix="grp",
        case_ids=None,
        harness="default",
        run_group=None,
        max_cases=25,
        max_steps=10,
        max_runtime=60.0,
        allow_repeats=True,
        json_output=True,
        allow_holdout=False,
    )

    assert [c["eval_set_id"] for c in calls] == [easy_id, medium_id, hard_id]
