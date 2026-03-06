"""Opus-first curriculum orchestration and public history helpers."""
from __future__ import annotations

import importlib.util
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

from mechanistic_agent.core import RegistrySet, RunStore
from mechanistic_agent.flower_curriculum import DEFAULT_INDEX_PATH, ensure_index, load_curriculum_index
from mechanistic_agent.prompt_assets import load_call_few_shot_examples, score_few_shot_example

OPUS_MODEL = "anthropic/claude-opus-4.6"
COURSE_PATH = Path("curriculum/course.yaml")
CHECKPOINTS_DIR = Path("curriculum/checkpoints")
GENERATED_DIR = Path("curriculum/generated")
LEADERBOARD_JSON_PATH = GENERATED_DIR / "leaderboard_opus.json"
README_CONTEXT_PATH = GENERATED_DIR / "readme_context.json"

DEFAULT_COURSE_CONFIG: Dict[str, Any] = {
    "name": "Mechanistic Curriculum",
    "focus_label": "Trainee Curriculum",
    "start_date": "2026-03-11",
    "timezone": "America/Denver",
    "release_time": {"hour": 17, "minute": 0},
    "default_harness": "default",
    "default_eval_set_name": "flower_100_default",
    "default_group_sizes": {"lesson": 4, "quiz": 6},
    "weekdays": {
        "0": {"kind": "lesson", "label": "Monday lesson"},
        "1": {"kind": "lesson", "label": "Tuesday lesson"},
        "2": {"kind": "lesson", "label": "Wednesday lesson"},
        "3": {"kind": "lesson", "label": "Thursday lesson"},
        "4": {"kind": "quiz", "label": "Friday quiz"},
    },
    "quiz_unlock": {
        "required_pass_count": 4,
        "required_batch_size": 6,
        "required_mean_quality": 0.70,
    },
    "modules": [
        {"id": "module_01", "number": 1, "label": "1-step reactions", "min_step_count": 1, "max_step_count": 1},
        {"id": "module_02", "number": 2, "label": "2-step reactions", "min_step_count": 2, "max_step_count": 2},
        {"id": "module_03", "number": 3, "label": "3-step reactions", "min_step_count": 3, "max_step_count": 3},
        {"id": "module_04", "number": 4, "label": "4-step reactions", "min_step_count": 4, "max_step_count": 4},
        {"id": "module_05", "number": 5, "label": "5-6 step reactions", "min_step_count": 5, "max_step_count": 6},
        {"id": "module_06", "number": 6, "label": "7-8 step reactions", "min_step_count": 7, "max_step_count": 8},
    ],
}


@dataclass(frozen=True)
class ReleaseSlot:
    release_date: str
    weekday_index: int
    label: str
    release_kind: str
    scheduled_publish_at: float
    scheduled_publish_at_iso: str


def _load_yaml_or_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return json.loads(json.dumps(DEFAULT_COURSE_CONFIG))
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return json.loads(json.dumps(DEFAULT_COURSE_CONFIG))
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(text)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    raise ValueError(f"Unable to parse curriculum config: {path}")


def load_course_config(base_dir: Path) -> Dict[str, Any]:
    raw = json.loads(json.dumps(DEFAULT_COURSE_CONFIG))
    loaded = _load_yaml_or_json(base_dir / COURSE_PATH)
    raw.update({key: value for key, value in loaded.items() if key not in {"modules", "weekdays", "release_time", "quiz_unlock", "default_group_sizes"}})
    raw["modules"] = list(loaded.get("modules") or raw["modules"])
    raw["weekdays"] = dict(loaded.get("weekdays") or raw["weekdays"])
    raw["release_time"] = dict(loaded.get("release_time") or raw["release_time"])
    raw["quiz_unlock"] = dict(loaded.get("quiz_unlock") or raw["quiz_unlock"])
    raw["default_group_sizes"] = dict(loaded.get("default_group_sizes") or raw["default_group_sizes"])
    return raw


def _now_in_course_tz(config: Dict[str, Any], now: datetime | None = None) -> datetime:
    tz = ZoneInfo(str(config.get("timezone") or "America/Denver"))
    current = now or datetime.now(tz)
    if current.tzinfo is None:
        current = current.replace(tzinfo=tz)
    return current.astimezone(tz)


def _course_start_date(config: Dict[str, Any]) -> date:
    raw = str(config.get("start_date") or "").strip()
    if not raw:
        return date(2026, 3, 11)
    return date.fromisoformat(raw)


def _scheduled_release_datetime(config: Dict[str, Any], release_day: date) -> datetime:
    tz = ZoneInfo(str(config.get("timezone") or "America/Denver"))
    release_time = config.get("release_time") or {}
    return datetime(
        release_day.year,
        release_day.month,
        release_day.day,
        int(release_time.get("hour", 17)),
        int(release_time.get("minute", 0)),
        0,
        0,
        tzinfo=tz,
    )


def _iter_release_days(config: Dict[str, Any], *, start_day: date, count: int) -> List[date]:
    weekdays = config.get("weekdays") or {}
    cursor = max(start_day, _course_start_date(config))
    days: List[date] = []
    while len(days) < count:
        if str(cursor.weekday()) in weekdays:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _next_release_slot(config: Dict[str, Any], now: datetime | None = None) -> Optional[ReleaseSlot]:
    current = _now_in_course_tz(config, now)
    start_day = max(current.date(), _course_start_date(config))
    for release_day in _iter_release_days(config, start_day=start_day, count=14):
        weekday_key = str(release_day.weekday())
        day_config = (config.get("weekdays") or {}).get(weekday_key)
        if not isinstance(day_config, dict):
            continue
        scheduled = _scheduled_release_datetime(config, release_day)
        if scheduled < current:
            continue
        return ReleaseSlot(
            release_date=release_day.isoformat(),
            weekday_index=release_day.weekday(),
            label=str(day_config.get("label") or day_config.get("kind") or "release"),
            release_kind=str(day_config.get("kind") or "lesson"),
            scheduled_publish_at=scheduled.timestamp(),
            scheduled_publish_at_iso=scheduled.isoformat(),
        )
    return None


def current_release_slot(config: Dict[str, Any], now: datetime | None = None) -> Optional[ReleaseSlot]:
    current = _now_in_course_tz(config, now)
    if current.date() < _course_start_date(config):
        return None
    weekday_key = str(current.weekday())
    day_config = (config.get("weekdays") or {}).get(weekday_key)
    if not isinstance(day_config, dict):
        return None
    scheduled = _scheduled_release_datetime(config, current.date())
    return ReleaseSlot(
        release_date=current.date().isoformat(),
        weekday_index=current.weekday(),
        label=str(day_config.get("label") or day_config.get("kind") or "release"),
        release_kind=str(day_config.get("kind") or "lesson"),
        scheduled_publish_at=scheduled.timestamp(),
        scheduled_publish_at_iso=scheduled.isoformat(),
    )


def _course_modules(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    modules = []
    for index, module in enumerate(config.get("modules") or [], start=1):
        item = dict(module)
        item.setdefault("number", index)
        modules.append(item)
    return modules


def _published_checkpoints(store: RunStore, *, model_name: str) -> List[Dict[str, Any]]:
    rows = store.list_curriculum_checkpoints(model_name=model_name, limit=500)
    rows.sort(key=lambda row: (str(row.get("release_date") or ""), float(row.get("created_at") or 0.0)))
    return rows


def _current_module(config: Dict[str, Any], store: RunStore, *, model_name: str) -> Dict[str, Any]:
    modules = _course_modules(config)
    if not modules:
        raise ValueError("Curriculum course config is missing modules")
    module_index = 0
    for checkpoint in _published_checkpoints(store, model_name=model_name):
        if str(checkpoint.get("release_kind") or "") != "quiz":
            continue
        summary = checkpoint.get("summary") or {}
        if bool(summary.get("quiz_passed")) and module_index < len(modules) - 1:
            module_index += 1
    return dict(modules[module_index])


def _calendar_dates(config: Dict[str, Any], current: datetime) -> List[date]:
    start_day = max(current.date(), _course_start_date(config))
    return _iter_release_days(config, start_day=start_day, count=10)


def _format_countdown(target: datetime, current: datetime) -> Dict[str, Any]:
    total_seconds = max(0, int((target - current).total_seconds()))
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    return {
        "days": days,
        "hours": hours,
        "minutes": minutes,
        "total_seconds": total_seconds,
        "label": f"{days}d {hours}h",
        "detailed_label": f"{days}d {hours}h {minutes}m",
    }


def _resolve_eval_set_id(store: RunStore, config: Dict[str, Any]) -> Optional[str]:
    desired_name = str(config.get("default_eval_set_name") or "flower_100_default")
    for item in store.list_eval_sets():
        if str(item.get("name") or "") == desired_name:
            return str(item.get("id") or "")
    return None


def _resolve_leaderboard_row(store: RunStore, config: Dict[str, Any], *, model_name: str) -> Optional[Dict[str, Any]]:
    eval_set_id = _resolve_eval_set_id(store, config)
    if not eval_set_id:
        return None
    for row in store.leaderboard(eval_set_id, limit=50):
        if str(row.get("model_name") or row.get("model") or "") == model_name and not row.get("is_baseline"):
            return row
    return None


def _entries_for_module(entries: Sequence[Dict[str, Any]], module: Dict[str, Any]) -> List[Dict[str, Any]]:
    min_step = int(module.get("min_step_count") or 1)
    max_step = int(module.get("max_step_count") or min_step)
    filtered = [dict(entry) for entry in entries if min_step <= int(entry.get("step_count") or 0) <= max_step]
    filtered.sort(key=lambda entry: (int(entry.get("global_rank") or 10**9), str(entry.get("case_id") or "")))
    return filtered


def _partition_module_pool(entries: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    quiz_pool: List[Dict[str, Any]] = []
    lesson_pool: List[Dict[str, Any]] = []
    for index, entry in enumerate(entries, start=1):
        if index % 5 == 0:
            quiz_pool.append(dict(entry))
        else:
            lesson_pool.append(dict(entry))
    return {"lesson": lesson_pool, "quiz": quiz_pool}


def _used_case_ids(store: RunStore, *, model_name: str, module_id: str) -> set[str]:
    used: set[str] = set()
    for release in store.list_curriculum_releases(model_name=model_name, limit=500):
        payload = release.get("payload") or {}
        if str(release.get("module_id") or "") != module_id:
            continue
        for case_id in payload.get("selected_case_ids") or []:
            used.add(str(case_id))
    for checkpoint in store.list_curriculum_checkpoints(model_name=model_name, limit=500):
        if str(checkpoint.get("module_id") or "") != module_id:
            continue
        summary = checkpoint.get("summary") or {}
        for case_id in summary.get("selected_case_ids") or []:
            used.add(str(case_id))
    return used


def _selected_entries_for_slot(
    store: RunStore,
    config: Dict[str, Any],
    *,
    model_name: str,
    module: Dict[str, Any],
    release_kind: str,
) -> List[Dict[str, Any]]:
    ensure_index(index_path=Path(config.get("curriculum_index_path") or DEFAULT_INDEX_PATH))
    index_path = Path(config.get("curriculum_index_path") or DEFAULT_INDEX_PATH)
    entries = load_curriculum_index(index_path)
    module_entries = _entries_for_module(entries, module)
    pools = _partition_module_pool(module_entries)
    available_pool = pools["quiz" if release_kind == "quiz" else "lesson"]
    used_case_ids = _used_case_ids(store, model_name=model_name, module_id=str(module.get("id") or ""))
    filtered = [entry for entry in available_pool if str(entry.get("case_id") or "") not in used_case_ids]
    batch_size = int((config.get("default_group_sizes") or {}).get(release_kind, 4))
    return filtered[:batch_size]


def _best_few_shot_score_for_model(call_name: str, base_dir: Path, *, model_name: str) -> float:
    examples = load_call_few_shot_examples(call_name, base_dir, model_name=model_name)
    if not examples:
        return 0.0
    scores = [
        score_few_shot_example(
            call_name,
            input_text=str(example.get("input") or ""),
            output_text=str(example.get("output") or ""),
            explicit_score=example.get("score") if isinstance(example.get("score"), (int, float)) else None,
        )
        for example in examples
    ]
    return max(scores) if scores else 0.0


def _load_evolve_module(base_dir: Path):
    script_path = base_dir / "scripts" / "evolve_harness.py"
    spec = importlib.util.spec_from_file_location("mechanistic_curriculum_evolve", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load evolve_harness module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _queue_payload_from_results(
    *,
    config: Dict[str, Any],
    slot: ReleaseSlot,
    module: Dict[str, Any],
    eval_run_id: str,
    case_results: List[Dict[str, Any]],
    examples_mined: Dict[str, int],
    selected_entries: Sequence[Dict[str, Any]],
    leaderboard_row: Optional[Dict[str, Any]],
    prompt_assets: Dict[str, Any],
    harness_sha: str,
) -> Dict[str, Any]:
    scores = [float(result.get("score") or 0.0) for result in case_results]
    pass_count = sum(1 for result in case_results if result.get("passed"))
    mean_score = (sum(scores) / len(scores)) if scores else 0.0
    quiz_unlock = config.get("quiz_unlock") or {}
    quiz_passed = (
        slot.release_kind != "quiz"
        or (
            pass_count >= int(quiz_unlock.get("required_pass_count", 4))
            and len(case_results) >= int(quiz_unlock.get("required_batch_size", 6))
            and mean_score >= float(quiz_unlock.get("required_mean_quality", 0.70))
        )
    )
    return {
        "course_name": config.get("name"),
        "release_date": slot.release_date,
        "release_kind": slot.release_kind,
        "release_label": slot.label,
        "scheduled_publish_at": slot.scheduled_publish_at,
        "scheduled_publish_at_iso": slot.scheduled_publish_at_iso,
        "module": module,
        "eval_run_id": eval_run_id,
        "selected_case_ids": [str(entry.get("case_id") or "") for entry in selected_entries],
        "selected_entries": [dict(entry) for entry in selected_entries],
        "mean_quality_score": round(mean_score, 4),
        "pass_count": pass_count,
        "case_count": len(case_results),
        "quiz_passed": quiz_passed,
        "examples_mined": examples_mined,
        "leaderboard_row": dict(leaderboard_row or {}),
        "prompt_assets": prompt_assets,
        "harness_sha": harness_sha,
        "case_results": [
            {
                key: value
                for key, value in result.items()
                if key not in {"step_outputs", "input_payload"}
            }
            for result in case_results
        ],
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_readme_context(base_dir: Path, store: RunStore, *, model_name: str = OPUS_MODEL, now: datetime | None = None) -> Dict[str, Any]:
    config = load_course_config(base_dir)
    current = _now_in_course_tz(config, now)
    module = _current_module(config, store, model_name=model_name)
    queue_rows = store.list_curriculum_releases(model_name=model_name, limit=100)
    checkpoints = store.list_curriculum_checkpoints(model_name=model_name, limit=50)
    slot = current_release_slot(config, current)
    next_slot = _next_release_slot(config, current)
    week_rows: List[Dict[str, Any]] = []
    queued_by_date = {(str(row.get("release_date") or ""), str(row.get("release_kind") or "")): row for row in queue_rows}
    checkpoint_by_date = {(str(row.get("release_date") or ""), str(row.get("release_kind") or "")): row for row in checkpoints}
    for item_date in _calendar_dates(config, current):
        weekday_key = str(item_date.weekday())
        day_cfg = (config.get("weekdays") or {}).get(weekday_key)
        if not isinstance(day_cfg, dict):
            continue
        kind = str(day_cfg.get("kind") or "lesson")
        date_key = item_date.isoformat()
        checkpoint = checkpoint_by_date.get((date_key, kind))
        queued = queued_by_date.get((date_key, kind))
        status = "scheduled"
        if checkpoint:
            status = "published"
        elif queued:
            status = str(queued.get("status") or "queued")
        release_dt = _scheduled_release_datetime(config, item_date)
        week_rows.append(
            {
                "date": date_key,
                "weekday": item_date.strftime("%A"),
                "label": str(day_cfg.get("label") or kind),
                "release_kind": kind,
                "status": status,
                "checkpoint_id": checkpoint.get("id") if checkpoint else None,
                "queue_id": queued.get("id") if queued else None,
                "scheduled_publish_at_iso": release_dt.isoformat(),
            }
        )
    model_slug = str(model_name or OPUS_MODEL).replace("/", "__")
    leaderboard_filename = f"leaderboard_{model_slug}.json"
    models_dir = base_dir / "skills" / "mechanistic" / "propose_mechanism_step" / "models"
    registered_trainees = []
    if models_dir.is_dir():
        for d in sorted(models_dir.iterdir()):
            if d.is_dir():
                registered_trainees.append({
                    "slug": d.name,
                    "path": f"skills/mechanistic/propose_mechanism_step/models/{d.name}/",
                })
    prompt_asset_links = {
        "shared_skills": "skills/mechanistic/",
        "override_guide": "docs/model_asset_overrides.md",
        "opus_base_override": f"skills/mechanistic/base_system/models/{model_slug}/SKILL.md",
        "opus_few_shot_dir": f"skills/mechanistic/propose_mechanism_step/models/{model_slug}/",
    }
    curriculum_links = {
        "calendar": "curriculum/generated/readme_context.json",
        "leaderboard": f"curriculum/generated/{leaderboard_filename}",
        "checkpoints": "curriculum/checkpoints/",
        "upcoming_reactions": "training_data/flower_curriculum_pngs/index.json",
        "operations": "docs/curriculum_operations.md",
        "history": "docs/history_and_reproducibility.md",
    }
    context = {
        "generated_at": current.isoformat(),
        "course_name": config.get("name"),
        "focus_label": config.get("focus_label") or "Trainee Curriculum",
        "timezone": config.get("timezone"),
        "model_name": model_name,
        "trainee_name": "Claude Opus",
        "student_label": "trainee",
        "start_date": str(config.get("start_date") or _course_start_date(config).isoformat()),
        "current_module": module,
        "today_slot": None if slot is None else {
            "release_date": slot.release_date,
            "label": slot.label,
            "release_kind": slot.release_kind,
            "scheduled_publish_at_iso": slot.scheduled_publish_at_iso,
        },
        "next_slot": None if next_slot is None else {
            "release_date": next_slot.release_date,
            "label": next_slot.label,
            "release_kind": next_slot.release_kind,
            "scheduled_publish_at_iso": next_slot.scheduled_publish_at_iso,
        },
        "latest_leaderboard_row": _resolve_leaderboard_row(store, config, model_name=model_name),
        "calendar": week_rows,
        "checkpoints": checkpoints,
        "registered_trainees": registered_trainees,
        "curriculum_links": curriculum_links,
        "prompt_asset_links": prompt_asset_links,
    }
    _write_json(base_dir / README_CONTEXT_PATH, context)
    _write_json(base_dir / GENERATED_DIR / leaderboard_filename, {"item": context["latest_leaderboard_row"]})
    return context


def render_curriculum_readme(base_dir: Path, store: RunStore, *, model_name: str = OPUS_MODEL, now: datetime | None = None) -> str:
    context = build_readme_context(base_dir, store, model_name=model_name, now=now)
    current_module = context.get("current_module") or {}
    latest = context.get("latest_leaderboard_row") or {}
    next_slot = context.get("next_slot") or {}
    links = context.get("curriculum_links") or {}
    prompt_links = context.get("prompt_asset_links") or {}
    registered_trainees = context.get("registered_trainees") or []
    trainee_links = " | ".join(
        f"[{t['slug']}]({t['path']})" for t in registered_trainees
    ) if registered_trainees else "_none registered yet_"
    model_slug = str(model_name or OPUS_MODEL).replace("/", "__")
    lines = [
        "# Mechanistic Curriculum",
        "",
        '<img align="right" src="docs/readme_ralph.png" alt="Ralph" width="260" />',
        "",
        "## Orchestration Modes",
        "",
        "RAlph mode provides iterative multi-attempt orchestration with budget controls for enhanced mechanism prediction reliability.",
        "",
        "## Program Status",
        "",
        f"- Course: `{context.get('course_name')}`",
        f"- Launch: `{context.get('start_date')}`",
        f"- Module: `Module {current_module.get('number', 1)}` — {current_module.get('label', 'n/a')}",
        "",
        f"**Trainees:** {trainee_links}",
        "",
        "Quick links: "
        f"[Checkpoints]({links.get('checkpoints')}) | "
        f"[Reactions]({links.get('upcoming_reactions')}) | "
        f"[Prompt guide]({prompt_links.get('override_guide')}) | "
        f"[History]({links.get('history')})",
        "",
        "## Current Two-Week Calendar",
        "",
    ]
    for item in context.get("calendar") or []:
        marker = "x" if str(item.get("status") or "") == "published" else " "
        status_text = f"{item.get('release_kind')} {item.get('status')}"
        lines.append(
            f"- [{marker}] {item.get('date')} {item.get('weekday')}: {item.get('label')} "
            f"({status_text}, release `{item.get('scheduled_publish_at_iso')}`)"
        )
    lines.extend(["", "## Trainee Progress Snapshot", ""])
    if latest:
        lines.extend(
            [
                f"- Trainee: `{context.get('trainee_name')}` — [leaderboard]({links.get('leaderboard')})",
                f"- Mean quality: `{float(latest.get('mean_quality_score') or 0.0):.3f}`",
                f"- Pass rate: `{float(latest.get('deterministic_pass_rate') or 0.0) * 100.0:.1f}%`",
                f"- Cases: `{latest.get('case_count') or 0}`",
                f"- Run group: `{latest.get('run_group_name') or 'n/a'}`",
            ]
        )
    else:
        for t in registered_trainees:
            t_lb = f"curriculum/generated/leaderboard_{t['slug']}.json"
            lines.append(f"- [`{t['slug']}`]({t['path']}) — quality: `—` pass-rate: `—` [leaderboard]({t_lb})")
        if not registered_trainees:
            lines.append("- No trainees registered yet.")
    lines.extend(["", "## Checkpoints", ""])
    all_checkpoints = context.get("checkpoints") or []
    for cp in all_checkpoints[:20]:
        summary = cp.get("summary") or {}
        manifest_path = cp.get("manifest_path") or ""
        manifest_link = f" [{cp.get('release_kind')} manifest]({manifest_path})" if manifest_path else ""
        cp_slug = str(cp.get("model_name") or model_name or OPUS_MODEL).replace("/", "__")
        lb_link = f" [leaderboard](curriculum/generated/leaderboard_{cp_slug}.json)"
        quality = float(summary.get("mean_quality_score") or 0.0)
        entry = (
            f"- `{cp.get('release_date')}` `{cp.get('release_kind')}` "
            f"Module {summary.get('module_number', '?')} "
            f"quality=`{quality:.3f}`"
        )
        if cp.get("release_kind") == "quiz":
            passed = "pass" if summary.get("quiz_passed") else "fail"
            entry += f" result=`{passed}`"
        entry += manifest_link + lb_link
        lines.append(entry)
    lines.extend(
        [
            "",
            "## How to Inspect Any Past Milestone",
            "",
            "1. Open the linked checkpoint manifest under `curriculum/checkpoints/`.",
            "2. Check out the recorded git tag or commit.",
            "3. Inspect the manifest for harness metadata plus resolved prompt and few-shot asset hashes.",
            "4. Compare the linked skill directory to the current trainee lane if you want to see prompt or few-shot drift.",
            "",
            "---",
            "",
            "## Developer",
            "",
            "### Harness Workflow Diagram",
            "",
            "The default mechanistic harness orchestrates pre-loop analysis, an iterative mechanism-step proposal loop, and post-step validation. "
            "The diagram below matches the flow shown in the frontend app's Progress panel:",
            "",
            "![Harness flow diagram](docs/harness_flow_snapshot.png)",
            "",
            "- **Pre-loop** (runs once): Check Atom Balance -> Identify Functional Groups -> Recommend pH -> Assess Reaction Conditions -> "
            "Predict Missing Reagents -> Map Atoms -> Map To Reaction Type",
            "- **Loop**: Propose Next Mechanism Step (LLM) -> Validate Mechanism Step -> Bond/Electron, Atom Balance, State Progress validators "
            "-> Retry or Continue? -> Target Products Reached? (yes -> Run Complete; no -> loop back)",
            "- **Decision gates**: Retry/Backtrack routing when validation fails; Paused when no branch points remain",
            "",
            "Regenerate the snapshot with `python scripts/capture_harness_mermaid.py`.",
            "",
            "### Quick Start",
            "",
            "- Start the app: `python main.py serve`",
            f"- Submit today’s trainee run: `python main.py curriculum submit --model-name {OPUS_MODEL}`",
            "- Publish queued releases: `python main.py curriculum publish-due`",
            "- Refresh the curriculum dashboard: `python main.py curriculum render-readme`",
            "",
            "### Contribution Methods",
            "",
            "- Submit an individual reaction locally through the UI or API and use it as evidence for later tracked changes.",
            "- Add or revise few-shot examples for a trainee lane under `skills/mechanistic/<call_name>/models/<model-slug>/few_shot.jsonl`.",
            "- Update prompt instructions in `SKILL.md` for a shared skill or trainee-specific override.",
            "- Propose harness changes under `harness_versions/` and tie them to eval results.",
            "- Add another trainee lane by introducing exact-model overrides and documenting its evidence path.",
            "",
            "### Docs",
            "",
            f"- Operations: [docs/curriculum_operations.md]({links.get('operations')})",
            f"- Prompt/few-shot overrides: [docs/model_asset_overrides.md]({prompt_links.get('override_guide')})",
            f"- History and reproducibility: [docs/history_and_reproducibility.md]({links.get('history')})",
            "",
        ]
    )
    content = "\n".join(lines)
    (base_dir / "README.md").write_text(content.rstrip() + "\n", encoding="utf-8")
    return content


def _git_run(base_dir: Path, args: List[str]) -> tuple[int, str, str]:
    proc = subprocess.run(args, cwd=base_dir, capture_output=True, text=True)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _git_metadata_for_release(base_dir: Path, *, release_date: str, release_kind: str, module_number: int) -> Dict[str, Any]:
    branch = f"codex/curriculum/opus/{release_date}-{release_kind}-m{module_number:02d}"
    tag = f"curriculum/opus/{release_date}-{release_kind}-m{module_number:02d}"
    commit_message = f"curriculum(opus): publish {release_date} {release_kind} module {module_number:02d}"
    metadata = {
        "branch": branch,
        "tag": tag,
        "commit_message": commit_message,
        "commit_sha": None,
        "pr": {},
        "git_available": False,
    }
    code, _, _ = _git_run(base_dir, ["git", "rev-parse", "--is-inside-work-tree"])
    if code != 0:
        return metadata
    metadata["git_available"] = True
    _git_run(base_dir, ["git", "checkout", "-B", branch])
    return metadata


def _publish_git_artifacts(base_dir: Path, *, files: Sequence[str], release_date: str, release_kind: str, module_number: int) -> Dict[str, Any]:
    metadata = _git_metadata_for_release(
        base_dir,
        release_date=release_date,
        release_kind=release_kind,
        module_number=module_number,
    )
    if not metadata.get("git_available"):
        return metadata
    relative_files = [str(path) for path in files if str(path).strip()]
    if relative_files:
        _git_run(base_dir, ["git", "add", *relative_files])
    _git_run(base_dir, ["git", "commit", "-m", str(metadata["commit_message"])])
    _, commit_sha, _ = _git_run(base_dir, ["git", "rev-parse", "HEAD"])
    metadata["commit_sha"] = commit_sha or None
    _git_run(base_dir, ["git", "tag", "-f", str(metadata["tag"])])
    return metadata


def build_curriculum_status(base_dir: Path, store: RunStore, *, model_name: str = OPUS_MODEL, now: datetime | None = None) -> Dict[str, Any]:
    config = load_course_config(base_dir)
    current = _now_in_course_tz(config, now)
    slot = current_release_slot(config, current)
    next_slot = _next_release_slot(config, current)
    module = _current_module(config, store, model_name=model_name)
    queue_rows = store.list_curriculum_releases(model_name=model_name, limit=50)
    checkpoints = store.list_curriculum_checkpoints(model_name=model_name, limit=25)
    today_queue = None
    if slot is not None:
        for row in queue_rows:
            if str(row.get("release_date") or "") == slot.release_date and str(row.get("release_kind") or "") == slot.release_kind:
                today_queue = row
                break
    return {
        "course": config,
        "timezone_now": current.isoformat(),
        "model_name": model_name,
        "trainee_name": "Claude Opus",
        "current_module": module,
        "today_slot": None if slot is None else {
            "release_date": slot.release_date,
            "release_kind": slot.release_kind,
            "label": slot.label,
            "scheduled_publish_at": slot.scheduled_publish_at,
            "scheduled_publish_at_iso": slot.scheduled_publish_at_iso,
        },
        "next_slot": None if next_slot is None else {
            "release_date": next_slot.release_date,
            "release_kind": next_slot.release_kind,
            "label": next_slot.label,
            "scheduled_publish_at": next_slot.scheduled_publish_at,
            "scheduled_publish_at_iso": next_slot.scheduled_publish_at_iso,
            "countdown": _format_countdown(datetime.fromisoformat(next_slot.scheduled_publish_at_iso), current),
        },
        "queued_release": today_queue,
        "latest_leaderboard_row": _resolve_leaderboard_row(store, config, model_name=model_name),
        "weekly_checklist": build_readme_context(base_dir, store, model_name=model_name, now=current).get("calendar") or [],
        "history": checkpoints,
    }


def submit_curriculum_release(base_dir: Path, store: RunStore, *, model_name: str = OPUS_MODEL, now: datetime | None = None) -> Dict[str, Any]:
    config = load_course_config(base_dir)
    slot = current_release_slot(config, now)
    if slot is None:
        raise ValueError("Curriculum runs are only scheduled Monday through Friday")
    existing = next(
        (
            row
            for row in store.list_curriculum_releases(model_name=model_name, limit=100)
            if str(row.get("release_date") or "") == slot.release_date and str(row.get("release_kind") or "") == slot.release_kind
        ),
        None,
    )
    if existing is not None:
        return existing

    module = _current_module(config, store, model_name=model_name)
    selected_entries = _selected_entries_for_slot(
        store,
        config,
        model_name=model_name,
        module=module,
        release_kind=slot.release_kind,
    )
    if not selected_entries:
        raise ValueError(f"No remaining curriculum cases for {module.get('label')} {slot.release_kind}")

    evolve = _load_evolve_module(base_dir)
    registry = RegistrySet(base_dir)
    source_store = store
    imported_id = evolve.ensure_default_flower_eval_set(source_store, base_dir=base_dir)
    eval_set_id = imported_id or _resolve_eval_set_id(store, config)
    if not eval_set_id:
        raise ValueError("Default FlowER eval set is not available")
    evolve_config = evolve.EvolutionConfig(
        model_name=model_name,
        harness=str(config.get("default_harness") or "default"),
        eval_set_id=eval_set_id,
        group_size=int((config.get("default_group_sizes") or {}).get(slot.release_kind, 4)),
        step_pass_target=int((config.get("quiz_unlock") or {}).get("required_pass_count", 4)),
        curriculum_index_path=base_dir / str(config.get("curriculum_index_path") or DEFAULT_INDEX_PATH),
    )
    prepared_batch = evolve.prepare_curriculum_batch(
        config=evolve_config,
        candidate_entries=selected_entries,
        current_step_count=int(module.get("min_step_count") or 1),
    )
    eval_run_id, case_results = evolve.run_curriculum_batch(
        evolve_config,
        prepared_batch=prepared_batch,
        store=store,
        base_dir=base_dir,
    )

    existing_hashes: Dict[str, set[str]] = {}
    best_scores_by_call: Dict[str, float] = {}
    for call_name in set(evolve.MINEABLE_SUBAGENTS.values()):
        examples = load_call_few_shot_examples(call_name, base_dir, model_name=model_name)
        existing_hashes[call_name] = {
            hashlib.sha256(str(ex.get("output") or "").encode()).hexdigest()[:16]
            for ex in examples
        }
        best_scores_by_call[call_name] = _best_few_shot_score_for_model(call_name, base_dir, model_name=model_name)
    mined = evolve.mine_few_shots(case_results, evolve_config, existing_hashes, best_scores_by_call)
    case_scores = {str(result.get("case_id") or ""): float(result.get("score") or 0.0) for result in case_results}
    examples_mined = evolve.apply_mined_examples(
        mined,
        store,
        base_dir,
        base_dir / GENERATED_DIR,
        False,
        "curriculum",
        case_scores,
        model_name=model_name,
    ) if mined else {}

    leaderboard_row = _resolve_leaderboard_row(store, config, model_name=model_name)
    prompt_assets = registry.prompt_step_map(model_name=model_name)
    harness_sha = registry.harness.load(str(config.get("default_harness") or "default")).version
    payload = _queue_payload_from_results(
        config=config,
        slot=slot,
        module=module,
        eval_run_id=eval_run_id,
        case_results=case_results,
        examples_mined=examples_mined,
        selected_entries=selected_entries,
        leaderboard_row=leaderboard_row,
        prompt_assets=prompt_assets,
        harness_sha=harness_sha,
    )
    queue_id = store.queue_curriculum_release(
        release_date=slot.release_date,
        model_name=model_name,
        module_id=str(module.get("id") or ""),
        release_kind=slot.release_kind,
        scheduled_publish_at=slot.scheduled_publish_at,
        eval_run_id=eval_run_id,
        payload=payload,
    )
    return store.get_curriculum_release(queue_id) or {"id": queue_id, "payload": payload}


def publish_curriculum_release(
    base_dir: Path,
    store: RunStore,
    *,
    queue_id: str,
    force: bool = False,
    now: datetime | None = None,
) -> Dict[str, Any]:
    queue_row = store.get_curriculum_release(queue_id)
    if queue_row is None:
        raise ValueError(f"Queued curriculum release not found: {queue_id}")
    if str(queue_row.get("status") or "") == "published" and not force:
        checkpoint_id = str(queue_row.get("published_checkpoint_id") or "")
        existing = store.get_curriculum_checkpoint(checkpoint_id) if checkpoint_id else None
        return existing or queue_row

    config = load_course_config(base_dir)
    current = _now_in_course_tz(config, now)
    scheduled_publish_at = float(queue_row.get("scheduled_publish_at") or 0.0)
    if not force and scheduled_publish_at > current.timestamp():
        raise ValueError("Release is queued but not yet due")

    payload = dict(queue_row.get("payload") or {})
    release_date = str(queue_row.get("release_date") or payload.get("release_date") or "")
    release_kind = str(queue_row.get("release_kind") or payload.get("release_kind") or "lesson")
    module = dict(payload.get("module") or {})
    module_number = int(module.get("number") or 1)

    context = build_readme_context(base_dir, store, model_name=str(queue_row.get("model_name") or OPUS_MODEL), now=current)
    render_curriculum_readme(base_dir, store, model_name=str(queue_row.get("model_name") or OPUS_MODEL), now=current)

    checkpoint_path = base_dir / CHECKPOINTS_DIR / release_date[:4] / f"{release_date}.json"
    manifest = {
        "release_date": release_date,
        "release_kind": release_kind,
        "published_at": current.isoformat(),
        "scheduled_publish_at": payload.get("scheduled_publish_at_iso"),
        "model_name": queue_row.get("model_name"),
        "eval_run_id": queue_row.get("eval_run_id"),
        "module": module,
        "leaderboard_snapshot": payload.get("leaderboard_row") or {},
        "prompt_assets": payload.get("prompt_assets") or {},
        "harness_sha": payload.get("harness_sha"),
        "selected_case_ids": payload.get("selected_case_ids") or [],
        "summary": {
            "module_number": module_number,
            "mean_quality_score": payload.get("mean_quality_score"),
            "pass_count": payload.get("pass_count"),
            "case_count": payload.get("case_count"),
            "quiz_passed": payload.get("quiz_passed"),
            "examples_mined": payload.get("examples_mined") or {},
            "selected_case_ids": payload.get("selected_case_ids") or [],
        },
        "history_preview": context.get("checkpoints") or [],
        "changed_files": [],
        "curriculum_png_index": (
            str((base_dir / "training_data" / "flower_curriculum_pngs" / "index.json").relative_to(base_dir))
            if (base_dir / "training_data" / "flower_curriculum_pngs" / "index.json").exists()
            else None
        ),
        "git": {},
    }
    _write_json(checkpoint_path, manifest)

    _model_slug_pub = str(queue_row.get("model_name") or OPUS_MODEL).replace("/", "__")
    git_files = [
        str(CHECKPOINTS_DIR / release_date[:4] / f"{release_date}.json"),
        str(GENERATED_DIR / f"leaderboard_{_model_slug_pub}.json"),
        str(README_CONTEXT_PATH),
        "README.md",
    ]
    model_slug = str(queue_row.get("model_name") or OPUS_MODEL).replace("/", "__")
    skills_root = base_dir / "skills" / "mechanistic"
    for path in sorted(skills_root.glob(f"*/models/{model_slug}/*")):
        git_files.append(str(path.relative_to(base_dir)))
    manifest["changed_files"] = list(git_files)
    git_metadata = _publish_git_artifacts(
        base_dir,
        files=git_files,
        release_date=release_date,
        release_kind=release_kind,
        module_number=module_number,
    )
    manifest["git"] = git_metadata
    _write_json(checkpoint_path, manifest)

    checkpoint_id = store.record_curriculum_checkpoint(
        release_date=release_date,
        model_name=str(queue_row.get("model_name") or OPUS_MODEL),
        module_id=str(queue_row.get("module_id") or module.get("id") or ""),
        release_kind=release_kind,
        manifest_path=str(checkpoint_path.relative_to(base_dir)),
        commit_sha=git_metadata.get("commit_sha"),
        git_tag=git_metadata.get("tag"),
        git_branch=git_metadata.get("branch"),
        pr_metadata=git_metadata.get("pr") if isinstance(git_metadata.get("pr"), dict) else {},
        summary=manifest["summary"],
    )
    store.update_curriculum_release(queue_id, status="published", payload=payload, published_checkpoint_id=checkpoint_id)
    return store.get_curriculum_checkpoint(checkpoint_id) or {"id": checkpoint_id, "summary": manifest["summary"]}


def publish_due_curriculum_releases(base_dir: Path, store: RunStore, *, now: datetime | None = None) -> List[Dict[str, Any]]:
    config = load_course_config(base_dir)
    current = _now_in_course_tz(config, now)
    published: List[Dict[str, Any]] = []
    for release in store.list_curriculum_releases(status="queued", limit=100):
        if float(release.get("scheduled_publish_at") or 0.0) > current.timestamp():
            continue
        published.append(publish_curriculum_release(base_dir, store, queue_id=str(release.get("id") or ""), now=current))
    return published


def curriculum_history(store: RunStore, *, model_name: str = OPUS_MODEL, limit: int = 100) -> List[Dict[str, Any]]:
    return store.list_curriculum_checkpoints(model_name=model_name, limit=limit)


def render_launchd_plist(base_dir: Path) -> str:
    program = str((base_dir / "main.py").resolve())
    cwd = str(base_dir.resolve())
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.mechanistic.curriculum.publish</string>
  <key>ProgramArguments</key>
  <array>
    <string>python</string>
    <string>{program}</string>
    <string>curriculum</string>
    <string>publish-due</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Weekday</key>
    <integer>1</integer>
    <key>Hour</key>
    <integer>17</integer>
    <key>Minute</key>
    <integer>0</integer>
  </dict>
  <key>WorkingDirectory</key>
  <string>{cwd}</string>
</dict>
</plist>
"""
