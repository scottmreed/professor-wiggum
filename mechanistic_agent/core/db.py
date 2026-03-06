"""SQLite persistence for the local-first mechanistic runtime."""
from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from mechanistic_agent.model_registry import get_model_family, get_model_provider, resolve_model_key
from mechanistic_agent.prompt_assets import resolve_call_name_from_step, traces_root

SCHEMA_VERSION = "2026_03_single_model_selection_v1"


class RunStore:
    """Repository wrapper over a local SQLite database."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _json_dumps(payload: Any) -> str:
        return json.dumps(payload, sort_keys=True)

    @staticmethod
    def _json_loads(raw: Optional[str], default: Any) -> Any:
        if not raw:
            return default
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return default

    @staticmethod
    def _model_pricing_sha256() -> str:
        pricing_path = Path(__file__).resolve().parent.parent / "model_pricing.json"
        if not pricing_path.exists():
            return ""
        digest = hashlib.sha256()
        digest.update(pricing_path.read_bytes())
        return digest.hexdigest()

    def init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS db_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    status TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    input_payload_json TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    prompt_bundle_hash TEXT NOT NULL,
                    skill_bundle_hash TEXT NOT NULL,
                    memory_bundle_hash TEXT NOT NULL,
                    harness_bundle_hash TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS run_events (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    ts REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    step_name TEXT,
                    payload_json TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_run_events_run_seq ON run_events(run_id, seq);

                CREATE TABLE IF NOT EXISTS step_outputs (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    step_name TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    retry_index INTEGER NOT NULL DEFAULT 0,
                    source TEXT NOT NULL DEFAULT 'llm',
                    model TEXT,
                    reasoning_level TEXT,
                    tool_name TEXT NOT NULL,
                    output_json TEXT NOT NULL,
                    validation_json TEXT,
                    accepted_bool INTEGER,
                    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_step_outputs_run_step_attempt ON step_outputs(run_id, step_name, attempt);

                CREATE TABLE IF NOT EXISTS arrow_push_annotations (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    attempt INTEGER NOT NULL,
                    retry_index INTEGER NOT NULL DEFAULT 0,
                    candidate_rank INTEGER,
                    source TEXT NOT NULL DEFAULT 'mechanism_loop',
                    prediction_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_arrow_push_annotations_run_step
                    ON arrow_push_annotations(run_id, step_index, attempt, retry_index);

                CREATE TABLE IF NOT EXISTS verification_decisions (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    step_name TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    decided_at REAL NOT NULL,
                    rationale TEXT,
                    decided_by TEXT,
                    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    step_name TEXT,
                    rating INTEGER,
                    label TEXT,
                    comment TEXT,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS memory_items (
                    id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL,
                    tags_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    active_bool INTEGER NOT NULL DEFAULT 1
                );
                CREATE INDEX IF NOT EXISTS idx_memory_scope_key ON memory_items(scope, key);

                CREATE TABLE IF NOT EXISTS assets (
                    id TEXT PRIMARY KEY,
                    asset_type TEXT NOT NULL,
                    path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    loaded_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_assets_type_path ON assets(asset_type, path);

                CREATE TABLE IF NOT EXISTS evaluations (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    judge_model TEXT NOT NULL,
                    score REAL,
                    summary_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS prompt_versions (
                    id TEXT PRIMARY KEY,
                    prompt_name TEXT NOT NULL,
                    call_name TEXT,
                    step_name TEXT,
                    version TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    path TEXT NOT NULL,
                    template_text TEXT,
                    shared_base_sha256 TEXT,
                    call_base_sha256 TEXT,
                    few_shot_sha256 TEXT,
                    prompt_bundle_sha256 TEXT,
                    created_at REAL NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_prompt_versions_sha
                    ON prompt_versions(sha256);

                CREATE TABLE IF NOT EXISTS run_step_prompts (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    step_name TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    prompt_version_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE,
                    FOREIGN KEY(prompt_version_id) REFERENCES prompt_versions(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_run_step_prompts_run_step
                    ON run_step_prompts(run_id, step_name, attempt);

                CREATE TABLE IF NOT EXISTS model_versions (
                    id TEXT PRIMARY KEY,
                    resolved_model_key TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    family TEXT NOT NULL,
                    pricing_sha256 TEXT NOT NULL,
                    reasoning_level TEXT,
                    created_at REAL NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_model_versions_identity
                    ON model_versions(
                        resolved_model_key,
                        provider,
                        family,
                        pricing_sha256,
                        COALESCE(reasoning_level, '')
                    );

                CREATE TABLE IF NOT EXISTS trace_records (
                    id TEXT PRIMARY KEY,
                    run_id TEXT,
                    step_name TEXT NOT NULL,
                    model TEXT,
                    reasoning_level TEXT,
                    prompt_version_id TEXT,
                    model_version_id TEXT,
                    score REAL,
                    source TEXT NOT NULL,
                    approved_bool INTEGER NOT NULL DEFAULT 0,
                    approval_label TEXT,
                    approval_notes TEXT,
                    approved_by TEXT,
                    approved_at REAL,
                    actor_id TEXT,
                    trace_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE SET NULL,
                    FOREIGN KEY(prompt_version_id) REFERENCES prompt_versions(id) ON DELETE SET NULL,
                    FOREIGN KEY(model_version_id) REFERENCES model_versions(id) ON DELETE SET NULL
                );
                CREATE INDEX IF NOT EXISTS idx_trace_records_step_score
                    ON trace_records(step_name, score DESC, created_at DESC);

                CREATE TABLE IF NOT EXISTS eval_sets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    source_path TEXT,
                    sha256 TEXT,
                    active_bool INTEGER NOT NULL DEFAULT 1,
                    purpose TEXT NOT NULL DEFAULT 'general',
                    exposed_in_ui_bool INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_eval_sets_name_version
                    ON eval_sets(name, version);

                CREATE TABLE IF NOT EXISTS eval_set_cases (
                    id TEXT PRIMARY KEY,
                    eval_set_id TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    expected_json TEXT,
                    tags_json TEXT NOT NULL,
                    FOREIGN KEY(eval_set_id) REFERENCES eval_sets(id) ON DELETE CASCADE
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_eval_set_cases_unique
                    ON eval_set_cases(eval_set_id, case_id);

                CREATE TABLE IF NOT EXISTS eval_runs (
                    id TEXT PRIMARY KEY,
                    eval_set_id TEXT NOT NULL,
                    run_group_name TEXT,
                    model TEXT,
                    model_name TEXT,
                    model_family TEXT,
                    thinking_level TEXT,
                    harness_bundle_hash TEXT,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(eval_set_id) REFERENCES eval_sets(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS eval_run_results (
                    id TEXT PRIMARY KEY,
                    eval_run_id TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    run_id TEXT,
                    score REAL,
                    pass_bool INTEGER,
                    cost_json TEXT,
                    latency_ms REAL,
                    summary_json TEXT NOT NULL,
                    FOREIGN KEY(eval_run_id) REFERENCES eval_runs(id) ON DELETE CASCADE,
                    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE SET NULL
                );
                CREATE INDEX IF NOT EXISTS idx_eval_run_results_eval_run
                    ON eval_run_results(eval_run_id, score DESC);

                CREATE TABLE IF NOT EXISTS few_shot_examples (
                    id TEXT PRIMARY KEY,
                    step_name TEXT NOT NULL,
                    example_key TEXT NOT NULL,
                    input_text TEXT NOT NULL,
                    output_text TEXT NOT NULL,
                    source_trace_id TEXT,
                    score REAL,
                    approved_bool INTEGER NOT NULL DEFAULT 0,
                    prompt_version_id TEXT,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(source_trace_id) REFERENCES trace_records(id) ON DELETE SET NULL,
                    FOREIGN KEY(prompt_version_id) REFERENCES prompt_versions(id) ON DELETE SET NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_few_shot_example_key
                    ON few_shot_examples(step_name, example_key);

                CREATE TABLE IF NOT EXISTS run_pauses (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    paused_at REAL NOT NULL,
                    decided_at REAL,
                    decision TEXT,
                    decided_by TEXT,
                    rationale TEXT,
                    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_run_pauses_run
                    ON run_pauses(run_id, paused_at DESC);

                CREATE TABLE IF NOT EXISTS ralph_attempts (
                    id TEXT PRIMARY KEY,
                    parent_run_id TEXT NOT NULL,
                    attempt_index INTEGER NOT NULL,
                    child_run_id TEXT,
                    harness_name TEXT NOT NULL,
                    parent_harness_sha TEXT NOT NULL,
                    harness_sha TEXT NOT NULL,
                    mutation_actions_json TEXT NOT NULL,
                    diff_summary_json TEXT NOT NULL,
                    stop_reason TEXT,
                    completion_promise_met_bool INTEGER,
                    cost_usd REAL,
                    created_at REAL NOT NULL,
                    completed_at REAL,
                    FOREIGN KEY(parent_run_id) REFERENCES runs(id) ON DELETE CASCADE,
                    FOREIGN KEY(child_run_id) REFERENCES runs(id) ON DELETE SET NULL
                );
                CREATE INDEX IF NOT EXISTS idx_ralph_attempts_parent_attempt
                    ON ralph_attempts(parent_run_id, attempt_index ASC);

                CREATE TABLE IF NOT EXISTS ralph_votes (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    attempt_index INTEGER NOT NULL,
                    step_index INTEGER NOT NULL,
                    candidate_a_json TEXT NOT NULL,
                    candidate_b_json TEXT NOT NULL,
                    vote TEXT NOT NULL,
                    confidence REAL,
                    source TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_ralph_votes_run_attempt
                    ON ralph_votes(run_id, attempt_index, step_index);

                CREATE TABLE IF NOT EXISTS curation_exports (
                    id TEXT PRIMARY KEY,
                    export_type TEXT NOT NULL,
                    path TEXT NOT NULL,
                    manifest_json TEXT NOT NULL,
                    created_by TEXT,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_curation_exports_type_created
                    ON curation_exports(export_type, created_at DESC);

                CREATE TABLE IF NOT EXISTS verification_results (
                    id TEXT PRIMARY KEY,
                    harness_version TEXT NOT NULL,
                    model_family TEXT NOT NULL,
                    step_name TEXT NOT NULL,
                    verified_model TEXT NOT NULL,
                    verified_reasoning TEXT,
                    baseline_score REAL NOT NULL,
                    step_score REAL NOT NULL,
                    eval_set_id TEXT,
                    eval_run_id TEXT,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(eval_set_id) REFERENCES eval_sets(id) ON DELETE SET NULL,
                    FOREIGN KEY(eval_run_id) REFERENCES eval_runs(id) ON DELETE SET NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_verification_results_version_family_step
                    ON verification_results(harness_version, model_family, step_name);

                CREATE TABLE IF NOT EXISTS verification_jobs (
                    id TEXT PRIMARY KEY,
                    model_family TEXT NOT NULL,
                    eval_set_id TEXT NOT NULL,
                    harness_version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress_json TEXT NOT NULL DEFAULT '{}',
                    result_json TEXT,
                    created_at REAL NOT NULL,
                    completed_at REAL
                );

                CREATE TABLE IF NOT EXISTS curriculum_release_queue (
                    id TEXT PRIMARY KEY,
                    release_date TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    module_id TEXT NOT NULL,
                    release_kind TEXT NOT NULL,
                    scheduled_publish_at REAL NOT NULL,
                    eval_run_id TEXT,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    published_checkpoint_id TEXT,
                    FOREIGN KEY(eval_run_id) REFERENCES eval_runs(id) ON DELETE SET NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_curriculum_release_queue_unique_slot
                    ON curriculum_release_queue(release_date, model_name, release_kind);

                CREATE TABLE IF NOT EXISTS curriculum_checkpoints (
                    id TEXT PRIMARY KEY,
                    release_date TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    module_id TEXT NOT NULL,
                    release_kind TEXT NOT NULL,
                    manifest_path TEXT NOT NULL,
                    commit_sha TEXT,
                    git_tag TEXT,
                    git_branch TEXT,
                    pr_json TEXT,
                    summary_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_curriculum_checkpoints_model_date
                    ON curriculum_checkpoints(model_name, release_date DESC);
                """
            )
            self._ensure_column(conn, "step_outputs", "retry_index", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "step_outputs", "source", "TEXT NOT NULL DEFAULT 'llm'")
            self._ensure_column(conn, "trace_records", "approved_bool", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "trace_records", "approval_label", "TEXT")
            self._ensure_column(conn, "trace_records", "approval_notes", "TEXT")
            self._ensure_column(conn, "trace_records", "approved_by", "TEXT")
            self._ensure_column(conn, "trace_records", "approved_at", "REAL")
            self._ensure_column(conn, "trace_records", "model_version_id", "TEXT")
            self._ensure_column(conn, "prompt_versions", "call_name", "TEXT")
            self._ensure_column(conn, "prompt_versions", "shared_base_sha256", "TEXT")
            self._ensure_column(conn, "prompt_versions", "call_base_sha256", "TEXT")
            self._ensure_column(conn, "prompt_versions", "few_shot_sha256", "TEXT")
            self._ensure_column(conn, "prompt_versions", "prompt_bundle_sha256", "TEXT")
            self._ensure_column(conn, "prompt_versions", "model_name", "TEXT")
            self._ensure_column(conn, "prompt_versions", "resolved_shared_base_path", "TEXT")
            self._ensure_column(conn, "prompt_versions", "resolved_call_base_path", "TEXT")
            self._ensure_column(conn, "prompt_versions", "resolved_few_shot_path", "TEXT")
            self._ensure_column(conn, "prompt_versions", "asset_scope", "TEXT")
            self._ensure_column(conn, "step_outputs", "usage_json", "TEXT")
            self._ensure_column(conn, "step_outputs", "cost_json", "TEXT")
            self._ensure_column(conn, "eval_runs", "model_name", "TEXT")
            self._ensure_column(conn, "eval_runs", "model_family", "TEXT")
            self._ensure_column(conn, "eval_runs", "thinking_level", "TEXT")
            self._ensure_column(conn, "eval_sets", "purpose", "TEXT NOT NULL DEFAULT 'general'")
            self._ensure_column(conn, "eval_sets", "exposed_in_ui_bool", "INTEGER NOT NULL DEFAULT 1")
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_trace_records_source_approved
                ON trace_records(source, approved_bool, score DESC, created_at DESC)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_versions (
                    id TEXT PRIMARY KEY,
                    resolved_model_key TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    family TEXT NOT NULL,
                    pricing_sha256 TEXT NOT NULL,
                    reasoning_level TEXT,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_model_versions_identity
                ON model_versions(
                    resolved_model_key,
                    provider,
                    family,
                    pricing_sha256,
                    COALESCE(reasoning_level, '')
                )
                """
            )
            conn.execute(
                "INSERT OR IGNORE INTO db_migrations(version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, time.time()),
            )
            conn.commit()

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        existing = {str(row[1]) for row in rows}
        if column in existing:
            return
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def create_run(
        self,
        *,
        mode: str,
        input_payload: Dict[str, Any],
        config: Dict[str, Any],
        prompt_bundle_hash: str,
        skill_bundle_hash: str,
        memory_bundle_hash: str,
        harness_bundle_hash: str = "",
    ) -> str:
        run_id = uuid.uuid4().hex
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs(
                    id, created_at, status, mode,
                    input_payload_json, config_json,
                    prompt_bundle_hash, skill_bundle_hash, memory_bundle_hash, harness_bundle_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    now,
                    "pending",
                    mode,
                    self._json_dumps(input_payload),
                    self._json_dumps(config),
                    prompt_bundle_hash,
                    skill_bundle_hash,
                    memory_bundle_hash,
                    harness_bundle_hash,
                ),
            )
            conn.commit()
        return run_id

    def get_run_row(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                return None
            payload = dict(row)
            payload["input_payload"] = self._json_loads(payload.pop("input_payload_json", None), {})
            payload["config"] = self._json_loads(payload.pop("config_json", None), {})
            return payload

    def set_run_status(self, run_id: str, status: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("UPDATE runs SET status = ? WHERE id = ?", (status, run_id))
            conn.commit()

    def delete_run(self, run_id: str) -> bool:
        """Delete a run and all cascaded data (events, step_outputs, run_step_prompts, etc.).

        Also deletes trace_records that reference this run_id (which use ON DELETE SET NULL
        and would otherwise be orphaned). Returns True if the run existed and was deleted.
        """
        import shutil

        with self._lock, self._connect() as conn:
            existing = conn.execute("SELECT id FROM runs WHERE id = ?", (run_id,)).fetchone()
            if existing is None:
                return False
            child_rows = conn.execute(
                "SELECT child_run_id FROM ralph_attempts WHERE parent_run_id = ?",
                (run_id,),
            ).fetchall()
            child_run_ids = [
                str(row[0])
                for row in child_rows
                if row and row[0]
            ]
            # Delete trace_records explicitly (FK is SET NULL, not CASCADE)
            conn.execute("DELETE FROM trace_records WHERE run_id = ?", (run_id,))
            for child_run_id in child_run_ids:
                conn.execute("DELETE FROM trace_records WHERE run_id = ?", (child_run_id,))
                conn.execute("DELETE FROM runs WHERE id = ?", (child_run_id,))
            # Delete the run — CASCADE removes events, step_outputs, run_step_prompts, etc.
            conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
            conn.commit()

        # Remove filesystem trace directory
        base_root = self.db_path.parent.parent if self.db_path.parent.name == "data" else self.db_path.parent
        run_dir = traces_root(base_root) / "runs" / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)

        return True

    def update_run_config(self, run_id: str, config: Dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE runs SET config_json = ? WHERE id = ?",
                (self._json_dumps(config), run_id),
            )
            conn.commit()

    def append_event(
        self,
        run_id: str,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        step_name: Optional[str] = None,
    ) -> int:
        payload = payload or {}
        event_id = uuid.uuid4().hex
        now = time.time()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) AS max_seq FROM run_events WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            max_seq = int(row["max_seq"]) if row else 0
            seq = max_seq + 1
            conn.execute(
                """
                INSERT INTO run_events(id, run_id, ts, event_type, step_name, payload_json, seq)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    run_id,
                    now,
                    event_type,
                    step_name,
                    self._json_dumps(payload),
                    seq,
                ),
            )
            conn.commit()
        return seq

    def list_events(self, run_id: str, *, after_seq: int = 0, limit: int = 500) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM run_events
                WHERE run_id = ? AND seq > ?
                ORDER BY seq ASC
                LIMIT ?
                """,
                (run_id, after_seq, limit),
            ).fetchall()
        output: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["payload"] = self._json_loads(item.pop("payload_json", None), {})
            output.append(item)
        return output

    def record_step_output(
        self,
        *,
        run_id: str,
        step_name: str,
        attempt: int,
        retry_index: int = 0,
        source: str = "llm",
        model: Optional[str],
        reasoning_level: Optional[str],
        tool_name: str,
        output: Dict[str, Any],
        validation: Optional[Dict[str, Any]],
        accepted_bool: Optional[bool] = None,
        usage: Optional[Dict[str, int]] = None,
        cost: Optional[Dict[str, float]] = None,
    ) -> str:
        row_id = uuid.uuid4().hex
        accepted_val = None if accepted_bool is None else int(bool(accepted_bool))
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO step_outputs(
                    id, run_id, step_name, attempt, retry_index, source, model, reasoning_level,
                    tool_name, output_json, validation_json, accepted_bool, usage_json, cost_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row_id,
                    run_id,
                    step_name,
                    attempt,
                    retry_index,
                    source,
                    model,
                    reasoning_level,
                    tool_name,
                    self._json_dumps(output),
                    self._json_dumps(validation) if validation is not None else None,
                    accepted_val,
                    self._json_dumps(usage) if usage is not None else None,
                    self._json_dumps(cost) if cost is not None else None,
                ),
            )
            conn.commit()
        return row_id

    def list_step_outputs(self, run_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM step_outputs
                WHERE run_id = ?
                ORDER BY step_name ASC, attempt ASC, retry_index ASC
                """,
                (run_id,),
            ).fetchall()
        output: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["output"] = self._json_loads(item.pop("output_json", None), {})
            item["validation"] = self._json_loads(item.pop("validation_json", None), None)
            accepted = item.get("accepted_bool")
            item["accepted_bool"] = None if accepted is None else bool(accepted)
            if "retry_index" in item and item["retry_index"] is not None:
                item["retry_index"] = int(item["retry_index"])
            if "source" in item and item["source"] is not None:
                item["source"] = str(item["source"])
            item["usage"] = self._json_loads(item.pop("usage_json", None), None)
            item["cost"] = self._json_loads(item.pop("cost_json", None), None)
            output.append(item)
        return output

    def record_arrow_push_annotation(
        self,
        *,
        run_id: str,
        step_index: int,
        attempt: int,
        retry_index: int = 0,
        candidate_rank: Optional[int] = None,
        source: str = "mechanism_loop",
        prediction: Dict[str, Any],
    ) -> str:
        row_id = uuid.uuid4().hex
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO arrow_push_annotations(
                    id, run_id, step_index, attempt, retry_index, candidate_rank, source, prediction_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row_id,
                    run_id,
                    int(step_index),
                    int(attempt),
                    int(retry_index),
                    None if candidate_rank is None else int(candidate_rank),
                    str(source or "mechanism_loop"),
                    self._json_dumps(prediction),
                    time.time(),
                ),
            )
            conn.commit()
        return row_id

    def list_arrow_push_annotations(
        self,
        run_id: str,
        *,
        step_index: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        params: List[Any] = [run_id]
        where = "run_id = ?"
        if step_index is not None:
            where += " AND step_index = ?"
            params.append(int(step_index))
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM arrow_push_annotations
                WHERE {where}
                ORDER BY step_index ASC, attempt ASC, retry_index ASC, created_at ASC
                """,
                params,
            ).fetchall()
        output: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["prediction"] = self._json_loads(item.pop("prediction_json", None), {})
            output.append(item)
        return output

    def get_run_cost_summary(self, run_id: str) -> Dict[str, Any]:
        """Aggregate token usage and cost across all step outputs for a run."""
        from mechanistic_agent.model_registry import update_cost_totals, update_usage_totals

        steps = self.list_step_outputs(run_id)
        usage_totals: Dict[str, int] = {}
        cost_totals: Dict[str, float] = {}
        per_step: List[Dict[str, Any]] = []

        for step in steps:
            step_usage = step.get("usage")
            step_cost = step.get("cost")
            if step_usage:
                update_usage_totals(usage_totals, step_usage)
            if step_cost:
                update_cost_totals(cost_totals, step_cost)
            if step_usage or step_cost:
                per_step.append({
                    "step_name": step.get("step_name"),
                    "attempt": step.get("attempt"),
                    "retry_index": step.get("retry_index"),
                    "model": step.get("model"),
                    "usage": step_usage,
                    "cost": step_cost,
                })

        return {
            "run_id": run_id,
            "total_usage": usage_totals or None,
            "total_cost": cost_totals or None,
            "step_costs": per_step,
        }

    def record_verification_decision(
        self,
        *,
        run_id: str,
        step_name: str,
        decision: str,
        rationale: Optional[str],
        decided_by: Optional[str],
    ) -> str:
        row_id = uuid.uuid4().hex
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO verification_decisions(id, run_id, step_name, decision, decided_at, rationale, decided_by)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (row_id, run_id, step_name, decision, time.time(), rationale, decided_by),
            )
            conn.commit()
        return row_id

    def update_step_acceptance(
        self,
        *,
        run_id: str,
        step_name: str,
        attempt: Optional[int],
        accepted: bool,
    ) -> None:
        with self._lock, self._connect() as conn:
            if attempt is None:
                row = conn.execute(
                    """
                    SELECT attempt FROM step_outputs
                    WHERE run_id = ? AND step_name = ?
                    ORDER BY attempt DESC
                    LIMIT 1
                    """,
                    (run_id, step_name),
                ).fetchone()
                if row is None:
                    return
                attempt = int(row["attempt"])
            conn.execute(
                """
                UPDATE step_outputs
                SET accepted_bool = ?
                WHERE run_id = ? AND step_name = ? AND attempt = ?
                """,
                (int(accepted), run_id, step_name, attempt),
            )
            conn.commit()

    def unaccepted_verified_steps(self, run_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM step_outputs
                WHERE run_id = ?
                  AND step_name = 'mechanism_synthesis'
                  AND validation_json IS NOT NULL
                  AND accepted_bool IS NULL
                ORDER BY attempt ASC
                """,
                (run_id,),
            ).fetchall()
        output: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["output"] = self._json_loads(item.pop("output_json", None), {})
            item["validation"] = self._json_loads(item.pop("validation_json", None), None)
            output.append(item)
        return output

    def record_feedback(
        self,
        *,
        run_id: str,
        step_name: Optional[str],
        rating: Optional[int],
        label: Optional[str],
        comment: Optional[str],
        payload: Optional[Dict[str, Any]],
    ) -> str:
        row_id = uuid.uuid4().hex
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feedback(id, run_id, step_name, rating, label, comment, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row_id,
                    run_id,
                    step_name,
                    rating,
                    label,
                    comment,
                    self._json_dumps(payload or {}),
                    time.time(),
                ),
            )
            conn.commit()
        return row_id

    def list_feedback(self, run_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM feedback WHERE run_id = ? ORDER BY created_at ASC",
                (run_id,),
            ).fetchall()
        output: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["payload"] = self._json_loads(item.pop("payload_json", None), {})
            output.append(item)
        return output

    def add_memory_item(
        self,
        *,
        scope: str,
        key: str,
        value: Dict[str, Any],
        source: str,
        confidence: Optional[float],
        tags: Optional[Iterable[str]],
        active: bool = True,
    ) -> str:
        row_id = uuid.uuid4().hex
        now = time.time()
        tags_list = [str(tag) for tag in (tags or []) if str(tag).strip()]
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_items(
                    id, scope, key, value_json, source, confidence,
                    tags_json, created_at, updated_at, active_bool
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row_id,
                    scope,
                    key,
                    self._json_dumps(value),
                    source,
                    confidence,
                    self._json_dumps(tags_list),
                    now,
                    now,
                    int(active),
                ),
            )
            conn.commit()
        return row_id

    def list_memory_items(self, *, scope: Optional[str] = None, active_only: bool = True) -> List[Dict[str, Any]]:
        clauses = ["1=1"]
        params: List[Any] = []
        if scope:
            clauses.append("scope = ?")
            params.append(scope)
        if active_only:
            clauses.append("active_bool = 1")

        query = (
            "SELECT * FROM memory_items WHERE "
            + " AND ".join(clauses)
            + " ORDER BY updated_at DESC"
        )
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        items: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["value"] = self._json_loads(item.pop("value_json", None), {})
            item["tags"] = self._json_loads(item.pop("tags_json", None), [])
            item["active_bool"] = bool(item.get("active_bool", 0))
            items.append(item)
        return items

    def record_assets(self, assets: List[Dict[str, Any]]) -> None:
        if not assets:
            return
        now = time.time()
        with self._lock, self._connect() as conn:
            for asset in assets:
                conn.execute(
                    """
                    INSERT INTO assets(id, asset_type, path, sha256, metadata_json, loaded_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uuid.uuid4().hex,
                        str(asset.get("asset_type", "unknown")),
                        str(asset.get("path", "")),
                        str(asset.get("sha256", "")),
                        self._json_dumps(asset.get("metadata", {})),
                        now,
                    ),
                )
            conn.commit()

    def record_evaluation(
        self,
        *,
        run_id: str,
        judge_model: str,
        score: Optional[float],
        summary: Dict[str, Any],
    ) -> str:
        row_id = uuid.uuid4().hex
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO evaluations(id, run_id, judge_model, score, summary_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (row_id, run_id, judge_model, score, self._json_dumps(summary), time.time()),
            )
            conn.commit()
        return row_id

    def upsert_prompt_versions(self, prompts: List[Dict[str, Any]]) -> Dict[str, str]:
        ids_by_step: Dict[str, str] = {}
        now = time.time()
        with self._lock, self._connect() as conn:
            for prompt in prompts:
                sha = str(
                    prompt.get("prompt_bundle_sha256")
                    or prompt.get("sha256")
                    or ""
                )
                if not sha:
                    continue
                current_step_name = str(prompt.get("step") or "")
                existing = conn.execute(
                    "SELECT id, step_name FROM prompt_versions WHERE sha256 = ? LIMIT 1",
                    (sha,),
                ).fetchone()
                if existing is not None:
                    prompt_id = str(existing["id"])
                    step_name = current_step_name or str(existing["step_name"] or "")
                else:
                    prompt_id = uuid.uuid4().hex
                    step_name = current_step_name
                    conn.execute(
                        """
                        INSERT INTO prompt_versions(
                            id, prompt_name, call_name, step_name, version, sha256, path, template_text,
                            shared_base_sha256, call_base_sha256, few_shot_sha256, prompt_bundle_sha256,
                            model_name, resolved_shared_base_path, resolved_call_base_path,
                            resolved_few_shot_path, asset_scope,
                            created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            prompt_id,
                            str(prompt.get("name") or prompt.get("call_name") or step_name or "prompt"),
                            str(prompt.get("call_name") or ""),
                            step_name,
                            str(prompt.get("version") or "0.0.0"),
                            sha,
                            str(prompt.get("path") or ""),
                            str(prompt.get("template") or ""),
                            str(prompt.get("shared_base_sha256") or ""),
                            str(prompt.get("call_base_sha256") or ""),
                            str(prompt.get("few_shot_sha256") or ""),
                            str(prompt.get("prompt_bundle_sha256") or sha),
                            str(prompt.get("model_name") or ""),
                            str(prompt.get("resolved_shared_base_path") or ""),
                            str(prompt.get("resolved_call_base_path") or ""),
                            str(prompt.get("resolved_few_shot_path") or ""),
                            str(prompt.get("asset_scope") or ""),
                            now,
                        ),
                    )
                if step_name:
                    ids_by_step[step_name] = prompt_id
            conn.commit()
        return ids_by_step

    def upsert_model_version(
        self,
        *,
        model_name: str,
        reasoning_level: Optional[str],
    ) -> Optional[str]:
        model_value = str(model_name or "").strip()
        if not model_value:
            return None
        try:
            resolved_model_key = resolve_model_key(model_value)
        except Exception:
            resolved_model_key = model_value
        provider = get_model_provider(resolved_model_key)
        family = get_model_family(resolved_model_key)
        pricing_sha = self._model_pricing_sha256()
        reasoning = (reasoning_level or "").strip() or None
        identity = "|".join(
            [
                resolved_model_key,
                provider,
                family,
                pricing_sha,
                reasoning or "",
            ]
        )
        model_version_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO model_versions(
                    id, resolved_model_key, provider, family, pricing_sha256, reasoning_level, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_version_id,
                    resolved_model_key,
                    provider,
                    family,
                    pricing_sha,
                    reasoning,
                    time.time(),
                ),
            )
            conn.commit()
        return model_version_id

    def bind_run_step_prompt(
        self,
        *,
        run_id: str,
        step_name: str,
        prompt_version_id: str,
        attempt: int = 0,
    ) -> str:
        row_id = uuid.uuid4().hex
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO run_step_prompts(id, run_id, step_name, attempt, prompt_version_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (row_id, run_id, step_name, attempt, prompt_version_id, time.time()),
            )
            conn.commit()
        return row_id

    def list_run_step_prompts(self, run_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT rsp.*, pv.prompt_name, pv.call_name, pv.version, pv.sha256, pv.path, pv.template_text,
                       pv.shared_base_sha256, pv.call_base_sha256, pv.few_shot_sha256, pv.prompt_bundle_sha256,
                       pv.model_name, pv.resolved_shared_base_path, pv.resolved_call_base_path,
                       pv.resolved_few_shot_path, pv.asset_scope
                FROM run_step_prompts rsp
                JOIN prompt_versions pv ON rsp.prompt_version_id = pv.id
                WHERE rsp.run_id = ?
                ORDER BY rsp.step_name ASC, rsp.attempt ASC
                """,
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_prompt_versions(
        self,
        *,
        step_name: Optional[str] = None,
        call_name: Optional[str] = None,
        sha256: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        clauses = ["1=1"]
        params: List[Any] = []
        if step_name:
            call_from_step = resolve_call_name_from_step(step_name)
            if call_from_step:
                clauses.append("(step_name = ? OR call_name = ?)")
                params.extend([step_name, call_from_step])
            else:
                clauses.append("step_name = ?")
                params.append(step_name)
        if call_name:
            clauses.append("call_name = ?")
            params.append(call_name)
        if sha256:
            clauses.append("sha256 = ?")
            params.append(sha256)
        params.append(limit)
        query = (
            "SELECT * FROM prompt_versions WHERE "
            + " AND ".join(clauses)
            + " ORDER BY created_at DESC LIMIT ?"
        )
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_prompt_version(self, prompt_version_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM prompt_versions WHERE id = ? LIMIT 1",
                (prompt_version_id,),
            ).fetchone()
        if row is None:
            return None
        return dict(row)

    def get_model_version(self, model_version_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM model_versions WHERE id = ? LIMIT 1",
                (model_version_id,),
            ).fetchone()
        if row is None:
            return None
        return dict(row)

    def resolve_run_step_prompt_id(
        self,
        *,
        run_id: str,
        step_name: str,
        attempt: Optional[int] = None,
    ) -> Optional[str]:
        with self._connect() as conn:
            if attempt is None:
                row = conn.execute(
                    """
                    SELECT prompt_version_id
                    FROM run_step_prompts
                    WHERE run_id = ? AND step_name = ?
                    ORDER BY attempt DESC
                    LIMIT 1
                    """,
                    (run_id, step_name),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT prompt_version_id
                    FROM run_step_prompts
                    WHERE run_id = ? AND step_name = ? AND attempt <= ?
                    ORDER BY attempt DESC
                    LIMIT 1
                    """,
                    (run_id, step_name, attempt),
                ).fetchone()
        if row is None:
            return None
        return str(row["prompt_version_id"])

    def _get_prompt_version_row(self, prompt_version_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not prompt_version_id:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM prompt_versions WHERE id = ? LIMIT 1",
                (prompt_version_id,),
            ).fetchone()
        if row is None:
            return None
        return dict(row)

    def _get_model_version_row(self, model_version_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not model_version_id:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM model_versions WHERE id = ? LIMIT 1",
                (model_version_id,),
            ).fetchone()
        if row is None:
            return None
        return dict(row)

    def _persist_trace_file(
        self,
        *,
        trace_id: str,
        run_id: Optional[str],
        step_name: str,
        prompt_version_id: Optional[str],
        model_version_id: Optional[str],
        approved: bool,
        trace: Dict[str, Any],
    ) -> None:
        base_root = self.db_path.parent.parent if self.db_path.parent.name == "data" else self.db_path.parent
        run_key = str(run_id or "unassigned")
        run_dir = traces_root(base_root) / "runs" / run_key
        run_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time() * 1000)
        file_name = f"{stamp}_{step_name}_{trace_id}.json"
        path = run_dir / file_name

        prompt_row = self._get_prompt_version_row(prompt_version_id)
        model_row = self._get_model_version_row(model_version_id)
        call_name = resolve_call_name_from_step(step_name)
        if isinstance(prompt_row, dict) and prompt_row.get("call_name"):
            call_name = str(prompt_row.get("call_name"))

        payload: Dict[str, Any] = {
            "trace_id": trace_id,
            "run_id": run_id,
            "step_name": step_name,
            "call_name": call_name,
            "approved_bool": bool(approved),
            "prompt_version_id": prompt_version_id,
            "model_version_id": model_version_id,
            "prompt_version": (
                {
                    "id": prompt_row.get("id"),
                    "call_name": prompt_row.get("call_name"),
                    "step_name": prompt_row.get("step_name"),
                    "prompt_bundle_sha256": prompt_row.get("prompt_bundle_sha256") or prompt_row.get("sha256"),
                    "shared_base_sha256": prompt_row.get("shared_base_sha256"),
                    "call_base_sha256": prompt_row.get("call_base_sha256"),
                    "few_shot_sha256": prompt_row.get("few_shot_sha256"),
                    "model_name": prompt_row.get("model_name"),
                    "resolved_shared_base_path": prompt_row.get("resolved_shared_base_path"),
                    "resolved_call_base_path": prompt_row.get("resolved_call_base_path"),
                    "resolved_few_shot_path": prompt_row.get("resolved_few_shot_path"),
                    "asset_scope": prompt_row.get("asset_scope"),
                }
                if isinstance(prompt_row, dict)
                else None
            ),
            "model_version": model_row,
            "trace": trace,
            "captured_at": time.time(),
        }
        path.write_text(self._json_dumps(payload), encoding="utf-8")

    def add_trace_record(
        self,
        *,
        step_name: str,
        trace: Dict[str, Any],
        source: str,
        run_id: Optional[str] = None,
        model: Optional[str] = None,
        reasoning_level: Optional[str] = None,
        prompt_version_id: Optional[str] = None,
        model_version_id: Optional[str] = None,
        score: Optional[float] = None,
        approved: bool = False,
        approval_label: Optional[str] = None,
        approval_notes: Optional[str] = None,
        approved_by: Optional[str] = None,
        approved_at: Optional[float] = None,
        actor_id: Optional[str] = None,
    ) -> str:
        row_id = uuid.uuid4().hex
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trace_records(
                    id, run_id, step_name, model, reasoning_level, prompt_version_id,
                    model_version_id,
                    score, source, approved_bool, approval_label, approval_notes, approved_by, approved_at,
                    actor_id, trace_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row_id,
                    run_id,
                    step_name,
                    model,
                    reasoning_level,
                    prompt_version_id,
                    model_version_id,
                    score,
                    source,
                    int(bool(approved)),
                    approval_label,
                    approval_notes,
                    approved_by,
                    approved_at,
                    actor_id,
                    self._json_dumps(trace),
                    time.time(),
                ),
            )
            conn.commit()
        self._persist_trace_file(
            trace_id=row_id,
            run_id=run_id,
            step_name=step_name,
            prompt_version_id=prompt_version_id,
            model_version_id=model_version_id,
            approved=bool(approved),
            trace=trace,
        )
        return row_id

    def list_trace_records(
        self,
        *,
        step_name: Optional[str] = None,
        source: Optional[str] = None,
        run_id: Optional[str] = None,
        approved_only: bool = False,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        clauses = ["1=1"]
        params: List[Any] = []
        if step_name:
            clauses.append("step_name = ?")
            params.append(step_name)
        if source:
            clauses.append("source = ?")
            params.append(source)
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        if approved_only:
            clauses.append("approved_bool = 1")
        params.append(limit)
        query = (
            "SELECT * FROM trace_records WHERE "
            + " AND ".join(clauses)
            + " ORDER BY COALESCE(score, -1) DESC, created_at DESC LIMIT ?"
        )
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        output: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["trace"] = self._json_loads(item.pop("trace_json", None), {})
            item["approved_bool"] = bool(item.get("approved_bool", 0))
            output.append(item)
        return output

    def get_trace_record(self, trace_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM trace_records WHERE id = ? LIMIT 1",
                (trace_id,),
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["trace"] = self._json_loads(item.pop("trace_json", None), {})
        item["approved_bool"] = bool(item.get("approved_bool", 0))
        return item

    def approve_trace_record(
        self,
        *,
        trace_id: str,
        approved: bool,
        label: Optional[str],
        notes: Optional[str],
        approved_by: Optional[str],
    ) -> bool:
        with self._lock, self._connect() as conn:
            result = conn.execute(
                """
                UPDATE trace_records
                SET approved_bool = ?, approval_label = ?, approval_notes = ?, approved_by = ?, approved_at = ?
                WHERE id = ?
                """,
                (
                    int(bool(approved)),
                    label,
                    notes,
                    approved_by,
                    time.time() if approved else None,
                    trace_id,
                ),
            )
            conn.commit()
            return bool(result.rowcount)

    def create_run_pause(
        self,
        *,
        run_id: str,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        row_id = uuid.uuid4().hex
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO run_pauses(id, run_id, reason, details_json, paused_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (row_id, run_id, reason, self._json_dumps(details or {}), time.time()),
            )
            conn.commit()
        return row_id

    def resolve_run_pause(
        self,
        *,
        pause_id: str,
        decision: str,
        decided_by: Optional[str],
        rationale: Optional[str],
    ) -> bool:
        with self._lock, self._connect() as conn:
            result = conn.execute(
                """
                UPDATE run_pauses
                SET decided_at = ?, decision = ?, decided_by = ?, rationale = ?
                WHERE id = ?
                """,
                (time.time(), decision, decided_by, rationale, pause_id),
            )
            conn.commit()
            return bool(result.rowcount)

    def get_latest_run_pause(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM run_pauses
                WHERE run_id = ?
                ORDER BY paused_at DESC
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["details"] = self._json_loads(item.pop("details_json", None), {})
        return item

    def create_ralph_attempt(
        self,
        *,
        parent_run_id: str,
        attempt_index: int,
        child_run_id: Optional[str],
        harness_name: str,
        parent_harness_sha: str,
        harness_sha: str,
        mutation_actions: Optional[List[Dict[str, Any]]] = None,
        diff_summary: Optional[Dict[str, Any]] = None,
    ) -> str:
        row_id = uuid.uuid4().hex
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ralph_attempts(
                    id, parent_run_id, attempt_index, child_run_id, harness_name,
                    parent_harness_sha, harness_sha, mutation_actions_json, diff_summary_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row_id,
                    parent_run_id,
                    int(attempt_index),
                    child_run_id,
                    harness_name,
                    parent_harness_sha,
                    harness_sha,
                    self._json_dumps(mutation_actions or []),
                    self._json_dumps(diff_summary or {}),
                    time.time(),
                ),
            )
            conn.commit()
        return row_id

    def complete_ralph_attempt(
        self,
        *,
        attempt_id: str,
        stop_reason: Optional[str],
        completion_promise_met: bool,
        cost_usd: float,
    ) -> bool:
        with self._lock, self._connect() as conn:
            result = conn.execute(
                """
                UPDATE ralph_attempts
                SET stop_reason = ?, completion_promise_met_bool = ?, cost_usd = ?, completed_at = ?
                WHERE id = ?
                """,
                (
                    stop_reason,
                    int(bool(completion_promise_met)),
                    float(cost_usd),
                    time.time(),
                    attempt_id,
                ),
            )
            conn.commit()
            return bool(result.rowcount)

    def list_ralph_attempts(self, parent_run_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM ralph_attempts
                WHERE parent_run_id = ?
                ORDER BY attempt_index ASC
                """,
                (parent_run_id,),
            ).fetchall()
        output: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["mutation_actions"] = self._json_loads(item.pop("mutation_actions_json", None), [])
            item["diff_summary"] = self._json_loads(item.pop("diff_summary_json", None), {})
            cpm = item.get("completion_promise_met_bool")
            item["completion_promise_met"] = None if cpm is None else bool(cpm)
            output.append(item)
        return output

    def record_ralph_vote(
        self,
        *,
        run_id: str,
        attempt_index: int,
        step_index: int,
        candidate_a: Dict[str, Any],
        candidate_b: Dict[str, Any],
        vote: str,
        confidence: Optional[float],
        source: str,
    ) -> str:
        row_id = uuid.uuid4().hex
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ralph_votes(
                    id, run_id, attempt_index, step_index, candidate_a_json, candidate_b_json,
                    vote, confidence, source, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row_id,
                    run_id,
                    int(attempt_index),
                    int(step_index),
                    self._json_dumps(candidate_a or {}),
                    self._json_dumps(candidate_b or {}),
                    vote,
                    confidence,
                    source,
                    time.time(),
                ),
            )
            conn.commit()
        return row_id

    def list_ralph_votes(self, run_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM ralph_votes
                WHERE run_id = ?
                ORDER BY created_at ASC
                """,
                (run_id,),
            ).fetchall()
        output: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["candidate_a"] = self._json_loads(item.pop("candidate_a_json", None), {})
            item["candidate_b"] = self._json_loads(item.pop("candidate_b_json", None), {})
            output.append(item)
        return output

    def record_curation_export(
        self,
        *,
        export_type: str,
        path: str,
        manifest: Dict[str, Any],
        created_by: Optional[str] = None,
    ) -> str:
        row_id = uuid.uuid4().hex
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO curation_exports(id, export_type, path, manifest_json, created_by, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    row_id,
                    export_type,
                    path,
                    self._json_dumps(manifest),
                    created_by,
                    time.time(),
                ),
            )
            conn.commit()
        return row_id

    def list_curation_exports(self, export_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        clauses = ["1=1"]
        params: List[Any] = []
        if export_type:
            clauses.append("export_type = ?")
            params.append(export_type)
        params.append(limit)
        query = (
            "SELECT * FROM curation_exports WHERE "
            + " AND ".join(clauses)
            + " ORDER BY created_at DESC LIMIT ?"
        )
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        output: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["manifest"] = self._json_loads(item.pop("manifest_json", None), {})
            output.append(item)
        return output

    def queue_curriculum_release(
        self,
        *,
        release_date: str,
        model_name: str,
        module_id: str,
        release_kind: str,
        scheduled_publish_at: float,
        eval_run_id: Optional[str],
        payload: Dict[str, Any],
        status: str = "queued",
    ) -> str:
        row_id = uuid.uuid4().hex
        now = time.time()
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                """
                SELECT id FROM curriculum_release_queue
                WHERE release_date = ? AND model_name = ? AND release_kind = ?
                LIMIT 1
                """,
                (release_date, model_name, release_kind),
            ).fetchone()
            if existing is not None:
                row_id = str(existing["id"])
                conn.execute(
                    """
                    UPDATE curriculum_release_queue
                    SET module_id = ?, scheduled_publish_at = ?, eval_run_id = ?, status = ?,
                        payload_json = ?, updated_at = ?, published_checkpoint_id = NULL
                    WHERE id = ?
                    """,
                    (
                        module_id,
                        scheduled_publish_at,
                        eval_run_id,
                        status,
                        self._json_dumps(payload),
                        now,
                        row_id,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO curriculum_release_queue(
                        id, release_date, model_name, module_id, release_kind,
                        scheduled_publish_at, eval_run_id, status, payload_json,
                        created_at, updated_at, published_checkpoint_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                    """,
                    (
                        row_id,
                        release_date,
                        model_name,
                        module_id,
                        release_kind,
                        scheduled_publish_at,
                        eval_run_id,
                        status,
                        self._json_dumps(payload),
                        now,
                        now,
                    ),
                )
            conn.commit()
        return row_id

    def list_curriculum_releases(
        self,
        *,
        status: Optional[str] = None,
        model_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        clauses = ["1=1"]
        params: List[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if model_name:
            clauses.append("model_name = ?")
            params.append(model_name)
        params.append(limit)
        query = (
            "SELECT * FROM curriculum_release_queue WHERE "
            + " AND ".join(clauses)
            + " ORDER BY release_date DESC, scheduled_publish_at DESC LIMIT ?"
        )
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        output: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["payload"] = self._json_loads(item.pop("payload_json", None), {})
            output.append(item)
        return output

    def get_curriculum_release(self, queue_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM curriculum_release_queue WHERE id = ? LIMIT 1",
                (queue_id,),
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["payload"] = self._json_loads(item.pop("payload_json", None), {})
        return item

    def update_curriculum_release(
        self,
        queue_id: str,
        *,
        status: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        published_checkpoint_id: Optional[str] = None,
    ) -> bool:
        updates: List[str] = ["updated_at = ?"]
        params: List[Any] = [time.time()]
        if status is not None:
            updates.append("status = ?")
            params.append(status)
        if payload is not None:
            updates.append("payload_json = ?")
            params.append(self._json_dumps(payload))
        if published_checkpoint_id is not None:
            updates.append("published_checkpoint_id = ?")
            params.append(published_checkpoint_id)
        params.append(queue_id)
        with self._lock, self._connect() as conn:
            result = conn.execute(
                f"UPDATE curriculum_release_queue SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            conn.commit()
            return bool(result.rowcount)

    def record_curriculum_checkpoint(
        self,
        *,
        release_date: str,
        model_name: str,
        module_id: str,
        release_kind: str,
        manifest_path: str,
        commit_sha: Optional[str],
        git_tag: Optional[str],
        git_branch: Optional[str],
        pr_metadata: Optional[Dict[str, Any]],
        summary: Dict[str, Any],
    ) -> str:
        row_id = uuid.uuid4().hex
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO curriculum_checkpoints(
                    id, release_date, model_name, module_id, release_kind,
                    manifest_path, commit_sha, git_tag, git_branch, pr_json,
                    summary_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row_id,
                    release_date,
                    model_name,
                    module_id,
                    release_kind,
                    manifest_path,
                    commit_sha,
                    git_tag,
                    git_branch,
                    self._json_dumps(pr_metadata or {}),
                    self._json_dumps(summary),
                    time.time(),
                ),
            )
            conn.commit()
        return row_id

    def list_curriculum_checkpoints(
        self,
        *,
        model_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        clauses = ["1=1"]
        params: List[Any] = []
        if model_name:
            clauses.append("model_name = ?")
            params.append(model_name)
        params.append(limit)
        query = (
            "SELECT * FROM curriculum_checkpoints WHERE "
            + " AND ".join(clauses)
            + " ORDER BY release_date DESC, created_at DESC LIMIT ?"
        )
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        output: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["pr"] = self._json_loads(item.pop("pr_json", None), {})
            item["summary"] = self._json_loads(item.pop("summary_json", None), {})
            output.append(item)
        return output

    def get_curriculum_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM curriculum_checkpoints WHERE id = ? LIMIT 1",
                (checkpoint_id,),
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["pr"] = self._json_loads(item.pop("pr_json", None), {})
        item["summary"] = self._json_loads(item.pop("summary_json", None), {})
        return item

    def add_eval_set(
        self,
        *,
        name: str,
        version: str,
        source_path: Optional[str],
        sha256: Optional[str],
        cases: List[Dict[str, Any]],
        active: bool = True,
        purpose: str = "general",
        exposed_in_ui: bool = True,
    ) -> str:
        normalized_purpose = str(purpose or "general").strip().lower() or "general"
        if normalized_purpose not in {"general", "leaderboard_holdout"}:
            raise ValueError(f"Unsupported eval set purpose: {purpose}")
        eval_set_id = uuid.uuid4().hex
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO eval_sets(
                    id, name, version, source_path, sha256, active_bool,
                    purpose, exposed_in_ui_bool, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    eval_set_id,
                    name,
                    version,
                    source_path,
                    sha256,
                    int(active),
                    normalized_purpose,
                    int(bool(exposed_in_ui)),
                    time.time(),
                ),
            )
            for case in cases:
                case_id = str(case.get("case_id") or case.get("id") or uuid.uuid4().hex)
                input_payload = case.get("input", {})
                expected_payload = case.get("expected")
                tags = case.get("tags") or []
                conn.execute(
                    """
                    INSERT INTO eval_set_cases(id, eval_set_id, case_id, input_json, expected_json, tags_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uuid.uuid4().hex,
                        eval_set_id,
                        case_id,
                        self._json_dumps(input_payload),
                        self._json_dumps(expected_payload) if expected_payload is not None else None,
                        self._json_dumps(tags),
                    ),
                )
            conn.commit()
        return eval_set_id

    def list_eval_sets(
        self,
        *,
        purpose: Optional[str] = None,
        exposed_in_ui: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if purpose is not None:
            clauses.append("purpose = ?")
            params.append(str(purpose))
        if exposed_in_ui is not None:
            clauses.append("exposed_in_ui_bool = ?")
            params.append(int(bool(exposed_in_ui)))

        query = "SELECT * FROM eval_sets"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_eval_set(self, eval_set_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM eval_sets WHERE id = ? LIMIT 1",
                (eval_set_id,),
            ).fetchone()
        if row is None:
            return None
        return dict(row)

    def list_eval_set_cases(self, eval_set_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM eval_set_cases
                WHERE eval_set_id = ?
                ORDER BY case_id ASC
                """,
                (eval_set_id,),
            ).fetchall()
        output: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["input"] = self._json_loads(item.pop("input_json", None), {})
            item["expected"] = self._json_loads(item.pop("expected_json", None), None)
            item["tags"] = self._json_loads(item.pop("tags_json", None), [])
            output.append(item)
        return output

    def create_eval_run(
        self,
        *,
        eval_set_id: str,
        run_group_name: Optional[str],
        model: Optional[str],
        harness_bundle_hash: Optional[str],
        model_name: Optional[str] = None,
        model_family: Optional[str] = None,
        thinking_level: Optional[str] = None,
        status: str = "running",
    ) -> str:
        row_id = uuid.uuid4().hex
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO eval_runs(
                    id, eval_set_id, run_group_name, model, model_name, model_family,
                    thinking_level, harness_bundle_hash, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row_id,
                    eval_set_id,
                    run_group_name,
                    model,
                    model_name,
                    model_family,
                    thinking_level,
                    harness_bundle_hash,
                    status,
                    time.time(),
                ),
            )
            conn.commit()
        return row_id

    def set_eval_run_status(self, eval_run_id: str, status: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE eval_runs SET status = ? WHERE id = ?",
                (status, eval_run_id),
            )
            conn.commit()

    def record_eval_run_result(
        self,
        *,
        eval_run_id: str,
        case_id: str,
        run_id: Optional[str],
        score: Optional[float],
        passed: Optional[bool],
        cost: Optional[Dict[str, Any]],
        latency_ms: Optional[float],
        summary: Dict[str, Any],
    ) -> str:
        row_id = uuid.uuid4().hex
        pass_value = None if passed is None else int(bool(passed))
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO eval_run_results(
                    id, eval_run_id, case_id, run_id, score, pass_bool, cost_json, latency_ms, summary_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row_id,
                    eval_run_id,
                    case_id,
                    run_id,
                    score,
                    pass_value,
                    self._json_dumps(cost or {}),
                    latency_ms,
                    self._json_dumps(summary),
                ),
            )
            conn.commit()
        return row_id

    def list_eval_run_results(self, eval_run_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM eval_run_results
                WHERE eval_run_id = ?
                ORDER BY case_id ASC
                """,
                (eval_run_id,),
            ).fetchall()
        output: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["cost"] = self._json_loads(item.pop("cost_json", None), {})
            item["summary"] = self._json_loads(item.pop("summary_json", None), {})
            pass_value = item.get("pass_bool")
            item["pass_bool"] = None if pass_value is None else bool(pass_value)
            output.append(item)
        return output

    def list_eval_run_results_many(self, eval_run_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        run_ids = [str(run_id).strip() for run_id in eval_run_ids if str(run_id).strip()]
        if not run_ids:
            return {}

        grouped: Dict[str, List[Dict[str, Any]]] = {run_id: [] for run_id in run_ids}
        chunk_size = 900  # Keep below SQLite host parameter limits.
        with self._connect() as conn:
            for index in range(0, len(run_ids), chunk_size):
                chunk = run_ids[index : index + chunk_size]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"""
                    SELECT * FROM eval_run_results
                    WHERE eval_run_id IN ({placeholders})
                    ORDER BY eval_run_id ASC, case_id ASC
                    """,
                    chunk,
                ).fetchall()
                for row in rows:
                    item = dict(row)
                    item["cost"] = self._json_loads(item.pop("cost_json", None), {})
                    item["summary"] = self._json_loads(item.pop("summary_json", None), {})
                    pass_value = item.get("pass_bool")
                    item["pass_bool"] = None if pass_value is None else bool(pass_value)
                    grouped.setdefault(str(item.get("eval_run_id") or ""), []).append(item)
        return grouped

    def list_eval_runs(self, eval_set_id: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM eval_runs"
        params: List[Any] = []
        if eval_set_id:
            query += " WHERE eval_set_id = ?"
            params.append(eval_set_id)
        query += " ORDER BY created_at DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_eval_run(self, eval_run_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM eval_runs WHERE id = ? LIMIT 1",
                (eval_run_id,),
            ).fetchone()
        if row is None:
            return None
        return dict(row)

    @staticmethod
    def _extract_case_step_count(case: Dict[str, Any]) -> int:
        expected = case.get("expected") or {}
        if isinstance(expected, dict):
            direct = expected.get("n_mechanistic_steps")
            if isinstance(direct, int):
                return max(0, int(direct))
            if isinstance(direct, float):
                return max(0, int(direct))
            known = expected.get("known_mechanism") or expected.get("verified_mechanism")
            if isinstance(known, dict):
                min_steps = known.get("min_steps")
                if isinstance(min_steps, int):
                    return max(0, int(min_steps))
                if isinstance(min_steps, float):
                    return max(0, int(min_steps))
                steps = known.get("steps")
                if isinstance(steps, list):
                    return max(0, len(steps))
        input_payload = case.get("input") or {}
        if isinstance(input_payload, dict):
            raw = input_payload.get("n_mechanistic_steps")
            if isinstance(raw, int):
                return max(0, int(raw))
            if isinstance(raw, float):
                return max(0, int(raw))
        return 0

    @staticmethod
    def _extract_result_step_count(
        result: Dict[str, Any],
        *,
        case_step_map: Dict[str, int],
    ) -> int:
        summary = result.get("summary") or {}
        if isinstance(summary, dict):
            for key in ("n_mechanistic_steps", "min_steps"):
                raw = summary.get(key)
                if isinstance(raw, int):
                    return max(0, int(raw))
                if isinstance(raw, float):
                    return max(0, int(raw))
        case_id = str(result.get("case_id") or "")
        return max(0, int(case_step_map.get(case_id, 0)))

    def _build_holdout_bucket_metrics(
        self,
        *,
        eval_set_id: str,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        cases = self.list_eval_set_cases(eval_set_id)
        case_step_map: Dict[str, int] = {}
        target_by_step: Dict[int, int] = {}
        for case in cases:
            case_id = str(case.get("case_id") or "")
            if not case_id:
                continue
            step_count = self._extract_case_step_count(case)
            if step_count <= 0:
                continue
            case_step_map[case_id] = step_count
            target_by_step[step_count] = target_by_step.get(step_count, 0) + 1

        bucket_scores: Dict[int, List[float]] = {}
        bucket_attempts: Dict[int, int] = {}
        bucket_passes: Dict[int, int] = {}
        for row in results:
            step_count = self._extract_result_step_count(row, case_step_map=case_step_map)
            if step_count <= 0:
                continue
            score = row.get("score")
            if isinstance(score, (int, float)):
                bucket_scores.setdefault(step_count, []).append(float(score))
            bucket_attempts[step_count] = bucket_attempts.get(step_count, 0) + 1
            passed = row.get("pass_bool")
            if passed is True:
                bucket_passes[step_count] = bucket_passes.get(step_count, 0) + 1

        per_step_scores: Dict[str, Any] = {}
        weighted_quality_num = 0.0
        weighted_pass_num = 0.0
        weighted_den = 0.0
        all_weights = 0.0
        eligible_bucket_count = 0

        for step_count in sorted(set(target_by_step) | set(bucket_attempts)):
            attempted = int(bucket_attempts.get(step_count, 0))
            target_cases = int(target_by_step.get(step_count, attempted))
            gate = min(6, max(1, target_cases))
            eligible = attempted >= gate
            quality_values = bucket_scores.get(step_count, [])
            mean_quality = (sum(quality_values) / len(quality_values)) if quality_values else 0.0
            pass_rate = (
                float(bucket_passes.get(step_count, 0)) / float(attempted)
                if attempted > 0
                else 0.0
            )
            per_step_scores[str(step_count)] = {
                "step_count": int(step_count),
                "mean_quality_score": round(mean_quality, 6),
                "pass_rate": round(pass_rate, 6),
                "completed_cases": attempted,
                "target_cases": target_cases,
                "eligible_for_aggregate": bool(eligible),
            }
            if eligible:
                weight = float(step_count)
                weighted_quality_num += weight * mean_quality
                weighted_pass_num += weight * pass_rate
                weighted_den += weight
                eligible_bucket_count += 1
            all_weights += float(step_count)

        weighted_quality_score = weighted_quality_num / weighted_den if weighted_den > 0 else 0.0
        weighted_pass_rate = weighted_pass_num / weighted_den if weighted_den > 0 else 0.0
        return {
            "weighted_quality_score": round(weighted_quality_score, 6),
            "weighted_pass_rate": round(weighted_pass_rate, 6),
            "per_step_scores": per_step_scores,
            "aggregate_weighting": "linear_step_count",
            "aggregate_gate_cases": 6,
            "eligible_bucket_count": eligible_bucket_count,
            "eligible_weight_sum": round(weighted_den, 6),
            "total_weight_sum": round(all_weights, 6),
        }

    def leaderboard(self, eval_set_id: str, *, limit: int = 20) -> List[Dict[str, Any]]:
        eval_set = self.get_eval_set(eval_set_id)
        is_holdout = str((eval_set or {}).get("purpose") or "general") == "leaderboard_holdout"
        runs = self.list_eval_runs(eval_set_id)
        if not runs:
            return []

        eval_run_ids = [str(run.get("id") or "") for run in runs if str(run.get("id") or "")]
        results_by_run = self.list_eval_run_results_many(eval_run_ids)

        scored_rows: List[Dict[str, Any]] = []
        for run in runs:
            eval_run_id = str(run.get("id") or "")
            if not eval_run_id:
                continue
            results = results_by_run.get(eval_run_id, [])
            if not results:
                continue
            scores = [float(item["score"]) for item in results if isinstance(item.get("score"), (int, float))]
            if not scores:
                continue
            passes = [item.get("pass_bool") for item in results if item.get("pass_bool") is not None]
            pass_rate = (sum(1 for item in passes if item) / len(passes)) if passes else 0.0
            total_cost = 0.0
            for item in results:
                cost = item.get("cost")
                if isinstance(cost, dict):
                    value = cost.get("total_cost")
                    if isinstance(value, (int, float)):
                        total_cost += float(value)

            # Aggregate per-subagent scores across all cases for this eval run.
            subagent_quality: Dict[str, List[float]] = {}
            subagent_pass: Dict[str, List[float]] = {}
            for item in results:
                summary = item.get("summary") or {}
                per_subagent = summary.get("subagent_scores") or {}
                for subagent_id, entry in per_subagent.items():
                    if not isinstance(entry, dict):
                        continue
                    q = entry.get("quality_score")
                    p = entry.get("pass_rate")
                    if isinstance(q, (int, float)):
                        subagent_quality.setdefault(subagent_id, []).append(float(q))
                    if isinstance(p, (int, float)):
                        subagent_pass.setdefault(subagent_id, []).append(float(p))
            per_subagent_agg: Dict[str, Any] = {}
            for subagent_id, q_vals in subagent_quality.items():
                p_vals = subagent_pass.get(subagent_id, [])
                per_subagent_agg[subagent_id] = {
                    "quality_score": round(sum(q_vals) / len(q_vals), 4),
                    "pass_rate": round(sum(p_vals) / len(p_vals), 4) if p_vals else 0.0,
                    "case_count": len(q_vals),
                }

            run_group = str(run.get("run_group_name") or "")
            is_baseline = run_group.startswith("harness_free_baseline")
            is_simulated = "[SIMULATED]" in run_group

            scored_rows.append(
                {
                    "eval_run_id": eval_run_id,
                    "eval_set_id": eval_set_id,
                    "model": run.get("model"),
                    "model_name": run.get("model_name") or run.get("model"),
                    "model_family": run.get("model_family"),
                    "thinking_level": run.get("thinking_level"),
                    "run_group_name": run_group,
                    "harness_bundle_hash": run.get("harness_bundle_hash"),
                    "status": run.get("status"),
                    "created_at": run.get("created_at"),
                    "mean_quality_score": sum(scores) / len(scores),
                    "deterministic_pass_rate": pass_rate,
                    "total_cost": total_cost,
                    "case_count": len(results),
                    "per_subagent_scores": per_subagent_agg,
                    "is_baseline": is_baseline,
                    "is_simulated": is_simulated,
                    "weighted_quality_score": sum(scores) / len(scores),
                    "weighted_pass_rate": pass_rate,
                    "per_step_scores": {},
                    "aggregate_weighting": None,
                    "aggregate_gate_cases": None,
                    "eligible_bucket_count": 0,
                    "eligible_weight_sum": 0.0,
                    "total_weight_sum": 0.0,
                }
            )
            if is_holdout:
                scored_rows[-1].update(
                    self._build_holdout_bucket_metrics(eval_set_id=eval_set_id, results=results)
                )

        if is_holdout:
            scored_rows.sort(
                key=lambda item: (
                    -float(item.get("weighted_quality_score", 0.0)),
                    -float(item.get("weighted_pass_rate", 0.0)),
                    float(item.get("total_cost", 0.0)),
                )
            )
        else:
            scored_rows.sort(
                key=lambda item: (
                    -float(item.get("mean_quality_score", 0.0)),
                    -float(item.get("deterministic_pass_rate", 0.0)),
                    float(item.get("total_cost", 0.0)),
                )
            )
        return scored_rows[:limit]

    def seed_simulated_leaderboard(
        self,
        eval_set_id: str,
        *,
        case_count: int = 5,
    ) -> Dict[str, Any]:
        """Insert clearly-labelled SIMULATED placeholder leaderboard rows.

        Covers each model family (openai, claude, gemini) × thinking level
        (none, low, high) × mode (harness, baseline).  All group names are
        prefixed with ``[SIMULATED]`` so they are easy to identify and remove.
        """
        from mechanistic_agent.core.baseline_runner import SIMULATED_GROUP_PREFIX

        _SIMULATED_CONFIGS: List[Dict[str, Any]] = [
            # (family, model_label, thinking_level, is_baseline, mean_score, pass_rate, subagent_grades)
            # ---- OpenAI harness ----
            {"family": "openai", "model": "gpt-5.4", "thinking": None,  "baseline": False,
             "score": 0.62, "pass_rate": 0.48, "subagents": {
                 "initial_conditions":         {"quality_score": 0.88, "pass_rate": 0.92},
                 "missing_reagents":           {"quality_score": 0.75, "pass_rate": 0.80},
                 "atom_mapping":               {"quality_score": 0.70, "pass_rate": 0.74},
                 "reaction_type_mapping":      {"quality_score": 0.82, "pass_rate": 0.85},
                 "mechanism_step_proposal":    {"quality_score": 0.60, "pass_rate": 0.64},
                 "bond_electron_validation":   {"quality_score": 0.55, "pass_rate": 0.55},
                 "atom_balance_validation":    {"quality_score": 0.68, "pass_rate": 0.68},
                 "state_progress_validation":  {"quality_score": 0.72, "pass_rate": 0.72},
             }},
            {"family": "openai", "model": "gpt-5.4", "thinking": "low", "baseline": False,
             "score": 0.70, "pass_rate": 0.56, "subagents": {
                 "initial_conditions":         {"quality_score": 0.91, "pass_rate": 0.94},
                 "missing_reagents":           {"quality_score": 0.80, "pass_rate": 0.84},
                 "atom_mapping":               {"quality_score": 0.76, "pass_rate": 0.79},
                 "reaction_type_mapping":      {"quality_score": 0.86, "pass_rate": 0.89},
                 "mechanism_step_proposal":    {"quality_score": 0.67, "pass_rate": 0.72},
                 "bond_electron_validation":   {"quality_score": 0.61, "pass_rate": 0.61},
                 "atom_balance_validation":    {"quality_score": 0.74, "pass_rate": 0.74},
                 "state_progress_validation":  {"quality_score": 0.77, "pass_rate": 0.77},
             }},
            {"family": "openai", "model": "gpt-5.4", "thinking": "high", "baseline": False,
             "score": 0.76, "pass_rate": 0.62, "subagents": {
                 "initial_conditions":         {"quality_score": 0.93, "pass_rate": 0.96},
                 "missing_reagents":           {"quality_score": 0.84, "pass_rate": 0.88},
                 "atom_mapping":               {"quality_score": 0.80, "pass_rate": 0.83},
                 "reaction_type_mapping":      {"quality_score": 0.89, "pass_rate": 0.92},
                 "mechanism_step_proposal":    {"quality_score": 0.73, "pass_rate": 0.78},
                 "bond_electron_validation":   {"quality_score": 0.66, "pass_rate": 0.66},
                 "atom_balance_validation":    {"quality_score": 0.79, "pass_rate": 0.79},
                 "state_progress_validation":  {"quality_score": 0.82, "pass_rate": 0.82},
             }},
            # ---- Claude harness ----
            {"family": "claude", "model": "claude-opus-4.6", "thinking": None,  "baseline": False,
             "score": 0.65, "pass_rate": 0.50, "subagents": {
                 "initial_conditions":         {"quality_score": 0.89, "pass_rate": 0.93},
                 "missing_reagents":           {"quality_score": 0.77, "pass_rate": 0.81},
                 "atom_mapping":               {"quality_score": 0.72, "pass_rate": 0.76},
                 "reaction_type_mapping":      {"quality_score": 0.83, "pass_rate": 0.87},
                 "mechanism_step_proposal":    {"quality_score": 0.63, "pass_rate": 0.67},
                 "bond_electron_validation":   {"quality_score": 0.58, "pass_rate": 0.58},
                 "atom_balance_validation":    {"quality_score": 0.70, "pass_rate": 0.70},
                 "state_progress_validation":  {"quality_score": 0.74, "pass_rate": 0.74},
             }},
            {"family": "claude", "model": "claude-opus-4.6", "thinking": "low", "baseline": False,
             "score": 0.73, "pass_rate": 0.58, "subagents": {
                 "initial_conditions":         {"quality_score": 0.92, "pass_rate": 0.95},
                 "missing_reagents":           {"quality_score": 0.82, "pass_rate": 0.86},
                 "atom_mapping":               {"quality_score": 0.78, "pass_rate": 0.81},
                 "reaction_type_mapping":      {"quality_score": 0.87, "pass_rate": 0.90},
                 "mechanism_step_proposal":    {"quality_score": 0.70, "pass_rate": 0.75},
                 "bond_electron_validation":   {"quality_score": 0.63, "pass_rate": 0.63},
                 "atom_balance_validation":    {"quality_score": 0.76, "pass_rate": 0.76},
                 "state_progress_validation":  {"quality_score": 0.79, "pass_rate": 0.79},
             }},
            {"family": "claude", "model": "claude-opus-4.6", "thinking": "high", "baseline": False,
             "score": 0.79, "pass_rate": 0.66, "subagents": {
                 "initial_conditions":         {"quality_score": 0.94, "pass_rate": 0.97},
                 "missing_reagents":           {"quality_score": 0.86, "pass_rate": 0.90},
                 "atom_mapping":               {"quality_score": 0.82, "pass_rate": 0.85},
                 "reaction_type_mapping":      {"quality_score": 0.90, "pass_rate": 0.93},
                 "mechanism_step_proposal":    {"quality_score": 0.76, "pass_rate": 0.81},
                 "bond_electron_validation":   {"quality_score": 0.69, "pass_rate": 0.69},
                 "atom_balance_validation":    {"quality_score": 0.81, "pass_rate": 0.81},
                 "state_progress_validation":  {"quality_score": 0.84, "pass_rate": 0.84},
             }},
            # ---- Gemini harness ----
            {"family": "gemini", "model": "gemini-2.5-pro", "thinking": None,  "baseline": False,
             "score": 0.58, "pass_rate": 0.44, "subagents": {
                 "initial_conditions":         {"quality_score": 0.85, "pass_rate": 0.89},
                 "missing_reagents":           {"quality_score": 0.71, "pass_rate": 0.75},
                 "atom_mapping":               {"quality_score": 0.66, "pass_rate": 0.70},
                 "reaction_type_mapping":      {"quality_score": 0.79, "pass_rate": 0.83},
                 "mechanism_step_proposal":    {"quality_score": 0.56, "pass_rate": 0.60},
                 "bond_electron_validation":   {"quality_score": 0.51, "pass_rate": 0.51},
                 "atom_balance_validation":    {"quality_score": 0.64, "pass_rate": 0.64},
                 "state_progress_validation":  {"quality_score": 0.68, "pass_rate": 0.68},
             }},
            {"family": "gemini", "model": "gemini-2.5-pro", "thinking": "low", "baseline": False,
             "score": 0.66, "pass_rate": 0.52, "subagents": {
                 "initial_conditions":         {"quality_score": 0.88, "pass_rate": 0.91},
                 "missing_reagents":           {"quality_score": 0.76, "pass_rate": 0.80},
                 "atom_mapping":               {"quality_score": 0.72, "pass_rate": 0.75},
                 "reaction_type_mapping":      {"quality_score": 0.83, "pass_rate": 0.86},
                 "mechanism_step_proposal":    {"quality_score": 0.63, "pass_rate": 0.67},
                 "bond_electron_validation":   {"quality_score": 0.58, "pass_rate": 0.58},
                 "atom_balance_validation":    {"quality_score": 0.70, "pass_rate": 0.70},
                 "state_progress_validation":  {"quality_score": 0.74, "pass_rate": 0.74},
             }},
            {"family": "gemini", "model": "gemini-2.5-pro", "thinking": "high", "baseline": False,
             "score": 0.72, "pass_rate": 0.58, "subagents": {
                 "initial_conditions":         {"quality_score": 0.91, "pass_rate": 0.94},
                 "missing_reagents":           {"quality_score": 0.80, "pass_rate": 0.84},
                 "atom_mapping":               {"quality_score": 0.76, "pass_rate": 0.79},
                 "reaction_type_mapping":      {"quality_score": 0.86, "pass_rate": 0.89},
                 "mechanism_step_proposal":    {"quality_score": 0.68, "pass_rate": 0.73},
                 "bond_electron_validation":   {"quality_score": 0.63, "pass_rate": 0.63},
                 "atom_balance_validation":    {"quality_score": 0.75, "pass_rate": 0.75},
                 "state_progress_validation":  {"quality_score": 0.78, "pass_rate": 0.78},
             }},
            # ---- OpenAI baseline (harness-free) ----
            {"family": "openai", "model": "gpt-5.4", "thinking": None,  "baseline": True,
             "score": 0.41, "pass_rate": 0.28, "subagents": {
                 "full_mechanism_baseline": {"quality_score": 0.41, "pass_rate": 0.28},
             }},
            {"family": "openai", "model": "gpt-5.4", "thinking": "low", "baseline": True,
             "score": 0.48, "pass_rate": 0.34, "subagents": {
                 "full_mechanism_baseline": {"quality_score": 0.48, "pass_rate": 0.34},
             }},
            {"family": "openai", "model": "gpt-5.4", "thinking": "high", "baseline": True,
             "score": 0.54, "pass_rate": 0.40, "subagents": {
                 "full_mechanism_baseline": {"quality_score": 0.54, "pass_rate": 0.40},
             }},
            # ---- Claude baseline ----
            {"family": "claude", "model": "claude-opus-4.6", "thinking": None,  "baseline": True,
             "score": 0.44, "pass_rate": 0.30, "subagents": {
                 "full_mechanism_baseline": {"quality_score": 0.44, "pass_rate": 0.30},
             }},
            {"family": "claude", "model": "claude-opus-4.6", "thinking": "low", "baseline": True,
             "score": 0.51, "pass_rate": 0.36, "subagents": {
                 "full_mechanism_baseline": {"quality_score": 0.51, "pass_rate": 0.36},
             }},
            {"family": "claude", "model": "claude-opus-4.6", "thinking": "high", "baseline": True,
             "score": 0.57, "pass_rate": 0.44, "subagents": {
                 "full_mechanism_baseline": {"quality_score": 0.57, "pass_rate": 0.44},
             }},
            # ---- Gemini baseline ----
            {"family": "gemini", "model": "gemini-2.5-pro", "thinking": None,  "baseline": True,
             "score": 0.38, "pass_rate": 0.24, "subagents": {
                 "full_mechanism_baseline": {"quality_score": 0.38, "pass_rate": 0.24},
             }},
            {"family": "gemini", "model": "gemini-2.5-pro", "thinking": "low", "baseline": True,
             "score": 0.45, "pass_rate": 0.31, "subagents": {
                 "full_mechanism_baseline": {"quality_score": 0.45, "pass_rate": 0.31},
             }},
            {"family": "gemini", "model": "gemini-2.5-pro", "thinking": "high", "baseline": True,
             "score": 0.51, "pass_rate": 0.38, "subagents": {
                 "full_mechanism_baseline": {"quality_score": 0.51, "pass_rate": 0.38},
             }},
        ]

        inserted_run_ids: List[str] = []
        now = time.time()

        for cfg in _SIMULATED_CONFIGS:
            mode_label = "harness_free_baseline" if cfg["baseline"] else "harness"
            group_name = f"{SIMULATED_GROUP_PREFIX} {mode_label}"
            thinking = cfg.get("thinking")
            model = str(cfg["model"])
            family = str(cfg["family"])

            eval_run_id = uuid.uuid4().hex
            with self._lock, self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO eval_runs(
                        id, eval_set_id, run_group_name, model, model_name, model_family,
                        thinking_level, harness_bundle_hash, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        eval_run_id,
                        eval_set_id,
                        group_name,
                        model,
                        model,
                        family,
                        thinking,
                        "simulated",
                        "completed",
                        now - (len(inserted_run_ids) * 10),
                    ),
                )

                mean_score = float(cfg["score"])
                pass_rate = float(cfg["pass_rate"])
                subagents = cfg.get("subagents") or {}

                for i in range(case_count):
                    rng = random.Random(hash((eval_run_id, i)))
                    s = max(0.0, min(1.0, mean_score + rng.uniform(-0.08, 0.08)))
                    passed = rng.random() < pass_rate

                    per_subagent_scores: Dict[str, Any] = {}
                    for sid, grades in subagents.items():
                        q = float(grades.get("quality_score", 0.5))
                        p = float(grades.get("pass_rate", 0.5))
                        per_subagent_scores[sid] = {
                            "quality_score": round(max(0.0, min(1.0, q + rng.uniform(-0.05, 0.05))), 4),
                            "pass_rate": round(max(0.0, min(1.0, p + rng.uniform(-0.05, 0.05))), 4),
                            "case_count": 1,
                        }

                    conn.execute(
                        """
                        INSERT INTO eval_run_results(
                            id, eval_run_id, case_id, run_id, score, pass_bool,
                            cost_json, latency_ms, summary_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            uuid.uuid4().hex,
                            eval_run_id,
                            f"simulated_case_{i+1}",
                            None,
                            round(s, 4),
                            1 if passed else 0,
                            "{}",
                            round(rng.uniform(2000, 30000), 1),
                            self._json_dumps({
                                "score": round(s, 4),
                                "passed": passed,
                                "eval_mode": "baseline" if cfg["baseline"] else "harness",
                                "is_simulated": True,
                                "subagent_scores": per_subagent_scores,
                            }),
                        ),
                    )
                conn.commit()

            inserted_run_ids.append(eval_run_id)

        return {
            "inserted_eval_run_count": len(inserted_run_ids),
            "eval_run_ids": inserted_run_ids,
            "note": (
                "These are SIMULATED placeholder rows. "
                "Delete them with DELETE /api/evals/seed-simulated-leaderboard "
                "once real eval data is available."
            ),
        }

    def delete_simulated_leaderboard_rows(self, eval_set_id: str) -> Dict[str, Any]:
        """Delete all eval runs whose run_group_name contains '[SIMULATED]'."""
        from mechanistic_agent.core.baseline_runner import SIMULATED_GROUP_PREFIX

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id FROM eval_runs WHERE eval_set_id = ? AND run_group_name LIKE ?",
                (eval_set_id, f"%{SIMULATED_GROUP_PREFIX}%"),
            ).fetchall()
            run_ids = [row[0] for row in rows]

        deleted = 0
        for rid in run_ids:
            with self._lock, self._connect() as conn:
                conn.execute("DELETE FROM eval_run_results WHERE eval_run_id = ?", (rid,))
                conn.execute("DELETE FROM eval_runs WHERE id = ?", (rid,))
                conn.commit()
                deleted += 1

        return {"deleted_eval_run_count": deleted}

    def add_few_shot_example(
        self,
        *,
        step_name: str,
        example_key: str,
        input_text: str,
        output_text: str,
        approved: bool = False,
        source_trace_id: Optional[str] = None,
        score: Optional[float] = None,
        prompt_version_id: Optional[str] = None,
    ) -> str:
        row_id = uuid.uuid4().hex
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO few_shot_examples(
                    id, step_name, example_key, input_text, output_text, source_trace_id,
                    score, approved_bool, prompt_version_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(step_name, example_key) DO UPDATE SET
                    input_text = excluded.input_text,
                    output_text = excluded.output_text,
                    source_trace_id = excluded.source_trace_id,
                    score = excluded.score,
                    approved_bool = excluded.approved_bool,
                    prompt_version_id = excluded.prompt_version_id
                """,
                (
                    row_id,
                    step_name,
                    example_key,
                    input_text,
                    output_text,
                    source_trace_id,
                    score,
                    int(bool(approved)),
                    prompt_version_id,
                    time.time(),
                ),
            )
            conn.commit()
        return row_id

    def list_few_shot_examples(
        self,
        *,
        step_name: Optional[str] = None,
        approved_only: bool = True,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        clauses = ["1=1"]
        params: List[Any] = []
        if step_name:
            clauses.append("step_name = ?")
            params.append(step_name)
        if approved_only:
            clauses.append("approved_bool = 1")
        params.append(limit)
        query = (
            "SELECT * FROM few_shot_examples WHERE "
            + " AND ".join(clauses)
            + " ORDER BY COALESCE(score, -1) DESC, created_at DESC LIMIT ?"
        )
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        output: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["approved_bool"] = bool(item.get("approved_bool", 0))
            output.append(item)
        return output

    def get_latest_evaluation(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM evaluations
                WHERE run_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["summary"] = self._json_loads(item.pop("summary_json", None), {})
        return item

    def list_evaluations(self, run_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM evaluations
                WHERE run_id = ?
                ORDER BY created_at DESC
                """,
                (run_id,),
            ).fetchall()
        output: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["summary"] = self._json_loads(item.pop("summary_json", None), {})
            output.append(item)
        return output

    def get_run_snapshot(self, run_id: str) -> Optional[Dict[str, Any]]:
        run = self.get_run_row(run_id)
        if run is None:
            return None
        events = self.list_events(run_id, after_seq=0, limit=5000)
        steps = self.list_step_outputs(run_id)
        feedback = self.list_feedback(run_id)
        evaluations = self.list_evaluations(run_id)
        latest_pause = self.get_latest_run_pause(run_id)
        ralph_attempts = self.list_ralph_attempts(run_id)
        ralph_votes = self.list_ralph_votes(run_id)
        run["events"] = events
        run["step_outputs"] = steps
        run["feedback"] = feedback
        run["evaluations"] = evaluations
        run["latest_pause"] = latest_pause
        run["ralph_attempts"] = ralph_attempts
        run["ralph_votes"] = ralph_votes
        if ralph_attempts:
            latest_attempt = ralph_attempts[-1]
            child_run_id = str(latest_attempt.get("child_run_id") or "").strip()
            run["ralph_latest_child_run_id"] = child_run_id or None
            if child_run_id:
                child_row = self.get_run_row(child_run_id)
                if child_row is not None:
                    run["ralph_latest_child_status"] = child_row.get("status")
        pending = self.unaccepted_verified_steps(run_id)
        run["pending_verification"] = [
            {
                "step_name": row["step_name"],
                "attempt": row["attempt"],
                "validation": row.get("validation"),
            }
            for row in pending
        ]
        run["cost_summary"] = self.get_run_cost_summary(run_id)
        return run

    # ---- Verification Results ----

    def upsert_verification_result(
        self,
        *,
        harness_version: str,
        model_family: str,
        step_name: str,
        verified_model: str,
        verified_reasoning: Optional[str],
        baseline_score: float,
        step_score: float,
        eval_set_id: Optional[str] = None,
        eval_run_id: Optional[str] = None,
    ) -> str:
        result_id = uuid.uuid4().hex
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO verification_results
                    (id, harness_version, model_family, step_name,
                     verified_model, verified_reasoning,
                     baseline_score, step_score, eval_set_id, eval_run_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(harness_version, model_family, step_name) DO UPDATE SET
                    id = excluded.id,
                    verified_model = excluded.verified_model,
                    verified_reasoning = excluded.verified_reasoning,
                    baseline_score = excluded.baseline_score,
                    step_score = excluded.step_score,
                    eval_set_id = excluded.eval_set_id,
                    eval_run_id = excluded.eval_run_id,
                    created_at = excluded.created_at
                """,
                (
                    result_id, harness_version, model_family, step_name,
                    verified_model, verified_reasoning,
                    baseline_score, step_score, eval_set_id, eval_run_id, now,
                ),
            )
            conn.commit()
        return result_id

    def get_verified_step_models(
        self, *, harness_version: str, model_family: str
    ) -> Dict[str, Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT step_name, verified_model, verified_reasoning,
                       baseline_score, step_score
                FROM verification_results
                WHERE harness_version = ? AND model_family = ?
                """,
                (harness_version, model_family),
            ).fetchall()
        return {
            row["step_name"]: {
                "model": row["verified_model"],
                "reasoning": row["verified_reasoning"],
                "baseline_score": row["baseline_score"],
                "step_score": row["step_score"],
            }
            for row in rows
        }

    def list_verification_history(
        self, *, model_family: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            if model_family:
                rows = conn.execute(
                    """
                    SELECT * FROM verification_results
                    WHERE model_family = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (model_family, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM verification_results
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return [dict(row) for row in rows]

    # ---- Verification Jobs ----

    def create_verification_job(
        self,
        *,
        model_family: str,
        eval_set_id: str,
        harness_version: str,
    ) -> str:
        job_id = uuid.uuid4().hex
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO verification_jobs
                    (id, model_family, eval_set_id, harness_version,
                     status, progress_json, created_at)
                VALUES (?, ?, ?, ?, 'pending', '{}', ?)
                """,
                (job_id, model_family, eval_set_id, harness_version, now),
            )
            conn.commit()
        return job_id

    def update_verification_job_progress(
        self, job_id: str, *, status: str, progress: Dict[str, Any]
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE verification_jobs
                SET status = ?, progress_json = ?
                WHERE id = ?
                """,
                (status, self._json_dumps(progress), job_id),
            )
            conn.commit()

    def complete_verification_job(
        self, job_id: str, *, status: str, result: Dict[str, Any]
    ) -> None:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE verification_jobs
                SET status = ?, result_json = ?, completed_at = ?
                WHERE id = ?
                """,
                (status, self._json_dumps(result), now, job_id),
            )
            conn.commit()

    def get_verification_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM verification_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["progress"] = self._json_loads(item.pop("progress_json", None), {})
        item["result"] = self._json_loads(item.pop("result_json", None), None)
        return item

    def list_verification_jobs(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM verification_jobs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        output: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["progress"] = self._json_loads(item.pop("progress_json", None), {})
            item["result"] = self._json_loads(item.pop("result_json", None), None)
            output.append(item)
        return output
