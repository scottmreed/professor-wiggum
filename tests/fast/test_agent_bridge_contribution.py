"""Fast tests for the keyless agent-bridge contribution workflow.

These lock the guarantees the contribution PRD relies on:

* **No silent fallback** — selecting the bridge always routes to the bridge,
  even when hosted provider keys are present (SOUL.md Guardrail 5).
* **Origin provenance** — bridge runs are stamped with a declared ``config.origin``
  block so the data's origin is auditable; hosted runs are untouched (additive).
* **Privacy / holdout isolation** — the request envelope can only ever carry the
  model-visible inputs, never run state, ground truth, or scoring hints.
"""
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from mechanistic_agent.agent_bridge import (
    MODEL_INPUT_KEYS,
    AgentBridgeAdapter,
    build_model_input,
    build_origin_provenance,
    origin_for_config,
)
from mechanistic_agent.llm import get_chat_model


# --- No silent fallback -----------------------------------------------------

def test_no_fallback_to_keyed_model_when_bridge_selected(tmp_path, monkeypatch) -> None:
    """Selecting the bridge must route to the bridge even with hosted keys set.

    A delegated run must never silently degrade into a keyed-model call.
    """
    monkeypatch.setenv("MECHANISTIC_AGENT_BRIDGE_DIR", str(tmp_path))
    # Hosted keys ARE present — the bridge must still win for a bridge model.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-be-used")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-be-used")
    adapter = get_chat_model("agent-bridge")
    assert isinstance(adapter, AgentBridgeAdapter)


# --- Origin provenance ------------------------------------------------------

def test_build_origin_provenance_defaults_and_env(monkeypatch) -> None:
    for var in (
        "MECHANISTIC_AGENT_BRIDGE_DECLARED_MODEL",
        "MECHANISTIC_AGENT_BRIDGE_RESPONDER_KIND",
        "MECHANISTIC_AGENT_BRIDGE_NOTES",
    ):
        monkeypatch.delenv(var, raising=False)
    default = build_origin_provenance("agent-bridge")
    assert default["responder"] == "agent-bridge"
    assert default["declared_underlying_model"] == "undeclared"
    assert default["responder_kind"] == "undeclared"
    assert default["budget_observability"] == "opaque"  # never implies free SOTA

    monkeypatch.setenv("MECHANISTIC_AGENT_BRIDGE_DECLARED_MODEL", "opus-4.8 (Hyperagent)")
    monkeypatch.setenv("MECHANISTIC_AGENT_BRIDGE_RESPONDER_KIND", "orchestrator_subagents")
    declared = build_origin_provenance("agent-bridge")
    assert declared["declared_underlying_model"] == "opus-4.8 (Hyperagent)"
    assert declared["responder_kind"] == "orchestrator_subagents"


def test_origin_for_config_detects_bridge_and_ignores_hosted(monkeypatch) -> None:
    monkeypatch.delenv("MECHANISTIC_ACTIVE_MODEL", raising=False)
    assert origin_for_config({"model": "agent-bridge", "model_name": "agent-bridge"})
    # Detection also covers per-step bridge use.
    assert origin_for_config({"model": "gpt-5.5", "step_models": {"x": "agent-bridge"}})
    # Hosted-only config gets no origin (stored config untouched).
    assert origin_for_config({"model": "anthropic/claude-opus-4.6"}) is None


def test_origin_for_config_honours_active_model_env(monkeypatch) -> None:
    """MECHANISTIC_ACTIVE_MODEL=agent-bridge forces every step through the bridge."""
    monkeypatch.setenv("MECHANISTIC_ACTIVE_MODEL", "agent-bridge")
    origin = origin_for_config({"model": "gpt-5.5", "model_name": "gpt-5.5"})
    assert origin is not None
    assert origin["responder"] == "agent-bridge"


def test_create_run_stamps_origin_for_bridge_only(tmp_path, monkeypatch) -> None:
    """RunStore.create_run injects origin for bridge runs, not hosted runs."""
    monkeypatch.delenv("MECHANISTIC_ACTIVE_MODEL", raising=False)
    from mechanistic_agent.core import RunStore

    store = RunStore(tmp_path / "data" / "mechanistic.db")
    bridge_run = store.create_run(
        mode="unverified",
        input_payload={"starting_materials": ["CCBr.[I-]"], "products": ["CCI.[Br-]"]},
        config={"model": "agent-bridge", "model_name": "agent-bridge"},
        prompt_bundle_hash="",
        skill_bundle_hash="",
    )
    origin = store.get_run_row(bridge_run)["config"].get("origin")
    assert origin and origin["responder"] == "agent-bridge"
    assert origin["budget_observability"] == "opaque"

    hosted_run = store.create_run(
        mode="unverified",
        input_payload={"starting_materials": ["CCBr"], "products": ["CCI"]},
        config={"model": "anthropic/claude-opus-4.6", "model_name": "anthropic/claude-opus-4.6"},
        prompt_bundle_hash="",
        skill_bundle_hash="",
    )
    assert "origin" not in store.get_run_row(hosted_run)["config"]


# --- Privacy / holdout isolation -------------------------------------------

def test_model_input_envelope_cannot_carry_ground_truth() -> None:
    """build_model_input returns ONLY messages/tools/tool_choice — the single place
    that decides what a responder may see. Holdout answers/scoring cannot ride along.
    """
    block = build_model_input(
        [{"role": "user", "content": "map it"}],
        [{"type": "function", "function": {"name": "atom_mapping_result"}}],
        {"type": "function", "function": {"name": "atom_mapping_result"}},
    )
    assert set(block.keys()) == set(MODEL_INPUT_KEYS)
    serialized = json.dumps(block)
    assert "expected" not in serialized and "ground_truth" not in serialized


# --- bridge-serve CLI -------------------------------------------------------

def test_bridge_serve_replay_writes_response(tmp_path) -> None:
    """`bridge-serve --replay --once` answers a pending request from a seed file."""
    import main as cli

    bridge_dir = tmp_path / "bridge"
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir(parents=True, exist_ok=True)

    adapter = AgentBridgeAdapter(model="agent-bridge", bridge_dir=str(bridge_dir))
    req = adapter._write_request(
        [{"role": "user", "content": "x"}],
        [{"type": "function", "function": {"name": "reaction_type_selection_result"}}],
        {"type": "function", "function": {"name": "reaction_type_selection_result"}},
    )
    # Seed a bare arguments object; bridge-serve wraps it for the forced tool.
    (seed_dir / req.name).write_text(json.dumps({"selected_label_exact": "Finkelstein halide exchange"}))

    result = CliRunner().invoke(
        cli.app,
        ["bridge-serve", "--bridge-dir", str(bridge_dir), "--replay", str(seed_dir), "--once"],
    )
    assert result.exit_code == 0, result.output

    response_path = bridge_dir / "responses" / req.name
    assert response_path.exists()
    payload = json.loads(response_path.read_text())
    call = payload["tool_calls"][0]
    assert call["name"] == "reaction_type_selection_result"
    assert call["arguments"]["selected_label_exact"] == "Finkelstein halide exchange"


# --- Keyless run summary never tracebacks on opaque cost ---------------------
# A completed agent-bridge run stores ``cost_summary['total_cost'] = None``
# (cost is opaque; no API spend recorded). The CLI run summary must coerce that
# to a numeric total instead of calling ``.get`` on ``None`` — otherwise every
# keyless run tracebacks *after* completing, breaking the contribution entry point.
import main as main_cli  # noqa: E402


def test_summary_total_cost_handles_opaque_bridge_cost():
    # Shape stored by an agent-bridge run: present key, ``None`` value.
    assert main_cli._summary_total_cost({"total_cost": None, "step_costs": []}) == 0.0


def test_summary_total_cost_reads_hosted_cost_and_tolerates_junk():
    assert main_cli._summary_total_cost({"total_cost": {"total_cost": 0.0123}}) == 0.0123
    # Defensive: missing block, non-dict block, and non-numeric value all → 0.0.
    assert main_cli._summary_total_cost({}) == 0.0
    assert main_cli._summary_total_cost(None) == 0.0
    assert main_cli._summary_total_cost({"total_cost": "n/a"}) == 0.0
    assert main_cli._summary_total_cost({"total_cost": {"total_cost": "x"}}) == 0.0


def test_active_model_env_forces_bridge_dispatch_over_run_model(monkeypatch) -> None:
    """MECHANISTIC_ACTIVE_MODEL must force step dispatch through the bridge even when
    the per-run (thread-local) model is a hosted id.

    Regression guard: before this fix the coordinator's thread-local active model
    shadowed MECHANISTIC_ACTIVE_MODEL, so a hosted model id + the bridge override
    raised "<provider> API key not configured" at the proposal step instead of
    dispatching keyless. Dispatch must agree with origin_for_config, which already
    treats the env var as authoritative.
    """
    from mechanistic_agent.tools import _resolve_step_model
    from mechanistic_agent.core.model_context import set_run_context, clear_run_context

    # Simulate a run attributed to a hosted model (what the coordinator sets).
    set_run_context(
        step_models={},
        step_reasoning={},
        active_model="anthropic/claude-opus-4.8",
        model_family="claude",
    )
    try:
        # Without the override, the run model is used.
        monkeypatch.delenv("MECHANISTIC_ACTIVE_MODEL", raising=False)
        assert _resolve_step_model("intermediates", "MECHANISTIC_INTERMEDIATE_MODEL") == "anthropic/claude-opus-4.8"

        # With the override set, every step is forced through the bridge.
        monkeypatch.setenv("MECHANISTIC_ACTIVE_MODEL", "agent-bridge")
        for step, env in [
            ("intermediates", "MECHANISTIC_INTERMEDIATE_MODEL"),
            ("atom_mapping", "MECHANISTIC_ATOM_MAPPING_MODEL"),
            ("reaction_type", "MECHANISTIC_REACTION_TYPE_MODEL"),
        ]:
            assert _resolve_step_model(step, env) == "agent-bridge"
    finally:
        clear_run_context()
