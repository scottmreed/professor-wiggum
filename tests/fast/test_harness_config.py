"""Tests for harness configuration loading, validation, and legacy overrides."""
from __future__ import annotations

from pathlib import Path

import pytest

from mechanistic_agent.core.types import FewShotSelectionConfig, HarnessConfig, ModuleSpec, RunConfig
from mechanistic_agent.core.registries import HarnessRegistry

HARNESS_DIR = Path(__file__).resolve().parents[2] / "harness_versions"


@pytest.fixture
def registry() -> HarnessRegistry:
    return HarnessRegistry(HARNESS_DIR)


@pytest.fixture
def default_config(registry: HarnessRegistry) -> HarnessConfig:
    return registry.load("default")


# ---------------------------------------------------------------------------
# Phase 1: basic loading
# ---------------------------------------------------------------------------


class TestDefaultHarnessLoads:
    def test_loads_without_error(self, default_config: HarnessConfig) -> None:
        assert default_config.name == "default"

    def test_has_version_sha(self, default_config: HarnessConfig) -> None:
        assert len(default_config.version) == 64  # SHA-256 hex

    def test_version_is_stable(self, registry: HarnessRegistry) -> None:
        a = registry.load("default")
        b = registry.load("default")
        assert a.version == b.version

    def test_pre_loop_count(self, default_config: HarnessConfig) -> None:
        assert len(default_config.pre_loop_modules) == 7

    def test_post_step_count(self, default_config: HarnessConfig) -> None:
        assert len(default_config.post_step_modules) == 5

    def test_all_modules_have_ids(self, default_config: HarnessConfig) -> None:
        for m in default_config.all_modules():
            assert m.id, f"Module missing id: {m}"

    def test_all_modules_have_labels(self, default_config: HarnessConfig) -> None:
        for m in default_config.all_modules():
            assert m.label, f"Module {m.id} missing label"


class TestPreLoopModuleOrder:
    EXPECTED_ORDER = [
        "balance_analysis",
        "functional_groups",
        "ph_recommendation",
        "initial_conditions",
        "missing_reagents",
        "atom_mapping",
        "reaction_type_mapping",
    ]

    def test_order_matches(self, default_config: HarnessConfig) -> None:
        ids = [m.id for m in default_config.pre_loop_modules]
        assert ids == self.EXPECTED_ORDER


class TestPostStepModuleOrder:
    EXPECTED_ORDER = [
        "bond_electron_validation",
        "atom_balance_validation",
        "state_progress_validation",
        "reflection",
        "step_atom_mapping",
    ]

    def test_order_matches(self, default_config: HarnessConfig) -> None:
        ids = [m.id for m in default_config.post_step_modules]
        assert ids == self.EXPECTED_ORDER


class TestModuleProperties:
    def test_balance_analysis_not_removable(self, default_config: HarnessConfig) -> None:
        m = next(m for m in default_config.pre_loop_modules if m.id == "balance_analysis")
        assert not m.removable
        assert not m.movable

    def test_functional_groups_removable(self, default_config: HarnessConfig) -> None:
        m = next(m for m in default_config.pre_loop_modules if m.id == "functional_groups")
        assert m.removable
        assert m.movable

    def test_validators_individually_removable(self, default_config: HarnessConfig) -> None:
        for vid in ("bond_electron_validation", "atom_balance_validation", "state_progress_validation"):
            m = next(m for m in default_config.post_step_modules if m.id == vid)
            assert m.removable, f"{vid} should be removable"
            assert not m.movable, f"{vid} should not be movable"

    def test_conditions_pair_group_key(self, default_config: HarnessConfig) -> None:
        ph = next(m for m in default_config.pre_loop_modules if m.id == "ph_recommendation")
        cond = next(m for m in default_config.pre_loop_modules if m.id == "initial_conditions")
        assert ph.group_key == "conditions_pair"
        assert cond.group_key == "conditions_pair"

    def test_llm_modules_have_correct_kind(self, default_config: HarnessConfig) -> None:
        llm_ids = {"initial_conditions", "missing_reagents", "atom_mapping", "reaction_type_mapping", "step_atom_mapping"}
        for m in default_config.all_modules():
            if m.id in llm_ids:
                assert m.kind == "llm", f"{m.id} should be llm"

    def test_deterministic_modules_have_correct_kind(self, default_config: HarnessConfig) -> None:
        det_ids = {"balance_analysis", "functional_groups", "ph_recommendation",
                   "bond_electron_validation", "atom_balance_validation",
                   "state_progress_validation", "reflection"}
        for m in default_config.all_modules():
            if m.id in det_ids:
                assert m.kind == "deterministic", f"{m.id} should be deterministic"


# ---------------------------------------------------------------------------
# Phase 1: legacy flag overrides
# ---------------------------------------------------------------------------


class TestLegacyOverrides:
    def test_disable_functional_groups(self, registry: HarnessRegistry) -> None:
        rc = RunConfig(model="gpt-4", functional_groups_enabled=False)
        config = registry.resolve_from_run_config(rc)
        fg = next(m for m in config.pre_loop_modules if m.id == "functional_groups")
        assert not fg.enabled

    def test_remove_missing_reagents_from_optional(self, registry: HarnessRegistry) -> None:
        rc = RunConfig(model="gpt-4", optional_llm_tools=["attempt_atom_mapping"])
        config = registry.resolve_from_run_config(rc)
        mr = next(m for m in config.pre_loop_modules if m.id == "missing_reagents")
        assert not mr.enabled

    def test_remove_atom_mapping_from_optional(self, registry: HarnessRegistry) -> None:
        rc = RunConfig(model="gpt-4", optional_llm_tools=["predict_missing_reagents"])
        config = registry.resolve_from_run_config(rc)
        am = next(m for m in config.pre_loop_modules if m.id == "atom_mapping")
        assert not am.enabled

    def test_reaction_template_off(self, registry: HarnessRegistry) -> None:
        rc = RunConfig(model="gpt-4", reaction_template_policy="off")
        config = registry.resolve_from_run_config(rc)
        rtm = next(m for m in config.pre_loop_modules if m.id == "reaction_type_mapping")
        assert not rtm.enabled

    def test_step_mapping_disabled(self, registry: HarnessRegistry) -> None:
        rc = RunConfig(model="gpt-4", step_mapping_enabled=False)
        config = registry.resolve_from_run_config(rc)
        sm = next(m for m in config.post_step_modules if m.id == "step_atom_mapping")
        assert not sm.enabled

    def test_default_config_all_enabled(self, registry: HarnessRegistry) -> None:
        rc = RunConfig(
            model="gpt-4",
            optional_llm_tools=["predict_missing_reagents", "attempt_atom_mapping"],
        )
        config = registry.resolve_from_run_config(rc)
        for m in config.all_modules():
            assert m.enabled, f"{m.id} should be enabled with default settings"

    def test_no_harness_path_uses_default(self, registry: HarnessRegistry) -> None:
        rc = RunConfig(model="gpt-4")
        config = registry.resolve_from_run_config(rc)
        assert config.name == "default"
        assert len(config.pre_loop_modules) == 7


# ---------------------------------------------------------------------------
# Phase 1: schema v2 fields
# ---------------------------------------------------------------------------


class TestHarnessSchemaV2:
    def test_schema_version(self, default_config: HarnessConfig) -> None:
        assert default_config.schema_version == "2.1"

    def test_tool_calling_mode(self, default_config: HarnessConfig) -> None:
        assert default_config.tool_calling_mode == "forced"

    def test_loop_module_present(self, default_config: HarnessConfig) -> None:
        assert default_config.loop_module is not None
        assert default_config.loop_module.get("id") == "mechanism_step_proposal"
        assert default_config.loop_module.get("prompt_call_name") == "propose_mechanism_step"

    def test_execution_note_present(self, default_config: HarnessConfig) -> None:
        assert default_config.execution_note

    def test_modules_have_step_field(self, default_config: HarnessConfig) -> None:
        for m in default_config.pre_loop_modules:
            assert m.step > 0, f"Module {m.id} missing step field"
        for m in default_config.post_step_modules:
            assert m.step > 0, f"Module {m.id} missing step field"

    def test_validator_modules_have_skill_ref(self, default_config: HarnessConfig) -> None:
        validator_ids = {"bond_electron_validation", "atom_balance_validation", "state_progress_validation"}
        for m in default_config.post_step_modules:
            if m.id in validator_ids:
                assert m.validator_skill == m.id, f"{m.id} should reference its skill"

    def test_metadata_has_changelog(self, default_config: HarnessConfig) -> None:
        changelog = default_config.metadata.get("changelog")
        assert isinstance(changelog, list) and len(changelog) >= 1

    def test_few_shot_defaults_present(self, default_config: HarnessConfig) -> None:
        assert default_config.few_shot_defaults.enabled is True
        assert default_config.few_shot_defaults.max_examples == 4
        assert default_config.few_shot_defaults.selection_strategy == "top_score"

    def test_few_shot_policy_resolves_for_reaction_type(self, default_config: HarnessConfig) -> None:
        policy = default_config.few_shot_policy_for_call("select_reaction_type")
        assert policy.max_examples == 4
        assert policy.selection_strategy == "top_score"


class TestNoToolsBaselineHarness:
    @pytest.fixture
    def no_tools_config(self, registry: HarnessRegistry) -> HarnessConfig:
        return registry.load("no_tools_baseline")

    def test_loads_without_error(self, no_tools_config: HarnessConfig) -> None:
        assert no_tools_config.name == "no_tools_baseline"

    def test_tool_calling_mode_none(self, no_tools_config: HarnessConfig) -> None:
        assert no_tools_config.tool_calling_mode == "none"

    def test_no_pre_loop_modules(self, no_tools_config: HarnessConfig) -> None:
        assert len(no_tools_config.pre_loop_modules) == 0

    def test_no_post_step_modules(self, no_tools_config: HarnessConfig) -> None:
        assert len(no_tools_config.post_step_modules) == 0

    def test_loop_module_is_text_completion(self, no_tools_config: HarnessConfig) -> None:
        assert no_tools_config.loop_module is not None
        assert no_tools_config.loop_module.get("kind") == "text_completion"


# ---------------------------------------------------------------------------
# Phase 1: serialization round-trip
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_round_trip(self, default_config: HarnessConfig) -> None:
        data = default_config.as_dict()
        restored = HarnessConfig.from_dict(data)
        assert len(restored.pre_loop_modules) == len(default_config.pre_loop_modules)
        assert len(restored.post_step_modules) == len(default_config.post_step_modules)
        for orig, rest in zip(default_config.all_modules(), restored.all_modules()):
            assert orig.id == rest.id
            assert orig.kind == rest.kind
            assert orig.enabled == rest.enabled

    def test_module_spec_round_trip(self) -> None:
        spec = ModuleSpec(
            id="test_mod",
            label="Test Module",
            kind="llm",
            phase="pre_loop",
            enabled=True,
            inputs=["a", "b"],
            outputs=["c"],
            custom=True,
            prompt_text="You are a test prompt.",
            few_shot=FewShotSelectionConfig(max_examples=2, selection_strategy="most_recent"),
        )
        data = spec.as_dict()
        restored = ModuleSpec.from_dict(data)
        assert restored.id == "test_mod"
        assert restored.custom is True
        assert restored.prompt_text == "You are a test prompt."
        assert restored.inputs == ["a", "b"]
        assert restored.few_shot is not None
        assert restored.few_shot.max_examples == 2
        assert restored.few_shot.selection_strategy == "most_recent"


# ---------------------------------------------------------------------------
# Phase 1: list versions
# ---------------------------------------------------------------------------


class TestListVersions:
    def test_lists_default(self, registry: HarnessRegistry) -> None:
        versions = registry.list_versions()
        names = [v["name"] for v in versions]
        assert "default" in names

    def test_version_has_sha(self, registry: HarnessRegistry) -> None:
        versions = registry.list_versions()
        for v in versions:
            assert len(v["version"]) == 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_enabled_pre_loop(self, default_config: HarnessConfig) -> None:
        enabled = default_config.enabled_pre_loop()
        assert len(enabled) == 7  # all enabled by default

    def test_enabled_post_step(self, default_config: HarnessConfig) -> None:
        enabled = default_config.enabled_post_step()
        assert len(enabled) == 5  # all enabled by default

    def test_disabled_module_filtered(self) -> None:
        config = HarnessConfig(
            pre_loop_modules=[
                ModuleSpec(id="a", label="A", kind="deterministic", phase="pre_loop", enabled=True),
                ModuleSpec(id="b", label="B", kind="deterministic", phase="pre_loop", enabled=False),
            ],
        )
        assert len(config.enabled_pre_loop()) == 1
        assert config.enabled_pre_loop()[0].id == "a"


# ---------------------------------------------------------------------------
# Phase 3: flow graph generation from harness config
# ---------------------------------------------------------------------------


_has_fastapi = True
try:
    import fastapi  # noqa: F401
except ImportError:
    _has_fastapi = False

_skip_no_fastapi = pytest.mark.skipif(not _has_fastapi, reason="fastapi not installed")


@_skip_no_fastapi
class TestFlowGraphGeneration:
    """Tests that generated flow nodes/edges match the old hardcoded constants."""

    def test_generated_node_ids_match_hardcoded(self, default_config: HarnessConfig) -> None:
        from mechanistic_agent.api.app import build_flow_node_specs, FLOW_NODE_SPECS

        generated = build_flow_node_specs(default_config)
        generated_ids = {n["id"] for n in generated}
        hardcoded_ids = {n["id"] for n in FLOW_NODE_SPECS}
        assert generated_ids == hardcoded_ids

    def test_generated_workflow_order_matches_hardcoded(self, default_config: HarnessConfig) -> None:
        from mechanistic_agent.api.app import build_workflow_order, WORKFLOW_ORDER

        generated = build_workflow_order(default_config)
        assert generated == WORKFLOW_ORDER

    def test_generated_edges_cover_hardcoded_sources_targets(self, default_config: HarnessConfig) -> None:
        from mechanistic_agent.api.app import build_flow_edges, FLOW_EDGES

        generated = build_flow_edges(default_config)
        gen_pairs = {(e["source"], e["target"]) for e in generated}
        hc_pairs = {(e["source"], e["target"]) for e in FLOW_EDGES}
        assert gen_pairs == hc_pairs

    def test_removing_module_removes_from_nodes_and_edges(self, default_config: HarnessConfig) -> None:
        from mechanistic_agent.api.app import build_flow_node_specs, build_flow_edges

        # Disable functional_groups
        for m in default_config.pre_loop_modules:
            if m.id == "functional_groups":
                m.enabled = False
                break

        nodes = build_flow_node_specs(default_config)
        node_ids = {n["id"] for n in nodes}
        assert "functional_groups" not in node_ids

        edges = build_flow_edges(default_config)
        for edge in edges:
            assert edge["source"] != "functional_groups"
            assert edge["target"] != "functional_groups"

        # Verify the chain skips correctly: balance_analysis → ph_recommendation
        edge_pairs = {(e["source"], e["target"]) for e in edges}
        assert ("balance_analysis", "ph_recommendation") in edge_pairs

    def test_removing_validator_removes_from_post_step(self, default_config: HarnessConfig) -> None:
        from mechanistic_agent.api.app import build_flow_node_specs, build_flow_edges

        for m in default_config.post_step_modules:
            if m.id == "bond_electron_validation":
                m.enabled = False
                break

        nodes = build_flow_node_specs(default_config)
        node_ids = {n["id"] for n in nodes}
        assert "bond_electron_validation" not in node_ids

        edges = build_flow_edges(default_config)
        for edge in edges:
            assert edge["source"] != "bond_electron_validation"
            assert edge["target"] != "bond_electron_validation"

    def test_pre_loop_order_preserved(self, default_config: HarnessConfig) -> None:
        from mechanistic_agent.api.app import build_flow_node_specs

        nodes = build_flow_node_specs(default_config)
        pre_loop_nodes = [n for n in nodes if n.get("phase") == "pre_loop"]
        ids = [n["id"] for n in pre_loop_nodes]
        expected = [m.id for m in default_config.enabled_pre_loop()]
        assert ids == expected
