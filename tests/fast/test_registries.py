from __future__ import annotations

from pathlib import Path

from mechanistic_agent.core.registries import RegistrySet

_SKILL_FRONTMATTER = """\
---
skill_type: mechanistic
call_name: assess_initial_conditions
steps: [initial_conditions]
phase: pre_loop
kind: llm
---

# Assess Initial Conditions

Description.

## Prompt

<!-- PROMPT_START -->
{prompt_body}
<!-- PROMPT_END -->
"""


def _make_skill_md(text: str) -> str:
    return _SKILL_FRONTMATTER.format(prompt_body=text)


def test_bundle_hash_changes_when_prompt_changes(tmp_path: Path) -> None:
    # New layout: skills/mechanistic/<call_name>/SKILL.md
    mechanistic_dir = tmp_path / "skills" / "mechanistic"
    base_system_dir = mechanistic_dir / "base_system"
    call_dir = mechanistic_dir / "assess_initial_conditions"
    project_skill = tmp_path / "skills" / "project" / "alpha"
    memory = tmp_path / "memory_packs"

    base_system_dir.mkdir(parents=True)
    call_dir.mkdir(parents=True)
    project_skill.mkdir(parents=True)
    memory.mkdir(parents=True)

    base_sys_frontmatter = "---\nskill_type: mechanistic\ncall_name: base_system\nkind: shared_base\n---\n## Prompt\n<!-- PROMPT_START -->\nbase\n<!-- PROMPT_END -->\n"
    (base_system_dir / "SKILL.md").write_text(base_sys_frontmatter, encoding="utf-8")
    (call_dir / "SKILL.md").write_text(_make_skill_md("first"), encoding="utf-8")
    (call_dir / "few_shot.jsonl").write_text("", encoding="utf-8")
    (project_skill / "SKILL.md").write_text("# project skill", encoding="utf-8")
    (memory / "pack.md").write_text("memory", encoding="utf-8")

    registry = RegistrySet(tmp_path)
    before = registry.bundle_hashes()

    (call_dir / "SKILL.md").write_text(_make_skill_md("second"), encoding="utf-8")
    after = registry.bundle_hashes()

    assert before["prompt_bundle_hash"] != after["prompt_bundle_hash"]
    assert before["skill_bundle_hash"] != after["skill_bundle_hash"]  # SKILL.md is also tracked by skills
    assert before["memory_bundle_hash"] == after["memory_bundle_hash"]


def test_prompt_frontmatter_parsed_for_step_and_version(tmp_path: Path) -> None:
    mechanistic_dir = tmp_path / "skills" / "mechanistic"
    base_system_dir = mechanistic_dir / "base_system"
    call_dir = mechanistic_dir / "assess_initial_conditions"
    (tmp_path / "skills" / "project" / "alpha").mkdir(parents=True)
    (tmp_path / "memory_packs").mkdir(parents=True)
    base_system_dir.mkdir(parents=True)
    call_dir.mkdir(parents=True)

    base_sys_frontmatter = "---\nskill_type: mechanistic\ncall_name: base_system\nkind: shared_base\n---\n## Prompt\n<!-- PROMPT_START -->\nshared\n<!-- PROMPT_END -->\n"
    (base_system_dir / "SKILL.md").write_text(base_sys_frontmatter, encoding="utf-8")
    (call_dir / "SKILL.md").write_text(_make_skill_md("Prompt body"), encoding="utf-8")
    (call_dir / "few_shot.jsonl").write_text("", encoding="utf-8")

    registry = RegistrySet(tmp_path)
    prompt_map = registry.prompt_step_map()

    assert "initial_conditions" in prompt_map
    assert prompt_map["initial_conditions"]["call_name"] == "assess_initial_conditions"
    assert "Prompt body" in str(prompt_map["initial_conditions"]["template"])
