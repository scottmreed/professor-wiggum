#!/usr/bin/env python3
"""One-time migration helper for legacy prompts/*.md -> prompt_versions/."""
from __future__ import annotations

import argparse
from pathlib import Path


def _read(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="Repository root")
    parser.add_argument("--force", action="store_true", help="Overwrite existing call base content")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    legacy_mechanism = repo / "prompts" / "mechanism_synthesis.md"
    legacy_intermediate = repo / "prompts" / "intermediate_prediction.md"
    target = repo / "prompt_versions" / "calls" / "propose_mechanism_step" / "base.md"
    target.parent.mkdir(parents=True, exist_ok=True)

    existing = _read(target)
    if existing and not args.force:
        print(f"Skip: {target.relative_to(repo)} already has content (use --force to overwrite)")
        return 0

    sections = []
    mech_text = _read(legacy_mechanism)
    inter_text = _read(legacy_intermediate)
    if mech_text:
        sections.append("## Migrated from prompts/mechanism_synthesis.md\n\n" + mech_text)
    if inter_text:
        sections.append("## Migrated from prompts/intermediate_prediction.md\n\n" + inter_text)
    if not sections:
        print("No legacy prompt files found; nothing to migrate.")
        return 0

    target.write_text("\n\n".join(sections).strip() + "\n", encoding="utf-8")
    print(f"Wrote migrated content to {target.relative_to(repo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
