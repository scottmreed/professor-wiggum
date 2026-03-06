#!/usr/bin/env python3
"""Capture a clean cropped snapshot of the full mermaid harness diagram.

Generates the mermaid flowchart from the default harness (matching the frontend
app view) and renders it to PNG using @mermaid-js/mermaid-cli.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_mermaid_source() -> str:
    """Build mermaid flowchart source matching the frontend app view."""
    from mechanistic_agent.api.app import build_flow_edges, build_flow_node_specs
    from mechanistic_agent.core.registries import HarnessRegistry

    registry = HarnessRegistry(PROJECT_ROOT / "harness_versions")
    harness = registry.load("default")
    nodes = build_flow_node_specs(harness)
    edges = build_flow_edges(harness)

    # Node styles matching app.js getFlowNodeStyle(kind, "pending")
    FLOW_KIND_FILL = {"llm": "#fff2d8", "deterministic": "#e7f4ea", "decision": "#e8eefb"}
    FLOW_KIND_STROKE = {"llm": "#c26b00", "deterministic": "#2f7d3b", "decision": "#3355aa"}

    lines = ["flowchart TD"]
    for node in nodes:
        nid = node["id"]
        label = node.get("label", nid).replace('"', "'")
        lines.append(f'  {nid}["{label}"]')
    for edge in edges:
        label = edge.get("label", "")
        label_part = f"|{label}|" if label else ""
        lines.append(f'  {edge["source"]} -->{label_part} {edge["target"]}')
    for node in nodes:
        kind = node.get("kind", "deterministic")
        fill = FLOW_KIND_FILL.get(kind, "#e7f4ea")
        stroke = FLOW_KIND_STROKE.get(kind, "#2f7d3b")
        lines.append(f'  style {node["id"]} fill:{fill},stroke:{stroke},stroke-width:1px,opacity:0.9')
    return "\n".join(lines)


def main() -> int:
    out_dir = PROJECT_ROOT / "docs"
    out_dir.mkdir(exist_ok=True)
    mmd_path = out_dir / "harness_flow.mmd"
    png_path = out_dir / "harness_flow_snapshot.png"

    mermaid_src = build_mermaid_source()
    mmd_path.write_text(mermaid_src, encoding="utf-8")

    try:
        subprocess.run(
            ["npx", "-y", "@mermaid-js/mermaid-cli", "-i", str(mmd_path), "-o", str(png_path), "-b", "transparent"],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
    except subprocess.CalledProcessError as e:
        print(f"mermaid-cli failed: {e.stderr or e}", file=sys.stderr)
        return 2
    except FileNotFoundError:
        print("npx not found. Install Node.js and ensure npx is on PATH.", file=sys.stderr)
        return 2

    print(f"Rendered harness diagram to {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
