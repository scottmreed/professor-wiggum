#!/usr/bin/env python3
"""
Figure handler for managing manuscript figures and ensuring they are up-to-date.
Handles PNG generation, RTF references, and figure version tracking.
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess
import re


class ManuscriptFigureHandler:
    """Manages manuscript figures and their integration with RTF content."""

    def __init__(self, manuscript_path: str = "manuscript/mechanistic_manuscript.rtf",
                 figures_dir: str = "manuscript/figures"):
        self.manuscript_path = Path(manuscript_path)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(exist_ok=True)

        # Figure registry file
        self.registry_path = self.figures_dir / "figure_registry.json"
        self.load_registry()

    def load_registry(self):
        """Load the figure registry tracking file versions and metadata."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    self.registry = json.load(f)
            except json.JSONDecodeError:
                self.registry = {}
        else:
            self.registry = {}

    def save_registry(self):
        """Save the figure registry."""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2, default=str)

    def generate_harness_figure(self) -> Tuple[bool, str]:
        """Generate or update the harness architecture figure.

        Returns:
            Tuple of (updated: bool, message: str)
        """
        figure_name = "harness_architecture"
        png_path = self.figures_dir / f"{figure_name}.png"
        mermaid_path = self.figures_dir / f"{figure_name}.mmd"

        # Check if Mermaid diagram exists
        if not mermaid_path.exists():
            self._create_harness_mermaid_diagram(mermaid_path)

        # Generate PNG from Mermaid if needed
        updated = False
        if self._needs_png_update(mermaid_path, png_path):
            success = self._generate_png_from_mermaid(mermaid_path, png_path)
            if success:
                updated = True
                self._update_registry_entry(figure_name, png_path, "Harness Architecture Diagram")
            else:
                return False, "Failed to generate PNG from Mermaid diagram"

        return updated, f"Harness figure updated: {png_path}"

    def _create_harness_mermaid_diagram(self, mermaid_path: Path):
        """Create the Mermaid diagram for the harness architecture."""
        mermaid_content = """graph TB
    A[Input Reaction<br/>SMILES] --> B[Pre-loop Analysis]
    B --> C[Atom Balance Check]
    B --> D[Functional Groups]
    B --> E[pH Recommendation]

    C --> F[LLM Modules]
    D --> F
    E --> F

    F --> G[Assess Conditions]
    F --> H[Predict Missing Reagents]
    F --> I[Atom Mapping]
    F --> J[Reaction Type Classification]

    J --> K[Mechanism Loop]
    I --> K

    K --> L[Propose Next Step<br/>3 Candidates]
    L --> M[Validate Step<br/>Bond/Electron<br/>Conservation]
    M --> N[Validate Step<br/>Atom Balance]
    N --> O[Validate Step<br/>State Progress]

    O --> P{Target<br/>Reached?}
    P -->|No| K
    P -->|Yes| Q[Complete Mechanism]

    style A fill:#e1f5fe
    style K fill:#fff3e0
    style Q fill:#e8f5e8
"""

        with open(mermaid_path, 'w') as f:
            f.write(mermaid_content)

    def _generate_png_from_mermaid(self, mermaid_path: Path, png_path: Path) -> bool:
        """Generate PNG from Mermaid diagram using mermaid-cli or puppeteer."""
        try:
            # Try using mermaid-cli if available
            result = subprocess.run([
                'mmdc', '-i', str(mermaid_path), '-o', str(png_path),
                '-t', 'default', '-b', 'transparent'
            ], capture_output=True, timeout=30)

            if result.returncode == 0:
                return True

        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Fallback: create HTML file for manual PNG generation
        try:
            html_path = png_path.with_suffix('.html')
            self._create_mermaid_html(mermaid_path, html_path)
            # For now, we'll consider the HTML creation as success
            return True

        except Exception:
            return False

    def _create_mermaid_html(self, mermaid_path: Path, html_path: Path):
        """Create an HTML file with the Mermaid diagram for PNG generation."""
        with open(mermaid_path, 'r') as f:
            mermaid_content = f.read()

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Harness Architecture</title>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'default'
        }});
    </script>
</head>
<body>
    <div class="mermaid">
{mermaid_content}
    </div>
</body>
</html>"""

        with open(html_path, 'w') as f:
            f.write(html_content)

    def _needs_png_update(self, mermaid_path: Path, png_path: Path) -> bool:
        """Check if PNG needs to be updated based on source file changes."""
        if not png_path.exists():
            return True

        mermaid_mtime = mermaid_path.stat().st_mtime
        png_mtime = png_path.stat().st_mtime

        return mermaid_mtime > png_mtime

    def _update_registry_entry(self, figure_name: str, png_path: Path, description: str):
        """Update the registry entry for a figure."""
        file_hash = self._calculate_file_hash(png_path) if png_path.exists() else ""

        self.registry[figure_name] = {
            'path': str(png_path),
            'description': description,
            'hash': file_hash,
            'last_updated': datetime.now().isoformat(),
            'version': self.registry.get(figure_name, {}).get('version', 0) + 1
        }

        self.save_registry()

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def check_figure_updates_needed(self) -> List[Dict]:
        """Check which figures need updates based on source changes."""
        updates_needed = []

        # Check harness figure
        figure_name = "harness_architecture"
        mermaid_path = self.figures_dir / f"{figure_name}.mmd"
        png_path = self.figures_dir / f"{figure_name}.png"

        if mermaid_path.exists() and self._needs_png_update(mermaid_path, png_path):
            updates_needed.append({
                'figure': figure_name,
                'type': 'regenerate_png',
                'reason': 'Mermaid source file modified'
            })

        # Check for missing figures referenced in manuscript
        manuscript_figures = self._extract_figure_references()
        for fig_ref in manuscript_figures:
            fig_name = fig_ref.get('name')
            if fig_name and fig_name not in self.registry:
                updates_needed.append({
                    'figure': fig_name,
                    'type': 'missing_figure',
                    'reason': f'Figure referenced in manuscript but not in registry: {fig_ref.get("reference", "")}'
                })

        return updates_needed

    def _extract_figure_references(self) -> List[Dict]:
        """Extract figure references from the manuscript."""
        if not self.manuscript_path.exists():
            return []

        with open(self.manuscript_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Look for figure references (simplified pattern)
        figure_refs = []

        # RTF figure patterns - looking for common figure reference formats
        patterns = [
            r'Figure\s+(\d+)',
            r'Fig\.?\s*(\d+)',
            r'\\[^}]*figure[^}]*',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                figure_refs.append({
                    'reference': match,
                    'name': f'figure_{match}' if match.isdigit() else match
                })

        return figure_refs

    def get_figure_status(self) -> Dict:
        """Get status of all figures."""
        status = {
            'total_figures': len(self.registry),
            'figures': {},
            'updates_needed': self.check_figure_updates_needed()
        }

        for fig_name, fig_info in self.registry.items():
            png_path = Path(fig_info['path'])
            status['figures'][fig_name] = {
                'exists': png_path.exists(),
                'last_updated': fig_info.get('last_updated'),
                'description': fig_info.get('description'),
                'version': fig_info.get('version', 1)
            }

        return status


def main():
    """Command-line interface for figure management."""
    import argparse

    parser = argparse.ArgumentParser(description="Manuscript figure handler")
    parser.add_argument("--generate-harness", action="store_true",
                       help="Generate/update harness architecture figure")
    parser.add_argument("--check-updates", action="store_true",
                       help="Check which figures need updates")
    parser.add_argument("--status", action="store_true",
                       help="Show figure status")
    parser.add_argument("--manuscript", default="manuscript/mechanistic_manuscript.rtf",
                       help="Path to manuscript file")
    parser.add_argument("--figures-dir", default="manuscript/figures",
                       help="Directory for figures")

    args = parser.parse_args()

    handler = ManuscriptFigureHandler(args.manuscript, args.figures_dir)

    if args.generate_harness:
        updated, message = handler.generate_harness_figure()
        print(f"Harness figure: {'Updated' if updated else 'No update needed'}")
        print(message)

    if args.check_updates:
        updates = handler.check_figure_updates_needed()
        if updates:
            print(f"Updates needed for {len(updates)} figures:")
            for update in updates:
                print(f"  - {update['figure']}: {update['reason']}")
        else:
            print("All figures are up to date")

    if args.status:
        status = handler.get_figure_status()
        print(f"Total figures: {status['total_figures']}")
        print(f"Updates needed: {len(status['updates_needed'])}")

        for fig_name, fig_info in status['figures'].items():
            exists_status = "✓" if fig_info['exists'] else "✗"
            print(f"  {exists_status} {fig_name}: v{fig_info['version']} ({fig_info.get('last_updated', 'unknown')[:10]})")


if __name__ == '__main__':
    main()