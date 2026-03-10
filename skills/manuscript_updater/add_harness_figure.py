#!/usr/bin/env python3
"""
Script to add the harness architecture figure to the manuscript.
Inserts figure reference and ensures the figure exists.
"""

import re
from pathlib import Path
from figure_handler import ManuscriptFigureHandler
from manuscript_integrator import ManuscriptIntegrator


def add_harness_figure_to_manuscript(manuscript_path: str = "manuscript/mechanistic_manuscript.rtf"):
    """Add the harness architecture figure to the manuscript."""

    # Initialize handlers
    figure_handler = ManuscriptFigureHandler(manuscript_path)
    integrator = ManuscriptIntegrator(manuscript_path)

    # Ensure the figure exists
    print("Generating harness architecture figure...")
    updated, message = figure_handler.generate_harness_figure()
    print(message)

    # Read the manuscript
    with open(manuscript_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Check if figure is already referenced
    if 'Figure 1' in content or 'harness architecture' in content.lower():
        print("Harness figure already referenced in manuscript.")
        return True

    # Find the Methods section
    methods_match = re.search(r'\\f1\\b\s*(1\.?\s*)?Methods?\\f0\\b0', content)
    if not methods_match:
        print("Could not find Methods section in manuscript.")
        return False

    methods_start = methods_match.end()

    # Find a good insertion point (after the initial description)
    # Look for the end of the first paragraph in Methods
    next_paragraph = content.find('\\par', methods_start + 100)
    if next_paragraph == -1:
        next_paragraph = methods_start + 500  # Fallback

    # Create figure reference RTF
    figure_rtf = """
\\par
\\pard\\qc\\b Figure 1: Mechanistic Agent Harness Architecture\\b0\\par
\\pard\\qc The harness integrates modular analysis components with LLM-powered proposal generation and deterministic validation. Pre-loop modules perform initial chemical analysis, while the mechanism loop iteratively proposes and validates elementary steps until target products are reached.\\par
\\par
"""

    # Insert the figure reference
    new_content = content[:next_paragraph] + figure_rtf + content[next_paragraph:]

    # Create backup
    integrator.create_backup()

    # Write updated content
    with open(manuscript_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print("Successfully added harness figure reference to manuscript.")
    print("Figure file: manuscript/figures/harness_architecture.png")
    print("Mermaid source: manuscript/figures/harness_architecture.mmd")

    return True


def main():
    """Command-line interface for adding harness figure."""
    import argparse

    parser = argparse.ArgumentParser(description="Add harness figure to manuscript")
    parser.add_argument("--manuscript", default="manuscript/mechanistic_manuscript.rtf",
                       help="Path to manuscript file")

    args = parser.parse_args()

    success = add_harness_figure_to_manuscript(args.manuscript)
    if success:
        print("✓ Harness figure added successfully")
    else:
        print("✗ Failed to add harness figure")


if __name__ == '__main__':
    main()