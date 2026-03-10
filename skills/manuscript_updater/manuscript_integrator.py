#!/usr/bin/env python3
"""
Manuscript integration tool for updating RTF content with new information
while preserving formatting and maintaining academic standards.
"""

import re
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import shutil
import os


class ManuscriptIntegrator:
    """Integrate updates into the mechanistic manuscript RTF file."""

    def __init__(self, manuscript_path: str = "manuscript/mechanistic_manuscript.rtf"):
        self.manuscript_path = Path(manuscript_path)
        self.backup_dir = self.manuscript_path.parent / "backups"
        self.backup_dir.mkdir(exist_ok=True)

    def create_backup(self) -> str:
        """Create a timestamped backup of the current manuscript."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"manuscript_backup_{timestamp}.rtf"

        if self.manuscript_path.exists():
            shutil.copy2(self.manuscript_path, backup_path)
            return str(backup_path)

        return ""

    def read_manuscript(self) -> str:
        """Read the current manuscript content."""
        if not self.manuscript_path.exists():
            return ""

        with open(self.manuscript_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    def write_manuscript(self, content: str):
        """Write updated content to the manuscript file."""
        # Create backup first
        self.create_backup()

        with open(self.manuscript_path, 'w', encoding='utf-8') as f:
            f.write(content)

    def identify_section_boundaries(self, content: str) -> Dict[str, Tuple[int, int]]:
        """Identify the start and end positions of major manuscript sections."""
        sections = {}

        # RTF section markers (looking for bold section headers)
        section_patterns = {
            'Abstract': r'\\f1\\b\s*Abstract\\f0\\b0',
            'Introduction': r'\\f1\\b\s*(1\.?\s*)?Introduction\\f0\\b0',
            'Methods': r'\\f1\\b\s*(1\.1\.?\s*)?Methods?\\f0\\b0',
            'Results': r'\\f1\\b\s*(2\.?\s*)?Results?\\f0\\b0',
            'Discussion': r'\\f1\\b\s*(3\.?\s*)?Discussion\\f0\\b0',
            'Related Work': r'\\f1\\b\s*(4\.?\s*)?(Related Work|Literature Review)\\f0\\b0',
            'Conclusions': r'\\f1\\b\s*(5\.?\s*)?Conclusions?\\f0\\b0',
            'Acknowledgments': r'\\f1\\b\s*Acknowledgments?\\f0\\b0',
            'References': r'\\f1\\b\s*References?\\f0\\b0'
        }

        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, content)
            if match:
                start_pos = match.start()

                # Find next section or end of document
                next_matches = []
                for other_pattern in section_patterns.values():
                    if other_pattern != pattern:
                        other_match = re.search(other_pattern, content[start_pos + 1:])
                        if other_match:
                            next_matches.append(start_pos + 1 + other_match.start())

                end_pos = min(next_matches) if next_matches else len(content)
                sections[section_name] = (start_pos, end_pos)

        return sections

    def update_section(self, section_name: str, new_content: str, action: str = "append") -> bool:
        """Update a specific section of the manuscript."""
        content = self.read_manuscript()
        sections = self.identify_section_boundaries(content)

        if section_name not in sections:
            print(f"Section '{section_name}' not found in manuscript")
            return False

        start_pos, end_pos = sections[section_name]
        section_content = content[start_pos:end_pos]

        if action == "append":
            # Find a good insertion point (before references or end of section)
            insert_pos = self._find_append_position(section_content, start_pos)
            updated_content = (
                content[:insert_pos] +
                self._format_rtf_text(new_content) +
                content[insert_pos:]
            )

        elif action == "replace":
            # Replace entire section content but keep header
            header_end = self._find_header_end(section_content, start_pos)
            updated_content = (
                content[:header_end] +
                self._format_rtf_text(new_content) +
                content[end_pos:]
            )

        elif action == "prepend":
            # Add before section content
            header_end = self._find_header_end(section_content, start_pos)
            updated_content = (
                content[:header_end] +
                self._format_rtf_text(new_content) +
                content[header_end:]
            )

        else:
            print(f"Unknown action: {action}")
            return False

        self.write_manuscript(updated_content)
        return True

    def add_attribution(self, contributor_name: str, contribution_type: str = "general"):
        """Add contributor attribution to acknowledgments section."""
        attribution_text = f", {contributor_name}"

        if contribution_type == "technical":
            attribution_text = f" and {contributor_name} for technical contributions"
        elif contribution_type == "methodological":
            attribution_text = f" and {contributor_name} for methodological insights"

        # This would need more sophisticated parsing of acknowledgments
        # For now, we'll append to the acknowledgments section
        return self.update_section("Acknowledgments", attribution_text, "append")

    def add_literature_citation(self, citation_text: str, section: str = "Related Work"):
        """Add a new literature citation to the specified section."""
        citation_block = f"\n\n{citation_text}"

        return self.update_section(section, citation_block, "append")

    def update_performance_metrics(self, new_metrics: Dict, section: str = "Results"):
        """Update performance metrics in the results section."""
        metrics_text = "\n\nUpdated Performance Metrics:\n"

        for metric_name, value in new_metrics.items():
            if isinstance(value, float):
                metrics_text += f"- {metric_name}: {value:.3f}\n"
            else:
                metrics_text += f"- {metric_name}: {value}\n"

        return self.update_section(section, metrics_text, "append")

    def add_competition_results(self, competition_data: Dict, section: str = "Results"):
        """Add competition benchmark results."""
        competition_text = f"\n\nCompetition Results ({competition_data.get('competition_name', 'Unknown')}):\n"
        competition_text += f"- Rank: {competition_data.get('rank', 'N/A')}\n"
        competition_text += f"- Score: {competition_data.get('score', 'N/A')}\n"
        competition_text += f"- Methodology: {competition_data.get('methodology', 'N/A')}\n"

        if 'date' in competition_data:
            competition_text += f"- Date: {competition_data['date']}\n"

        return self.update_section(section, competition_text, "append")

    def remove_obsolete_content(self, obsolete_patterns: List[str]):
        """Remove obsolete or outdated content based on patterns."""
        content = self.read_manuscript()
        original_content = content

        for pattern in obsolete_patterns:
            # Use word boundaries and be careful with RTF codes
            rtf_pattern = r'\s*' + re.escape(pattern) + r'\s*(?:\n|\\par\s*)+'
            content = re.sub(rtf_pattern, '', content, flags=re.IGNORECASE)

        if content != original_content:
            self.write_manuscript(content)
            return True

        return False

    def _find_append_position(self, section_content: str, section_start: int) -> int:
        """Find the best position to append new content in a section."""
        # Look for paragraph breaks or section endings
        # Prefer to insert before references or conclusions

        # Try to find position before "References" or similar
        ref_patterns = [r'\\f1\\b\s*References', r'\\par\s*$', r'\n\s*$']

        for pattern in ref_patterns:
            match = re.search(pattern, section_content)
            if match:
                return section_start + match.start()

        # Default to end of section
        return section_start + len(section_content)

    def _find_header_end(self, section_content: str, section_start: int) -> int:
        """Find the end of the section header."""
        # Look for the end of the bold formatting after the section title
        header_pattern = r'\\f0\\b0\s*'
        match = re.search(header_pattern, section_content)
        if match:
            return section_start + match.end()

        # Fallback: look for first paragraph break
        para_match = re.search(r'\\par\s*', section_content)
        if para_match:
            return section_start + para_match.end()

        return section_start + len(section_content[:200])  # First 200 chars

    def _format_rtf_text(self, text: str) -> str:
        """Format plain text for RTF insertion."""
        # Basic RTF formatting - this is simplified
        # In practice, you'd want more sophisticated RTF parsing

        # Escape special characters
        text = text.replace('\\', '\\\\')
        text = text.replace('{', '\\{')
        text = text.replace('}', '\\}')
        text = text.replace('\n', '\\par\n')

        # Wrap in RTF formatting
        rtf_text = f"\\f0\\b0 \\cf0 {text}"

        return rtf_text

    def validate_manuscript_integrity(self) -> Dict[str, bool]:
        """Validate that the manuscript structure is intact after updates."""
        content = self.read_manuscript()
        validation_results = {
            'has_rtf_header': content.startswith('{\\rtf1'),
            'has_rtf_footer': content.endswith('}'),
            'sections_present': len(self.identify_section_boundaries(content)) > 0,
            'readable': len(content) > 1000  # Basic length check
        }

        return validation_results

    def generate_update_summary(self, updates: List[Dict]) -> str:
        """Generate a summary of all manuscript updates."""
        summary = f"Manuscript Update Summary - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"

        for update in updates:
            summary += f"Section: {update.get('section', 'Unknown')}\n"
            summary += f"Action: {update.get('action', 'Unknown')}\n"
            summary += f"Type: {update.get('type', 'Unknown')}\n"

            if 'justification' in update:
                summary += f"Justification: {update['justification']}\n"

            if 'attributions' in update:
                summary += f"Attributions: {', '.join(update['attributions'])}\n"

            summary += "---\n"

        return summary


def main():
    """Command-line interface for testing the manuscript integrator."""
    integrator = ManuscriptIntegrator()

    print("Reading manuscript...")
    content = integrator.read_manuscript()
    print(f"Manuscript length: {len(content)} characters")

    print("\nIdentifying sections...")
    sections = integrator.identify_section_boundaries(content)
    print(f"Found sections: {list(sections.keys())}")

    print("\nValidating manuscript integrity...")
    validation = integrator.validate_manuscript_integrity()
    print(f"Validation results: {validation}")

    # Example updates (commented out to avoid accidental modifications)
    """
    print("\nAdding example attribution...")
    integrator.add_attribution("Dr. Jane Smith", "technical")

    print("Adding example literature citation...")
    citation = "Smith, J., & Johnson, A. (2024). Advances in mechanistic chemistry AI. Journal of Chemical AI, 12(3), 145-167."
    integrator.add_literature_citation(citation, "Related Work")

    print("Adding example performance metrics...")
    metrics = {
        "accuracy": 0.895,
        "precision": 0.912,
        "recall": 0.878
    }
    integrator.update_performance_metrics(metrics)
    """

    print("\nManuscript integration test completed successfully!")


if __name__ == '__main__':
    main()

