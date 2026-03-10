#!/usr/bin/env python3
"""
Main workflow for automated manuscript updates based on repository changes,
competition results, and new literature. Now includes figure management.
"""

import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

# Import our tools
from git_monitor import GitManuscriptMonitor
from clawdiator_scraper import ClawdiatorScraper
from literature_monitor import LiteratureMonitor
from manuscript_integrator import ManuscriptIntegrator
from figure_handler import ManuscriptFigureHandler


class ManuscriptUpdateWorkflow:
    """Orchestrate the complete manuscript update process."""

    def __init__(self, manuscript_path: str = "manuscript/mechanistic_manuscript.rtf"):
        self.manuscript_path = manuscript_path
        self.monitor = GitManuscriptMonitor()
        self.scraper = ClawdiatorScraper()
        self.literature = LiteratureMonitor()
        self.integrator = ManuscriptIntegrator(manuscript_path)
        self.figure_handler = ManuscriptFigureHandler(manuscript_path)

        # Update tracking
        self.update_log_path = Path(manuscript_path).parent / "update_log.json"
        self.load_update_log()

    def load_update_log(self):
        """Load the update history log."""
        if self.update_log_path.exists():
            try:
                with open(self.update_log_path, 'r') as f:
                    self.update_log = json.load(f)
            except json.JSONDecodeError:
                self.update_log = []
        else:
            self.update_log = []

    def save_update_log(self):
        """Save the update history log."""
        with open(self.update_log_path, 'w') as f:
            json.dump(self.update_log, f, indent=2, default=str)

    def run_full_update(self, dry_run: bool = True) -> Dict:
        """Run the complete manuscript update workflow."""
        print("Starting manuscript update workflow...")
        print(f"Dry run mode: {dry_run}")

        updates = []
        timestamp = datetime.now().isoformat()

        # 1. Check figure updates first
        print("\n1. Checking figure updates...")
        figure_updates = self._process_figure_updates()
        updates.extend(figure_updates)

        # 2. Monitor repository changes
        print("\n2. Monitoring repository changes...")
        repo_updates = self._process_repository_changes()
        updates.extend(repo_updates)

        # 3. Check competition updates
        print("\n3. Checking competition updates...")
        competition_updates = self._process_competition_updates()
        updates.extend(competition_updates)

        # 4. Monitor literature
        print("\n4. Monitoring new literature...")
        literature_updates = self._process_literature_updates()
        updates.extend(literature_updates)

        # 5. Apply updates to manuscript
        if not dry_run and updates:
            print("\n5. Applying updates to manuscript...")
            applied_updates = self._apply_updates(updates)
        else:
            applied_updates = updates

        # 6. Generate summary
        summary = {
            'timestamp': timestamp,
            'total_updates': len(updates),
            'applied_updates': len(applied_updates) if not dry_run else 0,
            'dry_run': dry_run,
            'updates': applied_updates
        }

        # Log the update
        self.update_log.append(summary)
        self.save_update_log()

        print(f"\nWorkflow completed. {len(updates)} updates identified.")
        if dry_run:
            print("Use --apply to actually update the manuscript.")

        return summary

    def _process_figure_updates(self) -> List[Dict]:
        """Process figure updates and ensure harness figure exists."""
        updates = []

        # Check if harness figure needs updating
        updated, message = self.figure_handler.generate_harness_figure()
        if updated:
            updates.append({
                'type': 'figure',
                'subtype': 'harness_diagram',
                'section': 'Methods',
                'action': 'add_figure',
                'content': 'Add harness architecture diagram reference',
                'justification': 'Updated harness architecture diagram for manuscript',
                'figure_path': 'manuscript/figures/harness_architecture.png'
            })

        # Check for other figure updates needed
        figure_status = self.figure_handler.check_figure_updates_needed()
        for fig_update in figure_status:
            updates.append({
                'type': 'figure',
                'subtype': fig_update['type'],
                'section': 'Methods',
                'action': 'update_figure',
                'content': f"Update figure: {fig_update['figure']}",
                'justification': fig_update['reason']
            })

        return updates

    def _process_repository_changes(self) -> List[Dict]:
        """Process repository changes for manuscript updates."""
        updates = []

        # Get recent commits
        commits = self.monitor.get_recent_commits(since_days=30)
        high_relevance_commits = [c for c in commits if c['manuscript_relevance'] == 'high']

        if high_relevance_commits:
            updates.append({
                'type': 'repository',
                'subtype': 'commits',
                'section': 'Methods',
                'action': 'append',
                'content': self._format_commit_updates(high_relevance_commits),
                'justification': f'Incorporating {len(high_relevance_commits)} significant repository changes',
                'attributions': list(set(c['author'] for c in high_relevance_commits))
            })

        # Get PR contributions
        prs = self.monitor.get_pr_contributions(since_days=30)
        significant_prs = [pr for pr in prs if pr['manuscript_impact'] == 'high']

        if significant_prs:
            for pr in significant_prs:
                updates.append({
                    'type': 'repository',
                    'subtype': 'pr_contribution',
                    'section': 'Acknowledgments',
                    'action': 'append',
                    'content': f"Special thanks to {', '.join(pr['contributors'])} for contributions to {pr['title']}.",
                    'justification': f'Attributing significant PR contribution: {pr["title"]}',
                    'attributions': pr['contributors']
                })

        # Get training results
        training_results = self.monitor.extract_training_results()
        significant_results = [r for r in training_results if r['manuscript_relevance'] == 'high']

        if significant_results:
            latest_result = max(significant_results, key=lambda x: x['timestamp'])
            updates.append({
                'type': 'repository',
                'subtype': 'performance',
                'section': 'Results',
                'action': 'append',
                'content': self._format_performance_updates(latest_result),
                'justification': 'Incorporating latest performance evaluation results'
            })

        # Get ralph loop progress
        ralph_loops = self.monitor.get_ralph_loop_progress()
        converged_loops = [loop for loop in ralph_loops if loop['convergence']]

        if converged_loops:
            updates.append({
                'type': 'repository',
                'subtype': 'ralph_convergence',
                'section': 'Methods',
                'action': 'append',
                'content': f"Recent ralph loop convergence achieved with {len(converged_loops)} optimization cycles completed.",
                'justification': 'Documenting successful optimization convergence'
            })

        return updates

    def _process_competition_updates(self) -> List[Dict]:
        """Process competition updates for manuscript integration."""
        updates = []

        # Get leaderboard data
        leaderboard = self.scraper.get_leaderboard_data()

        if 'entries' in leaderboard and leaderboard['entries']:
            # Find our position (assuming we can identify our entry)
            our_entry = None
            for entry in leaderboard['entries']:
                # This would need customization based on how we identify our submission
                if 'mechanistic' in entry.get('methodology', '').lower():
                    our_entry = entry
                    break

            if our_entry:
                updates.append({
                    'type': 'competition',
                    'subtype': 'leaderboard',
                    'section': 'Results',
                    'action': 'append',
                    'content': f"Current competitive positioning: Rank {our_entry['rank']} with score {our_entry['score']} in {leaderboard.get('competition_name', 'evaluation competition')}.",
                    'justification': 'Updating competitive benchmark results'
                })

        # Get recent competition updates
        recent_updates = self.scraper.get_competition_updates()

        high_impact_updates = [u for u in recent_updates if u.get('impact') == 'high']

        if high_impact_updates:
            for update in high_impact_updates[:2]:  # Limit to 2 most recent
                updates.append({
                    'type': 'competition',
                    'subtype': 'significant_update',
                    'section': 'Discussion',
                    'action': 'append',
                    'content': f"Recent competition development: {update['description']}",
                    'justification': 'Incorporating significant competition landscape changes'
                })

        return updates

    def _process_literature_updates(self) -> List[Dict]:
        """Process new literature for manuscript integration."""
        updates = []

        # Search for recent papers
        recent_papers = self.literature.search_recent_papers(months_back=6)
        high_relevance_papers = [p for p in recent_papers if p.get('manuscript_relevance') == 'high']

        if high_relevance_papers:
            # Generate citation suggestions
            suggestions = self.literature.generate_citation_suggestions(high_relevance_papers[:3])

            for suggestion in suggestions:
                paper = suggestion['paper']
                updates.append({
                    'type': 'literature',
                    'subtype': 'new_paper',
                    'section': suggestion['suggested_sections'][0],  # Use first suggested section
                    'action': 'append',
                    'content': f"{suggestion['citation_text']} {suggestion['discussion_points'][0]}",
                    'justification': f'Incorporating relevant new literature: {paper["title"][:50]}...'
                })

        return updates

    def _apply_updates(self, updates: List[Dict]) -> List[Dict]:
        """Apply updates to the manuscript."""
        applied_updates = []

        for update in updates:
            try:
                # Handle figure updates specially
                if update.get('type') == 'figure':
                    if update.get('subtype') == 'harness_diagram':
                        success = self.figure_handler.add_figure_reference_to_manuscript(
                            'harness_architecture', update.get('section', 'Methods')
                        )
                    else:
                        # Other figure updates would be handled here
                        success = True
                else:
                    # Regular manuscript updates
                    success = self.integrator.update_section(
                        update['section'],
                        update['content'],
                        update.get('action', 'append')
                    )

                if success:
                    applied_updates.append(update)
                    print(f"✓ Applied update to {update['section']}: {update['type']}")
                else:
                    print(f"✗ Failed to apply update to {update['section']}")

            except Exception as e:
                print(f"✗ Error applying update: {e}")

        return applied_updates

    def _format_commit_updates(self, commits: List[Dict]) -> str:
        """Format commit information for manuscript inclusion."""
        if not commits:
            return ""

        content = "\n\nRecent System Improvements:\n"
        for commit in commits[:3]:  # Limit to 3 most recent
            content += f"- {commit['message'][:80]}{'...' if len(commit['message']) > 80 else ''}\n"

        return content

    def _format_performance_updates(self, result: Dict) -> str:
        """Format performance results for manuscript inclusion."""
        content = "\n\nUpdated Performance Metrics:\n"

        if 'metrics' in result and result['metrics']:
            for metric_name, value in list(result['metrics'].items())[:5]:  # Limit to 5 metrics
                if isinstance(value, float):
                    content += f"- {metric_name}: {value:.3f}\n"
                else:
                    content += f"- {metric_name}: {value}\n"

        return content

    def check_update_needed(self) -> Dict:
        """Check if manuscript updates are needed without applying them."""
        print("Checking for needed manuscript updates...")

        updates_needed = {
            'figures': self.figure_handler.check_figure_updates_needed(),
            'repository': [],
            'competition': [],
            'literature': [],
            'total_updates': 0
        }

        # Check repository changes
        commits = self.monitor.get_recent_commits(since_days=7)  # Last week
        if any(c['manuscript_relevance'] == 'high' for c in commits):
            updates_needed['repository'].append('Recent high-impact commits')

        prs = self.monitor.get_pr_contributions(since_days=7)
        if any(pr['manuscript_impact'] == 'high' for pr in prs):
            updates_needed['repository'].append('Recent significant PRs')

        training_results = self.monitor.extract_training_results()
        recent_results = [r for r in training_results if r['manuscript_relevance'] == 'high']
        if recent_results:
            updates_needed['repository'].append('New performance results')

        # Check competition
        leaderboard = self.scraper.get_leaderboard_data()
        if leaderboard.get('entries'):
            updates_needed['competition'].append('Competition leaderboard available')

        # Check literature
        recent_papers = self.literature.search_recent_papers(months_back=1)  # Last month
        if any(p.get('manuscript_relevance') == 'high' for p in recent_papers):
            updates_needed['literature'].append('New relevant literature')

        # Calculate totals
        updates_needed['total_updates'] = (
            len(updates_needed['figures']) +
            len(updates_needed['repository']) +
            len(updates_needed['competition']) +
            len(updates_needed['literature'])
        )

        return updates_needed

    def get_update_history(self) -> List[Dict]:
        """Get the history of manuscript updates."""
        return self.update_log

    def validate_manuscript(self) -> bool:
        """Validate manuscript integrity after updates."""
        validation = self.integrator.validate_manuscript_integrity()
        return all(validation.values())


def main():
    """Command-line interface for the manuscript update workflow."""
    parser = argparse.ArgumentParser(description="Automated manuscript update workflow")
    parser.add_argument("--apply", action="store_true", help="Actually apply updates (default: dry run)")
    parser.add_argument("--manuscript", default="manuscript/mechanistic_manuscript.rtf",
                       help="Path to manuscript file")
    parser.add_argument("--check-updates", action="store_true",
                       help="Check what updates are needed without applying them")
    parser.add_argument("--history", action="store_true", help="Show update history")
    parser.add_argument("--validate", action="store_true", help="Validate manuscript integrity")

    args = parser.parse_args()

    workflow = ManuscriptUpdateWorkflow(args.manuscript)

    if args.validate:
        print("Validating manuscript integrity...")
        is_valid = workflow.validate_manuscript()
        print(f"Manuscript integrity: {'✓ Valid' if is_valid else '✗ Invalid'}")
        return

    if args.check_updates:
        print("Checking for needed updates...")
        updates_needed = workflow.check_update_needed()
        print(f"\nUpdates needed: {updates_needed['total_updates']}")

        for category, items in updates_needed.items():
            if category != 'total_updates' and items:
                print(f"\n{category.upper()}:")
                for item in items:
                    print(f"  - {item}")
        return

    if args.history:
        print("Manuscript update history:")
        history = workflow.get_update_history()
        for i, update in enumerate(history[-5:]):  # Show last 5
            print(f"{i+1}. {update['timestamp'][:19]}: {update['total_updates']} updates")
        return

    # Run the full workflow
    summary = workflow.run_full_update(dry_run=not args.apply)

    # Print summary
    print("\n" + "="*50)
    print("UPDATE SUMMARY")
    print("="*50)
    print(f"Timestamp: {summary['timestamp'][:19]}")
    print(f"Updates identified: {summary['total_updates']}")
    print(f"Updates applied: {summary['applied_updates']}")
    print(f"Dry run: {summary['dry_run']}")

    if summary['updates']:
        print("\nUpdates:")
        for update in summary['updates'][:5]:  # Show first 5
            print(f"  - {update['type']}: {update['section']} ({update['action']})")


if __name__ == '__main__':
    main()