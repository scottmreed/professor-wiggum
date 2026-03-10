#!/usr/bin/env python3
"""
Git repository monitor for tracking changes relevant to manuscript updates.
Extracts training runs, PR contributions, and performance changes.
"""

import subprocess
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import os


class GitManuscriptMonitor:
    """Monitor git repository for manuscript-relevant changes."""

    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).resolve()
        self.last_update_file = self.repo_path / "manuscript" / ".last_manuscript_update"

    def get_recent_commits(self, since_days: int = 30) -> List[Dict]:
        """Get commits since the last manuscript update."""
        since_date = self._get_last_update_date()
        if since_date:
            since_str = since_date.strftime("%Y-%m-%d")
        else:
            since_str = f"{since_days} days ago"

        cmd = [
            "git", "log", "--since", since_str,
            "--pretty=format:%H|%an|%ae|%ad|%s",
            "--date=iso", "--no-merges"
        ]

        try:
            result = subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True)
            result.check_returncode()

            commits = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split('|', 4)
                    if len(parts) >= 5:
                        commit = {
                            'hash': parts[0],
                            'author': parts[1],
                            'email': parts[2],
                            'date': parts[3],
                            'message': parts[4],
                            'manuscript_relevance': self._assess_commit_relevance(parts[4])
                        }
                        commits.append(commit)

            return commits

        except subprocess.CalledProcessError as e:
            print(f"Error getting commits: {e}")
            return []

    def get_pr_contributions(self, since_days: int = 30) -> List[Dict]:
        """Extract PR information and contributor details."""
        # This would integrate with GitHub API for full PR details
        # For now, we'll parse from commit messages and git log

        cmd = [
            "git", "log", "--since", f"{since_days} days ago",
            "--grep", "Merge pull request",
            "--pretty=format:%H|%an|%ae|%ad|%s|%b",
            "--date=iso"
        ]

        prs = []
        try:
            result = subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True)
            result.check_returncode()

            current_pr = None
            for line in result.stdout.strip().split('\n'):
                if "Merge pull request" in line:
                    if current_pr:
                        prs.append(current_pr)

                    parts = line.split('|', 5)
                    if len(parts) >= 6:
                        current_pr = {
                            'merge_commit': parts[0],
                            'merger': parts[1],
                            'date': parts[3],
                            'title': parts[4],
                            'description': parts[5],
                            'contributors': [],
                            'manuscript_impact': self._assess_pr_impact(parts[4], parts[5])
                        }
                elif current_pr and line.strip():
                    # Additional commit information
                    if "Co-authored-by:" in line:
                        author_match = re.search(r'Co-authored-by:\s*([^<]+)', line)
                        if author_match:
                            current_pr['contributors'].append(author_match.group(1).strip())

            if current_pr:
                prs.append(current_pr)

        except subprocess.CalledProcessError as e:
            print(f"Error getting PRs: {e}")

        return prs

    def extract_training_results(self) -> List[Dict]:
        """Extract training run results and performance metrics."""
        results = []

        # Look for result files in various locations
        result_patterns = [
            "local_contributions/runs/*/score_report.json",
            "tests/results/*.json",
            "data/evaluation_results/*.json"
        ]

        for pattern in result_patterns:
            for result_file in self.repo_path.glob(pattern):
                if result_file.stat().st_mtime > self._get_last_update_timestamp():
                    try:
                        with open(result_file, 'r') as f:
                            data = json.load(f)

                        result = {
                            'file': str(result_file),
                            'timestamp': datetime.fromtimestamp(result_file.stat().st_mtime).isoformat(),
                            'metrics': self._extract_metrics(data),
                            'manuscript_relevance': self._assess_result_impact(data)
                        }
                        results.append(result)

                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Error parsing {result_file}: {e}")

        return results

    def get_ralph_loop_progress(self) -> List[Dict]:
        """Extract ralph loop iterations and convergence data."""
        loops = []

        # Look for ralph-related files and commits
        ralph_files = list(self.repo_path.glob("**/ralph*")) + \
                     list(self.repo_path.glob("**/loop*"))

        for ralph_file in ralph_files:
            if ralph_file.is_file():
                try:
                    with open(ralph_file, 'r') as f:
                        content = f.read()

                    # Extract iteration numbers and metrics
                    iterations = re.findall(r'iteration[:\s]+(\d+)', content, re.IGNORECASE)
                    scores = re.findall(r'score[:\s]+([\d.]+)', content, re.IGNORECASE)

                    if iterations or scores:
                        loop_data = {
                            'file': str(ralph_file),
                            'iterations': [int(i) for i in iterations],
                            'scores': [float(s) for s in scores],
                            'convergence': self._check_convergence(scores)
                        }
                        loops.append(loop_data)

                except Exception as e:
                    print(f"Error processing ralph file {ralph_file}: {e}")

        return loops

    def _assess_commit_relevance(self, message: str) -> str:
        """Assess how relevant a commit is to manuscript updates."""
        message_lower = message.lower()

        if any(term in message_lower for term in ['performance', 'accuracy', 'evaluation', 'benchmark']):
            return 'high'
        elif any(term in message_lower for term in ['feature', 'improvement', 'fix', 'update']):
            return 'medium'
        elif any(term in message_lower for term in ['docs', 'readme', 'comment']):
            return 'low'
        else:
            return 'unknown'

    def _assess_pr_impact(self, title: str, description: str) -> str:
        """Assess the manuscript impact of a PR."""
        text = (title + " " + description).lower()

        if any(term in text for term in ['new model', 'architecture', 'methodology', 'significant']):
            return 'high'
        elif any(term in text for term in ['improvement', 'enhancement', 'feature']):
            return 'medium'
        elif any(term in text for term in ['bug fix', 'minor', 'refactor']):
            return 'low'
        else:
            return 'unknown'

    def _extract_metrics(self, data: Dict) -> Dict:
        """Extract key performance metrics from result data."""
        metrics = {}

        # Common metric patterns
        metric_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'score', 'performance']

        def extract_recursive(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    full_path = f"{path}.{key}" if path else key
                    if any(metric in key.lower() for metric in metric_keys):
                        if isinstance(value, (int, float)):
                            metrics[full_path] = value
                    else:
                        extract_recursive(value, full_path)
            elif isinstance(obj, list) and len(obj) > 0:
                # Check first few items for patterns
                for i, item in enumerate(obj[:3]):
                    if isinstance(item, dict):
                        extract_recursive(item, f"{path}[{i}]")

        extract_recursive(data)
        return metrics

    def _assess_result_impact(self, data: Dict) -> str:
        """Assess the manuscript impact of evaluation results."""
        # Look for significant performance indicators
        if 'accuracy' in str(data).lower():
            return 'high'
        elif any(key in data for key in ['leaderboard', 'benchmark', 'evaluation']):
            return 'medium'
        else:
            return 'low'

    def _check_convergence(self, scores: List[str]) -> bool:
        """Check if ralph loop scores show convergence."""
        if len(scores) < 3:
            return False

        try:
            float_scores = [float(s) for s in scores[-5:]]  # Last 5 scores
            if len(float_scores) >= 3:
                # Check if improvement has slowed significantly
                improvements = [float_scores[i+1] - float_scores[i] for i in range(len(float_scores)-1)]
                avg_improvement = sum(improvements) / len(improvements)
                return abs(avg_improvement) < 0.01  # Convergence threshold
        except ValueError:
            pass

        return False

    def _get_last_update_date(self) -> Optional[datetime]:
        """Get the date of the last manuscript update."""
        if self.last_update_file.exists():
            try:
                with open(self.last_update_file, 'r') as f:
                    date_str = f.read().strip()
                    return datetime.fromisoformat(date_str)
            except Exception:
                pass
        return None

    def _get_last_update_timestamp(self) -> float:
        """Get timestamp of last manuscript update."""
        last_date = self._get_last_update_date()
        return last_date.timestamp() if last_date else 0

    def update_last_check(self):
        """Update the timestamp of the last manuscript check."""
        with open(self.last_update_file, 'w') as f:
            f.write(datetime.now().isoformat())


def main():
    """Command-line interface for testing the monitor."""
    monitor = GitManuscriptMonitor()

    print("=== Recent Commits ===")
    commits = monitor.get_recent_commits()
    for commit in commits[:5]:  # Show first 5
        print(f"{commit['date'][:10]}: {commit['message'][:60]}... ({commit['manuscript_relevance']})")

    print("\n=== PR Contributions ===")
    prs = monitor.get_pr_contributions()
    for pr in prs[:3]:  # Show first 3
        print(f"{pr['date'][:10]}: {pr['title'][:50]}... ({pr['manuscript_impact']})")
        if pr['contributors']:
            print(f"  Contributors: {', '.join(pr['contributors'])}")

    print("\n=== Training Results ===")
    results = monitor.extract_training_results()
    for result in results[:3]:
        print(f"{result['timestamp'][:10]}: {result['file']} ({result['manuscript_relevance']})")
        if result['metrics']:
            print(f"  Metrics: {result['metrics']}")

    print("\n=== Ralph Loop Progress ===")
    loops = monitor.get_ralph_loop_progress()
    for loop in loops[:2]:
        print(f"{loop['file']}: {len(loop['iterations'])} iterations, converged: {loop['convergence']}")


if __name__ == '__main__':
    main()

