"""Run-scoped scratchpad for accumulating mechanism history.

The scratchpad is an ephemeral markdown file written to
``traces/runs/<run_id>/scratchpad.md``.  It records accepted steps,
validation retries, backtracks, and failed paths so that the coordinator
can selectively inject condensed history into LLM prompts without
bloating context windows.

Key design property: the LLM proposing a single step does NOT receive
the full scratchpad.  It only sees a short summary injected by the
coordinator — enough to avoid repeating failed approaches, but not so
much that it becomes confused by branch/backtrack mechanics it doesn't
need to reason about.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from mechanistic_agent.prompt_assets import traces_root


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def scratchpad_path(base_dir: Path, run_id: str) -> Path:
    """Return the canonical scratchpad path for a run."""
    return traces_root(base_dir) / "runs" / run_id / "scratchpad.md"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Write helpers — called by the coordinator at key events.
# All accept Optional[Path] for base_dir; when None the call is a no-op
# (happens in tests with mock stores that lack a db_path).
# ---------------------------------------------------------------------------

def init_scratchpad(
    base_dir: Optional[Path],
    run_id: str,
    starting_materials: List[str],
    products: List[str],
) -> None:
    """Create (or overwrite) the scratchpad header for a new run."""
    if base_dir is None:
        return
    path = scratchpad_path(base_dir, run_id)
    _ensure_parent(path)
    lines = [
        f"# Run scratchpad — {run_id}\n",
        f"Starting materials: {', '.join(starting_materials)}\n",
        f"Products: {', '.join(products)}\n",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def append_step_accepted(
    base_dir: Optional[Path],
    run_id: str,
    step_index: int,
    intermediate_smiles: Optional[str],
    resulting_state: List[str],
    candidate_rank: int,
) -> None:
    """Append a record of an accepted mechanism step."""
    if base_dir is None:
        return
    path = scratchpad_path(base_dir, run_id)
    _ensure_parent(path)
    block = (
        f"\n## Step {step_index} (accepted, rank {candidate_rank})\n"
        f"- Intermediate: {intermediate_smiles or 'n/a'}\n"
        f"- Resulting state: {', '.join(resulting_state)}\n"
    )
    with path.open("a", encoding="utf-8") as fh:
        fh.write(block)


def append_validation_retry(
    base_dir: Optional[Path],
    run_id: str,
    step_index: int,
    retry_index: int,
    failed_checks: List[str],
    hint: str = "",
) -> None:
    """Append a record of a validation retry at a given step."""
    if base_dir is None:
        return
    path = scratchpad_path(base_dir, run_id)
    _ensure_parent(path)
    checks_str = ", ".join(failed_checks) if failed_checks else "unknown"
    block = (
        f"\n### Retry {retry_index} at step {step_index}\n"
        f"- Failed checks: {checks_str}\n"
    )
    if hint:
        block += f"- Hint: {hint}\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(block)


def append_backtrack(
    base_dir: Optional[Path],
    run_id: str,
    reverted_to_step: int,
    alternative_rank: int,
    failed_path_steps: int,
    failure_reason: str = "",
) -> None:
    """Append a record of a backtrack event."""
    if base_dir is None:
        return
    path = scratchpad_path(base_dir, run_id)
    _ensure_parent(path)
    block = (
        f"\n## Backtrack\n"
        f"- Reverted to step {reverted_to_step}, trying alternative rank {alternative_rank}\n"
        f"- Failed path length: {failed_path_steps} steps\n"
    )
    if failure_reason:
        block += f"- Failure reason: {failure_reason}\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(block)


def append_failed_path(
    base_dir: Optional[Path],
    run_id: str,
    branch_step_index: int,
    candidate_rank: int,
    steps_taken: int,
    reason: str = "",
) -> None:
    """Append a condensed summary of a failed exploration path."""
    if base_dir is None:
        return
    path = scratchpad_path(base_dir, run_id)
    _ensure_parent(path)
    block = (
        f"\n### Failed path at branch step {branch_step_index}\n"
        f"- Candidate rank: {candidate_rank}\n"
        f"- Steps taken: {steps_taken}\n"
    )
    if reason:
        block += f"- Reason: {reason}\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(block)


# ---------------------------------------------------------------------------
# Read helpers — produce a condensed summary for LLM injection
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(r"^##\s+", re.MULTILINE)


def read_scratchpad_summary(
    base_dir: Optional[Path],
    run_id: str,
    *,
    max_chars: int = 2000,
) -> str:
    """Return a condensed scratchpad summary suitable for LLM context.

    Keeps the last few retry/backtrack/failed-path sections and truncates
    to *max_chars*.  Returns empty string if scratchpad doesn't exist or
    is empty.
    """
    if base_dir is None:
        return ""
    path = scratchpad_path(base_dir, run_id)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return ""

    # Split into top-level sections (## headings)
    sections = _SECTION_RE.split(text)
    # sections[0] is the header block; the rest are individual sections.
    header = sections[0].strip()

    # Keep all Backtrack / Failed path sections (important context) and
    # only the last 3 Step sections to avoid bloat.
    backtrack_sections: List[str] = []
    step_sections: List[str] = []
    retry_sections: List[str] = []

    for sec in sections[1:]:
        sec_stripped = sec.strip()
        if sec_stripped.startswith("Backtrack"):
            backtrack_sections.append(f"## {sec_stripped}")
        elif sec_stripped.startswith("Step"):
            step_sections.append(f"## {sec_stripped}")
        else:
            # Retry, Failed path, or other
            retry_sections.append(f"## {sec_stripped}")

    # Keep last 3 steps, all backtracks, last 3 retries/failed paths
    kept_steps = step_sections[-3:]
    kept_retries = retry_sections[-3:]

    parts = [header]
    parts.extend(kept_steps)
    parts.extend(backtrack_sections)
    parts.extend(kept_retries)

    summary = "\n\n".join(parts)
    if len(summary) > max_chars:
        summary = summary[:max_chars - 20] + "\n...(truncated)"
    return summary
