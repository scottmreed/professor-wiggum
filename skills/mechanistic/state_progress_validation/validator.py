"""State progress validator.

Ground truth implementation for the state_progress_validation mechanistic skill.
Verifies that a proposed mechanism step actually transforms the state: the resulting
species differ from the current species, and the starting materials have not simply
been returned unchanged.

Harness-specific overrides go in:
  harness_versions/<harness>/patches/state_progress_validation.py
"""
from __future__ import annotations

from typing import Any, Dict, List

from mechanistic_agent.core.types import StepValidationCheck


def validate_state_progress(
    payload: Dict[str, Any],
) -> StepValidationCheck:
    """Check that the mechanism step makes progress.

    Parameters
    ----------
    payload:
        Step output dict expected to contain:
        - ``unchanged_starting_materials_detected`` (bool)
        - ``resulting_state_changed`` (bool)
    """
    unchanged = bool(payload.get("unchanged_starting_materials_detected"))
    resulting_changed = bool(payload.get("resulting_state_changed"))
    return StepValidationCheck(
        name="state_progress",
        passed=(not unchanged and resulting_changed),
        details={
            "unchanged_starting_materials_detected": unchanged,
            "resulting_state_changed": resulting_changed,
        },
    )
