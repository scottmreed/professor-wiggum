"""Verification runner: finds the cheapest model per step that matches baseline score.

The bisection process:
1. Run the full eval set with the top model + highest reasoning → baseline score
2. For each LLM step, try models from cheapest to most expensive (with reasoning
   levels) until the score matches the baseline. Keep all OTHER steps at the baseline
   (top model).
3. Record the minimum verified model per step.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from mechanistic_agent.config import LLM_STEP_KEYS
from mechanistic_agent.model_registry import (
    get_family_models,
    get_family_supports_reasoning,
    get_top_family_model,
)

logger = logging.getLogger(__name__)


@dataclass
class StepVerificationResult:
    step_name: str
    verified_model: str
    verified_reasoning: Optional[str]
    score: float


@dataclass
class VerificationResult:
    harness_version: str
    model_family: str
    baseline_score: float
    step_results: List[StepVerificationResult]


class VerificationRunner:
    """Orchestrates model verification by testing each step independently."""

    def __init__(
        self,
        *,
        run_eval_fn: Callable[..., float],
        store: Any,
        harness_version: str,
    ) -> None:
        """
        Args:
            run_eval_fn: Callable that runs the eval set with given step_models
                and step_reasoning dicts, returning an aggregate score (float).
                Signature: run_eval_fn(eval_set_id, step_models, step_reasoning) -> float
            store: RunStore instance for persisting results.
            harness_version: Current harness version string.
        """
        self.run_eval_fn = run_eval_fn
        self.store = store
        self.harness_version = harness_version

    def run_verification(
        self,
        *,
        eval_set_id: str,
        model_family: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> VerificationResult:
        """Run the full verification process for a model family.

        Returns a VerificationResult with the minimum verified model per step.
        """
        family_models = get_family_models(model_family)
        if not family_models:
            raise ValueError(f"No models found for family: {model_family}")

        top_model = get_top_family_model(model_family)
        if not top_model:
            raise ValueError(f"No top model found for family: {model_family}")

        supports_reasoning = get_family_supports_reasoning(model_family)

        # The LLM steps we need to verify
        llm_steps = sorted(LLM_STEP_KEYS)

        # Baseline: all steps use top model + highest reasoning
        baseline_step_models = {step: top_model for step in llm_steps}
        baseline_step_reasoning = (
            {step: "highest" for step in llm_steps} if supports_reasoning else {}
        )

        if progress_callback:
            progress_callback({
                "phase": "baseline",
                "message": f"Running baseline with {top_model}",
                "current_step": 0,
                "total_steps": len(llm_steps) + 1,
            })

        logger.info("Running baseline with model=%s, reasoning=highest", top_model)
        baseline_score = self.run_eval_fn(
            eval_set_id, baseline_step_models, baseline_step_reasoning
        )
        logger.info("Baseline score: %.4f", baseline_score)

        # Build reasoning levels to try (cheapest first)
        reasoning_levels: List[Optional[str]] = [None]
        if supports_reasoning:
            reasoning_levels = ["lowest", "highest"]

        step_results: List[StepVerificationResult] = []

        for step_idx, step_name in enumerate(llm_steps):
            if progress_callback:
                progress_callback({
                    "phase": "bisect",
                    "message": f"Testing step: {step_name}",
                    "current_step": step_idx + 1,
                    "total_steps": len(llm_steps) + 1,
                    "step_name": step_name,
                })

            result = self._find_minimum_model(
                eval_set_id=eval_set_id,
                step_name=step_name,
                family_models=family_models,
                reasoning_levels=reasoning_levels,
                baseline_step_models=baseline_step_models,
                baseline_step_reasoning=baseline_step_reasoning,
                baseline_score=baseline_score,
            )
            step_results.append(result)

            # Store the result immediately
            self.store.upsert_verification_result(
                harness_version=self.harness_version,
                model_family=model_family,
                step_name=step_name,
                verified_model=result.verified_model,
                verified_reasoning=result.verified_reasoning,
                baseline_score=baseline_score,
                step_score=result.score,
                eval_set_id=eval_set_id,
            )

            logger.info(
                "Step %s: verified model=%s, reasoning=%s, score=%.4f",
                step_name, result.verified_model, result.verified_reasoning,
                result.score,
            )

        return VerificationResult(
            harness_version=self.harness_version,
            model_family=model_family,
            baseline_score=baseline_score,
            step_results=step_results,
        )

    def _find_minimum_model(
        self,
        *,
        eval_set_id: str,
        step_name: str,
        family_models: List[str],
        reasoning_levels: List[Optional[str]],
        baseline_step_models: Dict[str, str],
        baseline_step_reasoning: Dict[str, str],
        baseline_score: float,
    ) -> StepVerificationResult:
        """Find the cheapest model+reasoning for a single step that matches baseline."""

        for model in family_models:  # sorted cheapest-first
            for reasoning in reasoning_levels:
                # Override only this step; all others stay at baseline
                test_step_models = dict(baseline_step_models)
                test_step_models[step_name] = model

                test_step_reasoning = dict(baseline_step_reasoning)
                if reasoning:
                    test_step_reasoning[step_name] = reasoning
                else:
                    test_step_reasoning.pop(step_name, None)

                logger.info(
                    "Testing step=%s model=%s reasoning=%s",
                    step_name, model, reasoning,
                )
                score = self.run_eval_fn(
                    eval_set_id, test_step_models, test_step_reasoning
                )

                if score >= baseline_score:
                    return StepVerificationResult(
                        step_name=step_name,
                        verified_model=model,
                        verified_reasoning=reasoning,
                        score=score,
                    )

        # Nothing cheaper matched — use the top model
        top_model = family_models[-1]
        top_reasoning = reasoning_levels[-1] if reasoning_levels else None
        return StepVerificationResult(
            step_name=step_name,
            verified_model=top_model,
            verified_reasoning=top_reasoning,
            score=baseline_score,
        )
