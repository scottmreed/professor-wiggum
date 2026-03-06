---
skill_type: mechanistic
call_name: evaluate_run_judge
steps: [evaluation_judge]
phase: post_run
kind: llm
tool_schema: judge_result
version: 1
---

# Evaluate Run Judge

LLM judge that scores a completed mechanism run, identifies root-cause failure patterns,
and returns actionable recommendations for prompt or few-shot updates. Called after a
run completes (not during the harness loop).

## Inputs

- `run_progress` — summary of accepted steps, validations, and backtracking events
- `traces` — structured event log from the run
- `comparison_context` — known benchmark mechanism (if available)

## Outputs

- `quality_score` (float 0–1)
- `failure_cascade_root` (string) — most likely root cause of failures
- `harness_recommendations` (string[]) — actionable suggestions
- `prompt_update_targets` (string[]) — which skill prompts to update
- `few_shot_targets` (string[]) — which skill few-shot files to update

## Tool Schema

```json
{
  "type": "function",
  "function": {
    "name": "judge_result",
    "description": "Return scored evaluation with failure analysis and harness recommendations.",
    "parameters": {
      "type": "object",
      "properties": {
        "text": {"type": "string", "description": "Free-form analysis."},
        "quality_score": {"type": "number", "description": "0–1 quality assessment."},
        "failure_cascade_root": {"type": "string", "description": "Most likely root cause of failure chain."},
        "harness_recommendations": {"type": "array", "items": {"type": "string"}},
        "prompt_update_targets": {"type": "array", "items": {"type": "string"}},
        "few_shot_targets": {"type": "array", "items": {"type": "string"}}
      }
    }
  }
}
```

## Prompt

<!-- PROMPT_START -->
Judge the completed run using provided progress, traces, and comparison context.
Score quality, identify likely cascade-root failures, and return actionable harness recommendations.
Focus on concise, evidence-based feedback that can directly guide prompt and few-shot updates.
<!-- PROMPT_END -->

## Notes

- Few-shot examples in `few_shot.jsonl` show representative judgment patterns.
- PRs updating this skill require approved evidence traces at medium eval tier minimum.
