# Model Asset Overrides

Exact-model trainee lanes can override shared mechanistic prompt assets under:

- `skills/mechanistic/base_system/models/<model-slug>/SKILL.md`
- `skills/mechanistic/<call_name>/models/<model-slug>/SKILL.md`
- `skills/mechanistic/<call_name>/models/<model-slug>/few_shot.jsonl`

Model slugs replace `/` with `__`. For Claude Opus the slug is:

- `anthropic__claude-opus-4.6`

## Resolution policy

Prompt files:

1. exact-model override if present
2. shared mechanistic asset

Few-shot examples:

1. exact-model override examples first
2. shared examples as fallback

New mined examples for an exact-model trainee lane are written into that model's override path so they do not contaminate shared assets.
