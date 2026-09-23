---
name: pr-creation-helper
description: Prepare pull requests for the Mechanistic Agent repo by inspecting the diff, running fast tests, scanning for secrets/PII, deciding whether the change is behavior-changing, and drafting a PR body that satisfies docs/change_evidence_policy.md.
---

# PR Creation Helper for Mechanistic Agent

Use this skill when preparing a pull request or a local PR draft for this repository. The public contribution contract is [CONTRIBUTING.md](../../../CONTRIBUTING.md); the internal merge policy this skill enforces is [docs/change_evidence_policy.md](../../../docs/change_evidence_policy.md).

## Scratch storage

Write PR body previews and push summaries only under `local_contributions/pr_drafts/<slug>.md` (gitignored; never `git add`). Other scratch goes to ignored paths. After writing, confirm `git status` does not list the file.

## Workflow

1. **Inspect the change.** Confirm the repo root with `git rev-parse --show-toplevel`. Use `git status --short` and `git diff` / `git diff --staged` to see what changed.

2. **Classify: is it behavior-changing?** It is if the diff touches any of:
   - `skills/mechanistic/**` (prompts, few-shots, model lanes)
   - `harness_versions/**`
   - `mechanistic_agent/core/validators.py`, `core/subagents.py`, `core/coordinator.py`, `tool_schemas.py`
   - `mechanistic_agent/model_pricing.json`, `mechanistic_agent/llm.py`
   Everything else (bug fixes, infra, eval/leaderboard plumbing, docs) is not, and falls under the Infra Exception in the policy doc.

3. **Secrets/PII scan** before drafting. Grep for `api_key`, `OPENAI_API_KEY`, `sk-`, `authorization: Bearer`, `PRIVATE_KEY`, `BEGIN RSA PRIVATE KEY`, `BEGIN OPENSSH PRIVATE KEY`, and file patterns `.env*`, `*.pem`, `*.key`, `*credentials*`, `*secret*`, `*.pfx`, `*.p12`. `detect-secrets` or `trufflehog` may be used if already installed; do not install tools without permission. Call out any hit, propose redaction, and confirm it is not committed.

4. **Run fast tests.** Always:
   ```bash
   source .venv/bin/activate
   python -m pytest tests/fast/ -q
   ```
   Also run the targeted fast tests for the touched area (`test_model_registry.py`, `test_harness_config.py`, `test_coordination_topology.py`, `test_<subagent>.py`). If tests fail, capture the failing names and fix before drafting a body that claims success.

5. **If behavior-changing, run the gate and the required tier** from the policy doc's merge-bar table:
   - Prompt/few-shot: `PYTHONPATH=. python scripts/validate_prompt_trace_evidence.py --call <call_name>` and the `medium` tier.
   - Subagents/validators: `medium`, `hard` preferred.
   - Models: `easy` tier for the cost class.
   - Harness: `medium`.
   Compare against `LEADERBOARD.md` (or `python main.py leaderboard ...`). If the required tier does not improve, position the change as experimental or local-only rather than as a mergeable PR. If it does, regenerate the leaderboard with the CLI (never by hand).

6. **Draft the PR body** using `.github/pull_request_template.md` headings: Summary, Changes, Validation (exact commands and results), Behavior-changing (tick one; if yes, include evidence paths, eval commands, run IDs or run groups, and the before/after leaderboard delta), Checklist. For agent-bridge runs, include the origin fields (`responder`, `declared_underlying_model`, `budget_observability`, `responder_saw_ground_truth`, `official_holdout_exposed_to_agent`).

7. **Maintainer exceptions.** Follow the policy strictly by default. Deviate only when the user says they are acting as maintainer and the change is an infra exception, isolated maintenance, or an emergency/security hotfix. Never skip fast tests without the user explicitly accepting the risk. Add an "Exception / Maintainer Rationale" section stating which requirements are unmet, why that is acceptable, and what follow-up is planned.

8. **Final checks.** `git status` shows only intended changes; secrets scan clean or addressed; fast tests ran against the current diff; every referenced eval run has its command and run ID or run group; title is concise and names the change type.

## Commit subjects and branch names

Prefer Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`) when it helps reviewers. Branch names should be descriptive (`fix-validator-soft-pass`); no date prefixes are required.

## professor-wiggum: branches, working tree, and "conflicts"

This repo may use long-lived integration branches (for example `curriculum-workflow`) alongside `main`.

- A successful `git push` does not clear `git status`. Modified or untracked files after a push are uncommitted local work, not a failed push and not a merge conflict.
- Merges and fast-forwards update commits, not your working tree. Locally modified files stay modified until you commit, stash, or `git restore`. It is a conflict only if `git status` shows a merge state or files contain `<<<<<<<` markers.
- Choose the PR base branch explicitly on GitHub and say in the description which branch you integrated from.

## Optional manual release-branch merge

There is no cron-driven release automation. `.github/workflows/manual-release-branch-merge.yml` runs only on `workflow_dispatch`: it resolves today's date in America/Denver, looks for `release/YYYY-MM-DD`, merges it into `main` if ahead, and tags `release-YYYY-MM-DD`. Opening a normal PR and merging through GitHub is equally valid. Curriculum publish helpers in `mechanistic_agent/curriculum.py` use their own branch/tag patterns.

## Notes

- Edit this tracked skill when the workflow changes; keep it consistent with `docs/change_evidence_policy.md` and `.github/pull_request_template.md`.
- Prefer existing project scripts and commands over inventing new ones.
- When uncertain whether a change is behavior-changing, treat it as behavior-changing and say so in the PR body.
