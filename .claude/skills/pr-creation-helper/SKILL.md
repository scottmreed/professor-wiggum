---
name: pr-creation-helper
description: Create high-quality pull requests for the Mechanistic Agent repo by analyzing git status, running required tests, scanning for secrets/PII, and drafting PR descriptions that follow contribution templates while allowing carefully justified admin exceptions when appropriate.
---

# PR Creation Helper for Mechanistic Agent

## Purpose and Scope

Use this skill when preparing a pull request or local PR draft for the Mechanistic Agent repository.

This skill helps the agent:

- Inspect the current git status and diffs.
- Infer which contribution track(s) apply based on the changes.
- Run the required tests for the relevant track(s).
- Optionally run evals when the user requests or when needed for a leaderboard-claiming PR.
- Scan for accidental secrets, API keys, or PII before drafting.
- Generate a PR description using the appropriate template from `templates/contributions/`.
- For users with admin/maintainer privileges, carefully apply exceptions to the usual gates (e.g., infra-only changes) while documenting the rationale.

This repo keeps a **shared** copy at `.claude/skills/pr-creation-helper/SKILL.md` (tracked in git). For Cursor-wide reuse, you may mirror it to `~/.cursor/skills/pr-creation-helper/SKILL.md`.

## Storage and Non-Version-Control Rule

- **PR / push markdown previews** (draft bodies, “paste this into GitHub”, command summaries): write under **`local_contributions/pr_drafts/<slug>.md`**, where `<slug>` is a short descriptive name (for example `curriculum-workflow-updates.md`). A **date prefix** like `2026-05-05-<slug>.md` is optional for your own sorting; it is **not** required. That tree is **gitignored**—**never** `git add` these files. If a preview was created elsewhere, do not commit it; move or recreate under `pr_drafts/`. After writing, confirm `git status` does not list the file.
- Other scratch notes: prefer ignored paths (`/tmp`, or repo paths listed in `.gitignore`), not tracked docs.
- Do **not** put the skill’s canonical copy only under `.cursor/` inside the repo; use `.claude/skills/` as above.

## High-Level Workflow

When using this skill to help with a PR, follow this sequence:

1. **Identify context and track**
   - Confirm the repo root is `/Users/scottreed/PycharmProjects/professor-wiggum` (or detect via `git rev-parse --show-toplevel`).
   - Read `CONTRIBUTING.md` as needed, especially the **Core rule**, **Track definitions**, and **Required Baseline Checks** sections.
   - Use `git status --short` and (when helpful) `git diff` / `git diff --staged` to understand which files changed.
   - Infer which track(s) the changes map to:
     - Track 1 – few-shot examples under `skills/mechanistic/.../few_shot.jsonl`
     - Track 2 – new/changed subagents, validators, schemas, coordinator wiring
     - Track 3 – model catalog entries or adapters
     - Track 4 – harness configuration / topology in `harness_versions/`
     - Track 5 – single reaction submissions or local-only artifacts
   - If multiple tracks apply, treat the PR as the **strictest** combination (e.g., Track 2 + Track 4 → run both sets of tests/evals that make sense).

2. **Baseline safety and hygiene checks**
   - **Secrets/PII scan** before writing the PR description:
     - Use ripgrep or similar to look for likely secrets and keys:
       - Terms: `api_key`, `OPENAI_API_KEY`, `sk-`, `authorization: Bearer`, `PRIVATE_KEY`, `BEGIN RSA PRIVATE KEY`, `BEGIN OPENSSH PRIVATE KEY`.
       - File patterns: `.env`, `.env.*`, `*.pem`, `*.key`, `*credentials*`, `*secret*`, `*.pfx`, `*.p12`.
       - Known provider prefixes (OpenAI, Anthropic, Stripe, GitHub, etc.) if relevant.
     - If `detect-secrets` or `trufflehog` is already installed in the environment, you may run:
       - `detect-secrets scan` (or scoped to the repo) or
       - `trufflehog filesystem . --fail --only-verified`
       - Treat these as **optional** helpers; do not install new global tools without user permission.
     - For any suspicious matches:
       - Point them out explicitly.
       - Propose redaction or relocation to environment variables / secrets management.
       - Confirm they are **not** added to commits before proceeding.

3. **Run required tests (and evals when appropriate)**
   - Always ensure the Python virtual environment is active first:
     - `source .venv/bin/activate`
   - **Baseline fast tests** (for Tracks 1–4, and usually for infra exceptions too):
     - `python -m pytest tests/fast/ -q`
   - Additional per-track checks (only when the user intends to actually push a leaderboard PR, not for local dry-runs unless requested):
     - **Track 1 (few-shot)**:
       - `PYTHONPATH=. python scripts/validate_prompt_trace_evidence.py --call <call_name>`
       - `PYTHONPATH=. pytest tests/llm/test_eval_tiers.py --tier medium -k medium`
     - **Track 2 (subagents)**:
       - `python -m pytest tests/fast/ -q`
       - Relevant new fast tests, e.g. `PYTHONPATH=. pytest tests/fast/test_<subagent_name>.py`
       - `PYTHONPATH=. pytest tests/llm/test_eval_tiers.py --tier medium -k medium`
       - Optionally `--tier hard -k hard` if the user wants more evidence.
     - **Track 3 (models)**:
       - `PYTHONPATH=. pytest tests/fast/test_model_registry.py`
       - `PYTHONPATH=. pytest tests/llm/test_eval_tiers.py --tier easy -k easy`
     - **Track 4 (harness)**:
       - `PYTHONPATH=. pytest tests/fast/test_harness_config.py`
       - `PYTHONPATH=. pytest tests/fast/test_coordination_topology.py`
       - `PYTHONPATH=. pytest tests/llm/test_eval_tiers.py --tier medium -k medium`
   - When tests fail:
     - Capture the failing test name(s) and error summaries.
     - Help the user fix issues **before** drafting a final PR body that claims success.

4. **Evaluate leaderboard / eval runs when claiming improvements**
   - For any PR that **claims leaderboard improvement**, help the user:
     - Identify the relevant eval set and tier from `CONTRIBUTING.md` and `training_data/` docs.
     - Run the required eval tier(s) for the chosen model/harness using `python main.py eval ...` or `python main.py baseline ...` as described in `CONTRIBUTING.md`.
     - Compare results against the current leaderboard (usually via `python main.py leaderboard ...` or `python main.py leaderboard-official`).
     - If the change **does not** improve the required tier:
       - Recommend positioning the change as experimental or local-only, not as a mergeable leaderboard PR.

5. **Draft the PR body using repo templates**
   - Choose the correct base template from `templates/contributions/`:
     - Track 1 → `track1_few_shot_pr.md`
     - Track 2 → `track2_subagent_pr.md`
     - Track 3 → `track3_model_pr.md`
     - Track 4 → `track4_harness_pr.md`
     - Track 5 → `track5_single_reaction_submission.md` (usually used for local drafts under `local_contributions/`).
   - When drafting a PR:
     - Preserve the template’s **section headings**, checklists, and emphasis on leaderboard deltas.
     - Fill in:
       - Summary of changes (one short paragraph).
       - Motivation and context.
       - Exact commands used for tests and evals, with their results.
       - Before/after leaderboard deltas where applicable.
       - Any new files or artifacts (skills, harness variants, traces, etc.).
     - For any **markdown preview** of the PR body (or push summary), write **`local_contributions/pr_drafts/<slug>.md` only** (optional date in the filename if you want)—gitignored; do not stage or commit.

6. **Admin/maintainer-only deviations ("breaking the rules carefully")**

By default, **obey the contribution rules strictly**. Only deviate when:

- The user explicitly indicates they are acting in an admin/maintainer role **and**
- The deviation falls under one of the following narrow categories:
  - **Infra exception**: changes to eval/leaderboard workflow infrastructure that:
    - Do not claim a harness/prompt/model quality improvement.
    - Preserve existing track gates for actual quality-affecting changes.
    - Include targeted tests for the new infra behavior.
  - **Non-eval-impacting maintenance**: pure documentation, comments, or small refactors where the admin explicitly accepts the risk and the scope is clearly isolated.
  - **Emergency/security hotfix**: time-sensitive fixes where running full eval tiers is impractical, but fast tests and local validation still pass.

When applying an admin exception:

- **Never skip fast tests** unless the user explicitly overrules this and understands the risk; even then, recommend running at least a minimal subset.
- Add a dedicated **"Exception / Admin Rationale"** section in the PR body that:
  - States exactly which usual requirements are not met (e.g., missing eval tier run, incomplete leaderboard regeneration).
  - Justifies why this is acceptable for this PR (e.g., infra-only, no leaderboard claim, emergency security fix).
  - Clarifies whether follow-up work is planned (e.g., "We will run the full medium-tier eval and update the leaderboard in a subsequent PR").
- Avoid overusing this escape hatch; remind the user that the project’s philosophy is evidence-backed evolution.

7. **Final sanity checks before presenting the PR draft**

Before you present the draft PR description to the user, ensure:

- `git status` is clean or clearly shows only intentional changes.
- Secrets/PII scans are clean, or any issues are explicitly called out and addressed.
- Fast tests have been run recently enough that the results are still valid for the current diff.
- Any eval runs referenced in the PR body have:
  - Exact commands listed.
  - Run IDs or result summaries.
  - Clear before/after comparisons when claiming improvements.
- The PR title:
  - Is concise and descriptive.
  - Matches the main change type (few-shot, harness, model, subagent, infra).

## Commit subjects and branch names

- Prefer **Conventional Commits** style when it helps reviewers (`feat:`, `fix:`, `docs:`, `infra:`, `chore:`, etc.). **Do not** require date-stamped commit subjects (no obligation to prefix commits or branches with calendar dates).
- Branch names should be **descriptive** (e.g. `curriculum-workflow`, `fix-validator-soft-pass`). Avoid coupling branch names to a fixed release clock unless you deliberately choose that convention for a one-off merge workflow.

## Optional manual release-branch merge (GitHub Actions)

There is **no** cron-driven “5pm release” automation in this repo.

If maintainers use **`.github/workflows/manual-release-branch-merge.yml`**, it runs only on **`workflow_dispatch`**. At run time it resolves **today’s date in America/Denver** and looks for a remote branch named **`release/YYYY-MM-DD`** with that date, then merges it into `main` if it is ahead. Tags created by that workflow use the form **`release-YYYY-MM-DD`** (no time-of-day suffix).

- This is **optional** and **manual**; opening a normal PR and merging through GitHub is equally valid.
- Curriculum publish/git helpers use their **own** branch/tag patterns (see `mechanistic_agent/curriculum.py`); they are not tied to this workflow’s `release/…` naming.

## Usage Patterns

### 1. Full PR Prep (standard track)

When the user says something like:

- "Help me open a PR for these changes."
- "Draft a PR for this new harness variant."
- "Turn this model experiment into a PR."

You should:

1. Confirm the repo root and run `git status`.
2. Classify the changes into contribution tracks.
3. Run fast tests and any relevant additional tests/evals.
4. Perform a secrets/PII scan.
5. Synthesize a PR title and body using the appropriate template, filling in:
   - Summary, motivation, design, risks.
   - Test commands + results.
   - Eval commands + leaderboard deltas, if applicable.
6. Present the draft PR body in chat; if a file helps, write **`local_contributions/pr_drafts/<slug>.md`** only (ignored—never commit).

### 2. Local-Only Dry Run

When the user wants to practice or sandbox a PR:

1. Follow the same analysis steps (git status, tests, secrets scan).
2. Draft the PR body using the usual template.
3. Save it under **`local_contributions/pr_drafts/<slug>.md`** (gitignored; never commit).
4. Do not save PR previews under tracked paths (e.g. repo root, `docs/`).

### 3. Admin-Mode Infra or Emergency PR

When the user explicitly says they are acting in an admin/maintainer role and wants to relax certain gates:

1. Still run fast tests whenever possible.
2. Run a secrets scan regardless.
3. Draft the PR body with:
   - Clear scoping of the change.
   - Explicit note that no leaderboard claim is made (if true).
   - An **"Exception / Admin Rationale"** section documenting:
     - Which usual rules are being relaxed.
     - Why this is safe and necessary.
4. Encourage follow-up work to restore normal gates (e.g., post-hoc evals or leaderboard updates).

## Notes and Constraints

- Edit the tracked skill at `.claude/skills/pr-creation-helper/SKILL.md` when improving repo workflow; keep a personal mirror under `~/.cursor/skills/` only if you want the same rules outside this repo.
- **Never commit** PR preview markdown from `local_contributions/pr_drafts/` (directory is gitignored).
- Prefer existing project scripts and commands over inventing new ones.
- When uncertain about a track, err on the side of **stricter** requirements and transparency in the PR body.
- Keep this skill’s logic up to date with any future changes to `CONTRIBUTING.md` and templates by re-reading them when behavior appears mismatched.

