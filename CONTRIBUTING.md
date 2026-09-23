# Contributing to Mechanistic Agent

Mechanistic is primarily a maintainer- and agent-developed research project. The system improves through an evidence-gated evolution loop (see [SOUL.md](SOUL.md)); you do not need to understand or operate that loop to help.

There are three ways to contribute.

## 1. Report a chemistry problem

Found a reaction the agent gets wrong, or one it gets right that surprised you? Open an issue using the **Chemistry failure** template. Include the starting materials, the products, and what went wrong. A run ID or trace ID is helpful but not required.

Maintainers and automated agents turn these reports into eval cases, evidence traces, few-shot candidates, or code changes. Saying "this reaction failed" is a complete, useful contribution.

## 2. Report a software bug or an idea

Open an issue using the **Bug report** or **Feature request** template.

## 3. Submit code

Pull requests are welcome. Before opening one:

```bash
source .venv/bin/activate
python -m pytest tests/fast/ -q
```

That is the only check you need to run. You do **not** need to run the project's model evaluations, generate evidence traces, or regenerate the leaderboard.

Changes that affect prompts, few-shot examples, models, validators, or harness behavior undergo additional evidence and benchmark validation by the maintainers before merge. That process is documented in [docs/change_evidence_policy.md](docs/change_evidence_policy.md); it is a repository invariant, not a requirement on you.

Setup instructions are in [SETUP.md](SETUP.md). Please follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## professor-wiggum (maintainer fork)

This checkout may use **long-lived integration branches** (for example `curriculum-workflow`) in addition to `main`. Choose the PR **base branch** explicitly on GitHub and say which branch you integrated from. Agent-oriented PR workflow notes live in the tracked skill **[`.claude/skills/pr-creation-helper/SKILL.md`](.claude/skills/pr-creation-helper/SKILL.md)**.

## Attribution

Git and GitHub authorship are the attribution record. If you would like to be acknowledged in future manuscript updates, say so in your issue or PR and include your preferred name and handle (ORCID, GitHub, or email).

---

### Note on the former contribution tracks

Until September 2026 this project used a five-track public contribution model (Tracks 1–5, per-track PR templates under `templates/contributions/`, and a requirement that contributors beat the leaderboard before opening a PR). It was retired because the project's external contribution volume did not justify the overhead, and because the evidence-gating it described is better enforced internally than delegated to contributors. The old templates and guidance remain visible in git history; CHANGELOG entries and older PRs that refer to "Track N" are describing that model. The internal evidence policy that replaced it lives in [docs/change_evidence_policy.md](docs/change_evidence_policy.md).
