# Literature contributions (staging)

These are reaction cases taken from published papers. ChemIllusion's **Mechanisms from the literature** feature proposes them, and a person approves each one before it reaches this directory.

**How cases get here**
1. A ChemIllusion user analyses a paper.
2. The user runs Wiggum on one of the reactions it finds.
3. The user opts in to contribute that case.
4. Scott gets an email with a one-click approval link.
5. Approving opens a pull request here that adds one file: `<yyyymmdd>_<id>.json`.

ChemIllusion never merges these PRs. They go through normal review.

**Format.** Each file is a [custom eval set](../../docs/custom_eval_sets.md) array of cases:
- required: `id`, `starting_materials`, `products`;
- optional: `name`, `temperature_celsius`, `notes`, `tags`.

`notes` carry the DOI, the scheme locator and the ChemIllusion run id. A predicted mechanism is never stored as `verified_mechanism`, because nobody has checked it as a chemist.

**What these cases are not**
- They are not part of `eval_set.json`, `eval_tiers.json` or the leaderboard holdout.
- Nothing loads them automatically. The server's example menu reads only top-level `training_data/*.json`.
- To run one:

  ```bash
  python main.py import-eval-set --path training_data/literature_contributions/<file>.json --name <name>
  python main.py eval --eval-set-id <id> ...
  ```

  Add `--hide-products` to the eval to check that Wiggum predicts the paper's product.
- Promoting a case into a tier is a separate, evidence-gated change. See `docs/change_evidence_policy.md`.

`2026-10-09_chemillusion_demos.json` holds the three cases behind ChemIllusion's first public literature examples. See the PR that added it for the runs.
