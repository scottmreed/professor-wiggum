# Eval Tiers: Where Each Tier Comes From

This note answers, for `python main.py eval --tier {easy,medium,hard} ...`: which file
actually supplies the case IDs for a tier today, what happens when the resolved list
is empty, and what changed in the PR that populated
[`training_data/eval_tiers.json`](../training_data/eval_tiers.json)'s `medium`/`hard`
lists (PRD §19 Phase 0: "Populate `eval_tiers.json` medium and hard ... so acceptance
criteria are runnable").

## Two different CLI code paths, two different tier-resolution mechanisms

`main.py` has **two independent tier-resolution code paths** that must not be
confused:

1. **`python main.py eval --tier <t> ...`** (single tier, no `--all-tiers`) goes
   through the **development leaderboard route planner**
   (`_build_development_leaderboard_status` / `_resolve_development_tier_contexts`
   in `main.py`), documented in
   [development_leaderboard_routes.md](development_leaderboard_routes.md). This is
   also the path used by `--leaderboard-status-only`.
2. **`python main.py eval --all-tiers ...`** and **`python main.py baseline
   --all-tiers ...`** go through `_build_baseline_tier_execution_plan`, which reads
   a single tier-definitions file resolved by `_resolve_baseline_tier_definitions_path`
   (`training_data/baseline_tiers_clawdiator.json` if present, else
   `training_data/eval_tiers.json`) with **no route planner involved**.

### Path 1: `eval --tier <t>` (route planner)

The route planner treats two files as "synchronized views over the same
development mechanism pool" (its own docs' wording):

- [`training_data/eval_tiers.json`](../training_data/eval_tiers.json)
- [`training_data/baseline_tiers_clawdiator.json`](../training_data/baseline_tiers_clawdiator.json)

[`training_data/development_leaderboard_policy.json`](../training_data/development_leaderboard_policy.json)
picks, **per tier**, which file is "active" via `active_tier_sources`. As of this PR
the policy still says:

```json
"active_tier_sources": {
  "easy": "eval_tiers",
  "medium": "baseline_tiers_clawdiator",
  "hard": "baseline_tiers_clawdiator"
}
```

So today, concretely:

| Tier | Active source file | Case IDs |
| --- | --- | --- |
| `easy` | `eval_tiers.json` | 100 IDs (all of `eval_set.json`, 1-2 step FlowER defaults) |
| `medium` | `baseline_tiers_clawdiator.json` | 20 IDs (3-step) |
| `hard` | `baseline_tiers_clawdiator.json` | 60 IDs (4-6 step) |

The actual eval-set rows to run against come from a **third**, separate file,
[`training_data/baseline_tier_eval_set_map.json`](../training_data/baseline_tier_eval_set_map.json),
which maps each tier name to an `eval_set_id` already imported into the DB (see
`main.py import-eval-set`). The active tier-definitions file only supplies the
**subset of case IDs within that eval set** to run (`_resolve_development_tier_contexts`
intersects the tier's ID list against `resolved_eval_set.cases`).

This means: **`eval_tiers.json` having `medium: []` did not, by itself, break
`eval --tier medium` before this PR**, because the policy already routed `medium`
away from `eval_tiers.json` to `baseline_tiers_clawdiator.json` (which was already
populated with 20 IDs). `docs/development_leaderboard_routes.md` said as much:
"`medium` and `hard` remain Clawdiator-backed until the eval-facing tiers are
prepared." This PR is that preparation step — see "What changed" below.

### Path 2: `eval --all-tiers` / `baseline --all-tiers`

`_resolve_baseline_tier_definitions_path` picks **one** file for all three tiers
(no per-tier `active_tier_sources`): `baseline_tiers_clawdiator.json` if it exists,
else `eval_tiers.json`. Since `baseline_tiers_clawdiator.json` exists and already had
non-empty `medium`/`hard`, this path was also never actually exposed to
`eval_tiers.json`'s empty lists in practice.

## What happens when a tier list is empty?

**Before this PR**, behavior differed by path:

- **Path 2** (`--all-tiers`): `_build_baseline_tier_execution_plan` already raised
  `typer.BadParameter(f"Tier '{tier_name}' has 0 cases; sync eval_tiers.json or tier
  mapping")` if the resolved+intersected ID list for a tier came back empty. This
  path was already loud.
- **Path 1** (`eval --tier <t>`, the route planner): **no such check existed.**
  - The "seed" route (first run for a model+thinking scope) *did* raise if its
    tier's cases were empty (`"Development leaderboard seed route has 0 available
    cases for tier '<t>'"`) — but only for the **policy's `initial_qualifying.tier`**
    (`easy` by default), not for whatever tier the caller actually requested with
    `--tier`.
  - The "next" route silently **omitted itself** from the offered routes when the
    next tier's case list was empty (`if next_case_ids: routes.append(...)`), with
    no error.
  - Concretely: if a completed canonical `easy` row already existed for a
    model+thinking scope, and you ran `python main.py eval --tier medium
    --leaderboard-status-only` while `medium`'s *active* source resolved to 0 cases,
    the planner would print status and silently recommend route `same` — **re-running
    the `easy` tier** — with no indication that your `--tier medium` request was not
    actually honored. `--leaderboard-status-only` returned 0 (success) either way.

This is the gap PRD §19 Phase 0 flags ("does it silently run zero cases?"): the
honest answer for Path 1 was "worse — it can silently run the *wrong* tier with no
error," not merely "it runs zero cases."

**After this PR**, `eval_cmd` in `main.py` checks
`status["requested_tier_has_cases"]` (a field `_build_development_leaderboard_status`
already computed but never consulted) immediately after building the status, before
printing status or selecting a route:

```python
if not status.get("requested_tier_has_cases"):
    raise typer.BadParameter(
        f"Tier '{requested_tier}' resolves to 0 cases via its active source "
        f"'{source_name}' ({source_path}). Populate that tier's case list ..."
    )
```

This fires for **both** `--leaderboard-status-only` and a real run, before any
harness eval executes, and names the active source file so the fix is obvious. See
`tests/fast/test_baseline_tier_cli.py::test_eval_tier_status_only_fails_loudly_when_resolved_tier_is_empty`
and `::test_eval_tier_run_fails_loudly_when_resolved_tier_is_empty`.

## What changed: populating `eval_tiers.json` medium/hard

### How `easy` was actually chosen (the existing convention)

`easy` is **not** a 10-case sample. Despite `AGENTS.md`'s description of eval tiers
as "fixed 10 easy + 10 medium + 10 hard," the actual `eval_tiers.json` in this repo
has `easy` = **all 100** IDs of `training_data/eval_set.json`, in ranked order
(`_meta.ordering: "IDs remain in ranked order from eval_set.json"`). `initial_qualifying.case_count`
in `development_leaderboard_policy.json` (currently `10`) is what actually produces a
10-case *slice* at run time — the tier list itself is not pre-truncated to 10.

### Selection rule used for `medium`/`hard`

Given that, the analogous, already-established rule (not an invented one) is to use
**all** of the deterministic per-step-count sample already selected for the
`baseline_tiers_clawdiator.json` medium/hard tiers, copied verbatim:

- `medium` = the 20 IDs in `baseline_tiers_clawdiator.json["medium"]` (all have
  `n_mechanistic_steps == 3`).
- `hard` = the 60 IDs in `baseline_tiers_clawdiator.json["hard"]` (20 each of
  `n_mechanistic_steps` 4, 5, 6).

These IDs are themselves a **deterministic, documented** stratified sample: they are
a contiguous slice of `training_data/flower_mechanisms_multistep_report.json`'s
`selected_case_ids` (itself produced by `scripts/build_flower_mechanism_dataset.py`'s
stratified mode: `selection_policy: "stratified: per_step lowest-ranked conversions
per step-count tier"`, `per_step_requested: 20`, `max_step: 6`). No new sampling was
performed for this PR — copying an already-established selection keeps
`eval_tiers.json` and `baseline_tiers_clawdiator.json` synchronized, which is exactly
what `development_leaderboard_routes.md` says they are meant to be. This is why
`medium`/`hard` end up at 20/60 in this PR rather than a capped 10/10: it matches how
`easy` was actually built (all in-band cases, not a capped sample).

`eval_tiers.json`'s `_meta` records this provenance (`medium_hard_provenance`,
`holdout_disjointness` keys).

### Do the chosen IDs resolve to importable records?

Yes. `training_data/flower_mechanisms_multistep.json` (120 records: 20 each at
1-6 steps) has the same schema as `training_data/eval_set.json`
(`id`/`starting_materials`/`products`/`verified_mechanism`/...), which `main.py
import-eval-set --path <file>` imports generically (it only requires `id`,
`starting_materials`, `products`). `training_data/baseline_tier_eval_set_map.json`'s
`medium`/`hard` entries already reference `eval_set_id`s imported into the DB from
that file — this PR does not need to (and does not) re-import anything; it only
aligns `eval_tiers.json`'s ID lists with IDs that already resolve successfully via
that existing eval-set/tier-map wiring.

`training_data/flower_mechanisms_multistep.json` and its `_report.json` are
**gitignored, generated, local-only artifacts** (`training_data/*` is gitignored
except an explicit `!training_data/<file>` allow-list in `.gitignore`, and neither
file is on that list). They were present in the maintainer's working checkout used
to produce `baseline_tiers_clawdiator.json` but are not guaranteed to exist in every
checkout or in CI. `tests/fast/test_eval_tiers_structure.py` reflects this:
step-count/ID-membership checks against the multistep file are marked
`_skip_no_multistep` and skip cleanly when the file is absent; the
always-available invariant — `eval_tiers.json`'s `medium`/`hard` must equal
`baseline_tiers_clawdiator.json`'s `medium`/`hard` exactly — runs unconditionally and
does not depend on that file being present.

### Holdout disjointness

Checked directly against `../wiggum-data/training_data/leaderboard_holdout/eval_set_holdout.json`
(98 cases, resolved via `mechanistic_agent/data_paths.py::holdout_eval_set_path`,
present in this environment's sibling data checkout): **zero overlap** with the
`medium`/`hard` IDs (or `easy`). This holds structurally, not just for the specific
IDs chosen: the holdout is generated from FlowER's **test** split and uses IDs
prefixed `flower_test_...`, while `eval_set.json` / `flower_mechanisms_multistep.json`
are generated from the **train** split and use IDs prefixed `flower_...` (no
`_test_`). `tests/fast/test_eval_tiers_structure.py::test_medium_and_hard_ids_disjoint_from_holdout`
checks the ID-prefix invariant unconditionally (no holdout file required in CI);
the direct set-intersection check above was a one-time manual verification recorded
here since the holdout file is not committed.

## Net effect of this PR

- `eval_tiers.json` now has non-empty `easy` (100) / `medium` (20) / `hard` (60).
- `development_leaderboard_policy.json`'s `active_tier_sources` is **unchanged**
  (`medium`/`hard` still resolve via `baseline_tiers_clawdiator.json`) — this PR is
  infra/data-prep, not a routing change, and flipping `active_tier_sources` is a
  behavior decision left to the maintainer. Because `eval_tiers.json`'s `medium`/`hard`
  now equal `baseline_tiers_clawdiator.json`'s exactly, flipping the policy later
  would be a no-op for which case IDs get selected.
- `eval --tier <t>` and `--leaderboard-status-only` now fail loudly, before any
  harness run, if the *requested* tier's active source resolves to 0 cases —
  regardless of which file that source is.
