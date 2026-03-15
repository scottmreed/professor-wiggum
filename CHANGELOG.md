# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- (None)

### Changed
- (None)

### Fixed
- (None)

---

## [0.2.0] - 2026-03-14

### Added
- **Baseline evaluation support**: Harness-free baseline runs via CLI (`main.py baseline --all-tiers`), tier-based eval set mapping (`training_data/baseline_tier_eval_set_map.json`, `training_data/baseline_tiers_clawdiator.json`), and documentation in AGENTS.md and CONTRIBUTING.md.
- **API**: New endpoints for example progress and chemistry backend configuration; improved handling of chemistry backend options.
- **Reaction type taxonomy**: Reaction type templates and expanded reaction map training data (`reaction_type_templates.json`, `rxn_map_expanded.json`, `eval_mechanism_map.json`); catalog expanded from 51 to 53 entries.
- **Scoring and utilities**: Species normalization and comparison helpers in `smiles_utils.py`; overall balance metrics in evaluation scoring (`scoring.py`).
- **Leaderboard and curriculum**: Auto-generation of leaderboard tables and refresh of curriculum artifacts (LEADERBOARD.md); updated curriculum calendars and generated leaderboard JSON.
- **Clawdiators**: Submission structure and scoring system; clawdiators-test evaluation scaffolding.
- **Documentation**: Mechanism move notation (`docs/mechanism_move_notation.md`), Cursor remote bootstrap (`docs/cursor_remote_bootstrap.md`), Ralph overnight notes; custom eval sets guide (`docs/custom_eval_sets.md`).
- **Scripts and tests**: New local contribution scripts (e.g. `scripts/sync_examples.py`); new tests for baseline tier CLI, chemistry backend adapter, reaction type templates, rxn map expanded, and related training data.

### Changed
- **AGENTS.md**: Clarified verified vs unverified modes and where each is available (web UI vs CLI/API).
- **README.md**: Pre-loop analysis process and outputs; running baseline evaluations without the API server; rdkit_cli backend setup.
- **CONTRIBUTING.md**: Instructions for harness-free baseline runs and tier management.
- **.gitignore**: Additional training data files and traces for baseline evaluations; reaction type taxonomy and derived reaction map artifacts.
- **Harness**: Updates to `harness_versions/default` and `harness_versions/adaptive_default`; harness and manuscript alignment.
- **SETUP.md**: Baseline evaluation support and setup clarifications.
- **Skills**: Updates to base_system, predict_missing_reagents, and propose_mechanism_step (SKILL.md and few_shot.jsonl); atom_balance_validation validator.

### Fixed
- Reaction type matching for `alkyl_halide` (adjusted `consume` parameter in `tools.py` for correct matching behavior).
