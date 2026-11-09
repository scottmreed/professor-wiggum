---
skill_type: mechanistic
call_name: metal_center_validation
phase: post_step
kind: deterministic
implementation: validator.py
version: 1
---

# Check Metal-Centre Bookkeeping

For a step that changes a species containing a d-block metal, counts the metal before
and after with the covalent MLX model and checks that the change fits an elementary
organometallic step. Advisory by default. No LLM call.

## Algorithm

1. Collect the d-block-containing species in `current_state` and `resulting_state`
   (multisets, atom maps ignored). If none changed, return no check, so steps that
   leave the metal alone validate exactly as before.
2. Pair the consumed and formed species per metal. If a metal does not have exactly
   one species consumed and one formed, abstain.
3. Convert each side to house convention (`organometallic_chem.notation.to_house_convention`:
   an over-valent neutral N/P/As/O/S/Se donor on an ordinary bond to the metal becomes
   a dative L bond, `[M]C#O` becomes `[O+]#[C-]->[M]`). For an ambiguous drawing
   (η2-alkene vs metallacyclopropane, carbene carbon, bridging donor, more than one metal),
   abstain.
4. Count with `organometallic_chem.organometallic_electron_count.count_electrons`:
   oxidation state (OS), d count, valence electrons (VE), coordination number.
5. Findings:
   - `impossible_oxidation_state` (error): OS above the element's maximum (Pd/Ni 4,
     Pt 6, Cu/Ag 3, Au 5, Zn/Cd/Hg 2, otherwise the group number), below -2, or a
     negative d count.
   - `over_18_electrons`: error for groups 8–10, warning otherwise.
   - `declared_kind_mismatch`: the proposal's `metal_step_kind` disagrees with the
     observed (ΔOS, ΔVE) per `STEP_DELTAS`. A warning under `soft`, an error under `strict`.
   - `no_matching_step_kind` (warning): the observed deltas match no single elementary step.
   - `high_coordination_number` (warning): CN above 6.
6. The check fails only on an error. Warnings stay in `details.warnings` with
   `warning_only = true`.

## Inputs

- `current_state`, `resulting_state` (SMILES[]): covalent or house-convention drawings.
- `declared_kind` (optional): the candidate's `metal_step_kind`, one of
  `oxidative_addition`, `reductive_elimination`, `transmetalation`, `migratory_insertion`,
  `beta_hydride_elimination`, `ligand_association`, `ligand_dissociation`, `other`.
- `policy`: `RunConfig.metal_center_policy`, one of `off` | `soft` (default) | `strict`.
  A harness may set it with `run_config_defaults.metal_center_policy`.

## Outputs (`details`, schema `metal_center.v1`)

- `pairs[]`: `metal`, `before` and `after` (`smiles`, `house_smiles`, `oxidation_state`,
  `d_electrons`, `valence_electrons`, `coordination_number`, `mlx`, `summary`),
  `delta_oxidation_state`, `delta_valence_electrons`, `consistent_kinds`.
- `findings[]`: `{code, severity, message}`. Messages are chemistry sentences that are
  reused in reproposal feedback.
- `abstained`: the reason, or null.
- `policy`, `declared_kind`, plus `error` and `warnings` when present.

The coordinator also emits each result as a `metal_center_check` event and attaches it to
`candidate_validation_result.metal_center`. `/observatory` projects it onto the accepted path.

## Evidence

On FlowER train Pd mechanisms (ground-truth paths), all 340 metal-changing steps pass. Each
matches exactly one elementary step kind (121 oxidative additions, 121 reductive
eliminations, 105 transmetalations), with no warnings and no abstentions.
