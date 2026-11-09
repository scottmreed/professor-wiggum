"""Metal-bound donor atoms drawn with covalent M-L bonds are not radicals.

FlowER (and most LLM output) draws a phosphine on Pd as an ordinary single
bond to a neutral P: ``[Pd]([P](C)(C)C)``. RDKit reads that 4-bonded neutral P
as carrying one radical electron, which routed ordinary two-electron Pd steps
(ligand dissociation, oxidative addition, transmetalation) to strict radical
validation, and that path rejects every metal atom. 63 of 290 ground-truth
FlowER Pd steps lost arrow credit this way.
"""
from mechanistic_agent.core.mechanism_moves import normalize_electron_pushes
from mechanistic_agent.core.radical_validation import (
    radical_species_changed,
    requires_radical_validation,
    state_has_radicals,
)
from mechanistic_agent.quality_scoring import QualityStep, _bond_electron_valid

PD_BIS_PHOSPHINE = (
    "[Cl:1][Pd:2]([Cl:3])([P:4]([CH3:5])([CH3:6])[CH3:7])[P:8]([CH3:9])([CH3:10])[CH3:11]"
)
PD_MONO_PHOSPHINE = "[Cl:1][Pd:2]([Cl:3])[P:4]([CH3:5])([CH3:6])[CH3:7]"
FREE_PHOSPHINE = "[P:8]([CH3:9])([CH3:10])[CH3:11]"
DISSOCIATION_SMIRKS = f"{PD_BIS_PHOSPHINE}>>{PD_MONO_PHOSPHINE}.{FREE_PHOSPHINE}"
DISSOCIATION_PUSHES = [
    {"kind": "sigma_bond", "source_bond": ["2", "8"], "through_atom": "8",
     "target_atom": "8", "electrons": 2},
]


def test_covalent_phosphine_on_pd_is_not_a_radical_state():
    assert not state_has_radicals([PD_BIS_PHOSPHINE])
    assert not state_has_radicals([PD_MONO_PHOSPHINE])


def test_phosphine_dissociation_keeps_the_pair_arrow_path():
    states = [[PD_BIS_PHOSPHINE], [PD_MONO_PHOSPHINE, FREE_PHOSPHINE]]
    assert not radical_species_changed(states, DISSOCIATION_SMIRKS)
    moves = normalize_electron_pushes(DISSOCIATION_PUSHES)
    assert not requires_radical_validation(moves, states=states, reaction_smirks=DISSOCIATION_SMIRKS)


def test_phosphine_dissociation_scores_as_a_valid_pair_arrow_step():
    step = QualityStep(
        step_index=1,
        current_state=[PD_BIS_PHOSPHINE],
        resulting_state=[PD_MONO_PHOSPHINE, FREE_PHOSPHINE],
        reaction_smirks=DISSOCIATION_SMIRKS,
        electron_pushes=DISSOCIATION_PUSHES,
    )
    valid, error = _bond_electron_valid(step)
    assert valid, error


def test_organic_radical_next_to_a_metal_complex_still_counts():
    # The exemption covers only atoms bonded to the metal; a methyl radical in
    # the same state is still an open-shell species.
    assert state_has_radicals([PD_BIS_PHOSPHINE, "[CH3]"])
    assert radical_species_changed([[PD_BIS_PHOSPHINE, "[CH3]"], [PD_BIS_PHOSPHINE]])


def test_radical_elsewhere_in_a_metal_bound_ligand_still_counts():
    # A carbon radical two bonds from Pd is not the donor atom.
    assert state_has_radicals(["[Cl][Pd][P](C)(C)[CH2]"])


def test_explicit_fishhook_on_a_metal_step_still_selects_radical_path():
    moves = normalize_electron_pushes(
        [{"kind": "radical", "source_atom": "8", "target_atom": "2", "electrons": 1}]
    )
    states = [[PD_BIS_PHOSPHINE], [PD_MONO_PHOSPHINE, FREE_PHOSPHINE]]
    assert requires_radical_validation(moves, states=states, reaction_smirks=DISSOCIATION_SMIRKS)
