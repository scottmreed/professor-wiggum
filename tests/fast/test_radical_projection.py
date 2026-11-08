from mechanistic_agent.core.arrow_count import step_arrow_count
from mechanistic_agent.core.reaction_focus import build_reaction_focus


def test_pair_arrow_preference_does_not_miscount_fishhooks():
    assert step_arrow_count('[Cl:1][Cl:2]>>[Cl:1].[Cl:2]') is None
    assert step_arrow_count('[CH3:1].[CH3:2]>>[CH3:1][CH3:2]') is None


def test_observatory_reports_radical_changes_separately():
    focus = build_reaction_focus('[Cl:1][Cl:2]>>[Cl:1].[Cl:2]')
    assert focus['changed_radical_electrons'] == ['a1', 'a2']
    assert focus['core_atom_ids'] == ['a1', 'a2']
