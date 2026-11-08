import pytest
from mechanistic_agent.core.mechanism_moves import (
    MechanismMoveFormatError, normalize_electron_pushes, parse_mechanism_moves,
    serialize_mechanism_moves, synthesize_dbe_entries,
)


def test_fishhook_roundtrip_and_counts():
    notation = 'rad:1>2;pi:2-3>3:e1;et:4>5'
    moves = parse_mechanism_moves(notation)
    assert [m.electrons for m in moves] == [1, 1, 1]
    assert serialize_mechanism_moves(moves) == notation
    assert normalize_electron_pushes([m.as_dict() for m in moves]) == moves
    assert synthesize_dbe_entries([m.as_dict() for m in moves])


@pytest.mark.parametrize('push', [
    {'kind': 'radical', 'source_atom': '1', 'target_atom': '2', 'electrons': 2},
    {'kind': 'lone_pair', 'source_atom': '1', 'target_atom': '2', 'electrons': 0},
    {'kind': 'mystery', 'start_atom': '1', 'end_atom': '2'},
    {'notation': 'rad:1>2', 'electrons': 2},
    {'notation': 'rad:1>2', 'source_atom': '3'},
])
def test_invalid_push_is_not_silently_dropped(push):
    with pytest.raises(MechanismMoveFormatError):
        normalize_electron_pushes([push])


CASES = [
    ('[Cl:1][Cl:2]>>[Cl:1].[Cl:2]', 'sigma:1-2>2:e1;sigma:2-1>1:e1'),
    ('[CH3:1].[CH3:2]>>[CH3:1][CH3:2]', 'rad:1>2;rad:2>1'),
    ('[CH3:1].[H:2][CH3:3]>>[CH3:1][H:2].[CH3:3]', 'rad:1>2;sigma:3-2>1:e1;sigma:2-3>3:e1'),
    ('[CH3:1].[CH2:2]=[CH2:3]>>[CH3:1][CH2:2][CH2:3]', 'rad:1>2;pi:3-2>1:e1;pi:2-3>3:e1'),
    ('[CH3:1].[Cl:2]>>[CH3+:1].[Cl-:2]', 'et:1>2'),
]


@pytest.mark.parametrize('reaction,notation', CASES)
def test_radical_inventories_match(reaction, notation):
    from mechanistic_agent.core.radical_validation import validate_radical_moves
    result = validate_radical_moves(reaction, parse_mechanism_moves(notation))
    assert result['valid'], result


@pytest.mark.parametrize('reaction,notation', [
    (CASES[0][0], 'sigma:1-2>2'),  # pair heterolysis cannot describe homolysis
    (CASES[1][0], 'rad:1>2;rad:1>2'),  # same radical spent twice
    (CASES[0][0], 'sigma:1-2>2:e1'),  # missing the other electron
    ('[CH4:1].[CH3:2]>>[CH3:1][CH3:2]', 'rad:1>2;rad:2>1'),  # no radical at source
    (CASES[4][0], 'rad:1>2'),  # bond arrow cannot describe SET
    (CASES[0][0], 'pi:1-2>2:e1;pi:2-1>1:e1'),  # no pi source
])
def test_radical_invalid_inventory_rejected(reaction, notation):
    from mechanistic_agent.core.radical_validation import validate_radical_moves
    assert not validate_radical_moves(reaction, parse_mechanism_moves(notation))['valid']


def _runtime(reaction, notation, metadata=None, resulting=None):
    import json
    from mechanistic_agent.tools import predict_mechanistic_step
    left, right = reaction.split('>>')
    return json.loads(predict_mechanistic_step(
        step_index=0, current_state=left.split('.'), target_products=[],
        resulting_state=resulting or right.split('.'),
        electron_pushes=[m.as_dict() for m in parse_mechanism_moves(notation)],
        reaction_smirks=reaction + ' |mech:v1;' + (metadata or notation) + '|',
    ))


def test_runtime_rejects_wrong_homolysis_pair_arrows():
    assert not _runtime(CASES[0][0], 'sigma:1-2>2')['bond_electron_validation']['valid']


def test_runtime_rejects_metadata_contradiction():
    assert not _runtime(CASES[0][0], CASES[0][1], 'sigma:1-2>2')['bond_electron_validation']['valid']


def test_runtime_retains_duplicate_radical_species():
    result = _runtime(*CASES[0], resulting=['[Cl]', '[Cl]'])
    assert result['resulting_state'] == ['[Cl]', '[Cl]']
    assert result['bond_electron_validation']['valid']


def test_runtime_rejects_mismatched_state_even_with_consistent_mapped_arrows():
    result = _runtime(*CASES[0], resulting=['[Cl-]', '[Cl+]'])
    assert not result['bond_electron_validation']['valid']


@pytest.mark.parametrize('electrons', [True, 1.0, '1', 3, -1])
def test_notation_cannot_mask_invalid_count_type(electrons):
    with pytest.raises(MechanismMoveFormatError):
        normalize_electron_pushes([{'notation': 'rad:1>2', 'electrons': electrons}])


def test_closed_shell_electron_transfer():
    from mechanistic_agent.core.radical_validation import validate_radical_moves
    assert validate_radical_moves('[Cl-:1].[CH3+:2]>>[Cl:1].[CH3:2]', parse_mechanism_moves('et:1>2'))['valid']


@pytest.mark.parametrize('reaction', [
    '[Cl:1][Cl:1]>>[Cl:1].[Cl:1]',
    'Cl[Cl:2]>>[Cl:1].[Cl:2]',
    '[Cl:1][Cl:2]>>[Cl:1].[Br:2]',
])
def test_radical_requires_conserved_unique_atom_maps(reaction):
    from mechanistic_agent.core.radical_validation import validate_radical_moves
    assert not validate_radical_moves(reaction, parse_mechanism_moves(CASES[0][1]))['valid']


def test_radical_detection_includes_states_without_fishhooks():
    from mechanistic_agent.core.radical_validation import requires_radical_validation
    assert requires_radical_validation(CASES[0][0], parse_mechanism_moves('sigma:1-2>2'))
    assert not requires_radical_validation('[CH3:1][Cl:2]>>[CH3+:1].[Cl-:2]', parse_mechanism_moves('sigma:1-2>2'))


def test_radical_replay_heavy_policy_preserves_heavy_ids():
    from mechanistic_agent.core.mapped_state import MappedState, execute_candidate
    reaction, notation = CASES[2]
    left, right = reaction.split('>>')
    state = MappedState.from_smiles(left.split('.'), hydrogen_policy='heavy_atom')
    result = execute_candidate(state, reaction_smirks=reaction,
        electron_pushes=[m.as_dict() for m in parse_mechanism_moves(notation)], expected_resulting_state=right.split('.'))
    assert result.ok, result.as_dict()
    assert result.smirks_state_agreement
    assert set(result.preserved_ids) == set(state.map_to_pid.values())
    assert not result.new_ids and not result.lost_ids


def test_radical_replay_rejects_invalid_arrows_and_wrong_map_binding():
    from mechanistic_agent.core.mapped_state import MappedState, execute_candidate
    state = MappedState.from_smiles(['[Cl:1][Cl:2]'])
    bad_arrows = execute_candidate(state, reaction_smirks=CASES[0][0],
        electron_pushes=[m.as_dict() for m in parse_mechanism_moves('sigma:1-2>2')])
    assert not bad_arrows.ok
    renamed = CASES[0][0].replace(':1', ':3').replace(':2', ':4')
    bad_maps = execute_candidate(state, reaction_smirks=renamed,
        electron_pushes=[m.as_dict() for m in parse_mechanism_moves('sigma:3-4>4:e1;sigma:4-3>3:e1')])
    assert not bad_maps.ok
