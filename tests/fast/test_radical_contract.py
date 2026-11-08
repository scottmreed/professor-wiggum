"""Independent full-step and chain fixtures, not a prediction-quality benchmark."""
import json
from pathlib import Path

import pytest

from mechanistic_agent.core.mapped_state import MappedState, execute_candidate
from mechanistic_agent.core.mechanism_moves import parse_mechanism_moves
from mechanistic_agent.quality_scoring import QualityStep, _bond_electron_valid
from mechanistic_agent.tools import predict_mechanistic_step

FIXTURE = json.loads((Path(__file__).parents[1] / 'fixtures/radical_contract.json').read_text())


@pytest.mark.parametrize('case', FIXTURE['steps'] + FIXTURE['chain']['steps'], ids=lambda c: c['id'])
def test_runtime_and_quality_agree_on_authored_step(case):
    left, right = case['reaction_smirks'].split('>>')
    pushes = [m.as_dict() for m in parse_mechanism_moves(case['notation'])]
    smirks = case['reaction_smirks'] + ' |mech:v1;' + case['notation'] + '|'
    payload = json.loads(predict_mechanistic_step(
        step_index=1, current_state=left.split('.'), target_products=right.split('.'),
        resulting_state=right.split('.'), reaction_smirks=smirks, electron_pushes=pushes,
    ))
    assert payload['bond_electron_validation']['valid'], payload
    assert _bond_electron_valid(QualityStep(1, left.split('.'), right.split('.'), smirks, pushes))[0]
    assert [p['electrons'] for p in payload['electron_pushes']] == [p['electrons'] for p in pushes]


def test_chain_replay_preserves_radicals_multiplicity_and_atom_identity():
    steps = FIXTURE['chain']['steps']
    state = MappedState.from_smiles(steps[0]['reaction_smirks'].split('>>')[0].split('.'), hydrogen_policy='explicit')
    original_ids = set(state.map_to_pid.values())
    for step in steps:
        expected = step['reaction_smirks'].split('>>')[1].split('.')
        result = execute_candidate(state, reaction_smirks=step['reaction_smirks'],
                                   electron_pushes=[m.as_dict() for m in parse_mechanism_moves(step['notation'])],
                                   expected_resulting_state=expected)
        assert result.ok, result.as_dict()
        assert result.smirks_state_agreement, result.as_dict()
        assert set(result.preserved_ids) == original_ids
        assert not result.lost_ids and not result.new_ids
        state = result.next_state


def test_quality_rejects_pair_arrow_for_homolysis():
    case = FIXTURE['steps'][0]
    left, right = case['reaction_smirks'].split('>>')
    smirks = case['reaction_smirks'] + ' |mech:v1;sigma:1-2>2|'
    pushes = [m.as_dict() for m in parse_mechanism_moves('sigma:1-2>2')]
    assert not _bond_electron_valid(QualityStep(1, left.split('.'), right.split('.'), smirks, pushes))[0]
