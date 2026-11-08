"""Regression tests for the radical-support review findings.

The invariant: ordinary two-electron (pair-arrow) steps behave exactly as on
the pre-radical base; radical strictness applies only to explicit one-electron
moves or genuinely radical states.
"""
import json

import pytest

from mechanistic_agent.core.mechanism_moves import (
    MechanismMoveFormatError,
    normalize_electron_pushes,
    parse_mechanism_moves,
)
from mechanistic_agent.core.radical_validation import (
    radical_states_match,
    requires_radical_validation,
    validate_radical_moves,
)
from mechanistic_agent.quality_scoring import QualityStep, _bond_electron_valid
from mechanistic_agent.tools import MechanismStepPrediction, predict_mechanistic_step, propose_intermediates

# Hydroxide adds to acetone; a core-only SMIRKS whose bracket atoms carry no H
# (RDKit reads ``[C:1]`` as a radical carbon, which it is not).
CORE_SMIRKS = "[C:1]=[O:2].[O-:3]>>[C:1]([O-:2])[O:3] |mech:v1;lp:3>1;pi:1-2>2|"
CORE_PUSHES = [
    {"kind": "lone_pair", "source_atom": "3", "target_atom": "1", "electrons": 2},
    {"kind": "pi_bond", "source_bond": ["1", "2"], "through_atom": "2", "target_atom": "2", "electrons": 2},
]


def _predict(**kwargs):
    payload = predict_mechanistic_step(step_index=0, target_products=[], **kwargs)
    return json.loads(payload) if isinstance(payload, str) else payload


# --- Finding 1: radical path chosen from H-less SMIRKS atoms -----------------


def test_core_smirks_without_bracket_h_is_validated_as_pair_step():
    result = _predict(
        current_state=["CC(C)=O", "[OH-]"],
        resulting_state=["CC(C)(O)[O-]"],
        electron_pushes=CORE_PUSHES,
        reaction_smirks=CORE_SMIRKS,
    )
    assert result["bond_electron_validation"]["valid"] is True
    assert "error" not in result["bond_electron_validation"]


def test_pair_step_still_dedupes_resulting_state():
    result = _predict(
        current_state=["CC(C)=O", "[OH-]"],
        resulting_state=["CC(C)(O)[O-]", "CC(C)(O)[O-]"],
        electron_pushes=CORE_PUSHES,
        reaction_smirks=CORE_SMIRKS,
    )
    assert result["resulting_state"] == ["CC(C)(O)[O-]"]


def test_radical_detection_ignores_smirks_atoms_without_fishhooks():
    pair = parse_mechanism_moves("lp:3>1;pi:1-2>2")
    assert not requires_radical_validation(pair)
    assert not requires_radical_validation(pair, states=[["CC(C)=O", "[OH-]"], ["CC(C)(O)[O-]"]])
    assert requires_radical_validation(parse_mechanism_moves("sigma:1-2>2:e1;sigma:2-1>1:e1"))
    assert requires_radical_validation(pair, states=[["ClCl"], ["[Cl]", "[Cl]"]])


def test_mapped_execute_candidate_keeps_pair_route_for_core_smirks():
    from mechanistic_agent.core.mapped_state import MappedState, execute_candidate

    state = MappedState.from_smiles(["[CH3:4][C:1](=[O:2])[CH3:5]", "[OH-:3]"])
    result = execute_candidate(state, reaction_smirks=CORE_SMIRKS, electron_pushes=CORE_PUSHES)
    assert result.match_mode != "map_constrained_radical_inventory"
    assert result.ok, result.as_dict()


# --- Finding 2: pair-move pushes tolerated as on base --------------------------


BASE_TOLERATED = {
    # name: (pushes, expected [(notation, (bond_start, bond_end, through_atom))])
    "pi_through_start_with_notation": (
        [{"kind": "pi_bond", "source_bond": ["1", "2"], "through_atom": "1", "target_atom": "3", "notation": "pi:2-1>3"}],
        [("pi:2-1>3", (2, 1, 1))],
    ),
    "one_bad_one_good": (
        [{"kind": "lone_pair", "source_atom": "3", "target_atom": "1"},
         {"kind": "lone_pair", "source_atom": None, "target_atom": "1"}],
        [("lp:3>1", (None, None, None))],
    ),
    "electrons_str": (
        [{"kind": "lone_pair", "source_atom": "3", "target_atom": "1", "electrons": "2"}],
        [("lp:3>1", (None, None, None))],
    ),
    "electrons_float": (
        [{"kind": "lone_pair", "source_atom": "3", "target_atom": "1", "electrons": 2.0}],
        [("lp:3>1", (None, None, None))],
    ),
    "through_not_endpoint": (
        [{"kind": "sigma_bond", "source_bond": ["1", "2"], "through_atom": "3", "target_atom": "3"}],
        [("sigma:1-2>3", (1, 2, 3))],
    ),
    "notation_and_kind_space": (
        [{"kind": "lone pair", "source_atom": "3", "target_atom": "1", "notation": "lp:3>1"}],
        [("lp:3>1", (None, None, None))],
    ),
    "bad_notation_plus_fields": (
        [{"notation": "lp 3->1", "kind": "lone_pair", "source_atom": "3", "target_atom": "1"}],
        [],
    ),
    "dict_not_list": ({"kind": "lone_pair", "source_atom": "3", "target_atom": "1"}, []),
    "sigma_through_start_no_notation": (
        [{"kind": "sigma_bond", "source_bond": ["2", "3"], "through_atom": "2", "target_atom": "5"}],
        [("sigma:2-3>5", (2, 3, 2))],
    ),
    "unknown_kind_start_end": (
        [{"kind": "mystery", "start_atom": "1", "end_atom": "2"}],
        [("lp:1>2", (None, None, None))],
    ),
    "lone_pair_electrons_zero": (
        [{"kind": "lone_pair", "source_atom": "1", "target_atom": "2", "electrons": 0}],
        [("lp:1>2", (None, None, None))],
    ),
}


@pytest.mark.parametrize("name", sorted(BASE_TOLERATED))
def test_pair_pushes_normalize_as_on_base(name):
    pushes, expected = BASE_TOLERATED[name]
    moves = normalize_electron_pushes(pushes)
    got = [(m.as_dict()["notation"], (m.bond_start, m.bond_end, m.through_atom)) for m in moves]
    assert got == expected
    assert all(m.electrons == 2 for m in moves)


@pytest.mark.parametrize("push", [
    {"kind": "radical", "source_atom": None, "target_atom": "1"},
    {"kind": "sigma_bond", "source_bond": ["1", "2"], "through_atom": "3", "target_atom": "3", "electrons": 1},
    {"notation": "rad 1->2"},
    {"notation": "pi:1-2>2:e3"},
])
def test_malformed_one_electron_push_fails_closed(push):
    with pytest.raises(MechanismMoveFormatError):
        normalize_electron_pushes([push])


def test_single_one_electron_dict_fails_closed():
    with pytest.raises(MechanismMoveFormatError):
        normalize_electron_pushes({"kind": "radical", "source_atom": "1", "target_atom": "2"})


def test_one_bad_pair_candidate_does_not_drop_the_response():
    good = {
        "rank": 1,
        "intermediate_smiles": "CC(C)(O)[O-]",
        "reaction_description": "x",
        "reaction_smirks": "[CH3:4][C:1](=[O:2])[CH3:5].[OH-:3]>>[CH3:4][C:1]([O-:2])([OH:3])[CH3:5] |mech:v1;lp:3>1;pi:1-2>2|",
        "electron_pushes": CORE_PUSHES,
    }
    bad = dict(good, rank=2, electron_pushes=CORE_PUSHES + [
        {"kind": "sigma_bond", "source_bond": ["1", "4"], "through_atom": "2", "target_atom": "2", "electrons": 2},
    ])
    parsed = MechanismStepPrediction.model_validate(
        {"classification": "intermediate_step", "analysis": "a", "candidates": [good, bad]}
    )
    assert [len(c.electron_pushes) for c in parsed.candidates] == [2, 3]


def test_one_bad_fishhook_candidate_does_not_drop_the_response():
    good = {
        "rank": 1,
        "intermediate_smiles": "CC(C)(O)[O-]",
        "reaction_description": "x",
        "reaction_smirks": "[CH3:4][C:1](=[O:2])[CH3:5].[OH-:3]>>[CH3:4][C:1]([O-:2])([OH:3])[CH3:5] |mech:v1;lp:3>1;pi:1-2>2|",
        "electron_pushes": CORE_PUSHES,
    }
    bad = dict(good, rank=2, electron_pushes=[{"kind": "radical", "source_atom": None, "target_atom": "1"}])
    parsed = MechanismStepPrediction.model_validate(
        {"classification": "intermediate_step", "analysis": "a", "candidates": [good, bad]}
    )
    assert len(parsed.candidates) == 2


def _stub_llm(monkeypatch, payload):
    class _Response:
        usage = None

        def __init__(self) -> None:
            self.tool_calls = [{"arguments": json.dumps(payload)}]

    class _LLM:
        def invoke(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
            return _Response()

    monkeypatch.setattr("mechanistic_agent.tools.adapter_supports_forced_tools", lambda _model: True)
    monkeypatch.setattr("mechanistic_agent.tools.get_model_api_key", lambda *_a, **_k: "test-key")
    monkeypatch.setattr("mechanistic_agent.tools.get_chat_model", lambda *_a, **_k: _LLM())


def test_propose_keeps_good_candidate_when_fishhook_candidate_is_malformed(monkeypatch):
    good = {
        "rank": 1,
        "intermediate_smiles": "CCCl",
        "reaction_description": "sn2",
        "reaction_smirks": "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] |mech:v1;lp:4>2;sigma:2-3>3|",
        "electron_pushes": [
            {"kind": "lone_pair", "source_atom": "4", "target_atom": "2", "electrons": 2},
            {"kind": "sigma_bond", "source_bond": ["2", "3"], "through_atom": "3", "target_atom": "3", "electrons": 2},
        ],
    }
    bad = dict(good, rank=2, electron_pushes=[{"kind": "radical", "source_atom": None, "target_atom": "2"}])
    _stub_llm(monkeypatch, {"classification": "intermediate_step", "analysis": "a", "candidates": [good, bad]})
    payload = json.loads(propose_intermediates(
        starting_materials=["CCBr", "[Cl-]"], products=["CCCl", "[Br-]"], current_state=["CCBr", "[Cl-]"],
    ))
    ranks = [c.get("rank") for c in payload.get("candidates") or []]
    assert 1 in ranks and 2 not in ranks


def test_quality_scoring_pair_step_matches_base():
    step = QualityStep(1, ["CC(C)=O", "[OH-]"], ["CC(C)(O)[O-]"], CORE_SMIRKS, CORE_PUSHES)
    assert _bond_electron_valid(step) == (True, None)
    # A malformed fishhook push in a stored step must not crash scoring.
    bad = QualityStep(1, ["ClCl"], ["[Cl]", "[Cl]"], "[Cl:1][Cl:2]>>[Cl:1].[Cl:2]",
                      [{"kind": "radical", "source_atom": None, "target_atom": "1"}])
    valid, _error = _bond_electron_valid(bad)
    assert valid is False


def test_flower_curriculum_move_conversion_does_not_crash_on_through_start():
    from mechanistic_agent.flower_curriculum import convert_elementary_step

    # Two-arrow sigma move whose delivering atom is the first listed bond atom.
    try:
        convert_elementary_step("[CH3:1][Br:2].[OH-:3]>>[CH3:1][OH:3].[Br-:2]")
    except Exception as exc:  # noqa: BLE001
        assert type(exc).__name__ == "ConversionError", exc


# --- Finding 3: zero-net bond arrows cancel ------------------------------------

ETHANE_HAT = "[CH3:1].[H:2][CH2:3][CH3:4]>>[CH3:1][H:2].[CH2:3][CH3:4]"
PROPENE_HAT = "[CH3:1].[H:2][CH2:3][CH:4]=[CH2:5]>>[CH3:1][H:2].[CH2:3][CH:4]=[CH2:5]"
HAT_NOTATION = "rad:1>2;sigma:3-2>1:e1;sigma:2-3>3:e1"


def test_hat_baseline_valid():
    assert validate_radical_moves(ETHANE_HAT, parse_mechanism_moves(HAT_NOTATION))["valid"]
    assert validate_radical_moves(PROPENE_HAT, parse_mechanism_moves(HAT_NOTATION))["valid"]


@pytest.mark.parametrize("reaction,extra", [
    (ETHANE_HAT, "sigma:3-4>3"),
    (ETHANE_HAT, "sigma:3-4>3:e1"),
    (PROPENE_HAT, "pi:4-5>4:e1"),
    (PROPENE_HAT, "pi:5-4>5"),
])
def test_self_cancelling_bond_arrow_rejected(reaction, extra):
    result = validate_radical_moves(reaction, parse_mechanism_moves(HAT_NOTATION + ";" + extra))
    assert not result["valid"], result


# --- Finding 4: radical steps in the live loop ----------------------------------

HOMOLYSIS = "[Cl:1][Cl:2]>>[Cl:1].[Cl:2] |mech:v1;sigma:1-2>2:e1;sigma:2-1>1:e1|"
HOMOLYSIS_PUSHES = [m.as_dict() for m in parse_mechanism_moves("sigma:1-2>2:e1;sigma:2-1>1:e1")]


def test_radical_states_match_allows_spectators():
    assert radical_states_match(HOMOLYSIS, ["ClCl", "C1CCCCC1"], ["[Cl]", "[Cl]", "C1CCCCC1"])
    assert radical_states_match(HOMOLYSIS, ["C1CCCCC1", "ClCl"], ["C1CCCCC1", "[Cl]", "[Cl]"])
    assert not radical_states_match(HOMOLYSIS, ["ClCl", "C1CCCCC1"], ["[Cl]", "[Cl]"])
    assert not radical_states_match(HOMOLYSIS, ["ClCl", "C1CCCCC1"], ["[Cl]", "[Cl]", "CCCCCC"])
    assert not radical_states_match(HOMOLYSIS, ["ClCl"], ["[Cl]"])


def test_radical_step_with_spectator_validates():
    result = _predict(
        current_state=["ClCl", "C1CCCCC1"],
        resulting_state=["[Cl]", "[Cl]", "C1CCCCC1"],
        electron_pushes=HOMOLYSIS_PUSHES,
        reaction_smirks=HOMOLYSIS,
    )
    assert result["bond_electron_validation"]["valid"], result["bond_electron_validation"]


PEROXIDE = (
    "[CH3:1][C:2]([CH3:3])([CH3:4])[O:5][O:6][C:7]([CH3:8])([CH3:9])[CH3:10]>>"
    "[CH3:1][C:2]([CH3:3])([CH3:4])[O:5].[O:6][C:7]([CH3:8])([CH3:9])[CH3:10]"
    " |mech:v1;sigma:5-6>6:e1;sigma:6-5>5:e1|"
)
PEROXIDE_PUSHES = [m.as_dict() for m in parse_mechanism_moves("sigma:5-6>6:e1;sigma:6-5>5:e1")]


def test_peroxide_homolysis_with_spectator_validates():
    result = _predict(
        current_state=["CC(C)(C)OOC(C)(C)C", "C1CCCCC1"],
        resulting_state=["CC(C)(C)[O]", "CC(C)(C)[O]", "C1CCCCC1"],
        electron_pushes=PEROXIDE_PUSHES,
        reaction_smirks=PEROXIDE,
    )
    assert result["bond_electron_validation"]["valid"], result["bond_electron_validation"]
    assert result["resulting_state"] == ["CC(C)(C)[O]", "CC(C)(C)[O]", "C1CCCCC1"]


def test_propose_keeps_radical_multiplicity_in_resulting_state(monkeypatch):
    candidate = {
        "rank": 1,
        "intermediate_smiles": "CC(C)(C)[O]",
        "reaction_description": "peroxide homolysis",
        "reaction_smirks": PEROXIDE,
        "electron_pushes": PEROXIDE_PUSHES,
        "resulting_state": ["CC(C)(C)[O]", "CC(C)(C)[O]", "C1CCCCC1"],
    }
    _stub_llm(monkeypatch, {"classification": "intermediate_step", "analysis": "a", "candidates": [candidate]})
    payload = json.loads(propose_intermediates(
        starting_materials=["CC(C)(C)OOC(C)(C)C", "C1CCCCC1"], products=["CC(C)(C)O", "[CH]1CCCCC1"],
        current_state=["CC(C)(C)OOC(C)(C)C", "C1CCCCC1"],
    ))
    assert payload.get("candidates"), payload.get("rejected_candidates")
    assert payload["candidates"][0]["resulting_state"] == ["CC(C)(C)[O]", "CC(C)(C)[O]", "C1CCCCC1"]


def test_propose_still_dedupes_pair_candidate_resulting_state(monkeypatch):
    candidate = {
        "rank": 1,
        "intermediate_smiles": "CCCl",
        "reaction_description": "sn2",
        "reaction_smirks": "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] |mech:v1;lp:4>2;sigma:2-3>3|",
        "electron_pushes": [
            {"kind": "lone_pair", "source_atom": "4", "target_atom": "2", "electrons": 2},
            {"kind": "sigma_bond", "source_bond": ["2", "3"], "through_atom": "3", "target_atom": "3", "electrons": 2},
        ],
        "resulting_state": ["CCCl", "[Br-]", "[Br-]"],
    }
    _stub_llm(monkeypatch, {"classification": "intermediate_step", "analysis": "a", "candidates": [candidate]})
    payload = json.loads(propose_intermediates(
        starting_materials=["CCBr", "[Cl-]"], products=["CCCl", "[Br-]"], current_state=["CCBr", "[Cl-]"],
    ))
    assert payload["candidates"][0]["resulting_state"] == ["CCCl", "[Br-]"]


# --- Finding 8: metal / H atoms do not force the radical path ------------------


@pytest.mark.parametrize("extra", ["[Na]", "[Li]", "[K]", "[Mg]", "[Cu]", "[H]"])
def test_pair_step_with_metal_or_h_atom_scores_as_base(extra):
    step = QualityStep(1, ["CC(C)=O", "[OH-]", extra], ["CC(C)(O)[O-]", extra], CORE_SMIRKS, CORE_PUSHES)
    assert _bond_electron_valid(step) == (True, None)
    assert not requires_radical_validation(
        normalize_electron_pushes(CORE_PUSHES), states=[step.current_state, step.resulting_state]
    )


# --- Round 2, finding 1: an untouched radical spectator keeps the pair path ----

TEMPO = "CC1(C)CCCC(C)(C)N1[O]"
SN2_PUSHES = [
    {"kind": "lone_pair", "source_atom": "1", "target_atom": "3", "electrons": 2},
    {"kind": "sigma_bond", "source_bond": ["3", "4"], "through_atom": "4", "target_atom": "4", "electrons": 2},
]
SN2_CORE = "[O-:1].[CH3:3][Br:4]>>[O:1][CH3:3].[Br-:4] |mech:v1;lp:1>3;sigma:3-4>4|"
SN2_FULL = "[CH3:2][O-:1].[CH3:3][Br:4]>>[CH3:2][O:1][CH3:3].[Br-:4] |mech:v1;lp:1>3;sigma:3-4>4|"


@pytest.mark.parametrize("spectator", [TEMPO, "[O][O]"])
def test_pair_step_with_untouched_radical_spectator_matches_base(spectator):
    current, resulting = ["C[O-]", "CBr", spectator], ["COC", "[Br-]", spectator]
    assert not requires_radical_validation(
        normalize_electron_pushes(SN2_PUSHES), states=[current, resulting], reaction_smirks=SN2_CORE,
    )
    result = _predict(current_state=current, resulting_state=resulting, electron_pushes=SN2_PUSHES, reaction_smirks=SN2_CORE)
    assert result["bond_electron_validation"]["valid"] is True, result["bond_electron_validation"]
    assert result["resulting_state"] == resulting
    assert _bond_electron_valid(QualityStep(1, current, resulting, SN2_CORE, SN2_PUSHES)) == (True, None)


def test_pair_step_with_radical_spectator_still_dedupes():
    result = _predict(
        current_state=["C[O-]", "CBr", TEMPO, "O", "O"],
        resulting_state=["COC", "[Br-]", TEMPO, "O", "O"],
        electron_pushes=SN2_PUSHES,
        reaction_smirks=SN2_FULL,
    )
    assert result["resulting_state"] == ["COC", "[Br-]", TEMPO, "O"]


def test_mapped_executor_pair_step_with_radical_spectator_matches_base():
    from mechanistic_agent.core.mapped_state import advance_mapped_loop_state

    _snap, record = advance_mapped_loop_state(
        None, previous_state=["C[O-]", "CBr", TEMPO], stated_resulting_state=["COC", "[Br-]", TEMPO],
        reaction_smirks=SN2_CORE, electron_pushes=SN2_PUSHES,
    )
    assert record.get("executed") is True, record
    assert record.get("smirks_state_agreement") is True, record
    assert record.get("failure_category") is None


def test_pair_step_creating_a_radical_still_fails_closed():
    current, resulting = ["C[O-]", "CBr"], ["COC", "[Br]"]
    assert requires_radical_validation(normalize_electron_pushes(SN2_PUSHES), states=[current, resulting])
    result = _predict(current_state=current, resulting_state=resulting, electron_pushes=SN2_PUSHES, reaction_smirks=SN2_CORE)
    assert result["bond_electron_validation"]["valid"] is False
    assert _bond_electron_valid(QualityStep(1, current, resulting, SN2_CORE, SN2_PUSHES))[0] is False


def test_pair_arrows_touching_a_radical_species_select_strict_path():
    # Degenerate exchange: the radical species multiset is unchanged, but the
    # SMIRKS touches the chlorine radical.
    pair = parse_mechanism_moves("lp:1>2;sigma:2-3>3")
    smirks = "[Cl:1].[Cl:2][Cl:3]>>[Cl:1][Cl:2].[Cl:3]"
    assert requires_radical_validation(pair, states=[["[Cl]", "ClCl"], ["ClCl", "[Cl]"]], reaction_smirks=smirks)


# --- Round 2, finding 2: cancelling one-electron arrows and empty steps -------

CL2_HOMOLYSIS_CORE = "[Cl:1][Cl:2]>>[Cl:1].[Cl:2]"


def test_legitimate_homolysis_still_validates():
    assert validate_radical_moves(CL2_HOMOLYSIS_CORE, parse_mechanism_moves("sigma:1-2>2:e1;sigma:2-1>1:e1"))["valid"]


@pytest.mark.parametrize("reaction,notation", [
    (CL2_HOMOLYSIS_CORE, "sigma:1-2>2:e1;sigma:2-1>1:e1;et:1>2;et:2>1"),
    ("[Cl:1].[Cl:2]>>[Cl:1].[Cl:2]", "et:1>2;et:2>1"),
    ("[Cl:1].[Cl:2].[Cl:3]>>[Cl:1].[Cl:2].[Cl:3]", "et:1>2;et:2>3;et:3>1"),
])
def test_mutually_cancelling_moves_rejected(reaction, notation):
    assert not validate_radical_moves(reaction, parse_mechanism_moves(notation))["valid"]


def test_cancelling_et_pair_fails_quality_scoring():
    pushes = [{"notation": n} for n in ("et:1>2", "et:2>1")]
    step = QualityStep(1, ["[Cl]", "[Cl]"], ["[Cl]", "[Cl]"], "[Cl:1].[Cl:2]>>[Cl:1].[Cl:2]", pushes)
    assert _bond_electron_valid(step)[0] is False
    step = QualityStep(1, ["ClCl", "O"], ["[Cl]", "[Cl]", "O"], CL2_HOMOLYSIS_CORE,
                       [{"notation": n} for n in ("sigma:1-2>2:e1", "sigma:2-1>1:e1", "et:1>2", "et:2>1")])
    assert _bond_electron_valid(step)[0] is False


def test_empty_radical_steps_rejected():
    assert not validate_radical_moves(">>", [])["valid"]
    assert not validate_radical_moves("[Cl:1].[Cl:2]>>[Cl:1].[Cl:2]", [])["valid"]
    assert not radical_states_match(">>", ["[Cl]", "[Cl]"], ["[Cl]", "[Cl]"])


def test_radical_executor_rejects_empty_moves():
    from mechanistic_agent.core.mapped_state import MappedState, _execute_radical_candidate

    state = MappedState.from_smiles(["[Cl:1]", "[Cl:2]"])
    result = _execute_radical_candidate(
        state, "[Cl:1].[Cl:2]>>[Cl:1].[Cl:2]", [], expected_resulting_state=["[Cl]", "[Cl]"], step_label="t",
    )
    assert not result.ok


# --- Round 2, findings 3 and 4: metadata/push agreement --------------------------

RADICAL_SN2_STATES = (["C[O-]", "CBr", "[Cl]"], ["COC", "[Br-]", "[Cl]"])


def test_reordered_pair_bond_agrees_with_metadata():
    pushes = [
        {"kind": "lone_pair", "source_atom": "1", "target_atom": "3", "electrons": 2},
        {"kind": "sigma_bond", "source_bond": ["4", "3"], "through_atom": "4", "target_atom": "4", "electrons": 2},
    ]
    smirks = "[CH3:2][O-:1].[CH3:3][Br:4].[Cl:5]>>[CH3:2][O:1][CH3:3].[Br-:4].[Cl:5] |mech:v1;lp:1>3;sigma:3-4>4|"
    # [Cl:5] is touched by the SMIRKS, so the strict radical path runs.
    assert requires_radical_validation(normalize_electron_pushes(pushes), states=RADICAL_SN2_STATES, reaction_smirks=smirks)
    assert _bond_electron_valid(QualityStep(1, *RADICAL_SN2_STATES, smirks, pushes)) == (True, None)
    result = _predict(current_state=RADICAL_SN2_STATES[0], resulting_state=RADICAL_SN2_STATES[1],
                      electron_pushes=pushes, reaction_smirks=smirks)
    assert result["bond_electron_validation"]["valid"] is True, result["bond_electron_validation"]


def test_radical_step_without_mech_block_consistent_across_paths():
    pushes = [m.as_dict() for m in parse_mechanism_moves("sigma:1-2>2:e1;sigma:2-1>1:e1")]
    current, resulting = ["ClCl"], ["[Cl]", "[Cl]"]
    assert _bond_electron_valid(QualityStep(1, current, resulting, CL2_HOMOLYSIS_CORE, pushes)) == (True, None)
    result = _predict(current_state=current, resulting_state=resulting, electron_pushes=pushes,
                      reaction_smirks=CL2_HOMOLYSIS_CORE)
    assert result["bond_electron_validation"]["valid"] is True, result["bond_electron_validation"]


def test_radical_metadata_disagreement_fails_both_paths():
    pushes = [m.as_dict() for m in parse_mechanism_moves("sigma:1-2>2:e1;sigma:2-1>1:e1")]
    smirks = CL2_HOMOLYSIS_CORE + " |mech:v1;sigma:1-2>2:e1|"
    assert _bond_electron_valid(QualityStep(1, ["ClCl"], ["[Cl]", "[Cl]"], smirks, pushes))[0] is False
    result = _predict(current_state=["ClCl"], resulting_state=["[Cl]", "[Cl]"], electron_pushes=pushes, reaction_smirks=smirks)
    assert result["bond_electron_validation"]["valid"] is False
