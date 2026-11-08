from pathlib import Path

import pytest

from mechanistic_agent.radical_dataset import extract_radical_training_cases

RADICAL = '[CH3:1].[Cl:2]>>[CH3:1][Cl:2]|RC\n'


def test_extracts_deduplicated_step_cases_with_provenance(tmp_path):
    source = tmp_path / 'train.txt'
    source.write_text('[CH4:1]>>[CH4:1]|12\n' + RADICAL + RADICAL)
    cases, report = extract_radical_training_cases(source)
    assert len(cases) == 1
    assert cases[0]['starting_materials'] == ['[CH3:1]', '[Cl:2]']
    assert cases[0]['products'] == ['[CH3:1][Cl:2]']
    assert cases[0]['provenance']['source_split'] == 'train'
    assert cases[0]['provenance']['source_label'] == 'RC'
    assert cases[0]['provenance']['source_lines'] == [2, 3]
    assert 'verified_mechanism' not in cases[0]  # no fabricated arrows
    assert report['duplicate_rows'] == 1
    assert report['radical_rows'] == 2
    source.write_text(RADICAL)
    again, _ = extract_radical_training_cases(source)
    assert again[0]['id'] == cases[0]['id']


def test_rejects_test_and_validation_sources_before_reading(tmp_path):
    for filename in ('test.txt', 'val.txt', 'holdout.txt'):
        with pytest.raises(ValueError, match='train.txt'):
            extract_radical_training_cases(tmp_path / filename)


def test_bad_radical_rows_reported_and_no_silent_repair(tmp_path):
    source = tmp_path / 'train.txt'
    source.write_text('[CH4:1]>>[CH4:1]|RC\ninvalid>>invalid|RS\n')
    cases, report = extract_radical_training_cases(source)
    assert cases == []
    assert report['rejected_rows'] == 2


def test_limit_is_deterministic_and_counts_full_source(tmp_path):
    source = tmp_path / 'train.txt'
    source.write_text(RADICAL + '[OH:1].[CH3:2]>>[OH:1][CH3:2]|RS\n')
    cases, report = extract_radical_training_cases(source, limit=1)
    assert len(cases) == 1
    assert report['unique_cases'] == 2
    assert report['exported_cases'] == 1
    assert len(report['source_sha256']) == 64
