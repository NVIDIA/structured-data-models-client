# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Current-client checks cannot borrow proof from a frozen or missing peer."""

import pytest
from compare import assess, matrix
from test_contract import one, paired


def check(data, records, **kwargs):
    return assess(
        data,
        {},
        records,
        comparison='current-clients',
        candidate_commit='candidate',
        **kwargs,
    )


def test_current_mode_checks_contract_and_actual_language_peer():
    data = one()
    _, records = paired(data)
    report = check(data, records)
    assert report['goal_complete']
    assert {row['verdict'] for row in report['cells']} == {
        'Current contract verified',
        'Cross-language agreement',
    }
    assert 'Parity' not in matrix(report)
    java = next(key for key in records if key[3] == 'java')
    assert check(data, records, selected={java})['scope_complete']
    del records[(*java[:3], 'python')]
    assert not check(data, records, selected={java})['scope_complete']


@pytest.mark.parametrize(
    'mutation',
    ['native', 'phase', 'outcome', 'source', 'endpoint', 'results', 'route'],
)
def test_bad_python_peer_or_cross_language_difference_cannot_pass(mutation):
    data = one()
    _, records = paired(data)
    python = next(
        record for key, record in records.items() if key[3] == 'python'
    )
    if mutation == 'native':
        python['artifact']['native_hash'] = ''
    elif mutation == 'phase':
        python['phase'] = 'baseline'
    elif mutation == 'outcome':
        python['status'] = 'refused'
    elif mutation == 'source':
        python['source_snapshot'] = 'different'
    elif mutation == 'endpoint':
        python['endpoint']['version'] = 'different'
    elif mutation == 'results':
        python['semantic']['results'][0][1] = 9.0
    else:
        python['route']['legacy_hits'] = ['old-code']
    assert not check(data, records)['goal_complete']


def test_current_mode_pins_python_artifact_commit():
    data = one()
    _, records = paired(data)
    next(record for key, record in records.items() if key[3] == 'python')[
        'artifact'
    ]['commit'] = 'different'
    with pytest.raises(ValueError, match='requested commit'):
        check(data, records)


def test_local_mode_never_claims_live_acceptance_or_agreement():
    data = one()
    _, records = paired(data)
    for record in records.values():
        record['execution'] = 'local'
        record['semantic']['lifecycle'] = ['materialize']
        record['semantic']['results'] = []
    report = check(data, records, live=False)
    assert report['scope_complete']
    assert not report['goal_complete']
    assert 'Local check only' in matrix(report)
    assert 'Cross-language agreement' not in matrix(report)


def test_verdict_metadata_does_not_waive_a_supported_language_difference():
    data = one()
    case = data['scenarios'][0]
    case['expected_by_language'] = {
        'java': {**case['expected'], 'verdict': 'Improvement'}
    }
    _, records = paired(data)
    next(record for key, record in records.items() if key[3] == 'java')[
        'semantic'
    ]['results'][0][1] = 9.0
    assert not check(data, records)['goal_complete']


def test_declared_refusal_difference_is_checked_without_claiming_agreement():
    from test_contract import observation

    from contract import manifest

    data = one()
    case = next(
        case
        for case in manifest()['scenarios']
        if case['id'] == 'refusal.source-persistence'
    )
    case['connectors'] = ['memory']
    data['scenarios'] = [case]
    records = {}
    for language in ('python', 'java'):
        expected = case.get('expected_by_language', {}).get(
            language, case['expected']
        )
        record = observation(case, language=language)
        record.update(
            status=expected.get('outcome', 'refused'),
            semantic=None,
            error={key: expected[key] for key in ('stage', 'code', 'rule')},
        )
        records[(case['id'], 'memory', 'relational', language)] = record
    report = check(data, records)
    assert report['goal_complete']
    assert report['cells'][0]['verdict'] == 'Declared language difference'
    records[(case['id'], 'memory', 'relational', 'java')]['error']['code'] = (
        'OTHER'
    )
    assert not check(data, records)['goal_complete']


@pytest.mark.parametrize(
    'case_id', ['types.numeric.provider', 'relational.temporal-link']
)
@pytest.mark.parametrize('raw_reader', [False, True])
def test_current_comparison_never_applies_frozen_client_bug_waivers(
    monkeypatch, case_id, raw_reader
):
    import compare

    def forbidden(*args):
        pytest.fail('historical improvement waiver used for current clients')

    monkeypatch.setattr(compare, 'equivalent_legacy_decimal_cast', forbidden)
    monkeypatch.setattr(compare, 'equivalent_derived_entity_refs', forbidden)
    data = one()
    case = data['scenarios'][0]
    case['id'] = case_id
    if raw_reader:
        case['input_mode'] = 'raw-reader'
    case['connectors'] = ['duckdb']
    _, records = paired(data)
    for record in records.values():
        record['connector'] = 'duckdb'
    assert check(data, records)['goal_complete']
    next(record for key, record in records.items() if key[3] == 'java')[
        'semantic'
    ]['request'] = {'wrong': 'data'}
    assert not check(data, records)['goal_complete']
