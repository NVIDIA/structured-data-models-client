# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Current-client checks cannot borrow proof from a frozen or missing peer."""

import pytest
from compare import assess, matrix
from test_contract import current_records, one


def check(data, records, **kwargs):
    return assess(
        data,
        records,
        candidate_commit='candidate',
        **kwargs,
    )


def test_current_mode_checks_contract_and_actual_language_peer():
    data = one()
    records = current_records(data)
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
    ['native', 'phase', 'outcome', 'source', 'endpoint', 'results'],
)
def test_bad_python_peer_or_cross_language_difference_cannot_pass(mutation):
    data = one()
    records = current_records(data)
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
    assert not check(data, records)['goal_complete']


def test_current_mode_pins_python_artifact_commit():
    data = one()
    records = current_records(data)
    next(record for key, record in records.items() if key[3] == 'python')[
        'artifact'
    ]['commit'] = 'different'
    with pytest.raises(ValueError, match='requested commit'):
        check(data, records)


def test_local_mode_never_claims_live_acceptance_or_agreement():
    data = one()
    records = current_records(data)
    for record in records.values():
        record['execution'] = 'local'
        record['semantic']['lifecycle'] = ['materialize']
        record['semantic']['results'] = []
    report = check(data, records, live=False)
    assert report['scope_complete']
    assert not report['goal_complete']
    assert 'Local check only' in matrix(report)


@pytest.mark.parametrize(
    'feature', ['supplement', 'raw-reader', 'prediction-refusal']
)
def test_materialization_mode_does_not_require_unexecuted_model_calls(feature):
    from contract import fixture_hash

    data = one()
    case = data['scenarios'][0]
    if feature == 'supplement':
        case['supplement'] = {'kind': 'batch', 'prediction_calls': 2}
    elif feature == 'raw-reader':
        case['input_mode'] = 'raw-reader'
    else:
        case['expected'] = {
            'status': 'refused',
            'stage': 'predict',
            'code': 'UNSUPPORTED_DTYPE',
            'rule': None,
        }
    records = current_records(data)
    for record in records.values():
        record['fixture_hash'] = fixture_hash(case)
        record['execution'] = 'local'
        record['status'] = 'passed'
        record.pop('error', None)
        record['semantic']['lifecycle'] = ['materialize']
        record['semantic']['results'] = []
    report = check(data, records, live=False)
    assert report['scope_complete']
    assert not report['goal_complete']
    assert not check(data, records, live=True)['scope_complete']
    assert 'Cross-language agreement' not in matrix(report)


def test_verdict_metadata_does_not_waive_a_supported_language_difference():
    data = one()
    case = data['scenarios'][0]
    case['expected_by_language'] = {
        'java': {**case['expected'], 'verdict': 'Improvement'}
    }
    records = current_records(data)
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
def test_current_comparison_requires_exact_data_for_every_case(
    case_id, raw_reader
):
    data = one()
    case = data['scenarios'][0]
    case['id'] = case_id
    if raw_reader:
        case['input_mode'] = 'raw-reader'
    case['connectors'] = ['duckdb']
    records = current_records(data)
    for record in records.values():
        record['connector'] = 'duckdb'
    assert check(data, records)['goal_complete']
    next(record for key, record in records.items() if key[3] == 'java')[
        'semantic'
    ]['request'] = {'wrong': 'data'}
    assert not check(data, records)['goal_complete']


def test_public_schema_requires_current_native_engine(monkeypatch):
    """A missing native engine must fail rather than run host-side policy."""
    import pandas as pd
    from sdm_client import Schema
    from sdm_client._rust import NativeBoundaryError, run

    def unavailable():
        raise NativeBoundaryError('native engine unavailable for this check')

    monkeypatch.setattr(run, 'native', unavailable)
    with pytest.raises(NativeBoundaryError, match='native engine unavailable'):
        Schema.from_data(
            {'rows': pd.DataFrame({'id': [1, 2], 'value': [3, 4]})}
        )


def test_historical_execution_is_not_a_runner_mode():
    from python_runner import Incomplete, artifact

    with pytest.raises(Incomplete, match='only current installed clients'):
        artifact('unused', 'baseline')


def _public_query_and_model(model_name):
    import pandas as pd
    from sdm_client import KumoRelational, KumoTabular, PredictiveQuery, Schema

    schema = Schema.from_data(
        {
            'rows': pd.DataFrame(
                {'id': range(20), 'value': [float(i) for i in range(20)]}
            )
        },
        primary_keys={'rows': 'id'},
    )
    schema['rows']['value'].stype = 'numerical'
    query = PredictiveQuery('PREDICT rows.value FOR EACH rows.id', schema)
    model = KumoRelational() if model_name == 'relational' else KumoTabular()
    return query, model


@pytest.mark.parametrize('model_name', ['relational', 'tabular'])
def test_public_materialize_requires_native_after_schema_and_query_exist(
    monkeypatch, model_name
):
    from sdm_client._rust import NativeBoundaryError, run

    query, model = _public_query_and_model(model_name)

    def unavailable():
        raise NativeBoundaryError('native engine unavailable for materialize')

    monkeypatch.setattr(run, 'native', unavailable)
    with pytest.raises(
        NativeBoundaryError, match='unavailable for materialize'
    ):
        query.materialize(
            indices=[0, 1],
            model=model,
            **({'random_seed': 7} if model_name == 'relational' else {}),
        )


@pytest.mark.parametrize('model_name', ['relational', 'tabular'])
def test_public_predict_requires_native_before_transport(
    monkeypatch, model_name
):
    import requests
    from sdm_client import SDMClient
    from sdm_client._rust import NativeBoundaryError, run

    query, model = _public_query_and_model(model_name)
    request = query.materialize(
        indices=[0, 1],
        model=model,
        **({'random_seed': 7} if model_name == 'relational' else {}),
    )

    def unavailable():
        raise NativeBoundaryError('native engine unavailable for predict')

    def forbidden_send(*args, **kwargs):
        pytest.fail('prediction reached HTTP without the native engine')

    with SDMClient(url='http://localhost:1') as client:
        monkeypatch.setattr(requests.Session, 'send', forbidden_send)
        monkeypatch.setattr(run, 'native', unavailable)
        with pytest.raises(
            NativeBoundaryError, match='unavailable for predict'
        ):
            client.predict(request, model=model)
