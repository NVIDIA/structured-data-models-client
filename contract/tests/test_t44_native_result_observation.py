# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Semantic native owners supplement, and never overwrite, ordinal carriers."""

import copy

import pytest
from test_t44_prediction_observation import actual_observation, reader
from test_t136_forecast_capture import RUNNER


def presentation():
    return {
        'outputs': ['prediction'],
        'ranking': False,
        'rows': [
            {
                'row_index': i,
                'forecast_step': None,
                'id': {'i64': str(i + 7)},
                'rankings': None,
            }
            for i in range(2)
        ],
    }


def test_actual_builtin_result_storage_is_observed_without_custom_hooks():
    from sdm_api.generated.tfm_api import PredictionItem
    from sdm_client.result import PredictionResult

    result = PredictionResult(
        model='kumo-relational',
        request_id='actual',
        items=[
            PredictionItem(
                row_index=0,
                id=10**25,
                rankings=({'candidate': 9, 'score': 0.5},),
            )
        ],
    )
    assert RUNNER.native_result_snapshot(result) == {
        'outputs': [],
        'ranking': False,
        'rows': [
            {
                'row_index': 0,
                'forecast_step': None,
                'id': {'integer': str(10**25)},
                'rankings': [
                    {'candidate': {'i64': '9'}, 'score': '3fe0000000000000'}
                ],
            }
        ],
    }

    class Custom:
        def __getattribute__(self, name):
            raise AssertionError(name)

    object.__setattr__(result.items[0], 'id', Custom())
    assert RUNNER.native_result_snapshot(result) is None

    class CustomResult(PredictionResult):
        def __getattribute__(self, name):
            raise AssertionError(name)

    assert RUNNER.native_result_snapshot(object.__new__(CustomResult)) is None


@pytest.mark.parametrize('field', ['prediction', 'rankings'])
def test_ranking_candidate_companion_uses_the_actual_requested_native_field(
    field,
):
    from sdm_api.generated.tfm_api import PredictionItem
    from sdm_client.result import PredictionResult

    item = PredictionItem(
        row_index=0,
        id=7,
        **{
            field: [9, 8]
            if field == 'prediction'
            else (
                {'candidate': 9, 'score': 0.5},
                {'candidate': 8, 'score': 0.25},
            )
        },
    )
    result = PredictionResult(
        model='kumo-relational',
        request_id='actual',
        items=[item],
        outputs=[field],
    )
    shown = RUNNER.native_result_snapshot(result, ranking=True)['rows'][0]
    if field == 'prediction':
        assert shown['prediction_candidates'] == [{'i64': '9'}, {'i64': '8'}]
        assert shown['rankings'] is None
    else:
        assert 'prediction_candidates' not in shown
        assert [row['candidate'] for row in shown['rankings']] == [
            {'i64': '9'},
            {'i64': '8'},
        ]


@pytest.mark.parametrize('missing', ['prediction', 'rankings'])
def test_requested_native_ranking_field_cannot_disappear_behind_the_other(
    missing,
):
    from sdm_api.generated.tfm_api import PredictionItem
    from sdm_client.result import PredictionResult

    result = PredictionResult(
        model='kumo-relational',
        request_id='actual',
        outputs=['prediction', 'rankings'],
        items=[
            PredictionItem(
                row_index=0,
                id=7,
                prediction=[9],
                rankings=({'candidate': 9, 'score': 0.5},),
            )
        ],
    )
    shown = RUNNER.native_result_snapshot(result, ranking=True)
    table = {
        'columns': [
            {'name': name, 'type': kind}
            for name, kind in (
                ('answer_index', 'int64'),
                ('row_index', 'int64'),
                ('id', 'int64'),
                ('candidates', 'list'),
                ('scores', 'list'),
                ('rankings', 'list'),
            )
        ],
        'rows': [
            [
                {'i64': '0'},
                {'i64': '0'},
                {'i64': '0'},
                {'list': [{'i64': '0'}]},
                {'list': [{'f64_bits': '3fe0000000000000'}]},
                {'list': [{'utf8': 'null'}]},
            ]
        ],
    }
    reader()._presentation(
        shown, table, ranking=True, outputs=['prediction', 'rankings']
    )
    if missing == 'prediction':
        del shown['rows'][0]['prediction_candidates']
    else:
        shown['rows'][0]['rankings'] = None
    with pytest.raises(
        reader().ObservationError, match='requested native ranking'
    ):
        reader()._presentation(
            shown, table, ranking=True, outputs=['prediction', 'rankings']
        )


def test_native_semantic_projection_keeps_the_raw_ordinal_result():
    core, operations = actual_observation()
    final = core['predictions'][0]['results'][0]
    final['presentation'] = presentation()
    core['predicted']['presentation'] = copy.deepcopy(final['presentation'])
    for row in final['result']['rows']:
        row[1] = copy.deepcopy(row[0])
    core['predicted']['result'] = copy.deepcopy(final['result'])
    before = copy.deepcopy(core)
    value = reader().view(core, operations)['events'][0]['result']
    semantic = reader().semantic_result(value)
    assert [row[1] for row in semantic['result']['rows']] == [
        {'i64': '7'},
        {'i64': '8'},
    ]
    assert core == before
    assert final['result']['rows'][0][1] == {'i64': '0'}
    semantic['result']['rows'][1][1] = {'i64': '7'}
    assert semantic['result'] != reader().semantic_result(value)['result']


@pytest.mark.parametrize(
    'fault',
    ['short', 'row', 'extra', 'scalar', 'width', 'aggregation', 'outputs'],
)
def test_native_semantic_companion_is_complete_and_bound_to_actual_rows(fault):
    core, operations = actual_observation()
    shown = presentation()
    if fault == 'short':
        shown['rows'].pop()
    elif fault == 'row':
        shown['rows'][0]['row_index'] = 9
    elif fault == 'extra':
        shown['rows'][0]['guessed'] = True
    elif fault == 'scalar':
        shown['rows'][0]['id'] = {'i64': 'not an integer'}
    elif fault == 'width':
        shown['rows'][0]['id'] = {'i64': str(10**25)}
    elif fault == 'aggregation':
        shown['ranking'] = True
    else:
        shown['outputs'].append('rankings')
    core['predictions'][0]['results'][0]['presentation'] = shown
    core['predicted']['presentation'] = copy.deepcopy(shown)
    with pytest.raises(reader().ObservationError):
        reader().view(core, operations)


@pytest.mark.parametrize(
    'values',
    [
        (1.5, -0.0),
        (float('nan'), float('inf')),
        (True, False),
        (1, 'one'),
        (10**25, -(10**25)),
        (None, 1.5),
        (None, None),
    ],
)
def test_observed_native_ids_agree_with_complete_rust_written_ids(values):
    """The old float-key public journey also has a Rust utf8 carrier.

    Rust id_columns retains noninteger/mixed identities in id_as_written;
    that actual owner and a native presented owner need one semantic view.
    """
    import json

    from sdm_api.generated.tfm_api import PredictionItem
    from sdm_client.result import PredictionResult

    core, operations = actual_observation()
    actual = reader().view(core, operations)['events'][0]['result']
    native = copy.deepcopy(actual)
    native['presentation'] = RUNNER.native_result_snapshot(
        PredictionResult(
            model='kumo-relational',
            request_id='actual',
            outputs=['prediction'],
            items=[
                PredictionItem(row_index=i, id=v) for i, v in enumerate(values)
            ],
        )
    )
    rust = copy.deepcopy(actual)
    table = rust['result']
    names = [column['name'] for column in table['columns']]
    table['columns'][names.index('id')]['type'] = 'utf8'
    table['columns'].append({'name': 'id_as_written', 'type': 'utf8'})
    for row, value in zip(table['rows'], values, strict=True):
        written = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
        row[names.index('id')] = (
            None
            if value is None
            else {'utf8': value if type(value) is str else written}
        )
        row.append(None if value is None else {'utf8': written})
    before = copy.deepcopy((native, rust))
    assert reader().semantic_result(native) == reader().semantic_result(rust)
    assert (native, rust) == before
    rust['result']['rows'][0][names.index('id')] = {'utf8': 'another identity'}
    with pytest.raises(reader().ObservationError, match='written identity'):
        reader().semantic_result(rust)


@pytest.mark.parametrize('written', ['not JSON', '"another identity"'])
def test_written_identity_owner_cannot_disagree_with_its_actual_carrier(
    written,
):
    core, operations = actual_observation()
    actual = reader().view(core, operations)['events'][0]['result']
    table = actual['result']
    names = [column['name'] for column in table['columns']]
    table['columns'][names.index('id')]['type'] = 'utf8'
    table['columns'].append({'name': 'id_as_written', 'type': 'utf8'})
    for row in table['rows']:
        row[names.index('id')] = {'utf8': '1.5'}
        row.append({'utf8': written})
    with pytest.raises(reader().ObservationError, match='written identity'):
        reader().semantic_result(actual)


def test_nonprimitive_written_owners_and_absent_results_keep_their_exact_layout():
    for value in ({'result': None}, {'model': 'unobserved'}):
        assert reader().semantic_result(value) == value
    core, operations = actual_observation()
    actual = reader().view(core, operations)['events'][0]['result']
    table = actual['result']
    names = [column['name'] for column in table['columns']]
    table['columns'][names.index('id')]['type'] = 'utf8'
    table['columns'].append({'name': 'id_as_written', 'type': 'utf8'})
    for row in table['rows']:
        row[names.index('id')] = {'utf8': '[1]'}
        row.append({'utf8': '[1]'})
    assert reader().semantic_result(actual) == actual
