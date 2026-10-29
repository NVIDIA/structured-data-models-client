# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read actual prepared and nested materialized products without invented ops."""

from __future__ import annotations

import copy

import pytest

from sdm_client._rust import results
from sdm_client._rust.codec import CodecError
from sdm_client.errors import StructuredDataError


@pytest.mark.parametrize('attachment', [0, 17])
def test_prepared_materialize_is_its_own_typed_result(attachment):
    actual = {'op': 'prepare_materialize', 'value': attachment}
    decoded = results.decode(actual)
    assert type(decoded) is results.MaterializePrepared
    assert decoded.value == attachment
    assert results.expect(actual, results.MaterializePrepared) == decoded
    with pytest.raises(StructuredDataError) as caught:
        results.expect(actual, results.FramePrepared)
    assert caught.value.code == 'INTERNAL_ERROR'
    assert results.decode({'op': 'prepare_frames', 'value': attachment}) == (
        results.FramePrepared(attachment)
    )
    assert results.decode({'op': 'prepare_request', 'value': attachment}) == (
        results.Prepared(attachment)
    )


@pytest.mark.parametrize(
    'invalid',
    [
        {},
        {'value': True},
        {'value': -1},
        {'value': 1.0},
        {'value': '1'},
        {'value': None},
        {'value': 1, 'state': {}},
    ],
)
def test_prepared_materialize_refuses_missing_or_untyped_attachments(invalid):
    with pytest.raises(CodecError):
        results.decode({'op': 'prepare_materialize', **invalid})


def _materialized(kind='ordinary'):
    state = {'version': 2, 'payload': 11, 'query': 12}
    if kind == 'forecast':
        state['forecast'] = [
            {'step': 3, 'payload': 31, 'query': 32, 'request_id': 'later-3'},
            {'step': 2, 'payload': 21, 'query': 22},
        ]
    elif kind == 'batches':
        state['batches'] = [
            {'payload': 41, 'rows': 200, 'request_id': 'later-call'},
            {'payload': 42, 'rows': 7},
        ]
    elif kind == 'ranking':
        state['ranking'] = {
            'top_k': 2,
            'candidate': 1,
            'value': '"first"',
            'parts': [
                {
                    'candidate': 5,
                    'value': '"last"',
                    'payload': 51,
                    'query': 52,
                    'request_id': 'last-question',
                },
                {'candidate': 3, 'value': 'null', 'payload': 53, 'query': 54},
            ],
        }
    return {
        'state': state,
        'query': {
            'entity_table': 'items',
            'problem_type': 'regression',
            'windowed': False,
            'num_forecasts': 1,
            'text': 'resolved query',
        },
    }


@pytest.mark.parametrize('kind', ['ordinary', 'forecast', 'batches', 'ranking'])
def test_nested_materialized_reuses_complete_existing_state_validation(kind):
    actual = _materialized(kind)
    before = copy.deepcopy(actual)
    gathered = results.Gathered.from_contract(actual)
    assert actual == before
    assert type(gathered) is results.Gathered
    assert gathered.query == results.QueryFacts(
        entity_table='items',
        problem_type='regression',
        windowed=False,
        num_forecasts=1,
        text='resolved query',
    )
    assert gathered.state == results.GatheredState(
        version=2,
        payload=11,
        query=12,
        forecast=(results.Step(3, 31, 32), results.Step(2, 21, 22))
        if kind == 'forecast'
        else None,
        batches=(results.Batch(41, 200), results.Batch(42, 7))
        if kind == 'batches'
        else None,
        ranking=results.Ranking(
            top_k=2,
            candidate=1,
            value='"first"',
            parts=(
                results.Part(5, '"last"', 51, 52),
                results.Part(3, 'null', 53, 54),
            ),
        )
        if kind == 'ranking'
        else None,
    )
    # The legacy decoder and nested reader share one validation path.
    assert results.decode({'op': 'materialize', **actual}) == gathered


@pytest.mark.parametrize('explicit_null', [False, True])
def test_nested_optional_query_and_state_parts_keep_existing_none_meaning(
    explicit_null,
):
    actual = {'state': {'version': 2, 'payload': 11, 'query': 12}}
    if explicit_null:
        actual['query'] = None
        actual['state'].update(forecast=None, batches=None, ranking=None)
    gathered = results.Gathered.from_contract(actual)
    assert gathered == results.Gathered(
        state=results.GatheredState(2, 11, 12, None, None, None),
        query=None,
    )
    assert results.decode({'op': 'materialize', **actual}) == gathered


@pytest.mark.parametrize(
    ('kind', 'path', 'invalid'),
    [
        ('ordinary', ('state',), None),
        ('ordinary', ('state', 'version'), True),
        ('ordinary', ('state', 'payload'), -1),
        ('ordinary', ('state', 'query'), '12'),
        ('ordinary', ('state', 'extra'), 0),
        ('ordinary', ('extra',), {}),
        ('ordinary', ('op',), 'materialize'),
        ('ordinary', ('query', 'windowed'), 0),
        ('ordinary', ('query', 'num_forecasts'), -1),
        ('ordinary', ('query', 'problem_type'), 5),
        ('ordinary', ('query', 'text'), []),
        ('ordinary', ('query', 'extra'), None),
        ('forecast', ('state', 'forecast'), {}),
        ('forecast', ('state', 'forecast', 0, 'step'), -1),
        ('forecast', ('state', 'forecast', 0, 'payload'), False),
        ('forecast', ('state', 'forecast', 0, 'extra'), None),
        ('batches', ('state', 'batches'), {}),
        ('batches', ('state', 'batches', 0, 'rows'), True),
        ('batches', ('state', 'batches', 0, 'payload'), -1),
        ('batches', ('state', 'batches', 0, 'extra'), None),
        ('ranking', ('state', 'ranking', 'top_k'), True),
        ('ranking', ('state', 'ranking', 'candidate'), -1),
        ('ranking', ('state', 'ranking', 'value'), None),
        ('ranking', ('state', 'ranking', 'parts'), {}),
        ('ranking', ('state', 'ranking', 'parts', 0, 'query'), '52'),
        ('ranking', ('state', 'ranking', 'parts', 0, 'extra'), None),
    ],
)
def test_nested_materialized_rejects_malformed_complete_products(
    kind,
    path,
    invalid,
):
    actual = _materialized(kind)
    parent = actual
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = invalid
    with pytest.raises(CodecError):
        results.Gathered.from_contract(actual)
    # The operation envelope itself is not a field of nested Materialized.
    if path != ('op',):
        with pytest.raises(CodecError):
            results.decode({'op': 'materialize', **actual})


@pytest.mark.parametrize(
    ('kind', 'path'),
    [
        ('ordinary', ('state',)),
        ('ordinary', ('state', 'version')),
        ('ordinary', ('query', 'entity_table')),
        ('forecast', ('state', 'forecast', 0, 'query')),
        ('batches', ('state', 'batches', 0, 'rows')),
        ('ranking', ('state', 'ranking', 'parts', 0, 'value')),
    ],
)
def test_nested_materialized_cannot_omit_required_inner_fields(kind, path):
    actual = _materialized(kind)
    parent = actual
    for key in path[:-1]:
        parent = parent[key]
    del parent[path[-1]]
    with pytest.raises(CodecError):
        results.Gathered.from_contract(actual)
    with pytest.raises(CodecError):
        results.decode({'op': 'materialize', **actual})


def test_nested_materialized_never_invents_an_operation_envelope():
    actual = _materialized()
    with pytest.raises(CodecError, match='names no operation'):
        results.decode(actual)
    with pytest.raises(CodecError):
        results.Gathered.from_contract({'state': actual['state'], 'value': 17})
