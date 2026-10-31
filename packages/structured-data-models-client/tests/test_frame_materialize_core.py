# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public local graph materialization with a retained Rust snapshot."""

import json
import warnings

import pandas as pd
import pytest

from sdm_client import KumoRelational, PredictiveQuery, Schema, _routing


@pytest.fixture(autouse=True)
def frame_routes(monkeypatch):
    monkeypatch.setenv(_routing.ENVIRONMENT, ','.join(_routing.FRAME_ROUTES))


def materialize(query):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        request = query.materialize(
            indices=[1, 2], random_seed=7, model=KumoRelational()
        )
    return json.loads(json.dumps(request._payloads, default=str))


def schema():
    result = Schema.from_data(
        {
            'users': pd.DataFrame(
                {'id': range(1, 21), 'value': [float(i) for i in range(20)]}
            )
        },
        primary_keys={'users': 'id'},
    )
    result['users']['value'].stype = 'numerical'
    return result


def test_static_frame_materialization_is_repeatable():
    query = PredictiveQuery('PREDICT users.value FOR EACH users.id', schema())
    first = materialize(query)
    assert len(first) == 1
    assert first == materialize(query)


def test_materializer_snapshot_does_not_alias_input_buffers():
    data = schema()
    query = PredictiveQuery('PREDICT users.value FOR EACH users.id', data)
    first = materialize(query)
    data['users'].data.loc[:, 'value'] = 999.0
    assert materialize(query) == first
    assert (
        materialize(
            PredictiveQuery('PREDICT users.value FOR EACH users.id', data)
        )
        != first
    )


def test_static_payload_matches_old_client(monkeypatch):
    data = schema()
    first = materialize(
        PredictiveQuery('PREDICT users.value FOR EACH users.id', data)
    )
    monkeypatch.setenv(_routing.ENVIRONMENT, '')
    old = materialize(
        PredictiveQuery('PREDICT users.value FOR EACH users.id', data)
    )
    # The old local sampler overflows Timestamp.min when it floors to seconds
    # and multiplies back to nanoseconds, inventing a timestamp in 2262.
    # Rust correctly leaves the no-anchor sentinel out of the request.
    assert 'anchor_time_column' not in first[0]['task']
    for payload in old:
        payload['task'].pop('anchor_time_column')
        payload['schema']['instance_table']['columns'].pop('__kumo_anchor_time')
        for side in ('context', 'predict'):
            table = payload[side]['instance_table']
            position = table['columns'].index('__kumo_anchor_time')
            table['columns'].pop(position)
            for row in table['rows']:
                assert row.pop(position) == '2262-04-11T23:47:16.709551Z'
    assert first == old


def test_temporal_frame_materialization():
    data = Schema.from_data(
        {
            'users': pd.DataFrame(
                {'user_id': range(1, 21), 'age': range(20, 40)}
            ),
            'orders': pd.DataFrame(
                {
                    'order_id': range(100),
                    'user_id': [i % 20 + 1 for i in range(100)],
                    'price': [float(i % 10) for i in range(100)],
                    'time': pd.date_range('2024-01-01', periods=100, freq='D'),
                }
            ),
        },
        primary_keys={'users': 'user_id', 'orders': 'order_id'},
        time_columns={'orders': 'time'},
    )
    query = PredictiveQuery(
        'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id', data
    )
    first = materialize(query)
    assert len(first) == 1
    assert first == materialize(query)


@pytest.mark.parametrize('rank', ['', ' RANK TOP 2'])
def test_link_prediction_frame_materialization(rank):
    data = Schema.from_data(
        {
            'users': pd.DataFrame({'user_id': range(1, 21)}),
            'items': pd.DataFrame({'item_id': range(1, 6), 'price': range(5)}),
            'orders': pd.DataFrame(
                {
                    'order_id': range(100),
                    'user_id': [i % 20 + 1 for i in range(100)],
                    'item_id': [i % 5 + 1 for i in range(100)],
                    'time': pd.date_range('2024-01-01', periods=100, freq='D'),
                }
            ),
        },
        primary_keys={
            'users': 'user_id',
            'items': 'item_id',
            'orders': 'order_id',
        },
        time_columns={'orders': 'time'},
    )
    query = PredictiveQuery(
        f'PREDICT LIST_DISTINCT(orders.item_id, 0, 30, days){rank} FOR EACH users.user_id',
        data,
    )
    first = materialize(query)
    assert len(first) == 1
    assert first[0]['task']['kind'] == 'temporal_link_prediction'
    assert first == materialize(query)


def test_composite_neighbor_identity_is_materialized():
    data = Schema.from_data(
        {
            'users': pd.DataFrame(
                {
                    'user_id': range(1, 21),
                    'value': [float(i) for i in range(20)],
                }
            ),
            'orders': pd.DataFrame(
                {
                    'user_id': [i % 20 + 1 for i in range(60)],
                    'part': [i // 20 for i in range(60)],
                    'amount': [float(i) for i in range(60)],
                }
            ),
        },
        primary_keys={'users': 'user_id', 'orders': ['user_id', 'part']},
    )
    data['users']['value'].stype = 'numerical'
    first = materialize(
        PredictiveQuery('PREDICT users.value FOR EACH users.user_id', data)
    )
    assert len(first) == 1
    assert 'orders' in first[0]['schema']['related_tables']
