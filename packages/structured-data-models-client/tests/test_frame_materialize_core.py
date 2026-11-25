# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public local graph materialization with a retained Rust snapshot."""

import json
import warnings

import pandas as pd
import pytest

from sdm_client import KumoRelational, PredictiveQuery, Schema
from sdm_client.errors import StructuredDataError, UnsupportedRequestError


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


def test_static_payload_omits_the_no_anchor_sentinel():
    payload = materialize(
        PredictiveQuery('PREDICT users.value FOR EACH users.id', schema())
    )[0]
    assert payload['task'] == {
        'kind': 'regression',
        'target': {'column_name': 'TARGET', 'dtype': 'float32'},
        'entity_table_names': ['users'],
    }
    scoring = payload['predict']['instance_table']
    assert scoring['columns'] == ['instance_id', '__kumo_entity_ref_0']
    assert scoring['rows'] == [[20, 1], [21, 2]]
    context = payload['context']['instance_table']
    assert context['columns'] == [
        'instance_id',
        '__kumo_entity_ref_0',
        'TARGET',
    ]
    assert sorted((row[1], row[2]) for row in context['rows']) == [
        (key, float(key - 1)) for key in range(3, 21)
    ]
    assert 'anchor_time_column' not in payload['task']
    assert (
        '__kumo_anchor_time'
        not in payload['schema']['instance_table']['columns']
    )
    for side in ('context', 'predict'):
        assert (
            '__kumo_anchor_time'
            not in payload[side]['instance_table']['columns']
        )


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


def link_schema():
    return Schema.from_data(
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


def test_link_prediction_frame_materialization():
    data = link_schema()
    query = PredictiveQuery(
        'PREDICT LIST_DISTINCT(orders.item_id, 0, 30, days) FOR EACH users.user_id',
        data,
    )
    with pytest.raises(UnsupportedRequestError, match='RANK TOP k'):
        materialize(query)
    ranked = PredictiveQuery(
        'PREDICT LIST_DISTINCT(orders.item_id, 0, 30, days) RANK TOP 2 '
        'FOR EACH users.user_id',
        data,
    )
    with pytest.raises(StructuredDataError, match='candidates'):
        materialize(ranked)
    request = ranked.materialize(
        indices=[1, 2],
        random_seed=7,
        model=KumoRelational(),
        candidates=[1, 2, 3],
    )
    assert len(request.parts) == 3


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


@pytest.mark.parametrize('seed', range(8))
def test_generated_temporal_context_matches_independent_join_sum(seed):
    import random

    rng = random.Random(seed)
    count = rng.randrange(8, 18)
    width = rng.choice([3, 7, 14])
    keys = list(range(1, count + 1))
    orders = pd.DataFrame(
        {
            'order_id': range(100),
            'user_id': pd.array(
                [rng.choice(keys + [None, count + 100]) for _ in range(100)],
                dtype='Int64',
            ),
            'price': [float(rng.randrange(-3, 10)) for _ in range(100)],
            'time': [
                pd.Timestamp('2024-01-01')
                + pd.Timedelta(days=rng.randrange(60))
                for _ in range(100)
            ],
        }
    )
    requested = rng.sample(keys, 3)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data = Schema.from_data(
            {'users': pd.DataFrame({'user_id': keys}), 'orders': orders},
            primary_keys={'users': 'user_id', 'orders': 'order_id'},
            time_columns={'orders': 'time'},
        )
        request = PredictiveQuery(
            f'PREDICT SUM(orders.price, 0, {width}, days) FOR EACH users.user_id',
            data,
        ).materialize(
            indices=requested,
            model=KumoRelational(),
            random_seed=seed,
            num_context_samples=count,
            anchor_time=pd.Timestamp('2024-02-15'),
        )
    payload = request._payloads[0]
    context = payload['context']['instance_table']
    columns = context['columns']
    observed_keys = []
    for row in context['rows']:
        entity = row[columns.index('__kumo_entity_ref_0')]
        anchor = pd.Timestamp(row[columns.index('__kumo_anchor_time')])
        upper = anchor + pd.Timedelta(days=width)
        # Independently join source rows and apply the public (start, end] window.
        expected = sum(
            record.price
            for record in orders.itertuples(index=False)
            if not pd.isna(record.user_id)
            and record.user_id == entity
            and anchor < record.time.tz_localize('UTC') <= upper
        )
        assert row[columns.index('TARGET')] == expected
        observed_keys.append(entity)
    assert sorted(observed_keys) == keys
    scoring = payload['predict']['instance_table']
    entity_column = scoring['columns'].index('__kumo_entity_ref_0')
    assert [row[entity_column] for row in scoring['rows']] == requested


def test_sanitization_report_is_durable_and_includes_clean_tables():
    import dataclasses
    import pickle

    data = schema()
    query = PredictiveQuery('PREDICT users.value FOR EACH users.id', data)
    request = query.materialize(
        indices=[1, 2], random_seed=7, model=KumoRelational()
    )
    report = request.sanitization_report
    assert report.status == 'available'
    assert len(report.tables) == 1
    table = report.tables[0]
    assert (table.table, table.input_rows, table.output_rows) == (
        'users',
        20,
        20,
    )
    assert (
        table.null_primary_key_rows,
        table.duplicate_primary_key_rows,
        table.null_time_rows,
    ) == (0, 0, 0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        table.input_rows = 0
    data['users'].data.loc[:, 'value'] = 999.0
    again = query.materialize(
        indices=[1, 2], random_seed=7, model=KumoRelational()
    )
    assert again.sanitization_report == report
    assert pickle.loads(pickle.dumps(request)).sanitization_report == report


def test_sanitization_exclusive_drop_precedence_and_no_key_table():
    users = pd.DataFrame(
        {'id': range(20), 'value': [float(i) for i in range(20)]}
    )
    events = pd.DataFrame(
        {
            'id': pd.array(range(30), dtype='Int64'),
            'user_id': [i % 20 for i in range(30)],
            'time': pd.date_range('2024-01-01', periods=30),
        }
    )
    no_key = pd.DataFrame(
        {'value': [1.0, 2.0], 'time': pd.date_range('2024-01-01', periods=2)}
    )
    data = Schema.from_data(
        {'users': users, 'events': events, 'no_key': no_key},
        primary_keys={'users': 'id', 'events': 'id'},
        time_columns={'events': 'time', 'no_key': 'time'},
    )
    data['users']['value'].stype = 'numerical'
    data['events'].data.loc[25:, 'id'] = [25, 25, 26, 26, pd.NA]
    data['events'].data.loc[[25, 28, 29], 'time'] = pd.NaT
    data['no_key'].data.loc[1, 'time'] = pd.NaT
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter('always')
        request = PredictiveQuery(
            'PREDICT users.value FOR EACH users.id', data
        ).materialize(indices=[1, 2], random_seed=7, model=KumoRelational())
    tables = {row.table: row for row in request.sanitization_report.tables}
    row = tables['events']
    assert (
        row.input_rows,
        row.output_rows,
        row.null_primary_key_rows,
        row.duplicate_primary_key_rows,
        row.null_time_rows,
    ) == (30, 26, 1, 2, 1)
    row = tables['no_key']
    assert (
        row.input_rows,
        row.output_rows,
        row.null_primary_key_rows,
        row.duplicate_primary_key_rows,
        row.null_time_rows,
    ) == (2, 1, 0, 0, 1)
    assert tables['users'].output_rows == 20
    assert any('4 of 30 rows dropped' in str(w.message) for w in emitted)

    # Nonzero reasons must survive persistence too, not just clean zero counts.
    import pickle

    restored = pickle.loads(pickle.dumps(request))
    assert restored.sanitization_report == request.sanitization_report
    assert restored.payloads == request.payloads


def test_legacy_saved_requests_report_unavailable_or_not_applicable():
    import pickle

    from sdm_client import Request

    data = schema()
    prepared = PredictiveQuery(
        'PREDICT users.value FOR EACH users.id', data
    ).materialize(indices=[1], random_seed=7, model=KumoRelational())
    del prepared._sanitization_report
    assert (
        pickle.loads(pickle.dumps(prepared)).sanitization_report.status
        == 'unavailable'
    )
    direct = Request(
        pd.DataFrame({'x': [1.0, 2.0], 'y': [3.0, 4.0]}),
        pd.DataFrame({'x': [5.0]}),
        target='y',
        task='regression',
    )
    assert direct.sanitization_report.status == 'not_applicable'
    del direct._sanitization_report
    assert (
        pickle.loads(pickle.dumps(direct)).sanitization_report.status
        == 'not_applicable'
    )


@pytest.mark.parametrize('ranking', [False, True])
def test_decomposed_requests_retain_sanitization_report(ranking):
    data = link_schema()
    text = (
        'PREDICT LIST_DISTINCT(orders.item_id, 0, 30, days) RANK TOP 2 '
        if ranking
        else 'PREDICT COUNT(orders.*, 0, 30, days) FORECAST 2 TIMEFRAMES '
    ) + 'FOR EACH users.user_id'
    request = PredictiveQuery(text, data).materialize(
        indices=[1, 2],
        random_seed=7,
        model=KumoRelational(),
        **({'candidates': [1, 2, 3]} if ranking else {}),
    )
    assert request.sanitization_report.status == 'available'
    assert request.parts
    assert all(
        part.sanitization_report == request.sanitization_report
        for part in request.parts
    )
