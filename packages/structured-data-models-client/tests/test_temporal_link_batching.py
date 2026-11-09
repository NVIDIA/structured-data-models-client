# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public PQL batching obeys the model limit before any prediction is sent."""

from datetime import datetime, timedelta

import duckdb
import pandas as pd
import pytest
from adbc_driver_sqlite import dbapi as sqlite
from sdm_connectors import DuckDBSource, SQLiteSource

from sdm_client import KumoRelational, PredictiveQuery, Schema, SDMClient
from sdm_client.errors import StructuredDataError

QUERY = (
    'PREDICT LIST_DISTINCT(orders.item_id, 0, 30, days) FOR EACH users.user_id'
)


@pytest.fixture(params=['memory', 'duckdb', 'sqlite'])
def link_schema(request):
    frames = {
        'users': pd.DataFrame({'user_id': range(1, 202)}),
        'items': pd.DataFrame({'item_id': range(1, 6), 'price': range(5)}),
        'orders': pd.DataFrame(
            [
                (
                    i,
                    i % 201 + 1,
                    i % 5 + 1,
                    datetime(2024, 1, 1) + timedelta(days=i % 100),
                )
                for i in range(1005)
            ],
            columns=['order_id', 'user_id', 'item_id', 'time'],
        ),
    }
    options = dict(
        primary_keys={
            'users': 'user_id',
            'items': 'item_id',
            'orders': 'order_id',
        },
        time_columns={'orders': 'time'},
    )
    if request.param == 'memory':
        yield Schema.from_data(frames, **options)
        return
    connection = (
        duckdb.connect() if request.param == 'duckdb' else sqlite.connect()
    )
    try:
        with connection.cursor() as cursor:
            for statement in (
                'CREATE TABLE users(user_id BIGINT PRIMARY KEY)',
                'CREATE TABLE items(item_id BIGINT PRIMARY KEY, price DOUBLE)',
                'CREATE TABLE orders(order_id BIGINT PRIMARY KEY, '
                'user_id BIGINT REFERENCES users(user_id), '
                'item_id BIGINT REFERENCES items(item_id), time TIMESTAMP)',
            ):
                cursor.execute(statement)
            for name, frame in frames.items():
                cursor.executemany(
                    f'INSERT INTO {name} VALUES ({",".join("?" for _ in frame.columns)})',
                    [
                        tuple(
                            v.isoformat()
                            if isinstance(v, (datetime, pd.Timestamp))
                            else v
                            for v in row
                        )
                        for row in frame.itertuples(index=False, name=None)
                    ],
                )
        connection.commit()
        source = (
            DuckDBSource(connection)
            if request.param == 'duckdb'
            else SQLiteSource(connection)
        )
        yield Schema.from_source(
            source, {name: name for name in frames}, **options
        )
    finally:
        connection.close()


def materialize(schema, count=201, rows_per_call=None, query=QUERY):
    return PredictiveQuery(query, schema).materialize(
        indices=list(range(1, count + 1)),
        model=KumoRelational(),
        anchor_time=pd.Timestamp('2024-03-01'),
        num_context_samples=10,
        random_seed=7,
        rows_per_call=rows_per_call,
    )


@pytest.mark.parametrize(
    ('count', 'limit', 'sizes'),
    [
        (199, None, [199]),
        (200, None, [200]),
        (201, None, [200, 1]),
        (201, 100, [100, 100, 1]),
        (201, 1000, [200, 1]),
    ],
)
def test_pql_link_batches_preserve_every_entity(
    link_schema, count, limit, sizes
):
    request = materialize(link_schema, count, limit)
    assert [
        len(p['predict']['instance_table']['rows']) for p in request.payloads
    ] == sizes
    entities = []
    for payload in request.payloads:
        assert payload['task']['kind'] == 'temporal_link_prediction'
        table = payload['predict']['instance_table']
        column = table['columns'].index('__kumo_entity_ref_0')
        entities.extend(int(row[column]) for row in table['rows'])
    assert entities == list(range(1, count + 1))


def test_node_predictions_keep_the_ordinary_limit(link_schema):
    request = materialize(
        link_schema,
        query='PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id',
    )
    assert [
        len(p['predict']['instance_table']['rows']) for p in request.payloads
    ] == [201]


def test_link_batches_respect_total_call_limit(link_schema, monkeypatch):
    from sdm_client.materializers import relational

    monkeypatch.setattr(relational, 'MOST_CALLS', 1)
    with pytest.raises(StructuredDataError) as caught:
        materialize(link_schema)
    assert caught.value.code == 'UNSUPPORTED_REQUEST'
    assert caught.value.details['calls'] == 2
    assert caught.value.details['rows_per_call'] == 200


def test_link_batch_result_order_and_partial_failure(
    link_schema, requests_mock
):
    request = materialize(link_schema)
    calls = []
    fail = False

    def answer(call, response):
        payload = call.json()
        table = payload['predict']['instance_table']
        calls.append(len(table['rows']))
        if fail and len(calls) == 2:
            response.status_code = 500
            return {'error': 'second batch failed'}
        identity = table['columns'].index('instance_id')
        return {
            'model': 'kumo-relational',
            'id': 'links',
            'metadata': {},
            'predictions': [
                {
                    'row_index': i,
                    'id': str(row[identity]),
                    'rankings': [{'item': '3', 'score': 0.9}],
                }
                for i, row in reversed(list(enumerate(table['rows'])))
            ],
        }

    requests_mock.post('http://localhost/v1/predictions', json=answer)
    requests_mock.post('http://localhost/v1/sessions', status_code=405)
    with SDMClient('http://localhost', max_retries=0) as client:
        result = client.predict(request, model=KumoRelational())
        assert calls == [200, 1]
        assert result.to_pandas()['id'].astype(int).tolist() == list(
            range(1, 202)
        )
        calls.clear()
        fail = True
        with pytest.raises(StructuredDataError) as caught:
            client.predict(request, model=KumoRelational())
        assert calls == [200, 1]
        answered = caught.value.details['answered']
        assert answered.to_pandas()['id'].astype(int).tolist() == list(
            range(1, 201)
        )
