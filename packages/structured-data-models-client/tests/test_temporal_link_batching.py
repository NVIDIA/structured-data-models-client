# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Candidate-free temporal link prediction is refused before anything is read.

No released Kumo Relational model serves it, so building its request would
only spend a source's reads on a certain refusal.
"""

from datetime import datetime, timedelta

import duckdb
import pandas as pd
import pytest
from adbc_driver_sqlite import dbapi as sqlite
from sdm_connectors import DuckDBSource, SQLiteSource

from sdm_client import KumoRelational, PredictiveQuery, Schema, SDMClient
from sdm_client.errors import UnsupportedRequestError

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


def assert_refused(refused):
    assert refused.value.code == 'UNSUPPORTED_REQUEST'
    assert refused.value.details['rule'] == 'temporal_link_prediction'
    assert 'RANK TOP k with candidates=' in str(refused.value)


def test_candidate_free_link_prediction_is_refused_before_any_read(
    link_schema, source_reads
):
    source_reads.clear()
    with pytest.raises(UnsupportedRequestError) as refused:
        materialize(link_schema)
    assert_refused(refused)
    assert source_reads == []


def test_node_predictions_keep_the_ordinary_limit(link_schema, source_reads):
    source_reads.clear()
    request = materialize(
        link_schema,
        query='PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id',
    )
    assert [
        len(p['predict']['instance_table']['rows']) for p in request.payloads
    ] == [201]
    assert bool(source_reads) == (link_schema.source is not None)


def test_a_request_naming_the_link_task_is_refused_before_it_is_sent(
    link_schema, requests_mock
):
    """As a request saved by 1.0.0, which built one, would be."""
    request = materialize(
        link_schema,
        query='PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id',
    )
    request._core = None
    request._payloads[0]['task']['kind'] = 'temporal_link_prediction'
    requests_mock.post('http://localhost/v1/predictions', json={})
    requests_mock.post('http://localhost/v1/sessions', json={})
    with SDMClient('http://localhost', max_retries=0) as client:
        with pytest.raises(UnsupportedRequestError) as refused:
            client.predict(request, model=KumoRelational())
    assert_refused(refused)
    assert requests_mock.call_count == 0
