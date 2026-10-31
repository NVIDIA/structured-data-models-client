# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct warehouse link tasks preserve exact targets and historical context."""

from datetime import datetime, timedelta

import duckdb
import pandas as pd
import pytest
from adbc_driver_sqlite import dbapi as sqlite
from sdm_connectors import DuckDBSource, SQLiteSource

from sdm_client import (
    KumoRelational,
    PredictiveQuery,
    Schema,
    SDMClient,
    _routing,
)

QUERY = (
    'PREDICT LIST_DISTINCT(orders.item_id, 0, 30, days) FOR EACH users.user_id'
)


@pytest.fixture(params=['duckdb', 'sqlite'])
def source(request, monkeypatch):
    monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
    connection = (
        duckdb.connect() if request.param == 'duckdb' else sqlite.connect()
    )
    try:
        yield (
            connection,
            DuckDBSource(connection)
            if request.param == 'duckdb'
            else SQLiteSource(connection),
        )
    finally:
        connection.close()


def prepare(source, key_kind='integer'):
    connection, provider = source
    items = {
        'integer': list(range(1, 6)),
        'large_integer': [2**53 + i for i in range(1, 6)],
        'text': [f'sku-{i}' for i in range(1, 6)],
    }[key_kind]
    sql_type = 'VARCHAR' if key_kind == 'text' else 'BIGINT'
    cursor = connection.cursor()
    cursor.execute('CREATE TABLE users(user_id BIGINT PRIMARY KEY)')
    cursor.executemany(
        'INSERT INTO users VALUES (?)', [(i,) for i in range(1, 21)]
    )
    cursor.execute(
        f'CREATE TABLE items(item_id {sql_type} PRIMARY KEY, price DOUBLE)'
    )
    cursor.executemany(
        'INSERT INTO items VALUES (?, ?)',
        [(key, float(i)) for i, key in enumerate(items)],
    )
    cursor.execute(
        'CREATE TABLE orders(order_id BIGINT PRIMARY KEY, '
        'user_id BIGINT REFERENCES users(user_id), '
        f'item_id {sql_type} REFERENCES items(item_id), time TIMESTAMP)'
    )
    rows = [
        (
            i,
            i % 20 + 1,
            items[(i // 20) % 5],
            datetime(2024, 1, 1) + timedelta(days=i),
        )
        for i in range(100)
    ]
    cursor.executemany(
        'INSERT INTO orders VALUES (?, ?, ?, ?)',
        [(*row[:3], row[3].isoformat()) for row in rows],
    )
    connection.commit()
    cursor.close()
    schema = Schema.from_source(
        provider,
        {'users': 'users', 'items': 'items', 'orders': 'orders'},
        primary_keys={
            'users': 'user_id',
            'items': 'item_id',
            'orders': 'order_id',
        },
        time_columns={'orders': 'time'},
    )
    return schema, rows


def materialize(schema, query=QUERY, **options):
    return PredictiveQuery(query, schema).materialize(
        indices=[1, 2],
        model=KumoRelational(),
        anchor_time=pd.Timestamp('2024-03-01'),
        num_context_samples=10,
        random_seed=7,
        **options,
    )


@pytest.mark.parametrize('key_kind', ['integer', 'large_integer', 'text'])
def test_link_labels_and_related_rows_obey_each_example_time(source, key_kind):
    schema, orders = prepare(source, key_kind)
    request = materialize(schema)
    assert len(request.payloads) == 1
    payload = request.payloads[0]
    assert payload['task']['kind'] == 'temporal_link_prediction'
    assert payload['task']['entity_table_names'] == ['users', 'items']
    assert payload['task']['target']['dtype'] == 'stringlist'
    anchor_name = payload['task']['anchor_time_column']
    for half in ('context', 'predict'):
        instances = payload[half]['instance_table']
        columns = instances['columns']
        anchors = {}
        for row in instances['rows']:
            values = dict(zip(columns, row, strict=True))
            anchor = pd.Timestamp(values[anchor_name])
            anchors[values['instance_id']] = anchor
            if half == 'context':
                expected = {
                    str(item)
                    for _, user, item, when in orders
                    if user == values['__kumo_entity_ref_0']
                    and anchor < when <= anchor + pd.Timedelta(days=30)
                }
                assert set(values['TARGET']) == expected
        assert anchors
        related = payload[half]['related_tables']['orders']
        for row in related['rows']:
            values = dict(zip(related['columns'], row, strict=True))
            assert (
                pd.Timestamp(values['time']) <= anchors[values['instance_id']]
            )


def test_link_prediction_and_session_results_use_rankings(
    source, requests_mock
):
    schema, _ = prepare(source)
    request = materialize(schema)

    def answer(call, context):
        sent = call.json()
        assert sent['output']['fields'] == ['rankings']
        rows = sent['predict']['instance_table']
        identity = rows['columns'].index('instance_id')
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
                for i, row in reversed(list(enumerate(rows['rows'])))
            ],
        }

    requests_mock.post('http://localhost/v1/predictions', json=answer)
    requests_mock.post(
        'http://localhost/v1/sessions', json={'session_id': 'links'}
    )
    requests_mock.post(
        'http://localhost/v1/sessions/links/predictions', json=answer
    )
    deleted = requests_mock.delete(
        'http://localhost/v1/sessions/links', status_code=204
    )
    with SDMClient('http://localhost', max_retries=0) as client:
        direct = client.predict(request, model=KumoRelational())
        with client.fit(request, model=KumoRelational()) as fitted:
            scored = fitted.predict(request)
        pd.testing.assert_frame_equal(direct.to_pandas(), scored.to_pandas())
        assert direct.items == scored.items
    assert deleted.call_count == 1


def test_rank_top_keeps_candidate_classification_path(source):
    schema, _ = prepare(source)
    request = materialize(
        schema,
        QUERY.replace(' FOR EACH', ' RANK TOP 2 FOR EACH'),
        candidates=[1, 2, 3],
    )
    assert len(request.parts) == 3
    assert all(
        part.payloads[0]['task']['kind'] == 'binary_classification'
        for part in request.parts
    )
