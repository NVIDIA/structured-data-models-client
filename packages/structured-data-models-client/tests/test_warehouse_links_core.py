# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warehouse link questions: one asked without candidates is refused before
any read, and a ranking is asked as one classification per candidate."""

from datetime import datetime, timedelta

import duckdb
import pandas as pd
import pytest
from adbc_driver_sqlite import dbapi as sqlite
from sdm_connectors import DuckDBSource, SQLiteSource

from sdm_client import KumoRelational, PredictiveQuery, Schema
from sdm_client.errors import UnsupportedRequestError

QUERY = (
    'PREDICT LIST_DISTINCT(orders.item_id, 0, 30, days) FOR EACH users.user_id'
)


@pytest.fixture(params=['duckdb', 'sqlite'])
def source(request):
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


def test_candidate_free_links_are_refused_before_any_read(source, source_reads):
    schema, _ = prepare(source)
    source_reads.clear()
    with pytest.raises(UnsupportedRequestError) as refused:
        materialize(schema)
    assert refused.value.details['rule'] == 'temporal_link_prediction'
    assert source_reads == []


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
