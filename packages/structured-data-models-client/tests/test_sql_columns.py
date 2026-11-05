# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle

import adbc_driver_manager
import duckdb
import pandas as pd
import pytest

from sdm_client import KumoRelational, PredictiveQuery, Schema
from sdm_client.errors import StructuredDataError


@pytest.fixture(params=['duckdb', 'sqlite'])
def source(request):
    from sdm_connectors import DuckDBSource, SQLiteSource

    if request.param == 'duckdb':
        import duckdb

        connection = duckdb.connect()
        source = DuckDBSource(connection)
    else:
        import adbc_driver_sqlite.dbapi

        connection = adbc_driver_sqlite.dbapi.connect()
        source = SQLiteSource(connection)
    cursor = connection.cursor()
    cursor.execute(
        'CREATE TABLE orders(id INTEGER PRIMARY KEY, quantity INTEGER, price DOUBLE)'
    )
    cursor.executemany(
        'INSERT INTO orders VALUES (?, ?, ?)',
        [(i, i + 1, None if i == 2 else 2.5) for i in range(30)],
    )
    if request.param == 'sqlite':
        connection.commit()
    cursor.close()
    try:
        yield source
    finally:
        connection.close()


def test_sql_column_values_ownership_refresh_and_materialization(source):
    original = Schema.from_source(
        source,
        {'orders': 'orders'},
        primary_keys={'orders': 'id'},
        relationships=[],
    )
    original['orders']['quantity'].stype = 'categorical'
    added = original.with_sql_column('orders', 'line_total', 'price * quantity')
    assert 'line_total' not in original['orders'].columns
    assert added['orders'].primary_key == 'id'
    assert added['orders']['quantity'].stype == 'categorical'
    added['orders']['line_total'].stype = 'numerical'
    added.relationships = []
    added = added.infer_metadata()
    expected = {i: (i + 1) * 2.5 for i in range(30) if i != 2}
    for _ in range(2):
        output = PredictiveQuery(
            'PREDICT orders.line_total FOR EACH orders.id', added
        ).labels(100)
        assert dict(zip(output.ENTITY, output.TARGET, strict=True)) == expected
    prepared = PredictiveQuery(
        'PREDICT orders.line_total FOR EACH orders.id', added
    ).materialize(indices=[0, 1], random_seed=7, model=KumoRelational())
    assert prepared.predict_rows == 2
    with pytest.raises(TypeError, match='open connection'):
        pickle.dumps(prepared)
    second = added.with_sql_column('orders', 'double_price', 'price * 2')
    assert 'double_price' not in added['orders'].columns
    assert 'line_total' in second['orders'].columns


def test_sql_column_errors_leave_source_and_schema_unchanged(source):
    original = Schema.from_source(
        source, {'orders': 'orders'}, primary_keys={'orders': 'id'}
    )
    for table, name, expression in [
        ('orders', 'price', '1'),
        ('missing', 'value', '1'),
        ('orders', '__kumo_bad', '1'),
        ('orders', '', '1'),
        ('orders', 'computed', 'absent_column * 2'),
    ]:
        with pytest.raises(
            (StructuredDataError, duckdb.Error, adbc_driver_manager.Error)
        ) as caught:
            original.with_sql_column(table, name, expression)
        if isinstance(caught.value, StructuredDataError):
            assert caught.value.code in {
                'INVALID_REQUEST',
                'UNSUPPORTED_REQUEST',
                'PROVIDER_ERROR',
            }
        assert list(original['orders'].columns) == ['id', 'quantity', 'price']
    added = original.with_sql_column('orders', 'line_total', 'price * quantity')
    with pytest.raises(
        (StructuredDataError, duckdb.Error, adbc_driver_manager.Error)
    ) as caught:
        added.with_sql_column('orders', 'chained', 'line_total * 2')
    if isinstance(caught.value, StructuredDataError):
        assert caught.value.code in {
            'INVALID_REQUEST',
            'UNSUPPORTED_REQUEST',
            'PROVIDER_ERROR',
        }
    quoted = added.with_sql_column('orders', 'quoted name', 'price + 1')
    assert 'quoted name' in quoted['orders'].columns


def test_in_memory_sql_refusal_is_explicit():
    schema = Schema.from_data(
        {'orders': pd.DataFrame({'id': [1], 'price': [2.5]})}
    )
    with pytest.raises(StructuredDataError, match='source-backed'):
        schema.with_sql_column('orders', 'line_total', 'price * 2')


def test_source_changes_cannot_hide_a_calculated_dependency_or_shadow_it(
    source,
):
    original = Schema.from_source(
        source, {'orders': 'orders'}, primary_keys={'orders': 'id'}
    )
    added = original.with_sql_column('orders', 'line_total', 'price * quantity')
    with source.connection.cursor() as cursor:
        cursor.execute('ALTER TABLE orders ADD COLUMN line_total DOUBLE')
    with pytest.raises(StructuredDataError, match='conflicts'):
        PredictiveQuery(
            'PREDICT orders.line_total FOR EACH orders.id', added
        ).labels(100)


def test_dropped_physical_dependency_is_refused_on_refresh(source):
    original = Schema.from_source(
        source, {'orders': 'orders'}, primary_keys={'orders': 'id'}
    )
    added = original.with_sql_column('orders', 'line_total', 'price * quantity')
    with source.connection.cursor() as cursor:
        cursor.execute(
            'ALTER TABLE orders RENAME COLUMN price TO previous_price'
        )
    with pytest.raises(
        (StructuredDataError, duckdb.Error, adbc_driver_manager.Error)
    ):
        PredictiveQuery(
            'PREDICT orders.line_total FOR EACH orders.id', added
        ).labels(100)


@pytest.mark.parametrize('source', ['sqlite'], indirect=True)
def test_sqlite_calculated_rowid_cannot_change_sampling_identity(source):
    original = Schema.from_source(
        source, {'orders': 'orders'}, primary_keys={'orders': 'id'}
    )
    with pytest.raises(StructuredDataError, match='rowid'):
        original.with_sql_column('orders', 'RoWiD', 'id + 100')
    assert 'RoWiD' not in original['orders'].columns


def test_engine_projection_preserves_filtered_aliases_and_indexed_key(source):
    """Exercise the engine relation contract beneath semantic-source adapters."""
    import json

    from sdm_client._rust import codec, providers
    from sdm_client._rust.run import run
    from sdm_client._rust.schema import build
    from sdm_client._rust.spec import held_schema

    with source.connection.cursor() as cursor:
        cursor.execute(
            'CREATE TABLE indexed_orders(quantity INTEGER NOT NULL UNIQUE, price DOUBLE)'
        )
        cursor.execute(
            'INSERT INTO indexed_orders SELECT quantity, price FROM orders'
        )
    relation = {
        'source': ['indexed_orders'],
        'columns': [
            {'name': 'entity', 'column': 'quantity'},
            {'name': 'price', 'column': 'price'},
        ],
        'filter': 'quantity > 10',
    }
    choice, provider = providers.for_source(source)
    described = run(
        {
            'op': 'describe_source',
            'provider': choice,
            'source': 1,
            'check_keys': True,
        },
        {
            1: codec.attachment(
                {
                    'tables': [{'name': 'sales', 'source': relation}],
                    'primary_keys': {},
                    'time_columns': {},
                    'relationships': [],
                }
            )
        },
        provider=provider,
    )
    document = json.loads(described.take(described.result['description']))
    assert document['tables'][0]['primary_key'] == ['entity']
    assert document['tables'][0]['duplicate_keys']['v'] == '0'
    held = held_schema(build(Schema, source, document))
    held['tables'][0]['source'] = relation
    result = run(
        {'op': 'add_sql_column', 'provider': choice, 'schema': 1, 'input': 2},
        {
            1: codec.attachment(held),
            2: codec.attachment(
                {
                    'table': 'sales',
                    'name': 'line_total',
                    'expression': 'quantity * price',
                }
            ),
        },
        provider=provider,
    )
    changed = json.loads(result.take(result.result['description']))['tables'][0]
    assert changed['source'] == relation
    assert changed['primary_key'] == ['entity']
    assert changed['calculated_columns'] == [
        {'name': 'line_total', 'expression': 'quantity * price'}
    ]
