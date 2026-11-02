# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warehouse sampling respects selected source columns and exact key values."""

from __future__ import annotations

import os
import uuid
import warnings
from collections.abc import Iterator
from typing import Any

import pytest
from sdm_connectors import PostgresSource
from test_schema_copy_provenance import (
    ANCHOR,
    QUERY,
    _copies,
    _described,
    _shop,
)

from sdm_client import (
    Column,
    PredictiveQuery,
    Relationship,
    Schema,
    SourceTable,
    StructuredDataError,
)

NUMERIC_IDS = (90000000000000001, 90000000000000002)


def materialize(schema, indices=(0, 1, 2)):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return PredictiveQuery(QUERY, schema).materialize(
            indices=list(indices),
            anchor_time=ANCHOR,
        )


def _hand_built(source: Any, **extra: Column) -> Schema:
    r"""The shop written out by hand, column by column, as a user would
    write it, with ``extra`` columns on ``users``."""
    return Schema(
        {
            'users': SourceTable(
                'users',
                'users',
                {
                    'user_id': Column('user_id', 'int64', 'ID'),
                    'age': Column('age', 'int64', 'numerical'),
                    **extra,
                },
                primary_key='user_id',
            ),
            'orders': SourceTable(
                'orders',
                'orders',
                {
                    'order_id': Column('order_id', 'int64', 'ID'),
                    'user_id': Column('user_id', 'int64', 'ID'),
                    'price': Column('price', 'float64', 'numerical'),
                    'ts': Column('ts', 'timestamp[us]', 'timestamp'),
                },
                primary_key='order_id',
                time_column='ts',
            ),
        },
        [Relationship('orders', 'user_id', 'users', 'user_id')],
        source=source,
    )


@pytest.fixture
def numeric_source() -> Iterator[Any]:
    r"""A Postgres schema of its own whose accounts are keyed by a NUMERIC,
    which the provider hands over as exact text; dropped afterwards."""
    dsn = os.getenv('SDM_POSTGRES_TEST_DSN')
    if not dsn:
        if os.getenv('SDM_REQUIRE_POSTGRES'):
            pytest.fail(
                'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
            )
        pytest.skip('requires SDM_POSTGRES_TEST_DSN')
    import psycopg

    namespace = f'sdm_{uuid.uuid4().hex[:8]}'
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(f'CREATE SCHEMA {namespace}')
        try:
            connection.execute(f'SET search_path TO {namespace}')
            connection.execute(
                'CREATE TABLE users (user_id NUMERIC PRIMARY KEY, age BIGINT)'
            )
            connection.execute(
                'INSERT INTO users SELECT i + 90000000000000001, 20 + i % 50 '
                'FROM generate_series(0, 39) AS i'
            )
            connection.execute(
                'CREATE TABLE orders (order_id BIGINT PRIMARY KEY, user_id '
                'NUMERIC REFERENCES users (user_id), price DOUBLE PRECISION, '
                'ts TIMESTAMP)'
            )
            connection.execute(
                'INSERT INTO orders SELECT i, i % 40 + 90000000000000001, '
                "10 + i * 1.5, TIMESTAMP '2023-06-01' + i * INTERVAL '1 day' "
                'FROM generate_series(0, 399) AS i'
            )
            yield PostgresSource(connection, schema=namespace)
        finally:
            connection.execute(f'DROP SCHEMA {namespace} CASCADE')


def test_missing_declared_column_is_refused_by_name():
    schema = _hand_built(_shop(), ghost=Column('ghost', 'float64', 'numerical'))
    with pytest.raises(StructuredDataError, match='users.ghost') as refused:
        materialize(schema)
    assert refused.value.code == 'INVALID_REQUEST'
    assert 'the source has no such column' in str(refused.value)


def test_incompatible_declared_dtype_is_refused_by_name():
    schema = _described(_shop())
    schema['users']['age'].dtype = 'string'
    schema['users']['age'].stype = 'categorical'
    with pytest.raises(
        StructuredDataError, match='users.age is given as'
    ) as refused:
        materialize(schema)
    assert refused.value.code == 'INVALID_REQUEST'


def test_source_name_selects_the_actual_table():
    schema = _described(_shop())
    original = materialize(schema)
    schema['orders'].source_name = 'other_orders'
    doubled = materialize(schema)
    for before, after in (
        (original.context, doubled.context),
        (original.predict, doubled.predict),
    ):
        prices = before.related_tables['orders']['price'].tolist()
        assert prices
        assert after.related_tables['orders']['price'].tolist() == [
            2 * p for p in prices
        ]


def test_added_column_can_be_edited_and_sampled():
    schema = _described(_shop())
    del schema['users']['country']
    schema['users'].columns['country'] = Column('country', 'string', 'text')
    schema['users']['country'].stype = 'categorical'
    assert schema['users']['country'].stype == 'categorical'
    assert schema['users']['country'].stype_declared
    request = materialize(schema)
    assert set(request.predict.related_tables['users']['country']) <= {
        'US',
        'DE',
        'IN',
    }
    assert len(request.predict.related_tables['users']) > 0


@pytest.mark.parametrize('kind', ['BOOLEAN[]', 'STRUCT(a INTEGER, b VARCHAR)'])
def test_unselected_unsupported_source_column_is_not_read(kind):
    source = _shop()
    source.connection.execute(f'ALTER TABLE users ADD COLUMN unselected {kind}')
    request = materialize(_hand_built(source))
    assert 'unselected' not in request.predict.related_tables['users']
    assert len(request.predict.instance_table) == 3


def test_unsupported_column_added_after_describe_is_not_read():
    source = _shop()
    schema = _described(source)
    source.connection.execute(
        'ALTER TABLE users ADD COLUMN extra STRUCT(a INTEGER, b VARCHAR)'
    )
    request = materialize(schema)
    assert 'extra' not in request.predict.related_tables['users']
    assert len(request.predict.instance_table) == 3


@pytest.mark.parametrize('made', sorted(_copies()))
def test_postgres_numeric_keys_stay_exact_through_copies_and_edits(
    made, numeric_source
):
    schema = _copies()[made](_described(numeric_source))
    assert schema['users']['user_id'].dtype == 'string'
    assert schema['orders']['user_id'].dtype == 'string'

    def assert_key_types():
        assert schema['users']['user_id'].dtype == 'string'
        assert schema['orders']['user_id'].dtype == 'string'
        assert schema['orders']['user_id'].stype == 'ID'

    def check():
        assert_key_types()
        request = materialize(schema, NUMERIC_IDS)
        values = request.predict.related_tables['users']['user_id'].tolist()
        assert set(map(str, NUMERIC_IDS)) <= set(values)
        assert all(isinstance(v, str) for v in values)
        assert len(request.predict.instance_table) == 2

    check()
    schema.relationships = []
    assert_key_types()
    with pytest.raises(
        StructuredDataError, match='There is no foreign key'
    ) as refused:
        materialize(schema, NUMERIC_IDS)
    assert refused.value.code == 'INVALID_REQUEST'
    schema.relationships = [
        Relationship('orders', 'user_id', 'users', 'user_id')
    ]
    assert_key_types()
    schema['users'].primary_key = None
    assert_key_types()
    schema['users'].primary_key = 'user_id'
    check()
