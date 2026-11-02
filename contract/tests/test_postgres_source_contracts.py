# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public PostgreSQL materialization preserves source values and time zones."""

import os
import uuid

import pandas as pd
import psycopg
import pytest
from psycopg import sql
from sdm_client import PredictiveQuery, Schema
from sdm_connectors import PostgresSource


@pytest.fixture
def source():
    dsn = os.environ.get('SDM_POSTGRES_TEST_DSN')
    if not dsn:
        if os.environ.get('SDM_REQUIRE_POSTGRES'):
            pytest.fail('SDM_POSTGRES_TEST_DSN is required')
        pytest.skip('local PostgreSQL is not configured')
    namespace = 'source_contract_' + uuid.uuid4().hex[:12]
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(
            sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(namespace))
        )
        try:
            connection.execute(
                sql.SQL('SET search_path TO {}').format(
                    sql.Identifier(namespace)
                )
            )
            yield connection, PostgresSource(connection, schema=namespace)
        finally:
            connection.execute(
                sql.SQL('DROP SCHEMA {} CASCADE').format(
                    sql.Identifier(namespace)
                )
            )


def test_automatic_anchor_is_the_same_in_utc_and_new_york(source):
    connection, provider = source
    connection.execute('CREATE TABLE users(id INTEGER PRIMARY KEY, age REAL)')
    connection.execute(
        'INSERT INTO users SELECT n, n * 10 FROM generate_series(1,12) n'
    )
    connection.execute(
        'CREATE TABLE orders(id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(id), price REAL, ts TIMESTAMPTZ)'
    )
    connection.execute(
        "INSERT INTO orders SELECT n, (n % 12)+1, n*0.5, '2024-01-01 00:00:00+00'::timestamptz + n * interval '2 hours' FROM generate_series(1,1200) n"
    )
    observed = []
    for zone in ('UTC', 'America/New_York'):
        connection.execute("SELECT set_config('TimeZone', %s, false)", [zone])
        schema = Schema.from_source(provider)
        request = PredictiveQuery(
            'PREDICT SUM(orders.price, 0, 7, days) FOR EACH users.id', schema
        ).materialize(indices=[1], num_context_samples=3, random_seed=42)
        assert request.payloads
        rows = []
        for payload in request.payloads:
            for half in ('context', 'predict'):
                table = payload[half]['instance_table']
                at = table['columns'].index('__kumo_anchor_time')
                assert table['rows']
                if half == 'predict':
                    assert all(
                        pd.Timestamp(row[at])
                        == pd.Timestamp('2024-04-10T00:00:00Z')
                        for row in table['rows']
                    )
                rows.extend(
                    (half, pd.Timestamp(row[at])) for row in table['rows']
                )
        observed.append(rows)
    assert observed[0] == observed[1]


def test_domain_feature_and_column_x_preserve_text_key_joins(source):
    connection, provider = source
    connection.execute('CREATE DOMAIN qty AS integer NOT NULL')
    connection.execute(
        'CREATE TABLE users(user_id TEXT PRIMARY KEY, age INTEGER)'
    )
    connection.execute(
        "INSERT INTO users SELECT 'u'||i,20+i FROM generate_series(0,7) i"
    )
    connection.execute(
        'CREATE TABLE orders(order_id BIGINT PRIMARY KEY, user_id TEXT REFERENCES users(user_id), n qty, price DOUBLE PRECISION, ts TIMESTAMP, x INTEGER DEFAULT 7)'
    )
    connection.execute(
        "INSERT INTO orders(order_id,user_id,n,price,ts) SELECT i,'u'||(i%8),i,10+i*1.5,TIMESTAMP '2023-01-01'+i*interval '7 days' FROM generate_series(0,79) i"
    )
    schema = Schema.from_source(provider)
    request = PredictiveQuery(
        'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id', schema
    ).materialize(
        indices=['u0', 'u1'],
        anchor_time=pd.Timestamp('2024-03-01'),
        random_seed=7,
        num_context_samples=3,
    )
    assert request.payloads
    for payload in request.payloads:
        for half in ('context', 'predict'):
            tables = payload[half]['related_tables']
            users = [
                dict(zip(tables['users']['columns'], row, strict=True))
                for row in tables['users']['rows']
            ]
            orders = [
                dict(zip(tables['orders']['columns'], row, strict=True))
                for row in tables['orders']['rows']
            ]
            assert users and orders
            keys = {(row['instance_id'], row['user_id']) for row in users}
            for row in orders:
                assert row['user_id'] == f'u{int(row["n"]) % 8}'
                assert row['x'] == 7
                assert row['price'] == 10 + row['n'] * 1.5
                assert (row['instance_id'], row['user_id']) in keys


@pytest.mark.parametrize('base', [0, 10**17])
@pytest.mark.parametrize('constrained', [False, True])
def test_decimal_key_scales_join_exact_values_without_nearby_matches(
    source, base, constrained
):
    from decimal import Decimal

    from sdm_client import Relationship

    connection, provider = source
    user_type = 'NUMERIC(22,1)' if constrained else 'NUMERIC'
    order_type = 'NUMERIC(23,2)' if constrained else 'NUMERIC'
    connection.execute(
        f'CREATE TABLE users(user_id {user_type} PRIMARY KEY, age INTEGER)'
    )
    connection.execute(
        'INSERT INTO users SELECT (%s+i)::NUMERIC(22,1),20+i FROM generate_series(0,7) i',
        [base],
    )
    connection.execute(
        f'CREATE TABLE orders(order_id BIGINT PRIMARY KEY, user_id {order_type}, marker TEXT)'
    )
    connection.execute(
        "INSERT INTO orders SELECT i,(%s+i)::NUMERIC(23,2),'exact-'||i FROM generate_series(0,7) i",
        [base],
    )
    connection.execute(
        "INSERT INTO orders SELECT 100+i,%s+i+0.1,'nearby' FROM generate_series(0,7) i",
        [base],
    )
    schema = Schema.from_source(
        provider,
        relationships=[Relationship('orders', 'user_id', 'users', 'user_id')],
    )
    request = PredictiveQuery(
        'PREDICT users.age FOR EACH users.user_id', schema
    ).materialize(
        indices=[Decimal(base), Decimal(base + 1)],
        num_context_samples=3,
        random_seed=7,
    )
    assert request.payloads
    for payload in request.payloads:
        for half in ('context', 'predict'):
            tables = payload[half]['related_tables']
            users = [
                dict(zip(tables['users']['columns'], row, strict=True))
                for row in tables['users']['rows']
            ]
            orders = [
                dict(zip(tables['orders']['columns'], row, strict=True))
                for row in tables['orders']['rows']
            ]
            assert users and orders
            keys = {(row['instance_id'], str(row['user_id'])) for row in users}
            for row in orders:
                assert row['marker'] != 'nearby'
                assert (row['instance_id'], str(row['user_id'])) in keys
                assert Decimal(str(row['user_id'])) == Decimal(base) + int(
                    row['marker'].split('-')[1]
                )
