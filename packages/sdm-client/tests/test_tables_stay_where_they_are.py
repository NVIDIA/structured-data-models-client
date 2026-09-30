# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Schemas over tables left in DuckDB, SQLite and Postgres, sampled there with
SQL."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('sdm_relational_engine.relationallib')
duckdb = pytest.importorskip('duckdb')
pytest.importorskip('adbc_driver_sqlite')

from sdm_connectors import (  # noqa: E402
    DuckDBSource,
    PostgresSource,
    SQLiteSource,
    connect,
)

from sdm_client import (  # noqa: E402
    KumoTabular,
    PredictiveQuery,
    Relationship,
    Schema,
    SourceTable,
    StructuredDataError,
    Table,
)
from sdm_client.core.warnings import SDMWarning  # noqa: E402
from sdm_client.materializers import RelationalMaterializer  # noqa: E402

QUERY = 'PREDICT SUM(orders.amount, 0, 30, days) FOR EACH users.user_id'
TABLES = {'users': 'app_users', 'orders': 'app_orders'}


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    users = pd.DataFrame(
        {
            'user_id': range(1, 51),
            'age': rng.integers(18, 70, 50),
            'country': rng.choice(['US', 'DE', 'IN'], 50),
        }
    )
    # Hourly timestamps over a year, so windows of whole days land exactly on
    # stored rows: the edges are where a text comparison goes wrong.
    orders = pd.DataFrame(
        {
            'order_id': range(1, 1001),
            'user_id': rng.integers(1, 51, 1000),
            'amount': rng.gamma(2, 20, 1000).round(2),
            'ts': pd.Timestamp('2024-01-01')
            + pd.to_timedelta(rng.integers(0, 365 * 24, 1000), unit='h'),
        }
    )
    return users, orders


USERS, ORDERS = _frames()


@pytest.fixture
def duckdb_source() -> Any:
    connection = connect('duckdb')
    connection.register('users_frame', USERS)
    connection.register('orders_frame', ORDERS)
    connection.execute('CREATE TABLE app_users AS SELECT * FROM users_frame')
    connection.execute('CREATE TABLE app_orders AS SELECT * FROM orders_frame')
    yield DuckDBSource(connection)
    connection.close()


@pytest.fixture
def sqlite_source(tmp_path: Path) -> Any:
    path = tmp_path / 'sales.sqlite'
    sqlite3.connect(path).close()
    connection = connect('sqlite', str(path))
    # ADBC stores fractional seconds, which a text window bound must match.
    with connection.cursor() as cursor:
        cursor.adbc_ingest('app_users', USERS, mode='create')
        cursor.adbc_ingest('app_orders', ORDERS, mode='create')
    connection.commit()
    yield SQLiteSource(connection)
    connection.close()


@pytest.fixture
def postgres() -> Any:
    r"""A psycopg connection and a schema of its own, dropped afterwards."""
    dsn = os.getenv('SDM_POSTGRES_TEST_DSN')
    if not dsn:
        if os.getenv('SDM_REQUIRE_POSTGRES'):
            pytest.fail(
                'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
            )
        pytest.skip('SDM_POSTGRES_TEST_DSN is not set')
    psycopg = pytest.importorskip('psycopg')

    namespace = f'sdm_{uuid.uuid4().hex[:8]}'
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(f'CREATE SCHEMA {namespace}')
        try:
            yield connection, namespace
        finally:
            connection.execute(f'DROP SCHEMA {namespace} CASCADE')


def _python_rows(frame: pd.DataFrame) -> list[tuple[Any, ...]]:
    # psycopg adapts Python values, not numpy's.
    return [
        tuple(
            value.to_pydatetime() if isinstance(value, pd.Timestamp) else value
            for value in row.values()
        )
        for row in frame.to_dict('records')
    ]


@pytest.fixture
def postgres_source(postgres: Any) -> Any:
    connection, namespace = postgres
    for name, frame, columns in (
        ('app_users', USERS, 'user_id BIGINT, age BIGINT, country TEXT'),
        (
            'app_orders',
            ORDERS,
            'order_id BIGINT, user_id BIGINT, amount DOUBLE PRECISION, '
            'ts TIMESTAMP',
        ),
    ):
        connection.execute(f'CREATE TABLE {namespace}.{name} ({columns})')
        markers = ', '.join(['%s'] * len(frame.columns))
        with connection.cursor() as cursor:
            cursor.executemany(
                f'INSERT INTO {namespace}.{name} VALUES ({markers})',
                _python_rows(frame),
            )
    return PostgresSource(connection, schema=namespace)


@pytest.fixture(params=['duckdb', 'sqlite'])
def source(request: pytest.FixtureRequest) -> Any:
    return request.getfixturevalue(f'{request.param}_source')


def _schema(source: Any, **overrides: Any) -> Schema:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return Schema.from_source(source, TABLES, **overrides)


def _payload(schema: Any, seed: int = 7) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        request = PredictiveQuery(QUERY, schema).materialize(
            indices=[1, 2, 3], random_seed=seed
        )
    (payload,) = request._payloads
    return json.loads(json.dumps(payload, default=str))


def test_the_tables_are_described_without_being_read(source: Any) -> None:
    from sdm_relational_engine.rfm.base import DataBackend

    schema = _schema(source)

    assert schema.source is source
    assert all(isinstance(t, SourceTable) for t in schema.tables.values())
    assert not any(hasattr(t, 'data') for t in schema.tables.values())
    assert schema['users'].source_name == 'app_users'
    assert schema['users'].primary_key == 'user_id'
    assert schema['orders'].primary_key == 'order_id'
    assert schema['orders'].time_column == 'ts'
    assert schema['orders']['amount'].stype == 'numerical'
    assert schema['users']['country'].stype == 'categorical'
    assert schema.relationships == [
        Relationship('orders', 'user_id', 'users', 'user_id')
    ]
    assert RelationalMaterializer(schema)._build_graph().backend == (
        DataBackend(source.backend)
    )


@pytest.mark.parametrize('seed', [7, 11, 23])
def test_every_sampled_label_is_the_sum_the_rows_give(
    source: Any, seed: int
) -> None:
    r"""An order exactly on a window edge must count; SQLite compares bounds
    as text.
    """
    rows = _payload(_schema(source), seed)['context']['instance_table']['rows']

    wrong = []
    for _, user, anchor, label in rows:
        at = pd.Timestamp(anchor).tz_localize(None)
        inside = (
            (ORDERS.user_id == user)
            & (ORDERS.ts > at)
            & (ORDERS.ts <= at + pd.Timedelta(days=30))
        )
        expected = ORDERS.amount[inside].sum()
        if not np.isclose(label, expected, rtol=1e-5, atol=1e-3):
            wrong.append((user, anchor, label, expected))
    assert rows
    assert not wrong


def test_a_scored_entity_sees_the_history_it_would_in_memory(
    source: Any,
) -> None:
    def history(payload: dict[str, Any]) -> list[tuple[Any, ...]]:
        table = payload['predict']['related_tables']['orders']
        at = {name: i for i, name in enumerate(table['columns'])}
        return sorted(
            (row[at['user_id']], row[at['ts']], row[at['amount']])
            for row in table['rows']
        )

    in_place = _payload(_schema(source))
    in_memory = _payload({'users': USERS, 'orders': ORDERS})

    assert history(in_place) == history(in_memory)
    assert (
        in_place['predict']['instance_table']
        == in_memory['predict']['instance_table']
    )


_KEYED = (
    'CREATE TABLE {in_}u (user_id INTEGER PRIMARY KEY)',
    'CREATE TABLE {in_}o (order_id INTEGER PRIMARY KEY, '
    'buyer INTEGER REFERENCES {in_}u(user_id), ts TIMESTAMP)',
    'INSERT INTO {in_}u VALUES (1)',
    "INSERT INTO {in_}o VALUES (1, 1, '2024-01-01')",
)


@pytest.fixture(params=['duckdb', 'postgres'])
def keyed_source(request: pytest.FixtureRequest) -> Any:
    # Postgres names the referenced table with its schema; DuckDB does not.
    if request.param == 'duckdb':
        connection = duckdb.connect()
        for statement in _KEYED:
            connection.execute(statement.format(in_=''))
        return DuckDBSource(connection)
    connection, namespace = request.getfixturevalue('postgres')
    for statement in _KEYED:
        connection.execute(statement.format(in_=f'{namespace}.'))
    return PostgresSource(connection, schema=namespace)


def test_a_declared_foreign_key_links_what_names_would_not(
    keyed_source: Any,
) -> None:
    named = Schema.from_source(keyed_source, {'users': 'u', 'orders': 'o'})
    everything = Schema.from_source(keyed_source)

    assert named.relationships == [
        Relationship('orders', 'buyer', 'users', 'user_id')
    ]
    assert everything.relationships == [
        Relationship('o', 'buyer', 'u', 'user_id')
    ]


def test_overrides_mean_what_they_mean_on_from_data(
    duckdb_source: Any,
) -> None:
    schema = _schema(
        duckdb_source,
        primary_keys={'orders': 'order_id'},
        time_columns={'orders': 'ts'},
        relationships=[],
    )
    assert schema.relationships == []


def test_a_column_with_no_scalar_form_is_left_out_and_named() -> None:
    connection = duckdb.connect()
    connection.execute(
        'CREATE TABLE t (id INTEGER PRIMARY KEY, tags INTEGER[], x DOUBLE)'
    )
    connection.execute('INSERT INTO t VALUES (1, [1, 2], 0.5)')

    with pytest.warns(SDMWarning, match=r'leaves out tags'):
        schema = Schema.from_source(DuckDBSource(connection), {'t': 't'})

    assert list(schema['t'].columns) == ['id', 'x']


def test_a_column_left_out_after_a_question_is_not_sent_by_the_next(
    source: Any,
) -> None:
    schema = _schema(source)
    _payload(schema)
    del schema['users']['country']

    assert 'country' not in schema['users']
    payload = _payload(schema)
    assert (
        'country'
        not in payload['context']['related_tables']['users']['columns']
    )

    with pytest.raises(StructuredDataError, match='primary key; set that'):
        del schema['users']['user_id']
    with pytest.raises(StructuredDataError, match='time column; set that'):
        del schema['orders']['ts']
    with pytest.raises(StructuredDataError, match="has no column 'nope'"):
        del schema['users']['nope']


def test_the_tabular_model_is_refused_rather_than_fed_nothing(
    duckdb_source: Any,
) -> None:
    schema = Schema.from_source(
        duckdb_source, {'users': 'app_users'}, relationships=[]
    )
    query = PredictiveQuery('PREDICT users.age FOR EACH users.user_id', schema)

    with pytest.raises(StructuredDataError, match='only the relational'):
        query.materialize(indices=[1], model=KumoTabular())
    with pytest.raises(StructuredDataError, match='score with indices='):
        query.materialize(predict=USERS.head(1))


def test_a_one_table_source_schema_is_gathered_for_the_relational_model(
    duckdb_source: Any,
) -> None:
    schema = Schema.from_source(
        duckdb_source, {'users': 'app_users'}, relationships=[]
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        request = PredictiveQuery(
            'PREDICT users.age FOR EACH users.user_id', schema
        ).materialize(indices=[1, 2])

    assert request.predict_rows == 2
    assert request.schema.source is duckdb_source


@pytest.mark.parametrize(
    'call, match',
    [
        (
            lambda source: Schema.from_source(source.connection, TABLES),
            'needs an sdm_connectors',
        ),
        (lambda source: Schema.from_source(source, {}), 'leave tables out'),
        (
            lambda source: _schema(source, time_columns={'order': 'ts'}),
            "no table 'order'",
        ),
        (
            lambda source: Schema(
                {
                    'users': Table('users', USERS, {}),
                    'orders': _schema(source)['orders'],
                },
                source=source,
            ),
            'samples one or the other',
        ),
        (
            lambda source: Schema({'orders': _schema(source)['orders']}),
            'needs that source',
        ),
        (
            lambda source: Schema(
                {'users': Table('users', USERS, {})}, source=source
            ),
            'which a source does not',
        ),
    ],
    ids=[
        'a connection',
        'no tables',
        'an unknown override',
        'frames and tables in place',
        'tables in place without their source',
        'frames with a source',
    ],
)
def test_a_schema_that_cannot_be_sampled_is_refused(
    duckdb_source: Any, call: Any, match: str
) -> None:
    with pytest.raises(StructuredDataError, match=match):
        call(duckdb_source)


def test_a_snapshot_keeps_the_source(duckdb_source: Any) -> None:
    schema = _schema(duckdb_source)
    copied = schema.snapshot()

    assert copied.source is duckdb_source
    assert isinstance(copied['orders'], SourceTable)
    assert copied['orders'].source_name == 'app_orders'
    copied['orders']['amount'].stype = 'categorical'
    assert schema['orders']['amount'].stype == 'numerical'


def test_left_out_tables_mean_every_table_in_the_source(source: Any) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        schema = Schema.from_source(source)

    assert set(schema.tables) == {'app_users', 'app_orders'}
    assert schema['app_orders'].source_name == 'app_orders'
    assert set(schema['app_orders'].columns) == set(ORDERS.columns)
    assert schema['app_orders'].time_column == 'ts'


def test_a_question_repeats_none_of_the_reads_that_described_the_tables(
    postgres_source: Any,
) -> None:
    r"""The time range is still read per question, so new rows are seen."""
    import psycopg

    statements: list[str] = []

    class Recorded(psycopg.Cursor):
        def execute(self, query: Any, *args: Any, **kwargs: Any) -> Any:
            statements.append(str(query))
            return super().execute(query, *args, **kwargs)

    postgres_source.connection.cursor_factory = Recorded
    schema = _schema(postgres_source)
    described = list(statements)
    for seed in (7, 11):
        _payload(schema, seed)
    asked = statements[len(described) :]

    assert any('information_schema' in sql for sql in described)
    assert asked
    assert not set(described) & set(asked)


def test_a_question_cannot_change_what_the_next_one_is_given(
    source: Any,
) -> None:
    r"""Every gather writes keys, time columns and semantic types onto the
    tables it samples; those it is given are copies of what was read."""
    schema = _schema(source)
    changed = schema.snapshot()
    changed['users']['age'].stype = 'categorical'
    changed['users']['country'].stype = 'text'

    _payload(changed)
    graph = RelationalMaterializer(schema)._build_graph()

    assert str(graph['users']['age'].stype) == 'numerical'
    assert str(graph['users']['country'].stype) == 'categorical'
