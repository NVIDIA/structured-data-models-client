# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""``Source._run``: one statement in, what it reads out, as Arrow.

Each warehouse's quirks live behind it: SQLite and DuckDB read as their
drivers do, PostgreSQL types its rows by the declared column types and keeps
each read in its own transaction, and Snowflake binds with ``qmark`` for the
statement and puts the caller's style back.
"""

from __future__ import annotations

import datetime
import decimal
import json
import os
from collections.abc import Iterator
from typing import Any

import pyarrow as pa
import pytest

from sdm_connectors import Source
from sdm_connectors.source import adopt
from sdm_connectors.sql import rows_to_arrow

# Rows to Arrow.


def test_rows_become_columns_of_the_type_given_or_inferred() -> None:
    table = rows_to_arrow(
        [(1, 'a', None), (2, None, 2.5)],
        ['n', 's', 'x'],
        [pa.int16(), None, None],
    )
    assert table.column_names == ['n', 's', 'x']
    assert table.schema.field('n').type == pa.int16()
    assert table.schema.field('s').type == pa.string()
    assert table.to_pylist() == [
        {'n': 1, 's': 'a', 'x': None},
        {'n': 2, 's': None, 'x': 2.5},
    ]


def test_a_column_no_one_type_holds_holds_text() -> None:
    table = rows_to_arrow([(1,), ('a',), (None,)], ['mixed'])
    assert table.column('mixed').to_pylist() == ['1', 'a', None]


def test_a_value_its_declared_type_cannot_hold_is_refused() -> None:
    from sdm_connectors import ConnectorError

    with pytest.raises(ConnectorError, match="'grid'") as caught:
        rows_to_arrow(
            [(1, [[1, 2]]), (2, [3])],
            ['id', 'grid'],
            [pa.int64(), pa.list_(pa.int64())],
        )
    assert caught.value.code == 'READ_FAILED'
    assert caught.value.details['column'] == 'grid'


def test_no_rows_keep_their_columns() -> None:
    table = rows_to_arrow([], ['a', 'b'], [pa.int64(), None])
    assert table.num_rows == 0
    assert table.column_names == ['a', 'b']
    assert table.schema.field('a').type == pa.int64()


# SQLite and DuckDB.


def test_sqlite_binds_by_position_and_reads_arrow() -> None:
    dbapi = pytest.importorskip('adbc_driver_sqlite.dbapi')
    connection = dbapi.connect(':memory:')
    try:
        source = Source(connection)
        table = source._run('SELECT ? AS a, ? AS b', [1, "it's"])
        assert table.to_pylist() == [{'a': 1, 'b': "it's"}]
        assert source._run('SELECT 2 AS a', None).to_pylist() == [{'a': 2}]
    finally:
        connection.close()


def test_duckdb_reads_through_a_cursor_of_its_own() -> None:
    duckdb = pytest.importorskip('duckdb')
    connection = duckdb.connect()
    try:
        connection.execute('CREATE TABLE t AS SELECT 1 AS a')
        connection.execute('CREATE TEMP TABLE scratch AS SELECT 1 AS a')
        source = Source(connection)
        assert source._run('SELECT a + ? AS b FROM t', [1]).to_pylist() == [
            {'b': 2}
        ]
        # A cursor is a connection of its own: the caller's temporary
        # objects are theirs.
        with pytest.raises(duckdb.CatalogException):
            source._run('SELECT * FROM scratch')
    finally:
        connection.close()


# PostgreSQL.


@pytest.fixture
def postgres() -> Iterator[Any]:
    psycopg = pytest.importorskip('psycopg')
    dsn = os.getenv('SDM_POSTGRES_TEST_DSN')
    if not dsn:
        if os.getenv('SDM_REQUIRE_POSTGRES'):
            pytest.fail(
                'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
            )
        pytest.skip('SDM_POSTGRES_TEST_DSN is not set')
    connection = psycopg.connect(dsn)
    try:
        yield connection
    finally:
        connection.close()


def test_postgres_types_follow_the_declared_types(postgres: Any) -> None:
    from sdm_connectors.backends.postgres import TYPE_OID_KEY

    source = Source(postgres)
    table = source._run(
        'SELECT 1::int2 AS a, 1.5::numeric(5, 2) AS b, '
        """'{"k": [1, 2]}'::jsonb AS c, 0.1::float4 AS d, """
        "TIMESTAMPTZ '2024-01-02 03:04:05+02' AS e, "
        "'1 day'::interval AS f, ARRAY[1, 2]::int8[] AS g, "
        "'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11'::uuid AS h"
    )
    types = {field.name: field.type for field in table.schema}
    assert types == {
        'a': pa.int16(),
        'b': pa.string(),
        'c': pa.string(),
        'd': pa.float64(),
        'e': pa.timestamp('us', tz='UTC'),
        'f': pa.duration('us'),
        'g': pa.list_(pa.int64()),
        'h': pa.string(),
    }
    row = table.to_pylist()[0]
    # Exact, and as psycopg reads it: a numeric's digits, the document, and a
    # float4 as the double of its text.
    assert decimal.Decimal(row['b']) == decimal.Decimal('1.50')
    assert row['b'] == '1.50'
    assert json.loads(row['c']) == {'k': [1, 2]}
    assert row['d'] == 0.1
    assert row['e'] == datetime.datetime(
        2024, 1, 2, 1, 4, 5, tzinfo=datetime.timezone.utc
    )
    assert row['f'] == datetime.timedelta(days=1)
    assert row['g'] == [1, 2]
    assert row['h'] == 'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11'
    assert table.schema.field('b').metadata == {TYPE_OID_KEY: b'1700'}


def test_a_void_result_is_not_fetched(postgres: Any) -> None:
    table = Source(postgres)._run('SELECT setseed(%s)', [0.5])
    assert table.column_names == ['setseed']
    assert table.num_rows == 0
    assert table.schema.field('setseed').type == pa.null()


def test_postgres_binds_percent_s_by_position(postgres: Any) -> None:
    table = Source(postgres)._run('SELECT %s::text AS a', ["it's"])
    assert table.to_pylist() == [{'a': "it's"}]


# Snowflake, over a stand-in connection.


class _SnowflakeCursor:
    def __init__(self, connection: _Snowflake) -> None:
        self._connection = connection
        self.description: list[tuple[str, ...]] | None = None

    def __enter__(self) -> _SnowflakeCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        self._connection.seen.append(
            (sql, params, self._connection._paramstyle)
        )
        if 'fail' in sql:
            raise RuntimeError('SQL compilation error')
        self.description = [('name',), ('rows',)]

    def fetch_arrow_all(self, force_return_table: bool = False) -> pa.Table:
        from snowflake.connector.errors import NotSupportedError

        if self._connection.seen[-1][0].startswith('SHOW'):
            raise NotSupportedError('no Arrow for a metadata statement')
        return pa.table({'n': [1]})

    def fetchall(self) -> list[tuple[Any, ...]]:
        return [('ORDERS', 7)]


class _Snowflake:
    def __init__(self) -> None:
        self._paramstyle = 'pyformat'
        self.seen: list[tuple[str, Any, str]] = []

    def cursor(self) -> _SnowflakeCursor:
        return _SnowflakeCursor(self)


@pytest.fixture
def snowflake() -> _Snowflake:
    pytest.importorskip('snowflake.connector')
    return _Snowflake()


def test_snowflake_binds_with_qmark_and_puts_the_style_back(
    snowflake: _Snowflake,
) -> None:
    source = adopt(snowflake, 'snowflake')
    assert source._run('SELECT ? AS n', [1]).to_pylist() == [{'n': 1}]
    assert snowflake.seen[-1] == ('SELECT ? AS n', [1], 'qmark')
    assert snowflake._paramstyle == 'pyformat'


def test_snowflake_puts_the_style_back_when_a_statement_fails(
    snowflake: _Snowflake,
) -> None:
    # The connection is usually the caller's, so a failed statement must not
    # leave it in another style.
    source = adopt(snowflake, 'snowflake')
    with pytest.raises(RuntimeError):
        source._run('SELECT fail WHERE a = ?', [1])
    assert snowflake.seen[-1][2] == 'qmark'
    assert snowflake._paramstyle == 'pyformat'


def test_snowflake_leaves_the_style_alone_without_parameters(
    snowflake: _Snowflake,
) -> None:
    adopt(snowflake, 'snowflake')._run('SELECT 1 AS n')
    assert snowflake.seen[-1] == ('SELECT 1 AS n', None, 'pyformat')


def test_a_snowflake_metadata_statement_is_read_from_its_rows(
    snowflake: _Snowflake,
) -> None:
    table = adopt(snowflake, 'snowflake')._run('SHOW TABLES LIKE ?', ['O%'])
    assert table.to_pylist() == [{'name': 'ORDERS', 'rows': 7}]


def test_a_semantic_model_runs_in_its_warehouse(snowflake: _Snowflake) -> None:
    from sdm_connectors import SemanticModel, SnowflakeSource

    source = adopt(snowflake, 'snowflake')
    assert isinstance(source, SnowflakeSource)
    model = SemanticModel(source, 'SALES')
    assert model._run('SELECT 1 AS n').to_pylist() == [{'n': 1}]
