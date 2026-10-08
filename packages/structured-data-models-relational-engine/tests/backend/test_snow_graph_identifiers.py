# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Warehouse names passed to the Snowflake constructors must reach the driver
as data, not as SQL text.

Doubling the quote character is not sufficient escaping on Snowflake, which
also honours backslash escapes inside a single-quoted literal, so a name
containing ``\'`` closes the literal early and the rest executes. Names in a
value position are therefore bound; a name in an identifier position, which no
bind can stand in for, is double-quoted, a form a backslash cannot escape out
of.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

pytest.importorskip('snowflake.connector')
pytest.importorskip('yaml')

import sdm_relational_engine.rfm as rfm
from sdm_connectors.source import adopt
from sdm_relational_engine.rfm.backend.snow.table import SnowTable
from snowflake.connector import SnowflakeConnection

_INJECTION = "X' OR 1=1 --"
_BACKSLASH_INJECTION = (
    "x\\' UNION ALL SELECT CURRENT_USER() FROM INFORMATION_SCHEMA.TABLES --"
)
_SEMANTIC_VIEW_YAML = 'tables: []\n'


class _Cursor:
    def __init__(self, connection: _Connection) -> None:
        self._connection = connection

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def execute(self, sql: str, parameters: Any = None) -> None:
        self._connection.calls.append(
            (sql, parameters, self._connection._paramstyle)
        )

    def fetch_arrow_all(self, force_return_table: bool = False) -> pa.Table:
        sql = self._connection.calls[-1][0]
        # A one-row answer for a statement read by its first row:
        single = sql.lstrip().startswith(('SELECT SYSTEM$', 'SELECT CURRENT'))
        rows = [self._connection.row] if single else self._connection.rows
        width = len(rows[0]) if rows else 1
        names = [f'c{i}' for i in range(width)]
        if sql.lstrip().startswith('SHOW TABLES'):
            # `SHOW TABLES` is read by field name.
            names = [
                'created_on',
                'name',
                'database_name',
                'schema_name',
                'kind',
                'comment',
                'cluster_by',
                'rows',
            ][:width]
        return pa.table(
            {name: [row[i] for row in rows] for i, name in enumerate(names)}
        )


class _Connection(SnowflakeConnection):
    def __init__(
        self,
        scalar: Any = None,
        row: tuple | None = None,
        rows: list[tuple] | None = None,
    ) -> None:
        self.calls: list[tuple[str, Any, str]] = []
        self.row = row if row is not None else (scalar,)
        self.rows = rows if rows is not None else []
        self._paramstyle = 'pyformat'

    def cursor(self) -> _Cursor:  # type: ignore[override]
        return _Cursor(self)


def _discover(
    schema: str,
    database: str = 'MY_DB',
) -> tuple[str, Any, str]:
    connection = _Connection()

    # The fake connection discovers no tables, which the constructor rejects;
    # the statement under test has already been issued at that point:
    with pytest.raises(ValueError, match='No tables found'):
        rfm.Graph.from_snowflake(
            connection,
            database=database,
            schema=schema,
            verbose=False,
        )

    return connection.calls[0]


def test_from_snowflake_quotes_the_database_identifier() -> None:
    sql, _, _ = _discover('MY_SCHEMA')
    assert 'FROM "MY_DB".INFORMATION_SCHEMA.TABLES' in sql


def test_from_snowflake_binds_the_schema() -> None:
    sql, parameters, style = _discover('MY_SCHEMA')
    assert 'WHERE TABLE_SCHEMA = ?' in sql
    assert parameters == ['MY_SCHEMA']
    assert style == 'qmark'
    assert 'MY_SCHEMA' not in sql


@pytest.mark.parametrize(
    'schema',
    [
        _INJECTION,
        _BACKSLASH_INJECTION,
        "x\\\\' UNION ALL SELECT 1 --",
        "x'' UNION ALL SELECT 1 --",
        "$$' UNION ALL SELECT 1 --",
    ],
)
def test_from_snowflake_never_writes_the_schema_into_the_sql(
    schema: str,
) -> None:
    sql, parameters, _ = _discover(schema)
    assert parameters == [schema]
    assert 'UNION' not in sql.upper()
    assert "'" not in sql
    assert '\\' not in sql


def test_from_snowflake_quotes_an_injected_database() -> None:
    sql, _, _ = _discover(
        'MY_SCHEMA',
        database='MY_DB". INFORMATION_SCHEMA.TABLES; DROP TABLE T; --',
    )
    assert (
        'FROM "MY_DB"". INFORMATION_SCHEMA.TABLES; DROP TABLE T; --"'
        '.INFORMATION_SCHEMA.TABLES'
    ) in sql


def test_from_snowflake_restores_the_drivers_paramstyle() -> None:
    connection = _Connection()

    with pytest.raises(ValueError, match='No tables found'):
        rfm.Graph.from_snowflake(
            connection, database='MY_DB', schema='MY_SCHEMA', verbose=False
        )

    assert connection._paramstyle == 'pyformat'


@pytest.mark.parametrize(
    'name',
    [
        'MY_VIEW',
        "MY_VIEW' OR 1=1 --",
        "MY_VIEW\\' OR 1=1 --",
    ],
)
def test_from_snowflake_semantic_view_binds_the_view_name(name: str) -> None:
    connection = _Connection(scalar=_SEMANTIC_VIEW_YAML)

    # The view definition is empty, so the graph fails validation afterwards;
    # the statement under test has already been issued at that point:
    with pytest.raises(ValueError, match='At least one table'):
        rfm.Graph.from_snowflake_semantic_view(
            name,
            connection=connection,
            verbose=False,
        )

    sql, parameters, style = connection.calls[0]
    assert sql == 'SELECT SYSTEM$READ_YAML_FROM_SEMANTIC_VIEW(?)'
    assert parameters == [name]
    assert style == 'qmark'
    assert connection._paramstyle == 'pyformat'


class _NumRowsTable:
    r"""What ``_get_num_rows`` reads, over a stub connection."""

    _run = SnowTable._run
    _rows = SnowTable._rows
    _listed = SnowTable.__dict__['_listed']
    _named = SnowTable._named

    def __init__(self, source_name: str, rows: list | None = None) -> None:
        listed = [('', source_name) + ('',) * 5 + (7,)]
        self._connection = _Connection(rows=listed if rows is None else rows)
        self._source = adopt(self._connection, 'snowflake')
        self._source_name = source_name
        self._database = 'DB'
        self._schema = 'PUBLIC'
        self.source_name = f'DB.PUBLIC.{source_name}'


@pytest.mark.parametrize(
    'source_name',
    [
        'ORDERS',
        "ORDERS\\' IN SCHEMA X.Y; DROP TABLE T; --",
    ],
)
def test_snow_table_binds_the_source_name(source_name: str) -> None:
    table = _NumRowsTable(source_name)

    assert SnowTable._get_num_rows(table) == 7  # type: ignore[arg-type]

    sql, parameters, style = table._connection.calls[0]
    assert sql.startswith('SHOW TABLES LIKE ? IN SCHEMA "DB"."PUBLIC"')
    assert parameters == [source_name]
    assert style == 'qmark'
    assert source_name not in sql


def test_snow_table_counts_the_rows_of_the_table_named_exactly() -> None:
    # Regression test: `SHOW TABLES LIKE` matches case-insensitively and reads
    # `_` as any character, and the first table listed was taken, so the row
    # count, which bounds the time window neighbors are read from, could be
    # another table's.
    rows = [
        ('', name, '', '', '', '', '', count)
        for name, count in (('ORDERSX', 3), ('orders_', 5), ('ORDERS_', 7))
    ]
    table = _NumRowsTable('ORDERS_', rows)

    assert SnowTable._get_num_rows(table) == 7  # type: ignore[arg-type]


def test_snow_table_counts_the_one_table_named_alike_but_for_case() -> None:
    rows = [
        ('', name, '', '', '', '', '', count)
        for name, count in (('ORDERSX', 3), ('ORDERS', 7))
    ]
    table = _NumRowsTable('orders', rows)

    assert SnowTable._get_num_rows(table) == 7  # type: ignore[arg-type]


def test_snow_table_refuses_to_guess_between_tables_named_alike() -> None:
    # Regression test: with no table named exactly, the first one listed was
    # counted, so which count bounded the time window depended on the order
    # `SHOW TABLES` listed `Orders` and `ORDERS` in.
    rows = [
        ('', name, '', '', '', '', '', count)
        for name, count in (('Orders', 3), ('ORDERS', 7))
    ]
    table = _NumRowsTable('orders', rows)

    with pytest.raises(RuntimeError, match=r"\['ORDERS', 'Orders'\]"):
        SnowTable._get_num_rows(table)  # type: ignore[arg-type]


def test_snow_table_says_when_it_is_not_listed() -> None:
    table = _NumRowsTable('ORDERS', [])

    with pytest.raises(RuntimeError, match='SHOW TABLES does not list it'):
        SnowTable._get_num_rows(table)  # type: ignore[arg-type]
