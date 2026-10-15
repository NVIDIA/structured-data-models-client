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

import pytest

pytest.importorskip('snowflake.connector')
pytest.importorskip('yaml')

import sdm_relational_engine.rfm as rfm
from sdm_relational_engine.rfm.backend.snow.binding import paramstyle_attribute
from sdm_relational_engine.rfm.backend.snow.table import SnowTable
from snowflake.connector import SnowflakeConnection

_INJECTION = "X' OR 1=1 --"
_BACKSLASH_INJECTION = (
    "x\\' UNION ALL SELECT CURRENT_USER() FROM INFORMATION_SCHEMA.TABLES --"
)
_SEMANTIC_VIEW_YAML = 'tables: []\n'


def _style(connection: SnowflakeConnection) -> Any:
    return getattr(connection, paramstyle_attribute(connection))


class _Cursor:
    def __init__(self, connection: _Connection) -> None:
        self._connection = connection

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def execute(self, sql: str, parameters: Any = None) -> None:
        self._connection.calls.append(
            (sql, parameters, str(_style(self._connection)))
        )

    def fetchone(self) -> tuple:
        return self._connection.row

    def fetchall(self) -> list:
        return []


class _Connection(SnowflakeConnection):
    def __init__(self, scalar: Any = None, row: tuple | None = None) -> None:
        self.calls: list[tuple[str, Any, str]] = []
        self.row = row if row is not None else (scalar,)
        setattr(self, paramstyle_attribute(self), 'pyformat')

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
    assert parameters == ('MY_SCHEMA',)
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
    assert parameters == (schema,)
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

    assert _style(connection) == 'pyformat'


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
    assert parameters == (name,)
    assert style == 'qmark'
    assert _style(connection) == 'pyformat'


class _NumRowsTable:
    def __init__(self, source_name: str) -> None:
        self._connection = _Connection(row=('',) * 7 + (7,))
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
    assert parameters == (source_name,)
    assert style == 'qmark'
    assert source_name not in sql
