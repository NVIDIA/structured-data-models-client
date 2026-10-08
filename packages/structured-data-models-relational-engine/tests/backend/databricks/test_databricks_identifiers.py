# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Warehouse names passed to the Databricks constructors must reach the driver
as data, not as SQL text.

Spark honours backslash escapes inside a single-quoted literal, so doubling the
quote character leaves a name containing ``\'`` able to close the literal and
run the rest as SQL, proven live against ``information_schema``. Names in a
value position are therefore bound; the catalog, which sits in an identifier
position no bind can fill, is backtick-quoted, a form a backslash cannot escape
out of.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest
from sdm_relational_engine.rfm import Graph

try:
    from sdm_relational_engine.rfm.backend.databricks import (
        Connection,
        DatabricksTable,
    )
except ImportError:
    pytest.skip("'databricks' extension not installed", allow_module_level=True)

_BACKSLASH_INJECTION = (
    "x\\' UNION ALL SELECT current_user() FROM "
    'system.information_schema.tables --'
)


class _Cursor:
    def __init__(self, connection: _Connection) -> None:
        self._connection = connection

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def execute(self, sql: str, parameters: Any = None) -> None:
        self._connection.calls.append((sql, parameters))

    def fetchall_arrow(self) -> pa.Table:
        return pa.table({'name': pa.array([], pa.string())})


class _Connection(Connection):
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def cursor(self) -> _Cursor:  # type: ignore[override]
        return _Cursor(self)

    def close(self) -> None:
        pass

    def __del__(self) -> None:
        pass


def _discover(schema: str, catalog: str = 'MY_CAT') -> tuple[str, Any]:
    connection = _Connection()

    # The fake connection discovers no tables, which the constructor rejects;
    # the statement under test has already been issued at that point:
    with pytest.raises(ValueError, match='No tables found'):
        Graph.from_databricks(
            connection=connection,
            catalog=catalog,
            schema=schema,
            verbose=False,
        )

    return connection.calls[0]


def test_from_databricks_quotes_the_catalog_identifier() -> None:
    sql, _ = _discover('my_schema')
    assert 'FROM `MY_CAT`.information_schema.tables' in sql


def test_from_databricks_binds_the_schema() -> None:
    sql, parameters = _discover('my_schema')
    assert 'WHERE table_schema = ?' in sql
    assert parameters == ['my_schema']
    assert 'my_schema' not in sql


@pytest.mark.parametrize(
    'schema',
    [
        "x' UNION ALL SELECT 1 --",
        _BACKSLASH_INJECTION,
        "x\\\\' UNION ALL SELECT 1 --",
        "x'' UNION ALL SELECT 1 --",
    ],
)
def test_from_databricks_never_writes_the_schema_into_the_sql(
    schema: str,
) -> None:
    sql, parameters = _discover(schema)
    assert parameters == [schema]
    assert 'UNION' not in sql.upper()
    assert '\\' not in sql
    assert sql.count("'") == 2  # only the METRIC_VIEW table-type literal


def test_from_databricks_quotes_an_injected_catalog() -> None:
    sql, _ = _discover(
        'my_schema',
        catalog='c`.information_schema.tables; DROP TABLE t; --',
    )
    assert (
        'FROM `c``.information_schema.tables; DROP TABLE t; --`'
        '.information_schema.tables'
    ) in sql


class _Table(DatabricksTable):
    def __init__(self, connection: _Connection, source_name: str) -> None:
        self._connection = connection
        self._catalog = 'cat'
        self._schema = 'sch'
        self._source_name = source_name


@pytest.mark.parametrize(
    'source_name',
    [
        'orders',
        "orders\\' UNION ALL SELECT 1 --",
    ],
)
def test_databricks_table_binds_the_metadata_lookups(source_name: str) -> None:
    connection = _Connection()
    table = _Table(connection, source_name)

    with pytest.raises(ValueError, match='does not exist'):
        table._get_source_columns()
    assert table._get_source_foreign_keys() == []

    assert len(connection.calls) == 2
    for sql, parameters in connection.calls:
        assert 'lower(?)' in sql
        assert parameters == ['sch', source_name]
        assert source_name not in sql
        assert '\\' not in sql
