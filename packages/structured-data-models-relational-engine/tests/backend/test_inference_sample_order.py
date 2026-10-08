# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""The rows a warehouse table's types are inferred from.

A table infers each column's type from a sample of its rows, which a bare
``LIMIT`` took in whatever order the warehouse read its files or
micro-partitions, so a borderline column could be inferred differently on two
identical runs. The sample is now the rows a hash of the primary key, or of a
few columns, picks, ordered totally, and a sample of expressions is ordered
like the sample of columns it is joined to by position. Which columns are
hashed is tested in ``base/test_utils.py``. PostgreSQL's sample is tested against a server in
``postgres/test_postgres_graph.py``.
"""

from __future__ import annotations

import re
from typing import Any

import pyarrow as pa
import pytest
from predictive_query.schema.typing import Dtype
from sdm_relational_engine.rfm.base import ColumnSpec, SourceColumn

_COLUMNS = {
    'n': Dtype.int,
    'tags': Dtype.stringlist,
    'ts': Dtype.date,
    'blob': Dtype.unsupported,
    'odd': None,
}


class _Cursor:
    def __init__(self, calls: list[str], metadata: list[str]) -> None:
        self._calls = calls
        self._metadata = metadata
        self._table: pa.Table | None = None

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def execute(self, sql: str, parameters: Any = None) -> None:
        if sql.lstrip().upper().startswith(('DESCRIBE', 'SHOW')):
            # A table that says nothing of its size or kind, so it is read
            # whole, as a small one is.
            self._metadata.append(sql)
            self._table = pa.table({})
            return
        self._calls.append(sql)
        # One column per selected item: the order terms, which the table
        # drops, hold `-1`, and each projection holds the row number.
        *order, projections = re.split(
            r' AS __kumo_order_\d+__, ', sql.split(' FROM ')[0]
        )
        num_order = len(order)
        names = re.findall(r'[`"](\w+)[`"](?:, |$)', projections)
        self._table = pa.table(
            {
                **{f'__kumo_order_{i}__': [-1, -1] for i in range(num_order)},
                **{name: [0, 1] for name in names},
            }
        )

    def fetchall_arrow(self) -> pa.Table:
        assert self._table is not None
        return self._table

    def fetch_arrow_all(self, force_return_table: bool = False) -> pa.Table:
        assert self._table is not None
        return self._table


class _Connection:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.metadata: list[str] = []

    def cursor(self) -> _Cursor:
        return _Cursor(self.calls, self.metadata)


def _source_columns() -> dict[str, SourceColumn]:
    return {
        name: SourceColumn(
            name=name,
            dtype=dtype,
            is_primary_key=False,
            is_unique_key=False,
            is_nullable=True,
        )
        for name, dtype in _COLUMNS.items()
    }


def _table(cls: type) -> Any:
    table = cls.__new__(cls)
    table._connection = _Connection()
    table._catalog = table._database = 'cat'
    table._schema = 'sch'
    table._source_name = 't'
    table._row_filter = None
    vars(table)['_source_column_dict'] = _source_columns()
    return table


def _databricks_table() -> Any:
    pytest.importorskip('databricks.sql')
    from sdm_relational_engine.rfm.backend.databricks import DatabricksTable

    return _table(DatabricksTable)


def _snow_table() -> Any:
    pytest.importorskip('snowflake.connector')
    from sdm_relational_engine.rfm.backend.snow import SnowTable

    table = _table(SnowTable)
    table._zoned_expr_columns = set()
    return table


def test_databricks_samples_the_rows_a_hash_picks() -> None:
    table = _databricks_table()

    sample = table._get_source_sample_df()

    # Map, struct and unknown types are left out of the hash and the order,
    # and so are lists, which not every warehouse hashes:
    (sql,) = table._connection.calls
    assert sql == (
        'SELECT xxhash64(`n`, `ts`) AS __kumo_order_0__, '
        '`n` AS __kumo_order_1__, `ts` AS __kumo_order_2__, '
        '`n`, `tags`, `ts`, `blob`, `odd` FROM `cat`.`sch`.`t` '
        'ORDER BY 1, 2, 3 LIMIT 1000'
    )
    assert list(sample.columns) == list(_COLUMNS)


def test_snowflake_samples_the_rows_a_hash_picks() -> None:
    table = _snow_table()

    sample = table._get_source_sample_df()

    (sql,) = table._connection.calls
    assert sql == (
        'SELECT HASH("n", "ts") AS __kumo_order_0__, '
        '"n" AS __kumo_order_1__, "ts" AS __kumo_order_2__, '
        '"n", "tags", "ts", "blob", "odd" FROM "cat"."sch"."t" '
        'ORDER BY 1, 2, 3 LIMIT 1000'
    )
    assert list(sample.columns) == list(_COLUMNS)


@pytest.mark.parametrize('make', [_databricks_table, _snow_table])
def test_expressions_are_sampled_from_the_same_rows(make: Any) -> None:
    table = make()

    table._get_source_sample_df()
    sample = table._get_expr_sample_df(
        [ColumnSpec(name='ts', expr='CAST(ts AS DATE)', dtype=Dtype.date)]
    )

    # The order is selected ahead of the expression and read by position, so
    # an expression named like a source column cannot take its place in it:
    source_sql, expr_sql = table._connection.calls
    head = source_sql.split(', `n`, `tags`')[0].split(', "n", "tags"')[0]
    assert expr_sql.startswith(head + ', CAST(ts AS DATE) AS ')
    assert expr_sql.endswith('ORDER BY 1, 2, 3 LIMIT 1000')
    assert list(sample.columns) == ['ts']


@pytest.mark.parametrize(
    ('make', 'quote', 'hash_function'),
    [(_databricks_table, '`', 'xxhash64'), (_snow_table, '"', 'HASH')],
)
def test_a_declared_key_is_the_only_column_hashed(
    make: Any,
    quote: str,
    hash_function: str,
) -> None:
    table = make()
    vars(table)['_source_column_dict']['ts'].is_primary_key = True

    table._get_source_sample_df()

    (sql,) = table._connection.calls
    ts, n = f'{quote}ts{quote}', f'{quote}n{quote}'
    assert sql.startswith(
        f'SELECT {hash_function}({ts}) AS __kumo_order_0__, '
        f'{ts} AS __kumo_order_1__, {n} AS __kumo_order_2__, '
    )
