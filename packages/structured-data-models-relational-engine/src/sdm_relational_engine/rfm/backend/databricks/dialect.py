# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The SQL the engine runs against Databricks.

Each statement is written by a plain function that returns ``(sql, params)``
and touches no connection; the tables, the sampler and the metric view reader
hand what it returns to the source's ``_run``. Parameters are bound to ``?``
markers, natively, by a SQL warehouse or by a notebook's Spark session alike.
Identifiers are quoted with backticks.

A key list travels as one parameter holding a JSON array, which
``posexplode(from_json(...))`` turns into rows; a list longer than the 1 MB a
parameter may hold is split into chunks, each its own statement.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from predictive_query.schema.typing import Dtype

from sdm_relational_engine.rfm.base.source import KeyCountPart, Statement
from sdm_relational_engine.rfm.base.utils import (
    count_distinct_values,
    key_count_select,
    ordered_sample_sql,
    repeated_key_select,
)

# Catalog #####################################################################


def current_namespace() -> Statement:
    r"""The catalog and schema a bare name resolves in."""
    return 'SELECT current_catalog(), current_schema()', None


def list_tables(catalog: str, schema: str) -> Statement:
    r"""Every table of ``schema`` in the quoted ``catalog`` but its metric
    views, by name.
    """
    return (
        '\n'
        '                    SELECT table_name\n'
        f'                    FROM {catalog}.information_schema.tables\n'
        '                    WHERE table_schema = ?\n'
        "                      AND table_type != 'METRIC_VIEW'\n"
        '                    ORDER BY table_name\n'
        '                    ',
        [schema],
    )


def columns(catalog: str, schema: str, table: str) -> Statement:
    r"""The columns of ``schema.table`` in the quoted ``catalog``, in order.

    Unity Catalog stores ``table_schema`` and ``table_name`` folded to lower
    case, so the literals are folded, which keeps the look-up sargable.
    """
    return (
        f'SELECT c.column_name, c.full_data_type, c.is_nullable,\n'
        f'       c.table_catalog, c.table_schema, c.table_name,\n'
        f'       t.table_type\n'
        f'FROM {catalog}.information_schema.columns c\n'
        f'JOIN {catalog}.information_schema.tables t\n'
        f'  ON t.table_catalog = c.table_catalog\n'
        f' AND t.table_schema = c.table_schema\n'
        f' AND t.table_name = c.table_name\n'
        f'WHERE c.table_schema = lower(?)\n'
        f'  AND c.table_name = lower(?)\n'
        f'ORDER BY c.ordinal_position',
        [schema, table],
    )


def key_constraints(catalog: str, schema: str, table: str) -> Statement:
    r"""The primary and unique keys of ``schema.table`` in the quoted
    ``catalog``, by column.
    """
    return (
        f'SELECT tc.constraint_type, tc.constraint_name,\n'
        f'       kcu.column_name\n'
        f'FROM {catalog}.information_schema.'
        f'table_constraints tc\n'
        f'JOIN {catalog}.information_schema.'
        f'key_column_usage kcu\n'
        f'  ON tc.constraint_catalog = kcu.constraint_catalog\n'
        f' AND tc.constraint_schema = kcu.constraint_schema\n'
        f' AND tc.constraint_name = kcu.constraint_name\n'
        f'WHERE tc.table_schema = lower(?)\n'
        f'  AND tc.table_name = lower(?)\n'
        f"  AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE')",
        [schema, table],
    )


def foreign_keys(catalog: str, schema: str, table: str) -> Statement:
    r"""The foreign keys of ``schema.table`` in the quoted ``catalog``, with
    what each references.
    """
    return (
        f'SELECT tc.constraint_name, kcu.column_name,\n'
        f'       ccu.table_catalog, ccu.table_schema,\n'
        f'       ccu.table_name, ccu.column_name\n'
        f'FROM {catalog}.information_schema.'
        f'table_constraints tc\n'
        f'JOIN {catalog}.information_schema.'
        f'referential_constraints rc\n'
        f'  ON tc.constraint_catalog = rc.constraint_catalog\n'
        f' AND tc.constraint_schema = rc.constraint_schema\n'
        f' AND tc.constraint_name = rc.constraint_name\n'
        f'JOIN {catalog}.information_schema.'
        f'key_column_usage kcu\n'
        f'  ON tc.constraint_catalog = kcu.constraint_catalog\n'
        f' AND tc.constraint_schema = kcu.constraint_schema\n'
        f' AND tc.constraint_name = kcu.constraint_name\n'
        f'JOIN {catalog}.information_schema.'
        f'constraint_column_usage ccu\n'
        f'  ON rc.unique_constraint_catalog = '
        f'ccu.constraint_catalog\n'
        f' AND rc.unique_constraint_schema = '
        f'ccu.constraint_schema\n'
        f' AND rc.unique_constraint_name = ccu.constraint_name\n'
        f"WHERE tc.constraint_type = 'FOREIGN KEY'\n"
        f'  AND tc.table_schema = lower(?)\n'
        f'  AND tc.table_name = lower(?)\n'
        f'ORDER BY tc.constraint_name',
        [schema, table],
    )


def describe_extended(view: str) -> Statement:
    r"""The columns and details of the quoted ``view``, a metric view's
    definition among them.
    """
    return f'DESCRIBE TABLE EXTENDED {view}', None


def describe_detail(table: str) -> Statement:
    r"""The details of the quoted ``table``, its ``sizeInBytes`` among them,
    which a Delta table keeps in its log, so nothing is scanned.
    """
    return f'DESCRIBE DETAIL {table}', None


def bounded(table: str, num_rows: int) -> str:
    r"""``table`` cut to the first ``num_rows`` rows Databricks reads of it.

    Which rows those are is up to how it reads the table's files; a ``LIMIT``
    without an order stops reading once it has them.
    """
    return f'(SELECT * FROM {table} LIMIT {num_rows})'


def sample(
    table: str,
    projections: Sequence[str],
    order_terms: Sequence[str],
    num_rows: int,
) -> Statement:
    r"""``projections`` of the first ``num_rows`` rows of ``table`` in the
    order ``order_terms`` give, the terms selected first.
    """
    return ordered_sample_sql(table, projections, order_terms, num_rows), None


def holds_any(table: str, identity: str, values: Sequence[str]) -> Statement:
    r"""Whether a row of ``table`` holds one of ``values`` in ``identity``,
    the expression of a folded composite key: one row at most is read, and
    ``values`` travel as one JSON array.
    """
    return (
        f'SELECT 1 FROM {table}\n'
        f'WHERE ({identity}) IN (\n'
        f"  SELECT col FROM explode(from_json(?, 'array<string>'))\n"
        f')\n'
        f'LIMIT 1',
        [dumps(list(values))],
    )


def count_rows(table: str) -> Statement:
    r"""The rows of ``table``."""
    return f'SELECT COUNT(*) FROM {table}', None


# Keys ########################################################################


def key_counts(parts: Sequence[KeyCountPart]) -> Statement:
    r"""The rows, distinct keys and rows missing a key of each table, in one
    ``UNION ALL``. Spark counts several columns distinct, skipping a row
    missing any of them.
    """
    selects = [
        key_count_select(
            part.label,
            part.source,
            part.columns,
            part.unique,
            count_distinct_values(part.columns),
        )
        for part in parts
    ]
    return '\nUNION ALL\n'.join(selects), None


def repeated_key(table: str, columns: Sequence[str]) -> Statement:
    r"""One value of the key ``columns`` more than one row holds."""
    return repeated_key_select(table, columns), None


# Sampling ####################################################################


def _json_default(obj: Any) -> Any:
    # `numpy`/`pandas` scalars are not JSON-serializable out-of-the-box:
    if hasattr(obj, 'item'):
        return obj.item()
    return str(obj)


def dumps(obj: Any) -> str:
    r"""``obj`` as compact JSON.

    With no whitespace, so :func:`chunk_rows`' per-row size accounting, which
    adds one byte for the single element separator, matches the payload
    exactly; the default ``json.dumps`` inserts a ``', '`` separator,
    undercounting the payload by one byte per row.
    """
    return json.dumps(obj, separators=(',', ':'), default=_json_default)


def chunk_rows(rows: list, max_bytes: int = 900_000) -> list[list]:
    r"""``rows`` in chunks whose JSON stays below the native parameter size
    limit, 1 MB.
    """
    chunks: list[list] = []
    current: list = []
    current_bytes = 2  # Account for the enclosing '[]'.
    for row in rows:
        size = len(dumps(row)) + 1
        if current and current_bytes + size > max_bytes:
            chunks.append(current)
            current, current_bytes = [], 2
        current.append(row)
        current_bytes += size
    if current:
        chunks.append(current)
    return chunks or [[]]


def element_type(dtype: Dtype) -> str:
    r"""The Databricks SQL type a key of ``dtype`` is read from JSON as."""
    if dtype.is_int():
        return 'bigint'
    if dtype.is_float():
        return 'double'
    return 'string'


def coerce_id(value: Any, dtype: Dtype) -> Any:
    r"""A key as the native Python type matching ``dtype``, so that it
    round-trips through JSON into the declared Databricks SQL type.
    """
    if dtype.is_int():
        return int(value)
    if dtype.is_float():
        return float(value)
    return str(value)


def time_ranges(tables: Sequence[tuple[str, str]]) -> Statement:
    r"""The earliest and latest time of each ``(table, time column)``."""
    selects = [
        f'SELECT\n'
        f'  {index} as table_index,\n'
        f'  MIN({column}) as min_date,\n'
        f'  MAX({column}) as max_date\n'
        f'FROM {table}'
        for index, (table, column) in enumerate(tables)
    ]
    return '\nUNION ALL\n'.join(selects), None


def entity_rows(
    table: str,
    projections: Sequence[str],
    filters: Sequence[str],
    *,
    key: str,
    order_by: str,
    percent: float | None = None,
    random_seed: int | None = None,
    num_rows: int | None = None,
    ids: list | None = None,
    dtype: Dtype | None = None,
) -> Statement:
    r"""Rows of an entity table: ``num_rows`` of a ``percent`` sample, or the
    rows holding ``ids`` in ``key``, joined from a JSON parameter in the order
    of the ids.

    ``TABLESAMPLE (n ROWS)`` is ``LIMIT`` in Databricks, not random, so a
    random percentage is drawn instead. With ``random_seed`` it is repeatable,
    and the rows the sample holds beyond ``num_rows`` are dropped by a seeded
    hash of the key, so that the same ones are kept every time.
    """
    sql = f'SELECT {", ".join(projections)}\nFROM {table}'
    params: list | None = None
    if ids is None:
        sql += f'\nTABLESAMPLE ({percent} PERCENT)'
        if random_seed is not None:
            sql += f' REPEATABLE ({random_seed})'
    else:
        assert dtype is not None
        sql += (
            f'\nJOIN posexplode(from_json(?, '
            f"'array<{element_type(dtype)}>')) AS __ids__"
            f'(__pos__, __KUMO_ID__)\n'
            f'  ON {key} = __ids__.__KUMO_ID__'
        )
        params = [dumps([coerce_id(value, dtype) for value in ids])]
    if len(filters) > 0:
        sql += f'\nWHERE{" AND".join(filters)}'
    if ids is not None:
        sql += f'\nORDER BY __ids__.__pos__, {order_by}'
    elif random_seed is not None:
        sql += f'\nORDER BY xxhash64({int(random_seed)}, {key}), {order_by}'
    if ids is None:
        sql += f'\nLIMIT {num_rows}'
    return sql, params


def rows_by_key(
    table: str,
    projections: Sequence[str],
    *,
    key: str,
    dtype: Dtype,
    ids: list,
    order_by: Sequence[str],
) -> Statement:
    r"""The rows of ``table`` holding each of ``ids``, a chunk of them, in
    ``key``, with the id's position as ``__KUMO_BATCH__``, in that order and
    then ``order_by``.
    """
    sql = (
        f'WITH TMP AS (\n'
        f'  SELECT pos AS __KUMO_BATCH__, col AS __KUMO_ID__\n'
        f"  FROM posexplode(from_json(?, 'array<{element_type(dtype)}"
        f">'))\n"
        f')\n'
        f'SELECT '
        f'TMP.__KUMO_BATCH__ as __KUMO_BATCH__, '
        f'{", ".join(projections)}\n'
        f'FROM TMP\n'
        f'JOIN {table}\n'
        f'  ON {key} = TMP.__KUMO_ID__\n'
    )
    sql += f'ORDER BY {", ".join(["TMP.__KUMO_BATCH__", *order_by])}'
    return sql, [dumps(ids)]


def neighbors(
    table: str,
    projections: Sequence[str],
    *,
    key: str,
    dtype: Dtype,
    rows: list,
    windowed: bool,
    time: str | None,
    bounds: Sequence[str],
    order_by: str,
    num_neighbors: int,
) -> Statement:
    r"""For each of ``rows``, a chunk of keys, with end (``e``) and start
    (``s``) times if ``windowed``, the first ``num_neighbors`` rows of
    ``table`` holding it in ``key`` and in its window, in ``order_by``
    order. The ``bounds`` of every window let the warehouse skip data.
    """
    elem = element_type(dtype)
    select = ['pos AS __KUMO_BATCH__']
    if windowed:
        schema = f'array<struct<id:{elem},e:string,s:string>>'
        select.append('col.id AS __KUMO_ID__')
        select.append('col.e AS __KUMO_END_TIME__')
        select.append('col.s AS __KUMO_START_TIME__')
    else:
        schema = f'array<{elem}>'
        select.append('col AS __KUMO_ID__')
    sql = (
        f'WITH TMP AS (\n'
        f'  SELECT {", ".join(select)}\n'
        f"  FROM posexplode(from_json(?, '{schema}'))\n"
        f')\n'
        f'SELECT '
        f'TMP.__KUMO_BATCH__ as __KUMO_BATCH__, '
        f'{", ".join(projections)}\n'
        f'FROM TMP\n'
        f'JOIN {table}\n'
        f'  ON {key} = TMP.__KUMO_ID__\n'
    )
    if windowed:
        sql += (
            f' AND {time} <= TMP.__KUMO_END_TIME__\n'
            f' AND {time} > TMP.__KUMO_START_TIME__\n'
            f'WHERE {time} <= ?\n'
            f'  AND {time} > ?\n'
        )
    sql += 'QUALIFY ROW_NUMBER() OVER (\n  PARTITION BY TMP.__KUMO_BATCH__\n'
    sql += f'  ORDER BY {order_by}\n'
    sql += f') <= {num_neighbors}\n'
    # Returned in that order too. Without it the rows came back in the order
    # each engine read them, which serverless Spark does differently from a
    # SQL warehouse, and both the request and the answer differed. Chunks are
    # stitched in batch order, so ordering each keeps the whole.
    sql += f'ORDER BY TMP.__KUMO_BATCH__, {order_by}'
    return sql, [dumps(rows), *bounds]


def window(
    table: str,
    projections: Sequence[str],
    *,
    key: str,
    dtype: Dtype,
    rows: list,
    with_start: bool,
    time: str,
    bounds: Sequence[str],
    order_by: str,
) -> Statement:
    r"""For each of ``rows``, a chunk of keys with their end (``e``) and, if
    ``with_start``, start (``s``) times, the rows of ``table`` holding it in
    ``key`` in its window, in ``order_by`` order. The ``bounds`` of every
    window, bound like every other value, let the warehouse skip data.
    """
    elem = element_type(dtype)
    if with_start:
        schema = f'array<struct<id:{elem},e:string,s:string>>'
    else:
        schema = f'array<struct<id:{elem},e:string>>'
    select = [
        'pos AS __KUMO_BATCH__',
        'col.id AS __KUMO_ID__',
        'col.e AS __KUMO_END_TIME__',
    ]
    if with_start:
        select.append('col.s AS __KUMO_START_TIME__')
    sql = (
        f'WITH TMP AS (\n'
        f'  SELECT {", ".join(select)}\n'
        f"  FROM posexplode(from_json(?, '{schema}'))\n"
        f')\n'
        f'SELECT '
        f'TMP.__KUMO_BATCH__ as __KUMO_BATCH__, '
        f'{", ".join(projections)}\n'
        f'FROM TMP\n'
        f'JOIN {table}\n'
        f'  ON {key} = TMP.__KUMO_ID__\n'
        f' AND {time} <= TMP.__KUMO_END_TIME__\n'
    )
    if with_start:
        sql += f'AND {time} > TMP.__KUMO_START_TIME__\n'
    sql += f'WHERE {time} <= ?'
    if with_start:
        sql += f'\nAND {time} > ?'
    sql += f'\nORDER BY TMP.__KUMO_BATCH__, {order_by}'
    return sql, [dumps(rows), *bounds]
