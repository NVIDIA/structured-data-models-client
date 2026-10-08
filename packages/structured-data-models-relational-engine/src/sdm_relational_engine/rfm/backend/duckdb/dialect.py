# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The SQL the engine runs against DuckDB.

Each statement is written by a plain function that returns ``(sql, params)``
and touches no connection; the tables and the sampler hand what it returns to
the source's ``_run``. Parameters are bound to ``?`` markers.

A key list travels as one parameter holding a JSON array, which
``from_json`` reads into typed rows, so a lookup registers nothing on the
connection. Each value takes the DuckDB type of the Arrow type it has, the
type the key tables the engine used to register gave it, so it compares with
the table's columns as it did then.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
from sdm_connectors import quote_ident

from sdm_relational_engine.rfm.base.source import KeyCountPart, Statement
from sdm_relational_engine.rfm.base.utils import (
    count_distinct_values,
    key_count_select,
    repeated_key_select,
)

# A table is read by its bare name, which resolves in the current schema, so a
# namesake elsewhere must not lend it constraints.
_IN_CURRENT_SCHEMA = (
    'database_name = current_database() AND schema_name = current_schema()'
)

# Catalog #####################################################################


def list_tables() -> Statement:
    r"""Every table of the schema a bare name resolves in, by name."""
    return (
        'SELECT table_name FROM duckdb_tables() '
        'WHERE NOT temporary AND NOT internal '
        'AND database_name = current_database() '
        'AND schema_name = current_schema() '
        'ORDER BY table_name',
        None,
    )


def table_info(table: str) -> Statement:
    r"""The columns of ``table``, a quoted name."""
    return f'PRAGMA table_info({table})', None


def key_constraints(table: str) -> Statement:
    r"""The constraints of the table named ``table``, unquoted."""
    name = quote_ident(table, char="'")
    return (
        'SELECT constraint_type, constraint_column_names\n'
        'FROM duckdb_constraints()\n'
        f'WHERE table_name = {name}\n'
        f'  AND {_IN_CURRENT_SCHEMA}',
        None,
    )


def foreign_keys(table: str) -> Statement:
    r"""The foreign keys of the table named ``table``, unquoted."""
    name = quote_ident(table, char="'")
    return (
        'SELECT constraint_index, referenced_table,\n'
        '       constraint_column_names, referenced_column_names\n'
        'FROM duckdb_constraints()\n'
        f'WHERE table_name = {name}\n'
        f'  AND {_IN_CURRENT_SCHEMA}\n'
        "  AND constraint_type = 'FOREIGN KEY'\n"
        'ORDER BY constraint_index',
        None,
    )


def first_rows(
    table: str, projections: Sequence[str], num_rows: int
) -> Statement:
    r"""``projections`` of ``num_rows`` rows of ``table``."""
    return (
        f'SELECT {", ".join(projections)} FROM {table} LIMIT {num_rows}',
        None,
    )


def holds_any(table: str, identity: str, values: Sequence[str]) -> Statement:
    r"""Whether a row of ``table`` holds one of ``values`` in ``identity``,
    the expression of a folded composite key: one row at most is read, and
    ``values`` travel as one JSON array.
    """
    return (
        f'SELECT 1 FROM {table}\n'
        f'WHERE ({identity}) IN (\n'
        f'  SELECT unnest(from_json(?, \'["VARCHAR"]\'))\n'
        f')\n'
        f'LIMIT 1',
        [json.dumps(list(values), separators=(',', ':'))],
    )


def count_rows(table: str) -> Statement:
    r"""The rows of ``table``."""
    return f'SELECT COUNT(*) FROM {table}', None


# Keys ########################################################################


def count_distinct(columns: Sequence[str]) -> str:
    r"""The distinct values of a key, leaving out a row missing any part.

    A row of several values is not null when one of them is, so a row missing
    part of the key is left out by hand.
    """
    if len(columns) == 1:
        return count_distinct_values(columns)
    present = ' AND '.join(f'{column} IS NOT NULL' for column in columns)
    return f'COUNT(DISTINCT ({", ".join(columns)})) FILTER (WHERE {present})'


def key_counts(parts: Sequence[KeyCountPart]) -> Statement:
    r"""The rows, distinct keys and rows missing a key of each table, in one
    ``UNION ALL``.
    """
    selects = [
        key_count_select(
            part.label,
            part.source,
            part.columns,
            part.unique,
            count_distinct(part.columns),
        )
        for part in parts
    ]
    return '\nUNION ALL\n'.join(selects), None


def repeated_key(table: str, columns: Sequence[str]) -> Statement:
    r"""One value of the key ``columns`` more than one row holds."""
    return repeated_key_select(table, columns), None


# Sampling ####################################################################


def indices() -> Statement:
    r"""Every index, by table, with the expressions it indexes."""
    return 'SELECT table_name, expressions FROM duckdb_indexes()', None


def time_column(column: str, zoned: bool = False) -> str:
    r"""The time ``column`` as a ``TIMESTAMP``, as it is compared.

    A ``TIMESTAMP WITH TIME ZONE`` column, ``zoned``, is taken as its UTC
    wall-clock time, as the engine means every time: a ``CAST`` would take
    it in the connection's ``TimeZone``, which is the machine's own unless set.
    """
    if zoned:
        return f"timezone('UTC', {column})"
    return f'CAST({column} AS TIMESTAMP)'


def time_ranges(tables: Sequence[tuple[str, str]]) -> Statement:
    r"""The earliest and latest time of each ``(table, time)``."""
    selects = [
        f'SELECT\n'
        f'  {index} as table_index,\n'
        f'  MIN({time}) as min_date,\n'
        f'  MAX({time}) as max_date\n'
        f'FROM {table}'
        for index, (table, time) in enumerate(tables)
    ]
    return '\nUNION ALL\n'.join(selects), None


def random_rows(
    table: str,
    projections: Sequence[str],
    filters: Sequence[str],
    *,
    num_rows: int,
    random_seed: int | None,
    order_by: str,
) -> Statement:
    r"""A reservoir sample of ``num_rows`` rows of ``table`` holding
    ``filters``, repeatable for a seed, put in ``order_by`` order.

    DuckDB's sampling avoids SQLite's ``ORDER BY RANDOM()`` full sort.
    """
    sql = f'SELECT {", ".join(projections)}\n'
    sql += f'FROM (SELECT * FROM {table}'
    if len(filters) > 0:
        sql += f'\nWHERE{" AND".join(filters)}'
    sql += f')\nUSING SAMPLE reservoir({num_rows} ROWS)'
    if random_seed is not None:
        sql += f' REPEATABLE ({random_seed})'
    # The reservoir comes back in no defined order, and the sampled rows are
    # shuffled by position next, so they are put in one first:
    sql += f'\nORDER BY {order_by}'
    return sql, None


def rows_by_ids(
    table: str,
    projections: Sequence[str],
    filters: Sequence[str],
    *,
    key: str,
    ids: Sequence[Any],
    order_by: str,
) -> Statement:
    r"""The rows of ``table`` holding ``ids`` in ``key`` and ``filters``, in
    ``order_by`` order, the ids bound as parameters.
    """
    placeholders = ', '.join(['?'] * len(ids))
    conditions = [*filters, f' {key} IN ({placeholders})']
    sql = f'SELECT {", ".join(projections)}\nFROM {table}'
    sql += f'\nWHERE{" AND".join(conditions)}'
    sql += f'\nORDER BY {order_by}'
    return sql, list(ids)


def _duckdb_type(arrow_type: pa.DataType) -> str:
    r"""The DuckDB type of a key value of ``arrow_type``, as DuckDB reads
    an Arrow column of it.
    """
    if pa.types.is_boolean(arrow_type):
        return 'BOOLEAN'
    if pa.types.is_integer(arrow_type):
        width = arrow_type.bit_width
        signed = {8: 'TINYINT', 16: 'SMALLINT', 32: 'INTEGER', 64: 'BIGINT'}
        if pa.types.is_signed_integer(arrow_type):
            return signed[width]
        return 'U' + signed[width]
    if pa.types.is_float32(arrow_type):
        return 'FLOAT'
    if pa.types.is_floating(arrow_type):
        return 'DOUBLE'
    if pa.types.is_date(arrow_type):
        return 'DATE'
    if pa.types.is_timestamp(arrow_type):
        if arrow_type.tz is not None:
            return 'TIMESTAMPTZ'
        return {
            's': 'TIMESTAMP_S',
            'ms': 'TIMESTAMP_MS',
            'us': 'TIMESTAMP',
            'ns': 'TIMESTAMP_NS',
        }[arrow_type.unit]
    if pa.types.is_decimal(arrow_type):
        return f'DECIMAL({arrow_type.precision}, {arrow_type.scale})'
    return 'VARCHAR'


def _json_values(values: pa.Array) -> list[Any]:
    r"""``values`` as JSON holds them: a timestamp in ISO 8601 to its last
    digit, with its offset where it has a zone, a date in ISO 8601, a decimal
    as its exact text.
    """
    if pa.types.is_timestamp(values.type):
        spec = '%Y-%m-%dT%H:%M:%S%z' if values.type.tz else '%Y-%m-%dT%H:%M:%S'
        return pc.strftime(values, format=spec).to_pylist()
    if pa.types.is_date(values.type):
        return [
            None if v is None else v.isoformat() for v in values.to_pylist()
        ]
    if pa.types.is_decimal(values.type):
        return [None if v is None else str(v) for v in values.to_pylist()]
    if pa.types.is_floating(values.type):
        return [None if v is None or v != v else v for v in values.to_pylist()]
    if (
        pa.types.is_integer(values.type)
        or pa.types.is_boolean(values.type)
        or pa.types.is_string(values.type)
        or pa.types.is_large_string(values.type)
    ):
        return values.to_pylist()
    return [None if v is None else str(v) for v in values.to_pylist()]


@dataclass(frozen=True)
class _Keys:
    r"""The ``WITH`` clause that reads key columns from a JSON parameter, and
    that parameter.
    """

    sql: str
    param: str


def _keys(columns: Sequence[tuple[str, pa.Array]]) -> _Keys:
    r"""The ``WITH`` clause that reads key columns from a JSON parameter, as
    ``__kumo_keys__``, and that parameter.

    Each row of ``__kumo_keys__`` is one object of the JSON array:
    ``__kumo_batch__`` its position, and one column per ``(name, values)``
    in ``columns``, typed as DuckDB reads an Arrow column of the values' type.
    """
    fields = [f'f{index}' for index in range(len(columns))]
    structure = {
        field: _duckdb_type(array.type)
        for field, (_, array) in zip(fields, columns, strict=True)
    }
    values = [_json_values(array) for _, array in columns]
    payload = [dict(zip(fields, row, strict=True)) for row in zip(*values)]
    selects = ['k.i - 1 AS __kumo_batch__']
    selects += [
        f'k.v.{field} AS {name}'
        for field, (name, _) in zip(fields, columns, strict=True)
    ]
    shape = quote_ident(json.dumps([structure], separators=(',', ':')), "'")
    sql = (
        f'WITH __kumo_keys__ AS (\n'
        f'  SELECT {", ".join(selects)}\n'
        f'  FROM unnest(from_json(?, {shape}))\n'
        f'    WITH ORDINALITY AS k(v, i)\n'
        f')\n'
    )
    return _Keys(sql, json.dumps(payload, separators=(',', ':')))


def rows_by_key(
    table: str,
    key: str,
    ids: pa.Array,
    projections: Sequence[str],
    *,
    order_by: str,
) -> Statement:
    r"""For each id, one row of ``table`` holding it in ``key``, with the
    id's position as ``__kumo_batch__``, and in ``__kumo_matches__`` how many
    rows hold it. Of several, the first in ``order_by`` order, so the same row
    is kept however they are stored.
    """
    keys = _keys([('__kumo_id__', ids)])
    sql = (
        f'{keys.sql}'
        f'SELECT '
        f'tmp.__kumo_batch__, '
        f'COUNT(*) OVER (PARTITION BY tmp.__kumo_batch__) '
        f'AS __kumo_matches__, '
        f'{", ".join(projections)}\n'
        f'FROM __kumo_keys__ tmp\n'
        f'JOIN {table}\n'
        f'  ON {key} = tmp.__kumo_id__\n'
        f'QUALIFY ROW_NUMBER() OVER (\n'
        f'  PARTITION BY tmp.__kumo_batch__\n'
        f'  ORDER BY {order_by}\n'
        f') = 1'
    )
    return sql, [keys.param]


def neighbors(
    table: str,
    key: str,
    ids: pa.Array,
    projections: Sequence[str],
    *,
    num_neighbors: int,
    order_by: str,
    time: str | None = None,
    end_times: pa.Array | None = None,
) -> Statement:
    r"""For each id, the first ``num_neighbors`` rows of ``table`` holding it
    in ``key``, in ``order_by`` order, with the time ``time`` no later than its
    end time, if given.
    """
    columns = [('__kumo_id__', ids)]
    if end_times is not None:
        columns.append(('__kumo_time__', end_times))
    keys = _keys(columns)
    sql = (
        f'{keys.sql}'
        f'SELECT '
        f'tmp.__kumo_batch__, '
        f'{", ".join(projections)}\n'
        f'FROM __kumo_keys__ tmp\n'
        f'JOIN {table}\n'
        f'  ON {key} = tmp.__kumo_id__\n'
    )
    if end_times is not None:
        sql += f' AND {time} <= tmp.__kumo_time__\n'
    sql += 'QUALIFY ROW_NUMBER() OVER (\n  PARTITION BY tmp.__kumo_batch__\n'
    sql += f'  ORDER BY {order_by}\n'
    sql += f') <= {num_neighbors}\n'
    # Returned in that order too, so the rows reach the request the same way
    # however the table is stored.
    sql += f'ORDER BY tmp.__kumo_batch__, {order_by}'
    return sql, [keys.param]


def window(
    table: str,
    key: str,
    ids: pa.Array,
    projections: Sequence[str],
    *,
    time: str,
    end_times: pa.Array,
    start_times: pa.Array | None,
    order_by: str,
) -> Statement:
    r"""For each id, the rows of ``table`` holding it in ``key`` whose time
    ``time`` is in its window: no later than its end, and later than its
    start, if given. In ``order_by`` order, which is the order they are
    aggregated in.
    """
    columns = [('__kumo_id__', ids), ('__kumo_end__', end_times)]
    if start_times is not None:
        columns.append(('__kumo_start__', start_times))
    keys = _keys(columns)
    sql = (
        f'{keys.sql}'
        f'SELECT '
        f'tmp.__kumo_batch__, '
        f'{", ".join(projections)}\n'
        f'FROM __kumo_keys__ tmp\n'
        f'JOIN {table}\n'
        f'  ON {key} = tmp.__kumo_id__\n'
        f' AND {time} <= tmp.__kumo_end__'
    )
    if start_times is not None:
        sql += f'\n AND {time} > tmp.__kumo_start__'
    sql += f'\nORDER BY tmp.__kumo_batch__, {order_by}'
    return sql, [keys.param]
