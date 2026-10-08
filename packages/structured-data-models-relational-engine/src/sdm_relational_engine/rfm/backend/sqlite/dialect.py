# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The SQL the engine runs against SQLite.

Each statement is written by a plain function that returns ``(sql, params)``
and touches no connection; the tables and the sampler hand what it returns to
the source's ``_run``. Parameters are bound to ``?`` markers.

A key list travels as one parameter holding a JSON array, which
``json_each`` turns into rows, so a lookup writes nothing to the database.
Each value is cast to the type SQLite gave the key tables the engine used to
write, so it compares with the table's columns as it did then.
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
    random_state,
    repeated_key_select,
)

# Catalog #####################################################################


def list_tables() -> Statement:
    r"""Every table in the database, by name."""
    return (
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
        None,
    )


def table_info(table: str) -> Statement:
    r"""The columns of ``table``, a quoted name."""
    return f'PRAGMA table_info({table})', None


def index_list(table: str) -> Statement:
    r"""The indices of ``table``, a quoted name."""
    return f'PRAGMA index_list({table})', None


def index_info(index: str) -> Statement:
    r"""The columns of the index named ``index``."""
    return f'PRAGMA index_info({quote_ident(index)})', None


def foreign_key_list(table: str) -> Statement:
    r"""The foreign keys of ``table``, a quoted name."""
    return f'PRAGMA foreign_key_list({table})', None


def first_rows(
    table: str, projections: Sequence[str], num_rows: int
) -> Statement:
    r"""``projections`` of the first ``num_rows`` rows of ``table``, in
    ``rowid`` order.
    """
    return (
        f'SELECT {", ".join(projections)} '
        f'FROM {table} '
        f'ORDER BY rowid '
        f'LIMIT {num_rows}',
        None,
    )


def holds_any(table: str, identity: str, values: Sequence[str]) -> Statement:
    r"""Whether a row of ``table`` holds one of ``values`` in ``identity``,
    the expression of a folded composite key: one row at most is read, and
    ``values`` travel as one JSON array.
    """
    return (
        f'SELECT 1 FROM {table}\n'
        f'WHERE ({identity}) IN (SELECT value FROM json_each(?))\n'
        f'LIMIT 1',
        [json.dumps(list(values), separators=(',', ':'))],
    )


# Keys ########################################################################


def count_distinct(table: str, columns: Sequence[str]) -> str:
    r"""The distinct values of a key, leaving out a row missing any part.

    SQLite counts one column distinct at a time, so the pairs are counted in
    a subquery.
    """
    if len(columns) == 1:
        return count_distinct_values(columns)
    present = ' AND '.join(f'{column} IS NOT NULL' for column in columns)
    return (
        f'(SELECT COUNT(*) FROM (SELECT DISTINCT {", ".join(columns)} '
        f'FROM {table} WHERE {present}))'
    )


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
            count_distinct(part.source, part.columns),
        )
        for part in parts
    ]
    return '\nUNION ALL\n'.join(selects), None


def repeated_key(table: str, columns: Sequence[str]) -> Statement:
    r"""One value of the key ``columns`` more than one row holds."""
    return repeated_key_select(table, columns), None


# Sampling ####################################################################

# The modulus of the seeded row order: a prime, so the hash below is a
# polynomial over a field, and below 2**31, so a product of two residues stays
# inside SQLite's 64-bit integers.
_HASH_PRIME = 2**31 - 1


def seeded_hash(random_seed: int, ref: str) -> str:
    r"""A hash of the integer ``ref`` that shuffles rows, repeatably for a seed.

    SQLite's ``RANDOM()`` takes no seed, so ``ORDER BY RANDOM()`` drew a
    different sample on every call. Rows are ordered by this hash instead: a
    cubic polynomial modulo a prime, whose four coefficients are drawn from
    ``random_state(random_seed)``. Over uniform coefficients such a polynomial
    hashes any four distinct values below the prime independently and
    uniformly, so the rows that hash lowest are spread over the table like a
    random draw, and a different seed draws different ones. It is not an
    exactly uniform draw of every larger set of rows, which no family of
    hashes this small gives.

    Args:
        random_seed: The seed the coefficients are drawn from.
        ref: The integer to hash, a key column or ``rowid``.
    """
    p = _HASH_PRIME
    c0, c1, c2, c3 = (
        int(c) for c in random_state(random_seed).randint(0, p, size=4)
    )
    x = f'(({ref} % {p} + {p}) % {p})'
    return (
        f'(((({c3} * {x} + {c2}) % {p}) * {x} + {c1}) % {p} * {x} + {c0}) % {p}'
    )


def first_time(table: str, column: str) -> Statement:
    r"""One stored value of the time ``column`` of ``table``, to read how it
    is written.
    """
    return (
        f'SELECT {column} FROM {table} '
        f'WHERE {column} IS NOT NULL ORDER BY rowid LIMIT 1',
        None,
    )


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
    order_by: str | None = None,
    num_rows: int | None = None,
    key: str | None = None,
    ids: Sequence[Any] | None = None,
) -> Statement:
    r"""Rows of an entity table: ``num_rows`` of them in ``order_by``, a
    random draw without one, or the rows holding ``ids`` in ``key``.

    Args:
        table: The table, quoted.
        projections: What is read of each row.
        filters: Conditions every row read holds, each with a leading space.
        order_by: The order the rows are drawn or returned in.
        num_rows: How many rows to draw.
        key: The key ``ids`` are looked up in.
        ids: The ids to read, bound as parameters.
    """
    conditions = list(filters)
    params: list[Any] | None = None
    if ids is not None:
        placeholders = ', '.join(['?'] * len(ids))
        conditions.append(f' {key} IN ({placeholders})')
        params = list(ids)
    sql = f'SELECT {", ".join(projections)}\nFROM {table}'
    if len(conditions) > 0:
        sql += f'\nWHERE{" AND".join(conditions)}'
    if ids is None and order_by is None:
        sql += f'\nORDER BY RANDOM() LIMIT {num_rows}'
    elif ids is None:
        sql += f'\nORDER BY {order_by} LIMIT {num_rows}'
    else:
        sql += f'\nORDER BY {order_by}'
    return sql, params


# Each key value's SQLite type, by the Arrow type of the values: the type the
# key tables written with `adbc_ingest` gave it, so it compares as it did.
def _sqlite_type(arrow_type: pa.DataType) -> str | None:
    if pa.types.is_integer(arrow_type) or pa.types.is_boolean(arrow_type):
        return 'INTEGER'
    if pa.types.is_floating(arrow_type):
        return 'REAL'
    if (
        pa.types.is_string(arrow_type)
        or pa.types.is_large_string(arrow_type)
        or pa.types.is_date(arrow_type)
    ):
        return 'TEXT'
    return None


def _json_values(values: pa.Array) -> list[Any]:
    r"""``values`` as JSON holds them, a timestamp or date as the text
    ``adbc_ingest`` writes for it.
    """
    if pa.types.is_timestamp(values.type):
        return pc.strftime(values, format='%Y-%m-%dT%H:%M:%S').to_pylist()
    if pa.types.is_date(values.type):
        return [
            None if v is None else v.isoformat() for v in values.to_pylist()
        ]
    if pa.types.is_floating(values.type):
        return [None if v is None or v != v else v for v in values.to_pylist()]
    return values.to_pylist()


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

    Each row of ``tmp`` is one element of the JSON array: ``__kumo_batch__``
    its position, and one column per ``(name, values)`` in ``columns``. A
    single column is passed as an array of values, several as an array of
    arrays.
    """
    names = [name for name, _ in columns]
    values = [_json_values(array) for _, array in columns]
    selects = ['key AS __kumo_batch__']
    for index, (name, array) in enumerate(columns):
        ref = (
            'value'
            if len(columns) == 1
            else f"json_extract(value, '$[{index}]')"
        )
        cast = _sqlite_type(array.type)
        selects.append(
            f'{ref} AS {name}'
            if cast is None
            else f'CAST({ref} AS {cast}) AS {name}'
        )
    payload: list[Any] = (
        values[0] if len(names) == 1 else [list(row) for row in zip(*values)]
    )
    sql = (
        f'WITH __kumo_keys__ AS (\n'
        f'  SELECT {", ".join(selects)}\n'
        f'  FROM json_each(?)\n'
        f')\n'
    )
    return _Keys(sql, json.dumps(payload, separators=(',', ':')))


def rows_by_key(
    table: str,
    key: str,
    ids: pa.Array,
    projections: Sequence[str],
    *,
    unique: bool,
    tie_break: Sequence[str],
    count_matches: bool,
) -> Statement:
    r"""For each id, one row of ``table`` holding it in ``key``, with the
    id's position as ``__kumo_batch__``.

    Of several rows holding an id, the first in ``tie_break``, then ``rowid``,
    order, so the same row is kept however they are stored. With
    ``count_matches``, ``__kumo_matches__`` counts the rows holding each id.
    """
    keys = _keys([('__kumo_id__', ids)])
    matches = (
        f'(SELECT COUNT(*) FROM {table} '
        f'WHERE {key} == tmp.__kumo_id__) as __kumo_matches__, '
        if count_matches
        else ''
    )
    sql = (
        f'{keys.sql}'
        f'SELECT '
        f'tmp.__kumo_batch__ as __kumo_batch__, '
        f'{matches}'
        f'{", ".join(projections)}\n'
        f'FROM __kumo_keys__ tmp\n'
        f'JOIN {table} ent\n'
    )
    if unique:
        sql += f'  ON {key} = tmp.__kumo_id__'
    else:
        order_by = ', '.join([*tie_break, 'rowid'])
        sql += (
            f'  ON ent.rowid = (\n'
            f'    SELECT rowid\n'
            f'    FROM {table}\n'
            f'    WHERE {key} == tmp.__kumo_id__\n'
            f'    ORDER BY {order_by}\n'
            f'    LIMIT 1\n'
            f')'
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
    in ``key``, in ``order_by`` order, and with the time ``time`` no later
    than its end time, if given as text spelled as the column is.
    """
    columns = [('__kumo_id__', ids)]
    if end_times is not None:
        columns.append(('__kumo_time__', end_times))
    keys = _keys(columns)
    sql = (
        f'{keys.sql}'
        f'SELECT '
        f'tmp.__kumo_batch__ as __kumo_batch__, '
        f'{", ".join(projections)}\n'
        f'FROM __kumo_keys__ tmp\n'
        f'JOIN {table} fact\n'
        f'ON fact.rowid IN (\n'
        f'  SELECT rowid\n'
        f'  FROM {table}\n'
        f'  WHERE {key} = tmp.__kumo_id__\n'
    )
    if end_times is not None:
        sql += f'  AND {time} <= tmp.__kumo_time__\n'
    sql += f'  ORDER BY {order_by}\n'
    sql += f'  LIMIT {num_neighbors}\n)\n'
    # Returned in that order too, so the rows reach the request the same way
    # however the table is stored.
    sql += f'ORDER BY __kumo_batch__, {order_by}'
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
    start, if given; both as text spelled as the column is. In ``order_by``
    order, which is the order they are aggregated in.
    """
    columns = [('__kumo_id__', ids), ('__kumo_end__', end_times)]
    if start_times is not None:
        columns.append(('__kumo_start__', start_times))
    keys = _keys(columns)
    sql = (
        f'{keys.sql}'
        f'SELECT '
        f'tmp.__kumo_batch__ as __kumo_batch__, '
        f'{", ".join(projections)}\n'
        f'FROM __kumo_keys__ tmp\n'
        f'JOIN {table}\n'
        f'  ON {key} = tmp.__kumo_id__\n'
        f' AND {time} <= tmp.__kumo_end__'
    )
    if start_times is not None:
        sql += f'\n AND {time} > tmp.__kumo_start__'
    sql += f'\nORDER BY __kumo_batch__, {order_by}'
    return sql, [keys.param]
