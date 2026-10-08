# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The SQL the engine runs against PostgreSQL.

Each statement is written by a plain function that returns ``(sql, params)``
and touches no connection; the tables and the sampler hand what it returns to
the source's ``_run``. Parameters are bound to ``%s`` markers.

A key list travels as one parameter holding a JSON array, which
``jsonb_array_elements`` turns into rows; a list longer than one parameter
should carry is split into chunks, each its own statement.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

import pandas as pd
from predictive_query.schema.typing import Dtype

from sdm_relational_engine.rfm.base.source import KeyCountPart, Statement
from sdm_relational_engine.rfm.base.utils import (
    count_distinct_values,
    key_count_select,
    ordered_sample_sql,
    repeated_key_select,
)

# Catalog #####################################################################


def current_schema() -> Statement:
    r"""The schema a bare name resolves in."""
    return 'SELECT current_schema()', None


def list_tables(schema: str) -> Statement:
    r"""Every base table of ``schema``, by name."""
    return (
        'SELECT table_name\n'
        'FROM information_schema.tables\n'
        'WHERE table_schema = %s\n'
        "  AND table_type = 'BASE TABLE'\n"
        'ORDER BY table_name',
        [schema],
    )


def columns(schema: str, table: str) -> Statement:
    r"""The columns of ``schema.table``, in order, with their types as
    PostgreSQL writes them.
    """
    return (
        'SELECT c.column_name, c.data_type, c.udt_name,\n'
        '       c.numeric_scale, c.is_nullable, c.table_schema,\n'
        '       c.table_name,\n'
        '       pg_catalog.format_type(a.atttypid, a.atttypmod)\n'
        'FROM information_schema.columns c\n'
        'JOIN pg_catalog.pg_namespace n\n'
        '  ON n.nspname = c.table_schema\n'
        'JOIN pg_catalog.pg_class rel\n'
        '  ON rel.relnamespace = n.oid AND rel.relname = c.table_name\n'
        'JOIN pg_catalog.pg_attribute a\n'
        '  ON a.attrelid = rel.oid AND a.attname = c.column_name\n'
        ' AND a.attnum > 0 AND NOT a.attisdropped\n'
        'WHERE c.table_schema = %s AND c.table_name = %s\n'
        'ORDER BY c.ordinal_position',
        [schema, table],
    )


def key_constraints(schema: str, table: str) -> Statement:
    r"""The primary, unique and foreign keys of ``schema.table``, by
    column.
    """
    return (
        'SELECT tc.constraint_type, tc.constraint_name,\n'
        '       kcu.column_name, kcu.ordinal_position\n'
        'FROM information_schema.table_constraints tc\n'
        'JOIN information_schema.key_column_usage kcu\n'
        '  ON tc.constraint_catalog = kcu.constraint_catalog\n'
        ' AND tc.constraint_schema = kcu.constraint_schema\n'
        ' AND tc.constraint_name = kcu.constraint_name\n'
        'WHERE tc.table_schema = %s AND tc.table_name = %s\n'
        '  AND tc.constraint_type IN '
        "('PRIMARY KEY', 'UNIQUE', 'FOREIGN KEY')",
        [schema, table],
    )


def foreign_keys(schema: str, table: str) -> Statement:
    r"""The foreign keys of ``schema.table``, with what each references."""
    return (
        'SELECT tc.constraint_name, kcu.column_name,\n'
        '       target_kcu.table_schema, target_kcu.table_name,\n'
        '       target_kcu.column_name, kcu.ordinal_position\n'
        'FROM information_schema.table_constraints tc\n'
        'JOIN information_schema.referential_constraints rc\n'
        '  ON tc.constraint_catalog = rc.constraint_catalog\n'
        ' AND tc.constraint_schema = rc.constraint_schema\n'
        ' AND tc.constraint_name = rc.constraint_name\n'
        'JOIN information_schema.key_column_usage kcu\n'
        '  ON tc.constraint_catalog = kcu.constraint_catalog\n'
        ' AND tc.constraint_schema = kcu.constraint_schema\n'
        ' AND tc.constraint_name = kcu.constraint_name\n'
        'JOIN information_schema.key_column_usage target_kcu\n'
        '  ON rc.unique_constraint_catalog = '
        'target_kcu.constraint_catalog\n'
        ' AND rc.unique_constraint_schema = '
        'target_kcu.constraint_schema\n'
        ' AND rc.unique_constraint_name = '
        'target_kcu.constraint_name\n'
        ' AND target_kcu.ordinal_position = '
        'kcu.position_in_unique_constraint\n'
        "WHERE tc.constraint_type = 'FOREIGN KEY'\n"
        '  AND tc.table_schema = %s AND tc.table_name = %s\n'
        'ORDER BY tc.constraint_name, kcu.ordinal_position',
        [schema, table],
    )


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


def relation_size(table: str) -> Statement:
    r"""The kind of the relation ``table``, a quoted name, and the blocks it
    holds now, every partition's for a partitioned table; no row when it
    names nothing ``to_regclass`` resolves.

    The blocks are counted from the files, so a table never analyzed is
    counted too, which ``reltuples`` and ``relpages`` are not.
    """
    return (
        'SELECT c.relkind,\n'
        "       CASE WHEN c.relkind = 'p'\n"
        '            THEN (SELECT COALESCE(SUM(pg_relation_size(p.relid)), 0)'
        '::bigint\n'
        '                  FROM pg_partition_tree(c.oid) AS p)\n'
        '            ELSE pg_relation_size(c.oid)\n'
        "       END / current_setting('block_size')::bigint\n"
        'FROM pg_catalog.pg_class AS c\n'
        'WHERE c.oid = to_regclass(%s)',
        [table],
    )


def block_sample(table: str, percent: str) -> str:
    r"""``table`` with a repeatable sample of ``percent`` of its blocks."""
    return f'{table} TABLESAMPLE SYSTEM ({percent}) REPEATABLE (0)'


def holds_any(table: str, identity: str, values: Sequence[str]) -> Statement:
    r"""Whether a row of ``table`` holds one of ``values`` in ``identity``,
    the expression of a folded composite key: one row at most is read, and
    ``values`` travel as one JSON array.
    """
    return (
        f'SELECT 1 FROM {table}\n'
        f'WHERE ({identity}) IN (\n'
        f'  SELECT jsonb_array_elements_text(%s::jsonb)\n'
        f')\n'
        f'LIMIT 1',
        [dumps(list(values))],
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


def _json_default(obj: Any) -> Any:
    if hasattr(obj, 'item'):
        return obj.item()
    return str(obj)


def dumps(obj: Any) -> str:
    r"""``obj`` as compact JSON, numpy scalars as Python ones and anything
    else JSON has no value for as its text.
    """
    return json.dumps(obj, separators=(',', ':'), default=_json_default)


def chunk_rows(rows: list, max_bytes: int = 900_000) -> list[list]:
    r"""``rows`` in chunks whose JSON stays below ``max_bytes``."""
    chunks: list[list] = []
    current: list = []
    current_bytes = 2
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
    r"""The type a key of ``dtype`` with no declared type is read as."""
    if dtype.is_int():
        return 'bigint'
    if dtype.is_float():
        return 'double precision'
    return 'text'


def coerce_id(value: Any, dtype: Dtype, postgres_type: str) -> Any:
    r"""``value`` as JSON holds a key of ``dtype``, declared as
    ``postgres_type``: a ``numeric`` key as its text, so no digit is lost.
    """
    if hasattr(value, 'item'):
        value = value.item()
    if dtype.is_int():
        return int(value)
    if dtype.is_float():
        if postgres_type.lower().startswith(('numeric', 'decimal')):
            return str(value)
        return float(value)
    return str(value)


def serialize_time(value: Any, utc: bool = False) -> str | None:
    r"""A time bound as JSON holds it.

    With ``utc``, a naive bound, which the engine means as UTC, is written
    with ``+00:00``: a ``timestamptz`` column, or a domain over one, then
    reads the instant meant whatever the session's ``TimeZone``, and a
    ``timestamp`` or ``date`` column ignores the offset. A bound for a text
    column is left bare, so it compares with the text as it always did.
    """
    if pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    if utc:
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize('UTC')
        else:
            timestamp = timestamp.tz_convert('UTC')
    return timestamp.isoformat()


def postgres_seed(random_seed: int) -> float:
    r"""Map an integer seed stably into PostgreSQL's ``[-1, 1]`` range.

    Hashing avoids the guaranteed collisions every 2,000,001 values that a
    direct modulo mapping would introduce. PostgreSQL accepts a double, so the
    final mapping necessarily has finite precision.
    """
    digest = hashlib.blake2b(
        str(random_seed).encode('ascii'), digest_size=8
    ).digest()
    value = int.from_bytes(digest, byteorder='big', signed=False)
    return 2.0 * value / ((1 << 64) - 1) - 1.0


def tie_break_term(ref: str, is_string: bool) -> str:
    r"""The ``ORDER BY`` term of a column in a tie-break.

    json and xml read as strings but cannot be ordered, and a domain can hide
    either, so every string column is ordered by its text.
    """
    if is_string:
        return f'CAST({ref} AS text)'
    return ref


def json_cte(
    postgres_type: str,
    fields: tuple[str, ...] = (),
    time_type: str = 'timestamp',
) -> str:
    r"""The ``WITH TMP`` clause that reads the key list, the statement's
    first parameter, as rows: ``__KUMO_BATCH__`` the position, ``__KUMO_ID__``
    the key, and the end (``e``) and start (``s``) times of each in
    ``fields``.
    """
    select = ['(ordinality - 1)::bigint AS __KUMO_BATCH__']
    if fields:
        select.append(f"(value->>'id')::{postgres_type} AS __KUMO_ID__")
        if 'e' in fields:
            select.append(f"(value->>'e')::{time_type} AS __KUMO_END_TIME__")
        if 's' in fields:
            select.append(f"(value->>'s')::{time_type} AS __KUMO_START_TIME__")
    else:
        select.append(f"(value #>> '{{}}')::{postgres_type} AS __KUMO_ID__")
    return (
        'WITH TMP AS (\n'
        f'  SELECT {", ".join(select)}\n'
        '  FROM jsonb_array_elements(%s::jsonb) WITH ORDINALITY\n'
        '       AS input(value, ordinality)\n'
        ')\n'
    )


def time_ranges(tables: Sequence[tuple[str, str]]) -> Statement:
    r"""The earliest and latest time of each ``(table, time column)``."""
    selects = [
        f'SELECT {index} AS table_index, MIN({column}) AS '
        f'min_date, MAX({column}) AS max_date\n'
        f'FROM {table}'
        for index, (table, column) in enumerate(tables)
    ]
    return '\nUNION ALL\n'.join(selects), None


def set_seed(random_seed: int) -> Statement:
    r"""Seed ``random()`` for the session, from ``random_seed``."""
    return 'SELECT setseed(%s)', [postgres_seed(random_seed)]


def random_rows(
    table: str,
    projections: Sequence[str],
    filters: Sequence[str],
    num_rows: int,
) -> Statement:
    r"""``num_rows`` rows of ``table`` holding ``filters``, drawn by
    ``random()``.
    """
    sql = f'SELECT {", ".join(projections)}\nFROM {table}'
    if filters:
        sql += f'\nWHERE {" AND ".join(filters)}'
    sql += f'\nORDER BY random()\nLIMIT {num_rows}'
    return sql, None


def rows_by_ids(
    table: str,
    projections: Sequence[str],
    filters: Sequence[str],
    *,
    key: str,
    postgres_type: str,
    ids: list,
    order_by: str,
) -> Statement:
    r"""The rows of ``table`` holding ``ids``, a chunk of them, in ``key``,
    and ``filters``, in the order of the ids, then ``order_by``.
    """
    sql = json_cte(postgres_type)
    sql += (
        f'SELECT {", ".join(projections)}\n'
        'FROM TMP\n'
        f'JOIN {table}\n'
        f'  ON {key} = TMP.__KUMO_ID__'
    )
    if filters:
        sql += f'\nWHERE {" AND ".join(filters)}'
    sql += f'\nORDER BY TMP.__KUMO_BATCH__, {order_by}'
    return sql, [dumps(ids)]


def rows_by_key(
    table: str,
    projections: Sequence[str],
    *,
    key: str,
    postgres_type: str,
    ids: list,
    order_by: str,
) -> Statement:
    r"""For each of ``ids``, a chunk of them, the first row of ``table``
    holding it in ``key`` in ``order_by`` order, with how many do.
    """
    sql = json_cte(postgres_type)
    sql += (
        'SELECT * FROM (\n'
        '  SELECT TMP.__KUMO_BATCH__, '
        f'{", ".join(projections)},\n'
        '         ROW_NUMBER() OVER (PARTITION BY '
        'TMP.__KUMO_BATCH__\n'
        f'           ORDER BY {order_by}) AS __KUMO_ROW__,\n'
        '         COUNT(*) OVER (PARTITION BY TMP.__KUMO_BATCH__) '
        'AS __KUMO_MATCHES__\n'
        '  FROM TMP\n'
        f'  JOIN {table}\n'
        f'    ON {key} = TMP.__KUMO_ID__\n'
        ') AS sampled\n'
        'WHERE __KUMO_ROW__ = 1\n'
        'ORDER BY __KUMO_BATCH__'
    )
    return sql, [dumps(ids)]


def neighbors(
    table: str,
    projections: Sequence[str],
    *,
    key: str,
    postgres_type: str,
    rows: list,
    fields: tuple[str, ...],
    time: str | None,
    time_type: str,
    order_by: str,
    num_neighbors: int,
) -> Statement:
    r"""For each of ``rows``, a chunk of keys, with an end time where
    ``fields`` holds ``e``, the first ``num_neighbors`` rows of ``table``
    holding it in ``key`` and no later than its end, in ``order_by`` order.
    """
    sql = json_cte(postgres_type, fields, time_type)
    sql += (
        'SELECT TMP.__KUMO_BATCH__, sampled.*\n'
        'FROM TMP\n'
        'JOIN LATERAL (\n'
        f'  SELECT {", ".join(projections)},\n'
        f'         ROW_NUMBER() OVER (ORDER BY {order_by}) '
        'AS __KUMO_ROW__\n'
        f'  FROM {table}\n'
        f'  WHERE {key} = TMP.__KUMO_ID__\n'
    )
    if fields:
        sql += f'    AND {time} <= TMP.__KUMO_END_TIME__\n'
    sql += (
        f'  ORDER BY {order_by}\n'
        f'  LIMIT {num_neighbors}\n'
        ') AS sampled ON TRUE\n'
        'ORDER BY __KUMO_BATCH__, __KUMO_ROW__'
    )
    return sql, [dumps(rows)]


def window(
    table: str,
    projections: Sequence[str],
    *,
    key: str,
    postgres_type: str,
    rows: list,
    fields: tuple[str, ...],
    time: str,
    time_type: str,
    order_by: str,
) -> Statement:
    r"""For each of ``rows``, a chunk of keys with their end and, where
    ``fields`` holds ``s``, start times, the rows of ``table`` holding it in
    ``key`` in its window, in ``order_by`` order.
    """
    sql = json_cte(postgres_type, fields, time_type)
    sql += (
        'SELECT TMP.__KUMO_BATCH__, '
        f'{", ".join(projections)}\n'
        'FROM TMP\n'
        f'JOIN {table}\n'
        f'  ON {key} = TMP.__KUMO_ID__\n'
        f' AND {time} <= TMP.__KUMO_END_TIME__\n'
    )
    if 's' in fields:
        sql += f' AND {time} > TMP.__KUMO_START_TIME__\n'
    sql += f'ORDER BY TMP.__KUMO_BATCH__, {order_by}'
    return sql, [dumps(rows)]
