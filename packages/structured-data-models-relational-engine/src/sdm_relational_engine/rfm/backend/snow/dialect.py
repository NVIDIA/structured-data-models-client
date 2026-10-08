# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The SQL the engine runs against Snowflake.

Each statement is written by a plain function that returns ``(sql, params)``
and touches no connection; the tables, the sampler and the semantic view
reader hand what it returns to the source's ``_run``. Parameters are bound to
``?`` markers, by Snowflake itself, which is how a caller's name or value
reaches it as data rather than as SQL text.

A key list travels as one parameter holding a JSON array, which
``FLATTEN(PARSE_JSON(...))`` turns into rows.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from predictive_query.schema.typing import Dtype
from sdm_connectors import quote_ident

from sdm_relational_engine.rfm.base.source import KeyCountPart, Statement
from sdm_relational_engine.rfm.base.utils import (
    count_distinct_values,
    key_count_select,
    ordered_sample_sql,
    repeated_key_select,
)

#: The most rows ``SAMPLE ROW`` draws.
MAX_SAMPLE_ROWS = 1_000_000

# Catalog #####################################################################


def current_namespace() -> Statement:
    r"""The database and schema a bare name resolves in."""
    return 'SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()', None


def list_tables(database: str, schema: str) -> Statement:
    r"""Every table of ``schema`` in ``database``, by name."""
    return (
        '\n'
        '                    SELECT TABLE_NAME\n'
        f'                    FROM {quote_ident(database)}'
        '.INFORMATION_SCHEMA.TABLES\n'
        '                    WHERE TABLE_SCHEMA = ?\n'
        '                    ORDER BY TABLE_NAME\n'
        '                    ',
        [schema],
    )


def describe_table(table: str) -> Statement:
    r"""The columns of ``table``, a quoted name."""
    return f'DESCRIBE TABLE {table}', None


def imported_keys(table: str) -> Statement:
    r"""The foreign keys of ``table``, a quoted name."""
    return f'SHOW IMPORTED KEYS IN TABLE {table}', None


def tables_in_schema(database: str, schema: str) -> Statement:
    r"""Every table of ``schema`` in ``database``, with whether it is a
    hybrid table.
    """
    return (
        f'SHOW TABLES IN SCHEMA {quote_ident(database)}.{quote_ident(schema)}',
        None,
    )


def tables_like(database: str, schema: str, table: str) -> Statement:
    r"""The tables of ``schema`` in ``database`` named like ``table``, with
    their row counts. ``LIKE`` matches case-insensitively, and ``_`` matches
    any character.
    """
    return (
        f'SHOW TABLES LIKE ? '
        f'IN SCHEMA {quote_ident(database)}.'
        f'{quote_ident(schema)}',
        [table],
    )


def semantic_view(name: str) -> Statement:
    r"""The YAML definition of the semantic view ``name``."""
    return 'SELECT SYSTEM$READ_YAML_FROM_SEMANTIC_VIEW(?)', [name]


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


def block_sample(percent: str, seed: int) -> str:
    r"""The clause after a table's name that keeps ``percent`` of its
    micro-partitions, the same ones for the same ``seed`` while the table is
    unchanged.
    """
    return f'SAMPLE SYSTEM ({percent}) SEED ({seed})'


def holds_any(table: str, identity: str, values: Sequence[str]) -> Statement:
    r"""Whether a row of ``table`` holds one of ``values`` in ``identity``,
    the expression of a folded composite key: one row at most is read, and
    ``values`` travel as one JSON array.
    """
    return (
        f'SELECT 1 FROM {table}\n'
        f'WHERE ({identity}) IN (\n'
        f'  SELECT f.value::VARCHAR\n'
        f'  FROM TABLE(FLATTEN(INPUT => PARSE_JSON(?))) f\n'
        f')\n'
        f'LIMIT 1',
        [json.dumps(list(values))],
    )


# Keys ########################################################################


def key_counts(parts: Sequence[KeyCountPart]) -> Statement:
    r"""The rows, distinct keys and rows missing a key of each table, in one
    ``UNION ALL``. Snowflake counts several columns distinct, skipping a row
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


def seeded_key_hash(
    key: str, dtype: Dtype | None, is_date: bool, random_seed: int
) -> str:
    r"""A hash of the seed and a row's ``key``, the same in every session.

    ``TO_VARCHAR`` renders dates, times and binary values as the session's
    ``*_OUTPUT_FORMAT`` and ``TIMEZONE`` say, so hashing that text would draw
    different rows from one seed under different settings. Each key is
    rendered in a form no session parameter changes: strings as they are,
    numbers and booleans by ``TO_VARCHAR``, which no parameter formats, dates
    and times with an explicit format, timestamps as nanoseconds since the
    epoch and binary values as hex. A ``DATE`` is told from a timestamp by its
    source type, ``is_date``, as ``DATE_PART`` gives no ``EPOCH_NANOSECOND``
    of a ``DATE``.

    Args:
        key: The key column, quoted.
        dtype: Its type.
        is_date: Whether its source type is ``DATE``.
        random_seed: The seed of the draw.
    """
    if dtype == Dtype.string:
        text = key
    elif dtype == Dtype.date and is_date:
        text = f"TO_VARCHAR({key}, 'YYYY-MM-DD')"
    elif dtype == Dtype.date:  # Every `TIMESTAMP` type.
        text = f'TO_VARCHAR(DATE_PART(EPOCH_NANOSECOND, {key}))'
    elif dtype == Dtype.time:
        text = f"TO_VARCHAR({key}, 'HH24:MI:SS.FF9')"
    elif dtype == Dtype.binary:
        text = f"TO_VARCHAR({key}, 'HEX')"
    else:
        text = f'TO_VARCHAR({key})'
    return f"MD5_NUMBER_LOWER64('{int(random_seed)}:' || {text})"


def entity_rows(
    table: str,
    projections: Sequence[str],
    filters: Sequence[str],
    *,
    order_by: str,
    num_rows: int,
    seeded: str | None = None,
    key: str | None = None,
    ids: Sequence[Any] | None = None,
    prefilter: str | None = None,
) -> Statement:
    r"""Rows of an entity table: ``num_rows`` drawn by ``SAMPLE ROW``, the
    ``num_rows`` lowest in the ``seeded`` hash, or the rows holding ``ids``
    in ``key``, bound as parameters, in ``order_by`` order.

    Snowflake seeds only ``SYSTEM`` sampling, which keeps whole blocks rather
    than a number of rows, so a seeded draw is the rows whose key hashes
    lowest with the seed instead: as random, and the same every time. A
    seeded draw from a large table hashes only the micro-partitions
    ``prefilter``, a :func:`block_sample` clause, keeps.
    """
    conditions = list(filters)
    params: list[Any] | None = None
    if ids is not None:
        placeholders = ', '.join(['?'] * len(ids))
        conditions.append(f' {key} IN ({placeholders})')
        params = list(ids)
    if ids is not None or seeded is not None:
        sql = f'SELECT {", ".join(projections)}\nFROM {table}'
        if seeded is not None and prefilter is not None:
            sql += f'\n{prefilter}'
    else:
        sql = (
            f'SELECT {", ".join(projections)}\n'
            f'FROM {table}\n'
            f'SAMPLE ROW ({num_rows} ROWS)'
        )
    if len(conditions) > 0:
        sql += f'\nWHERE{" AND".join(conditions)}'
    if ids is not None:
        sql += f'\nORDER BY {order_by}'
    elif seeded is not None:
        sql += f'\nORDER BY {seeded}, {order_by}\nLIMIT {num_rows}'
    return sql, params


#: How a bound on a zoned time column is written, and the format Snowflake
#: reads it with: explicit, so no ``TIMESTAMP_INPUT_FORMAT`` changes it.
ZONED_BOUND_FORMAT = '%Y-%m-%dT%H:%M:%S+00:00'
_ZONED_BOUND_SNOWFLAKE_FORMAT = 'YYYY-MM-DD"T"HH24:MI:SSTZH:TZM'

#: How a bound on a naive or ``DATE`` time column is written.
NAIVE_BOUND_FORMAT = '%Y-%m-%d %H:%M:%S'


def zone_of(snowflake_type: str | None) -> str | None:
    r"""``LTZ`` or ``TZ`` for a ``TIMESTAMP_LTZ`` or ``TIMESTAMP_TZ`` type as
    ``DESCRIBE TABLE`` spells it, and ``None`` for any other, naive ones and
    ``DATE`` among them.
    """
    spelled = (snowflake_type or '').strip().upper()
    if spelled.startswith('TIMESTAMP_LTZ'):
        return 'LTZ'
    if spelled.startswith('TIMESTAMP_TZ'):
        return 'TZ'
    return None


def _bound(value: str, zone: str | None) -> str:
    r"""A window bound, ``value``, as the time column compares with it.

    A naive bound is read as ``TIMESTAMP_NTZ``, as it always was. For a
    ``TIMESTAMP_LTZ`` or ``TIMESTAMP_TZ`` column that would be converted in
    the session's ``TIMEZONE``, America/Los_Angeles by default, so the bound
    carries ``+00:00`` and is read as the instant it names.
    """
    if zone is None:
        return f'{value}::TIMESTAMP_NTZ'
    return f"TO_TIMESTAMP_{zone}({value}, '{_ZONED_BOUND_SNOWFLAKE_FORMAT}')"


def _outer_bound(zone: str | None) -> str:
    r"""The placeholder of a bound outside the key list."""
    if zone is None:
        return '?'
    return f"TO_TIMESTAMP_{zone}(?, '{_ZONED_BOUND_SNOWFLAKE_FORMAT}')"


def _key_list_time(index: int, zone: str | None) -> str:
    r"""The ``index``-th item of a key list entry, a time bound."""
    if zone is None:
        return _bound(f'f.value[{index}]', zone)
    return _bound(f'f.value[{index}]::VARCHAR', zone)


def _cast(dtype: Dtype) -> str:
    if dtype.is_int():
        return 'NUMBER'
    if dtype.is_float():
        return 'FLOAT'
    return 'VARCHAR'


def rows_by_key(
    table: str,
    projections: Sequence[str],
    *,
    key: str,
    dtype: Dtype,
    ids: list,
    order_by: Sequence[str],
) -> Statement:
    r"""The rows of ``table`` holding each of ``ids`` in ``key``, with the
    id's position as ``__KUMO_BATCH__``, in that order and then
    ``order_by``.
    """
    sql = 'WITH TMP as (\n  SELECT\n    f.index as __KUMO_BATCH__,\n'
    sql += f'    f.value::{_cast(dtype)} as __KUMO_ID__\n'
    sql += (
        f'  FROM TABLE(FLATTEN(INPUT => PARSE_JSON(?))) f\n'
        f')\n'
        f'SELECT '
        f'TMP.__KUMO_BATCH__ as __KUMO_BATCH__, '
        f'{", ".join(projections)}\n'
        f'FROM TMP\n'
        f'JOIN {table}\n'
        f'  ON {key} = TMP.__KUMO_ID__\n'
    )
    sql += f'ORDER BY {", ".join(["TMP.__KUMO_BATCH__", *order_by])}'
    return sql, [json.dumps(ids)]


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
    zone: str | None = None,
) -> Statement:
    r"""For each of ``rows``, a key with, if ``windowed``, its end and start
    time, the first ``num_neighbors`` rows of ``table`` holding it in ``key``
    and in its window, in ``order_by`` order. The ``bounds`` of every window
    let Snowflake prune partitions. ``zone`` is the time column's, as
    :func:`zone_of` says.
    """
    sql = 'WITH TMP as (\n  SELECT\n    f.index as __KUMO_BATCH__,\n'
    sql += f'    f.value[0]::{_cast(dtype)} as __KUMO_ID__'
    if windowed:
        sql += (
            ',\n'
            f'    {_key_list_time(1, zone)} as __KUMO_END_TIME__,\n'
            f'    {_key_list_time(2, zone)} as __KUMO_START_TIME__'
        )
    sql += (
        f'\n'
        f'  FROM TABLE(FLATTEN(INPUT => PARSE_JSON(?))) f\n'
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
            f'WHERE {time} <= {_outer_bound(zone)}\n'
            f'  AND {time} > {_outer_bound(zone)}\n'
        )
    sql += 'QUALIFY ROW_NUMBER() OVER (\n  PARTITION BY TMP.__KUMO_BATCH__\n'
    sql += f'  ORDER BY {order_by}\n'
    sql += f') <= {num_neighbors}\n'
    # Returned in that order too, so the rows reach the request the same way
    # however the warehouse reads them.
    sql += f'ORDER BY TMP.__KUMO_BATCH__, {order_by}'
    return sql, [json.dumps(rows), *bounds]


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
    zone: str | None = None,
) -> Statement:
    r"""For each of ``rows``, a key with its end and, if ``with_start``,
    start time, the rows of ``table`` holding it in ``key`` in its window, in
    ``order_by`` order. The ``bounds`` of every window, bound like every
    other value, let Snowflake prune partitions. ``zone`` is the time
    column's, as :func:`zone_of` says.
    """
    sql = 'WITH TMP as (\n  SELECT\n    f.index as __KUMO_BATCH__,\n'
    sql += f'    f.value[0]::{_cast(dtype)} as __KUMO_ID__,\n'
    sql += f'    {_key_list_time(1, zone)} as __KUMO_END_TIME__'
    if with_start:
        sql += f',\n    {_key_list_time(2, zone)} as __KUMO_START_TIME__'
    sql += (
        f'\n'
        f'  FROM TABLE(FLATTEN(INPUT => PARSE_JSON(?))) f\n'
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
    sql += f'WHERE {time} <= {_outer_bound(zone)}'
    if with_start:
        sql += f'\nAND {time} > {_outer_bound(zone)}'
    sql += f'\nORDER BY TMP.__KUMO_BATCH__, {order_by}'
    return sql, [json.dumps(rows), *bounds]
