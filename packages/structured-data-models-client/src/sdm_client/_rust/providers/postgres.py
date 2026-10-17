# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The Postgres provider (T2.3): it answers a fetch step by running the
statements its plan carries, in order, on one cursor of the source's
psycopg connection, and hands back the last one's rows as an Arrow table,
built from the driver's values with no pandas, an empty result keeping its
schema.

The statements are the core's (``sdm-data`` renders Postgres's SQL); this
module writes none. Each request gets its own cursor, in a transaction of
its own as the reference's reads take one (``own_transaction``): none under
autocommit, a savepoint inside a transaction the caller opened, so a failed
read leaves the caller's connection usable.

A key set a plan names (the seeds of a ``neighbors`` or ``lookup``
request) is bound as the dialect binds it: the statement's one parameter,
``$1``, as JSON text, an array of the key set's rows, each an array of its
cells' texts in the key set's column order (``null`` for a null), which the
statement reads in the source's own types. Each cell's text is pyarrow's
cast of the value to text, exact for every type a key or a seed time
takes. psycopg writes a parameter ``%s`` and a ``%`` as ``%%``, so such a
statement's text is given to it so written (:func:`_psycopg_text`); one
whose ``$1`` is not its only one cannot be, and is not run. A plan this
provider cannot run, one for a host provider, one with a scalar parameter
or more than one key set in a statement, or one naming a key set it was
not handed, raises, and the run loop hands the core a provider error.

Each result column's Arrow type is read from the column's Postgres type,
by the type's name in the driver's registry, never by its number (a type
the registry does not name, a user's composite or enum, is read as text,
which is how the driver reads it); the values are the driver's, converted
by pyarrow. Where the driver's value would lose what the server holds, the
column is read as the server writes it instead: ``json`` and ``jsonb`` as
their text, and an ``interval`` from its text, months, days and
microseconds kept apart (``month_day_nano_interval``, each value pyarrow's
``MonthDayNano``). A ``timestamptz`` is built as the instant in UTC,
whatever the session's zone.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

#: Rows per Arrow batch; a larger result comes as several.
BATCH_ROWS = 100_000

#: The Arrow type of each Postgres type read as the driver gives it, by the
#: type's name in the driver's registry; ``numeric``, ``timestamptz``, the
#: types read as text and the arrays are read by :func:`_arrow_type`.
_SCALARS = {
    'bool': 'bool_',
    'int2': 'int16',
    'int4': 'int32',
    'int8': 'int64',
    'oid': 'int64',
    # Read as the reference reads it, a double of the value the server
    # writes (its shortest text), never the float32 it holds.
    'float4': 'float64',
    'float8': 'float64',
    'bytea': 'binary',
    'date': 'date32',
}

#: Postgres types read as text: the driver gives a ``str`` (or, for these,
#: a value whose ``str`` is the server's text), and Arrow has no type of
#: their own.
_TEXT = frozenset(
    {
        'text',
        'varchar',
        'bpchar',
        'char',
        'name',
        'uuid',
        'xml',
        'inet',
        'cidr',
        'macaddr',
        'macaddr8',
        'json',
        'jsonb',
        'money',
        'bit',
        'varbit',
        'timetz',
    }
)

#: The types read as the server's own text, not the driver's value.
_SERVER_TEXT = ('json', 'jsonb')


class PlanNotRun(Exception):
    r"""A plan this provider does not run. Its text names no value."""


class PostgresProvider:
    r"""Answers fetch steps on ``connection``, a psycopg connection."""

    def __init__(
        self, connection: Any, *, batch_rows: int = BATCH_ROWS
    ) -> None:
        self._connection = connection
        self._batch_rows = batch_rows

    def __call__(
        self, step: dict[str, Any], keys: Mapping[int, Any] | None = None
    ) -> Any:
        plan = step['plan']
        if plan['plan'] != 'sql':
            raise PlanNotRun("a host plan is not a SQL provider's to run")
        statements = plan['statements']
        if not statements or any(
            statement['dialect'] != 'postgres' for statement in statements
        ):
            raise PlanNotRun('a plan with no statement or another dialect')
        bound: list[str | None] = []
        for statement in statements:
            params = statement['params']
            if len(params) > 1:
                raise PlanNotRun('a statement binding more than one key set')
            if not params:
                bound.append(None)
                continue
            param = params[0]
            if param['param'] != 'key_set':
                raise PlanNotRun('a plan with a scalar parameter')
            attachment = param['keys']['attachment']
            if keys is None or attachment not in keys:
                raise PlanNotRun('a plan naming a key set not handed over')
            bound.append(key_set_json(_arrow_table(keys[attachment])))
        connection = self._connection
        with _own_transaction(connection), connection.cursor() as cursor:
            _read_as_the_server_writes(cursor)
            for statement, text in zip(statements, bound, strict=True):
                if text is None:
                    cursor.execute(statement['text'])
                else:
                    cursor.execute(_psycopg_text(statement['text']), (text,))
            return _result(cursor, connection, self._batch_rows)


def _own_transaction(connection: Any) -> Any:
    r"""The transaction one read runs in, as the reference's reads take
    one: none under autocommit, else a savepoint inside the caller's
    transaction or one of its own."""
    if connection.autocommit:
        return nullcontext()
    return connection.transaction()


def _psycopg_text(text: str) -> str:
    r"""A statement binding its key set as ``$1``, as psycopg reads a
    statement with a parameter: each ``%`` doubled and ``$1`` as ``%s``,
    which psycopg sends to the server as ``$1`` again.

    Raises:
        PlanNotRun: ``$1`` is not in the text exactly once (a name holding
            it), so its parameter is not told apart.
    """
    if text.count('$1') != 1:
        raise PlanNotRun('a statement whose parameter is not told apart')
    return text.replace('%', '%%').replace('$1', '%s')


def _read_as_the_server_writes(cursor: Any) -> None:
    r"""The cursor's loaders for the types read as the server writes them,
    on this cursor alone: ``json`` and ``jsonb`` as their text, and
    ``interval`` as its months, days and nanoseconds (:func:`interval`)."""
    from psycopg.adapt import Loader
    from psycopg.types.string import TextLoader

    class _Interval(Loader):
        def load(self, data: Any) -> Any:
            import pyarrow as pa

            return pa.MonthDayNano(interval(bytes(data).decode()))

    adapters = cursor.adapters
    for name in _SERVER_TEXT:
        adapters.register_loader(name, TextLoader)
    adapters.register_loader('interval', _Interval)


#: The units of an interval's text in Postgres's default style, as months
#: or days.
_INTERVAL_UNITS = {
    'year': ('months', 12),
    'years': ('months', 12),
    'mon': ('months', 1),
    'mons': ('months', 1),
    'day': ('days', 1),
    'days': ('days', 1),
}


def interval(text: str) -> tuple[int, int, int]:
    r"""An interval as Postgres writes it in its default style
    (``IntervalStyle`` ``postgres``: ``1 year 2 mons -3 days +04:05:06.5``),
    as months, days and nanoseconds, kept apart as the server keeps them.

    Raises:
        PlanNotRun: The text is in another style, which the session chose.
    """
    months = days = nanos = 0
    words = text.split()
    at = 0
    while at < len(words):
        word = words[at]
        if ':' in word:
            sign = -1 if word.startswith('-') else 1
            clock = word.lstrip('+-').split(':')
            if len(clock) != 3:
                raise PlanNotRun('an interval in a style other than postgres')
            whole, _, fraction = clock[2].partition('.')
            seconds = int(clock[0]) * 3600 + int(clock[1]) * 60 + int(whole)
            nanos += sign * (
                seconds * 10**9 + int((fraction + '000000000')[:9])
            )
            at += 1
            continue
        if at + 1 >= len(words) or words[at + 1] not in _INTERVAL_UNITS:
            raise PlanNotRun('an interval in a style other than postgres')
        part, scale = _INTERVAL_UNITS[words[at + 1]]
        if part == 'months':
            months += int(word) * scale
        else:
            days += int(word) * scale
        at += 2
    return months, days, nanos


def _result(cursor: Any, connection: Any, batch_rows: int) -> Any:
    r"""The last statement's rows as a pyarrow table, in batches of at most
    ``batch_rows``, each column typed by its Postgres type."""
    import pyarrow as pa

    description = cursor.description
    if description is None:
        raise PlanNotRun('a plan whose last statement returns no rows')
    types = [_arrow_type(column, connection) for column in description]
    schema = pa.schema(
        [
            pa.field(column.name, kind)
            for column, kind in zip(description, types, strict=True)
        ]
    )
    # One read of every row, as the reference reads a result
    # (``fetchall``), built into batches of at most ``batch_rows``.
    rows = cursor.fetchall()
    batches = []
    for start in range(0, len(rows), batch_rows):
        columns = list(zip(*rows[start : start + batch_rows], strict=True))
        batches.append(
            pa.record_batch(
                [
                    _array(values, kind)
                    for values, kind in zip(columns, types, strict=True)
                ],
                schema=schema,
            )
        )
    return pa.Table.from_batches(batches, schema=schema)


def _array(values: Any, kind: Any) -> Any:
    import pyarrow as pa

    return pa.array(values, type=kind)


def _arrow_type(column: Any, connection: Any) -> Any:
    r"""The Arrow type of a result column, read from its Postgres type by
    the type's name in the driver's registry."""
    import pyarrow as pa

    info = connection.adapters.types.get(column.type_code)
    if info is None:
        # A type the driver does not name (a user's composite or enum),
        # which it reads as text.
        return pa.string()
    if info.array_oid == column.type_code and info.oid != column.type_code:
        return pa.list_(_element_type(info.name))
    return _scalar_type(info.name, column, connection)


def _element_type(name: str) -> Any:
    r"""An array element's Arrow type: the scalar's, a ``numeric`` element
    read as the driver's value, of no declared scale (text)."""
    import pyarrow as pa

    if name in _SCALARS:
        return getattr(pa, _SCALARS[name])()
    if name == 'timestamp':
        return pa.timestamp('us')
    if name == 'time':
        return pa.time64('us')
    return pa.string()


def _scalar_type(name: str, column: Any, connection: Any) -> Any:
    import pyarrow as pa

    if name in _SCALARS:
        return getattr(pa, _SCALARS[name])()
    if name == 'numeric':
        precision, scale = column.precision, column.scale
        if precision is None:
            # Declared without a precision: any value, held as its text.
            return pa.string()
        if precision <= 38:
            return pa.decimal128(precision, scale or 0)
        return pa.decimal256(precision, scale or 0)
    if name == 'timestamp':
        return pa.timestamp('us')
    if name == 'timestamptz':
        # The instant, in UTC whatever the session's zone, so the type is
        # the same on every session (the comparator's vocabulary holds it).
        return pa.timestamp('us', tz='UTC')
    if name == 'time':
        return pa.time64('us')
    if name == 'interval':
        return pa.month_day_nano_interval()
    return pa.string()


def key_set_json(table: Any) -> str:
    r"""A key set as the dialect binds it: a JSON array of its rows, each
    an array of its cells' texts in the table's column order, ``null`` for
    a null."""
    import pyarrow as pa
    import pyarrow.compute as pc

    columns = []
    for column in table.columns:
        kind = column.type
        if pa.types.is_binary(kind) or pa.types.is_large_binary(kind):
            texts = [
                None if value is None else '\\x' + value.hex()
                for value in column.to_pylist()
            ]
        elif pa.types.is_timestamp(kind) and kind.tz is None:
            texts = [
                None if ticks is None else _isoformat(ticks, kind.unit)
                for ticks in column.cast(pa.int64()).to_pylist()
            ]
        else:
            texts = pc.cast(column, pa.string()).to_pylist()
        columns.append(texts)
    rows = [list(row) for row in zip(*columns, strict=True)] if columns else []
    return json.dumps(rows, separators=(',', ':'), ensure_ascii=False)


_PER_SECOND = {'s': 1, 'ms': 10**3, 'us': 10**6, 'ns': 10**9}


def _isoformat(ticks: int, unit: str) -> str:
    r"""A zone-less time as pandas' ``Timestamp.isoformat`` writes it, the
    text the reference binds a seed time as: ``YYYY-MM-DDTHH:MM:SS``, then
    six fractional digits where it has microseconds and nine where it has
    nanoseconds, none where it has neither. A text time column is compared
    with it as text, as the reference compares it."""
    import datetime as dt

    nanos = ticks * (10**9 // _PER_SECOND[unit])
    seconds, fraction = divmod(nanos, 10**9)
    moment = dt.datetime(1970, 1, 1) + dt.timedelta(seconds=seconds)
    text = moment.strftime('%Y-%m-%dT%H:%M:%S')
    if moment.year < 1000:
        text = f'{moment.year:04d}' + text[text.index('-') :]
    if fraction % 1000:
        return f'{text}.{fraction:09d}'
    if fraction:
        return f'{text}.{fraction // 1000:06d}'
    return text


def _arrow_table(stream: Any) -> Any:
    r"""A key set the core sent, an object exporting an Arrow stream, read
    whole as a pyarrow table, through the PyCapsule interface pyarrow 14.0.1
    reads (decision E16)."""
    import pyarrow as pa

    # The private importer, because pyarrow 14.0.1, the floor, has no public
    # reader of a stream capsule (RecordBatchReader.from_stream is 15's).
    return pa.RecordBatchReader._import_from_c_capsule(
        stream.__arrow_c_stream__()
    ).read_all()
