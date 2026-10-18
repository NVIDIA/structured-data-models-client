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
statement's text is given to it so written (:func:`_psycopg_text`), its
``$1`` the one outside quotes; one holding no such ``$1``, or several, is
not run. A plan this
provider cannot run, one for a host provider, one with a scalar parameter
or more than one key set in a statement, or one naming a key set it was
not handed, raises, and the run loop hands the core a provider error.

Each result column's Arrow type is read from the column's Postgres type,
by the type's name in the driver's registry, never by its number (a type
the registry does not name, a user's composite or enum, is read as text,
which is how the driver reads it); the values are the driver's, converted
by pyarrow. The driver reads each value as it reads the reference's (its
own loaders, so the statements' results are the reference's), and a value
is converted after the read where the core takes it another way: a
``json`` or ``jsonb`` value as the reference's text of it (compact JSON,
its field marked as JSON, Arrow's ``arrow.json``),
an ``interval`` (the driver's ``timedelta``, a month counted as 30 days as
the reference's driver counts it) as pyarrow's ``MonthDayNano``. A
``timestamptz`` is built as the instant in UTC, whatever the session's
zone.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import nullcontext
from decimal import Decimal
from typing import Any

#: The field metadata key marking a column of numbers handed over as their
#: exact text (``sdm_contract::data::EXACT_NUMBER_MARKER``).
EXACT_NUMBER_MARKER = 'sdm:exact_number'
#: The range of an ``int64``.
_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1
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
    }
)

#: The types read as the reference's text of their value (:func:`json_text`).
_JSON = ('json', 'jsonb')


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
    statement with a parameter: each ``%`` doubled, quoted or not (psycopg
    reads every ``%`` as a placeholder's start), and the parameter, the one
    ``$1`` outside a quoted identifier (``"..."``) and a text literal
    (``'...'``, or ``E'...'`` with its backslash escapes), the only quoting
    the dialect writes, as ``%s``, which psycopg sends to the server as
    ``$1`` again. A ``$1`` inside quotes, a relation's name holding it
    (``"users$1"``), is text and stays.

    Raises:
        PlanNotRun: The text holds no such ``$1``, or more than one, so its
            parameter is not told apart.
    """
    out: list[str] = []
    found = 0
    at = 0
    while at < len(text):
        char = text[at]
        if char == '"' or char == "'":
            escaped = char == "'" and at > 0 and text[at - 1] in 'Ee'
            end = at + 1
            while end < len(text):
                if escaped and text[end] == '\\':
                    end += 2
                    continue
                if text[end] == char:
                    if text[end + 1 : end + 2] == char:
                        end += 2
                        continue
                    break
                end += 1
            out.append(text[at : end + 1].replace('%', '%%'))
            at = end + 1
        elif text.startswith('$1', at) and not text[at + 2 : at + 3].isdigit():
            out.append('%s')
            found += 1
            at += 2
        else:
            out.append('%%' if char == '%' else char)
            at += 1
    if found != 1:
        raise PlanNotRun('a statement whose parameter is not told apart')
    return ''.join(out)


def json_text(value: Any) -> str | None:
    r"""A ``json`` or ``jsonb`` value as the reference writes it
    (``_sanitize_postgres_frame``'s ``stringify``): an object or array as
    compact JSON, ASCII only, a JSON null as a null, and any other value as
    Python's text of it."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(',', ':'))
    return str(value)


def _interval(value: Any) -> Any:
    r"""An ``interval`` the driver read (a ``timedelta``) as pyarrow's
    ``MonthDayNano``: its days, and the rest in nanoseconds."""
    import pyarrow as pa

    nanos = (value.seconds * 10**6 + value.microseconds) * 1000
    return pa.MonthDayNano((0, value.days, nanos))


def _after_read(column: Any, connection: Any) -> Any:
    r"""How a result column's driver values are converted after the read
    (:func:`json_text`, :func:`_interval`), an array's each element; None
    for a column taken as the driver reads it."""
    info = connection.adapters.types.get(column.type_code)
    if info is None:
        return None
    array = info.array_oid == column.type_code and info.oid != column.type_code
    one = (
        json_text
        if info.name in _JSON
        else _interval
        if info.name == 'interval'
        else None
    )
    if one is None:
        return None
    if not array:
        return one

    def each(value: Any) -> Any:
        if isinstance(value, list):
            return [each(item) for item in value]
        return None if value is None else one(value)

    return each


def _result(cursor: Any, connection: Any, batch_rows: int) -> Any:
    r"""The last statement's rows as a pyarrow table, in batches of at most
    ``batch_rows``, each column typed by its Postgres type."""
    import pyarrow as pa

    description = cursor.description
    if description is None:
        raise PlanNotRun('a plan whose last statement returns no rows')
    # One read of every row, as the reference reads a result
    # (``fetchall``), built into batches of at most ``batch_rows``.
    rows = cursor.fetchall()
    converts = [_after_read(column, connection) for column in description]
    if any(converts):
        rows = [
            tuple(
                value if convert is None or value is None else convert(value)
                for convert, value in zip(converts, row, strict=True)
            )
            for row in rows
        ]
    fields = []
    for at, column in enumerate(description):
        kind = _arrow_type(column, connection)
        fields.append(
            kind.field(column.name, (row[at] for row in rows))
            if isinstance(kind, _Numeric)
            else pa.field(column.name, kind)
        )
    schema = pa.schema(fields)
    types = [field.type for field in fields]
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


class _Numeric:
    r"""A ``numeric`` column no Arrow decimal holds.

    Declared without a precision, or with a negative scale (which the
    core's wire takes no decimal of), it is handed over as its exact text,
    marked :data:`EXACT_NUMBER_MARKER`, and the core reads it as the
    reference's Postgres table does, by its role: a key's text, any other
    column's double. Declared past the 76 digits Arrow's widest decimal
    holds, it is read by its declaration (the reference's ``_to_dtype``): of
    a scale of zero as an integer, a whole number past ``int64`` as its
    digits (where the reference overflows); else as a double."""

    def __init__(self, *, exact: bool, whole: bool) -> None:
        self.exact = exact
        self.whole = whole

    def field(self, name: str, values: Any) -> Any:
        import pyarrow as pa

        if self.exact:
            return pa.field(
                name, pa.string(), metadata={EXACT_NUMBER_MARKER: 'true'}
            )
        if not self.whole:
            return pa.field(name, pa.float64())
        for value in values:
            if value is not None and not _INT64_MIN <= value <= _INT64_MAX:
                return pa.field(name, pa.string())
        return pa.field(name, pa.int64())


def _array(values: Any, kind: Any) -> Any:
    import pyarrow as pa

    return pa.array([_cell(value, kind) for value in values], type=kind)


def _cell(value: Any, kind: Any) -> Any:
    r"""A driver's value in its column's Arrow form: a value of a type read
    as text (a ``uuid``, an ``inet``, a ``cidr``) as the reference's text of
    it, Python's ``str``; a ``numeric`` no Arrow decimal holds as the
    integer, text or double :class:`_Numeric` reads it as; an array's elements
    each so, a multi-dimensional array's in row-major order (Arrow's list
    holds one dimension; the core sends no list); anything else as the
    driver gives it."""
    import pyarrow as pa

    if value is None:
        return None
    if pa.types.is_list(kind):
        return [_cell(item, kind.value_type) for item in _elements(value)]
    if isinstance(value, Decimal) and pa.types.is_integer(kind):
        return int(value)
    if pa.types.is_string(kind) and not isinstance(value, str):
        return str(value)
    if pa.types.is_floating(kind) and isinstance(value, Decimal):
        return float(value)
    if (
        pa.types.is_decimal(kind)
        and isinstance(value, Decimal)
        and value.is_nan()
    ):
        # A numeric's NaN, which no Arrow decimal holds: missing, as the
        # reference's double of it is.
        return None
    if pa.types.is_time(kind) and value.tzinfo is not None:
        # A timetz's time of day, its offset left out, as the reference
        # reads a time.
        return value.replace(tzinfo=None)
    return value


def _elements(value: Any) -> list[Any]:
    r"""An array's elements, a multi-dimensional array's in row-major
    order."""
    found: list[Any] = []
    for item in value:
        if isinstance(item, list):
            found.extend(_elements(item))
        else:
            found.append(item)
    return found


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
    r"""An array element's Arrow type: the scalar's; a ``numeric`` element,
    whose array carries no scale of its own, a double, as the reference
    reads one; any other the text the driver gives."""
    import pyarrow as pa

    if name in _SCALARS:
        return getattr(pa, _SCALARS[name])()
    if name == 'numeric':
        return pa.float64()
    if name == 'timestamp':
        return pa.timestamp('us')
    if name == 'time':
        return pa.time64('us')
    return pa.string()


def _scalar_type(name: str, column: Any, connection: Any) -> Any:
    r"""A scalar column's Arrow type, or a :class:`_Numeric` for a
    ``numeric`` no Arrow decimal holds."""
    import pyarrow as pa

    if name in _SCALARS:
        return getattr(pa, _SCALARS[name])()
    if name == 'numeric':
        precision, scale = column.precision, column.scale
        if precision is None or (scale or 0) < 0 or precision > 76:
            # Declared without a precision or scale, with a negative scale,
            # which the core's wire takes no decimal of, or past the 76
            # digits Arrow's widest decimal holds: its exact text, or read
            # by its declaration and values (:class:`_Numeric`).
            return _Numeric(
                exact=precision is None or (scale or 0) < 0,
                whole=scale == 0,
            )
        if precision <= 38:
            return pa.decimal128(precision, scale or 0)
        return pa.decimal256(precision, scale or 0)
    if name == 'timestamp':
        return pa.timestamp('us')
    if name == 'timestamptz':
        # The instant, in UTC whatever the session's zone, so the type is
        # the same on every session (the comparator's vocabulary holds it).
        return pa.timestamp('us', tz='UTC')
    if name in ('time', 'timetz'):
        # A timetz as its time of day (:func:`_cell`), as the reference
        # reads a time.
        return pa.time64('us')
    if name == 'interval':
        return pa.month_day_nano_interval()
    if name in _JSON:
        # The reference's text of the value (json_text), marked as JSON
        # (Arrow's arrow.json over a string).
        return pa.json_()
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
