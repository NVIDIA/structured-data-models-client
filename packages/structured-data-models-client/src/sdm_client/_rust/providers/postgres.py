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
cells in the key set's column order (``null`` for a null), which the
statement reads in the source's own types. The core hands the key set over
with the cells the reference spells otherwise than Arrow's text of them
(a time without a zone, binary) already as that text (``sdm-data``'s
Postgres value rules, T2.14d), and each cell is written as Arrow's text of
it.
psycopg writes a parameter ``%s`` and a ``%`` as ``%%``, so such a
statement's text is given to it so written (:func:`_psycopg_text`), its
``$1`` the one outside quotes; one holding no such ``$1``, or several, is
not run. A plan this
provider cannot run, one for a host provider, one with a scalar parameter
or more than one key set in a statement, or one naming a key set it was
not handed, raises, and the run loop hands the core a provider error.

Each result column's Arrow type is read from the column's Postgres type,
by the type's name in the driver's registry, never by its number (a type
the registry does not name, a user's composite or enum, is read as text,
which is how the driver reads it); the values are the driver's, moved into
Arrow by pyarrow, and how the reference reads each is the core's
(``sdm-data``'s Postgres value rules, T2.14d). The driver reads each value
as it reads the reference's (its own loaders, so the statements' results
are the reference's): a ``json`` or ``jsonb`` value is handed over as the
JSON text of the value the driver read, its field marked as JSON (Arrow's
``arrow.json`` metadata, :data:`JSON_TAG`); an ``interval`` (the driver's
``timedelta``, a month counted as 30 days as the reference's driver counts
it) as pyarrow's ``MonthDayNano`` of it, its field tagged ``sdm:interval``
= ``timedelta`` (:data:`INTERVAL_TAG`); a ``numeric`` no Arrow decimal
holds as each value's exact text, its declaration marked
(:func:`_numeric_field`). A ``timestamptz`` is built as the instant in UTC,
whatever the session's zone.
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
#: The field metadata key naming the declared scale of a column of numbers
#: handed over as their exact text because their declaration is past the
#: digits an Arrow decimal holds
#: (``sdm_contract::data::DECLARED_SCALE_MARKER``).
DECLARED_SCALE_MARKER = 'sdm:declared_scale'
#: The field metadata of Arrow's ``arrow.json`` extension over a string,
#: marking a column of JSON text.
JSON_TAG = {
    'ARROW:extension:name': 'arrow.json',
    'ARROW:extension:metadata': '',
}
#: The field metadata key, and its value, marking an interval column built
#: from the driver's ``timedelta``s, a month counted as 30 days as psycopg
#: counts it.
INTERVAL_TAG = ('sdm:interval', 'timedelta')
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

#: The types handed over as the JSON text of their value.
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
            bound.append(_bound_text(_arrow_table(keys[attachment])))
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
    fields = [_field(column, connection) for column in description]
    schema = pa.schema(fields)
    batches = []
    for start in range(0, len(rows), batch_rows):
        columns = list(zip(*rows[start : start + batch_rows], strict=True))
        batches.append(
            pa.record_batch(
                [
                    _array(values, field)
                    for values, field in zip(columns, fields, strict=True)
                ],
                schema=schema,
            )
        )
    return pa.Table.from_batches(batches, schema=schema)


def _field(column: Any, connection: Any) -> Any:
    r"""A result column's Arrow field: its type by its Postgres type
    (:func:`_arrow_type`), and its metadata (:func:`_tags`)."""
    import pyarrow as pa

    kind = _arrow_type(column, connection)
    if isinstance(kind, pa.Field):
        return kind.with_name(column.name)
    return pa.field(column.name, kind, metadata=_tags(column, connection))


def _numeric_field(precision: Any, scale: Any) -> Any:
    r"""A ``numeric`` column no Arrow decimal holds, as each value's exact
    text, its declaration marked: declared without a precision, or with a
    negative scale (which the core's wire takes no decimal of),
    :data:`EXACT_NUMBER_MARKER`, which the core reads by the column's role;
    declared past the 76 digits Arrow's widest decimal holds,
    :data:`DECLARED_SCALE_MARKER`, its declared scale, which the core reads
    by its declaration."""
    import pyarrow as pa

    if precision is None or (scale or 0) < 0:
        return pa.field('', pa.string(), metadata={EXACT_NUMBER_MARKER: 'true'})
    return pa.field(
        '', pa.string(), metadata={DECLARED_SCALE_MARKER: str(scale or 0)}
    )


def _tags(column: Any, connection: Any) -> dict[str, str] | None:
    r"""A result column's field metadata: a ``json`` or ``jsonb`` column's
    text marked as JSON (Arrow's ``arrow.json`` extension, written as its
    metadata over a string, which every supported pyarrow keeps); an
    ``interval`` column's built from the driver's ``timedelta``s
    (:data:`INTERVAL_TAG`); none for any other."""
    info = connection.adapters.types.get(column.type_code)
    if info is None or info.oid != column.type_code:
        return None
    if info.name in _JSON:
        return dict(JSON_TAG)
    if info.name == 'interval':
        return dict([INTERVAL_TAG])
    return None


def _array(values: Any, field: Any) -> Any:
    import pyarrow as pa

    kind = field.type
    if (field.metadata or {}).get(b'ARROW:extension:name') == b'arrow.json':
        # The JSON text of the value the driver read; what the reference
        # makes of it is the core's.
        return pa.array(
            [
                None
                if value is None
                else json.dumps(
                    value, separators=(',', ':'), ensure_ascii=False
                )
                for value in values
            ],
            type=kind,
        )
    return pa.array([_cell(value, kind) for value in values], type=kind)


def _cell(value: Any, kind: Any) -> Any:
    r"""A driver's value in its column's Arrow form: a value of a type read
    as text (a ``uuid``, an ``inet``, a ``cidr``) as the reference's text of
    it, Python's ``str``, a ``numeric`` no Arrow decimal holds among them;
    an array's elements
    each so, a multi-dimensional array's in row-major order (Arrow's list
    holds one dimension; the core sends no list); anything else as the
    driver gives it."""
    import pyarrow as pa

    if value is None:
        return None
    if pa.types.is_list(kind):
        return [_cell(item, kind.value_type) for item in _elements(value)]
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
    r"""A scalar column's Arrow type, or the field of a ``numeric`` no
    Arrow decimal holds (:func:`_numeric_field`)."""
    import pyarrow as pa

    if name in _SCALARS:
        return getattr(pa, _SCALARS[name])()
    if name == 'numeric':
        precision, scale = column.precision, column.scale
        if precision is None or (scale or 0) < 0 or precision > 76:
            # Declared without a precision or scale, with a negative scale,
            # which the core's wire takes no decimal of, or past the 76
            # digits Arrow's widest decimal holds: its exact text, its
            # declaration marked (:func:`_numeric_field`).
            return _numeric_field(precision, scale)
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
    return pa.string()


def _bound_text(table: Any) -> str:
    r"""A key set as the dialect binds it: a JSON array of its rows, each
    an array of its cells' texts in the table's column order, ``null`` for
    a null. Each cell's text is Arrow's text of the value (pyarrow's cast),
    the core having handed over as their text the columns the reference
    spells otherwise (a time without a zone, binary)."""
    import pyarrow as pa
    import pyarrow.compute as pc

    columns = [
        pc.cast(column, pa.string()).to_pylist() for column in table.columns
    ]
    rows = [list(row) for row in zip(*columns, strict=True)] if columns else []
    return json.dumps(rows, separators=(',', ':'), ensure_ascii=False)


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
