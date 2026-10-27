# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Postgres rows as Arrow, each column typed by its Postgres type as
psycopg's registry names it: the one mapping the Postgres provider
(:mod:`postgres`) hands the core, and the one the contract's recording mode
keeps a Lakebase or Postgres read as (``contract/runners/run_python.py``,
which loads this file by its path, beside a reference client that has no
provider). It imports nothing of this package."""

from __future__ import annotations

import json
from datetime import datetime, timezone
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


def rows_arrow(
    description: Any,
    rows: list[Any],
    connection: Any,
    batch_rows: int = BATCH_ROWS,
) -> Any:
    r"""``rows``, read on a psycopg cursor whose ``description`` they
    answer, as a pyarrow table in batches of at most ``batch_rows``, each
    column typed by its Postgres type (:func:`field`) on ``connection``."""
    import pyarrow as pa

    fields = [field(column, connection) for column in description]
    schema = pa.schema(fields)
    batches = []
    for start in range(0, len(rows), batch_rows):
        columns = list(zip(*rows[start : start + batch_rows], strict=True))
        batches.append(
            pa.record_batch(
                [
                    _array(values, kind)
                    for values, kind in zip(columns, fields, strict=True)
                ],
                schema=schema,
            )
        )
    return pa.Table.from_batches(batches, schema=schema)


def field(column: Any, connection: Any) -> Any:
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


def _is_json(field: Any) -> bool:
    return (field.metadata or {}).get(b'ARROW:extension:name') == b'arrow.json'


def _json_text(value: Any) -> str | None:
    r"""The JSON text of a value the driver read, ASCII only, so a lone
    surrogate a document escapes (which Python's reader keeps) travels as
    its escape; what the reference makes of it is the core's."""
    if value is None:
        return None
    return json.dumps(value, separators=(',', ':'))


def _array(values: Any, field: Any) -> Any:
    import pyarrow as pa

    kind = field.type
    if _is_json(field):
        return pa.array([_json_text(value) for value in values], type=kind)
    if pa.types.is_list(kind) and _is_json(kind.value_field):
        # A JSON array's elements, each as its JSON text: an element that
        # is itself a JSON array stays one value (never flattened).
        return pa.array(
            [
                None if value is None else [_json_text(item) for item in value]
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
        return str(_utc_text_value(value))
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


def _utc_text_value(value: Any) -> Any:
    """Native encoding for the core-selected zoned_text_utc result policy.

    This is used only for a text Arrow cell, including an array element.
    Scalar timestamp columns keep their existing Arrow conversion.
    Range and multirange formatting remains the driver's own formatting.
    Standalone capture reads retain the same native UTC encoding.
    """
    if isinstance(value, datetime):
        offset = value.utcoffset()
        if offset is None:
            return value
        try:
            return str(value.astimezone(timezone.utc))
        except OverflowError:
            # A readable wall date can represent UTC year 0 or 10000.
            # Shift by a Gregorian cycle, retaining its actual fixed offset,
            # so conversion preserves the instant without losing support.
            shift = 400 if value.year < 400 else -400
            shifted = value.replace(
                year=value.year + shift,
                tzinfo=timezone(offset),
            ).astimezone(timezone.utc)
            return f'{shifted.year - shift:04d}' + str(shifted)[4:]

    # A Postgres native result has psycopg available; keep the dependency
    # lazy so importing this standalone Arrow adapter needs no driver.
    try:
        from psycopg.types.multirange import Multirange
        from psycopg.types.range import Range
    except ModuleNotFoundError as error:
        if error.name == 'psycopg':
            return value
        raise

    if isinstance(value, Range) and not value.isempty:
        return Range(
            _utc_text_value(value.lower),
            _utc_text_value(value.upper),
            value.bounds,
        )
    if isinstance(value, Multirange):
        return Multirange([_utc_text_value(span) for span in value])
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
        if info.name in _JSON:
            # A JSON array's elements, each the JSON text of the value the
            # driver read, marked as JSON as a JSON column's are.
            return pa.list_(pa.field('item', pa.string(), metadata=JSON_TAG))
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
