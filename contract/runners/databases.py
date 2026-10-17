# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The contract loader: a case's DuckDB-built tables, loaded into SQLite
or Postgres through Arrow (Phase 2 design, choice 2; T2.1a).

    load(duck, 'sqlite', path)       # a new SQLite file
    load(duck, 'postgres', dsn)      # the DSN's current schema

Every table of the DuckDB connection's ``main`` schema is read as Arrow and
created in the target with each column stored as the type map says
(:data:`TYPE_MAP`, the same entries as ``contract/type-map-vectors.json``'s
``map``): an engine's own type wherever it has one, and otherwise a stored
form the map names (``*_value``): canonical decimal text, JSON text for a
nested value, ISO text for a time. A table's primary key, unique keys,
``NOT NULL`` columns and foreign keys are carried as DuckDB declares them,
and its rows are inserted in the order DuckDB scans them. A value the
target cannot hold exactly (a nanosecond on Postgres, a map with keys that
are not text) is refused by name (:class:`Unloadable`), never rounded: an
input an engine can't load is an owner's exclusion, not a silent change.

The reference and both core hosts read the same loaded database.
:func:`canonical_decimal_text` is the one definition of a decimal's text,
which ``sdm-testkit``'s key binders implement too; both run the vector.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator
from contextlib import closing
from decimal import Decimal
from pathlib import Path
from typing import Any

#: The databases a case's tables load into, beside DuckDB itself.
TARGETS = ('sqlite', 'postgres')

#: The type map, by Arrow type family: DuckDB's type, and the type each
#: engine declares; ``<engine>_value`` names the stored form of a type the
#: engine lacks (``canonical``: :func:`canonical_decimal_text`; ``json``:
#: JSON text; ``iso``: ISO 8601 text; ``digits``: the integer's decimal
#: digits), and its absence means the engine's own type holds the value.
#: ``<engine>_refuses`` names a value of the type the engine cannot hold at
#: all (SQLite stores a NaN as NULL), which the loader refuses by name.
#: ``(p,s)`` is the column's precision and scale; ``(n)`` the column's own
#: composite type, created beside its table and named by
#: :func:`composite_name`. A value past what the engine's
#: type holds (a nanosecond on Postgres's microsecond times) is refused.
TYPE_MAP: dict[str, dict[str, str]] = {
    'boolean': {
        'duckdb': 'BOOLEAN',
        'sqlite': 'BOOLEAN',
        'postgres': 'boolean',
    },
    'int8': {'duckdb': 'TINYINT', 'sqlite': 'INTEGER', 'postgres': 'smallint'},
    'int16': {
        'duckdb': 'SMALLINT',
        'sqlite': 'INTEGER',
        'postgres': 'smallint',
    },
    'int32': {'duckdb': 'INTEGER', 'sqlite': 'INTEGER', 'postgres': 'integer'},
    'int64': {'duckdb': 'BIGINT', 'sqlite': 'INTEGER', 'postgres': 'bigint'},
    'uint8': {
        'duckdb': 'UTINYINT',
        'sqlite': 'INTEGER',
        'postgres': 'smallint',
    },
    'uint16': {
        'duckdb': 'USMALLINT',
        'sqlite': 'INTEGER',
        'postgres': 'integer',
    },
    'uint32': {'duckdb': 'UINTEGER', 'sqlite': 'INTEGER', 'postgres': 'bigint'},
    'uint64': {
        'duckdb': 'UBIGINT',
        'sqlite': 'TEXT',
        'sqlite_value': 'digits',
        'postgres': 'numeric(20,0)',
    },
    'float32': {
        'duckdb': 'FLOAT',
        'sqlite': 'REAL',
        'sqlite_refuses': 'NaN',
        'postgres': 'real',
    },
    'float64': {
        'duckdb': 'DOUBLE',
        'sqlite': 'REAL',
        'sqlite_refuses': 'NaN',
        'postgres': 'double precision',
    },
    'decimal': {
        'duckdb': 'DECIMAL(p,s)',
        'sqlite': 'TEXT',
        'sqlite_value': 'canonical',
        'postgres': 'numeric(p,s)',
    },
    'string': {'duckdb': 'VARCHAR', 'sqlite': 'TEXT', 'postgres': 'text'},
    'binary': {'duckdb': 'BLOB', 'sqlite': 'BLOB', 'postgres': 'bytea'},
    'date': {
        'duckdb': 'DATE',
        'sqlite': 'DATE',
        'sqlite_value': 'iso',
        'postgres': 'date',
    },
    'time': {
        'duckdb': 'TIME',
        'sqlite': 'TIME',
        'sqlite_value': 'iso',
        'postgres': 'time',
    },
    'timestamp': {
        'duckdb': 'TIMESTAMP',
        'sqlite': 'TIMESTAMP',
        'sqlite_value': 'iso',
        'postgres': 'timestamp',
    },
    'timestamptz': {
        'duckdb': 'TIMESTAMP WITH TIME ZONE',
        'sqlite': 'TIMESTAMPTZ',
        'sqlite_value': 'iso',
        'postgres': 'timestamp with time zone',
    },
    'interval': {
        'duckdb': 'INTERVAL',
        'sqlite': 'INTERVAL',
        'sqlite_value': 'iso',
        'postgres': 'interval',
    },
    'list': {
        'duckdb': 'T[] or T[n]',
        'sqlite': 'TEXT',
        'sqlite_value': 'json',
        'postgres': 'T[]',
    },
    'struct': {
        'duckdb': 'STRUCT(...)',
        'sqlite': 'TEXT',
        'sqlite_value': 'json',
        'postgres': '(n)',
    },
    'map': {
        'duckdb': 'MAP(VARCHAR, T)',
        'sqlite': 'TEXT',
        'sqlite_value': 'json',
        'postgres': 'jsonb',
        'postgres_value': 'json',
    },
    'union': {
        'duckdb': 'UNION(...)',
        'sqlite': 'TEXT',
        'sqlite_value': 'json',
        'postgres': 'jsonb',
        'postgres_value': 'json',
    },
}


class Unloadable(Exception):
    r"""A table the target cannot hold as the type map says, named by
    table and column; never a value."""


def canonical_decimal_text(value: Decimal) -> str:
    r"""A decimal's one text: plain positional notation, no exponent, no
    trailing fractional zeros, no trailing point, ``-0`` written ``0``; so
    equal values of any scale have one text."""
    if not value.is_finite():
        raise ValueError('a decimal that is not finite has no canonical text')
    if value.is_zero():
        return '0'
    text = format(value, 'f')
    if '.' in text:
        text = text.rstrip('0').rstrip('.')
    return text


# --- the type map, per Arrow type ------------------------------------------------


def family(kind: Any) -> str:
    r"""The type map's entry for an Arrow type."""
    import pyarrow as pa

    t = pa.types
    checks: list[tuple[Callable[[Any], bool], str]] = [
        (t.is_boolean, 'boolean'),
        (t.is_int8, 'int8'),
        (t.is_int16, 'int16'),
        (t.is_int32, 'int32'),
        (t.is_int64, 'int64'),
        (t.is_uint8, 'uint8'),
        (t.is_uint16, 'uint16'),
        (t.is_uint32, 'uint32'),
        (t.is_uint64, 'uint64'),
        (t.is_float32, 'float32'),
        (t.is_float64, 'float64'),
        (t.is_decimal, 'decimal'),
        (lambda k: t.is_string(k) or t.is_large_string(k), 'string'),
        (lambda k: t.is_binary(k) or t.is_large_binary(k), 'binary'),
        (t.is_date32, 'date'),
        (t.is_time, 'time'),
        (lambda k: t.is_timestamp(k) and k.tz is None, 'timestamp'),
        (lambda k: t.is_timestamp(k) and k.tz is not None, 'timestamptz'),
        (t.is_interval, 'interval'),
        (t.is_map, 'map'),
        (
            lambda k: (
                t.is_list(k) or t.is_large_list(k) or t.is_fixed_size_list(k)
            ),
            'list',
        ),
        (t.is_struct, 'struct'),
        (t.is_union, 'union'),
    ]
    for check, name in checks:
        if check(kind):
            return name
    raise KeyError(str(kind))


def quote(name: str) -> str:
    r"""An identifier quoted alone, as SQLite and Postgres both read it."""
    return '"' + name.replace('"', '""') + '"'


# --- values ------------------------------------------------------------------------

_UNIT = {'s': 1, 'ms': 10**3, 'us': 10**6, 'ns': 10**9}


def _instant_text(ticks: int, unit: str, zoned: bool) -> str:
    r"""A timestamp's ISO text, as many fractional digits as its unit
    holds and none when they are all zero; a zoned one in UTC with its
    offset."""
    import datetime as dt

    per = _UNIT[unit]
    seconds, fraction = divmod(ticks, per)
    moment = dt.datetime(1970, 1, 1) + dt.timedelta(seconds=seconds)
    text = moment.strftime('%Y-%m-%d %H:%M:%S')
    if moment.year < 1000:
        text = f'{moment.year:04d}' + text[text.index('-') :]
    if fraction:
        digits = len(str(per)) - 1
        text += '.' + f'{fraction:0{digits}d}'.rstrip('0')
    return text + ('+00:00' if zoned else '')


def _time_text(ticks: int, unit: str) -> str:
    per = _UNIT[unit]
    seconds, fraction = divmod(ticks, per)
    hours, rest = divmod(seconds, 3600)
    text = f'{hours:02d}:{rest // 60:02d}:{rest % 60:02d}'
    if fraction:
        digits = len(str(per)) - 1
        text += '.' + f'{fraction:0{digits}d}'.rstrip('0')
    return text


def _interval_text(months: int, days: int, nanos: int) -> str:
    r"""An interval's ISO 8601 duration, its three parts kept apart."""
    seconds = Decimal(nanos).scaleb(-9)
    return f'P{months}M{days}DT{canonical_decimal_text(seconds)}S'


def _cells(array: Any) -> list[Any]:
    r"""An Arrow array's values as plain Python data: integers, floats,
    decimals, text, bytes, lists and dicts; a temporal value as its
    ``(kind, ...)`` parts, read from its integer storage so no precision
    passes through a library's own type; ``None`` for a null."""
    import pyarrow as pa

    kind = array.type
    name = family(kind)
    valid = array.is_valid().to_pylist() if len(array) else []
    if name in ('timestamp', 'timestamptz'):
        ticks = array.cast(pa.int64()).to_pylist()
        return [
            ('instant', tick, kind.unit, name == 'timestamptz') if ok else None
            for tick, ok in zip(ticks, valid, strict=True)
        ]
    if name == 'time':
        storage = pa.int64() if kind.bit_width == 64 else pa.int32()
        ticks = array.cast(storage).to_pylist()
        return [
            ('time', tick, kind.unit) if ok else None
            for tick, ok in zip(ticks, valid, strict=True)
        ]
    if name == 'date':
        days = array.cast(pa.int32()).to_pylist()
        return [
            ('date', day) if ok else None
            for day, ok in zip(days, valid, strict=True)
        ]
    if name == 'interval':
        return [
            None if value is None else ('interval', *value)
            for value in array.to_pylist()
        ]
    if name == 'list':
        # The child storage whole, each row's slice of it by the offsets,
        # so a null row's slot (which a fixed-size list keeps, and a
        # variable one may) never shifts its neighbours' values.
        flat = _cells(array.values)
        if pa.types.is_fixed_size_list(kind):
            size = kind.list_size
            offsets = [(array.offset + i) * size for i in range(len(array) + 1)]
        else:
            offsets = array.offsets.to_pylist()
        return [
            flat[offsets[i] : offsets[i + 1]] if ok else None
            for i, ok in enumerate(valid)
        ]
    if name == 'map':
        entries = array.values.flatten()
        keys, items = _cells(entries[0]), _cells(entries[1])
        offsets = array.offsets.to_pylist()
        return [
            _Map(
                zip(
                    keys[offsets[i] : offsets[i + 1]],
                    items[offsets[i] : offsets[i + 1]],
                    strict=True,
                )
            )
            if ok
            else None
            for i, ok in enumerate(valid)
        ]
    if name == 'struct':
        fields = [
            (kind.field(i).name, _cells(child))
            for i, child in enumerate(array.flatten())
        ]
        return [
            {field: values[row] for field, values in fields} if ok else None
            for row, ok in enumerate(valid)
        ]
    if name == 'union':
        codes = array.type_codes.to_pylist()
        children = {
            code: (kind.field(i).name, _cells(array.field(i)))
            for i, code in enumerate(kind.type_codes)
        }
        dense = kind.mode == 'dense'
        offsets = array.offsets.to_pylist() if dense else None
        found = []
        for row, code in enumerate(codes):
            member, values = children[code]
            at = offsets[row] if offsets is not None else row
            found.append({member: values[at]})
        return found
    return array.to_pylist()


class _Map(list[tuple[Any, Any]]):
    r"""A map cell: its entries in order, a map even when it has none."""


def _json_value(value: Any, where: str) -> Any:
    r"""A cell as JSON data: a decimal and a big integer as text; a float
    JSON has no form for (NaN, an infinity) refused by name."""
    if isinstance(value, float) and not math.isfinite(value):
        raise Unloadable(
            f'{where}: a float JSON has no form for, inside a nested value'
        )
    if value is None or isinstance(value, (bool, str, float)):
        return value
    if isinstance(value, int):
        return value if -(2**53) <= value <= 2**53 else str(value)
    if isinstance(value, Decimal):
        return canonical_decimal_text(value)
    if isinstance(value, _Map):
        keys = [key for key, _ in value]
        if not all(isinstance(key, str) for key in keys):
            raise Unloadable(f'{where}: a map whose keys are not text')
        return {key: _json_value(item, where) for key, item in value}
    if isinstance(value, list):
        return [_json_value(item, where) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item, where) for key, item in value.items()}
    if isinstance(value, tuple):
        return _iso(value)
    if isinstance(value, bytes):
        raise Unloadable(f'{where}: bytes inside a nested value')
    raise Unloadable(f'{where}: a value of {type(value).__name__}')


def _iso(value: tuple[Any, ...]) -> str:
    import datetime as dt

    kind = value[0]
    if kind == 'instant':
        return _instant_text(value[1], value[2], value[3])
    if kind == 'time':
        return _time_text(value[1], value[2])
    if kind == 'date':
        return (dt.date(1970, 1, 1) + dt.timedelta(days=value[1])).isoformat()
    return _interval_text(value[1], value[2], value[3])


def _json(value: Any, where: str) -> str:
    return json.dumps(
        _json_value(value, where),
        ensure_ascii=False,
        separators=(',', ':'),
        allow_nan=False,
    )


def _sqlite_value(value: Any, name: str, where: str) -> Any:
    r"""A cell as SQLite stores it, by the type map's entry ``name``."""
    if value is None:
        return None
    if name == 'decimal':
        return canonical_decimal_text(value)
    if name == 'uint64':
        return str(value)
    if name == 'boolean':
        return int(value)
    if name in ('float32', 'float64') and math.isnan(value):
        # SQLite stores a NaN as NULL: a NaN no SQLite database can hold is
        # refused by name, never loaded as a null (the type map's
        # sqlite_refuses).
        raise Unloadable(f'{where}: a NaN, which SQLite stores as NULL')
    if name in ('date', 'time', 'timestamp', 'timestamptz', 'interval'):
        return _iso(value)
    if name in ('list', 'struct', 'map', 'union'):
        return _json(value, where)
    return value


# --- Postgres ----------------------------------------------------------------------


def composite_name(path: list[str]) -> str:
    r"""The name of the composite type of a struct at ``path`` (its table,
    column and members' names): a digest of the path written as JSON, so
    no two paths share one, whatever their names hold (a ``.`` included),
    it fits Postgres's 63 bytes, and every run gives the same."""
    import hashlib

    encoded = json.dumps(path, ensure_ascii=True, separators=(',', ':'))
    return 'sdm_' + hashlib.sha256(encoded.encode()).hexdigest()[:32]


def _postgres_type(
    kind: Any, path: list[str], create: Callable[[str], Any]
) -> str:
    r"""The Postgres type of an Arrow type at ``path`` (its table, column
    and members' names). A struct's is a composite type
    (:func:`composite_name`), which ``create`` (a statement's runner)
    creates after the composite types of its members, at any depth, so
    every type a composite names exists before it; a list's element may be
    any type but a list (Postgres arrays are not lists of lists)."""
    import pyarrow as pa

    name = family(kind)
    if name == 'decimal':
        return f'numeric({kind.precision},{kind.scale})'
    if name == 'list':
        element = kind.value_type
        if family(element) == 'list':
            raise Unloadable(f'{".".join(path)}: a list of lists')
        return _postgres_type(element, path, create) + '[]'
    if name == 'struct':
        members = ', '.join(
            f'{quote(member.name)} '
            + _postgres_type(member.type, [*path, member.name], create)
            for member in kind
        )
        composite = composite_name(path)
        create(f'CREATE TYPE {quote(composite)} AS ({members})')
        return quote(composite)
    if name == 'time' and pa.types.is_time32(kind):
        return 'time'
    return TYPE_MAP[name]['postgres']


def _whole_micros(ticks: int, unit: str, where: str) -> None:
    if unit == 'ns' and ticks % 1000:
        raise Unloadable(f'{where}: a time past microseconds')


def _postgres_text(value: Any, name: str, where: str) -> str | None:
    r"""A cell as Postgres's text input reads it."""
    if value is None:
        return None
    if name == 'decimal':
        return canonical_decimal_text(value)
    if name == 'boolean':
        return 't' if value else 'f'
    if name in ('float32', 'float64'):
        return repr(float(value))
    if name in ('timestamp', 'timestamptz', 'time'):
        _whole_micros(value[1], value[2], where)
        return _iso(value)
    if name == 'date':
        return _iso(value)
    if name == 'interval':
        months, days, nanos = value[1:]
        if nanos % 1000:
            raise Unloadable(f'{where}: an interval past microseconds')
        return f'{months} mons {days} days {nanos // 1000} microseconds'
    if name in ('map', 'union'):
        # jsonb holds no NUL, which JSON's text would carry escaped: the
        # value's own text, its keys included, is what is checked.
        if _holds_nul(value):
            raise Unloadable(
                f'{where}: text holding a NUL, which Postgres refuses'
            )
        return _json(value, where)
    if name == 'binary':
        return '\\x' + bytes(value).hex()
    if name in ('list', 'struct'):
        raise AssertionError('nested values are written by _nested')
    return str(value)


def _quoted_element(text: str | None, null: str) -> str:
    if text is None:
        return null
    return '"' + text.replace('\\', '\\\\').replace('"', '\\"') + '"'


def _nested(value: Any, kind: Any, where: str) -> str | None:
    r"""A list or struct cell as Postgres's array or composite text input."""
    if value is None:
        return None
    name = family(kind)
    if name == 'list':
        element = kind.value_type
        return (
            '{'
            + ','.join(
                _quoted_element(_element(item, element, where), 'NULL')
                for item in value
            )
            + '}'
        )
    return (
        '('
        + ','.join(
            _quoted_element(
                _element(value[kind.field(i).name], kind.field(i).type, where),
                '',
            )
            for i in range(kind.num_fields)
        )
        + ')'
    )


def _element(value: Any, kind: Any, where: str) -> str | None:
    name = family(kind)
    if name in ('list', 'struct'):
        return _nested(value, kind, where)
    return _postgres_text(value, name, where)


# --- tables --------------------------------------------------------------------------


def _tables(duck: Any) -> list[str]:
    r"""The main schema's tables, each after the tables its foreign keys
    reference, else in name order."""
    names = [
        row[0]
        for row in duck.execute(
            'SELECT table_name FROM duckdb_tables() '
            "WHERE schema_name = 'main' AND NOT temporary ORDER BY table_name"
        ).fetchall()
    ]
    needs = {name: set() for name in names}
    for table, target in duck.execute(
        'SELECT table_name, referenced_table FROM duckdb_constraints() '
        "WHERE schema_name = 'main' AND constraint_type = 'FOREIGN KEY'"
    ).fetchall():
        if table in needs and target in needs and target != table:
            needs[table].add(target)
    ordered: list[str] = []
    while needs:
        ready = sorted(name for name, wanted in needs.items() if not wanted)
        if not ready:
            raise Unloadable(
                'foreign keys that reference each other in a cycle'
            )
        for name in ready:
            ordered.append(name)
            del needs[name]
        for wanted in needs.values():
            wanted.difference_update(ready)
    return ordered


def _constraints(duck: Any, table: str) -> tuple[set[str], list[str]]:
    r"""A table's ``NOT NULL`` columns, and its key and foreign-key clauses."""
    not_null: set[str] = set()
    clauses: list[str] = []
    for kind, columns, target, targets in duck.execute(
        'SELECT constraint_type, constraint_column_names, referenced_table, '
        'referenced_column_names FROM duckdb_constraints() '
        "WHERE schema_name = 'main' AND table_name = ? "
        'ORDER BY constraint_index',
        [table],
    ).fetchall():
        listed = ', '.join(quote(column) for column in columns)
        if kind == 'NOT NULL':
            not_null.update(columns)
        elif kind == 'PRIMARY KEY':
            clauses.append(f'PRIMARY KEY ({listed})')
        elif kind == 'UNIQUE':
            clauses.append(f'UNIQUE ({listed})')
        elif kind == 'FOREIGN KEY':
            clauses.append(
                f'FOREIGN KEY ({listed}) REFERENCES {quote(target)} ('
                + ', '.join(quote(column) for column in targets)
                + ')'
            )
    return not_null, clauses


def _read(duck: Any, table: str) -> Any:
    reader = duck.execute(f'SELECT * FROM {quote(table)}')
    fetch = getattr(reader, 'to_arrow_table', None) or reader.fetch_arrow_table
    return fetch()


#: DuckDB types whose Arrow form loses what they hold (a time's offset, a
#: bit string read as DuckDB's own bytes), refused by name rather than
#: loaded as something else.
LOST_IN_ARROW = ('time with time zone', 'bit')


def _holds_lost(kind: Any) -> bool:
    r"""Whether a DuckDB type is, or holds at any depth, one of
    :data:`LOST_IN_ARROW`, read from DuckDB's own type tree: a struct's
    field names are names, never types."""
    if kind.id in LOST_IN_ARROW:
        return True
    if kind.id in ('struct', 'list', 'array', 'map', 'union'):
        return any(
            _holds_lost(child)
            for _, child in kind.children
            if hasattr(child, 'id')
        )
    return False


def _lost(duck: Any, table: str) -> None:
    relation = duck.table(quote(table))
    for column, kind in zip(relation.columns, relation.types, strict=True):
        if _holds_lost(kind):
            raise Unloadable(
                f'{table}.{column}: {kind}, which Arrow does not carry whole'
            )


def _rows(
    data: Any, convert: list[Callable[[Any], Any]], table: str
) -> Iterator[tuple]:
    r"""The rows of ``data``, each column converted; a value no conversion
    takes (a time past what Python's dates hold, text Postgres cannot hold)
    is refused by name."""
    columns = []
    for i, cell in enumerate(convert):
        where = f'{table}.{data.schema.field(i).name}'
        try:
            values = [cell(v) for v in _cells(data.column(i).combine_chunks())]
        except Unloadable:
            raise
        except (OverflowError, ValueError) as error:
            raise Unloadable(
                f'{where}: a value the target cannot take '
                f'({type(error).__name__})'
            ) from None
        columns.append(values)
    return zip(*columns, strict=True) if columns else iter(())


def _holds_nul(value: Any) -> bool:
    r"""Whether a cell's text, at any depth, a map's keys and a struct's
    names included, holds a NUL."""
    if isinstance(value, str):
        return '\x00' in value
    if isinstance(value, dict):
        return any(_holds_nul(k) or _holds_nul(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return any(_holds_nul(item) for item in value)
    return False


def _no_nul(text: Any, where: str) -> Any:
    r"""Postgres holds no NUL in text, nor in jsonb."""
    if isinstance(text, str) and '\x00' in text:
        raise Unloadable(f'{where}: text holding a NUL, which Postgres refuses')
    return text


def _load_sqlite(duck: Any, target: Path) -> None:
    import sqlite3

    target = Path(target)
    if target.exists():
        raise FileExistsError(f'{target.name}: the SQLite target exists')
    with closing(sqlite3.connect(target)) as connection:
        for table in _tables(duck):
            _lost(duck, table)
            data = _read(duck, table)
            not_null, clauses = _constraints(duck, table)
            columns, convert = [], []
            for field in data.schema:
                where = f'{table}.{field.name}'
                try:
                    name = family(field.type)
                except KeyError:
                    raise Unloadable(
                        f'{where}: the type map has no {field.type}'
                    ) from None
                declared = TYPE_MAP[name]['sqlite']
                columns.append(
                    f'{quote(field.name)} {declared}'
                    + (' NOT NULL' if field.name in not_null else '')
                )
                convert.append(
                    lambda value, n=name, w=where: _sqlite_value(value, n, w)
                )
            connection.execute(
                f'CREATE TABLE {quote(table)} ('
                + ', '.join(columns + clauses)
                + ')'
            )
            marks = ', '.join('?' for _ in columns)
            connection.executemany(
                f'INSERT INTO {quote(table)} VALUES ({marks})',
                _rows(data, convert, table),
            )
        connection.commit()


def _load_postgres(duck: Any, dsn: str) -> None:
    import psycopg

    with psycopg.connect(dsn, connect_timeout=30) as connection:
        for table in _tables(duck):
            _lost(duck, table)
            data = _read(duck, table)
            not_null, clauses = _constraints(duck, table)
            columns, convert = [], []
            for field in data.schema:
                where = f'{table}.{field.name}'
                try:
                    name = family(field.type)
                except KeyError:
                    raise Unloadable(
                        f'{where}: the type map has no {field.type}'
                    ) from None
                declared = _postgres_type(
                    field.type, [table, field.name], connection.execute
                )
                columns.append(
                    f'{quote(field.name)} {declared}'
                    + (' NOT NULL' if field.name in not_null else '')
                )
                if name in ('list', 'struct'):
                    convert.append(
                        lambda value, k=field.type, w=where: _no_nul(
                            _nested(value, k, w), w
                        )
                    )
                else:
                    convert.append(
                        lambda value, n=name, w=where: _no_nul(
                            _postgres_text(value, n, w), w
                        )
                    )
            connection.execute(
                f'CREATE TABLE {quote(table)} ('
                + ', '.join(columns + clauses)
                + ')'
            )
            with connection.cursor().copy(
                f'COPY {quote(table)} FROM STDIN'
            ) as copy:
                for row in _rows(data, convert, table):
                    copy.write_row(row)


def load(duck: Any, database: str, target: str | Path) -> None:
    r"""Every table of ``duck``'s main schema, loaded into ``database``
    (``'sqlite'``: a new file at ``target``; ``'postgres'``: the libpq DSN
    ``target``, the tables created in its current schema), each column
    stored as the type map says; :class:`Unloadable` names a column the
    target cannot hold."""
    if database == 'sqlite':
        _load_sqlite(duck, Path(target))
    elif database == 'postgres':
        _load_postgres(duck, str(target))
    else:
        raise ValueError(f'no loader for {database!r}; it is one of {TARGETS}')


def non_native(database: str) -> list[str]:
    r"""The type map's entries ``database`` stores in a form of its own
    (``<database>_value``), the cells check 1's evidence lists."""
    return sorted(
        name for name, entry in TYPE_MAP.items() if f'{database}_value' in entry
    )
