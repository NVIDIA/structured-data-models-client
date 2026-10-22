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

#: The databases a case's tables load into, beside DuckDB itself: the two
#: local ones, and the warehouses' engines (T3.3), where Lakebase is a
#: Postgres server in a schema of its own.
TARGETS = ('sqlite', 'postgres', 'databricks', 'snowflake', 'lakebase')

#: The engines loaded by SQL statements alone (:func:`warehouse_statements`).
WAREHOUSE_ENGINES = ('databricks', 'snowflake')

#: Each warehouse leg's engine, whose column of the type map it loads by:
#: both Databricks legs read the same Unity Catalog tables, and Lakebase is
#: Postgres 17 (design, choice 1).
ENGINE = {
    'databricks-sql': 'databricks',
    'databricks-spark': 'databricks',
    'snowflake': 'snowflake',
    'lakebase': 'postgres',
}

#: The type map, by Arrow type family: DuckDB's type, and the type each
#: engine declares; ``<engine>_value`` names the stored form of a type the
#: engine lacks (``canonical``: :func:`canonical_decimal_text`; ``json``:
#: JSON text; ``iso``: ISO 8601 text; ``digits``: the integer's decimal
#: digits), and its absence means the engine's own type holds the value.
#: ``<engine>_refuses`` names a value of the type the engine cannot hold at
#: all (SQLite stores a NaN as NULL), which the loader refuses by name; it
#: refuses a float -0.0 on SQLite too, which SQLite stores as 0.0.
#: ``(p,s)`` is the column's precision and scale; ``(n)`` the column's own
#: composite type, created beside its table and named by
#: :func:`composite_name`; ``(f)`` the fractional digits of the column's
#: unit (0, 3, 6 or 9). A value past what the engine's
#: type holds (a nanosecond on Postgres's microsecond times) is refused.
#: Databricks and Snowflake (T3.3) have no unsigned integers, so an
#: unsigned column is declared in a type holding its whole range, as on
#: Postgres; a type an engine has no column type for at all (a time of day
#: or an interval on Databricks) is stored in a form of its own and listed
#: by :func:`non_native`, never counted as covered there (KO2-47).
TYPE_MAP: dict[str, dict[str, str]] = {
    'boolean': {
        'duckdb': 'BOOLEAN',
        'sqlite': 'BOOLEAN',
        'postgres': 'boolean',
        'databricks': 'BOOLEAN',
        'snowflake': 'BOOLEAN',
    },
    'int8': {
        'duckdb': 'TINYINT',
        'sqlite': 'INTEGER',
        'postgres': 'smallint',
        'databricks': 'TINYINT',
        'snowflake': 'NUMBER(38,0)',
    },
    'int16': {
        'duckdb': 'SMALLINT',
        'sqlite': 'INTEGER',
        'postgres': 'smallint',
        'databricks': 'SMALLINT',
        'snowflake': 'NUMBER(38,0)',
    },
    'int32': {
        'duckdb': 'INTEGER',
        'sqlite': 'INTEGER',
        'postgres': 'integer',
        'databricks': 'INT',
        'snowflake': 'NUMBER(38,0)',
    },
    'int64': {
        'duckdb': 'BIGINT',
        'sqlite': 'INTEGER',
        'postgres': 'bigint',
        'databricks': 'BIGINT',
        'snowflake': 'NUMBER(38,0)',
    },
    'uint8': {
        'duckdb': 'UTINYINT',
        'sqlite': 'INTEGER',
        'postgres': 'smallint',
        'databricks': 'SMALLINT',
        'snowflake': 'NUMBER(38,0)',
    },
    'uint16': {
        'duckdb': 'USMALLINT',
        'sqlite': 'INTEGER',
        'postgres': 'integer',
        'databricks': 'INT',
        'snowflake': 'NUMBER(38,0)',
    },
    'uint32': {
        'duckdb': 'UINTEGER',
        'sqlite': 'INTEGER',
        'postgres': 'bigint',
        'databricks': 'BIGINT',
        'snowflake': 'NUMBER(38,0)',
    },
    'uint64': {
        'duckdb': 'UBIGINT',
        'sqlite': 'TEXT',
        'sqlite_value': 'digits',
        'postgres': 'numeric(20,0)',
        'databricks': 'DECIMAL(20,0)',
        'snowflake': 'NUMBER(38,0)',
    },
    'float32': {
        'duckdb': 'FLOAT',
        'sqlite': 'REAL',
        'sqlite_refuses': 'NaN',
        'postgres': 'real',
        'databricks': 'FLOAT',
        'snowflake': 'FLOAT',
    },
    'float64': {
        'duckdb': 'DOUBLE',
        'sqlite': 'REAL',
        'sqlite_refuses': 'NaN',
        'postgres': 'double precision',
        'databricks': 'DOUBLE',
        'snowflake': 'FLOAT',
    },
    'decimal': {
        'duckdb': 'DECIMAL(p,s)',
        'sqlite': 'TEXT',
        'sqlite_value': 'canonical',
        'postgres': 'numeric(p,s)',
        'databricks': 'DECIMAL(p,s)',
        'snowflake': 'NUMBER(p,s)',
    },
    'string': {
        'duckdb': 'VARCHAR',
        'sqlite': 'TEXT',
        'postgres': 'text',
        'databricks': 'STRING',
        'snowflake': 'VARCHAR',
    },
    'binary': {
        'duckdb': 'BLOB',
        'sqlite': 'BLOB',
        'postgres': 'bytea',
        'databricks': 'BINARY',
        'snowflake': 'BINARY',
    },
    'date': {
        'duckdb': 'DATE',
        'sqlite': 'DATE',
        'sqlite_value': 'iso',
        'postgres': 'date',
        'databricks': 'DATE',
        'snowflake': 'DATE',
    },
    'time': {
        'duckdb': 'TIME',
        'sqlite': 'TIME',
        'sqlite_value': 'iso',
        'postgres': 'time',
        'databricks': 'STRING',
        'databricks_value': 'iso',
        'snowflake': 'TIME(f)',
    },
    'timestamp': {
        'duckdb': 'TIMESTAMP',
        'sqlite': 'TIMESTAMP',
        'sqlite_value': 'iso',
        'postgres': 'timestamp',
        'databricks': 'TIMESTAMP_NTZ',
        'snowflake': 'TIMESTAMP_NTZ(f)',
    },
    'timestamptz': {
        'duckdb': 'TIMESTAMP WITH TIME ZONE',
        'sqlite': 'TIMESTAMPTZ',
        'sqlite_value': 'iso',
        'postgres': 'timestamp with time zone',
        'databricks': 'TIMESTAMP',
        'snowflake': 'TIMESTAMP_LTZ(f)',
    },
    'interval': {
        'duckdb': 'INTERVAL',
        'sqlite': 'INTERVAL',
        'sqlite_value': 'iso',
        'postgres': 'interval',
        'databricks': 'STRING',
        'databricks_value': 'iso',
        'snowflake': 'VARCHAR',
        'snowflake_value': 'iso',
    },
    'list': {
        'duckdb': 'T[] or T[n]',
        'sqlite': 'TEXT',
        'sqlite_value': 'json',
        'postgres': 'T[]',
        'databricks': 'ARRAY<T>',
        'snowflake': 'ARRAY',
        'snowflake_value': 'json',
    },
    'struct': {
        'duckdb': 'STRUCT(...)',
        'sqlite': 'TEXT',
        'sqlite_value': 'json',
        'postgres': '(n)',
        'databricks': 'STRUCT<...>',
        'snowflake': 'OBJECT',
        'snowflake_value': 'json',
    },
    'map': {
        'duckdb': 'MAP(VARCHAR, T)',
        'sqlite': 'TEXT',
        'sqlite_value': 'json',
        'postgres': 'jsonb',
        'postgres_value': 'json',
        'databricks': 'MAP<K, V>',
        'snowflake': 'OBJECT',
        'snowflake_value': 'json',
    },
    'union': {
        'duckdb': 'UNION(...)',
        'sqlite': 'TEXT',
        'sqlite_value': 'json',
        'postgres': 'jsonb',
        'postgres_value': 'json',
        'databricks': 'STRING',
        'databricks_value': 'json',
        'snowflake': 'VARIANT',
        'snowflake_value': 'json',
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


def _sqlite_value(
    value: Any, name: str, where: str, refuse_negative_zero: bool = True
) -> Any:
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
    if (
        refuse_negative_zero
        and name in ('float32', 'float64')
        and value == 0
        and math.copysign(1, value) < 0
    ):
        # SQLite stores a REAL -0.0 as 0.0 (a whole REAL is written as an
        # integer): refused by name, as a NaN is, never loaded changed.
        raise Unloadable(f'{where}: -0.0, which SQLite stores as 0.0')
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
    kind: Any, path: list[str], create: Callable[[str], Any], schema: str
) -> str:
    r"""The Postgres type of an Arrow type at ``path`` (its table, column
    and members' names). A struct's is a composite type
    (:func:`composite_name`) in ``schema``, which ``create`` (a statement's
    runner) creates after the composite types of its members, at any depth,
    so every type a composite names exists before it; a list's element may
    be any type but a list (Postgres arrays are not lists of lists)."""
    import pyarrow as pa

    name = family(kind)
    if name == 'decimal':
        return f'numeric({kind.precision},{kind.scale})'
    if name == 'list':
        element = kind.value_type
        if family(element) == 'list':
            raise Unloadable(f'{".".join(path)}: a list of lists')
        return _postgres_type(element, path, create, schema) + '[]'
    if name == 'struct':
        members = ', '.join(
            f'{quote(member.name)} '
            + _postgres_type(member.type, [*path, member.name], create, schema)
            for member in kind
        )
        composite = f'{quote(schema)}.{quote(composite_name(path))}'
        create(f'CREATE TYPE {composite} AS ({members})')
        return composite
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


def _constraints(
    duck: Any, table: str, prefix: str = ''
) -> tuple[set[str], list[str]]:
    r"""A table's ``NOT NULL`` columns, and its key and foreign-key clauses,
    each referenced table's name after ``prefix`` (its quoted schema and a
    dot, where the target qualifies it)."""
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
                f'FOREIGN KEY ({listed}) REFERENCES {prefix}{quote(target)} ('
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
    # A relation over the quoted name: DuckDB's table() reads a name with
    # a double quote in it as written, not unescaped.
    relation = duck.sql(f'FROM {quote(table)}')
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


def _load_sqlite(
    duck: Any, target: Path, refuse_negative_zero: bool = True
) -> None:
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
                    lambda value, n=name, w=where: _sqlite_value(
                        value, n, w, refuse_negative_zero
                    )
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


def _load_postgres(
    duck: Any, target: Any, schema: str | None = None, replace: bool = False
) -> list[str]:
    r"""The tables into Postgres: ``target`` a libpq DSN, or an open psycopg
    connection (Lakebase's, whose credential the host holds), committed
    here. Every name is qualified by ``schema`` (by default the
    connection's current schema, read from the server), so a case table
    named as a catalog one (``pg_roles``) is the case's, never
    ``pg_catalog``'s. With ``replace``, each table and composite type of
    the case already in the schema is dropped first (Lakebase's one test
    schema holds case after case). Returns the composite types created,
    each qualified."""
    import psycopg

    own = isinstance(target, str)
    connection = psycopg.connect(target, connect_timeout=30) if own else target
    try:
        if schema is None:
            row = connection.execute('SELECT current_schema()').fetchone()
            if row is None or row[0] is None:
                raise Unloadable('the connection has no current schema')
            schema = str(row[0])
        tables = _tables(duck)
        if replace:
            for table in reversed(tables):
                connection.execute(
                    f'DROP TABLE IF EXISTS {quote(schema)}.{quote(table)} '
                    'CASCADE'
                )

        created: list[str] = []

        def create(statement: str) -> None:
            composite = statement.split(' AS (', 1)[0][len('CREATE TYPE ') :]
            if replace:
                connection.execute(f'DROP TYPE IF EXISTS {composite} CASCADE')
            connection.execute(statement)
            created.append(composite)

        for table in tables:
            _lost(duck, table)
            data = _read(duck, table)
            not_null, clauses = _constraints(duck, table, f'{quote(schema)}.')
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
                    field.type, [table, field.name], create, schema
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
            qualified = f'{quote(schema)}.{quote(table)}'
            connection.execute(
                f'CREATE TABLE {qualified} ('
                + ', '.join(columns + clauses)
                + ')'
            )
            with connection.cursor().copy(
                f'COPY {qualified} FROM STDIN'
            ) as copy:
                for row in _rows(data, convert, table):
                    copy.write_row(row)
        connection.commit()
        return created
    finally:
        if own:
            connection.close()


# --- the warehouses (T3.3) ----------------------------------------------------------
#
# Databricks and Snowflake are loaded by SQL statements alone, run through
# the caller's statement runner (a DB-API connection's cursor, Spark's
# ``spark.sql``, or either wrapped in the live write guard, KO3-51), so the
# loader needs no bulk API and every write names its table fully
# qualified, as the guard requires. Each cell is written as a literal the
# engine reads back exactly: text and bytes as hexadecimal, a decimal as
# its canonical text cast to the column's type, a float as its shortest
# round-trip text, a time as ISO text with its offset; Snowflake's
# ``VALUES`` takes constants only, so its rows are text constants each
# column's ``SELECT`` expression converts.

#: What a warehouse leaves out of a table's constraints, beside its types:
#: Databricks keeps primary and foreign keys only, a foreign key only when
#: it references the other table's primary key.
CONSTRAINTS_NOT_STORED = {
    'databricks': 'UNIQUE, and a FOREIGN KEY not referencing a PRIMARY KEY',
}

#: Rows per ``INSERT``, and the most text one may hold, so a statement
#: stays well within each engine's limit.
_BATCH_ROWS = 500
_BATCH_TEXT = 256 * 1024

_FRACTION_DIGITS = {'s': 0, 'ms': 3, 'us': 6, 'ns': 9}

#: The characters Delta reads as column-name syntax unless the table maps
#: its columns by name.
_DELTA_NAME_SYNTAX = set(' ,;{}()\n\t=')


def backtick(name: str) -> str:
    r"""An identifier quoted with backticks, as Databricks reads it."""
    return '`' + name.replace('`', '``') + '`'


def _unit_digits(kind: Any) -> int:
    return _FRACTION_DIGITS[kind.unit]


def column_type(kind: Any, engine: str, where: str) -> str:
    r"""The column type ``engine`` (``databricks`` or ``snowflake``)
    declares for the Arrow type ``kind``: the type map's, with a decimal's
    precision and scale, a time's fractional digits, and Databricks'
    nested types spelled out; :class:`Unloadable` names ``where`` for a
    type it cannot hold in any form."""
    if engine not in WAREHOUSE_ENGINES:
        raise ValueError(f'no warehouse engine {engine!r}')
    try:
        name = family(kind)
    except KeyError:
        raise Unloadable(f'{where}: the type map has no {kind}') from None
    entry = TYPE_MAP[name]
    if name == 'decimal':
        if kind.precision > 38:
            raise Unloadable(
                f'{where}: DECIMAL({kind.precision},{kind.scale}), past the '
                f'38 digits {engine} holds'
            )
        base = 'DECIMAL' if engine == 'databricks' else 'NUMBER'
        return f'{base}({kind.precision},{kind.scale})'
    if engine == 'databricks' and f'{engine}_value' not in entry:
        if name == 'list':
            return f'ARRAY<{column_type(kind.value_type, engine, where)}>'
        if name == 'struct':
            members = ', '.join(
                f'{backtick(member.name)}: '
                + column_type(member.type, engine, where)
                for member in kind
            )
            return f'STRUCT<{members}>'
        if name == 'map':
            return (
                f'MAP<{column_type(kind.key_type, engine, where)}, '
                f'{column_type(kind.item_type, engine, where)}>'
            )
    declared = entry[engine]
    if declared.endswith('(f)'):
        return declared[:-3] + f'({_unit_digits(kind)})'
    return declared


def _stored_type(kind: Any, engine: str) -> Any:
    r"""The Arrow type a column of ``kind`` is sent to ``engine`` as: the
    type itself where the engine holds it; a wider signed integer, or a
    decimal of 20 digits, for an unsigned one Databricks lacks; text for
    a stored form (``<engine>_value``); a nanosecond time as microseconds
    on Databricks; a nested type of stored members."""
    import pyarrow as pa

    name = family(kind)
    if f'{engine}_value' in TYPE_MAP[name]:
        return pa.string()
    if engine == 'databricks':
        widened = {
            'uint8': pa.int16(),
            'uint16': pa.int32(),
            'uint32': pa.int64(),
            'uint64': pa.decimal128(20, 0),
        }
        if name in widened:
            return widened[name]
        if name in ('timestamp', 'timestamptz') and kind.unit == 'ns':
            return pa.timestamp('us', kind.tz)
        if name == 'list':
            return pa.list_(_stored_type(kind.value_type, engine))
        if name == 'struct':
            return pa.struct(
                [
                    pa.field(m.name, _stored_type(m.type, engine), m.nullable)
                    for m in kind
                ]
            )
        if name == 'map':
            return pa.map_(
                _stored_type(kind.key_type, engine),
                _stored_type(kind.item_type, engine),
            )
    return kind


def _stored_cell(value: Any, kind: Any, engine: str, where: str) -> Any:
    r"""One cell of :func:`_cells` as :func:`_stored_type` holds it."""
    if value is None:
        return None
    name = family(kind)
    if name == 'union' and all(item is None for item in value.values()):
        # Arrow's unions carry no nulls of their own: DuckDB's NULL union
        # is its member's null, which a warehouse stores as NULL.
        return None
    stored = TYPE_MAP[name].get(f'{engine}_value')
    if stored == 'iso':
        return _iso(value)
    if stored == 'json':
        return _json(value, where)
    if name == 'uint64' and engine == 'databricks':
        return Decimal(value)
    if name in ('timestamp', 'timestamptz', 'time', 'date'):
        if name != 'date' and engine == 'databricks':
            _whole_micros(value[1], value[2], where)
        return _temporal(value, kind)
    if name == 'list':
        return [
            _stored_cell(item, kind.value_type, engine, where) for item in value
        ]
    if name == 'struct':
        return {
            m.name: _stored_cell(value[m.name], m.type, engine, where)
            for m in kind
        }
    if name == 'map':
        return [
            (
                _stored_cell(k, kind.key_type, engine, where),
                _stored_cell(v, kind.item_type, engine, where),
            )
            for k, v in value
        ]
    return value


def _temporal(value: tuple[Any, ...], kind: Any) -> Any:
    r"""A temporal cell's ticks (a date's days), which pyarrow builds the
    column from."""
    del kind
    return value[1]


def _stored_array(cells: list[Any], kind: Any, stored: Any) -> Any:
    r"""An Arrow array of ``stored`` from the stored cells of a ``kind``
    column, temporal ones from their ticks so nothing passes through a
    library's own type."""
    import pyarrow as pa

    name = family(kind)
    if name in ('timestamp', 'timestamptz') and pa.types.is_timestamp(stored):
        per = (
            _UNIT[kind.unit] // _UNIT[stored.unit]
            if kind.unit != stored.unit
            else 1
        )
        ticks = [None if c is None else c // per for c in cells]
        return pa.array(ticks, pa.int64()).cast(stored)
    if name == 'time' and pa.types.is_time(stored):
        width = pa.int64() if stored.bit_width == 64 else pa.int32()
        return pa.array(cells, width).cast(stored)
    if name == 'date' and pa.types.is_date(stored):
        return pa.array(cells, pa.int32()).cast(stored)
    if name == 'list' and pa.types.is_list(stored):
        flat = [item for cell in cells if cell is not None for item in cell]
        values = _stored_array(flat, kind.value_type, stored.value_type)
        offsets, at = [0], 0
        for cell in cells:
            at += 0 if cell is None else len(cell)
            offsets.append(at)
        mask = pa.array([cell is None for cell in cells])
        return pa.ListArray.from_arrays(
            pa.array(offsets, pa.int32()), values, mask=mask
        ).cast(stored)
    if name == 'struct' and pa.types.is_struct(stored):
        children = [
            _stored_array(
                [None if c is None else c[m.name] for c in cells],
                m.type,
                stored.field(i).type,
            )
            for i, m in enumerate(kind)
        ]
        mask = pa.array([cell is None for cell in cells])
        return pa.StructArray.from_arrays(
            children, fields=list(stored), mask=mask
        )
    if name == 'map' and pa.types.is_map(stored):
        keys = _stored_array(
            [k for cell in cells if cell is not None for k, _ in cell],
            kind.key_type,
            stored.key_type,
        )
        items = _stored_array(
            [v for cell in cells if cell is not None for _, v in cell],
            kind.item_type,
            stored.item_type,
        )
        offsets, at = [0], 0
        for cell in cells:
            at += 0 if cell is None else len(cell)
            offsets.append(at)
        mask = pa.array([cell is None for cell in cells])
        return pa.MapArray.from_arrays(
            pa.array(offsets, pa.int32()), keys, items, mask=mask
        ).cast(stored)
    return pa.array(cells, stored)


def stored(duck: Any, table: str, engine: str) -> Any:
    r"""DuckDB's ``table`` as the loader sends it to ``engine``: each column
    in the type map's stored form (:func:`_stored_type`), its rows in
    DuckDB's order; :class:`Unloadable` names ``table.column`` for a type
    or a value the engine cannot hold exactly (a nanosecond on Databricks,
    which holds microseconds)."""
    import pyarrow as pa

    if engine not in WAREHOUSE_ENGINES:
        raise ValueError(f'no warehouse engine {engine!r}')
    _lost(duck, table)
    data = _read(duck, table)
    columns, fields = [], []
    for i, field in enumerate(data.schema):
        where = f'{table}.{field.name}'
        column_type(field.type, engine, where)
        kind = _stored_type(field.type, engine)
        try:
            cells = [
                _stored_cell(value, field.type, engine, where)
                for value in _cells(data.column(i).combine_chunks())
            ]
            array = _stored_array(cells, field.type, kind)
        except Unloadable:
            raise
        except (OverflowError, ValueError, pa.ArrowInvalid) as error:
            raise Unloadable(
                f'{where}: a value {engine} cannot take '
                f'({type(error).__name__})'
            ) from None
        columns.append(array)
        fields.append(pa.field(field.name, kind, field.nullable))
    return pa.Table.from_arrays(columns, schema=pa.schema(fields))


def _hex(data: bytes) -> str:
    return data.hex().upper()


def _float_text(value: float) -> str:
    if math.isnan(value):
        return 'NaN'
    if math.isinf(value):
        return 'Infinity' if value > 0 else '-Infinity'
    return repr(float(value))


def _databricks_literal(value: Any, kind: Any, where: str) -> str:
    r"""A stored cell (:func:`_cells` of :func:`stored`'s column) as a
    Databricks SQL literal of the column's type."""
    declared = column_type(kind, 'databricks', where)
    if value is None:
        return f'CAST(NULL AS {declared})'
    name = family(kind)
    if name == 'boolean':
        return 'TRUE' if value else 'FALSE'
    if name in ('int8', 'int16', 'int32', 'int64'):
        return f'CAST({int(value)} AS {declared})'
    if name in ('float32', 'float64'):
        return f"CAST('{_float_text(value)}' AS {declared})"
    if name == 'decimal':
        return f"CAST('{canonical_decimal_text(value)}' AS {declared})"
    if name == 'string':
        return f"CAST(X'{_hex(value.encode())}' AS STRING)"
    if name == 'binary':
        return f"X'{_hex(bytes(value))}'"
    if name == 'date':
        return f"DATE'{_iso(value)}'"
    if name in ('timestamp', 'timestamptz'):
        return f"CAST('{_iso(value)}' AS {declared})"
    if name == 'list':
        items = ', '.join(
            _databricks_literal(item, kind.value_type, where) for item in value
        )
        return f'CAST(ARRAY({items}) AS {declared})'
    if name == 'struct':
        # A struct casts to a struct member by member in order, so the
        # members' own names are the declared type's, whatever they hold.
        members = ', '.join(
            f"'f{i}', " + _databricks_literal(value[m.name], m.type, where)
            for i, m in enumerate(kind)
        )
        return f'CAST(NAMED_STRUCT({members}) AS {declared})'
    if name == 'map':
        keys = [repr(key) for key, _ in value]
        if len(set(keys)) != len(keys):
            raise Unloadable(f'{where}: a map with a key twice')
        entries = ', '.join(
            _databricks_literal(key, kind.key_type, where)
            + ', '
            + _databricks_literal(item, kind.item_type, where)
            for key, item in value
        )
        return f'CAST(MAP({entries}) AS {declared})'
    raise Unloadable(f'{where}: {kind}, which the loader cannot write')


def _snowflake_text(value: Any, name: str, where: str) -> str | None:
    r"""A stored cell as the text constant Snowflake's ``SELECT`` converts
    (:func:`_snowflake_reader`); ``name`` the column's family."""
    if value is None:
        return None
    if 'snowflake_value' in TYPE_MAP[name] or name == 'string':
        return _hex(value.encode())
    if name == 'boolean':
        return 'true' if value else 'false'
    if name in ('float32', 'float64'):
        text = _float_text(value)
        return {'Infinity': 'inf', '-Infinity': '-inf'}.get(text, text)
    if name == 'decimal':
        return canonical_decimal_text(value)
    if name == 'binary':
        return _hex(bytes(value))
    if name in ('date', 'time', 'timestamp', 'timestamptz'):
        return _iso(value)
    return str(int(value))


def _snowflake_reader(name: str, declared: str, column: str) -> str:
    r"""The expression reading a column of :func:`_snowflake_text`'s text
    constants as ``declared``."""
    stored = TYPE_MAP[name].get('snowflake_value')
    if stored == 'json':
        return f'PARSE_JSON(HEX_DECODE_STRING({column}))::{declared}'
    if stored is not None or name == 'string':
        return f'HEX_DECODE_STRING({column})'
    if name == 'binary':
        return f"TO_BINARY({column}, 'HEX')"
    return f'CAST({column} AS {declared})'


def _sql_text(text: str | None) -> str:
    r"""A text constant for Snowflake: every text :func:`_snowflake_text`
    writes is digits, letters, signs, points, colons and spaces, so it
    needs no escape; anything else is a fault here."""
    if text is None:
        return 'NULL'
    if not all(c.isalnum() or c in '+-.: ' for c in text):
        raise AssertionError(f'unexpected text constant {text!r}')
    return f"'{text}'"


def _qualified(engine: str, namespace: tuple[str, ...], table: str) -> str:
    quoting = backtick if engine == 'databricks' else quote
    return '.'.join(quoting(part) for part in (*namespace, table))


def _warehouse_constraints(
    engine: str, duck: Any, table: str, namespace: tuple[str, ...]
) -> tuple[set[str], list[str], list[str]]:
    r"""A table's ``NOT NULL`` columns, the key clauses ``engine`` stores
    and a note for each it does not (:data:`CONSTRAINTS_NOT_STORED`)."""
    quoting = backtick if engine == 'databricks' else quote
    not_null: set[str] = set()
    clauses: list[str] = []
    notes: list[str] = []
    primary: dict[str, tuple[str, ...]] = {}
    for name, columns in duck.execute(
        'SELECT table_name, constraint_column_names FROM duckdb_constraints() '
        "WHERE schema_name = 'main' AND constraint_type = 'PRIMARY KEY'"
    ).fetchall():
        primary[name] = tuple(columns)
    rows = duck.execute(
        'SELECT constraint_type, constraint_column_names, referenced_table, '
        'referenced_column_names FROM duckdb_constraints() '
        "WHERE schema_name = 'main' AND table_name = ? "
        'ORDER BY constraint_index',
        [table],
    ).fetchall()
    for kind, columns, target, targets in rows:
        listed = ', '.join(quoting(column) for column in columns)
        if kind == 'NOT NULL':
            not_null.update(columns)
        elif kind == 'PRIMARY KEY':
            not_null.update(columns)
            clauses.append(
                f'CONSTRAINT {quoting(table + "_pk")} PRIMARY KEY ({listed})'
                if engine == 'databricks'
                else f'PRIMARY KEY ({listed})'
            )
        elif kind == 'UNIQUE':
            if engine == 'databricks':
                notes.append(
                    f'{table}: UNIQUE ({", ".join(columns)}) not stored'
                )
            else:
                clauses.append(f'UNIQUE ({listed})')
        elif kind == 'FOREIGN KEY':
            if engine == 'databricks' and primary.get(target) != tuple(targets):
                notes.append(
                    f'{table}: FOREIGN KEY ({", ".join(columns)}) not stored, '
                    'as it references no PRIMARY KEY'
                )
                continue
            reference = (
                f'FOREIGN KEY ({listed}) REFERENCES '
                f'{_qualified(engine, namespace, target)} ('
                + ', '.join(quoting(column) for column in targets)
                + ')'
            )
            if engine == 'databricks':
                index = sum(
                    clause.startswith('CONSTRAINT') for clause in clauses
                )
                reference = (
                    f'CONSTRAINT {quoting(f"{table}_fk{index}")} ' + reference
                )
            clauses.append(reference)
    return not_null, clauses, notes


def _batches(rows: list[str]) -> Iterator[list[str]]:
    batch: list[str] = []
    size = 0
    for row in rows:
        if batch and (
            len(batch) >= _BATCH_ROWS or size + len(row) > _BATCH_TEXT
        ):
            yield batch
            batch, size = [], 0
        batch.append(row)
        size += len(row)
    if batch:
        yield batch


def warehouse_statements(
    duck: Any, engine: str, namespace: tuple[str, ...]
) -> tuple[list[str], list[str]]:
    r"""The statements loading every table of ``duck``'s main schema into
    ``engine``'s schema ``namespace`` (catalog or database, then schema),
    each table dropped first, every value as :func:`stored` sends it, and
    a note for each constraint the engine does not store. A value the
    engine cannot hold is refused by name (:class:`Unloadable`)."""
    if engine not in WAREHOUSE_ENGINES:
        raise ValueError(f'no warehouse engine {engine!r}')
    if len(namespace) != 2 or not all(namespace):
        raise ValueError('a warehouse schema is named by two parts')
    quoting = backtick if engine == 'databricks' else quote
    tables = _tables(duck)
    statements = [
        f'DROP TABLE IF EXISTS {_qualified(engine, namespace, table)}'
        for table in reversed(tables)
    ]
    notes: list[str] = []
    for table in tables:
        source = _read(duck, table)
        data = stored(duck, table, engine)
        names = [field.name for field in source.schema]
        not_null, clauses, left_out = _warehouse_constraints(
            engine, duck, table, namespace
        )
        notes.extend(left_out)
        families = [family(field.type) for field in source.schema]
        declared = [
            column_type(field.type, engine, f'{table}.{field.name}')
            for field in source.schema
        ]
        columns = [
            f'{quoting(name)} {kind}'
            + (' NOT NULL' if name in not_null else '')
            for name, kind in zip(names, declared, strict=True)
        ]
        qualified = _qualified(engine, namespace, table)
        create = (
            f'CREATE TABLE {qualified} (' + ', '.join(columns + clauses) + ')'
        )
        if engine == 'databricks' and any(
            set(name) & _DELTA_NAME_SYNTAX for name in names
        ):
            create += " TBLPROPERTIES ('delta.columnMapping.mode' = 'name')"
        statements.append(create)
        if engine == 'databricks':
            convert = [
                lambda value, k=field.type, w=f'{table}.{field.name}': (
                    _databricks_literal(value, k, w)
                )
                for field in data.schema
            ]
            rows = [
                '(' + ', '.join(row) + ')'
                for row in _rows(data, convert, table)
            ]
            listed = ', '.join(quoting(name) for name in names)
            for batch in _batches(rows):
                statements.append(
                    f'INSERT INTO {qualified} ({listed}) VALUES '
                    + ', '.join(batch)
                )
        else:
            convert = [
                lambda value, n=name, w=f'{table}.{field.name}': _sql_text(
                    _snowflake_text(value, n, w)
                )
                for name, field in zip(families, data.schema, strict=True)
            ]
            rows = [
                '(' + ', '.join(row) + ')'
                for row in _rows(data, convert, table)
            ]
            listed = ', '.join(quoting(name) for name in names)
            readers = ', '.join(
                _snowflake_reader(name, kind, f'column{i + 1}')
                for i, (name, kind) in enumerate(
                    zip(families, declared, strict=True)
                )
            )
            for batch in _batches(rows):
                statements.append(
                    f'INSERT INTO {qualified} ({listed}) SELECT {readers} '
                    'FROM VALUES ' + ', '.join(batch)
                )
    return statements, notes


def statement_runner(connection: Any) -> Callable[[str], Any]:
    r"""A statement runner over a DB-API connection (Databricks SQL's, the
    Snowflake connector's, either wrapped in the live write guard): each
    statement on a cursor of its own."""

    def run(statement: str) -> None:
        cursor = connection.cursor()
        try:
            cursor.execute(statement)
        finally:
            cursor.close()

    return run


def load(
    duck: Any,
    database: str,
    target: Any,
    *,
    namespace: tuple[str, ...] | None = None,
    refuse_negative_zero: bool = True,
) -> list[str]:
    r"""Every table of ``duck``'s main schema, loaded into ``database``,
    each column stored as the type map says; :class:`Unloadable` names a
    column the target cannot hold. Returns a note for each constraint the
    target does not store (:data:`CONSTRAINTS_NOT_STORED`).

    - ``'sqlite'``: a new file at ``target``;
    - ``'postgres'``: the libpq DSN ``target``, the tables created in its
      current schema;
    - ``'lakebase'``: ``target`` a DSN or an open psycopg connection, the
      tables created in ``namespace``'s one schema, replacing the case's
      tables and types already there;
    - ``'databricks'``, ``'snowflake'``: ``target`` a DB-API connection or a
      statement runner (Spark's ``spark.sql``; the write guard's wrapped
      connection), the tables created in ``namespace`` (catalog or
      database, then schema), replacing the case's tables already there
      (:func:`warehouse_statements`);
    - a warehouse leg (:data:`ENGINE`'s keys): ``target`` the recording
      mode's ``WarehouseTarget`` (``run_python.py``, T3.2), loaded by
      :func:`load_leg`.

    A float -0.0, which SQLite stores as 0.0, is refused on SQLite unless
    ``refuse_negative_zero`` is false: a caller whose expected answers
    already read the zeros as SQLite stores them (the provider kit's
    ``zeros``) loads it as 0.0; a contract case, which names the value it
    holds, never does."""
    if database in ENGINE and hasattr(target, 'connection'):
        return load_leg(duck, database, target)
    if database == 'sqlite':
        _load_sqlite(duck, Path(target), refuse_negative_zero)
        return []
    if database == 'postgres':
        _load_postgres(duck, str(target))
        return []
    if database == 'lakebase':
        if namespace is None or len(namespace) != 1 or not namespace[0]:
            raise ValueError('a Lakebase schema is named by one part')
        _load_postgres(duck, target, namespace[0], replace=True)
        return []
    if database in ('databricks', 'snowflake'):
        if namespace is None:
            raise ValueError(f'a {database} load needs its schema')
        statements, notes = warehouse_statements(duck, database, namespace)
        run = target if callable(target) else statement_runner(target)
        for statement in statements:
            run(statement)
        return notes
    raise ValueError(f'no loader for {database!r}; it is one of {TARGETS}')


class CaseSchema:
    r"""One warehouse leg's test schema across a run's cases (C-1, E-0):
    each :meth:`load` first drops every table (and Lakebase composite type)
    the run loaded before, children first, so the schema holds exactly the
    case's tables when the old client reads its catalog; :meth:`clear`
    drops what is left. Only what this object loaded is ever dropped: the
    schema's other objects (a view a case reads) are never touched.

    ``target`` is what :func:`load` takes for ``database``: a statement
    runner or DB-API connection for Databricks and Snowflake, an open
    psycopg connection for Lakebase."""

    def __init__(
        self, database: str, target: Any, namespace: tuple[str, ...]
    ) -> None:
        if database not in ('databricks', 'snowflake', 'lakebase'):
            raise ValueError(f'no warehouse schema on {database!r}')
        self.database = database
        self.target = target
        self.namespace = tuple(namespace)
        self.tables: list[str] = []
        self.types: list[str] = []
        self.views: list[tuple[str, str]] = []

    def _run(self, statement: str) -> None:
        if self.database == 'lakebase':
            self.target.execute(statement)
        elif callable(self.target):
            self.target(statement)
        else:
            statement_runner(self.target)(statement)

    def clear(self) -> None:
        r"""Drops every view, table and type loaded so far, the last
        first."""
        for kind, name in reversed(self.views):
            qualified = _qualified(
                'postgres' if self.database == 'lakebase' else self.database,
                self.namespace,
                name,
            )
            what = 'SEMANTIC VIEW' if kind == 'semantic_view' else 'VIEW'
            self._run(f'DROP {what} IF EXISTS {qualified}')
        self.views = []
        for table in reversed(self.tables):
            if self.database == 'lakebase':
                self._run(
                    f'DROP TABLE IF EXISTS {quote(self.namespace[0])}.'
                    f'{quote(table)} CASCADE'
                )
            else:
                self._run(
                    'DROP TABLE IF EXISTS '
                    + _qualified(self.database, self.namespace, table)
                )
        for composite in reversed(self.types):
            self._run(f'DROP TYPE IF EXISTS {composite} CASCADE')
        if self.database == 'lakebase':
            self.target.commit()
        self.tables, self.types = [], []

    def create_view(self, kind: str, name: str, yaml: str) -> None:
        r"""The case's view, created in the schema from its YAML: a
        Databricks metric view (``CREATE VIEW ... WITH METRICS LANGUAGE
        YAML``), or a Snowflake semantic view
        (``SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML``, the view named by the
        YAML); dropped by the next :meth:`load` or :meth:`clear`."""
        if '$$' in yaml:
            raise Unloadable(f'{name}: a view YAML holding $$')
        if kind == 'metric_view' and self.database == 'databricks':
            qualified = _qualified('databricks', self.namespace, name)
            self._run(
                f'CREATE OR REPLACE VIEW {qualified} WITH METRICS LANGUAGE '
                f'YAML AS $$\n{yaml}\n$$'
            )
        elif kind == 'semantic_view' and self.database == 'snowflake':
            where = '.'.join(quote(part) for part in self.namespace)
            self._run(
                'CALL SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML('
                f"'{where.replace(chr(39), chr(39) * 2)}', $$\n{yaml}\n$$)"
            )
        else:
            raise Unloadable(
                f'{name}: a {kind} is not created on {self.database}'
            )
        self.views.append((kind, name))

    def load(self, duck: Any) -> list[str]:
        r"""The case's tables in the schema, alone; :func:`load`'s notes."""
        self.clear()
        self.tables = _tables(duck)
        if self.database == 'lakebase':
            self.types = _load_postgres(
                duck, self.target, self.namespace[0], replace=True
            )
            return []
        return load(duck, self.database, self.target, namespace=self.namespace)


#: Each leg's test schema across the run's cases, by leg and schema: the
#: recording mode opens a connection per case, and the schema must hold
#: one case's tables at a time (:class:`CaseSchema`).
_SCHEMAS: dict[tuple[str, ...], CaseSchema] = {}


def _spark_runner(session: Any) -> Callable[[str], Any]:
    r"""A statement runner over a Spark session."""

    def run(statement: str) -> None:
        session.sql(statement).collect()

    return run


def load_leg(duck: Any, leg: str, target: Any) -> list[str]:
    r"""The case's tables in ``leg``'s test schema, alone, through
    ``target`` (``run_python.py``'s ``WarehouseTarget``: ``connection``,
    the leg's open connection or Spark session; ``names``, the private
    ``catalog`` and ``schema``; ``view``, the case's resolved
    ``input.view``): the tables an earlier case of the run loaded there
    dropped first, then each table, and, for a view given by its YAML, the
    view created from it. Returns the loader's notes."""
    names = dict(target.names)
    if leg == 'lakebase':
        namespace: tuple[str, ...] = (names.get('schema', ''),)
        database, writer = 'lakebase', target.connection
    else:
        namespace = (names.get('catalog', ''), names.get('schema', ''))
        database = ENGINE[leg]
        writer = (
            _spark_runner(target.connection)
            if leg == 'databricks-spark'
            else target.connection
        )
    if not all(namespace):
        raise Unloadable(f'{leg}: the test schema is not named')
    key = (leg, *namespace)
    schema = _SCHEMAS.get(key)
    if schema is None:
        schema = _SCHEMAS[key] = CaseSchema(database, writer, namespace)
    schema.target = writer
    notes = schema.load(duck)
    view = target.view
    if view and view.get('yaml'):
        schema.create_view(view['kind'], view['name'], view['yaml'])
    return notes


def non_native(database: str) -> list[str]:
    r"""The type map's entries ``database`` stores in a form of its own
    (``<database>_value``), the cells check 1's evidence lists."""
    return sorted(
        name for name, entry in TYPE_MAP.items() if f'{database}_value' in entry
    )
