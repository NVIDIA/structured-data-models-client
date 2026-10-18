# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""L5, the conversion tables (T2.4; design, locked check 4; KO2-47), run in
CI's ``l5-conversion (<cell>)`` on pandas 2 and 3, each cell's pyarrow and
NumPy, beside the pinned Postgres service.

- The drivers' table. A table holding a column of each type of the type
  map (``contract/type-map-vectors.json``) is built on DuckDB and loaded
  into SQLite and Postgres by the contract loader
  (``contract/runners/databases.py``), each column stored as the map says.
  Each column is read through each database's provider, planned by the
  core's data layer and admitted by the core (the test-only fetch): its
  Arrow type is :data:`CONVERSION`'s, recorded from the three providers
  and reviewed, or the core refuses it by name; so a unit, scale or type
  change in one driver's path fails. On DuckDB and SQLite each is the type
  the reference's own reads give it (``contract/l5-vectors.json``'s
  ``drivers``, recorded from the reference by
  ``contract/tools/l5_record.py`` in every cell), but the two the core
  refuses by name.
- Without loss: each admitted column holds what the database stores, as
  that database's own driver reads it (DuckDB's, ``sqlite3``, psycopg),
  value by value, nulls and order included.
- Identical Arrow on the types all three store: the columns whose Arrow
  type is the same on all three are exactly :data:`ALL_THREE`, the type
  map's types each engine stores as its own, and their admitted arrays are
  equal across the three, metadata included.
- The evidence (KO2-47): every type an engine stores in another form is
  the type map's, and is printed per engine with the form it is stored in;
  never counted as covered by its replacement.
- The presentation table: each layout of the result the core hands the
  host (``sdm-request``'s ``predict.rs``), read as the host reads it
  (``_rust/predict.py``'s ``_rows``, ``_id`` and ``_as_answered``) and
  presented by ``PredictionResult.to_pandas()``: each cell is the answer's
  JSON value, and each column's dtype is :data:`PRESENTATION`'s for this
  pandas major, recorded from the reference (pandas 3) and python-old
  (pandas 2) presenting the same answers (``contract/l5-vectors.json``,
  T1.31's rule).

Where ``SDM_POSTGRES_TEST_DSN`` names no server and ``SDM_REQUIRE_POSTGRES``
is not set, the Postgres leg has nothing to hold and the drivers' tests
return without asserting; with it set, a missing server fails (H26).
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sqlite3
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from sdm_client._rust import codec
from sdm_client._rust.run import run

ROOT = Path(__file__).resolve().parents[3]
TYPE_MAP = json.loads(
    (ROOT / 'contract/type-map-vectors.json').read_text(encoding='utf-8')
)['map']
DATABASES = ('duckdb', 'sqlite', 'postgres')

#: Each column of the table and the type map's type it holds.
COLUMNS = {
    'b': 'boolean',
    'i8': 'int8',
    'i16': 'int16',
    'i32': 'int32',
    'i64': 'int64',
    'u8': 'uint8',
    'u16': 'uint16',
    'u32': 'uint32',
    'u64': 'uint64',
    'f32': 'float32',
    'f64': 'float64',
    'dec': 'decimal',
    's': 'string',
    'bin': 'binary',
    'd': 'date',
    'tm': 'time',
    'ts': 'timestamp',
    'tstz': 'timestamptz',
    'iv': 'interval',
    'li': 'list',
    'st': 'struct',
    'mp': 'map',
    'un': 'union',
}

#: The table, built on DuckDB: each column's extremes, an empty or zero
#: value where the type has one, and a null; no NaN, which SQLite refuses
#: (the type map's ``sqlite_refuses``).
SETUP = [
    'CREATE TABLE t (id BIGINT PRIMARY KEY, b BOOLEAN, i8 TINYINT, '
    'i16 SMALLINT, i32 INTEGER, i64 BIGINT, u8 UTINYINT, u16 USMALLINT, '
    'u32 UINTEGER, u64 UBIGINT, f32 FLOAT, f64 DOUBLE, dec DECIMAL(38,10), '
    's VARCHAR, bin BLOB, d DATE, tm TIME, ts TIMESTAMP, tstz TIMESTAMPTZ, '
    'iv INTERVAL, li INTEGER[], st STRUCT(a INTEGER, b VARCHAR), '
    'mp MAP(VARCHAR, INTEGER), un UNION(n INTEGER, s VARCHAR))',
    'INSERT INTO t VALUES (1, true, -128, -32768, -2147483648, '
    '-9223372036854775808, 0, 0, 0, 0, 1.5, 0.1, '
    "12345678901234567890.0123456789, 'a', '\\x00\\x01'::BLOB, "
    "'1970-01-01', '00:00:00', '2024-01-02 03:04:05.123456', "
    "'2024-01-02 03:04:05.123456+00', INTERVAL 1 DAY, [1, 2], "
    "{'a': 1, 'b': 'x'}, MAP {'k': 1}, 2::UNION(n INTEGER, s VARCHAR))",
    'INSERT INTO t VALUES (2, false, 127, 32767, 2147483647, '
    '9223372036854775807, 255, 65535, 4294967295, 18446744073709551615, '
    "-0.25, -1e300, -0.5, 'é ü', ''::BLOB, '2024-02-29', "
    "'23:59:59.123456', '1969-12-31 23:59:59.999999', "
    "'1999-12-31 23:00:00+00', INTERVAL '1 month 2 days 3 seconds', [], "
    "{'a': NULL, 'b': NULL}, MAP {}, 'z'::UNION(n INTEGER, s VARCHAR))",
    'INSERT INTO t (id) VALUES (3)',
]

REFUSED = 'refused UNSUPPORTED_TYPE'

#: The session zone every connection here reads in, DuckDB's and
#: Postgres's, as CI's runners have it (a TIMESTAMPTZ's Arrow zone is the
#: session's on DuckDB).
ZONE = 'Etc/UTC'
PINNED = f'-c TimeZone={ZONE}'

#: The conversion table: each column's admitted Arrow type on each
#: database, as :func:`_arrow_type` spells it, or the core's refusal by
#: code. Recorded from the three providers through the core, the same in
#: every cell, and reviewed; a change here is a change of what reaches the
#: core.
CONVERSION = {
    'b': {'duckdb': 'bool', 'sqlite': 'int64', 'postgres': 'bool'},
    'i8': {'duckdb': 'int8', 'sqlite': 'int64', 'postgres': 'int16'},
    'i16': {'duckdb': 'int16', 'sqlite': 'int64', 'postgres': 'int16'},
    'i32': {'duckdb': 'int32', 'sqlite': 'int64', 'postgres': 'int32'},
    'i64': {'duckdb': 'int64', 'sqlite': 'int64', 'postgres': 'int64'},
    'u8': {'duckdb': 'uint8', 'sqlite': 'int64', 'postgres': 'int16'},
    'u16': {'duckdb': 'uint16', 'sqlite': 'int64', 'postgres': 'int32'},
    'u32': {'duckdb': 'uint32', 'sqlite': 'int64', 'postgres': 'int64'},
    'u64': {
        'duckdb': 'uint64',
        'sqlite': 'string',
        'postgres': 'decimal128(20, 0)',
    },
    'f32': {'duckdb': 'float', 'sqlite': 'double', 'postgres': 'double'},
    'f64': {'duckdb': 'double', 'sqlite': 'double', 'postgres': 'double'},
    'dec': {
        'duckdb': 'decimal128(38, 10)',
        'sqlite': 'string',
        'postgres': 'decimal128(38, 10)',
    },
    's': {'duckdb': 'string', 'sqlite': 'string', 'postgres': 'string'},
    'bin': {'duckdb': 'binary', 'sqlite': 'binary', 'postgres': 'binary'},
    'd': {
        'duckdb': 'date32[day]',
        'sqlite': 'string',
        'postgres': 'date32[day]',
    },
    'tm': {
        'duckdb': 'time64[us]',
        'sqlite': 'string',
        'postgres': 'time64[us]',
    },
    'ts': {
        'duckdb': 'timestamp[us]',
        'sqlite': 'string',
        'postgres': 'timestamp[us]',
    },
    'tstz': {
        'duckdb': 'timestamp[us, tz=Etc/UTC]',
        'sqlite': 'string',
        'postgres': 'timestamp[us, tz=UTC]',
    },
    'iv': {'duckdb': REFUSED, 'sqlite': 'string', 'postgres': REFUSED},
    'li': {
        'duckdb': 'list<l: int32>',
        'sqlite': 'string',
        'postgres': 'list<item: int32>',
    },
    'st': {
        'duckdb': 'struct<a: int32, b: string>',
        'sqlite': 'string',
        'postgres': 'string',
    },
    'mp': {
        'duckdb': 'map<string, int32>',
        'sqlite': 'string',
        'postgres': 'string arrow.json',
    },
    'un': {
        'duckdb': REFUSED,
        'sqlite': 'string',
        'postgres': 'string arrow.json',
    },
}

#: The type map's types each of the three stores as its own type, whose
#: Arrow is identical on all three.
ALL_THREE = {'i64', 'f64', 's', 'bin'}


def _arrow_type(field: pa.Field) -> str:
    r"""``field``'s type as pyarrow spells it, an extension type as its
    storage type and name: pyarrow from 19 reads Arrow's canonical
    ``arrow.json`` as an extension type, and the floor's 14 keeps it as the
    field's metadata."""
    kind = field.type
    name = None
    if isinstance(kind, pa.BaseExtensionType):
        name = kind.extension_name
        kind = kind.storage_type
    elif field.metadata and b'ARROW:extension:name' in field.metadata:
        name = field.metadata[b'ARROW:extension:name'].decode()
    return str(kind) + (f' {name}' if name else '')


def _loader() -> Any:
    spec = importlib.util.spec_from_file_location(
        't24_databases', ROOT / 'contract/runners/databases.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _admitted(provider: Any, dialect: str, column: str) -> pa.Table | str:
    r"""``column`` (with ``id``) read through ``provider`` and admitted by
    the core, or the core's refusal by code."""
    from sdm_client.errors import StructuredDataError

    request = {
        'request': 'scan',
        'table': {'source': ['t']},
        'columns': ['id', column],
        'filters': [],
        'limit': None,
    }
    fetch = {
        'request': request,
        'provider': {'provider': 'sql', 'dialect': dialect},
        'echo': True,
    }
    try:
        finished = run(
            {'op': 'test', 'test': 'fetch', 'fetch': 1},
            {1: codec.attachment(fetch)},
            provider=provider,
        )
    except StructuredDataError as error:
        return f'refused {error.code}'
    stream = finished.take(finished.result['data'])
    table = pa.RecordBatchReader._import_from_c_capsule(
        stream.__arrow_c_stream__()
    ).read_all()
    return table.sort_by('id')


class Source:
    r"""The table on each database, each database's provider and its own
    driver's read of a column."""

    def __init__(self, tmp: Path, dsn: str | None) -> None:
        self.duck = duckdb.connect()
        # The zone names a TIMESTAMPTZ's Arrow zone; pinned for every cursor
        # (GLOBAL), as CI's runners have it, so the table holds anywhere.
        self.duck.execute(f"SET GLOBAL TimeZone = '{ZONE}'")
        for statement in SETUP:
            self.duck.execute(statement)
        loader = _loader()
        self.path = tmp / 'l5.sqlite'
        loader.load(self.duck, 'sqlite', self.path)
        self.dsn = dsn
        self.database: str | None = None
        self.scoped: str | None = None
        if dsn is not None:
            import psycopg
            from psycopg.conninfo import make_conninfo

            self.database = f'sdm_l5_{uuid.uuid4().hex[:16]}'
            with psycopg.connect(dsn, autocommit=True) as admin:
                admin.execute(f'CREATE DATABASE {self.database}')
            self.scoped = make_conninfo(dsn, dbname=self.database)
            loader.load(self.duck, 'postgres', self.scoped)

    def providers(self) -> dict[str, Any]:
        import adbc_driver_sqlite.dbapi
        import psycopg

        from sdm_client._rust.providers.duckdb import DuckDBProvider
        from sdm_client._rust.providers.postgres import PostgresProvider
        from sdm_client._rust.providers.sqlite import SqliteProvider

        assert self.scoped is not None
        return {
            'duckdb': DuckDBProvider(self.duck),
            'sqlite': SqliteProvider(
                adbc_driver_sqlite.dbapi.connect(str(self.path))
            ),
            'postgres': PostgresProvider(
                psycopg.connect(self.scoped, autocommit=True, options=PINNED)
            ),
        }

    def own_read(self, database: str, column: str) -> list[Any]:
        r"""``column`` in ``id`` order as ``database``'s own driver reads
        it: DuckDB's Python rows, ``sqlite3``'s, psycopg's."""
        sql = f'SELECT "{column}" FROM t ORDER BY id'
        if database == 'duckdb':
            return [row[0] for row in self.duck.execute(sql).fetchall()]
        if database == 'sqlite':
            with sqlite3.connect(self.path) as connection:
                return [row[0] for row in connection.execute(sql).fetchall()]
        import psycopg

        assert self.scoped is not None
        with psycopg.connect(self.scoped, options=PINNED) as connection:
            return [row[0] for row in connection.execute(sql).fetchall()]

    def drop(self) -> None:
        self.duck.close()
        if self.dsn is not None and self.database is not None:
            import psycopg

            with psycopg.connect(self.dsn, autocommit=True) as admin:
                admin.execute(
                    f'DROP DATABASE IF EXISTS {self.database} WITH (FORCE)'
                )


@pytest.fixture(scope='module')
def source(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Source | None]:
    dsn = os.environ.get('SDM_POSTGRES_TEST_DSN')
    if not dsn:
        assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
            'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
        )
        yield None
        return
    made = Source(tmp_path_factory.mktemp('l5'), dsn)
    try:
        yield made
    finally:
        made.drop()


Read = dict[str, dict[str, pa.Table | str]]


def _read(providers: dict[str, Any]) -> Read:
    r"""Every column through every database's provider and the core."""
    return {
        column: {
            database: _admitted(providers[database], database, column)
            for database in DATABASES
        }
        for column in COLUMNS
    }


def _cells(table: pa.Table, column: str) -> list[Any]:
    r"""``column``'s cells as Python values, each as the database's own
    driver gives that type: a map as a dict, an ``arrow.json`` text as the
    value it holds."""
    field = table.schema.field(column)
    values = table.column(column).to_pylist()
    if pa.types.is_map(field.type):
        return [None if v is None else dict(v) for v in values]
    if _arrow_type(field).endswith(' arrow.json'):
        return [None if v is None else json.loads(v) for v in values]
    return values


def _problems(read: Read, own: Callable[[str, str], list[Any]]) -> list[str]:
    r"""What fails L5 in ``read``: a type other than :data:`CONVERSION`'s,
    a value other than the database's own read, the identical set other
    than :data:`ALL_THREE` or an array of it differing between databases."""
    problems = []
    for column, by_database in read.items():
        for database, admitted in by_database.items():
            where = f'{column} ({COLUMNS[column]}) on {database}'
            wanted = CONVERSION[column][database]
            got = (
                admitted
                if isinstance(admitted, str)
                else _arrow_type(admitted.schema.field(column))
            )
            if got != wanted:
                problems.append(f'{where}: {got}, not {wanted}')
                continue
            if isinstance(admitted, str):
                continue
            if admitted.column('id').to_pylist() != [1, 2, 3]:
                problems.append(f'{where}: rows {admitted.column("id")}')
                continue
            cells, stored = _cells(admitted, column), own(database, column)
            if cells != stored:
                problems.append(f'{where}: {cells!r}, stored {stored!r}')
    same = {
        column
        for column, by_database in CONVERSION.items()
        if REFUSED not in by_database.values()
        and len(set(by_database.values())) == 1
    }
    if same != ALL_THREE:
        problems.append(f'identical on all three: {sorted(same)}')
    for column in sorted(same & set(read)):
        tables = read[column]
        first = tables['duckdb']
        for database in DATABASES[1:]:
            other = tables[database]
            if isinstance(first, str) or isinstance(other, str):
                continue
            if not (
                first.schema.field(column).equals(
                    other.schema.field(column), check_metadata=True
                )
                and first.column(column).equals(other.column(column))
            ):
                problems.append(f'{column}: {database} differs from duckdb')
    return problems


def test_each_type_reaches_the_core_as_the_conversion_table_says_without_loss(
    source: Source | None,
) -> None:
    if source is None:
        return
    problems = _problems(_read(source.providers()), source.own_read)
    assert not problems, '\n'.join(problems)


def test_each_type_an_engine_stores_in_another_form_is_the_type_map_s() -> None:
    # KO2-47's evidence, derived from the type map, never written here: a
    # type SQLite stores in another form (its `sqlite_value`) reaches the
    # core as that form's text, a Postgres JSON value as arrow.json text;
    # every other type of the table is each engine's own.
    evidence: dict[str, list[str]] = {'sqlite': [], 'postgres': []}
    for column, kind in COLUMNS.items():
        mapped = TYPE_MAP[kind]
        for database in evidence:
            form = mapped.get(f'{database}_value')
            if form is not None:
                evidence[database].append(f'{kind} as {form}')
                # SQLite's text, and Postgres's JSON as Arrow's arrow.json.
                wanted = (
                    'string arrow.json' if database == 'postgres' else 'string'
                )
                assert CONVERSION[column][database] == wanted, (
                    column,
                    database,
                )
    for database, forms in evidence.items():
        print(f'{database} stores in another form: {", ".join(forms)}')
    assert {COLUMNS[c] for c in ALL_THREE} <= {
        kind
        for kind, mapped in TYPE_MAP.items()
        if not any(key.endswith('_value') for key in mapped)
    }


class _Seeded:
    r"""A provider whose answer has ``column`` cast to ``to``: a unit or
    scale change in one driver's path."""

    def __init__(self, inner: Any, column: str, to: pa.DataType) -> None:
        self.inner, self.column, self.to = inner, column, to

    def __call__(self, step: dict[str, Any], keys: Any = None) -> Any:
        answer = self.inner(step, keys)
        table = pa.RecordBatchReader._import_from_c_capsule(
            answer.__arrow_c_stream__()
        ).read_all()
        if self.column in table.column_names:
            at = table.column_names.index(self.column)
            table = table.set_column(
                at,
                self.column,
                table.column(at).cast(self.to, safe=False),
            )
        return table


@pytest.mark.parametrize(
    ('column', 'to'),
    [
        ('ts', pa.timestamp('ms')),
        ('dec', pa.decimal128(38, 4)),
        ('tstz', pa.timestamp('ns', tz='UTC')),
    ],
)
def test_a_unit_or_scale_change_in_one_driver_s_path_fails(
    source: Source | None, column: str, to: pa.DataType
) -> None:
    if source is None:
        return
    providers = source.providers()
    providers['postgres'] = _Seeded(providers['postgres'], column, to)
    read = {column: {d: _admitted(providers[d], d, column) for d in DATABASES}}
    problems = [
        p for p in _problems(read, source.own_read) if p.startswith(column)
    ]
    assert problems, (column, to)


# --- the presentation table -----------------------------------------------------


def _result(columns: dict[str, pa.Array]) -> pa.Table:
    n = len(next(iter(columns.values())))
    return pa.table({**columns, 'answer_index': pa.array(range(n), pa.int64())})


def _probabilities(values: list[dict[str, float] | None]) -> pa.Array:
    kind = pa.map_(pa.string(), pa.float64())
    return pa.array(
        [None if v is None else list(v.items()) for v in values], type=kind
    )


#: Each layout of the result the core hands the host (``predict.rs``):
#: the core's Arrow, and each row as the endpoint answered it, the JSON
#: values the reference's items hold.
LAYOUTS: dict[str, tuple[pa.Table, list[dict[str, Any]]]] = {
    'regression, float answers': (
        _result(
            {
                'row_index': pa.array([0, 1, 2], pa.int64()),
                'id': pa.array([7, 8, None], pa.int64()),
                'prediction': pa.array([1.5, None, -2.25], pa.float64()),
            }
        ),
        [
            {'row_index': 0, 'id': 7, 'prediction': 1.5},
            {'row_index': 1, 'id': 8, 'prediction': None},
            {'row_index': 2, 'id': None, 'prediction': -2.25},
        ],
    ),
    'regression, integer answers': (
        _result(
            {
                'row_index': pa.array([0, 1], pa.int64()),
                'id': pa.array([1, 2], pa.int64()),
                'prediction': pa.array([3, -9007199254740993], pa.int64()),
            }
        ),
        [
            {'row_index': 0, 'id': 1, 'prediction': 3},
            {'row_index': 1, 'id': 2, 'prediction': -9007199254740993},
        ],
    ),
    'regression, an integer widened among floats': (
        _result(
            {
                'row_index': pa.array([0, 1, 2], pa.int64()),
                'id': pa.array([1, 2, 3], pa.int64()),
                'prediction': pa.array([1.0, 2.5, None], pa.float64()),
                'prediction_as_answered': pa.array(['1', '2.5', None]),
            }
        ),
        [
            {'row_index': 0, 'id': 1, 'prediction': 1},
            {'row_index': 1, 'id': 2, 'prediction': 2.5},
            {'row_index': 2, 'id': 3, 'prediction': None},
        ],
    ),
    'binary classification, text ids': (
        _result(
            {
                'row_index': pa.array([0, 1, 2], pa.int64()),
                'id': pa.array(['a', None, 'é'], pa.string()),
                'prediction': pa.array([True, False, None], pa.bool_()),
                'probabilities': _probabilities(
                    [{'True': 0.75, 'False': 0.25}, {'True': 0.0}, None]
                ),
            }
        ),
        [
            {
                'row_index': 0,
                'id': 'a',
                'prediction': True,
                'probabilities': {'True': 0.75, 'False': 0.25},
            },
            {
                'row_index': 1,
                'id': None,
                'prediction': False,
                'probabilities': {'True': 0.0},
            },
            {'row_index': 2, 'id': 'é', 'prediction': None},
        ],
    ),
    'multiclass, ids of mixed types': (
        _result(
            {
                'row_index': pa.array([0, 1, 2], pa.int64()),
                'id': pa.array(['7', 'b', None], pa.string()),
                'id_as_written': pa.array(['7', '"b"', None]),
                'prediction': pa.array(['red', None, 'blue'], pa.string()),
            }
        ),
        [
            {'row_index': 0, 'id': 7, 'prediction': 'red'},
            {'row_index': 1, 'id': 'b', 'prediction': None},
            {'row_index': 2, 'id': None, 'prediction': 'blue'},
        ],
    ),
    'forecast, a step per row': (
        _result(
            {
                'row_index': pa.array([0, 0, 1], pa.int64()),
                'id': pa.array([5, 5, 6], pa.int64()),
                'forecast_step': pa.array([1, 2, None], pa.int64()),
                'prediction': pa.array([0.5, 0.25, 1e300], pa.float64()),
            }
        ),
        [
            {'row_index': 0, 'id': 5, 'forecast_step': 1, 'prediction': 0.5},
            {'row_index': 0, 'id': 5, 'forecast_step': 2, 'prediction': 0.25},
            {'row_index': 1, 'id': 6, 'prediction': 1e300},
        ],
    ),
}

#: Each layout's presented columns and their dtypes, by pandas major:
#: recorded by ``contract/tools/l5_record.py`` from the reference in
#: ``ref-pandas3`` (pandas 3) and python-old in ``dbr17-pandas2`` (pandas 2),
#: never written here (T1.31).
L5 = json.loads((ROOT / 'contract/l5-vectors.json').read_text(encoding='utf-8'))
PRESENTATION: dict[str, dict[str, dict[str, str]]] = {
    layout: {major: L5['presentation'][major][layout] for major in ('2', '3')}
    for layout in LAYOUTS
}


def _recorder() -> Any:
    spec = importlib.util.spec_from_file_location(
        't24_l5_record', ROOT / 'contract/tools/l5_record.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_recorded_tables_are_the_reference_s_for_these_inputs() -> None:
    # The recorder's inputs are this file's: the table it reads and the
    # answers it presents; the presentation was recorded by the reference in
    # ref-pandas3 and python-old in dbr17-pandas2, for every layout.
    recorder = _recorder()
    assert recorder.SETUP == SETUP
    assert {name: answered for name, (_, answered) in LAYOUTS.items()} == (
        recorder.ANSWERED
    )
    assert L5['recorded_by'] == 'contract/tools/l5_record.py'
    assert L5['by']['ref-pandas3']['runner'] == 'python-ref'
    assert L5['by']['dbr17-pandas2']['runner'] == 'python-old'
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    assert f'reference: {L5["reference"]}' in env
    for major in ('2', '3'):
        assert sorted(L5['presentation'][major]) == sorted(LAYOUTS)


def test_duckdb_and_sqlite_reach_the_core_as_the_reference_s_drivers_read() -> (
    None
):
    # The reference reads DuckDB through to_arrow_table() and SQLite through
    # ADBC's fetch_arrow_table(), so on those two each admitted type is the
    # one the reference's own read gives, recorded in each of its cells;
    # where the core refuses a type by name (an interval and a union on
    # DuckDB, as the type set records) the reference reads it. The reference
    # reads Postgres as Python rows, with no Arrow type to hold the core's
    # to; its values are held to psycopg's above.
    refused = set()
    for cell, read in L5['drivers'].items():
        assert sorted(read) == sorted(COLUMNS), cell
        for column, by_database in read.items():
            for database, kind in by_database.items():
                wanted = CONVERSION[column][database]
                if wanted == REFUSED:
                    refused.add((column, database))
                    continue
                assert wanted == kind, (cell, column, database, wanted, kind)
    assert refused == {('iv', 'duckdb'), ('un', 'duckdb')}


def _presented(table: pa.Table) -> pd.DataFrame:
    r"""``table`` read as the host reads the core's result, its items
    built as ``_rust/predict.py``'s ``predict`` builds them, presented."""
    from sdm_api.generated.tfm_api import PredictionItem

    from sdm_client._rust.predict import _as_answered, _id, _rows
    from sdm_client.result import PredictionResult

    rows = sorted(_rows(table), key=lambda row: row['answer_index'])
    items = [
        PredictionItem(
            row_index=row['row_index'],
            id=_id(row),
            forecast_step=row.get('forecast_step'),
            prediction=_as_answered(row),
            probabilities=None
            if row.get('probabilities') is None
            else dict(row['probabilities']),
        )
        for row in rows
    ]
    return PredictionResult(model='m', request_id='r', items=items).to_pandas()


def _same(cell: Any, answered: Any) -> bool:
    r"""A presented cell is the answer's JSON value: a missing one null or
    NaN, a boolean a boolean, a number or text equal."""
    if answered is None:
        return cell is None or (isinstance(cell, float) and math.isnan(cell))
    if isinstance(answered, bool):
        return isinstance(cell, (bool, np.bool_)) and bool(cell) is answered
    return not isinstance(cell, (bool, np.bool_)) and bool(cell == answered)


@pytest.mark.parametrize('layout', list(LAYOUTS))
def test_each_result_layout_is_presented_as_the_table_says(layout: str) -> None:
    table, answered = LAYOUTS[layout]
    frame = _presented(table)
    major = pd.__version__.split('.')[0]
    dtypes = {name: str(dtype) for name, dtype in frame.dtypes.items()}
    assert dtypes == PRESENTATION[layout][major], dtypes
    for position, item in enumerate(answered):
        for name, value in item.items():
            if name == 'probabilities':
                for key, p in value.items():
                    cell = frame.at[position, f'probability_{key}']
                    assert cell == p, (layout, position, key)
                continue
            cell = frame.at[position, name]
            assert _same(cell, value), (layout, position, name, cell, value)
