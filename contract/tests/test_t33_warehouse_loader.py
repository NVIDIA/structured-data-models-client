# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T3.3's acceptance tests for the loader into the four warehouses (Phase 3
design, choice 1; T3.3's brief, "The loader"), written by the package's
test author before the implementation, offline: no warehouse is reached.

What each engine stores is read from the old client at 43e55f4f:
Databricks' columns and types from ``information_schema.columns``
(``RE/backend/databricks/table.py:212-246``), Snowflake's ``_to_dtype``
(``RE/backend/snow/table.py:331-383``: ``NUMBER(p,s)``, ``TIMESTAMP*``,
``TIME``, the unsupported ``VARIANT``, ``OBJECT`` and ``MAP``), and
Lakebase as a Postgres connection (``CN/databricks.py:178-262``,
``RE/backend/postgres/table.py:146-184``). ``RE/`` is
``packages/structured-data-models-relational-engine/src/sdm_relational_engine/rfm/``
and ``CN/`` is ``packages/structured-data-models-connectors/src/sdm_connectors/``.

The interface tested (``contract/runners/databases.py``), named here
because nothing names it before the implementation:

- ``TYPE_MAP[<family>]`` gains a ``databricks`` and a ``snowflake`` entry,
  with ``<engine>_value`` naming the stored form of a type the engine has no
  column type for (KO2-47: listed by :func:`non_native`, never counted as
  covered), as SQLite's and Postgres's entries do; the vector file's
  ``map`` is the same (``test_contract_loader.py`` holds them equal);
- ``ENGINE``: each warehouse leg's engine in the type map
  (``lakebase`` is a Postgres server, so it is ``postgres``);
- ``column_type(kind, engine, where) -> str``: the column type a table's
  ``CREATE TABLE`` declares for the Arrow type ``kind`` on ``engine``
  (``databricks`` or ``snowflake``), :class:`Unloadable` naming ``where``
  for a type it cannot hold in any form;
- ``stored(duck, table, engine) -> pyarrow.Table``: DuckDB's ``table`` as
  the loader sends it to ``engine``, each column in the type map's stored
  form, its rows in DuckDB's order; :class:`Unloadable` names
  ``table.column`` for a value the engine cannot hold exactly.

Each test names the brief line it proves. Each test listed in
``contract/tests/t33-pending.json``'s ``pending`` (as ``py:<function
name>``) failed when written and is held to failing (``xfail``, strict)
until the implementer takes it out; ``pending`` stays within
``authored``, the test author's list.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import uuid
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNERS = ROOT / 'contract/runners'
PENDING_FILE = Path(__file__).with_name('t33-pending.json')
#: The two engines the type map gains; Lakebase reads as Postgres.
WAREHOUSE_ENGINES = ('databricks', 'snowflake')


def _module(name: str) -> ModuleType:
    key = f't33_loader_{name}'
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, RUNNERS / f'{name}.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


databases = _module('databases')


def _pending_names(key: str) -> set[str]:
    return set(json.loads(PENDING_FILE.read_text(encoding='utf-8'))[key])


def pending(test: Any) -> Any:
    r"""Holds ``test`` to failing while ``t33-pending.json`` lists it."""
    if f'py:{test.__name__}' in _pending_names('pending'):
        return pytest.mark.xfail(
            strict=True,
            reason='T3.3: failing until implemented; once it passes, take '
            f'py:{test.__name__} out of contract/tests/t33-pending.json',
        )(test)
    return test


def test_t33_the_pending_list_only_shrinks() -> None:
    assert _pending_names('pending') <= _pending_names('authored')


# --- helpers ---------------------------------------------------------------


def _duck(*statements: str) -> Any:
    import duckdb

    duck = duckdb.connect()
    duck.execute("SET TimeZone = 'UTC'")
    for statement in statements:
        duck.execute(statement)
    return duck


def _arrow(statement: str) -> Any:
    r"""The Arrow type of the one column ``statement`` selects."""
    duck = _duck()
    reader = duck.execute(statement)
    fetch = getattr(reader, 'to_arrow_table', None) or reader.fetch_arrow_table
    return fetch().schema.field(0).type


def _plain(declared: str) -> str:
    r"""A declared type compared as written apart from case, spaces and
    quoting: ``STRUCT<`age`: INT>`` is ``STRUCT<AGE:INT>``."""
    return re.sub(r'[\s`"]', '', declared).upper()


#: The integer types' ranges, by the name an engine declares them with.
_SIGNED = {
    'databricks': {
        'TINYINT': 2**7,
        'BYTE': 2**7,
        'SMALLINT': 2**15,
        'SHORT': 2**15,
        'INT': 2**31,
        'INTEGER': 2**31,
        'BIGINT': 2**63,
        'LONG': 2**63,
    },
    # Every Snowflake integer type is NUMBER(38,0).
    'snowflake': {
        name: 10**38
        for name in (
            'NUMBER',
            'INT',
            'INTEGER',
            'BIGINT',
            'SMALLINT',
            'TINYINT',
            'BYTEINT',
        )
    },
}


def _holds(declared: str, engine: str, low: int, high: int) -> bool:
    r"""Whether ``declared`` holds every integer from ``low`` to ``high``."""
    plain = _plain(declared)
    exact = re.fullmatch(r'(?:DECIMAL|NUMERIC|NUMBER)\((\d+),0\)', plain)
    if exact:
        bound = 10 ** int(exact.group(1))
        return -bound < low and high < bound
    bound = _SIGNED[engine].get(plain)
    return bound is not None and -bound <= low and high < bound


# --- the type map ----------------------------------------------------------


@pending
def test_t33_the_type_map_has_each_warehouse_engine() -> None:
    r"""Brief, "The loader": Phase 2's loader extended with a type map per
    engine, in ``databases.py`` and the vector file together; each
    warehouse leg loads through its engine's entries (Lakebase is a
    Postgres server)."""
    vectors = json.loads((ROOT / 'contract/type-map-vectors.json').read_text())
    for name, entry in databases.TYPE_MAP.items():
        for engine in WAREHOUSE_ENGINES:
            assert engine in entry, (name, engine)
            assert engine in vectors['map'][name], (name, engine)
    assert databases.ENGINE == {
        'databricks-sql': 'databricks',
        'databricks-spark': 'databricks',
        'snowflake': 'snowflake',
        'lakebase': 'postgres',
    }


#: (DuckDB's expression, engine, the declared types accepted), each from
#: the engine's own type names: Databricks' Delta column types, and
#: Snowflake's as ``_to_dtype`` reads them.
DECLARED = [
    ('TRUE', 'databricks', {'BOOLEAN'}),
    ('TRUE', 'snowflake', {'BOOLEAN'}),
    ("'x'", 'databricks', {'STRING'}),
    ("'x'", 'snowflake', {'VARCHAR', 'STRING', 'TEXT'}),
    ("'x'::BLOB", 'databricks', {'BINARY'}),
    ("'x'::BLOB", 'snowflake', {'BINARY', 'VARBINARY'}),
    ("DATE '2024-01-01'", 'databricks', {'DATE'}),
    ("DATE '2024-01-01'", 'snowflake', {'DATE'}),
    ('1::DOUBLE', 'databricks', {'DOUBLE'}),
    (
        '1::DOUBLE',
        'snowflake',
        {'FLOAT', 'DOUBLE', 'DOUBLEPRECISION', 'FLOAT8', 'REAL'},
    ),
    ('1::FLOAT', 'databricks', {'FLOAT', 'REAL'}),
    ('1.5::DECIMAL(38,6)', 'databricks', {'DECIMAL(38,6)'}),
    ('1.5::DECIMAL(38,6)', 'snowflake', {'NUMBER(38,6)', 'DECIMAL(38,6)'}),
    ('1.5::DECIMAL(9,3)', 'databricks', {'DECIMAL(9,3)'}),
    ('1.5::DECIMAL(9,3)', 'snowflake', {'NUMBER(9,3)', 'DECIMAL(9,3)'}),
    # Databricks' TIMESTAMP is an instant in UTC read in the session's
    # zone; TIMESTAMP_NTZ is the wall clock.
    ("TIMESTAMP '2024-01-01 00:00:00'", 'databricks', {'TIMESTAMP_NTZ'}),
    ("TIMESTAMPTZ '2024-01-01 00:00:00+00'", 'databricks', {'TIMESTAMP'}),
    # Snowflake: the wall clock is TIMESTAMP_NTZ, and a zoned instant is
    # TIMESTAMP_LTZ, read in the session's zone (the brief's appended
    # section: the session-zone case, snowflake.zone.ltz-time).
    (
        "TIMESTAMP '2024-01-01 00:00:00'",
        'snowflake',
        {f'TIMESTAMP_NTZ{p}' for p in ('', '(6)', '(9)')},
    ),
    (
        "TIMESTAMPTZ '2024-01-01 00:00:00+00'",
        'snowflake',
        {f'TIMESTAMP_LTZ{p}' for p in ('', '(6)', '(9)')},
    ),
    # A nanosecond wall clock: Snowflake's TIMESTAMP_NTZ holds nine digits
    # (nine when no precision is written).
    (
        "TIMESTAMP_NS '2024-01-01 00:00:00.000000001'",
        'snowflake',
        {'TIMESTAMP_NTZ', 'TIMESTAMP_NTZ(9)'},
    ),
    # Databricks' nested types, as the struct-feature case
    # (databricks.value.struct-feature) loads one.
    (
        "{'age': 1::INTEGER, 'country': 'x'}",
        'databricks',
        {
            'STRUCT<AGE:INT,COUNTRY:STRING>',
            'STRUCT<AGE:INTEGER,COUNTRY:STRING>',
        },
    ),
    ('[1, 2]::BIGINT[]', 'databricks', {'ARRAY<BIGINT>'}),
    ("map(['k'], [2::BIGINT])", 'databricks', {'MAP<STRING,BIGINT>'}),
]


@pending
@pytest.mark.parametrize(
    ('expression', 'engine', 'accepted'),
    DECLARED,
    ids=[f'{engine}:{expression}' for expression, engine, _ in DECLARED],
)
def test_t33_each_engine_declares_its_own_types(
    expression: str, engine: str, accepted: set[str]
) -> None:
    r"""Brief, "The loader" and "Input axes": each engine's own type where
    it has one: ``DECIMAL`` to 38 digits with its scale, each engine's
    timestamp kinds (Databricks' ``TIMESTAMP`` and ``TIMESTAMP_NTZ``,
    Snowflake's ``TIMESTAMP_NTZ`` and ``_LTZ``), Databricks' ``STRUCT``,
    ``ARRAY`` and ``MAP``."""
    kind = _arrow(f'SELECT {expression} AS v')
    declared = databases.column_type(kind, engine, 't.v')
    assert _plain(declared) in {_plain(a) for a in accepted}, (kind, declared)


#: The unsigned integers' largest values, by DuckDB type.
UNSIGNED = {
    'UTINYINT': 2**8 - 1,
    'USMALLINT': 2**16 - 1,
    'UINTEGER': 2**32 - 1,
    'UBIGINT': 2**64 - 1,
}


@pending
@pytest.mark.parametrize('engine', WAREHOUSE_ENGINES)
def test_t33_unsigned_integers_keep_every_value(engine: str) -> None:
    r"""Brief, "The loader": Databricks and Snowflake have no unsigned
    integers; each unsigned column is declared in a type that holds its
    whole range, and its largest value and zero load unchanged."""
    columns = ', '.join(
        f'{value}::{kind} AS "{kind}"' for kind, value in UNSIGNED.items()
    )
    duck = _duck(
        f'CREATE TABLE t AS SELECT {columns} UNION ALL SELECT '
        + ', '.join(f'0::{kind}' for kind in UNSIGNED)
    )
    data = databases._read(duck, 't')
    loaded = databases.stored(duck, 't', engine)
    for kind, value in UNSIGNED.items():
        declared = databases.column_type(
            data.schema.field(kind).type, engine, f't.{kind}'
        )
        assert _holds(declared, engine, 0, value), (kind, declared)
        found = [
            None if cell is None else int(Decimal(str(cell)))
            for cell in loaded.column(kind).to_pylist()
        ]
        assert found == [value, 0], (kind, found)


@pending
@pytest.mark.parametrize('engine', WAREHOUSE_ENGINES)
def test_t33_a_decimal_of_38_digits_keeps_every_digit(engine: str) -> None:
    r"""Brief, "The loader": ``DECIMAL`` up to 38 digits, a key past 64 bits
    included (the Snowflake case snowflake.key.number-38-0.past-int64 keys
    10**25 and up), with its scale."""
    duck = _duck(
        'CREATE TABLE t AS SELECT * FROM (VALUES '
        "('10000000000000000000000007'::DECIMAL(38,0), "
        "'-12345678901234567890123456789012.345678'::DECIMAL(38,6))) t(k, v)"
    )
    data = databases._read(duck, 't')
    for name, precision, scale in (('k', 38, 0), ('v', 38, 6)):
        declared = _plain(
            databases.column_type(
                data.schema.field(name).type, engine, f't.{name}'
            )
        )
        assert re.fullmatch(
            rf'(DECIMAL|NUMBER|NUMERIC)\({precision},{scale}\)', declared
        ), (name, declared)
    row = databases.stored(duck, 't', engine).to_pylist()
    assert [Decimal(str(row[0]['k'])), Decimal(str(row[0]['v']))] == [
        Decimal('10000000000000000000000007'),
        Decimal('-12345678901234567890123456789012.345678'),
    ]


@pending
def test_t33_types_an_engine_cannot_store_are_listed() -> None:
    r"""Brief, "The loader": a type an engine cannot store is listed, and its
    cell never counts (KO2-47). No column holds a time of day on
    Databricks; neither engine holds an interval with months, days and a
    time together, nor a union."""
    assert {'time', 'interval', 'union'} <= set(
        databases.non_native('databricks')
    )
    assert {'interval', 'union'} <= set(databases.non_native('snowflake'))
    # Listed is stored in a form of its own: the time loads, and is no
    # longer an Arrow time.
    import pyarrow as pa

    duck = _duck("CREATE TABLE t AS SELECT TIME '12:00:01.5' AS v")
    loaded = databases.stored(duck, 't', 'databricks')
    assert not pa.types.is_time(loaded.schema.field('v').type)
    assert loaded.num_rows == 1 and loaded.column('v').null_count == 0


@pending
def test_t33_a_nanosecond_is_refused_where_the_engine_holds_microseconds() -> (
    None
):
    r"""Brief, "Input axes": nanosecond times are refused by name where the
    engine holds microseconds (Databricks), never rounded; a nanosecond
    value with whole microseconds loads there, and Snowflake, whose
    timestamps hold nine digits, keeps the nanosecond."""
    duck = _duck(
        "CREATE TABLE t AS SELECT TIMESTAMP_NS '2023-01-01 00:00:00.000000001' AS ns"
    )
    with pytest.raises(databases.Unloadable, match=r't\.ns'):
        databases.stored(duck, 't', 'databricks')
    whole = _duck(
        "CREATE TABLE t AS SELECT TIMESTAMP_NS '2023-01-01 00:00:00.000001' AS ns"
    )
    assert databases.stored(whole, 't', 'databricks').num_rows == 1
    import pyarrow as pa

    kept = databases.stored(duck, 't', 'snowflake').column('ns')
    ticks = kept.cast(pa.timestamp('ns')).cast(pa.int64()).to_pylist()
    assert ticks == [1672531200 * 10**9 + 1]


#: One column of each family DuckDB builds, a value and then a NULL.
EVERY_FAMILY = (
    'TRUE',
    '1::TINYINT',
    '1::SMALLINT',
    '1::INTEGER',
    '1::BIGINT',
    '1::UTINYINT',
    '1::USMALLINT',
    '1::UINTEGER',
    '1::UBIGINT',
    '1.5::FLOAT',
    '1.5::DOUBLE',
    '1.5::DECIMAL(10,2)',
    "'x'",
    "'x'::BLOB",
    "DATE '2024-01-01'",
    "TIME '12:00:00'",
    "TIMESTAMP '2024-01-01 00:00:00'",
    "TIMESTAMPTZ '2024-01-01 00:00:00+00'",
    'to_months(1)',
    '[1, 2]::BIGINT[]',
    "{'a': 1::INTEGER}",
    "map(['k'], [2::BIGINT])",
    'union_value(n := 3)::UNION(n INTEGER, s VARCHAR)',
)


@pending
@pytest.mark.parametrize('engine', WAREHOUSE_ENGINES)
def test_t33_a_null_stays_null_in_every_type(engine: str) -> None:
    r"""Brief, "Input axes": nulls in every type, on each engine: every
    family's column loads (in its own type or its stored form), its value
    still a value and its NULL still a NULL."""
    names = [f'c{i}' for i in range(len(EVERY_FAMILY))]
    values = ', '.join(
        f'{expression} AS {name}'
        for expression, name in zip(EVERY_FAMILY, names, strict=True)
    )
    duck = _duck(
        f'CREATE TABLE t AS SELECT 0 AS id, {values} '
        'UNION ALL BY NAME SELECT 1 AS id'
    )
    data = databases._read(duck, 't')
    for name in names:
        databases.column_type(data.schema.field(name).type, engine, f't.{name}')
    loaded = databases.stored(duck, 't', engine)
    assert loaded.num_rows == 2
    for name, expression in zip(names, EVERY_FAMILY, strict=True):
        cells = loaded.column(name).to_pylist()
        assert cells[0] is not None and cells[1] is None, (expression, cells)


def _load_bases() -> dict[str, list[str]]:
    return _module('run_python').load_bases()


S1_CASES = sorted((ROOT / 'contract/cases/s1').glob('*.json'))


@pending
@pytest.mark.parametrize('engine', WAREHOUSE_ENGINES)
def test_t33_every_s1_case_loads_or_is_refused_by_name(engine: str) -> None:
    r"""Choice 1: every S1 case runs on every warehouse leg whose engine
    stores its data, by the default that runs it on DuckDB; a table an
    engine cannot hold is refused naming its column (:class:`Unloadable`),
    never by another error, and a table it holds keeps every row."""
    import duckdb

    bases = _load_bases()
    loaded = 0
    for path in S1_CASES:
        case = json.loads(path.read_text())
        given = case['input']
        duck = duckdb.connect()
        for statement in [*bases[given['base']], *given.get('setup', [])]:
            duck.execute(statement)
        for table in databases._tables(duck):
            data = databases._read(duck, table)
            try:
                for field in data.schema:
                    databases.column_type(
                        field.type, engine, f'{table}.{field.name}'
                    )
                stored = databases.stored(duck, table, engine)
            except databases.Unloadable as refused:
                assert str(refused).startswith(f'{table}.'), (
                    path.stem,
                    refused,
                )
                continue
            assert stored.num_rows == data.num_rows, (path.stem, table)
            assert stored.column_names == data.column_names, (path.stem, table)
            loaded += 1
        duck.close()
    # Not vacuous: most cases' tables load.
    assert loaded > len(S1_CASES), loaded


# --- Lakebase, on a local Postgres -----------------------------------------


def _postgres_dsn() -> str:
    dsn = os.environ.get('SDM_POSTGRES_TEST_DSN')
    if not dsn:
        # Required (a Postgres job): that fails; elsewhere it is skipped,
        # never passed.
        assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
            'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
        )
        pytest.skip('no Postgres server named (SDM_POSTGRES_TEST_DSN)')
    return dsn


@pending
def test_t33_a_table_a_catalog_relation_shadows_loads_in_its_schema() -> None:
    r"""The brief's appended section: the loader qualifies its ``COPY`` by
    schema, or the ``pg_roles`` case (lakebase.name.shadowed-by-catalog)
    does not load: an unqualified ``pg_roles`` names ``pg_catalog``'s view
    first, whatever the search path."""
    dsn = _postgres_dsn()
    import psycopg
    from psycopg.conninfo import make_conninfo

    duck = _duck(
        'CREATE TABLE pg_roles AS SELECT * FROM (VALUES (0, 10.5), (1, 20.5)) '
        't(user_id, v)'
    )
    schema = f'loader_{uuid.uuid4().hex[:12]}'
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute(f'CREATE SCHEMA {schema}')
    try:
        scoped = make_conninfo(dsn, options=f'-c search_path={schema}')
        databases.load(duck, 'postgres', scoped)
        with psycopg.connect(dsn) as connection:
            rows = connection.execute(
                f'SELECT user_id FROM {schema}.pg_roles ORDER BY user_id'
            ).fetchall()
    finally:
        with psycopg.connect(dsn, autocommit=True) as admin:
            admin.execute(f'DROP SCHEMA {schema} CASCADE')
    assert rows == [(0,), (1,)]
