# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The contract loader's own tests (``contract/runners/databases.py``,
T2.1a): its type map is the vector file's, every S1 case's tables load on
SQLite with each column declared as the map says, and a value Postgres
cannot hold exactly is refused by name. The acceptance tests are
``tests/test_t21a_type_map.py``'s.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sqlite3
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNERS = ROOT / 'contract/runners'


def _module(name: str) -> ModuleType:
    key = f'loader_tests_{name}'
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, RUNNERS / f'{name}.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


databases = _module('databases')
runner = _module('run_python')


def test_the_type_map_is_the_vector_file_s() -> None:
    vectors = json.loads((ROOT / 'contract/type-map-vectors.json').read_text())
    assert vectors['map'] == databases.TYPE_MAP
    assert databases.non_native('sqlite') == [
        'date',
        'decimal',
        'interval',
        'list',
        'map',
        'struct',
        'time',
        'timestamp',
        'timestamptz',
        'uint64',
        'union',
    ]
    assert databases.non_native('postgres') == ['map', 'union']


def _duck(case: dict[str, object]) -> object:
    import duckdb

    bases = runner.load_bases()
    duck = duckdb.connect()
    given = case['input']
    assert isinstance(given, dict)
    for statement in [*bases[given['base']], *given.get('setup', [])]:
        duck.execute(statement)
    return duck


@pytest.mark.parametrize(
    'path',
    sorted((ROOT / 'contract/cases/s1').glob('*.json')),
    ids=lambda path: path.stem,
)
def test_every_s1_case_loads_on_sqlite(path: Path, tmp_path: Path) -> None:
    r"""Each table, its columns declared by the map and its rows in
    DuckDB's order, every value kept (a JSON or ISO text read back). A case
    whose ``databases`` leave SQLite out (T2.12's float cases, under their
    exclusion) holds a value SQLite cannot store, which the loader refuses
    by name."""
    case = json.loads(path.read_text())
    duck = _duck(case)
    target = tmp_path / 'case.sqlite'
    if 'sqlite' not in runner.case_databases(case):
        with pytest.raises(databases.Unloadable):
            databases.load(duck, 'sqlite', target)
        return
    databases.load(duck, 'sqlite', target)
    connection = sqlite3.connect(target)
    try:
        for table in databases._tables(duck):
            data = databases._read(duck, table)
            declared = {
                row[1]: row[2]
                for row in connection.execute(
                    f'PRAGMA table_info({databases.quote(table)})'
                )
            }
            for field in data.schema:
                entry = databases.TYPE_MAP[databases.family(field.type)]
                assert declared[field.name] == entry['sqlite'], field
            count = connection.execute(
                f'SELECT count(*) FROM {databases.quote(table)}'
            ).fetchone()
            assert count == (data.num_rows,)
    finally:
        connection.close()


def test_nested_and_temporal_values_are_stored_as_the_map_says(
    tmp_path: Path,
) -> None:
    import duckdb

    duck = duckdb.connect()
    duck.execute(
        "CREATE TABLE t AS SELECT {'a': 1, 'b': 'x'} AS s, map(['k'], [2]) AS m, "
        'union_value(n := 3)::UNION(n INTEGER, s VARCHAR) AS u, '
        '[1, NULL]::BIGINT[] AS l, '
        "TIMESTAMP_NS '2023-01-01 00:00:00.000001' AS ns, "
        "TIME '12:00:01.5' AS t, to_months(1) + to_days(2) AS i, "
        "DATE '0999-01-02' AS d, 18446744073709551615::UBIGINT AS big"
    )
    target = tmp_path / 'nested.sqlite'
    databases.load(duck, 'sqlite', target)
    connection = sqlite3.connect(target)
    try:
        row = connection.execute('SELECT * FROM t').fetchone()
    finally:
        connection.close()
    assert row == (
        '{"a":1,"b":"x"}',
        '{"k":2}',
        '{"n":3}',
        '[1,null]',
        '2023-01-01 00:00:00.000001',
        '12:00:01.5',
        'P1M2DT0S',
        '0999-01-02',
        '18446744073709551615',
    )


def test_a_nanosecond_postgres_cannot_hold_is_refused_by_name() -> None:
    with pytest.raises(databases.Unloadable, match='t.ts'):
        databases._postgres_text(
            ('instant', 1, 'ns', False), 'timestamp', 't.ts'
        )
    assert (
        databases._postgres_text(
            ('instant', 1000, 'ns', False), 'timestamp', 't'
        )
        == '1970-01-01 00:00:00.000001'
    )


def test_python_old_runs_on_duckdb_only(tmp_path: Path) -> None:
    r"""Fix 4 (the orchestrator's decision): off DuckDB the reference side is
    python-ref's committed lines, and the old route is refused by name."""
    import subprocess

    done = subprocess.run(
        [
            sys.executable,
            '-I',
            str(RUNNERS / 'run_python.py'),
            '--mode',
            'python-old',
            '--database',
            'sqlite',
            '--out',
            str(tmp_path / 'old.jsonl'),
            str(ROOT / 'contract/cases/s1/base.regression.json'),
        ],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 1, done.stdout
    assert 'python-old runs on DuckDB only' in done.stderr
    assert not (tmp_path / 'old.jsonl').exists()


def _server_or_nothing(database: str) -> bool:
    import os

    if database != 'postgres' or os.environ.get('SDM_POSTGRES_TEST_DSN'):
        return True
    # No server named: required (a Postgres job), that fails; elsewhere
    # there is nothing to hold.
    assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
        'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
    )
    return False


def _called(database: str, case_id: str) -> dict[str, object]:
    path = next((ROOT / 'contract/cases').rglob(f'{case_id}.json'))
    case = json.loads(path.read_text())
    outcome, _, driver, _, _ = runner._call(
        case, runner.load_bases(), True, database
    )
    return {'outcome': outcome, 'driver': driver}


@pytest.mark.parametrize('database', ['sqlite', 'postgres'])
def test_the_statement_log_holds_the_capture_complete(database: str) -> None:
    r"""Fix 4: off DuckDB the query log is the driver's own statement log,
    and the capture of a run over the loaded database is complete against
    it: every statement counted and its text the one the capture recorded."""
    if not _server_or_nothing(database):
        return
    found = _called(database, 'base.regression')
    driver = found['driver']
    assert isinstance(driver, dict)
    logged, capture = driver['query_log'], driver['capture']
    assert logged['statements'] == len(capture['statements']) > 0
    assert logged['bindings'] == len(capture['bindings'])
    assert logged['compared'] > 0 and logged['differing'] == []
    check = _module_capture_check()
    assert check.complete({'case': 'x@' + database, **found}) == []


def _module_capture_check() -> ModuleType:
    return _module('capture_check')


def test_postgres_result_types_are_named_never_numbered() -> None:
    r"""Fix 4: a composite type the loader creates is recorded by its name,
    never by the number the server gives it, so two runs over databases of
    their own record the same types."""
    if not _server_or_nothing('postgres'):
        return

    def types(found: dict[str, object]) -> list[list[str]]:
        driver = found['driver']
        assert isinstance(driver, dict)
        return [
            result['values']['types']
            for statement in driver['capture']['statements']
            for result in statement['results']
            if result['values']['types'] is not None
        ]

    first, second = (_called('postgres', 'nested.struct') for _ in range(2))
    assert types(first) == types(second)
    named = [kind for kinds in types(first) for kind in kinds]
    assert any(kind.startswith('sdm_') for kind in named), named
    assert not any(kind.isdigit() for kind in named), named


def _sqlite_rows(tmp_path: Path, *statements: str) -> list[tuple[object, ...]]:
    import duckdb

    duck = duckdb.connect()
    for statement in statements:
        duck.execute(statement)
    target = tmp_path / 'rows.sqlite'
    databases.load(duck, 'sqlite', target)
    connection = sqlite3.connect(target)
    try:
        return connection.execute('SELECT * FROM t ORDER BY 1').fetchall()
    finally:
        connection.close()


def test_a_null_array_keeps_its_neighbours_values(tmp_path: Path) -> None:
    r"""T2.1a fix 1: a NULL fixed-size or variable list before and between
    others moves no value to another row."""
    assert _sqlite_rows(
        tmp_path,
        'CREATE TABLE t AS SELECT * FROM (VALUES (0, NULL::INTEGER[2]), '
        '(1, [10, 20]::INTEGER[2]), (2, NULL), (3, [30, 40]::INTEGER[2])) '
        't(id, v)',
    ) == [(0, None), (1, '[10,20]'), (2, None), (3, '[30,40]')]


def test_an_empty_map_stays_a_map(tmp_path: Path) -> None:
    r"""T2.1a fix 1: an empty map is the empty object, alone and nested."""
    assert _sqlite_rows(
        tmp_path,
        'CREATE TABLE t AS SELECT * FROM (VALUES '
        "(0, map([]::VARCHAR[], []::INTEGER[]), {'m': map([]::VARCHAR[], []::INTEGER[])}), "
        "(1, map(['k'], [1]), {'m': map(['k'], [1])})) t(id, v, s)",
    ) == [(0, '{}', '{"m":{}}'), (1, '{"k":1}', '{"m":{"k":1}}')]


def test_a_nan_sqlite_cannot_hold_is_refused_by_name(tmp_path: Path) -> None:
    r"""T2.1a fix 1: SQLite stores a NaN as NULL, so the loader refuses it,
    naming the column, rather than load a null (``sqlite_refuses``)."""
    assert databases.TYPE_MAP['float64']['sqlite_refuses'] == 'NaN'
    with pytest.raises(databases.Unloadable, match=r't\.v: a NaN'):
        _sqlite_rows(tmp_path, "CREATE TABLE t AS SELECT 'NaN'::DOUBLE AS v")


def test_a_negative_zero_sqlite_cannot_hold_is_refused_by_name(
    tmp_path: Path,
) -> None:
    r"""T2.12: SQLite stores a REAL -0.0 as 0.0, so the loader refuses it,
    naming the column, rather than load another value; 0.0 still loads."""
    for name in ('minus', 'minus32', 'plus'):
        (tmp_path / name).mkdir()
    with pytest.raises(databases.Unloadable, match=r't\.v: -0\.0'):
        _sqlite_rows(
            tmp_path / 'minus', "CREATE TABLE t AS SELECT '-0.0'::DOUBLE AS v"
        )
    with pytest.raises(databases.Unloadable, match=r't\.v: -0\.0'):
        _sqlite_rows(
            tmp_path / 'minus32', "CREATE TABLE t AS SELECT '-0.0'::FLOAT AS v"
        )
    assert _sqlite_rows(
        tmp_path / 'plus', 'CREATE TABLE t AS SELECT 0.0::DOUBLE AS v'
    ) == [(0.0,)]


def test_nested_composites_load_on_postgres() -> None:
    r"""T2.1a fix 1: a struct in a struct, and a list of structs, load on
    Postgres as composite types created inner first."""
    import os
    import uuid

    dsn = os.environ.get('SDM_POSTGRES_TEST_DSN')
    if not dsn:
        # No server named: required (a Postgres job), that fails; elsewhere
        # there is nothing to hold.
        assert not os.environ.get('SDM_REQUIRE_POSTGRES')
        return
    import duckdb
    import psycopg
    from psycopg.conninfo import make_conninfo

    duck = duckdb.connect()
    duck.execute(
        "CREATE TABLE t AS SELECT 1 AS id, {'a': {'b': 1}} AS v, "
        "[{'c': 2, 'd': 'x y'}, NULL] AS w"
    )
    schema = f'loader_{uuid.uuid4().hex[:12]}'
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute(f'CREATE SCHEMA {schema}')
    try:
        scoped = make_conninfo(dsn, options=f'-c search_path={schema}')
        databases.load(duck, 'postgres', scoped)
        with psycopg.connect(scoped) as connection:
            row = connection.execute(
                'SELECT (v).a, ((v).a).b, w[1], w[2] IS NULL FROM t'
            ).fetchone()
    finally:
        with psycopg.connect(dsn, autocommit=True) as admin:
            admin.execute(f'DROP SCHEMA {schema} CASCADE')
    assert row == ('(1)', 1, '(2,"x y")', True)


def test_a_list_of_dates_keeps_every_date(tmp_path: Path) -> None:
    r"""Fix 1 (round 1's P2-2): a list of dates, alone and in a struct, is
    every date's ISO text, never mistaken for a map."""
    assert _sqlite_rows(
        tmp_path,
        "CREATE TABLE t AS SELECT 1 AS id, [DATE '2020-01-01', "
        "DATE '2021-06-30'] AS v, {'a': [DATE '2020-01-01', "
        "DATE '2021-06-30']} AS s",
    ) == [
        (
            1,
            '["2020-01-01","2021-06-30"]',
            '{"a":["2020-01-01","2021-06-30"]}',
        )
    ]


@pytest.mark.parametrize(
    ('statement', 'named'),
    [
        ("CREATE TABLE t AS SELECT TIMETZ '12:00:00+05' AS v", 't.v'),
        ("CREATE TABLE t AS SELECT '101'::BIT AS v", 't.v'),
        (
            "CREATE TABLE t AS SELECT TIMESTAMP '9999-12-31' + INTERVAL 2 DAY AS v",
            't.v',
        ),
    ],
    ids=['timetz', 'bit', 'past year 9999'],
)
def test_input_the_loader_cannot_carry_is_refused_by_name(
    statement: str, named: str, tmp_path: Path
) -> None:
    r"""Fix 1 (round 1's P3-2): a type Arrow does not carry whole, and a
    time past what the loader writes, are refused naming the column."""
    with pytest.raises(databases.Unloadable, match=re.escape(named)):
        _sqlite_rows(tmp_path, statement)


@pytest.mark.parametrize(
    'path',
    sorted((ROOT / 'contract/cases/s1').glob('*.json')),
    ids=lambda path: path.stem,
)
def test_every_s1_case_loads_on_postgres(path: Path) -> None:
    r"""Fix 1 (round 1's P3-5): the Postgres twin of the SQLite load, where
    a server is named; required, a missing one fails."""
    import os
    import uuid

    dsn = os.environ.get('SDM_POSTGRES_TEST_DSN')
    if not dsn:
        assert not os.environ.get('SDM_REQUIRE_POSTGRES')
        return
    import psycopg
    from psycopg.conninfo import make_conninfo

    duck = _duck(json.loads(path.read_text()))
    schema = f'loader_{uuid.uuid4().hex[:12]}'
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute(f'CREATE SCHEMA {schema}')
    try:
        scoped = make_conninfo(dsn, options=f'-c search_path={schema}')
        databases.load(duck, 'postgres', scoped)
        with psycopg.connect(scoped) as connection:
            for table in databases._tables(duck):
                [(count,)] = connection.execute(
                    f'SELECT count(*) FROM {databases.quote(table)}'
                ).fetchall()
                assert count == databases._read(duck, table).num_rows, table
    finally:
        with psycopg.connect(dsn, autocommit=True) as admin:
            admin.execute(f'DROP SCHEMA {schema} CASCADE')


def test_a_field_named_as_a_lost_type_is_a_name(tmp_path: Path) -> None:
    r"""Fix 2: struct fields named ``BIT`` and ``TIME WITH TIME ZONE`` are
    names, read from DuckDB's type tree, and their integers load."""
    assert _sqlite_rows(
        tmp_path,
        "CREATE TABLE t AS SELECT 1 AS id, {'BIT': 42, "
        "'TIME WITH TIME ZONE': 7} AS v",
    ) == [(1, '{"BIT":42,"TIME WITH TIME ZONE":7}')]


@pytest.mark.parametrize(
    'value',
    [
        "map(['k'], [chr(0)])",
        "map([chr(0)], ['v'])",
        'union_value(s := chr(0))',
    ],
    ids=['map value', 'map key', 'union'],
)
def test_a_nul_inside_json_is_refused_by_name_on_postgres(value: str) -> None:
    r"""Fix 2: a NUL in a map's or union's text, which jsonb cannot hold, is
    refused naming the column before COPY sees it."""
    import os

    dsn = os.environ.get('SDM_POSTGRES_TEST_DSN')
    if not dsn:
        assert not os.environ.get('SDM_REQUIRE_POSTGRES')
        return
    import duckdb

    duck = duckdb.connect()
    duck.execute(f'CREATE TABLE t AS SELECT {value} AS v')
    with pytest.raises(databases.Unloadable, match=r't\.v: text holding a NUL'):
        databases.load(duck, 'postgres', dsn)


def test_a_table_name_with_a_double_quote_loads(tmp_path: Path) -> None:
    r"""Fix 4 (round 3's P3-1): the type check reads a quoted name as the
    table's, so a name holding ``"`` loads."""
    import duckdb

    duck = duckdb.connect()
    duck.execute('CREATE TABLE "my ""t"" x" AS SELECT 1 AS id')
    target = tmp_path / 'quoted.sqlite'
    databases.load(duck, 'sqlite', target)
    connection = sqlite3.connect(target)
    try:
        assert connection.execute('SELECT id FROM "my ""t"" x"').fetchall() == [
            (1,)
        ]
    finally:
        connection.close()


def test_composite_names_never_collide() -> None:
    r"""Fix 4 (round 3's composite-name finding): a ``.`` in a name makes no
    two struct paths share a composite type's name, and each fits
    Postgres's 63 bytes."""
    one = databases.composite_name(['t', 'v', 'a'])
    other = databases.composite_name(['t', 'v.a'])
    assert one != other
    assert len(one.encode()) <= 63
    assert one == databases.composite_name(['t', 'v', 'a'])
