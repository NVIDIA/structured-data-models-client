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
    DuckDB's order, every value kept (a JSON or ISO text read back)."""
    duck = _duck(json.loads(path.read_text()))
    target = tmp_path / 'case.sqlite'
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


def _old_client(database: str, tmp_path: Path) -> dict[str, object]:
    import os
    import subprocess

    out = tmp_path / f'{database}.jsonl'
    done = subprocess.run(
        [
            sys.executable,
            '-I',
            str(RUNNERS / 'run_python.py'),
            '--mode',
            'python-old',
            '--database',
            database,
            '--out',
            str(out),
            str(ROOT / 'contract/cases/s1/base.regression.json'),
        ],
        cwd=ROOT,
        env=dict(os.environ),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stderr
    [line] = [json.loads(text) for text in out.read_text().splitlines()]
    return line


@pytest.mark.parametrize('database', ['sqlite', 'postgres'])
def test_the_runner_reads_the_loaded_database_through_the_capture(
    database: str, tmp_path: Path
) -> None:
    r"""``run_python.py --database``: the old client over the loaded
    database, its observation keyed ``<case>@<database>``, every statement
    and result captured through the driver's own connection, and no DuckDB
    query log."""
    import os

    if database == 'postgres' and not os.environ.get('SDM_POSTGRES_TEST_DSN'):
        # No server named: required (a Postgres job), that fails; elsewhere
        # there is nothing to hold, as in test_postgres_required.py.
        assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
            'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
        )
        return
    line = _old_client(database, tmp_path)
    assert line['case'] == f'base.regression@{database}'
    assert line['outcome']['result'] == 'success'
    driver = line['driver']
    assert isinstance(driver, dict) and driver['query_log'] is None
    capture = driver['capture']
    assert capture['statements'] and not capture['incomplete']
    assert any(statement['results'] for statement in capture['statements'])


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
