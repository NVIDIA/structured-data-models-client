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
        if os.environ.get('SDM_REQUIRE_POSTGRES'):
            pytest.fail(
                'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
            )
        pytest.skip('SDM_POSTGRES_TEST_DSN is not set')
    line = _old_client(database, tmp_path)
    assert line['case'] == f'base.regression@{database}'
    assert line['outcome']['result'] == 'success'
    driver = line['driver']
    assert isinstance(driver, dict) and driver['query_log'] is None
    capture = driver['capture']
    assert capture['statements'] and not capture['incomplete']
    assert any(statement['results'] for statement in capture['statements'])
