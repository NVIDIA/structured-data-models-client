# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The driver capture's controls (design, "Driver capture"; decision Q2).

Run in an environment with the client, the relational engine and DuckDB:

    python -m pytest contract/tests/test_capture.py -q

The same file runs in the reference's environment too, where the
dispatch tests read the reference's own ``sdm_connectors`` and engine.

- Dispatch: the proxy passes the reference's backend dispatch on class
  module, ``reader._query_handle``'s ``isinstance(cursor,
  type(connection))``, and ``Graph.from_duckdb``'s ``isinstance(connection,
  DuckDBPyConnection)``, which takes the connection as given rather than
  opening its own.
- The scripted cursor control: over a scripted driver that records every
  call it gets, the proxy makes the same calls once, in order, and gives
  back the same answers, through derived cursors, typed parameters,
  several batches and an empty result.
- The same over DuckDB itself, its query log counting what ran.
- Conservation on ``base.regression`` on this environment's runner:
  capture on and off give the same public projection, and the captured
  counts equal DuckDB's query log.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

import duckdb
import pyarrow as pa
import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNERS = ROOT / 'contract/runners'


def _module(name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, RUNNERS / f'{name}.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_python = _module('run_python')
capture_check = _module('capture_check')


def _shop() -> Any:
    connection = duckdb.connect()
    for statement in run_python.load_bases()['shop']:
        connection.execute(statement)
    return connection


# --- dispatch ----------------------------------------------------------------


def test_the_backend_dispatch_reads_the_proxy_as_duckdb() -> None:
    from sdm_connectors import DuckDBSource, Source
    from sdm_connectors.source import _backend_of

    connection = duckdb.connect()
    proxy = run_python.capture(connection, run_python.DriverLog())
    assert _backend_of(connection) == 'duckdb'
    assert _backend_of(proxy) == 'duckdb'
    assert type(Source(proxy)) is DuckDBSource
    assert DuckDBSource(proxy).connection is proxy


def test_a_derived_cursor_is_the_proxy_type_as_the_driver_s_is() -> None:
    from sdm_connectors.reader import _query_handle

    connection = duckdb.connect()
    log = run_python.DriverLog()
    proxy = run_python.capture(connection, log)
    cursor = proxy.cursor()
    assert isinstance(connection.cursor(), type(connection))
    assert isinstance(cursor, type(proxy))
    assert type(cursor) is type(proxy)
    # The reader runs on the connection itself, as it does on the driver's,
    # and closes the duplicate it was handed.
    assert _query_handle(connection) == (connection, False)
    handle, owned = _query_handle(proxy)
    assert handle is proxy and owned is False
    assert {'event': 'close', 'handle': 2} in log.events


def test_graph_from_duckdb_takes_the_proxy_as_its_connection() -> None:
    from sdm_relational_engine.rfm import Graph

    connection = _shop()
    log = run_python.DriverLog()
    proxy = run_python.capture(connection, log)
    # Graph.from_duckdb opens its own connection for anything that is not a
    # DuckDBPyConnection; the proxy must be taken as one.
    assert isinstance(proxy, duckdb.DuckDBPyConnection)
    graph = Graph.from_duckdb(proxy, infer_metadata=False, verbose=False)
    assert sorted(graph.tables) == ['orders', 'users']
    connections = [t._connection for t in graph.tables.values()]  # type: ignore[attr-defined]
    assert all(c is proxy for c in connections)
    assert getattr(graph, '_connection', None) is None
    assert log.statements[0]['sql'].startswith('SELECT table_name FROM')


# --- the scripted cursor control -----------------------------------------------


class _Script:
    r"""A scripted driver: each statement's rows, given in batches, and a
    record of every call any of its handles gets."""

    def __init__(self, answers: dict[str, list[list[tuple[Any, ...]]]]):
        self.answers = answers
        self.calls: list[tuple[Any, ...]] = []
        self.handles = 0


class ScriptedConnection:
    def __init__(self, script: _Script) -> None:
        self._script = script
        self._handle = script.handles
        script.handles += 1
        self._pending: list[tuple[Any, ...]] = []
        self._batches: list[list[tuple[Any, ...]]] = []

    def _call(self, *call: Any) -> None:
        self._script.calls.append((self._handle, *call))

    def cursor(self) -> ScriptedConnection:
        self._call('cursor')
        return ScriptedConnection(self._script)

    def execute(self, query: str, parameters: Any = None) -> Any:
        self._call('execute', query, parameters)
        self._batches = [list(b) for b in self._script.answers[query]]
        self._pending = [row for batch in self._batches for row in batch]
        return self

    @property
    def description(self) -> list[tuple[Any, ...]]:
        # As DuckDB's: the columns of the result last run and their types;
        # reading it is no call.
        return [('a', 'BIGINT'), ('b', 'VARCHAR')]

    def fetchone(self) -> Any:
        self._call('fetchone')
        return self._pending.pop(0) if self._pending else None

    def fetchmany(self, size: int = 1) -> list[tuple[Any, ...]]:
        self._call('fetchmany', size)
        taken, self._pending = self._pending[:size], self._pending[size:]
        return taken

    def fetchall(self) -> list[tuple[Any, ...]]:
        self._call('fetchall')
        taken, self._pending = self._pending, []
        return taken

    def to_arrow_table(self, batch_size: int = 1_000_000) -> pa.Table:
        self._call('to_arrow_table', batch_size)
        schema = pa.schema([('a', pa.int64()), ('b', pa.string())])
        batches = [
            pa.record_batch(
                [
                    pa.array([r[0] for r in batch], pa.int64()),
                    pa.array([r[1] for r in batch], pa.string()),
                ],
                schema=schema,
            )
            for batch in self._batches
        ]
        self._pending = []
        return pa.Table.from_batches(batches, schema=schema)

    def register(self, name: str, value: Any) -> Any:
        self._call('register', name, value.num_rows)
        return self

    def close(self) -> None:
        self._call('close')

    def __enter__(self) -> ScriptedConnection:
        self._call('enter')
        return self

    def __exit__(self, *exc: Any) -> None:
        self._call('exit')


ScriptedConnection.__module__ = 'duckdb.scripted'

_ANSWERS: dict[str, list[list[tuple[Any, ...]]]] = {
    'rows': [[(1, 'x'), (2, None)], [(3, "a'b")], [(4, '')]],
    'empty': [[]],
    'with parameters': [[(7, 'p')]],
}
_PARAMETERS = [
    2**63 + 1,
    Decimal('0.100000000000000001'),
    datetime(2024, 3, 1, 12, 30),
    None,
    'text',
    b'\x00\x01',
    True,
]


def _drive(connection: Any) -> list[Any]:
    r"""The script: what a caller does and reads, through derived cursors,
    with typed parameters, several batches and an empty result."""
    seen: list[Any] = []
    with connection.cursor() as cursor:
        cursor.execute('rows')
        seen.append(cursor.fetchone())
        seen.append(cursor.fetchmany(2))
        seen.append(cursor.fetchall())
        seen.append(cursor.fetchall())
        seen.append(cursor.fetchone())
    with connection.cursor() as cursor:
        seen.append(cursor.execute('empty').fetchall())
        seen.append(cursor.execute('empty').to_arrow_table())
        seen.append(cursor.execute('rows').to_arrow_table(batch_size=2))
        derived = cursor.cursor()
        derived.register('seeds', pa.table({'k': [1, 2, 3]}))
        seen.append(derived.execute('with parameters', _PARAMETERS).fetchall())
        derived.close()
    return seen


def test_the_scripted_cursor_control() -> None:
    plain = _Script(_ANSWERS)
    expected = _drive(ScriptedConnection(plain))
    scripted = _Script(_ANSWERS)
    log = run_python.DriverLog()
    got = _drive(run_python.capture(ScriptedConnection(scripted), log))
    # The same answers, and the driver got the same calls, once each, in
    # the same order: nothing consumed twice, dropped or reordered.
    assert got == expected
    assert scripted.calls == plain.calls
    for table in (got[6], got[7]):
        assert isinstance(table, pa.Table)
    assert [b.num_rows for b in got[7].to_batches()] == [2, 1, 1]
    assert got[6].num_rows == 0 and got[6].schema.names == ['a', 'b']
    # The parameters reached the driver as the very objects passed.
    sent = next(
        c for c in scripted.calls if c[1:3] == ('execute', 'with parameters')
    )
    assert all(a is b for a, b in zip(sent[3], _PARAMETERS, strict=True))
    # The capture saw each statement, every read and the binding.
    assert [s['sql'] for s in log.statements] == [
        'rows',
        'empty',
        'empty',
        'rows',
        'with parameters',
    ]
    assert [s['handle'] for s in log.statements] == [1, 2, 2, 2, 3]
    reads = [[r['method'] for r in s['results']] for s in log.statements]
    assert reads == [
        ['fetchone', 'fetchmany', 'fetchall', 'fetchall', 'fetchone'],
        ['fetchall'],
        ['to_arrow_table'],
        ['to_arrow_table'],
        ['fetchall'],
    ]
    rows = [[r['rows'] for r in s['results']] for s in log.statements]
    assert rows == [[1, 2, 1, 0, 0], [0], [0], [4], [1]]
    assert [p['type'] for p in log.statements[-1]['parameters']] == [
        'builtins.int',
        'decimal.Decimal',
        'datetime.datetime',
        'builtins.NoneType',
        'builtins.str',
        'builtins.bytes',
        'builtins.bool',
    ]
    assert log.bindings == [
        {
            'handle': 3,
            'stage': None,
            'name': 'seeds',
            'type': 'pyarrow.lib.Table',
            'rows': 3,
            'schema': [{'name': 'k', 'type': 'int64'}],
            'digest': run_python._arrow_digest(pa.table({'k': [1, 2, 3]})),
            'values': run_python.arrow_values(pa.table({'k': [1, 2, 3]})),
        }
    ]
    # Each result read has its values in canonical form, with the driver's
    # column names and types (T1.11e); the rows read, not those left.
    first = log.statements[0]['results'][0]['values']
    assert first == run_python.result_values(
        ['a', 'b'], ['BIGINT', 'VARCHAR'], [(1, 'x')]
    )
    assert log.incomplete == [] and log.passed == []
    # Batch boundaries do not reach a result's digest (decision H9).
    one = pa.Table.from_batches([got[7].combine_chunks().to_batches()[0]])
    assert run_python._arrow_digest(one) == run_python._arrow_digest(got[7])


def test_a_result_the_capture_cannot_record_is_reported() -> None:
    log = run_python.DriverLog()
    proxy = run_python.capture(duckdb.connect(), log)
    proxy.execute('SELECT 1 AS a').to_arrow_reader()
    assert log.incomplete == ['to_arrow_reader']


def test_the_control_over_duckdb_and_its_query_log() -> None:
    def drive(capturing: bool) -> tuple[list[Any], list[str], Any]:
        connection = duckdb.connect()
        connection.execute('CREATE TABLE t AS SELECT range AS a FROM range(5)')
        run_python.enable_query_log(connection)
        log = run_python.DriverLog()
        handle = (
            run_python.capture(connection, log) if capturing else connection
        )
        seen: list[Any] = []
        with handle.cursor() as cursor:
            cursor.register('seeds', pa.table({'k': [1, 3, 9]}))
            seen.append(
                cursor.execute(
                    'SELECT a FROM t JOIN seeds ON a = k ORDER BY a'
                ).fetchall()
            )
            seen.append(
                cursor.execute(
                    'SELECT a FROM t WHERE a >= ? ORDER BY a', [2]
                ).to_arrow_table(batch_size=2)
            )
            seen.append(
                cursor.execute('SELECT a FROM t WHERE a < 0').to_arrow_table()
            )
            seen.append(cursor.execute("PRAGMA table_info('t')").fetchall())
        logged = run_python.query_log(connection)
        return seen, logged, log

    off, logged_off, _ = drive(False)
    on, logged_on, log = drive(True)
    assert on == off
    assert on[2].num_rows == 0 and on[2].schema.names == ['a']
    # DuckDB ran the same statements, as many times, with or without it.
    assert len(logged_on) == len(logged_off)
    assert [t for t in logged_on if t] == [t for t in logged_off if t]
    assert len(log.statements) == sum(1 for text in logged_on if text) == 4
    assert len(log.bindings) == sum(1 for text in logged_on if not text) == 1


# --- conservation on base.regression -----------------------------------------------

Run = list[dict[str, Any]]


@pytest.fixture(scope='module')
def pair() -> tuple[Run, Run]:
    r"""``base.regression`` on this environment's runner, with the capture
    and without."""
    cases = run_python.inventory_cases(['base.regression'])
    on = run_python.run('python-old', cases, None, driver_capture=True)
    off = run_python.run('python-old', cases, None, driver_capture=False)
    return on, off


def test_capture_conserves_base_regression(pair: tuple[Run, Run]) -> None:
    on, off = pair
    assert capture_check.check(on, off, ['base.regression']) == []
    assert on[0]['outcome']['result'] == 'success'
    assert on[0]['outcome']['stage'] == 'predict'
    assert len(on[0]['driver']['capture']['statements']) == 16
    assert len(on[0]['driver']['capture']['bindings']) == 3
    logged = on[0]['driver']['query_log']
    assert (logged['compared'], logged['differing']) == (14, [])


def _metadata(run: Run) -> dict[str, Any]:
    metadata: dict[str, Any] = run[0]['public']['payloads'][0]['parsed'][
        'metadata'
    ]
    return metadata


def test_only_the_request_id_s_value_is_left_out(pair: tuple[Run, Run]) -> None:
    on, off = copy.deepcopy(pair)
    # Another random value of the same type passes.
    _metadata(on)['request_id'] = 'sdm-000000000000'
    assert capture_check.check(on, off) == []
    # Deleted from one run only: fails.
    del _metadata(on)['request_id']
    problems = capture_check.check(on, off)
    assert any('request_id' in p for p in problems), problems
    # Deleted from both: the field's absence is compared, and agrees.
    del _metadata(off)['request_id']
    assert capture_check.check(on, off) == []
    # Of another type in one run: fails.
    on, off = copy.deepcopy(pair)
    _metadata(on)['request_id'] = {'number': '7'}
    problems = capture_check.check(on, off)
    assert any('request_id' in p for p in problems), problems
    # Any other metadata field still counts.
    on, off = copy.deepcopy(pair)
    _metadata(on)['source_query'] = 'PREDICT x'
    assert capture_check.check(on, off) != []


def test_missing_or_repeated_evidence_fails(pair: tuple[Run, Run]) -> None:
    on, off = pair
    assert capture_check.check([], []) != []
    assert capture_check.check(on, []) != []
    assert capture_check.check([], off) != []
    twice = capture_check.check(on + on, off + off)
    assert any('more than once' in p for p in twice), twice
    missing = capture_check.check(on, off, ['base.regression', 'base.binary'])
    assert missing == ['base.binary: required, and not in both runs']


def test_a_capture_off_observation_in_the_captured_run_fails(
    pair: tuple[Run, Run], tmp_path: Path
) -> None:
    on, off = pair
    for name, run in (('on', off), ('off', off)):
        (tmp_path / name).write_text(
            ''.join(json.dumps(item) + '\n' for item in run)
        )
    done = subprocess.run(
        [
            sys.executable,
            str(RUNNERS / 'capture_check.py'),
            '--on',
            str(tmp_path / 'on'),
            '--off',
            str(tmp_path / 'off'),
            '--require',
            'base.regression',
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 1, done.stderr
    assert 'capture: FAILED' in done.stdout
    assert 'Traceback' not in done.stderr
    assert 'the capture was off' in done.stderr
    # And two empty files fail too.
    for name in ('on', 'off'):
        (tmp_path / name).write_text('')
    done = subprocess.run(
        [
            sys.executable,
            str(RUNNERS / 'capture_check.py'),
            '--on',
            str(tmp_path / 'on'),
            '--off',
            str(tmp_path / 'off'),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 1 and 'capture: FAILED' in done.stdout


def test_a_statement_text_other_than_the_one_run_fails(
    pair: tuple[Run, Run],
) -> None:
    on, off = copy.deepcopy(pair)
    on[0]['driver']['query_log']['differing'] = [3]
    assert any('ran as text other' in p for p in capture_check.check(on, off))


# --- what a connection hands back as it is ------------------------------------------


def test_a_connection_s_arrow_reader_passes_through_unchanged() -> None:
    def read(capturing: bool) -> tuple[type, pa.Table, list[str]]:
        log = run_python.DriverLog()
        connection = duckdb.connect()
        handle = (
            run_python.capture(connection, log) if capturing else connection
        )
        reader = handle.execute('select 1 AS a, 2 AS b').arrow()
        return type(reader), reader.read_all(), log.incomplete

    plain_type, plain, _ = read(False)
    captured_type, captured, incomplete = read(True)
    assert captured_type is plain_type
    assert captured.equals(plain)
    # The reader is the caller's to consume, so the capture says it could
    # not record it, rather than read it first.
    assert incomplete == ['arrow']


def test_named_parameters_are_recorded_by_name_and_value() -> None:
    log = run_python.DriverLog()
    proxy = run_python.capture(duckdb.connect(), log)
    plain = duckdb.connect().execute('select $value', {'value': 7}).fetchall()
    assert proxy.execute('select $value', {'value': 7}).fetchall() == plain
    assert proxy.execute(
        query='select $value + 1', parameters={'value': 7}
    ).fetchall() == [(8,)]
    assert log.statements[0]['parameters'] == [
        {'name': 'value', 'type': 'builtins.int', 'value': '7'}
    ]
    assert log.statements[1]['sql'] == 'select $value + 1'
    assert log.statements[1]['parameters'] == [
        {'name': 'value', 'type': 'builtins.int', 'value': '7'}
    ]
