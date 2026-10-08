# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P4.loop: the toy run passes with the Rust replay host and with the Python
SQLite host.

- The run loop's crate exports ``Run``, ``Step``, ``Query``, ``Outcome`` and
  ``Diagnostic`` in the brief's shape: a crate using them so must compile.
- Its replay-host tests (named ``replay``) run and pass.
- From Python, the toy run "count the rows of table T" asks for one query,
  which a ``SQLiteSource`` runs through ``Source._run`` (the Phase 3 seam),
  and gives back the count; it reads the count only from what it is given,
  and a failure or an unreadable result comes back as a Python exception,
  never a Rust panic.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from _rc import (
    CACHE,
    RUST,
    cargo_metadata,
    find_crate,
    install_local_wheel,
    run,
)

PROBE = r"""
#![allow(dead_code, unused_imports, unreachable_patterns, clippy::all)]

use rc::{Diagnostic, Outcome, Query, Run, Step};

fn step<T>(step: Step<T>) {
    match step {
        Step::RunQueries(queries) => {
            let queries: Vec<Query> = queries;
            for query in queries {
                let Query {
                    id: _,
                    sql: _,
                    params: _,
                    columns: _,
                    ..
                } = query;
            }
        }
        Step::Done(value) => {
            let _: T = value;
        }
        _ => {}
    }
}

fn outcome(outcome: Outcome) {
    match outcome {
        Outcome::Data { .. } => {}
        Outcome::Error {
            sqlstate: _,
            vendor_code: _,
            message: _,
            ..
        } => {}
        _ => {}
    }
}

fn diagnostic(diagnostic: Diagnostic) {
    let Diagnostic {
        code: _,
        message: _,
        ..
    } = diagnostic;
}
"""


@pytest.fixture(scope='module')
def core() -> dict[str, Any]:
    return find_crate('core', cargo_metadata())


def test_the_run_loop_types_have_the_briefs_shape(core, tmp_path):
    probe = tmp_path / 'probe'
    (probe / 'src').mkdir(parents=True)
    manifest_dir = Path(core['manifest_path']).parent
    (probe / 'Cargo.toml').write_text(
        '[package]\nname = "run-loop-probe"\nversion = "0.0.0"\n'
        'edition = "2021"\npublish = false\n\n[workspace]\n\n'
        f'[dependencies]\nrc = {{ package = "{core["name"]}", '
        f'path = "{manifest_dir}" }}\n'
    )
    (probe / 'src' / 'lib.rs').write_text(PROBE)
    lock = RUST / 'Cargo.lock'
    if lock.exists():
        shutil.copy(lock, probe / 'Cargo.lock')
    toolchain = RUST / 'rust-toolchain.toml'
    if toolchain.exists():
        shutil.copy(toolchain, probe / 'rust-toolchain.toml')
    env = dict(os.environ)
    env['CARGO_TARGET_DIR'] = str(CACHE / 'target-loop-probe')
    run(['cargo', 'check', '--quiet'], cwd=probe, env=env)


def test_the_rust_replay_host_runs_the_toy_run(core):
    proc = run(['cargo', 'test', '-p', core['name'], 'replay'], cwd=RUST)
    passed = sum(
        int(n)
        for n in re.findall(r'test result: ok\. (\d+) passed', proc.stdout)
    )
    assert passed > 0, (
        f'no test named replay ran in {core["name"]}:\n{proc.stdout[-4000:]}'
    )


# --- The Python SQLite host ----------------------------------------------------


@pytest.fixture(scope='module')
def native():
    install_local_wheel()
    sys.modules.pop('sdm_core', None)
    sys.modules.pop('sdm_core._native', None)
    from sdm_core import _native

    return _native


@pytest.fixture
def source(tmp_path):
    import adbc_driver_sqlite.dbapi as adbc
    from sdm_connectors import SQLiteSource

    path = tmp_path / 'shop.sqlite'
    connection = adbc.connect(str(path))
    with connection.cursor() as cursor:
        cursor.execute('CREATE TABLE users (id INTEGER, name TEXT)')
        cursor.execute('CREATE TABLE orders (id INTEGER, user_id INTEGER)')
        cursor.execute('CREATE TABLE empty (id INTEGER)')
        cursor.execute("INSERT INTO users VALUES (1, 'a'), (2, 'b'), (3, NULL)")
        cursor.execute(
            'WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n '
            'WHERE i < 12345) INSERT INTO orders SELECT i, i % 3 FROM n'
        )
    connection.commit()
    yield SQLiteSource(connection)
    connection.close()


class Host:
    r"""Runs every query a run asks for through ``Source._run``."""

    def __init__(self, native: Any, source: Any) -> None:
        self.native = native
        self.source = source
        self.queries: list[Any] = []

    def drive(self, run: Any) -> Any:
        step = run.next()
        while step.kind == 'run_queries':
            assert step.queries, 'a run_queries step with no queries'
            for query in step.queries:
                self.queries.append(query)
                assert isinstance(query.sql, str)
                assert isinstance(list(query.params), list)
                columns = list(query.columns)
                assert columns and all(isinstance(c, str) for c in columns)
                try:
                    table = self.source._run(query.sql, query.params or None)
                except Exception as error:
                    outcome = self.native.Outcome.error(
                        getattr(error, 'sqlstate', None),
                        getattr(error, 'vendor_code', None),
                        str(error),
                    )
                else:
                    assert table.column_names == columns, (
                        f'the run expects {columns}, the query returns '
                        f'{table.column_names}'
                    )
                    outcome = self.native.Outcome.data(table)
                run.submit(query.id, outcome)
            step = run.next()
        assert step.kind == 'done', f'unknown step kind {step.kind!r}'
        return step.value


@pytest.mark.parametrize(
    ('table', 'count'), [('users', 3), ('orders', 12345), ('empty', 0)]
)
def test_the_toy_run_counts_rows_through_source_run(
    native, source, table, count
):
    host = Host(native, source)
    value = host.drive(native.count_rows(table))
    assert type(value) is int and value == count
    assert len(host.queries) == 1, 'the toy run asks for exactly one query'


def first_query(native, table: str = 'users') -> tuple[Any, Any]:
    run = native.count_rows(table)
    step = run.next()
    assert step.kind == 'run_queries' and len(step.queries) == 1
    return run, step.queries[0]


def test_the_count_comes_from_the_submitted_result(native):
    run, query = first_query(native)
    table = pa.table({query.columns[0]: pa.array([42], pa.int64())})
    run.submit(query.id, native.Outcome.data(table))
    step = run.next()
    assert step.kind == 'done' and step.value == 42


def test_a_database_error_is_raised_with_its_message(native, source):
    run = native.count_rows('no_such_table')
    host = Host(native, source)
    with pytest.raises(Exception) as raised:
        host.drive(run)
    assert not isinstance(raised.value, AssertionError), raised.value
    assert 'no_such_table' in str(raised.value)


def test_a_submitted_error_is_raised_with_its_message(native):
    run, query = first_query(native)
    run.submit(
        query.id,
        native.Outcome.error('42P01', 7, 'acceptance: the table is gone'),
    )
    with pytest.raises(Exception, match='acceptance: the table is gone'):
        run.next()


def test_an_unknown_query_id_is_refused(native):
    run, query = first_query(native)
    other = 'not-an-id' if not isinstance(query.id, str) else query.id + '?'
    table = pa.table({query.columns[0]: pa.array([1], pa.int64())})
    with pytest.raises(Exception):  # noqa: B017 - any error but a panic
        run.submit(other, native.Outcome.data(table))
    with pytest.raises(Exception):  # noqa: B017 - any error but a panic
        run.submit(12345678, native.Outcome.data(table))


@pytest.mark.parametrize('shape', ['no rows', 'wrong column'])
def test_an_unreadable_result_is_an_exception_not_a_panic(native, shape):
    run, query = first_query(native)
    if shape == 'no rows':
        table = pa.table({query.columns[0]: pa.array([], pa.int64())})
    else:
        table = pa.table({'acceptance_other': pa.array([5], pa.int64())})
    with pytest.raises(Exception):  # noqa: B017 - any error but a panic
        run.submit(query.id, native.Outcome.data(table))
        run.next()
