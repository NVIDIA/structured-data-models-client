# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T3.20e: keep failures raised while fetching and reject lost reads."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

HERE = Path(__file__).resolve().parent
NAME = 't320e_recording_fixtures'
spec = importlib.util.spec_from_file_location(
    NAME, HERE / 'test_t32_recording_own.py'
)
assert spec is not None and spec.loader is not None
OWN = importlib.util.module_from_spec(spec)
sys.modules[NAME] = OWN
spec.loader.exec_module(OWN)
run, check = OWN.run, OWN.check


@pytest.mark.parametrize(
    ('capture', 'method'),
    [
        (run._CapturedDatabricks, 'fetchall'),
        (run._CapturedDatabricks, 'fetchmany'),
        (run._CapturedDatabricks, 'fetchone'),
        (run._CapturedDatabricks, 'fetchall_arrow'),
        (run._CapturedDatabricks, 'fetchmany_arrow'),
        (run._CapturedSnowflake, 'fetch_arrow_all'),
        (run._CapturedSnowflake, 'fetch_pandas_all'),
        (run._CapturedPsycopg, 'fetchall'),
    ],
)
def test_a_fetch_failure_is_kept_on_its_executed_statement(
    capture: Any, method: str
) -> None:
    # The live Databricks FK query executes, then fetchall raises
    # ArrowInvalid for duplicate field names; the old client catches it.
    failure = pa.ArrowInvalid('Multiple matches for FieldRef.Name(table_name)')
    calls: list[str] = []

    class Cursor:
        def execute(self, sql: str) -> Cursor:
            calls.append(sql)
            return self

        def __getattr__(self, name: str) -> Any:
            if name != method:
                raise AttributeError(name)

            def fail(*args: Any, **kwargs: Any) -> Any:
                calls.append(name)
                raise failure

            return fail

    log = run.DriverLog()
    log.recorder = run.Recorder()
    cursor = capture(Cursor(), log)
    cursor.execute('SELECT * FROM information_schema.referential_constraints')
    with pytest.raises(pa.ArrowInvalid) as raised:
        getattr(cursor, method)()
    assert raised.value is failure
    assert calls == [log.statements[0]['sql'], method]
    assert log.statements[0]['error'] == 'ArrowInvalid'
    assert log.statements[0]['results'] == []
    assert log.recorder.kept == {}
    assert log.incomplete == []


def test_a_fetch_failure_without_a_statement_is_incomplete() -> None:
    class Cursor:
        def fetchall(self) -> Any:
            raise pa.ArrowInvalid('no executed statement')

    log = run.DriverLog()
    cursor = run._CapturedDatabricks(Cursor(), log)
    with pytest.raises(pa.ArrowInvalid):
        cursor.fetchall()
    assert log.incomplete == ['fetchall:no_statement']


def _unread(sql: str, leg: str) -> tuple[Any, Any, Any]:
    case, line, tables = OWN._capture()
    if leg != line['leg']:
        line['leg'] = leg
        line['case'] = run.case_key(case['id'], leg)
        line['observation']['case'] = line['case']
        line['versions']['psycopg'] = '3.2.0'
    statement = {
        'handle': 1,
        'stage': 'materialize',
        'sql': sql,
        'parameters': [{'name': '0', 'type': 'builtins.float', 'value': '0.5'}],
        'results': [],
        'error': None,
    }
    line['statements'].append(statement)
    line['observation']['driver']['capture']['statements'].append(
        copy.deepcopy(statement)
    )
    return case, line, tables


@pytest.mark.parametrize('sql', ['SELECT setseed(%s)', ' select SETSEED ( %s ) ; '])
def test_only_the_lakebase_seed_state_change_may_be_unread(sql: str) -> None:
    case, line, tables = _unread(sql, 'lakebase')
    assert check.capture_problems(line, case, tables) == []


@pytest.mark.parametrize(
    ('sql', 'leg'),
    [
        ('SELECT * FROM information_schema.referential_constraints', 'databricks-sql'),
        ('SELECT setseed(%s)', 'databricks-sql'),
        ('SELECT setseed(%s), user_id FROM users', 'lakebase'),
        ('SELECT setseed(%s); SELECT user_id FROM users', 'lakebase'),
        ('SELECT setseed(%s) FROM users', 'lakebase'),
        ('SELECT current_schema()', 'lakebase'),
    ],
)
def test_unread_catalog_data_and_combined_seed_queries_still_fail(
    sql: str, leg: str
) -> None:
    case, line, tables = _unread(sql, leg)
    assert any(
        'a query with no result read' in problem
        for problem in check.capture_problems(line, case, tables)
    )


def test_a_genuine_fetch_failure_passes_full_recording_validation() -> None:
    case, line, tables = _unread(
        'SELECT * FROM information_schema.referential_constraints',
        'databricks-sql',
    )
    for statements in (
        line['statements'],
        line['observation']['driver']['capture']['statements'],
    ):
        statements[-1]['error'] = 'ArrowInvalid'
    assert check.capture_problems(line, case, tables) == []
