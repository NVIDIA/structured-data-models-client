# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A successful Snowflake row fallback is not a failed SQL statement."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest


def _module(name: str, path: Path) -> Any:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[name] = loaded
        spec.loader.exec_module(loaded)
    return sys.modules[name]


OWN = _module(
    't310_fallback_recording_fixtures',
    Path(__file__).with_name('test_t32_recording_own.py'),
)
run = OWN.run


class NotSupportedError(Exception):
    """The optional driver's class identity, supplied by a fake driver API."""


@pytest.fixture(autouse=True)
def _snowflake_error_api(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import ModuleType

    errors = ModuleType('snowflake.connector.errors')
    errors.NotSupportedError = NotSupportedError
    monkeypatch.setitem(sys.modules, 'snowflake.connector.errors', errors)


class Cursor:
    description = [('name', 2)]

    def __init__(self) -> None:
        self.execute_error: Exception | None = None
        self.arrow_error: Exception = NotSupportedError('rows only')
        self.row_error: Exception | None = None
        self.calls: list[str] = []

    def execute(self, sql: str) -> Cursor:
        self.calls.append(sql)
        if self.execute_error is not None:
            raise self.execute_error
        return self

    def fetch_arrow_all(self, **kwargs: Any) -> Any:
        self.calls.append('fetch_arrow_all')
        raise self.arrow_error

    def fetchall(self) -> list[tuple[str]]:
        self.calls.append('fetchall')
        if self.row_error is not None:
            raise self.row_error
        return [('user_id',)]

    def fetchone(self) -> tuple[str]:
        self.calls.append('fetchone')
        return ('user_id',)


def _captured() -> tuple[Any, Any, Cursor]:
    driver = Cursor()
    log = run.DriverLog()
    log.stage = 'from_source'
    cursor = run._CapturedSnowflake(driver, log)
    cursor.execute('DESCRIBE TABLE users')
    return cursor, log, driver


def _unsupported(cursor: Any) -> None:
    with pytest.raises(NotSupportedError):
        cursor.fetch_arrow_all(force_return_table=True)


@pytest.mark.parametrize('subclass', [False, True])
def test_rows_fallback_clears_only_its_unsupported_arrow_status(
    subclass: bool,
) -> None:
    cursor, log, driver = _captured()
    if subclass:
        driver.arrow_error = type('RowsOnlySubtype', (NotSupportedError,), {})(
            'rows only'
        )
    expected_class = type(driver.arrow_error).__name__
    _unsupported(cursor)
    assert log.statements[0]['error'] == expected_class
    assert cursor.fetchall() == [('user_id',)]
    statement = log.statements[0]
    assert statement['error'] is None
    assert statement['fetch_errors'] == [
        {'method': 'fetch_arrow_all', 'error': expected_class}
    ]
    assert statement['results'][0]['rows'] == 1
    assert driver.calls == [
        'DESCRIBE TABLE users',
        'fetch_arrow_all',
        'fetchall',
    ]


@pytest.mark.parametrize(
    'failure', [RuntimeError('fetch failed'), pa.ArrowInvalid('bad rows')]
)
def test_fatal_arrow_failure_survives_a_later_read(failure: Exception) -> None:
    cursor, log, driver = _captured()
    driver.arrow_error = failure
    with pytest.raises(type(failure)) as raised:
        cursor.fetch_arrow_all()
    assert raised.value is failure
    cursor.fetchall()
    assert log.statements[0]['error'] == type(failure).__name__
    assert log.statements[0]['fetch_errors'] == [
        {'method': 'fetch_arrow_all', 'error': type(failure).__name__}
    ]


def test_failed_execute_is_not_recovered_using_a_stale_result() -> None:
    cursor, log, driver = _captured()
    driver.execute_error = RuntimeError('execution failed')
    with pytest.raises(RuntimeError):
        cursor.execute('DESCRIBE TABLE missing')
    _unsupported(cursor)
    cursor.fetchall()
    assert log.statements[-1]['error'] == 'RuntimeError'
    assert log.statements[-1]['fetch_errors'] == [
        {'method': 'fetch_arrow_all', 'error': 'NotSupportedError'}
    ]


def test_failed_row_fallback_remains_failed_and_keeps_both_attempts() -> None:
    cursor, log, driver = _captured()
    _unsupported(cursor)
    driver.row_error = RuntimeError('row fetch failed')
    with pytest.raises(RuntimeError):
        cursor.fetchall()
    assert log.statements[0]['error'] == 'RuntimeError'
    assert log.statements[0]['fetch_errors'] == [
        {'method': 'fetch_arrow_all', 'error': 'NotSupportedError'},
        {'method': 'fetchall', 'error': 'RuntimeError'},
    ]


def test_other_successful_read_cannot_later_clear_the_error() -> None:
    cursor, log, _ = _captured()
    _unsupported(cursor)
    cursor.fetchone()
    cursor.fetchall()
    assert log.statements[0]['error'] == 'NotSupportedError'


def test_a_prior_read_prevents_reclassifying_a_later_fetch_failure() -> None:
    cursor, log, _ = _captured()
    cursor.fetchone()
    _unsupported(cursor)
    cursor.fetchall()
    assert log.statements[0]['error'] == 'NotSupportedError'


def test_new_execution_does_not_recover_the_previous_statement() -> None:
    cursor, log, _ = _captured()
    _unsupported(cursor)
    cursor.execute('DESCRIBE TABLE orders')
    cursor.fetchall()
    assert [statement['error'] for statement in log.statements] == [
        'NotSupportedError',
        None,
    ]


def test_another_cursor_cannot_recover_the_failed_cursor() -> None:
    cursor, log, _ = _captured()
    _unsupported(cursor)
    other = run._CapturedSnowflake(Cursor(), log)
    other.execute('DESCRIBE TABLE orders')
    other.fetchall()
    assert [statement['error'] for statement in log.statements] == [
        'NotSupportedError',
        None,
    ]


def test_failed_fetch_evidence_is_checked_as_part_of_the_whole_recording() -> (
    None
):
    import copy

    case, line, tables = OWN._capture()
    evidence = [{'method': 'fetch_arrow_all', 'error': 'NotSupportedError'}]
    line['statements'][0]['fetch_errors'] = copy.deepcopy(evidence)
    captured = line['observation']['driver']['capture']['statements'][0]
    captured['fetch_errors'] = copy.deepcopy(evidence)
    assert OWN.check.capture_problems(line, case, tables) == []
    del line['statements'][0]['fetch_errors']
    assert any(
        'fetch_errors is not what its reference line captured' in problem
        for problem in OWN.check.capture_problems(line, case, tables)
    )


@pytest.mark.parametrize('field', ['method', 'error'])
def test_malformed_failed_fetch_evidence_does_not_validate(field: str) -> None:
    case, line, tables = OWN._capture()
    evidence = {'method': 'fetch_arrow_all', 'error': 'NotSupportedError'}
    del evidence[field]
    line['observation']['driver']['capture']['statements'][0][
        'fetch_errors'
    ] = [evidence]
    assert OWN.check.capture_problems(line, case, tables)


def test_absent_driver_does_not_replace_a_fake_cursor_fetch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    original_import = builtins.__import__

    def without_snowflake(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == 'snowflake' or name.startswith('snowflake.'):
            raise ModuleNotFoundError('optional Snowflake driver is absent')
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, 'snowflake.connector.errors')
    monkeypatch.setattr(builtins, '__import__', without_snowflake)
    cursor, log, driver = _captured()
    failure = pa.ArrowInvalid('the original fake driver failure')
    driver.arrow_error = failure
    with pytest.raises(pa.ArrowInvalid) as raised:
        cursor.fetch_arrow_all()
    assert raised.value is failure
    assert driver.calls == ['DESCRIBE TABLE users', 'fetch_arrow_all']
    assert log.statements[0]['error'] == 'ArrowInvalid'
    assert log.statements[0]['results'] == []
    assert log.statements[0]['fetch_errors'] == [
        {'method': 'fetch_arrow_all', 'error': 'ArrowInvalid'}
    ]


def test_an_unrelated_same_name_exception_is_not_a_supported_fallback() -> None:
    foreign_type = type('NotSupportedError', (RuntimeError,), {})
    failure = foreign_type('not the optional driver class')
    cursor, log, driver = _captured()
    driver.arrow_error = failure
    with pytest.raises(foreign_type) as raised:
        cursor.fetch_arrow_all()
    assert raised.value is failure
    cursor.fetchall()
    assert log.statements[0]['error'] == 'NotSupportedError'
