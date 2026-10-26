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
from snowflake.connector.errors import NotSupportedError

ROOT = Path(__file__).resolve().parents[2]


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


def test_public_schema_records_successful_rows_only_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _module(
        't310_fallback_public_source_fixtures',
        ROOT
        / 'packages/structured-data-models-client/tests'
        / 'test_snowflake_from_source.py',
    )
    warehouse = fixture._Warehouse('base.binary')
    log = run.DriverLog()
    log.stage = 'from_source'
    captured = run._CapturedSnowflake(warehouse, log)
    schema = fixture._described('core', monkeypatch, captured, **fixture._NAMED)
    assert set(schema.tables) == {'users', 'orders'}
    metadata = [
        statement
        for statement in log.statements
        if statement.get('fetch_errors')
    ]
    assert metadata
    for statement in metadata:
        assert statement['sql'].lstrip().startswith(('SHOW', 'DESCRIBE'))
        assert statement['error'] is None
        assert statement['fetch_errors'] == [
            {'method': 'fetch_arrow_all', 'error': 'NotSupportedError'}
        ]
        assert statement['results'][0]['method'] == 'fetchall'
    assert log.incomplete == []
    assert log.passed == []
    assert any(event['event'] == 'close' for event in log.events)


def test_rows_fallback_clears_only_its_unsupported_arrow_status() -> None:
    cursor, log, driver = _captured()
    _unsupported(cursor)
    assert log.statements[0]['error'] == 'NotSupportedError'
    assert cursor.fetchall() == [('user_id',)]
    statement = log.statements[0]
    assert statement['error'] is None
    assert statement['fetch_errors'] == [
        {'method': 'fetch_arrow_all', 'error': 'NotSupportedError'}
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


def test_public_materialize_keeps_the_reference_large_key_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from decimal import Decimal

    import pandas as pd
    from sdm_client import PredictiveQuery, StructuredDataError

    fixture = _module(
        't310_fallback_public_source_fixtures',
        ROOT
        / 'packages/structured-data-models-client/tests'
        / 'test_snowflake_from_source.py',
    )
    warehouse = fixture._Warehouse('snowflake.key.number-38-0.past-int64')
    schema = fixture._described(
        'core', monkeypatch, warehouse, **fixture._NAMED
    )
    query = PredictiveQuery(
        'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id',
        schema,
    )
    with pytest.raises(StructuredDataError) as raised:
        query.materialize(
            indices=[Decimal(10**25), Decimal(10**25 + 1)],
            anchor_time=pd.Timestamp('2024-03-01'),
        )
    assert raised.value.code == 'INVALID_REQUEST'
    assert raised.value._sentence == 'snowflake_integer_key_range'
    assert 'Snowflake' in str(raised.value)
    assert 'signed 64-bit range' in str(raised.value)
