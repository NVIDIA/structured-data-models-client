# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The recording mode's round-1 findings (T3.2 fix 5, Codex r1 1 to 3):

- Snowflake's session time zone is read with ``SHOW PARAMETERS``, its
  ``value`` column, not a ``CURRENT_SETTING`` Snowflake does not have;
- two identical executions on one cursor or Spark session are two
  statements, each keeping its own result;
- the recording's completeness holds each statement whole (parameters,
  error, every read) and each Arrow result to its read.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pyarrow as pa
import pytest

HERE = Path(__file__).resolve().parent


def _own() -> ModuleType:
    name = 't32_recording_own_fixtures'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, HERE / 'test_t32_recording_own.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


OWN = _own()
run, check = OWN.run, OWN.check


# --- 1. Snowflake's session time zone ------------------------------------------------


class SnowflakeCursor:
    r"""A Snowflake cursor: SHOW PARAMETERS answers as Snowflake documents
    it (key, value, default, level, description, type); CURRENT_SETTING is
    no Snowflake function."""

    def __init__(self) -> None:
        self.description: Any = None
        self._rows: list[tuple[Any, ...]] = []

    def execute(self, sql: str) -> None:
        if 'CURRENT_SETTING' in sql.upper():
            raise RuntimeError(
                'SQL compilation error: Unknown function CURRENT_SETTING'
            )
        assert sql == "SHOW PARAMETERS LIKE 'TIMEZONE' IN SESSION", sql
        names = ('key', 'value', 'default', 'level', 'description', 'type')
        self.description = [(name, 2) for name in names]
        self._rows = [
            ('TIMEZONE', 'Europe/Berlin', 'America/Los_Angeles', 'SESSION',
             'time zone', 'STRING'),
        ]  # fmt: skip

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows

    def fetch_arrow_all(self, force_return_table: bool = False) -> Any:
        raise AssertionError('a SHOW answer is no Arrow result')

    def close(self) -> None:
        pass


class SnowflakeConnection:
    def cursor(self) -> SnowflakeCursor:
        return SnowflakeCursor()


def test_snowflake_s_session_zone_is_read_with_show_parameters() -> None:
    assert (
        run.session_zone('snowflake', SnowflakeConnection()) == 'Europe/Berlin'
    )


# --- 2. identical executions keep their own results -----------------------------------


def test_identical_reads_on_one_cursor_each_keep_their_result() -> None:
    log = run.DriverLog()
    log.stage = 'from_source'
    log.recorder = run.Recorder()
    cursor = run._CapturedDatabricks(
        OWN.FakeCursor({'SELECT 1': pa.table({'x': [1]})}), log
    )
    for _ in range(2):
        cursor.execute('SELECT 1')
        cursor.fetchall_arrow()
    assert len(log.statements) == 2
    assert sorted(log.recorder.kept) == [(0, 0), (1, 0)]


def test_identical_reads_on_one_spark_session_each_keep_their_result() -> None:
    class Frame:
        columns = ['a']

        def toArrow(self) -> Any:  # noqa: N802 - Spark's name
            return pa.table({'a': [1]})

    class Session:
        version = '4.0.0'

        def sql(self, query: str, args: Any = None) -> Frame:
            return Frame()

    log = run.DriverLog()
    log.stage = 'from_source'
    log.recorder = run.Recorder()
    session = run._CapturedSpark(Session(), log)
    for _ in range(2):
        session.sql('SELECT 1').toArrow()
    assert len(log.statements) == 2
    assert sorted(log.recorder.kept) == [(0, 0), (1, 0)]


# --- 3. completeness holds each statement whole ----------------------------------------


def _first(line: dict[str, Any]) -> dict[str, Any]:
    first: dict[str, Any] = line['observation']['driver']['capture'][
        'statements'
    ][0]
    return first


def _lost_read(line: dict[str, Any]) -> None:
    first = _first(line)
    first['results'].append(copy.deepcopy(first['results'][0]))


def _lost_parameter(line: dict[str, Any]) -> None:
    _first(line)['parameters'] = [
        {'name': '0', 'type': 'builtins.int', 'value': '1'}
    ]


def _changed_error(line: dict[str, Any]) -> None:
    _first(line)['error'] = 'RuntimeError: refused'


def _other_columns(line: dict[str, Any]) -> None:
    for statement, seen in zip(
        line['statements'],
        line['observation']['driver']['capture']['statements'],
        strict=True,
    ):
        for result, kept in zip(
            statement['results'], seen['results'], strict=True
        ):
            columns = (kept.get('values') or {}).get('columns')
            if columns:
                renamed = [f'{name}_x' for name in columns]
                for read in (result, kept):
                    read['values']['columns'] = renamed
                return


@pytest.mark.parametrize(
    ('fault', 'named'),
    [
        (_lost_read, 'its results'),
        (_lost_parameter, 'its parameters'),
        (_changed_error, 'its error'),
        (_other_columns, 'held'),
    ],
)
def test_a_recording_missing_part_of_a_statement_fails(
    fault: Any, named: str
) -> None:
    case, line, tables = OWN._capture()
    assert check.capture_problems(line, case, tables) == []
    fault(line)
    problems = check.capture_problems(line, case, tables)
    assert any(named in problem for problem in problems), problems
