# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The Snowflake provider (T3.6) on fake cursors: it runs a fetch step's
statements in order on one cursor of the source's connection, each with
the connection's parameter style switched to ``qmark`` for that statement
and restored after (``SNOW/binding.py:15-38``), and hands back the last
one's result: the connector's Arrow (``fetch_arrow_all(force_return_table=
True)``), or, for a result the connector gives as rows only (``SHOW``,
``DESCRIBE``: ``fetch_arrow_all`` raises ``NotSupportedError``), Arrow the
row adapter builds from ``cursor.description`` and ``fetchall()``: every
column nullable text, in the description's order, each non-null cell its
``str``, an empty result keeping its columns.

A Snowpark session reaches the provider as its own connector connection
(``CN/source.py:367-368``). A driver's errno is the provider error's vendor
code, which the core reads (``2003``, an object not found). The provider
writes no SQL, folds no case and retries nothing.
"""

from __future__ import annotations

import datetime
from typing import Any

import pyarrow as pa
import pytest
from snowflake.connector import SnowflakeConnection
from snowflake.connector.errors import NotSupportedError, ProgrammingError

from sdm_client._rust import run as rust_run
from sdm_client._rust.providers import for_source
from sdm_client._rust.providers.snowflake import SnowflakeProvider
from sdm_connectors import SnowflakeSource
from sdm_connectors.providers import PlanNotRun


class _Cursor:
    r"""A connector cursor: each statement's answer from its connection,
    Arrow for a ``SELECT``, rows only for a ``SHOW`` or ``DESCRIBE``."""

    def __init__(self, connection: _Connection) -> None:
        self._connection = connection
        self._answer: pa.Table | None = None
        self._rows_only = False
        self.description: list[tuple[Any, ...]] | None = None
        self.closed = False

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.closed = True

    def execute(self, command: str, params: Any = None) -> _Cursor:
        connection = self._connection
        connection.calls.append(
            (
                command,
                None if params is None else list(params),
                connection._paramstyle,
            )
        )
        answer = connection.answers[command]
        if isinstance(answer, BaseException):
            raise answer
        self._answer = answer
        self._rows_only = command.startswith(('SHOW', 'DESCRIBE'))
        self.description = [
            (name, 2, None, None, None, None, True)
            for name in answer.column_names
        ]
        return self

    def fetch_arrow_all(
        self, force_return_table: bool = False
    ) -> pa.Table | None:
        assert self._answer is not None
        if self._rows_only:
            raise NotSupportedError
        self._connection.fetched.append(force_return_table)
        if self._answer.num_rows == 0 and not force_return_table:
            return None
        return self._answer

    def fetchall(self) -> list[tuple[Any, ...]]:
        assert self._answer is not None
        columns = [column.to_pylist() for column in self._answer.columns]
        return list(zip(*columns, strict=True))


class _Connection(SnowflakeConnection):
    r"""A connector connection answering each statement by its text."""

    def __init__(self, answers: dict[str, Any]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, list[Any] | None, str]] = []
        self.fetched: list[bool] = []
        self.cursors: list[_Cursor] = []
        self._paramstyle = 'pyformat'

    def cursor(self) -> _Cursor:  # type: ignore[override]
        cursor = _Cursor(self)
        self.cursors.append(cursor)
        return cursor

    def close(self, retry: bool = True) -> None:
        return None


class Session:
    r"""A Snowpark session, recognised by its class's name and module, which
    hands over its own connector connection."""

    __module__ = 'snowflake.snowpark.session'

    def __init__(self, connection: _Connection) -> None:
        self.connection = connection


def _scalar(value: str) -> dict[str, Any]:
    return {'param': 'value', 'value': {'t': 'utf8', 'v': value}}


def _step(*statements: tuple[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        'plan': {
            'plan': 'sql',
            'statements': [
                {'dialect': 'snowflake', 'text': text, 'params': params}
                for text, params in statements
            ],
            'expect': {'expect': 'discover'},
        }
    }


_SELECT = 'SELECT "user_id" FROM "DB"."S"."users" LIMIT 1000'
_DESCRIBE = 'DESCRIBE TABLE "DB"."S"."users"'
_SHOW = 'SHOW TABLES LIKE ? IN SCHEMA "DB"."S"'
_ROWS = 'SELECT "rows" FROM TABLE(RESULT_SCAN(LAST_QUERY_ID())) LIMIT 1'


def _arrow(table: Any) -> pa.Table:
    if isinstance(table, pa.Table):
        return table
    return pa.RecordBatchReader._import_from_c_capsule(
        table.__arrow_c_stream__()
    ).read_all()


def test_a_select_is_answered_with_the_connectors_arrow() -> None:
    rows = pa.table({'user_id': pa.array([0, 1], pa.int8())})
    connection = _Connection({_SELECT: rows})
    answer = _arrow(SnowflakeProvider(connection)(_step((_SELECT, []))))
    assert answer.equals(rows)
    # The whole table, an empty one included.
    assert connection.fetched == [True]
    empty = pa.table({'user_id': pa.array([], pa.int8())})
    connection = _Connection({_SELECT: empty})
    answer = _arrow(SnowflakeProvider(connection)(_step((_SELECT, []))))
    assert answer.schema == empty.schema
    assert answer.num_rows == 0


def test_rows_the_connector_gives_only_as_rows_are_handed_over_as_text() -> (
    None
):
    described = pa.table(
        {
            'name': ['user_id', 'ts'],
            'type': ['NUMBER(38,0)', 'TIMESTAMP_NTZ(6)'],
            'null?': ['N', 'Y'],
            'default': pa.array([None, None], pa.null()),
            'rows': [8, None],
            'created_on': [
                datetime.datetime(2026, 10, 23, 0, 34, 52),
                None,
            ],
            'is_hybrid': [True, False],
        }
    )
    connection = _Connection({_DESCRIBE: described})
    answer = _arrow(SnowflakeProvider(connection)(_step((_DESCRIBE, []))))
    # Every column nullable text, in the description's order.
    assert answer.column_names == described.column_names
    assert all(field.type == pa.string() for field in answer.schema)
    assert all(field.nullable for field in answer.schema)
    assert answer.to_pydict() == {
        'name': ['user_id', 'ts'],
        'type': ['NUMBER(38,0)', 'TIMESTAMP_NTZ(6)'],
        'null?': ['N', 'Y'],
        'default': [None, None],
        'rows': ['8', None],
        'created_on': ['2026-10-23 00:34:52', None],
        'is_hybrid': ['True', 'False'],
    }


def test_an_empty_row_result_keeps_its_columns() -> None:
    nothing = pa.table(
        {
            'created_on': pa.array([], pa.null()),
            'pk_name': pa.array([], pa.null()),
        }
    )
    show = 'SHOW IMPORTED KEYS IN TABLE "DB"."S"."users"'
    connection = _Connection({show: nothing})
    answer = _arrow(SnowflakeProvider(connection)(_step((show, []))))
    assert answer.column_names == ['created_on', 'pk_name']
    assert answer.num_rows == 0
    assert all(field.type == pa.string() for field in answer.schema)


def test_each_statement_is_run_in_order_bound_in_qmark_style_then_restored() -> (
    None
):
    listing = pa.table({'name': ['users'], 'rows': [8]})
    counted = pa.table({'rows': pa.array([8], pa.int8())})
    connection = _Connection({_SHOW: listing, _ROWS: counted})
    answer = _arrow(
        SnowflakeProvider(connection)(
            _step((_SHOW, [_scalar('users')]), (_ROWS, []))
        )
    )
    # The last statement's answer.
    assert answer.equals(counted)
    assert connection.calls == [
        (_SHOW, ['users'], 'qmark'),
        (_ROWS, None, 'qmark'),
    ]
    # The caller's connection is left as it was.
    assert connection._paramstyle == 'pyformat'
    assert all(cursor.closed for cursor in connection.cursors)


def test_the_parameter_style_is_restored_after_a_failure() -> None:
    connection = _Connection(
        {
            _DESCRIBE: ProgrammingError(
                msg='Object does not exist or not authorized.',
                errno=2003,
                sqlstate='02000',
            )
        }
    )
    with pytest.raises(ProgrammingError):
        SnowflakeProvider(connection)(_step((_DESCRIBE, [])))
    assert connection._paramstyle == 'pyformat'
    assert all(cursor.closed for cursor in connection.cursors)


def test_a_drivers_errno_is_the_provider_errors_vendor_code() -> None:
    error = ProgrammingError(
        msg='SQL compilation error: Object \'"DB"."S"."users"\' does not '
        'exist or not authorized.',
        errno=2003,
        sqlstate='02000',
    )
    provided, driver = rust_run._driver(
        error, SnowflakeProvider(_Connection({}))
    )
    assert provided['vendor_code'] == '2003'
    assert provided['sqlstate'] == '02000'
    # The driver's message stays in the host.
    assert 'does not exist' not in repr(provided)
    assert driver == 'snowflake.connector.errors.ProgrammingError'


def test_a_plan_the_provider_does_not_run_is_refused_before_anything_runs() -> (
    None
):
    connection = _Connection({_SELECT: pa.table({'user_id': [0]})})
    other = _step((_SELECT, []))
    other['plan']['statements'][0]['dialect'] = 'databricks'
    with pytest.raises(PlanNotRun):
        SnowflakeProvider(connection)(other)
    with pytest.raises(PlanNotRun):
        SnowflakeProvider(connection)({'plan': {'plan': 'host', 'expect': {}}})
    assert connection.calls == []


def test_a_snowflake_source_names_its_dialect_namespace_and_provider() -> None:
    connection = _Connection({})
    choice, provider = for_source(
        SnowflakeSource(connection, database='SALES', schema='public')
    )
    # The parts as given, never upper-cased (``CN/sql.py:247``).
    assert choice == {
        'provider': 'sql',
        'dialect': 'snowflake',
        'namespace': ['SALES', 'public'],
    }
    assert isinstance(provider, SnowflakeProvider)
    choice, _ = for_source(SnowflakeSource(connection, database='SALES'))
    assert choice['namespace'] == ['SALES', None]
    choice, _ = for_source(SnowflakeSource(connection))
    assert 'namespace' not in choice


def test_a_snowpark_session_is_read_through_its_own_connection() -> None:
    rows = pa.table({'user_id': pa.array([0, 1], pa.int8())})
    connection = _Connection({_SELECT: rows})
    source = SnowflakeSource(Session(connection), database='DB', schema='S')
    assert source.kind == 'snowflake'
    _, provider = for_source(source)
    answer = _arrow(provider(_step((_SELECT, []))))
    assert answer.equals(rows)
    assert connection.calls == [(_SELECT, None, 'qmark')]
