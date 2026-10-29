# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Snowflake raw Arrow scopes preserve native reads and ordinary fallbacks."""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pytest
from sdm_connectors.providers import PlanNotRun
from snowflake.connector.errors import NotSupportedError

from sdm_client._rust.native_source_rows import (
    SourceRowsBatch,
    read_source_rows,
)
from sdm_client._rust.providers.snowflake import SnowflakeProvider


def _step(*statements):
    return {
        'id': 4,
        'request': {'request': 'scan'},
        'plan': {
            'plan': 'sql',
            'statements': [
                {'dialect': 'snowflake', 'text': text, 'params': params}
                for text, params in statements
            ],
            'expect': {'expect': 'discover'},
        },
    }


class Connection:
    def __init__(
        self, *, arrow_error=None, execute_error=None, close_error=None
    ):
        self._paramstyle = 'pyformat'
        self.events = []
        self.table = pa.table({'value': [2**53 + 1, None]})
        self.arrow_error = arrow_error
        self.execute_error = execute_error
        self.close_error = close_error
        self.closed = False

    def cursor(self):
        self.events.append(('cursor', self._paramstyle))
        return Cursor(self)


class PublicStyleConnection(Connection):
    @property
    def paramstyle(self):
        return self._paramstyle

    @paramstyle.setter
    def paramstyle(self, value):
        self._paramstyle = value


class Cursor:
    description = [('value',)]

    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        self.connection.events.append(('enter', self.connection._paramstyle))
        return self

    def __exit__(self, error_type, error, traceback):
        if error is not None:
            self.connection.events.append(
                ('unwind', error, self.connection._paramstyle)
            )
        self.close()

    def execute(self, text, parameters=None):
        connection = self.connection
        connection.events.append(
            ('execute', text, parameters, connection._paramstyle)
        )
        if connection.execute_error is not None:
            raise connection.execute_error

    def fetch_arrow_all(self, *, force_return_table):
        connection = self.connection
        connection.events.append(
            ('arrow', force_return_table, connection._paramstyle)
        )
        if connection.arrow_error is not None:
            raise connection.arrow_error
        return connection.table

    def fetchall(self):
        self.connection.events.append(('rows', self.connection._paramstyle))
        return [[2**53 + 1], [None]]

    def close(self):
        connection = self.connection
        connection.events.append(('close', connection._paramstyle))
        connection.closed = True
        if connection.close_error is not None:
            raise connection.close_error


@pytest.mark.parametrize('connection_type', [Connection, PublicStyleConnection])
def test_native_cursor_and_style_scope_encloses_one_execute_and_exact_arrow_read(
    connection_type,
):
    connection = connection_type()
    provider = SnowflakeProvider(connection)
    with provider.source_rows(_step(('SELECT value', [])), {}) as table:
        assert table is connection.table
        assert not connection.closed
        assert connection._paramstyle == 'qmark'
        connection.events.append('consumer')
    assert connection.closed
    assert connection._paramstyle == 'pyformat'
    assert connection.events == [
        ('cursor', 'qmark'),
        ('enter', 'qmark'),
        ('execute', 'SELECT value', None, 'qmark'),
        ('arrow', True, 'qmark'),
        'consumer',
        ('close', 'qmark'),
    ]


def test_native_port_binds_the_actual_core_text_and_scalar_parameters_in_order():
    connection = Connection()
    step = _step(
        ('SELECT ?', [{'param': 'text', 'attachment': 8}]),
        (
            'SELECT ?',
            [{'param': 'value', 'value': {'t': 'utf8', 'v': 'given'}}],
        ),
    )
    keys = {8: pa.table({'keys': ['[[3]]']})}
    with SnowflakeProvider(connection).source_rows(step, keys) as table:
        assert table is connection.table
    assert [event for event in connection.events if event[0] == 'execute'] == [
        ('execute', 'SELECT ?', ('[[3]]',), 'qmark'),
        ('execute', 'SELECT ?', ('given',), 'qmark'),
    ]
    assert [event for event in connection.events if event[0] == 'arrow'] == [
        ('arrow', True, 'qmark')
    ]


def test_native_arrow_not_supported_error_is_preserved_without_scalar_fallback():
    original = NotSupportedError('raw Arrow unavailable')
    connection = Connection(arrow_error=original)
    with pytest.raises(NotSupportedError) as caught:
        with SnowflakeProvider(connection).source_rows(
            _step(('SELECT value', [])), {}
        ):
            pytest.fail('a failed native Arrow read yielded data')
    assert caught.value is original
    assert connection.closed
    assert connection._paramstyle == 'pyformat'
    assert not any(event[0] == 'rows' for event in connection.events)
    assert connection.events[-2:] == [
        ('unwind', original, 'qmark'),
        ('close', 'qmark'),
    ]


def test_ordinary_call_retains_arrow_to_scalar_text_fallback_and_its_old_style_scope():
    connection = Connection(arrow_error=NotSupportedError())
    table = SnowflakeProvider(connection)(_step(('SHOW value', [])), {})
    assert table.to_pydict() == {'value': [str(2**53 + 1), None]}
    assert table.schema.field('value').type == pa.string()
    assert table.schema.field('value').nullable
    assert connection.events == [
        ('cursor', 'pyformat'),
        ('execute', 'SHOW value', None, 'qmark'),
        ('arrow', True, 'pyformat'),
        ('rows', 'pyformat'),
        ('close', 'pyformat'),
    ]
    assert connection._paramstyle == 'pyformat'


def test_real_native_arrow_conversion_runs_after_cursor_close_and_style_restoration():
    connection = Connection()
    step = _step(('SELECT value', []))
    result = read_source_rows(SnowflakeProvider(connection), [step], {4: {}})
    assert isinstance(result, SourceRowsBatch)
    assert result.rows is connection.table

    def actual_type(dtype):
        assert connection.closed
        assert connection._paramstyle == 'pyformat'
        return pd.ArrowDtype(dtype)

    frame = result.rows.to_pandas(types_mapper=actual_type)
    assert frame['value'].iloc[0] == 2**53 + 1
    assert frame['value'].iloc[1] is pd.NA
    assert frame['value'].dtype == pd.ArrowDtype(pa.int64())


@pytest.mark.parametrize('position', ['execute', 'body', 'close'])
def test_native_scope_restores_style_on_exact_failure_and_closes_cursor(
    position,
):
    original = RuntimeError('selected failure')
    connection = Connection(
        execute_error=original if position == 'execute' else None,
        close_error=original if position == 'close' else None,
    )
    with pytest.raises(RuntimeError) as caught:
        with SnowflakeProvider(connection).source_rows(
            _step(('SELECT value', [])), {}
        ):
            if position == 'body':
                raise original
    assert caught.value is original
    assert connection.closed
    assert connection._paramstyle == 'pyformat'


def test_unsupported_plan_fails_before_cursor_creation_or_style_mutation():
    connection = Connection()
    step = _step(('SELECT value', []))
    step['plan']['statements'][0]['dialect'] = 'duckdb'
    with pytest.raises(PlanNotRun):
        with SnowflakeProvider(connection).source_rows(step, {}):
            pytest.fail('an unsupported plan yielded rows')
    assert connection.events == []
    assert connection._paramstyle == 'pyformat'


@pytest.mark.parametrize('error_type', [KeyboardInterrupt, SystemExit])
def test_process_control_propagates_through_native_scope_cleanup(error_type):
    original = error_type()
    connection = Connection(arrow_error=original)
    with pytest.raises(error_type) as caught:
        with SnowflakeProvider(connection).source_rows(
            _step(('SELECT value', [])), {}
        ):
            pytest.fail('process control must propagate')
    assert caught.value is original
    assert connection.closed
    assert connection._paramstyle == 'pyformat'
