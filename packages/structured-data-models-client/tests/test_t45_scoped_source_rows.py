# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Useful provider data stays inside its actual cursor lifetime for conversion."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from decimal import Decimal
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pytest

from sdm_client._rust.providers.duckdb import DuckDBProvider
from sdm_client._rust.providers.postgres import PostgresProvider, ScalarRows


def plan(dialect):
    return {
        'plan': {
            'plan': 'sql',
            'statements': [
                {'dialect': dialect, 'text': 'SELECT exact_rows', 'params': []}
            ],
            'result_conversions': [{'conversion': 'zoned_text_utc'}]
            if dialect == 'postgres'
            else [],
        }
    }


class Cursor:
    def __init__(self, events, result, description=()):
        self.events, self.result, self.description = events, result, description

    def __enter__(self):
        self.events.append('cursor-enter')
        return self

    def __exit__(self, kind, value, traceback):
        self.events.append(('cursor-exit', kind, value))

    def execute(self, text):
        self.events.append(('execute', text))

    def to_arrow_table(self, size):
        self.events.append(('arrow-read', size))
        return self.result

    def fetchall(self):
        self.events.append('rows-read')
        return self.result


@pytest.mark.parametrize('failure', [False, True])
def test_actual_arrow_native_construction_runs_before_cursor_exit(failure):
    events = []
    table = pa.table({'id': pa.array([2**53 + 1, None], type=pa.int64())})
    cursor = Cursor(events, table)
    provider = DuckDBProvider(SimpleNamespace(cursor=lambda: cursor))
    error = ValueError('native source conversion failed')
    try:
        with provider.source_rows(plan('duckdb')) as source:
            assert source is table
            events.append('native-source')
            frame = source.to_pandas(types_mapper=pd.ArrowDtype)
            assert frame['id'].iloc[0] == 2**53 + 1
            if failure:
                raise error
    except ValueError as raised:
        assert failure and raised is error
    assert events[:4] == [
        'cursor-enter',
        ('execute', 'SELECT exact_rows'),
        ('arrow-read', 1_000_000),
        'native-source',
    ]
    assert events[4] == (
        'cursor-exit',
        ValueError if failure else None,
        error if failure else None,
    )


def test_ordinary_arrow_port_uses_same_single_read_without_importing_pandas(
    monkeypatch,
):
    events = []
    table = pa.table({'id': [1, None]})
    provider = DuckDBProvider(
        SimpleNamespace(cursor=lambda: Cursor(events, table))
    )
    monkeypatch.setitem(sys.modules, 'pandas', None)
    assert provider(plan('duckdb')) is table
    assert events == [
        'cursor-enter',
        ('execute', 'SELECT exact_rows'),
        ('arrow-read', 1_000_000),
        ('cursor-exit', None, None),
    ]


@pytest.mark.parametrize(
    'precision,scale', [(38, 0), (76, 41), (90, -129), (None, None)]
)
def test_scalar_source_retains_full_description_and_exact_unboxed_cells(
    precision, scale
):
    events = []
    description = [
        SimpleNamespace(
            name='exact', type_code=1700, precision=precision, scale=scale
        )
    ]
    raw = [(2**53 + 1,), (Decimal('12345678901234567890.123456789'),), (None,)]
    cursor = Cursor(events, raw, description)
    connection = SimpleNamespace(
        cursor=lambda: cursor, autocommit=True, transaction=nullcontext
    )
    provider = PostgresProvider(connection)
    with provider.source_rows(plan('postgres')) as source:
        assert type(source) is ScalarRows
        assert source.description is description and source.rows is raw
        assert source.description[0].precision == precision
        assert source.description[0].scale == scale
        frame = pd.DataFrame(
            source.rows,
            columns=[column.name for column in source.description],
            dtype=object,
        )
        assert frame.iloc[0, 0] == 2**53 + 1
        assert frame.iloc[1, 0] is raw[1][0]
        assert frame.iloc[2, 0] is None
        events.append('native-source')
    assert events == [
        'cursor-enter',
        ('execute', 'SELECT exact_rows'),
        'rows-read',
        'native-source',
        ('cursor-exit', None, None),
    ]


def test_source_context_unwinds_interrupt_without_reclassifying_it_as_driver_failure():
    events = []
    cursor = Cursor(events, pa.table({'id': []}))
    provider = DuckDBProvider(SimpleNamespace(cursor=lambda: cursor))
    interrupt = KeyboardInterrupt('native source cancelled')
    with pytest.raises(KeyboardInterrupt) as raised:
        with provider.source_rows(plan('duckdb')):
            raise interrupt
    assert raised.value is interrupt
    assert events[-1] == ('cursor-exit', KeyboardInterrupt, interrupt)
