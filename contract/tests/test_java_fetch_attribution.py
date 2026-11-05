# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Java driver captures preserve fetch-purpose boundaries and SQL logs."""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import pyarrow as pa
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'contract/runners'))
import run_java  # noqa: E402


def step(*purposes):
    return {
        'event': 'stepped',
        'step': {
            'step': 'fetch',
            'requests': [
                {} if purpose is None else {'purpose': purpose}
                for purpose in purposes
            ],
        },
    }


def statement(sql, handle=1):
    return {
        'event': 'statement',
        'stage': 'materialize',
        'handle': handle,
        'sql': sql,
    }


def test_actual_locate_marker_preserves_full_audit_and_filters_only_refresh():
    sink = pa.BufferOutputStream()
    table = pa.table({'key': [1]})
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    events = [
        step('locate'),
        {'event': 'cursor'},
        statement('SELECT current_columns'),
        {
            'event': 'bound',
            'stage': 'materialize',
            'handle': 1,
            'name': 'keys',
            'sql': 'CREATE TEMPORARY TABLE keys AS SELECT 1',
            'arrow': base64.b64encode(sink.getvalue().to_pybytes()).decode(),
        },
        step(None),
        {'event': 'cursor'},
        statement('SELECT actual_rows', 2),
    ]
    logged = [
        'SELECT current_columns',
        'CREATE TEMPORARY TABLE keys AS SELECT 1',
        'SELECT actual_rows',
    ]
    driver = run_java.driver_record(events, logged)
    assert driver['capture']['statements'][0]['purpose'] == 'locate'
    assert driver['capture']['bindings'][0]['purpose'] == 'locate'
    assert 'purpose' not in driver['capture']['statements'][1]
    assert driver['query_log'] == {
        'statements': 2,
        'bindings': 1,
        'compared': 2,
        'differing': [],
    }
    wrong = run_java.driver_record(events, ['SELECT wrong', *logged[1:]])
    assert wrong['query_log']['differing'] == [0]


@pytest.mark.parametrize(
    'reset',
    [
        step(None),
        step('locate', None),
        step('locate', 'other'),
        {'event': 'stepped', 'step': {'step': 'done'}},
        {'event': 'started'},
    ],
)
def test_a_refresh_marker_cannot_leak_to_other_fetches_or_operations(reset):
    events = [
        step('locate'),
        statement('SELECT refresh'),
        reset,
        statement('SELECT actual_rows'),
    ]
    capture = run_java.driver_record(events, None)['capture']
    assert capture['statements'][0]['purpose'] == 'locate'
    assert 'purpose' not in capture['statements'][1]
