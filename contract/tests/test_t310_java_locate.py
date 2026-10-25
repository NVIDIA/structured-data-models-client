# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Java's actual fetch-purpose evidence and truthful warehouse cell claims."""

from __future__ import annotations

import base64
import importlib.util
import json
import sys
from pathlib import Path

import pyarrow as pa
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'contract/runners'))
import run_java  # noqa: E402


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


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
    compare = module('t310_java_compare', 'scripts/rust_core/compare.py')
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
    assert compare.counts({'driver': driver}) == {
        'from_source': 0,
        'materialize': 1,
        'predict': 0,
    }
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


def test_java_types_cells_are_witnessed_without_removing_ignored_inputs():
    coverage = module('t310_java_coverage', 'scripts/rust_core/coverage.py')
    case = json.loads(
        (ROOT / 'contract/cases/s2/java.databricks-sql.types.json').read_text()
    )
    captured = json.loads(
        (
            ROOT
            / 'contract/captures/databricks-sql/java.databricks-sql.types/capture.json'
        ).read_text()
    )
    assert set(case['cells']) <= coverage.observed(case, captured['reference'])
    assert any(
        ' AS token' in sql and ' AS tags' in sql
        for sql in case['input']['setup']
    )
