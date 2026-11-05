# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Java runner preserves provider selection, SQL evidence and endpoint bytes."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / 'contract/runners'))

import run_java  # noqa: E402

CASES = ROOT / 'contract/cases'


def _case(name: str) -> dict[str, Any]:
    for path in (CASES / 's1').glob('*.json'):
        case = json.loads(path.read_text(encoding='utf-8'))
        if case['id'] == name:
            return case
    raise AssertionError(name)


def test_t312_an_s1_case_on_java_s_paths_has_a_leg_on_each_of_its_databases() -> (
    None
):
    case = _case('base.binary')
    for database in ('duckdb', 'sqlite', 'postgres'):
        assert run_java.on_java(case, database), database
    # T3.13 extends this same warehouse-capable case to Java's SQL leg.
    assert run_java.on_java(case, 'databricks-sql')


def test_t312_a_ranking_or_a_forecast_has_no_java_leg() -> None:
    for name in ('rank.top-k', 'forecast.lag-1'):
        case = _case(name)
        assert not any(
            run_java.on_java(case, database)
            for database in ('duckdb', 'sqlite', 'postgres')
        ), name


def test_t312_s0_runs_on_duckdb_s_leg_alone() -> None:
    case = json.loads(
        (CASES / 's0/s0.accept.sum-window.json').read_text(encoding='utf-8')
    )
    assert run_java.on_java(case, 'duckdb')
    assert not run_java.on_java(case, 'sqlite')


def test_t312_duckdb_s_log_counts_a_key_set_kept_as_a_table_once() -> None:
    import pyarrow as pa

    sink = pa.BufferOutputStream()
    table = pa.table({'k': pa.array([1, 2], pa.int64())})
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    ipc = base64.b64encode(sink.getvalue().to_pybytes()).decode()
    keep = 'CREATE TEMPORARY TABLE "keys" AS SELECT * FROM "__sdm_keys_1"'
    events = [
        {'event': 'cursor', 'stage': 'materialize'},
        {
            'event': 'bound',
            'stage': 'materialize',
            'handle': 1,
            'name': 'keys',
            'sql': keep,
            'arrow': ipc,
        },
        {
            'event': 'statement',
            'stage': 'materialize',
            'handle': 1,
            'sql': 'SELECT 1',
        },
        {
            'event': 'answered',
            'stage': 'materialize',
            'arrow': ipc,
            'columns': [{'type': 'BIGINT', 'precision': 0, 'scale': 0}],
        },
    ]
    record = run_java.driver_record(events, ['', keep, 'SELECT 1'], 'duckdb')
    assert record['query_log'] == {
        'statements': 1,
        'bindings': 1,
        'compared': 1,
        'differing': [],
    }
    assert len(record['capture']['bindings']) == 1
    assert (
        record['capture']['statements'][0]['results'][0]['method']
        == 'query_arrow'
    )
    assert record['capture']['statements'][0]['results'][0]['values'][
        'types'
    ] == ['BIGINT']


def test_t312_postgres_columns_are_described_as_psycopg_displays_them() -> None:
    import pyarrow as pa

    table = pa.table({'a': [1], 'b': ['x'], 'c': [1], 'd': [[1]]})
    columns = [
        {'type': 'int4', 'precision': 10, 'scale': 0},
        {'type': 'varchar', 'precision': 3, 'scale': 0},
        {'type': 'numeric', 'precision': 10, 'scale': 2},
        {'type': '_int4', 'precision': 10, 'scale': 0},
    ]
    assert run_java.described(table, columns, 'postgres') == [
        'int4',
        'varchar(3)',
        'numeric(10,2)',
        'int4[]',
    ]
    assert run_java.described(table, columns, 'sqlite') == [
        'int64',
        'string',
        'int64',
        'list<item: int64>',
    ]


def test_t312_a_call_the_endpoint_received_otherwise_fails_the_runner() -> None:
    case = _case('base.binary')
    body = b'{"model": "kumo-relational"}'
    events = [
        {
            'event': 'stepped',
            'stage': 'predict',
            'step': {
                'message': 'step',
                'step': 'call_model',
                'id': 1,
                'body': 4,
            },
        },
        {
            'event': 'taken',
            'stage': 'predict',
            'id': 4,
            'bytes': base64.b64encode(body).decode(),
        },
    ]
    assert run_java.calls_sent(events) == [body]
    run_java.check_sent(
        case,
        events,
        SimpleNamespace(requests=[('POST', '/v1/predictions', body)]),
    )
    # Seeded: the endpoint received a body the core did not write.
    other = SimpleNamespace(
        requests=[('POST', '/v1/predictions', body.replace(b' ', b''))]
    )
    with pytest.raises(run_java.RunnerFailure, match='other bytes'):
        run_java.check_sent(case, events, other)
