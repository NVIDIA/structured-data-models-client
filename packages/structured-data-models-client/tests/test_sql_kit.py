# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SQL kit from Python: a built-in source's typed request answered by
the statements the kit writes, run on the caller's own DuckDB through the
provider that runs the core's statements today."""

from __future__ import annotations

import json

import pyarrow as pa
import pytest

duckdb = pytest.importorskip('duckdb')

from sdm_client._native import CoreError, SqlKit  # noqa: E402
from sdm_client._rust.providers.duckdb import DuckDBProvider  # noqa: E402


def _encode(value: object) -> bytes:
    return json.dumps(value).encode()


@pytest.fixture
def connection():
    conn = duckdb.connect()
    conn.execute(
        'CREATE TABLE users AS SELECT * FROM (VALUES (1, 25), (2, 40), (3, 35)) '
        'AS t(user_id, age)'
    )
    yield conn
    conn.close()


def _runner(connection, asked):
    provider = DuckDBProvider(connection)

    def run(plan: bytes, request: bytes, keys):
        step = {'plan': json.loads(plan), 'request': json.loads(request)}
        asked.append(step)
        return provider(step, dict(keys))

    return run


def test_a_duckdb_lookup_is_answered_by_the_kit_on_the_caller_s_connection(
    connection,
):
    kit = SqlKit(_encode({'provider': 'sql', 'dialect': 'duckdb'}))
    capabilities = json.loads(kit.capabilities())
    assert 'lookup' in capabilities['optional']
    seeds = pa.table(
        {
            'key': pa.array([3, 1], pa.int64()),
            'batch': pa.array([0, 1], pa.int64()),
        }
    )
    request = {
        'request': 'lookup',
        'table': {'source': ['users']},
        'columns': ['user_id', 'age'],
        'key': ['user_id'],
        'seeds': {
            'attachment': 2,
            'columns': [
                {'name': 'key', 'nullable': True, 'type': {'type': 'int64'}},
                {'name': 'batch', 'nullable': False, 'type': {'type': 'int64'}},
            ],
        },
    }
    asked: list[dict] = []
    answer = kit.answer(
        _encode(request), {2: seeds}, _runner(connection, asked)
    )
    table = pa.table(answer)
    assert len(asked) == 1
    assert asked[0]['plan']['plan'] == 'sql'
    rows = sorted(
        zip(
            table.column('__kumo_batch__').to_pylist(),
            table.column('user_id').to_pylist(),
            table.column('age').to_pylist(),
            strict=True,
        )
    )
    assert rows == [(0, 3, 35), (1, 1, 25)]


def test_a_driver_exception_is_raised_as_it_is(connection):
    kit = SqlKit(_encode({'provider': 'sql', 'dialect': 'duckdb'}))
    request = {
        'request': 'scan',
        'table': {'source': ['users']},
        'columns': ['user_id'],
        'filters': [],
        'limit': None,
    }

    def failing(plan, request, keys):
        raise RuntimeError('the driver failed')

    with pytest.raises(RuntimeError, match='the driver failed'):
        kit.answer(_encode(request), {}, failing)


def test_a_source_the_kit_writes_no_sql_for_is_refused():
    with pytest.raises(CoreError):
        SqlKit(_encode({'provider': 'sql', 'dialect': 'oracle'}))
