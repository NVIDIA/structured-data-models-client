# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Python host's DuckDB provider, through the core (T1.11b).

Each fetch is planned by the core's data layer for DuckDB, answered by the
provider, admitted by the core and echoed back by the test-only fetch, and
held to ``contract/provider-vectors.json``, which the pure-Rust host's
provider tests read too: a result the driver hands over in several batches
is admitted as one attachment of its rows, the same as when handed over
whole; an empty one keeps its schema; and a ``TIMESTAMPTZ`` column reads in
the database zone's wall clock, a zone set for the source's session alone
being seen by no read, as in the reference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa

from sdm_client._rust import codec
from sdm_client._rust.providers.duckdb import DuckDBProvider
from sdm_client._rust.run import run

VECTORS = json.loads(
    (
        Path(__file__).resolve().parents[3] / 'contract/provider-vectors.json'
    ).read_text(encoding='utf-8')
)


def _database(zone: str | None = None) -> Any:
    r"""A database holding the vectors' tables, and the zone ``zone``
    names set."""
    connection = duckdb.connect()
    for statement in VECTORS['setup']:
        connection.execute(statement)
    if zone is not None:
        connection.execute(VECTORS[zone])
    return connection


def _admitted(
    connection: Any, request: dict[str, Any], batch_rows: int = 1_000_000
) -> tuple[pa.Table, list[int]]:
    r"""What the core admitted of ``request``'s answer, and how many batches
    the provider handed over."""
    provider = DuckDBProvider(connection, batch_rows=batch_rows)
    handed: list[int] = []

    def answer(step: dict[str, Any]) -> Any:
        table = provider(step)
        handed.append(len(table.to_batches()))
        return table

    fetch = {
        'request': request,
        'provider': {'provider': 'sql', 'dialect': 'duckdb'},
        'echo': True,
    }
    finished = run(
        {'op': 'test', 'test': 'fetch', 'fetch': 1},
        {1: codec.attachment(fetch)},
        provider=answer,
    )
    stream = finished.take(finished.result['data'])
    data = pa.RecordBatchReader._import_from_c_capsule(
        stream.__arrow_c_stream__()
    ).read_all()
    return data, handed


def test_a_fetch_in_several_batches_is_one_attachment_of_its_rows() -> None:
    connection = _database()
    several = VECTORS['several']
    data, handed = _admitted(
        connection, several['request'], several['batch_rows']
    )
    assert handed == [3]
    whole, handed = _admitted(connection, several['request'])
    assert handed == [1]
    assert data.equals(whole)
    assert data.num_rows == several['rows']
    rows = data.to_pydict()
    assert sorted(rows['n']) == list(range(several['rows']))
    assert all(
        name == f'r{n}' for n, name in zip(rows['n'], rows['name'], strict=True)
    )


def test_an_empty_fetch_keeps_its_schema() -> None:
    empty = VECTORS['empty']
    data, handed = _admitted(_database(), empty['request'])
    assert handed == [0]
    assert data.num_rows == 0
    names = {'int64': pa.int64(), 'utf8': pa.string()}
    assert data.schema == pa.schema(
        [(name, names[kind]) for name, kind in empty['fields']]
    )


def _bounds(zone: str | None) -> list[list[int]]:
    data, _ = _admitted(_database(zone), VECTORS['time_bounds']['request'])
    # Zone-less: a wall clock, as counts of microseconds.
    assert data.schema.field('min_date').type == pa.timestamp('us')
    as_counts = data.cast(
        pa.schema(
            [
                ('table_index', pa.int32()),
                ('min_date', pa.int64()),
                ('max_date', pa.int64()),
            ]
        )
    )
    return [list(row.values()) for row in as_counts.to_pylist()]


def test_a_timestamptz_column_reads_in_the_database_zone_s_wall_clock() -> None:
    assert _bounds('zone') == VECTORS['time_bounds']['rows']
    # A zone set for the source's own session is not one its reads see, as
    # in the reference, whose reads are on cursors too.
    assert _bounds('session_zone') == _bounds(None)
