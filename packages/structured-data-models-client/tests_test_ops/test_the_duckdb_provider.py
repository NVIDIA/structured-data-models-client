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
being seen by no read, as in the reference. From T1.11c, a key set the core
sends with a step is bound by the provider exactly: keys that collide as
float64 (an unsigned key past int64, 38-digit and scale-18 decimals) each
find their own row, which keys forced to float64 do not (decision H22), and a
key set past the 64 KB control cap travels as an attachment beside a step
under it.
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

    def answer(step: dict[str, Any], keys: Any) -> Any:
        table = provider(step, keys)
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


def _seeds(kind: dict[str, Any], values: list[str]) -> pa.Table:
    r"""A key-set case's seeds: its keys, of the type named, from their
    exact text, and their batch positions."""
    from decimal import Decimal

    if kind['type'] == 'decimal128':
        arrow = pa.decimal128(kind['precision'], kind['scale'])
        keys = pa.array([Decimal(v) for v in values], type=arrow)
    else:
        keys = pa.array(
            [int(v) for v in values], type=pa.type_for_alias(kind['type'])
        )
    return pa.table(
        {'key': keys, 'batch': pa.array(range(len(values)), pa.int64())}
    )


def _keyed(
    connection: Any, request: dict[str, Any], seeds: pa.Table
) -> tuple[pa.Table, dict[str, Any]]:
    r"""A lookup with ``seeds``, through the core: the core attaches them to
    the step as its own key set, the provider binds them and answers, and
    the core echoes what it admitted. Also the step as the provider saw it,
    and the key set it was handed."""
    from sdm_client._rust.codec import CONTROL_CAP

    provider = DuckDBProvider(connection)
    seen: dict[str, Any] = {}

    def answer(step: dict[str, Any], keys: Any) -> Any:
        seen['step'] = step
        seen['keys'] = {
            attachment: pa.RecordBatchReader._import_from_c_capsule(
                stream.__arrow_c_stream__()
            ).read_all()
            for attachment, stream in keys.items()
        }
        seen['control'] = len(json.dumps(step).encode())
        assert seen['control'] < CONTROL_CAP
        return provider(step, {a: t for a, t in seen['keys'].items()})

    kinds = {
        'uint64': {'type': 'uint64'},
        'int64': {'type': 'int64'},
        'double': {'type': 'float64'},
    }
    request = dict(request)
    request['seeds'] = {
        'attachment': 0,
        'columns': [
            {
                'name': field.name,
                'type': kinds.get(str(field.type))
                or {
                    'type': 'decimal128',
                    'precision': field.type.precision,
                    'scale': field.type.scale,
                },
                'nullable': field.nullable,
            }
            for field in seeds.schema
        ],
    }
    fetch = {
        'request': request,
        'provider': {'provider': 'sql', 'dialect': 'duckdb'},
        'echo': True,
    }
    finished = run(
        {'op': 'test', 'test': 'keyed', 'fetch': 1, 'keys': 2},
        {1: codec.attachment(fetch), 2: seeds},
        provider=answer,
    )
    stream = finished.take(finished.result['data'])
    data = pa.RecordBatchReader._import_from_c_capsule(
        stream.__arrow_c_stream__()
    ).read_all()
    return data, seen


def _each_seed_finds_its_own_row(data: pa.Table, labels: list[str]) -> bool:
    found: dict[int, str] = {}
    for row in data.to_pylist():
        if row['__kumo_matches__'] != 1 or row['__kumo_batch__'] in found:
            return False
        found[row['__kumo_batch__']] = row['label']
    return [found.get(i) for i in range(len(labels))] == labels


def test_keys_that_collide_as_float64_bind_join_and_answer_exactly() -> None:
    connection = _database()
    cases = VECTORS['keys']['cases']
    assert len(cases) == 3
    for case in cases:
        exact = _seeds(case['type'], case['values'])
        data, seen = _keyed(connection, case['request'], exact)
        # The provider was handed the keys exactly as sent.
        [handed] = seen['keys'].values()
        assert handed.equals(exact), case['column']
        assert _each_seed_finds_its_own_row(data, case['labels']), case[
            'column'
        ]
        # The values collide as float64, and keys forced to it fail.
        forced = exact.set_column(
            0, 'key', exact.column('key').cast(pa.float64(), safe=False)
        )
        assert len(set(forced.column('key').to_pylist())) < len(case['values'])
        data, _ = _keyed(connection, case['request'], forced)
        assert not _each_seed_finds_its_own_row(data, case['labels']), case[
            'column'
        ]


def test_a_key_set_past_the_control_cap_travels_as_an_attachment() -> None:
    from sdm_client._rust.codec import CONTROL_CAP

    large = VECTORS['large']
    values = [str(i % large['modulo']) for i in range(large['rows'])]
    seeds = _seeds({'type': 'int64'}, values)
    assert seeds.nbytes > CONTROL_CAP
    data, seen = _keyed(_database(), large['request'], seeds)
    assert seen['control'] < CONTROL_CAP
    assert data.num_rows == large['rows']
    for row in data.to_pylist():
        assert row['name'] == f'r{row["__kumo_batch__"] % large["modulo"]}'


KINDS = json.loads(
    (
        Path(__file__).resolve().parents[3] / 'contract/request-kinds.json'
    ).read_text(encoding='utf-8')
)


def _cell(value: Any) -> Any:
    r"""A cell as ``contract/request-kinds.json`` writes one."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, list):
        return [_cell(item) for item in value]
    raise AssertionError(f'a cell of {type(value).__name__}')


def _canonical(table: pa.Table) -> list[list[Any]]:
    columns = []
    for field, column in zip(table.schema, table.columns, strict=True):
        if pa.types.is_timestamp(field.type):
            column = column.cast(pa.int64())
        elif pa.types.is_date32(field.type):
            column = column.cast(pa.int32())
        columns.append([_cell(value) for value in column.to_pylist()])
    rows = [list(row) for row in zip(*columns, strict=True)]
    return sorted(
        rows,
        key=lambda row: json.dumps(
            row, separators=(',', ':'), ensure_ascii=False
        ),
    )


def _base(name: str) -> Any:
    r"""A connection holding base ``name`` of ``contract/bases.yaml``, read
    without a YAML library (this job installs none): its statements are the
    plain list items under its name, comments between them skipped."""
    lines = (
        (Path(__file__).resolve().parents[3] / 'contract/bases.yaml')
        .read_text(encoding='utf-8')
        .splitlines()
    )
    connection = duckdb.connect()
    for line in lines[lines.index(f'  {name}:') + 1 :]:
        if line.startswith('  #'):
            continue
        if not line.startswith('  - '):
            break
        connection.execute(line[len('  - ') :])
    return connection


def test_every_request_kind_runs_on_every_base_as_the_vectors_say() -> None:
    # T1.23: every data request kind, planned by the core's data layer for
    # DuckDB, answered by this host's provider and admitted, is the Arrow
    # schema and rows contract/request-kinds.json records, which the
    # pure-Rust host's provider is held to as well.
    fetches = KINDS['fetches']
    kinds = {fetch['request']['request'] for fetch in fetches}
    assert kinds == {
        'describe',
        'scan',
        'preview',
        'count_rows',
        'key_counts',
        'time_bounds',
        'sample_entities',
        'neighbors',
        'lookup',
    }
    # Every kind on every base, not only every kind and every base.
    assert {(f['request']['request'], f['base']) for f in fetches} == {
        (kind, base)
        for kind in kinds
        for base in ('shop', 'declared', 'adventureworks')
    }
    connections: dict[str, Any] = {}
    for fetch in fetches:
        connection = connections.setdefault(fetch['base'], _base(fetch['base']))
        if 'keys' in fetch:
            seeds = pa.table(
                {
                    'key': pa.array(fetch['keys'], pa.int64()),
                    'batch': pa.array(range(len(fetch['keys'])), pa.int64()),
                }
            )
            data, _ = _keyed(connection, fetch['request'], seeds)
        else:
            data, _ = _admitted(connection, fetch['request'])
        where = f'{fetch["base"]}: {fetch["name"]}'
        assert [[f.name, str(f.type)] for f in data.schema] == fetch[
            'schema'
        ], where
        assert _canonical(data) == fetch['rows'], where
    # Not vacuous: rows on every base, a key set bound on each.
    assert all(
        any(f['rows'] for f in fetches if f['base'] == base)
        for base in connections
    )
    assert sum(1 for f in fetches if 'keys' in f and f['rows']) == 6
