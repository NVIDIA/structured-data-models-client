# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Python host's Postgres provider, through the core (T2.3).

Written by T2.3's independent test author before the implementation, from
the design's row ("dialect, psycopg provider (Arrow without pandas)", "key
sets bound as the dialect binds them", "L4 goldens executed", choice 12's
relation forms) and ``test_the_duckdb_provider.py``, whose vectors this
reads: each fetch is planned by the core's data layer for Postgres
(``{'provider': 'sql', 'dialect': 'postgres'}``, under the database's
``public`` schema since T3.7), answered by the provider
over a ``psycopg`` connection to a real Postgres, admitted by the core and
echoed back by the test-only fetch. The data is the DuckDB-built tables the
vectors name, loaded into a database of their own by the contract loader
(``contract/runners/databases.py``), as the cases' are.

- L4 executed on Postgres: every request kind but ``describe`` (whose
  answer is each dialect's own catalog shape, read into the contract's by
  ``sdm-data``; the S1 cases judge it through ``Schema.from_source``), on
  every base of ``contract/request-kinds.json``, is the Arrow schema and
  rows that file records for DuckDB, except a seeded sample, whose rows
  are the engine's own draw: as many distinct keys as asked, each a key
  the table holds; and a key lookup, which Postgres answers as the
  reference's own lookup statement does (``REFERENCE_LOOKUP``, read from
  the reference's committed ``@postgres`` lines): the batch, the row's
  columns, then ``__kumo_row__`` (1, the first row of its batch) and
  ``__kumo_matches__``, the same rows as DuckDB's.
- Key sets bound as the dialect binds them (``P1 d:131``): keys that
  collide as float64 (an unsigned key past int64, 38-digit and scale-18
  decimals, stored as Postgres ``numeric``) each find their own row, which
  the same keys forced to float64 do not; a key set past the control cap
  travels as an attachment.
- Results are complete: a fetch of 5,000 rows is all of them, and an empty
  one keeps its schema.
- Choice 12's relation forms Postgres takes, executed: one part (a dotted
  name is one table, never split on its dot; a double quote in a name
  doubled) and two (schema and table) each read the table their parts
  name and no other.

The provider is ``sdm_client._rust.providers.postgres.PostgresProvider``,
built over a ``psycopg`` connection and called as the DuckDB provider is
(``provider(step, keys)``): the one name this test assumes of the
implementation (:data:`PROVIDER`).

Where ``SDM_POSTGRES_TEST_DSN`` names no server and ``SDM_REQUIRE_POSTGRES``
is not set, the tests that need one have nothing to hold and return without
asserting, as T2.1a's Postgres tests do; with it set, a missing server
fails (H26). Expected to fail
until T2.3's implementation lands: no Postgres provider module, and the
core reports no Postgres dialect.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import uuid
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pytest

from sdm_client._rust import codec
from sdm_client._rust.run import run

ROOT = Path(__file__).resolve().parents[3]
VECTORS = json.loads(
    (ROOT / 'contract/provider-vectors.json').read_text(encoding='utf-8')
)
KINDS = json.loads(
    (ROOT / 'contract/request-kinds.json').read_text(encoding='utf-8')
)
#: The provider under test, module and class.
PROVIDER = ('sdm_client._rust.providers.postgres', 'PostgresProvider')
#: Under the schema each test's database holds its tables in, ``public``,
#: as every Postgres read names one (T3.7): the source's, or its session's
#: current one, read once per run.
POSTGRES = {'provider': 'sql', 'dialect': 'postgres', 'namespace': ['public']}


def _loader() -> Any:
    spec = importlib.util.spec_from_file_location(
        't23_databases', ROOT / 'contract/runners/databases.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _provider(connection: Any) -> Any:
    module, name = PROVIDER
    return getattr(importlib.import_module(module), name)(connection)


class Server:
    r"""A Postgres server's databases, each made for a test and dropped
    after it."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self.made: list[str] = []

    def database(self, statements: list[str]) -> Any:
        r"""A connection to a new database holding the tables DuckDB builds
        from ``statements``, loaded by the contract loader."""
        import psycopg
        from psycopg.conninfo import make_conninfo

        name = f'sdm_t23_{uuid.uuid4().hex[:16]}'
        with psycopg.connect(self.dsn, autocommit=True) as admin:
            admin.execute(f'CREATE DATABASE {name}')
        self.made.append(name)
        scoped = make_conninfo(self.dsn, dbname=name)
        duck = duckdb.connect()
        try:
            for statement in statements:
                duck.execute(statement)
            _loader().load(duck, 'postgres', scoped)
        finally:
            duck.close()
        return psycopg.connect(scoped, autocommit=True)

    def drop(self) -> None:
        import psycopg

        with psycopg.connect(self.dsn, autocommit=True) as admin:
            for name in self.made:
                admin.execute(f'DROP DATABASE IF EXISTS {name} WITH (FORCE)')


@pytest.fixture(scope='module')
def server() -> Iterator[Server | None]:
    dsn = os.environ.get('SDM_POSTGRES_TEST_DSN')
    if not dsn:
        assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
            'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
        )
        yield None
        return
    made = Server(dsn)
    try:
        yield made
    finally:
        made.drop()


@pytest.fixture(scope='module')
def vectors_db(server: Server | None) -> Any:
    r"""``contract/provider-vectors.json``'s tables on Postgres; None
    where no server is named."""
    if server is None:
        return None
    return server.database(VECTORS['setup'])


def _fetch(request: dict[str, Any]) -> dict[str, Any]:
    return {'request': request, 'provider': POSTGRES, 'echo': True}


def _table(finished: Any) -> pa.Table:
    stream = finished.take(finished.result['data'])
    return pa.RecordBatchReader._import_from_c_capsule(
        stream.__arrow_c_stream__()
    ).read_all()


def _admitted(connection: Any, request: dict[str, Any]) -> pa.Table:
    r"""What the core admitted of ``request``'s answer on Postgres."""
    provider = _provider(connection)
    finished = run(
        {'op': 'test', 'test': 'fetch', 'fetch': 1},
        {1: codec.attachment(_fetch(request))},
        provider=provider,
    )
    return _table(finished)


def _keyed(
    connection: Any, request: dict[str, Any], seeds: pa.Table
) -> tuple[pa.Table, dict[str, Any]]:
    r"""A keyed fetch with ``seeds``, through the core: the core attaches
    them to the step as its own key set, the provider binds them and
    answers. Also the step's control size and the key set handed over."""
    from sdm_client._rust.codec import CONTROL_CAP

    provider = _provider(connection)
    seen: dict[str, Any] = {}

    def answer(step: dict[str, Any], keys: Any) -> Any:
        seen['keys'] = {
            attachment: pa.RecordBatchReader._import_from_c_capsule(
                stream.__arrow_c_stream__()
            ).read_all()
            for attachment, stream in keys.items()
        }
        seen['control'] = len(json.dumps(step).encode())
        assert seen['control'] < CONTROL_CAP
        return provider(step, dict(seen['keys']))

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
    finished = run(
        {'op': 'test', 'test': 'keyed', 'fetch': 1, 'keys': 2},
        {1: codec.attachment(_fetch(request)), 2: seeds},
        provider=answer,
    )
    return _table(finished), seen


def _seeds(kind: dict[str, Any], values: list[str]) -> pa.Table:
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


def test_the_core_reports_a_postgres_provider() -> None:
    # The staging rule's marker (T2.1a): the core legs run on Postgres
    # exactly when the core reports it.
    from sdm_client import _native

    assert 'postgres' in _native.PROVIDERS


def test_a_fetch_is_all_its_rows_and_an_empty_one_keeps_its_schema(
    vectors_db: Any,
) -> None:
    if vectors_db is None:
        return
    several = VECTORS['several']
    data = _admitted(vectors_db, several['request'])
    assert data.num_rows == several['rows']
    rows = data.to_pydict()
    assert sorted(rows['n']) == list(range(several['rows']))
    assert all(
        name == f'r{n}' for n, name in zip(rows['n'], rows['name'], strict=True)
    )
    empty = VECTORS['empty']
    data = _admitted(vectors_db, empty['request'])
    assert data.num_rows == 0
    names = {'int64': pa.int64(), 'utf8': pa.string()}
    assert data.schema == pa.schema(
        [(name, names[kind]) for name, kind in empty['fields']]
    )


def _each_seed_finds_its_own_row(data: pa.Table, labels: list[str]) -> bool:
    found: dict[int, str] = {}
    for row in data.to_pylist():
        if row['__kumo_matches__'] != 1 or row['__kumo_batch__'] in found:
            return False
        found[row['__kumo_batch__']] = row['label']
    return [found.get(i) for i in range(len(labels))] == labels


def test_keys_that_collide_as_float64_bind_join_and_answer_exactly(
    vectors_db: Any,
) -> None:
    if vectors_db is None:
        return
    cases = VECTORS['keys']['cases']
    assert len(cases) == 3
    for case in cases:
        exact = _seeds(case['type'], case['values'])
        data, seen = _keyed(vectors_db, case['request'], exact)
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
        data, _ = _keyed(vectors_db, case['request'], forced)
        assert not _each_seed_finds_its_own_row(data, case['labels']), case[
            'column'
        ]


def test_a_key_set_past_the_control_cap_travels_as_an_attachment(
    vectors_db: Any,
) -> None:
    if vectors_db is None:
        return
    from sdm_client._rust.codec import CONTROL_CAP

    large = VECTORS['large']
    values = [str(i % large['modulo']) for i in range(large['rows'])]
    seeds = _seeds({'type': 'int64'}, values)
    assert seeds.nbytes > CONTROL_CAP
    data, seen = _keyed(vectors_db, large['request'], seeds)
    assert seen['control'] < CONTROL_CAP
    assert data.num_rows == large['rows']
    for row in data.to_pylist():
        assert row['name'] == f'r{row["__kumo_batch__"] % large["modulo"]}'


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


def _reference_lookups() -> list[tuple[str, list[str], list[str]]]:
    r"""Each key lookup the reference ran on Postgres, from its committed
    ``@postgres`` lines (``contract/reference-env.txt``'s file): the
    statement that keeps the first row of each batch, with its result's
    column names and Postgres types."""
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    commit = next(
        line.split(': ', 1)[1].strip()
        for line in env.splitlines()
        if line.startswith('reference: ')
    )
    found = []
    path = ROOT / f'contract/reference/{commit}.jsonl'
    for line in path.read_text(encoding='utf-8').splitlines():
        observation = json.loads(line)
        if not observation['case'].endswith('@postgres'):
            continue
        capture = (observation.get('driver') or {}).get('capture') or {}
        for statement in capture.get('statements') or []:
            if 'WHERE __KUMO_ROW__ = 1' not in statement['sql']:
                continue
            for result in statement['results']:
                values = result['values']
                found.append(
                    (statement['sql'], values['columns'], values['types'])
                )
    return found


def _helpers(columns: list[str]) -> tuple[list[str], list[str]]:
    r"""The helper columns (``__kumo_*``) before and after a result's
    row columns."""
    before = []
    for name in columns:
        if not name.startswith('__kumo_'):
            break
        before.append(name)
    after = []
    for name in reversed(columns):
        if not name.startswith('__kumo_'):
            break
        after.insert(0, name)
    return before, after


#: The reference's key lookup on Postgres: the helper columns before and
#: after the row's own, the same in every committed lookup, each an int8.
REFERENCE_LOOKUP = (['__kumo_batch__'], ['__kumo_row__', '__kumo_matches__'])


def test_the_reference_s_postgres_lookup_has_that_layout() -> None:
    # Not hand-written: every key lookup the reference ran on Postgres in
    # its committed lines has REFERENCE_LOOKUP's helper columns, as int8,
    # and keeps only the first row of each batch.
    lookups = _reference_lookups()
    assert len(lookups) > 10
    for sql, columns, types in lookups:
        assert _helpers(columns) == REFERENCE_LOOKUP, columns
        typed = dict(zip(columns, types, strict=True))
        assert all(
            typed[name] == 'int8' for part in REFERENCE_LOOKUP for name in part
        ), (columns, types)
        assert 'ROW_NUMBER() OVER (PARTITION BY TMP.__KUMO_BATCH__' in sql


def _as_the_reference_looks_up(fetch: dict[str, Any]) -> dict[str, Any]:
    r"""A lookup fetch of ``contract/request-kinds.json`` with its schema
    and rows in :data:`REFERENCE_LOOKUP`'s layout: the batch, the row's
    columns, ``__kumo_row__`` (1) and ``__kumo_matches__``."""
    before, after = REFERENCE_LOOKUP
    names = [name for name, _ in fetch['schema']]
    typed = dict(fetch['schema'])
    own = [name for name in names if not name.startswith('__kumo_')]
    order = before + own + after
    rows = []
    for row in fetch['rows']:
        cells = dict(zip(names, row, strict=True))
        cells['__kumo_row__'] = '1'
        rows.append([cells[name] for name in order])
    return {
        **fetch,
        'schema': [[name, typed.get(name, 'int64')] for name in order],
        'rows': sorted(
            rows,
            key=lambda row: json.dumps(
                row, separators=(',', ':'), ensure_ascii=False
            ),
        ),
    }


def _base(name: str) -> list[str]:
    r"""The statements of base ``name`` of ``contract/bases.yaml``: the
    plain list items under its name, comments between them skipped."""
    lines = (
        (ROOT / 'contract/bases.yaml').read_text(encoding='utf-8').splitlines()
    )
    out = []
    for line in lines[lines.index(f'  {name}:') + 1 :]:
        if line.startswith('  #'):
            continue
        if not line.startswith('  - '):
            break
        out.append(line[len('  - ') :])
    return out


def test_every_request_kind_runs_on_postgres_as_on_duckdb(
    server: Server | None,
) -> None:
    if server is None:
        return
    # L4 executed on Postgres: every request kind the core plans, but the
    # catalog's, on every base, answered by Postgres and admitted, is the
    # schema and rows contract/request-kinds.json records on DuckDB, a
    # lookup's in the reference's Postgres layout.
    fetches = [
        f for f in KINDS['fetches'] if f['request']['request'] != 'describe'
    ]
    kinds = {fetch['request']['request'] for fetch in fetches}
    assert kinds == {
        'scan',
        'preview',
        'count_rows',
        'key_counts',
        'time_bounds',
        'sample_entities',
        'neighbors',
        'lookup',
    }
    assert {(f['request']['request'], f['base']) for f in fetches} == {
        (kind, base)
        for kind in kinds
        for base in ('shop', 'declared', 'adventureworks')
    }
    connections: dict[str, Any] = {}
    problems = []
    for fetch in fetches:
        if fetch['request']['request'] == 'lookup':
            fetch = _as_the_reference_looks_up(fetch)
        base = fetch['base']
        if base not in connections:
            connections[base] = server.database(_base(base))
        connection = connections[base]
        if 'keys' in fetch:
            seeds = pa.table(
                {
                    'key': pa.array(fetch['keys'], pa.int64()),
                    'batch': pa.array(range(len(fetch['keys'])), pa.int64()),
                }
            )
            data, _ = _keyed(connection, fetch['request'], seeds)
        else:
            data = _admitted(connection, fetch['request'])
        where = f'{base}: {fetch["name"]}'
        schema = [[f.name, str(f.type)] for f in data.schema]
        if schema != fetch['schema']:
            problems.append(f'{where}: schema {schema}, not {fetch["schema"]}')
            continue
        rows = _canonical(data)
        if fetch['request']['request'] == 'sample_entities':
            # The engine's own seeded draw: as many distinct keys as asked,
            # each one the table holds.
            whole = _canonical(
                _admitted(
                    connection,
                    {
                        'request': 'scan',
                        'table': fetch['request']['table'],
                        'columns': fetch['request']['columns'],
                        'filters': fetch['request']['filters'],
                        'limit': None,
                    },
                )
            )
            wanted = min(fetch['request']['n'], len(whole))
            if not (
                len(rows) == wanted
                and len({json.dumps(r) for r in rows}) == wanted
                and all(row in whole for row in rows)
            ):
                problems.append(f'{where}: {rows} is no sample of {whole}')
        elif rows != fetch['rows']:
            problems.append(f'{where}: rows {rows}, not {fetch["rows"]}')
    assert not problems, '\n'.join(problems)


def test_each_relation_form_postgres_takes_reads_the_table_it_names(
    server: Server | None,
) -> None:
    if server is None:
        return
    connection = server.database([])
    for statement in (
        'CREATE TABLE t (v integer PRIMARY KEY)',
        'INSERT INTO t VALUES (1)',
        'CREATE TABLE "a.b" (v integer)',
        'INSERT INTO "a.b" VALUES (2)',
        'CREATE SCHEMA a',
        'CREATE TABLE a.b (v integer)',
        'INSERT INTO a.b VALUES (3)',
        'CREATE SCHEMA s',
        'CREATE TABLE s.t (v integer)',
        'INSERT INTO s.t VALUES (4)',
        'CREATE TABLE "say ""hi""" (v integer)',
        'INSERT INTO "say ""hi""" VALUES (5)',
    ):
        connection.execute(statement)
    for parts, value in [
        (['t'], 1),
        # A dotted name is one table, never split on its dot.
        (['a.b'], 2),
        (['public', 't'], 1),
        (['a', 'b'], 3),
        (['s', 't'], 4),
        # A part holding a double quote, the quote doubled.
        (['say "hi"'], 5),
        (['public', 'say "hi"'], 5),
    ]:
        data = _admitted(
            connection,
            {
                'request': 'scan',
                'table': {'source': parts},
                'columns': ['v'],
                'filters': [],
                'limit': None,
            },
        )
        assert data.column('v').to_pylist() == [value], parts
