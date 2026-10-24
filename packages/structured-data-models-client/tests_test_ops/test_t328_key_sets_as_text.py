# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Key sets bound by each dialect, sent as attachments (T3.28, design S4).

The core writes a Databricks or Postgres key set as the JSON text its
statement reads, splits it past the old client's budget of 900,000 bytes
(``_chunk_rows``, ``DB/sampler.py:32-44``, ``PG/sampler.py:125-182``), sends
one fetch per chunk in one step, each chunk's text a one-cell Arrow
attachment the fetch's plan names (never in the control message), and joins
the answers in order; any chunk failing fails the request. The providers
bind the text they are handed and write no JSON.

Each test here runs a keyed fetch through the core's test-only operation:
the core attaches the seeds as its own key set, as an operation's body
does, and the run loop hands each fetch's attachments to the provider. On
Databricks the provider is the client's own, over a fake warehouse that
reads each text as ``posexplode(from_json(?, ...))`` does; on Postgres it is
the client's own over a real server, where ``SDM_POSTGRES_TEST_DSN`` names
one.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from sdm_client._rust import codec
from sdm_client._rust.providers.databricks import DatabricksProvider
from sdm_client._rust.providers.postgres import PostgresProvider
from sdm_client._rust.run import run
from sdm_client.errors import StructuredDataError

ROOT = Path(__file__).resolve().parents[3]
VECTORS = json.loads(
    (ROOT / 'contract/provider-vectors.json').read_text(encoding='utf-8')
)
#: The old client's budget of a key set's text, in bytes.
BUDGET = 900_000
BATCH = '__kumo_batch__'


# The old client's chunking (DB/sampler.py:32-44 at 43e55f4f), transcribed
# as the oracle.
def _reference_dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(',', ':'))


def _reference_chunk_rows(rows: list, max_bytes: int) -> list[list]:
    chunks: list[list] = []
    current: list = []
    current_bytes = 2
    for row in rows:
        size = len(_reference_dumps(row)) + 1
        if current and current_bytes + size > max_bytes:
            chunks.append(current)
            current, current_bytes = [], 2
        current.append(row)
        current_bytes += size
    if current:
        chunks.append(current)
    return chunks or [[]]


def _keyed(
    provider: Any,
    dialect: str,
    request: dict[str, Any],
    seeds: pa.Table,
    seen: list[dict[str, Any]],
) -> pa.Table:
    r"""A keyed fetch of ``request`` against ``seeds`` through the core on
    ``dialect``, each fetch the provider answers kept in ``seen``: the
    step's control size and the attachments handed over with it."""
    from sdm_client._rust.codec import CONTROL_CAP

    def answer(step: dict[str, Any], keys: Any) -> Any:
        handed = {
            attachment: pa.RecordBatchReader._import_from_c_capsule(
                stream.__arrow_c_stream__()
            ).read_all()
            for attachment, stream in keys.items()
        }
        control = len(json.dumps(step).encode())
        seen.append({'step': step, 'keys': handed, 'control': control})
        assert control < CONTROL_CAP
        return provider(step, dict(handed))

    request = dict(request)
    request['seeds'] = {
        'attachment': 0,
        'columns': [
            {'name': 'key', 'type': {'type': 'int64'}, 'nullable': False},
            {'name': 'batch', 'type': {'type': 'int64'}, 'nullable': False},
        ],
    }
    fetch = {
        'request': request,
        'provider': {'provider': 'sql', 'dialect': dialect},
        'echo': True,
    }
    finished = run(
        {'op': 'test', 'test': 'keyed', 'fetch': 1, 'keys': 2},
        {1: codec.attachment(fetch), 2: seeds},
        provider=answer,
    )
    stream = finished.take(finished.result['data'])
    return pa.RecordBatchReader._import_from_c_capsule(
        stream.__arrow_c_stream__()
    ).read_all()


def _seeds(keys: list[int]) -> pa.Table:
    return pa.table(
        {
            'key': pa.array(keys, pa.int64()),
            'batch': pa.array(range(len(keys)), pa.int64()),
        }
    )


def _text(handed: dict[int, pa.Table]) -> str:
    r"""The one text a fetch was handed: one attachment, one cell."""
    [table] = handed.values()
    assert table.num_columns == 1 and table.num_rows == 1
    assert pa.types.is_string(table.schema.field(0).type)
    return table.column(0)[0].as_py()


# --- Databricks, over a fake warehouse ---------------------------------------------

LOOKUP = {
    'request': 'lookup',
    'table': {'source': ['main', 'shop', 'users']},
    'key': ['user_id'],
    'columns': ['user_id'],
}
_OFFSET = re.compile(r'\bpos \+ (\d+) AS\b')


class _Cursor:
    def __init__(self, warehouse: _Warehouse) -> None:
        self._warehouse = warehouse
        self._result: Any = None

    def execute(self, operation: str, parameters: Any = None) -> None:
        self._warehouse.executed.append((operation, parameters))
        if len(self._warehouse.executed) == self._warehouse.failing:
            raise self._warehouse.failure
        self._result = _posexplode(operation, parameters)

    def fetchall_arrow(self) -> Any:
        return self._result

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *exc: object) -> None:
        pass


class _Warehouse:
    """A SQL warehouse connection, answering a lookup as its statement's
    ``posexplode(from_json(?, ...))`` reads the text bound: each seed at its
    position plus the offset the statement adds, with its key."""

    def __init__(self, failing: int = 0) -> None:
        self.executed: list[tuple[str, Any]] = []
        self.failing = failing
        self.failure = type(
            'ServerOperationError',
            (Exception,),
            {'__module__': 'databricks.sql.exc'},
        )('[INSUFFICIENT_PERMISSIONS] denied. SQLSTATE: 42501')

    def cursor(self) -> _Cursor:
        return _Cursor(self)


def _posexplode(operation: str, parameters: Any) -> pa.Table:
    seeds = json.loads(parameters[0])
    found = _OFFSET.search(operation)
    offset = int(found.group(1)) if found else 0
    return pa.table(
        {
            BATCH: pa.array(range(offset, offset + len(seeds)), pa.int32()),
            'user_id': pa.array(seeds, pa.int64()),
        }
    )


def test_a_databricks_key_set_past_the_budget_is_one_fetch_per_chunk_joined() -> (
    None
):
    keys = [100_000 + index for index in range(300_000)]
    expected = _reference_chunk_rows(keys, BUDGET)
    assert len(expected) >= 3
    warehouse = _Warehouse()
    seen: list[dict[str, Any]] = []
    data = _keyed(
        DatabricksProvider(warehouse), 'databricks', LOOKUP, _seeds(keys), seen
    )
    # One fetch per chunk, all asked in one step, each handed its own text,
    # the old client's chunk byte for byte.
    assert len(seen) == len(expected)
    sent = [_text(each['keys']) for each in seen]
    assert sent == [_reference_dumps(chunk) for chunk in expected]
    assert all(len(text) <= BUDGET for text in sent)
    assert [parameters for _, parameters in warehouse.executed] == [
        [text] for text in sent
    ]
    # Each chunk's statement moves its seeds by those before it.
    offsets = [_OFFSET.search(operation) for operation, _ in warehouse.executed]
    assert offsets[0] is None
    moved = [int(found.group(1)) for found in offsets[1:] if found]
    assert moved == [
        sum(len(chunk) for chunk in expected[:at])
        for at in range(1, len(expected))
    ]
    # The answers joined in order: every seed at its place.
    assert data.column(BATCH).to_pylist() == list(range(len(keys)))
    assert data.column('user_id').to_pylist() == keys


def test_a_later_databricks_chunk_failing_fails_the_request() -> None:
    keys = [100_000 + index for index in range(300_000)]
    warehouse = _Warehouse(failing=2)
    seen: list[dict[str, Any]] = []
    with pytest.raises(StructuredDataError) as raised:
        _keyed(
            DatabricksProvider(warehouse),
            'databricks',
            LOOKUP,
            _seeds(keys),
            seen,
        )
    assert len(seen) >= 2
    assert raised.value.code == 'PROVIDER_ERROR'
    assert '42501' in repr(vars(raised.value))


# --- Postgres, over a real server ----------------------------------------------------


def _loader() -> Any:
    spec = importlib.util.spec_from_file_location(
        't328_databases', ROOT / 'contract/runners/databases.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def vectors_db() -> Iterator[Any]:
    r"""``contract/provider-vectors.json``'s tables in a Postgres database
    of their own, dropped after; None where no server is named."""
    dsn = os.environ.get('SDM_POSTGRES_TEST_DSN')
    if not dsn:
        assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
            'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
        )
        yield None
        return
    import duckdb
    import psycopg
    from psycopg.conninfo import make_conninfo

    name = f'sdm_t328_{uuid.uuid4().hex[:16]}'
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute(f'CREATE DATABASE {name}')
    try:
        scoped = make_conninfo(dsn, dbname=name)
        duck = duckdb.connect()
        try:
            for statement in VECTORS['setup']:
                duck.execute(statement)
            _loader().load(duck, 'postgres', scoped)
        finally:
            duck.close()
        with psycopg.connect(scoped, autocommit=True) as connection:
            yield connection
    finally:
        with psycopg.connect(dsn, autocommit=True) as admin:
            admin.execute(f'DROP DATABASE IF EXISTS {name} WITH (FORCE)')


def test_a_postgres_key_set_past_the_budget_is_one_fetch_per_chunk_joined(
    vectors_db: Any,
) -> None:
    if vectors_db is None:
        return
    large = VECTORS['large']
    modulo = large['modulo']
    count = 150_000
    keys = [index % modulo for index in range(count)]
    seen: list[dict[str, Any]] = []
    data = _keyed(
        PostgresProvider(vectors_db),
        'postgres',
        large['request'],
        _seeds(keys),
        seen,
    )
    # The rows' cells as text, the batch carried in each, split as the old
    # client splits: one fetch per chunk.
    rows = [[str(key), str(at)] for at, key in enumerate(keys)]
    expected = _reference_chunk_rows(rows, BUDGET)
    assert len(expected) >= 2
    sent = [_text(each['keys']) for each in seen]
    assert sent == [_reference_dumps(chunk) for chunk in expected]
    assert all(len(text.encode()) <= BUDGET for text in sent)
    # The answers joined: every seed found its row.
    assert data.num_rows == count
    for row in data.to_pylist():
        assert row['name'] == f'r{row[BATCH] % modulo}'
    assert sorted(data.column(BATCH).to_pylist()) == list(range(count))
