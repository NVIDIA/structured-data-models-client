# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DuckDB's supported versions share public SQL and Arrow ownership contracts."""

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

import duckdb
import pyarrow as pa
import pytest
from sdm_connectors import DuckDBSource, read_table

from sdm_client import KumoRelational, PredictiveQuery, Schema
from sdm_client._rust.providers.duckdb import DuckDBProvider


def plan(*queries, params=()):
    return {
        'plan': {
            'plan': 'sql',
            'statements': [
                {'dialect': 'duckdb', 'text': query, 'params': list(params)}
                for query in queries
            ],
        }
    }


def test_relation_read_preserves_arrow_batches_keys_and_empty_schema():
    connection = duckdb.connect()
    keys = pa.table({'id': pa.array([2**63 + 1, 2**63 + 2], pa.uint64())})
    provider = DuckDBProvider(connection, batch_rows=1)
    params = [
        {'param': 'key_set', 'keys': {'attachment': 1}, 'relation': 'keys'}
    ]
    table = provider(
        plan(
            'CREATE TEMP TABLE input AS SELECT id FROM keys',
            'SELECT id, CAST(id - 9223372036854775808 AS DECIMAL(38, 0)) AS value '
            'FROM input ORDER BY id',
            params=params,
        ),
        {1: keys},
    )
    assert len(table.to_batches()) == 2
    assert table.to_pydict() == {
        'id': [2**63 + 1, 2**63 + 2],
        'value': [Decimal(1), Decimal(2)],
    }
    empty = provider(plan('SELECT NULL::BIGINT AS id WHERE false'))
    assert empty.num_rows == 0
    assert empty.schema == pa.schema([('id', pa.int64())])
    # Both the temporary table and key registration belong to the child cursor.
    with pytest.raises(duckdb.CatalogException):
        connection.execute('SELECT * FROM input')
    connection.close()
    # Returned Arrow buffers remain valid beyond cursor and parent lifetime.
    assert table['id'].to_pylist() == [2**63 + 1, 2**63 + 2]


def test_provider_does_not_use_copying_dbapi_arrow_output():
    connection = duckdb.connect()
    calls = []

    class Cursor:
        def __init__(self):
            self.actual = connection.cursor()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.actual.close()

        def execute(self, query):
            calls.append(('execute', query))
            return self.actual.execute(query)

        def sql(self, query):
            calls.append(('sql', query))
            return self.actual.sql(query)

        def to_arrow_table(self, *_):
            pytest.fail('DB-API Arrow output introduces an extra result copy')

    class Connection:
        def cursor(self):
            return Cursor()

    table = DuckDBProvider(Connection())(
        plan(
            'CREATE TEMP TABLE value AS SELECT 1 AS n',
            'INSERT INTO value VALUES (2)',
            'SELECT n FROM value ORDER BY n',
        )
    )
    assert table.to_pydict() == {'n': [1, 2]}
    assert [kind for kind, _ in calls] == ['execute', 'execute', 'sql']
    connection.close()


def test_cursor_reads_preserve_parent_pending_result_transaction_and_settings():
    with duckdb.connect() as connection:
        connection.execute('CREATE TABLE committed AS SELECT 42 AS n')
        connection.execute('BEGIN')
        connection.execute('CREATE TABLE uncommitted AS SELECT 43 AS n')
        pending = connection.execute('SELECT * FROM range(4) ORDER BY range')
        assert pending.fetchone() == (0,)
        provider = DuckDBProvider(connection)
        assert provider(plan('SELECT * FROM committed')).to_pydict() == {
            'n': [42]
        }
        with pytest.raises(duckdb.CatalogException):
            provider(plan('SELECT * FROM missing'))
        assert pending.fetchall() == [(1,), (2,), (3,)]
        assert connection.execute('SELECT * FROM uncommitted').fetchone() == (
            43,
        )
        connection.execute('ROLLBACK')
        with pytest.raises(duckdb.CatalogException):
            connection.execute('SELECT * FROM uncommitted')
        # Concurrent provider calls each use a separate cursor, on both versions.
        with ThreadPoolExecutor(max_workers=2) as pool:
            tables = list(
                pool.map(
                    lambda _: provider(plan('SELECT * FROM committed')),
                    range(4),
                )
            )
        assert [t.to_pydict() for t in tables] == [{'n': [42]}] * 4


def test_public_catalog_quoted_names_and_materialization_use_the_same_sql():
    with duckdb.connect() as connection:
        connection.execute(
            'CREATE TABLE "items""quoted.name" '
            '(id BIGINT PRIMARY KEY, value DOUBLE, nullable VARCHAR, '
            'precise DECIMAL(38, 18), "at" TIMESTAMPTZ)'
        )
        connection.execute(
            'INSERT INTO "items""quoted.name" SELECT i, i * 2.0, '
            "CASE WHEN i % 2 = 0 THEN NULL ELSE 'text' END, "
            "1.234567890123456789, TIMESTAMPTZ '2024-01-01 12:34:56+00' "
            'FROM range(12) r(i)'
        )
        loaded = read_table(
            connection, query='SELECT * FROM "items""quoted.name" ORDER BY id'
        )
        assert loaded['id'].tolist() == list(range(12))
        schema = Schema.from_source(
            DuckDBSource(connection),
            {'items': 'items"quoted.name'},
            primary_keys={'items': 'id'},
            time_columns={'items': None},
        )
        changed = schema.with_sql_column('items', 'doubled', '"value" * 2')
        changed['items']['doubled'].stype = 'numerical'
        request = PredictiveQuery(
            'PREDICT items.value FOR EACH items.id', changed
        ).materialize(
            indices=[1, 2],
            model=KumoRelational(),
            num_context_samples=4,
            random_seed=7,
        )
        assert len(request.payloads) == 1
        assert (
            len(request.payloads[0]['predict']['instance_table']['rows']) == 2
        )
        assert connection.execute('SELECT 1').fetchone() == (1,)
