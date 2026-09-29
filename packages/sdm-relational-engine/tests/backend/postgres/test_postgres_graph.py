# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import uuid
from collections.abc import Generator
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sdm_relational_engine.rfm import Graph, KumoRelational

try:
    import psycopg
    from sdm_connectors import connect, read
    from sdm_relational_engine.rfm.backend.postgres import PostgresSampler
except ImportError:
    if os.getenv('SDM_REQUIRE_POSTGRES'):
        raise
    pytest.skip("'postgres' extension not installed", allow_module_level=True)


class _RecordingCursor:
    def __init__(self, cursor: Any, statements: list[str]) -> None:
        self._cursor = cursor
        self._statements = statements

    def __enter__(self) -> _RecordingCursor:
        self._cursor.__enter__()
        return self

    def __exit__(self, *args: Any) -> Any:
        return self._cursor.__exit__(*args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)

    def execute(self, statement: str, *args: Any, **kwargs: Any) -> Any:
        self._statements.append(statement)
        self._cursor.execute(statement, *args, **kwargs)
        return self


class _RecordingConnection:
    def __init__(self, connection: Any) -> None:
        self._connection = connection
        self.statements: list[str] = []

    def cursor(self, *args: Any, **kwargs: Any) -> _RecordingCursor:
        return _RecordingCursor(
            self._connection.cursor(*args, **kwargs), self.statements
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)


@pytest.fixture()
def postgres_schema(
    postgres_dsn: str,
) -> Generator[tuple[str, str], None, None]:
    r"""A disposable integration schema when a PostgreSQL DSN is supplied."""
    dsn = postgres_dsn
    schema = f'sdm_test_{uuid.uuid4().hex}'
    connection = connect('postgres', dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f'CREATE SCHEMA "{schema}"')
            cursor.execute(
                f'CREATE TABLE "{schema}".customers ('
                'customer_id BIGINT PRIMARY KEY, name TEXT NOT NULL)'
            )
            cursor.execute(
                f'CREATE TABLE "{schema}".orders ('
                'order_id BIGINT PRIMARY KEY, '
                'customer_id BIGINT NOT NULL REFERENCES '
                f'"{schema}".customers(customer_id), '
                'ordered_at TIMESTAMP NOT NULL)'
            )
            cursor.execute(
                f'CREATE INDEX orders_customer_time_idx ON "{schema}".orders '
                '(customer_id, ordered_at DESC, order_id)'
            )
            cursor.execute(
                f'CREATE TABLE "{schema}".uuid_customers ('
                'customer_id UUID PRIMARY KEY, name TEXT NOT NULL)'
            )
            cursor.execute(
                f'CREATE TABLE "{schema}".uuid_events ('
                'event_id UUID PRIMARY KEY, '
                'customer_id UUID NOT NULL REFERENCES '
                f'"{schema}".uuid_customers(customer_id), '
                'occurred_at TIMESTAMPTZ NOT NULL, amount NUMERIC(30, 10))'
            )
            cursor.execute(
                f'CREATE INDEX uuid_events_customer_time_idx ON '
                f'"{schema}".uuid_events '
                '(customer_id, occurred_at DESC, event_id)'
            )
            cursor.execute(
                f'CREATE TABLE "{schema}".numeric_customers ('
                'customer_id NUMERIC(30, 10) PRIMARY KEY, name TEXT NOT NULL)'
            )
            cursor.execute(
                f'CREATE TABLE "{schema}".numeric_events ('
                'event_id BIGINT PRIMARY KEY, '
                'customer_id NUMERIC(30, 10) NOT NULL REFERENCES '
                f'"{schema}".numeric_customers(customer_id))'
            )
            cursor.execute(
                f'INSERT INTO "{schema}".customers '
                "SELECT value, 'customer-' || value "
                'FROM generate_series(1, 20) AS value'
            )
            cursor.execute(
                f'INSERT INTO "{schema}".orders '
                'SELECT value, 1 + (value % 20), '
                "TIMESTAMP '2025-01-01' + value * INTERVAL '1 day' "
                'FROM generate_series(1, 1000) AS value'
            )
            cursor.execute(
                f'INSERT INTO "{schema}".orders VALUES '
                "(1001, 1, TIMESTAMP '2010-01-01'), "
                "(1002, 1, TIMESTAMP '2011-01-01')"
            )
            customer_id = uuid.UUID('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa')
            cursor.execute(
                f'INSERT INTO "{schema}".uuid_customers VALUES (%s, %s)',
                (customer_id, 'uuid-customer'),
            )
            event_rows = [
                (
                    uuid.UUID(f'00000000-0000-0000-0000-{number:012d}'),
                    customer_id,
                    timestamp,
                    '12345678901234567890.1234567890',
                )
                for number, timestamp in (
                    (1, '2025-01-02T03:04:05.123455+00:00'),
                    (2, '2025-01-02T03:04:05.123456+00:00'),
                    (3, '2025-01-02T03:04:05.123456+00:00'),
                    (4, '2025-01-02T03:04:05.123457+00:00'),
                )
            ]
            cursor.executemany(
                f'INSERT INTO "{schema}".uuid_events VALUES (%s, %s, %s, %s)',
                event_rows,
            )
            exact_numeric_id = '12345678901234567890.1234567890'
            cursor.execute(
                f'INSERT INTO "{schema}".numeric_customers VALUES (%s, %s)',
                (exact_numeric_id, 'numeric-customer'),
            )
            cursor.execute(
                f'INSERT INTO "{schema}".numeric_events VALUES (%s, %s)',
                (1, exact_numeric_id),
            )
        connection.commit()
    finally:
        connection.close()

    try:
        yield dsn, schema
    finally:
        connection = connect('postgres', dsn)
        try:
            with connection.cursor() as cursor:
                cursor.execute(f'DROP SCHEMA "{schema}" CASCADE')
            connection.commit()
        finally:
            connection.close()


def test_from_postgres_discovers_constraints_and_pushes_down_sampling(
    postgres_schema: tuple[str, str],
) -> None:
    dsn, schema = postgres_schema
    flat_table = read(
        'postgres',
        conninfo=dsn,
        table=f'{schema}.customers',
    )
    assert len(flat_table) == 20

    graph = Graph.from_postgres(
        connection=dsn,
        schema=schema,
        verbose=False,
    )
    try:
        assert graph.backend == 'postgres'
        assert set(graph.tables) == {
            'customers',
            'numeric_customers',
            'numeric_events',
            'orders',
            'uuid_customers',
            'uuid_events',
        }
        assert len(graph.edges) == 3

        recording = _RecordingConnection(graph._connection)
        for table in graph.tables.values():
            table._connection = recording
        sampler = PostgresSampler(graph, verbose=False)
        columns = {column.name for column in graph['customers'].columns}
        sample = sampler._sample_entity_table(
            'customers', columns, num_rows=7, random_seed=42
        )

        assert len(sample) == 7
        sampling_sql = [
            sql for sql in recording.statements if 'ORDER BY random()' in sql
        ]
        assert len(sampling_sql) == 1
        assert 'LIMIT 7' in sampling_sql[0]
    finally:
        assert graph._connection is not None
        graph._connection.close()
        graph._connection = None


def test_postgres_sampler_preserves_native_keys_and_exact_time_semantics(
    postgres_schema: tuple[str, str],
) -> None:
    dsn, schema = postgres_schema
    graph = Graph.from_postgres(
        connection=dsn,
        schema=schema,
        verbose=False,
    )
    try:
        graph['orders'].time_column = 'ordered_at'
        graph['uuid_events'].time_column = 'occurred_at'
        recording = _RecordingConnection(graph._connection)
        for table in graph.tables.values():
            table._connection = recording
        sampler = PostgresSampler(graph, verbose=False)

        customer_id = 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'
        missing_id = 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb'
        entity = sampler._sample_entity_table(
            'uuid_customers',
            {'customer_id', 'name'},
            num_rows=1,
            entity_ids=[customer_id, customer_id, missing_id],
        )
        assert entity['customer_id'].astype(str).tolist() == [
            customer_id,
            customer_id,
        ]

        by_pkey, pkey_batch = sampler._by_pkey(
            'uuid_customers',
            pd.Series([customer_id, customer_id, missing_id]),
            {'customer_id', 'name'},
        )
        assert by_pkey['customer_id'].astype(str).tolist() == [
            customer_id,
            customer_id,
        ]
        assert pkey_batch.tolist() == [0, 1]

        exact_numeric_id = '12345678901234567890.1234567890'
        assert graph['numeric_customers']['customer_id'].dtype.value == 'string'
        numeric_entity = sampler._sample_entity_table(
            'numeric_customers',
            {'customer_id', 'name'},
            num_rows=1,
            entity_ids=[exact_numeric_id],
        )
        assert numeric_entity['customer_id'].tolist() == [exact_numeric_id]
        numeric_neighbors, numeric_batch = sampler._by_fkey(
            'numeric_events',
            'customer_id',
            pd.Series([exact_numeric_id]),
            num_neighbors=1,
            anchor_time=None,
            columns={'event_id', 'customer_id'},
        )
        assert numeric_neighbors['customer_id'].tolist() == [exact_numeric_id]
        assert numeric_batch.tolist() == [0]

        old_neighbors, old_batch = sampler._by_fkey(
            'orders',
            'customer_id',
            pd.Series([1]),
            num_neighbors=2,
            anchor_time=pd.Series([pd.Timestamp('2012-01-01')]),
            columns={'order_id', 'customer_id', 'ordered_at'},
        )
        assert old_neighbors['order_id'].tolist() == [1002, 1001]
        assert old_batch.tolist() == [0, 0]

        anchor = pd.Timestamp('2025-01-02T03:04:05.123456+00:00')
        latest, latest_batch = sampler._by_fkey(
            'uuid_events',
            'customer_id',
            pd.Series([customer_id]),
            num_neighbors=2,
            anchor_time=pd.Series([anchor]),
            columns={'event_id', 'customer_id', 'occurred_at'},
        )
        assert latest['event_id'].astype(str).tolist() == [
            '00000000-0000-0000-0000-000000000002',
            '00000000-0000-0000-0000-000000000003',
        ]
        assert latest_batch.tolist() == [0, 0]

        bounded, bounded_batch = sampler._by_time(
            'uuid_events',
            'customer_id',
            pd.Series([customer_id]),
            anchor_time=pd.Series([anchor]),
            min_offset=pd.DateOffset(microseconds=-1),
            max_offset=pd.DateOffset(microseconds=0),
            columns={'event_id', 'customer_id', 'occurred_at'},
        )
        assert bounded['event_id'].astype(str).tolist() == [
            '00000000-0000-0000-0000-000000000002',
            '00000000-0000-0000-0000-000000000003',
        ]
        assert bounded_batch.tolist() == [0, 0]

        open_start, open_batch = sampler._by_time(
            'uuid_events',
            'customer_id',
            pd.Series([customer_id]),
            anchor_time=pd.Series([anchor]),
            min_offset=None,
            max_offset=pd.DateOffset(microseconds=0),
            columns={'event_id'},
        )
        assert open_start['event_id'].astype(str).tolist() == [
            '00000000-0000-0000-0000-000000000001',
            '00000000-0000-0000-0000-000000000002',
            '00000000-0000-0000-0000-000000000003',
        ]
        assert np.array_equal(open_batch, np.zeros(3, dtype=np.int64))

        pkey_sql = [
            sql
            for sql in recording.statements
            if 'WHERE __KUMO_ROW__ = 1' in sql
        ]
        assert len(pkey_sql) == 1

        neighbor_sql = [
            sql for sql in recording.statements if 'JOIN LATERAL' in sql
        ]
        assert len(neighbor_sql) == 3
        assert all('LIMIT ' in sql for sql in neighbor_sql)
        assert all('__KUMO_START_TIME__' not in sql for sql in neighbor_sql)
        assert sum('LIMIT 2' in sql for sql in neighbor_sql) == 2
        assert sum('LIMIT 1' in sql for sql in neighbor_sql) == 1

        window_sql = [
            sql
            for sql in recording.statements
            if 'ORDER BY TMP.__KUMO_BATCH__' in sql
            and 'JOIN LATERAL' not in sql
            and '__KUMO_END_TIME__' in sql
        ]
        assert len(window_sql) == 2
        assert all(
            'AND "occurred_at" <= TMP.__KUMO_END_TIME__' in sql
            for sql in window_sql
        )
        assert sum('__KUMO_START_TIME__' in sql for sql in window_sql) == 1
    finally:
        assert graph._connection is not None
        graph._connection.close()
        graph._connection = None


def test_seeded_postgres_sampling_repeats_across_connections(
    postgres_schema: tuple[str, str],
) -> None:
    dsn, schema = postgres_schema

    def sample(seed: int) -> list[int]:
        graph = Graph.from_postgres(
            connection=dsn,
            schema=schema,
            tables=['customers'],
            verbose=False,
        )
        try:
            sampler = PostgresSampler(graph, verbose=False)
            frame = sampler._sample_entity_table(
                'customers', {'customer_id'}, 10, random_seed=seed
            )
            return frame['customer_id'].tolist()
        finally:
            assert graph._connection is not None
            graph._connection.close()
            graph._connection = None

    assert sample(42) == sample(42)
    assert sample(42) != sample(2_000_043)


def test_composite_postgres_constraints_fail_loudly(
    postgres_schema: tuple[str, str],
) -> None:
    dsn, schema = postgres_schema
    connection = connect('postgres', dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f'CREATE TABLE "{schema}".composite_parent ('
                'left_id BIGINT, right_id BIGINT, '
                'PRIMARY KEY (left_id, right_id))'
            )
            cursor.execute(
                f'CREATE TABLE "{schema}".unique_parent ('
                'id BIGINT PRIMARY KEY, left_id BIGINT, right_id BIGINT, '
                'UNIQUE (left_id, right_id))'
            )
            cursor.execute(
                f'CREATE TABLE "{schema}".composite_child ('
                'id BIGINT PRIMARY KEY, left_id BIGINT, right_id BIGINT, '
                'FOREIGN KEY (left_id, right_id) REFERENCES '
                f'"{schema}".unique_parent(left_id, right_id))'
            )
            cursor.execute(
                f'INSERT INTO "{schema}".composite_parent VALUES (1, 1)'
            )
            cursor.execute(
                f'INSERT INTO "{schema}".unique_parent VALUES (1, 1, 1)'
            )
            cursor.execute(
                f'INSERT INTO "{schema}".composite_child VALUES (1, 1, 1)'
            )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(Exception, match='Composite primary key constraint'):
        Graph.from_postgres(
            connection=dsn,
            schema=schema,
            tables=['composite_parent'],
            verbose=False,
        )

    with pytest.raises(Exception, match='Composite foreign key constraint'):
        Graph.from_postgres(
            connection=dsn,
            schema=schema,
            tables=['unique_parent', 'composite_child'],
            verbose=False,
        )


_COUNT_ORDERS = (
    'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH customers.customer_id'
)


def _materialize(graph: Graph, query: str) -> None:
    model = KumoRelational(graph, verbose=False)
    task = model._get_task_table(
        model._parse_query(query), indices=[1, 2], random_seed=0
    )
    model.materialize_task(task, random_seed=0, verbose=False)


def test_reading_leaves_no_transaction_or_locks_behind(
    postgres_schema: tuple[str, str],
) -> None:
    """Without autocommit, psycopg keeps the first read's transaction, with its
    locks, until a commit, and one failed read then refuses every later one.
    """
    dsn, schema = postgres_schema
    connection = connect('postgres', dsn)
    other = connect('postgres', dsn, autocommit=True)
    try:
        graph = Graph.from_postgres(
            connection,
            tables=['customers', 'orders'],
            schema=schema,
            verbose=False,
        )
        _materialize(graph, _COUNT_ORDERS)
        idle = psycopg.pq.TransactionStatus.IDLE
        assert connection.info.transaction_status == idle
        # A read left open would hold its locks and block this writer.
        other.execute("SET lock_timeout = '2s'")
        other.execute(f'ALTER TABLE "{schema}".orders ADD COLUMN extra INT')

        other.execute(f'ALTER TABLE "{schema}".orders RENAME TO gone')
        with pytest.raises(psycopg.errors.UndefinedTable):
            _materialize(graph, _COUNT_ORDERS)
        assert connection.execute('SELECT 1').fetchone() == (1,)
    finally:
        connection.close()
        other.close()


def test_a_keyless_table_with_json_and_xml_columns_samples(
    postgres_schema: tuple[str, str],
) -> None:
    """A keyless table breaks timestamp ties on every scalar column, and json
    and xml read as strings but have no ordering operator.
    """
    dsn, schema = postgres_schema
    connection = connect('postgres', dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f'CREATE DOMAIN "{schema}".payload AS JSON')
            cursor.execute(
                f'CREATE TABLE "{schema}".notes ('
                'customer_id BIGINT NOT NULL REFERENCES '
                f'"{schema}".customers(customer_id), '
                'noted_at TIMESTAMP NOT NULL, body JSON, doc XML, '
                f'meta "{schema}".payload)'
            )
            cursor.execute(
                f'INSERT INTO "{schema}".notes '
                "SELECT 1 + (value % 20), TIMESTAMP '2025-01-01' "
                "+ (value / 5) * INTERVAL '1 day', "
                "json_build_object('n', value), "
                "xmlelement(name n, value), json_build_object('m', value) "
                'FROM generate_series(1, 1000) AS value'
            )
        connection.commit()
        graph = Graph.from_postgres(
            connection,
            tables=['customers', 'notes'],
            schema=schema,
            verbose=False,
        )
        _materialize(
            graph,
            'PREDICT COUNT(notes.*, 0, 30, days) FOR EACH customers.customer_id',
        )
    finally:
        connection.close()
