# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import collections
import os
import uuid
from collections.abc import Generator

import numpy as np
import pandas as pd
import pytest
from sdm_relational_engine.rfm import Graph, KumoRelational

try:
    import psycopg
    from sdm_connectors import read_table
    from sdm_relational_engine.rfm.backend.postgres import PostgresSampler
except ImportError:
    if os.getenv('SDM_REQUIRE_POSTGRES'):
        raise
    pytest.skip("'postgres' extension not installed", allow_module_level=True)


@pytest.fixture()
def postgres_schema(
    postgres_dsn: str,
) -> Generator[tuple[str, str], None, None]:
    r"""A disposable integration schema when a PostgreSQL DSN is supplied."""
    dsn = postgres_dsn
    schema = f'sdm_test_{uuid.uuid4().hex}'
    connection = psycopg.connect(dsn)
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
                'FROM generate_series(1, 100) AS value'
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
        connection = psycopg.connect(dsn)
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
    with psycopg.connect(dsn) as connection:
        flat_table = read_table(connection, table=f'{schema}.customers')
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

        sampler = PostgresSampler(graph, verbose=False)
        columns = {column.name for column in graph['customers'].columns}
        sample = sampler._sample_entity_table(
            'customers', columns, num_rows=7, random_seed=42
        )

        assert len(sample) == 7
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
    connection = psycopg.connect(dsn)
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
    connection = psycopg.connect(dsn)
    other = psycopg.connect(dsn, autocommit=True)
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
    connection = psycopg.connect(dsn)
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


def test_names_that_need_quoting_are_sampled(postgres_dsn: str) -> None:
    """Every statement names a table through its quoted source name, so a
    schema and table needing quotes sample like plain ones.
    """
    schema = f'Sales {uuid.uuid4().hex[:8]}'
    connection = psycopg.connect(postgres_dsn, autocommit=True)
    try:
        connection.execute(f'CREATE SCHEMA "{schema}"')
        connection.execute(
            f'CREATE TABLE "{schema}"."Customers" '
            '(customer_id BIGINT PRIMARY KEY)'
        )
        connection.execute(
            f'CREATE TABLE "{schema}"."Order Items" ('
            'item_id BIGINT PRIMARY KEY, customer_id BIGINT REFERENCES '
            f'"{schema}"."Customers"(customer_id), ordered_at TIMESTAMP)'
        )
        connection.execute(
            f'INSERT INTO "{schema}"."Customers" '
            'SELECT value FROM generate_series(1, 20) AS value'
        )
        connection.execute(
            f'INSERT INTO "{schema}"."Order Items" '
            'SELECT value, 1 + (value % 20), '
            "TIMESTAMP '2025-01-01' + value * INTERVAL '1 day' "
            'FROM generate_series(1, 200) AS value'
        )
        graph = Graph.from_postgres(
            connection,
            tables=[
                dict(name='customers', source_name='Customers'),
                dict(name='items', source_name='Order Items'),
            ],
            schema=schema,
            verbose=False,
        )
        assert [tuple(edge) for edge in graph.edges] == [
            ('items', 'customer_id', 'customers')
        ]
        _materialize(
            graph,
            'PREDICT COUNT(items.*, 0, 30, days) FOR EACH customers.customer_id',
        )

        sampler = PostgresSampler(graph, verbose=False)
        entities = sampler._sample_entity_table(
            'customers', {'customer_id'}, num_rows=5, random_seed=1
        )
        neighbours, _ = sampler._by_fkey(
            'items',
            'customer_id',
            pd.Series([1, 2]),
            num_neighbors=3,
            anchor_time=pd.Series(pd.to_datetime(['2025-06-01'] * 2)),
            columns={'item_id', 'customer_id', 'ordered_at'},
        )
        assert len(entities) == 5
        assert len(neighbours) == 6
    finally:
        connection.execute(f'DROP SCHEMA "{schema}" CASCADE')
        connection.close()


@pytest.mark.parametrize('keyed', [True, False])
def test_the_inference_sample_is_the_same_rows_however_they_are_stored(
    postgres_dsn: str,
    keyed: bool,
) -> None:
    """Types are inferred from a sample of 1,000 rows, which a bare `LIMIT`
    took in whatever order the table was scanned. The same rows stored in
    another order now give the same sample, and a sample of expressions holds
    the rows of the sample of columns, in the same order, even when an
    expression is named like a source column.
    """
    from sdm_relational_engine.rfm.backend.postgres import PostgresTable

    schema = f'sdm_test_{uuid.uuid4().hex}'
    key = 'PRIMARY KEY' if keyed else ''
    values = [(7 * i) % 1500 for i in range(1500)]
    connection = psycopg.connect(postgres_dsn, autocommit=True)
    try:
        connection.execute(f'CREATE SCHEMA "{schema}"')
        samples = []
        for name, stored in (('ahead', values), ('behind', values[::-1])):
            connection.execute(
                f'CREATE TABLE "{schema}".{name} (n BIGINT {key}, m TEXT)'
            )
            with connection.cursor() as cursor:
                cursor.executemany(
                    f'INSERT INTO "{schema}".{name} VALUES (%s, %s)',
                    [(n, f'm{n}') for n in stored],
                )
            table = PostgresTable(
                connection,
                name,
                schema=schema,
                columns=[
                    'n',
                    dict(name='m', expr='CAST(n AS TEXT)', dtype='string'),
                    dict(name='double', expr='2 * n', dtype='int'),
                ],
                primary_key=None,
            )
            source = table._source_sample_df
            expr = table._get_expr_sample_df(
                [table['m'], table['double']]
            ).reset_index(drop=True)
            assert len(source) == len(expr) == 1000
            assert expr['m'].astype(str).tolist() == [
                str(n) for n in source['n']
            ]
            assert (expr['double'] == 2 * source['n']).all()
            samples.append(source['n'].tolist())

        assert samples[0] == samples[1]
        if keyed:  # The lowest keys, read through the key's index.
            assert samples[0] == list(range(1000))
        else:  # Spread over the table by the hash.
            assert 0 < sum(n >= 1000 for n in samples[0]) < 1000
    finally:
        connection.execute(f'DROP SCHEMA "{schema}" CASCADE')
        connection.close()


def test_an_unkeyed_inference_sample_follows_no_session_setting(
    postgres_dsn: str,
) -> None:
    """An unkeyed table's inference sample was ordered by a hash of each
    row's text, which `TimeZone`, `IntervalStyle` and `extra_float_digits`
    format, so two sessions inferred types from different rows. The rows are
    now hashed in their binary form. psycopg cannot read a `TIMESTAMPTZ`
    under a `DateStyle` other than ISO, or an `INTERVAL` under an
    `IntervalStyle` other than `postgres`, so the test sets the other two.
    """
    from sdm_relational_engine.rfm.backend.postgres import PostgresTable

    schema = f'sdm_test_{uuid.uuid4().hex}'
    connection = psycopg.connect(postgres_dsn, autocommit=True)
    try:
        connection.execute(f'CREATE SCHEMA "{schema}"')
        connection.execute(
            f'CREATE TABLE "{schema}".t '
            '(n BIGINT, at TIMESTAMPTZ, f DOUBLE PRECISION, i INTERVAL, '
            'd DATE)'
        )
        connection.execute(
            f'INSERT INTO "{schema}".t '
            "SELECT n, TIMESTAMPTZ '2024-01-01 00:00:00+00' + n * "
            "INTERVAL '37 minutes', n / 7.0, n * INTERVAL '90 seconds', "
            "DATE '2024-01-01' + n FROM generate_series(0, 1499) AS n"
        )
        samples = []
        for settings in (
            (),
            (
                "SET TimeZone = 'Pacific/Kiritimati'",
                'SET extra_float_digits = -3',
            ),
        ):
            session = psycopg.connect(postgres_dsn, autocommit=True)
            try:
                for setting in settings:
                    session.execute(setting)
                table = PostgresTable(
                    session,
                    't',
                    schema=schema,
                    columns=['n', 'at', 'f', 'd'],
                    primary_key=None,
                )
                samples.append(table._source_sample_df['n'].tolist())
            finally:
                session.close()

        assert len(samples[0]) == 1000
        assert samples[0] == samples[1]
    finally:
        connection.execute(f'DROP SCHEMA "{schema}" CASCADE')
        connection.close()


def test_an_unkeyed_table_with_a_column_without_a_binary_form_loads(
    postgres_dsn: str,
) -> None:
    """An unkeyed table's inference sample is hashed in each row's binary
    form, which `aclitem` and arrays of it lack. Such a table loaded before
    the sample was ordered, then failed with `no binary output function`, and
    loads again: such columns only break ties, by their text.
    """
    schema = f'sdm_test_{uuid.uuid4().hex}'
    connection = psycopg.connect(postgres_dsn, autocommit=True)
    try:
        connection.execute(f'CREATE SCHEMA "{schema}"')
        connection.execute(
            f'CREATE TABLE "{schema}".grants '
            '(n BIGINT, item aclitem, acl aclitem[])'
        )
        connection.execute(
            f'INSERT INTO "{schema}".grants '
            "SELECT n % 1200, 'postgres=r/postgres', "
            "ARRAY['postgres=r/postgres', 'postgres=w/postgres']::aclitem[] "
            'FROM generate_series(0, 1499) AS n'
        )
        graph = Graph.from_postgres(
            connection,
            tables=[dict(name='grants', source_name='grants', columns=['n'])],
            schema=schema,
            verbose=False,
        )
        sample = graph['grants']._source_sample_df
        assert len(sample) == 1000
        assert sample['n'].between(0, 1199).all()
    finally:
        connection.execute(f'DROP SCHEMA "{schema}" CASCADE')
        connection.close()


def test_an_unkeyed_table_without_a_readable_column_is_sampled_spread(
    postgres_dsn: str,
) -> None:
    """A table with no column of a type the backend reads was sampled by its
    rows' text alone, so the sample was the rows that sort first by text:
    here 750 'a' rows and 250 'b' rows of a half-and-half enum column. Such
    a table is now ordered by a hash of that text first.
    """
    schema = f'sdm_test_{uuid.uuid4().hex}'
    connection = psycopg.connect(postgres_dsn, autocommit=True)
    try:
        connection.execute(f'CREATE SCHEMA "{schema}"')
        connection.execute(f"CREATE TYPE \"{schema}\".mood AS ENUM ('a', 'b')")
        connection.execute(
            f'CREATE TABLE "{schema}".t (m "{schema}".mood, iv INTERVAL)'
        )
        connection.execute(
            f'INSERT INTO "{schema}".t '
            "SELECT CASE WHEN n % 2 = 0 THEN 'a' ELSE 'b' END::"
            f'"{schema}".mood, n * INTERVAL \'1 second\' '
            'FROM generate_series(0, 1499) AS n'
        )
        first_by_text = connection.execute(
            f'SELECT m::text FROM "{schema}".t '
            'ORDER BY CAST(ROW(m, iv) AS text) COLLATE "C" LIMIT 1000'
        ).fetchall()
        assert collections.Counter(m for (m,) in first_by_text) == {
            'a': 750,
            'b': 250,
        }

        graph = Graph.from_postgres(
            connection,
            tables=[dict(name='t', source_name='t', columns=['m'])],
            schema=schema,
            verbose=False,
        )
        sample = graph['t']._source_sample_df
        assert len(sample) == 1000
        counts = collections.Counter(sample['m'].astype(str))
        assert 400 < counts['a'] < 600
        assert 400 < counts['b'] < 600
    finally:
        connection.execute(f'DROP SCHEMA "{schema}" CASCADE')
        connection.close()


@pytest.mark.parametrize(
    ('target', 'prefiltered'),
    [(64, True), (1, False)],
    ids=['block-sample', 'too-few-rows'],
)
def test_a_large_unkeyed_table_samples_its_blocks(
    postgres_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
    target: int,
    prefiltered: bool,
) -> None:
    """An unkeyed table is ordered by a hash of every row, which reads every
    block. From `_SAMPLE_PREFILTER_MIN_BLOCKS` blocks it reads a repeatable
    sample of them first, a power-of-two fraction holding at least
    `_SAMPLE_PREFILTER_TARGET_BLOCKS`; when that holds fewer rows than the
    sample needs, the table is read whole, and every later read of it reads
    what that settled on. The thresholds are shrunk here, so a table of
    100,000 rows, about 540 blocks, counts as large.
    """
    from sdm_relational_engine.rfm.backend.postgres import PostgresTable

    monkeypatch.setattr(PostgresTable, '_SAMPLE_PREFILTER_MIN_BLOCKS', 64)
    monkeypatch.setattr(
        PostgresTable, '_SAMPLE_PREFILTER_TARGET_BLOCKS', target
    )
    schema = f'sdm_test_{uuid.uuid4().hex}'
    connection = psycopg.connect(postgres_dsn, autocommit=True)
    try:
        connection.execute(f'CREATE SCHEMA "{schema}"')
        connection.execute(
            f'CREATE TABLE "{schema}".big AS SELECT n, \'m\' || n AS m '
            'FROM generate_series(1, 100000) AS n'
        )
        from sdm_connectors.source import Source

        sent: list[str] = []
        original = Source._run

        def run(self: Source, sql: str, params: object = None) -> object:
            sent.append(sql)
            return original(self, sql, params)  # type: ignore[arg-type]

        monkeypatch.setattr(Source, '_run', run)
        samples = []
        for _ in range(2):
            sent.clear()
            table = PostgresTable(
                connection,
                'big',
                schema=schema,
                columns=['n', dict(name='twice', expr='2 * n', dtype='int')],
                primary_key=None,
            )
            source = table._source_sample_df
            expr = table._get_expr_sample_df([table['twice']])
            samples.append(source['n'].tolist())
            reads = [sql for sql in sent if '__kumo_order_' in sql]
            assert sum('pg_relation_size' in sql for sql in sent) == 1
            # The first read, of whichever columns, settles where every read
            # reads from.
            if prefiltered:
                assert all(
                    'TABLESAMPLE SYSTEM (12.5) REPEATABLE (0)' in sql
                    for sql in reads
                )
            else:
                assert 'TABLESAMPLE SYSTEM (0.1953125)' in reads[0]
                assert all('TABLESAMPLE' not in sql for sql in reads[1:])
                assert reads[1] == reads[0].replace(
                    ' TABLESAMPLE SYSTEM (0.1953125) REPEATABLE (0)', ''
                )
            assert len(source) == 1000
            assert (
                expr['twice'].to_numpy() == 2 * source['n'].to_numpy()
            ).all()
        assert samples[0] == samples[1]
    finally:
        connection.execute(f'DROP SCHEMA "{schema}" CASCADE')
        connection.close()
