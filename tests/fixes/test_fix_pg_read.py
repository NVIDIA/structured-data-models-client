# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P55.pg-read: the four PostgreSQL reading fixes from DV14.

* A connection made with ``row_factory=dict_row`` or ``namedtuple_row``
  predicts through the public API exactly as a default one does; today it
  fails with a misleading error.
* ``timetz`` keeps its offset: ``PostgresSource._run`` reads it as its text,
  and the engine's values carry their zone, as before Phase 3, so a time
  column that can't be parsed reports its values with their offsets.
* ``rows_to_arrow`` refuses values that don't fit a column's declared type,
  naming the column, rather than turning the column into text; so does
  reading a two-dimensional array through ``PostgresSource._run``. A column
  without a declared type still holds mixed values as text.
* A repeated ``uuid``, ``inet`` or ``cidr`` key is reported with the value's
  Python form, as before Phase 3: ``UUID('...')``, ``IPv4Address('...')``,
  ``IPv6Network('...')``.
"""

from __future__ import annotations

import uuid
import warnings
from typing import Any

import pytest
from _fixes import (
    CLOCK_QUERIES,
    clock_options,
    clock_schema,
    drop_postgres_schemas,
    load_postgres_clock,
    postgres_dsn,
    predict,
)


@pytest.fixture(scope='module')
def dsn() -> str:
    return postgres_dsn()


@pytest.fixture()
def schema(dsn: str) -> Any:
    import psycopg

    name = f'fix_pg_{uuid.uuid4().hex[:10]}'
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(f'CREATE SCHEMA "{name}"')
    try:
        yield name
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(f'DROP SCHEMA "{name}" CASCADE')


# Row factories ###############################################################


@pytest.fixture(scope='module')
def clock(dsn: str) -> Any:
    names = load_postgres_clock(dsn, f'fix_rows_{uuid.uuid4().hex[:10]}')
    try:
        yield names['timestamp']
    finally:
        drop_postgres_schemas(dsn, list(names.values()))


def _payloads(dsn: str, schema: str, row_factory: Any) -> Any:
    import psycopg
    from sdm_connectors import PostgresSource

    options: dict[str, Any] = {'autocommit': True}
    if row_factory is not None:
        options['row_factory'] = row_factory
    connection = psycopg.connect(dsn, **options)
    try:
        built = clock_schema(PostgresSource(connection, schema=schema))
        return predict(
            built, CLOCK_QUERIES['count'], **clock_options(anchor=True)
        )
    finally:
        connection.close()


@pytest.mark.parametrize('factory', ('dict_row', 'namedtuple_row'))
def test_a_connection_with_another_row_factory_predicts(
    dsn: str, clock: str, factory: str
) -> None:
    from psycopg import rows

    want = _payloads(dsn, clock, None)
    assert _payloads(dsn, clock, getattr(rows, factory)) == want


def test_the_callers_row_factory_is_left_as_it_was(
    dsn: str, clock: str
) -> None:
    import psycopg
    from psycopg.rows import dict_row
    from sdm_connectors import PostgresSource

    connection = psycopg.connect(dsn, autocommit=True, row_factory=dict_row)
    try:
        table = PostgresSource(connection, schema=clock)._run(
            'SELECT 1 AS one, 2 AS two'
        )
        assert table.to_pylist() == [{'one': 1, 'two': 2}]
        assert connection.row_factory is dict_row
        assert connection.execute('SELECT 1 AS one').fetchone() == {'one': 1}
    finally:
        connection.close()


# timetz ######################################################################


def test_source_reads_timetz_as_its_text(dsn: str) -> None:
    import psycopg
    import pyarrow as pa
    from sdm_connectors import PostgresSource

    connection = psycopg.connect(dsn, autocommit=True)
    try:
        table = PostgresSource(connection)._run(
            "SELECT '12:00:00+05:30'::timetz AS t, NULL::timetz AS n"
        )
    finally:
        connection.close()
    assert table.schema.field('t').type == pa.string()
    assert table.column('t').to_pylist() == ['12:00:00+05:30']
    assert table.column('n').to_pylist() == [None]


def test_timetz_values_keep_their_zone(dsn: str, schema: str) -> None:
    r"""As DV14 measured before Phase 3: a ``timetz`` column read as a time
    column reports its unparsed values with their offsets."""
    import psycopg
    from sdm_relational_engine.rfm.backend.postgres import PostgresTable

    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(
            f'CREATE TABLE "{schema}".c (id BIGINT PRIMARY KEY, v timetz)'
        )
        literals = ["'12:00:00+05:30'", "'23:59:59-12'", 'NULL']
        rows = ', '.join(f'({i}, {literals[i % 3]})' for i in range(1, 25))
        connection.execute(f'INSERT INTO "{schema}".c VALUES {rows}')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            table = PostgresTable(connection, 'c', schema=schema)
            assert len(table._source_sample_df) == 24
    messages = [str(w.message) for w in caught]
    found = [m for m in messages if "time column 'v'" in m]
    assert found, messages
    assert (
        'datetime.time(12, 0, tzinfo=datetime.timezone('
        'datetime.timedelta(seconds=19800)))'
    ) in found[0], found[0]
    assert (
        'datetime.time(23, 59, 59, tzinfo=datetime.timezone('
        'datetime.timedelta(days=-1, seconds=43200)))'
    ) in found[0], found[0]


# Values that don't fit their type ############################################


def test_rows_to_arrow_refuses_values_that_do_not_fit() -> None:
    import pyarrow as pa
    from sdm_connectors.sql import rows_to_arrow

    with pytest.raises(Exception, match='grid') as caught:
        rows_to_arrow(
            [[1, [[1, 2], [3, 4]]], [2, [5]]],
            ['id', 'grid'],
            [pa.int64(), pa.list_(pa.int64())],
        )
    assert not isinstance(caught.value, AssertionError)


def test_rows_to_arrow_still_reads_undeclared_mixed_values_as_text() -> None:
    import pyarrow as pa
    from sdm_connectors.sql import rows_to_arrow

    table = rows_to_arrow([[1, 'a'], ['x', 2]], ['p', 'q'])
    assert table.schema.field('p').type == pa.string()
    assert table.column('p').to_pylist() == ['1', 'x']
    assert table.column('q').to_pylist() == ['a', '2']


def test_source_refuses_a_two_dimensional_array(dsn: str) -> None:
    import psycopg
    from sdm_connectors import PostgresSource

    connection = psycopg.connect(dsn, autocommit=True)
    try:
        with pytest.raises(Exception, match='grid') as caught:
            PostgresSource(connection)._run(
                "SELECT '{{1,2},{3,4}}'::int4[] AS grid"
            )
        assert not isinstance(caught.value, AssertionError)
        # The connection is still usable.
        assert connection.execute('SELECT 1').fetchone() == (1,)
    finally:
        connection.close()


# Repeated keys ###############################################################


@pytest.mark.parametrize(
    ('column', 'example'),
    [
        ('u', "UUID('a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11')"),
        ('i', "IPv4Address('10.0.0.1')"),
        ('i6', "IPv6Address('2001:db8::1')"),
        ('n', "IPv4Network('10.0.0.0/8')"),
        ('n6', "IPv6Network('2001:db8::/32')"),
    ],
)
@pytest.mark.parametrize('distinct', (190, 390))
def test_a_repeated_key_is_reported_as_before_phase_3(
    dsn: str, schema: str, column: str, example: str, distinct: int
) -> None:
    r"""190 distinct values beside three repeats is more than 1% repeated, and
    refused; 390 is less, and warned about."""
    import psycopg
    from sdm_client import Schema
    from sdm_connectors import PostgresSource

    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(
            f'CREATE TABLE "{schema}".d (u uuid, i inet, i6 inet, n cidr, '
            'n6 cidr, x int)'
        )
        connection.execute(
            f'INSERT INTO "{schema}".d SELECT '
            "'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', '10.0.0.1', "
            "'2001:db8::1', '10.0.0.0/8', '2001:db8::/32', g "
            'FROM generate_series(1, 3) AS g'
        )
        connection.execute(
            f'INSERT INTO "{schema}".d SELECT '
            "('00000000-0000-4000-8000-' || lpad(g::text, 12, '0'))::uuid, "
            "('10.1.' || g / 250 || '.' || g % 250)::inet, "
            "('2001:db8::' || to_hex(g))::inet, "
            "('10.2.' || g / 250 || '.' || g % 250 || '/32')::cidr, "
            "('2001:db8:' || to_hex(g) || '::/48')::cidr, g "
            f'FROM generate_series(10, {9 + distinct}) AS g'
        )
    connection = psycopg.connect(dsn, autocommit=True)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                Schema.from_source(
                    PostgresSource(connection, schema=schema),
                    {'d': 'd'},
                    primary_keys={'d': column},
                )
                messages = [str(w.message) for w in caught]
            except Exception as error:  # noqa: BLE001 - the refusal is read
                messages = [str(error)]
    finally:
        connection.close()
    reported = [m for m in messages if 'e.g.' in m]
    assert reported, messages
    assert f'e.g. {example}' in reported[0], reported[0]
