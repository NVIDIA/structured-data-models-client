# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public PostgreSQL sampling keeps unorderable string types in tie-breaks."""

import os
import uuid

import pytest
from sdm_connectors import PostgresSource

from sdm_client import KumoRelational, PredictiveQuery, Schema, SDMClient


@pytest.fixture
def postgres_notes():
    dsn = os.environ.get('SDM_POSTGRES_TEST_DSN')
    if not dsn:
        if os.environ.get('SDM_REQUIRE_POSTGRES') == '1':
            pytest.fail('SDM_POSTGRES_TEST_DSN is required')
        pytest.skip('SDM_POSTGRES_TEST_DSN is not set')
    psycopg = pytest.importorskip('psycopg')
    name = 'sdm_keyless_' + uuid.uuid4().hex[:12]
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(f'CREATE SCHEMA {name}')
        try:
            connection.execute(f'CREATE DOMAIN {name}.payload AS JSON')
            connection.execute(
                f'CREATE TABLE {name}.users(id BIGINT PRIMARY KEY, value DOUBLE PRECISION)'
            )
            connection.execute(
                f'INSERT INTO {name}.users SELECT i, i*1.5 FROM generate_series(1,20) i'
            )
            yield connection, name
        finally:
            connection.execute(f'DROP SCHEMA {name} CASCADE')


def prepare(postgres_notes, kind):
    connection, name = postgres_notes
    dtype = f'{name}.payload' if kind == 'domain' else kind
    connection.execute(
        f'CREATE TABLE {name}.notes('
        f'user_id BIGINT REFERENCES {name}.users(id), noted_at TIMESTAMP, '
        f'a_number INTEGER, "body quoted" {dtype})'
    )
    expression = (
        'xmlelement(name v, v)' if kind == 'xml' else "json_build_object('v',v)"
    )
    connection.execute(
        f"INSERT INTO {name}.notes SELECT i, TIMESTAMP '2024-01-01', n, "
        f'CASE WHEN v IS NULL THEN NULL ELSE {expression} END '
        'FROM generate_series(1,20)i CROSS JOIN '
        "(VALUES (2,'z'),(10,'a'),(2,'a'),(2,NULL)) a(n,v)"
    )
    schema = Schema.from_source(
        PostgresSource(connection, schema=name),
        {'users': 'users', 'notes': 'notes'},
        primary_keys={'users': 'id'},
        time_columns={'notes': 'noted_at'},
    )
    assert schema['notes'].primary_key is None
    # Model-facing categorical declarations are public input configuration.
    schema['notes']['body quoted'].stype = 'categorical'
    return schema


def materialize(schema, neighbors=2):
    return PredictiveQuery(
        'PREDICT users.value FOR EACH users.id', schema
    ).materialize(
        indices=[1, 2],
        model=KumoRelational(),
        num_context_samples=10,
        num_neighbors=[neighbors],
        random_seed=7,
    )


@pytest.mark.parametrize('kind', ['json', 'xml', 'domain'])
def test_keyless_strings_keep_numeric_order_and_nulls(postgres_notes, kind):
    schema = prepare(postgres_notes, kind)
    expected = (
        ['<v>a</v>', '<v>z</v>']
        if kind == 'xml'
        else ['{"v" : "a"}', '{"v" : "z"}']
    )
    for _ in range(2):
        table = materialize(schema).payloads[0]['predict']['related_tables'][
            'notes'
        ]
        columns = table['columns']
        numbers = [row[columns.index('a_number')] for row in table['rows']]
        bodies = [row[columns.index('body quoted')] for row in table['rows']]
        assert numbers == [
            2,
            2,
            2,
            2,
        ]  # Numeric2 precedes10, not text ordering.
        assert bodies == expected * 2
    table = materialize(schema, 3).payloads[0]['predict']['related_tables'][
        'notes'
    ]
    assert [
        row[table['columns'].index('body quoted')] for row in table['rows']
    ] == (expected + [None]) * 2
    assert postgres_notes[0].execute('SELECT 1').fetchone() == (1,)


@pytest.mark.live_nim
def test_postgres_keyless_read_to_prediction(postgres_notes):
    url = os.environ.get('SDM_RELATIONAL_URL')
    if not url:
        pytest.skip('SDM_RELATIONAL_URL is required')
    request = materialize(prepare(postgres_notes, 'json'))
    with SDMClient(url=url, timeout=180) as client:
        result = client.predict(request, model=KumoRelational())
    assert result.to_pandas()['id'].astype(int).tolist() == [1, 2]
