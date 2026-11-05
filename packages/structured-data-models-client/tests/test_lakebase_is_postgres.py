# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Postgres session schemas are resolved and qualified afresh for every run."""

from __future__ import annotations

import json
import os
import uuid
import warnings
from collections.abc import Iterator
from typing import Any

import pytest
from sdm_connectors import PostgresSource

from sdm_client import PredictiveQuery, Schema, StructuredDataError

QUERY = 'PREDICT users.target FOR EACH users.id'


@pytest.fixture
def server() -> Iterator[str]:
    r"""The DSN of a database made for the test, dropped after it."""
    dsn = os.getenv('SDM_POSTGRES_TEST_DSN')
    if not dsn:
        if os.getenv('SDM_REQUIRE_POSTGRES'):
            pytest.fail(
                'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
            )
        pytest.skip('SDM_POSTGRES_TEST_DSN is not set')
    psycopg = pytest.importorskip('psycopg')
    from psycopg.conninfo import make_conninfo

    name = f'sdm_t37_{uuid.uuid4().hex[:12]}'
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute(f'CREATE DATABASE {name}')
    try:
        yield make_conninfo(dsn, dbname=name)
    finally:
        with psycopg.connect(dsn, autocommit=True) as admin:
            admin.execute(f'DROP DATABASE {name} WITH (FORCE)')


def _users(connection: Any, schema: str, scale: int) -> None:
    r"""``schema.users``, twenty keyed rows whose target is ``scale`` times
    their key modulo three."""
    connection.execute(f'CREATE SCHEMA {schema}')
    connection.execute(
        f'CREATE TABLE {schema}.users '
        '(id BIGINT PRIMARY KEY, target DOUBLE PRECISION)'
    )
    connection.execute(
        f'INSERT INTO {schema}.users '
        f'SELECT i, {scale} * (i % 3) FROM generate_series(1, 20) AS i'
    )


@pytest.fixture
def connection(server: str) -> Iterator[Any]:
    r"""A connection whose search path is a schema of the test's own,
    ``shop``, holding ``users``; the source is given no schema."""
    import psycopg

    with psycopg.connect(server, autocommit=True) as connection:
        _users(connection, 'shop', 1)
        connection.execute('SET search_path TO shop')
        yield connection


def _payload(schema: Schema) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        request = PredictiveQuery(QUERY, schema).materialize(
            indices=[1, 2, 3], random_seed=7
        )
    return json.loads(json.dumps(request._payloads, default=str))


def _described(connection: Any, tables: Any = None) -> Schema:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_source(
            PostgresSource(connection), tables or {'users': 'users'}
        )


def _recording(connection: Any) -> list[str]:
    r"""Every statement run on ``connection`` from now on, in order."""
    import psycopg

    statements: list[str] = []

    class Recorded(psycopg.Cursor):
        def execute(self, query: Any, *args: Any, **kwargs: Any) -> Any:
            statements.append(str(query))
            return super().execute(query, *args, **kwargs)

    connection.cursor_factory = Recorded
    return statements


def test_temporary_table_does_not_shadow_qualified_source(connection):
    connection.execute('CREATE TEMPORARY TABLE users (id BIGINT, other TEXT)')
    connection.execute("INSERT INTO users VALUES (1, 'temporary')")
    schema = _described(connection)
    assert sorted(schema['users'].columns) == ['id', 'target']
    assert 'temporary' not in json.dumps(_payload(schema))


def test_every_statement_names_schema_and_resolves_it_once_per_run(connection):
    statements = _recording(connection)
    schema = _described(connection)
    described = list(statements)
    _payload(schema)
    for run in (described, statements[len(described) :]):
        assert run
        asking = [s for s in run if 'current_schema' in s]
        assert len(asking) == 1
        for statement in run:
            assert 'search_path' not in statement
            assert 'regclass' not in statement
            if 'users' in statement and statement not in asking:
                assert '"shop"."users"' in statement or (
                    "'shop'" in statement and "'users'" in statement
                ), statement


def test_missing_session_schema_names_the_remedy(connection):
    connection.execute('SET search_path TO no_such_schema')
    with pytest.raises(StructuredDataError) as caught:
        _described(connection)
    assert caught.value.code == 'INVALID_REQUEST'
    assert str(caught.value) == (
        "[INVALID_REQUEST] No current 'schema' set. "
        'Please specify the PostgreSQL schema manually'
    )


def test_dotted_name_is_a_literal_table_name(connection):
    with pytest.raises(StructuredDataError) as caught:
        _described(connection, {'users': 'shop.users'})
    assert (
        "Table 'shop.shop.users' does not exist in the PostgreSQL database"
        in str(caught.value)
    )


def test_changed_search_path_is_observed_at_next_materialize(connection):
    _users(connection, 'outlet', 10)
    schema = _described(connection)
    first = _payload(schema)
    connection.execute('SET search_path TO outlet')
    second = _payload(schema)
    assert second == _payload(_described(connection))
    assert len(first) == len(second) == 1
    target = first[0]['task']['target']['column_name']
    for before, after in zip(first, second, strict=True):
        old = before['context']['instance_table']
        new = after['context']['instance_table']
        position = old['columns'].index(target)
        assert old['columns'] == new['columns']
        assert [row[position] for row in new['rows']] == [
            10 * row[position] for row in old['rows']
        ]
