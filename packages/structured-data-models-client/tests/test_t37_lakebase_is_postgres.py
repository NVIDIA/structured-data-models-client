# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Lakebase is Postgres (T3.7): a ``lakebase()`` connection is a psycopg
connection, read as a ``PostgresSource``, and the one change on the core
route is that every Postgres source given no schema reads its session's
schema once per run and names it in every statement, as the old client
reads ``current_schema()`` once and writes ``"schema"."table"`` into every
read (``old-behaviour/T3.7-lakebase.md``, rows 16, 21 and 38).

Each case runs through the public API on a local Postgres, with the session's
search path set to a schema of the test's own and no schema given the
source, on the old route and with every core route forced:

- a table a temporary table of its name shadows is read from the schema, as
  the old client reads it, where the core refused it before (row 21);
- every statement the core sends names the schema, and the session's schema
  is asked once per run;
- a session whose search path holds no schema is refused in the old
  client's words (row 17);
- a dotted name is one table in the schema, refused on both routes as the
  old client refuses it (rows 20 and 22);
- a session switched to another schema between two materializes is read
  where it now points, the core keeping nothing between runs (plan B,
  register P3A-1), where the old client reads where it first read.

Where ``SDM_POSTGRES_TEST_DSN`` names no server and ``SDM_REQUIRE_POSTGRES`` is
not set, there is nothing to hold.
"""

from __future__ import annotations

import json
import os
import uuid
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from sdm_connectors import PostgresSource

from sdm_client import PredictiveQuery, Schema, StructuredDataError, _routing

CORE = ','.join(sorted(_routing.CORE_ROUTED))
QUERY = 'PREDICT users.target FOR EACH users.id'


def _on_both_routes(
    monkeypatch: pytest.MonkeyPatch, call: Callable[[], Any]
) -> dict[str, Any]:
    r"""``call``'s answer on the old route and with every core route forced."""
    answers = {}
    for route, forced in (('old', ''), ('core', CORE)):
        monkeypatch.setenv(_routing.ENVIRONMENT, forced)
        answers[route] = call()
    return answers


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


def test_a_table_a_temporary_table_shadows_is_read_from_the_schema(
    monkeypatch: pytest.MonkeyPatch, connection: Any
) -> None:
    # A temporary table of the same name, first on the session's search
    # path: an unqualified name would read it, a qualified one never does.
    connection.execute('CREATE TEMPORARY TABLE users (id BIGINT, other TEXT)')
    connection.execute("INSERT INTO users VALUES (1, 'temporary')")

    def read() -> Any:
        schema = _described(connection)
        return sorted(schema['users'].columns), _payload(schema)

    answers = _on_both_routes(monkeypatch, read)
    assert answers['core'] == answers['old']
    columns, payload = answers['core']
    assert columns == ['id', 'target']
    assert 'temporary' not in json.dumps(payload)


def test_every_statement_names_the_schema_and_the_session_is_asked_once_a_run(
    monkeypatch: pytest.MonkeyPatch, connection: Any
) -> None:
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    statements = _recording(connection)
    schema = _described(connection)
    described = list(statements)
    _payload(schema)
    materialized = statements[len(described) :]

    for run in (described, materialized):
        assert run
        asking = [s for s in run if 'current_schema' in s]
        assert len(asking) == 1, asking
        for statement in run:
            assert 'search_path' not in statement
            assert 'regclass' not in statement
            if 'users' in statement and statement not in asking:
                assert '"shop"."users"' in statement or (
                    "'shop'" in statement and "'users'" in statement
                ), statement


def test_a_session_with_no_schema_is_refused_in_the_old_client_s_words(
    monkeypatch: pytest.MonkeyPatch, connection: Any
) -> None:
    connection.execute('SET search_path TO no_such_schema')

    def refused() -> tuple[str, str]:
        with pytest.raises((StructuredDataError, ValueError)) as caught:
            _described(connection)
        return type(caught.value).__name__, str(caught.value)

    answers = _on_both_routes(monkeypatch, refused)
    words = (
        "No current 'schema' set. Please specify the PostgreSQL schema manually"
    )
    # The old client raises the engine's own ValueError; the core refuses by
    # name, INVALID_REQUEST, in the same words, as a StructuredDataError
    # says them, after its code (register P3E-6).
    assert answers['old'] == ('ValueError', words)
    assert answers['core'] == (
        'StructuredDataError',
        f'[INVALID_REQUEST] {words}',
    )


def test_a_dotted_name_is_one_table_in_the_schema_on_both_routes(
    monkeypatch: pytest.MonkeyPatch, connection: Any
) -> None:
    def refused() -> str:
        with pytest.raises(StructuredDataError) as caught:
            _described(connection, {'users': 'shop.users'})
        return str(caught.value)

    answers = _on_both_routes(monkeypatch, refused)
    assert answers['core'] == answers['old']
    assert (
        "Table 'shop.shop.users' does not exist in the PostgreSQL database"
        in answers['core']
    )


def test_a_session_switched_to_another_schema_is_read_where_it_now_points(
    monkeypatch: pytest.MonkeyPatch, connection: Any
) -> None:
    # Another schema whose users' targets are ten times shop's.
    _users(connection, 'outlet', 10)

    def two_runs() -> tuple[Any, Any]:
        connection.execute('SET search_path TO shop')
        schema = _described(connection)
        first = _payload(schema)
        connection.execute('SET search_path TO outlet')
        return first, _payload(schema)

    answers = _on_both_routes(monkeypatch, two_runs)
    # The old client reads where it first read; the core reads each run's
    # session as it stands (register P3A-1).
    old_first, old_second = answers['old']
    assert old_second == old_first
    core_first, core_second = answers['core']
    assert core_first == old_first
    assert core_second != core_first
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    assert core_second == _payload(_described(connection))
