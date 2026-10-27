# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Three answers the core route gives as the old route does (T3.8 fix 2,
from Codex's first review round): each case runs on the old route and with
every core route forced, through the public API, and the two must agree.

- A schema built by hand reads only the columns it keeps: a column it leaves
  out is never read, however its values look.
- A day-first date literal in a query the core gathers alone is warned of,
  in the reference's words.
- A Postgres table a source given no schema lacks is named under the schema
  the read looked in.
"""

from __future__ import annotations

import json
import os
import re
import uuid
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import pandas as pd
import pytest

duckdb = pytest.importorskip('duckdb')

from sdm_connectors import DuckDBSource, PostgresSource  # noqa: E402

from sdm_client import (  # noqa: E402
    Column,
    PredictiveQuery,
    Schema,
    SourceTable,
    StructuredDataError,
    _routing,
)

CORE = ','.join(sorted(_routing.CORE_ROUTED))
#: The code the core route puts before a warning's words (P3E-2).
CODE = re.compile(r'^\[[A-Z][A-Z0-9_]*\] ')


def _on_both_routes(
    monkeypatch: pytest.MonkeyPatch, call: Callable[[], Any]
) -> dict[str, Any]:
    r"""``call``'s answer on the old route and with every core route forced."""
    answers = {}
    for route, forced in (('old', ''), ('core', CORE)):
        monkeypatch.setenv(_routing.ENVIRONMENT, forced)
        answers[route] = call()
    return answers


def test_a_column_a_schema_built_by_hand_leaves_out_is_never_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A timestamp past what nanoseconds hold, in a column the schema does not
    # keep: the reference reads only the columns a schema describes
    # (`_copy_for_one_gather`), so it never meets it.
    connection = duckdb.connect()
    connection.execute(
        'CREATE TABLE users AS SELECT i::BIGINT AS id, (i % 3)::DOUBLE AS '
        "target, TIMESTAMP '10000-01-01' AS ignored_time FROM range(20) t(i)"
    )
    schema = Schema(
        {
            'users': SourceTable(
                'users',
                'users',
                {
                    'id': Column('id', 'int64', 'ID'),
                    'target': Column('target', 'float64', 'numerical'),
                },
                primary_key='id',
            )
        },
        source=DuckDBSource(connection),
    )

    def materialized() -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            request = PredictiveQuery(
                'PREDICT users.target FOR EACH users.id', schema
            ).materialize(indices=[1, 2], random_seed=7)
        return json.loads(json.dumps(request._payloads, default=str))

    answers = _on_both_routes(monkeypatch, materialized)
    assert answers['core'] == answers['old']
    (payload,) = answers['core']
    assert 'ignored_time' not in json.dumps(payload)


def test_a_key_over_a_column_left_out_marks_no_column_it_keeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The source keys the table on (ts, omitted); the schema leaves `omitted`
    # out and names `id`. The reference never reads the source's key here,
    # so the schema's key stands (T3.8 fix 3, Codex round 2).
    connection = duckdb.connect()
    connection.execute(
        'CREATE TABLE users (id BIGINT, target DOUBLE, ts TIMESTAMP, '
        'omitted BIGINT, PRIMARY KEY (ts, omitted))'
    )
    connection.execute(
        "INSERT INTO users SELECT i, i % 3, TIMESTAMP '2024-01-01' + i * "
        'INTERVAL 1 DAY, i FROM range(20) t(i)'
    )
    schema = Schema(
        {
            'users': SourceTable(
                'users',
                'users',
                {
                    'id': Column('id', 'int64', 'ID'),
                    'target': Column('target', 'float64', 'numerical'),
                    'ts': Column('ts', 'timestamp[us]', 'timestamp'),
                },
                primary_key='id',
            )
        },
        source=DuckDBSource(connection),
    )

    def materialized() -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            request = PredictiveQuery(
                'PREDICT users.target FOR EACH users.id', schema
            ).materialize(indices=[1, 2], random_seed=7)
        return json.loads(json.dumps(request._payloads, default=str))

    answers = _on_both_routes(monkeypatch, materialized)
    assert answers['core'] == answers['old']


def _buys() -> Schema:
    connection = duckdb.connect()
    connection.execute(
        'CREATE TABLE users AS SELECT i::BIGINT AS uid FROM range(20) t(i)'
    )
    connection.execute(
        'CREATE TABLE buys AS SELECT i::BIGINT AS bid, (i % 20)::BIGINT AS '
        "uid, TIMESTAMP '2025-06-01' + i * INTERVAL 1 DAY AS stamp "
        'FROM range(300) t(i)'
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_source(
            DuckDBSource(connection),
            primary_keys={'users': 'uid', 'buys': 'bid'},
            time_columns={'buys': 'stamp'},
        )


def test_one_question_in_two_spellings_is_one_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A request keeps its query as the validated query writes it back, so a
    # session pins one question however it was spelled (T3.8 fix 3, Claude
    # round 2).
    def asked() -> list[Any]:
        schema = _buys()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            requests = [
                PredictiveQuery(text, schema).materialize(
                    indices=[1, 2], random_seed=7
                )
                for text in (
                    'PREDICT COUNT(buys.*, 0, 30, days) FOR EACH users.uid',
                    'predict   count(buys.*, 0, 30, days)\nfor each users.uid',
                )
            ]
        return [(r.query, r._question) for r in requests]

    answers = _on_both_routes(monkeypatch, asked)
    assert answers['core'] == answers['old']
    (first, second) = answers['core']
    assert first == second


def test_a_forecast_step_s_refusal_names_the_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refused() -> str:
        schema = _buys()
        with (
            warnings.catch_warnings(),
            pytest.raises(StructuredDataError) as caught,
        ):
            warnings.simplefilter('ignore')
            PredictiveQuery(
                'PREDICT COUNT(buys.*, 0, 30, days) FORECAST 40 TIMEFRAMES '
                'FOR EACH users.uid',
                schema,
            ).materialize(indices=[1, 2])
        return str(caught.value)

    answers = _on_both_routes(monkeypatch, refused)
    assert answers['core'] == answers['old']
    assert 'This is forecast step 10 of 40.' in answers['core']


def test_a_day_first_literal_is_warned_of_when_the_core_gathers_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = duckdb.connect()
    connection.execute(
        'CREATE TABLE users AS SELECT i::BIGINT AS id, (i % 3)::DOUBLE AS '
        "target, TIMESTAMP '2024-01-01' + i * INTERVAL 1 DAY AS ts "
        'FROM range(40) t(i)'
    )

    def said() -> list[str]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            schema = Schema.from_source(
                DuckDBSource(connection),
                primary_keys={'users': 'id'},
                time_columns={'users': 'ts'},
            )
            PredictiveQuery(
                'PREDICT users.target FOR EACH users.id '
                'WHERE users.ts > 13/01/2024',
                schema,
            ).materialize(
                indices=[20, 21],
                random_seed=7,
                anchor_time=pd.Timestamp('2024-03-01'),
            )
        return [
            CODE.sub('', str(warning.message))
            for warning in caught
            if 'Parsing dates' in str(warning.message)
        ]

    answers = _on_both_routes(monkeypatch, said)
    # The old route says it each time it parses the literal; the words are
    # what a caller reads.
    assert answers['old'], answers
    assert set(answers['core']) == set(answers['old'])


@pytest.fixture
def postgres() -> Iterator[Any]:
    r"""A psycopg connection whose current schema is one of its own."""
    dsn = os.getenv('SDM_POSTGRES_TEST_DSN')
    if not dsn:
        if os.getenv('SDM_REQUIRE_POSTGRES'):
            pytest.fail(
                'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
            )
        pytest.skip('SDM_POSTGRES_TEST_DSN is not set')
    psycopg = pytest.importorskip('psycopg')
    namespace = f'sdm_{uuid.uuid4().hex[:8]}'
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(f'CREATE SCHEMA {namespace}')
        connection.execute(f'SET search_path TO {namespace}')
        connection.execute(
            'CREATE TABLE users (id BIGINT, target DOUBLE PRECISION)'
        )
        connection.execute(
            'INSERT INTO users SELECT i, i % 3 FROM generate_series(1, 20) AS i'
        )
        try:
            yield connection, namespace
        finally:
            connection.execute(f'DROP SCHEMA {namespace} CASCADE')


def test_a_missing_postgres_table_is_named_under_the_schema_looked_in(
    monkeypatch: pytest.MonkeyPatch, postgres: Any
) -> None:
    connection, namespace = postgres

    def refused() -> str:
        with pytest.raises(StructuredDataError) as caught:
            Schema.from_source(
                PostgresSource(connection), {'users': 'users', 'gone': 'nope'}
            )
        return str(caught.value)

    answers = _on_both_routes(monkeypatch, refused)
    assert answers['core'] == answers['old']
    assert f"Table '{namespace}.nope' does not exist" in answers['core']
