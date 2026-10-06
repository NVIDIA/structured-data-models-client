# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A key of tables left in their source is held to the rules a key of frames
is, counted where the rows are.

The same frames are described twice, once in memory and once from DuckDB,
SQLite and Postgres, and the two have to say the same about their keys. A key
that repeats used to pass silently from a source: a scored id matched one of
its rows at random, and the context held the same entity twice at one anchor
time, with labels that could disagree.
"""

from __future__ import annotations

import io
import json
import os
import pickle
import re
import uuid
import warnings
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('sdm_relational_engine.relationallib')
duckdb = pytest.importorskip('duckdb')
dbapi = pytest.importorskip('adbc_driver_sqlite.dbapi')

from sdm_connectors import (  # noqa: E402
    DuckDBSource,
    PostgresSource,
    SQLiteSource,
)

from sdm_client import (  # noqa: E402
    PredictiveQuery,
    Schema,
    StructuredDataError,
)

TABLES = {'users': 'app_users', 'orders': 'app_orders'}
TEMPORAL = 'PREDICT SUM(orders.amount, 0, 30, days) FOR EACH users.user_id'
STATIC = 'PREDICT users.country FOR EACH users.user_id'


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    ids = [f'u{i:03d}' for i in range(200)]
    users = pd.DataFrame(
        {
            'user_id': ids,
            'age': rng.integers(18, 70, 200),
            'country': rng.choice(['US', 'DE', 'IN'], 200),
        }
    )
    orders = pd.DataFrame(
        {
            'order_id': range(1, 601),
            'user_id': rng.choice(ids, 600),
            'amount': rng.gamma(2, 20, 600).round(2),
            'ts': pd.Timestamp('2024-01-01')
            + pd.to_timedelta(rng.integers(0, 365 * 24, 600), unit='h'),
        }
    )
    return users, orders


USERS, ORDERS = _frames()


def _repeating(*ids: str) -> pd.DataFrame:
    r"""The users, with each of ``ids`` held by a second, identical row."""
    copies = USERS[USERS.user_id.isin(ids)]
    return pd.concat([USERS, copies], ignore_index=True)


def _missing_one() -> pd.DataFrame:
    row = pd.DataFrame({'user_id': [None], 'age': [40], 'country': ['US']})
    return pd.concat([USERS, row], ignore_index=True)


Build = Callable[..., Any]


def _postgres_type(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series):
        return 'BIGINT'
    if pd.api.types.is_float_dtype(series):
        return 'DOUBLE PRECISION'
    if pd.api.types.is_datetime64_any_dtype(series):
        return 'TIMESTAMP'
    return 'TEXT'


def _python_rows(frame: pd.DataFrame) -> list[tuple[Any, ...]]:
    # psycopg adapts Python values, not numpy's.
    return [
        tuple(
            value.to_pydatetime()
            if isinstance(value, pd.Timestamp)
            else value.item()
            if isinstance(value, np.generic)
            else value
            for value in row
        )
        for row in frame.itertuples(index=False)
    ]


@pytest.fixture
def postgres() -> Iterator[Any]:
    r"""A psycopg connection and a schema of its own, dropped afterwards."""
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
        try:
            yield connection, namespace
        finally:
            connection.execute(f'DROP SCHEMA {namespace} CASCADE')


@pytest.fixture
def build_duckdb() -> Iterator[Build]:
    connections = []

    def build(users: pd.DataFrame, orders: pd.DataFrame = ORDERS) -> Any:
        connection = duckdb.connect()
        connections.append(connection)
        connection.register('users_frame', users)
        connection.register('orders_frame', orders)
        connection.execute(
            'CREATE TABLE app_users AS SELECT * FROM users_frame'
        )
        connection.execute(
            'CREATE TABLE app_orders AS SELECT * FROM orders_frame'
        )
        return DuckDBSource(connection)

    yield build
    for connection in connections:
        connection.close()


@pytest.fixture
def build_sqlite(tmp_path: Path) -> Iterator[Build]:
    connections = []

    def build(users: pd.DataFrame, orders: pd.DataFrame = ORDERS) -> Any:
        path = tmp_path / f'{uuid.uuid4().hex}.sqlite'
        connection = dbapi.connect(str(path))
        connections.append(connection)
        with connection.cursor() as cursor:
            cursor.adbc_ingest('app_users', users, mode='create')
            cursor.adbc_ingest('app_orders', orders, mode='create')
        connection.commit()
        return SQLiteSource(connection)

    yield build
    for connection in connections:
        connection.close()


@pytest.fixture
def build_postgres(postgres: Any) -> Build:
    connection, namespace = postgres

    def build(
        users: pd.DataFrame,
        orders: pd.DataFrame = ORDERS,
        *,
        constrained: tuple[str, ...] = (),
    ) -> Any:
        prefix = f'{namespace}.'
        for name, frame, key in (
            ('app_users', users, 'user_id'),
            ('app_orders', orders, 'order_id'),
        ):
            columns = ', '.join(
                f'{column} {_postgres_type(frame[column])}'
                + (
                    ' PRIMARY KEY'
                    if name in constrained and column == key
                    else ''
                )
                for column in frame.columns
            )
            connection.execute(f'DROP TABLE IF EXISTS {prefix}{name}')
            connection.execute(f'CREATE TABLE {prefix}{name} ({columns})')
            markers = ', '.join(['%s'] * len(frame.columns))
            with connection.cursor() as cursor:
                cursor.executemany(
                    f'INSERT INTO {prefix}{name} VALUES ({markers})',
                    _python_rows(frame),
                )
        return PostgresSource(connection, schema=namespace)

    return build


@pytest.fixture(params=['duckdb', 'sqlite', 'postgres'])
def build(request: pytest.FixtureRequest) -> Build:
    return request.getfixturevalue(f'build_{request.param}')


# --- the same frames say the same about their keys, in memory or in place ---


class _Outcome:
    r"""What describing the tables said about their keys."""

    def __init__(self, describe: Callable[[], Schema]) -> None:
        self.error: str | None = None
        self.key: Any = None
        self.repeats: int | None = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                schema = describe()
            except StructuredDataError as error:
                self.error = _plain(str(error))
            else:
                self.key = schema['users'].primary_key
                self.repeats = schema['users'].duplicate_keys
        self.said = sorted(
            {
                _plain(str(warning.message))
                for warning in caught
                if re.search(
                    r'identifies a row|named like a key|no value',
                    str(warning.message),
                )
            }
        )


def _plain(message: str) -> str:
    r"""A message without what only one route can say.

    Which repeated value is shown is whichever a scan meets first, and each
    route names its own remedy: a frame is deduplicated, a source is not.
    """
    message = re.sub(r'e\.g\. .*?(?=\. |, so |$)', 'e.g. X', message)
    message = message.split(' Keep one row per ')[0]
    return message.split('. Name another key with primary_keys=')[0]


CASES = {
    'an inferred key repeating once': (_repeating('u007'), None),
    'an inferred key repeating past the tolerance': (
        _repeating(*[f'u{i:03d}' for i in range(0, 200, 10)]),
        None,
    ),
    'an inferred key missing a value': (_missing_one(), None),
    'a named key repeating once': (_repeating('u007'), 'user_id'),
    'a named key repeating past the tolerance': (
        _repeating(*[f'u{i:03d}' for i in range(0, 200, 10)]),
        'user_id',
    ),
    'a named key missing a value': (_missing_one(), 'user_id'),
    'a named composite key repeating': (USERS, ['country', 'age']),
}


@pytest.mark.parametrize('case', CASES)
def test_a_source_says_what_its_frames_say_about_a_key(
    build: Build, case: str
) -> None:
    users, key = CASES[case]
    stated = {'primary_keys': {'users': key}} if key is not None else {}
    in_memory = _Outcome(
        lambda: Schema.from_data({'users': users, 'orders': ORDERS}, **stated)
    )
    in_place = _Outcome(
        lambda: Schema.from_source(build(users), TABLES, **stated)
    )

    if in_memory.error is None:
        assert in_place.error is None
        assert in_place.said == in_memory.said
        assert in_place.key == in_memory.key
        assert in_place.repeats == in_memory.repeats
    else:
        # Frames infer a key before a named one replaces it, and can warn
        # about the first on the way to refusing the second.
        assert in_place.error is not None
        assert in_place.error.startswith(in_memory.error)


def test_the_cases_cover_every_outcome(build_duckdb: Build) -> None:
    r"""The parity above means little if every case passed quietly."""
    every = {'a key that holds': (USERS, None), **CASES}
    outcomes = {
        case: _Outcome(
            lambda users=users, key=key: Schema.from_source(
                build_duckdb(users),
                TABLES,
                **({'primary_keys': {'users': key}} if key else {}),
            )
        )
        for case, (users, key) in every.items()
    }

    assert outcomes['a key that holds'].said == []
    assert outcomes['a key that holds'].repeats == 0
    assert outcomes['an inferred key repeating once'].repeats == 1
    assert outcomes['an inferred key repeating past the tolerance'].key is None
    assert outcomes['an inferred key repeating past the tolerance'].said
    assert outcomes['an inferred key missing a value'].said
    assert outcomes['a named key repeating once'].repeats == 1
    for refused in (
        'a named key repeating past the tolerance',
        'a named key missing a value',
        'a named composite key repeating',
    ):
        assert outcomes[refused].error is not None, refused


def test_a_refusal_says_how_to_go_on(build: Build) -> None:
    users = _repeating(*[f'u{i:03d}' for i in range(0, 200, 10)])
    with pytest.raises(StructuredDataError) as caught:
        Schema.from_source(
            build(users), TABLES, primary_keys={'users': 'user_id'}
        )

    message = str(caught.value)
    assert re.search(
        r'users\.user_id is not unique enough to identify a row: 20 of 220 '
        r"rows repeat a value already used, e\.g\. 'u\d{3}'\. Name another "
        r'key with primary_keys=, or pass check_keys=False to skip \(one '
        r'full scan per table\)\.',
        message,
    ), message
    assert 'does not enforce' not in message


def test_a_composite_key_is_counted_as_one_value(build: Build) -> None:
    lines = USERS.assign(line=1)
    repeated = pd.concat(
        [lines, USERS.head(3).assign(line=2), USERS.head(1).assign(line=1)],
        ignore_index=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(
            build(repeated),
            TABLES,
            primary_keys={'users': ['user_id', 'line']},
        )

    assert schema['users'].primary_key == ('user_id', 'line')
    assert schema['users'].duplicate_keys == 1


# --- what is counted, and what is not -------------------------------------


@pytest.fixture
def statements(postgres: Any) -> list[str]:
    import psycopg

    said: list[str] = []

    class Recorded(psycopg.Cursor):
        def execute(self, query: Any, *args: Any, **kwargs: Any) -> Any:
            said.append(str(query))
            return super().execute(query, *args, **kwargs)

    connection, _ = postgres
    connection.cursor_factory = Recorded
    return said


def _counts(statements: list[str]) -> list[str]:
    return [sql for sql in statements if 'COUNT(CASE WHEN' in sql]


def test_one_statement_counts_every_table(
    build_postgres: Build, statements: list[str]
) -> None:
    source = build_postgres(USERS)
    statements.clear()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(source, TABLES)

    (counted,) = _counts(statements)
    assert counted.count('UNION ALL') == 1
    assert schema['users'].duplicate_keys == 0
    assert schema['orders'].duplicate_keys == 0


def test_a_key_the_source_enforces_is_not_counted(
    build_postgres: Build, statements: list[str]
) -> None:
    source = build_postgres(USERS, constrained=('app_users',))
    statements.clear()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(source, TABLES)

    (counted,) = _counts(statements)
    assert 'app_orders' in counted
    assert 'app_users' not in counted
    assert schema['users'].primary_key == 'user_id'
    assert schema['users'].duplicate_keys == 0


def test_nothing_is_counted_when_asked_not_to(
    build_postgres: Build, statements: list[str]
) -> None:
    source = build_postgres(_repeating('u007'))
    statements.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        schema = Schema.from_source(source, TABLES, check_keys=False)

    assert _counts(statements) == []
    assert not [w for w in caught if 'identifies a row' in str(w.message)]
    assert schema['users'].duplicate_keys is None
    assert schema['orders'].duplicate_keys is None


# --- what sampling does with a key that repeats anyway ----------------------


def _materialize(schema: Schema, query: str, ids: list[str], seed: int) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        request = PredictiveQuery(query, schema).materialize(
            indices=ids, random_seed=seed
        )
    (payload,) = request._payloads
    return json.loads(json.dumps(payload, default=str))


def _unchecked(build: Build, users: pd.DataFrame) -> Schema:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_source(build(users), TABLES, check_keys=False)


@pytest.mark.parametrize('query', [TEMPORAL, STATIC])
def test_an_id_two_rows_hold_cannot_be_scored(build: Build, query: str) -> None:
    schema = _unchecked(build, _repeating('u007'))

    with pytest.raises(StructuredDataError, match=r"Cannot score 'u007'"):
        _materialize(schema, query, ['u001', 'u007'], seed=7)
    assert _materialize(schema, query, ['u001'], seed=7)['predict']


@pytest.mark.parametrize('query', [TEMPORAL, STATIC])
@pytest.mark.parametrize('seed', [7, 11])
def test_the_context_holds_an_entity_once_at_each_anchor_time(
    build: Build, query: str, seed: int
) -> None:
    repeated = [f'u{i:03d}' for i in range(0, 200, 10)]
    schema = _unchecked(build, _repeating(*repeated))

    table = _materialize(schema, query, ['u001'], seed)['context'][
        'instance_table'
    ]
    entity = table['columns'].index('__kumo_entity_ref_0')
    anchor = table['columns'].index('__kumo_anchor_time')
    pairs = Counter((row[entity], row[anchor]) for row in table['rows'])

    assert {row[entity] for row in table['rows']} & set(repeated)
    assert [pair for pair, seen in pairs.items() if seen > 1] == []


# --- a schema pickled before it could have a source -------------------------


def test_a_schema_pickled_before_sources_still_loads() -> None:
    r"""A schema pickled by a release without sources held its tables and
    relationships and nothing else. It loads as the schema of frames it was.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data({'users': USERS, 'orders': ORDERS})
    # Its relationships and tables point back at it, so every reference is
    # written the old way, not only the outermost one.
    old_state = {
        key: value
        for key, value in vars(schema).items()
        if key not in ('_source', '_engine_tables')
    }

    class _AsReleased(pickle.Pickler):
        def reducer_override(self, obj: Any) -> Any:
            if obj is schema:
                return object.__new__, (Schema,), old_state
            return NotImplemented

    written = io.BytesIO()
    _AsReleased(written).dump(schema)
    loaded = pickle.loads(written.getvalue())

    assert isinstance(loaded, Schema)
    assert loaded.source is None
    assert loaded.relationships == schema.relationships
    assert loaded.snapshot()['users'].primary_key == 'user_id'
    assert _materialize(loaded, TEMPORAL, ['u001'], seed=7)['predict']
