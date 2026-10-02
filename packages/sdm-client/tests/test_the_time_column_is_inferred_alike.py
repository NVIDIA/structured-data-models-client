# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Frames and the same rows in a source infer the same time column.

The two routes disagreed. `from_data` took a lone timestamp unchecked and
none of several, and the engine behind `from_source` took the first of two
that tied and otherwise the one whose values start earliest. On
AdventureWorks that made `customers.ModifiedDate`, one date later than every
order, the time column, and every windowed question about a customer was
refused. Both now call one rule, and each case here is one where they once
disagreed or where the rule decides something new.
"""

from __future__ import annotations

import os
import uuid
import warnings
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

from sdm_client import PredictiveQuery, Schema  # noqa: E402

# More rows than the engine reads to describe a table, so a column constant
# in the rows read is checked against the whole table.
ROWS = 1_200
HOURS = pd.Series(pd.date_range('2023-01-01', periods=ROWS, freq='h'))
LATER = HOURS + pd.Timedelta(days=3)
ONE_DAY = pd.Series([pd.Timestamp('2025-09-11')] * ROWS)
LATE_CHANGE = pd.Series(
    [pd.Timestamp('2023-01-01')] * 1_000
    + list(pd.date_range('2023-02-01', periods=ROWS - 1_000, freq='D'))
)
# Later than HOURS in the rows the engine reads, and earlier past them.
EARLIEST_PAST_THE_ROWS_READ = pd.Series(
    list(HOURS[:1_000] + pd.Timedelta(days=2))
    + list(pd.date_range('2020-01-01', periods=ROWS - 1_000, freq='D'))
)
BIRTHS = pd.Series(pd.date_range('1960-01-01', periods=ROWS, freq='W'))

CASES: dict[str, tuple[dict[str, pd.Series], str | None]] = {
    'an order date beside an edit stamp': (
        {'ModifiedDate': HOURS, 'OrderDate': HOURS},
        'OrderDate',
    ),
    'a lone edit stamp, as in a log of changes': (
        {'updated_at': HOURS},
        'updated_at',
    ),
    'two edit stamps and nothing else': (
        {'updated_at': HOURS, 'last_login': LATER},
        None,
    ),
    'a lone load stamp': ({'_fivetran_synced': HOURS}, None),
    'a lone constant stamp': ({'ordered': ONE_DAY}, None),
    'a stamp constant only in its first rows': (
        {'ordered': LATE_CHANGE},
        'ordered',
    ),
    'a creation stamp beside another': (
        {'shipped_at': HOURS, 'created_at': LATER},
        'created_at',
    ),
    'two event stamps': (
        {'shipped_at': LATER, 'ordered_at': HOURS},
        'ordered_at',
    ),
    'an order header': (
        {
            'OrderDate': HOURS,
            'DueDate': LATER + pd.Timedelta(days=9),
            'ShipDate': LATER,
            'ModifiedDate': LATER,
        },
        'OrderDate',
    ),
    'the earliest only past the rows read': (
        {'ordered_at': HOURS, 'shipped_at': EARLIEST_PAST_THE_ROWS_READ},
        'shipped_at',
    ),
    'a hire date beside a birth date': (
        {'BirthDate': BIRTHS, 'HireDate': HOURS},
        'HireDate',
    ),
    'a lone date of birth': ({'dob': BIRTHS}, None),
}


def _frames(
    stamps: dict[str, pd.Series], rows: int = ROWS
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(0)
    customers = pd.DataFrame(
        {
            'customer_id': range(50),
            'ModifiedDate': [pd.Timestamp('2025-09-11')] * 50,
        }
    )
    orders = pd.DataFrame(
        {
            'order_id': range(rows),
            'customer_id': rng.integers(0, 50, rows),
            'amount': rng.gamma(2, 20, rows).round(2),
            **{name: values[:rows] for name, values in stamps.items()},
        }
    )
    return {'customers': customers, 'orders': orders}


TIME_COLUMN_NOTICE = (
    'Some time columns were inferred, or refused, on a guess about what a '
    'column holds:\n'
)

Build = Callable[[dict[str, pd.DataFrame]], Any]


@pytest.fixture
def build_duckdb() -> Iterator[Build]:
    connections = []

    def build(frames: dict[str, pd.DataFrame]) -> Any:
        connection = duckdb.connect()
        connections.append(connection)
        for name, frame in frames.items():
            connection.register(f'{name}_frame', frame)
            connection.execute(
                f'CREATE TABLE {name} AS SELECT * FROM {name}_frame'
            )
        return DuckDBSource(connection)

    yield build
    for connection in connections:
        connection.close()


@pytest.fixture
def build_sqlite(tmp_path: Path) -> Iterator[Build]:
    connections = []

    def build(frames: dict[str, pd.DataFrame]) -> Any:
        connection = dbapi.connect(str(tmp_path / f'{uuid.uuid4().hex}.db'))
        connections.append(connection)
        with connection.cursor() as cursor:
            for name, frame in frames.items():
                cursor.adbc_ingest(name, frame, mode='create')
        connection.commit()
        return SQLiteSource(connection)

    yield build
    for connection in connections:
        connection.close()


def _postgres_type(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series):
        return 'BIGINT'
    if pd.api.types.is_float_dtype(series):
        return 'DOUBLE PRECISION'
    return 'TIMESTAMP'


@pytest.fixture
def build_postgres() -> Iterator[Build]:
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

        def build(frames: dict[str, pd.DataFrame]) -> Any:
            for name, frame in frames.items():
                columns = ', '.join(
                    f'"{column}" {_postgres_type(frame[column])}'
                    for column in frame.columns
                )
                connection.execute(
                    f'CREATE TABLE {namespace}.{name} ({columns})'
                )
                markers = ', '.join(['%s'] * len(frame.columns))
                rows = frame.astype(object).to_numpy().tolist()
                with connection.cursor() as cursor:
                    cursor.executemany(
                        f'INSERT INTO {namespace}.{name} VALUES ({markers})',
                        rows,
                    )
            return PostgresSource(connection, schema=namespace)

        try:
            yield build
        finally:
            connection.execute(f'DROP SCHEMA {namespace} CASCADE')


@pytest.fixture(params=['duckdb', 'sqlite', 'postgres'])
def build(request: pytest.FixtureRequest) -> Build:
    return request.getfixturevalue(f'build_{request.param}')


def _described(describe: Callable[[], Schema]) -> tuple[Schema, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        schema = describe()
    said = [
        str(item.message)
        for item in caught
        if str(item.message).startswith(TIME_COLUMN_NOTICE)
    ]
    return schema, said


@pytest.mark.parametrize('case', CASES)
def test_a_source_infers_the_time_column_its_frames_do(
    build: Build, case: str
) -> None:
    stamps, expected = CASES[case]
    frames = _frames(stamps)

    in_memory, said_in_memory = _described(lambda: Schema.from_data(frames))
    in_place, said_in_place = _described(
        lambda: Schema.from_source(build(frames))
    )

    for schema in (in_memory, in_place):
        assert schema['orders'].time_column == expected
        assert schema['customers'].time_column is None
    assert said_in_place == said_in_memory


def test_a_one_row_table_keeps_its_timestamp(build: Build) -> None:
    """One value is no evidence that a column never changes."""
    frames = _frames({'ordered': ONE_DAY}, rows=1)
    in_memory, _ = _described(lambda: Schema.from_data(frames))
    in_place, _ = _described(lambda: Schema.from_source(build(frames)))
    assert in_memory['orders'].time_column == 'ordered'
    assert in_place['orders'].time_column == 'ordered'


def test_every_guess_is_named_in_one_notice_with_why_and_the_override() -> None:
    frames = _frames({'ordered': ONE_DAY})
    frames['order_status_updates'] = pd.DataFrame(
        {'order_id': range(ROWS), 'updated_at': HOURS}
    )
    frames['shipments'] = pd.DataFrame(
        {'order_id': range(ROWS), 'shipped_at': LATER, 'packed_at': HOURS}
    )
    _, said = _described(lambda: Schema.from_data(frames))
    assert said == [
        TIME_COLUMN_NOTICE
        + '- customers.ModifiedDate was not taken as the time column because '
        'every row holds 2025-09-11 00:00:00. If it does say when a row '
        "became true, name it: time_columns={'customers': 'ModifiedDate'}.\n"
        '- order_status_updates.updated_at was taken as the time column '
        'although it looks like when a row was last changed, because the '
        'table has no other timestamp. If it does say when a row became '
        'true, name it: '
        "time_columns={'order_status_updates': 'updated_at'}.\n"
        '- orders.ordered was not taken as the time column because every row '
        'holds 2025-09-11 00:00:00. If it does say when a row became true, '
        "name it: time_columns={'orders': 'ordered'}.\n"
        '- shipments.packed_at was taken as the time column because its '
        "values start earliest of 'shipped_at' and 'packed_at'. If another "
        'says when a row became true, name it: '
        "time_columns={'shipments': 'shipped_at'}."
    ]


def test_a_table_given_its_time_column_is_not_told_about_the_others() -> None:
    _, said = _described(
        lambda: Schema.from_data(
            _frames({'ModifiedDate': HOURS}),
            time_columns={'orders': 'ModifiedDate'},
        )
    )
    assert not any('- orders.' in notice for notice in said)


def test_a_customer_can_be_counted_once_its_edit_stamp_is_refused(
    build_duckdb: Build,
) -> None:
    """The AdventureWorks question, refused while `customers.ModifiedDate`
    was its time column."""
    frames = _frames({'OrderDate': HOURS, 'ModifiedDate': HOURS})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(build_duckdb(frames))
        request = PredictiveQuery(
            'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH '
            'customers.customer_id',
            schema,
        ).materialize(indices=[1, 2, 3])
    assert request is not None
