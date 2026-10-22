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

import ast
import functools
import os
import re
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

from sdm_client import PredictiveQuery, Relationship, Schema  # noqa: E402

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
# The epoch where no order was cancelled, which is most of them.
PLACEHOLDERS = pd.Series([pd.Timestamp('1970-01-01')] * ROWS).where(
    pd.Series(range(ROWS)) % 50 != 0, HOURS + pd.Timedelta(days=1)
)
# Set on one order in a hundred, and earlier than every order date.
CANCELLED = (HOURS - pd.Timedelta(days=30)).where(
    pd.Series(range(ROWS)) % 100 == 0
)

CASES: dict[str, tuple[dict[str, pd.Series], str | None]] = {
    'an order date beside an edit stamp': (
        {'ModifiedDate': HOURS, 'OrderDate': HOURS},
        'OrderDate',
    ),
    'a lone edit stamp on a table nothing links to': (
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
    'a mostly empty stamp beside the order date': (
        {'cancelled_at': CANCELLED, 'order_date': HOURS},
        'order_date',
    ),
    'a lone mostly empty stamp': ({'cancelled_at': CANCELLED}, None),
    'a placeholder-filled stamp beside the order date': (
        {'order_date': HOURS, 'cancelled_at': PLACEHOLDERS},
        'order_date',
    ),
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

#: The code the core's route puts before a warning's words (decisions
#: E41(i), E59(d)); the two routes are compared on the words after it.
CORE_CODE = re.compile(r'^\[[A-Z][A-Z0-9_]*\] ')


def _bare(message: object) -> str:
    r"""A warning's words, without the core route's leading ``[CODE] ``."""
    return CORE_CODE.sub('', str(message))


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
    if pd.api.types.is_string_dtype(series):
        return 'TEXT'
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
                held = frame.astype(object)
                rows = held.where(frame.notna(), None).to_numpy().tolist()
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
        _bare(item.message)
        for item in caught
        if _bare(item.message).startswith(TIME_COLUMN_NOTICE)
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


@pytest.mark.parametrize('surrogate', [False, True])
@pytest.mark.parametrize('stamp', ['last_login', 'deleted_at'])
def test_a_log_takes_its_edit_stamp_and_a_linked_table_does_not(
    build: Build, stamp: str, surrogate: bool
) -> None:
    """`order_status_updates` appends a row per change, and nothing links
    to it, so its `updated_at` is when each row was written. `users` holds
    one row per user, edited in place, and `orders` links to it, so its
    `last_login` or `deleted_at` is no user's time.

    With an `update_id`, a frame takes it as the log's key and a source
    does not, which is why the key cannot decide this.
    """
    stamps = pd.Series(pd.date_range('2024-01-01', periods=50, freq='D'))
    log = {
        'order_id': [i % 100 for i in range(ROWS)],
        'updated_at': HOURS,
    }
    if surrogate:
        log = {'update_id': range(ROWS), **log}
    frames = {
        'users': pd.DataFrame({'user_id': range(50), stamp: stamps}),
        'orders': pd.DataFrame(
            {'order_id': range(100), 'user_id': [i % 50 for i in range(100)]}
        ),
        'order_status_updates': pd.DataFrame(log),
    }
    in_memory, said_in_memory = _described(lambda: Schema.from_data(frames))
    in_place, said_in_place = _described(
        lambda: Schema.from_source(build(frames))
    )
    for schema in (in_memory, in_place):
        assert schema['users'].time_column is None
        assert schema['order_status_updates'].time_column == 'updated_at'
    assert said_in_place == said_in_memory


def test_a_link_the_caller_names_refuses_the_edit_stamp_too(
    build: Build,
) -> None:
    """The links decide, and the caller's are the final ones."""
    frames = {
        'accounts': pd.DataFrame(
            {'handle': [f'h{i}' for i in range(50)], 'updated_at': HOURS[:50]}
        ),
        'logins': pd.DataFrame(
            {'account': [f'h{i % 50}' for i in range(ROWS)], 'at': HOURS}
        ),
    }
    keys = {'accounts': 'handle'}
    links = [Relationship('logins', 'account', 'accounts', 'handle')]
    source = build(frames)
    for describe in (
        functools.partial(Schema.from_data, frames, primary_keys=keys),
        functools.partial(Schema.from_source, source, primary_keys=keys),
    ):
        assert describe()['accounts'].time_column == 'updated_at'
        linked = _described(functools.partial(describe, relationships=links))
        assert linked[0]['accounts'].time_column is None


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
        {'order_id': [i % 100 for i in range(ROWS)], 'updated_at': HOURS}
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
        'table has no other timestamp and no table links to it, so it reads '
        'as a log of changes. If it does not say when a row became true, clear '
        "it: time_columns={'order_status_updates': None}.\n"
        '- orders.ordered was not taken as the time column because every row '
        'holds 2025-09-11 00:00:00. If it does say when a row became true, '
        "name it: time_columns={'orders': 'ordered'}.\n"
        '- shipments.packed_at was taken as the time column because its '
        "values start earliest of 'shipped_at' and 'packed_at'. If another "
        'says when a row became true, name it: '
        "time_columns={'shipments': 'shipped_at'}. If none does, clear it: "
        "time_columns={'shipments': None}."
    ]


def _suggested(notice: str) -> list[dict[str, str | None]]:
    return [
        ast.literal_eval(found)
        for found in re.findall(r'time_columns=(\{[^}]*\})', notice)
    ]


def test_every_override_a_notice_suggests_does_what_it_says(
    build: Build,
) -> None:
    """Each remedy is applied, on both paths, and the table it names ends
    with the time column it says and leaves the notice."""
    frames = _frames({'ordered': ONE_DAY})
    frames['order_status_updates'] = pd.DataFrame(
        {'order_id': [i % 100 for i in range(ROWS)], 'updated_at': HOURS}
    )
    frames['shipments'] = pd.DataFrame(
        {'order_id': range(ROWS), 'shipped_at': LATER, 'packed_at': HOURS}
    )
    _, said = _described(lambda: Schema.from_data(frames))
    suggested = _suggested(said[0])
    assert len(suggested) == 5
    assert {'order_status_updates': None} in suggested
    assert {'shipments': None} in suggested
    source = build(frames)
    for override in suggested:
        [(table, column)] = override.items()
        for schema, notices in (
            _described(
                functools.partial(
                    Schema.from_data, frames, time_columns=override
                )
            ),
            _described(
                functools.partial(
                    Schema.from_source, source, time_columns=override
                )
            ),
        ):
            assert schema[table].time_column == column
            assert not any(f'- {table}.' in notice for notice in notices)


def test_a_table_given_its_time_column_is_not_told_about_the_others() -> None:
    _, said = _described(
        lambda: Schema.from_data(
            _frames({'ModifiedDate': HOURS}),
            time_columns={'orders': 'ModifiedDate'},
        )
    )
    assert not any('- orders.' in notice for notice in said)


def test_a_column_the_whole_table_could_not_check_is_kept_and_named(
    build_duckdb: Build, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The query over the whole table times out, as it can on a large
    view. Every row read holds one day, which the first rows of a table
    kept in time order often do, so the column is kept, and said.
    """
    from sdm_relational_engine.rfm.backend.duckdb import DuckDBTable

    fetch = DuckDBTable._fetch_rows

    def timed_out(self: DuckDBTable, sql: str) -> list[tuple]:
        if sql.startswith('SELECT COUNT(*), MIN('):
            raise RuntimeError('canceling statement due to statement timeout')
        return fetch(self, sql)

    monkeypatch.setattr(DuckDBTable, '_fetch_rows', timed_out)
    source = build_duckdb(_frames({'ordered': ONE_DAY}))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        schema = Schema.from_source(source)
    assert schema['orders'].time_column == 'ordered'
    # Said once, by the schema, not again by the engine it reads through.
    said = [
        _bare(item.message)
        for item in caught
        if 'orders.ordered' in str(item.message)
    ]
    assert said == [
        TIME_COLUMN_NOTICE
        + '- customers.ModifiedDate was not taken as the time column because '
        'every row holds 2025-09-11 00:00:00. If it does say when a row '
        "became true, name it: time_columns={'customers': 'ModifiedDate'}.\n"
        '- orders.ordered was taken as the time column although every one of '
        'the 1,000 rows read holds 2025-09-11 00:00:00, because the query '
        'over the whole table failed and the rows past them were not '
        'checked. If the whole table is like that, clear it: '
        "time_columns={'orders': None}."
    ]


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
