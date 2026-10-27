# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Record the reference's answers for ``test_t38_forced_routes.py``.

Run with the reference checkout's interpreter, from this repository's root,
with a throwaway Postgres named (a ``postgres:17`` container, as CI runs)::

    SDM_POSTGRES_TEST_DSN=postgresql://... \
    ~/code/sdmc-reference/.venv/bin/python -I \
        packages/structured-data-models-client/tests/t38_forced_routes_record.py

T3.8 closes Phase 2's flip dry run (every core route forced:
``docs/design/rust-core/phase-2/evidence/flip/``), whose 118 failures fall
in the families of the carried rows P2-T2.16-9, -10 and -11. Each scenario
here is one input of one family, as a user meets it: frames or statements
loaded into DuckDB, SQLite and Postgres (a schema of its own, as the flip's
tests load it), then the public calls (``Schema.from_source``,
``Schema(...)`` with :class:`SourceTable`, ``PredictiveQuery.materialize``).
The reference answers each on each database, and this script keeps what
the family is about (:func:`observe`): the schema's keys, time columns,
duplicate counts, columns and relationships, the family's notices, a
refusal's class, code and words, and a payload's parts that do not depend
on the draw. Nothing here is written by hand but the inputs; every expected
value is the reference's. A repeated key's example (``e.g. ...``) is
whichever repeated value the reference's scan met first, which differs
from run to run on DuckDB; ``examples`` records every value it could have
been, and the test reads any of them as the reference's.

The families (:data:`FAMILIES`), each scenario naming its own:

- ``time``: P2-T2.16-9, the "Some time columns..." notice and the time
  columns it explains, over one candidate and several, the earliest, an
  edit stamp, a load stamp, a constant, a date of birth, a mostly empty
  stamp, a placeholder, all rejected and a caller-named one; a timestamp,
  a zoned timestamp, a date and text; every table listed and the tables
  named; links inferred and given. Two lines of the notice are out of
  reach here: ``instead_of`` (a dimension a metric or semantic view
  derives) and ``unconfirmed`` (the whole-table query failing, which only
  the old engine's internals can make happen).
- ``keys``: P2-T2.16-10, a key counted where its rows are: stated
  (``primary_keys=``), inferred and declared in the catalog; no repeat,
  one, exactly 1% and just over; a value missing; composite, decimal and
  past 2^53; ``check_keys=False``; and an id two rows hold, asked to be
  scored.
- ``refusals``: P2-T2.16-10 and -11, the reference's refusals of a source
  call in its own words, not the core's generic "the request is not one
  the core accepts".
- ``postgres``: P2-T2.16-11, a ``PostgresSource`` read in the schema it
  names (``schema=``, or the connection's ``search_path``), never in
  another that holds a table of the same name.
- ``described``: P2-T2.16-11, ``materialize`` of a schema the core did not
  describe: built by hand, or described on the old route (the reference
  materializes any :class:`Schema`, ``query.py:87``).

``observe`` and the scenarios are imported by the test, so both routes are
observed by the same code. The recorded file is
``t38-forced-routes.json``, keyed ``<scenario>@<database>``.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import sys
import uuid
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

OUT = Path(__file__).with_name('t38-forced-routes.json')
#: The variable forcing paths onto the core (``sdm_client._routing``), which
#: the reference does not have.
ROUTES = 'SDM_CORE_ROUTES'
DATABASES = ('duckdb', 'sqlite', 'postgres')

#: Each family's carried row and what it covers, as the brief words them.
FAMILIES = {
    'time': (
        'P2-T2.16-9: the "Some time columns..." warning the old route gives, '
        'and the time columns it explains'
    ),
    'keys': (
        'P2-T2.16-10: a key counted where its rows are, its 1% and 99% '
        'thresholds, its warnings and its refusals'
    ),
    'refusals': (
        'P2-T2.16-10 and -11: a source call refused INVALID_REQUEST in the '
        "reference's words"
    ),
    'postgres': 'P2-T2.16-11: a Postgres source read in the schema it names',
    'described': (
        'P2-T2.16-11: materialize of a schema not described on the core '
        '(RouteError)'
    ),
}

#: The notices each family compares, by their words (the core's ``[CODE] ``
#: prefix aside, decisions E41(i) and E59(d)).
NOTICES = {
    'time': r'^Some time columns were inferred',
    'keys': r'identifies a row|named like a key|no value|not unique enough',
    'refusals': r'identifies a row|named like a key|no value',
    'postgres': (
        r'^Some time columns|identifies a row|named like a key|no value|'
        r'refers to|cannot match'
    ),
    'described': r'identifies a row|named like a key|no value|refers to',
}

ANCHOR = '2024-12-01'
TEMPORAL = 'PREDICT SUM(orders.amount, 0, 30, days) FOR EACH users.user_id'
STATIC = 'PREDICT users.country FOR EACH users.user_id'

# --- the data ---------------------------------------------------------------

ROWS = 1_200
HOURS = pd.Series(pd.date_range('2023-01-01', periods=ROWS, freq='h'))
LATER = HOURS + pd.Timedelta(days=3)
ONE_DAY = pd.Series([pd.Timestamp('2025-09-11')] * ROWS)
LATE_CHANGE = pd.Series(
    [pd.Timestamp('2023-01-01')] * 1_000
    + list(pd.date_range('2023-02-01', periods=ROWS - 1_000, freq='D'))
)
EARLIEST_PAST_THE_ROWS_READ = pd.Series(
    list(HOURS[:1_000] + pd.Timedelta(days=2))
    + list(pd.date_range('2020-01-01', periods=ROWS - 1_000, freq='D'))
)
BIRTHS = pd.Series(pd.date_range('1960-01-01', periods=ROWS, freq='W'))
PLACEHOLDERS = pd.Series([pd.Timestamp('1970-01-01')] * ROWS).where(
    pd.Series(range(ROWS)) % 50 != 0, HOURS + pd.Timedelta(days=1)
)
CANCELLED = (HOURS - pd.Timedelta(days=30)).where(
    pd.Series(range(ROWS)) % 100 == 0
)
DAYS = pd.Series(pd.date_range('2020-01-01', periods=ROWS, freq='D'))


def _text(stamps: pd.Series) -> pd.Series:
    return stamps.dt.strftime('%Y-%m-%d %H:%M:%S')


def _dates(stamps: pd.Series) -> pd.Series:
    return pd.Series([value.date() for value in stamps], dtype=object)


def _orders_and_customers(
    stamps: dict[str, pd.Series], rows: int = ROWS
) -> dict[str, pd.DataFrame]:
    r"""``customers`` with a constant edit stamp, and ``orders`` linking to
    it with ``stamps`` (``test_the_time_column_is_inferred_alike.py``'s
    frames)."""
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


def _users_and_orders() -> tuple[pd.DataFrame, pd.DataFrame]:
    r"""``test_a_key_is_counted_where_its_rows_are.py``'s frames: 200 users
    keyed by text, 600 orders over a year."""
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


USERS, ORDERS = _users_and_orders()


def _day(i: int) -> str:
    r"""Midnight ``i`` days after 2024-01-01, as a literal each database
    reads as a timestamp."""
    return str(pd.Timestamp('2024-01-01') + pd.Timedelta(days=i))


def _repeating(*ids: str, users: pd.DataFrame = USERS) -> pd.DataFrame:
    r"""``users`` with each of ``ids`` held by a second, identical row."""
    copies = users[users.user_id.isin(ids)]
    return pd.concat([users, copies], ignore_index=True)


def _missing_one(users: pd.DataFrame = USERS) -> pd.DataFrame:
    row = pd.DataFrame({'user_id': [None], 'age': [40], 'country': ['US']})
    return pd.concat([users, row], ignore_index=True)


def _users_and_orders_with(users: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {'app_users': users, 'app_orders': ORDERS}


#: The users of ``_users_and_orders``, 198 of them, so two repeated rows
#: are exactly 1% of 200 and three are just over 1% of 201.
FEWER = USERS.head(198)


def _numbered_users(ids: list[Any], *, dtype: str) -> pd.DataFrame:
    r"""Users keyed by ``ids`` of ``dtype``, each with an order."""
    users = pd.DataFrame(
        {
            'user_id': pd.Series(ids, dtype=dtype),
            'country': ['US', 'DE'] * (len(ids) // 2) + ['IN'] * (len(ids) % 2),
        }
    )
    return users


# --- loading the data where a user keeps it ---------------------------------


@dataclass
class Place:
    r"""Where a scenario's tables were loaded: its source, and what built
    it."""

    database: str
    source: Any
    connection: Any
    #: The Postgres schema the tables are in, or ``None``.
    namespace: str | None = None
    #: Postgres tables created outside ``namespace`` (``public``), dropped
    #: after the scenario.
    outside: list[str] = field(default_factory=list)
    #: Names unique to this run, by the scenario's name for them.
    names: dict[str, str] = field(default_factory=dict)

    def qualified(self, name: str) -> str:
        return f'{self.namespace}.{name}' if self.namespace else name


def _postgres_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return 'BOOLEAN'
    if pd.api.types.is_integer_dtype(series):
        return 'BIGINT'
    if pd.api.types.is_float_dtype(series):
        return 'DOUBLE PRECISION'
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return 'TIMESTAMPTZ'
    if pd.api.types.is_datetime64_any_dtype(series):
        return 'TIMESTAMP'
    sample = series.dropna()
    if len(sample) and isinstance(sample.iloc[0], dt.date):
        return 'DATE'
    if len(sample) and isinstance(sample.iloc[0], Decimal):
        return 'NUMERIC(38, 0)'
    return 'TEXT'


def _python_value(value: Any) -> Any:
    # psycopg adapts Python values, not numpy's or pandas'.
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_frames(place: Place, frames: dict[str, pd.DataFrame]) -> None:
    r"""Each frame as a table of its name, its column types the frame's,
    as the flip's tests load them."""
    connection = place.connection
    for name, frame in frames.items():
        if place.database == 'duckdb':
            connection.register(f'{name}_frame', frame)
            connection.execute(
                f'CREATE TABLE {name} AS SELECT * FROM {name}_frame'
            )
            connection.unregister(f'{name}_frame')
        elif place.database == 'sqlite':
            with connection.cursor() as cursor:
                cursor.adbc_ingest(name, frame, mode='create')
            connection.commit()
        else:
            columns = ', '.join(
                f'"{column}" {_postgres_type(frame[column])}'
                for column in frame.columns
            )
            connection.execute(
                f'CREATE TABLE {place.qualified(name)} ({columns})'
            )
            markers = ', '.join(['%s'] * len(frame.columns))
            rows = [
                tuple(_python_value(value) for value in row)
                for row in frame.astype(object).itertuples(index=False)
            ]
            with connection.cursor() as cursor:
                cursor.executemany(
                    f'INSERT INTO {place.qualified(name)} VALUES ({markers})',
                    rows,
                )


def run_statements(place: Place, statements: list[str]) -> None:
    r"""``statements``, each with ``{in_}`` naming where tables go."""
    prefix = f'{place.namespace}.' if place.namespace else ''
    for statement in statements:
        sql = statement.format(in_=prefix, **place.names)
        if place.database == 'sqlite':
            with place.connection.cursor() as cursor:
                cursor.execute(sql)
            place.connection.commit()
        else:
            place.connection.execute(sql)


def _postgres_dsn() -> str | None:
    return os.environ.get('SDM_POSTGRES_TEST_DSN') or None


@contextmanager
def placed(database: str, scratch: Path) -> Iterator[Place]:
    r"""An empty place in ``database``: an in-memory DuckDB, a SQLite file
    under ``scratch``, or a Postgres schema of its own, dropped after."""
    from sdm_connectors import DuckDBSource, PostgresSource, SQLiteSource

    if database == 'duckdb':
        import duckdb

        duck = duckdb.connect()
        try:
            yield Place(database, DuckDBSource(duck), duck)
        finally:
            duck.close()
    elif database == 'sqlite':
        from adbc_driver_sqlite import dbapi

        path = scratch / f'{uuid.uuid4().hex}.sqlite'
        lite = dbapi.connect(str(path))
        try:
            yield Place(database, SQLiteSource(lite), lite)
        finally:
            lite.close()
    else:
        import psycopg

        dsn = _postgres_dsn()
        assert dsn is not None, 'SDM_POSTGRES_TEST_DSN names no server'
        namespace = f'sdm_{uuid.uuid4().hex[:8]}'
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(f'CREATE SCHEMA {namespace}')
            place = Place(
                database,
                PostgresSource(connection, schema=namespace),
                connection,
                namespace,
            )
            try:
                yield place
            finally:
                connection.execute('RESET search_path')
                for name in place.outside:
                    connection.execute(f'DROP TABLE IF EXISTS {name} CASCADE')
                connection.execute(f'DROP SCHEMA {namespace} CASCADE')


# --- what a call answered ---------------------------------------------------


def bare(message: str) -> str:
    r"""A notice's words without the core's ``[CODE] `` prefix, which every
    warning the core gives carries (decisions E41(i), E59(d))."""
    return re.sub(r'^\[[A-Z][A-Z0-9_]*\] ', '', message)


def _refusal(error: BaseException) -> dict[str, Any]:
    return {
        'outcome': 'refusal',
        'class': type(error).__name__,
        'code': getattr(error, 'code', None),
        'message': str(error),
    }


def _key(value: Any) -> Any:
    if value is None or isinstance(value, str):
        return value
    return list(value)


def _schema(schema: Any) -> dict[str, Any]:
    return {
        'tables': {
            name: {
                'source_name': getattr(table, 'source_name', None),
                'primary_key': _key(table.primary_key),
                'time_column': table.time_column,
                'duplicate_keys': getattr(table, 'duplicate_keys', None),
                'columns': sorted(table.columns),
            }
            for name, table in sorted(schema.tables.items())
        },
        'relationships': sorted(
            [
                link.source_table,
                link.source_column,
                link.target_table,
                link.target_column,
            ]
            for link in schema.relationships
        ),
    }


@contextmanager
def _noticed(pattern: str | None, into: list[list[str]]) -> Iterator[None]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        yield
    if pattern is None:
        return
    into.extend(
        sorted(
            [type(item.message).__name__, bare(str(item.message))]
            for item in caught
            if re.search(pattern, bare(str(item.message)))
        )
    )


def _rows(table: dict[str, Any] | None) -> Any:
    if table is None:
        return None
    return {
        'columns': list(table['columns']),
        'rows': sorted(json.dumps(row, default=str) for row in table['rows']),
    }


def _shape(tables: dict[str, Any] | None) -> Any:
    return {
        name: sorted(table['columns'])
        for name, table in sorted((tables or {}).items())
    }


def payload_parts(request: Any) -> dict[str, Any]:
    r"""What a materialized request holds that the draw does not decide:
    its task (a classification's classes as a set), the scored rows and
    their history, and the context's tables and columns. A context row, and
    the order the classes are met in, are the draw's, which SQLite cannot
    seed."""
    (payload,) = request._payloads
    payload = json.loads(json.dumps(payload, default=str))
    predict = payload.get('predict') or {}
    context = payload.get('context') or {}
    task = payload.get('task') or {}
    target = task.get('target') or {}
    if isinstance(target.get('classes'), list):
        # Their order is the context draw's, which SQLite cannot seed.
        target['classes'] = sorted(target['classes'], key=str)
    return {
        'outcome': 'payload',
        'parts': len(request._payloads),
        'task': task,
        'predict': _rows(predict.get('instance_table')),
        'predict_related': {
            name: _rows(table)
            for name, table in sorted(
                (predict.get('related_tables') or {}).items()
            )
        },
        'context_columns': list(
            (context.get('instance_table') or {}).get('columns') or []
        ),
        'context_related': _shape(context.get('related_tables')),
    }


def materialized(
    schema: Any,
    query: str,
    indices: list[Any],
    *,
    seed: int = 7,
) -> dict[str, Any]:
    from sdm_client import PredictiveQuery

    request = PredictiveQuery(query, schema).materialize(
        indices=indices,
        random_seed=seed,
        anchor_time=pd.Timestamp(ANCHOR),
    )
    return payload_parts(request)


# --- the scenarios ----------------------------------------------------------

Act = Callable[[Place], Any]


@dataclass
class Scenario:
    r"""One input of one family: what is loaded, and the calls made."""

    name: str
    family: str
    #: What the scenario holds the core to, in a line.
    covers: str
    #: The calls, in the place the data was loaded; their answer, or the
    #: scenario's steps' answers as a dict.
    act: Act
    #: Frames loaded before ``act``, or statements run before it.
    frames: Callable[[], dict[str, pd.DataFrame]] | None = None
    statements: list[str] = field(default_factory=list)
    databases: tuple[str, ...] = DATABASES
    #: Names made unique per run (``{name}`` in ``statements``).
    unique: tuple[str, ...] = ()
    #: The values a refusal or notice may give as its example (``e.g.``),
    #: when a key repeats: any repeated value, as a scan meets it.
    examples: Callable[[], list[str]] | None = None


SCENARIOS: dict[str, Scenario] = {}


def scenario(*args: Any, **kwargs: Any) -> Scenario:
    found = Scenario(*args, **kwargs)
    assert found.name not in SCENARIOS, found.name
    assert found.family in FAMILIES, found.family
    SCENARIOS[found.name] = found
    return found


def describe(
    pattern: str | None,
    call: Callable[[], Any],
) -> dict[str, Any]:
    r"""``call``'s schema and the notices ``pattern`` selects, or its
    refusal."""
    from sdm_client import StructuredDataError

    notices: list[list[str]] = []
    try:
        with _noticed(pattern, notices):
            schema = call()
    except StructuredDataError as error:
        return {**_refusal(error), 'notices': notices}
    return {'outcome': 'schema', **_schema(schema), 'notices': notices}


def answer(call: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    r"""``call``'s answer, or its refusal (a ``StructuredDataError``; any
    other exception, the core's ``RouteError`` among them, propagates)."""
    from sdm_client import StructuredDataError

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return call()
    except StructuredDataError as error:
        return _refusal(error)


def from_source(
    family: str, tables: dict[str, str] | None = None, **options: Any
) -> Act:
    r"""``Schema.from_source`` over the place's source."""
    from sdm_client import Schema

    def act(place: Place) -> Any:
        given = (
            None
            if tables is None
            else {k: v.format(**place.names) for k, v in tables.items()}
        )
        return describe(
            NOTICES[family],
            lambda: Schema.from_source(place.source, given, **options),
        )

    return act


# The time column (P2-T2.16-9). ``test_the_time_column_is_inferred_alike.py``'s
# cases, and the types and modes its frames do not reach.

_TIME_CASES: dict[str, dict[str, pd.Series]] = {
    'an-order-date-beside-an-edit-stamp': {
        'ModifiedDate': HOURS,
        'OrderDate': HOURS,
    },
    'a-lone-edit-stamp-on-a-table-nothing-links-to': {'updated_at': HOURS},
    'two-edit-stamps-and-nothing-else': {
        'updated_at': HOURS,
        'last_login': LATER,
    },
    'a-lone-load-stamp': {'_fivetran_synced': HOURS},
    'a-lone-constant-stamp': {'ordered': ONE_DAY},
    'a-stamp-constant-only-in-its-first-rows': {'ordered': LATE_CHANGE},
    'a-creation-stamp-beside-another': {
        'shipped_at': HOURS,
        'created_at': LATER,
    },
    'two-event-stamps': {'shipped_at': LATER, 'ordered_at': HOURS},
    'an-order-header': {
        'OrderDate': HOURS,
        'DueDate': LATER + pd.Timedelta(days=9),
        'ShipDate': LATER,
        'ModifiedDate': LATER,
    },
    'the-earliest-only-past-the-rows-read': {
        'ordered_at': HOURS,
        'shipped_at': EARLIEST_PAST_THE_ROWS_READ,
    },
    'a-hire-date-beside-a-birth-date': {'BirthDate': BIRTHS, 'HireDate': HOURS},
    'a-lone-date-of-birth': {'dob': BIRTHS},
    'a-mostly-empty-stamp-beside-the-order-date': {
        'cancelled_at': CANCELLED,
        'order_date': HOURS,
    },
    'a-lone-mostly-empty-stamp': {'cancelled_at': CANCELLED},
    'a-placeholder-filled-stamp-beside-the-order-date': {
        'order_date': HOURS,
        'cancelled_at': PLACEHOLDERS,
    },
    # Every timestamp of one table refused, each for its own reason.
    'every-candidate-rejected': {
        'dob': BIRTHS,
        '_fivetran_synced': HOURS,
        'ordered': ONE_DAY,
    },
    # The earliest of two zoned timestamps.
    'two-zoned-stamps': {
        'ordered_at': HOURS.dt.tz_localize('UTC'),
        'shipped_at': LATER.dt.tz_localize('UTC'),
    },
    # The earliest of two text columns holding timestamps.
    'two-text-stamps': {
        'ordered_at': _text(HOURS),
        'shipped_at': _text(LATER),
    },
    # The earliest of two dates.
    'two-dates': {
        'ordered_on': _dates(DAYS),
        'shipped_on': _dates(DAYS + pd.Timedelta(days=2)),
    },
    # A date beside the timestamp it truncates.
    'a-date-beside-its-timestamp': {
        'ordered_at': HOURS,
        'ordered_on': _dates(HOURS.dt.normalize()),
    },
}

for _case, _stamps in _TIME_CASES.items():
    for _mode, _tables in (
        ('every-table', None),
        ('tables-named', {'customers': 'customers', 'orders': 'orders'}),
    ):
        scenario(
            f'time.{_case}.{_mode}',
            'time',
            f'the time notice and time columns: {_case.replace("-", " ")}, '
            f'{_mode.replace("-", " ")}',
            from_source('time', _tables),
            frames=lambda stamps=_stamps: _orders_and_customers(stamps),
        )

scenario(
    'time.the-caller-names-one-table',
    'time',
    'a table whose time column the caller named gets no line; the others do',
    from_source('time', None, time_columns={'orders': 'shipped_at'}),
    frames=lambda: _orders_and_customers(
        {'ordered_at': HOURS, 'shipped_at': LATER}
    ),
)
scenario(
    'time.the-caller-clears-one-table',
    'time',
    'a table the caller cleared gets no line; the others do',
    from_source('time', None, time_columns={'customers': None}),
    frames=lambda: _orders_and_customers(
        {'ordered_at': HOURS, 'shipped_at': LATER}
    ),
)
scenario(
    'time.a-one-row-table-keeps-its-timestamp',
    'time',
    'one row is no evidence a column never changes',
    from_source('time'),
    frames=lambda: _orders_and_customers({'ordered': ONE_DAY}, rows=1),
)


def _log(stamp: str, surrogate: bool) -> dict[str, pd.DataFrame]:
    r"""A log of changes nothing links to, and users edited in place that
    orders link to (``test_a_log_takes_its_edit_stamp_and_a_linked_table_
    does_not``)."""
    stamps = pd.Series(pd.date_range('2024-01-01', periods=50, freq='D'))
    log: dict[str, Any] = {
        'order_id': [i % 100 for i in range(ROWS)],
        'updated_at': HOURS,
    }
    if surrogate:
        log = {'update_id': range(ROWS), **log}
    return {
        'users': pd.DataFrame({'user_id': range(50), stamp: stamps}),
        'orders': pd.DataFrame(
            {'order_id': range(100), 'user_id': [i % 50 for i in range(100)]}
        ),
        'order_status_updates': pd.DataFrame(log),
    }


for _stamp in ('last_login', 'deleted_at'):
    for _surrogate in (False, True):
        scenario(
            f'time.a-log-and-a-linked-table.{_stamp}'
            + ('.with-a-surrogate' if _surrogate else ''),
            'time',
            'a log takes its edit stamp, a table linked to does not',
            from_source('time'),
            frames=lambda s=_stamp, g=_surrogate: _log(s, g),
        )


def _several_guesses() -> dict[str, pd.DataFrame]:
    frames = _orders_and_customers({'ordered': ONE_DAY})
    frames['order_status_updates'] = pd.DataFrame(
        {'order_id': [i % 100 for i in range(ROWS)], 'updated_at': HOURS}
    )
    frames['shipments'] = pd.DataFrame(
        {'order_id': range(ROWS), 'shipped_at': LATER, 'packed_at': HOURS}
    )
    return frames


scenario(
    'time.every-guess-in-one-notice',
    'time',
    'every guess named in one notice, with why and the override',
    from_source('time'),
    frames=_several_guesses,
)
for _table, _column in (
    ('customers', 'ModifiedDate'),
    ('order_status_updates', None),
    ('orders', 'ordered'),
    ('shipments', 'shipped_at'),
    ('shipments', None),
):
    scenario(
        f'time.the-override-for-{_table}-{_column or "none"}',
        'time',
        'each override the notice suggests does what it says',
        from_source('time', None, time_columns={_table: _column}),
        frames=_several_guesses,
    )


def _accounts() -> dict[str, pd.DataFrame]:
    return {
        'accounts': pd.DataFrame(
            {'handle': [f'h{i}' for i in range(50)], 'updated_at': HOURS[:50]}
        ),
        'logins': pd.DataFrame(
            {'account': [f'h{i % 50}' for i in range(ROWS)], 'at': HOURS}
        ),
    }


def _given_link(place: Place) -> Any:
    from sdm_client import Relationship

    return from_source(
        'time',
        None,
        primary_keys={'accounts': 'handle'},
        relationships=[Relationship('logins', 'account', 'accounts', 'handle')],
    )(place)


scenario(
    'time.a-link-the-caller-names-refuses-the-edit-stamp',
    'time',
    "the links decide, and the caller's are the final ones",
    _given_link,
    frames=_accounts,
)
scenario(
    'time.no-link-keeps-the-edit-stamp',
    'time',
    'without the link, the edit stamp of a table nothing links to is kept',
    from_source('time', None, primary_keys={'accounts': 'handle'}),
    frames=_accounts,
)

# Keys (P2-T2.16-10).

TABLES = {'users': 'app_users', 'orders': 'app_orders'}
EVERY_TENTH = [f'u{i:03d}' for i in range(0, 200, 10)]


def _repeated_examples(
    users: Callable[[], pd.DataFrame], columns: tuple[str, ...]
) -> Callable[[], list[str]]:
    def examples() -> list[str]:
        frame = users()
        repeated = frame[frame.duplicated(list(columns), keep=False)]
        values = (
            repeated[list(columns)]
            .astype(object)
            .itertuples(index=False, name=None)
        )
        return sorted(
            {
                repr(
                    tuple(_python_value(v) for v in value)
                    if len(columns) > 1
                    else _python_value(value[0])
                )
                for value in values
            }
        )

    return examples


def _key_scenario(
    name: str,
    covers: str,
    users: Callable[[], pd.DataFrame],
    key: Any = None,
    **options: Any,
) -> None:
    stated = {'primary_keys': {'users': key}} if key is not None else {}
    columns = ('user_id',) if key is None or key == 'user_id' else tuple(key)
    scenario(
        f'keys.{name}',
        'keys',
        covers,
        from_source('keys', TABLES, **stated, **options),
        frames=lambda: _users_and_orders_with(users()),
        examples=_repeated_examples(users, columns),
    )


_key_scenario('inferred.holds', 'an inferred key that holds', lambda: USERS)
_key_scenario(
    'inferred.repeating-once',
    'an inferred key repeating once is kept and warned about',
    lambda: _repeating('u007'),
)
_key_scenario(
    'inferred.repeating-exactly-one-percent',
    'an inferred key at exactly 99% distinct is kept',
    lambda: _repeating('u001', 'u002', users=FEWER),
)
_key_scenario(
    'inferred.repeating-just-over-one-percent',
    'an inferred key just under 99% distinct is declined',
    lambda: _repeating('u001', 'u002', 'u003', users=FEWER),
)
_key_scenario(
    'inferred.repeating-past-the-tolerance',
    'an inferred key past the tolerance is declined, and said',
    lambda: _repeating(*EVERY_TENTH),
)
_key_scenario(
    'inferred.missing-a-value',
    'an inferred key missing a value is kept and warned about',
    _missing_one,
)
_key_scenario(
    'stated.holds',
    'a stated key that holds',
    lambda: USERS,
    'user_id',
)
_key_scenario(
    'stated.repeating-once',
    'a stated key repeating once is kept and warned about',
    lambda: _repeating('u007'),
    'user_id',
)
_key_scenario(
    'stated.repeating-exactly-one-percent',
    'a stated key repeating exactly 1% of its rows is kept and warned about',
    lambda: _repeating('u001', 'u002', users=FEWER),
    'user_id',
)
_key_scenario(
    'stated.repeating-just-over-one-percent',
    'a stated key repeating just over 1% of its rows is refused',
    lambda: _repeating('u001', 'u002', 'u003', users=FEWER),
    'user_id',
)
_key_scenario(
    'stated.repeating-past-the-tolerance',
    'a stated key past the tolerance is refused, saying how to go on',
    lambda: _repeating(*EVERY_TENTH),
    'user_id',
)
_key_scenario(
    'stated.missing-a-value',
    'a stated key missing a value is refused',
    _missing_one,
    'user_id',
)
_key_scenario(
    'stated.composite-repeating-past-the-tolerance',
    'a stated composite key past the tolerance is refused',
    lambda: USERS,
    ['country', 'age'],
)
_key_scenario(
    'stated.composite-repeating-once',
    'a composite key is counted as one value',
    lambda: pd.concat(
        [
            USERS.assign(line=1),
            USERS.head(3).assign(line=2),
            USERS.head(1).assign(line=1),
        ],
        ignore_index=True,
    ),
    ['user_id', 'line'],
)
_key_scenario(
    'stated.repeating-past-the-tolerance.unchecked',
    'check_keys=False counts nothing and says nothing',
    lambda: _repeating(*EVERY_TENTH),
    'user_id',
    check_keys=False,
)
_key_scenario(
    'inferred.repeating-once.unchecked',
    'check_keys=False counts nothing and says nothing (inferred)',
    lambda: _repeating('u007'),
    check_keys=False,
)

#: Whole numbers past 2^53 that one float64 cannot tell apart: a count in
#: doubles finds repeats where there are none.
PAST_2_53 = [2**53 + i for i in range(1, 41)]


def _big_users(repeat: bool) -> pd.DataFrame:
    ids = PAST_2_53 + ([PAST_2_53[0]] if repeat else [])
    return _numbered_users(ids, dtype='int64')


def _big_orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'order_id': range(1, 121),
            'user_id': pd.Series(
                [PAST_2_53[i % len(PAST_2_53)] for i in range(120)],
                dtype='int64',
            ),
            'amount': [float(i % 7) for i in range(120)],
            'ts': pd.Timestamp('2024-01-01')
            + pd.to_timedelta(range(0, 120 * 24, 24), unit='h'),
        }
    )


for _repeat in (False, True):
    _users = lambda repeat=_repeat: _big_users(repeat)  # noqa: E731
    for _stated in (None, 'user_id'):
        scenario(
            'keys.past-2-53.'
            + ('stated' if _stated else 'inferred')
            + ('.repeating-once' if _repeat else '.holds'),
            'keys',
            'a key past 2^53 counted exactly, each value its own',
            from_source(
                'keys',
                TABLES,
                **({'primary_keys': {'users': _stated}} if _stated else {}),
            ),
            frames=lambda users=_users: {
                'app_users': users(),
                'app_orders': _big_orders(),
            },
            examples=_repeated_examples(_users, ('user_id',)),
        )

#: A declared key the source enforces is not counted.
scenario(
    'keys.declared-in-the-catalog',
    'keys',
    'a key the catalog declares and the source enforces',
    from_source('keys', {'users': 'u', 'orders': 'o'}),
    statements=[
        'CREATE TABLE {in_}u (user_id BIGINT PRIMARY KEY, country TEXT)',
        'CREATE TABLE {in_}o (order_id BIGINT PRIMARY KEY, '
        'user_id BIGINT, amount DOUBLE PRECISION, ts TIMESTAMP)',
        "INSERT INTO {in_}u VALUES (1, 'US'), (2, 'DE'), (3, 'IN')",
        'INSERT INTO {in_}o VALUES '
        + ', '.join(
            f"({i}, {1 + i % 3}, {i}.5, '{_day(i)}')" for i in range(1, 61)
        ),
    ],
    databases=('duckdb', 'postgres'),
)


def _scores(query: str, ids: list[str]) -> Act:
    r"""``materialize`` of ``ids`` over a schema whose ``users`` key holds
    ``u007`` twice, its keys unchecked."""
    from sdm_client import Schema

    def act(place: Place) -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            schema = Schema.from_source(place.source, TABLES, check_keys=False)
        return answer(lambda: materialized(schema, query, ids))

    return act


for _query_name, _query in (('temporal', TEMPORAL), ('static', STATIC)):
    for _ids in (['u001', 'u007'], ['u007'], ['u001']):
        scenario(
            f'keys.score.{_query_name}.{"-".join(_ids)}',
            'keys',
            'an id two rows hold cannot be scored; one row holds can'
            if 'u007' in _ids
            else 'an id one row holds is scored beside a repeated key',
            _scores(_query, _ids),
            frames=lambda: _users_and_orders_with(_repeating('u007')),
        )

# Refusals in the reference's words (P2-T2.16-10 and -11).

scenario(
    'refusals.a-missing-table',
    'refusals',
    'a table the source does not hold is refused naming it',
    from_source('refusals', {'users': 'app_users', 'orders': 'nope'}),
    frames=lambda: _users_and_orders_with(USERS),
)
scenario(
    'refusals.an-empty-source',
    'refusals',
    'a source holding no table is refused saying so',
    from_source('refusals'),
)
scenario(
    'refusals.a-time-column-the-table-lacks',
    'refusals',
    'a time column the table does not have is refused',
    from_source('refusals', TABLES, time_columns={'orders': 'nope'}),
    frames=lambda: _users_and_orders_with(USERS),
)
scenario(
    'refusals.a-key-column-the-table-lacks',
    'refusals',
    'a key column the table does not have is refused',
    from_source('refusals', TABLES, primary_keys={'users': 'nope'}),
    frames=lambda: _users_and_orders_with(USERS),
)
scenario(
    'refusals.a-key-for-a-table-not-named',
    'refusals',
    'a key for a table the schema does not have is refused',
    from_source('refusals', TABLES, primary_keys={'nope': 'user_id'}),
    frames=lambda: _users_and_orders_with(USERS),
)

# Postgres: the schema a source names (P2-T2.16-11).


def _shadowed(tables: dict[str, str] | None) -> Act:
    r"""``public`` holds a table of each scenario table's name, of other
    columns and rows; the source names its own schema."""

    def act(place: Place) -> Any:
        if place.database == 'postgres':
            for name in place.names.values():
                place.connection.execute(
                    f'CREATE TABLE public.{name} (other_id TEXT, n BIGINT)'
                )
                place.connection.execute(
                    f"INSERT INTO public.{name} VALUES ('a', 1), ('a', 1)"
                )
                place.outside.append(f'public.{name}')
        return from_source('postgres', tables)(place)

    return act


_SHADOW_STATEMENTS = [
    'CREATE TABLE {in_}{users} (user_id BIGINT PRIMARY KEY, country TEXT)',
    'CREATE TABLE {in_}{orders} (order_id BIGINT PRIMARY KEY, '
    'user_id BIGINT REFERENCES {in_}{users}(user_id), '
    'amount DOUBLE PRECISION, ts TIMESTAMP)',
    "INSERT INTO {in_}{users} VALUES (1, 'US'), (2, 'DE'), (3, 'IN')",
    'INSERT INTO {in_}{orders} VALUES '
    + ', '.join(
        f"({i}, {1 + i % 3}, {i}.5, '{_day(i)}')" for i in range(1, 61)
    ),
]

scenario(
    'postgres.a-table-only-in-its-schema',
    'postgres',
    'a table in the schema the source names, and in no other',
    from_source('postgres', {'users': '{users}', 'orders': '{orders}'}),
    statements=_SHADOW_STATEMENTS,
    unique=('users', 'orders'),
)
scenario(
    'postgres.every-table-of-its-schema',
    'postgres',
    'every table of the schema the source names, and no other',
    from_source('postgres'),
    statements=_SHADOW_STATEMENTS,
    unique=('users', 'orders'),
)
scenario(
    'postgres.a-table-public-also-holds',
    'postgres',
    'a table named alike in public is not the one read',
    _shadowed({'users': '{users}', 'orders': '{orders}'}),
    statements=_SHADOW_STATEMENTS,
    unique=('users', 'orders'),
    databases=('postgres',),
)
scenario(
    'postgres.every-table-when-public-holds-others',
    'postgres',
    'every table of its schema, not those of public named alike',
    _shadowed(None),
    statements=_SHADOW_STATEMENTS,
    unique=('users', 'orders'),
    databases=('postgres',),
)


def _search_path(place: Place) -> Any:
    r"""A source naming no schema reads the connection's own
    (``current_schema()``), here the place's, set as its search path."""
    from sdm_connectors import PostgresSource

    place.connection.execute(f'SET search_path TO {place.namespace}')
    place.source = PostgresSource(place.connection)
    return from_source('postgres', {'users': '{users}', 'orders': '{orders}'})(
        place
    )


scenario(
    'postgres.the-connection-s-search-path',
    'postgres',
    "a source naming no schema reads the connection's current schema",
    _search_path,
    statements=_SHADOW_STATEMENTS,
    unique=('users', 'orders'),
    databases=('postgres',),
)


def _renamed(place: Place, observed: dict[str, Any]) -> dict[str, Any]:
    r"""``observed`` with this run's unique names put back as the
    scenario's (``<users>``), and its Postgres schema as ``<schema>``, so
    runs compare."""
    back = {unique: f'<{name}>' for name, unique in place.names.items()}
    if place.namespace is not None:
        back[place.namespace] = '<schema>'

    def put(value: Any) -> Any:
        if isinstance(value, str):
            for unique, name in back.items():
                value = value.replace(unique, name)
            return value
        if isinstance(value, list):
            return [put(item) for item in value]
        if isinstance(value, dict):
            return {put(k): put(v) for k, v in value.items()}
        return value

    return put(observed)


def _case_apart(place: Place) -> Any:
    return from_source(
        'postgres', {'Parent': 'Parent', 'parent': 'parent', 'child': 'child'}
    )(place)


scenario(
    'postgres.a-foreign-key-lands-on-the-name-that-matches-exactly',
    'postgres',
    'a foreign key to "Parent" lands there, not on parent',
    _case_apart,
    statements=[
        'CREATE TABLE {in_}"Parent" (parent_id INTEGER PRIMARY KEY)',
        'CREATE TABLE {in_}parent (parent_id INTEGER PRIMARY KEY)',
        'CREATE TABLE {in_}child (child_id INTEGER PRIMARY KEY, '
        'parent_id INTEGER REFERENCES {in_}"Parent"(parent_id))',
        'INSERT INTO {in_}"Parent" VALUES (1)',
        'INSERT INTO {in_}parent VALUES (1)',
        'INSERT INTO {in_}child VALUES (1, 1)',
    ],
    databases=('postgres',),
)
_REGIONS = [
    'CREATE TABLE {in_}region (code TEXT UNIQUE, slug TEXT UNIQUE)',
    'CREATE TABLE {in_}store (store_id INTEGER PRIMARY KEY, ref TEXT UNIQUE)',
    'CREATE TABLE {in_}sale (sale_id INTEGER PRIMARY KEY, '
    'region_code TEXT REFERENCES {in_}region(code), '
    'store_ref TEXT REFERENCES {in_}store(ref), ts TIMESTAMP)',
    "INSERT INTO {in_}region VALUES ('n', 'north'), ('s', 'south')",
    "INSERT INTO {in_}store VALUES (1, 'a'), (2, 'b')",
    'INSERT INTO {in_}sale VALUES '
    + ', '.join(
        f"({i}, '{'n' if i % 2 == 0 else 's'}', "
        f"'{'a' if i % 3 == 0 else 'b'}', '{_day(i)}')"
        for i in range(1, 366)
    ),
]


def _regions(place: Place) -> Any:
    from sdm_client import PredictiveQuery, Schema, StructuredDataError

    notices: list[list[str]] = []

    try:
        with _noticed(NOTICES['postgres'], notices):
            schema = Schema.from_source(place.source)
    except StructuredDataError as error:
        return {'describe': {**_refusal(error), 'notices': notices}}
    described = {'outcome': 'schema', **_schema(schema), 'notices': notices}

    def labels() -> dict[str, Any]:
        request: Any = PredictiveQuery(
            'PREDICT COUNT(sale.*, 0, 30, days) FOR EACH region.code', schema
        ).materialize(indices=['n'], random_seed=7)
        (payload,) = request._payloads
        context = json.loads(json.dumps(payload, default=str))['context']
        return {
            'outcome': 'labels',
            'labels': sorted(
                {
                    json.dumps([row[1], row[3]])
                    for row in context['instance_table']['rows']
                }
            ),
        }

    return {'describe': described, 'materialize': answer(labels)}


scenario(
    'postgres.a-foreign-key-to-a-unique-column-keys-a-keyless-table',
    'postgres',
    'a foreign key to a unique column keys a keyless table, and its labels',
    _regions,
    statements=_REGIONS,
    databases=('duckdb', 'postgres'),
)


def _described_and_asked(query: str, ids: list[Any]) -> Act:
    r"""``Schema.from_source`` and ``materialize``, both in the source's
    schema."""
    from sdm_client import Schema

    def act(place: Place) -> Any:
        described = from_source('postgres', TABLES)(place)
        if described['outcome'] != 'schema':
            return {'describe': described}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            schema = Schema.from_source(place.source, TABLES)
        return {
            'describe': described,
            'materialize': answer(lambda: materialized(schema, query, ids)),
        }

    return act


scenario(
    'postgres.described-and-asked.temporal',
    'postgres',
    "a question over a schema read in the source's schema",
    _described_and_asked(TEMPORAL, ['u001', 'u002', 'u003']),
    frames=lambda: _users_and_orders_with(USERS),
)
scenario(
    'postgres.described-and-asked.static',
    'postgres',
    "a static question over a schema read in the source's schema",
    _described_and_asked(STATIC, ['u001', 'u002', 'u003']),
    frames=lambda: _users_and_orders_with(USERS),
)

# A schema the core did not describe (P2-T2.16-11's RouteError).


def _by_hand(described: Any, source: Any, **source_names: str) -> Any:
    r"""``described`` rebuilt from its parts, so nothing it read is kept."""
    from sdm_client import Schema, SourceTable

    return Schema(
        {
            name: SourceTable(
                name,
                source_names.get(name, table.source_name),
                dict(table.columns),
                primary_key=table.primary_key,
                time_column=table.time_column,
            )
            for name, table in described.tables.items()
        },
        described.relationships,
        source=source,
    )


def _from_scratch(source: Any) -> Any:
    r"""A schema written out by hand, column by column, as a user would
    write one for tables they know."""
    from sdm_client import Column, Relationship, Schema, SourceTable

    return Schema(
        {
            'users': SourceTable(
                'users',
                'app_users',
                {
                    'user_id': Column('user_id', 'string', 'ID'),
                    'age': Column('age', 'int64', 'numerical'),
                    'country': Column('country', 'string', 'categorical'),
                },
                primary_key='user_id',
            ),
            'orders': SourceTable(
                'orders',
                'app_orders',
                {
                    'order_id': Column('order_id', 'int64', 'ID'),
                    'user_id': Column('user_id', 'string', 'ID'),
                    'amount': Column('amount', 'float64', 'numerical'),
                    'ts': Column('ts', 'timestamp[us]', 'timestamp'),
                },
                primary_key='order_id',
                time_column='ts',
            ),
        },
        [Relationship('orders', 'user_id', 'users', 'user_id')],
        source=source,
    )


def _not_described_on_the_core(
    build: str, query: str, ids: list[Any], **names: str
) -> Act:
    from sdm_client import Schema

    def act(place: Place) -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if build == 'old-route':
                # Described where the old route still runs, then asked
                # where every route is the core's: a schema a session made
                # before the switch.
                forced = os.environ.get(ROUTES)
                os.environ[ROUTES] = ''
                try:
                    schema = Schema.from_source(place.source, TABLES)
                finally:
                    if forced is not None:
                        os.environ[ROUTES] = forced
                    else:
                        os.environ.pop(ROUTES, None)
            elif build == 'from-scratch':
                schema = _from_scratch(place.source)
            else:
                described = Schema.from_source(place.source, TABLES)
                schema = _by_hand(described, place.source, **names)
        return answer(lambda: materialized(schema, query, ids))

    return act


for _build in ('by-hand', 'from-scratch', 'old-route'):
    for _query_name, _query in (('temporal', TEMPORAL), ('static', STATIC)):
        scenario(
            f'described.{_build}.{_query_name}',
            'described',
            f'materialize of a schema {_build.replace("-", " ")}, as the '
            'reference materializes any Schema',
            _not_described_on_the_core(
                _build, _query, ['u001', 'u002', 'u003']
            ),
            frames=lambda: _users_and_orders_with(USERS),
        )
scenario(
    'described.by-hand.a-missing-table',
    'described',
    'a schema by hand naming a table the source lacks is refused naming it',
    _not_described_on_the_core('by-hand', TEMPORAL, ['u001'], orders='nope'),
    frames=lambda: _users_and_orders_with(USERS),
)
scenario(
    'described.by-hand.a-repeated-id',
    'described',
    'a schema by hand over a repeated key: an id two rows hold is refused',
    _not_described_on_the_core('by-hand', TEMPORAL, ['u001', 'u007']),
    frames=lambda: _users_and_orders_with(_repeating('u007')),
)


# --- running a scenario -----------------------------------------------------


def observe(found: Scenario, database: str, scratch: Path) -> Any:
    r"""``found``'s answer on ``database``, on the route the environment
    selects."""
    with placed(database, scratch) as place:
        place.names = {
            name: f'{name}_{uuid.uuid4().hex[:8]}' for name in found.unique
        }
        if found.frames is not None:
            load_frames(place, found.frames())
        run_statements(place, found.statements)
        return _renamed(place, found.act(place))


def keys(databases: tuple[str, ...] = DATABASES) -> list[str]:
    r"""Every ``<scenario>@<database>``, in order."""
    return [
        f'{name}@{database}'
        for name, found in SCENARIOS.items()
        for database in found.databases
        if database in databases
    ]


def main() -> int:
    import tempfile

    import sdm_relational_engine  # noqa: F401 - the reference's engine

    if _postgres_dsn() is None:
        print('SDM_POSTGRES_TEST_DSN names no server', file=sys.stderr)
        return 2
    reference = Path.home() / 'code/sdmc-reference'
    commit = subprocess.run(
        ['git', '-C', str(reference), 'rev-parse', 'HEAD'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    recorded: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as scratch:
        for key in keys():
            name, database = key.rsplit('@', 1)
            found = SCENARIOS[name]
            recorded[key] = {
                'family': found.family,
                'covers': found.covers,
                'examples': None
                if found.examples is None
                else found.examples(),
                'expected': observe(found, database, Path(scratch)),
            }
            print(key, file=sys.stderr)
    OUT.write_text(
        json.dumps(
            {
                'recorded_by': Path(__file__).name,
                'reference': commit,
                'families': FAMILIES,
                'scenarios': recorded,
            },
            indent=1,
            sort_keys=True,
        )
        + '\n',
        encoding='utf-8',
    )
    print(f'{len(recorded)} answers written to {OUT}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
