# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P55.tz-e2e: time windows don't depend on the session's time zone.

A prediction runs through the public API (``Schema.from_source``,
``PredictiveQuery.materialize``, ``SDMClient.predict``, to the corpus'
stand-in NIM) over the clock: hourly events, so a window moved by any offset
holds different events. The same events are stored with the time column
naive, zoned, as a domain over a zoned type, as a date and as text.

* A zoned column, under any session zone, must give exactly the payloads the
  naive column gives under UTC: window bounds name UTC instants.
* Every column kind must give the same payloads under every session zone.
* No payload may hold an event after the anchor of the example it belongs
  to, whatever the zone; text columns are excluded, since they compare as
  text (a follow-up outside this phase).

DuckDB is checked through ``duckdb.connect(config={'TimeZone': ...})`` and,
the way most users hit it, through a process whose ``TZ`` isn't UTC.
SQLite and in-memory data are guards: unaffected before and after the fix.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import pytest
from _fixes import (
    CLOCK_QUERIES,
    clock_payloads,
    drop_postgres_schemas,
    duckdb_clock,
    late_rows,
    load_postgres_clock,
    postgres_dsn,
    run_child,
)

ZONES = ('UTC', 'America/New_York', 'Asia/Kolkata', 'Pacific/Chatham')
ZONED = ('timestamptz', 'domain')
LABELS = [
    f'{name}@{anchor}'
    for name in CLOCK_QUERIES
    for anchor in ('default', 'explicit')
]


#: A date or text column holds no event inside a six-hour window after a
#: midday anchor, so every example would be inactive; it is asked to count.
QUESTIONS = {
    'timestamp': tuple(CLOCK_QUERIES),
    'timestamptz': tuple(CLOCK_QUERIES),
    'domain': tuple(CLOCK_QUERIES),
    'date': ('count',),
    'text': ('count',),
}


def _differences(
    got: dict[str, list[Any]], want: dict[str, list[Any]]
) -> list[str]:
    out = []
    for label in LABELS:
        if label not in got and label not in want:
            continue
        if json.dumps(got[label], sort_keys=True) != json.dumps(
            want[label], sort_keys=True
        ):
            out.append(label)
    return out


def _assert_no_late_rows(payloads: dict[str, list[Any]]) -> None:
    for label, sent in payloads.items():
        late = late_rows(sent, 'events', 'happened_at')
        assert not late, (
            f'{label}: {len(late)} event(s) after their example anchor, '
            f'e.g. {late[:3]}'
        )


# PostgreSQL #################################################################


@pytest.fixture(scope='module')
def postgres_clock() -> Any:
    dsn = postgres_dsn()
    names = load_postgres_clock(dsn, f'fix_tz_{uuid.uuid4().hex[:10]}')
    try:
        yield dsn, names
    finally:
        drop_postgres_schemas(dsn, list(names.values()))


_postgres_cache: dict[tuple[str, str], dict[str, list[Any]]] = {}


def _postgres_payloads(
    postgres_clock: Any, kind: str, zone: str
) -> dict[str, list[Any]]:
    import psycopg
    from sdm_connectors import PostgresSource

    dsn, names = postgres_clock
    key = (kind, zone)
    if key not in _postgres_cache:
        connection = psycopg.connect(
            dsn, autocommit=True, options=f'-c TimeZone={zone}'
        )
        try:
            source = PostgresSource(connection, schema=names[kind])
            _postgres_cache[key] = clock_payloads(source, QUESTIONS[kind])
        finally:
            connection.close()
    return _postgres_cache[key]


@pytest.mark.parametrize('zone', ZONES)
@pytest.mark.parametrize('kind', ZONED)
def test_postgres_zoned_column_predicts_as_naive_utc(
    postgres_clock: Any, kind: str, zone: str
) -> None:
    want = _postgres_payloads(postgres_clock, 'timestamp', 'UTC')
    got = _postgres_payloads(postgres_clock, kind, zone)
    _assert_no_late_rows(got)
    differing = _differences(got, want)
    assert not differing, (
        f'a {kind} column under TimeZone={zone} sent different payloads '
        f'from the naive column under UTC for {differing}'
    )


@pytest.mark.parametrize('zone', ZONES[1:])
@pytest.mark.parametrize('kind', ('timestamp', 'date', 'text'))
def test_postgres_other_columns_ignore_the_session_zone(
    postgres_clock: Any, kind: str, zone: str
) -> None:
    want = _postgres_payloads(postgres_clock, kind, 'UTC')
    got = _postgres_payloads(postgres_clock, kind, zone)
    if kind != 'text':
        _assert_no_late_rows(got)
    differing = _differences(got, want)
    assert not differing, (
        f'a {kind} column under TimeZone={zone} sent different payloads '
        f'from the same column under UTC for {differing}'
    )


def test_postgres_the_late_event_is_in_the_table(postgres_clock: Any) -> None:
    r"""The check above is not vacuous: the event three hours after the
    explicit anchor exists, and a naive UTC run leaves it out of history."""
    import psycopg

    dsn, names = postgres_clock
    connection = psycopg.connect(dsn, autocommit=True)
    try:
        found = connection.execute(
            f'SELECT count(*) FROM "{names["timestamptz"]}".events '
            "WHERE happened_at = TIMESTAMPTZ '2024-03-15 15:00:00+00'"
        ).fetchone()
    finally:
        connection.close()
    assert found is not None and found[0] >= 1
    sent = _postgres_payloads(postgres_clock, 'timestamp', 'UTC')
    assert sent['count@explicit'][0]['predict']['related_tables']['events'][
        'rows'
    ]


# DuckDB #####################################################################


_duckdb_cache: dict[tuple[str, str | None], dict[str, list[Any]]] = {}


def _duckdb_payloads(kind: str, zone: str | None) -> dict[str, list[Any]]:
    from sdm_connectors import DuckDBSource

    key = (kind, zone)
    if key not in _duckdb_cache:
        connection = duckdb_clock(kind, zone)
        try:
            _duckdb_cache[key] = clock_payloads(DuckDBSource(connection))
        finally:
            connection.close()
    return _duckdb_cache[key]


@pytest.mark.parametrize('zone', ZONES)
def test_duckdb_zoned_column_predicts_as_naive_utc(zone: str) -> None:
    want = _duckdb_payloads('timestamp', 'UTC')
    got = _duckdb_payloads('timestamptz', zone)
    _assert_no_late_rows(got)
    differing = _differences(got, want)
    assert not differing, (
        f'a TIMESTAMPTZ column under TimeZone={zone} sent different payloads '
        f'from the naive column under UTC for {differing}'
    )


@pytest.mark.parametrize('zone', ZONES[1:])
def test_duckdb_naive_column_ignores_the_zone(zone: str) -> None:
    want = _duckdb_payloads('timestamp', 'UTC')
    got = _duckdb_payloads('timestamp', zone)
    _assert_no_late_rows(got)
    assert not _differences(got, want)


@pytest.mark.parametrize('kind', ('timestamptz', 'timestamp'))
def test_duckdb_in_a_process_outside_utc(kind: str) -> None:
    r"""A laptop's own zone, with nothing set on the connection."""
    want = _duckdb_payloads('timestamp', 'UTC')
    env = dict(os.environ, TZ='America/New_York')
    got = run_child('duckdb-clock', kind, env=env)
    _assert_no_late_rows(got)
    differing = _differences(got, want)
    assert not differing, (
        f'a {kind} column in a process with TZ=America/New_York sent '
        f'different payloads from the naive column under UTC for {differing}'
    )


# SQLite and in-memory (guards: they pass before and after the fix) #########


@pytest.mark.parametrize('backend', ('sqlite', 'memory'))
def test_local_backends_ignore_the_process_zone(backend: str) -> None:
    utc = run_child('local-clock', backend, env=dict(os.environ, TZ='UTC'))
    other = run_child(
        'local-clock', backend, env=dict(os.environ, TZ='America/New_York')
    )
    _assert_no_late_rows(other)
    assert not _differences(other, utc)


def test_zoned_frames_predict_as_naive_ones() -> None:
    naive = run_child('local-clock', 'memory', env=dict(os.environ, TZ='UTC'))
    zoned = run_child(
        'local-clock',
        'memory-zoned',
        env=dict(os.environ, TZ='America/New_York'),
    )
    assert not _differences(zoned, naive)
