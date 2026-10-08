# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Helpers shared by the phase 5.5 acceptance checks.

The interface the checks rely on is written down in ``CONTRACT.md``.

Run as a script, this module is the child process some checks start, under a
chosen ``TZ`` or with the code of ``BASE`` ahead of this checkout's:

    python tests/fixes/_fixes.py duckdb-clock KIND [TIMEZONE]
    python tests/fixes/_fixes.py local-clock BACKEND
    python tests/fixes/_fixes.py warehouse KIND TIME_TYPE [CONFIG]
    python tests/fixes/_fixes.py postgres-sample DSN SCHEMA TABLE

Each prints one JSON document on its last line of output.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
FIXES = REPO / 'tests' / 'fixes'
PARITY_SCRIPTS = REPO / 'scripts' / 'parity'
TRANSCRIPTS = REPO / 'tests' / 'parity' / 'transcripts'

#: The commit phase 5.5 started from.
BASE = 'f9951d76a0bbd07e74382d3e742813a9c6df6dd8'

PACKAGES = (
    'sdm-api',
    'sdm-pql',
    'sdm-connectors',
    'sdm-client',
    'sdm-relational-engine',
)

if str(PARITY_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PARITY_SCRIPTS))
if str(FIXES) not in sys.path:
    sys.path.insert(0, str(FIXES))


# Environment ################################################################


def postgres_dsn() -> str:
    r"""The test database's DSN; a skip without one, unless one is required."""
    import pytest

    dsn = os.getenv('SDM_POSTGRES_TEST_DSN')
    if dsn:
        return dsn
    if os.getenv('SDM_REQUIRE_POSTGRES'):
        pytest.fail(
            'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
        )
    pytest.skip('SDM_POSTGRES_TEST_DSN is not set')


def git(*args: str) -> str:
    return subprocess.run(
        ['git', *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def output(proc: subprocess.CompletedProcess[str]) -> str:
    return (
        f'exit {proc.returncode}\n--- stdout\n{proc.stdout[-20000:]}\n'
        f'--- stderr\n{proc.stderr[-20000:]}'
    )


def base_sources(destination: Path) -> list[Path]:
    r"""The packages' sources at ``BASE``, with this checkout's compiled
    engine extension, which phase 5.5 does not change. Extracted once per
    destination."""
    paths = [f'packages/{p}/src' for p in PACKAGES]
    marker = destination / '.extracted'
    if not marker.exists():
        archive = subprocess.run(
            ['git', 'archive', BASE, *paths],
            cwd=REPO,
            capture_output=True,
            check=True,
        ).stdout
        destination.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ['tar', '-x', '-C', str(destination)], input=archive, check=True
        )
        engine = REPO / 'packages' / 'sdm-relational-engine' / 'src'
        for pattern in ('*.so', '*.pyd'):
            for built in engine.rglob(pattern):
                target = destination / built.relative_to(REPO)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(built, target)
        marker.write_text(BASE)
    return [destination / p for p in paths]


def base_env(destination: Path, **extra: str) -> dict[str, str]:
    r"""An environment whose Python imports the packages of ``BASE``."""
    env = dict(os.environ)
    sources = base_sources(destination)
    env['PYTHONPATH'] = os.pathsep.join(
        [str(p) for p in sources] + [env.get('PYTHONPATH', '')]
    ).rstrip(os.pathsep)
    env.update(extra)
    return env


def run_child(
    *args: str, env: dict[str, str] | None = None, timeout: float = 900
) -> Any:
    r"""Run this module as a child and return the JSON it printed last."""
    proc = subprocess.run(
        [sys.executable, str(FIXES / '_fixes.py'), *args],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert proc.returncode == 0, output(proc)
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert lines, output(proc)
    return json.loads(lines[-1])


def assert_imports_base(env: dict[str, str], destination: Path) -> None:
    r"""The child really imports the code of ``BASE``."""
    proc = subprocess.run(
        [
            sys.executable,
            '-c',
            'import sdm_relational_engine, sdm_client, sdm_connectors; '
            'print(sdm_relational_engine.__file__); '
            'print(sdm_client.__file__); print(sdm_connectors.__file__)',
        ],
        cwd=destination,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, output(proc)
    for line in proc.stdout.split():
        assert line.startswith(str(destination)), (
            f'the code of BASE was not the code imported: {proc.stdout}'
        )


# Statements #################################################################


@contextmanager
def statements() -> Iterator[list[tuple[str, list[Any] | None]]]:
    r"""Every statement any source runs meanwhile, with its parameters."""
    from sdm_connectors.source import Source

    log: list[tuple[str, list[Any] | None]] = []
    original = Source._run

    def _run(self: Any, sql: str, params: Sequence[Any] | None = None) -> Any:
        log.append((sql, None if params is None else list(params)))
        return original(self, sql, params)

    Source._run = _run  # type: ignore[method-assign]
    try:
        yield log
    finally:
        Source._run = original  # type: ignore[method-assign]


def flat(sql: str) -> str:
    r"""A statement on one line."""
    return ' '.join(sql.split())


# Predictions ################################################################


def normalized(payloads: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out = json.loads(json.dumps(list(payloads)))
    for payload in out:
        metadata = payload.get('metadata')
        if isinstance(metadata, dict) and 'request_id' in metadata:
            metadata['request_id'] = '<request-id>'
    return out


def predict(schema: Any, query: str, **options: Any) -> list[dict[str, Any]]:
    r"""The payloads a prediction sends, through the public client, to the
    corpus' stand-in NIM."""
    from _parity.nim import SyntheticSession
    from sdm_client import PredictiveQuery, SDMClient

    session = SyntheticSession()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        request = PredictiveQuery(query, schema).materialize(**options)
        client = SDMClient(url='http://localhost:8000')
        client._transport._session.close()
        client._transport._session = session
        with client:
            client.predict(request, model='kumo-relational').to_pandas()
    return normalized(session.payloads)


def late_rows(
    payloads: Sequence[dict[str, Any]], table: str, column: str
) -> list[tuple[str, Any, str, str]]:
    r"""Every row of ``table`` a payload holds whose ``column`` is after the
    anchor time of the example it belongs to: ``(part, instance, time,
    anchor)``. History must end at the anchor."""
    import pandas as pd

    late = []
    for payload in payloads:
        for part in ('context', 'predict'):
            section = payload.get(part) or {}
            instances = section.get('instance_table')
            related = (section.get('related_tables') or {}).get(table)
            if instances is None or related is None:
                continue
            columns = instances['columns']
            ids = columns.index('instance_id')
            at = columns.index('__kumo_anchor_time')
            anchors = {
                row[ids]: pd.Timestamp(row[at]) for row in instances['rows']
            }
            columns = related['columns']
            owner = columns.index('instance_id')
            when = columns.index(column)
            for row in related['rows']:
                if row[when] is None:
                    continue
                anchor = anchors[row[owner]]
                if pd.Timestamp(row[when]) > anchor:
                    late.append((part, row[owner], row[when], str(anchor)))
    return late


# The clock ##################################################################
#
# Ten users, and three weeks of events at minutes past each hour, so a window
# moved by any whole or half-hour offset holds different events. One event
# sits three hours after the explicit anchor.

CLOCK_START = '2024-03-01 00:00:00'
CLOCK_ANCHOR = '2024-03-15 12:00:00'
CLOCK_LATE_EVENT = 999_999
CLOCK_TABLES = {'users': 'users', 'events': 'events'}
CLOCK_KEYS = {'users': 'user_id', 'events': 'event_id'}
CLOCK_TIMES = {'events': 'happened_at'}
CLOCK_QUERIES = {
    'count': 'PREDICT COUNT(events.*, 0, 6, hours) FOR EACH users.user_id',
    'inactivity': (
        'PREDICT COUNT(events.*, 0, 6, hours) = 0 FOR EACH users.user_id'
    ),
}


def clock_frames() -> tuple[Any, Any]:
    r"""``users`` and ``events``; ``happened_at`` is naive UTC."""
    import numpy as np
    import pandas as pd

    users = pd.DataFrame({'user_id': np.arange(1, 11, dtype='int64')})
    start = pd.Timestamp(CLOCK_START)
    rows = []
    event_id = 1
    for hour in range(24 * 21):
        for user in range(1, 11):
            if (hour + 3 * user) % (user + 1) == 0:
                rows.append(
                    (
                        event_id,
                        user,
                        start
                        + pd.Timedelta(hours=hour)
                        + pd.Timedelta(minutes=(7 * user) % 60),
                        float((hour * user) % 13),
                    )
                )
                event_id += 1
    rows.append(
        (
            CLOCK_LATE_EVENT,
            1,
            pd.Timestamp(CLOCK_ANCHOR) + pd.Timedelta(hours=3),
            1000.0,
        )
    )
    events = pd.DataFrame(
        rows, columns=['event_id', 'user_id', 'happened_at', 'value']
    )
    events['event_id'] = events['event_id'].astype('int64')
    events['user_id'] = events['user_id'].astype('int64')
    return users, events


def clock_options(anchor: bool) -> dict[str, Any]:
    import pandas as pd

    options: dict[str, Any] = {
        'indices': [1, 2, 3, 4, 5],
        'num_context_samples': 20,
        'random_seed': 7,
    }
    if anchor:
        options['anchor_time'] = pd.Timestamp(CLOCK_ANCHOR)
    return options


def clock_schema(source: Any) -> Any:
    from sdm_client import Schema

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return Schema.from_source(
            source,
            CLOCK_TABLES,
            primary_keys=CLOCK_KEYS,
            time_columns=CLOCK_TIMES,
        )


def clock_payloads(
    source: Any, questions: Sequence[str] = tuple(CLOCK_QUERIES)
) -> dict[str, list[dict[str, Any]]]:
    r"""Each clock question, with the default and the explicit anchor."""
    schema = clock_schema(source)
    out = {}
    for name in questions:
        query = CLOCK_QUERIES[name]
        for anchor in (False, True):
            label = f'{name}@{"explicit" if anchor else "default"}'
            out[label] = predict(schema, query, **clock_options(anchor))
    return out


#: How each Postgres time column kind is made from the naive one.
POSTGRES_KINDS = {
    'timestamp': 'happened_at',
    'timestamptz': "happened_at AT TIME ZONE 'UTC'",
    'domain': "(happened_at AT TIME ZONE 'UTC')::{schema}.instant",
    'date': 'happened_at::date',
    'text': "to_char(happened_at, 'YYYY-MM-DD HH24:MI:SS')",
}


def load_postgres_clock(dsn: str, schema: str) -> dict[str, str]:
    r"""The clock in one schema per time column kind; their names."""
    import psycopg

    users, events = clock_frames()
    names = {kind: f'{schema}_{kind}' for kind in POSTGRES_KINDS}
    connection = psycopg.connect(
        dsn, autocommit=True, options='-c TimeZone=UTC'
    )
    try:
        for kind, expression in POSTGRES_KINDS.items():
            name = names[kind]
            connection.execute(f'DROP SCHEMA IF EXISTS "{name}" CASCADE')
            connection.execute(f'CREATE SCHEMA "{name}"')
            if kind == 'domain':
                connection.execute(
                    f'CREATE DOMAIN "{name}".instant AS timestamptz'
                )
            connection.execute(
                f'CREATE TABLE "{name}".users (user_id BIGINT PRIMARY KEY)'
            )
            with connection.cursor() as cursor:
                cursor.executemany(
                    f'INSERT INTO "{name}".users VALUES (%s)',
                    [(int(u),) for u in users['user_id']],
                )
            connection.execute(
                f'CREATE TABLE "{name}".naive (event_id BIGINT, '
                'user_id BIGINT, happened_at TIMESTAMP, '
                'value DOUBLE PRECISION)'
            )
            with connection.cursor() as cursor:
                cursor.executemany(
                    f'INSERT INTO "{name}".naive VALUES (%s, %s, %s, %s)',
                    [
                        (int(e), int(u), t.to_pydatetime(), float(v))
                        for e, u, t, v in events.itertuples(index=False)
                    ],
                )
            column = expression.format(schema=f'"{name}"')
            connection.execute(
                f'CREATE TABLE "{name}".events AS SELECT event_id, user_id, '
                f'{column} AS happened_at, value FROM "{name}".naive'
            )
            connection.execute(f'DROP TABLE "{name}".naive')
            connection.execute(
                f'ALTER TABLE "{name}".events ADD PRIMARY KEY (event_id)'
            )
            connection.execute(
                f'ALTER TABLE "{name}".events ADD FOREIGN KEY (user_id) '
                f'REFERENCES "{name}".users (user_id)'
            )
    finally:
        connection.close()
    return names


def drop_postgres_schemas(dsn: str, names: Sequence[str]) -> None:
    import psycopg

    connection = psycopg.connect(dsn, autocommit=True)
    try:
        for name in names:
            connection.execute(f'DROP SCHEMA IF EXISTS "{name}" CASCADE')
    finally:
        connection.close()


def duckdb_clock(kind: str, timezone: str | None) -> Any:
    r"""The clock in an in-memory DuckDB database; ``kind`` is ``timestamp``
    or ``timestamptz``, and ``timezone`` its ``TimeZone`` setting, or
    ``None`` to leave it to the process."""
    import duckdb

    users, events = clock_frames()
    config = {} if timezone is None else {'TimeZone': timezone}
    connection = duckdb.connect(config=config)
    connection.register('clock_users', users)
    connection.register('clock_events', events)
    column = {
        'timestamp': 'happened_at',
        'timestamptz': "timezone('UTC', happened_at)",
    }[kind]
    connection.execute('CREATE TABLE users AS SELECT * FROM clock_users')
    connection.execute(
        f'CREATE TABLE events AS SELECT event_id, user_id, '
        f'{column} AS happened_at, value FROM clock_events'
    )
    connection.unregister('clock_users')
    connection.unregister('clock_events')
    return connection


def sqlite_clock(path: Path) -> Any:
    import adbc_driver_sqlite.dbapi as sqlite
    import pyarrow as pa

    users, events = clock_frames()
    events = events.assign(
        happened_at=events['happened_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
    )
    connection = sqlite.connect(str(path))
    with connection.cursor() as cursor:
        cursor.adbc_ingest('users', pa.Table.from_pandas(users))
        cursor.adbc_ingest('events', pa.Table.from_pandas(events))
    connection.commit()
    return connection


def memory_clock(zoned: bool) -> Any:
    from sdm_client import Schema

    users, events = clock_frames()
    if zoned:
        events = events.assign(
            happened_at=events['happened_at'].dt.tz_localize('UTC')
        )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return Schema.from_data(
            {'users': users, 'events': events},
            primary_keys=CLOCK_KEYS,
            time_columns=CLOCK_TIMES,
        )


def memory_payloads(zoned: bool) -> dict[str, list[dict[str, Any]]]:
    schema = memory_clock(zoned)
    out = {}
    for name, query in CLOCK_QUERIES.items():
        for anchor in (False, True):
            label = f'{name}@{"explicit" if anchor else "default"}'
            out[label] = predict(schema, query, **clock_options(anchor))
    return out


# Children ###################################################################


def _child_duckdb_clock(kind: str, timezone: str | None = None) -> Any:
    from sdm_connectors import DuckDBSource

    return clock_payloads(DuckDBSource(duckdb_clock(kind, timezone)))


def _child_local_clock(backend: str) -> Any:
    import tempfile

    from sdm_connectors import SQLiteSource

    if backend == 'memory':
        return memory_payloads(zoned=False)
    if backend == 'memory-zoned':
        return memory_payloads(zoned=True)
    if backend == 'sqlite':
        with tempfile.TemporaryDirectory() as tmp:
            connection = sqlite_clock(Path(tmp) / 'clock.db')
            try:
                return clock_payloads(SQLiteSource(connection))
            finally:
                connection.close()
    raise ValueError(backend)


def _child_warehouse(kind: str, time_type: str, config: str = '{}') -> Any:
    r"""A stub warehouse's statements and payloads. ``config`` is JSON:
    ``Warehouse`` fields (``listings``, ``details``, ``prefiltered_rows``,
    ``nullable_keys``), and ``questions``, the names of
    ``_warehouse_stub.QUESTIONS`` to ask (all by default; none builds the
    schema only)."""
    import _warehouse_stub as stub

    options = json.loads(config)
    names = options.pop('questions', list(stub.QUESTIONS))
    warehouse = stub.Warehouse(
        kind=kind, time_type=None if time_type == '-' else time_type, **options
    )
    if names:
        questions = {name: stub.QUESTIONS[name] for name in names}
        payloads = stub.predict(warehouse, questions)
    else:
        stub.build_schema(warehouse)
        payloads = {}
    return {
        'statements': stub.statements_json(warehouse),
        'payloads': payloads,
    }


def _child_postgres_sample(dsn: str, schema: str, table: str) -> Any:
    import psycopg
    from sdm_relational_engine.rfm.backend.postgres import PostgresTable

    connection = psycopg.connect(dsn, autocommit=True)
    try:
        with statements() as log:
            sampled = PostgresTable(
                connection, table, schema=schema
            )._source_sample_df
        return {
            'statements': [[sql, params] for sql, params in log],
            'rows': len(sampled),
        }
    finally:
        connection.close()


_CHILDREN: dict[str, Callable[..., Any]] = {
    'duckdb-clock': _child_duckdb_clock,
    'local-clock': _child_local_clock,
    'warehouse': _child_warehouse,
    'postgres-sample': _child_postgres_sample,
}


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] not in _CHILDREN:
        print(__doc__, file=sys.stderr)
        return 2
    result = _CHILDREN[argv[1]](*argv[2:])
    print(json.dumps(result, default=str, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
