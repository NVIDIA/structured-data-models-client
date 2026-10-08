# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Helpers shared by the phase 3 acceptance checks.

The interface the checks rely on is written down in ``CONTRACT.md``.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SEAM = REPO / 'tests' / 'seam'
ENGINE_SRC = REPO / 'packages' / 'sdm-relational-engine' / 'src'
CLIENT_SRC = REPO / 'packages' / 'sdm-client' / 'src'
CONNECTORS_SRC = REPO / 'packages' / 'sdm-connectors' / 'src'
TRANSCRIPTS = REPO / 'tests' / 'parity' / 'transcripts'
RUN_CORPUS = REPO / 'scripts' / 'parity' / 'run_corpus.py'

#: The commit phase 3 started from: its recordings are the reference.
BASE = '6b3eda8737e37ff1b395ebf96d22ace1abd2502a'

#: The driver calls the engine must leave to the connectors.
DRIVER_CALLS = frozenset(
    {
        'cursor',
        'execute',
        'executemany',
        'fetchall',
        'fetchone',
        'fetchmany',
        'fetchall_arrow',
        'fetch_arrow_all',
        'fetch_arrow_table',
        'to_arrow_table',
        'toArrow',
        'register',
        'adbc_ingest',
        'commit',
    }
)

#: The backends that write SQL, by their directory under ``rfm/backend``.
DIALECT_BACKENDS = ('databricks', 'snow', 'postgres', 'duckdb', 'sqlite')


def run_python(
    *args: Any, env: dict[str, str] | None = None, timeout: float = 900
) -> subprocess.CompletedProcess[str]:
    r"""Run a Python script from the repository root."""
    return subprocess.run(
        [sys.executable, *(str(a) for a in args)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def output(proc: subprocess.CompletedProcess[str]) -> str:
    return (
        f'exit {proc.returncode}\n--- stdout\n{proc.stdout[-20000:]}\n'
        f'--- stderr\n{proc.stderr[-20000:]}'
    )


def git(*args: str) -> str:
    return subprocess.run(
        ['git', *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


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


# The shop the push-down checks predict over.


def shop_frames() -> tuple[Any, Any]:
    r"""Forty users and four hundred orders over 2024, repeatably."""
    import numpy as np
    import pandas as pd

    rs = np.random.RandomState(20261002)
    users = pd.DataFrame(
        {
            'user_id': np.arange(1, 41, dtype='int64'),
            'age': rs.randint(18, 70, 40).astype('int64'),
            'country': rs.choice(['DE', 'FR', 'US'], 40),
        }
    )
    n = 400
    orders = pd.DataFrame(
        {
            'order_id': np.arange(1000, 1000 + n, dtype='int64'),
            'user_id': rs.randint(1, 41, n).astype('int64'),
            'price': rs.rand(n).round(2),
            'ordered_at': pd.Timestamp('2024-01-01')
            + pd.to_timedelta(rs.randint(0, 365 * 24, n), unit='h'),
        }
    )
    return users, orders


#: (query, materialize options) asked of each push-down source.
QUESTIONS: dict[str, tuple[str, dict[str, Any]]] = {
    'count': (
        'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id',
        {'indices': [1, 2, 3], 'random_seed': 7},
    ),
    'sum': (
        'PREDICT SUM(orders.price, 0, 14, days) FOR EACH users.user_id',
        {'indices': [4, 5], 'num_neighbors': [3, 3], 'random_seed': 23},
    ),
}


def predict(source: Any, tables: dict[str, str] | None = None) -> list[Any]:
    r"""Each question's payloads, gathered from ``source`` where it is."""
    import warnings

    from sdm_client import PredictiveQuery, Schema
    from sdm_client.adapters.relational import RelationalAdapter

    payloads: list[Any] = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        schema = Schema.from_source(source, tables)
        for query, options in QUESTIONS.values():
            request = PredictiveQuery(query, schema).materialize(**options)
            payloads.extend(
                RelationalAdapter().prediction_plan(request, ('prediction',))
            )
    for payload in payloads:
        assert payload['context']['instance_table']['rows'], (
            'the prediction gathered no context; nothing was read'
        )
    return payloads


# Who calls a driver.

# Frames that are the drivers themselves, the stubs standing in for them, or
# the plumbing between: skipped when looking for who made a call.
_SKIPPED_PREFIXES = (
    'adbc_driver_manager',
    'adbc_driver_sqlite',
    'psycopg',
    'duckdb',
    '_duckdb',
    'databricks.',
    'snowflake.',
    'pyspark',
    'contextlib',
    'functools',
    'abc',
    '_seam',
    '_warehouse',
)


def call_path() -> tuple[str, str | None]:
    r"""Who made the driver call being handled: ``(caller, entry)``.

    ``caller`` is the module of the nearest frame outside the drivers, the
    stubs and ``sdm_connectors``. ``entry`` is the function of the outermost
    ``sdm_connectors`` frame below it, the one ``caller`` called, or ``None``
    when ``caller`` called the driver itself.
    """
    entry: str | None = None
    for frame_info in inspect.stack()[1:]:
        module = frame_info.frame.f_globals.get('__name__', '')
        if module.startswith(_SKIPPED_PREFIXES):
            continue
        if module == 'sdm_connectors' or module.startswith('sdm_connectors.'):
            entry = frame_info.function
            continue
        return module, entry
    return '', entry


def through_run(caller: str, entry: str | None) -> bool:
    r"""Whether a driver call the engine caused went through ``Source._run``.

    ``_run`` must be the first ``sdm_connectors`` function the engine's own
    code called; a backend's module-level ``run`` may only be reached
    beneath it.
    """
    return not caller.startswith('sdm_relational_engine') or entry == '_run'


@dataclass
class DriverCall:
    method: str
    sql: str | None
    params: Any
    caller: str
    entry: str | None


@dataclass
class DriverLog:
    calls: list[DriverCall] = field(default_factory=list)

    @property
    def statements(self) -> list[DriverCall]:
        return [c for c in self.calls if c.sql is not None]

    def not_through_run(self) -> list[DriverCall]:
        r"""The calls the engine made other than through ``_run``."""
        return [c for c in self.calls if not through_run(c.caller, c.entry)]


def _wrap(
    cls: type, name: str, log: DriverLog, sql_arg: str | None
) -> Callable[..., Any]:
    original = cls.__dict__[name]

    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        sql = None
        params = None
        if sql_arg is not None:
            bound = inspect.signature(original).bind(self, *args, **kwargs)
            sql = bound.arguments.get(sql_arg)
            params = bound.arguments.get('parameters') or bound.arguments.get(
                'params'
            )
            if not isinstance(sql, str):
                sql = repr(sql)
        log.calls.append(DriverCall(name, sql, params, *call_path()))
        return original(self, *args, **kwargs)

    wrapper.__wrapped__ = original  # type: ignore[attr-defined]
    return wrapper


@contextmanager
def watch_classes(
    methods: dict[type, dict[str, str | None]],
) -> Iterator[DriverLog]:
    r"""Log every call of the named methods: ``{cls: {method: sql_arg}}``."""
    log = DriverLog()
    saved: list[tuple[type, str, Any]] = []
    try:
        for cls, names in methods.items():
            for name, sql_arg in names.items():
                if name not in cls.__dict__:
                    continue
                saved.append((cls, name, cls.__dict__[name]))
                setattr(cls, name, _wrap(cls, name, log, sql_arg))
        yield log
    finally:
        for cls, name, original in reversed(saved):
            setattr(cls, name, original)


_FETCHES = {
    'fetchall': None,
    'fetchone': None,
    'fetchmany': None,
    'fetch_arrow_table': None,
    'fetch_df': None,
    'fetch_record_batch': None,
}


@contextmanager
def watch_sqlite() -> Iterator[DriverLog]:
    r"""Every ADBC call a SQLite run makes, and who made it."""
    from adbc_driver_manager import dbapi

    with watch_classes(
        {
            dbapi.Connection: {
                'cursor': None,
                'commit': None,
                'rollback': None,
            },
            dbapi.Cursor: {
                'execute': 'operation',
                'executemany': 'operation',
                'adbc_ingest': None,
                **_FETCHES,
            },
        }
    ) as log:
        yield log


@contextmanager
def watch_postgres() -> Iterator[DriverLog]:
    r"""Every psycopg call a PostgreSQL run makes, and who made it."""
    import psycopg

    with watch_classes(
        {
            psycopg.Connection: {
                'cursor': None,
                'commit': None,
                'rollback': None,
                'execute': 'query',
                'transaction': None,
            },
            psycopg.Cursor: {
                'execute': 'query',
                'executemany': 'query',
                **_FETCHES,
            },
        }
    ) as log:
        yield log


def sqlite_database(path: Path, *, autocommit: bool = False) -> Any:
    r"""A SQLite file holding the shop, and an ADBC connection to it."""
    from adbc_driver_sqlite import dbapi

    users, orders = shop_frames()
    connection = dbapi.connect(str(path))
    with connection.cursor() as cursor:
        cursor.adbc_ingest('users', users, mode='create')
        cursor.adbc_ingest('orders', orders, mode='create')
    connection.commit()
    connection.close()
    return dbapi.connect(str(path), autocommit=autocommit)


def duckdb_database(path: Path, *, read_only: bool = False) -> Any:
    r"""A DuckDB file holding the shop, and a connection to it."""
    import duckdb

    users, orders = shop_frames()
    connection = duckdb.connect(str(path))
    connection.register('users_frame', users)
    connection.register('orders_frame', orders)
    connection.execute('CREATE TABLE users AS SELECT * FROM users_frame')
    connection.execute('CREATE TABLE orders AS SELECT * FROM orders_frame')
    connection.close()
    return duckdb.connect(str(path), read_only=read_only)


def postgres_schema(dsn: str, schema: str, *, autocommit: bool) -> Any:
    r"""A fresh PostgreSQL schema holding the shop, and a connection."""
    import psycopg

    users, orders = shop_frames()
    with psycopg.connect(dsn, autocommit=True) as setup:
        setup.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        setup.execute(f'CREATE SCHEMA "{schema}"')
        setup.execute(
            f'CREATE TABLE "{schema}".users (user_id BIGINT PRIMARY KEY, '
            f'age BIGINT, country TEXT)'
        )
        setup.execute(
            f'CREATE TABLE "{schema}".orders (order_id BIGINT PRIMARY KEY, '
            f'user_id BIGINT REFERENCES "{schema}".users(user_id), '
            f'price DOUBLE PRECISION, ordered_at TIMESTAMP)'
        )
        with setup.cursor() as cursor:
            cursor.executemany(
                f'INSERT INTO "{schema}".users VALUES (%s, %s, %s)',
                [
                    (int(r.user_id), int(r.age), str(r.country))
                    for r in users.itertuples()
                ],
            )
            cursor.executemany(
                f'INSERT INTO "{schema}".orders VALUES (%s, %s, %s, %s)',
                [
                    (
                        int(r.order_id),
                        int(r.user_id),
                        float(r.price),
                        r.ordered_at.to_pydatetime(),
                    )
                    for r in orders.itertuples()
                ],
            )
    return psycopg.connect(dsn, autocommit=autocommit)


def drop_postgres_schema(dsn: str, schema: str) -> None:
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as setup:
        setup.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


# Transcripts.


def load_transcript(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_arrow(path: Path) -> Any:
    import pyarrow as pa

    data = path.read_bytes()
    try:
        return pa.ipc.open_file(pa.py_buffer(data)).read_all()
    except pa.ArrowInvalid:
        return pa.ipc.open_stream(pa.py_buffer(data)).read_all()


def scenario_dirs(root: Path) -> dict[str, Path]:
    r"""Each recorded scenario's directory, by its path under ``root``."""
    return {
        p.parent.relative_to(root).as_posix(): p.parent
        for p in sorted(root.rglob('transcript.json'))
    }


def export_base_transcripts(destination: Path) -> Path:
    r"""The recordings as they were at ``BASE``, written under ``destination``."""
    archive = subprocess.run(
        ['git', 'archive', BASE, 'tests/parity/transcripts'],
        cwd=REPO,
        capture_output=True,
        check=True,
    ).stdout
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ['tar', '-x', '-C', str(destination)], input=archive, check=True
    )
    return destination / 'tests' / 'parity' / 'transcripts'


def json_array_param(value: Any) -> bool:
    r"""Whether a parameter is a key list passed as JSON text."""
    if not isinstance(value, str) or not value.lstrip().startswith('['):
        return False
    try:
        return isinstance(json.loads(value), list)
    except ValueError:
        return False


def passes_keys_as_json(step: dict[str, Any]) -> bool:
    params = step.get('params')
    if step.get('op') != 'execute' or not isinstance(params, list):
        return False
    return any(json_array_param(p) for p in params)


def strict(value: Any) -> str:
    r"""A JSON value as canonical text: ``1`` and ``1.0`` differ."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False)
