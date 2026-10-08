# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The verifier's routes (``CONTRACT.md``, "Routes"): public calls, offline,
that take the six branches, made through today's Python.

Each is the route ``investigations/P77-coverage-gaps-verifier.md`` gives
(its probes are ``rc-p77-tmp/verify77/findings.py``), over the parity
corpus's generated data. Each function makes the calls and returns what the
last one gave; what it raised propagates.
"""

from __future__ import annotations

import contextlib
import itertools
import os
import sys
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

# The real coverage.py, before scripts/parity, which holds a coverage.py of
# its own, goes on the path.
import coverage  # noqa: F401

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _path in (
    REPO / 'scripts' / 'parity',
    REPO / 'tests' / 'rust_core',
    REPO / 'tests' / 'rust_core' / 'sampling',
    REPO / 'tests' / 'rust_core' / 'sampling' / 'driver',
    REPO / 'tests' / 'rust_core' / 'rfm',
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

COUNT = 'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id'
POSTGRES_DSN = 'SDM_POSTGRES_TEST_DSN'
_SCHEMAS = itertools.count()


def frames(name: str = 'shop') -> dict[str, Any]:
    from _parity.data import generate

    return generate(name, 0)


def sqlite_source(tables: dict[str, Any], *statements: str) -> Any:
    import adbc_driver_sqlite.dbapi as dbapi
    from sdm_connectors import SQLiteSource

    path = Path(tempfile.mkdtemp(prefix='sdm-p78-sqlite-')) / 'data.sqlite'
    connection = dbapi.connect(str(path))
    with connection.cursor() as cursor:
        for name, frame in tables.items():
            cursor.adbc_ingest(name, frame, mode='create')
        for statement in statements:
            cursor.execute(statement)
    connection.commit()
    return SQLiteSource(connection)


def duckdb_source(tables: dict[str, Any], *statements: str) -> Any:
    from _parity.runner import _duckdb
    from sdm_connectors import DuckDBSource

    connection = _duckdb(tables, {'column_types': {}, 'session': {}})
    for statement in statements:
        connection.execute(statement)
    return DuckDBSource(connection)


@contextlib.contextmanager
def postgres_source(tables: dict[str, Any], *statements: str) -> Iterator[Any]:
    r"""A PostgreSQL source over ``tables``, in a schema of its own that is
    dropped afterwards; ``{schema}`` in a statement is that schema."""
    import psycopg
    from _parity.runner import load_postgres
    from sdm_connectors import PostgresSource

    dsn = os.environ.get(POSTGRES_DSN)
    assert dsn, f'{POSTGRES_DSN} is not set: the phase env names the server'
    schema = f'sdm_p78_{os.getpid()}_{next(_SCHEMAS)}'
    connection = psycopg.connect(dsn, autocommit=True)
    try:
        connection.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')
        load_postgres(connection, schema, tables, {})
        for statement in statements:
            connection.execute(statement.format(schema=schema))
        yield PostgresSource(connection, schema=schema)
    finally:
        with contextlib.suppress(Exception):
            connection.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')
        connection.close()


def ask(schema: Any, **gather: Any) -> Any:
    from sdm_client import KumoRelational, PredictiveQuery

    return PredictiveQuery(COUNT, schema).materialize(
        model=KumoRelational(), **gather
    )


# --- The routes -------------------------------------------------------------------


def text_of_lists() -> Any:
    r"""A text column edited in place after ``Schema.from_data``: its first
    cell text, the rest tuples (verifier F1)."""
    from sdm_client import Schema

    tables = frames()
    schema = Schema.from_data(tables)
    users = tables['users']
    users['country'] = ['x y'] + [('a', 'b')] * (len(users) - 1)
    schema['users']['country'].stype = 'text'
    return ask(schema, indices=[30, 31], num_context_samples=3, random_seed=1)


def _list_column(schema: Any) -> Any:
    from sdm_client import Column

    assert 'tags' not in schema['users'].columns, (
        'Schema.from_source kept the list column it should leave out'
    )
    schema['users'].columns['tags'] = Column(
        'tags', 'string', 'multicategorical'
    )
    return ask(schema, indices=[1, 2], num_context_samples=5, random_seed=7)


def list_column_duckdb() -> Any:
    r"""A DuckDB ``VARCHAR[]`` column ``Schema.from_source`` leaves out, put
    back as multicategorical (verifier F2)."""
    from sdm_client import Schema

    source = duckdb_source(
        frames(),
        'ALTER TABLE users ADD COLUMN tags VARCHAR[]',
        "UPDATE users SET tags = ['a', 'b']",
    )
    return _list_column(Schema.from_source(source))


def list_column_postgres() -> Any:
    r"""The same with a PostgreSQL ``text[]`` column (verifier F2)."""
    from sdm_client import Schema

    with postgres_source(
        frames(),
        'ALTER TABLE {schema}.users ADD COLUMN tags text[]',
        "UPDATE {schema}.users SET tags = ARRAY['a', 'b']",
    ) as source:
        return _list_column(Schema.from_source(source))


def no_valid_key(tables: dict[str, Any] | None = None) -> Any:
    r"""A key column set to NaN in place after ``Schema.from_data``, in a
    table without a time column (verifier F4)."""
    import numpy as np
    from sdm_client import Schema

    tables = frames() if tables is None else tables
    schema = Schema.from_data(tables)
    tables['products']['product_id'] = np.nan
    return ask(schema, indices=[1, 2], num_context_samples=5, random_seed=7)


def entity_times_at_max_memory() -> Any:
    r"""Every entity time ``pd.Timestamp.max``, in memory (verifier F3)."""
    import pandas as pd
    from sdm_client import Schema

    tables = frames('shop-referrals')
    users = tables['users']
    users['joined_at'] = pd.Series([pd.Timestamp.max] * len(users))
    schema = Schema.from_data(tables)
    schema['users'].time_column = 'joined_at'
    return ask(schema, indices=[1, 2], num_context_samples=5, random_seed=7)


def entity_times_at_max_sqlite() -> Any:
    r"""The same on SQLite, the times stored as the text of that instant
    (verifier F3)."""
    from sdm_client import Schema

    source = sqlite_source(
        frames('shop-referrals'),
        "UPDATE users SET joined_at = '2262-04-11 23:47:16.854775807'",
    )
    schema = Schema.from_source(source)
    schema['users'].time_column = 'joined_at'
    return ask(schema, indices=[1, 2], num_context_samples=5, random_seed=7)


def _unseeded(schema: Any) -> Any:
    return ask(schema, indices=[1, 2], num_context_samples=30, random_seed=None)


def unseeded_sqlite() -> Any:
    r"""``random_seed=None`` over SQLite (verifier F5)."""
    from sdm_client import Schema

    return _unseeded(Schema.from_source(sqlite_source(frames())))


def unseeded_postgres() -> Any:
    r"""``random_seed=None`` over PostgreSQL (verifier F5)."""
    from sdm_client import Schema

    with postgres_source(frames()) as source:
        return _unseeded(Schema.from_source(source))


#: Each branch's routes, by the branch's name.
ROUTES: dict[str, dict[str, Callable[[], Any]]] = {
    'text-of-lists': {'memory': text_of_lists},
    'list-column-on-sql': {
        'duckdb': list_column_duckdb,
        'postgres': list_column_postgres,
    },
    'no-valid-key': {'memory': no_valid_key},
    'entity-times-at-max': {
        'memory': entity_times_at_max_memory,
        'sqlite': entity_times_at_max_sqlite,
    },
    'sqlite-unseeded': {'sqlite': unseeded_sqlite},
    'postgres-unseeded': {'postgres': unseeded_postgres},
}
