#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Records the Postgres value rules' expectations (T2.14d; MID2 P2-3) into
``contract/l5-vectors.json``'s ``postgres_values``, beside L5's tables.

Written by T2.14b's test author (T2.14d rides along in that landing),
before the implementation. Each rule is one the Postgres provider
(``_rust/providers/postgres.py``) holds at T2.14a's ledger: an interval
read as the driver's ``timedelta`` (a month folded into 30 days, no months
left), a JSON scalar as Python's text of it (JSON ``true`` as ``True``), a
``numeric`` past ``int64`` or past 76 digits read by its declaration, the
key set bound as JSON text, and a seed time written as pandas'
``Timestamp.isoformat`` writes it. Run twice, from this repository's root,
each run writing its own part and keeping the other's, with
``SDM_POSTGRES_TEST_DSN`` naming a Postgres server (a database is made for
the run and dropped after it)::

    <reference interpreter> -I contract/tools/postgres_values_record.py \
        --reference CHECKOUT
    <test-ops candidate interpreter> -I contract/tools/postgres_values_record.py

- With ``--reference`` (the reference checkout at
  ``contract/reference-env.txt``'s commit, its environment's interpreter):
  for each column of :data:`COLUMNS`, each in a table of its own
  (:data:`SETUP`), ``driver``, psycopg's rows of ``SELECT id, v ... ORDER
  BY id`` as the reference's Postgres backend reads them (its ``_frame``),
  each value's canonical cell (``run_python.py``'s ``canonical_cell``: what
  the comparator holds a statement's result to); and ``reference``, the
  same rows after the reference's own reading of the column's declared type
  (``PostgresTable._to_dtype`` over ``information_schema``, then
  ``_sanitize_postgres_frame``), each value's canonical cell, or the
  exception that reading raises, by class, or ``unsupported`` where the
  reference leaves the column out. For :data:`SEEDS`, ``seed_times``: the
  text the reference binds each seed's key and time as
  (``PostgresSampler._coerce_id`` and ``_serialize_time``).
- Without it (the candidate's client, a test-ops build): for each column,
  ``arrow``, the Arrow type the core admits it as through the Postgres
  provider (the test-only fetch), as ``test_l5_conversion.py`` spells a
  type, or the core's refusal by code; the conversion table's practice, so
  a change of what reaches the core is a change here, reviewed.
"""

from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contract/l5-vectors.json'
SECTION = 'postgres_values'

#: Each column, its Postgres type, its values (``id`` 1 onwards, a null
#: last) and the rule of the provider it exercises.
COLUMNS: dict[str, dict[str, Any]] = {
    'iv': {
        'type': 'interval',
        'values': ["'1 month 2 days 3 seconds'", "'-1 year'", "'0'"],
        'rule': "an interval read as the driver's timedelta, months 0",
    },
    'js': {
        'type': 'jsonb',
        'values': [
            "'true'",
            "'false'",
            "'3'",
            '\'"x"\'',
            '\'{"a":1}\'',
            "'null'",
        ],
        'rule': "a JSON scalar as Python's text of it (true as True)",
    },
    'jn': {
        'type': 'json',
        'values': ["'true'", "'2.50'", '\'[1, "b"]\''],
        'rule': "a JSON scalar as Python's text of it (true as True)",
    },
    'nw': {
        'type': 'numeric(100,0)',
        'values': ['9223372036854775808', '1', '-9223372036854775809'],
        'rule': 'a whole numeric past 76 digits and past int64',
    },
    'nf': {
        'type': 'numeric(90,3)',
        'values': ['1.5', '-2.25', '12345678901234567.125'],
        'rule': 'a fractional numeric past 76 digits, read as a double',
    },
    'nx': {
        'type': 'numeric',
        'values': ['12345678901234567890.123', '1', '0.1'],
        'rule': 'a numeric with no precision, read as its exact text',
    },
}
#: The tables: one per column, ``id`` and the column's values as ``v``,
#: then one of events keyed by user and time, which :data:`SEEDS` seed.
SETUP = [
    statement
    for name, column in COLUMNS.items()
    for statement in (
        f'CREATE TABLE t_{name} (id bigint PRIMARY KEY, v {column["type"]})',
        f'INSERT INTO t_{name} VALUES '
        + ', '.join(
            f'({at}, {value})'
            for at, value in enumerate([*column['values'], 'NULL'], start=1)
        ),
    )
] + [
    'CREATE TABLE events (uid bigint, at timestamp, v bigint)',
    "INSERT INTO events VALUES (1, '2024-01-02 03:04:05', 10), "
    "(1, '2024-01-02 03:04:05.5', 11), (2, '2024-01-03 00:00:00', 12)",
]
#: The seeds of a keyed read of ``events`` by ``uid`` within a window
#: ending at each seed's time: each a key and its end time in nanoseconds
#: since 1970-01-01 on the wall clock (whole seconds, microseconds,
#: nanoseconds, before 1970).
SEEDS = [
    [1, 1704164645000000000],
    [1, 1704164645500000000],
    [2, 1704240000000000001],
    [2, -86399999999000],
]


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def runner() -> ModuleType:
    return _module('pgv_run_python', ROOT / 'contract/runners/run_python.py')


def cells(values: list[Any]) -> list[Any]:
    r"""Each value's canonical cell (``run_python.py``'s ``canonical_cell``)."""
    odd: list[str] = []
    return [runner().canonical_cell(value, odd) for value in values]


class Database:
    r"""A Postgres database made for one run, :data:`SETUP` in it."""

    def __init__(self, dsn: str) -> None:
        import psycopg
        from psycopg.conninfo import make_conninfo

        self.dsn = dsn
        self.name = f'sdm_pgv_{uuid.uuid4().hex[:16]}'
        with psycopg.connect(dsn, autocommit=True) as admin:
            admin.execute(f'CREATE DATABASE {self.name}')
        self.scoped = make_conninfo(dsn, dbname=self.name)
        with psycopg.connect(self.scoped, autocommit=True) as connection:
            for statement in SETUP:
                connection.execute(statement)

    def connect(self) -> Any:
        import psycopg

        return psycopg.connect(
            self.scoped, autocommit=True, options='-c TimeZone=Etc/UTC'
        )

    def drop(self) -> None:
        import psycopg

        with psycopg.connect(self.dsn, autocommit=True) as admin:
            admin.execute(f'DROP DATABASE IF EXISTS {self.name} WITH (FORCE)')


def _arrow_type(field: Any) -> str:
    r"""``field``'s type as ``test_l5_conversion.py`` spells it."""
    import pyarrow as pa

    kind = field.type
    name = None
    if isinstance(kind, pa.BaseExtensionType):
        name = kind.extension_name
        kind = kind.storage_type
    elif field.metadata and b'ARROW:extension:name' in field.metadata:
        name = field.metadata[b'ARROW:extension:name'].decode()
    return str(kind) + (f' {name}' if name else '')


def admitted(provider: Any, name: str) -> Any:
    r"""``t_<name>``'s ``id`` and ``v`` read through ``provider`` and
    admitted by the core (the test-only fetch), in ``id`` order, or the
    core's refusal as ``refused <code>``."""
    import pyarrow as pa
    from sdm_client._rust import codec
    from sdm_client._rust.run import run
    from sdm_client.errors import StructuredDataError

    fetch = {
        'request': {
            'request': 'scan',
            'table': {'source': [f't_{name}']},
            'columns': ['id', 'v'],
            'filters': [],
            'limit': None,
        },
        # Under the database's schema, as every Postgres read names one
        # (T3.7).
        'provider': {
            'provider': 'sql',
            'dialect': 'postgres',
            'namespace': ['public'],
        },
        'echo': True,
    }
    try:
        finished = run(
            {'op': 'test', 'test': 'fetch', 'fetch': 1},
            {1: codec.attachment(fetch)},
            provider=provider,
        )
    except StructuredDataError as error:
        return f'refused {error.code}'
    stream = finished.take(finished.result['data'])
    return (
        pa.RecordBatchReader._import_from_c_capsule(stream.__arrow_c_stream__())
        .read_all()
        .sort_by('id')
    )
