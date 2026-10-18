#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Records the reference's answers to T2.3's round 1 regressions on
Postgres into ``contract/postgres-regression-vectors.json``.

Run with the reference checkout's interpreter (the environment of
``contract/reference-env.txt``), from this repository's root, with
``SDM_POSTGRES_TEST_DSN`` naming a Postgres 17 server the recorder may
create databases on::

    ~/code/sdmc-reference/.venv/bin/python -I \
        contract/tools/postgres_regressions_record.py

Written by T2.3's independent test author from the round 1 review's
findings, each a public-API call over a ``psycopg`` connection, in a
database of its own whose tables the vector's ``setup`` builds with
Postgres's own types (which the contract loader does not make):

- ``time.new-york-auto-anchor``: ``timestamptz`` facts read in a session
  whose ``TimeZone`` is America/New_York, materialized with no anchor time,
  so the automatic anchor and every training anchor come from the time
  bounds (the reference's: 2024-04-10T00:00:00Z);
- ``type.<name>``: a feature column of ``uuid``, ``inet``, unconstrained
  ``numeric``, ``numeric[]`` or ``uuid[]`` beside the key and target;
- ``relation.dollar``: the entity table named ``users$1``, a legal
  quoted identifier holding what a bind marker looks like, read through
  ``tables={'users': 'users$1'}`` and asked a keyed question;
- from the second review, over the shop's shape (``users``, ``orders``),
  asked ``SHOP_QUESTION``: ``key.real-fractional-linked``, a ``real`` key
  of ``items`` holding fractional values (``i + 0.1``) that ``orders``
  links to; ``key.real-fractional-entity``, the entity table's ``real``
  key holding them, its entities given as those floats;
  ``join.domain-not-null-text-key``, ``orders`` holding a ``NOT NULL``
  domain column and joined on a text key; ``column.dollar``, an entity
  column named ``cost$1``, read by the keyed lookup of ``users``; and
  ``type.timetz``, a ``timetz`` column read by ``Schema.from_source``.

For each the reference's public answer is kept: the outcome (result,
stage, exception class and ``StructuredDataError`` code), the public
schema (``run_python``'s projection), the digest of the request's
payloads with ``metadata.request_id`` left out (as the comparator's
payload rule does), each anchor time the payloads carry, and each
Postgres column's type by name (``format_type``). ``observe`` is imported
by ``contract/tests/test_t23_postgres_regressions.py``, so python-core is
observed by the same code, forced onto the core's Postgres routes.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import uuid
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contract/postgres-regression-vectors.json'
FORMAT = 'sdm-postgres-regression-vectors/1'
#: How a register entry names the vectors it covers, as the key vectors'.
REGISTER_REF = 'contract/postgres-regression-vectors.json#'

_USERS = 'CREATE TABLE users(id INTEGER PRIMARY KEY, age REAL, feature {})'
_FILL = 'INSERT INTO users SELECT n, n * 10, {} FROM generate_series(1, 12) n'
_UUID = "('00000000-0000-0000-0000-' || lpad(n::text, 12, '0'))::uuid"
#: Each feature type of the round 1 probe: its Postgres type and values.
FEATURES = {
    'uuid': ('uuid', _UUID),
    'inet': ('inet', "('192.168.0.' || n)::inet"),
    'numeric': ('numeric', 'n + 0.25'),
    'numeric-array': ('numeric[]', 'ARRAY[n::numeric]'),
    'uuid-array': ('uuid[]', f'ARRAY[{_UUID}]'),
}
#: The second review's shape: ``users``, and ``orders`` whose columns
#: before ``price`` and ``ts`` each vector names, and its question.
_SHOP_USERS = 'CREATE TABLE users ({})'
_SHOP_ORDERS = (
    'CREATE TABLE orders (order_id bigint PRIMARY KEY, {}, '
    'price double precision, ts timestamp)'
)
_SHOP_FACTS = (
    "10 + i * 1.5, TIMESTAMP '2023-01-01' + make_interval(days => i * 7)"
)
SHOP_QUESTION = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
#: The keyed question and its entities, as the review's probe asks it.
KEYED = {
    'pql': 'PREDICT users.age FOR EACH users.id',
    'materialize': {
        'indices': [1],
        'num_context_samples': 3,
        'random_seed': 42,
    },
}

VECTORS: list[dict[str, Any]] = [
    {
        'id': 'time.new-york-auto-anchor',
        'session': ["SET TimeZone TO 'America/New_York'"],
        'setup': [
            'CREATE TABLE users(id INTEGER PRIMARY KEY, age REAL)',
            'INSERT INTO users SELECT n, n * 10 FROM generate_series(1, 12) n',
            'CREATE TABLE orders(id INTEGER PRIMARY KEY, user_id INTEGER '
            'REFERENCES users(id), price REAL, ts TIMESTAMPTZ)',
            'INSERT INTO orders SELECT n, (n % 12) + 1, n * 0.5, '
            "'2024-01-01 00:00:00+00'::timestamptz + n * interval '2 hours' "
            'FROM generate_series(1, 1200) n',
        ],
        'pql': 'PREDICT SUM(orders.price, 0, 7, days) FOR EACH users.id',
        'materialize': copy.deepcopy(KEYED['materialize']),
        'columns': [['orders', 'ts']],
    },
    *[
        {
            'id': f'type.{name}',
            'session': [],
            'setup': [_USERS.format(kind), _FILL.format(values)],
            **copy.deepcopy(KEYED),
            'columns': [['users', 'feature']],
        }
        for name, (kind, values) in FEATURES.items()
    ],
    {
        'id': 'relation.dollar',
        'session': [],
        'setup': [
            _USERS.format('text'),
            _FILL.format("'abc'"),
            'ALTER TABLE users RENAME TO "users$1"',
        ],
        'tables': {'users': 'users$1'},
        **copy.deepcopy(KEYED),
        'columns': [['users$1', 'id']],
    },
    *[
        {
            'id': name,
            'session': [],
            'setup': setup,
            'pql': SHOP_QUESTION,
            'materialize': {
                'indices': indices,
                'anchor_time': '2024-03-01',
                'random_seed': 42,
            },
            'columns': columns,
        }
        for name, setup, indices, columns in [
            (
                'key.real-fractional-linked',
                [
                    _SHOP_USERS.format(
                        'user_id bigint PRIMARY KEY, age integer'
                    ),
                    'INSERT INTO users SELECT i, 20 + i '
                    'FROM generate_series(0, 7) i',
                    'CREATE TABLE items (item_id real PRIMARY KEY, cat text)',
                    'INSERT INTO items SELECT i + 0.1, '
                    "(ARRAY['a', 'b', 'c'])[1 + i % 3] "
                    'FROM generate_series(0, 4) i',
                    _SHOP_ORDERS.format(
                        'user_id bigint REFERENCES users(user_id), '
                        'item_id real REFERENCES items(item_id)'
                    ),
                    'INSERT INTO orders SELECT i, i % 8, (i % 5) + 0.1, '
                    f'{_SHOP_FACTS} FROM generate_series(0, 79) i',
                ],
                [0, 1],
                [['items', 'item_id'], ['orders', 'item_id']],
            ),
            (
                'key.real-fractional-entity',
                [
                    _SHOP_USERS.format('user_id real PRIMARY KEY, age integer'),
                    'INSERT INTO users SELECT i + 0.1, 20 + i '
                    'FROM generate_series(0, 7) i',
                    _SHOP_ORDERS.format(
                        'user_id real REFERENCES users(user_id)'
                    ),
                    'INSERT INTO orders SELECT i, (i % 8) + 0.1, '
                    f'{_SHOP_FACTS} FROM generate_series(0, 79) i',
                ],
                [0.1, 1.1],
                [['users', 'user_id'], ['orders', 'user_id']],
            ),
            (
                'join.domain-not-null-text-key',
                [
                    'CREATE DOMAIN qty AS integer NOT NULL',
                    _SHOP_USERS.format('user_id text PRIMARY KEY, age integer'),
                    "INSERT INTO users SELECT 'u' || i, 20 + i "
                    'FROM generate_series(0, 7) i',
                    _SHOP_ORDERS.format(
                        'user_id text REFERENCES users(user_id), n qty'
                    ),
                    "INSERT INTO orders SELECT i, 'u' || (i % 8), i, "
                    f'{_SHOP_FACTS} FROM generate_series(0, 79) i',
                ],
                ['u0', 'u1'],
                [['users', 'user_id'], ['orders', 'n']],
            ),
            (
                'column.dollar',
                [
                    _SHOP_USERS.format(
                        'user_id bigint PRIMARY KEY, "cost$1" integer'
                    ),
                    'INSERT INTO users SELECT i, 20 + i '
                    'FROM generate_series(0, 7) i',
                    _SHOP_ORDERS.format(
                        'user_id bigint REFERENCES users(user_id)'
                    ),
                    'INSERT INTO orders SELECT i, i % 8, '
                    f'{_SHOP_FACTS} FROM generate_series(0, 79) i',
                ],
                [0, 1],
                [['users', 'cost$1']],
            ),
            (
                'type.timetz',
                [
                    _SHOP_USERS.format(
                        'user_id bigint PRIMARY KEY, x timetz, age integer'
                    ),
                    "INSERT INTO users SELECT i, TIME WITH TIME ZONE '12:00+02' "
                    '+ make_interval(mins => i), 20 + i '
                    'FROM generate_series(0, 7) i',
                    _SHOP_ORDERS.format(
                        'user_id bigint REFERENCES users(user_id)'
                    ),
                    'INSERT INTO orders SELECT i, i % 8, '
                    f'{_SHOP_FACTS} FROM generate_series(0, 79) i',
                ],
                [0, 1],
                [['users', 'x']],
            ),
        ]
    ],
]


def _runner() -> ModuleType:
    name = 'run_python'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / 'contract/runners/run_python.py'
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def payloads_projection(payloads: Any) -> dict[str, Any] | None:
    r"""The digest of the payloads, ``metadata.request_id`` left out, and
    every anchor time they carry, sorted; null with no payload."""
    if not payloads:
        return None
    bodies = copy.deepcopy([dict(each) for each in payloads])
    anchors: set[str] = set()
    for body in bodies:
        (body.get('metadata') or {}).pop('request_id', None)
        for part in ('context', 'predict'):
            table = (body.get(part) or {}).get('instance_table') or {}
            columns = table.get('columns') or []
            if '__kumo_anchor_time' in columns:
                at = columns.index('__kumo_anchor_time')
                anchors.update(str(row[at]) for row in table.get('rows') or [])
    text = json.dumps(bodies, sort_keys=True, separators=(',', ':'))
    return {
        'digest': hashlib.sha256(text.encode('utf-8')).hexdigest(),
        'anchors': sorted(anchors),
    }


def _database(dsn: str) -> tuple[str, str]:
    import psycopg
    from psycopg.conninfo import make_conninfo

    name = f'sdm_t23_{uuid.uuid4().hex[:16]}'
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute(f'CREATE DATABASE {name}')
    return name, make_conninfo(dsn, dbname=name)


def _drop(dsn: str, name: str) -> None:
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute(f'DROP DATABASE IF EXISTS {name} WITH (FORCE)')


def observe(vector: dict[str, Any], dsn: str, core: bool) -> dict[str, Any]:
    r"""The public answer to ``vector`` over a new database on ``dsn``'s
    server: on the core's Postgres routes with ``core``, else on the
    interpreter's own client (the reference's, when recording)."""
    import psycopg

    name, scoped = _database(dsn)
    previous = os.environ.pop('SDM_CORE_ROUTES', None)
    try:
        with psycopg.connect(scoped, autocommit=True) as connection:
            for statement in vector['setup']:
                connection.execute(statement)
        types = {}
        with psycopg.connect(scoped) as connection:
            for table, column in vector['columns']:
                row = connection.execute(
                    'SELECT format_type(a.atttypid, a.atttypmod) '
                    'FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid '
                    'WHERE c.relname = %s AND a.attname = %s AND a.attnum > 0',
                    [table, column],
                ).fetchone()
                types[f'{table}.{column}'] = row[0] if row else None
        answer: dict[str, Any] = {
            'outcome': {
                'result': 'success',
                'stage': 'materialize',
                'error': None,
                'code': None,
            },
            'schema': None,
            'payloads': None,
            'postgres': types,
        }
        if core:
            from sdm_client import _routing

            os.environ['SDM_CORE_ROUTES'] = ','.join(
                path
                for path in _routing.PATHS
                if path.endswith('/postgres') and path in _routing.CORE_ROUTED
            )
        from sdm_client import PredictiveQuery, Schema
        from sdm_connectors import PostgresSource

        with psycopg.connect(scoped, autocommit=True) as connection:
            for statement in vector['session']:
                connection.execute(statement)
            stage = 'from_source'
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    schema = Schema.from_source(
                        PostgresSource(connection), tables=vector.get('tables')
                    )
                    answer['schema'] = _runner()._schema_projection(schema)
                    stage = 'materialize'
                    arguments = dict(vector['materialize'])
                    if 'anchor_time' in arguments:
                        import pandas as pd

                        arguments['anchor_time'] = pd.Timestamp(
                            arguments['anchor_time']
                        )
                    request = PredictiveQuery(
                        vector['pql'], schema
                    ).materialize(**arguments)
                answer['payloads'] = payloads_projection(request.payloads)
            except Exception as error:  # noqa: BLE001 - the answer observed
                answer['outcome'] = {
                    'result': 'refusal',
                    'stage': stage,
                    'error': type(error).__name__,
                    'code': getattr(error, 'code', None),
                }
        return answer
    finally:
        os.environ.pop('SDM_CORE_ROUTES', None)
        if previous is not None:
            os.environ['SDM_CORE_ROUTES'] = previous
        _drop(dsn, name)


def main() -> int:
    import pandas as pd
    import psycopg
    import pyarrow as pa

    reference = Path.home() / 'code/sdmc-reference'
    commit = subprocess.run(
        ['git', '-C', str(reference), 'rev-parse', 'HEAD'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    import sdm_client

    if not Path(sdm_client.__file__).resolve().is_relative_to(reference):
        print("sdm_client is not the reference checkout's", file=sys.stderr)
        return 1
    dsn = os.environ.get('SDM_POSTGRES_TEST_DSN')
    if not dsn:
        print('SDM_POSTGRES_TEST_DSN names no server', file=sys.stderr)
        return 1
    with psycopg.connect(dsn) as connection:
        server = connection.info.server_version // 10000
    vectors = []
    for vector in VECTORS:
        entry = copy.deepcopy(vector)
        entry['reference'] = observe(vector, dsn, core=False)
        vectors.append(entry)
        print(vector['id'], entry['reference']['outcome'], file=sys.stderr)
    document = {
        'format': FORMAT,
        'recorded_by': 'contract/tools/postgres_regressions_record.py',
        'reference': commit,
        'cell': 'ref-pandas3',
        'environment': {
            'python': platform.python_version(),
            'pandas': pd.__version__,
            'pyarrow': pa.__version__,
            'psycopg': psycopg.__version__,
            'postgres': server,
        },
        'register_ref': REGISTER_REF,
        'vectors': vectors,
    }
    OUT.write_text(json.dumps(document, indent=1) + '\n', encoding='utf-8')
    print(f'{len(vectors)} vectors written', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
