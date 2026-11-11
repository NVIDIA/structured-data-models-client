#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Postgres scenarios: independent inputs and current-client test helpers."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
#: The zone of every connection the recorder opens, set before anything is
#: read, so no answer depends on the recorder's environment (``PGTZ``) or
#: the server's default. The source reads every cursor through one
#: connection, which a vector's ``session`` statements then set further
#: (only ``time.new-york-auto-anchor``'s, on purpose); its ``setup`` names
#: no zone, as ``tests/test_contract_cases.py`` holds every vector file to.
UTC = "SET TimeZone TO 'UTC'"

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
    # integral values past 2**53, within int64, in a numeric
    # wider than Arrow's decimals; the reference keeps them exact.
    'numeric-100-0-past-2-53': ('numeric(100,0)', '9007199254740992 + n'),
    # json and jsonb features, which must read on PyArrow's
    # floor (14.0.1) as on its newest; and a NaN in a numeric(100,0),
    # which the reference sends as null.
    'json': ('json', "json_build_object('a', n)"),
    'jsonb': ('jsonb', "jsonb_build_object('a', n)"),
    'numeric-100-0-nan': (
        'numeric(100,0)',
        "CASE WHEN n = 1 THEN 'NaN'::numeric ELSE n END",
    ),
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
                    "INSERT INTO users SELECT i, '12:00+02'::timetz "
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
# the domain vector's orders also holding a column named x,
# which a seed key's row must not be read as.
VECTORS.append(
    {
        **copy.deepcopy(
            next(
                v for v in VECTORS if v['id'] == 'join.domain-not-null-text-key'
            )
        ),
        'id': 'join.domain-not-null-text-key-column-x',
    }
)
VECTORS[-1]['setup'].append('ALTER TABLE orders ADD COLUMN x integer DEFAULT 7')
VECTORS[-1]['columns'].append(['orders', 'x'])
# unconstrained numeric keys at both ends, which the
# reference sends as their exact text; small, and past 2**53, where a
# double would merge the eight keys into one.
for _name, _base in (('small', 0), ('past-2-53', 10**17)):
    VECTORS.append(
        {
            'id': f'key.numeric-unconstrained-{_name}',
            'session': [],
            'setup': [
                _SHOP_USERS.format('user_id numeric PRIMARY KEY, age integer'),
                f'INSERT INTO users SELECT {_base} + i, 20 + i '
                'FROM generate_series(0, 7) i',
                _SHOP_ORDERS.format(
                    'user_id numeric REFERENCES users(user_id)'
                ),
                f'INSERT INTO orders SELECT i, {_base} + (i % 8), '
                f'{_SHOP_FACTS} FROM generate_series(0, 79) i',
            ],
            'pql': SHOP_QUESTION,
            'materialize': {
                'indices': [_base, _base + 1],
                'anchor_time': '2024-03-01',
                'random_seed': 42,
            },
            'columns': [['users', 'user_id'], ['orders', 'user_id']],
        }
    )
# a composite primary key, and a composite foreign
# key, which the reference refuses at Schema.from_source (an unnamed
# ValueError: its graph discovery takes single-column keys only).
for _name, _setup, _columns in (
    (
        'key.composite-primary',
        [
            'CREATE TABLE users (user_id bigint, region int, age integer, '
            'PRIMARY KEY (user_id, region))',
            'INSERT INTO users SELECT i, 1, 20 + i FROM generate_series(0, 7) i',
            _SHOP_ORDERS.format('user_id bigint'),
            'INSERT INTO orders SELECT i, i % 8, '
            f'{_SHOP_FACTS} FROM generate_series(0, 79) i',
        ],
        [['users', 'user_id'], ['users', 'region']],
    ),
    (
        'key.composite-foreign',
        [
            'CREATE TABLE users (user_id bigint PRIMARY KEY, region int, '
            'age integer, UNIQUE (user_id, region))',
            'INSERT INTO users SELECT i, 1, 20 + i FROM generate_series(0, 7) i',
            'CREATE TABLE orders (order_id bigint PRIMARY KEY, user_id bigint '
            'REFERENCES users(user_id), region int, price double precision, '
            'ts timestamp, FOREIGN KEY (user_id, region) '
            'REFERENCES users(user_id, region))',
            'INSERT INTO orders SELECT i, i % 8, 1, '
            f'{_SHOP_FACTS} FROM generate_series(0, 79) i',
        ],
        [['orders', 'user_id'], ['orders', 'region']],
    ),
):
    VECTORS.append(
        {
            'id': _name,
            'session': [],
            'setup': _setup,
            'pql': SHOP_QUESTION,
            'materialize': {
                'indices': [0, 1],
                'anchor_time': '2024-03-01',
                'random_seed': 42,
            },
            'columns': _columns,
        }
    )


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
