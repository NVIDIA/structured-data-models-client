#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact key scenarios: independent inputs and current-client test helpers."""

from __future__ import annotations

import copy
import importlib.util
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contract/exact-key-vectors.json'
FORMAT = 'sdm-exact-key-vectors/1'
#: How a register entry names the vectors it covers, as the key vectors'.
REGISTER_REF = 'contract/exact-key-vectors.json#'

#: The first id, 2**53: past it a double holds only every other integer.
FIRST = 2**53
#: How many users, each with three orders.
USERS = 20
#: The entity asked for: 2**53 + 1, which a double reads as 2**53.
ASKED = FIRST + 1

STATIC = 'PREDICT users.age FOR EACH users.user_id'
SUM = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
LINK = {
    'source_table': 'orders',
    'source_column': 'user_id',
    'target_table': 'users',
    'target_column': 'user_id',
}
KEYS = {'users': 'user_id', 'orders': 'order_id'}


def setup(declared: bool) -> list[str]:
    r"""The tables, with the key and the link declared in SQL or not."""
    key = ' PRIMARY KEY' if declared else ''
    link = ' REFERENCES users (user_id)' if declared else ''
    return [
        f'CREATE TABLE users (user_id numeric{key}, age double precision)',
        f'INSERT INTO users SELECT {FIRST} + i, 20 + i '
        f'FROM generate_series(0, {USERS - 1}) i',
        f'CREATE TABLE orders (order_id bigint{key}, user_id numeric{link}, '
        'price double precision, ts timestamp)',
        f'INSERT INTO orders SELECT i, {FIRST} + i % {USERS}, 10 + i * 1.5, '
        "TIMESTAMP '2024-01-01' + make_interval(days => i) "
        f'FROM generate_series(0, {3 * USERS - 1}) i',
    ]


def _vector(
    vector_id: str,
    declared: bool,
    pql: str,
    from_source: dict[str, Any],
    edits: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        'id': vector_id,
        'setup': setup(declared),
        'from_source': from_source,
        'edits': edits,
        'pql': pql,
        'materialize': {
            'indices': [ASKED],
            'anchor_time': '2024-02-01',
            'random_seed': 42,
        },
        'columns': [['users', 'user_id'], ['orders', 'user_id']],
    }


VECTORS: list[dict[str, Any]] = [
    _vector(
        'pk.given', False, STATIC, {'primary_keys': {'users': 'user_id'}}, []
    ),
    _vector('pk.catalog', True, STATIC, {}, []),
    _vector(
        'fk.given',
        False,
        SUM,
        {'primary_keys': dict(KEYS), 'relationships': [dict(LINK)]},
        [],
    ),
    _vector(
        'fk.edited',
        False,
        SUM,
        {'primary_keys': dict(KEYS), 'relationships': []},
        [{'op': 'set_relationships', 'relationships': [dict(LINK)]}],
    ),
    _vector('fk.catalog', True, SUM, {}, []),
]
#: Each vector whose key is the caller's, and the control it must answer
#: as: the same tables with the key declared in SQL, asked the same.
CONTROL = {
    'pk.given': 'pk.catalog',
    'fk.given': 'fk.catalog',
    'fk.edited': 'fk.catalog',
}


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _regressions() -> ModuleType:
    return _module(
        'postgres_scenarios',
        ROOT / 'contract/support/postgres_scenarios.py',
    )


def _relationships(links: list[dict[str, str]]) -> list[Any]:
    from sdm_client import Relationship

    return [Relationship(**link) for link in links]


def observe(vector: dict[str, Any], dsn: str) -> dict[str, Any]:
    """Run the vector through the public client in a temporary database."""
    import psycopg

    regressions = _regressions()
    name, scoped = regressions._database(dsn)
    try:
        with psycopg.connect(scoped, autocommit=True) as connection:
            connection.execute(regressions.UTC)
            for statement in vector['setup']:
                connection.execute(statement)
            types = {}
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
        from sdm_client import PredictiveQuery, Schema
        from sdm_connectors import PostgresSource

        with psycopg.connect(scoped, autocommit=True) as connection:
            connection.execute(regressions.UTC)
            stage = 'from_source'
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    given = copy.deepcopy(vector['from_source'])
                    if 'relationships' in given:
                        given['relationships'] = _relationships(
                            given['relationships']
                        )
                    schema = Schema.from_source(
                        PostgresSource(connection), **given
                    )
                    for edit in vector['edits']:
                        # The one edit the vectors make: the links set.
                        assert edit['op'] == 'set_relationships', edit
                        schema.relationships = _relationships(
                            edit['relationships']
                        )
                    answer['schema'] = regressions._runner()._schema_projection(
                        schema
                    )
                    stage = 'materialize'
                    arguments = dict(vector['materialize'])
                    import pandas as pd

                    arguments['anchor_time'] = pd.Timestamp(
                        arguments['anchor_time']
                    )
                    request = PredictiveQuery(
                        vector['pql'], schema
                    ).materialize(**arguments)
                answer['payloads'] = regressions.payloads_projection(
                    request.payloads
                )
            except Exception as error:  # noqa: BLE001 - the answer observed
                answer['outcome'] = {
                    'result': 'refusal',
                    'stage': stage,
                    'error': type(error).__name__,
                    'code': getattr(error, 'code', None),
                }
        return answer
    finally:
        regressions._drop(dsn, name)
