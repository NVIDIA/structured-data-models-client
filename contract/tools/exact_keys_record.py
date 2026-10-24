#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Records the reference's answers to T2.21's Postgres vectors into
``contract/exact-key-vectors.json``.

Run with the reference checkout's interpreter (the environment of
``contract/reference-env.txt``), from this repository's root, with
``SDM_POSTGRES_TEST_DSN`` naming a Postgres 17 server the recorder may
create databases on::

    ~/code/sdmc-reference/.venv/bin/python -I contract/tools/exact_keys_record.py

Written by T2.21's test author from the mid-phase design check's
reproduction: a key column of Postgres's unconstrained ``numeric`` (the one
type the Postgres provider hands over as exact text, marked
``sdm:exact_number``, which the contract loader does not make), holding 20
ids from 2**53 (9007199254740992), where a double tells 2**53 and 2**53 + 1
apart no more. ``users(user_id numeric, age double precision)`` and
``orders(order_id bigint, user_id numeric, price double precision, ts
timestamp)``, each user with three orders. Each vector names how the key is
made the key:

- ``pk.given``: the reproduction, the key given to ``Schema.from_source``
  (``primary_keys={'users': 'user_id'}``), asked ``PREDICT users.age FOR
  EACH users.user_id`` for 2**53 + 1;
- ``fk.given``: both keys and the link from ``orders.user_id`` given,
  asked ``PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id``;
- ``fk.edited``: both keys given and no link (``relationships=[]``), the
  link then set on the schema (``schema.relationships = [...]``), asked
  the same;
- ``pk.catalog`` and ``fk.catalog``, the controls: the same tables with the
  key and the link declared in SQL (``PRIMARY KEY``, ``REFERENCES``),
  nothing given, asked each question.

For each the reference's public answer is kept, as
``postgres_regressions_record.py`` keeps it (whose helpers this uses):
the outcome (result, stage, exception class and ``StructuredDataError``
code), the public schema (``run_python``'s projection), the digest of the
request's payloads with ``metadata.request_id`` left out, the anchor times
they carry, and each key column's Postgres type by name (``format_type``).
``observe`` is imported by ``contract/tests/test_t221_exact_keys.py``, so
python-core is observed by the same code, forced onto the core's Postgres
routes.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import platform
import subprocess
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
        'postgres_regressions_record',
        ROOT / 'contract/tools/postgres_regressions_record.py',
    )


def _relationships(links: list[dict[str, str]]) -> list[Any]:
    from sdm_client import Relationship

    return [Relationship(**link) for link in links]


def observe(vector: dict[str, Any], dsn: str, core: bool) -> dict[str, Any]:
    r"""The public answer to ``vector`` over a new database on ``dsn``'s
    server: on the core's Postgres routes with ``core``, else on the
    interpreter's own client (the reference's, when recording)."""
    import psycopg

    regressions = _regressions()
    name, scoped = regressions._database(dsn)
    previous = os.environ.get('SDM_CORE_ROUTES')
    os.environ['SDM_CORE_ROUTES'] = ''
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
        if core:
            from sdm_client import _routing

            os.environ['SDM_CORE_ROUTES'] = ','.join(
                path
                for path in _routing.PATHS
                if path.endswith('/postgres') and path in _routing.CORE_ROUTED
            )
        else:
            os.environ['SDM_CORE_ROUTES'] = ''
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
        os.environ.pop('SDM_CORE_ROUTES', None)
        if previous is not None:
            os.environ['SDM_CORE_ROUTES'] = previous
        regressions._drop(dsn, name)


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
        'recorded_by': 'contract/tools/exact_keys_record.py',
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
