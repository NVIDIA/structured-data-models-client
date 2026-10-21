#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Records the reference's answers to T2.14c's Postgres vectors into
``contract/numeric-spelling-vectors.json``.

Run with the reference checkout's interpreter (the environment of
``contract/reference-env.txt``), from this repository's root, with
``SDM_POSTGRES_TEST_DSN`` naming a Postgres 17 server the recorder may
create databases on::

    ~/code/sdmc-reference/.venv/bin/python -I \
        contract/tools/numeric_spelling_record.py

Written by T2.14c's test author from the whole-phase review's finding
(round 1, Codex, P2): a key column of Postgres's unconstrained ``numeric``
(the one type the Postgres provider hands over as exact text, marked
``sdm:exact_number``), keyed and linked by the caller with no constraint in
the database, whose two ends spell the same numbers differently: each
user's ``user_id`` written ``i.0`` and each order's ``user_id`` written
``i.00``, as Postgres keeps a ``numeric``'s scale. The core reads both ends
as text (register P2G-2), kept each spelling, so ``1.0`` and ``1.00`` were
different keys, and the second hop failed with ``INTERNAL_ERROR`` ("a row
looked up was already reached"), where the reference answers. The review's
reproduction: ``users(user_id numeric, age double precision)`` holding 8
users and ``orders(order_id bigint, user_id numeric, price double
precision, ts timestamp)`` holding 80 orders, asked ``PREDICT
SUM(orders.price, 0, 30, days) FOR EACH users.user_id`` for ``'0.0'`` and
``'1.0'``. Each vector:

- ``given.default`` and ``given.hops-2``: the keys and the link given to
  ``Schema.from_source`` (``primary_keys``, ``relationships``), nothing
  declared, materialized at the default number of hops (two) and at
  ``num_hops=2``;
- ``given.same-spelling``: the same, each order's ``user_id`` written
  ``i.0`` as its user's is, at the default number of hops;
- ``catalog.default``, ``catalog.hops-2``, ``catalog.same-spelling.default``
  and ``catalog.same-spelling.hops-2``: the same tables with the keys and
  the link declared in SQL (``PRIMARY KEY``, ``REFERENCES``), nothing
  given, asked the same, the link spelled ``i.00`` or ``i.0``. P2G-2
  answers a key the caller gives as the same tables with the key declared
  are; ``EXPECTED`` names the answer each vector is held to on the core.

For each the reference's public answer is kept by
``exact_keys_record.observe`` (T2.21's, itself built on
``postgres_regressions_record.py``'s helpers): the outcome, the public
schema, the digest of the request's payloads with ``metadata.request_id``
left out, the anchor times they carry, and each key column's Postgres type
by name. ``contract/tests/test_t214c_numeric_spelling.py`` calls the same
``observe`` on the core's Postgres routes.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contract/numeric-spelling-vectors.json'
FORMAT = 'sdm-numeric-spelling-vectors/1'
#: How a register entry names the vectors it covers, as the key vectors'.
REGISTER_REF = 'contract/numeric-spelling-vectors.json#'

#: How many users; each has ten orders.
USERS = 8
#: How each end spells its numbers: the key ``i.0``, the link ``i.00``.
KEY_SPELLING = '.0'
LINK_SPELLING = '.00'
QUESTION = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
LINK = {
    'source_table': 'orders',
    'source_column': 'user_id',
    'target_table': 'users',
    'target_column': 'user_id',
}
KEYS = {'users': 'user_id', 'orders': 'order_id'}
#: The review's materialize: two users, given as the key spells them.
MATERIALIZE = {
    'indices': ['0.0', '1.0'],
    'anchor_time': '2023-05-01',
    'random_seed': 7,
    'num_context_samples': 12,
}


def setup(declared: bool, link_spelling: str = LINK_SPELLING) -> list[str]:
    r"""The review's tables, the key and the link declared in SQL or not;
    each end's numbers cast from text, so Postgres keeps their scale: the
    key's ``i.0``, the link's ``link_spelling``."""
    key = ' PRIMARY KEY' if declared else ''
    link = ' REFERENCES users (user_id)' if declared else ''
    return [
        f'CREATE TABLE users (user_id numeric{key}, age double precision)',
        f"INSERT INTO users SELECT (i::text || '{KEY_SPELLING}')::numeric, "
        f'20 + i FROM generate_series(0, {USERS - 1}) i',
        f'CREATE TABLE orders (order_id bigint{key}, user_id numeric{link}, '
        'price double precision, ts timestamp)',
        f'INSERT INTO orders SELECT i, ((i % {USERS})::text || '
        f"'{link_spelling}')::numeric, 10 + i * 1.5, "
        "TIMESTAMP '2023-01-01' + make_interval(days => i * 7) "
        f'FROM generate_series(0, {10 * USERS - 1}) i',
    ]


def _vector(
    vector_id: str,
    declared: bool,
    hops: int | None,
    link_spelling: str = LINK_SPELLING,
) -> dict[str, Any]:
    materialize = copy.deepcopy(MATERIALIZE)
    if hops is not None:
        materialize['num_hops'] = hops
    return {
        'id': vector_id,
        'setup': setup(declared, link_spelling),
        'from_source': (
            {}
            if declared
            else {'primary_keys': dict(KEYS), 'relationships': [dict(LINK)]}
        ),
        'edits': [],
        'pql': QUESTION,
        'materialize': materialize,
        'columns': [['users', 'user_id'], ['orders', 'user_id']],
    }


VECTORS: list[dict[str, Any]] = [
    _vector('given.default', False, None),
    _vector('given.hops-2', False, 2),
    _vector('given.same-spelling', False, None, KEY_SPELLING),
    _vector('catalog.default', True, None),
    _vector('catalog.hops-2', True, 2),
    _vector('catalog.same-spelling.default', True, None, KEY_SPELLING),
    _vector('catalog.same-spelling.hops-2', True, 2, KEY_SPELLING),
]
#: Each vector and the reference's answer it is held to on the core: the
#: same tables with the keys declared in SQL (P2G-2: a key the caller gives
#: is answered as a declared one is) and each link spelled as its key is
#: (one number, one key: the reference's own answer to the keys given does
#: not change with the link's spelling), asked the same.
EXPECTED = {
    'given.default': 'catalog.same-spelling.default',
    'given.hops-2': 'catalog.same-spelling.hops-2',
    'given.same-spelling': 'catalog.same-spelling.default',
    'catalog.default': 'catalog.same-spelling.default',
    'catalog.hops-2': 'catalog.same-spelling.hops-2',
    'catalog.same-spelling.default': 'catalog.same-spelling.default',
    'catalog.same-spelling.hops-2': 'catalog.same-spelling.hops-2',
}


def _exact_keys() -> ModuleType:
    name = 'exact_keys_record'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / 'contract/tools/exact_keys_record.py'
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def observe(vector: dict[str, Any], dsn: str, core: bool) -> dict[str, Any]:
    r"""The public answer to ``vector`` over a new database on ``dsn``'s
    server, as ``exact_keys_record.observe`` observes it: on the core's
    Postgres routes with ``core``, else on the interpreter's own client."""
    answer: dict[str, Any] = _exact_keys().observe(vector, dsn, core)
    return answer


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
        'recorded_by': 'contract/tools/numeric_spelling_record.py',
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
