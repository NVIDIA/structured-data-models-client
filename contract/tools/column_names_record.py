#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Records the reference's answers to T2.18's column-name vectors into
``contract/column-name-vectors.json``.

Run with the reference checkout's interpreter (the environment of
``contract/reference-env.txt``, the ``ref-pandas3`` cell), from this
repository's root, ``SDM_POSTGRES_TEST_DSN`` naming a Postgres 17 server
the recorder may create databases on::

    ~/code/sdmc-reference/.venv/bin/python -I contract/tools/column_names_record.py

Written by T2.18's test author (KO2-36: any column name works). A vector
is one character in one column's name, in one role, on one database:

- the characters: each the reference's ``schema.py`` leaves a column out
  for, a dot, a backtick, a line feed and a carriage return, in the name
  ``a<character>b``, spelled in the query backtick-quoted, a backtick
  doubled inside the quotes and a line break kept as it is;
- the roles, over the contract's ``shop`` base (users and their orders):
  ``feature``, a column of users the query does not name; ``target``, a
  text class of users predicted; ``entity``, users' declared primary key,
  which ``orders.user_id`` references, the query's entity; ``aggregation``, the
  column of orders the target sums;
- the databases: DuckDB, and SQLite and Postgres loaded from the DuckDB
  tables by the contract loader, as a case is (``run_python._open``).

Each role is recorded once more on each database with the column named
``a_b``, its *twin*: the same data and query under a name the reference
can say. Column names should not matter (KO2-36), so the core's answer for
a vector is its twin's reference answer with the name ``a_b`` read as the
vector's name and the twin's query as the vector's; the vector's own
reference answer is what ``main`` does, a difference T2.18 registers.

The answer is the public calls' (``Schema.from_source`` then
``materialize``): the outcome (success, or the refusal's class,
``StructuredDataError.code`` and the call it ended, no message text,
H17), the public schema as ``run_python.py`` projects it, the number of
SDMWarnings each call gave, and the request's payloads as JSON, each
``metadata.request_id`` left out (for a vector, as the twin's are kept
whole, their :func:`digest`). :func:`run` is imported by
``contract/tests/test_t218_column_names.py``, so ``python-core`` is run by
the same code.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contract/column-name-vectors.json'
FORMAT = 'sdm-column-name-vectors/1'
DATABASES = ('duckdb', 'sqlite', 'postgres')
CHARACTERS = {
    'dot': '.',
    'backtick': '`',
    'line-feed': '\n',
    'carriage-return': '\r',
}
#: The roles recorded. ``filter`` (a column of orders filtering the
#: target's aggregation) is kept in :func:`scenario` but recorded under
#: ``contract/deferred/t210-column-filter/`` until T2.10 lifts the core's
#: ``target_filter`` rule, which refuses any filter, a plain name's too.
ROLES = ('feature', 'target', 'entity', 'aggregation')
TWIN = 'a_b'


def name_of(character: str) -> str:
    r"""The column's name holding ``character``."""
    return f'a{character}b'


def spelled(name: str) -> str:
    r"""``name`` as a query names it: bare where the grammar's identifier
    spells it, else backtick-quoted, a backtick inside doubled."""
    if name == TWIN:
        return name
    return '`' + name.replace('`', '``') + '`'


def _quoted(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def scenario(role: str, name: str) -> dict[str, Any]:
    r"""The DuckDB statements after the ``shop`` base, the query and
    ``materialize``'s keywords of ``role`` with its column named ``name``."""
    column, query = _quoted(name), spelled(name)
    summed = {'indices': [0, 1], 'anchor_time': '2024-03-01'}
    if role == 'feature':
        return {
            'setup': [
                'CREATE OR REPLACE TABLE users AS SELECT *, '
                f'(user_id % 3)::INTEGER AS {column} FROM users'
            ],
            'pql': 'PREDICT SUM(orders.price, 0, 30, days) '
            'FOR EACH users.user_id',
            'materialize': summed,
        }
    if role == 'target':
        return {
            'setup': [
                'CREATE OR REPLACE TABLE users AS SELECT *, '
                f"['p', 'q', 'r'][1 + user_id // 3] AS {column} FROM users"
            ],
            'pql': f'PREDICT users.{query} FOR EACH users.user_id',
            'materialize': {'indices': [0, 1]},
        }
    if role == 'entity':
        return {
            'setup': [
                'DROP TABLE orders',
                'DROP TABLE users',
                f'CREATE TABLE users ({column} BIGINT PRIMARY KEY, '
                'age INTEGER, country VARCHAR)',
                'CREATE TABLE orders (order_id BIGINT PRIMARY KEY, '
                f'user_id BIGINT REFERENCES users ({column}), price DOUBLE, '
                'ts TIMESTAMP)',
                "INSERT INTO users SELECT i, 20 + i, ['US', 'DE', 'IN']"
                '[1 + i % 3] FROM range(8) t(i)',
                'INSERT INTO orders SELECT i, i % 8, 10 + i * 1.5, '
                "TIMESTAMP '2023-01-01' + to_days((i * 7)::INTEGER) "
                'FROM range(80) t(i)',
            ],
            'pql': f'PREDICT SUM(orders.price, 0, 30, days) '
            f'FOR EACH users.{query}',
            'materialize': summed,
        }
    if role == 'filter':
        return {
            'setup': [
                'CREATE OR REPLACE TABLE orders AS SELECT *, '
                f'(order_id % 3)::INTEGER AS {column} FROM orders'
            ],
            'pql': f'PREDICT COUNT(orders.* WHERE orders.{query} = 1, '
            '0, 30, days) FOR EACH users.user_id',
            'materialize': summed,
        }
    if role == 'aggregation':
        return {
            'setup': [
                'CREATE OR REPLACE TABLE orders AS SELECT *, '
                f'(price * 2)::DOUBLE AS {column} FROM orders'
            ],
            'pql': f'PREDICT SUM(orders.{query}, 0, 30, days) '
            'FOR EACH users.user_id',
            'materialize': summed,
        }
    raise ValueError(f'no role {role!r}')


def vectors() -> list[dict[str, Any]]:
    r"""Every vector and every twin, in a fixed order: by database, then
    role, the twin first, then each character."""
    out = []
    for database in DATABASES:
        for role in ROLES:
            out.append(
                {
                    'id': f'twin.{role}@{database}',
                    'database': database,
                    'role': role,
                    'character': None,
                    'name': TWIN,
                    **scenario(role, TWIN),
                }
            )
            for label, character in CHARACTERS.items():
                name = name_of(character)
                out.append(
                    {
                        'id': f'{label}.{role}@{database}',
                        'database': database,
                        'role': role,
                        'character': label,
                        'name': name,
                        'twin': f'twin.{role}@{database}',
                        **scenario(role, name),
                    }
                )
    return out


def _runner() -> Any:
    if 'run_python' not in sys.modules:
        path = ROOT / 'contract/runners/run_python.py'
        spec = importlib.util.spec_from_file_location('run_python', path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules['run_python'] = module
        spec.loader.exec_module(module)
    return sys.modules['run_python']


def _plain(value: Any) -> Any:
    r"""A payload as JSON reads it back: its numbers and texts, nothing of
    the Python objects that carried them."""
    return json.loads(json.dumps(value, default=str))


def _payloads(payloads: Any) -> list[Any]:
    bodies = _plain(list(payloads))
    for body in bodies:
        (body.get('metadata') or {}).pop('request_id', None)
    return bodies


def digest(payloads: list[Any] | None) -> str | None:
    r"""The SHA-256 of the payloads as canonical JSON (sorted keys, no
    spaces); null with none."""
    if payloads is None:
        return None
    text = json.dumps(payloads, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def run(vector: dict[str, Any]) -> dict[str, Any]:
    r"""One vector through the public calls on its database, on whatever
    route the environment gives: its outcome, public schema, SDMWarnings
    per call and payloads."""
    from sdm_client import PredictiveQuery, Schema, SDMWarning

    runner = _runner()
    case = {
        'id': vector['id'],
        'input': {'base': 'shop', 'setup': vector['setup']},
    }
    connection, cleanup, _ = runner._open(
        case, runner.load_bases(), vector['database']
    )
    answer: dict[str, Any] = {'schema': None, 'payloads': None}
    told: dict[str, int] = {}
    stage = 'from_source'
    try:
        source = runner._source(vector['database'], connection)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always')
            try:
                try:
                    schema = Schema.from_source(source)
                finally:
                    told[stage] = sum(
                        issubclass(w.category, SDMWarning) for w in seen
                    )
                answer['schema'] = _plain(runner._schema_projection(schema))
                stage = 'materialize'
                del seen[:]
                query = PredictiveQuery(vector['pql'], schema)
                try:
                    request = query.materialize(
                        **runner._materialize_arguments(vector['materialize'])
                    )
                finally:
                    told[stage] = sum(
                        issubclass(w.category, SDMWarning) for w in seen
                    )
                answer['outcome'] = {'result': 'success', 'stage': stage}
                answer['payloads'] = _payloads(request.payloads)
            except Exception as error:  # noqa: BLE001 - the call's outcome
                answer['outcome'] = {
                    'result': 'refusal',
                    'stage': stage,
                    'error': type(error).__name__,
                    'code': getattr(error, 'code', None),
                }
    finally:
        cleanup()
    answer['sdm_warnings'] = told
    return answer


def main() -> int:
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
    if not os.environ.get('SDM_POSTGRES_TEST_DSN'):
        print('SDM_POSTGRES_TEST_DSN names no server', file=sys.stderr)
        return 1
    import duckdb
    import pandas as pd
    import psycopg
    import pyarrow as pa

    with psycopg.connect(os.environ['SDM_POSTGRES_TEST_DSN']) as connection:
        server = connection.info.server_version // 10000
    recorded = []
    for vector in vectors():
        answer = run(vector)
        if vector['character'] is not None:
            # What main does with a name it cannot say; held by its digest,
            # since the core's answer is held to the twin's.
            answer['payloads'] = digest(answer['payloads'])
        recorded.append({**vector, 'reference': answer})
        print(
            vector['id'], recorded[-1]['reference']['outcome'], file=sys.stderr
        )
    document = {
        'format': FORMAT,
        'recorded_by': 'contract/tools/column_names_record.py',
        'reference': commit,
        'cell': 'ref-pandas3',
        'environment': {
            'python': platform.python_version(),
            'pandas': pd.__version__,
            'pyarrow': pa.__version__,
            'duckdb': duckdb.__version__,
            'psycopg': psycopg.__version__,
            'postgres': server,
        },
        'twin': TWIN,
        'vectors': recorded,
    }
    OUT.write_text(json.dumps(document, indent=1) + '\n', encoding='utf-8')
    print(f'{len(recorded)} vectors written', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
