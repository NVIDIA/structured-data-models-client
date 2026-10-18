#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Records the reference's answers to T2.2's round 1 SQLite regressions
into ``contract/sqlite-regression-vectors.json``.

Run with the reference checkout's interpreter (the environment of
``contract/reference-env.txt``, the ``ref-pandas3`` cell), from this
repository's root::

    ~/code/sdmc-reference/.venv/bin/python -I contract/tools/sqlite_regressions_record.py

Written by T2.2's test author from the round 1 and round 2 reviews'
repros. Each scenario is a SQLite database made on an in-memory ADBC
connection by its ``setup``, read by ``Schema.from_source`` (over the
tables its ``tables`` names, or every table) and asked its ``query``
(default ``PREDICT users.age FOR EACH users.id``) by ``materialize`` with
its ``materialize`` keywords:

- ``seeds-table``, ``seeds-table-numbered``, ``seeds-table-upper``: the
  caller's own temporary table holding ``'keep me'`` beside the source,
  named as the provider's scratch relations are or were (``__sdm_seeds__``,
  ``__sdm_seeds_1__``, and ``__SDM_SEEDS_1__``, which SQLite takes for the
  same name); ``temp_after`` is that table after the call, columns and
  rows;
- ``float-keys``: an ``INT PRIMARY KEY`` holding 101.5 to 107.5 (SQLite
  stores each as a REAL), so the seeded context sample is the reference's
  for a key whose storage class differs from its declared type;
- ``null-key``: an ``INT PRIMARY KEY`` holding a NULL beside 101 to 106
  (SQLite lets a primary key that is not a rowid alias hold one);
- ``integer-fractions``: a column declared ``INTEGER`` holding fractions
  (``i.5``), described as its declaration types it;
- ``composite-key``: a composite primary key of an ``INTEGER`` column
  holding repeated ids and a ``TEXT`` one, described as the reference
  reads it (the ``INTEGER`` column as SQLite's rowid alias), and refused
  at ``materialize`` as there;
- ``partial-unique``: a partial unique index on the key column, read as
  unique, with the reference's statements.

The reference's answer is kept as the calls gave it: the outcome (success,
or the refusal's class, ``StructuredDataError.code`` and the call it ended,
no message text, H17), the public schema as ``run_python.py`` projects it,
the request's payloads as JSON, and the statements the driver ran per
public call (``run_python.py``'s ``StatementLog``, the setup outside it).
:func:`run` is imported by ``contract/tests/test_t22_r1_regressions.py``,
so ``python-core`` is run by the same code.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contract/sqlite-regression-vectors.json'
FORMAT = 'sdm-sqlite-regression-vectors/1'
QUERY = 'PREDICT users.age FOR EACH users.id'


def _seeds(name: str) -> dict[str, Any]:
    return {
        'setup': [
            'CREATE TABLE users (id INTEGER PRIMARY KEY, age REAL)',
            'INSERT INTO users VALUES (101, 10), (102, 20), (103, 30), '
            '(104, 40), (105, 50), (106, 60)',
            f'CREATE TEMP TABLE "{name}" (important TEXT)',
            f'INSERT INTO temp."{name}" VALUES (\'keep me\')',
        ],
        'tables': {'users': 'users'},
        'materialize': {
            'indices': [101],
            'num_context_samples': 3,
            'random_seed': 42,
        },
        'temp_table': name,
    }


def _shop(
    users: list[str], score: str = "'c' || (i % 3)", ids: str = 'i'
) -> list[str]:
    r"""Round 2's shop: ``users`` made by its statements and filled with
    40 rows of ``ids`` and ``score`` over i from 0 to 39, and 200 orders
    over users 0 to 39 in two months."""
    return [
        *users,
        'CREATE TABLE orders (order_id INTEGER, user_id INTEGER, '
        'ts TIMESTAMP, price REAL)',
        'WITH RECURSIVE n(i) AS (SELECT 0 UNION ALL SELECT i + 1 FROM n '
        f'WHERE i < 39) INSERT INTO users SELECT {ids}, {score} FROM n',
        'WITH RECURSIVE n(i) AS (SELECT 0 UNION ALL SELECT i + 1 FROM n '
        'WHERE i < 199) INSERT INTO orders SELECT i, i % 40, '
        "printf('2024-0%d-%02d 00:00:00', 1 + i % 2, 10 + i % 15), i + 0.5 "
        'FROM n',
    ]


#: Round 2's question, entities and anchor time.
SUMMED = {
    'query': 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id',
    'materialize': {'indices': [0, 1], 'anchor_time': '2024-03-01'},
}

SCENARIOS: dict[str, dict[str, Any]] = {
    'seeds-table': _seeds('__sdm_seeds__'),
    'seeds-table-numbered': _seeds('__sdm_seeds_1__'),
    'seeds-table-upper': _seeds('__SDM_SEEDS_1__'),
    'float-keys': {
        'setup': [
            'CREATE TABLE users (id INT PRIMARY KEY, age REAL)',
            'INSERT INTO users VALUES (101.5, 10), (102.5, 20), (103.5, 30), '
            '(104.5, 40), (105.5, 50), (106.5, 60), (107.5, 70)',
        ],
        'tables': {'users': 'users'},
        'materialize': {
            'indices': [101.5],
            'num_context_samples': 3,
            'random_seed': 42,
        },
    },
    'null-key': {
        'setup': [
            'CREATE TABLE users (id INT PRIMARY KEY, age REAL)',
            'INSERT INTO users VALUES (NULL, 99), (101, 10), (102, 20), '
            '(103, 30), (104, 40), (105, 50), (106, 60)',
        ],
        'tables': {'users': 'users'},
        'materialize': {
            'indices': [101],
            'num_context_samples': 3,
            'random_seed': 42,
        },
    },
    'integer-fractions': {
        'setup': _shop(
            ['CREATE TABLE users (user_id INTEGER, score INTEGER)'],
            score='i + 0.5',
        ),
        **SUMMED,
    },
    'composite-key': {
        'setup': _shop(
            [
                'CREATE TABLE users (user_id INTEGER, country TEXT, '
                'PRIMARY KEY (user_id, country))'
            ],
            score="'c' || i",
            ids='i % 30',
        ),
        **SUMMED,
    },
    'partial-unique': {
        'setup': _shop(
            [
                'CREATE TABLE users (user_id INTEGER, country TEXT)',
                'CREATE UNIQUE INDEX uu ON users(user_id) WHERE user_id >= 0',
            ]
        ),
        **SUMMED,
    },
}


def _plain(value: Any) -> Any:
    r"""A payload as JSON reads it back: its numbers and texts, nothing of
    the Python objects that carried them."""
    return json.loads(json.dumps(value, default=str))


def _runner() -> Any:
    import importlib.util

    if 'run_python' not in sys.modules:
        path = ROOT / 'contract/runners/run_python.py'
        spec = importlib.util.spec_from_file_location('run_python', path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules['run_python'] = module
        spec.loader.exec_module(module)
    return sys.modules['run_python']


def run(scenario: dict[str, Any]) -> dict[str, Any]:
    r"""One scenario through the public calls, on whatever route the
    environment gives: its outcome, public schema, payloads, statements per
    call and the caller's temporary table after the call."""
    import adbc_driver_sqlite.dbapi
    import pandas as pd
    from sdm_client import PredictiveQuery, Schema, StructuredDataError
    from sdm_connectors import SQLiteSource

    runner = _runner()
    connection = adbc_driver_sqlite.dbapi.connect()
    cursor = connection.cursor()
    try:
        for statement in scenario['setup']:
            cursor.execute(statement)
        answer: dict[str, Any] = {'schema': None, 'payloads': None}
        statements: dict[str, int] = {}
        keywords = dict(scenario['materialize'])
        if 'anchor_time' in keywords:
            keywords['anchor_time'] = pd.Timestamp(keywords['anchor_time'])
        stage = 'from_source'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                with runner.StatementLog('sqlite') as log:
                    if 'tables' in scenario:
                        schema = Schema.from_source(
                            SQLiteSource(connection), tables=scenario['tables']
                        )
                    else:
                        schema = Schema.from_source(SQLiteSource(connection))
                statements[stage] = sum(1 for text in log.texts if text)
                answer['schema'] = _plain(runner._schema_projection(schema))
                stage = 'materialize'
                with runner.StatementLog('sqlite') as log:
                    try:
                        request = PredictiveQuery(
                            scenario.get('query', QUERY), schema
                        ).materialize(**keywords)
                    finally:
                        statements[stage] = sum(1 for t in log.texts if t)
                answer['outcome'] = {'result': 'success', 'stage': stage}
                answer['payloads'] = _plain(list(request.payloads))
            except Exception as error:  # noqa: BLE001 - the call's outcome
                answer['outcome'] = {
                    'result': 'refusal',
                    'stage': stage,
                    'error': type(error).__name__,
                    'code': (
                        error.code
                        if isinstance(error, StructuredDataError)
                        else None
                    ),
                }
        answer['statements'] = statements
        if 'temp_table' in scenario:
            cursor.execute(f'SELECT * FROM temp."{scenario["temp_table"]}"')
            answer['temp_after'] = {
                'columns': [column[0] for column in cursor.description],
                'rows': _plain([list(row) for row in cursor.fetchall()]),
            }
        return answer
    finally:
        cursor.close()
        connection.close()


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
    scenarios = {}
    for name, scenario in SCENARIOS.items():
        scenarios[name] = {**scenario, 'reference': run(scenario)}
        print(name, scenarios[name]['reference']['outcome'], file=sys.stderr)
    document = {
        'format': FORMAT,
        'recorded_by': 'contract/tools/sqlite_regressions_record.py',
        'reference': commit,
        'cell': 'ref-pandas3',
        'environment': {'python': platform.python_version()},
        'query': QUERY,
        'scenarios': scenarios,
    }
    OUT.write_text(json.dumps(document, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
