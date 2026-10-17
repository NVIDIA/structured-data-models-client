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

Written by T2.2's test author from the round 1 review's repros. Each
scenario is a SQLite database made on an in-memory ADBC connection by its
``setup``, read by ``Schema.from_source(SQLiteSource(connection),
tables={'users': 'users'})`` and asked ``PREDICT users.age FOR EACH
users.id`` by ``materialize(indices=..., num_context_samples=3,
random_seed=42)``:

- ``seeds-table``: the caller's own temporary table ``__sdm_seeds__``
  holding ``'keep me'`` beside the source; ``temp_after`` is that table
  after the call, columns and rows;
- ``float-keys``: an ``INT PRIMARY KEY`` holding 101.5 to 107.5 (SQLite
  stores each as a REAL), so the seeded context sample is the reference's
  for a key whose storage class differs from its declared type;
- ``null-key``: an ``INT PRIMARY KEY`` holding a NULL beside 101 to 106
  (SQLite lets a primary key that is not a rowid alias hold one).

The reference's answer is kept as the call gave it: the outcome (success,
or the refusal's class and ``StructuredDataError.code``, no message text,
H17) and the request's payloads as JSON. :func:`run` is imported by
``contract/tests/test_t22_r1_regressions.py``, so ``python-core`` is run
by the same code.
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

SCENARIOS: dict[str, dict[str, Any]] = {
    'seeds-table': {
        'setup': [
            'CREATE TABLE users (id INTEGER PRIMARY KEY, age REAL)',
            'INSERT INTO users VALUES (101, 10), (102, 20), (103, 30), '
            '(104, 40), (105, 50), (106, 60)',
            'CREATE TEMP TABLE __sdm_seeds__ (important TEXT)',
            "INSERT INTO temp.__sdm_seeds__ VALUES ('keep me')",
        ],
        'indices': [101],
        'temp_table': '__sdm_seeds__',
    },
    'float-keys': {
        'setup': [
            'CREATE TABLE users (id INT PRIMARY KEY, age REAL)',
            'INSERT INTO users VALUES (101.5, 10), (102.5, 20), (103.5, 30), '
            '(104.5, 40), (105.5, 50), (106.5, 60), (107.5, 70)',
        ],
        'indices': [101.5],
    },
    'null-key': {
        'setup': [
            'CREATE TABLE users (id INT PRIMARY KEY, age REAL)',
            'INSERT INTO users VALUES (NULL, 99), (101, 10), (102, 20), '
            '(103, 30), (104, 40), (105, 50), (106, 60)',
        ],
        'indices': [101],
    },
}


def _plain(value: Any) -> Any:
    r"""A payload as JSON reads it back: its numbers and texts, nothing of
    the Python objects that carried them."""
    return json.loads(json.dumps(value, default=str))


def run(scenario: dict[str, Any]) -> dict[str, Any]:
    r"""One scenario through the public calls, on whatever route the
    environment gives: its outcome, payloads and the caller's temporary
    table after the call."""
    import adbc_driver_sqlite.dbapi
    from sdm_client import PredictiveQuery, Schema, StructuredDataError
    from sdm_connectors import SQLiteSource

    connection = adbc_driver_sqlite.dbapi.connect()
    cursor = connection.cursor()
    try:
        for statement in scenario['setup']:
            cursor.execute(statement)
        answer: dict[str, Any] = {'payloads': None}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                schema = Schema.from_source(
                    SQLiteSource(connection), tables={'users': 'users'}
                )
                request = PredictiveQuery(QUERY, schema).materialize(
                    indices=list(scenario['indices']),
                    num_context_samples=3,
                    random_seed=42,
                )
                answer['outcome'] = {'result': 'success', 'code': None}
                answer['payloads'] = _plain(list(request.payloads))
            except Exception as error:  # noqa: BLE001 - the call's outcome
                answer['outcome'] = {
                    'result': 'refusal',
                    'error': type(error).__name__,
                    'code': (
                        error.code
                        if isinstance(error, StructuredDataError)
                        else None
                    ),
                }
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
