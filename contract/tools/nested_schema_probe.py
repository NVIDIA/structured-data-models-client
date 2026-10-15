#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T1.32c's old-against-core probe (decision E49), a local pre-push probe,
not a CI gate: ``Schema.from_source`` over generated DuckDB sources holding
binary and list columns, on the reference and on the core route, compared
by their public description.

    python -I contract/tools/nested_schema_probe.py run \
        --reference-python REF --candidate-python CAND [--seed 31] [--n 60]
    python -I contract/tools/nested_schema_probe.py run \
        --reference-python REF --candidate-python CAND --targeted

``REF`` is the reference checkout's environment, ``CAND`` one holding the
candidate's client; ``run`` sets ``SDM_CORE_ROUTES`` to every DuckDB route
of ``Schema.from_source`` for the candidate. Each source is ``users`` (1 to
300 rows) and ``orders`` (three times as many), under the 1,000 rows of a
preview, with one to three columns added to one of them: binary, null
binary, lists of binary (empty or null), fixed-size lists of binary and of
integers, and lists of integers, floats and text, under names that say a
key, a time or nothing. ``--targeted`` runs five sources instead: a binary
primary key, a binary foreign key by name, a table of a key and binary
alone, binary named as a time column, and an all-null binary named as a
key.

What is compared, per source: the outcome (success, or the exception's
class and ``StructuredDataError.code``); each table's columns with their
dtype and stype, its key and its time column; and the body of each warning
that leaves a column out, with the core's ``[COLUMNS_LEFT_OUT]`` prefix set
aside. ``compare`` counts the sources that agree and those where the
reference fails unnamed and the core leaves a fixed-size list out
(register P1B-9), and prints every other difference; it exits 1 on any.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

KINDS = [
    "encode('x' || {k})::BLOB",
    'NULL::BLOB',
    "CASE WHEN {k} % 2 = 0 THEN encode('y') END",
    "[encode('a'), NULL]::BLOB[]",
    '[]::BLOB[]',
    'NULL::BLOB[]',
    "[encode('a'), encode('b')]::BLOB[2]",
    '[{k}, {k}+1]::INTEGER[]',
    "['s' || {k}]",
    '[{k} * 1.5]::DOUBLE[]',
    '[{k}, NULL]::BIGINT[2]',
    "from_hex('00ff')",
]
NAMES = ['f', 'payload', 'created_at', 'blob_id', 'g', 'updated_at', 'raw']
ORDERS = (
    'CREATE TABLE orders AS SELECT range AS order_id, range % {users} AS '
    "user_id, TIMESTAMP '2024-01-01' + INTERVAL (range) DAY AS ts "
    'FROM range({orders})'
)
TARGETED = {
    'blob-pk': [
        'CREATE TABLE users (user_id BLOB PRIMARY KEY, age INTEGER)',
        "INSERT INTO users SELECT encode('u' || range), range FROM range(20)",
        ORDERS.format(users=20, orders=60),
    ],
    'blob-fk': [
        'CREATE TABLE users AS SELECT range AS user_id, range AS age '
        'FROM range(20)',
        "CREATE TABLE orders AS SELECT range AS order_id, encode('u' || "
        "(range % 20)) AS user_id, TIMESTAMP '2024-01-01' + INTERVAL "
        '(range) DAY AS ts FROM range(60)',
    ],
    'blob-only': [
        "CREATE TABLE users AS SELECT range AS user_id, encode('a') AS b "
        'FROM range(20)',
        ORDERS.format(users=20, orders=60),
    ],
    'blob-time-name': [
        'CREATE TABLE users AS SELECT range AS user_id, range AS age '
        'FROM range(20)',
        'CREATE TABLE orders AS SELECT range AS order_id, range % 20 AS '
        "user_id, encode('2024-01-0' || (1 + range % 9)) AS ts, TIMESTAMP "
        "'2024-01-01' + INTERVAL (range) DAY AS created_at FROM range(60)",
    ],
    'blob-all-null-key-name': [
        'CREATE TABLE users AS SELECT range AS user_id, NULL::BLOB AS '
        'user_blob_id FROM range(20)',
        ORDERS.format(users=20, orders=60),
    ],
}


def generated(seed: int, n: int) -> dict[str, list[str]]:
    r"""``n`` sources from ``seed``, each its statements."""
    rnd = random.Random(seed)
    out = {}
    for i in range(n):
        columns: list[tuple[str, str]] = []
        for _ in range(rnd.randint(1, 3)):
            taken = [name for name, _ in columns]
            name = rnd.choice([c for c in NAMES if c not in taken])
            columns.append((name, rnd.choice(KINDS)))
        table = rnd.choice(['users', 'orders'])
        key = 'user_id' if table == 'users' else 'order_id'
        rows = rnd.choice([1, 3, 20, 120, 300])
        added = ', '.join(
            f'{kind.format(k=key)} AS "{name}"' for name, kind in columns
        )
        out[f'g{i:02d}'] = [
            'CREATE TABLE users AS SELECT range AS user_id, 20 + range % 50 '
            "AS age, ['us', 'de', 'fr'][1 + range % 3] AS country "
            f'FROM range({rows})',
            'CREATE TABLE orders AS SELECT range AS order_id, range % '
            f'{rows} AS user_id, (range % 97) * 1.25 AS price, TIMESTAMP '
            "'2024-01-01' + INTERVAL (range % 60) DAY AS ts "
            f'FROM range({rows * 3})',
            f'CREATE OR REPLACE TABLE {table} AS SELECT *, {added} '
            f'FROM {table}',
        ]
    return out


def probe(sources: dict[str, list[str]]) -> dict[str, Any]:
    r"""Each source's public description, in this interpreter."""
    import duckdb
    from sdm_client import Schema
    from sdm_connectors import DuckDBSource

    out: dict[str, Any] = {}
    for name, statements in sources.items():
        connection = duckdb.connect()
        for statement in statements:
            connection.execute(statement)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always')
            try:
                schema = Schema.from_source(DuckDBSource(connection))
                got: dict[str, Any] = {
                    'outcome': 'success',
                    'tables': {
                        table_name: {
                            'columns': [
                                [c.name, str(c.dtype), str(c.stype)]
                                for c in table.columns.values()
                            ],
                            'primary_key': table.primary_key,
                            'time_column': table.time_column,
                        }
                        for table_name, table in sorted(schema.tables.items())
                    },
                }
            except Exception as error:  # noqa: BLE001 - any is an answer
                got = {
                    'outcome': 'refusal',
                    'error': type(error).__name__,
                    'code': getattr(error, 'code', None),
                }
        got['left_out'] = sorted(
            re.sub(r'^\[COLUMNS_LEFT_OUT\] ', '', str(w.message))
            for w in seen
            if 'leaves out' in str(w.message)
        )
        out[name] = {'statements': statements, 'got': got}
    return out


def compare(old: dict[str, Any], core: dict[str, Any]) -> int:
    r"""Prints the counts and every unclassified difference; 1 on any."""
    counts: dict[str, int] = {}
    unclassified = 0
    for name, entry in old.items():
        mine, theirs = core[name]['got'], entry['got']
        fixed = re.search(r'\]::\w+\[2\]', ' '.join(entry['statements']))
        if mine == theirs:
            kind = 'agree'
        elif (
            theirs['outcome'] == 'refusal'
            and theirs['code'] is None
            and mine['outcome'] == 'success'
            and fixed
        ):
            kind = 'P1B-9: the reference fails unnamed, the core leaves out'
        else:
            kind = 'UNCLASSIFIED'
            unclassified += 1
            print(f'{name}: reference {theirs}\n  core {mine}')
        counts[kind] = counts.get(kind, 0) + 1
    print(json.dumps(counts, indent=1))
    return 1 if unclassified or not old else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    one = commands.add_parser('probe')
    one.add_argument('out', type=Path)
    run = commands.add_parser('run')
    run.add_argument('--reference-python', required=True)
    run.add_argument('--candidate-python', required=True)
    for command in (one, run):
        command.add_argument('--seed', type=int, default=31)
        command.add_argument('--n', type=int, default=60)
        command.add_argument('--targeted', action='store_true')
    args = parser.parse_args()
    sources = TARGETED if args.targeted else generated(args.seed, args.n)
    if args.command == 'probe':
        args.out.write_text(json.dumps(probe(sources), indent=1, default=str))
        return 0
    routes = subprocess.run(
        [
            args.candidate_python,
            '-c',
            'from sdm_client import _routing; '
            "print(','.join(_routing.FROM_SOURCE_DUCKDB))",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    with tempfile.TemporaryDirectory() as tmp:
        outs = {}
        for side, python, extra in (
            ('old', args.reference_python, {}),
            ('core', args.candidate_python, {'SDM_CORE_ROUTES': routes}),
        ):
            outs[side] = Path(tmp) / f'{side}.json'
            flags = ['--targeted'] if args.targeted else []
            subprocess.run(
                [python, '-I', __file__, 'probe', str(outs[side])]
                + ['--seed', str(args.seed), '--n', str(args.n), *flags],
                env={**os.environ, **extra},
                check=True,
            )
        old, core = (json.loads(outs[s].read_text()) for s in ('old', 'core'))
    return compare(old, core)


if __name__ == '__main__':
    sys.exit(main())
