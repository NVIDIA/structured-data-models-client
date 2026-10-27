#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T1.32e part 1's old-against-core probe (decision E49), a local pre-push
probe, not a CI gate: ``Schema.from_source`` over generated DuckDB sources
whose time column is chosen over the whole table, on the reference and on
the core route, compared by their public description and warnings.

    python -I contract/tools/whole_table_time_probe.py run \
        --reference-python REF --candidate-python CAND [--seed 32] [--n 50]
    python -I contract/tools/whole_table_time_probe.py run \
        --reference-python REF --candidate-python CAND --targeted

``REF`` is the reference checkout's environment, ``CAND`` one holding the
candidate's client; ``run`` sets ``SDM_CORE_ROUTES`` to every DuckDB route
of ``Schema.from_source`` for the candidate. Each source is ``users`` (40
rows, keyed ``user_id``) and ``events`` (999, 1,000, 1,001, 2,500 or 5,000
rows, linked to ``users`` by ``user_id``, so the schema's warning of the
time columns guessed is given), keyed by ``event_id`` (an integer or text)
or by nothing, with two to four time columns drawn from timestamps, dates,
times of day, zoned timestamps and ISO text, under names that say a
creation, an edit, a load, a birth or nothing, each filled by a pattern:
every row, nulls throughout, values only past the first 1,000 rows, one
value in the first 1,000 and others past them, mostly empty past them, a
placeholder (1900-01-01) past them, or one value throughout; a fifth of
the sources name a time column. ``--targeted`` runs the sources whose
answers the reference's environment decides (a zoned timestamp whose
preview and whole table disagree).

What is compared, per source: the outcome (success, or the exception's
class and ``StructuredDataError.code``); each table's columns with their
dtype and stype, its key and its time column; and the warnings' bodies,
the core's ``[CODE]`` prefix set aside. ``compare`` counts the sources that
agree and those register P1B-39 explains (the reference, without pytz,
cannot read a zoned timestamp's whole table and falls back to its
preview), and prints every other difference; it exits 1 on any, or on no
evidence.
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

#: A time column's kinds: its values from the row number ``i``.
KINDS = {
    'timestamp': "TIMESTAMP '2023-01-01' + to_minutes(({v})::INTEGER)",
    'date': "DATE '2022-06-01' + (({v}) % 400)::INTEGER",
    'time': "TIME '08:00:00' + to_seconds((({v}) % 3600)::INTEGER)",
    'zoned': "TIMESTAMPTZ '2023-01-01 00:00:00+00' + to_minutes(({v})::INTEGER)",
    'text': "strftime(DATE '2022-06-01' + (({v}) % 400)::INTEGER, '%Y-%m-%d')",
}
#: How a column is filled: an expression of the row number ``i`` for its
#: value, or NULL.
PATTERNS = {
    'every': 'i',
    'nulls': 'CASE WHEN i % 7 = 0 THEN NULL ELSE i END',
    'late': 'CASE WHEN i >= 1000 THEN i END',
    'late-change': 'CASE WHEN i < 1000 THEN 0 ELSE i END',
    'sparse-late': 'CASE WHEN i < 1000 THEN i END',
    'constant': '0',
    'earlier-late': 'CASE WHEN i < 1000 THEN i + 900000 ELSE i END',
}
NAMES = [
    'created_at',
    'updated_at',
    'loaded_at',
    'birth_date',
    'happened_at',
    'shipped',
    'billed_on',
    'ModifiedDate',
]
SIZES = [999, 1000, 1001, 2500, 5000]
USERS = 'CREATE TABLE users AS SELECT range AS user_id, range % 9 AS tier FROM range(40)'


def column(name: str, kind: str, pattern: str) -> str:
    if pattern == 'placeholder':
        return (
            f"CASE WHEN i = 1500 THEN TIMESTAMP '1900-01-01' ELSE "
            f'{KINDS["timestamp"].format(v="i")} END AS "{name}"'
        )
    value = PATTERNS[pattern]
    return (
        f'CASE WHEN ({value}) IS NULL THEN NULL ELSE '
        f'{KINDS[kind].format(v=value)} END AS "{name}"'
    )


def events(rows: int, key: str, columns: list[str]) -> str:
    keyed = {
        'int': 'i::BIGINT AS event_id, ',
        'text': "'e' || i::VARCHAR AS event_id, ",
        'none': '',
    }[key]
    return (
        f'CREATE TABLE events AS SELECT {keyed}i % 40 AS user_id, '
        f'(i % 97)::DOUBLE AS amount, {", ".join(columns)} '
        f'FROM range({rows}) t(i)'
    )


def generated(seed: int, n: int) -> dict[str, dict[str, Any]]:
    r"""``n`` sources from ``seed``, each its statements and the
    ``time_columns`` named, if any."""
    rnd = random.Random(seed)
    out = {}
    for i in range(n):
        names = rnd.sample(NAMES, rnd.randint(2, 4))
        columns = []
        for name in names:
            pattern = rnd.choice([*PATTERNS, 'placeholder'])
            kind = rnd.choice(list(KINDS))
            columns.append(column(name, kind, pattern))
        named = {'events': rnd.choice(names)} if rnd.random() < 0.2 else None
        out[f'g{i:02d}'] = {
            'statements': [
                USERS,
                events(
                    rnd.choice(SIZES),
                    rnd.choice(['int', 'int', 'text', 'none']),
                    columns,
                ),
            ],
            'time_columns': named,
        }
    return out


#: Sources whose answers the reference's environment decides.
TARGETED = {
    f'zoned-{pattern}-{rows}': {
        'statements': [
            USERS,
            events(rows, 'int', [column('happened_at', 'zoned', pattern)]),
        ],
        'time_columns': None,
    }
    for pattern in ('late-change', 'sparse-late', 'late', 'earlier-late')
    for rows in (1001, 5000)
}


def probe(sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    r"""Each source's public description and warnings, in this
    interpreter."""
    import duckdb
    from sdm_client import Schema
    from sdm_connectors import DuckDBSource

    out: dict[str, Any] = {}
    for name, source in sources.items():
        connection = duckdb.connect()
        for statement in source['statements']:
            connection.execute(statement)
        arguments = (
            {'time_columns': source['time_columns']}
            if source['time_columns']
            else {}
        )
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always')
            try:
                schema = Schema.from_source(
                    DuckDBSource(connection), **arguments
                )
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
        got['warnings'] = [
            [w.category.__name__, re.sub(r'^\[[A-Z_]+\] ', '', str(w.message))]
            for w in seen
        ]
        out[name] = {'source': source, 'got': got}
    return out


def zoned_whole_table(source: dict[str, Any]) -> bool:
    r"""Whether the source holds a zoned timestamp asked of the whole
    table, which the reference without pytz cannot read (P1B-39)."""
    return any(
        'TIMESTAMPTZ' in statement for statement in source['statements']
    ) and not any('range(999)' in s for s in source['statements'])


def compare(
    old: dict[str, Any], core: dict[str, Any], verbose: bool = False
) -> int:
    r"""Prints the counts and every unclassified difference; 1 on any."""
    counts: dict[str, int] = {}
    unclassified = 0
    for name, entry in old.items():
        mine, theirs = core[name]['got'], entry['got']
        if mine == theirs:
            kind = 'agree'
        elif zoned_whole_table(entry['source']):
            kind = 'P1B-39: the reference reads no zoned whole table'
            if verbose:
                print(f'{name}: reference {theirs}\n  core {mine}')
        else:
            kind = 'UNCLASSIFIED'
            unclassified += 1
            print(
                f'{name}: {entry["source"]}\n  reference {theirs}\n  core {mine}'
            )
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
        command.add_argument('--seed', type=int, default=32)
        command.add_argument('--n', type=int, default=50)
        command.add_argument('--targeted', action='store_true')
    run.add_argument('--verbose', action='store_true')
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
            "print(','.join(sorted({*_routing.FROM_SOURCE_DUCKDB, "
            '*_routing.FROM_SOURCE_DUCKDB_TABLES})))',
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    with tempfile.TemporaryDirectory() as tmp:
        outs = {}
        for side, python, extra in (
            ('old', args.reference_python, {'SDM_CORE_ROUTES': ''}),
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
    return compare(old, core, args.verbose)


if __name__ == '__main__':
    sys.exit(main())
