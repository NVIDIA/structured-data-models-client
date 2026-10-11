#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Writes the ``Schema`` pickles the reference client makes (T1.24).

    python -I contract/fixtures/pickles/write.py --reference CHECKOUT [--write]

Run by the reference environment's interpreter (``contract/reference-env.txt``:
the reference commit's client, Python 3.12, pandas 3), it builds each schema
of :data:`SCHEMAS` with the reference checkout's client, pickles it with
protocol 4, and records beside the pickles, in ``expected.json``, the
reference's own projection of each: each table's columns with their data
type, meaning, what each meant before the schema joined on it and whether
it was declared, the key, the time column and the duplicate-key count, and
the relationships, as the client reads them back from the pickle. The
client's ``contract/tests/test_schema_pickles.py`` loads each pickle and
holds what it reads to that projection.

A schema of tables in their source holds its source, which no client pickles
(``Source.__reduce__`` refuses it, an open connection), so every schema
here is a schema of frames: one table, two tables linked by name, and two
tables edited after they were built (a column's meaning declared, a key
moved, a time column cleared, the relationships replaced).

The script refuses to run but from the reference checkout given, at the
reference commit ``contract/reference-env.txt`` names. Without ``--write``
it builds the pickles and compares them with the committed ones, and the
exit status says whether they are the same; CI never runs it.
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REFERENCE_ENV = HERE.parents[1] / 'reference-env.txt'


def _reference_commit() -> str:
    for line in REFERENCE_ENV.read_text(encoding='utf-8').splitlines():
        if line.startswith('reference: '):
            return line.split(': ', 1)[1].strip()
    raise SystemExit('reference-env.txt names no reference commit')


def _schemas() -> dict[str, Any]:
    import pandas as pd
    from sdm_client import Relationship, Schema

    users = pd.DataFrame(
        {
            'user_id': [1, 2, 3, 4, 5, 6],
            'age': [31, 45, 27, 38, 52, 29],
            'score': [0.5, 1.25, 2.0, 3.5, 0.75, 1.5],
        }
    )
    orders = pd.DataFrame(
        {
            'order_id': list(range(12)),
            'user_id': [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
            'price': [float(i) * 1.5 for i in range(12)],
            'ts': pd.date_range('2024-01-01', periods=12, freq='D'),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        one = Schema.from_data({'orders': orders})
        two = Schema.from_data(
            {'users': users, 'orders': orders},
            primary_keys={'users': 'user_id', 'orders': 'order_id'},
        )
        edited = Schema.from_data(
            {'users': users, 'orders': orders},
            primary_keys={'users': 'user_id', 'orders': 'order_id'},
            time_columns={'orders': 'ts'},
            relationships=[
                Relationship('orders', 'user_id', 'users', 'user_id')
            ],
        )
        edited['users']['age'].stype = 'categorical'
        edited['orders'].time_column = None
        edited['users'].primary_key = 'age'
        edited.relationships = []
    return {'one-table': one, 'two-tables': two, 'edited': edited}


def projection(schema: Any) -> dict[str, Any]:
    r"""What a schema says, as the reference and the candidate both read it."""
    tables = []
    for name, table in schema.tables.items():
        key = table.primary_key
        tables.append(
            {
                'name': name,
                'kind': type(table).__name__,
                'columns': [
                    [
                        column,
                        c.dtype,
                        c.stype,
                        c._stype_before_key,
                        c.stype_declared,
                    ]
                    for column, c in table.columns.items()
                ],
                'primary_key': key
                if key is None or isinstance(key, str)
                else list(key),
                'time_column': table.time_column,
                'duplicate_keys': table.duplicate_keys,
                'rows': None
                if getattr(table, 'data', None) is None
                else len(table.data),
            }
        )
    return {
        'tables': tables,
        'links': [
            [
                link.source_table,
                link.source_column,
                link.target_table,
                link.target_column,
            ]
            for link in schema.relationships
        ],
        'source': schema.source is not None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--write', action='store_true')
    given = parser.parse_args(argv)
    reference = given.reference.resolve()
    head = subprocess.run(
        ['git', '-C', str(reference), 'rev-parse', 'HEAD'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != _reference_commit():
        raise SystemExit(f'{reference} is at {head}, not the reference commit')
    sys.path.insert(
        0, str(reference / 'packages/structured-data-models-client/src')
    )
    import pandas as pd
    import sdm_client

    client = Path(sdm_client.__file__).resolve()
    if not client.is_relative_to(reference):
        raise SystemExit(f'sdm_client is {client}, not the reference checkout')
    expected: dict[str, Any] = {
        'reference': head,
        'python': '.'.join(map(str, sys.version_info[:3])),
        'pandas': pd.__version__,
        'protocol': 4,
        'schemas': {},
    }
    written: dict[str, bytes] = {}
    for name, schema in _schemas().items():
        buffer = io.BytesIO()
        pickle.dump(schema, buffer, protocol=4)
        written[f'{name}.pkl'] = buffer.getvalue()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            loaded = pickle.loads(buffer.getvalue())
        expected['schemas'][name] = projection(loaded)
    written['expected.json'] = (
        json.dumps(expected, indent=2, sort_keys=True) + '\n'
    ).encode()
    if given.write:
        for file, data in written.items():
            (HERE / file).write_bytes(data)
        return 0
    same = all(
        (HERE / file).exists() and (HERE / file).read_bytes() == data
        for file, data in written.items()
    )
    print("the committed pickles are the reference's" if same else 'differ')
    return 0 if same else 1


if __name__ == '__main__':
    sys.exit(main())
