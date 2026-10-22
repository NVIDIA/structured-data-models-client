#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Records the reference's side of ``contract/type-set-vectors.json``
(T1.33's type set): what DuckDB hands over and what the reference makes of
it, for every vector and ``reached`` row with a ``witness``.

Run with the reference checkout's interpreter (the environment of
``contract/reference-env.txt``, the ``ref-pandas3`` cell), from this
repository's root::

    TZ=America/Los_Angeles ~/code/sdmc-reference/.venv/bin/python -I \
        contract/tools/type_set_record.py

The process's zone is named because ``duckdb_arrow_by_setting``'s
``default`` reads it: a TIMESTAMPTZ with no zone set comes over in the
process's zone, recorded in Los Angeles. Every witness sets its zone with
``SET GLOBAL TimeZone``, which every cursor takes (C16).

Written by T1.33's independent test author. A ``witness`` is the DuckDB
statements, after ``contract/bases.yaml``'s ``shop``, that add the column
``f`` of the type to ``users``, as ``contract/tools/nested_type_record.py``'s
vectors' ``setup`` is; this script runs each through that recorder's
``run`` and ``project`` (``nested.list-int``'s case, the public calls on
``python-ref``) and writes, beside the witness:

- ``duckdb_arrow``: ``f``'s Arrow type as the reference environment's
  DuckDB hands it over (``nested_type_record.duckdb_types``), an extension
  type by its extension name;
- ``reference_answer``: the reference's answer as ``project`` reads it
  (outcome and stage, and what ``Schema.from_source`` made of ``f``: its
  dtype and stype, or ``"left out"``);
- ``reference``: its class where ``describe_source`` answers, the
  vectors' terms: ``refusal <CODE>`` (``refusal UNNAMED`` for an
  exception with no code) where the reference failed at ``from_source``,
  else ``success``.

It also measures ``duckdb_arrow_by_setting``: for each DuckDB type of
:data:`TYPES` under each session setting of :data:`SETTINGS`, the Arrow
type DuckDB hands over, so the vectors can be held to cover every form a
setting gives. Nothing else in the file is written here: every
expectation of the core is the test author's.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contract/type-set-vectors.json'
sys.path.insert(0, str(ROOT / 'contract/tools'))

import nested_type_record as N  # noqa: E402

#: The DuckDB types measured, each an expression of one value.
TYPES: dict[str, str] = {
    'BOOLEAN': 'true',
    'TINYINT': '1::TINYINT',
    'SMALLINT': '1::SMALLINT',
    'INTEGER': '1',
    'BIGINT': '1::BIGINT',
    'HUGEINT': '1::HUGEINT',
    'UTINYINT': '1::UTINYINT',
    'USMALLINT': '1::USMALLINT',
    'UINTEGER': '1::UINTEGER',
    'UBIGINT': '1::UBIGINT',
    'UHUGEINT': '1::UHUGEINT',
    'VARINT': '1::VARINT',
    'FLOAT': '1::FLOAT',
    'DOUBLE': '1::DOUBLE',
    'DECIMAL(9,2)': '1::DECIMAL(9,2)',
    'DECIMAL(18,2)': '1::DECIMAL(18,2)',
    'DECIMAL(38,10)': '1::DECIMAL(38,10)',
    'VARCHAR': "'a'",
    'BLOB': "'\\xAA'::BLOB",
    'BIT': "'101'::BIT",
    'UUID': 'uuid()',
    'JSON': "'{}'::JSON",
    'DATE': "DATE '2024-01-01'",
    'TIME': "TIME '12:00:00'",
    'TIME_NS': "'12:00:00'::TIME_NS",
    'TIMETZ': "'12:00:00+01'::TIMETZ",
    'TIMESTAMP': "TIMESTAMP '2024-01-01'",
    'TIMESTAMP_S': "'2024-01-01'::TIMESTAMP_S",
    'TIMESTAMP_MS': "'2024-01-01'::TIMESTAMP_MS",
    'TIMESTAMP_NS': "'2024-01-01'::TIMESTAMP_NS",
    'TIMESTAMPTZ': "TIMESTAMPTZ '2024-01-01 00:00:00+00'",
    'INTERVAL': 'INTERVAL 1 DAY',
    'ENUM(3)': "'a'::e3",
    'ENUM(300)': "'v1'::e300",
    'ENUM(70000)': "'v1'::e70000",
    'INTEGER[]': '[1, 2]',
    'VARCHAR[]': "['a']",
    'INTEGER[2]': '[1, 2]::INTEGER[2]',
    'STRUCT(a VARCHAR)': "{'a': 'x'}",
    'MAP(VARCHAR, BIGINT)': "map(['a'], [1::BIGINT])",
    'UNION(n INTEGER, s VARCHAR)': (
        'union_value(n := 1)::UNION(n INTEGER, s VARCHAR)'
    ),
}

#: The enum types :data:`TYPES` casts to.
ENUMS = [
    "CREATE TYPE e3 AS ENUM ('a', 'b', 'c')",
    "CREATE TYPE e300 AS ENUM (SELECT 'v' || range FROM range(300))",
    "CREATE TYPE e70000 AS ENUM (SELECT 'v' || range FROM range(70000))",
]

#: The session settings that change what DuckDB hands over, by name.
SETTINGS: dict[str, list[str]] = {
    'default': [],
    'arrow_lossless_conversion': ['SET arrow_lossless_conversion = true'],
    'arrow_large_buffer_size': ['SET arrow_large_buffer_size = true'],
    'string_views': [
        "SET arrow_output_version = '1.4'",
        'SET produce_arrow_string_view = true',
    ],
    'list_views': [
        "SET arrow_output_version = '1.4'",
        'SET arrow_output_list_view = true',
    ],
    'arrow_output_version_1.5': ["SET arrow_output_version = '1.5'"],
    'TimeZone_UTC': ["SET GLOBAL TimeZone = 'UTC'"],
}


def by_setting() -> dict[str, dict[str, str]]:
    r"""Each DuckDB type's Arrow type under each setting."""
    import duckdb

    found: dict[str, dict[str, str]] = {}
    for name, expression in TYPES.items():
        for setting, statements in SETTINGS.items():
            connection = duckdb.connect()
            try:
                for statement in ENUMS + statements:
                    connection.execute(statement)
                table = connection.execute(
                    f'SELECT {expression} AS f'
                ).to_arrow_table()
                found.setdefault(name, {})[setting] = str(
                    table.schema.field('f').type
                )
            finally:
                connection.close()
    return found


def reference_class(answer: dict[str, Any]) -> str:
    r"""The reference's answer where ``describe_source`` answers."""
    outcome = answer['outcome']
    if outcome['result'] == 'refusal' and outcome['stage'] == 'from_source':
        return N.outcome_class(outcome)
    return 'success'


def main() -> int:
    import duckdb
    import numpy as np
    import pandas as pd
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
    document = json.loads(OUT.read_text(encoding='utf-8'))
    rows = [
        row
        for row in document['vectors'] + document['reached']
        if row.get('witness')
    ]
    runs = [
        {'id': row['id'], 'table': 'users', 'setup': row['witness']}
        for row in rows
    ]
    with tempfile.TemporaryDirectory(prefix='t133-record-') as tmp:
        observed = N.run('python-ref', runs, Path(tmp), reference=reference)
    for row, run in zip(rows, runs, strict=True):
        answer = N.project(observed[row['id']], run)
        row['duckdb_arrow'] = N.duckdb_types(run)['duckdb_arrow']
        row['reference_answer'] = {
            'outcome': answer['outcome'],
            'described': answer['described'],
        }
        row['reference'] = reference_class(answer)
        print(
            row['id'],
            row['duckdb_arrow'],
            row['reference'],
            answer['described'],
            file=sys.stderr,
        )
    document['duckdb_arrow_by_setting'] = by_setting()
    document['recorded'] = {
        'by': 'contract/tools/type_set_record.py',
        'reference': commit,
        'cell': 'ref-pandas3',
        'environment': {
            'python': platform.python_version(),
            'pandas': pd.__version__,
            'pyarrow': pa.__version__,
            'numpy': np.__version__,
            'duckdb': duckdb.__version__,
        },
    }
    OUT.write_text(json.dumps(document, indent=1) + '\n', encoding='utf-8')
    print(f'{len(rows)} witnessed rows recorded', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
