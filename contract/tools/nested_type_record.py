#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Records the reference's handling of nested column types into
``contract/nested-type-vectors.json`` (T1.32c's vectors).

Run with the reference checkout's interpreter (the environment of
``contract/reference-env.txt``, the ``ref-pandas3`` cell), from this
repository's root::

    ~/code/sdmc-reference/.venv/bin/python -I contract/tools/nested_type_record.py

Written by T1.32c's independent test author before the implementation.
Each vector is ``contract/bases.yaml``'s ``shop`` with one column ``f``
added to ``users`` (or to ``orders``, the fact table the request's context
reads) by a DuckDB expression (``TYPES``): lists of each scalar type DuckDB
has, empty and null lists and null items, nested lists, lists of structs
and of maps, fixed-size lists (ARRAY), structs, maps, a union, and the
scalar binary (BLOB) and interval columns ``column_type`` also refuses.
DuckDB hands no Arrow ``duration`` over (an INTERVAL is
``month_day_nano_interval``), so the interval columns are the vectors of
that refusal. Some vectors set DuckDB's Arrow options first, so the list
reaches the host as a ``large_list`` or a ``list_view``.

Each vector is a case in ``contract/cases``' format, ``nested.list-int``'s
with the vector's setup, written to a temporary directory and run by
``contract/runners/run_python.py --mode python-ref`` through the public
calls (``Schema.from_source``, ``materialize`` with ``indices=[0, 1]``,
``SDMClient.predict`` against the scripted endpoint), and this script keeps
the reference's answer as ``project`` reads it from the observation:

- ``outcome``: success, or the refusal's class, ``StructuredDataError.code``
  and stage (no message text, H17);
- ``described``: what ``Schema.from_source`` made of ``f``: its dtype and
  stype, ``"left out"`` when its table is described without it, or null
  when there is no schema;
- ``warnings``: the warning classes, in order;
- ``payload``: in the request the scripted endpoint received, ``f``'s entry
  in the payload's schema and its cells in the ``predict`` table's rows,
  each null when ``f`` is not there; null when no request was made.

and, beside it, what DuckDB hands over: ``f``'s catalog type
(``typeof``) and its Arrow type (``duckdb_arrow``), observed with the
reference environment's DuckDB.

``register`` marks the vectors where the core is expected to differ, so
that it may only under a register entry naming the vector
(``REGISTER_REF``). It is derived from the reference's answer, never
chosen: the reference fails with an unnamed exception (no
``StructuredDataError`` code) or with ``INTERNAL_ERROR`` (E66(b)'s class:
a named refusal or a success in its place is a change to register). Every
other vector is held to the reference exactly. This decides no policy.

Nothing here is written by hand but the types: every expected value is the
reference's. ``TYPES``, ``case_for``, ``run`` and ``project`` are imported
by ``contract/tests/test_t132c_nested_types.py``, so ``python-core`` is
observed and projected by the same code.
"""

from __future__ import annotations

import copy
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contract/nested-type-vectors.json'
TEMPLATE = ROOT / 'contract/cases/s1/nested.list-int.json'
RUNNER = ROOT / 'contract/runners/run_python.py'
FORMAT = 'sdm-nested-type-vectors/1'
#: How a register entry names the vectors it covers, in its ``refs``: this
#: prefix and a vector's id, or a glob of ids (``fnmatch``), for example
#: ``contract/nested-type-vectors.json#struct.*``.
REGISTER_REF = 'contract/nested-type-vectors.json#'
#: Each vector's case id is this prefix and the vector's id.
CASE_PREFIX = 't132c.'

_LARGE = 'SET arrow_large_buffer_size = true'
_VIEW = [
    'SET arrow_output_list_view = true',
    "SET arrow_output_version = '1.4'",
]


def _users(expression: str) -> list[str]:
    return [
        f'CREATE OR REPLACE TABLE users AS SELECT *, {expression} AS f '
        'FROM users'
    ]


def _orders(expression: str) -> list[str]:
    return [
        f'CREATE OR REPLACE TABLE orders AS SELECT *, {expression} AS f '
        'FROM orders'
    ]


def _pair(
    sql_type: str, first: str = 'user_id', second: str = 'user_id + 1'
) -> str:
    return f'[{first}, {second}]::{sql_type}[]'


#: The column types, each a vector: its id, the table ``f`` is added to,
#: and the setup that adds it (after ``shop``'s statements). The ARRAYs of
#: ARRAYs and of lists are built with ``array_value``: DuckDB 1.5.6's cast
#: of a nested list literal to them (``[[user_id, 1], [2, 3]]::INTEGER[2][2]``)
#: intermittently read a garbage value and failed the setup on a loaded
#: machine.
TYPES: list[dict[str, Any]] = [
    # lists of each scalar type
    {
        'id': 'list.boolean',
        'table': 'users',
        'setup': _users('[user_id % 2 = 0, NULL]::BOOLEAN[]'),
    },
    {'id': 'list.tinyint', 'table': 'users', 'setup': _users(_pair('TINYINT'))},
    {
        'id': 'list.smallint',
        'table': 'users',
        'setup': _users(_pair('SMALLINT')),
    },
    {'id': 'list.integer', 'table': 'users', 'setup': _users(_pair('INTEGER'))},
    {'id': 'list.bigint', 'table': 'users', 'setup': _users(_pair('BIGINT'))},
    {'id': 'list.hugeint', 'table': 'users', 'setup': _users(_pair('HUGEINT'))},
    {
        'id': 'list.utinyint',
        'table': 'users',
        'setup': _users(_pair('UTINYINT')),
    },
    {
        'id': 'list.usmallint',
        'table': 'users',
        'setup': _users(_pair('USMALLINT')),
    },
    {
        'id': 'list.uinteger',
        'table': 'users',
        'setup': _users(_pair('UINTEGER')),
    },
    {'id': 'list.ubigint', 'table': 'users', 'setup': _users(_pair('UBIGINT'))},
    {
        'id': 'list.float',
        'table': 'users',
        'setup': _users(_pair('FLOAT', 'user_id + 0.5')),
    },
    {
        'id': 'list.double',
        'table': 'users',
        'setup': _users(_pair('DOUBLE', 'user_id + 0.5')),
    },
    {
        'id': 'list.decimal-18-2',
        'table': 'users',
        'setup': _users(_pair('DECIMAL(18, 2)', 'user_id + 0.25')),
    },
    {
        'id': 'list.decimal-38-10',
        'table': 'users',
        'setup': _users(_pair('DECIMAL(38, 10)', 'user_id + 0.25')),
    },
    {
        'id': 'list.varchar',
        'table': 'users',
        'setup': _users("['a' || user_id, NULL]::VARCHAR[]"),
    },
    {
        'id': 'list.uuid',
        'table': 'users',
        'setup': _users(
            "[('00000000-0000-0000-0000-' || lpad(user_id::VARCHAR, 12, '0'))::UUID]"
        ),
    },
    {
        'id': 'list.date',
        'table': 'users',
        'setup': _users("[DATE '2024-01-01' + user_id::INTEGER]"),
    },
    {
        'id': 'list.timestamp',
        'table': 'users',
        'setup': _users("[TIMESTAMP '2024-01-01' + to_days(user_id::INTEGER)]"),
    },
    {
        'id': 'list.timestamptz',
        'table': 'users',
        'setup': [
            "SET TimeZone = 'UTC'",
            *_users(
                "[TIMESTAMPTZ '2024-01-01 00:00:00+00' + to_days(user_id::INTEGER)]"
            ),
        ],
    },
    {
        'id': 'list.time',
        'table': 'users',
        'setup': _users("[TIME '10:00:00' + to_minutes(user_id::INTEGER)]"),
    },
    {
        'id': 'list.interval',
        'table': 'users',
        'setup': _users('[to_days(user_id::INTEGER)]'),
    },
    {
        'id': 'list.blob',
        'table': 'users',
        'setup': _users("[('x' || user_id)::BLOB]"),
    },
    # empty and null lists, and null items
    {
        'id': 'list.mixed-empty-null',
        'table': 'users',
        'setup': _users(
            '(CASE user_id % 4 WHEN 0 THEN [1, 2] WHEN 1 THEN [] WHEN 2 THEN NULL ELSE [3] END)::INTEGER[]'
        ),
    },
    {
        'id': 'list.all-empty',
        'table': 'users',
        'setup': _users('[]::INTEGER[]'),
    },
    {
        'id': 'list.all-null',
        'table': 'users',
        'setup': _users('NULL::INTEGER[]'),
    },
    {
        'id': 'list.null-items',
        'table': 'users',
        'setup': _users('[NULL, user_id]::INTEGER[]'),
    },
    {
        'id': 'list.only-null-items',
        'table': 'users',
        'setup': _users('[NULL]::INTEGER[]'),
    },
    {
        'id': 'list.all-empty-varchar',
        'table': 'users',
        'setup': _users('[]::VARCHAR[]'),
    },
    # lists as DuckDB's Arrow options hand them over
    {
        'id': 'list.integer-large',
        'table': 'users',
        'setup': [_LARGE, *_users(_pair('INTEGER'))],
    },
    {
        'id': 'list.varchar-large',
        'table': 'users',
        'setup': [_LARGE, *_users("['a' || user_id]::VARCHAR[]")],
    },
    {
        'id': 'list.integer-view',
        'table': 'users',
        'setup': [*_VIEW, *_users(_pair('INTEGER'))],
    },
    # nested lists, and lists of structs and maps
    {
        'id': 'list.list-integer',
        'table': 'users',
        'setup': _users('[[user_id], [user_id + 1, 2]]::INTEGER[][]'),
    },
    {
        'id': 'list.list-varchar',
        'table': 'users',
        'setup': _users("[['a' || user_id]]::VARCHAR[][]"),
    },
    {
        'id': 'list.list-empty',
        'table': 'users',
        'setup': _users('[[]]::INTEGER[][]'),
    },
    {
        'id': 'list.struct',
        'table': 'users',
        'setup': _users("[{'a': user_id::INTEGER}]"),
    },
    {
        'id': 'list.map',
        'table': 'users',
        'setup': _users("[map(['k'], [user_id::INTEGER])]"),
    },
    # fixed-size lists (ARRAY)
    {
        'id': 'array.bigint-2',
        'table': 'users',
        'setup': _users('[user_id, user_id + 1]::BIGINT[2]'),
    },
    {
        'id': 'array.integer-3',
        'table': 'users',
        'setup': _users('[user_id, user_id + 1, 0]::INTEGER[3]'),
    },
    {
        'id': 'array.double-2',
        'table': 'users',
        'setup': _users('[user_id + 0.5, 1.5]::DOUBLE[2]'),
    },
    {
        'id': 'array.varchar-2',
        'table': 'users',
        'setup': _users("['a' || user_id, 'b']::VARCHAR[2]"),
    },
    {
        'id': 'array.boolean-2',
        'table': 'users',
        'setup': _users('[true, user_id % 2 = 0]::BOOLEAN[2]'),
    },
    {'id': 'array.null', 'table': 'users', 'setup': _users('NULL::INTEGER[2]')},
    {
        'id': 'array.array',
        'table': 'users',
        'setup': _users(
            'array_value(array_value(user_id::INTEGER, 1::INTEGER), array_value(2::INTEGER, 3::INTEGER))'
        ),
    },
    {
        'id': 'array.list',
        'table': 'users',
        'setup': _users('array_value([user_id::INTEGER], []::INTEGER[])'),
    },
    # structs
    {
        'id': 'struct.integer-varchar',
        'table': 'users',
        'setup': _users("{'a': user_id::INTEGER, 'b': 'x' || user_id}"),
    },
    {
        'id': 'struct.integer',
        'table': 'users',
        'setup': _users("{'a': user_id::INTEGER}"),
    },
    {
        'id': 'struct.varchar',
        'table': 'users',
        'setup': _users("{'a': 'x' || user_id}"),
    },
    {
        'id': 'struct.double',
        'table': 'users',
        'setup': _users("{'a': (user_id + 0.5)::DOUBLE}"),
    },
    {
        'id': 'struct.nested',
        'table': 'users',
        'setup': _users("{'a': {'b': user_id::INTEGER}}"),
    },
    {
        'id': 'struct.list',
        'table': 'users',
        'setup': _users("{'a': [user_id::INTEGER]}"),
    },
    {
        'id': 'struct.null',
        'table': 'users',
        'setup': _users('NULL::STRUCT(a INTEGER)'),
    },
    {
        'id': 'struct.null-fields',
        'table': 'users',
        'setup': _users("{'a': NULL::INTEGER}"),
    },
    # maps
    {
        'id': 'map.varchar-integer',
        'table': 'users',
        'setup': _users("map(['k'], [user_id::INTEGER])"),
    },
    {
        'id': 'map.integer-varchar',
        'table': 'users',
        'setup': _users("map([user_id::INTEGER], ['x'])"),
    },
    {
        'id': 'map.varchar-list',
        'table': 'users',
        'setup': _users("map(['k'], [[user_id::INTEGER]])"),
    },
    {
        'id': 'map.empty',
        'table': 'users',
        'setup': _users('map([], [])::MAP(VARCHAR, INTEGER)'),
    },
    {
        'id': 'map.null',
        'table': 'users',
        'setup': _users('NULL::MAP(VARCHAR, INTEGER)'),
    },
    # a union
    {
        'id': 'union.integer-varchar',
        'table': 'users',
        'setup': _users(
            'union_value(n := user_id::INTEGER)::UNION(n INTEGER, s VARCHAR)'
        ),
    },
    # binary and interval, the scalars column_type also refuses
    {
        'id': 'scalar.blob',
        'table': 'users',
        'setup': _users("('x' || user_id)::BLOB"),
    },
    {'id': 'scalar.blob-null', 'table': 'users', 'setup': _users('NULL::BLOB')},
    {
        'id': 'scalar.interval',
        'table': 'users',
        'setup': _users('to_days(user_id::INTEGER)'),
    },
    {
        'id': 'scalar.interval-null',
        'table': 'users',
        'setup': _users('NULL::INTERVAL'),
    },
    # on the fact table, which the request's context reads
    {
        'id': 'orders.list-integer',
        'table': 'orders',
        'setup': _orders('[order_id::INTEGER]'),
    },
    {
        'id': 'orders.array-integer-2',
        'table': 'orders',
        'setup': _orders('[order_id::INTEGER, 1]::INTEGER[2]'),
    },
    {
        'id': 'orders.struct',
        'table': 'orders',
        'setup': _orders("{'a': order_id::INTEGER}"),
    },
    {
        'id': 'orders.map',
        'table': 'orders',
        'setup': _orders("map(['k'], [order_id::INTEGER])"),
    },
    {
        'id': 'orders.blob',
        'table': 'orders',
        'setup': _orders("('x' || order_id)::BLOB"),
    },
    {
        'id': 'orders.interval',
        'table': 'orders',
        'setup': _orders('to_days(order_id::INTEGER)'),
    },
]


# --- the cases -----------------------------------------------------------------


def case_for(
    vector: dict[str, Any], routes: list[str] | None = None
) -> dict[str, Any]:
    r"""The vector as a case: ``nested.list-int``'s, its setup the vector's,
    on ``routes`` (none: the runner's own choice, as the reference has)."""
    template = json.loads(TEMPLATE.read_text(encoding='utf-8'))
    case = copy.deepcopy(template)
    case['id'] = CASE_PREFIX + vector['id']
    case['status'] = 'diagnostic'
    case['routes'] = list(routes or [])
    case['cells'] = []
    case['input']['setup'] = list(vector['setup'])
    case['expect'] = {'outcome': 'success', 'stage': 'predict'}
    case['why'] = f'T1.32c vector {vector["id"]}'
    case['source'] = 'contract/tools/nested_type_record.py'
    case.pop('register', None)
    return case


def run(
    mode: str,
    vectors: list[dict[str, Any]],
    directory: Path,
    *,
    routes: list[str] | None = None,
    reference: Path | None = None,
    capture: bool = True,
) -> dict[str, dict[str, Any]]:
    r"""``vectors`` as cases through ``run_python.py --mode mode``, in this
    interpreter, from ``directory``: each observation by vector id."""
    cases = directory / 'cases'
    cases.mkdir(parents=True, exist_ok=True)
    for vector in vectors:
        case = case_for(vector, routes)
        (cases / f'{case["id"]}.json').write_text(
            json.dumps(case, indent=1), encoding='utf-8'
        )
    out = directory / f'{mode}{"" if capture else "-off"}.jsonl'
    command = [sys.executable, '-I', str(RUNNER), '--mode', mode]
    if reference is not None:
        command += ['--reference', str(reference)]
    if not capture:
        command.append('--no-capture')
    command += ['--out', str(out), str(cases)]
    done = subprocess.run(
        command, capture_output=True, text=True, cwd=ROOT, check=False
    )
    if done.returncode != 0:
        raise RuntimeError(f'{mode} run failed:\n{done.stdout}\n{done.stderr}')
    observed = {}
    for line in out.read_text(encoding='utf-8').splitlines():
        observation = json.loads(line)
        observed[observation['case'][len(CASE_PREFIX) :]] = observation
    return observed


# --- the public answer ---------------------------------------------------------


def _described(schema: Any, table: str) -> Any:
    if not isinstance(schema, dict):
        return None
    for entry in schema.get('tables') or []:
        if entry.get('name') != table:
            continue
        for column in entry.get('columns') or []:
            if column.get('name') == 'f':
                return {
                    'dtype': column.get('dtype'),
                    'stype': column.get('stype'),
                }
        return 'left out'
    return None


def _payload(payloads: Any, table: str) -> Any:
    if not payloads:
        return None
    parsed = payloads[0].get('parsed') or {}
    schema = (
        ((parsed.get('schema') or {}).get('related_tables') or {})
        .get(table, {})
        .get('columns', {})
    )
    rows = ((parsed.get('predict') or {}).get('related_tables') or {}).get(
        table
    )
    cells = None
    if rows is not None and 'f' in rows.get('columns', []):
        at = rows['columns'].index('f')
        cells = [row[at] for row in rows.get('rows', [])]
    return {'schema': schema.get('f'), 'cells': cells}


def project(
    observation: dict[str, Any], vector: dict[str, Any]
) -> dict[str, Any]:
    r"""The public answer of one run of ``vector``, as both Python runners
    record it (``run_python.py``'s observation)."""
    outcome = observation['outcome']
    public = observation.get('public') or {}
    host = observation.get('host') or {}
    return {
        'outcome': {
            'result': outcome['result'],
            'stage': outcome['stage'],
            'error': outcome.get('error'),
            'code': outcome.get('code'),
        },
        'described': _described(public.get('schema'), vector['table']),
        'warnings': list(host.get('warnings') or []),
        'payload': _payload(public.get('payloads'), vector['table']),
    }


def register(answer: dict[str, Any]) -> list[str]:
    r"""Why a host may differ from the reference on this vector only under a
    register entry: the reference's answer, never a choice. Empty: held to
    the reference exactly."""
    outcome = answer['outcome']
    if outcome['result'] != 'refusal':
        return []
    if outcome['code'] is None:
        return [
            f'the reference fails with an unnamed {outcome["error"]} at '
            f'{outcome["stage"]} (E66(b): a named refusal or a success in '
            'its place is a change to register)'
        ]
    if outcome['code'] == 'INTERNAL_ERROR':
        return [
            f'the reference fails with INTERNAL_ERROR at {outcome["stage"]} '
            '(E66(b): a named refusal or a success in its place is a change '
            'to register)'
        ]
    return []


# --- what DuckDB hands over ----------------------------------------------------


def shop() -> list[str]:
    r"""``contract/bases.yaml``'s ``shop`` statements."""
    lines = (
        (ROOT / 'contract/bases.yaml').read_text(encoding='utf-8').splitlines()
    )
    start = lines.index('  shop:') + 1
    statements = []
    for line in lines[start:]:
        if not line.startswith('  - '):
            break
        statements.append(line[len('  - ') :])
    return statements


def duckdb_types(vector: dict[str, Any]) -> dict[str, str]:
    r"""``f``'s DuckDB catalog type and the Arrow type DuckDB hands over."""
    import duckdb

    connection = duckdb.connect()
    try:
        for statement in shop() + vector['setup']:
            connection.execute(statement)
        table = vector['table']
        catalog = connection.execute(
            f'SELECT typeof(f) FROM {table} LIMIT 1'
        ).fetchone()
        arrow = connection.execute(f'SELECT f FROM {table}').to_arrow_table()
        return {
            'typeof': str(catalog[0]) if catalog else '',
            'duckdb_arrow': str(arrow.schema.field('f').type),
        }
    finally:
        connection.close()


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
    with tempfile.TemporaryDirectory(prefix='t132c-record-') as tmp:
        observed = run('python-ref', TYPES, Path(tmp), reference=reference)
    vectors = []
    for vector in TYPES:
        answer = project(observed[vector['id']], vector)
        why = register(answer)
        vectors.append(
            {
                'id': vector['id'],
                'table': vector['table'],
                'setup': vector['setup'],
                **duckdb_types(vector),
                'reference': answer,
                'register': {'required': bool(why), 'why': why},
            }
        )
        outcome = answer['outcome']
        print(
            vector['id'],
            outcome['result'],
            outcome['stage'],
            outcome['error'] or '',
            outcome['code'] or '',
            answer['described'],
            '(register)' if why else '',
            file=sys.stderr,
        )
    document = {
        'format': FORMAT,
        'recorded_by': 'contract/tools/nested_type_record.py',
        'reference': commit,
        'cell': 'ref-pandas3',
        'environment': {
            'python': platform.python_version(),
            'pandas': pd.__version__,
            'pyarrow': pa.__version__,
            'numpy': np.__version__,
            'duckdb': duckdb.__version__,
        },
        'template': 'contract/cases/s1/nested.list-int.json',
        'register_ref': REGISTER_REF,
        'vectors': vectors,
    }
    OUT.write_text(json.dumps(document, indent=1) + '\n', encoding='utf-8')
    required = sum(v['register']['required'] for v in vectors)
    print(
        f'{len(vectors)} vectors written to {OUT.relative_to(ROOT)}, '
        f'{required} marked register entry required',
        file=sys.stderr,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
