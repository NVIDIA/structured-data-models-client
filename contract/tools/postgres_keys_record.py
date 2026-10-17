#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Records the reference's handling of key types on Postgres into
``contract/postgres-key-vectors.json`` (T2.3's key-type and decimal-key
pair matrices).

Run with the reference checkout's interpreter (the environment of
``contract/reference-env.txt``, the ``ref-pandas3`` cell), from this
repository's root, with ``SDM_POSTGRES_TEST_DSN`` naming a Postgres 17
server the runner may create databases on::

    ~/code/sdmc-reference/.venv/bin/python -I \
        contract/tools/postgres_keys_record.py

Written by T2.3's independent test author before the implementation.
Each vector is ``contract/tools/decimal_record.py``'s: ``contract/bases.yaml``'s
``shop`` changed by its ``setup`` (DuckDB statements), asked that
recorder's question for its own ``indices`` (default ``[0, 1]``), written
as a case and run by ``contract/runners/run_python.py --mode python-ref
--database postgres``: the contract loader (``databases.py``) loads the
tables into a new Postgres database as its type map says, and the public
calls run over a ``psycopg`` connection to it. The vectors:

- ``pair.<primary>.<foreign>``: the key-type pair matrix on Postgres
  (design, choice 5; lesson 19): ``users.user_id`` (the primary key) and
  ``orders.user_id`` (the foreign key) each of one of nine Postgres key
  types, every ordered pair, the diagonal included: ``smallint``,
  ``integer``, ``bigint``, ``numeric(20,0)`` (DuckDB's UBIGINT, the
  unsigned 64-bit key), ``numeric(38,0)`` (a big integer), ``numeric(12,4)``
  (a decimal with a scale), ``real``, ``double precision`` and ``text``,
  all holding the shop's keys 0 to 7;
- ``key.*``: every decimal key vector of ``decimal-vectors.json`` (T1.32b's
  and T2.0c's, the decimal-key pair matrix among them: equal keys across
  scales, ``key.pair.d10-2.d12-4`` and its neighbours, and keys float64
  would merge, ``key.pair-merging.*``), on Postgres;
- ``fifteen.*`` (KO2-35): keys equal to 15 significant digits
  (1 + i * 10**-16 for i in 0 to 7), which float64 merges into one, of one
  type at both ends and across scales (``numeric(17,16)`` against
  ``numeric(18,17)``), the entities given as decimals.

For each vector the reference's public answer is kept as
``decimal_record.project`` reads it (outcome, schema, warnings, each
request's digest and tables, the result, and the key columns), the
statements it ran per public call (``compare.py``'s counts, the capture
complete against the driver's statement log), the ``register`` mark
(``empty_null_record.register``: derived, never chosen), and, read from the
loaded database, each key column's Postgres type by name (``format_type``,
never an OID). And ``joined``: the digest of the requests' tables with every
key cell read exactly and mapped back to the shop's key 0 to 7 (less the
vector's ``offset``, over its ``unit``), each table's rows sorted, so two
vectors whose requests join the same rows to the same entities have the same
``joined``, whatever their key types; the test holds the core's to the
same-type control's (``pair.i64.i64``).

Nothing here is written by hand but the inputs. ``VECTORS``, ``run``,
``joined`` and ``project`` are imported by
``contract/tests/test_t23_postgres_keys.py``, so ``python-core`` is observed
and projected by the same code.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contract/postgres-key-vectors.json'
RUNNER = ROOT / 'contract/runners/run_python.py'
FORMAT = 'sdm-postgres-key-vectors/1'
#: How a register entry names the vectors it covers, in its ``refs``: this
#: prefix, a vector's id or a glob of ids (``fnmatch``), and the transition
#: it declares, ``<from> -> <to>``, each ``success`` or ``refusal <CODE>``.
REGISTER_REF = 'contract/postgres-key-vectors.json#'
#: Each vector's case id is this prefix and the vector's id.
CASE_PREFIX = 't23.'
DATABASE = 'postgres'


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


#: T1.32b's recorder: the cases, the projection and the decimal key
#: vectors are its, so both databases' vectors are read by one code.
R = _module('decimal_record', ROOT / 'contract/tools/decimal_record.py')
E = R.E
QUESTION = R.QUESTION
INDICES = R.INDICES
ANCHOR = R.ANCHOR
KEYS = R.KEYS

# --- the vectors ---------------------------------------------------------------

#: The key types of the pair matrix: a tag, the DuckDB type the setup gives
#: the column, and the Postgres type the contract loader stores it as.
TYPES: list[tuple[str, str, str]] = [
    ('i16', 'SMALLINT', 'smallint'),
    ('i32', 'INTEGER', 'integer'),
    ('i64', 'BIGINT', 'bigint'),
    ('u64', 'UBIGINT', 'numeric(20,0)'),
    ('d38-0', 'DECIMAL(38, 0)', 'numeric(38,0)'),
    ('d12-4', 'DECIMAL(12, 4)', 'numeric(12,4)'),
    ('f32', 'FLOAT', 'real'),
    ('f64', 'DOUBLE', 'double precision'),
    ('text', 'VARCHAR', 'text'),
]
#: The pair whose requests every other pair's must join as: one integer
#: type at both ends, the shop as the base builds it.
CONTROL = 'pair.i64.i64'


def _alter(table: str, sql: str, using: str = '') -> str:
    return f'ALTER TABLE {table} ALTER user_id TYPE {sql}' + (
        f' USING {using}' if using else ''
    )


def _pairs() -> list[dict[str, Any]]:
    out = []
    for primary, primary_sql, primary_pg in TYPES:
        for foreign, foreign_sql, foreign_pg in TYPES:
            out.append(
                {
                    'id': f'pair.{primary}.{foreign}',
                    'columns': KEYS,
                    'setup': [
                        _alter('users', primary_sql),
                        _alter('orders', foreign_sql),
                    ],
                    'postgres_types': [primary_pg, foreign_pg],
                }
            )
    return out


def _decimal_keys() -> list[dict[str, Any]]:
    r"""``decimal-vectors.json``'s key vectors, as that recorder names them;
    the merging pairs' keys sit at 10**17."""
    out = []
    for vector in R.VECTORS:
        if not vector['id'].startswith('key.'):
            continue
        entry = copy.deepcopy(vector)
        if vector['id'].startswith('key.pair-merging.'):
            entry['offset'] = str(10**17)
        out.append(entry)
    return out


#: 1 + i * 10**-16: equal to 15 significant digits, one double for all.
_FIFTEEN = "1 + user_id * '0.0000000000000001'::DECIMAL(17, 16)"


def _fifteen() -> list[dict[str, Any]]:
    indices = ['1.0000000000000000', '1.0000000000000001']
    out = []
    for primary, foreign in [((17, 16), (17, 16)), ((17, 16), (18, 17))]:
        name = f'fifteen.{R.tag(*primary)}.{R.tag(*foreign)}'
        out.append(
            {
                'id': name,
                'columns': KEYS,
                'setup': [
                    _alter('users', R.sql_type(*primary), _FIFTEEN),
                    _alter('orders', R.sql_type(*foreign), _FIFTEEN),
                ],
                'indices': indices,
                'indices_as': 'decimal',
                'offset': '1',
                'unit': '0.0000000000000001',
            }
        )
    return out


#: The vectors: each its id, the key ``columns`` it is about, its setup
#: after ``shop``'s statements, and where it varies them its entities
#: (``indices``, read as decimals with ``indices_as``), and how its keys map
#: back to the shop's 0 to 7 (``offset``, ``unit``; default 0 and 1).
VECTORS: list[dict[str, Any]] = _pairs() + _decimal_keys() + _fifteen()

# --- the cases -----------------------------------------------------------------


def on_postgres(route: str) -> str:
    r"""A DuckDB route or path of the candidate's routing, as the same
    operation and mode over a Postgres source."""
    head, backend = route.rsplit('/', 1)
    assert backend == 'duckdb', route
    return f'{head}/{DATABASE}'


def case_for(vector: dict[str, Any], core: bool = False) -> dict[str, Any]:
    r"""The vector as a case: ``decimal_record.case_for``'s, its paths and,
    with ``core``, its routes those of the same calls over a Postgres
    source, the core's provider the Postgres dialect, and its entities read
    as decimals where the vector says so."""
    case = R.case_for(vector, core)
    case['id'] = CASE_PREFIX + vector['id']
    case['paths'] = [on_postgres(path) for path in case['paths']]
    case['routes'] = [on_postgres(route) for route in case['routes']]
    for operation in ('describe_source', 'materialize'):
        case['core'][operation]['provider'] = {
            'provider': 'sql',
            'dialect': DATABASE,
        }
    if 'indices_as' in vector:
        case['input']['materialize']['indices_as'] = vector['indices_as']
    case['why'] = f'T2.3 vector {vector["id"]}'
    case['source'] = 'contract/tools/postgres_keys_record.py'
    return case


def run(
    mode: str,
    vectors: list[dict[str, Any]],
    directory: Path,
    *,
    core: bool = False,
    reference: Path | None = None,
) -> dict[str, dict[str, Any]]:
    r"""``vectors`` as cases through ``run_python.py --mode mode --database
    postgres``, in this interpreter, from ``directory``, forced onto the
    core's Postgres routes with ``core``: each observation by vector id.
    Raises ``RuntimeError`` with the runner's output when the run fails as
    a whole (no Postgres server named, or no core provider for it)."""
    cases = directory / 'cases'
    cases.mkdir(parents=True, exist_ok=True)
    for vector in vectors:
        case = case_for(vector, core)
        (cases / f'{case["id"]}.json').write_text(
            json.dumps(case, indent=1), encoding='utf-8'
        )
    out = directory / f'{mode}.jsonl'
    command = [sys.executable, '-I', str(RUNNER), '--mode', mode]
    if reference is not None:
        command += ['--reference', str(reference)]
    command += ['--database', DATABASE, '--out', str(out), str(cases)]
    done = subprocess.run(
        command, capture_output=True, text=True, cwd=ROOT, check=False
    )
    if done.returncode != 0:
        raise RuntimeError(f'{mode} run failed:\n{done.stdout}\n{done.stderr}')
    observed = {}
    suffix = f'@{DATABASE}'
    for line in out.read_text(encoding='utf-8').splitlines():
        observation = json.loads(line)
        key = observation['case']
        assert key.startswith(CASE_PREFIX) and key.endswith(suffix), key
        observed[key[len(CASE_PREFIX) : -len(suffix)]] = observation
    return observed


# --- the answer ----------------------------------------------------------------

project = R.project
statements = R.statements
outcome_class = E.outcome_class
register = E.register

#: The key columns of a request: the entity reference and the tables'
#: ``user_id``.
KEY_COLUMNS = frozenset({'__kumo_entity_ref_0', 'user_id'})


def exact(value: Decimal) -> str:
    r"""A decimal's one text: positional, no trailing fractional zeros, no
    trailing point, ``-0`` written ``0``."""
    text = format(value.normalize(), 'f')
    if '.' in text:
        text = text.rstrip('0').rstrip('.')
    return '0' if text in ('-0', '') else text


def _key(cell: Any, offset: Decimal, unit: Decimal) -> str:
    r"""A key cell, sent as a number, as text or as a float, read exactly
    and mapped back to the shop's key."""
    raw = cell['number'] if isinstance(cell, dict) else cell
    return exact((Decimal(str(raw)) - offset) / unit)


def joined(observation: dict[str, Any], vector: dict[str, Any]) -> str | None:
    r"""The digest of every request's tables, each key cell read exactly
    and mapped back to the shop's key (less ``offset``, over ``unit``),
    each table's rows sorted: which rows each request joined to which
    entity, whatever the key's type. Null when no request was made."""
    payloads = (observation.get('public') or {}).get('payloads')
    if not payloads:
        return None
    offset = Decimal(vector.get('offset', '0'))
    unit = Decimal(vector.get('unit', '1'))
    requests = []
    for payload in payloads:
        body = payload.get('parsed') or {}
        tables = {}
        for part in ('context', 'predict'):
            section = body.get(part) or {}
            found = {'instance_table': section.get('instance_table')}
            found.update(section.get('related_tables') or {})
            for name, table in found.items():
                if not isinstance(table, dict):
                    continue
                columns = table['columns']
                rows = [
                    [
                        _key(cell, offset, unit)
                        if column in KEY_COLUMNS and cell is not None
                        else cell
                        for column, cell in zip(columns, row, strict=True)
                    ]
                    for row in table['rows']
                ]
                tables[f'{part}.{name}'] = [
                    columns,
                    sorted(rows, key=lambda row: json.dumps(row)),
                ]
        requests.append(tables)
    text = json.dumps(requests, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def postgres_holds(vector: dict[str, Any]) -> dict[str, str]:
    r"""Each key column's Postgres type by name (``format_type``), as the
    contract loader stores the vector's tables, read in a database of its
    own on ``SDM_POSTGRES_TEST_DSN``'s server."""
    import uuid

    import duckdb
    import psycopg
    from psycopg.conninfo import make_conninfo

    loader = _module('databases', ROOT / 'contract/runners/databases.py')
    dsn = os.environ['SDM_POSTGRES_TEST_DSN']
    name = f'sdm_t23_{uuid.uuid4().hex[:16]}'
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute(f'CREATE DATABASE {name}')
    try:
        duck = duckdb.connect()
        try:
            for statement in E.shop() + vector['setup']:
                duck.execute(statement)
            scoped = make_conninfo(dsn, dbname=name)
            loader.load(duck, DATABASE, scoped)
        finally:
            duck.close()
        out = {}
        with psycopg.connect(scoped) as connection:
            for table, column in vector['columns']:
                row = connection.execute(
                    'SELECT format_type(a.atttypid, a.atttypmod) '
                    'FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid '
                    'WHERE c.relname = %s AND a.attname = %s AND a.attnum > 0',
                    [table, column],
                ).fetchone()
                out[f'{table}.{column}'] = row[0] if row else None
        return out
    finally:
        with psycopg.connect(dsn, autocommit=True) as admin:
            admin.execute(f'DROP DATABASE {name} WITH (FORCE)')


def main() -> int:
    import duckdb
    import numpy as np
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
    if not os.environ.get('SDM_POSTGRES_TEST_DSN'):
        print('SDM_POSTGRES_TEST_DSN names no server', file=sys.stderr)
        return 1
    with psycopg.connect(os.environ['SDM_POSTGRES_TEST_DSN']) as connection:
        server = connection.info.server_version // 10000
    only = sys.argv[1:]
    chosen = [
        v
        for v in VECTORS
        if not only or any(v['id'].startswith(o) for o in only)
    ]
    with tempfile.TemporaryDirectory(prefix='t23-record-') as tmp:
        observed = run('python-ref', chosen, Path(tmp), reference=reference)
    vectors = []
    for vector in chosen:
        observation = observed[vector['id']]
        answer = project(observation, vector)
        why = register(answer)
        entry = {key: copy.deepcopy(value) for key, value in vector.items()}
        entry['postgres'] = postgres_holds(vector)
        entry['reference'] = answer
        entry['statements'] = statements(observation)
        entry['joined'] = joined(observation, vector)
        entry['register'] = {'required': bool(why), 'why': why}
        vectors.append(entry)
        outcome = answer['outcome']
        print(
            vector['id'],
            outcome['result'],
            outcome['stage'],
            outcome['error'] or '',
            outcome['code'] or '',
            '(register)' if why else '',
            file=sys.stderr,
        )
    if only:
        print(json.dumps(vectors, indent=1))
        return 0
    document = {
        'format': FORMAT,
        'recorded_by': 'contract/tools/postgres_keys_record.py',
        'reference': commit,
        'cell': 'ref-pandas3',
        'environment': {
            'python': platform.python_version(),
            'pandas': pd.__version__,
            'pyarrow': pa.__version__,
            'numpy': np.__version__,
            'duckdb': duckdb.__version__,
            'psycopg': psycopg.__version__,
            'postgres': server,
        },
        'template': 'contract/cases/s1/decimal.feature-38-0-wide.json',
        'question': QUESTION,
        'indices': INDICES,
        'anchor_time': ANCHOR,
        'register_ref': REGISTER_REF,
        'control': CONTROL,
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
