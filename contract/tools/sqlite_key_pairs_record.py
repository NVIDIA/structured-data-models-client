#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Records the reference's joins over every pair of key types on SQLite
into ``contract/sqlite-key-pair-vectors.json`` (T2.2's vectors).

Run with the reference checkout's interpreter (the environment of
``contract/reference-env.txt``, the ``ref-pandas3`` cell), from this
repository's root::

    ~/code/sdmc-reference/.venv/bin/python -I contract/tools/sqlite_key_pairs_record.py

Written by T2.2's test author before the implementation (design, choice
5 and lesson 19: every package that binds or casts keys runs the key-type
pair matrix on its own engine; T2.2's row: the key-type pair matrix on
SQLite, and the decimal-key pair matrix with equal keys across scales).
Each vector is ``contract/bases.yaml``'s ``shop`` with ``users.user_id``
(the primary key) and ``orders.user_id`` (the foreign key) changed by its
``setup`` (DuckDB statements), loaded into SQLite by the contract loader
(``contract/runners/databases.py``, the one loader the reference and both
core hosts read), and asked ``decimal.feature-38-0-wide``'s question for
its ``indices`` (default ``[0, 1]``):

- ``pair.<primary>.<foreign>``: every ordered pair of the key types a link
  allows (lesson 19: signed, unsigned and big integers, decimals with and
  without scale, floats, text), holding the shop's ids 0 to 7;
- ``wide.<primary>.<foreign>``: keys past what float64 tells apart
  (10**17 + i) and past int64 (2**63 + i);
- ``decimal.<id>``: T2.0c's decimal-key pair matrix and its decimal
  against integer keys (``contract/tools/decimal_record.py``'s
  ``key.pair*``, ``key.foreign-only.*`` and ``key.primary-only.*``, with
  their same-type control ``key.both.d9-0``), the setup theirs: equal keys
  across scales (1.00 against 1.0000, both ways; scale 0 against 2), and
  keys float64 would merge (10**17 + i at DECIMAL(38, 0) and (38, 2));
  and equal keys across scales with a fraction (``key.pair-fraction``:
  i + 0.25 at DECIMAL(10, 2) against 0.2500 at DECIMAL(12, 4), both
  ways), the entities given as the texts '0.25' and '1.25'.

The type map stores every integer up to UINTEGER as ``INTEGER``, a float
as ``REAL``, and a UBIGINT, a HUGEINT, a decimal and text as ``TEXT`` (a
decimal as its canonical text, so equal values of any scale have one
text): ``sqlite`` records, for each key column, the type SQLite declares
and each stored value as SQLite's ``quote()`` writes it, so what each
vector puts to the hosts is visible, not assumed.

The reference runs each vector as a case through
``contract/runners/run_python.py --mode python-ref --database sqlite``,
and this script keeps its answer as ``decimal_record.py``'s ``project``
reads it (``reference``: the outcome, public schema, warnings, each
request's digest as the ``payload`` rule compares it and its tables' row
counts, the result, and the key columns as they reached the model), the
statements per public call (``statements``), and whether each request the
reference sent joins every row to its own entity (``joined``, by
:func:`joins`: the shop's facts make a wrong join visible, whatever rows
SQLite's sampling draws). ``case`` is the case each host runs (its routes
left empty: the Python test forces the candidate's own).

``register`` marks where a host may differ from the reference: the
reference fails unnamed or with ``INTERNAL_ERROR`` (T1.32d's rule), or it
answers with a wrong join. It is derived, never chosen; a host differing
on any vector needs a register entry whose ``refs`` name it
(``REGISTER_REF`` + its id).

Nothing here is written by hand but the inputs. ``VECTORS``, ``joins`` and
``REGISTER_REF`` are imported by ``contract/tests/test_t22_sqlite.py``.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import platform
import sqlite3
import subprocess
import sys
import tempfile
from contextlib import closing
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contract/sqlite-key-pair-vectors.json'
RUNNER = ROOT / 'contract/runners/run_python.py'
FORMAT = 'sdm-sqlite-key-pair-vectors/1'
REGISTER_REF = 'contract/sqlite-key-pair-vectors.json#'
DATABASE = 'sqlite'


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


#: T1.32b's recorder: the case template, the projection and the statement
#: counts are its, so both families' vectors are read by one code.
R = _module('decimal_record', ROOT / 'contract/tools/decimal_record.py')
KEYS = R.KEYS

# --- the vectors -------------------------------------------------------------------

#: The key types of the matrix, by tag: lesson 19's list.
TYPES = {
    'bigint': 'BIGINT',
    'ubigint': 'UBIGINT',
    'hugeint': 'HUGEINT',
    'd18-0': 'DECIMAL(18, 0)',
    'd10-2': 'DECIMAL(10, 2)',
    'double': 'DOUBLE',
    'float': 'FLOAT',
    'varchar': 'VARCHAR',
}


def _alter(table: str, sql: str, using: str = '') -> str:
    return R._alter_key(table, 'user_id', sql, using)


def _pairs() -> list[dict[str, Any]]:
    return [
        {
            'id': f'pair.{p}.{f}',
            'columns': KEYS,
            'setup': [_alter('users', TYPES[p]), _alter('orders', TYPES[f])],
        }
        for p in TYPES
        for f in TYPES
    ]


def _wide() -> list[dict[str, Any]]:
    out = []
    # 10**17 + i: a double holds every 16th integer there.
    base = 10**17
    for p, f in [
        ('bigint', 'bigint'),
        ('bigint', 'd18-0'),
        ('d18-0', 'bigint'),
    ]:
        out.append(
            {
                'id': f'wide.{p}.{f}',
                'columns': KEYS,
                'setup': [
                    _alter('users', TYPES[p], f'user_id + {base}'),
                    _alter('orders', TYPES[f], f'user_id + {base}'),
                ],
                'indices': [base, base + 1],
            }
        )
    # 2**63 + i: past int64.
    base = 2**63
    for p, f in [('ubigint', 'ubigint'), ('ubigint', 'hugeint')]:
        out.append(
            {
                'id': f'wide.{p}.{f}',
                'columns': KEYS,
                'setup': [
                    _alter(
                        'users',
                        TYPES[p],
                        f'user_id::{TYPES[p]} + {base}::{TYPES[p]}',
                    ),
                    _alter(
                        'orders',
                        TYPES[f],
                        f'user_id::{TYPES[f]} + {base}::{TYPES[f]}',
                    ),
                ],
                'indices': [base, base + 1],
            }
        )
    return out


#: T2.0c's decimal-key vectors this matrix takes, with their control.
DECIMAL_KEYS = ('key.pair', 'key.foreign-only.', 'key.primary-only.')
CONTROL = 'key.both.d9-0'


def _decimals() -> list[dict[str, Any]]:
    out = []
    for vector in R.VECTORS:
        if vector['id'].startswith(DECIMAL_KEYS) or vector['id'] == CONTROL:
            taken = copy.deepcopy(vector)
            taken['id'] = f'decimal.{vector["id"]}'
            out.append(taken)
    # Equal keys across scales with a fraction (i + 0.25 at DECIMAL(10, 2)
    # and as 0.2500 at DECIMAL(12, 4)), both ways: one canonical text.
    for (pp, ps), (fp, fs) in [((10, 2), (12, 4)), ((12, 4), (10, 2))]:
        primary, foreign = R.sql_type(pp, ps), R.sql_type(fp, fs)
        out.append(
            {
                'id': f'decimal.key.pair-fraction.{R.tag(pp, ps)}.'
                f'{R.tag(fp, fs)}',
                'columns': KEYS,
                'setup': [
                    _alter('users', primary, f"user_id + '0.25'::{primary}"),
                    _alter('orders', foreign, f"user_id + '0.25'::{foreign}"),
                ],
                'indices': ['0.25', '1.25'],
            }
        )
    return out


#: The vectors: each its id, its key columns, its setup after ``shop``'s
#: statements, and its entities where it varies them.
VECTORS: list[dict[str, Any]] = _pairs() + _wide() + _decimals()


def indices(vector: dict[str, Any]) -> list[Any]:
    return list(vector.get('indices', R.INDICES))


def offset(vector: dict[str, Any]) -> Decimal:
    r"""What the vector adds to the shop's ids 0 to 7: its first entity,
    exactly."""
    return Decimal(str(indices(vector)[0]))


def case_for(vector: dict[str, Any]) -> dict[str, Any]:
    r"""The vector as the case each host runs: ``decimal_record.py``'s, on
    no routes (a Python runner forces its own) and named ``t22.<id>``."""
    case = R.case_for({**vector, 'id': vector['id']})
    case['id'] = f't22.{vector["id"]}'
    case['why'] = f'T2.2 vector {vector["id"]}'
    case['source'] = 'contract/tools/sqlite_key_pairs_record.py'
    return case


# --- the join check ----------------------------------------------------------------

#: The key columns of a request: the entity reference and the key.
KEY_COLUMNS = frozenset({'__kumo_entity_ref_0', 'user_id'})


def _number(cell: Any) -> Decimal:
    r"""A cell's exact value, sent as a number or as its digits."""
    return Decimal(cell['number'] if isinstance(cell, dict) else str(cell))


def joins(body: Any, base: Decimal) -> list[str]:
    r"""Where a request joins a row to another entity than its own, by the
    shop's facts (``users``: ``age`` is 20 + id; ``orders``: ``user_id`` is
    ``order_id % 8`` and ``price`` 10 + 1.5 * ``order_id``), the keys read
    exactly and less ``base``: each related row's key is its instance's
    entity, and its facts are that entity's. Empty: every row joins right.
    The predicted rows are the entities asked for."""
    wrong: list[str] = []
    for part in ('context', 'predict'):
        section = body[part]
        instances = section['instance_table']
        at = instances['columns']
        entity = {
            json.dumps(row[at.index('instance_id')]): _number(
                row[at.index('__kumo_entity_ref_0')]
            )
            - base
            for row in instances['rows']
        }
        for name, table in section['related_tables'].items():
            columns = table['columns']
            for row in table['rows']:
                cell = dict(zip(columns, row, strict=True))
                own = entity[json.dumps(cell['instance_id'])]
                if 'user_id' in cell and (
                    cell['user_id'] is None
                    or _number(cell['user_id']) - base != own
                ):
                    wrong.append(f'{part}.{name}: user_id of entity {own}')
                if name == 'users' and _number(cell['age']) - 20 != own:
                    wrong.append(f'{part}.users: age of entity {own}')
                if name == 'orders':
                    order = (_number(cell['price']) - 10) / Decimal('1.5')
                    if order % 8 != own:
                        wrong.append(f'{part}.orders: order of entity {own}')
    return wrong


def predicted(body: Any, base: Decimal) -> list[Decimal]:
    r"""The entities the request's ``predict`` part asks about, less
    ``base``."""
    instances = body['predict']['instance_table']
    at = instances['columns'].index('__kumo_entity_ref_0')
    return [_number(row[at]) - base for row in instances['rows']]


def joined(observation: dict[str, Any], vector: dict[str, Any]) -> bool | None:
    r"""Whether every request of a run joins every row to its own entity and
    asks about the entities given; None when it sent none."""
    payloads = (observation.get('public') or {}).get('payloads') or []
    if not payloads:
        return None
    base = offset(vector)
    asked = [Decimal(str(i)) - base for i in indices(vector)]
    return all(
        not joins(p['parsed'], base) and predicted(p['parsed'], base) == asked
        for p in payloads
    )


# --- what SQLite holds -------------------------------------------------------------


def sqlite_holds(vector: dict[str, Any]) -> dict[str, Any]:
    r"""Each key column as the contract loader stores it: the type SQLite
    declares and each stored value as ``quote()`` writes it, in rowid
    order."""
    import duckdb

    loader = _module('databases', ROOT / 'contract/runners/databases.py')
    duck = duckdb.connect()
    try:
        for statement in R.E.shop() + vector['setup']:
            duck.execute(statement)
        with tempfile.TemporaryDirectory(prefix='t22-holds-') as directory:
            path = Path(directory) / 'case.sqlite'
            loader.load(duck, DATABASE, path)
            with closing(sqlite3.connect(path)) as connection:
                out = {}
                for table, column in vector['columns']:
                    declared = {
                        row[1]: row[2]
                        for row in connection.execute(
                            f'PRAGMA table_info({loader.quote(table)})'
                        )
                    }[column]
                    values = [
                        row[0]
                        for row in connection.execute(
                            f'SELECT quote({loader.quote(column)}) FROM '
                            f'{loader.quote(table)} ORDER BY rowid LIMIT 8'
                        )
                    ]
                    out[f'{table}.{column}'] = {
                        'declared': declared,
                        'values': values,
                    }
                return out
    finally:
        duck.close()


# --- the reference's runs ----------------------------------------------------------


def run(
    mode: str,
    vectors: list[dict[str, Any]],
    directory: Path,
    *,
    routes: Any = None,
    reference: Path | None = None,
) -> dict[str, dict[str, Any]]:
    r"""``vectors`` as cases through ``run_python.py --mode mode --database
    sqlite``, in this interpreter: each observation by vector id. ``routes``
    gives a vector's case its routes (a Python core runner's)."""
    cases = directory / 'cases'
    cases.mkdir(parents=True, exist_ok=True)
    for vector in vectors:
        case = case_for(vector)
        if routes is not None:
            case['routes'] = routes(vector)
        (cases / f'{case["id"]}.json').write_text(
            json.dumps(case, indent=1), encoding='utf-8'
        )
    out = directory / f'{mode}.jsonl'
    command = [sys.executable, '-I', str(RUNNER), '--mode', mode]
    if reference is not None:
        command += ['--reference', str(reference)]
    command += ['--database', DATABASE, '--out', str(out), str(cases)]
    done = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    if done.returncode != 0:
        raise RuntimeError(f'{mode} run failed:\n{done.stdout}\n{done.stderr}')
    observed = {}
    for line in out.read_text(encoding='utf-8').splitlines():
        observation = json.loads(line)
        name = observation['case'].removesuffix(f'@{DATABASE}')
        observed[name.removeprefix('t22.')] = observation
    return observed


def register(answer: dict[str, Any], right: bool | None) -> list[str]:
    r"""Why a host may differ from the reference on this vector only under a
    register entry: T1.32d's marks, and a wrong join."""
    why = list(R.register(answer))
    if right is False:
        why.append('the reference joins a row to another entity')
    return why


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
    with tempfile.TemporaryDirectory(prefix='t22-record-') as tmp:
        observed = run('python-ref', VECTORS, Path(tmp), reference=reference)
    vectors = []
    for vector in VECTORS:
        observation = observed[vector['id']]
        answer = R.project(observation, vector)
        right = joined(observation, vector)
        why = register(answer, right)
        entry = {key: copy.deepcopy(value) for key, value in vector.items()}
        entry['case'] = case_for(vector)
        entry['sqlite'] = sqlite_holds(vector)
        entry['reference'] = answer
        entry['joined'] = right
        entry['statements'] = R.statements(observation)
        entry['register'] = {'required': bool(why), 'why': why}
        vectors.append(entry)
        outcome = answer['outcome']
        print(
            vector['id'],
            outcome['result'],
            outcome['stage'],
            outcome['code'] or '',
            {'joined': right},
            '(register)' if why else '',
            file=sys.stderr,
        )
    document = {
        'format': FORMAT,
        'recorded_by': 'contract/tools/sqlite_key_pairs_record.py',
        'reference': commit,
        'cell': 'ref-pandas3',
        'database': DATABASE,
        'environment': {
            'python': platform.python_version(),
            'pandas': pd.__version__,
            'pyarrow': pa.__version__,
            'numpy': np.__version__,
            'duckdb': duckdb.__version__,
            'sqlite': sqlite3.sqlite_version,
        },
        'question': R.QUESTION,
        'register_ref': REGISTER_REF,
        'payload_exclusions': R.E.exclusions(),
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
