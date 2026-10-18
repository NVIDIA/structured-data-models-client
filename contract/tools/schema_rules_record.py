#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Records the reference for T2.12's vectors: the key-type pair matrix
for composite keys on all three databases (``contract/composite-key-
vectors.json``) and FR P3-8's key-cast vectors (``contract/key-cast-p3-8-
vectors.json``).

Run with the reference checkout's interpreter (the environment of
``contract/reference-env.txt``, the ``ref-pandas3`` cell), from this
repository's root, with ``SDM_POSTGRES_TEST_DSN`` naming a Postgres 17
server the runner may create databases on::

    ~/code/sdmc-reference/.venv/bin/python -I \
        contract/tools/schema_rules_record.py composite
    ~/code/sdmc-reference/.venv/bin/python -I \
        contract/tools/schema_rules_record.py key-cast

Written by T2.12's test author before the implementation. Nothing here is
written by hand but the inputs.

``composite``: the key-type pair matrix for composite keys (design,
choice 5: "composite keys follow the same rule column by column"; lesson
19). Each vector is ``contract/bases.yaml``'s ``shop`` changed by its
``setup`` (DuckDB statements) so that a key of two columns has the types of
one ordered pair of :data:`TYPES`, the diagonal included, in one of three
forms:

- ``given.<a>.<b>``: ``orders`` keyed by ``order_id`` (type a) and
  ``line`` (type b, ``order_id % 3``), the key given to
  ``Schema.from_source`` as ``primary_keys={'orders': ['order_id',
  'line']}``;
- ``declared.<a>.<b>``: ``orders`` keyed by ``order_no`` (a) and ``line``
  (b), the key declared by the catalog (``PRIMARY KEY (order_no, line)``);
- ``foreign.<a>.<b>``: ``users`` keyed by ``user_id`` (a) and ``region``
  (b, ``user_id % 3``) and ``orders`` referring to it by a foreign key of
  the same two columns, both declared by the catalog; the entities are
  given as the reference reads a key of type a (integers, or the texts
  ``'0'`` and ``'1'`` for text and, as decimals, a decimal).

Each is written as a case (``case_for``) and run by
``contract/runners/run_python.py --mode python-ref`` on DuckDB, and with
``--database sqlite`` and ``postgres``, where the contract loader
(``databases.py``) loads the tables as its type map says, the keys as
DuckDB declares them. For each database the reference's answer is kept as
:func:`project` reads it: the outcome (class, stage, error class, code), the
key of each table and the dtype and meaning of the vector's key columns as
``Schema.from_source`` described them, the digest of the whole public
schema, and the digest of the requests the scripted endpoint received (the
``payload`` rule's: each parsed request, its request id left out).

``key-cast``: FR P3-8 (migration-plan.md section 8), the key cast's edges,
each a vector of ``contract/tools/key_cast_record.py``'s form (its key
types, its ``record``, its register marks), on DuckDB:

- ``utf8.decimal.1e-7``: ``Decimal('1E-7')`` into a text key;
- ``int64.time.timedelta`` and ``int64.time.time-of-day``: a ``timedelta``
  and a ``time`` of one microsecond into an integer key (which holds 1);
- the float of 2**63 into int64 is ``key-cast-vectors.json``'s
  ``int64.float.2pow63``, recorded there (the reference refuses it at
  6298913c, INVALID_REQUEST; it no longer saturates), and named, not
  repeated, here.

The fourth edge, an ``arrow.bool8`` tag under run-end encoding, is the
host's admission of a column no reference writes (DuckDB never writes
one): ``sdm-host``'s ``tests/t212_bool8_runs.rs`` holds it.

``VECTORS``, ``case_for``, ``run``, ``project`` and ``KEY_CAST`` are
imported by ``contract/tests/test_t212_schema_rules.py``, so ``python-core``
is observed and projected by the same code.
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
from datetime import time, timedelta
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'contract/runners/run_python.py'
TEMPLATE = ROOT / 'contract/cases/s1/rule.composite-key-given.json'
COMPOSITE = ROOT / 'contract/composite-key-vectors.json'
KEY_CAST_OUT = ROOT / 'contract/key-cast-p3-8-vectors.json'
COMPOSITE_FORMAT = 'sdm-composite-key-vectors/1'
KEY_CAST_FORMAT = 'sdm-key-cast-p3-8-vectors/1'
#: How a register entry names the vectors it covers, in its ``refs``: this
#: prefix, ``<vector>@<database>`` (a glob, ``fnmatch``), and the
#: transition it declares, ``<from> -> <to>``, each ``success`` or
#: ``refusal <CODE>`` (``UNNAMED`` for an error with no code).
COMPOSITE_REF = 'contract/composite-key-vectors.json#'
KEY_CAST_REF = 'contract/key-cast-p3-8-vectors.json#'
#: Each vector's case id is this prefix and the vector's id.
CASE_PREFIX = 't212.'
DATABASES = ('duckdb', 'sqlite', 'postgres')
QUESTION = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
ANCHOR = '2024-03-01'


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


# --- the composite key vectors -------------------------------------------------

#: The key column types of the pair matrix: a tag and the DuckDB type the
#: setup gives the column (the contract loader stores each on SQLite and
#: Postgres as its type map says: INTEGER, REAL, TEXT; integer, bigint,
#: numeric(12,4), double precision, text).
TYPES: list[tuple[str, str]] = [
    ('i32', 'INTEGER'),
    ('i64', 'BIGINT'),
    ('d12-4', 'DECIMAL(12, 4)'),
    ('f64', 'DOUBLE'),
    ('text', 'VARCHAR'),
]
FORMS = ('given', 'declared', 'foreign')


def _given(a: str, b: str) -> dict[str, Any]:
    return {
        'setup': [
            f'CREATE OR REPLACE TABLE orders AS SELECT CAST(order_id AS {a}) '
            f'AS order_id, user_id, price, ts, CAST(order_id % 3 AS {b}) AS '
            'line FROM orders'
        ],
        'from_source': {'primary_keys': {'orders': ['order_id', 'line']}},
        'key': ['orders', ['order_id', 'line']],
    }


def _declared(a: str, b: str) -> dict[str, Any]:
    return {
        'setup': [
            f'CREATE TABLE lines (order_no {a}, line {b}, user_id BIGINT, '
            'price DOUBLE, ts TIMESTAMP, PRIMARY KEY (order_no, line))',
            f'INSERT INTO lines SELECT CAST(order_id AS {a}), '
            f'CAST(order_id % 3 AS {b}), user_id, price, ts FROM orders',
            'DROP TABLE orders',
            'ALTER TABLE lines RENAME TO orders',
        ],
        'key': ['orders', ['order_no', 'line']],
    }


def _foreign(a: str, b: str) -> dict[str, Any]:
    return {
        'setup': [
            'CREATE TABLE u0 AS SELECT * FROM users',
            'CREATE TABLE o0 AS SELECT * FROM orders',
            'DROP TABLE orders',
            'DROP TABLE users',
            f'CREATE TABLE users (user_id {a}, region {b}, age INTEGER, '
            'country VARCHAR, PRIMARY KEY (user_id, region))',
            f'INSERT INTO users SELECT CAST(user_id AS {a}), '
            f'CAST(user_id % 3 AS {b}), age, country FROM u0',
            f'CREATE TABLE orders (order_id BIGINT PRIMARY KEY, user_id {a}, '
            f'region {b}, price DOUBLE, ts TIMESTAMP, FOREIGN KEY (user_id, '
            'region) REFERENCES users (user_id, region))',
            f'INSERT INTO orders SELECT order_id, CAST(user_id AS {a}), '
            f'CAST(user_id % 3 AS {b}), price, ts FROM o0 ORDER BY order_id',
            'DROP TABLE o0',
            'DROP TABLE u0',
        ],
        'key': ['users', ['user_id', 'region']],
    }


def _entities(form: str, a: str) -> dict[str, Any]:
    r"""The entities, ``users.user_id`` 0 and 1, as the reference reads a
    key of type ``a``: only ``foreign`` changes ``users.user_id``."""
    if form != 'foreign' or a in ('i32', 'i64', 'f64'):
        return {'indices': [0, 1]}
    if a == 'text':
        return {'indices': ['0', '1']}
    return {'indices': ['0', '1'], 'indices_as': 'decimal'}


def _vectors() -> list[dict[str, Any]]:
    build = {'given': _given, 'declared': _declared, 'foreign': _foreign}
    out = []
    for form in FORMS:
        for a, a_sql in TYPES:
            for b, b_sql in TYPES:
                vector = {
                    'id': f'{form}.{a}.{b}',
                    'form': form,
                    'types': [a, b],
                }
                vector.update(build[form](a_sql, b_sql))
                vector.update(_entities(form, a))
                out.append(vector)
    return out


VECTORS: list[dict[str, Any]] = _vectors()


def case_for(vector: dict[str, Any]) -> dict[str, Any]:
    r"""The vector as a case: ``rule.composite-key-given``'s routes, the
    vector's setup, source arguments and entities, through ``predict``."""
    case = json.loads(TEMPLATE.read_text(encoding='utf-8'))
    case['id'] = CASE_PREFIX + vector['id']
    case['paths'] = []
    case['input'] = {'base': 'shop', 'setup': list(vector['setup'])}
    if 'from_source' in vector:
        case['input']['from_source'] = copy.deepcopy(vector['from_source'])
    case['input']['pql'] = QUESTION
    case['input']['materialize'] = {
        'indices': list(vector['indices']),
        'anchor_time': ANCHOR,
    }
    if 'indices_as' in vector:
        case['input']['materialize']['indices_as'] = vector['indices_as']
    case['until'] = 'predict'
    case['core'] = {'describe_source': case['core']['describe_source']}
    keys = (vector.get('from_source') or {}).get('primary_keys', {})
    case['core']['describe_source']['source']['primary_keys'] = {
        t: {'value': k} for t, k in keys.items()
    }
    case['expect'] = {'outcome': 'success', 'stage': 'predict'}
    case['why'] = f'T2.12 composite key vector {vector["id"]}'
    case['source'] = 'contract/tools/schema_rules_record.py'
    return case


def run(
    mode: str,
    vectors: list[dict[str, Any]],
    database: str,
    directory: Path,
    *,
    reference: Path | None = None,
) -> dict[str, dict[str, Any]]:
    r"""``vectors`` as cases through ``run_python.py --mode mode`` on
    ``database``, in this interpreter: each observation by vector id.
    Raises ``RuntimeError`` with the runner's output when the run fails as
    a whole."""
    cases = directory / f'cases-{database}'
    cases.mkdir(parents=True, exist_ok=True)
    for vector in vectors:
        case = case_for(vector)
        (cases / f'{case["id"]}.json').write_text(
            json.dumps(case, indent=1), encoding='utf-8'
        )
    out = directory / f'{mode}-{database}.jsonl'
    command = [sys.executable, '-I', str(RUNNER), '--mode', mode]
    if reference is not None:
        command += ['--reference', str(reference)]
    if database != 'duckdb':
        command += ['--database', database]
    command += ['--out', str(out), str(cases)]
    done = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        check=False,
        timeout=3600,
    )
    if done.returncode != 0:
        raise RuntimeError(
            f'{mode} on {database} failed:\n{done.stdout}\n{done.stderr}'
        )
    suffix = '' if database == 'duckdb' else f'@{database}'
    observed = {}
    for line in out.read_text(encoding='utf-8').splitlines():
        observation = json.loads(line)
        key = observation['case']
        assert key.startswith(CASE_PREFIX) and key.endswith(suffix), key
        observed[key[len(CASE_PREFIX) : len(key) - len(suffix)]] = observation
    return observed


def _digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def payloads(observation: dict[str, Any]) -> list[Any]:
    r"""Each request the scripted endpoint received, parsed, as the
    ``payload`` rule compares it: its request id, drawn at random for each
    request, left out (``compare-policy.yaml``)."""
    out = []
    for payload in (observation.get('public') or {}).get('payloads') or []:
        parsed = copy.deepcopy(payload.get('parsed'))
        if isinstance(parsed, dict) and isinstance(
            parsed.get('metadata'), dict
        ):
            parsed['metadata'].pop('request_id', None)
        out.append(parsed)
    return out


def outcome(observation: dict[str, Any]) -> list[Any]:
    found = observation['outcome']
    return [found['result'], found['stage'], found['error'], found['code']]


def outcome_class(projected: dict[str, Any]) -> str:
    r"""``success`` or ``refusal <CODE>`` (``UNNAMED`` with no code): a
    register entry's transition is written from one of these to another."""
    result, _, _, code = projected['outcome']
    return 'success' if result == 'success' else f'refusal {code or "UNNAMED"}'


def project(
    observation: dict[str, Any], vector: dict[str, Any]
) -> dict[str, Any]:
    r"""What a host is held to of its answer to ``vector``: its outcome, the
    keys and the vector's key columns as described, the public schema's
    digest, and the requests' digest (null where none was made)."""
    schema = (observation.get('public') or {}).get('schema')
    keys = None
    columns = None
    if schema is not None:
        keys = {t['name']: t['primary_key'] for t in schema['tables']}
        table, names = vector['key']
        columns = {}
        for each in schema['tables']:
            if each['name'] != table:
                continue
            for column in each['columns']:
                if column['name'] in names:
                    columns[column['name']] = [column['dtype'], column['stype']]
    sent = payloads(observation)
    return {
        'outcome': outcome(observation),
        'keys': keys,
        'columns': columns,
        'schema': None if schema is None else _digest(schema),
        'requests': len(sent),
        'payloads': _digest(sent) if sent else None,
    }


def record_composite(reference: Path) -> int:
    answers: dict[str, dict[str, Any]] = {v['id']: {} for v in VECTORS}
    with tempfile.TemporaryDirectory(prefix='t212-record-') as tmp:
        for database in DATABASES:
            observed = run(
                'python-ref', VECTORS, database, Path(tmp), reference=reference
            )
            for vector in VECTORS:
                answers[vector['id']][database] = project(
                    observed[vector['id']], vector
                )
    vectors = []
    for vector in VECTORS:
        entry = {k: v for k, v in vector.items()}
        entry['reference'] = answers[vector['id']]
        vectors.append(entry)
        print(
            vector['id'],
            *(outcome_class(answers[vector['id']][d]) for d in DATABASES),
            file=sys.stderr,
        )
    document = {
        'format': COMPOSITE_FORMAT,
        'recorded_by': 'contract/tools/schema_rules_record.py composite',
        'reference': _reference_commit(reference),
        'cell': 'ref-pandas3',
        'environment': _environment(),
        'query': QUESTION,
        'anchor_time': ANCHOR,
        'register_ref': COMPOSITE_REF,
        'types': [list(t) for t in TYPES],
        'vectors': vectors,
    }
    COMPOSITE.write_text(
        json.dumps(document, indent=1) + '\n', encoding='utf-8'
    )
    print(
        f'{len(vectors)} vectors written to {COMPOSITE.name}', file=sys.stderr
    )
    return 0


# --- FR P3-8's key-cast vectors ------------------------------------------------


def key_cast() -> ModuleType:
    return _module(
        'key_cast_record', ROOT / 'contract/tools/key_cast_record.py'
    )


#: The key type (``key_cast_record``'s key id) and the entities of each
#: vector, by vector name; the vector's id is ``<key>.<name>``.
KEY_CAST: list[tuple[str, str, Any]] = [
    ('utf8', 'decimal.1e-7', lambda: [Decimal('1E-7')]),
    ('int64', 'time.timedelta', lambda: [timedelta(microseconds=1)]),
    ('int64', 'time.time-of-day', lambda: [time(0, 0, 0, 1)]),
]
#: FR P3-8's float of 2**63 into int64, recorded in key-cast-vectors.json.
RECORDED_ELSEWHERE = 'int64.float.2pow63'


def record_key_cast(reference: Path) -> int:
    kc = key_cast()
    keys = {key['id']: key for key in kc.all_keys()}
    vectors = []
    for key_id, name, values in KEY_CAST:
        vector = kc.record(keys[key_id], name, values())
        vectors.append(vector)
        print(vector['id'], vector['reference']['public'], file=sys.stderr)
    kc.derive_marks(vectors)
    document = {
        'format': KEY_CAST_FORMAT,
        'recorded_by': 'contract/tools/schema_rules_record.py key-cast',
        'reference': _reference_commit(reference),
        'cell': 'ref-pandas3',
        'environment': _environment(),
        'query': kc.QUERY,
        'anchor_time': kc.ANCHOR,
        'register_ref': KEY_CAST_REF,
        'recorded_elsewhere': [
            {'id': RECORDED_ELSEWHERE, 'file': 'contract/key-cast-vectors.json'}
        ],
        'keys': [
            {'id': key_id, 'setup': keys[key_id]['setup']}
            for key_id in sorted({k for k, _, _ in KEY_CAST})
        ],
        'vectors': vectors,
    }
    KEY_CAST_OUT.write_text(
        json.dumps(document, indent=1) + '\n', encoding='utf-8'
    )
    print(
        f'{len(vectors)} vectors written to {KEY_CAST_OUT.name}',
        file=sys.stderr,
    )
    return 0


# --- the run ---------------------------------------------------------------------


def _reference_commit(reference: Path) -> str:
    return subprocess.run(
        ['git', '-C', str(reference), 'rev-parse', 'HEAD'],
        capture_output=True,
        text=True,
        check=True,
        stdin=subprocess.DEVNULL,
    ).stdout.strip()


def _environment() -> dict[str, str]:
    import duckdb
    import numpy as np
    import pandas as pd
    import pyarrow as pa

    return {
        'python': platform.python_version(),
        'pandas': pd.__version__,
        'pyarrow': pa.__version__,
        'numpy': np.__version__,
        'duckdb': duckdb.__version__,
    }


def main() -> int:
    reference = Path.home() / 'code/sdmc-reference'
    import sdm_client

    if not Path(sdm_client.__file__).resolve().is_relative_to(reference):
        print("sdm_client is not the reference checkout's", file=sys.stderr)
        return 1
    if sys.argv[1:] == ['composite']:
        if not os.environ.get('SDM_POSTGRES_TEST_DSN'):
            print('SDM_POSTGRES_TEST_DSN names no server', file=sys.stderr)
            return 1
        return record_composite(reference)
    if sys.argv[1:] == ['key-cast']:
        return record_key_cast(reference)
    print('usage: schema_rules_record.py composite|key-cast', file=sys.stderr)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
