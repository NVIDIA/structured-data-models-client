#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Records the reference's handling of empty tables, nulls and refused
questions into ``contract/empty-null-vectors.json`` (T1.32d's vectors).

Run with the reference checkout's interpreter (the environment of
``contract/reference-env.txt``, the ``ref-pandas3`` cell), from this
repository's root::

    ~/code/sdmc-reference/.venv/bin/python -I contract/tools/empty_null_record.py

Written by T1.32d's independent test author before the implementation.
Each vector is ``contract/bases.yaml``'s ``shop`` changed by its ``setup``
(DuckDB statements), and asked ``null.feature-all-null``'s question unless
it names its own ``pql``, for its own ``indices`` (default ``[0, 1]``), of
the tables its ``from_source`` names (default every table):

- ``empty.*``: tables with no rows, the fact table, the entity table, both,
  a third table beside them, named or not, a view with no rows; and one
  row (``edge.*`` also);
- ``null.*``: a column ``f`` of every DuckDB type, every value null, on the
  entity table and on the fact table; null primary keys, null foreign keys,
  null times and null targets, some, all and one entity's;
- ``refuse.*``: what ``refuse.unknown-column`` probes, a question the
  schema cannot answer, by every part of the question that names a column
  or a table, and the refusals of a context the request leaves empty or
  unanswerable;
- ``edge.*``: one row, every entity asked for, an entity with no facts, an
  unknown entity, an empty entity list.

Each vector is a case in ``contract/cases``' format, written to a temporary
directory and run by ``contract/runners/run_python.py --mode python-ref``
through the public calls (``Schema.from_source``, ``materialize``,
``SDMClient.predict`` against the scripted endpoint), and this script keeps
the reference's answer as ``project`` reads it from the observation:

- ``outcome``: success, or the refusal's class, ``StructuredDataError.code``
  and stage (no message text, H17);
- ``schema``: the public schema ``Schema.from_source`` made, as the runner
  records it (each table's columns with dtype and stype, its key, its time
  column, the links), or null when there is none;
- ``warnings``: the warning classes, in order;
- ``payload``: each request the scripted endpoint received: its method and
  path, the SHA-256 of its parsed body as the ``payload`` rule of
  ``contract/compare-policy.yaml`` compares it (the policy's payload
  exclusions masked to their type, each number as its kind and exact
  value), and, per table the body carries rows of, the count of rows and
  of null cells; null when no request was made;
- ``result``: the result's projection as the runner records it.

and, beside it, what DuckDB holds after the setup: each table's row count
and, where the vector adds a column ``f``, its catalog type.

``register`` marks the vectors where the core is expected to differ. It is
derived from the reference's answer, never chosen: the reference fails
with an unnamed exception (no ``StructuredDataError`` code) or with
``INTERNAL_ERROR`` (E66(b)'s class: a named refusal or a success in its
place is a change to register). The mark decides no policy: any vector,
marked or not, may differ from the reference only under an approved
register entry whose ``refs`` name it with exactly the transition it makes,
``REGISTER_REF`` + ``<id or glob> <from> -> <to>``, each side ``success``
or ``refusal <CODE>`` (``outcome_class``; the reference's unnamed
exception is ``refusal UNNAMED``).

Nothing here is written by hand but the inputs: every expected value is
the reference's. ``VECTORS``, ``case_for``, ``run`` and ``project`` are
imported by ``contract/tests/test_t132d_empty_null_vectors.py``, so
``python-core`` is observed and projected by the same code.
"""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import re
import subprocess
import sys
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contract/empty-null-vectors.json'
TEMPLATE = ROOT / 'contract/cases/s1/null.feature-all-null.json'
POLICY = ROOT / 'contract/compare-policy.yaml'
RUNNER = ROOT / 'contract/runners/run_python.py'
FORMAT = 'sdm-empty-null-vectors/1'
#: How a register entry names the vectors it covers, in its ``refs``: this
#: prefix, a vector's id or a glob of ids (``fnmatch``), and the transition
#: it declares, for example
#: ``contract/empty-null-vectors.json#empty.* refusal UNNAMED -> refusal
#: UNSUPPORTED_REQUEST``.
REGISTER_REF = 'contract/empty-null-vectors.json#'
#: Each vector's case id is this prefix and the vector's id.
CASE_PREFIX = 't132d.'
#: The question every vector asks unless it names its own.
QUESTION = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
#: The entities every vector asks for unless it names its own.
INDICES = [0, 1]


def _add(table: str, expression: str) -> list[str]:
    return [
        f'CREATE OR REPLACE TABLE {table} AS SELECT *, {expression} AS f '
        f'FROM {table}'
    ]


def _named(*tables: str) -> dict[str, Any]:
    return {'tables': {name: name for name in tables}}


#: The all-null column types: the vector's id suffix and DuckDB's type.
NULL_TYPES: list[tuple[str, str]] = [
    ('boolean', 'BOOLEAN'),
    ('tinyint', 'TINYINT'),
    ('smallint', 'SMALLINT'),
    ('integer', 'INTEGER'),
    ('bigint', 'BIGINT'),
    ('hugeint', 'HUGEINT'),
    ('utinyint', 'UTINYINT'),
    ('usmallint', 'USMALLINT'),
    ('uinteger', 'UINTEGER'),
    ('ubigint', 'UBIGINT'),
    ('float', 'FLOAT'),
    ('double', 'DOUBLE'),
    ('decimal-18-2', 'DECIMAL(18, 2)'),
    ('decimal-38-10', 'DECIMAL(38, 10)'),
    ('varchar', 'VARCHAR'),
    ('uuid', 'UUID'),
    ('date', 'DATE'),
    ('time', 'TIME'),
    ('timestamp', 'TIMESTAMP'),
    ('timestamp-s', 'TIMESTAMP_S'),
    ('timestamp-ms', 'TIMESTAMP_MS'),
    ('timestamp-ns', 'TIMESTAMP_NS'),
    ('timestamptz', 'TIMESTAMPTZ'),
    ('blob', 'BLOB'),
    ('interval', 'INTERVAL'),
    ('list-integer', 'INTEGER[]'),
]
#: The all-null column types also added to the fact table.
FACT_NULL_TYPES = ['integer', 'double', 'varchar', 'timestamp', 'boolean']


def _null_columns() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for suffix, sql_type in NULL_TYPES:
        setup = _add('users', f'NULL::{sql_type}')
        if sql_type == 'TIMESTAMPTZ':
            setup = ["SET GLOBAL TimeZone = 'UTC'", *setup]
        out.append({'id': f'null.{suffix}', 'table': 'users', 'setup': setup})
    for suffix, sql_type in NULL_TYPES:
        if suffix in FACT_NULL_TYPES:
            out.append(
                {
                    'id': f'null.orders-{suffix}',
                    'table': 'orders',
                    'setup': _add('orders', f'NULL::{sql_type}'),
                }
            )
    out += [
        # DuckDB's own type for a bare NULL, and a column null but for one
        # value.
        {
            'id': 'null.untyped',
            'table': 'users',
            'setup': _add('users', 'NULL'),
        },
        {
            'id': 'null.all-but-one-integer',
            'table': 'users',
            'setup': _add('users', 'CASE WHEN user_id = 0 THEN 7 END::INTEGER'),
        },
        {
            'id': 'null.all-but-one-varchar',
            'table': 'users',
            'setup': _add(
                'users', "CASE WHEN user_id = 0 THEN 'x' END::VARCHAR"
            ),
        },
        # Every column of the entity table but its key null.
        {
            'id': 'null.every-feature',
            'table': None,
            'setup': ['UPDATE users SET age = NULL, country = NULL'],
        },
    ]
    return out


#: The vectors: each its id, the table its column ``f`` is on (or null),
#: and its setup, after ``shop``'s statements; and, where it varies them,
#: its question (``pql``), its entities (``indices``) and the tables it
#: describes (``from_source``, as ``Schema.from_source``'s keywords).
VECTORS: list[dict[str, Any]] = [
    # --- empty tables -----------------------------------------------------
    {'id': 'empty.orders', 'table': None, 'setup': ['DELETE FROM orders']},
    {'id': 'empty.users', 'table': None, 'setup': ['DELETE FROM users']},
    {
        'id': 'empty.both',
        'table': None,
        'setup': ['DELETE FROM orders', 'DELETE FROM users'],
    },
    {
        'id': 'empty.third-table',
        'table': None,
        'setup': ['CREATE TABLE extra (extra_id BIGINT, v INTEGER)'],
    },
    {
        'id': 'empty.third-table-not-named',
        'table': None,
        'setup': ['CREATE TABLE extra (extra_id BIGINT, v INTEGER)'],
        'from_source': _named('users', 'orders'),
    },
    {
        'id': 'empty.third-table-named',
        'table': None,
        'setup': ['CREATE TABLE extra (extra_id BIGINT, v INTEGER)'],
        'from_source': _named('users', 'orders', 'extra'),
    },
    {
        'id': 'empty.orders-named',
        'table': None,
        'setup': ['DELETE FROM orders'],
        'from_source': _named('users', 'orders'),
    },
    {
        'id': 'empty.view',
        'table': None,
        'setup': ['CREATE VIEW extra AS SELECT * FROM orders WHERE false'],
    },
    {
        'id': 'empty.orders-filtered-out',
        'table': None,
        'setup': [
            'CREATE OR REPLACE TABLE orders AS SELECT * FROM orders WHERE false'
        ],
    },
    # --- one row ----------------------------------------------------------
    {
        'id': 'edge.one-order',
        'table': None,
        'setup': ['DELETE FROM orders WHERE order_id > 0'],
    },
    {
        'id': 'edge.one-user',
        'table': None,
        'setup': [
            'DELETE FROM users WHERE user_id > 0',
            'DELETE FROM orders WHERE user_id > 0',
        ],
        'indices': [0],
    },
    {
        'id': 'edge.one-user-two-asked',
        'table': None,
        'setup': [
            'DELETE FROM users WHERE user_id > 0',
            'DELETE FROM orders WHERE user_id > 0',
        ],
    },
    {
        'id': 'edge.one-user-of-two-with-orders',
        'table': None,
        'setup': [
            'DELETE FROM users WHERE user_id > 1',
            'DELETE FROM orders WHERE user_id > 0',
        ],
    },
    # --- the entities -----------------------------------------------------
    {'id': 'edge.one-entity', 'table': None, 'setup': [], 'indices': [0]},
    {
        'id': 'edge.every-entity',
        'table': None,
        'setup': [],
        'indices': [0, 1, 2, 3, 4, 5, 6, 7],
    },
    {
        'id': 'edge.entity-without-facts',
        'table': None,
        'setup': ["INSERT INTO users VALUES (99, 50, 'US')"],
        'indices': [0, 99],
    },
    {
        'id': 'edge.entities-without-facts',
        'table': None,
        'setup': ['DELETE FROM orders WHERE user_id IN (0, 1)'],
    },
    {
        'id': 'edge.unknown-entity',
        'table': None,
        'setup': [],
        'indices': [0, 12345],
    },
    {
        'id': 'edge.only-unknown-entities',
        'table': None,
        'setup': [],
        'indices': [12345, 12346],
    },
    {'id': 'edge.no-entities', 'table': None, 'setup': [], 'indices': []},
    {
        'id': 'edge.no-entities-empty-orders',
        'table': None,
        'setup': ['DELETE FROM orders'],
        'indices': [],
    },
    # --- all-null columns of each type ------------------------------------
    *_null_columns(),
    # --- null keys --------------------------------------------------------
    {
        'id': 'null.key-one',
        'table': None,
        'setup': ['UPDATE users SET user_id = NULL WHERE user_id = 7'],
    },
    {
        'id': 'null.key-asked-for',
        'table': None,
        'setup': ['UPDATE users SET user_id = NULL WHERE user_id = 1'],
    },
    {
        'id': 'null.key-all',
        'table': None,
        'setup': ['UPDATE users SET user_id = NULL'],
    },
    {
        'id': 'null.key-fact-some',
        'table': None,
        'setup': ['UPDATE orders SET order_id = NULL WHERE order_id % 10 = 0'],
    },
    {
        'id': 'null.key-named-some',
        'table': None,
        'setup': ['UPDATE users SET user_id = NULL WHERE user_id = 7'],
        'from_source': {'primary_keys': {'users': 'user_id'}},
    },
    # --- null foreign keys ------------------------------------------------
    {
        'id': 'null.fk-some',
        'table': None,
        'setup': ['UPDATE orders SET user_id = NULL WHERE order_id % 10 = 0'],
    },
    {
        'id': 'null.fk-most',
        'table': None,
        'setup': ['UPDATE orders SET user_id = NULL WHERE order_id % 10 <> 0'],
    },
    {
        'id': 'null.fk-all',
        'table': None,
        'setup': ['UPDATE orders SET user_id = NULL'],
    },
    {
        'id': 'null.fk-of-an-asked-entity',
        'table': None,
        'setup': ['UPDATE orders SET user_id = NULL WHERE user_id = 0'],
    },
    {
        'id': 'null.fk-in-the-window',
        'table': None,
        'setup': [
            "UPDATE orders SET user_id = NULL WHERE ts >= TIMESTAMP '2023-12-01'"
        ],
    },
    # --- null times -------------------------------------------------------
    {
        'id': 'null.time-some',
        'table': None,
        'setup': ['UPDATE orders SET ts = NULL WHERE order_id % 10 = 0'],
    },
    {
        'id': 'null.time-most',
        'table': None,
        'setup': ['UPDATE orders SET ts = NULL WHERE order_id % 10 <> 0'],
    },
    {
        'id': 'null.time-all-but-one',
        'table': None,
        'setup': ['UPDATE orders SET ts = NULL WHERE order_id > 0'],
    },
    {
        'id': 'null.time-all',
        'table': None,
        'setup': ['UPDATE orders SET ts = NULL'],
    },
    {
        'id': 'null.time-of-an-asked-entity',
        'table': None,
        'setup': ['UPDATE orders SET ts = NULL WHERE user_id = 0'],
    },
    {
        'id': 'null.time-in-the-window',
        'table': None,
        'setup': [
            "UPDATE orders SET ts = NULL WHERE ts >= TIMESTAMP '2023-12-01'"
        ],
    },
    {
        'id': 'null.time-date',
        'table': None,
        'setup': [
            'CREATE OR REPLACE TABLE orders AS SELECT * REPLACE '
            '(CASE WHEN order_id % 10 = 0 THEN NULL ELSE ts::DATE END AS ts) '
            'FROM orders'
        ],
    },
    {
        'id': 'null.time-timestamptz',
        'table': None,
        'setup': [
            "SET GLOBAL TimeZone = 'UTC'",
            'CREATE OR REPLACE TABLE orders AS SELECT * REPLACE '
            '(CASE WHEN order_id % 10 = 0 THEN NULL ELSE ts::TIMESTAMPTZ END '
            'AS ts) FROM orders',
        ],
    },
    {
        'id': 'null.time-named',
        'table': None,
        'setup': ['UPDATE orders SET ts = NULL WHERE order_id % 10 = 0'],
        'from_source': {'time_columns': {'orders': 'ts'}},
    },
    # --- null targets -----------------------------------------------------
    {
        'id': 'null.target-all',
        'table': None,
        'setup': ['UPDATE orders SET price = NULL'],
    },
    {
        'id': 'null.target-some',
        'table': None,
        'setup': ['UPDATE orders SET price = NULL WHERE order_id % 3 = 0'],
    },
    {
        'id': 'null.target-of-an-asked-entity',
        'table': None,
        'setup': ['UPDATE orders SET price = NULL WHERE user_id = 0'],
    },
    {
        'id': 'null.target-all-count',
        'table': None,
        'setup': ['UPDATE orders SET price = NULL'],
        'pql': 'PREDICT COUNT(orders.price, 0, 30, days) FOR EACH users.user_id',
    },
    {
        'id': 'null.target-all-avg',
        'table': None,
        'setup': ['UPDATE orders SET price = NULL'],
        'pql': 'PREDICT AVG(orders.price, 0, 30, days) FOR EACH users.user_id',
    },
    {
        'id': 'null.target-all-classification',
        'table': None,
        'setup': ['UPDATE orders SET price = NULL'],
        'pql': (
            'PREDICT SUM(orders.price, 0, 30, days) > 0 FOR EACH users.user_id'
        ),
    },
    {
        'id': 'null.filter-column-all',
        'table': None,
        'setup': ['UPDATE users SET age = NULL'],
        'pql': (
            'PREDICT COUNT(orders.* WHERE users.age > 30, 0, 30, days) '
            'FOR EACH users.user_id'
        ),
    },
    # --- refused questions ------------------------------------------------
    {
        'id': 'refuse.unknown-target-column',
        'table': None,
        'setup': [],
        'pql': 'PREDICT SUM(orders.cost, 0, 30, days) FOR EACH users.user_id',
    },
    {
        'id': 'refuse.unknown-target-table',
        'table': None,
        'setup': [],
        'pql': 'PREDICT SUM(payments.amount, 0, 30, days) FOR EACH users.user_id',
    },
    {
        'id': 'refuse.unknown-entity-column',
        'table': None,
        'setup': [],
        'pql': 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.nope',
    },
    {
        'id': 'refuse.unknown-entity-table',
        'table': None,
        'setup': [],
        'pql': 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH people.user_id',
    },
    {
        'id': 'refuse.unknown-count-column',
        'table': None,
        'setup': [],
        'pql': 'PREDICT COUNT(orders.cost, 0, 30, days) FOR EACH users.user_id',
    },
    {
        'id': 'refuse.unknown-filter-column',
        'table': None,
        'setup': [],
        'pql': (
            'PREDICT COUNT(orders.* WHERE orders.cost > 1, 0, 30, days) '
            'FOR EACH users.user_id'
        ),
    },
    {
        'id': 'refuse.unknown-entity-filter-column',
        'table': None,
        'setup': [],
        'pql': (
            'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id '
            'WHERE users.nope > 1'
        ),
    },
    {
        'id': 'refuse.column-of-another-case',
        'table': None,
        'setup': [],
        'pql': 'PREDICT SUM(orders.PRICE, 0, 30, days) FOR EACH users.user_id',
    },
    {
        'id': 'refuse.column-left-out',
        'table': 'orders',
        'setup': _add('orders', "('x' || order_id)::BLOB"),
        'pql': 'PREDICT SUM(orders.f, 0, 30, days) FOR EACH users.user_id',
    },
    {
        'id': 'refuse.target-of-an-empty-table',
        'table': None,
        'setup': ['CREATE TABLE extra (extra_id BIGINT, v INTEGER)'],
        'from_source': _named('users', 'orders'),
        'pql': 'PREDICT SUM(extra.v, 0, 30, days) FOR EACH users.user_id',
    },
    {
        'id': 'null.target-new-column',
        'table': 'orders',
        'setup': _add('orders', 'NULL::DOUBLE'),
        'pql': 'PREDICT SUM(orders.f, 0, 30, days) FOR EACH users.user_id',
    },
    {
        'id': 'refuse.one-class-in-context',
        'table': None,
        'setup': [],
        'pql': (
            'PREDICT COUNT(orders.*, 0, 30, days) > 1000 FOR EACH users.user_id'
        ),
    },
    {
        'id': 'edge.every-entity-classification',
        'table': None,
        'setup': [],
        'pql': (
            'PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id'
        ),
        'indices': [0, 1, 2, 3, 4, 5, 6, 7],
    },
]


# --- the cases -----------------------------------------------------------------


def core_indices(indices: list[int]) -> list[dict[str, str]]:
    r"""The entities as the core's contract scalars (the case's
    ``core.materialize.indices``)."""
    return [{'t': 'i64', 'v': str(index)} for index in indices]


def core_source(from_source: dict[str, Any] | None) -> dict[str, Any]:
    r"""``Schema.from_source``'s keywords as the core's source spec (the
    case's ``core.describe_source.source``), as the client's bridge builds
    it: each key and time column as the contract's ``Given`` (T1.32h)."""
    given = from_source or {}
    return {
        'tables': [
            {'name': name, 'source': source}
            for name, source in (given.get('tables') or {}).items()
        ],
        'primary_keys': {
            table: {'value': [key] if isinstance(key, str) else list(key)}
            for table, key in (given.get('primary_keys') or {}).items()
        },
        'time_columns': {
            table: {'value': column}
            for table, column in (given.get('time_columns') or {}).items()
        },
    }


def core_routes(vector: dict[str, Any]) -> list[str]:
    r"""Every DuckDB route of the three public calls the vector makes, as
    the candidate's routing names them: ``Schema.from_source``'s with
    tables named where the vector names them (T1.24), with every table
    otherwise; ``materialize``'s and ``predict``'s."""
    from sdm_client import _routing

    named = bool((vector.get('from_source') or {}).get('tables'))
    return list(
        (
            _routing.FROM_SOURCE_DUCKDB_TABLES
            if named
            else _routing.FROM_SOURCE_DUCKDB
        )
        + _routing.MATERIALIZE_DUCKDB
        + _routing.PREDICT_DUCKDB
    )


def case_for(vector: dict[str, Any], core: bool = False) -> dict[str, Any]:
    r"""The vector as a case: ``null.feature-all-null``'s, its setup,
    question, entities and tables the vector's; with ``core``, forced onto
    the core's routes (``core_routes``), and otherwise on none (the
    runner's own choice, as the reference has)."""
    template = json.loads(TEMPLATE.read_text(encoding='utf-8'))
    case = copy.deepcopy(template)
    case['id'] = CASE_PREFIX + vector['id']
    case['status'] = 'diagnostic'
    case['routes'] = core_routes(vector) if core else []
    case['cells'] = []
    case['input']['setup'] = list(vector['setup'])
    if not case['input']['setup']:
        del case['input']['setup']
    case['input']['pql'] = vector.get('pql', QUESTION)
    indices = list(vector.get('indices', INDICES))
    case['input']['materialize']['indices'] = indices
    case['core']['describe_source']['source'] = core_source(
        vector.get('from_source')
    )
    if 'from_source' in vector:
        case['input']['from_source'] = copy.deepcopy(vector['from_source'])
    if indices:
        case['core']['materialize']['indices'] = core_indices(indices)
    else:
        # The case format's core input holds at least one entity: an empty
        # list is carried to the core by the public call alone.
        del case['core']['materialize']
        del case['core']['predict']
    case['expect'] = {'outcome': 'success', 'stage': 'predict'}
    case['why'] = f'T1.32d vector {vector["id"]}'
    case['source'] = 'contract/tools/empty_null_record.py'
    case.pop('register', None)
    return case


def run(
    mode: str,
    vectors: list[dict[str, Any]],
    directory: Path,
    *,
    core: bool = False,
    reference: Path | None = None,
    capture: bool = True,
) -> dict[str, dict[str, Any]]:
    r"""``vectors`` as cases through ``run_python.py --mode mode``, in this
    interpreter, from ``directory``, forced onto the core with ``core``:
    each observation by vector id."""
    cases = directory / 'cases'
    cases.mkdir(parents=True, exist_ok=True)
    for vector in vectors:
        case = case_for(vector, core)
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

_INTEGER = re.compile(r'-?[0-9]+')


def exclusions() -> list[list[str]]:
    r"""The paths the ``payload`` rule of ``contract/compare-policy.yaml``
    leaves out of a payload (each value masked, its presence and type
    kept), read from the policy's text."""
    paths: list[list[str]] = []
    block: dict[str, str] = {}
    lines = POLICY.read_text(encoding='utf-8').splitlines()
    inside = False
    for line in lines:
        if line.startswith('exclusions:'):
            inside = True
            continue
        if inside and line and not line.startswith((' ', '#')):
            break
        if not inside:
            continue
        stripped = line.strip()
        if stripped.startswith('- id:'):
            block = {}
        if ':' in stripped and not stripped.startswith('#'):
            key, _, value = stripped.lstrip('- ').partition(':')
            block[key.strip()] = value.strip()
            if key.strip() == 'path' and block.get('in') == 'payload':
                paths.append(
                    [p.strip() for p in value.strip().strip('[]').split(',')]
                )
    return paths


def _masked(value: Any, path: list[str]) -> Any:
    if not path or not isinstance(value, dict) or path[0] not in value:
        return value
    out = dict(value)
    if len(path) == 1:
        out[path[0]] = f'<{type(value[path[0]]).__name__}>'
    else:
        out[path[0]] = _masked(value[path[0]], path[1:])
    return out


def number_form(text: str) -> list[Any]:
    r"""A number token as the ``payload`` rule compares it: its kind (an
    integer or a float token), its sign, and its exact value as significant
    digits and a power of ten, so ``1e-05``, ``1e-5`` and ``0.00001``
    agree, ``1`` and ``1.0`` do not (their kinds), nor ``-0.0`` and ``0.0``
    (their signs). Read without rounding: no context, no float."""
    kind = 'integer' if _INTEGER.fullmatch(text) else 'float'
    sign, digits, exponent = Decimal(text).as_tuple()
    if not isinstance(exponent, int):
        return [kind, bool(sign), text.lower().lstrip('+-'), 0]
    significant = ''.join(str(d) for d in digits).lstrip('0')
    if not significant:
        return [kind, bool(sign), '0', 0]
    stripped = significant.rstrip('0')
    return [
        kind,
        bool(sign),
        stripped,
        exponent + len(significant) - len(stripped),
    ]


def _canonical(value: Any) -> Any:
    r"""A parsed payload with each number (``{"number": text}``) as
    ``number_form`` reads it."""
    if isinstance(value, dict):
        if set(value) == {'number'} and isinstance(value['number'], str):
            return {'number': number_form(value['number'])}
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


def digest(parsed: Any) -> str:
    r"""The SHA-256 of a parsed payload as the ``payload`` rule compares
    it: the policy's exclusions masked to their type, numbers as
    ``number_form`` reads them, written as JSON with sorted keys, no
    spaces and UTF-8 text unescaped (``sdm-testkit``'s
    ``tests/t132d_empty_null_vectors.rs`` writes the same bytes)."""
    for path in exclusions():
        parsed = _masked(parsed, path)
    canonical = json.dumps(
        _canonical(parsed),
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def tables_with_rows(value: Any, at: str = '') -> dict[str, Any]:
    r"""Each object of a body holding ``columns`` and ``rows`` (a table the
    body carries), by its path: its row count and null cells."""
    found: dict[str, Any] = {}
    if isinstance(value, dict):
        rows = value.get('rows')
        if isinstance(rows, list) and isinstance(value.get('columns'), list):
            found[at or '.'] = {
                'rows': len(rows),
                'nulls': sum(
                    cell is None
                    for row in rows
                    if isinstance(row, list)
                    for cell in row
                ),
            }
        for key, item in value.items():
            found.update(tables_with_rows(item, f'{at}.{key}' if at else key))
    return found


def payload_projection(payloads: Any) -> Any:
    r"""Each request the endpoint received: method, path, the digest of its
    parsed body as the ``payload`` rule compares it, and the tables it
    carries rows of; null when no request was made."""
    if not payloads:
        return None
    out = []
    for each in payloads:
        out.append(
            {
                'method': each.get('method'),
                'path': each.get('path'),
                'digest': digest(each.get('parsed')),
                'tables': tables_with_rows(each.get('parsed')),
            }
        )
    return out


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
        'schema': public.get('schema'),
        'warnings': list(host.get('warnings') or []),
        'payload': payload_projection(public.get('payloads')),
        'result': public.get('result'),
    }


#: The class of a refusal with no ``StructuredDataError`` code (the
#: reference's unnamed ``RuntimeError`` or ``ValueError``).
UNNAMED = 'UNNAMED'


def outcome_class(outcome: dict[str, Any]) -> str:
    r"""``success`` or ``refusal <CODE>`` of an answer's ``outcome`` (its
    ``result`` and ``code``): a register entry's transition is written from
    one of these to another."""
    if outcome['result'] == 'success':
        return 'success'
    return f'refusal {outcome.get("code") or UNNAMED}'


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


# --- what DuckDB holds ---------------------------------------------------------


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


def duckdb_holds(vector: dict[str, Any]) -> dict[str, Any]:
    r"""Each table's (and view's) row count after the setup and, where the
    vector adds ``f``, its catalog type."""
    import duckdb

    connection = duckdb.connect()
    try:
        for statement in shop() + vector['setup']:
            connection.execute(statement)
        names = [
            name
            for (name,) in connection.execute(
                'SELECT table_name FROM information_schema.tables '
                'ORDER BY table_name'
            ).fetchall()
        ]
        rows = {
            name: connection.execute(
                f'SELECT count(*) FROM "{name}"'
            ).fetchone()[0]
            for name in names
        }
        f_type = None
        if vector['table'] is not None:
            found = connection.execute(
                'SELECT data_type FROM information_schema.columns '
                "WHERE table_name = ? AND column_name = 'f'",
                [vector['table']],
            ).fetchone()
            f_type = found[0] if found else None
        return {'rows': rows, 'f_type': f_type}
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
    with tempfile.TemporaryDirectory(prefix='t132d-record-') as tmp:
        observed = run('python-ref', VECTORS, Path(tmp), reference=reference)
    vectors = []
    for vector in VECTORS:
        answer = project(observed[vector['id']], vector)
        why = register(answer)
        entry = {key: copy.deepcopy(value) for key, value in vector.items()}
        entry['duckdb'] = duckdb_holds(vector)
        entry['reference'] = answer
        entry['register'] = {'required': bool(why), 'why': why}
        vectors.append(entry)
        outcome = answer['outcome']
        print(
            vector['id'],
            outcome['result'],
            outcome['stage'],
            outcome['error'] or '',
            outcome['code'] or '',
            answer['warnings'],
            '(register)' if why else '',
            file=sys.stderr,
        )
    document = {
        'format': FORMAT,
        'recorded_by': 'contract/tools/empty_null_record.py',
        'reference': commit,
        'cell': 'ref-pandas3',
        'environment': {
            'python': platform.python_version(),
            'pandas': pd.__version__,
            'pyarrow': pa.__version__,
            'numpy': np.__version__,
            'duckdb': duckdb.__version__,
        },
        'template': 'contract/cases/s1/null.feature-all-null.json',
        'question': QUESTION,
        'indices': INDICES,
        'register_ref': REGISTER_REF,
        'payload_exclusions': exclusions(),
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
