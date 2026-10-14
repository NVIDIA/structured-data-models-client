#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Writes ``contract/given-vectors.json``: T1.32h's vectors of the keys and
time columns ``describe_source`` is given (register entry P1B-12), the
reference's answer recorded beside each that a caller can give.

Run with the reference checkout's interpreter (the environment of
``contract/reference-env.txt``, the ``ref-pandas3`` cell), from this
repository's root::

    ~/code/sdmc-reference/.venv/bin/python -I contract/tools/given_record.py

Written by T1.32h's independent test author before the implementation,
from the design of 2026-10-13 (``p1b12-design-2026-10-13.md``, revised
after Codex's review), section 4's table and section 6's plan. Every vector
is over ``contract/bases.yaml``'s ``shop``, every table described (no
``tables``), and carries:

- ``spec``: the ``SourceSpec`` the core is given, as JSON: from contract 23
  each key ``{"value": [names]}``, ``{"value": null}`` or ``"unreadable"``,
  each time column ``{"value": name}``, ``{"value": null}`` or
  ``"unreadable"`` (``Given``). A ``wire`` vector's spec is malformed on
  purpose, a host's bug: what no host may write.
- ``core``: the core's answer, from the design, never measured:
  ``outcome`` (``success`` or ``refusal``); ``at`` (``start``: refused
  before the run exists, nothing fetched; ``run``: after the source's
  tables were read, at least one fetch answered); the ``code``; the
  error's ``fields``, exactly (a value ``"*"`` is a field present with any
  value: the decoder's ``line`` and ``column``); and for a success, the
  key and time column each table is described with.
- ``from_source``: for a vector a caller can give through the public
  call, ``Schema.from_source``'s keywords as JSON (a ``case`` vector).
  The reference's answer to that call is recorded as ``reference``
  (``result``, ``stage``, the exception's class and
  ``StructuredDataError.code``, no message, H17; and the key and time
  column of each table it described), through
  ``contract/runners/run_python.py --mode python-ref``, the vector made a
  case in ``contract/cases``' format by :func:`case_for`. Other vectors
  carry ``reference: null`` and say why.
- ``register``: where the core's class of answer (``success`` or
  ``refusal <CODE>``) differs from the reference's, what covers it: an
  ``entry`` with the ``transition`` it must declare in its ``refs``
  (``contract/given-vectors.json#<id> <from> -> <to>``, the entry
  approved), or ``phase2`` naming the rule of decision E92's class (a
  composite key stays refused by name until Phase 2). Null otherwise.
- ``owner``: ``D3`` or ``D4`` for a vector of a decision still the
  owner's, which its own commit adds and can drop; null otherwise.

``edit`` vectors are ``edit_schema``'s: the same value given to the
schema's setter after the source is described (``set``), the edit the core
is given (``edit``), and ``pair``, the ``describe_source`` vector whose
refusal it shares (code, message and ``rule``; the ``field`` stays the
operation's own, ``primary_key``/``time_column`` against
``primary_keys``/``time_columns``) or whose ``null`` it reads otherwise.

``VECTORS``, ``case_for``, ``run`` and ``project`` are imported by
``contract/tests/test_t132h_given_vectors.py``, so ``python-core`` is run
and read by the same code.
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
OUT = ROOT / 'contract/given-vectors.json'
RUNNER = ROOT / 'contract/runners/run_python.py'
FORMAT = 'sdm-given-vectors/1'
REGISTER_REF = 'contract/given-vectors.json#'
CASE_PREFIX = 't132h.'
QUESTION = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
PROVIDER = {'provider': 'sql', 'dialect': 'duckdb'}

U = 'unreadable'


def V(value: Any) -> dict[str, Any]:
    r"""``Given::Value``, as the contract writes it."""
    return {'value': value}


def spec(
    primary_keys: dict[str, Any] | None = None,
    time_columns: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        'tables': [],
        'primary_keys': primary_keys or {},
        'time_columns': time_columns or {},
    }


def refused(code: str, at: str = 'run', **fields: Any) -> dict[str, Any]:
    return {'outcome': 'refusal', 'at': at, 'code': code, 'fields': fields}


def described(
    orders: tuple[list[str] | None, str | None],
    users: tuple[list[str] | None, str | None] = (['user_id'], None),
) -> dict[str, Any]:
    r"""A success: each table's key (its columns) and time column."""
    return {
        'outcome': 'success',
        'at': 'run',
        'tables': {
            'orders': {'primary_key': orders[0], 'time_column': orders[1]},
            'users': {'primary_key': users[0], 'time_column': users[1]},
        },
    }


#: The decoder's refusal of an attachment outside the contract's shape:
#: before the run exists, naming the attachment; ``category`` (``data`` or
#: ``syntax``, as serde classifies the fault), ``line`` and ``column`` are
#: serde's, any value.
MALFORMED = refused(
    'INVALID_MESSAGE',
    at='start',
    attachment=1,
    category='*',
    line='*',
    column='*',
)
NOT_NAMES = {
    'field': 'primary_keys',
    'table': 'orders',
    'rule': 'key_not_names',
}
NOT_A_NAME = {
    'field': 'time_columns',
    'table': 'orders',
    'rule': 'time_column_not_a_name',
}
EMPTY_KEY = {'field': 'primary_keys', 'table': 'orders'}
COMPOSITE = refused(
    'UNSUPPORTED_REQUEST', operation='describe_source', rule='composite_key'
)
ORDERS_INFERRED = (['order_id'], 'ts')


def case(
    id: str,
    from_source: dict[str, Any],
    given: dict[str, Any],
    core: dict[str, Any],
    register: dict[str, Any] | None = None,
    owner: str | None = None,
) -> dict[str, Any]:
    return {
        'id': id,
        'kind': 'case',
        'owner': owner,
        'from_source': from_source,
        'spec': given,
        'core': core,
        'register': register,
    }


def wire(id: str, given: dict[str, Any], why: str) -> dict[str, Any]:
    return {
        'id': id,
        'kind': 'wire',
        'owner': None,
        'spec': given,
        'core': MALFORMED,
        'register': None,
        'reference_why': why,
    }


HOST_BUG = (
    'a malformed SourceSpec, what no host may write: no caller value is '
    'written this way'
)

VECTORS: list[dict[str, Any]] = [
    # --- keys a caller can give, as section 4's rows 1-16 ---------------
    case(
        'key.name',
        {'primary_keys': {'orders': 'order_id'}},
        spec({'orders': V(['order_id'])}),
        described(ORDERS_INFERRED),
    ),
    case(
        'key.list',
        {'primary_keys': {'orders': ['order_id']}},
        spec({'orders': V(['order_id'])}),
        described(ORDERS_INFERRED),
    ),
    case(
        'key.null',
        {'primary_keys': {'orders': None}},
        spec({'orders': V(None)}),
        refused('INVALID_REQUEST', **EMPTY_KEY),
    ),
    case(
        'key.int',
        {'primary_keys': {'orders': 3}},
        spec({'orders': U}),
        refused('INVALID_REQUEST', **NOT_NAMES),
    ),
    case(
        'key.float',
        {'primary_keys': {'orders': 1.5}},
        spec({'orders': U}),
        refused('INVALID_REQUEST', **NOT_NAMES),
    ),
    case(
        'key.bool',
        {'primary_keys': {'orders': True}},
        spec({'orders': U}),
        refused('INVALID_REQUEST', **NOT_NAMES),
    ),
    case(
        'key.nested',
        {'primary_keys': {'orders': [['order_id']]}},
        spec({'orders': U}),
        refused('INVALID_REQUEST', **NOT_NAMES),
    ),
    case(
        'key.mapping',
        {'primary_keys': {'orders': {'order_id': 1}}},
        spec({'orders': U}),
        refused('INVALID_REQUEST', **NOT_NAMES),
    ),
    case(
        'key.empty',
        {'primary_keys': {'orders': []}},
        spec({'orders': V([])}),
        refused('INVALID_REQUEST', **EMPTY_KEY),
    ),
    case(
        'key.member-int',
        {'primary_keys': {'orders': ['order_id', 3]}},
        spec({'orders': U}),
        refused('INVALID_REQUEST', **NOT_NAMES),
    ),
    case(
        'key.member-null',
        {'primary_keys': {'orders': ['order_id', None]}},
        spec({'orders': U}),
        refused('INVALID_REQUEST', **NOT_NAMES),
    ),
    case(
        'key.twice',
        {'primary_keys': {'orders': ['order_id', 'order_id']}},
        spec({'orders': V(['order_id', 'order_id'])}),
        COMPOSITE,
        {'phase2': 'composite_key'},
    ),
    case(
        'key.composite',
        {'primary_keys': {'orders': ['order_id', 'user_id']}},
        spec({'orders': V(['order_id', 'user_id'])}),
        COMPOSITE,
        {'phase2': 'composite_key'},
    ),
    case(
        'key.missing-column',
        {'primary_keys': {'orders': 'nope'}},
        spec({'orders': V(['nope'])}),
        refused('INVALID_REQUEST', **EMPTY_KEY, column='nope'),
    ),
    # --- time columns a caller can give, rows 18-22 ----------------------
    case(
        'time.name',
        {'time_columns': {'orders': 'ts'}},
        spec(time_columns={'orders': V('ts')}),
        described(ORDERS_INFERRED),
    ),
    case(
        'time.null',
        {'time_columns': {'orders': None}},
        spec(time_columns={'orders': V(None)}),
        described((['order_id'], None)),
    ),
    case(
        'time.int',
        {'time_columns': {'orders': 3}},
        spec(time_columns={'orders': U}),
        refused('INVALID_REQUEST', **NOT_A_NAME),
    ),
    case(
        'time.float',
        {'time_columns': {'orders': 1.5}},
        spec(time_columns={'orders': U}),
        refused('INVALID_REQUEST', **NOT_A_NAME),
    ),
    case(
        'time.bool',
        {'time_columns': {'orders': True}},
        spec(time_columns={'orders': U}),
        refused('INVALID_REQUEST', **NOT_A_NAME),
    ),
    case(
        'time.list',
        {'time_columns': {'orders': ['ts']}},
        spec(time_columns={'orders': U}),
        refused('INVALID_REQUEST', **NOT_A_NAME),
        {
            'entry': 'P1B-12',
            'transition': 'refusal UNNAMED -> refusal INVALID_REQUEST',
        },
    ),
    case(
        'time.mapping',
        {'time_columns': {'orders': {'ts': 1}}},
        spec(time_columns={'orders': U}),
        refused('INVALID_REQUEST', **NOT_A_NAME),
        {
            'entry': 'P1B-12',
            'transition': 'refusal UNNAMED -> refusal INVALID_REQUEST',
        },
    ),
    case(
        'time.missing-column',
        {'time_columns': {'orders': 'nope'}},
        spec(time_columns={'orders': V('nope')}),
        refused(
            'INVALID_REQUEST',
            field='time_columns',
            table='orders',
            column='nope',
        ),
    ),
    # --- the ordering promise, section 2, steps 3 to 6 --------------------
    case(
        'order.key-table-first',
        {'primary_keys': {'nope': 3}},
        spec({'nope': U}),
        refused('INVALID_REQUEST', field='primary_keys', table='nope'),
    ),
    case(
        'order.time-table-first',
        {'time_columns': {'nope': 3}},
        spec(time_columns={'nope': U}),
        refused('INVALID_REQUEST', field='time_columns', table='nope'),
    ),
    case(
        'order.keys-before-times',
        {'primary_keys': {'orders': 3}, 'time_columns': {'orders': 3}},
        spec({'orders': U}, {'orders': U}),
        refused('INVALID_REQUEST', **NOT_NAMES),
    ),
    case(
        'order.key-null-before-time',
        {'primary_keys': {'orders': None}, 'time_columns': {'orders': 3}},
        spec({'orders': V(None)}, {'orders': U}),
        refused('INVALID_REQUEST', **EMPTY_KEY),
    ),
    case(
        'order.tables-by-name',
        {'primary_keys': {'users': 3, 'orders': 'nope'}},
        spec({'users': U, 'orders': V(['nope'])}),
        refused(
            'INVALID_REQUEST',
            field='primary_keys',
            table='orders',
            column='nope',
        ),
    ),
    case(
        'order.missing-column-before-composite',
        {'primary_keys': {'orders': ['order_id', 'nope']}},
        spec({'orders': V(['order_id', 'nope'])}),
        refused(
            'INVALID_REQUEST',
            field='primary_keys',
            table='orders',
            column='nope',
        ),
    ),
    # --- the wire: what no host may write (section 6A) -------------------
    wire('wire.key-bare-list', spec({'orders': ['order_id']}), HOST_BUG),
    wire('wire.key-bare-string', spec({'orders': 'order_id'}), HOST_BUG),
    wire('wire.key-bare-null', spec({'orders': None}), HOST_BUG),
    wire('wire.key-value-number', spec({'orders': V(3)}), HOST_BUG),
    wire('wire.key-value-string', spec({'orders': V('order_id')}), HOST_BUG),
    wire('wire.key-value-nested', spec({'orders': V([['a']])}), HOST_BUG),
    wire(
        'wire.key-value-member-number', spec({'orders': V(['a', 3])}), HOST_BUG
    ),
    wire(
        'wire.key-value-member-null', spec({'orders': V(['a', None])}), HOST_BUG
    ),
    wire(
        'wire.key-unreadable-object',
        spec({'orders': {'unreadable': True}}),
        HOST_BUG,
    ),
    wire(
        'wire.key-extra-field',
        spec({'orders': {'value': ['a'], 'x': 1}}),
        HOST_BUG,
    ),
    wire('wire.key-unknown-tag', spec({'orders': {'names': ['a']}}), HOST_BUG),
    wire('wire.key-tag-cased', spec({'orders': 'Unreadable'}), HOST_BUG),
    wire(
        'wire.time-bare-string', spec(time_columns={'orders': 'ts'}), HOST_BUG
    ),
    wire('wire.time-bare-null', spec(time_columns={'orders': None}), HOST_BUG),
    wire(
        'wire.time-value-number', spec(time_columns={'orders': V(3)}), HOST_BUG
    ),
    wire(
        'wire.time-value-list',
        spec(time_columns={'orders': V(['ts'])}),
        HOST_BUG,
    ),
    wire(
        'wire.time-unreadable-object',
        spec(time_columns={'orders': {'unreadable': True}}),
        HOST_BUG,
    ),
    wire(
        'wire.time-extra-field',
        spec(time_columns={'orders': {'value': 'ts', 'x': 1}}),
        HOST_BUG,
    ),
    # --- edit_schema beside describe_source (section 3) ------------------
    {
        'id': 'edit.key-not-names',
        'kind': 'edit',
        'owner': None,
        'set': {'table': 'orders', 'attribute': 'primary_key', 'value': 3},
        'edit': {'edit': 'primary_key', 'table': 'orders', 'key': U},
        'core': refused(
            'INVALID_REQUEST',
            field='primary_key',
            table='orders',
            rule='key_not_names',
        ),
        'pair': 'key.int',
        'register': None,
        'reference_why': 'the setter on the core route: '
        'tests/test_the_schema_edits.py holds it to the reference',
    },
    {
        'id': 'edit.time-not-a-name',
        'kind': 'edit',
        'owner': None,
        'set': {'table': 'orders', 'attribute': 'time_column', 'value': 3},
        'edit': {'edit': 'time_column', 'table': 'orders', 'column': U},
        'core': refused(
            'INVALID_REQUEST',
            field='time_column',
            table='orders',
            rule='time_column_not_a_name',
        ),
        'pair': 'time.int',
        'register': None,
        'reference_why': 'the setter on the core route: '
        'tests/test_the_schema_edits.py holds it to the reference',
    },
    {
        'id': 'edit.key-null',
        'kind': 'edit',
        'owner': None,
        'set': {'table': 'orders', 'attribute': 'primary_key', 'value': None},
        'edit': {'edit': 'primary_key', 'table': 'orders', 'key': V(None)},
        'core': described((None, 'ts')),
        'pair': 'key.null',
        'register': None,
        'reference_why': 'the setter clears the key, as the reference does '
        '(E67); describe_source refuses the same null as an empty key',
    },
    {
        'id': 'edit.time-null',
        'kind': 'edit',
        'owner': None,
        'set': {'table': 'orders', 'attribute': 'time_column', 'value': None},
        'edit': {'edit': 'time_column', 'table': 'orders', 'column': V(None)},
        'core': described((['order_id'], None)),
        'pair': 'time.null',
        'register': None,
        'reference_why': 'the setter clears the time column, as '
        'describe_source does',
    },
]


def case_for(vector: dict[str, Any], core: bool = False) -> dict[str, Any]:
    r"""A ``case`` vector as a case of ``contract/cases``' format, run to
    ``from_source``; with ``core``, forced onto ``Schema.from_source``'s
    DuckDB route of every table, otherwise on none."""
    routes: list[str] = []
    if core:
        from sdm_client import _routing

        routes = list(_routing.FROM_SOURCE_DUCKDB)
    return {
        'format': 'sdm-case/1',
        'id': CASE_PREFIX + vector['id'],
        'status': 'diagnostic',
        'slice': 'S1',
        'paths': [],
        'routes': routes,
        'cells': [],
        'input': {
            'base': 'shop',
            'from_source': copy.deepcopy(vector['from_source']),
            'pql': QUESTION,
            'materialize': {'indices': [0, 1], 'anchor_time': '2024-03-01'},
        },
        'until': 'from_source',
        'core': {
            'describe_source': {
                'provider': PROVIDER,
                'source': copy.deepcopy(vector['spec']),
                'check_keys': True,
            }
        },
        'expect': {'outcome': 'success', 'stage': 'from_source'},
        'why': f'T1.32h vector {vector["id"]}',
        'source': 'contract/tools/given_record.py',
    }


def run(
    mode: str,
    vectors: list[dict[str, Any]],
    directory: Path,
    *,
    core: bool = False,
    reference: Path | None = None,
) -> dict[str, dict[str, Any]]:
    r"""The ``case`` vectors among ``vectors`` through ``run_python.py
    --mode mode``, in this interpreter, from ``directory``: each
    observation by vector id."""
    cases = directory / 'cases'
    cases.mkdir(parents=True, exist_ok=True)
    for vector in vectors:
        if vector['kind'] != 'case':
            continue
        made = case_for(vector, core)
        (cases / f'{made["id"]}.json').write_text(
            json.dumps(made, indent=1), encoding='utf-8'
        )
    out = directory / f'{mode}.jsonl'
    command = [sys.executable, '-I', str(RUNNER), '--mode', mode]
    if reference is not None:
        command += ['--reference', str(reference)]
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


def _key(value: Any) -> list[str] | None:
    r"""A table's key as the contract writes it: its columns, or none."""
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


def project(observation: dict[str, Any]) -> dict[str, Any]:
    r"""The public answer of one run, as both Python runners record it:
    the outcome (no message, H17) and, for a success, each table's key and
    time column."""
    outcome = observation['outcome']
    schema = (observation.get('public') or {}).get('schema')
    tables = None
    if schema is not None:
        tables = {
            table['name']: {
                'primary_key': _key(table['primary_key']),
                'time_column': table['time_column'],
            }
            for table in schema['tables']
        }
    return {
        'result': outcome['result'],
        'stage': outcome['stage'],
        'error': outcome.get('error'),
        'code': outcome.get('code'),
        'tables': tables,
    }


#: The class of a refusal with no ``StructuredDataError`` code.
UNNAMED = 'UNNAMED'


def reference_class(reference: dict[str, Any]) -> str:
    r"""``success`` or ``refusal <CODE>`` of a recorded reference answer."""
    if reference['result'] == 'success':
        return 'success'
    return f'refusal {reference.get("code") or UNNAMED}'


def core_class(core: dict[str, Any]) -> str:
    r"""``success`` or ``refusal <CODE>`` of a vector's core answer."""
    if core['outcome'] == 'success':
        return 'success'
    return f'refusal {core["code"]}'


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
    with tempfile.TemporaryDirectory(prefix='t132h-record-') as tmp:
        observed = run('python-ref', VECTORS, Path(tmp), reference=reference)
    vectors = []
    for vector in VECTORS:
        entry = copy.deepcopy(vector)
        if vector['kind'] == 'case':
            entry['reference'] = project(observed[vector['id']])
            entry['reference_why'] = None
            print(
                vector['id'],
                reference_class(entry['reference']),
                entry['reference']['error'] or '',
                '->',
                core_class(vector['core']),
                file=sys.stderr,
            )
        else:
            entry['reference'] = None
        vectors.append(entry)
    document = {
        'format': FORMAT,
        'recorded_by': 'contract/tools/given_record.py',
        'reference': commit,
        'cell': 'ref-pandas3',
        'environment': {
            'python': platform.python_version(),
            'pandas': pd.__version__,
            'pyarrow': pa.__version__,
            'numpy': np.__version__,
            'duckdb': duckdb.__version__,
        },
        'base': 'shop',
        'provider': PROVIDER,
        'question': QUESTION,
        'register_ref': REGISTER_REF,
        'vectors': vectors,
    }
    OUT.write_text(json.dumps(document, indent=1) + '\n', encoding='utf-8')
    print(
        f'{len(vectors)} vectors written to {OUT.relative_to(ROOT)}',
        file=sys.stderr,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
