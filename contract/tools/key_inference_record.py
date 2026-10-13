#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Records the reference's key rules into ``contract/key-inference-vectors.json``
(T1.32e part 1's key vectors: the owner's ruling of 2026-10-13 to port the
reference's key inference, so that one null in a key never refuses a whole
table).

Run with the reference checkout's interpreter (the environment of
``contract/reference-env.txt``, the ``ref-pandas3`` cell), from this
repository's root::

    ~/code/sdmc-reference/.venv/bin/python -I contract/tools/key_inference_record.py

Written by T1.32e's independent test author before the implementation.
The reference infers a table's key from names and a preview of at most
1,000 rows, then ``Schema.from_source`` counts every row of the table
(``_check_source_keys``): an inferred key whose distinct values (among the
rows holding one) are fewer than 99% of the rows is declined, the table
left keyless, with a word when another table names the column and it is
nearly a key; one kept with missing values or repeats warns; a key named
in ``primary_keys`` holding a null is refused, and one repeating a value in
more than 1% of rows is refused, up to 1% warned. A keyless table read by
foreign key is then sampled without a key (the core's
``keyless_neighbors``).

Each vector is two generated DuckDB tables: ``accounts`` (``account_id``,
``region``), the entity, and ``payments`` (``payment_id``, ``account_id``
referring to it, ``paid_at``, ``amount``), the facts; ``contract/bases.yaml``'s
``shop`` is the base, its two tables dropped. :data:`VECTORS` varies one
key: its distinct share (100%, 99.5%, exactly 99%, 98.9%, 50%), its nulls
(1, 2, 10, all), its type (BIGINT, VARCHAR, DECIMAL(18, 0)), whether it is
inferred or named in ``primary_keys``, and the table's size: 200 rows,
1,000 (a full preview), or 5,000 with every repeat and null after the
first 1,000 rows, so the rule must hold over the whole table, not a
preview. The data is generated: these vectors hold ingestion and schema
inference, not a model.

Each vector is a case in ``contract/cases``' format run by
``contract/runners/run_python.py --mode python-ref`` through the public
``Schema.from_source`` and ``PredictiveQuery.materialize`` (``PREDICT
SUM(payments.amount, 0, 30, days) FOR EACH accounts.account_id``, the
first two accounts, anchored 2024-03-01), and this script keeps the
reference's answer as :func:`project` reads it from the observation:

- ``outcome``: success, or the refusal's class, ``StructuredDataError.code``
  and stage (no message text, H17);
- ``schema``: each table as the public schema describes it (each column's
  dtype and stype, the key chosen or None, the time column, the
  duplicate-key count) and the links; null when there is no schema;
- ``warnings``: the warning classes, in order.

``register`` marks the vectors where the core is expected to differ,
derived from the reference's answer, never chosen: the reference fails
with an unnamed exception (no ``StructuredDataError`` code) or with
``INTERNAL_ERROR`` (E66(b)'s class). The mark decides no policy: any
vector may differ from the reference only under an approved register
entry whose ``refs`` name it with exactly the transition it makes,
``REGISTER_REF`` + ``<id or glob> <from> -> <to>``, each side ``success``
or ``refusal <CODE>`` (the reference's unnamed exception is ``refusal
UNNAMED``), T1.32a's binding-transition form.

Nothing here is written by hand but the tables: every expected value is
the reference's. ``VECTORS``, ``case_for``, ``run`` and ``project`` are
imported by ``contract/tests/test_t132e_key_inference.py``, so
``python-core`` is observed and projected by the same code.
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
OUT = ROOT / 'contract/key-inference-vectors.json'
RUNNER = ROOT / 'contract/runners/run_python.py'
FORMAT = 'sdm-key-inference-vectors/1'
#: How a register entry names the vectors it covers, in its ``refs``: this
#: prefix, a vector's id or a glob of ids (``fnmatch``), and the
#: transition it declares, for example
#: ``contract/key-inference-vectors.json#nulls-text-* refusal UNNAMED ->
#: refusal UNSUPPORTED_REQUEST``.
REGISTER_REF = 'contract/key-inference-vectors.json#'
#: Each vector's case id is this prefix and the vector's id.
CASE_PREFIX = 't132e.key.'
#: The reference's preview: repeats and nulls of a 5,000-row table all
#: come after it.
PREVIEW_ROWS = 1_000
PQL = 'PREDICT SUM(payments.amount, 0, 30, days) FOR EACH accounts.account_id'
ANCHOR = '2024-03-01'
#: 2024-03-01 in nanoseconds since the epoch, as the core's input spells it.
ANCHOR_NS = '1709251200000000000'

_KINDS = {
    'int': ('{v}::BIGINT', 'BIGINT'),
    'text': ("'k' || ({v})::VARCHAR", 'VARCHAR'),
    'decimal': ('({v})::DECIMAL(18, 0)', 'DECIMAL(18, 0)'),
}


def key_expression(
    rows: int, *, repeats: int = 0, nulls: int = 0, kind: str = 'int'
) -> str:
    r"""A key of ``rows`` rows: row ``i`` holds ``i``, except the last
    ``nulls`` rows, which hold none, and the ``repeats`` rows before them,
    which repeat the first ``repeats`` values (so a 5,000-row table's
    first 1,000 rows are unique and hold a value). ``nulls`` of ``rows``
    is every row null."""
    form, sql_type = _KINDS[kind]
    if nulls >= rows:
        return f'NULL::{sql_type}'
    tail = rows - nulls
    value = (
        f'CASE WHEN i >= {tail - repeats} THEN i - {tail - repeats} ELSE i END'
    )
    keyed = form.format(v=value)
    return f'CASE WHEN i >= {tail} THEN NULL ELSE {keyed} END'


def setup(
    *,
    payments: int,
    payment_key: str,
    accounts: int = 50,
    account_key: str | None = None,
    account_kind: str = 'int',
) -> list[str]:
    r"""The statements that build the two tables (after ``shop``'s, whose
    tables they drop). Each payment refers to account ``i % accounts``."""
    account_form = _KINDS[account_kind][0]
    if account_key is None:
        account_key = account_form.format(v='i')
    refers = account_form.format(v=f'i % {accounts}')
    return [
        'DROP TABLE orders',
        'DROP TABLE users',
        f'CREATE TABLE accounts AS SELECT {account_key} AS account_id, '
        "['north', 'south', 'east'][1 + i % 3] AS region "
        f'FROM range({accounts}) t(i)',
        f'CREATE TABLE payments AS SELECT {payment_key} AS payment_id, '
        f'{refers} AS account_id, '
        "TIMESTAMP '2023-06-01 00:00:00' + to_minutes((i * 90)::INTEGER) "
        'AS paid_at, (i % 13)::DOUBLE AS amount '
        f'FROM range({payments}) t(i)',
    ]


def _vector(
    id: str,
    *,
    table: str,
    rows: int,
    statements: list[str],
    key: dict[str, Any],
    primary_keys: dict[str, str] | None = None,
    account_kind: str = 'int',
) -> dict[str, Any]:
    return {
        'id': id,
        'table': table,
        'rows': rows,
        'key': key,
        'setup': statements,
        'primary_keys': primary_keys,
        'account_kind': account_kind,
    }


def _fact(
    id: str,
    rows: int,
    *,
    repeats: int = 0,
    nulls: int = 0,
    kind: str = 'int',
    named: bool = False,
) -> dict[str, Any]:
    r"""A vector varying ``payments.payment_id``."""
    return _vector(
        id,
        table='payments',
        rows=rows,
        key={'kind': kind, 'nulls': nulls, 'repeats': repeats},
        statements=setup(
            payments=rows,
            payment_key=key_expression(
                rows, repeats=repeats, nulls=nulls, kind=kind
            ),
        ),
        primary_keys={'payments': 'payment_id'} if named else None,
    )


def _entity(
    id: str,
    rows: int,
    *,
    repeats: int = 0,
    nulls: int = 0,
    kind: str = 'int',
) -> dict[str, Any]:
    r"""A vector varying ``accounts.account_id``, which every payment
    refers to."""
    return _vector(
        id,
        table='accounts',
        rows=rows,
        key={'kind': kind, 'nulls': nulls, 'repeats': repeats},
        statements=setup(
            payments=400,
            payment_key='i::BIGINT',
            accounts=rows,
            account_key=key_expression(
                rows, repeats=repeats, nulls=nulls, kind=kind
            ),
            account_kind=kind,
        ),
        account_kind=kind,
    )


#: The vectors: ids name the key's case and, last, the varied table's rows.
VECTORS: list[dict[str, Any]] = [
    # an inferred fact key's distinct share: 100%, 99.5%, exactly 99%,
    # 98.9%, 50%; at 5,000 rows every repeat comes after the preview
    *[
        _fact(f'share-{name}.{rows}', rows, repeats=rows - int(rows * share))
        for rows in (1_000, 5_000)
        for name, share in (
            ('100', 1.0),
            ('99.5', 0.995),
            ('99', 0.99),
            ('98.9', 0.989),
            ('50', 0.5),
        )
    ],
    # the controls: a clean key of a table under a preview, which the core
    # keys today
    _fact('share-100.200', 200),
    _fact('named-clean.200', 200, named=True),
    # an inferred fact key with 1, 2 and 10 nulls, then none held
    *[
        _fact(f'nulls-{n}.{rows}', rows, nulls=n)
        for rows in (200, 5_000)
        for n in (1, 2, 10)
    ],
    *[_fact(f'all-null.{rows}', rows, nulls=rows) for rows in (200, 5_000)],
    # the key's type: text and decimal, unique and with a null
    *[
        _fact(f'{kind}-unique.1000', 1_000, kind=kind)
        for kind in ('text', 'decimal')
    ],
    *[
        _fact(f'nulls-1-{kind}.{rows}', rows, nulls=1, kind=kind)
        for kind in ('text', 'decimal')
        for rows in (200, 5_000)
    ],
    # the entity's key: a null, every row null, a text key with a null
    _entity('entity-nulls-1.200', 200, nulls=1),
    _entity('entity-nulls-1.5000', 5_000, nulls=1),
    _entity('entity-all-null.200', 200, nulls=200),
    _entity('entity-nulls-1-text.200', 200, nulls=1, kind='text'),
    # a key declined on the table a foreign key points at: nearly a key
    # (95%, warned of as named elsewhere) and half one
    _entity('entity-share-95.1000', 1_000, repeats=50),
    _entity('entity-share-95.5000', 5_000, repeats=250),
    _entity('entity-share-50.1000', 1_000, repeats=500),
    # a key named in primary_keys: clean, one null, repeats within and
    # past 1%, and text and decimal keys holding a null
    *[
        _fact(f'named-clean.{rows}', rows, named=True)
        for rows in (1_000, 5_000)
    ],
    *[
        _fact(f'named-null-1.{rows}', rows, nulls=1, named=True)
        for rows in (200, 5_000)
    ],
    *[
        _fact(
            f'named-repeats-0.5.{rows}', rows, repeats=rows // 200, named=True
        )
        for rows in (1_000, 5_000)
    ],
    *[
        _fact(
            f'named-repeats-1.1.{rows}',
            rows,
            repeats=rows * 11 // 1_000,
            named=True,
        )
        for rows in (1_000, 5_000)
    ],
    *[
        _fact(
            f'named-null-1-{kind}.{rows}', rows, nulls=1, kind=kind, named=True
        )
        for kind in ('text', 'decimal')
        for rows in (200, 5_000)
    ],
]


# --- the cases -----------------------------------------------------------------


def _indices(vector: dict[str, Any]) -> tuple[list[Any], list[dict[str, str]]]:
    r"""The two accounts asked about, as the public call and the core's
    input spell them."""
    if vector['account_kind'] == 'text':
        return ['k0', 'k1'], [{'t': 'utf8', 'v': f'k{i}'} for i in (0, 1)]
    return [0, 1], [{'t': 'i64', 'v': str(i)} for i in (0, 1)]


def materializes(vector: dict[str, Any]) -> bool:
    r"""Whether the vector's case goes on to ``materialize``: a decimal
    entity key's indices are no input the core's case spells, so such a
    vector stops at ``from_source`` (none does today)."""
    return vector['account_kind'] in ('int', 'text')


def case_for(
    vector: dict[str, Any], routes: list[str] | None = None
) -> dict[str, Any]:
    r"""The vector as a case: ``Schema.from_source`` over its two tables,
    with its ``primary_keys``, then ``materialize``, on ``routes`` (none:
    the runner's own choice, as the reference has)."""
    named = vector['primary_keys']
    public, core = _indices(vector)
    until = 'materialize' if materializes(vector) else 'from_source'
    case: dict[str, Any] = {
        'format': 'sdm-case/1',
        'id': CASE_PREFIX + vector['id'],
        'status': 'diagnostic',
        'slice': 'S1',
        'paths': [],
        'routes': list(routes or []),
        'cells': [],
        'input': {
            'base': 'shop',
            'setup': list(vector['setup']),
            'pql': PQL,
            'materialize': {'indices': public, 'anchor_time': ANCHOR},
        },
        'until': until,
        'core': {
            'describe_source': {
                'provider': {'provider': 'sql', 'dialect': 'duckdb'},
                'source': {
                    'tables': [],
                    'primary_keys': {
                        table: [column]
                        for table, column in (named or {}).items()
                    },
                    'time_columns': {},
                },
                'check_keys': True,
            }
        },
        'expect': {'outcome': 'success', 'stage': until},
        'why': f'T1.32e key vector {vector["id"]}',
        'source': 'contract/tools/key_inference_record.py',
    }
    if until == 'materialize':
        case['core']['materialize'] = {
            'provider': {'provider': 'sql', 'dialect': 'duckdb'},
            'indices': core,
            'anchor_time': {
                't': 'ts',
                'v': ANCHOR_NS,
                'unit': 'ns',
                'zone': None,
            },
            'seed': None,
        }
    if named is not None:
        case['input']['from_source'] = {'primary_keys': dict(named)}
    return case


def run(
    mode: str,
    vectors: list[dict[str, Any]],
    directory: Path,
    *,
    routes: list[str] | None = None,
    reference: Path | None = None,
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


# --- the public answer ---------------------------------------------------------


def _schema(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return None
    return {
        'tables': [
            {
                'name': entry.get('name'),
                'columns': [
                    {
                        'name': column.get('name'),
                        'dtype': column.get('dtype'),
                        'stype': column.get('stype'),
                    }
                    for column in entry.get('columns') or []
                ],
                'primary_key': entry.get('primary_key'),
                'time_column': entry.get('time_column'),
                'duplicate_keys': entry.get('duplicate_keys'),
            }
            for entry in sorted(
                schema.get('tables') or [], key=lambda t: str(t.get('name'))
            )
        ],
        'links': sorted(schema.get('links') or []),
    }


def project(observation: dict[str, Any]) -> dict[str, Any]:
    r"""The public answer of one run of a vector, as both Python runners
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
        'schema': _schema(public.get('schema')),
        'warnings': list(host.get('warnings') or []),
    }


#: The class of a refusal with no ``StructuredDataError`` code (the
#: reference's unnamed exception).
UNNAMED = 'UNNAMED'


def outcome_class(outcome: dict[str, Any]) -> str:
    r"""``success`` or ``refusal <CODE>`` of an answer's ``outcome``."""
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


def duckdb_facts(vector: dict[str, Any]) -> dict[str, Any]:
    r"""The varied key as DuckDB holds it, over the whole table and its
    first 1,000 rows: rows, rows holding a value, distinct values, and the
    catalog type, so each vector is seen to be the key it names."""
    import duckdb

    table = vector['table']
    column = 'payment_id' if table == 'payments' else 'account_id'
    connection = duckdb.connect()
    try:
        for statement in shop() + vector['setup']:
            connection.execute(statement)
        facts: dict[str, Any] = {}
        for scope, source in (
            ('whole', table),
            ('preview', f'(SELECT * FROM {table} LIMIT {PREVIEW_ROWS})'),
        ):
            rows, held, distinct = connection.execute(
                f'SELECT count(*), count({column}), count(DISTINCT {column}) '
                f'FROM {source}'
            ).fetchone()
            facts[scope] = {'rows': rows, 'held': held, 'distinct': distinct}
        facts['type'] = connection.execute(
            f'SELECT typeof({column}) FROM {table} LIMIT 1'
        ).fetchone()[0]
        return facts
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
    with tempfile.TemporaryDirectory(prefix='t132e-key-record-') as tmp:
        observed = run('python-ref', VECTORS, Path(tmp), reference=reference)
    recorded = []
    for vector in VECTORS:
        answer = project(observed[vector['id']])
        why = register(answer)
        recorded.append(
            {
                **vector,
                'until': case_for(vector)['until'],
                'duckdb': duckdb_facts(vector),
                'reference': answer,
                'register': {'required': bool(why), 'why': why},
            }
        )
        outcome = answer['outcome']
        keys = (
            {t['name']: t['primary_key'] for t in answer['schema']['tables']}
            if answer['schema']
            else None
        )
        print(
            vector['id'],
            outcome['result'],
            outcome['stage'],
            outcome['error'] or '',
            outcome['code'] or '',
            keys,
            answer['warnings'],
            '(register)' if why else '',
            file=sys.stderr,
        )
    document = {
        'format': FORMAT,
        'recorded_by': 'contract/tools/key_inference_record.py',
        'reference': commit,
        'cell': 'ref-pandas3',
        'environment': {
            'python': platform.python_version(),
            'pandas': pd.__version__,
            'pyarrow': pa.__version__,
            'numpy': np.__version__,
            'duckdb': duckdb.__version__,
        },
        'pql': PQL,
        'anchor': ANCHOR,
        'preview_rows': PREVIEW_ROWS,
        'register_ref': REGISTER_REF,
        'vectors': recorded,
    }
    OUT.write_text(json.dumps(document, indent=1) + '\n', encoding='utf-8')
    required = sum(v['register']['required'] for v in recorded)
    print(
        f'{len(recorded)} vectors written to {OUT.relative_to(ROOT)}, '
        f'{required} marked register entry required',
        file=sys.stderr,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
