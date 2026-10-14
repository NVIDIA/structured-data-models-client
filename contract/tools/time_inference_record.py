#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Records the reference's whole-table time inference into
``contract/time-inference-vectors.json`` (T1.32e part 1's vectors).

Run with the reference checkout's interpreter (the environment of
``contract/reference-env.txt``, the ``ref-pandas3`` cell), from this
repository's root::

    ~/code/sdmc-reference/.venv/bin/python -I contract/tools/time_inference_record.py

Written by T1.32e's independent test author before the implementation.
The reference reads at most 1,000 rows (``_NUM_SAMPLE_ROWS``) to infer a
table's metadata; when it read exactly that many, rows may be unread, so
its time column's choice (``predictive_query.time_columns``) asks the
source for each candidate's ``MIN``, ``MAX`` and ``COUNT`` over the whole
table: whether a column is mostly empty or holds one value, and where it
starts, are then what every row says. The core refuses any such table
today (``time_whole_table``), even where ``time_columns`` names the
column.

Each vector is one generated DuckDB table, ``events``, of :data:`SIZES`
rows (999, one short of the preview; 1,000, a full preview; 1,001, one
row unread; 5,000), built by a :data:`SCENARIOS` entry: a key
``event_id``, a number, and one or more timestamp, date, time-of-day or
timestamp-with-zone columns that are not the key, some with nulls, some
whose values differ from the first 1,000 rows only after them; with and
without ``time_columns`` naming a column (or ``None``). ``contract/bases.yaml``'s
``shop`` is the base, its two tables dropped, so ``events`` is the
source's only table. The data is generated: these vectors hold ingestion
and schema inference, not a model.

Each vector is a case in ``contract/cases``' format, ``until``
``from_source``, written to a temporary directory and run by
``contract/runners/run_python.py --mode python-ref`` through the public
``Schema.from_source``, and this script keeps the reference's answer as
:func:`project` reads it from the observation:

- ``outcome``: success, or the refusal's class, ``StructuredDataError.code``
  and stage (no message text, H17);
- ``schema``: ``events`` as the public schema describes it: each column's
  name, dtype and stype, the key, the time column (the one chosen) and the
  duplicate-key count; null when there is no schema;
- ``warnings``: the warning classes, in order.

``register`` marks the vectors where the core is expected to differ. It is
derived from the reference's answer, never chosen: the reference fails
with an unnamed exception (no ``StructuredDataError`` code) or with
``INTERNAL_ERROR`` (E66(b)'s class: a named refusal or a success in its
place is a change to register). The mark decides no policy: any vector,
marked or not, may differ from the reference only under an approved
register entry whose ``refs`` name it with exactly the transition it makes,
``REGISTER_REF`` + ``<id or glob> <from> -> <to>``, each side ``success``
or ``refusal <CODE>`` (``outcome_class``; the reference's unnamed
exception is ``refusal UNNAMED``), T1.32a's binding-transition form.

Nothing here is written by hand but the tables: every expected value is
the reference's. ``SCENARIOS``, ``SIZES``, ``vectors``, ``case_for``,
``run`` and ``project`` are imported by
``contract/tests/test_t132e_time_inference.py``, so ``python-core`` is
observed and projected by the same code.
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
OUT = ROOT / 'contract/time-inference-vectors.json'
RUNNER = ROOT / 'contract/runners/run_python.py'
FORMAT = 'sdm-time-inference-vectors/1'
#: How a register entry names the vectors it covers, in its ``refs``: this
#: prefix, a vector's id or a glob of ids (``fnmatch``), and the transition
#: it declares, for example
#: ``contract/time-inference-vectors.json#late-values.* success -> refusal
#: UNSUPPORTED_REQUEST``.
REGISTER_REF = 'contract/time-inference-vectors.json#'
#: Each vector's case id is this prefix and the vector's id.
CASE_PREFIX = 't132e.'
#: The table each vector describes.
TABLE = 'events'
#: The reference's preview (``_NUM_SAMPLE_ROWS``): a preview of exactly
#: this many rows may leave rows unread.
PREVIEW_ROWS = 1_000
#: The table sizes: one short of a full preview, a full preview, one row
#: past it, and several previews.
SIZES = (999, 1_000, 1_001, 5_000)

#: A row's minutes since 2023-01-01, the plain timestamp's values.
_MINUTES = "TIMESTAMP '2023-01-01 00:00:00' + to_minutes(i::INTEGER)"


def _late(early: str, late: str) -> str:
    r"""``early`` in the first 1,000 rows, ``late`` in every row after."""
    return f'CASE WHEN i < {PREVIEW_ROWS} THEN {early} ELSE {late} END'


#: The scenarios, each a vector at every size of :data:`SIZES`: its id,
#: the columns of ``events`` beside ``event_id`` and ``amount``, each a
#: name and a DuckDB expression of the row number ``i`` (0 to rows - 1),
#: and ``time_columns`` passed to ``Schema.from_source`` (left out when
#: absent). A name is quoted as it is written.
SCENARIOS: list[dict[str, Any]] = [
    # one timestamp, the plainest table a real source has
    {
        'id': 'one-timestamp',
        'columns': [('happened_at', _MINUTES)],
    },
    {
        'id': 'one-timestamp-named',
        'columns': [('happened_at', _MINUTES)],
        'time_columns': {TABLE: 'happened_at'},
    },
    # a timestamp, a date and a time of day; the date starts earliest
    {
        'id': 'three-kinds',
        'columns': [
            ('happened_at', _MINUTES),
            ('due_on', "DATE '2022-06-01' + (i % 400)::INTEGER"),
            ('at_time', "TIME '08:00:00' + to_seconds((i % 3600)::INTEGER)"),
        ],
    },
    {
        'id': 'three-kinds-named',
        'columns': [
            ('happened_at', _MINUTES),
            ('due_on', "DATE '2022-06-01' + (i % 400)::INTEGER"),
            ('at_time', "TIME '08:00:00' + to_seconds((i % 3600)::INTEGER)"),
        ],
        'time_columns': {TABLE: 'happened_at'},
    },
    {
        'id': 'three-kinds-named-date',
        'columns': [
            ('happened_at', _MINUTES),
            ('due_on', "DATE '2022-06-01' + (i % 400)::INTEGER"),
            ('at_time', "TIME '08:00:00' + to_seconds((i % 3600)::INTEGER)"),
        ],
        'time_columns': {TABLE: 'due_on'},
    },
    {
        'id': 'three-kinds-named-time-of-day',
        'columns': [
            ('happened_at', _MINUTES),
            ('due_on', "DATE '2022-06-01' + (i % 400)::INTEGER"),
            ('at_time', "TIME '08:00:00' + to_seconds((i % 3600)::INTEGER)"),
        ],
        'time_columns': {TABLE: 'at_time'},
    },
    {
        'id': 'three-kinds-named-none',
        'columns': [
            ('happened_at', _MINUTES),
            ('due_on', "DATE '2022-06-01' + (i % 400)::INTEGER"),
            ('at_time', "TIME '08:00:00' + to_seconds((i % 3600)::INTEGER)"),
        ],
        'time_columns': {TABLE: None},
    },
    # a timestamp with its zone
    {
        'id': 'timestamptz',
        'columns': [
            (
                'happened_at',
                "TIMESTAMPTZ '2023-01-01 00:00:00+00' + to_minutes(i::INTEGER)",
            )
        ],
    },
    # nulls in the time column: a third of the rows, then most of them
    {
        'id': 'nulls',
        'columns': [
            (
                'happened_at',
                f'CASE WHEN i % 3 = 0 THEN NULL ELSE {_MINUTES} END',
            )
        ],
    },
    {
        'id': 'nulls-named',
        'columns': [
            (
                'happened_at',
                f'CASE WHEN i % 3 = 0 THEN NULL ELSE {_MINUTES} END',
            )
        ],
        'time_columns': {TABLE: 'happened_at'},
    },
    {
        'id': 'mostly-null',
        'columns': [
            ('happened_at', f'CASE WHEN i % 5 = 0 THEN {_MINUTES} END')
        ],
    },
    # values only after the first 1,000 rows: a preview holds none
    {
        'id': 'late-values',
        'columns': [('happened_at', _late('NULL', _MINUTES))],
    },
    {
        'id': 'late-values-named',
        'columns': [('happened_at', _late('NULL', _MINUTES))],
        'time_columns': {TABLE: 'happened_at'},
    },
    # one value in the first 1,000 rows, others after: a preview reads it
    # as constant
    {
        'id': 'late-change',
        'columns': [
            ('happened_at', _late("TIMESTAMP '2023-01-01 00:00:00'", _MINUTES))
        ],
    },
    {
        'id': 'late-change-named',
        'columns': [
            ('happened_at', _late("TIMESTAMP '2023-01-01 00:00:00'", _MINUTES))
        ],
        'time_columns': {TABLE: 'happened_at'},
    },
    # two timestamps; the second starts earlier, but only after the first
    # 1,000 rows
    {
        'id': 'earliest-after-preview',
        'columns': [
            ('shipped_at', _MINUTES),
            (
                'billed_at',
                _late(
                    "TIMESTAMP '2023-06-01 00:00:00' + to_minutes(i::INTEGER)",
                    "TIMESTAMP '2020-01-01 00:00:00' + to_minutes(i::INTEGER)",
                ),
            ),
        ],
    },
    {
        'id': 'earliest-after-preview-named',
        'columns': [
            ('shipped_at', _MINUTES),
            (
                'billed_at',
                _late(
                    "TIMESTAMP '2023-06-01 00:00:00' + to_minutes(i::INTEGER)",
                    "TIMESTAMP '2020-01-01 00:00:00' + to_minutes(i::INTEGER)",
                ),
            ),
        ],
        'time_columns': {TABLE: 'shipped_at'},
    },
    # a placeholder (1900-01-01) in the last row is the whole table's
    # minimum: no start, so the other column starts earlier
    {
        'id': 'placeholder-after-preview',
        'columns': [
            (
                'shipped_at',
                'CASE WHEN i = {rows} - 1 '
                "THEN TIMESTAMP '1900-01-01 00:00:00' ELSE "
                "TIMESTAMP '2023-01-01 00:00:00' + to_minutes(i::INTEGER) END",
            ),
            (
                'billed_at',
                "TIMESTAMP '2022-01-01 00:00:00' + to_minutes(i::INTEGER)",
            ),
        ],
    },
    # one value in every row
    {
        'id': 'constant',
        'columns': [('happened_at', "TIMESTAMP '2023-01-01 00:00:00'")],
    },
    # AdventureWorks' stamp of the last edit, alone, and beside an order date
    {
        'id': 'edit-stamp-only',
        'columns': [('"ModifiedDate"', _MINUTES)],
    },
    {
        'id': 'edit-stamp-and-date',
        'columns': [
            ('"OrderDate"', "DATE '2022-06-01' + (i % 900)::INTEGER"),
            ('"ModifiedDate"', _MINUTES),
        ],
    },
    {
        'id': 'edit-stamp-and-date-named',
        'columns': [
            ('"OrderDate"', "DATE '2022-06-01' + (i % 900)::INTEGER"),
            ('"ModifiedDate"', _MINUTES),
        ],
        'time_columns': {TABLE: 'OrderDate'},
    },
    # names the time column cannot take: refused by name, at every size
    {
        'id': 'named-missing-column',
        'columns': [('happened_at', _MINUTES)],
        'time_columns': {TABLE: 'no_such_column'},
    },
    {
        'id': 'named-the-key',
        'columns': [('happened_at', _MINUTES)],
        'time_columns': {TABLE: 'event_id'},
    },
    {
        'id': 'named-a-number',
        'columns': [('happened_at', _MINUTES)],
        'time_columns': {TABLE: 'amount'},
    },
    {
        'id': 'named-missing-table',
        'columns': [('happened_at', _MINUTES)],
        'time_columns': {'no_such_table': 'happened_at'},
    },
]


#: The rows of a fallback vector's view, and the row from which reading
#: its time columns raises: past the first row group, so a preview of
#: 1,000 rows (one thread, insertion order) never reads it.
FALLBACK_ROWS = 200_000
FALLBACK_FROM = 150_000
#: The message the view raises with.
FALLBACK_ERROR = 'no whole-table read'

#: The fallback scenarios (owner, 2026-10-13: P1B-38 rejected, the core
#: matches the reference's fallback). ``events`` is a view whose every
#: time column raises from row :data:`FALLBACK_FROM` on, so the
#: reference's one query over every row (``COUNT(*)`` and each
#: candidate's ``MIN``, ``MAX`` and ``COUNT``) fails, and it decides from
#: its preview: a column the preview would refuse for its values is kept
#: as unconfirmed, behind every column it would not, and, when taken,
#: warned of (``TimeColumnWarning``, ``taken_unchecked``). Each is one
#: vector of :data:`FALLBACK_ROWS` rows.
FALLBACK: list[dict[str, Any]] = [
    # the preview's choice stands, nothing in doubt
    {
        'id': 'fallback-plain',
        'columns': [('happened_at', _MINUTES)],
    },
    # one value in the preview: refused by it, kept unconfirmed, warned of
    {
        'id': 'fallback-unconfirmed',
        'columns': [
            ('happened_at', _late("TIMESTAMP '2023-01-01 00:00:00'", _MINUTES))
        ],
    },
    # no value in the preview: the same
    {
        'id': 'fallback-unconfirmed-empty',
        'columns': [('happened_at', _late('NULL', _MINUTES))],
    },
    # a doubted column beside a plain one: the plain one is taken
    {
        'id': 'fallback-doubted-beside-plain',
        'columns': [
            ('shipped_at', _late("TIMESTAMP '2020-01-01 00:00:00'", _MINUTES)),
            (
                'billed_at',
                "TIMESTAMP '2022-01-01 00:00:00' + to_minutes(i::INTEGER)",
            ),
        ],
    },
    # named: no inference asked for
    {
        'id': 'fallback-unconfirmed-named',
        'columns': [
            ('happened_at', _late("TIMESTAMP '2023-01-01 00:00:00'", _MINUTES))
        ],
        'time_columns': {TABLE: 'happened_at'},
    },
]


def fallback_setup(scenario: dict[str, Any]) -> list[str]:
    r"""The statements that build the view ``events`` of a fallback
    scenario (after ``shop``'s, whose ``orders`` they drop): its rows in a
    table of another schema, which the source does not list, each row
    referring to one of ``shop``'s ``users``, so the schema has a link
    and the reference says how it chose (it says nothing of a lone
    table's time column)."""
    columns = ', '.join(
        f'(CASE WHEN i >= {FALLBACK_FROM} THEN error({FALLBACK_ERROR!r}) '
        f'ELSE {expression} END) AS {name}'
        for name, expression in scenario['columns']
    )
    return [
        'SET threads = 1',
        'DROP TABLE orders',
        'CREATE SCHEMA hidden',
        f'CREATE TABLE hidden.raw AS SELECT i FROM range({FALLBACK_ROWS}) t(i)',
        f'CREATE VIEW {TABLE} AS SELECT i::BIGINT AS event_id, '
        f'(i % 8)::BIGINT AS user_id, (i % 97)::DOUBLE AS amount, '
        f'{columns} FROM hidden.raw',
    ]


def setup(scenario: dict[str, Any], rows: int) -> list[str]:
    r"""The statements that build ``events`` of ``rows`` rows (after
    ``shop``'s, whose tables they drop)."""
    columns = ', '.join(
        f'({expression.format(rows=rows)}) AS {name}'
        for name, expression in scenario['columns']
    )
    return [
        'DROP TABLE orders',
        'DROP TABLE users',
        f'CREATE TABLE {TABLE} AS SELECT i::BIGINT AS event_id, '
        f'(i % 97)::DOUBLE AS amount, {columns} FROM range({rows}) t(i)',
    ]


def vectors() -> list[dict[str, Any]]:
    r"""Every scenario at every size, then every fallback scenario at
    :data:`FALLBACK_ROWS`: its id (``<scenario>.<rows>``), rows,
    setup, ``time_columns`` (null when not passed) and ``tables`` (the
    tables named, null when every table is taken)."""
    out = []
    for scenario in SCENARIOS:
        for rows in SIZES:
            out.append(
                {
                    'id': f'{scenario["id"]}.{rows}',
                    'scenario': scenario['id'],
                    'rows': rows,
                    'setup': setup(scenario, rows),
                    'time_columns': scenario.get('time_columns'),
                    'tables': None,
                }
            )
    for scenario in FALLBACK:
        out.append(
            {
                'id': f'{scenario["id"]}.{FALLBACK_ROWS}',
                'scenario': scenario['id'],
                'rows': FALLBACK_ROWS,
                'setup': fallback_setup(scenario),
                'time_columns': scenario.get('time_columns'),
                # A view is listed only when named.
                'tables': {TABLE: TABLE, 'users': 'users'},
            }
        )
    return out


# --- the cases -----------------------------------------------------------------


def case_for(
    vector: dict[str, Any], routes: list[str] | None = None
) -> dict[str, Any]:
    r"""The vector as a case: ``Schema.from_source`` over its table, with
    its ``time_columns``, on ``routes`` (none: the runner's own choice, as
    the reference has)."""
    named = vector['time_columns']
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
            'pql': f'PREDICT COUNT({TABLE}.*, 0, 30, days) FOR EACH '
            f'{TABLE}.event_id',
            'materialize': {'indices': [0, 1], 'anchor_time': '2024-03-01'},
        },
        'until': 'from_source',
        'core': {
            'describe_source': {
                'provider': {'provider': 'sql', 'dialect': 'duckdb'},
                'source': {
                    'tables': [
                        {'name': name, 'source': source}
                        for name, source in (vector.get('tables') or {}).items()
                    ],
                    'primary_keys': {},
                    'time_columns': {
                        table: {'value': column}
                        for table, column in (named or {}).items()
                    },
                },
                'check_keys': True,
            }
        },
        'expect': {'outcome': 'success', 'stage': 'from_source'},
        'why': f'T1.32e vector {vector["id"]}',
        'source': 'contract/tools/time_inference_record.py',
    }
    given: dict[str, Any] = {}
    if vector.get('tables'):
        given['tables'] = dict(vector['tables'])
    if named is not None:
        given['time_columns'] = dict(named)
    if given:
        case['input']['from_source'] = given
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


def _table(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return None
    for entry in schema.get('tables') or []:
        if entry.get('name') == TABLE:
            return {
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
    return 'not described'


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
        'schema': _table(public.get('schema')),
        'warnings': list(host.get('warnings') or []),
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


def duckdb_facts(vector: dict[str, Any]) -> dict[str, Any]:
    r"""What the table holds, read in DuckDB: its rows, and each column's
    catalog type, rows holding a value, distinct values and minimum as
    text, over the whole table and over its first 1,000 rows (a preview),
    so each vector is seen to be the table it names."""
    import duckdb

    connection = duckdb.connect()
    try:
        for statement in shop() + vector['setup']:
            connection.execute(statement)
        rows = connection.execute(f'SELECT count(*) FROM {TABLE}').fetchone()
        described = connection.execute(f'DESCRIBE {TABLE}').fetchall()
        columns = {}
        for name, kind, *_ in described:
            quoted = '"' + name.replace('"', '""') + '"'
            facts = {}
            for scope, source in (
                ('whole', TABLE),
                (
                    'preview',
                    f'(SELECT * FROM {TABLE} ORDER BY event_id '
                    f'LIMIT {PREVIEW_ROWS})',
                ),
            ):
                try:
                    held, distinct, low = connection.execute(
                        f'SELECT count({quoted}), count(DISTINCT {quoted}), '
                        f'min({quoted})::VARCHAR FROM {source}'
                    ).fetchone()
                except duckdb.Error:
                    # A fallback vector's time column: reading it whole
                    # raises, as the vector means it to.
                    facts[scope] = 'raises'
                    continue
                facts[scope] = {'held': held, 'distinct': distinct, 'min': low}
            columns[name] = {'type': kind, **facts}
        return {'rows': rows[0] if rows else 0, 'columns': columns}
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
    every = vectors()
    with tempfile.TemporaryDirectory(prefix='t132e-record-') as tmp:
        observed = run('python-ref', every, Path(tmp), reference=reference)
    recorded = []
    for vector in every:
        answer = project(observed[vector['id']])
        why = register(answer)
        recorded.append(
            {
                **vector,
                'duckdb': duckdb_facts(vector),
                'reference': answer,
                'register': {'required': bool(why), 'why': why},
            }
        )
        outcome = answer['outcome']
        schema = answer['schema']
        print(
            vector['id'],
            outcome['result'],
            outcome['stage'],
            outcome['error'] or '',
            outcome['code'] or '',
            schema.get('time_column') if isinstance(schema, dict) else schema,
            answer['warnings'],
            '(register)' if why else '',
            file=sys.stderr,
        )
    document = {
        'format': FORMAT,
        'recorded_by': 'contract/tools/time_inference_record.py',
        'reference': commit,
        'cell': 'ref-pandas3',
        'environment': {
            'python': platform.python_version(),
            'pandas': pd.__version__,
            'pyarrow': pa.__version__,
            'numpy': np.__version__,
            'duckdb': duckdb.__version__,
        },
        'table': TABLE,
        'preview_rows': PREVIEW_ROWS,
        'sizes': list(SIZES),
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
