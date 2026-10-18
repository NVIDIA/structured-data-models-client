# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T2.3's round 1 regressions on Postgres, through the public API, held to
the reference.

    SDM_POSTGRES_TEST_DSN=... python -m pytest \
        contract/tests/test_t23_postgres_regressions.py -q

Written by T2.3's independent test author from the round 1 review's
findings. ``contract/postgres-regression-vectors.json`` is recorded from
python-ref on Postgres 17 by ``contract/tools/postgres_regressions_record.py``,
whose ``observe`` this test calls on the core's Postgres routes, so both
are observed by the same code: ``Schema.from_source(PostgresSource(...))``
and ``PredictiveQuery(...).materialize(...)`` over a database of the
vector's own.

- ``time.new-york-auto-anchor``: in a session whose ``TimeZone`` is
  America/New_York, the automatic anchor and every training anchor are the
  reference's (the latest 2024-04-10T00:00:00Z): a zoned time bound keeps
  its UTC instant.
- ``type.*``: a ``uuid``, ``inet``, unconstrained ``numeric``,
  ``numeric[]`` or ``uuid[]`` column: the schema and the request are the
  reference's.
- ``relation.dollar``: a keyed materialize over the relation ``users$1``
  succeeds with the reference's request.
- From the second review: a ``real`` key holding fractional values joins
  its linked table (``key.real-fractional-linked``) and finds its entities
  (``key.real-fractional-entity``); a table with a ``NOT NULL`` domain
  column is joined on a text key (``join.domain-not-null-text-key``); a
  column named ``cost$1`` is read by a keyed lookup (``column.dollar``); a
  ``timetz`` column is read by ``Schema.from_source`` (``type.timetz``).
- From Codex's second round: integral values past 2**53 in a
  ``numeric(100,0)`` stay exact (``type.numeric-100-0-past-2-53``), and a
  text-key join reads its seed rows whatever the table's columns are named,
  ``x`` included (``join.domain-not-null-text-key-column-x``).

Each answer (outcome, public schema, the payloads' digest and anchors) is
the reference's, or differs exactly as an approved register entry's ref
declares (``contract/postgres-regression-vectors.json#<id or glob> <from>
-> <to>``); a proposed entry excuses nothing. Where
``SDM_POSTGRES_TEST_DSN`` names no server and ``SDM_REQUIRE_POSTGRES`` is
not set, the vector tests have nothing to hold and return without
asserting, as T2.1a's Postgres tests do; with it set, a missing server
fails (H26).
"""

from __future__ import annotations

import fnmatch
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
VECTORS = ROOT / 'contract/postgres-regression-vectors.json'
REGISTER = ROOT / 'docs/design/rust-core/register.yaml'


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


P = _module(
    'postgres_regressions_record',
    ROOT / 'contract/tools/postgres_regressions_record.py',
)
RECORDED: dict[str, Any] = json.loads(VECTORS.read_text(encoding='utf-8'))
BY_ID = {vector['id']: vector for vector in RECORDED['vectors']}
TRANSITION = re.compile(
    re.escape(P.REGISTER_REF)
    + r'(?P<glob>\S+) (?P<old>success|refusal [A-Z][A-Z_]*)'
    + r' -> (?P<new>success|refusal [A-Z][A-Z_]*)'
)


def _class(outcome: dict[str, Any]) -> str:
    if outcome['result'] == 'success':
        return 'success'
    return f'refusal {outcome.get("code") or "UNNAMED"}'


def covered(vector_id: str, old: str, new: str) -> str | None:
    r"""The approved entry whose ``refs`` name the vector with exactly the
    transition ``old -> new``, if any."""
    document = yaml.safe_load(REGISTER.read_text(encoding='utf-8'))
    for entry in document['entries']:
        if entry.get('status') != 'approved':
            continue
        for ref in entry.get('refs') or []:
            found = TRANSITION.fullmatch(ref) if isinstance(ref, str) else None
            if (
                found is not None
                and (found['old'], found['new']) == (old, new)
                and fnmatch.fnmatchcase(vector_id, found['glob'])
            ):
                return str(entry['id'])
    return None


@pytest.fixture(scope='module')
def dsn() -> str | None:
    found = os.environ.get('SDM_POSTGRES_TEST_DSN')
    if not found:
        assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
            'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
        )
        return None
    return found


@pytest.mark.parametrize(
    'vector_id', [pytest.param(v['id'], id=v['id']) for v in P.VECTORS]
)
def test_python_core_answers_as_the_reference_on_postgres(
    vector_id: str, dsn: str | None
) -> None:
    if dsn is None:
        return
    vector = BY_ID[vector_id]
    want = vector['reference']
    got = P.observe(vector, dsn, core=True)
    if got == want:
        return
    old, new = _class(want['outcome']), _class(got['outcome'])
    if covered(vector_id, old, new) is not None:
        return
    parts = sorted(k for k in want if got.get(k) != want[k])
    anchors = [
        (side.get('payloads') or {}).get('anchors', [])[-1:]
        for side in (got, want)
    ]
    raise AssertionError(
        f'{vector_id}: {parts} differ: got {got["outcome"]}, the reference '
        f'{want["outcome"]}; latest anchor {anchors[0]}, the reference '
        f'{anchors[1]}; no approved entry declares {old} -> {new}'
    )


def test_the_vectors_are_the_reference_s_on_postgres() -> None:
    assert RECORDED['format'] == P.FORMAT
    assert RECORDED['recorded_by'] == (
        'contract/tools/postgres_regressions_record.py'
    )
    assert RECORDED['register_ref'] == P.REGISTER_REF
    assert RECORDED['environment']['postgres'] == 17
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    assert f'reference: {RECORDED["reference"]}' in env
    assert [v['id'] for v in RECORDED['vectors']] == [
        v['id'] for v in P.VECTORS
    ]
    for kind in P.VECTORS:
        vector = BY_ID[kind['id']]
        for key in kind:
            assert vector[key] == kind[key], (kind['id'], key)
    # Not vacuous: the reference answers each, a request made, every column
    # the type the vector names, read by name; the automatic anchor is the
    # one the review measured, in UTC.
    for vector in RECORDED['vectors']:
        answer = vector['reference']
        assert answer['outcome']['result'] == 'success', vector['id']
        assert answer['payloads'] is not None, vector['id']
    for name, (kind, _) in P.FEATURES.items():
        held = BY_ID[f'type.{name}']['reference']['postgres']
        assert held == {'users.feature': kind}, name
    anchors = BY_ID['time.new-york-auto-anchor']['reference']['payloads'][
        'anchors'
    ]
    assert anchors[-1] == '2024-04-10T00:00:00.000000Z'
    assert len(anchors) > 1
    time = BY_ID['time.new-york-auto-anchor']
    assert time['reference']['postgres'] == {
        'orders.ts': 'timestamp with time zone'
    }
    assert "'America/New_York'" in time['session'][0]
    assert BY_ID['relation.dollar']['reference']['postgres'] == {
        'users$1.id': 'integer'
    }
    # The second review's: each column the type its finding names.
    for vector_id, held in {
        'key.real-fractional-linked': {
            'items.item_id': 'real',
            'orders.item_id': 'real',
        },
        'key.real-fractional-entity': {
            'users.user_id': 'real',
            'orders.user_id': 'real',
        },
        'join.domain-not-null-text-key': {
            'users.user_id': 'text',
            'orders.n': 'qty',
        },
        'column.dollar': {'users.cost$1': 'integer'},
        'join.domain-not-null-text-key-column-x': {
            'users.user_id': 'text',
            'orders.n': 'qty',
            'orders.x': 'integer',
        },
        'type.timetz': {'users.x': 'time with time zone'},
    }.items():
        assert BY_ID[vector_id]['reference']['postgres'] == held, vector_id
