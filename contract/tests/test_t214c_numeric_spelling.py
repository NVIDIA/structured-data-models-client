# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.14c's Postgres regression on ``python-core`` (the package's test
author's): one number spelled two ways is one key.

    SDM_POSTGRES_TEST_DSN=... python -m pytest \
        contract/tests/test_t214c_numeric_spelling.py -q

Written from the whole-phase review's finding (round 1, Codex, P2; the
package's rides, item 6) and the reference (python-ref at
``contract/reference-env.txt``'s commit), before the fix. A key column of
Postgres's unconstrained ``numeric``, keyed and linked by the caller with no
constraint in the database, each user's ``user_id`` written ``i.0`` and each
order's ``i.00``: Postgres keeps each value's scale, so its text differs
while its number does not. The core read both ends as their text (P2G-2)
and took the two spellings for two keys; the second hop then reached a user
it had already reached and failed with ``INTERNAL_ERROR``, at the default
two hops and at ``num_hops=2``, where python-ref answers. The fix (the
rides: exact numeric equality in graph key identity, no float rounding, per
P2G-2 and P2C-3) makes ``1.0`` and ``1.00`` one key.

``contract/numeric-spelling-vectors.json`` is recorded from python-ref on
Postgres 17 by ``contract/tools/numeric_spelling_record.py``, whose
``observe`` (T2.21's) this test calls on the core's Postgres routes:
``Schema.from_source(PostgresSource(...), primary_keys=..., relationships=
...)`` and ``PredictiveQuery(...).materialize(...)`` over a database of the
vector's own. Each test's one line says what it proves:

- "the reproduction answers as the reference does": python-core answers
  ``given.default`` and ``given.hops-2`` (the review's reproduction) as
  python-ref answers the same tables with the keys declared and each link
  spelled as its key (``catalog.same-spelling.*``): its outcome, public
  schema (each ``user_id`` as text, P2G-2) and request (the payloads'
  digest). python-ref reads the keys given as doubles, so its own answer
  to the keys given is that one with each key a number: the same for
  either spelling (checked below), and P2G-2's text in place of the
  double;
- "keys declared in the catalog too": ``catalog.default`` and
  ``catalog.hops-2``, the same spellings with the keys declared, answered
  the same. python-ref answers these with each user reached twice (once by
  each spelling: it keys a declared ``numeric`` by its text), which is the
  defect the fix removes, so the core answers as for one spelling and the
  difference is registered;
- "the controls are unchanged": ``given.same-spelling`` and
  ``catalog.same-spelling.*``, each link spelled as its key, answered as
  python-ref answers ``catalog.same-spelling.*`` (they do today);
- "every difference from python-ref is registered": each vector whose
  expected answer is not python-ref's own answer to it is named, with
  ``success -> success``, by an approved register entry's refs.

``sdm-testkit``'s ``tests/t214c_numeric_spelling.rs`` holds ``rust-host``'s
graph key identity to the same: ``1.0`` and ``1.00`` one key, ``1.0`` and
``1.01`` two.

Gating: ``t214c-pending.json``'s ``pending`` names, as ``py:<key>``, the
keys whose test failed when written. Each must still fail (strict); once
one passes it fails, naming itself, until it is taken out. ``pending``
stays within ``authored``. Where ``SDM_POSTGRES_TEST_DSN`` names no server
and ``SDM_REQUIRE_POSTGRES`` is not set, the Postgres keys have nothing to
hold and return without asserting; with it set, a missing server fails
(H26).
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
VECTORS = ROOT / 'contract/numeric-spelling-vectors.json'
REGISTER = ROOT / 'docs/design/rust-core/register.yaml'
PENDING = Path(__file__).with_name('t214c-pending.json')


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


R = _module(
    'numeric_spelling_record',
    ROOT / 'contract/tools/numeric_spelling_record.py',
)
RECORDED: dict[str, Any] = json.loads(VECTORS.read_text(encoding='utf-8'))
BY_ID = {vector['id']: vector for vector in RECORDED['vectors']}
VECTOR_IDS = tuple(v['id'] for v in R.VECTORS)
#: The vectors whose expected answer is not python-ref's own answer to it.
DIFFERING = tuple(v for v in VECTOR_IDS if R.EXPECTED[v] != v)
RUST_KEYS = (
    'rust:spelling.same@memory-marked',
    'rust:spelling.equal@memory-marked',
    'rust:spelling.different@memory-marked',
)


def held(name: str, check: Any) -> None:
    r"""Runs ``check``: it must pass, or, while ``py:<name>`` is pending,
    fail."""
    gating = json.loads(PENDING.read_text(encoding='utf-8'))
    if f'py:{name}' not in gating['pending']:
        check()
        return
    try:
        check()
    except Exception:  # noqa: BLE001 - pending: failing, as when written
        return
    raise AssertionError(
        f'{name} passes now: take py:{name} out of '
        'contract/tests/t214c-pending.json pending'
    )


def test_the_pending_list_only_shrinks() -> None:
    r"""Control: ``pending`` names only keys the test author wrote, and
    ``authored`` is every vector's key and every differing vector's
    register key on python-core, and the Rust test's keys."""
    gating = json.loads(PENDING.read_text(encoding='utf-8'))
    assert set(gating['pending']) <= set(gating['authored'])
    assert set(gating['authored']) == (
        {f'py:vector:{v}' for v in VECTOR_IDS}
        | {f'py:register:{v}' for v in DIFFERING}
        | set(RUST_KEYS)
    )


def _server() -> str | None:
    r"""The Postgres server named; where none is, one must not be required
    (H26)."""
    found = os.environ.get('SDM_POSTGRES_TEST_DSN')
    if found:
        return found
    assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
        'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
    )
    return None


def _dtypes(answer: dict[str, Any]) -> dict[str, str]:
    return {
        table['name']: column['dtype']
        for table in answer['schema']['tables']
        for column in table['columns']
        if column['name'] == 'user_id'
    }


def test_the_vectors_are_the_reference_s_on_postgres() -> None:
    r"""The vectors are python-ref's on Postgres 17, recorded from the
    recorder's own vectors, and not vacuous: each key column an
    unconstrained ``numeric``; the users' keys written ``i.0`` and the
    orders' links ``i.00`` but in the ``same-spelling`` controls; the keys
    given only in ``given.*``; python-ref answering each, its answer to the
    keys given the same whatever the link's spelling and the hops (the
    reproduction succeeds there), and its answer to the keys declared
    changed by the link's spelling (the defect a declared key shows on
    python-ref, which the core does not copy)."""
    assert RECORDED['format'] == R.FORMAT
    assert RECORDED['recorded_by'] == (
        'contract/tools/numeric_spelling_record.py'
    )
    assert RECORDED['register_ref'] == R.REGISTER_REF
    assert RECORDED['environment']['postgres'] == 17
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    assert f'reference: {RECORDED["reference"]}' in env
    assert [v['id'] for v in RECORDED['vectors']] == list(VECTOR_IDS)
    for kind in R.VECTORS:
        for name in kind:
            assert BY_ID[kind['id']][name] == kind[name], (kind['id'], name)
    assert set(R.EXPECTED) == set(VECTOR_IDS)
    assert set(R.EXPECTED.values()) == {
        'catalog.same-spelling.default',
        'catalog.same-spelling.hops-2',
    }
    for vector in RECORDED['vectors']:
        vector_id = vector['id']
        answer = vector['reference']
        assert answer['postgres'] == {
            'users.user_id': 'numeric',
            'orders.user_id': 'numeric',
        }, vector_id
        assert answer['outcome']['result'] == 'success', vector_id
        assert answer['payloads'] is not None, vector_id
        setup = '\n'.join(vector['setup'])
        assert "|| '.0')::numeric, 20 + i" in setup, vector_id
        link = '.0' if 'same-spelling' in vector_id else '.00'
        assert f"::text || '{link}')::numeric, 10 + i" in setup, vector_id
        declared = vector_id.startswith('catalog.')
        assert ('PRIMARY KEY' in setup) == declared, vector_id
        assert ('REFERENCES' in setup) == declared, vector_id
        if declared:
            assert vector['from_source'] == {}, vector_id
            assert _dtypes(answer) == {'users': 'string', 'orders': 'string'}
        else:
            assert vector['from_source'] == {
                'primary_keys': {'users': 'user_id', 'orders': 'order_id'},
                'relationships': [R.LINK],
            }, vector_id
            # python-ref reads a key the caller gives as doubles.
            assert _dtypes(answer) == {
                'users': 'float64',
                'orders': 'float64',
            }, vector_id
        hops = vector['materialize'].get('num_hops')
        assert hops == (2 if vector_id.endswith('hops-2') else None), vector_id
        assert vector['materialize']['indices'] == ['0.0', '1.0']

    def digest(vector_id: str) -> str:
        return str(BY_ID[vector_id]['reference']['payloads']['digest'])

    # The keys given: one answer, whatever the link's spelling or the hops.
    assert digest('given.default') == digest('given.hops-2')
    assert digest('given.default') == digest('given.same-spelling')
    # The keys declared: one answer at either number of hops, but another
    # where the link is spelled otherwise than its key.
    assert digest('catalog.default') == digest('catalog.hops-2')
    assert digest('catalog.same-spelling.default') == digest(
        'catalog.same-spelling.hops-2'
    )
    assert digest('catalog.default') != digest('catalog.same-spelling.default')
    # And the answer expected differs from python-ref's own exactly where
    # DIFFERING says.
    for vector_id in VECTOR_IDS:
        expected = R.EXPECTED[vector_id]
        assert (
            BY_ID[vector_id]['reference'] != BY_ID[expected]['reference']
        ) == (vector_id in DIFFERING), vector_id


def _projection(answer: dict[str, Any]) -> dict[str, Any]:
    r"""An answer as it is held to the one expected: its outcome, public
    schema and request (the payloads' digest and anchors)."""
    return {k: answer[k] for k in ('outcome', 'schema', 'payloads')}


@pytest.mark.parametrize('vector_id', VECTOR_IDS)
def test_python_core_answers_one_number_as_one_key(vector_id: str) -> None:
    r"""Exits "the reproduction answers as the reference does", "keys
    declared in the catalog too" and "the controls are unchanged":
    python-core answers the vector exactly as python-ref answers
    ``EXPECTED``'s vector, the same tables with the keys declared and each
    link spelled as its key: the outcome, the public schema and the request;
    and each key column is the Postgres type the vector names."""
    dsn = _server()
    if dsn is None:
        return

    def check() -> None:
        expected = R.EXPECTED[vector_id]
        want = _projection(BY_ID[expected]['reference'])
        observed = R.observe(BY_ID[vector_id], dsn, True)
        assert observed['postgres'] == BY_ID[vector_id]['reference']['postgres']
        got = _projection(observed)
        found = sorted(k for k in want if got.get(k) != want[k])
        assert not found, (
            f'{vector_id}: {found} differ from python-ref on {expected}: got '
            f'{got["outcome"]}'
        )

    held(f'vector:{vector_id}', check)


TRANSITION = re.compile(
    re.escape(R.REGISTER_REF)
    + r'(?P<glob>\S+) (?P<old>success|refusal [A-Z][A-Z_]*)'
    + r' -> (?P<new>success|refusal [A-Z][A-Z_]*)'
)


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


@pytest.mark.parametrize('vector_id', DIFFERING)
def test_each_difference_from_the_reference_is_registered(
    vector_id: str,
) -> None:
    r"""Exit "every difference from python-ref is registered": where the
    answer expected is not python-ref's own (a key given read as text, not
    doubles, P2G-2; a declared key's two spellings one key, not two), an
    approved entry names the vector with ``success -> success``."""

    def check() -> None:
        assert covered(vector_id, 'success', 'success'), (
            f'no approved entry names {R.REGISTER_REF}{vector_id} '
            'success -> success'
        )

    held(f'register:{vector_id}', check)


def test_a_transition_names_its_vectors_alone() -> None:
    r"""Control: :data:`TRANSITION` reads a ref of this file alone, with
    its glob and both outcomes."""
    ref = f'{R.REGISTER_REF}given.* success -> success'
    found = TRANSITION.fullmatch(ref)
    assert found is not None
    assert (found['glob'], found['old'], found['new']) == (
        'given.*',
        'success',
        'success',
    )
    assert fnmatch.fnmatchcase('given.hops-2', found['glob'])
    assert not fnmatch.fnmatchcase('catalog.default', found['glob'])
    other = 'contract/exact-key-vectors.json#given.* success -> success'
    assert TRANSITION.fullmatch(other) is None
