# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T2.3's key types on Postgres, on ``python-core``, held to the reference.

    SDM_POSTGRES_TEST_DSN=... python -m pytest \
        contract/tests/test_t23_postgres_keys.py -q

Written by T2.3's independent test author from the design's row ("the
key-type pair matrix on Postgres", "the decimal-key pair matrix, equal keys
across scales included", "key sets bound as the dialect binds them"),
choice 5 and lesson 19, before the implementation.
``contract/postgres-key-vectors.json`` is recorded from the reference on
Postgres by ``contract/tools/postgres_keys_record.py``, whose ``run``,
``project`` and ``joined`` this test calls, so both routes are observed the
same way: each vector a case run by ``run_python.py --database postgres``,
its tables loaded into a new Postgres database by the contract loader, the
public calls over a ``psycopg`` connection, here forced onto the core's
Postgres routes.

- Exit "the key-type pair matrix on Postgres" and "the decimal-key pair
  matrix": every vector's public answer on ``python-core`` is the
  reference's (outcome, schema, warnings, each request's digest and tables,
  the result, the key columns), and where the outcome is the reference's,
  so are the statements per public call. An approved register entry
  excuses a difference only as its ``refs`` declare it, each naming
  vectors by id or glob: ``contract/postgres-key-vectors.json#<id or
  glob> <from> -> <to>`` (each side ``success`` or ``refusal <CODE>``)
  declares the answer differs with that transition, its counts with it;
  ``contract/postgres-key-vectors.json#<id or glob> statements`` declares
  the statement counts alone differ, every answer the reference's (as
  P2C-2's fewer Postgres statements, KO2-58), and excuses no answer
  difference. Without such a ref a difference fails, and a proposed entry
  excuses nothing.
- Lesson 19, "a wrong row is P1": every pair vector (``pair.*``,
  ``key.pair*``, ``fifteen.*``) the core answers sends requests that join
  the same rows to the same entities as the same-type control
  (``pair.i64.i64``, as the reference sends it), every key read exactly;
  a pair the core refuses by name sends nothing to judge. So keys equal
  across scales join (``key.pair.d10-2.d12-4`` and the others), and keys
  float64 would merge stay distinct (``key.pair-merging.*``,
  ``fifteen.*``, KO2-35), where the reference merges them.
- The vectors themselves: recorded at the reference of
  ``contract/reference-env.txt`` on Postgres 17; every vector of the
  recorder present with its inputs; each key column stored as the Postgres
  type its name says (``format_type``, by name); the register marks the
  recorder derives; and not vacuous: the reference joins the equal-valued
  decimal pairs as the control and merges the fifteen-digit keys, so the
  pair tests judge something.

Where ``SDM_POSTGRES_TEST_DSN`` names no server the tests that need one
are skipped, unless ``SDM_REQUIRE_POSTGRES`` is set (a Postgres job), where
they fail (H26).

Expected to fail until T2.3's implementation lands: the runner refuses a
``python-core`` run on Postgres while the core reports no Postgres provider
(``sdm_client._native.PROVIDERS``), and the routing has no core route for
the Postgres paths.
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
VECTORS = ROOT / 'contract/postgres-key-vectors.json'
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
    'postgres_keys_record', ROOT / 'contract/tools/postgres_keys_record.py'
)
RECORDED: dict[str, Any] = json.loads(VECTORS.read_text(encoding='utf-8'))
BY_ID = {vector['id']: vector for vector in RECORDED['vectors']}
#: The inputs a vector carries, each as the recorder wrote it.
INPUTS = ('columns', 'setup', 'indices', 'indices_as', 'offset', 'unit')
#: The pair vectors: two key columns of different (or, on the diagonal, the
#: same) types holding the shop's keys, which must join as one type does.
PAIRS = [
    v['id']
    for v in RECORDED['vectors']
    if v['id'].startswith(('pair.', 'key.pair', 'fifteen.'))
]
TRANSITION = re.compile(
    re.escape(P.REGISTER_REF)
    + r'(?P<glob>\S+) (?P<old>success|refusal [A-Z][A-Z_]*)'
    + r' -> (?P<new>success|refusal [A-Z][A-Z_]*)'
)
#: A ref declaring that the statement counts alone differ.
COUNTS = re.compile(re.escape(P.REGISTER_REF) + r'(?P<glob>\S+) statements')


def _entries() -> list[dict[str, Any]]:
    document = yaml.safe_load(REGISTER.read_text(encoding='utf-8'))
    return list(document['entries'])


def covered(vector_id: str, old: str | None, new: str | None) -> str | None:
    r"""The approved entry whose ``refs`` name the vector with exactly the
    transition ``old -> new``, or, with both None, as differing in its
    statement counts alone; if any."""
    for entry in _entries():
        if entry.get('status') != 'approved':
            continue
        for ref in entry.get('refs') or []:
            if not isinstance(ref, str):
                continue
            if old is None:
                found = COUNTS.fullmatch(ref)
            else:
                found = TRANSITION.fullmatch(ref)
                if found is not None and (found['old'], found['new']) != (
                    old,
                    new,
                ):
                    found = None
            if found is not None and fnmatch.fnmatchcase(
                vector_id, found['glob']
            ):
                return str(entry['id'])
    return None


def problem_of(
    vector: dict[str, Any], got: dict[str, Any], counted: dict[str, int]
) -> str | None:
    r"""Why ``got`` and the statements counted fail the vector, or None:
    the answer the reference's, its counts too or excused by a counts-only
    ref; or the answer different exactly as an approved transition ref
    declares, which excuses its counts with it."""
    want = vector['reference']
    if got == want:
        if counted == vector['statements']:
            return None
        if covered(vector['id'], None, None) is not None:
            return None
        return (
            f'{vector["id"]}: statements {counted}, the reference '
            f'{vector["statements"]}; no approved entry declares '
            f'{P.REGISTER_REF}{vector["id"]} statements'
        )
    old = P.outcome_class(want['outcome'])
    new = P.outcome_class(got['outcome'])
    if covered(vector['id'], old, new) is not None:
        return None
    parts = sorted(k for k in want if got.get(k) != want[k])
    return (
        f'{vector["id"]}: {parts} differ: got {got["outcome"]}, the '
        f'reference {want["outcome"]}; no approved entry declares '
        f'{old} -> {new}'
    )


def _server() -> None:
    r"""Skip where no Postgres server is named, unless one is required."""
    if os.environ.get('SDM_POSTGRES_TEST_DSN'):
        return
    assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
        'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
    )
    pytest.skip('SDM_POSTGRES_TEST_DSN names no Postgres server')


@pytest.fixture(scope='module')
def core_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    r"""Every vector through ``python-core`` on Postgres, forced onto the
    core's Postgres routes: one run, or the reason the run failed whole."""
    _server()
    tmp = tmp_path_factory.mktemp('t23-keys')
    try:
        return P.run('python-core', P.VECTORS, tmp, core=True)
    except RuntimeError as error:
        return {'': str(error)}


def _observation(core_runs: dict[str, Any], vector_id: str) -> dict[str, Any]:
    if '' in core_runs:
        pytest.fail(core_runs[''])
    observation = core_runs[vector_id]
    assert observation['core'] is not None, (
        f'{vector_id}: the core was not reached on Postgres'
    )
    return observation


@pytest.mark.parametrize(
    'vector_id',
    [pytest.param(v['id'], id=v['id']) for v in RECORDED['vectors']],
)
def test_python_core_answers_each_key_vector_as_the_reference(
    vector_id: str, core_runs: dict[str, Any]
) -> None:
    vector = BY_ID[vector_id]
    observation = _observation(core_runs, vector_id)
    got = P.project(observation, vector)
    problem = problem_of(vector, got, P.statements(observation))
    assert problem is None, problem


@pytest.mark.parametrize('vector_id', [pytest.param(i, id=i) for i in PAIRS])
def test_a_key_pair_joins_as_one_type_does_on_the_core(
    vector_id: str, core_runs: dict[str, Any]
) -> None:
    vector = BY_ID[vector_id]
    observation = _observation(core_runs, vector_id)
    if observation['outcome']['result'] != 'success':
        # A pair refused by name sends nothing to judge here; whether it
        # may be refused is the parity test's (the reference's answer or a
        # registered transition), and a refusal without a code is no name.
        assert observation['outcome']['code'] is not None, (
            f'{vector_id}: refused without a code: {observation["outcome"]}'
        )
        return
    control = BY_ID[P.CONTROL]['joined']
    assert P.joined(observation, vector) == control, (
        f'{vector_id}: the core joins other rows, or other entities, than '
        f'the same-type control {P.CONTROL} (lesson 19: a wrong row)'
    )


# --- the vectors ---------------------------------------------------------------


def test_the_vectors_are_the_reference_s_on_postgres() -> None:
    assert RECORDED['format'] == P.FORMAT
    assert RECORDED['recorded_by'] == 'contract/tools/postgres_keys_record.py'
    assert RECORDED['cell'] == 'ref-pandas3'
    assert RECORDED['register_ref'] == P.REGISTER_REF
    assert RECORDED['control'] == P.CONTROL
    assert RECORDED['question'] == P.QUESTION
    assert RECORDED['indices'] == P.INDICES
    assert RECORDED['anchor_time'] == P.ANCHOR
    assert RECORDED['environment']['postgres'] == 17
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    assert f'reference: {RECORDED["reference"]}' in env
    assert [v['id'] for v in RECORDED['vectors']] == [
        v['id'] for v in P.VECTORS
    ]
    for kind in P.VECTORS:
        vector = BY_ID[kind['id']]
        for key in INPUTS:
            assert vector.get(key) == kind.get(key), (kind['id'], key)
        # Derived from the reference's answer, never edited.
        why = P.register(vector['reference'])
        assert vector['register'] == {'required': bool(why), 'why': why}, kind[
            'id'
        ]
        assert set(vector['statements']) == {
            'from_source',
            'materialize',
            'predict',
        }
        assert vector['statements']['from_source'] > 0, kind['id']


def test_each_key_column_is_stored_as_its_postgres_type() -> None:
    # Not vacuous: each key column is the Postgres type the vector's name
    # says, read by name from the loaded database; the matrix holds every
    # ordered pair of the nine types.
    tags = [tag for tag, _, _ in P.TYPES]
    assert {v for v in PAIRS if v.startswith('pair.')} == {
        f'pair.{a}.{b}' for a in tags for b in tags
    }
    assert len({pg for _, _, pg in P.TYPES}) == len(P.TYPES)
    for vector in RECORDED['vectors']:
        held = vector['postgres']
        assert list(held) == [f'{t}.{c}' for t, c in vector['columns']], vector[
            'id'
        ]
        if vector['id'].startswith('pair.'):
            assert [held['users.user_id'], held['orders.user_id']] == (
                vector['postgres_types']
            ), vector['id']
            continue
        # A decimal key the setup gives a column is Postgres's numeric of
        # the same precision and scale; a column left alone is the shop's
        # bigint.
        for name, kind in held.items():
            table, column = name.split('.')
            wanted = 'bigint'
            for statement in vector['setup']:
                found = re.match(
                    rf'ALTER TABLE {table} ALTER {column} TYPE '
                    r'DECIMAL\((\d+), (\d+)\)',
                    statement,
                )
                if found:
                    wanted = f'numeric({found[1]},{found[2]})'
            assert kind == wanted, (vector['id'], held)
    across = BY_ID['key.pair.d10-2.d12-4']['postgres']
    assert across == {
        'users.user_id': 'numeric(10,2)',
        'orders.user_id': 'numeric(12,4)',
    }
    fifteen = BY_ID['fifteen.d17-16.d18-17']['postgres']
    assert fifteen == {
        'users.user_id': 'numeric(17,16)',
        'orders.user_id': 'numeric(18,17)',
    }


def test_the_pair_matrices_judge_something() -> None:
    control = BY_ID[P.CONTROL]
    assert control['reference']['outcome']['result'] == 'success'
    assert control['joined'] is not None
    # The reference joins equal decimal keys across scales as one type
    # does, both ways: the core must too.
    for vector_id in (
        'key.pair.d10-2.d12-4',
        'key.pair.d12-4.d10-2',
        'key.pair.d18-0.d20-2',
        'key.pair.d20-2.d18-0',
    ):
        assert BY_ID[vector_id]['joined'] == control['joined'], vector_id
    # Every other pair the reference answers joins as the control too,
    # whatever the two types, but where float64 merges the keys: 10**17 + i
    # and keys equal to 15 significant digits, where the reference joins
    # other rows than the control, which the core may not (KO2-35).
    merging = {
        'key.pair-merging.d38-0.d38-2',
        'key.pair-merging.d38-2.d38-0',
        'fifteen.d17-16.d17-16',
        'fifteen.d17-16.d18-17',
    }
    answered = {
        vector_id
        for vector_id in PAIRS
        if BY_ID[vector_id]['reference']['outcome']['result'] == 'success'
    }
    assert {
        vector_id
        for vector_id in answered
        if BY_ID[vector_id]['joined'] != control['joined']
    } == merging
    # Both float types and text among the pairs answered.
    assert {'pair.f32.f64', 'pair.f64.f32', 'pair.text.text'} <= answered
    for vector_id in sorted(merging):
        reference = BY_ID[vector_id]['reference']
        assert reference['outcome']['result'] == 'success', vector_id
        assert BY_ID[vector_id]['joined'] != control['joined'], vector_id
