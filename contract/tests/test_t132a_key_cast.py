# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.32a's cast of the entities to the key's type on ``python-core``, held
to the reference's rule-table vectors.

    python -m pytest contract/tests/test_t132a_key_cast.py -q

Written by the package's independent test author from the brief, the
design's ledger row ("The entities cast to the key's type") and the
reference, before the implementation. ``contract/key-cast-vectors.json``
is recorded from the reference by ``contract/tools/key_cast_record.py``,
whose ``observe`` this test calls, so both routes are observed by the same
code; ``sdm-testkit``'s ``tests/t132a_key_cast.rs`` holds ``rust-host`` to
the same vectors, so the two hosts agree through them.

- Exit "the entities' cast runs in the core, both hosts giving the same
  answers and error codes over the rule table's vectors": each vector's
  key type described and materialized forced onto the core
  (``SDM_CORE_ROUTES``: ``Schema.from_source``'s and ``materialize``'s
  DuckDB routes) with the vector's entities as a caller passes them. The
  answer must be the reference's: a request whose predicted rows carry the
  reference's entities, in order, or a refusal of the same class, code and
  stage. INVALID_REQUEST against INTERNAL_ERROR is part of the answer
  (E41(l)).
- A vector the recorder marked ``register.required`` (the reference fails
  with INTERNAL_ERROR, its pandas cast changes an entity's kind or value,
  or the core cannot tell it from another vector the reference answers
  otherwise) may differ only where an entry of
  ``docs/design/rust-core/register.yaml``, proposed or approved, names it
  in its ``refs`` (``contract/key-cast-vectors.json#<id or glob>``); every
  other vector is held to the reference exactly, and no vector may end in
  ``UNSUPPORTED_REQUEST`` (a rule the core does not take yet).
- The vectors themselves: recorded at the reference of
  ``contract/reference-env.txt``, every key type and input of the recorder
  present, each input read back exactly, the host's column and the
  register marks what the recorder derives (so neither is edited by hand),
  and not vacuous.

Gating (``contract/tests/t132a_gate.py``): ``t132a-pending.json``'s
``pending["python-core"]`` lists the vectors that failed here when this was
written. Each still runs and must still fail; one that passes fails, naming
itself, until it is taken out of the list, which may only shrink (within
``authored["python-core"]``). The implementer removes ids as they pass and
changes nothing else of this file, the vectors or the recorder.
"""

from __future__ import annotations

import fnmatch
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
VECTORS = ROOT / 'contract/key-cast-vectors.json'
REGISTER = ROOT / 'docs/design/rust-core/register.yaml'


def _recorder() -> ModuleType:
    name = 'key_cast_record'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / 'contract/tools/key_cast_record.py'
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _gate() -> ModuleType:
    name = 't132a_gate'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / 'contract/tests/t132a_gate.py'
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


R = _recorder()
G = _gate()
RECORDED: dict[str, Any] = json.loads(VECTORS.read_text(encoding='utf-8'))
KEYS = {key['id']: key for key in RECORDED['keys']}


def registered(vector_id: str) -> str | None:
    r"""The register entry, proposed or approved, whose ``refs`` name the
    vector (by id or a glob of ids), or None."""
    document = yaml.safe_load(REGISTER.read_text(encoding='utf-8'))
    for entry in document['entries']:
        if entry.get('status') not in ('proposed', 'approved'):
            continue
        for ref in entry.get('refs') or []:
            if not isinstance(ref, str) or not ref.startswith(R.REGISTER_REF):
                continue
            if fnmatch.fnmatchcase(vector_id, ref[len(R.REGISTER_REF) :]):
                return str(entry['id'])
    return None


def problem_of(vector: dict[str, Any], got: dict[str, Any]) -> str | None:
    r"""Why ``got`` fails the vector, or None when it is the reference's
    answer, or a marked vector's registered difference."""
    want = vector['reference']['public']
    if got.get('code') == 'UNSUPPORTED_REQUEST':
        return f'refused by a rule the core does not take yet: {got}'
    if got == want:
        return None
    if vector['register']['required'] and registered(vector['id']):
        return None
    marked = (
        'marked register entry required, and no entry names it'
        if vector['register']['required']
        else 'held to the reference exactly'
    )
    return f'{vector["id"]}: got {got}, the reference {want} ({marked})'


def _cases() -> list[Any]:
    return [
        pytest.param(vector, id=vector['id']) for vector in RECORDED['vectors']
    ]


@pytest.fixture
def core_route(monkeypatch: pytest.MonkeyPatch) -> None:
    from sdm_client import _routing

    monkeypatch.setenv(
        _routing.ENVIRONMENT,
        ','.join(_routing.FROM_SOURCE_DUCKDB + _routing.MATERIALIZE_DUCKDB),
    )


@pytest.mark.parametrize('vector', _cases())
def test_python_core_casts_the_entities_as_the_reference(
    vector: dict[str, Any], core_route: None
) -> None:
    def check() -> None:
        values = [R.decode(tagged) for tagged in vector['input']]
        got = R.observe(KEYS[vector['key']], values)
        problem = problem_of(vector, got)
        assert problem is None, problem

    G.held('python-core', vector['id'], check)


# --- the vectors ---------------------------------------------------------------


def test_the_vectors_are_the_reference_s() -> None:
    assert RECORDED['format'] == R.FORMAT
    assert RECORDED['recorded_by'] == 'contract/tools/key_cast_record.py'
    assert RECORDED['cell'] == 'ref-pandas3'
    assert RECORDED['register_ref'] == R.REGISTER_REF
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    assert f'reference: {RECORDED["reference"]}' in env
    # Every key type and every input the recorder names, each once.
    assert [key['id'] for key in RECORDED['keys']] == [
        key['id'] for key in R.KEYS
    ]
    for key in R.KEYS:
        assert KEYS[key['id']]['setup'] == key['setup']
    wanted = [f'{key["id"]}.{name}' for key in R.KEYS for name in R.inputs(key)]
    assert [v['id'] for v in RECORDED['vectors']] == wanted


def test_each_input_reads_back_exactly() -> None:
    for vector in RECORDED['vectors']:
        values = [R.decode(tagged) for tagged in vector['input']]
        assert [R.encode(v) for v in values] == vector['input'], vector['id']


def test_the_host_column_and_the_marks_are_the_recorder_s() -> None:
    # Derived, never edited: the host's natural column (pyarrow at the
    # reference cell's pin) and the register marks from the answers.
    import copy

    vectors = copy.deepcopy(RECORDED['vectors'])
    for vector in vectors:
        values = [R.decode(tagged) for tagged in vector['input']]
        assert R.host_arrow(values) == vector['host_arrow'], vector['id']
        why = R.register(values, vector['reference']['public'])
        vector['register'] = {'required': bool(why), 'why': why}
    R.alike(vectors)
    for derived, recorded in zip(vectors, RECORDED['vectors'], strict=True):
        assert derived['register'] == recorded['register'], derived['id']


def test_the_vectors_are_not_vacuous() -> None:
    answers = [v['reference']['public'] for v in RECORDED['vectors']]
    outcomes = [(a['outcome'], a.get('code')) for a in answers]
    assert outcomes.count(('success', None)) > 100
    assert outcomes.count(('refusal', 'INVALID_REQUEST')) > 100
    assert outcomes.count(('refusal', 'INTERNAL_ERROR')) > 100
    assert set(outcomes) == {
        ('success', None),
        ('refusal', 'INVALID_REQUEST'),
        ('refusal', 'INTERNAL_ERROR'),
    }
    marked = [v for v in RECORDED['vectors'] if v['register']['required']]
    assert 0 < len(marked) < len(RECORDED['vectors'])
    for vector in RECORDED['vectors']:
        assert vector['register']['required'] == bool(vector['register']['why'])
    # The reference's own cast was observed wherever its engine cast.
    assert all(v['reference']['cast'] for v in RECORDED['vectors'])
    # Each key type has successes and refusals of both codes, or fails
    # everywhere with INTERNAL_ERROR (the reference's decimal and
    # large_string keys).
    for key in KEYS:
        mine = {
            (
                v['reference']['public']['outcome'],
                v['reference']['public'].get('code'),
            )
            for v in RECORDED['vectors']
            if v['key'] == key
        }
        assert ('success', None) in mine or mine == {
            ('refusal', 'INTERNAL_ERROR'),
            ('refusal', 'INVALID_REQUEST'),
        }, key


# --- the gating ----------------------------------------------------------------


def test_the_pending_lists_only_shrink() -> None:
    assert G.only_shrinks() == []
    ids = {v['id'] for v in RECORDED['vectors']}
    for host in ('python-core', 'rust-host'):
        assert G.pending(host) <= ids, f'{host}: ids that are no vector'


def test_the_gate_holds_a_pending_test_to_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A listed test passes only while its check fails, and an unlisted one
    # only while its check passes.
    def passing() -> None:
        pass

    def failing() -> None:
        raise AssertionError('seeded')

    monkeypatch.setattr(G, 'pending', lambda group: {'seeded.listed'})
    G.held('python-core', 'seeded.listed', failing)
    with pytest.raises(AssertionError, match='passes now'):
        G.held('python-core', 'seeded.listed', passing)
    G.held('python-core', 'seeded.unlisted', passing)
    with pytest.raises(AssertionError, match='seeded'):
        G.held('python-core', 'seeded.unlisted', failing)


def test_a_difference_is_held_unless_marked_and_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Fail closed: the comparison itself, on a seeded difference.
    vector = next(
        v for v in RECORDED['vectors'] if not v['register']['required']
    )
    marked = next(v for v in RECORDED['vectors'] if v['register']['required'])
    other = {
        'outcome': 'refusal',
        'stage': 'materialize',
        'error': 'StructuredDataError',
        'code': 'NOT_THE_REFERENCE_S',
    }
    assert problem_of(vector, vector['reference']['public']) is None
    assert problem_of(vector, other) is not None
    unported = dict(other, code='UNSUPPORTED_REQUEST')
    assert problem_of(marked, unported) is not None
    monkeypatch.setattr(sys.modules[__name__], 'registered', lambda _: None)
    assert problem_of(marked, other) is not None
    monkeypatch.setattr(
        sys.modules[__name__], 'registered', lambda _: 'SEEDED-1'
    )
    assert problem_of(marked, other) is None
    # An unmarked vector is never excused by an entry.
    assert problem_of(vector, other) is not None
