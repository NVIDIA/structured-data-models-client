# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.32c's nested column types on ``python-core``, held to the reference's
vectors.

    python -m pytest contract/tests/test_t132c_nested_types.py -q

Written by the package's independent test author from the brief (lift the
``column_type`` refusal this family owns, so nested columns are described,
materialized and predicted as the reference does, or refused by name where
registered), ``contract/refusal-maps/nested.json`` and the reference, before
the implementation. ``contract/nested-type-vectors.json`` is recorded from
the reference by ``contract/tools/nested_type_record.py``, whose ``run``
and ``project`` this test calls, so both routes are observed and projected
by the same code; ``sdm-testkit``'s ``tests/t132c_nested_types.rs`` holds
``rust-host`` to the same vectors, so the two hosts agree through them.

- Exits "``python-core`` and ``rust-host`` agree on ``core`` fields" and
  "every difference is registered", over the family's types: each vector
  (``shop`` with a column ``f`` of one DuckDB type: lists of each scalar
  type, empty and null lists, nested lists, fixed-size lists, structs,
  maps, a union, binary and interval) run by ``run_python.py --mode
  python-core`` through ``Schema.from_source``, ``materialize`` and
  ``predict``, forced onto every DuckDB route of the three. Its public
  answer must be the reference's: the outcome (result, stage, class,
  code), what the schema made of ``f`` (described, left out), the warning
  classes, and ``f`` in the request the endpoint received.
- A vector the recorder marked ``register.required`` (the reference fails
  unnamed or with INTERNAL_ERROR) may differ only where an entry of
  ``docs/design/rust-core/register.yaml``, proposed or approved, names it
  in its ``refs`` (``contract/nested-type-vectors.json#<id or glob>``);
  every other vector is held to the reference exactly. No vector may end
  in ``UNSUPPORTED_REQUEST``, a rule the core does not take yet: the brief
  lifts ``column_type`` for this family.
- The vectors themselves: recorded at the reference of
  ``contract/reference-env.txt``, every type of the recorder present with
  its setup, the register marks what the recorder derives, and not vacuous.

Gating (``contract/tests/t132c_gate.py``): ``t132c-pending.json``'s
``pending["python-core"]`` lists the vectors that failed here when this was
written. Each still runs and must still fail; one that passes fails, naming
itself, until it is taken out of the list, which may only shrink (within
``authored["python-core"]``). The implementer removes ids as they pass and
changes nothing else of this file, the vectors or the recorder.
"""

from __future__ import annotations

import copy
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
VECTORS = ROOT / 'contract/nested-type-vectors.json'
REGISTER = ROOT / 'docs/design/rust-core/register.yaml'


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


R = _module('nested_type_record', ROOT / 'contract/tools/nested_type_record.py')
G = _module('t132c_gate', ROOT / 'contract/tests/t132c_gate.py')
RECORDED: dict[str, Any] = json.loads(VECTORS.read_text(encoding='utf-8'))
BY_ID = {vector['id']: vector for vector in RECORDED['vectors']}


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
    want = vector['reference']
    if got['outcome'].get('code') == 'UNSUPPORTED_REQUEST':
        return f'refused by a rule the core does not take yet: {got["outcome"]}'
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


@pytest.fixture(scope='module')
def core_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    r"""Every vector through ``python-core``, forced onto the core: one run."""
    from sdm_client import _routing

    routes = list(
        _routing.FROM_SOURCE_DUCKDB
        + _routing.MATERIALIZE_DUCKDB
        + _routing.PREDICT_DUCKDB
    )
    tmp = tmp_path_factory.mktemp('t132c-vectors')
    return R.run('python-core', R.TYPES, tmp, routes=routes)


@pytest.mark.parametrize(
    'vector_id',
    [pytest.param(v['id'], id=v['id']) for v in RECORDED['vectors']],
)
def test_python_core_handles_the_nested_type_as_the_reference(
    vector_id: str, core_runs: dict[str, Any]
) -> None:
    vector = BY_ID[vector_id]

    def check() -> None:
        observation = core_runs[vector_id]
        assert observation['core'] is not None, 'the core was not reached'
        got = R.project(observation, vector)
        problem = problem_of(vector, got)
        assert problem is None, problem

    G.held('python-core', vector_id, check)


# --- the vectors ---------------------------------------------------------------


def test_the_vectors_are_the_reference_s() -> None:
    assert RECORDED['format'] == R.FORMAT
    assert RECORDED['recorded_by'] == 'contract/tools/nested_type_record.py'
    assert RECORDED['cell'] == 'ref-pandas3'
    assert RECORDED['register_ref'] == R.REGISTER_REF
    assert RECORDED['template'] == 'contract/cases/s1/nested.list-int.json'
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    assert f'reference: {RECORDED["reference"]}' in env
    # Every type the recorder names, once, with its table and setup.
    assert [v['id'] for v in RECORDED['vectors']] == [t['id'] for t in R.TYPES]
    for kind in R.TYPES:
        vector = BY_ID[kind['id']]
        assert vector['table'] == kind['table'], kind['id']
        assert vector['setup'] == kind['setup'], kind['id']


def test_the_marks_are_the_recorder_s() -> None:
    # Derived from the reference's answer, never edited.
    for vector in RECORDED['vectors']:
        why = R.register(vector['reference'])
        assert vector['register'] == {'required': bool(why), 'why': why}, (
            vector['id']
        )


def test_each_vector_is_a_case_the_runner_takes() -> None:
    # The cases run_python.py is handed validate, ids unique, and differ
    # from the template only where the vector does.
    import tempfile

    runner = _module('run_python', ROOT / 'contract/runners/run_python.py')
    template = json.loads(R.TEMPLATE.read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory() as tmp:
        for kind in R.TYPES:
            case = R.case_for(kind)
            assert case['input']['setup'] == kind['setup']
            assert case['input']['pql'] == template['input']['pql']
            assert case['core'] == template['core']
            (Path(tmp) / f'{case["id"]}.json').write_text(
                json.dumps(case), encoding='utf-8'
            )
        loaded = runner.load_cases([Path(tmp)])
    assert len(loaded) == len(R.TYPES)


def test_the_vectors_are_not_vacuous() -> None:
    answers = [v['reference'] for v in RECORDED['vectors']]
    successes = [a for a in answers if a['outcome']['result'] == 'success']
    refusals = [a for a in answers if a['outcome']['result'] == 'refusal']
    assert len(successes) > 20 and len(refusals) > 20
    # A success reached predict's result, its f left out and out of the
    # request; a refusal stopped where the reference stopped.
    for answer in successes:
        assert answer['outcome']['stage'] == 'predict'
        assert answer['payload'] is not None
    marked = [v for v in RECORDED['vectors'] if v['register']['required']]
    assert 0 < len(marked) < len(RECORDED['vectors'])
    # The kinds the brief names are all here.
    ids = set(BY_ID)
    for prefix in ('list.', 'array.', 'struct.', 'map.', 'union.', 'scalar.'):
        assert any(i.startswith(prefix) for i in ids), prefix
    for wanted in (
        'list.all-empty',
        'list.all-null',
        'list.list-integer',
        'scalar.blob',
        'scalar.interval',
    ):
        assert wanted in ids
    # DuckDB's own types were seen, so each vector is the type it names.
    arrow = {v['id']: v['duckdb_arrow'] for v in RECORDED['vectors']}
    assert arrow['list.integer-large'].startswith('large_list<')
    assert arrow['list.integer-view'].startswith('list_view<')
    assert arrow['array.bigint-2'].startswith('fixed_size_list<')
    assert arrow['map.varchar-integer'].startswith('map<')
    assert arrow['scalar.blob'] == 'binary'


# --- the gating ----------------------------------------------------------------


def test_the_pending_lists_only_shrink() -> None:
    assert G.only_shrinks() == []
    ids = set(BY_ID)
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
    plain = next(
        v for v in RECORDED['vectors'] if not v['register']['required']
    )
    marked = next(v for v in RECORDED['vectors'] if v['register']['required'])
    other = copy.deepcopy(plain['reference'])
    other['outcome'] = {
        'result': 'refusal',
        'stage': 'from_source',
        'error': 'StructuredDataError',
        'code': 'NOT_THE_REFERENCE_S',
    }
    assert problem_of(plain, plain['reference']) is None
    assert problem_of(plain, other) is not None
    # Each part of the answer counts: f described where it was left out,
    # another warning, f in the request.
    for part, value in (
        ('described', {'dtype': 'int64', 'stype': 'numerical'}),
        ('warnings', []),
        ('payload', {'schema': {'dtype': 'int64'}, 'cells': None}),
    ):
        changed = copy.deepcopy(plain['reference'])
        changed[part] = value
        assert problem_of(plain, changed) is not None, part
    unported = copy.deepcopy(other)
    unported['outcome']['code'] = 'UNSUPPORTED_REQUEST'
    monkeypatch.setattr(
        sys.modules[__name__], 'registered', lambda _: 'SEEDED-1'
    )
    assert problem_of(marked, unported) is not None
    assert problem_of(marked, other) is None
    # An unmarked vector is never excused by an entry.
    assert problem_of(plain, other) is not None
    monkeypatch.setattr(sys.modules[__name__], 'registered', lambda _: None)
    assert problem_of(marked, other) is not None


def test_the_register_reading_is_by_ref_and_status(tmp_path: Path) -> None:
    # A ref names a vector by id or glob; an entry neither proposed nor
    # approved, or a ref of another file, names none.
    document = {
        'entries': [
            {
                'id': 'A-1',
                'status': 'approved',
                'refs': [R.REGISTER_REF + 'struct.*'],
            },
            {
                'id': 'A-2',
                'status': 'rejected',
                'refs': [R.REGISTER_REF + 'map.*'],
            },
            {
                'id': 'A-3',
                'status': 'proposed',
                'refs': ['contract/other.json#map.*'],
            },
        ]
    }
    seeded = tmp_path / 'register.yaml'
    seeded.write_text(yaml.safe_dump(document), encoding='utf-8')
    module = sys.modules[__name__]
    original = module.REGISTER
    try:
        module.REGISTER = seeded  # type: ignore[attr-defined]
        assert registered('struct.integer') == 'A-1'
        assert registered('map.empty') is None
        assert registered('list.integer') is None
    finally:
        module.REGISTER = original  # type: ignore[attr-defined]
