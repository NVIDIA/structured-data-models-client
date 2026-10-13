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
- A vector may differ from the reference only under an approved entry of
  ``docs/design/rust-core/register.yaml`` whose ``refs`` name it with the
  transition it makes (T1.32a's binding-transition form):
  ``contract/nested-type-vectors.json#<id or glob> <from> -> <to>``, each
  side ``success`` or ``refusal <CODE>``, from the reference's answer to
  the host's (the reference's unnamed exception is ``refusal UNNAMED``).
  A ref covers exactly the transition it declares, nothing else; a
  proposed entry excuses nothing, so its vectors stay pending until the
  owner approves it. So the core's ``UNSUPPORTED_REQUEST`` (rule
  ``column_type``) passes where an approved entry declares it, P1B-10's
  struct and map for instance, and fails everywhere else. The recorder's
  ``register.required`` marks where a difference is expected (the
  reference fails unnamed or with INTERNAL_ERROR); it excuses nothing by
  itself, and an unmarked vector (BLOB, which the reference leaves out
  with its warning) is held to the reference exactly unless an approved
  entry declares its transition.
- Reviewers: only the owner approves an entry, but a ref attached to an
  already approved entry (P1B-7, P1B-9, P1B-10) takes effect here at once.
  Each newly attached ref therefore needs a reviewer's check that the
  entry's approved ``new_behavior`` really is that transition for those
  vectors, and that its glob names no vector the owner did not approve.
- Every vector ref of a proposed or approved entry must name a transition
  in that form; a malformed ref fails rather than being read as covering
  nothing (or everything).
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
import re
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


#: A vector ref and the transition it declares.
TRANSITION = re.compile(
    re.escape(R.REGISTER_REF)
    + r'(?P<glob>\S+) (?P<old>success|refusal [A-Z][A-Z_]*)'
    + r' -> (?P<new>success|refusal [A-Z][A-Z_]*)'
)


def transitions(
    document: dict[str, Any] | None = None,
) -> tuple[list[tuple[str, str, str, str, str]], list[str]]:
    r"""Each vector ref of the register's proposed or approved entries, as
    (entry, status, glob, from, to), and the refs naming vectors that
    declare no transition in the form (malformed)."""
    if document is None:
        document = yaml.safe_load(REGISTER.read_text(encoding='utf-8'))
    out: list[tuple[str, str, str, str, str]] = []
    malformed: list[str] = []
    for entry in document['entries']:
        status = entry.get('status')
        if status not in ('proposed', 'approved'):
            continue
        for ref in entry.get('refs') or []:
            if not isinstance(ref, str) or not ref.startswith(R.REGISTER_REF):
                continue
            found = TRANSITION.fullmatch(ref)
            if found is None:
                malformed.append(f'{entry["id"]}: {ref}')
                continue
            out.append(
                (
                    str(entry['id']),
                    str(status),
                    found['glob'],
                    found['old'],
                    found['new'],
                )
            )
    return out, malformed


def covered(
    vector_id: str, old: str, new: str
) -> tuple[str | None, str | None]:
    r"""The approved entry whose ref names the vector with exactly the
    transition ``old -> new``, and a proposed one that would, if any."""
    approved = proposed = None
    for entry, status, glob, frm, to in transitions()[0]:
        if (frm, to) != (old, new) or not fnmatch.fnmatchcase(vector_id, glob):
            continue
        if status == 'approved':
            approved = approved or entry
        else:
            proposed = proposed or entry
    return approved, proposed


def problem_of(vector: dict[str, Any], got: dict[str, Any]) -> str | None:
    r"""Why ``got`` fails the vector, or None when it is the reference's
    answer, or a difference an approved entry declares exactly."""
    want = vector['reference']
    if got == want:
        return None
    old = R.outcome_class(want['outcome'])
    new = R.outcome_class(got['outcome'])
    approved, proposed = covered(vector['id'], old, new)
    if approved is not None:
        return None
    why = (
        f'{proposed} declares it but is only proposed'
        if proposed is not None
        else 'no approved entry declares it'
    )
    return (
        f'{vector["id"]}: got {got}, the reference {want}; the transition '
        f'{old} -> {new}: {why}'
    )


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


def test_every_vector_ref_names_its_transition() -> None:
    assert transitions()[1] == []


def test_a_difference_passes_only_under_an_approved_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Fail closed: the comparison itself, on seeded differences and
    # seeded register entries.
    module = sys.modules[__name__]
    plain = BY_ID['list.integer']
    marked = BY_ID['struct.integer']
    assert R.outcome_class(marked['reference']['outcome']) == 'refusal UNNAMED'
    unported = copy.deepcopy(marked['reference'])
    unported['outcome'] = {
        'result': 'refusal',
        'stage': 'from_source',
        'error': 'UnsupportedRequestError',
        'code': 'UNSUPPORTED_REQUEST',
    }
    entries: list[tuple[str, str, str, str, str]] = []
    monkeypatch.setattr(module, 'transitions', lambda: (entries, []))
    assert problem_of(plain, plain['reference']) is None
    # No entry: held exactly, marked or not.
    assert problem_of(marked, unported) is not None
    # A proposed entry excuses nothing.
    entries[:] = [
        (
            'P-1',
            'proposed',
            'struct.*',
            'refusal UNNAMED',
            'refusal UNSUPPORTED_REQUEST',
        )
    ]
    assert 'only proposed' in (problem_of(marked, unported) or '')
    # An approved one covers exactly its transition and its glob.
    entries[:] = [
        (
            'A-1',
            'approved',
            'struct.*',
            'refusal UNNAMED',
            'refusal UNSUPPORTED_REQUEST',
        )
    ]
    assert problem_of(marked, unported) is None
    assert problem_of(BY_ID['map.empty'], unported) is not None
    described = copy.deepcopy(unported)
    described['outcome'] = dict(
        unported['outcome'], result='success', stage='predict', code=None
    )
    assert problem_of(marked, described) is not None
    entries[:] = [
        (
            'A-1',
            'approved',
            'struct.*',
            'refusal INTERNAL_ERROR',
            'refusal UNSUPPORTED_REQUEST',
        )
    ]
    assert problem_of(marked, unported) is not None
    # An unmarked vector is held exactly, and excused only the same way:
    # BLOB, which the reference leaves out with its warning.
    blob = BY_ID['scalar.blob']
    assert not blob['register']['required']
    refused = copy.deepcopy(blob['reference'])
    refused['outcome'] = unported['outcome']
    refused['described'] = refused['payload'] = None
    refused['warnings'] = []
    entries[:] = []
    assert problem_of(blob, refused) is not None
    entries[:] = [
        (
            'A-2',
            'approved',
            'scalar.blob',
            'success',
            'refusal UNSUPPORTED_REQUEST',
        )
    ]
    assert problem_of(blob, refused) is None
    # Each part of a success counts: f described where it was left out,
    # another warning, f in the request (success -> success, undeclared).
    entries[:] = []
    for part, value in (
        ('described', {'dtype': 'int64', 'stype': 'numerical'}),
        ('warnings', []),
        ('payload', {'schema': {'dtype': 'int64'}, 'cells': None}),
    ):
        changed = copy.deepcopy(plain['reference'])
        changed[part] = value
        assert problem_of(plain, changed) is not None, part


def test_the_register_reading_is_by_ref_status_and_transition() -> None:
    # Refs of proposed and approved entries are read; a rejected entry's
    # and another file's are not; a ref naming vectors with no transition
    # in the form is malformed.
    ref = R.REGISTER_REF
    document = {
        'entries': [
            {
                'id': 'A-1',
                'status': 'approved',
                'refs': [
                    f'{ref}struct.* refusal UNNAMED -> refusal UNSUPPORTED_REQUEST'
                ],
            },
            {
                'id': 'A-2',
                'status': 'proposed',
                'refs': [f'{ref}array.* refusal UNNAMED -> success'],
            },
            {
                'id': 'A-3',
                'status': 'rejected',
                'refs': [f'{ref}map.* refusal UNNAMED -> success'],
            },
            {
                'id': 'A-4',
                'status': 'approved',
                'refs': [
                    'contract/other.json#map.* success -> success',
                    f'{ref}map.*',
                    f'{ref}map.* refusal unnamed -> success',
                ],
            },
        ]
    }
    read, malformed = transitions(document)
    assert read == [
        (
            'A-1',
            'approved',
            'struct.*',
            'refusal UNNAMED',
            'refusal UNSUPPORTED_REQUEST',
        ),
        ('A-2', 'proposed', 'array.*', 'refusal UNNAMED', 'success'),
    ]
    assert malformed == [
        f'A-4: {ref}map.*',
        f'A-4: {ref}map.* refusal unnamed -> success',
    ]
