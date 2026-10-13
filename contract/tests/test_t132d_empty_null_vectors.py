# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.32d's empty tables, nulls and refused questions on ``python-core``,
held to the reference's vectors.

    python -m pytest contract/tests/test_t132d_empty_null_vectors.py -q

Written by the package's independent test author from the brief (admit the
``empty.*``, ``null.*`` and ``refuse.*`` cases, lift the ``empty_table``
refusal this family owns, matching the reference or registering each
difference), ``contract/refusal-maps/empty-null.json`` and the reference,
before the implementation. ``contract/empty-null-vectors.json`` is recorded
from the reference by ``contract/tools/empty_null_record.py``, whose
``run`` and ``project`` this test calls, so both routes are observed and
projected by the same code; ``sdm-testkit``'s
``tests/t132d_empty_null_vectors.rs`` holds ``rust-host`` to the same
vectors, so the two hosts agree through them.

- Exits "``python-core`` and ``rust-host`` agree on ``core`` fields" and
  "every difference is registered", over the family's inputs: each vector
  (``shop`` with tables emptied, a column of every type all null, null
  keys, foreign keys, times and targets, one row, entity lists that are
  empty, unknown or everything, and questions naming what the schema does
  not hold) run by ``run_python.py --mode python-core`` through
  ``Schema.from_source``, ``materialize`` and ``predict``, forced onto
  every DuckDB route of the three. Its public answer must be the
  reference's: the outcome (result, stage, class, code), the public schema,
  the warning classes, each request the endpoint received (its digest as
  the ``payload`` rule compares it, and the rows and null cells of each
  table it carries), and the result.
- A vector may differ from the reference only under an approved entry of
  ``docs/design/rust-core/register.yaml`` whose ``refs`` name it with the
  transition it makes (T1.32a's binding-transition form):
  ``contract/empty-null-vectors.json#<id or glob> <from> -> <to>``, each
  side ``success`` or ``refusal <CODE>``, from the reference's answer to
  the host's (the reference's unnamed exception is ``refusal UNNAMED``).
  A ref covers exactly the transition it declares, nothing else; a
  proposed entry excuses nothing, so its vectors stay pending until the
  owner approves it. So the core's ``INVALID_REQUEST`` for an empty table
  (rule ``empty_table``; ``UNSUPPORTED_REQUEST`` until the owner's switch of
  2026-10-13, T1.32g) passes where an approved entry declares it (P1B-11)
  and fails everywhere else. The recorder's ``register.required`` marks
  where a difference is expected (the reference fails unnamed or with
  INTERNAL_ERROR); it excuses nothing by itself, and an unmarked vector
  (a null key, which the reference refuses with INVALID_REQUEST at
  ``materialize``) is held to the reference exactly unless an approved
  entry declares its transition.
- Reviewers: only the owner approves an entry, but a ref attached to an
  already approved entry (P1B-11) takes effect here at once. Each newly
  attached ref therefore needs a reviewer's check that the entry's
  approved ``new_behavior`` really is that transition for those vectors,
  and that its glob names no vector the owner did not approve.
- Every vector ref of a proposed or approved entry must name a transition
  in that form; a malformed ref fails rather than being read as covering
  nothing (or everything).
- The owner's switch of P1B-11 (2026-10-13, T1.32g): each vector the
  reference refuses unnamed at ``from_source`` over an empty table is
  refused by ``python-core`` at ``from_source`` with ``INVALID_REQUEST``
  (class ``StructuredDataError``) in place of ``UNSUPPORTED_REQUEST``.
  The change is the code alone, and with it the class the host raises it
  as: the stage stays. The register declares exactly ``refusal UNNAMED -> refusal
  INVALID_REQUEST`` for each, by an approved entry, and declares
  ``-> refusal UNSUPPORTED_REQUEST`` for none. ``rust-host`` is held to
  the same through the declared transition: its per-vector check passes
  only on a code an approved entry declares.
- The vectors themselves: recorded at the reference of
  ``contract/reference-env.txt``, every vector of the recorder present with
  its inputs, the register marks what the recorder derives, DuckDB holding
  what each vector's name says (empty tables, one row, all-null columns of
  the type named), and not vacuous.

Gating (``contract/tests/t132d_gate.py``): ``t132d-pending.json``'s
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
VECTORS = ROOT / 'contract/empty-null-vectors.json'
REGISTER = ROOT / 'docs/design/rust-core/register.yaml'


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


R = _module('empty_null_record', ROOT / 'contract/tools/empty_null_record.py')
G = _module('t132d_gate', ROOT / 'contract/tests/t132d_gate.py')
RECORDED: dict[str, Any] = json.loads(VECTORS.read_text(encoding='utf-8'))
BY_ID = {vector['id']: vector for vector in RECORDED['vectors']}
#: The inputs a vector carries, each as the recorder wrote it.
INPUTS = ('table', 'setup', 'pql', 'indices', 'from_source')


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
    parts = sorted(k for k in want if got.get(k) != want[k])
    return (
        f'{vector["id"]}: {parts} differ: got {got["outcome"]}, the '
        f'reference {want["outcome"]}; the transition {old} -> {new}: {why}'
    )


@pytest.fixture(scope='module')
def core_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    r"""Every vector through ``python-core``, forced onto the core: one run."""
    tmp = tmp_path_factory.mktemp('t132d-vectors')
    return R.run('python-core', R.VECTORS, tmp, core=True)


@pytest.mark.parametrize(
    'vector_id',
    [pytest.param(v['id'], id=v['id']) for v in RECORDED['vectors']],
)
def test_python_core_handles_the_input_as_the_reference(
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
    assert RECORDED['recorded_by'] == 'contract/tools/empty_null_record.py'
    assert RECORDED['cell'] == 'ref-pandas3'
    assert RECORDED['register_ref'] == R.REGISTER_REF
    assert (
        RECORDED['template'] == 'contract/cases/s1/null.feature-all-null.json'
    )
    assert RECORDED['question'] == R.QUESTION
    assert RECORDED['indices'] == R.INDICES
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    assert f'reference: {RECORDED["reference"]}' in env
    # Every vector the recorder names, once, with its inputs.
    assert [v['id'] for v in RECORDED['vectors']] == [
        v['id'] for v in R.VECTORS
    ]
    for kind in R.VECTORS:
        vector = BY_ID[kind['id']]
        for key in INPUTS:
            assert vector.get(key) == kind.get(key), (kind['id'], key)


def test_the_marks_are_the_recorder_s() -> None:
    # Derived from the reference's answer, never edited.
    for vector in RECORDED['vectors']:
        why = R.register(vector['reference'])
        assert vector['register'] == {'required': bool(why), 'why': why}, (
            vector['id']
        )


def test_each_vector_is_a_case_the_runner_takes() -> None:
    # The cases run_python.py is handed validate, ids unique, and differ
    # from the template only where the vector does; the core is handed the
    # entities and tables the public call is.
    import tempfile

    runner = _module('run_python', ROOT / 'contract/runners/run_python.py')
    template = json.loads(R.TEMPLATE.read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory() as tmp:
        for kind in R.VECTORS:
            for core in (False, True):
                case = R.case_for(kind, core=core)
                assert case['input'].get('setup', []) == kind['setup']
                assert case['input']['pql'] == kind.get('pql', R.QUESTION)
                indices = kind.get('indices', R.INDICES)
                assert case['input']['materialize']['indices'] == indices
                assert case['input']['base'] == template['input']['base']
                if indices:
                    assert case['core']['materialize']['indices'] == [
                        {'t': 'i64', 'v': str(i)} for i in indices
                    ]
                    assert (
                        case['core']['predict'] == template['core']['predict']
                    )
                named = (kind.get('from_source') or {}).get('tables') or {}
                assert [
                    t['name']
                    for t in case['core']['describe_source']['source']['tables']
                ] == list(named)
                assert bool(case['routes']) == core
            (Path(tmp) / f'{case["id"]}.json').write_text(
                json.dumps(case), encoding='utf-8'
            )
        loaded = runner.load_cases([Path(tmp)])
    assert len(loaded) == len(R.VECTORS)


def test_duckdb_holds_what_each_vector_names() -> None:
    # The inputs are what their names say: recorded from DuckDB beside the
    # reference's answer.
    for vector in RECORDED['vectors']:
        rows = vector['duckdb']['rows']
        vid = vector['id']
        if vid.startswith('empty.orders') or vid in (
            'empty.both',
            'edge.no-entities-empty-orders',
        ):
            assert rows['orders'] == 0, vid
        if vid in ('empty.users', 'empty.both'):
            assert rows['users'] == 0, vid
        if vid.startswith('empty.third-table') or vid == 'empty.view':
            assert rows['extra'] == 0, vid
        if vid == 'edge.one-order':
            assert rows['orders'] == 1, vid
        if vid.startswith('edge.one-user'):
            assert rows['users'] in (1, 2), vid
        if vector['table'] is not None:
            assert vector['duckdb']['f_type'], vid
    types = {
        v['id']: v['duckdb']['f_type']
        for v in RECORDED['vectors']
        if v['table'] is not None
    }
    spelled = {'TIMESTAMPTZ': 'TIMESTAMP WITH TIME ZONE'}
    for suffix, sql_type in R.NULL_TYPES:
        # DuckDB's own spelling of the type named.
        want = spelled.get(sql_type, sql_type.replace(', ', ','))
        assert types[f'null.{suffix}'] == want, suffix
    for suffix in R.FACT_NULL_TYPES:
        assert types[f'null.orders-{suffix}'] == types[f'null.{suffix}']
    # An all-null column the schema describes reaches the request as null
    # cells (a column left out, BLOB's, reaches nothing).
    described = 0
    for vector in RECORDED['vectors']:
        answer = vector['reference']
        if not vector['id'].startswith('null.') or vector['table'] is None:
            continue
        if answer['payload'] is None:
            continue
        table = vector['table']
        columns = [
            column['name']
            for entry in answer['schema']['tables']
            if entry['name'] == table
            for column in entry['columns']
        ]
        if 'f' not in columns:
            continue
        described += 1
        cells = answer['payload'][0]['tables'][
            f'context.related_tables.{table}'
        ]
        assert cells['nulls'] > 0, vector['id']
    assert described > 20


def test_the_vectors_are_not_vacuous() -> None:
    answers = {v['id']: v['reference'] for v in RECORDED['vectors']}
    outcomes = [a['outcome'] for a in answers.values()]
    successes = [o for o in outcomes if o['result'] == 'success']
    refusals = [o for o in outcomes if o['result'] == 'refusal']
    assert len(successes) > 40 and len(refusals) > 25
    # A success reached predict's result and its request; a refusal stopped
    # where the reference stopped, and made no request.
    for answer in answers.values():
        if answer['outcome']['result'] == 'success':
            assert answer['outcome']['stage'] == 'predict'
            assert answer['payload'] and answer['result']
        else:
            assert answer['payload'] is None
    marked = [v for v in RECORDED['vectors'] if v['register']['required']]
    assert 0 < len(marked) < len(RECORDED['vectors'])
    # Every stage refuses somewhere, named and unnamed.
    stages = {(o['stage'], o['code']) for o in refusals}
    assert ('from_source', None) in stages
    assert ('materialize', 'INVALID_REQUEST') in stages
    assert ('predict', 'UNSUPPORTED_REQUEST') in stages
    # The kinds the brief names are all here.
    ids = set(answers)
    for prefix in ('empty.', 'null.', 'refuse.', 'edge.'):
        assert sum(i.startswith(prefix) for i in ids) >= 5, prefix
    for wanted in (
        'empty.orders',
        'empty.users',
        'edge.one-order',
        'edge.no-entities',
        'null.every-feature',
        'null.key-one',
        'null.fk-all',
        'null.time-all',
        'null.target-all',
        'refuse.unknown-target-column',
    ):
        assert wanted in ids
    # The nulls reach the request: an all-null feature is sent as nulls,
    # an all-null target's rows too, and null foreign keys add no rows.
    feature = answers['null.integer']['payload'][0]['tables']
    assert feature['context.related_tables.users']['nulls'] > 0
    target = answers['null.target-all']['payload'][0]['tables']
    assert target['context.related_tables.orders']['nulls'] > 0
    orphans = answers['null.fk-all']['payload'][0]['tables']
    assert 'context.related_tables.orders' not in orphans


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


#: The vectors of an empty table the reference refuses unnamed at
#: ``from_source`` ("Table '...' is empty"): P1B-11's.
EMPTY_TABLES = (
    'empty.orders',
    'empty.users',
    'empty.both',
    'empty.third-table',
    'empty.third-table-named',
    'empty.orders-named',
    'empty.orders-filtered-out',
    'edge.no-entities-empty-orders',
)
#: What the owner chose for them (P1B-11, 2026-10-13): the code alone
#: changes, from ``UNSUPPORTED_REQUEST``.
EMPTY_TABLE_REFUSAL = {
    'result': 'refusal',
    'stage': 'from_source',
    'error': 'StructuredDataError',
    'code': 'INVALID_REQUEST',
}


def test_the_empty_tables_are_the_reference_s_unnamed_refusals() -> None:
    # The list is the evidence's: every vector reading an empty table that
    # the reference refuses unnamed at from_source, and no other.
    found = [
        v['id']
        for v in RECORDED['vectors']
        if R.outcome_class(v['reference']['outcome']) == 'refusal UNNAMED'
        and v['reference']['outcome']['stage'] == 'from_source'
        and 0 in v['duckdb']['rows'].values()
    ]
    assert sorted(found) == sorted(EMPTY_TABLES)


def test_an_empty_table_is_refused_invalid_request(
    core_runs: dict[str, Any], request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        entries = transitions()[0]
        problems = []
        for vid in EMPTY_TABLES:
            declared = [
                (entry, status, old, new)
                for entry, status, glob, old, new in entries
                if fnmatch.fnmatchcase(vid, glob)
            ]
            if not any(
                status == 'approved'
                and (old, new) == ('refusal UNNAMED', 'refusal INVALID_REQUEST')
                for _, status, old, new in declared
            ):
                problems.append(
                    f'{vid}: no approved entry declares refusal UNNAMED -> '
                    f'refusal INVALID_REQUEST ({declared})'
                )
            for entry, _, _, new in declared:
                if new == 'refusal UNSUPPORTED_REQUEST':
                    problems.append(f'{vid}: {entry} still declares {new}')
            got = R.project(core_runs[vid], BY_ID[vid])['outcome']
            if got != EMPTY_TABLE_REFUSAL:
                problems.append(f'{vid}: python-core answered {got}')
        assert problems == [], '\n'.join(problems)

    G.held('exits', f'py:{request.node.name}', check)


def test_a_difference_passes_only_under_an_approved_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Fail closed: the comparison itself, on seeded differences and
    # seeded register entries.
    module = sys.modules[__name__]
    plain = BY_ID['null.integer']
    marked = BY_ID['empty.orders']
    assert marked['register']['required']
    assert R.outcome_class(marked['reference']['outcome']) == 'refusal UNNAMED'
    named = copy.deepcopy(marked['reference'])
    named['outcome'] = dict(EMPTY_TABLE_REFUSAL)
    entries: list[tuple[str, str, str, str, str]] = []
    monkeypatch.setattr(module, 'transitions', lambda: (entries, []))
    assert problem_of(plain, plain['reference']) is None
    # No entry: held exactly, marked or not.
    assert problem_of(marked, named) is not None
    # A proposed entry excuses nothing.
    entries[:] = [
        (
            'P-1',
            'proposed',
            'empty.*',
            'refusal UNNAMED',
            'refusal INVALID_REQUEST',
        )
    ]
    assert 'only proposed' in (problem_of(marked, named) or '')
    # An approved one covers exactly its transition and its glob.
    entries[:] = [
        (
            'A-1',
            'approved',
            'empty.*',
            'refusal UNNAMED',
            'refusal INVALID_REQUEST',
        )
    ]
    assert problem_of(marked, named) is None
    assert problem_of(BY_ID['null.interval'], named) is not None
    described = copy.deepcopy(named)
    described['outcome'] = dict(
        named['outcome'], result='success', stage='predict', code=None
    )
    assert problem_of(marked, described) is not None
    unsupported = copy.deepcopy(named)
    unsupported['outcome'] = dict(
        named['outcome'],
        error='UnsupportedRequestError',
        code='UNSUPPORTED_REQUEST',
    )
    assert problem_of(marked, unsupported) is not None
    # An unmarked vector is held exactly, and excused only the same way: a
    # null key, which the reference refuses with INVALID_REQUEST at
    # materialize.
    key = BY_ID['null.key-one']
    assert not key['register']['required']
    assert R.outcome_class(key['reference']['outcome']) == (
        'refusal INVALID_REQUEST'
    )
    early = copy.deepcopy(key['reference'])
    early['outcome'] = dict(unsupported['outcome'])
    early['schema'] = None
    entries[:] = []
    assert problem_of(key, early) is not None
    entries[:] = [
        (
            'A-2',
            'approved',
            'null.key-*',
            'refusal INVALID_REQUEST',
            'refusal UNSUPPORTED_REQUEST',
        )
    ]
    assert problem_of(key, early) is None
    # Each part of a success counts: the schema, the warnings, the request
    # and the result (success -> success, undeclared).
    entries[:] = []
    for part, value in (
        ('schema', None),
        ('warnings', []),
        ('payload', None),
        ('result', None),
    ):
        changed = copy.deepcopy(plain['reference'])
        changed[part] = value
        assert changed != plain['reference'], part
        assert problem_of(plain, changed) is not None, part
    # A null cell more in the request, its digest kept, still differs.
    changed = copy.deepcopy(plain['reference'])
    tables = changed['payload'][0]['tables']
    tables['context.related_tables.users']['nulls'] += 1
    assert problem_of(plain, changed) is not None


def test_the_payload_projection_is_the_payload_rule_s() -> None:
    # Numbers by kind and exact value, the policy's exclusions masked:
    # 1e-05 and 1e-5 agree, 1 and 1.0 do not, nor -0.0 and 0.0; the
    # request id is left out, its type kept.
    assert R.exclusions() == [['metadata', 'request_id']]

    def body(number: str, request_id: Any = 'r-1') -> list[dict[str, Any]]:
        return [
            {
                'method': 'POST',
                'path': '/v1/predictions',
                'parsed': {
                    'metadata': {'request_id': request_id},
                    'context': {
                        'instance_table': {
                            'columns': ['x'],
                            'rows': [[{'number': number}], [None]],
                        }
                    },
                },
            }
        ]

    def digest(number: str, request_id: Any = 'r-1') -> str:
        return str(R.payload_projection(body(number, request_id))[0]['digest'])

    assert digest('1e-05') == digest('1e-5') == digest('0.00001')
    assert digest('1') != digest('1.0')
    assert digest('-0.0') != digest('0.0')
    assert digest('1', 'r-1') == digest('1', 'r-2')
    assert digest('1', 'r-1') != digest('1', 7)
    [projected] = R.payload_projection(body('1'))
    assert projected['tables'] == {
        'context.instance_table': {'rows': 2, 'nulls': 1}
    }
    assert R.payload_projection([]) is None


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
                    f'{ref}empty.* refusal UNNAMED -> refusal INVALID_REQUEST'
                ],
            },
            {
                'id': 'A-2',
                'status': 'proposed',
                'refs': [f'{ref}null.key-* refusal INVALID_REQUEST -> success'],
            },
            {
                'id': 'A-3',
                'status': 'rejected',
                'refs': [f'{ref}empty.* refusal UNNAMED -> success'],
            },
            {
                'id': 'A-4',
                'status': 'approved',
                'refs': [
                    'contract/nested-type-vectors.json#map.* success -> success',
                    f'{ref}empty.*',
                    f'{ref}empty.* refusal unnamed -> success',
                ],
            },
        ]
    }
    read, malformed = transitions(document)
    assert read == [
        (
            'A-1',
            'approved',
            'empty.*',
            'refusal UNNAMED',
            'refusal INVALID_REQUEST',
        ),
        ('A-2', 'proposed', 'null.key-*', 'refusal INVALID_REQUEST', 'success'),
    ]
    assert malformed == [
        f'A-4: {ref}empty.*',
        f'A-4: {ref}empty.* refusal unnamed -> success',
    ]
