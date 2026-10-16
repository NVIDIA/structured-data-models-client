# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.32b's decimals on ``python-core``, held to the reference's vectors.

    python -m pytest contract/tests/test_t132b_decimal_vectors.py -q

Written by the package's independent test author from the brief (admit the
``decimal.*`` cases; lift ``target_type`` for decimal targets, the
static-decimal refusal behind ``null.decimal-18-2`` and decimal features'
``wire_dtype``; match the reference, its sum's failure at the column's
declared precision included, or register each difference with a binding
transition), ``contract/refusal-maps/decimal.json`` and the reference,
before the implementation. ``contract/decimal-vectors.json`` is recorded
from the reference by ``contract/tools/decimal_record.py``, whose ``run``
and ``project`` this test calls, so both routes are observed and projected
by the same code; ``sdm-testkit``'s ``tests/t132b_decimal_vectors.rs``
holds ``rust-host`` to the same vectors, so the two hosts agree through
them.

- Exits "``python-core`` and ``rust-host`` agree on ``core`` fields" and
  "every difference is registered", over the family's inputs: each vector
  (DECIMAL(p, s) for p in 4, 9, 18, 19, 38 and s in 0, 2, 10, p, as
  features of the entity and the fact table, regression and
  classification targets, keys and columns beside the time column; small
  values, values at the declared precision, zero, negatives, nulls; sums
  at and past the declared precision; HUGEINT and UHUGEINT past 38 digits)
  run by ``run_python.py --mode python-core`` through
  ``Schema.from_source``, ``materialize`` and ``predict``, forced onto
  every DuckDB route of the three. Its public answer must be the
  reference's: the outcome (result, stage, class, code), the public schema
  (each column's dtype and stype), the warning classes, each request (its
  digest as the ``payload`` rule compares it, the rows and null cells of
  each table), the result, and for each decimal column of the vector its
  dtype and stype and how it reached the model (each part's wire dtype
  and stype, and its cells as sent).
- Exit "statement counts equal the reference's, cross-checked against
  DuckDB's query log", over the vectors: where the answer's outcome is the
  reference's (success, or the same refusal), the statements run per public
  call equal the reference's (``statements``, as ``compare.py`` counts
  them), and the capture is complete against DuckDB's log.
- A vector may differ from the reference only under an approved entry of
  ``docs/design/rust-core/register.yaml`` whose ``refs`` name it with the
  transition it makes (T1.32a's binding-transition form):
  ``contract/decimal-vectors.json#<id or glob> <from> -> <to>``, each side
  ``success`` or ``refusal <CODE>``, from the reference's answer to the
  host's (the reference's unnamed exception is ``refusal UNNAMED``). A ref
  covers exactly the transition it declares; a proposed entry excuses
  nothing, so its vectors stay pending until it is approved (a
  success<->refusal change by the owner, E66(b), E73(b)). The recorder's
  ``register.required`` marks where a difference is expected (the
  reference fails unnamed or with INTERNAL_ERROR: its static decimal
  targets and decimal keys); it excuses nothing by itself, and an unmarked
  vector (a sum past the declared precision, which the reference refuses
  with INVALID_REQUEST at ``materialize``) is held to the reference exactly
  unless an approved entry declares its transition.
- Reviewers: a ref attached to an already approved entry takes effect here
  at once; each newly attached ref needs a reviewer's check that the
  entry's approved ``new_behavior`` really is that transition for those
  vectors, and that its glob names no vector the approval does not cover.
- T1.32d's ``null.decimal-18-2`` (an all-null DECIMAL(18, 2) feature) is
  this family's to clear and is not recorded again: its expectation is
  ``contract/empty-null-vectors.json``'s, which T1.32d's tests hold both
  hosts to; the exit is that it is no longer pending there.
- The vectors themselves: recorded at the reference of
  ``contract/reference-env.txt``, every vector of the recorder present with
  its inputs and the core inputs the recorder derives, the register marks
  what the recorder derives, DuckDB holding the types each vector's name
  says, and not vacuous.

Gating (``contract/tests/t132b_gate.py``): ``t132b-pending.json``'s
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
VECTORS = ROOT / 'contract/decimal-vectors.json'
REGISTER = ROOT / 'docs/design/rust-core/register.yaml'


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


R = _module('decimal_record', ROOT / 'contract/tools/decimal_record.py')
G = _module('t132b_gate', ROOT / 'contract/tests/t132b_gate.py')
RECORDED: dict[str, Any] = json.loads(VECTORS.read_text(encoding='utf-8'))
BY_ID = {vector['id']: vector for vector in RECORDED['vectors']}
#: The inputs a vector carries, each as the recorder wrote it.
INPUTS = ('columns', 'setup', 'pql', 'indices', 'anchor_time', 'from_source')

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


def problem_of(
    vector: dict[str, Any],
    got: dict[str, Any],
    counted: dict[str, int] | None = None,
) -> str | None:
    r"""Why ``got`` (and the statements counted, when given) fails the
    vector, or None when it is the reference's answer, or a difference an
    approved entry declares exactly. Where the outcome is the reference's,
    the statement counts are held to the reference's too."""
    want = vector['reference']
    old = R.outcome_class(want['outcome'])
    new = R.outcome_class(got['outcome'])
    if got == want:
        if counted is not None and counted != vector['statements']:
            return (
                f'{vector["id"]}: statements {counted}, the reference '
                f'{vector["statements"]}'
            )
        return None
    approved, proposed = covered(vector['id'], old, new)
    if approved is not None and old != new:
        return None
    if approved is not None and old == new:
        # success -> success, or the same refusal: a difference within the
        # outcome is declared, the counts are still the reference's.
        if counted is not None and counted != vector['statements']:
            return (
                f'{vector["id"]}: statements {counted}, the reference '
                f'{vector["statements"]}'
            )
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
    tmp = tmp_path_factory.mktemp('t132b-vectors')
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
        counted = R.statements(observation)
        got = R.project(observation, vector)
        problem = problem_of(vector, got, counted)
        assert problem is None, problem

    G.held('python-core', vector_id, check)


# --- the vectors ---------------------------------------------------------------


def test_the_vectors_are_the_reference_s() -> None:
    assert RECORDED['format'] == R.FORMAT
    assert RECORDED['recorded_by'] == 'contract/tools/decimal_record.py'
    assert RECORDED['cell'] == 'ref-pandas3'
    assert RECORDED['register_ref'] == R.REGISTER_REF
    assert (
        RECORDED['template']
        == 'contract/cases/s1/decimal.feature-38-0-wide.json'
    )
    assert RECORDED['question'] == R.QUESTION
    assert RECORDED['indices'] == R.INDICES
    assert RECORDED['anchor_time'] == R.ANCHOR
    assert RECORDED['payload_exclusions'] == R.E.exclusions()
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    assert f'reference: {RECORDED["reference"]}' in env
    # Every vector the recorder names, once, with its inputs, and the core
    # inputs the recorder derives from them.
    assert [v['id'] for v in RECORDED['vectors']] == [
        v['id'] for v in R.VECTORS
    ]
    for kind in R.VECTORS:
        vector = BY_ID[kind['id']]
        for key in INPUTS:
            assert vector.get(key) == kind.get(key), (kind['id'], key)
        indices = kind.get('indices', R.INDICES)
        assert vector['core_indices'] == R.core_indices(indices), kind['id']
        anchor = kind.get('anchor_time', R.ANCHOR)
        assert vector['core_anchor'] == R.core_anchor(anchor), kind['id']


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
    # entities, anchor and tables the public call is.
    import tempfile

    runner = _module('run_python', ROOT / 'contract/runners/run_python.py')
    template = json.loads(R.TEMPLATE.read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory() as tmp:
        for kind in R.VECTORS:
            for core in (False, True):
                case = R.case_for(kind, core=core)
                assert case['input']['setup'] == kind['setup']
                assert case['input']['pql'] == kind.get('pql', R.QUESTION)
                indices = kind.get('indices', R.INDICES)
                assert case['input']['materialize']['indices'] == indices
                anchor = kind.get('anchor_time', R.ANCHOR)
                assert case['input']['materialize'].get('anchor_time') == (
                    anchor
                )
                assert case['input']['base'] == template['input']['base']
                assert (
                    case['core']['materialize']['indices']
                    == BY_ID[kind['id']]['core_indices']
                )
                assert (
                    case['core']['materialize']['anchor_time']
                    == BY_ID[kind['id']]['core_anchor']
                )
                assert case['core']['predict'] == template['core']['predict']
                assert bool(case['routes']) == core
            (Path(tmp) / f'{case["id"]}.json').write_text(
                json.dumps(case), encoding='utf-8'
            )
        loaded = runner.load_cases([Path(tmp)])
    assert len(loaded) == len(R.VECTORS)
    # The template's own core anchor is the recorder's reading of its date.
    assert template['core']['materialize']['anchor_time'] == R.core_anchor(
        template['input']['materialize']['anchor_time']
    )


#: DuckDB's spelling of the types a vector's id names.
_TYPE = re.compile(r'\.d(?P<p>[0-9]+)-(?P<s>[0-9]+)(\.|$)')


def test_duckdb_holds_what_each_vector_names() -> None:
    # The inputs are what their names say, recorded from DuckDB beside the
    # reference's answer: a d<p>-<s> vector's decimal column is
    # DECIMAL(p,s) in the catalog and decimal128(p, s) in Arrow.
    named = 0
    for vector in RECORDED['vectors']:
        found = _TYPE.search(vector['id'])
        if found is None or vector['id'].startswith(('key.', 'class.static')):
            continue
        p, s = found['p'], found['s']
        decimal = [
            held
            for held in vector['duckdb'].values()
            if held['type'].startswith('DECIMAL')
        ]
        assert decimal, vector['id']
        for held in decimal:
            assert held == {
                'type': f'DECIMAL({p},{s})',
                'arrow': f'decimal128({p}, {s})',
            }, vector['id']
        named += 1
    assert named > 150
    for vector in RECORDED['vectors']:
        if vector['id'].startswith(('key.both', 'key.at-precision')):
            for column in ('users.user_id', 'orders.user_id'):
                assert vector['duckdb'][column]['type'].startswith('DECIMAL')
    # Past 38 digits DuckDB hands HUGEINT and UHUGEINT over as
    # decimal128(38, 0); no type DuckDB gives here is decimal256, so the
    # family's decimal256 inputs are the host's (T1.32a's entities), not a
    # source's. A DuckDB that produces one fails here and is seen.
    for vector in RECORDED['vectors']:
        for held in vector['duckdb'].values():
            assert not held['arrow'].startswith('decimal256'), vector['id']
    for vector_id in ('wide.hugeint-max', 'wide.uhugeint-max'):
        assert BY_ID[vector_id]['duckdb']['users.f']['arrow'] == (
            'decimal128(38, 0)'
        )


def _request(vector_id: str, column: str) -> dict[str, Any]:
    requests = BY_ID[vector_id]['reference']['columns'][column]['request']
    assert requests and len(requests) == 1, vector_id
    return dict(requests[0])


def test_the_vectors_are_not_vacuous() -> None:
    answers = {v['id']: v['reference'] for v in RECORDED['vectors']}
    outcomes = [a['outcome'] for a in answers.values()]
    successes = [o for o in outcomes if o['result'] == 'success']
    refusals = [o for o in outcomes if o['result'] == 'refusal']
    assert len(successes) > 120
    # At 6298913c (main's eb6c68eb reads a DECIMAL column as int64 or
    # float64 where its values fit) the reference refuses six, each by
    # name: a sum past the declared precision of a decimal whose values do
    # not fit int64 (kept as a decimal), and the four decimal time columns.
    assert sorted(
        i for i, a in answers.items() if a['outcome']['result'] == 'refusal'
    ) == [
        'overflow.d19-0.sum-past',
        'overflow.d38-0.sum-past',
        'time.epoch-declared',
        'time.epoch-in-place',
        'time.epoch-in-place-declared',
        'time.yyyymmdd-declared',
    ]
    assert {o['code'] for o in refusals} == {'INVALID_REQUEST'}
    for answer in answers.values():
        if answer['outcome']['result'] == 'success':
            assert answer['outcome']['stage'] == 'predict'
            assert answer['payload'] and answer['result']
        else:
            assert answer['payload'] is None
    # None fails unnamed or with INTERNAL_ERROR at 6298913c, so none is
    # marked; test_a_difference_passes_only_under_an_approved_transition
    # holds the marking on a seeded vector.
    marked = [v for v in RECORDED['vectors'] if v['register']['required']]
    assert marked == []
    assert all(o['code'] != 'INTERNAL_ERROR' for o in outcomes)
    # The brief's spread: every type, as a feature on both tables and as a
    # summed target, with small values and values at the precision.
    for p, s in R.TYPES:
        for prefix in ('feature.users', 'feature.orders', 'target.sum'):
            for shape in ('spread', 'max'):
                assert f'{prefix}.{R.tag(p, s)}.{shape}' in answers
    assert {(p, s) for p, s in R.TYPES} >= {
        (4, 0),
        (4, 2),
        (4, 4),
        (9, 9),
        (18, 10),
        (19, 19),
        (38, 2),
        (38, 10),
        (38, 38),
    }
    # The kinds the brief names are all here.
    for prefix in (
        'feature.',
        'wide.',
        'target.',
        'overflow.',
        'class.',
        'key.',
        'time.',
    ):
        assert sum(i.startswith(prefix) for i in answers) >= 5, prefix
    # The tracker's T1.27 note, as the reference at 6298913c measures it:
    # a sum past the declared precision of a decimal it reads as int64
    # (DECIMAL(4, 0): 9999 + 1) is taken now (b413e9ad refused it
    # INVALID_REQUEST), as is the sum at it; a DECIMAL(38, 0) whose values
    # do not fit int64 stays a decimal, and its sum past 38 digits is still
    # refused INVALID_REQUEST at materialize, the sum at it taken.
    for vector_id in ('overflow.d4-0.sum-past', 'overflow.d4-0.sum-at'):
        assert answers[vector_id]['outcome']['result'] == 'success', vector_id
    target = _request('overflow.d4-0.sum-at', 'instance_table.TARGET')
    assert '{"number":"9999.0"}' in target['context']['cells']
    past = answers['overflow.d38-0.sum-past']['outcome']
    assert (past['result'], past['stage'], past['code']) == (
        'refusal',
        'materialize',
        'INVALID_REQUEST',
    )
    assert answers['overflow.d38-0.sum-at']['outcome']['result'] == 'success'
    # A static decimal target failed b413e9ad with INTERNAL_ERROR (P1B-18's
    # class); at 6298913c the reference answers each, unmarked.
    three = [i for i in answers if i.startswith('class.static-three.')]
    assert len(three) >= 5
    for vector_id in three:
        assert answers[vector_id]['outcome']['result'] == 'success'
        assert not BY_ID[vector_id]['register']['required']
    # How decimals reach the model: a DECIMAL(38, 0) past int64 as text,
    # every digit kept; a DECIMAL(4, 0) as int64 numbers; a scale as
    # float64 numbers; nulls as nulls.
    wide = _request('feature.users.d38-0.max', 'users.f')['context']
    assert wide['dtype'] == 'string'
    assert '"99999999999999999999999999999999999999"' in wide['cells']
    narrow = _request('feature.users.d4-0.max', 'users.f')['context']
    assert narrow['dtype'] == 'int64'
    assert '{"number":"9999"}' in narrow['cells']
    scaled = _request('feature.orders.d4-2.spread', 'orders.f')['context']
    assert scaled['dtype'] == 'float64'
    assert 'null' in scaled['cells']
    empty = _request('feature.users.d4-0.null', 'users.f')['context']
    assert set(empty['cells']) == {'null'}
    # DuckDB's UHUGEINT past HUGEINT reaches the reference as decimal128
    # wrapped below zero: what the source gave is what the vectors hold.
    wrapped = _request('wide.uhugeint-max', 'users.f')['context']
    assert '{"number":"-1"}' in wrapped['cells']
    # The statements per public call, as counted, for every vector.
    for vector in RECORDED['vectors']:
        assert set(vector['statements']) == {
            'from_source',
            'materialize',
            'predict',
        }
        assert vector['statements']['from_source'] > 0, vector['id']


# --- T1.32d's null.decimal-18-2 -------------------------------------------------


def test_null_decimal_18_2_is_cleared(request: pytest.FixtureRequest) -> None:
    # T1.32d's vector, this family's to clear (its static-decimal
    # wire_dtype refusal): held to its own expectation by T1.32d's tests on
    # both hosts, it passes once it is out of their pending lists.
    def check() -> None:
        empty_null = json.loads(
            (ROOT / 'contract/empty-null-vectors.json').read_text(
                encoding='utf-8'
            )
        )
        [vector] = [
            v for v in empty_null['vectors'] if v['id'] == 'null.decimal-18-2'
        ]
        assert vector['setup'] == [
            'CREATE OR REPLACE TABLE users AS SELECT *, '
            'NULL::DECIMAL(18, 2) AS f FROM users'
        ]
        assert vector['reference']['outcome']['result'] == 'success'
        pending = json.loads(
            (ROOT / 'contract/tests/t132d-pending.json').read_text(
                encoding='utf-8'
            )
        )['pending']
        for host in ('python-core', 'rust-host'):
            assert 'null.decimal-18-2' not in pending[host], host

    G.held('exits', f'py:{request.node.name}', check)


# --- the gating ----------------------------------------------------------------


def test_the_pending_lists_only_shrink() -> None:
    assert G.only_shrinks() == []
    ids = set(BY_ID)
    for host in ('python-core', 'rust-host'):
        assert G.pending(host) <= ids, f'{host}: ids that are no vector'


def test_the_gate_holds_a_pending_test_to_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    # Fail closed: the comparison itself, on seeded differences and seeded
    # register entries.
    module = sys.modules[__name__]
    plain = BY_ID['feature.users.d4-2.spread']
    # class.static-three.d4-0 as the reference answered it at b413e9ad
    # (INTERNAL_ERROR at materialize), seeded: no recorded vector fails so
    # at 6298913c. Its mark is the recorder's own, derived from it.
    marked = copy.deepcopy(BY_ID['class.static-three.d4-0'])
    marked['reference']['outcome'] = {
        'result': 'refusal',
        'stage': 'materialize',
        'error': 'StructuredDataError',
        'code': 'INTERNAL_ERROR',
    }
    marked['reference']['payload'] = None
    why = R.register(marked['reference'])
    marked['register'] = {'required': bool(why), 'why': why}
    assert marked['register']['required']
    assert R.outcome_class(marked['reference']['outcome']) == (
        'refusal INTERNAL_ERROR'
    )
    named = copy.deepcopy(marked['reference'])
    named['outcome'] = {
        'result': 'refusal',
        'stage': 'materialize',
        'error': 'UnsupportedRequestError',
        'code': 'UNSUPPORTED_REQUEST',
    }
    entries: list[tuple[str, str, str, str, str]] = []
    monkeypatch.setattr(module, 'transitions', lambda: (entries, []))
    counts = plain['statements']
    assert problem_of(plain, plain['reference'], counts) is None
    # No entry: held exactly, marked or not.
    assert problem_of(marked, named) is not None
    # A proposed entry excuses nothing.
    entries[:] = [
        (
            'P-1',
            'proposed',
            'class.static-*',
            'refusal INTERNAL_ERROR',
            'refusal UNSUPPORTED_REQUEST',
        )
    ]
    assert 'only proposed' in (problem_of(marked, named) or '')
    # An approved one covers exactly its transition and its glob.
    entries[:] = [
        (
            'A-1',
            'approved',
            'class.static-*',
            'refusal INTERNAL_ERROR',
            'refusal UNSUPPORTED_REQUEST',
        )
    ]
    assert problem_of(marked, named) is None
    other = BY_ID['key.both.d4-0']
    assert problem_of(other, named) is not None
    succeeded = copy.deepcopy(named)
    succeeded['outcome'] = dict(
        named['outcome'], result='success', stage='predict', code=None
    )
    assert problem_of(marked, succeeded) is not None
    # An unmarked vector, a sum past the declared precision refused by
    # name, is held exactly, and excused only the same way.
    past = BY_ID['overflow.d38-0.sum-past']
    assert not past['register']['required']
    taken = copy.deepcopy(BY_ID['overflow.d38-0.sum-at']['reference'])
    entries[:] = []
    assert problem_of(past, taken) is not None
    entries[:] = [
        (
            'A-2',
            'approved',
            'overflow.*-past',
            'refusal INVALID_REQUEST',
            'success',
        )
    ]
    assert problem_of(past, taken) is None
    # Each part of a success counts: the schema, the warnings, the request,
    # the result and the decimal columns (success -> success, undeclared).
    entries[:] = []
    for part, value in (
        ('schema', None),
        ('warnings', []),
        ('payload', None),
        ('result', None),
        ('columns', {}),
    ):
        changed = copy.deepcopy(plain['reference'])
        changed[part] = value
        assert changed != plain['reference'], part
        assert problem_of(plain, changed) is not None, part
    # A decimal sent another way, its digest kept, still differs: a cell's
    # text, and the wire dtype.
    changed = copy.deepcopy(plain['reference'])
    [request] = changed['columns']['users.f']['request']
    cells = request['context']['cells']
    first = sorted(cells)[0]
    cells['{"number":"0.2500"}'] = cells.pop(first)
    assert problem_of(plain, changed) is not None
    changed = copy.deepcopy(plain['reference'])
    changed['columns']['users.f']['request'][0]['context']['dtype'] = 'string'
    assert problem_of(plain, changed) is not None
    changed = copy.deepcopy(plain['reference'])
    changed['columns']['users.f']['described']['stype'] = 'timestamp'
    assert problem_of(plain, changed) is not None
    # The counts: the reference's answer with a statement more fails; under
    # a declared same-outcome difference too.
    more = dict(counts, materialize=counts['materialize'] + 1)
    assert problem_of(plain, plain['reference'], more) is not None
    entries[:] = [('A-3', 'approved', 'feature.users.*', 'success', 'success')]
    assert problem_of(plain, changed, counts) is None
    assert problem_of(plain, changed, more) is not None


def test_the_columns_projection_reads_each_part_of_the_request() -> None:
    # Seeded bodies: a feature in the related tables, the target in the
    # instance table, a part that lacks the column, and no request.
    body = {
        'schema': {
            'instance_table': {
                'columns': {
                    'TARGET': {'dtype': 'float32', 'stype': 'numerical'}
                }
            },
            'related_tables': {
                'users': {
                    'columns': {
                        'f': {'dtype': 'string', 'stype': 'categorical'}
                    }
                }
            },
        },
        'context': {
            'instance_table': {
                'columns': ['instance_id', 'TARGET'],
                'rows': [
                    [{'number': '0'}, {'number': '1.5'}],
                    [{'number': '1'}, None],
                ],
            },
            'related_tables': {
                'users': {
                    'columns': ['instance_id', 'f'],
                    'rows': [[{'number': '0'}, '12'], [{'number': '1'}, '12']],
                }
            },
        },
        'predict': {
            'instance_table': {
                'columns': ['instance_id'],
                'rows': [[{'number': '2'}]],
            },
            'related_tables': {},
        },
    }
    observation = {
        'public': {
            'schema': {
                'tables': [
                    {
                        'name': 'users',
                        'columns': [
                            {
                                'name': 'f',
                                'dtype': 'int64',
                                'stype': 'categorical',
                            }
                        ],
                    }
                ]
            },
            'payloads': [{'parsed': body}],
        }
    }
    vector = {'columns': [['instance_table', 'TARGET'], ['users', 'f']]}
    projected = R.columns_projection(observation, vector)
    assert projected == {
        'instance_table.TARGET': {
            'described': None,
            'request': [
                {
                    'context': {
                        'dtype': 'float32',
                        'stype': 'numerical',
                        'cells': {'null': 1, '{"number":"1.5"}': 1},
                    },
                    'predict': None,
                }
            ],
        },
        'users.f': {
            'described': {'dtype': 'int64', 'stype': 'categorical'},
            'request': [
                {
                    'context': {
                        'dtype': 'string',
                        'stype': 'categorical',
                        'cells': {'"12"': 2},
                    },
                    'predict': None,
                }
            ],
        },
    }
    observation['public']['payloads'] = []
    assert (
        R.columns_projection(observation, vector)['users.f']['request'] is None
    )


def test_the_register_reading_is_by_ref_status_and_transition() -> None:
    ref = R.REGISTER_REF
    document = {
        'entries': [
            {
                'id': 'A-1',
                'status': 'approved',
                'refs': [
                    f'{ref}class.static-* refusal INTERNAL_ERROR -> '
                    'refusal UNSUPPORTED_REQUEST'
                ],
            },
            {
                'id': 'A-2',
                'status': 'proposed',
                'refs': [f'{ref}key.* refusal INTERNAL_ERROR -> success'],
            },
            {
                'id': 'A-3',
                'status': 'rejected',
                'refs': [f'{ref}key.* refusal INTERNAL_ERROR -> success'],
            },
            {
                'id': 'A-4',
                'status': 'approved',
                'refs': [
                    'contract/empty-null-vectors.json#null.* success -> success',
                    f'{ref}key.*',
                    f'{ref}key.* refusal internal_error -> success',
                ],
            },
        ]
    }
    read, malformed = transitions(document)
    assert read == [
        (
            'A-1',
            'approved',
            'class.static-*',
            'refusal INTERNAL_ERROR',
            'refusal UNSUPPORTED_REQUEST',
        ),
        ('A-2', 'proposed', 'key.*', 'refusal INTERNAL_ERROR', 'success'),
    ]
    assert malformed == [
        f'A-4: {ref}key.*',
        f'A-4: {ref}key.* refusal internal_error -> success',
    ]
