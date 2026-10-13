# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.32e part 2's multicategorical port on ``python-core``, held to the
reference's vectors.

    python -m pytest contract/tests/test_t132e_multicategorical.py -q

Written by the package's independent test author from the brief (port the
reference's multi-category detection and feature handling faithfully: its
stype for multi-category text, its separator detection, how it encodes the
values on the wire; stop misfiring on ordinary text; lift the
``multicategorical`` refusal) and the reference, before the
implementation. ``contract/multicategorical-vectors.json`` is recorded from
the reference by ``contract/tools/multicategorical_record.py``, whose
``run`` and ``project`` this test calls, so both routes are observed and
projected by the same code; ``sdm-testkit``'s
``tests/t132e_multicategorical.rs`` holds ``rust-host`` to the same
vectors, so the two hosts agree through them.

- "Port the reference's multi-category detection and feature handling
  faithfully": each vector (text columns with each separator the reference
  counts and some it does not, the 1.5 ratio's and the 100 parts'
  boundaries, times of day, URLs, ratios, free text with an occasional
  separator, two separators in one column and a tie between them, nulls,
  empty strings, a single category, 1,200-row tables whose values past row
  500 or the 1,000-row preview differ, the column on the fact table, and
  questions naming it) run by ``run_python.py --mode python-core`` through
  ``Schema.from_source``, ``materialize`` and ``predict``, forced onto
  every DuckDB route of the three. Its public answer must be the
  reference's: the outcome, the public schema (each column's dtype and
  stype), the warning classes, each request (its digest as the ``payload``
  rule compares it, and the rows and null cells of each table it carries),
  the result, and ``wire``: how the column travels (the request schema's
  dtype and stype for it, and its cells).
- "Lift the ``multicategorical`` refusal" and "stop misfiring on ordinary
  text": no vector is refused by the rule ``multicategorical`` (or by any
  rule naming it), on the vectors' own evidence.
- A vector may differ from the reference only under an approved entry of
  ``docs/design/rust-core/register.yaml`` whose ``refs`` name it with the
  transition it makes (T1.32a's binding-transition form):
  ``contract/multicategorical-vectors.json#<id or glob> <from> -> <to>``,
  each side ``success`` or ``refusal <CODE>`` (the reference's unnamed
  exception is ``refusal UNNAMED``). A proposed entry excuses nothing. The
  recorder's ``register.required`` marks where a difference is expected
  (the reference fails unnamed or with INTERNAL_ERROR) and excuses nothing
  by itself.
- Reviewers: only the owner approves an entry, but a ref attached to an
  already approved entry takes effect here at once; each newly attached ref
  needs a reviewer's check that the entry's approved behaviour is that
  transition for exactly the vectors its glob names.
- The vectors themselves: recorded at the reference of
  ``contract/reference-env.txt``, every vector of the recorder present with
  its inputs, the marks the recorder's, DuckDB holding what each vector's
  name says, the reference's own answers what the brief claims of it, and
  not vacuous.

Gating (``contract/tests/t132e_gate.py``): ``t132e-pending.json``'s
``pending["mc-python-core"]`` lists the vectors that failed here when this
was written, and ``pending["p2-exits"]`` (as ``py:<test id>``) the exits.
Each still runs and must still fail; one that passes fails, naming itself,
until it is taken out of the list, which may only shrink. The implementer
removes ids as they pass and changes nothing else of this file, the
vectors or the recorder.
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
VECTORS = ROOT / 'contract/multicategorical-vectors.json'
REGISTER = ROOT / 'docs/design/rust-core/register.yaml'


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


R = _module(
    'multicategorical_record',
    ROOT / 'contract/tools/multicategorical_record.py',
)
G = _module('t132e_gate', ROOT / 'contract/tests/t132e_gate.py')
RECORDED: dict[str, Any] = json.loads(VECTORS.read_text(encoding='utf-8'))
BY_ID = {vector['id']: vector for vector in RECORDED['vectors']}
INPUTS = ('table', 'setup', 'pql', 'indices')
#: The parts of an answer, as the recorder projects it.
PARTS = ('outcome', 'schema', 'warnings', 'payload', 'result', 'wire')

TRANSITION = re.compile(
    re.escape(R.REGISTER_REF)
    + r'(?P<glob>\S+) (?P<old>success|refusal [A-Z][A-Z_]*)'
    + r' -> (?P<new>success|refusal [A-Z][A-Z_]*)'
)


def transitions(
    document: dict[str, Any] | None = None,
) -> tuple[list[tuple[str, str, str, str, str]], list[str]]:
    r"""Each vector ref of the register's proposed or approved entries, as
    (entry, status, glob, from, to), and the refs naming these vectors that
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
    if all(got.get(part) == want.get(part) for part in PARTS):
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
    parts = [part for part in PARTS if got.get(part) != want.get(part)]
    return (
        f'{vector["id"]}: {parts} differ: got {got["outcome"]}, the '
        f'reference {want["outcome"]}; the transition {old} -> {new}: {why}'
    )


def rule_of(error: BaseException) -> str | None:
    r"""The rule a refusal names in its ``details`` (as
    ``contract/runners/refusal_map.py`` reads it), or None."""
    details = getattr(error, 'details', None)
    rule = details.get('rule') if isinstance(details, dict) else None
    return rule if isinstance(rule, str) else None


def described_rules(vectors: list[dict[str, Any]]) -> dict[str, str | None]:
    r"""Each vector's ``Schema.from_source`` on the core's route, in this
    process: the rule its refusal names, None where it names none or there
    is no refusal."""
    import os

    import duckdb
    from sdm_client import Schema, StructuredDataError, _routing
    from sdm_connectors import DuckDBSource

    out: dict[str, str | None] = {}
    previous = os.environ.get(_routing.ENVIRONMENT)
    os.environ[_routing.ENVIRONMENT] = ','.join(_routing.FROM_SOURCE_DUCKDB)
    try:
        for vector in vectors:
            connection = duckdb.connect()
            try:
                for statement in R.EN.shop() + vector['setup']:
                    connection.execute(statement)
                try:
                    Schema.from_source(DuckDBSource(connection))
                    out[vector['id']] = None
                except StructuredDataError as error:
                    out[vector['id']] = rule_of(error)
            finally:
                connection.close()
    finally:
        if previous is None:
            os.environ.pop(_routing.ENVIRONMENT, None)
        else:
            os.environ[_routing.ENVIRONMENT] = previous
    return out


@pytest.fixture(scope='module')
def core_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    r"""Every vector through ``python-core``, forced onto the core: one run."""
    tmp = tmp_path_factory.mktemp('t132e2-vectors')
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

    G.held('mc-python-core', vector_id, check)


def test_no_text_is_refused_for_holding_a_separator() -> None:
    # The brief: lift the multicategorical refusal; ordinary text (a time
    # of day, a URL, a ratio) is no longer refused for a ':' or a ';'. Read
    # from each refusal's own rule, on every vector, so a refusal under
    # another code or stage is caught too.
    def check() -> None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rules = described_rules(R.VECTORS)
        assert set(rules) == set(BY_ID)
        refused = {v: r for v, r in rules.items() if r == 'multicategorical'}
        assert refused == {}, sorted(refused)

    G.held(
        'p2-exits', 'py:test_no_text_is_refused_for_holding_a_separator', check
    )


def test_the_rule_reading_fails_closed() -> None:
    # The reading finds the rule in a refusal's details, and none where
    # the details name none or are not a mapping.
    class Refused(Exception):
        def __init__(self, details: Any) -> None:
            super().__init__('refused')
            self.details = details

    assert rule_of(Refused({'rule': 'multicategorical'})) == 'multicategorical'
    assert rule_of(Refused({'rule': None})) is None
    assert rule_of(Refused(None)) is None
    assert rule_of(Refused(['multicategorical'])) is None
    assert rule_of(ValueError('multicategorical')) is None
    # On this tree's own evidence: a separator in the first 500 values is
    # read by the probe (refused by the rule today, or described once the
    # port lands); a column with none is described.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rules = described_rules(
            [BY_ID['sep.semicolon'], BY_ID['text.plain-categories']]
        )
    assert rules['text.plain-categories'] is None
    assert rules['sep.semicolon'] in (None, 'multicategorical')


# --- the vectors ---------------------------------------------------------------


def test_the_vectors_are_the_reference_s() -> None:
    assert RECORDED['format'] == R.FORMAT
    assert (
        RECORDED['recorded_by'] == 'contract/tools/multicategorical_record.py'
    )
    assert RECORDED['cell'] == 'ref-pandas3'
    assert RECORDED['register_ref'] == R.REGISTER_REF
    assert RECORDED['template'] == 'contract/cases/s1/feature.bool.json'
    assert RECORDED['question'] == R.QUESTION
    assert RECORDED['indices'] == R.INDICES
    assert RECORDED['payload_exclusions'] == [['metadata', 'request_id']]
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    assert f'reference: {RECORDED["reference"]}' in env
    assert [v['id'] for v in RECORDED['vectors']] == [
        v['id'] for v in R.VECTORS
    ]
    for kind in R.VECTORS:
        vector = BY_ID[kind['id']]
        for key in INPUTS:
            assert vector.get(key) == kind.get(key), (kind['id'], key)


def test_the_marks_are_the_recorder_s() -> None:
    for vector in RECORDED['vectors']:
        why = R.register(vector['reference'])
        assert vector['register'] == {'required': bool(why), 'why': why}, (
            vector['id']
        )


def test_each_vector_is_a_case_the_runner_takes() -> None:
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
                assert case['input']['base'] == template['input']['base']
                assert case['core']['materialize']['indices'] == [
                    {'t': 'i64', 'v': str(i)} for i in indices
                ]
                assert case['core']['predict'] == template['core']['predict']
                assert bool(case['routes']) == core
            (Path(tmp) / f'{case["id"]}.json').write_text(
                json.dumps(case), encoding='utf-8'
            )
        loaded = runner.load_cases([Path(tmp)])
    assert len(loaded) == len(R.VECTORS)


def _f_stypes(vector: dict[str, Any]) -> dict[str, str]:
    schema = vector['reference']['schema'] or {'tables': []}
    return {
        table['name']: column['stype']
        for table in schema['tables']
        for column in table['columns']
        if column['name'] == 'f'
    }


def test_duckdb_holds_what_each_vector_names() -> None:
    for vector in RECORDED['vectors']:
        vid = vector['id']
        held = vector['duckdb']
        tables = (
            ['users', 'orders']
            if vector['table'] == 'both'
            else [vector['table']]
        )
        assert sorted(held['f']) == sorted(tables), vid
        for table in tables:
            assert held['f'][table]['type'] == 'VARCHAR', vid
        if vid.startswith('large.') or vid.startswith('sep.parts-'):
            assert held['rows']['users'] == R.LARGE, vid
            assert vector['indices'] == R.LARGE_INDICES, vid
        elif vid.endswith('-8-rows'):
            assert held['rows']['users'] == 8, vid
        else:
            assert held['rows']['users'] == R.ROWS, vid
    f = {v['id']: v['duckdb']['f'] for v in RECORDED['vectors']}
    # Each recognized separator, in its own vector.
    for name in R.SEPARATORS:
        counts = f[f'sep.{name}']['users']['first_500']
        assert counts[name] > 0 and sum(counts.values()) == counts[name]
    # None the reference counts in the controls.
    for vid in ('sep.comma', 'sep.space', 'sep.slash', 'text.plain-unique'):
        assert sum(f[vid]['users']['whole'].values()) == 0, vid
    # Past the preview: the separators start after row 500 or 1,000, or
    # stop there.
    late = f['large.plain-then-multi-after-1000']['users']
    assert late['preview']['semicolon'] == 0 < late['whole']['semicolon']
    late = f['large.plain-then-multi-after-500']['users']
    assert late['first_500']['semicolon'] == 0 < late['preview']['semicolon']
    early = f['large.multi-then-plain-after-1000']['users']
    assert early['preview']['semicolon'] == early['whole']['semicolon'] > 0
    nulls = f['large.nulls-first-500']['users']
    assert nulls['nulls'] == 500 and nulls['first_500']['semicolon'] == 0
    # The times of day and ratios hold a colon, and nothing else counted.
    for vid in ('text.time-of-day', 'text.ratio', 'text.aspect-ratio'):
        counts = f[vid]['users']['first_500']
        assert counts['colon'] > 0 and counts['semicolon'] == 0, vid
    # Both separators in the mixed vectors, the majority as named.
    majority = f['mixed.pipe-majority-multi']['users']['first_500']
    assert majority['pipe'] > majority['semicolon'] > 0
    majority = f['mixed.semicolon-majority-not-multi']['users']['first_500']
    assert majority['semicolon'] > majority['pipe'] > 0
    for vid in ('mixed.tie-pipe-first', 'mixed.tie-semicolon-first'):
        tie = f[vid]['users']['first_500']
        assert tie['pipe'] == tie['semicolon'] > 0, vid


def test_the_reference_answers_as_the_brief_says_of_it() -> None:
    # What the vectors show the reference does, so the port's target is
    # stated, not assumed: a recognized separator makes multi-category
    # text; times of day and ratios are read as times, never as
    # categories; URLs and free text are text or categorical; the most
    # frequent separator decides, the first seen on a tie; the first 500
    # rows decide, whatever comes after.
    stype = {v['id']: _f_stypes(v) for v in RECORDED['vectors']}
    for name in R.SEPARATORS:
        assert stype[f'sep.{name}'] == {'users': 'multicategorical'}, name
    for vid in ('sep.comma', 'sep.space', 'sep.slash', 'sep.ratio-at-1.5'):
        assert stype[vid] == {'users': 'categorical'}, vid
    assert stype['sep.ratio-over-1.5'] == {'users': 'multicategorical'}
    assert stype['sep.parts-100'] == {'users': 'multicategorical'}
    assert stype['sep.parts-101']['users'] != 'multicategorical'
    for vid in (
        'text.time-of-day',
        'text.time-of-day-seconds',
        'text.ratio',
        'text.aspect-ratio',
    ):
        assert stype[vid] == {'users': 'timestamp'}, vid
    for vid in (
        'text.url-unique',
        'text.url-few',
        'text.url-with-port-and-query',
        'text.free-occasional-colon',
        'text.free-occasional-semicolon',
        'text.free-few-occasional-colon',
        'text.mac-address',
        'text.ipv6',
        'text.windows-path',
    ):
        assert stype[vid]['users'] in ('text', 'categorical'), vid
    assert stype['mixed.pipe-majority-multi']['users'] == 'multicategorical'
    assert stype['mixed.semicolon-majority-not-multi']['users'] == (
        'categorical'
    )
    assert stype['mixed.tie-pipe-first']['users'] == 'multicategorical'
    assert stype['mixed.tie-semicolon-first']['users'] == 'categorical'
    assert stype['large.plain-then-multi-after-500']['users'] == 'categorical'
    assert stype['large.multi-then-plain-after-500']['users'] == (
        'multicategorical'
    )
    assert stype['large.multi-many-parts-after-1000']['users'] == (
        'multicategorical'
    )
    for vid in ('single.one-value', 'single.one-value-with-separator'):
        assert stype[vid] == {'users': 'categorical'}, vid
    assert stype['fact.and-entity'] == {
        'orders': 'multicategorical',
        'users': 'multicategorical',
    }


def test_the_wire_encoding_is_recorded_and_not_vacuous() -> None:
    # A multi-category column travels as text, each value whole: declared
    # dtype string and stype multicategorical in the request's schema, its
    # cells strings (or nulls), not lists of parts. Values past the preview
    # reach the request, as the entities asked for.
    seen = 0
    for vector in RECORDED['vectors']:
        answer = vector['reference']
        if answer['outcome']['result'] != 'success':
            assert answer['wire'] is None, vector['id']
            continue
        stypes = _f_stypes(vector)
        [tables] = answer['wire']
        assert tables, vector['id']
        for path, column in tables.items():
            table = path.rsplit('.', 1)[1]
            assert column['declared']['stype'] == stypes[table], (
                vector['id'],
                path,
            )
            assert column['rows'] > 0, (vector['id'], path)
            assert len(column['digest']) == 64
            if stypes[table] == 'multicategorical':
                seen += 1
                assert column['declared']['dtype'] == 'string'
                assert set(column['kinds']) <= {'string', 'null'}, path
    assert seen > 40
    late = BY_ID['large.multi-new-parts-after-1000']['reference']['wire'][0]
    assert late['predict.related_tables.users']['cells'] == [
        'a;b',
        'a;c',
        'x;y',
        'x',
    ]
    nulls = BY_ID['null.some']['reference']['wire'][0]
    assert nulls['context.related_tables.users']['nulls'] > 0


def test_the_wire_reading_is_the_payload_rule_s() -> None:
    # The cells' digest reads numbers by kind and exact value and ignores
    # nothing else; a summary lists sorted distinct cells; cells are kept
    # whole up to WHOLE rows.
    assert R.cells_digest(['a;b', None]) != R.cells_digest(['a;b'])
    assert R.cells_digest(['a;b', 'c']) != R.cells_digest(['c', 'a;b'])
    assert R.cells_digest([{'number': '1e-5'}]) == R.cells_digest(
        [{'number': '0.00001'}]
    )
    assert R.cells_digest([{'number': '1'}]) != R.cells_digest(
        [{'number': '1.0'}]
    )
    small = R.column_on_the_wire(
        {'dtype': 'string', 'stype': 'multicategorical'}, ['b;a', None, 'a']
    )
    assert small['rows'] == 3 and small['nulls'] == 1
    assert small['kinds'] == ['null', 'string']
    assert small['listed'] == ['a', 'b;a', None]
    assert small['distinct'] == 3
    assert small['cells'] == ['b;a', None, 'a']
    big = R.column_on_the_wire(None, ['x'] * (R.WHOLE + 1))
    assert 'cells' not in big and big['distinct'] == 1
    body = {
        'schema': {
            'related_tables': {
                'users': {
                    'columns': {
                        'f': {'dtype': 'string', 'stype': 'multicategorical'}
                    }
                }
            }
        },
        'context': {
            'related_tables': {
                'users': {'columns': ['user_id', 'f'], 'rows': [[1, 'a;b']]}
            }
        },
    }
    [tables] = R.wire([{'parsed': body}])
    assert tables['context.related_tables.users']['declared'] == {
        'dtype': 'string',
        'stype': 'multicategorical',
    }
    assert tables['context.related_tables.users']['cells'] == ['a;b']
    assert R.wire([]) is None


def test_the_vectors_are_not_vacuous() -> None:
    answers = {v['id']: v['reference'] for v in RECORDED['vectors']}
    outcomes = [a['outcome'] for a in answers.values()]
    successes = [o for o in outcomes if o['result'] == 'success']
    refusals = [o for o in outcomes if o['result'] == 'refusal']
    assert len(successes) > 50 and len(refusals) > 5
    for answer in answers.values():
        if answer['outcome']['result'] == 'success':
            assert answer['outcome']['stage'] == 'predict'
            assert answer['payload'] and answer['result'] and answer['wire']
        else:
            assert answer['payload'] is None
    marked = [v for v in RECORDED['vectors'] if v['register']['required']]
    assert 0 < len(marked) < len(RECORDED['vectors'])
    ids = set(answers)
    for prefix in ('sep.', 'text.', 'mixed.', 'null.', 'empty.', 'single.'):
        assert sum(i.startswith(prefix) for i in ids) >= 2, prefix
    assert sum(i.startswith('large.') for i in ids) >= 8
    assert sum(i.startswith('fact.') for i in ids) >= 3
    assert sum(i.startswith('question.') for i in ids) >= 3
    stypes = [s for v in RECORDED['vectors'] for s in _f_stypes(v).values()]
    for stype in ('multicategorical', 'categorical', 'text', 'timestamp'):
        assert stype in stypes, stype


# --- the gating ----------------------------------------------------------------


def test_the_pending_lists_only_shrink() -> None:
    assert G.only_shrinks() == []
    ids = set(BY_ID)
    for host in ('mc-python-core', 'mc-rust-host'):
        assert G.pending(host) <= ids, f'{host}: ids that are no vector'


def test_the_gate_holds_a_pending_test_to_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def passing() -> None:
        pass

    def failing() -> None:
        raise AssertionError('seeded')

    monkeypatch.setattr(G, 'pending', lambda group: {'seeded.listed'})
    G.held('mc-python-core', 'seeded.listed', failing)
    with pytest.raises(AssertionError, match='passes now'):
        G.held('mc-python-core', 'seeded.listed', passing)
    G.held('mc-python-core', 'seeded.unlisted', passing)
    with pytest.raises(AssertionError, match='seeded'):
        G.held('mc-python-core', 'seeded.unlisted', failing)


def test_every_vector_ref_names_its_transition() -> None:
    assert transitions()[1] == []


def test_a_difference_passes_only_under_an_approved_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = sys.modules[__name__]
    plain = BY_ID['sep.semicolon']
    marked = BY_ID['question.static-target']
    assert marked['register']['required']
    assert R.outcome_class(marked['reference']['outcome']) == (
        'refusal INTERNAL_ERROR'
    )
    named = copy.deepcopy(marked['reference'])
    named['outcome'] = dict(named['outcome'], code='UNSUPPORTED_REQUEST')
    entries: list[tuple[str, str, str, str, str]] = []
    monkeypatch.setattr(module, 'transitions', lambda: (entries, []))
    assert problem_of(plain, plain['reference']) is None
    # No entry: held exactly, marked or not.
    assert problem_of(marked, named) is not None
    old, new = 'refusal INTERNAL_ERROR', 'refusal UNSUPPORTED_REQUEST'
    # A proposed entry excuses nothing.
    entries[:] = [('P-1', 'proposed', 'question.*', old, new)]
    assert 'only proposed' in (problem_of(marked, named) or '')
    # An approved one covers exactly its transition and its glob.
    entries[:] = [('A-1', 'approved', 'question.*', old, new)]
    assert problem_of(marked, named) is None
    assert problem_of(BY_ID['question.first-of-fact'], named) is not None
    entries[:] = [('A-1', 'approved', 'question.*', old, 'success')]
    assert problem_of(marked, named) is not None
    # The core's refusal of multi-category text in place of the
    # reference's success is a success -> refusal transition: held unless
    # an approved entry declares exactly it.
    refused = copy.deepcopy(plain['reference'])
    refused['outcome'] = {
        'result': 'refusal',
        'stage': 'from_source',
        'error': 'UnsupportedRequestError',
        'code': 'UNSUPPORTED_REQUEST',
    }
    for part in ('schema', 'payload', 'result', 'wire'):
        refused[part] = None
    entries[:] = []
    assert problem_of(plain, refused) is not None
    entries[:] = [('A-2', 'approved', 'sep.*', 'success', new)]
    assert problem_of(plain, refused) is None
    # Each part of a success counts, the wire included (success ->
    # success, undeclared).
    entries[:] = []
    for part in PARTS[1:]:
        changed = copy.deepcopy(plain['reference'])
        changed[part] = None
        assert changed != plain['reference'], part
        assert problem_of(plain, changed) is not None, part
    # The column sent as categorical, or as lists of parts, fails.
    for edit in (
        lambda c: c.update(
            declared={'dtype': 'string', 'stype': 'categorical'}
        ),
        lambda c: c.update(kinds=['list']),
        lambda c: c.update(digest='0' * 64),
    ):
        changed = copy.deepcopy(plain['reference'])
        edit(changed['wire'][0]['context.related_tables.users'])
        assert problem_of(plain, changed) is not None


def test_the_register_reading_is_by_ref_status_and_transition() -> None:
    ref = R.REGISTER_REF
    document = {
        'entries': [
            {
                'id': 'A-1',
                'status': 'approved',
                'refs': [
                    f'{ref}question.* refusal INTERNAL_ERROR -> refusal '
                    'UNSUPPORTED_REQUEST'
                ],
            },
            {
                'id': 'A-2',
                'status': 'proposed',
                'refs': [
                    f'{ref}text.time-* refusal INVALID_REQUEST -> success'
                ],
            },
            {
                'id': 'A-3',
                'status': 'rejected',
                'refs': [f'{ref}sep.* success -> refusal UNSUPPORTED_REQUEST'],
            },
            {
                'id': 'A-4',
                'status': 'approved',
                'refs': [
                    'contract/empty-null-vectors.json#empty.* success -> success',
                    f'{ref}sep.*',
                    f'{ref}sep.* success -> refusal unsupported',
                ],
            },
        ]
    }
    read, malformed = transitions(document)
    assert read == [
        (
            'A-1',
            'approved',
            'question.*',
            'refusal INTERNAL_ERROR',
            'refusal UNSUPPORTED_REQUEST',
        ),
        (
            'A-2',
            'proposed',
            'text.time-*',
            'refusal INVALID_REQUEST',
            'success',
        ),
    ]
    assert malformed == [
        f'A-4: {ref}sep.*',
        f'A-4: {ref}sep.* success -> refusal unsupported',
    ]
