# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.32e part 1's whole-table time inference on ``python-core``, held to
the reference's vectors.

    python -m pytest contract/tests/test_t132e_time_inference.py -q

Written by the package's independent test author from the brief (honour a
named ``time_columns``; with none named, infer the time column as the
reference does over the whole table, not a 1,000-row preview; lift the
``time_whole_table`` refusal; register any remaining difference with a
binding transition) and the reference, before the implementation.
``contract/time-inference-vectors.json`` is recorded from the reference by
``contract/tools/time_inference_record.py``, whose ``vectors``, ``run``
and ``project`` this test calls, so both routes are observed and projected
by the same code; ``sdm-testkit``'s ``tests/t132e_time_inference.rs`` holds
``rust-host`` to the same vectors, so the two hosts agree through them.

- Brief, scope 1 ("infer the time column the way the reference does over
  the whole table" and "lift the ``time_whole_table`` refusal"), over
  generated tables of 999, 1,000, 1,001 and 5,000 rows: timestamp, date,
  time-of-day and zoned columns that are not the key, nulls in the time
  column, values that differ from the first 1,000 rows only after them,
  with and without ``time_columns``. Each vector runs through
  ``run_python.py --mode python-core``'s ``Schema.from_source``, forced
  onto the core's DuckDB routes. Its public answer must be the
  reference's: the outcome (result, stage, class, code), ``events`` as the
  schema describes it (each column's dtype and stype, the key, the time
  column chosen, the duplicate-key count) and the warning classes.
- Brief, scope 1 ("honour a named ``time_columns``, no inference, no
  refusal"), the package's exit: on every vector of 1,000 rows or more
  whose ``time_columns`` names a column the reference takes, the core's
  answer is a success whose time column is that column
  (:func:`test_a_named_time_column_is_honoured_on_a_large_table`), however
  the inference would have chosen.
- Brief, scope 1 ("register any remaining difference from the reference
  with a binding transition"): a vector may differ from the reference only
  under an approved entry of ``docs/design/rust-core/register.yaml`` whose
  ``refs`` name it with the transition it makes, T1.32a's form:
  ``contract/time-inference-vectors.json#<id or glob> <from> -> <to>``,
  each side ``success`` or ``refusal <CODE>``. A ref covers exactly the
  transition it declares; a proposed entry excuses nothing. The
  recorder's ``register.required`` marks where a difference is expected
  (the reference fails unnamed or with INTERNAL_ERROR); it excuses nothing
  by itself. At recording, the reference answered every vector with a
  success or a named INVALID_REQUEST, so no vector is marked: every one
  is held to the reference exactly unless the owner approves an entry.
- Negative controls (the brief template's): a ``time_columns`` naming a
  column that does not exist, the key, a number, or a table that does not
  exist is refused by name (INVALID_REQUEST at ``from_source``) at every
  size, as the reference refuses it.
- The vectors themselves: recorded at the reference of
  ``contract/reference-env.txt``, every vector of the recorder present with
  its setup, the marks the recorder derives, and DuckDB's own counts show
  each table is what it names (a column whose preview is constant but
  whose table is not, nulls, the four column kinds).

Gating (``contract/tests/t132e_gate.py``): ``t132e-pending.json``'s
``pending`` lists the tests that failed here when this was written. Each
still runs and must still fail; one that passes fails, naming itself,
until it is taken out of the list, which may only shrink (within
``authored``). The implementer removes ids as they pass and changes
nothing else of this file, the vectors or the recorder.
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
VECTORS = ROOT / 'contract/time-inference-vectors.json'
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
    'time_inference_record', ROOT / 'contract/tools/time_inference_record.py'
)
G = _module('t132e_gate', ROOT / 'contract/tests/t132e_gate.py')
RECORDED: dict[str, Any] = json.loads(VECTORS.read_text(encoding='utf-8'))
BY_ID = {vector['id']: vector for vector in RECORDED['vectors']}


def named_column(vector: dict[str, Any]) -> str | None:
    r"""The column ``time_columns`` names for ``events``, if it names one."""
    named = vector['time_columns'] or {}
    column = named.get(R.TABLE)
    return column if isinstance(column, str) else None


#: The exit's vectors: 1,000 rows or more, ``time_columns`` naming a column
#: of ``events``, and the reference taking it (derived from the vectors).
HONOURED = [
    vector['id']
    for vector in RECORDED['vectors']
    if vector['rows'] >= R.PREVIEW_ROWS
    and named_column(vector) is not None
    and vector['reference']['outcome']['result'] == 'success'
    and vector['reference']['schema']['time_column'] == named_column(vector)
]


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

    routes = sorted(
        {*_routing.FROM_SOURCE_DUCKDB, *_routing.FROM_SOURCE_DUCKDB_TABLES}
    )
    tmp = tmp_path_factory.mktemp('t132e-vectors')
    return R.run('python-core', R.vectors(), tmp, routes=routes)


@pytest.mark.parametrize(
    'vector_id',
    [pytest.param(v['id'], id=v['id']) for v in RECORDED['vectors']],
)
def test_python_core_infers_the_time_column_as_the_reference(
    vector_id: str, core_runs: dict[str, Any]
) -> None:
    vector = BY_ID[vector_id]

    def check() -> None:
        observation = core_runs[vector_id]
        assert observation['core'] is not None, 'the core was not reached'
        got = R.project(observation)
        problem = problem_of(vector, got)
        assert problem is None, problem

    G.held('python-core', vector_id, check)


@pytest.mark.parametrize('vector_id', [pytest.param(i, id=i) for i in HONOURED])
def test_a_named_time_column_is_honoured_on_a_large_table(
    vector_id: str, core_runs: dict[str, Any]
) -> None:
    r"""The exit: ``time_columns`` naming a column of a table of 1,000 rows
    or more is taken as named, with no refusal, on the core route."""
    vector = BY_ID[vector_id]
    named = named_column(vector)

    def check() -> None:
        observation = core_runs[vector_id]
        assert observation['core'] is not None, 'the core was not reached'
        got = R.project(observation)
        assert got['outcome']['result'] == 'success', got['outcome']
        assert isinstance(got['schema'], dict), got['schema']
        assert got['schema']['time_column'] == named, got['schema']

    G.held(
        'exits',
        f'py:test_a_named_time_column_is_honoured_on_a_large_table[{vector_id}]',
        check,
    )


# --- the vectors ---------------------------------------------------------------


def test_the_vectors_are_the_reference_s() -> None:
    assert RECORDED['format'] == R.FORMAT
    assert RECORDED['recorded_by'] == 'contract/tools/time_inference_record.py'
    assert RECORDED['cell'] == 'ref-pandas3'
    assert RECORDED['register_ref'] == R.REGISTER_REF
    assert RECORDED['table'] == R.TABLE
    assert RECORDED['preview_rows'] == R.PREVIEW_ROWS == 1_000
    assert RECORDED['sizes'] == list(R.SIZES) == [999, 1_000, 1_001, 5_000]
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    assert f'reference: {RECORDED["reference"]}' in env
    # Every vector the recorder makes, once, with its rows, setup and
    # time_columns.
    made = R.vectors()
    assert [v['id'] for v in RECORDED['vectors']] == [v['id'] for v in made]
    for vector in made:
        recorded = BY_ID[vector['id']]
        for key in ('scenario', 'rows', 'setup', 'time_columns', 'tables'):
            assert recorded[key] == vector[key], (vector['id'], key)


def test_the_marks_are_the_recorder_s() -> None:
    # Derived from the reference's answer, never edited.
    for vector in RECORDED['vectors']:
        why = R.register(vector['reference'])
        assert vector['register'] == {'required': bool(why), 'why': why}, (
            vector['id']
        )


def test_each_vector_is_a_case_the_runner_takes() -> None:
    # The cases run_python.py is handed validate, ids unique; time_columns
    # reaches both the public call and the core's describe_source.
    import tempfile

    runner = _module('run_python', ROOT / 'contract/runners/run_python.py')
    with tempfile.TemporaryDirectory() as tmp:
        for vector in R.vectors():
            case = R.case_for(vector)
            assert case['until'] == 'from_source'
            assert case['input']['setup'] == vector['setup']
            named = vector['time_columns']
            source = case['core']['describe_source']['source']
            assert source['time_columns'] == (named or {})
            given = {}
            if vector['tables']:
                given['tables'] = vector['tables']
                assert source['tables'] == [
                    {'name': n, 'source': t}
                    for n, t in vector['tables'].items()
                ]
            else:
                assert source['tables'] == []
            if named is not None:
                given['time_columns'] = named
            if given:
                assert case['input']['from_source'] == given
            else:
                assert 'from_source' not in case['input']
            (Path(tmp) / f'{case["id"]}.json').write_text(
                json.dumps(case), encoding='utf-8'
            )
        loaded = runner.load_cases([Path(tmp)])
    assert len(loaded) == len(RECORDED['vectors'])


def test_the_vectors_are_not_vacuous() -> None:
    vectors = RECORDED['vectors']
    answers = [v['reference'] for v in vectors]
    successes = [a for a in answers if a['outcome']['result'] == 'success']
    refusals = [a for a in answers if a['outcome']['result'] == 'refusal']
    assert len(successes) >= 60 and len(refusals) >= 12
    for answer in successes:
        assert answer['outcome']['stage'] == 'from_source'
        assert isinstance(answer['schema'], dict)
    for answer in refusals:
        assert answer['outcome']['code'] == 'INVALID_REQUEST', answer
    # Every size the brief names, for every scenario, and most of them past
    # a full preview.
    for scenario in R.SCENARIOS:
        rows = sorted(
            v['rows'] for v in vectors if v['scenario'] == scenario['id']
        )
        assert rows == [999, 1_000, 1_001, 5_000], scenario['id']
    # With and without time_columns, and naming None.
    assert any(v['time_columns'] is None for v in vectors)
    assert any(named_column(v) for v in vectors)
    assert any(v['time_columns'] == {R.TABLE: None} for v in vectors)
    # The exit's vectors: every large size, several scenarios.
    assert {BY_ID[i]['rows'] for i in HONOURED} >= {1_000, 1_001, 5_000}
    assert len({BY_ID[i]['scenario'] for i in HONOURED}) >= 6
    # The time column chosen differs with the size where the brief says it
    # must: a preview would choose otherwise than the whole table.
    chosen = {
        v['id']: (v['reference']['schema'] or {}).get('time_column')
        for v in vectors
        if isinstance(v['reference']['schema'], dict)
    }
    assert chosen['late-change.1000'] is None
    assert chosen['late-change.1001'] == 'happened_at'
    assert chosen['late-values.1001'] is None
    assert chosen['late-values.5000'] == 'happened_at'
    assert chosen['earliest-after-preview.1000'] == 'shipped_at'
    assert chosen['earliest-after-preview.1001'] == 'billed_at'
    # DuckDB's own counts: the tables are what they name.
    facts = {v['id']: v['duckdb'] for v in vectors}
    for vector in vectors:
        assert facts[vector['id']]['rows'] == vector['rows'], vector['id']
    kinds = {
        column['type']
        for fact in facts.values()
        for column in fact['columns'].values()
    }
    assert {'TIMESTAMP', 'DATE', 'TIME', 'TIMESTAMP WITH TIME ZONE'} <= kinds
    late = facts['late-change.5000']['columns']['happened_at']
    assert late['preview']['distinct'] == 1 and late['whole']['distinct'] > 1
    empty = facts['late-values.5000']['columns']['happened_at']
    assert empty['preview']['held'] == 0 and empty['whole']['held'] == 4_000
    nulls = facts['nulls.5000']['columns']['happened_at']
    assert 0 < nulls['whole']['held'] < 5_000
    starts = facts['earliest-after-preview.5000']['columns']
    assert (
        starts['billed_at']['preview']['min']
        > starts['shipped_at']['preview']['min']
    )
    assert (
        starts['billed_at']['whole']['min']
        < starts['shipped_at']['whole']['min']
    )
    placeholder = facts['placeholder-after-preview.5000']['columns'][
        'shipped_at'
    ]
    assert placeholder['whole']['min'].startswith('1900-01-01')
    assert not placeholder['preview']['min'].startswith('1900')


def test_the_fallback_vectors_are_the_preview_s_choice() -> None:
    r"""The owner's decision of 2026-10-13 (P1B-38 rejected: the core
    matches the reference's fallback): where the one query over every row
    fails, the reference decides from its preview, keeps a column the
    preview would refuse as unconfirmed, and says so when it takes one.
    The vectors are seen to be that: a view of 200,000 rows whose time
    columns raise when read whole, the preview's 1,000 rows readable."""
    fallback = [
        v for v in RECORDED['vectors'] if v['scenario'].startswith('fallback-')
    ]
    assert {v['scenario'] for v in fallback} == {f['id'] for f in R.FALLBACK}
    for vector in fallback:
        assert vector['rows'] == R.FALLBACK_ROWS
        assert vector['tables'] == {R.TABLE: R.TABLE, 'users': 'users'}
        assert vector['reference']['outcome']['result'] == 'success'
        facts = vector['duckdb']
        assert facts['rows'] == R.FALLBACK_ROWS
        times = [
            name
            for name, column in facts['columns'].items()
            if column['type'].startswith('TIMESTAMP')
        ]
        assert times, vector['id']
        for name in times:
            assert facts['columns'][name]['whole'] == 'raises', (
                vector['id'],
                name,
            )
    chosen = {
        v['scenario']: (
            v['reference']['schema']['time_column'],
            v['reference']['warnings'],
        )
        for v in fallback
    }
    assert chosen == {
        'fallback-plain': ('happened_at', []),
        'fallback-unconfirmed': ('happened_at', ['SDMWarning']),
        'fallback-unconfirmed-empty': ('happened_at', ['SDMWarning']),
        'fallback-doubted-beside-plain': ('billed_at', []),
        'fallback-unconfirmed-named': ('happened_at', []),
    }


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
    # Fail closed: the comparison itself, on seeded differences and seeded
    # register entries.
    module = sys.modules[__name__]
    large = BY_ID['one-timestamp.5000']
    assert R.outcome_class(large['reference']['outcome']) == 'success'
    refused = copy.deepcopy(large['reference'])
    refused['outcome'] = {
        'result': 'refusal',
        'stage': 'from_source',
        'error': 'UnsupportedRequestError',
        'code': 'UNSUPPORTED_REQUEST',
    }
    refused['schema'] = None
    refused['warnings'] = []
    entries: list[tuple[str, str, str, str, str]] = []
    monkeypatch.setattr(module, 'transitions', lambda: (entries, []))
    assert problem_of(large, large['reference']) is None
    # Today's time_whole_table refusal, with no entry: fails.
    assert problem_of(large, refused) is not None
    # A proposed entry excuses nothing.
    entries[:] = [
        (
            'P-1',
            'proposed',
            'one-timestamp.*',
            'success',
            'refusal UNSUPPORTED_REQUEST',
        )
    ]
    assert 'only proposed' in (problem_of(large, refused) or '')
    # An approved one covers exactly its transition and its glob.
    entries[:] = [
        (
            'A-1',
            'approved',
            'one-timestamp.*',
            'success',
            'refusal UNSUPPORTED_REQUEST',
        )
    ]
    assert problem_of(large, refused) is None
    assert problem_of(BY_ID['nulls.5000'], refused) is not None
    entries[:] = [
        (
            'A-1',
            'approved',
            'one-timestamp.*',
            'success',
            'refusal INVALID_REQUEST',
        )
    ]
    assert problem_of(large, refused) is not None
    # A success with another answer: each part counts, the time column
    # chosen above all.
    chose = BY_ID['earliest-after-preview.5000']
    assert chose['reference']['schema']['time_column'] == 'billed_at'
    entries[:] = []
    for part, change in (
        ('time_column', 'shipped_at'),
        ('time_column', None),
        ('primary_key', None),
        ('duplicate_keys', 1),
    ):
        got = copy.deepcopy(chose['reference'])
        got['schema'][part] = change
        assert problem_of(chose, got) is not None, part
    got = copy.deepcopy(chose['reference'])
    got['schema']['columns'][-1]['dtype'] = 'string'
    assert problem_of(chose, got) is not None
    got = copy.deepcopy(chose['reference'])
    got['warnings'] = ['UserWarning']
    assert problem_of(chose, got) is not None
    # A named refusal held: another code, or a success, fails.
    control = BY_ID['named-missing-column.5000']
    assert R.outcome_class(control['reference']['outcome']) == (
        'refusal INVALID_REQUEST'
    )
    assert problem_of(control, refused) is not None
    taken = copy.deepcopy(large['reference'])
    assert problem_of(control, taken) is not None


def test_the_exit_is_derived_and_fails_closed() -> None:
    # Every exit vector names a column of events and is past a full
    # preview; a vector the reference refuses is no exit vector.
    for vector_id in HONOURED:
        vector = BY_ID[vector_id]
        assert vector['rows'] >= 1_000
        assert (
            named_column(vector) == vector['reference']['schema']['time_column']
        )
    assert 'named-missing-column.5000' not in HONOURED
    assert 'three-kinds-named-none.5000' not in HONOURED
    assert 'late-change-named.1001' in HONOURED
    assert 'late-values-named.5000' in HONOURED


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
                'refs': [f'{ref}late-* success -> refusal UNSUPPORTED_REQUEST'],
            },
            {
                'id': 'A-2',
                'status': 'proposed',
                'refs': [f'{ref}nulls.* success -> refusal INVALID_REQUEST'],
            },
            {
                'id': 'A-3',
                'status': 'rejected',
                'refs': [f'{ref}constant.* success -> success'],
            },
            {
                'id': 'A-4',
                'status': 'approved',
                'refs': [
                    'contract/other.json#late-* success -> success',
                    f'{ref}late-*',
                    f'{ref}late-* Success -> success',
                ],
            },
        ]
    }
    read, malformed = transitions(document)
    assert read == [
        ('A-1', 'approved', 'late-*', 'success', 'refusal UNSUPPORTED_REQUEST'),
        ('A-2', 'proposed', 'nulls.*', 'success', 'refusal INVALID_REQUEST'),
    ]
    assert malformed == [
        f'A-4: {ref}late-*',
        f'A-4: {ref}late-* Success -> success',
    ]
