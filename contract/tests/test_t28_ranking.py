# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.8's exits on ``python-core`` (the package's test author's): ranking,
end to end (the design's T2.8a and T2.8b rows).

    SDM_POSTGRES_TEST_DSN=... python -m pytest \
        contract/tests/test_t28_ranking.py -q

Written from the design's rows (each item an exit) and the reference
(python-ref at ``contract/reference-env.txt``'s commit), before the
implementation. ``sdm-testkit``'s ``tests/t28_ranking.rs`` holds
``rust-host`` to the same cases and vectors.

The S1 cases, ``contract/cases/s1/rank.*.json`` and ``forecast.*.json``,
run on DuckDB, SQLite and Postgres over the shop base with ``items`` and
``buys`` (a ranking of the items each user buys in 30 days). The scripted
endpoint answers no probabilities, so they judge the materialize side
(T2.8a): each part's payload, and every refusal before a call. The
vectors, ``contract/ranking-vectors.json``, recorded from the reference by
``contract/tools/ranking_record.py``, whose ``observe`` these tests call on
the core's routes, judge the predict side (T2.8b) on DuckDB, each model
call answered by a script with the probabilities a ranking orders by.

Each test's one line says the exit it proves:

- control: each case is the design row item it names (:data:`CASES`),
  runs on all three databases, and python-ref's line for it on each is
  recorded at its case file, its outcome the case's;
- control: the vectors are the reference's, their inputs the recorder's,
  each exit has one, and they are not vacuous (ties, a cut, an unscored
  candidate, a NaN, an off-type answer);
- control: each case refused before any call is refused, on the
  reference, by the rule its argument meets, and the same call with that
  argument changed is gathered (the recorder's ``controls``);
- T2.8a, "``top_k``, candidates, the decomposition in ``sdm-request``;
  ``problem_type`` for ranking; the forecast's ``top_k`` and
  ``MOST_PARTS`` refusals in the core": python-core answers each case on
  each database as python-ref does (its outcome, and each request the
  scripted endpoint received but its request id), and the core does it: the
  host hands the core the question as asked (one ``materialize``, its query
  not a part's), and a refusal is the core's (the stage's core operation
  ran and refused with the reference's code);
- T2.8b, "scores, rankings, ``prediction_statistic``, PD-1, reassembly;
  ``response_field`` for rankings; ``instance_id_type``,
  ``prediction_type``": python-core answers each vector as python-ref
  does: the result through ``to_pandas()`` and ``.items``, its outputs, the
  candidates left unscored and the notices' classes, every call's asked
  outputs and status, or the refusal's code, class, the HTTP status an
  endpoint's refusal carries and, for a part, the sentence saying which
  candidate it was asked for.

PD-1 ("NaN ranks last") cannot be reached from the public calls: the
reference refuses a non-finite probability, the only score a ranking
orders by, ``INVALID_RESPONSE`` said of the part's candidate, so the
vectors ``pd1.*`` hold the core to that refusal; a NaN inside a
``rankings`` member an answer carries is given back as it came, never
ordered.

Gating: ``t28-pending.json``'s ``pending`` names, as ``py:<case key>`` and
``py:vector:<name>``, the keys whose test failed when written. Each must
still fail (strict); once one passes it fails, naming itself, until it is
taken out. ``pending`` stays within ``authored``. Where
``SDM_POSTGRES_TEST_DSN`` names no server and ``SDM_REQUIRE_POSTGRES`` is
not set, the Postgres keys have nothing to hold and return without
asserting; with it set, a missing server fails (H26).
"""

from __future__ import annotations

import copy
import functools
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'contract/runners/run_python.py'
RECORDER = ROOT / 'contract/tools/ranking_record.py'
VECTORS = ROOT / 'contract/ranking-vectors.json'
CASE_DIR = ROOT / 'contract/cases/s1'
PENDING = Path(__file__).with_name('t28-pending.json')
DATABASES = ('duckdb', 'sqlite', 'postgres')

#: Each case and the design row item it proves.
CASES = {
    'rank.top-k': 'T2.8a: candidates, the decomposition, problem_type for '
    "ranking, the query's top_k; each part's payload",
    'rank.top-k-option': 'T2.8a: the top_k option, sent with no part, and '
    'candidates in the order given',
    'rank.top-k-zero': 'T2.8a: top_k outside 1 to 20, refused',
    'rank.no-candidates': 'T2.8a: a ranking names its candidates',
    'rank.candidates-empty': 'T2.8a: no candidate to rank, refused',
    'rank.candidates-not-a-ranking': 'T2.8a: candidates beside a query that '
    'ranks nothing, refused',
    'rank.most-parts': 'T2.8a: more candidates than MOST_PARTS, refused',
    'rank.batched': 'T2.8c: the batched-ranking refusal (more entities than '
    'one call scores)',
    'rank.unscored-candidate': 'T2.8a and T2.8b: a candidate whose context '
    'holds one class left out, unscored, before any call',
    'rank.all-unscored': 'T2.8b: no candidate left to rank, refused before any '
    'call',
    'forecast.top-k': "T2.8a: the forecast's top_k refusal in the core",
    'forecast.most-parts': "T2.8a: the forecast's MOST_PARTS refusal in the "
    'core',
}
CASE_IDS = tuple(CASES)
#: The sentence the reference adds to a part's refusal.
PART = 'This is the question asked for candidate '


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def runner() -> ModuleType:
    return _module('t28_run_python', RUNNER)


def recorder() -> ModuleType:
    return _module('t28_ranking_record', RECORDER)


def key(case_id: str, database: str) -> str:
    return runner().case_key(case_id, database)


CASE_KEYS = tuple(key(c, d) for d in DATABASES for c in CASE_IDS)


@functools.cache
def vectors() -> dict[str, Any]:
    return json.loads(VECTORS.read_text(encoding='utf-8'))


VECTOR_NAMES = tuple(v['name'] for v in vectors()['vectors'])


def on_rust_host(vector: dict[str, Any]) -> bool:
    r"""Whether rust-host replays ``vector``: a call a case can make (its
    materialize arguments a case's, no edit, the default outputs), whose
    answers its JSON reader takes as Python's does (no NaN in a member it
    gives back as written)."""
    return (
        set(vector['materialize'])
        <= {'indices', 'anchor_time', 'candidates', 'top_k'}
        and vector['edit'] is None
        and not vector['predict']
        and 'nan-score' not in vector['name']
    )


def held(name: str, check: Callable[[], None]) -> None:
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
        'contract/tests/t28-pending.json pending'
    )


def test_the_pending_list_only_shrinks() -> None:
    r"""Control: ``pending`` names only keys the test author wrote, and
    ``authored`` is every case's key on each database and every vector on
    python-core, and on rust-host each case on DuckDB and SQLite (its
    Postgres pass replays python-core's fetches, which ``cross-host
    (postgres)`` holds) and each vector it replays."""
    gating = json.loads(PENDING.read_text(encoding='utf-8'))
    assert set(gating['pending']) <= set(gating['authored'])
    rust_vectors = [v['name'] for v in vectors()['vectors'] if on_rust_host(v)]
    assert set(gating['authored']) == (
        {f'py:{k}' for k in CASE_KEYS}
        | {f'py:vector:{n}' for n in VECTOR_NAMES}
        | {f'rust:{k}' for k in CASE_KEYS if not k.endswith('@postgres')}
        | {f'rust:vector:{n}' for n in rust_vectors}
    )


# --- the cases -------------------------------------------------------------------


@functools.cache
def cases() -> dict[str, tuple[dict[str, Any], str]]:
    r"""The T2.8 cases by id, each with its digest, as the runner loads
    them."""
    loaded = runner().load_cases([CASE_DIR / f'{c}.json' for c in CASE_IDS])
    return {case['id']: (case, digest) for case, digest in loaded}


@functools.cache
def reference_lines() -> dict[str, dict[str, Any]]:
    r"""python-ref's committed lines, by key."""
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    commit = next(
        line.split(': ', 1)[1].strip()
        for line in env.splitlines()
        if line.startswith('reference: ')
    )
    path = ROOT / f'contract/reference/{commit}.jsonl'
    return {
        json.loads(line)['case']: json.loads(line)
        for line in path.read_text(encoding='utf-8').splitlines()
    }


def _reference_commit() -> str:
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    return next(
        line.split(': ', 1)[1].strip()
        for line in env.splitlines()
        if line.startswith('reference: ')
    )


def test_the_cases_are_the_design_rows() -> None:
    r"""Control, "S1 ranking cases at materialize and at predict": each case
    is the row item :data:`CASES` names, an acceptance case of S1 on all
    three databases over one data set, and not vacuous: each input is the
    one its item needs."""
    data = recorder().data_case()
    for case_id, (case, _) in cases().items():
        given = case['input']
        spec = given['materialize']
        assert case['status'] == 'acceptance' and case['slice'] == 'S1'
        assert runner().case_databases(case) == DATABASES, case_id
        assert given['base'] == 'shop', case_id
        assert given['setup'] == data['input']['setup'], case_id
        ranks = 'RANK TOP' in given['pql']
        assert ranks == (
            case_id.startswith('rank.')
            and not case_id.endswith('not-a-ranking')
        ), case_id
        if case_id.startswith('forecast.'):
            assert 'FORECAST' in given['pql']
            assert 'candidates' not in spec

    def spec_of(c: str) -> dict[str, Any]:
        return cases()[c][0]['input']['materialize']

    assert spec_of('rank.top-k')['candidates'] == [0, 1, 2]
    assert 'top_k' not in spec_of('rank.top-k')
    assert spec_of('rank.top-k-option')['top_k'] == 1
    assert spec_of('rank.top-k-option')['candidates'] == [2, 0]
    assert spec_of('rank.top-k-zero')['top_k'] == 0
    assert 'candidates' not in spec_of('rank.no-candidates')
    assert spec_of('rank.candidates-empty')['candidates'] == []
    assert len(spec_of('rank.most-parts')['candidates']) == 201
    assert len(spec_of('rank.batched')['indices']) == 10_001
    assert spec_of('rank.unscored-candidate')['candidates'] == [0, 9, 1]
    assert spec_of('rank.all-unscored')['candidates'] == [9, 8]
    # The cases answered at predict are answered with probabilities, as a
    # model answers a ranking's parts (the scripted endpoint's
    # probabilities, input.probabilities), so their order is judged: by
    # the endpoint's rule, the users of rank.top-k do not rank the
    # candidates in the order given.
    answered = {c for c, n in CALLS.items() if n}
    for case_id, (case, _) in cases().items():
        assert case['input'].get('probabilities', False) == (
            case_id in answered
        ), case_id
    by_rule = [
        sorted(range(3), key=lambda k: -((3 * k + 5 * row) % 9 + 1))[:2]
        for row in range(3)
    ]
    assert any(top != [0, 1] for top in by_rule), by_rule
    assert spec_of('forecast.top-k')['top_k'] == 2
    assert 'FORECAST 201 ' in cases()['forecast.most-parts'][0]['input']['pql']
    # The candidates 9 and 8 name no item, so no user buys them: a context
    # of one class; 0, 1 and 2 are bought.
    connection = _built(cases()['rank.top-k'][0])
    try:
        bought = {
            row[0]
            for row in connection.execute(
                'SELECT DISTINCT item_id FROM buys'
            ).fetchall()
        }
    finally:
        connection.close()
    assert {0, 1, 2} <= bought and not {8, 9} & bought


def _built(case: dict[str, Any]) -> Any:
    import duckdb

    connection = duckdb.connect()
    for statement in [
        *runner().load_bases()[case['input']['base']],
        *case['input'].get('setup', []),
    ]:
        connection.execute(statement)
    return connection


def _outcome(line: dict[str, Any]) -> tuple[Any, ...]:
    outcome = line['outcome']
    return (
        outcome['result'],
        outcome['stage'],
        outcome.get('error'),
        outcome.get('code'),
    )


#: How many calls python-ref makes for each case ending at predict: one
#: per scored candidate.
CALLS = {
    'rank.top-k': 3,
    'rank.top-k-option': 2,
    'rank.unscored-candidate': 2,
    'rank.all-unscored': 0,
}


@pytest.mark.parametrize('case_key', CASE_KEYS)
def test_python_ref_s_line_is_at_its_case_file(case_key: str) -> None:
    r"""Control, "cases written for all three databases": python-ref's line
    for the case on the database, recorded at the case file, ends as the
    case's ``expect`` says, with one call per scored candidate."""
    case_id = case_key.split('@')[0]
    case, digest = cases()[case_id]
    line = reference_lines().get(case_key)
    assert line is not None, f'{case_key}: no python-ref line'
    assert line['identity']['runner'] == 'python-ref'
    assert line['identity']['inputs']['case'] == digest
    expect = case['expect']
    assert _outcome(line) == (
        expect['outcome'],
        expect['stage'],
        expect.get('error'),
        expect.get('code'),
    ), case_key
    assert len(line['public']['payloads']) == CALLS.get(case_id, 0), case_key


# --- the vectors -----------------------------------------------------------------


def test_the_vectors_are_the_reference_s_and_not_vacuous() -> None:
    r"""Control: the vectors were recorded from the reference by the
    committed recorder, their inputs are the recorder's, every exit has one,
    and they exercise what they name."""
    recorded = vectors()
    assert recorded['recorded_by'] == 'contract/tools/ranking_record.py'
    assert recorded['reference'] == _reference_commit()
    assert recorded['data_case'] == recorder().DATA_CASE == 'rank.top-k'
    assert recorded['exits'] == recorder().EXITS
    inputs = [
        {k: v for k, v in each.items() if k != 'expected'}
        for each in recorded['vectors']
    ]
    assert inputs == json.loads(json.dumps(recorder().VECTORS))
    assert {v['covers'] for v in inputs} == set(recorder().EXITS)
    by = {v['name']: v['expected'] for v in recorded['vectors']}
    # Ties and every order: entity 1 scores candidates 0 and 1 alike, and
    # keeps them in the candidates' order.
    ranked = by['scores.rank-top-2']
    assert ranked['outcome'] == 'result' and len(ranked['calls']) == 3
    assert ranked['columns'] == ['row_index', 'id', 'prediction', 'rankings']
    row = ranked['items'][1]
    assert row['prediction'] == {'list': [{'int': '0'}, {'int': '1'}]}
    scores = [m['map'][1][1] for m in row['rankings']['list']]
    assert scores[0] == scores[1]
    # top_k: the option cuts to one, or keeps three past the query's two.
    assert len(by['top-k.option-1']['items'][0]['prediction']['list']) == 1
    assert len(by['top-k.option-3']['items'][0]['prediction']['list']) == 3
    assert by['top-k.outputs-rankings']['columns'] == [
        'row_index',
        'id',
        'rankings',
    ]
    assert by['top-k.outputs-probabilities']['code'] == 'INVALID_REQUEST'
    # An unscored candidate: no call for it, named, one notice.
    unscored = by['candidates.unscored-one']
    assert unscored['unscored'] == {'list': [{'int': '9'}]}
    assert len(unscored['calls']) == 2 and unscored['notices'] == ['SDMWarning']
    assert by['candidates.all-unscored']['calls'] == []
    assert len(by['candidates.repeated']['calls']) == 2
    # PD-1: a NaN never ranks; refused of the candidate it was asked for.
    for name, candidate, calls in (
        ('pd1.nan-first-call', 0, 1),
        ('pd1.nan-later-call', 1, 2),
        ('pd1.nan-as-text', 0, 1),
    ):
        nan = by[name]
        assert (nan['outcome'], nan['code']) == ('refused', 'INVALID_RESPONSE')
        assert f'{PART}{candidate}, one of 3' in nan['message'], name
        assert len(nan['calls']) == calls, name
        assert 'NaN' in nan['calls'][-1]['answer'] or (
            '"nan"' in nan['calls'][-1]['answer']
        )
    failed = by['parts.failed-call-1']
    assert failed['class'] == 'NimRequestError' and len(failed['calls']) == 2
    assert f'{PART}1, one of 3' in failed['message']
    # response_field: the answer's scores and rankings come back as given.
    assert 'scores' in by['response-field.binary-scores']['columns']
    assert 'rankings' in by['response-field.binary-rankings']['columns']
    nan_member = by['response-field.binary-rankings-nan-score']
    assert {'f64': 'nan'} in [
        m['map'][1][1] for m in nan_member['items'][0]['rankings']['list']
    ]
    assert 'scores' in by['response-field.regression-scores']['columns']
    assert by['response-field.ranking-part-scores']['outcome'] == 'result'
    # prediction_statistic: refused for a ranking; sent as given otherwise.
    assert by['prediction-statistic.ranking']['code'] == 'INVALID_REQUEST'
    sent = by['prediction-statistic.edited-not-text']['calls'][0]['output']
    assert sent['prediction_statistic'] == 7
    mean = by['prediction-statistic.regression-mean']['calls'][0]['output']
    assert mean['prediction_statistic'] == 'mean'
    # instance_id_type and prediction_type: off-type values answered.
    assert by['instance-id.float']['outcome'] == 'result'
    assert by['prediction-type.binary-float']['dtypes'][2] == 'float64'
    assert by['prediction-type.binary-text']['rows'][0][2] == {'str': 'maybe'}
    assert by['prediction-type.multiclass-integer']['rows'][0][2] == {
        'int': '3'
    }


#: What the reference's refusal of each case refused before any call says,
#: so it is refused for the argument the case names.
REFUSED_FOR = {
    'rank.top-k-zero': 'top_k is 0',
    'rank.no-candidates': 'they have to be named',
    'rank.candidates-empty': 'candidates is empty',
    'rank.candidates-not-a-ranking': 'does not rank anything',
    'rank.most-parts': '201 candidates means 201 calls',
    'rank.batched': 'A part is not batched',
    'forecast.top-k': 'a forecast ranks nothing',
    'forecast.most-parts': '201 forecast steps means 201 calls',
}


def test_each_case_is_refused_for_the_argument_it_names() -> None:
    r"""Control, not vacuous: each case the reference refuses at
    materialize is refused, on the reference, by the rule its argument
    meets (the recorder's ``controls``: the case's own call and its
    message), and the same call with that argument changed is gathered, so
    the case is refused for nothing else."""
    recorded = {c['case']: c for c in vectors()['controls']}
    at_materialize = {
        case_id
        for case_id, (case, _) in cases().items()
        if case['expect']['stage'] == 'materialize'
    }
    assert set(recorded) == at_materialize == set(REFUSED_FOR)
    assert recorded.keys() == recorder().CONTROLS.keys()
    for case_id, each in recorded.items():
        case, _ = cases()[case_id]
        refused = each['refused']
        assert (refused['outcome'], refused['code']) == (
            'refused',
            case['expect']['code'],
        ), case_id
        assert REFUSED_FOR[case_id] in refused['message'], refused['message']
        assert each['change'] == recorder().CONTROLS[case_id]
        assert each['control']['outcome'] == 'result', (case_id, each)


# --- python-core against the reference ---------------------------------------------


def _server() -> bool:
    r"""Whether a Postgres server is named; where none is, one must not be
    required (H26)."""
    if os.environ.get('SDM_POSTGRES_TEST_DSN'):
        return True
    assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
        'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
    )
    return False


@functools.cache
def core_run(database: str) -> dict[str, Any]:
    r"""python-core over the T2.8 cases on ``database``, as that database's
    job runs them: each observation by its key, or under ``''`` why the run
    failed whole."""
    with tempfile.TemporaryDirectory(prefix='t28-') as tmp:
        given = Path(tmp) / 'cases'
        given.mkdir()
        for case_id in CASE_IDS:
            shutil.copy(CASE_DIR / f'{case_id}.json', given)
        out = Path(tmp) / 'python-core.jsonl'
        done = subprocess.run(
            [
                sys.executable,
                '-I',
                str(RUNNER),
                '--mode',
                'python-core',
                '--database',
                database,
                '--out',
                str(out),
                str(given),
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            check=False,
            timeout=1800,
        )
        if done.returncode != 0:
            return {'': done.stdout + done.stderr}
        return {
            json.loads(line)['case']: json.loads(line)
            for line in out.read_text(encoding='utf-8').splitlines()
        }


def payloads(line: dict[str, Any]) -> list[Any]:
    r"""Each request the scripted endpoint received, parsed, as the
    ``payload`` rule compares it: its request id, drawn at random for every
    request, left out (``compare-policy.yaml``)."""
    out = []
    for payload in line['public']['payloads']:
        parsed = copy.deepcopy(payload.get('parsed'))
        if isinstance(parsed, dict) and isinstance(
            parsed.get('metadata'), dict
        ):
            parsed['metadata'].pop('request_id', None)
        out.append(parsed)
    return out


def differences(want: Any, got: Any, at: str = '') -> list[str]:
    r"""Where ``got`` differs from ``want``, a few paths."""
    if isinstance(want, dict) and isinstance(got, dict):
        found = []
        for name in sorted(set(want) | set(got)):
            found += differences(want.get(name), got.get(name), f'{at}/{name}')
        return found[:8]
    if (
        isinstance(want, list)
        and isinstance(got, list)
        and len(want) == len(got)
    ):
        found = []
        for i, (w, g) in enumerate(zip(want, got, strict=True)):
            found += differences(w, g, f'{at}[{i}]')
        return found[:8]
    if want != got:
        if isinstance(want, list) and isinstance(got, list):
            return [
                f'{at}: {len(got)} items where the reference has {len(want)}'
            ]
        return [f'{at}: {got!r:.80} where the reference has {want!r:.80}']
    return []


@pytest.mark.parametrize('case_key', CASE_KEYS)
def test_python_core_answers_each_case_as_the_reference_does(
    case_key: str,
) -> None:
    r"""T2.8a's exits, "``top_k``, candidates, the decomposition in
    ``sdm-request``; ``problem_type`` for ranking; the forecast's ``top_k``
    and ``MOST_PARTS`` refusals in the core": python-core runs the case on
    the database on the core route, its outcome and each request the
    scripted endpoint received are python-ref's, and the core does the
    work: the host hands it the question as asked (one ``materialize``,
    never a part's query) and a refusal is the core's, given by the core
    operation of the stage it ends at with the reference's code."""
    database = case_key.partition('@')[2] or 'duckdb'
    if database == 'postgres' and not _server():
        return

    def check() -> None:
        run = core_run(database)
        assert '' not in run, run.get('')
        line = run.get(case_key)
        assert line is not None, f'{case_key}: not observed'
        case, _ = cases()[case_key.split('@')[0]]
        want = reference_lines()[case_key]
        assert line['identity']['native_loaded'] and line['core'] is not None
        assert _outcome(line) == _outcome(want), (
            f'{case_key}: {_outcome(line)} where python-ref has '
            f'{_outcome(want)}'
        )
        found = differences(payloads(want), payloads(line))
        assert not found, f'{case_key}: the payload differs: ' + '; '.join(
            found
        )
        if want['outcome']['result'] == 'success':
            # The ranking put back together: its columns and rows' digest.
            assert line['public']['result'] == want['public']['result'], (
                f'{case_key}: the result differs: '
                + '; '.join(
                    differences(
                        want['public']['result'], line['public']['result']
                    )
                )
            )
        ran = line['identity']['operations']
        asked = [
            each
            for each in line['core']['inputs']
            if each.get('op') == 'materialize'
        ]
        assert len(asked) == 1, (
            f'{case_key}: the core was asked {len(asked)} materialize '
            'operations, where it is asked the question once and splits it'
        )
        if ' WHERE ' not in case['input']['pql']:
            assert ' WHERE ' not in str(asked[0].get('query')), (
                f"{case_key}: the core was asked a part's query, "
                f'{asked[0].get("query")!r}, not the question'
            )
        stage = want['outcome']['stage']
        assert stage in ran, f'{case_key}: the core ran no {stage}: {ran}'
        if want['outcome']['result'] == 'refusal':
            error = line['core'].get('error') or {}
            assert error.get('code') == want['outcome']['code'], (
                f"{case_key}: the refusal is not the core's: the core "
                f'answered {error!r}'
            )

    held(case_key, check)


def _routes() -> str:
    from sdm_client import _routing

    return ','.join(
        dict.fromkeys(
            [
                *_routing.FROM_SOURCE_DUCKDB,
                *_routing.MATERIALIZE_DUCKDB,
                *_routing.PREDICT_DUCKDB,
            ]
        )
    )


def _dtypes(dtypes: list[str] | None) -> list[str] | None:
    r"""A frame's dtypes, pandas 2's ``object`` and pandas 3's ``str`` for
    a text column read as one."""
    if dtypes is None:
        return None
    return ['object' if d == 'str' else d for d in dtypes]


def held_to(got: dict[str, Any], want: dict[str, Any], name: str) -> None:
    r"""``got`` is the reference's outcome ``want``, as T2.8b's exits
    compare it."""
    assert (got['stage'], got['outcome']) == (want['stage'], want['outcome']), (
        f'{name}: {got["stage"]} {got["outcome"]} '
        f'({got.get("class")} {got.get("code")}: {got.get("message")}), '
        f'where the reference ends {want["stage"]} {want["outcome"]}'
    )
    assert [(c['output'], c['status']) for c in got['calls']] == [
        (c['output'], c['status']) for c in want['calls']
    ], f"{name}: the calls made are not the reference's"
    if want['outcome'] == 'refused':
        assert got['code'] == want['code'], f'{name}: {got["message"]}'
        assert want['class'] in got['mro'], f'{name}: {got["mro"]}'
        # An endpoint's refusal carries its HTTP status to the user, a local
        # error none.
        assert got['status_code'] == want['status_code'], (
            f'{name}: status {got["status_code"]!r} where the reference '
            f'gives {want["status_code"]!r}'
        )
        if PART in want['message']:
            said = want['message'][want['message'].index(PART) :]
            assert said in got['message'], (
                f'{name}: the refusal does not say {said!r}: {got["message"]}'
            )
        return
    for field in ('outputs', 'columns', 'rows', 'items', 'unscored', 'notices'):
        assert got.get(field) == want.get(field), (
            f'{name}: {field}: '
            + '; '.join(differences(want.get(field), got.get(field)))
        )
    assert _dtypes(got.get('dtypes')) == _dtypes(want.get('dtypes')), name


@pytest.mark.parametrize('name', VECTOR_NAMES)
def test_python_core_answers_each_vector_as_the_reference_does(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""T2.8b's exits, "scores, rankings, ``prediction_statistic``, PD-1,
    reassembly; ``response_field`` for rankings; ``instance_id_type``,
    ``prediction_type``": python-core, on the core route over DuckDB,
    answers the vector as python-ref does: its result and ``.items``, or
    its refusal, and the calls it made."""
    spec = next(v for v in vectors()['vectors'] if v['name'] == name)
    inputs = {k: v for k, v in spec.items() if k != 'expected'}
    monkeypatch.setenv('SDM_CORE_ROUTES', _routes())

    def check() -> None:
        held_to(recorder().observe(inputs), spec['expected'], name)

    held(f'vector:{name}', check)
