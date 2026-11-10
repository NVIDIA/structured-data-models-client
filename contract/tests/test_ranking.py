# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ranking preserves candidate scoring, ordering, cutoffs and invalid-request diagnostics.

Current-client checks use independently expected values and scripted endpoint responses.
No archived client runtime is executed.
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
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'contract/runners/run_python.py'
RECORDER = ROOT / 'contract/support/ranking_scenarios.py'
CASE_DIR = ROOT / 'contract/cases/s1'

DATABASES = ('duckdb', 'sqlite', 'postgres')

#: Each case and the design row item it proves.
CASES = {
    'rank.top-k': 'candidates, the decomposition, problem_type for '
    "ranking, the query's top_k; each part's payload",
    'rank.top-k-option': 'the top_k option, sent with no part, and '
    'candidates in the order given',
    'rank.top-k-zero': 'top_k outside 1 to 20, refused',
    'rank.no-candidates': 'a ranking names its candidates',
    'rank.candidates-empty': 'no candidate to rank, refused',
    'rank.candidates-not-a-ranking': 'candidates beside a query that '
    'ranks nothing, refused',
    'rank.most-parts': 'more candidates than MOST_PARTS, refused',
    'rank.batched': 'the batched-ranking refusal (more entities than '
    'one call scores)',
    'rank.unscored-candidate': 'a candidate whose context '
    'holds one class left out, unscored, before any call',
    'rank.all-unscored': 'no candidate left to rank, refused before any call',
    'rank.entities-by-candidates.over': 'the entities gathered once '
    'per candidate past the 100,000 one materialize holds (501 by 200), '
    'refused before anything is read',
    'rank.entities-by-candidates.at': 'the same at exactly 100,000 '
    '(500 by 200), not refused by that ceiling',
    'forecast.top-k': "the forecast's top_k refusal in the core",
    'forecast.most-parts': "the forecast's MOST_PARTS refusal in the core",
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
    return _module('test_ranking_run_python', RUNNER)


def recorder() -> ModuleType:
    return _module('ranking_scenarios', RECORDER)


def key(case_id: str, database: str) -> str:
    return runner().case_key(case_id, database)


CASE_KEYS = tuple(key(c, d) for d in DATABASES for c in CASE_IDS)


# --- the cases -------------------------------------------------------------------


@functools.cache
def cases() -> dict[str, tuple[dict[str, Any], str]]:
    r"""The cases by id, each with its digest, as the runner loads
    them."""
    loaded = runner().load_cases([CASE_DIR / f'{c}.json' for c in CASE_IDS])
    return {case['id']: (case, digest) for case, digest in loaded}


def test_the_cases_are_the_design_rows() -> None:
    r"""Control, "S1 ranking cases at materialize and at predict": each case
    is the row item:data:`CASES` names, an acceptance case of S1 on all
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
    # The entity ceiling across a ranking's parts: one past it, and at it.
    for case_id, entities in (
        ('rank.entities-by-candidates.over', 501),
        ('rank.entities-by-candidates.at', 500),
    ):
        assert spec_of(case_id)['indices'] == list(range(entities))
        assert spec_of(case_id)['candidates'] == list(range(200))
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


# --- the vectors -----------------------------------------------------------------


#: What the reference's refusal of each case refused before any call says,
#: so it is refused for the argument the case names.
REFUSED_FOR = {
    'rank.top-k-zero': 'top_k is 0',
    'rank.no-candidates': 'they have to be named',
    'rank.candidates-empty': 'candidates is empty',
    'rank.candidates-not-a-ranking': 'does not rank anything',
    'rank.most-parts': '201 candidates means 201 calls',
    'rank.batched': 'A part is not batched',
    'rank.entities-by-candidates.over': (
        '501 entities gathered 200 times over, 100,200 in all, is past the '
        '100,000'
    ),
    # At the ceiling the call is gathered, and refused only for the
    # entities 8 to 499, which name no user.
    'rank.entities-by-candidates.at': 'The primary keys [8, 9,',
    'forecast.top-k': 'a forecast ranks nothing',
    'forecast.most-parts': '201 forecast steps means 201 calls',
}


def test_each_case_is_refused_for_the_argument_it_names() -> None:
    """Exercise each invalid argument and its corrected control on the client."""
    recorded = {
        c['case']: c
        for c in recorder().controls(
            {key: case for key, (case, _) in cases().items()}
        )
    }
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
        if case_id == 'rank.entities-by-candidates.at':
            # Not vacuous: at 100,000 the ceiling does not refuse.
            assert 'in all, is past the' not in refused['message']
        assert each['control']['outcome'] == 'result', (case_id, each)


# --- python-core against the reference ---------------------------------------------


def _server() -> bool:
    r"""Whether a Postgres server is named; where none is, one must not be
    required."""
    if os.environ.get('SDM_POSTGRES_TEST_DSN'):
        return True
    assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
        'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
    )
    return False


@functools.cache
def core_run(database: str) -> dict[str, Any]:
    r"""python-core over the cases on ``database``, as that database's
    job runs them: each observation by its key, or under ``''`` why the run
    failed whole."""
    with tempfile.TemporaryDirectory(prefix='ranking-') as tmp:
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


@pytest.mark.parametrize('case_key', CASE_KEYS)
def test_current_client_decomposes_rankings_and_refuses_invalid_arguments(
    case_key: str,
) -> None:
    """Assert current-client behavior and native routing without recorded payload equality."""
    database = case_key.partition('@')[2] or 'duckdb'
    if database == 'postgres' and not _server():
        pytest.skip('Postgres test DSN is not configured')

    def check() -> None:
        run = core_run(database)
        assert '' not in run, run.get('')
        line = run.get(case_key)
        assert line is not None, f'{case_key}: not observed'
        case, _ = cases()[case_key.split('@')[0]]
        expect = case['expect']
        assert line['identity']['native_loaded'] and line['core'] is not None
        assert _outcome(line) == (
            expect['outcome'],
            expect['stage'],
            expect.get('error'),
            expect.get('code'),
        ), case_key
        sent = payloads(line)
        assert len(sent) == CALLS.get(case['id'], 0), case_key
        for body in sent:
            assert body['task']['kind'] == 'binary_classification'
            assert body['task']['target']['positive_class'] == 'true'
            assert body['task']['target']['classes'] == ['false', 'true']
            assert (
                body['task']['target']['column_name']
                not in body['predict']['instance_table']['columns']
            )
        if expect['outcome'] == 'success':
            result = line['public']['result']
            assert result['rows'] == len(
                case['input']['materialize']['indices']
            )
            assert [column['name'] for column in result['columns']] == [
                'row_index',
                'id',
                'prediction',
                'rankings',
            ]
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
        stage = expect['stage']
        assert stage in ran, f'{case_key}: the core ran no {stage}: {ran}'
        if expect['outcome'] == 'refusal':
            error = line['core'].get('error') or {}
            assert error.get('code') == expect['code'], (
                f"{case_key}: the refusal is not the core's: the core "
                f'answered {error!r}'
            )

    check()


@pytest.mark.parametrize(
    ('name', 'limit'),
    [('scores.rank-top-2', 2), ('top-k.option-1', 1), ('top-k.option-3', 3)],
)
def test_ranked_candidates_follow_scripted_scores_and_stable_ties(
    name: str, limit: int
) -> None:
    """Rank the supplied endpoint scores, preserving candidate order on ties."""
    spec = copy.deepcopy(
        next(v for v in recorder().VECTORS if v['name'] == name)
    )
    result = recorder().observe(spec)
    assert result['outcome'] == 'result'
    candidates = spec['materialize']['candidates']
    scores = spec['script']['table']
    assert len(result['calls']) == len(candidates)
    assert len(result['items']) == len(spec['materialize']['indices'])
    for row_index, item in enumerate(result['items']):
        # This oracle uses only the authored endpoint input, never captured output.
        order = sorted(
            range(len(candidates)), key=lambda i: -scores[i][row_index]
        )
        expected = [{'int': str(candidates[i])} for i in order[:limit]]
        assert item['prediction'] == {'list': expected}
        assert item['row_index'] == {'int': str(row_index)}
        assert item['id'] == {
            'int': str(spec['materialize']['indices'][row_index])
        }
        rankings = [
            {'candidate': candidates[i], 'score': scores[i][row_index]}
            for i in order[:limit]
        ]
        assert item['rankings'] == recorder()._cell(rankings)


def _observe_authored_ranking(name: str) -> dict[str, Any]:
    spec = copy.deepcopy(
        next(v for v in recorder().VECTORS if v['name'] == name)
    )
    return recorder().observe(spec)


def _assert_remaining_candidate_scores(
    result: dict[str, Any], candidates: list[int]
) -> None:
    assert len(result['items']) == 3
    for row_index, item in enumerate(result['items']):
        order = sorted(
            range(len(candidates)),
            key=lambda i: -recorder().SCORES[i][row_index],
        )
        assert item['prediction'] == recorder()._cell(
            [candidates[i] for i in order]
        )
        assert item['rankings'] == recorder()._cell(
            [
                {
                    'candidate': candidates[i],
                    'score': recorder().SCORES[i][row_index],
                }
                for i in order
            ]
        )


def test_unscored_and_duplicate_candidates_do_not_make_extra_calls() -> None:
    result = _observe_authored_ranking('candidates.unscored-one')
    assert result['outcome'] == 'result'
    assert result['unscored'] == {'list': [{'int': '9'}]}
    assert len(result['calls']) == 2
    assert result['notices'] == ['SDMWarning']
    _assert_remaining_candidate_scores(result, [0, 1])
    for item in result['items']:
        assert {'int': '9'} not in item['prediction']['list']
    result = _observe_authored_ranking('candidates.all-unscored')
    assert result['outcome'] == 'refused'
    assert result['calls'] == []
    result = _observe_authored_ranking('candidates.repeated')
    assert result['outcome'] == 'result'
    assert len(result['calls']) == 2
    _assert_remaining_candidate_scores(result, [1, 0])
    for item in result['items']:
        values = item['prediction']['list']
        assert len(values) == len(
            {json.dumps(v, sort_keys=True) for v in values}
        )


@pytest.mark.parametrize(
    ('name', 'candidate', 'calls'),
    [
        ('pd1.nan-first-call', 0, 1),
        ('pd1.nan-later-call', 1, 2),
        ('pd1.nan-as-text', 0, 1),
    ],
)
def test_invalid_ranking_scores_stop_at_the_offending_candidate(
    name: str, candidate: int, calls: int
) -> None:
    result = _observe_authored_ranking(name)
    assert result['outcome'] == 'refused'
    assert result['code'] == 'INVALID_RESPONSE'
    assert f'{PART}{candidate}, one of 3' in result['message']
    assert len(result['calls']) == calls


@pytest.mark.parametrize('database', DATABASES)
@pytest.mark.parametrize('after_unscored', [False, True])
def test_ranking_transport_failure_preserves_completed_part_count(
    database: str, after_unscored: bool
) -> None:
    if database == 'postgres' and not _server():
        pytest.skip('Postgres test DSN is not configured')
    name = 'parts.raised-on-call-1'
    if after_unscored:
        name += '.after-unscored'
    if database != 'duckdb':
        name += f'@{database}'
    result = _observe_authored_ranking(name)
    assert result['class'] == 'ValueError'
    assert result['details'] == {'parts_answered': 1}
    assert len(result['calls']) == 1


def test_ranking_endpoint_failure_identifies_its_candidate() -> None:
    result = _observe_authored_ranking('parts.failed-call-1')
    assert result['class'] == 'NimRequestError'
    assert result['status_code'] == 500
    assert len(result['calls']) == 2
    assert f'{PART}1, one of 3' in result['message']


@pytest.mark.parametrize(
    'name',
    [
        'response-field.binary-scores',
        'response-field.binary-rankings',
        'response-field.binary-rankings-nan-score',
        'response-field.regression-scores',
        'prediction-type.binary-float',
        'prediction-type.binary-text',
        'prediction-type.binary-integer',
        'prediction-type.multiclass-integer',
        'prediction-type.multiclass-bool',
    ],
)
def test_response_fields_preserve_authored_endpoint_values(name: str) -> None:
    spec = next(v for v in recorder().VECTORS if v['name'] == name)
    result = _observe_authored_ranking(name)
    assert result['outcome'] == 'result'
    script = spec['script']
    if name == 'prediction-type.binary-float':
        assert (
            result['dtypes'][result['columns'].index('prediction')] == 'float64'
        )
    expected = (
        json.loads('{' + script['carried'] + '}')
        if 'carried' in script
        else {'prediction': json.loads(script['prediction'])}
    )
    # Binary class 1 is represented as the boolean target class.
    if name == 'prediction-type.binary-integer':
        expected['prediction'] = True
    assert len(result['items']) == len(spec['materialize']['indices'])
    for item in result['items']:
        for field, value in expected.items():
            assert field in result['columns']
            assert item[field] == recorder()._cell(value)


@pytest.mark.parametrize('database', DATABASES)
@pytest.mark.parametrize(
    'kind',
    [
        'date-and-text',
        'timestamp-and-text',
        'decimal-past-i128',
        'int-past-i128',
        'int-past-76-digits',
    ],
)
def test_candidate_identity_survives_connector_roundtrip(
    database: str, kind: str
) -> None:
    if database == 'postgres' and not _server():
        pytest.skip('Postgres test DSN is not configured')
    name = f'candidates.{kind}' + (
        f'@{database}' if database != 'duckdb' else ''
    )
    spec = next(v for v in recorder().VECTORS if v['name'] == name)
    result = _observe_authored_ranking(name)
    assert result['outcome'] == 'result'
    assert result['unscored'] == {'list': []}
    candidates = [
        recorder()._candidate(v) for v in spec['materialize']['candidates']
    ]
    assert len(result['calls']) == len(candidates)
    assert len(result['items']) == len(spec['materialize']['indices'])
    scores = spec['script']['table']
    for row_index, item in enumerate(result['items']):
        order = sorted(
            range(len(candidates)), key=lambda i: -scores[i][row_index]
        )
        expected = [recorder()._cell(candidates[i]) for i in order[:2]]
        assert item['prediction']['list'] == expected


@pytest.mark.parametrize(
    ('name', 'statistic'),
    [
        ('prediction-statistic.regression-mean', 'mean'),
        ('prediction-statistic.edited-not-text', 7),
    ],
)
def test_prediction_statistic_reaches_endpoint(
    name: str, statistic: Any
) -> None:
    result = _observe_authored_ranking(name)
    assert result['outcome'] == 'result'
    assert len(result['calls']) == 1
    assert result['calls'][0]['output']['prediction_statistic'] == statistic


@pytest.mark.parametrize(
    ('name', 'stage', 'code', 'calls'),
    [
        ('top-k.outputs-probabilities', 'predict', 'INVALID_REQUEST', 0),
        ('top-k.outputs-none', 'predict', 'INVALID_REQUEST', 0),
        ('candidates.all-unscored', 'predict', 'UNSUPPORTED_REQUEST', 0),
        ('candidates.text-key', 'materialize', 'INVALID_REQUEST', 0),
        ('parts.no-probabilities', 'predict', 'INTERNAL_ERROR', 3),
        ('prediction-statistic.ranking', 'materialize', 'INVALID_REQUEST', 0),
        ('response-field.binary-scores-nan', 'predict', 'INVALID_RESPONSE', 1),
        ('instance-id.bool', 'predict', 'INVALID_RESPONSE', 1),
    ],
)
def test_ranking_refusals_preserve_stage_code_and_call_boundary(
    name: str, stage: str, code: str, calls: int
) -> None:
    result = _observe_authored_ranking(name)
    assert result['outcome'] == 'refused'
    assert result['stage'] == stage
    assert result['code'] == code
    assert len(result['calls']) == calls


@pytest.mark.parametrize(
    ('name', 'fields'),
    [
        ('top-k.outputs-rankings', ['rankings']),
        ('top-k.outputs-prediction', ['prediction']),
        ('response-field.ranking-part-scores', ['prediction', 'rankings']),
        ('response-field.ranking-part-rankings', ['prediction', 'rankings']),
    ],
)
def test_ranking_output_selection_and_part_fields(
    name: str, fields: list[str]
) -> None:
    spec = next(v for v in recorder().VECTORS if v['name'] == name)
    result = _observe_authored_ranking(name)
    assert result['outcome'] == 'result'
    assert result['columns'] == ['row_index', 'id', *fields]
    assert len(result['calls']) == 3
    assert len(result['items']) == len(spec['materialize']['indices'])
    scores = spec['script']['table']
    candidates = spec['materialize']['candidates']
    for row_index, item in enumerate(result['items']):
        order = sorted(range(3), key=lambda i: -scores[i][row_index])[:2]
        if 'prediction' in fields:
            assert item['prediction'] == recorder()._cell(
                [candidates[i] for i in order]
            )
        if 'rankings' in fields:
            assert item['rankings'] == recorder()._cell(
                [
                    {'candidate': candidates[i], 'score': scores[i][row_index]}
                    for i in order
                ]
            )


def test_float_instance_identity_is_accepted() -> None:
    result = _observe_authored_ranking('instance-id.float')
    assert result['outcome'] == 'result'
    assert len(result['calls']) == 1
    assert len(result['items']) == 2
