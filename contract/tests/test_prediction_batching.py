# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prediction batching preserves row limits, request-size boundaries and partial answers.

Current-client checks use retained expected values and scripted endpoint responses.
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
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'contract/runners/run_python.py'
RECORDER = ROOT / 'contract/support/batch_prediction_scenarios.py'
CASES = ROOT / 'contract/cases/s1'

DATABASES = ('duckdb', 'sqlite', 'postgres')

EIGHT = list(range(8))
#: The invalid ``rows_per_call`` values and their cases.
INVALID = {
    'multicall.rows-per-call.zero': 0,
    'multicall.rows-per-call.negative': -1,
    'multicall.rows-per-call.fraction': 2.5,
    'multicall.rows-per-call.whole-float': 3.0,
    'multicall.rows-per-call.bool': True,
    'multicall.rows-per-call.text': '3',
}
#: Values that only lower the cap, so 8 entities go in one call.
LOWERED = {
    'multicall.rows-per-call.above-the-cap': 10_001,
    'multicall.rows-per-call.past-int64': 2**64,
}
#: How many rows each call the endpoint received scores, by case, on the
#: reference: the batches each case is about (a refusal at materialize
#: sends none).
CALLS = {
    'multicall.rows-per-call-3': [3, 3, 2],
    'multicall.fail-call-2': [3, 3, 2],
    'multicall.fail-call-0': [3],
    'multicall.rows-per-call.above-the-cap': [8],
    'multicall.rows-per-call.past-int64': [8],
    'multicall.entities-per-call.10001': [10_000, 1],
    'multicall.calls.100': [1] * 100,
    'multicall.forecast.one-call': [3, 3],
    'multicall.forecast.fail-call-1': [3, 3],
}
#: The refusals at materialize, by case: the code, and the details the
#: reference gives (the ceilings' counts, the 30 MiB body's mark).
REFUSED = {
    **{case: ('INVALID_REQUEST', {}) for case in INVALID},
    'multicall.forecast.batched': (
        'UNSUPPORTED_REQUEST',
        {'entities': 8, 'rows_per_call': 3},
    ),
}
#: The failed calls, by case: the call failed, and what the error carries
#: of the calls before it (``calls_answered`` and ``rows_answered``, or a
#: forecast's ``parts_answered``), none where the first call failed.
FAILED = {
    'multicall.fail-call-2': (2, {'calls_answered': 2, 'rows_answered': 6}),
    'multicall.fail-call-0': (0, {}),
    'multicall.forecast.fail-call-1': (1, {'parts_answered': 1}),
}
#: The 30 MiB body cases: the remedy the client names, by case.
BODY = {
    'multicall.body-30mib.one-row': 'A call already scores one row',
    'multicall.body-30mib.several-rows': (
        'What comes down is one of three. For an ordinary question it is '
        'the call: materialize(rows_per_call=N) below the 3 this one holds'
    ),
}
#: The cases this client departs from the reference on, on purpose: it holds a
#: request to no size of its own (no entity, call or body ceiling), so each is
#: gathered where the reference refused it at materialize.
DEPARTED = (
    'multicall.entities.100001',
    'multicall.calls.101',
    *BODY,
)
CASE_IDS = (
    'multicall.rows-per-call-3',
    'multicall.fail-call-2',
    'multicall.fail-call-0',
    *INVALID,
    *LOWERED,
    'multicall.entities-per-call.10001',
    'multicall.entities.100000',
    'multicall.entities.100001',
    'multicall.calls.100',
    'multicall.calls.101',
    *BODY,
    'multicall.forecast.one-call',
    'multicall.forecast.batched',
    'multicall.forecast.fail-call-1',
)


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def runner() -> ModuleType:
    return _module('test_prediction_batching_run_python', RUNNER)


def recorder() -> ModuleType:
    return _module('batch_prediction_scenarios', RECORDER)


def key(case_id: str, database: str) -> str:
    return runner().case_key(case_id, database)


KEYS_ALL = tuple(key(c, d) for d in DATABASES for c in CASE_IDS)
#: The case keys observed with a call raising an error that is not the
#: client's own (``batch_prediction_scenarios.RAISED``): what each call answered
#: before it, kept on it, by case.
RAISED = {
    'multicall.rows-per-call-3': {'calls_answered': 2, 'rows_answered': 6},
    'multicall.forecast.one-call': {'parts_answered': 1},
}
RAISED_KEYS = tuple(key(c, d) for d in DATABASES for c in RAISED)


def test_every_multi_call_case_is_named_here() -> None:
    r"""Control: the cases here are every ``multicall.*`` case, once."""
    assert len(set(CASE_IDS)) == len(CASE_IDS)
    assert set(CASE_IDS) == set(recorder().case_ids())


# --- the cases -------------------------------------------------------------------


@functools.cache
def cases() -> dict[str, tuple[dict[str, Any], str]]:
    r"""The cases by id, each with its digest, as the runner loads
    them."""
    loaded = runner().load_cases([CASES / f'{c}.json' for c in CASE_IDS])
    return {case['id']: (case, digest) for case, digest in loaded}


def _case(case_id: str) -> dict[str, Any]:
    case, _ = cases()[case_id]
    assert case['status'] == 'acceptance' and case['slice'] == 'S1'
    assert runner().case_databases(case) == DATABASES
    return case


def _built(case: dict[str, Any]) -> Any:
    r"""The case's DuckDB database, its base and setup run, as the runner
    builds it."""
    import duckdb

    connection = duckdb.connect()
    for statement in [
        *runner().load_bases()[case['input']['base']],
        *case['input'].get('setup', []),
    ]:
        connection.execute(statement)
    return connection


def test_rows_per_call_3_over_8_entities_and_a_failure_on_call_2() -> None:
    r"""Exits "``rows_per_call=3`` over 8 entities" and "a failure on call
    2": the shop's 8 users asked in calls of 3 through predict, and the same
    with the call numbered 2 (the third, the scripted endpoint numbering
    from zero) or the first failed."""
    for case_id, fail_call in (
        ('multicall.rows-per-call-3', None),
        ('multicall.fail-call-2', 2),
        ('multicall.fail-call-0', 0),
    ):
        case = _case(case_id)
        given = case['input']
        assert given['base'] == 'shop' and 'setup' not in given
        assert given['materialize']['indices'] == EIGHT
        assert given['materialize']['rows_per_call'] == 3
        assert given.get('fail_call') == fail_call
        assert case['until'] == 'predict'
        assert 'FORECAST' not in given['pql']


@pytest.mark.parametrize('case_id', [*INVALID, *LOWERED])
def test_each_rows_per_call_value_is_the_one_named(case_id: str) -> None:
    r"""Exit "invalid ``rows_per_call`` values": each case passes its value
    as written, of its own type (a float with nothing after the point is
    no whole number to the reference), over the shop's 8 users; the two
    that only lower the cap are whole numbers past it."""
    case = _case(case_id)
    wanted = {**INVALID, **LOWERED}[case_id]
    given = case['input']['materialize']['rows_per_call']
    assert type(given) is type(wanted) and given == wanted
    assert case['input']['materialize']['indices'] == EIGHT
    assert 'fail_call' not in case['input']


def test_each_ceiling_s_boundary() -> None:
    r"""Exit "each ceiling's boundary": 10,001 entities (one past a call's
    10,000), 100,000 and 100,001 entities (the most one materialize
    gathers, and one past), 100 and 101 calls of one entity (the most
    calls, and one past); each on generated users holding more rows than
    it asks about, so context is left."""
    for case_id, count, rows_per_call, until in (
        ('multicall.entities-per-call.10001', 10_001, None, 'predict'),
        ('multicall.entities.100000', 100_000, None, 'materialize'),
        ('multicall.entities.100001', 100_001, None, 'materialize'),
        ('multicall.calls.100', 100, 1, 'predict'),
        ('multicall.calls.101', 101, 1, 'predict'),
    ):
        case = _case(case_id)
        given = case['input']['materialize']
        assert given['indices'] == list(range(count)), case_id
        assert given.get('rows_per_call') == rows_per_call, case_id
        assert case['until'] == until, case_id
        connection = _built(case)
        try:
            users = connection.execute('SELECT count(*) FROM users').fetchone()
            tables = connection.execute(
                "SELECT count(*) FROM duckdb_tables() WHERE schema_name = 'main'"
            ).fetchone()
        finally:
            connection.close()
        assert users[0] > count and tables[0] == 1, case_id


@pytest.mark.parametrize('case_id', list(BODY))
def test_a_call_past_the_30_mib_body(case_id: str) -> None:
    r"""Exit "the 30 MiB refusal with one row and with several": one
    entity, or three, over users whose context rows carry more than 30 MiB
    of one text column."""
    case = _case(case_id)
    assert case['input']['materialize']['indices'] == (
        [0] if case_id.endswith('one-row') else [0, 1, 2]
    )
    assert 'rows_per_call' not in case['input']['materialize']
    connection = _built(case)
    try:
        rows, size = connection.execute(
            'SELECT count(*), sum(strlen(cat)) FROM users'
        ).fetchone()
    finally:
        connection.close()
    assert rows > 1_000 and size > 30 * 1024 * 1024


def test_the_forecasts() -> None:
    r"""Exits "a forecast over one call" and "the batched-forecast
    refusal": a two-step forecast over as many entities as one call scores
    (``rows_per_call=3`` over 3), over more (3 over 8), and over one call a
    step with the second step's call failed."""
    for case_id, indices, fail_call in (
        ('multicall.forecast.one-call', [0, 1, 2], None),
        ('multicall.forecast.batched', EIGHT, None),
        ('multicall.forecast.fail-call-1', [0, 1, 2], 1),
    ):
        case = _case(case_id)
        given = case['input']
        assert ' FORECAST 2 TIMEFRAMES ' in given['pql'], case_id
        assert given['materialize']['indices'] == indices, case_id
        assert given['materialize']['rows_per_call'] == 3, case_id
        assert given.get('fail_call') == fail_call, case_id


def _expected(case: dict[str, Any]) -> tuple[Any, ...]:
    expect = case['expect']
    return (
        expect['outcome'],
        expect['stage'],
        expect.get('error'),
        expect.get('code'),
    )


def _outcome(line: dict[str, Any]) -> tuple[Any, ...]:
    outcome = line['outcome']
    return (
        outcome['result'],
        outcome['stage'],
        outcome.get('error'),
        outcome.get('code'),
    )


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
    r"""python-core over the cases on ``database``, as that
    database's job runs them: each observation by its key, or under ``''``
    why the run failed whole."""
    with tempfile.TemporaryDirectory(prefix='prediction_batching-') as tmp:
        given = Path(tmp) / 'cases'
        given.mkdir()
        for case_id in CASE_IDS:
            shutil.copy(CASES / f'{case_id}.json', given)
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


def same_outcome(want: tuple[Any, ...], got: tuple[Any, ...]) -> bool:
    r"""Two outcomes the same, but that the core route raises a refusal
    coded ``UNSUPPORTED_REQUEST`` as ``UnsupportedRequestError``, a
    subclass of the reference's ``StructuredDataError`` (B-20,
    approved by the owner, E78)."""
    if want == got:
        return True
    return (
        want[0] == got[0] == 'refusal'
        and want[1:2] == got[1:2]
        and want[3] == got[3] == 'UNSUPPORTED_REQUEST'
        and (want[2], got[2])
        == ('StructuredDataError', 'UnsupportedRequestError')
    )


@pytest.mark.parametrize('case_key', KEYS_ALL)
def test_current_client_batches_without_label_or_entity_leakage(
    case_key: str,
) -> None:
    """Assert current-client behavior and native routing without recorded payload equality."""
    database = case_key.partition('@')[2] or 'duckdb'
    if database == 'postgres' and not _server():
        pytest.skip('Postgres test DSN is not configured')
    if case_key.split('@')[0] in DEPARTED:
        pytest.skip('a departure from the reference, checked on its own')

    def check() -> None:
        run = core_run(database)
        assert '' not in run, run.get('')
        line = run.get(case_key)
        assert line is not None, f'{case_key}: not observed'
        case, _ = cases()[case_key.split('@')[0]]
        identity = line['identity']
        assert identity['native_loaded'] and line['core'] is not None
        missing = sorted(set(case['core']) - set(identity['operations']))
        assert not missing, f'{case_key}: core operations not run: {missing}'
        assert same_outcome(_expected(case), _outcome(line)), (
            case_key,
            _outcome(line),
        )
        sent = payloads(line)
        batches = [body['predict']['instance_table'] for body in sent]
        counts = CALLS.get(case['id'], [])
        assert [len(batch['rows']) for batch in batches] == counts, case_key
        scored = []
        context_positions = {}
        for body, batch in zip(sent, batches, strict=True):
            columns = batch['columns']
            assert body['task']['target']['column_name'] not in columns
            entity_at = columns.index('__kumo_entity_ref_0')
            entities = [int(row[entity_at]['number']) for row in batch['rows']]
            scored.extend(entities)
            if case['id'] in {
                'multicall.entities-per-call.10001',
                'multicall.calls.100',
            }:
                context = body['context']['instance_table']
                context_at = context['columns'].index('__kumo_entity_ref_0')
                context_entities = {
                    int(row[context_at]['number']) for row in context['rows']
                }
                assert context_entities.isdisjoint(entities), (
                    'Scored entities leaked into their own training context'
                )
                assert context['rows'], 'Large-case context must not disappear'
                instance_at = context['columns'].index('instance_id')
                target_at = context['columns'].index('TARGET')
                for row in context['rows']:
                    entity = int(row[context_at]['number'])
                    instance = int(row[instance_at]['number'])
                    assert float(row[target_at]['number']) == float(entity % 7)
                    assert (
                        context_positions.setdefault(entity, instance)
                        == instance
                    )
                related = body['context']['related_tables']['users']
                assert related['columns'] == ['instance_id', 'user_id']
                assert related['rows'] == [
                    [row[instance_at], row[context_at]]
                    for row in context['rows']
                ]

        indices = case['input']['materialize']['indices']
        expected_entities = indices * (
            2 if 'FORECAST 2' in case['input']['pql'] else 1
        )
        assert scored == expected_entities[: sum(counts)], case_key
        if (
            case['expect']['outcome'] == 'success'
            and case['until'] == 'predict'
        ):
            assert line['public']['result']['rows'] == len(expected_entities)
            assert [
                column['name'] for column in line['public']['result']['columns']
            ][:2] == ['row_index', 'id']

    check()


@functools.cache
def core_carried(case_key: str) -> dict[str, Any]:
    r"""python-core's observation of the case, by the recorder's code."""
    case_id, _, database = case_key.partition('@')
    case, _ = cases()[case_id]
    return recorder().observe(case, database or 'duckdb')


def assert_scored_rows(result, indices, steps=None):
    """Independent endpoint rule: regression scores 1.0 in requested order."""
    columns = (
        ['row_index', 'id']
        + (['forecast_step'] if steps else [])
        + ['prediction']
    )
    expected = [
        [i, entity] + ([step] if step is not None else []) + [1.0]
        for i, entity in enumerate(indices)
        for step in (steps or [None])
    ]
    assert result['columns'] == columns
    assert result['count'] == len(result['rows']) == len(expected)
    assert result['rows'] == expected
    assert all(type(row[-1]) is float for row in result['rows'])


@pytest.mark.parametrize('case_key', KEYS_ALL)
def test_batching_preserves_rows_limits_and_partial_answers(
    case_key: str,
) -> None:
    case_id, _, database = case_key.partition('@')
    if database == 'postgres' and not _server():
        pytest.skip('Postgres test DSN is not configured')
    if case_id in DEPARTED:
        pytest.skip('a departure from the reference, checked on its own')
    case, _ = cases()[case_id]
    got = core_carried(case_key)
    assert same_outcome(_expected(case), _outcome(got))
    assert got['calls'] == CALLS.get(case_id, [])
    indices = case['input']['materialize']['indices']
    if _expected(case)[0] == 'success' and case['until'] == 'predict':
        steps = [1, 2] if case_id == 'multicall.forecast.one-call' else None
        assert_scored_rows(got['result'], indices, steps)
    else:
        assert got['result'] is None
    details = got['details'] or {}
    if case_id in REFUSED:
        code, expected_details = REFUSED[case_id]
        assert got['outcome']['code'] == code
        for name, value in expected_details.items():
            assert details[name] == value
    if case_id in FAILED:
        call, carried = FAILED[case_id]
        for name, value in {
            'call': call,
            'error': 'scripted_failure',
            **carried,
        }.items():
            assert details[name] == value
        if carried:
            count = sum(CALLS[case_id][:call])
            steps = [1] if 'parts_answered' in carried else None
            assert_scored_rows(got['answered'], indices[:count], steps)
        else:
            assert got['answered'] is None
    else:
        assert got['answered'] is None
    if case_id in BODY:
        assert (
            got['opening']
            == '[INVALID_REQUEST] Context size exceeds the 30MB limit.'
        )
        assert got['remedy'].startswith(BODY[case_id])
        for control in ['num_context_samples', 'num_neighbors', 'columns']:
            assert control in got['remedy']
        if case_id.endswith('several-rows'):
            for advice in [
                'same entities',
                'more and smaller calls',
                'no call is small enough',
                'ranking or a forecast',
                'never batched',
                'entities',
            ]:
                assert advice in got['remedy']
    else:
        assert got['remedy'] is None
    if case_id in INVALID:
        assert 'rows_per_call' in got['opening']


@pytest.mark.parametrize('case_key', RAISED_KEYS)
def test_transport_errors_keep_completed_answers(case_key: str) -> None:
    case_id, _, database = case_key.partition('@')
    if database == 'postgres' and not _server():
        pytest.skip('Postgres test DSN is not configured')
    case, _ = cases()[case_id]
    call = recorder().RAISED[case_id]
    got = recorder().observe(case, database or 'duckdb', raise_on_call=call)
    assert _outcome(got) == ('refusal', 'predict', 'ValueError', None)
    assert got['calls'] == CALLS[case_id][:call]
    for name, value in RAISED[case_id].items():
        assert got['details'][name] == value
    count = sum(CALLS[case_id][:call])
    indices = case['input']['materialize']['indices'][:count]
    steps = [1] if 'parts_answered' in RAISED[case_id] else None
    assert_scored_rows(got['answered'], indices, steps)
    assert got['result'] is None


@pytest.mark.parametrize('case_id', DEPARTED)
def test_python_core_gathers_and_predicts_what_the_reference_refused(
    case_id: str,
) -> None:
    r"""The client sets no most on the entities, the calls or the body: each
    case is answered."""
    got = core_carried(case_id)
    assert got['outcome']['result'] == 'success', got['outcome']


#: A tiny seeded three-call fixture: five users scored ``user_id % 3``,
#: three of them asked one per call, sampled with a fixed seed.
TINY = {
    'base': 'shop',
    'setup': [
        'DROP TABLE orders',
        'DROP TABLE users',
        'CREATE TABLE users AS SELECT i::BIGINT AS user_id, '
        '(i % 3)::DOUBLE AS score FROM range(5) t(i)',
    ],
    'pql': 'PREDICT users.score FOR EACH users.user_id',
    'materialize': {'indices': [0, 1, 2], 'rows_per_call': 1, 'random_seed': 7},
}
#: The users in the order the seed samples them, each at its instance id.
TINY_SAMPLE = [3, 1, 2, 4, 0]


def test_three_seeded_calls_send_whole_requests() -> None:
    r"""Each of the three calls is the whole request: the shared task,
    schema, options and query, and its own context, which is the seeded
    sample less the user that call scores, each labelled with its score,
    and that user scored at the instance id after the sample's."""
    from sdm_client import KumoRelational, PredictiveQuery, Schema, SDMClient

    run = runner()
    connection, cleanup, _ = run._open(
        {'input': TINY}, run.load_bases(), 'duckdb'
    )
    try:
        with (
            recorder().endpoint().serve() as (url, received),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter('ignore')
            schema = Schema.from_source(run._source('duckdb', connection))
            request = PredictiveQuery(TINY['pql'], schema).materialize(
                **run._materialize_arguments(TINY['materialize'])
            )
            with SDMClient(url) as client:
                client.predict(request, model=KumoRelational())
    finally:
        cleanup()
    sent = []
    for _, path, body in received.requests:
        assert path == '/v1/predictions'
        parsed = json.loads(body)
        parsed['metadata'].pop('request_id')
        sent.append(parsed)

    def table(columns: list[str], rows: list[list[Any]]) -> dict[str, Any]:
        return {'columns': columns, 'format': 'arrays', 'rows': rows}

    def half(rows: list[list[Any]], columns: list[str]) -> dict[str, Any]:
        return {
            'instance_table': table(columns, rows),
            'related_tables': {
                'users': table(
                    ['instance_id', 'user_id'], [row[:2] for row in rows]
                )
            },
        }

    shared = {
        'model': 'kumo-relational',
        'task': {
            'kind': 'regression',
            'target': {'column_name': 'TARGET', 'dtype': 'float32'},
            'entity_table_names': ['users'],
        },
        'schema': {
            'instance_table': {
                'columns': {
                    'instance_id': {
                        'dtype': 'int64',
                        'nullable': False,
                        'stype': 'ID',
                    },
                    '__kumo_entity_ref_0': {'dtype': 'int64', 'stype': 'ID'},
                    'TARGET': {'dtype': 'float32', 'stype': 'numerical'},
                },
                'primary_key': 'instance_id',
            },
            'related_tables': {
                'users': {
                    'columns': {
                        'instance_id': {
                            'dtype': 'int64',
                            'nullable': False,
                            'stype': 'ID',
                        },
                        'user_id': {'dtype': 'int64', 'nullable': False},
                    },
                    'primary_key': ['instance_id', 'user_id'],
                }
            },
            'relationships': [
                {
                    'source_columns': ['__kumo_entity_ref_0'],
                    'target_table': 'users',
                    'target_columns': ['user_id'],
                }
            ],
        },
        'output': {'fields': ['prediction'], 'prediction_statistic': 'median'},
        'inference': {'run_mode': 'fast', 'use_prediction_time': False},
        'metadata': {
            'num_context_examples': len(TINY_SAMPLE),
            'num_prediction_examples': 1,
            'source_query': TINY['pql'],
        },
    }
    expected = []
    for scored in TINY['materialize']['indices']:
        context = [
            [instance, user, float(user % 3)]
            for instance, user in enumerate(TINY_SAMPLE)
            if user != scored
        ]
        expected.append(
            {
                **shared,
                'context': half(
                    context, ['instance_id', '__kumo_entity_ref_0', 'TARGET']
                ),
                'predict': half(
                    [[len(TINY_SAMPLE), scored]],
                    ['instance_id', '__kumo_entity_ref_0'],
                ),
            }
        )
    assert sent == expected
