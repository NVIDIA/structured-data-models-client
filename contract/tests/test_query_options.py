# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query options preserve endpoint payloads, filtering, result fields and diagnostics.

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
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'contract/runners/run_python.py'
RECORDER = ROOT / 'contract/support/query_option_scenarios.py'
ENDPOINT = ROOT / 'tests/support/scripted_endpoint.py'
CASES = ROOT / 'contract/cases/s1'

DATABASES = ('duckdb', 'sqlite', 'postgres')

#: The entity filter cases, by id: the filter's condition, as the
#: reference's notice quotes it (none where the validator refuses the
#: query first).
FILTERED = {
    'filter.entity-column': 'users.age > 22',
    'filter.entity-text': "users.country = 'DE'",
    'filter.entity-aggregation': 'COUNT(orders.*, -60, 0, days) > 0',
    'filter.matches-none': 'users.age > 100',
    'filter.event-column': None,
    'filter.ranking': 'users.age > 22',
}
#: The named-entity cases, by id: the entities the query names.
NAMED_ENTITIES = {
    'ids.one': '= 1',
    'ids.in-list-other-indices': 'IN (1, 2)',
    'ids.absent': '= 99',
    'ids.wrong-type': "= 'a'",
}
ASSUMING = (
    'assuming.regression',
    'assuming.binary',
    'assuming.no-examples',
    'assuming.static',
)
EVALUATE = ('evaluate.regression', 'evaluate.binary')
#: The explanation cases, by id: whether the query says EXPLAIN, the
#: outputs predict asks for, and whether the endpoint explains.
EXPLAIN = {
    'explain.query': (True, None, False),
    'explain.answered': (True, ['prediction', 'explanation'], True),
    'explain.explanation-only': (False, ['explanation'], True),
    'explain.not-answered': (False, ['prediction', 'explanation'], False),
    'explain.two-entities': (True, ['prediction', 'explanation'], True),
    'explain.forecast': (False, ['prediction', 'explanation'], True),
    'metadata.items-explained': (False, ['prediction', 'explanation'], True),
}
METADATA = (
    'metadata.items',
    'metadata.items-across-calls',
    'metadata.items-explained',
)
#: The null target transform cases, by id: the target_transforms given.
TRANSFORMS = {
    'transforms.null': [None],
    'transforms.null-beside-a-name': [None, 'quantile'],
    'transforms.whole-null': None,
    'transforms.null-on-a-classification': [None],
}
#: The inference options given as text, by case: the options, and the
#: inference_config the reference sends for them (none where it refuses).
CONFIG_TEXT = {
    'config-text.accepted': (
        {
            'num_estimators': ' 3 ',
            'column_shuffle': 'yes',
            'hop_shuffle': 'off',
            'category_shuffle': '1',
        },
        {
            'num_estimators': 3,
            'column_shuffle': True,
            'hop_shuffle': False,
            'category_shuffle': True,
        },
    ),
    'config-text.estimators-whole-float': (
        {'num_estimators': '4.0'},
        {'num_estimators': 4},
    ),
    'config-text.class-shuffle': (
        {'class_shuffle': 'true'},
        {'class_shuffle': True},
    ),
    'config-text.estimators-word': ({'num_estimators': 'five'}, None),
    'config-text.estimators-out-of-range': ({'num_estimators': '5'}, None),
    'config-text.shuffle-word': ({'column_shuffle': 'maybe'}, None),
}
#: U1's case: a day-first date literal, then one the query is refused for.
DAY_FIRST = 'date.day-first-then-packed'
CASE_IDS = (
    *FILTERED,
    *NAMED_ENTITIES,
    *ASSUMING,
    *EVALUATE,
    *(c for c in EXPLAIN if c.startswith('explain.')),
    *METADATA,
    *TRANSFORMS,
    *CONFIG_TEXT,
    DAY_FIRST,
)
#: How many rows each call the endpoint received scores, by case, on the
#: reference (none where it refuses before a call).
CALLS = {
    **{
        case_id: [2]
        for case_id in (
            'filter.entity-column',
            'filter.entity-text',
            'filter.entity-aggregation',
            'ids.in-list-other-indices',
            'ids.absent',
            'assuming.regression',
            'assuming.binary',
            *EVALUATE,
            'explain.query',
            'metadata.items',
            'config-text.accepted',
            'config-text.estimators-whole-float',
            'config-text.class-shuffle',
        )
    },
    'filter.ranking': [2, 2, 2],
    'ids.one': [1],
    'explain.answered': [1],
    'explain.explanation-only': [1],
    'explain.not-answered': [1],
    'explain.forecast': [1, 1],
    'metadata.items-across-calls': [2, 1],
    'metadata.items-explained': [1],
}
#: The warnings the exits name, by case, each as its class and how its
#: text opens; every other case gives none of them.
WARNED = {
    **{
        case_id: [('SDMWarning', f'the entity filter {condition!r} selects')]
        for case_id, condition in FILTERED.items()
        if condition is not None
    },
    'explain.not-answered': [
        ('SDMWarning', "kumo-relational answered without ['explanation']")
    ],
    DAY_FIRST: [('UserWarning', 'Parsing dates in %d/%m/%Y format')],
}
#: What a refusal's message names, by case, as the reference's names it:
#: the filter, the column or the option it is about. The brief: a refusal
#: these options reach names what the reference's names, never only "the
#: request is not one the core accepts".
NAMED = {
    'filter.matches-none': ['users.age > 100'],
    'filter.event-column': ['orders.price'],
    'ids.wrong-type': ['users.user_id'],
    'assuming.no-examples': ['context examples'],
    'assuming.static': ['users.age > 22'],
    'explain.two-entities': ['explanation'],
    'transforms.null': ['target_transforms'],
    'transforms.null-beside-a-name': ['target_transforms'],
    'transforms.whole-null': ['target_transforms'],
    'transforms.null-on-a-classification': ['target_transforms'],
    'config-text.estimators-word': ['num_estimators'],
    'config-text.estimators-out-of-range': ['num_estimators'],
    'config-text.shuffle-word': ['column_shuffle'],
}


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def runner() -> ModuleType:
    return _module('test_query_options_run_python', RUNNER)


def recorder() -> ModuleType:
    return _module('query_option_scenarios', RECORDER)


def endpoint() -> ModuleType:
    return _module('scripted_endpoint', ENDPOINT)


def key(case_id: str, database: str) -> str:
    return runner().case_key(case_id, database)


KEYS_ALL = tuple(key(c, d) for d in DATABASES for c in CASE_IDS)
NAMED_KEYS = tuple(
    key(c, d) for d in DATABASES for c in NAMED if c != 'explain.two-entities'
)


def test_every_option_case_is_named_here() -> None:
    r"""Control: the cases here are every case, once, and each
    refusal named in ``NAMED`` is a case."""
    assert len(set(CASE_IDS)) == len(CASE_IDS)
    assert set(CASE_IDS) == set(recorder().case_ids())
    assert set(NAMED) <= set(CASE_IDS)
    assert set(CALLS) <= set(CASE_IDS) and set(WARNED) <= set(CASE_IDS)


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
    assert case['input']['base'] == 'shop'
    assert case['input']['materialize']['anchor_time'] == '2023-05-01'
    return case


@pytest.mark.parametrize('case_id', list(FILTERED))
def test_each_entity_filter_case_filters_the_entity(case_id: str) -> None:
    r"""Exit "``entity_filter``": the query's entity carries a WHERE
    filter (a column of the entity table, a text, an aggregation, one no
    entity passes, a fact table's column, and a ranking's)."""
    case = _case(case_id)
    pql = case['input']['pql']
    entity = pql[pql.index(' FOR EACH users.user_id WHERE ') :]
    condition = FILTERED[case_id]
    assert condition is None or entity.endswith(f'WHERE {condition}')
    if case_id == 'filter.ranking':
        assert ' RANK TOP 2 ' in pql
        assert case['input']['materialize']['candidates'] == [0, 1, 2]
        assert case['input']['probabilities'] is True


@pytest.mark.parametrize('case_id', list(NAMED_ENTITIES))
def test_each_named_entity_case_names_entities(case_id: str) -> None:
    r"""Exit "``entity_ids``": the query names its entities (``FOR
    users.user_id =...`` or ``IN (...)``) in place of ``FOR EACH``."""
    pql = _case(case_id)['input']['pql']
    assert 'FOR EACH' not in pql
    assert pql.endswith(f' FOR users.user_id {NAMED_ENTITIES[case_id]}')


def test_the_assuming_evaluate_and_explain_cases() -> None:
    r"""Exits "``assuming``", "``evaluate``" and "``explain``": each case
    says ASSUMING, EVALUATE or EXPLAIN as it names; an explanation case
    asks predict for an explanation, the endpoint explaining where it
    says, over one or two entities (the archive refused two)."""
    for case_id in ASSUMING:
        assert ' ASSUMING ' in _case(case_id)['input']['pql'], case_id
    for case_id in EVALUATE:
        assert _case(case_id)['input']['pql'].startswith('EVALUATE PREDICT ')
    for case_id, (says, outputs, explains) in EXPLAIN.items():
        given = _case(case_id)['input']
        assert given['pql'].startswith('EXPLAIN ') == says, case_id
        assert given.get('outputs') == outputs, case_id
        assert given.get('explanation', False) == explains, case_id
        if outputs:
            count = 2 if case_id == 'explain.two-entities' else 1
            assert len(given['materialize']['indices']) == count, case_id


def test_the_item_metadata_null_transform_and_text_option_cases() -> None:
    r"""Exits "``response_field`` for item metadata",
    "``target_transform_null``", "``inference_config_text``" and U1: the
    endpoint gives item metadata where a case says (one over two calls);
    each transform case's target_transforms holds a null, or is null; each
    text case's options are text; U1's query reads a day-first literal
    before the one it is refused for."""
    for case_id in METADATA:
        assert _case(case_id)['input']['item_metadata'] is True, case_id
    assert (
        _case('metadata.items-across-calls')['input']['materialize'][
            'rows_per_call'
        ]
        == 2
    )
    for case_id, transforms in TRANSFORMS.items():
        config = _case(case_id)['input']['materialize']['inference_config']
        assert config == {'target_transforms': transforms}, case_id
    for case_id, (options, _) in CONFIG_TEXT.items():
        config = _case(case_id)['input']['materialize']['inference_config']
        assert config == options, case_id
        assert all(isinstance(v, str) for v in config.values()), case_id
    pql = _case(DAY_FIRST)['input']['pql']
    assert pql.index('13/01/2023') < pql.index('99999-01-01')
    assert _case(DAY_FIRST)['until'] == 'materialize'


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


def _plain(value: Any) -> Any:
    r"""A parsed payload's value with each number as the number it writes
    (the runner keeps a number as ``{"number": <its text>}``)."""
    if isinstance(value, dict):
        if set(value) == {'number'} and isinstance(value['number'], str):
            text = value['number']
            return float(text) if any(c in text for c in '.eE') else int(text)
        return {name: _plain(item) for name, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _explained(call: int, index: int) -> dict[str, Any]:
    r"""The scripted endpoint's explanation of row ``index`` of call
    ``call``, as JSON writes it."""
    return json.loads(json.dumps(endpoint()._explained(call, index)))


def _item_metadata(call: int, index: int) -> dict[str, Any]:
    return {'call': call, 'row_index': index, 'source': 'scripted'}


#: The explanation the reference's result holds, by case: the endpoint's
#: for the one row explained, a forecast's under each step; none for
#: every other case.
EXPLANATIONS = {
    'explain.answered': lambda: _explained(0, 0),
    'explain.explanation-only': lambda: _explained(0, 0),
    'metadata.items-explained': lambda: _explained(0, 0),
    'explain.forecast': lambda: {
        'by_forecast_step': {'1': _explained(0, 0), '2': _explained(1, 0)}
    },
}
#: Each item's metadata on the reference's result, by case, with its
#: row_index (numbered across calls); every other case's items carry none.
ITEM_METADATA = {
    'metadata.items': [[0, _item_metadata(0, 0)], [1, _item_metadata(0, 1)]],
    'metadata.items-across-calls': [
        [0, _item_metadata(0, 0)],
        [1, _item_metadata(0, 1)],
        [2, _item_metadata(1, 0)],
    ],
    'metadata.items-explained': [[0, _item_metadata(0, 0)]],
}


@pytest.mark.parametrize('case_key', KEYS_ALL)
def test_query_options_preserve_authored_behavior(case_key: str) -> None:
    """Execute the cases against authored outcomes and endpoint facts."""
    case_id = case_key.split('@')[0]
    database = case_key.partition('@')[2] or 'duckdb'
    if database == 'postgres' and not _server():
        pytest.skip('Postgres test DSN is not configured')
    case, _ = cases()[case_id]
    vector = core_carried(case_key)
    expected = _expected(case)
    if case_id == 'explain.two-entities':
        expected = ('success', 'predict', None, None)
    assert same_outcome(expected, _outcome(vector))
    assert vector['calls'] == (
        [2] if case_id == 'explain.two-entities' else CALLS.get(case_id, [])
    )
    expected_warnings = WARNED.get(case_id, [])
    if case_id == 'explain.query':
        expected_warnings = [
            ('SDMWarning', "kumo-relational answered without ['explanation']")
        ]
    warned = [
        (kind, text[: len(opening)])
        for kind, text in vector['warnings']
        for wanted, opening in expected_warnings
        if kind == wanted
    ]
    assert warned == expected_warnings, vector['warnings']
    assert len(vector['warnings']) == len(expected_warnings)
    if case_id in FILTERED and FILTERED[case_id] is not None:
        text = vector['warnings'][0][1]
        count = len(case['input']['materialize']['indices'])
        assert f'does not narrow the {count} rows named for scoring' in text
        assert 'Pass only indices that satisfy the condition' in text
    if case_id in {'explain.query', 'explain.not-answered'}:
        text = vector['warnings'][0][1]
        assert 'no column of to_pandas() will carry it' in text
        assert 'rest of the request is unaffected' in text
    explanation = EXPLANATIONS.get(case_id)
    expected_explanation = explanation() if explanation else None
    if case_id == 'explain.two-entities':
        expected_explanation = _explained(0, 0)
    assert vector['explanation'] == expected_explanation
    if vector['item_metadata'] is not None:
        assert vector['item_metadata'] == ITEM_METADATA.get(
            case_id,
            [[pair[0], None] for pair in vector['item_metadata']],
        )
    else:
        assert case_id not in ITEM_METADATA
    if case_id in NAMED_ENTITIES and vector['result']:
        at = vector['result']['columns'].index('id')
        assert [row[at] for row in vector['result']['rows']] == (
            case['input']['materialize']['indices']
        )
    if case_id in NAMED and case_id != 'explain.two-entities':
        for word in NAMED[case_id]:
            assert word in vector['message'], (word, vector['message'])


def assert_request_options(case: dict[str, Any], sent: list[Any]) -> None:
    case_id = case['id']
    for payload in sent:
        query = payload['metadata']['source_query']
        if case_id == 'filter.ranking':
            # One binary question per candidate, each keeping the filter.
            assert query.endswith(
                ' FOR EACH users.user_id WHERE users.age > 22'
            )
            assert query.startswith(
                'PREDICT COUNT(buys.* WHERE buys.item_id = '
            )
        elif case_id != 'explain.forecast':
            # A forecast is asked a step at a time, each its own query.
            assert query == case['input']['pql']
    if case_id in CONFIG_TEXT and CONFIG_TEXT[case_id][1] is not None:
        for payload in sent:
            assert (
                payload['inference']['inference_config']
                == (CONFIG_TEXT[case_id][1])
            )


@pytest.mark.parametrize('database', DATABASES)
def test_each_explained_entity_retains_its_own_raw_response(
    database, monkeypatch
):
    if database == 'postgres' and not _server():
        pytest.skip('Postgres test DSN is not configured')
    record = recorder()
    project = record.projection
    observed = []

    def inspect(result):
        observed.extend(
            (item.row_index, item.explanation) for item in result.items
        )
        return project(result)

    monkeypatch.setattr(record, 'projection', inspect)
    got = record.observe(cases()['explain.two-entities'][0], database)
    assert got['outcome']['result'] == 'success'
    assert observed == [(0, _explained(0, 0)), (1, _explained(0, 1))]
    assert got['explanation'] == _explained(0, 0)
    assert got['calls'] == [2]
    assert got['warnings'] == []


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
    with tempfile.TemporaryDirectory(prefix='query_options-') as tmp:
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
def test_current_client_sends_requested_query_and_output_options(
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
        identity = line['identity']
        assert identity['native_loaded'] and line['core'] is not None
        missing = sorted(set(case['core']) - set(identity['operations']))
        assert not missing, f'{case_key}: core operations not run: {missing}'
        case_id = case['id']
        expected = _expected(case)
        if case_id == 'explain.two-entities':
            # Multiple-entity explanations are supported by the current API.
            expected = ('success', 'predict', None, None)
        assert same_outcome(expected, _outcome(line)), (
            case_key,
            _outcome(line),
        )
        sent = [_plain(p) for p in payloads(line)]
        counts = (
            [2] if case_id == 'explain.two-entities' else CALLS.get(case_id, [])
        )
        assert [
            len(body['predict']['instance_table']['rows']) for body in sent
        ] == counts
        for body in sent:
            outputs = case['input'].get('outputs')
            if outputs is not None:
                assert body['output']['fields'] == outputs
            if case['input']['pql'].startswith('EXPLAIN '):
                assert 'explanation' in body['output']['fields']
        assert_request_options(case, sent)

    check()


@functools.cache
def core_carried(case_key: str) -> dict[str, Any]:
    r"""python-core's observation of the case, by the recorder's code."""
    case_id, _, database = case_key.partition('@')
    case, _ = cases()[case_id]
    return recorder().observe(case, database or 'duckdb')


@pytest.mark.parametrize('case_key', KEYS_ALL)
def test_query_option_results_preserve_requested_rows(case_key: str) -> None:
    """Check output shape, identity and values from the scripted response rule."""
    case_id, _, database = case_key.partition('@')
    if database == 'postgres' and not _server():
        pytest.skip('Postgres test DSN is not configured')
    case, _ = cases()[case_id]
    got = core_carried(case_key)
    expected = _expected(case)
    if case_id == 'explain.two-entities':
        expected = ('success', 'predict', None, None)
    assert same_outcome(expected, _outcome(got))
    if expected[0] != 'success':
        assert got['result'] is None
        assert got['item_metadata'] is None
        return
    indices = case['input']['materialize']['indices']
    forecast = case_id == 'explain.forecast'
    ranking = case_id == 'filter.ranking'
    columns = ['row_index', 'id']
    if forecast:
        columns.append('forecast_step')
    columns.append('prediction')
    if ranking:
        columns.append('rankings')
    result = got['result']
    assert result['columns'] == columns
    assert (
        result['count']
        == len(result['rows'])
        == len(indices) * (2 if forecast else 1)
    )
    identities = [(i, entity) for i, entity in enumerate(indices)]
    if forecast:
        identities *= 2
    assert [tuple(row[:2]) for row in result['rows']] == identities
    if forecast:
        assert [row[2] for row in result['rows']] == [1, 2]
    binary = case_id in {
        'assuming.binary',
        'evaluate.binary',
        'filter.entity-text',
        'explain.explanation-only',
        'metadata.items-across-calls',
        'config-text.class-shuffle',
    }
    for i, row in enumerate(result['rows']):
        prediction = row[columns.index('prediction')]
        if ranking:
            candidates = case['input']['materialize']['candidates']
            scores = [
                (candidate, ((3 * call + 5 * i) % 9 + 1) / 10)
                for call, candidate in enumerate(candidates)
            ]
            ranked = sorted(scores, key=lambda pair: -pair[1])[:2]
            assert prediction == [candidate for candidate, _ in ranked]
            assert row[-1] == [
                {'candidate': c, 'score': score} for c, score in ranked
            ]
        elif binary:
            assert prediction is False
        else:
            assert type(prediction) is float and prediction == 1.0
    expected_metadata = ITEM_METADATA.get(
        case_id, [[i, None] for i, _ in identities]
    )
    assert got['item_metadata'] == expected_metadata


@pytest.mark.parametrize('case_key', NAMED_KEYS)
def test_invalid_query_options_name_the_supplied_input(case_key: str) -> None:
    r"""The brief's carried item, for the refusals options reach:
    python-core refuses as the reference does and its message names what
    the reference's names (the filter, the column, the option), never only
    that the request is not one the core accepts."""
    case_id, _, database = case_key.partition('@')
    if database == 'postgres' and not _server():
        pytest.skip('Postgres test DSN is not configured')

    def check() -> None:
        case, _ = cases()[case_id]
        got = core_carried(case_key)
        expected = _expected(case)
        actual = _outcome({'outcome': got['outcome']})
        assert same_outcome(expected, actual), (case_key, expected, actual)
        missing = [w for w in NAMED[case_id] if w not in (got['message'] or '')]
        assert not missing, (
            f'{case_key}: the refusal does not name {missing}: '
            f'{got["message"]!r:.300}'
        )

    check()
