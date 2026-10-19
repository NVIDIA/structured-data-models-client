# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.13's exits on ``python-core`` (the package's test author's): the
query and option long tail.

    SDM_POSTGRES_TEST_DSN=... python -m pytest \
        contract/tests/test_t213_options.py -q

Written from the design's T2.13 row (each item an exit: ``entity_filter``,
``entity_ids``, ``assuming``, ``evaluate``, ``explain``,
``target_transform_null``, ``inference_config_text``; ``response_field``
for explanations and item metadata; U1) and the reference (python-ref at
``contract/reference-env.txt``'s commit), before the implementation.
``sdm-testkit``'s ``tests/t213_options.rs`` holds ``rust-host`` to the
same exits.

The cases, ``contract/cases/s1/{filter,ids,assuming,evaluate,explain,
metadata,transforms,config-text,date}.*.json``, each run on DuckDB, SQLite
and Postgres over the shop base, anchored at 2023-05-01; the values the
reference takes and the values it refuses, for each rule (see
``CASE_IDS``). The scripted endpoint answers an explanation and each
item's metadata where a case says so (``input.explanation`` and
``input.item_metadata``).

Each test's one line says the exit it proves. Two kinds of evidence, both
from the reference: its committed lines (``run_python.py``: the outcome,
each request the endpoint received, the result's digest) and
``contract/t213-option-vectors.json`` (``option_record.py``: the rows each
call scores, the result's rows, its explanation, each item's metadata, the
warnings the exits name, a refusal's message and details), which
``python-core`` is observed against by the same code.

Gating: ``t213-pending.json``'s ``pending`` names, as ``py:<case key>``
(the line), ``py-carried:<case key>`` (the vectors) and ``py-named:<case
key>`` (a refusal's words), the keys whose test failed when written. Each
must still fail (strict); once one passes it fails, naming itself, until
it is taken out. ``pending`` stays within ``authored``. Where
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
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'contract/runners/run_python.py'
RECORDER = ROOT / 'contract/tools/option_record.py'
ENDPOINT = ROOT / 'contract/runners/scripted_endpoint.py'
CASES = ROOT / 'contract/cases/s1'
PENDING = Path(__file__).with_name('t213-pending.json')
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
    return _module('t213_run_python', RUNNER)


def recorder() -> ModuleType:
    return _module('option_record', RECORDER)


def endpoint() -> ModuleType:
    return _module('scripted_endpoint', ENDPOINT)


def key(case_id: str, database: str) -> str:
    return runner().case_key(case_id, database)


KEYS_ALL = tuple(key(c, d) for d in DATABASES for c in CASE_IDS)
NAMED_KEYS = tuple(key(c, d) for d in DATABASES for c in NAMED)


def held(name: str, check: Any) -> None:
    r"""Runs ``check``: it must pass, or, while ``name`` is pending, fail."""
    gating = json.loads(PENDING.read_text(encoding='utf-8'))
    if name not in gating['pending']:
        check()
        return
    try:
        check()
    except Exception:  # noqa: BLE001 - pending: failing, as when written
        return
    raise AssertionError(
        f'{name} passes now: take it out of '
        'contract/tests/t213-pending.json pending'
    )


def test_the_pending_list_only_shrinks() -> None:
    r"""Control: ``pending`` names only keys the test author wrote, and
    ``authored`` is every case's key on each database on python-core, line
    and vectors, each refusal's words there, and on DuckDB and SQLite on
    rust-host (its Postgres pass replays python-core's fetches, which
    ``cross-host (postgres)`` holds)."""
    gating = json.loads(PENDING.read_text(encoding='utf-8'))
    assert set(gating['pending']) <= set(gating['authored'])
    assert set(gating['authored']) == (
        {f'py:{k}' for k in KEYS_ALL}
        | {f'py-carried:{k}' for k in KEYS_ALL}
        | {f'py-named:{k}' for k in NAMED_KEYS}
        | {f'rust:{k}' for k in KEYS_ALL if not k.endswith('@postgres')}
    )


def test_every_option_case_is_named_here() -> None:
    r"""Control: the cases here are every T2.13 case, once, and each
    refusal named in ``NAMED`` is a case."""
    assert len(set(CASE_IDS)) == len(CASE_IDS)
    assert set(CASE_IDS) == set(recorder().case_ids())
    assert set(NAMED) <= set(CASE_IDS)
    assert set(CALLS) <= set(CASE_IDS) and set(WARNED) <= set(CASE_IDS)


# --- the cases -------------------------------------------------------------------


@functools.cache
def cases() -> dict[str, tuple[dict[str, Any], str]]:
    r"""The T2.13 cases by id, each with its digest, as the runner loads
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
    users.user_id = ...`` or ``IN (...)``) in place of ``FOR EACH``."""
    pql = _case(case_id)['input']['pql']
    assert 'FOR EACH' not in pql
    assert pql.endswith(f' FOR users.user_id {NAMED_ENTITIES[case_id]}')


def test_the_assuming_evaluate_and_explain_cases() -> None:
    r"""Exits "``assuming``", "``evaluate``" and "``explain``": each case
    says ASSUMING, EVALUATE or EXPLAIN as it names; an explanation case
    asks predict for an explanation, the endpoint explaining where it
    says, over one entity but where two are refused."""
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


# --- the reference ---------------------------------------------------------------


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


@functools.cache
def vectors() -> dict[str, Any]:
    return recorder().load()


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
def test_python_ref_s_line_is_at_its_case_file(case_key: str) -> None:
    r"""Exit "cases per rule", on DuckDB, SQLite and Postgres: python-ref's
    line for the case on the database, recorded at the case file, ends as
    the case expects, with a request for each call."""
    case_id = case_key.split('@')[0]
    case, digest = cases()[case_id]
    line = reference_lines().get(case_key)
    assert line is not None, f'{case_key}: no python-ref line'
    assert line['identity']['runner'] == 'python-ref'
    assert line['identity']['inputs']['case'] == digest
    assert _outcome(line) == _expected(case)
    assert len(line['public']['payloads']) == len(CALLS.get(case_id, []))


@pytest.mark.parametrize('case_key', KEYS_ALL)
def test_the_reference_answers_each_case_as_it_names(case_key: str) -> None:
    r"""The cases exercise what they name, on the reference (so none is
    vacuous): the rows each call scores; the entity filter's notice
    quoting the filter, and the explanation's absence noticed; the
    entities named in the query sent as written while the indices given
    are scored; EVALUATE and EXPLAIN sent in the query; the endpoint's
    explanation on the result and its metadata on each item; the text
    options sent as the engine reads them; the day-first warning beside
    U1's refusal; and each refusal's message naming what it is about."""
    case_id = case_key.split('@')[0]
    case, digest = cases()[case_id]
    document = vectors()
    assert document['recorded_by'] == 'contract/tools/option_record.py'
    line = reference_lines()[case_key]
    assert (
        document['reference'] == line['identity']['reference']['commit']['sha']
    )
    vector = document['vectors'].get(case_key)
    assert vector is not None, f'{case_key}: no vector'
    assert vector['case'] == digest
    assert _outcome({'outcome': vector['outcome']}) == _expected(case)
    assert vector['calls'] == CALLS.get(case_id, [])
    warned = [
        (kind, text[: len(opening)])
        for kind, text in vector['warnings']
        for wanted, opening in WARNED.get(case_id, [])
        if kind == wanted
    ]
    assert warned == WARNED.get(case_id, []), vector['warnings']
    assert len(vector['warnings']) == len(WARNED.get(case_id, []))
    explanation = EXPLANATIONS.get(case_id)
    assert vector['explanation'] == (explanation() if explanation else None)
    if vector['item_metadata'] is not None:
        assert vector['item_metadata'] == ITEM_METADATA.get(
            case_id,
            [[pair[0], None] for pair in vector['item_metadata']],
        )
    else:
        assert case_id not in ITEM_METADATA
    sent = [_plain(p) for p in payloads(line)]
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
    if case_id in NAMED_ENTITIES and vector['result']:
        at = vector['result']['columns'].index('id')
        assert [row[at] for row in vector['result']['rows']] == (
            case['input']['materialize']['indices']
        )
    if case_id in CONFIG_TEXT and CONFIG_TEXT[case_id][1] is not None:
        for payload in sent:
            assert (
                payload['inference']['inference_config']
                == (CONFIG_TEXT[case_id][1])
            )
    if case_id in NAMED:
        for word in NAMED[case_id]:
            assert word in vector['message'], (word, vector['message'])


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
    r"""python-core over the T2.13 cases on ``database``, as that
    database's job runs them: each observation by its key, or under ``''``
    why the run failed whole."""
    with tempfile.TemporaryDirectory(prefix='t213-') as tmp:
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


def same_outcome(want: tuple[Any, ...], got: tuple[Any, ...]) -> bool:
    r"""Two outcomes the same, but that the core route raises a refusal
    coded ``UNSUPPORTED_REQUEST`` as ``UnsupportedRequestError``, a
    subclass of the reference's ``StructuredDataError`` (register P1B-20,
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
def test_python_core_answers_each_case_as_the_reference_does(
    case_key: str,
) -> None:
    r"""Exits "``entity_filter``", "``entity_ids``", "``assuming``",
    "``evaluate``", "``explain``", "``target_transform_null``" and
    "``inference_config_text``", each "cases per rule": python-core runs
    the case on the database on the core route (the native module loaded,
    each of the case's core operations run), and its outcome, each request
    the scripted endpoint received (the context the filter or the
    assumption gathered, the query as sent, the options as read) and its
    result through ``to_pandas()`` (columns, row count and the rows'
    digest) are python-ref's."""
    database = case_key.partition('@')[2] or 'duckdb'
    if database == 'postgres' and not _server():
        return

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
        want = reference_lines()[case_key]
        assert same_outcome(_outcome(want), _outcome(line)), (
            f'{case_key}: {_outcome(line)} where python-ref has {_outcome(want)}'
        )
        found = differences(payloads(want), payloads(line))
        assert not found, f'{case_key}: the payload differs: ' + '; '.join(
            found
        )
        found = differences(want['public']['result'], line['public']['result'])
        assert not found, f'{case_key}: the result differs: ' + '; '.join(found)

    held(f'py:{case_key}', check)


@functools.cache
def core_carried(case_key: str) -> dict[str, Any]:
    r"""python-core's observation of the case, by the recorder's code."""
    case_id, _, database = case_key.partition('@')
    case, _ = cases()[case_id]
    return recorder().observe(case, database or 'duckdb', core=True)


@pytest.mark.parametrize('case_key', KEYS_ALL)
def test_python_core_carries_what_the_reference_carries(case_key: str) -> None:
    r"""Exits "``response_field`` for explanations and item metadata",
    U1 and the notices of ``entity_filter`` and ``explain``: python-core,
    on the core route, sends calls of the reference's sizes, answers with
    the reference's rows, row by row, the reference's explanation
    (``PredictionResult.explanation``, a forecast's by step) and each
    item's metadata, and gives the warnings the reference gives (the
    entity filter's notice, an explanation asked for and not given, the
    day-first literal's warning kept on U1's refusal), each by class and
    words, a core code's prefix aside."""
    database = case_key.partition('@')[2] or 'duckdb'
    if database == 'postgres' and not _server():
        return

    def check() -> None:
        want = vectors()['vectors'][case_key]
        got = core_carried(case_key)
        outcome = (
            _outcome({'outcome': want['outcome']}),
            _outcome({'outcome': got['outcome']}),
        )
        assert same_outcome(*outcome), (
            f'{case_key}: {outcome[1]} where python-ref has {outcome[0]}'
        )
        assert got['calls'] == want['calls'], (
            f'{case_key}: calls of {got["calls"]} rows where python-ref '
            f'sends {want["calls"]}'
        )
        found = differences(want['result'], got['result'], '/result')
        found += differences(
            want['explanation'], got['explanation'], '/explanation'
        )
        found += differences(
            want['item_metadata'], got['item_metadata'], '/item_metadata'
        )
        assert not found, f'{case_key}: ' + '; '.join(found)
        assert sorted(map(tuple, got['warnings'])) == sorted(
            map(tuple, want['warnings'])
        ), (
            f'{case_key}: warns {got["warnings"]} where python-ref warns '
            f'{want["warnings"]}'
        )

    held(f'py-carried:{case_key}', check)


@pytest.mark.parametrize('case_key', NAMED_KEYS)
def test_python_core_names_what_the_reference_names(case_key: str) -> None:
    r"""The brief's carried item, for the refusals T2.13's options reach:
    python-core refuses as the reference does and its message names what
    the reference's names (the filter, the column, the option), never only
    that the request is not one the core accepts."""
    case_id, _, database = case_key.partition('@')
    if database == 'postgres' and not _server():
        return

    def check() -> None:
        want = vectors()['vectors'][case_key]
        got = core_carried(case_key)
        outcome = (
            _outcome({'outcome': want['outcome']}),
            _outcome({'outcome': got['outcome']}),
        )
        assert same_outcome(*outcome), (
            f'{case_key}: {outcome[1]} where python-ref has {outcome[0]}'
        )
        missing = [w for w in NAMED[case_id] if w not in (got['message'] or '')]
        assert not missing, (
            f'{case_key}: the refusal does not name {missing}: '
            f'{got["message"]!r:.300}'
        )

    held(f'py-named:{case_key}', check)
