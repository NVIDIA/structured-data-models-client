# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request state preserves resolved facts, repeated candidates and endpoint response bytes.

Current-client checks use retained expected values and scripted endpoint responses.
No archived client runtime is executed.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'contract/runners/run_python.py'
RECORDER = ROOT / 'contract/support/request_state_scenarios.py'
CASES = ROOT / 'contract/cases/s1'

PREFIX = 'host:'
REPEATS_CASE = 'rank.repeated-candidates'
DATABASES = ('duckdb', 'sqlite', 'postgres')

#: This file's tests gated by name, beside the case keys.
NAMED = (
    'response-bytes',
    'no-re-encoding',
    'facts-from-the-core',
    'state-version-named',
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
    return _module('test_request_state_endpoint_bytes_run_python', RUNNER)


def recorder() -> ModuleType:
    return _module('request_state_scenarios', RECORDER)


def endpoint() -> ModuleType:
    return recorder().endpoint()


CASE_KEYS = tuple(runner().case_key(REPEATS_CASE, d) for d in DATABASES)


# --- on the core route, in process -------------------------------------------


class _Spy:
    r"""``sdm_client._native`` as the host sees it, each model call's answer
    the host hands back recorded, and ``done`` messages given to ``edit``
    before the host reads them. It wraps the one function through which the
    host imports the native module (``sdm_client._rust.run.native``), as
    ``run_python.py``'s boundary does."""

    def __init__(self, edit: Any = None) -> None:
        self.answers: list[bytes] = []
        self.edit = edit

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from sdm_client._rust import codec, run

        imported = run.native
        spy = self

        class Handle:
            def __init__(self, handle: Any) -> None:
                self._handle = handle

            def next(self) -> Any:
                control, sent = self._handle.next()
                message = codec.read(control, sent)
                if (
                    spy.edit is not None
                    and message.get('message') == 'step'
                    and message.get('step') == 'done'
                ):
                    edited = spy.edit(message)
                    if edited is not None:
                        return codec.encode(edited), sent
                return control, sent

            def reply(self, message: bytes, attachments: Any) -> None:
                read = json.loads(message)
                if read.get('message') == 'model_reply':
                    outcome = read.get('outcome') or {}
                    body = outcome.get('body')
                    if body is not None:
                        spy.answers.append(bytes(attachments[body]))
                self._handle.reply(message, attachments)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._handle, name)

        class Run:
            def __init__(self, native: Any) -> None:
                self._native = native

            def start(self, message: bytes, attachments: Any) -> Handle:
                return Handle(self._native.Run.start(message, attachments))

        class Native:
            def __init__(self, native: Any) -> None:
                self._native = native
                self.Run = Run(native)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._native, name)

        monkeypatch.setattr(run, 'native', lambda: Native(imported()))


@contextmanager
def compact_endpoint(**told: Any) -> Iterator[tuple[str, Any, list[bytes]]]:
    r"""The scripted endpoint writing each answer as compact JSON text,
    non-ASCII as itself (``separators=(',', ':')``, ``ensure_ascii=False``),
    as a real endpoint's framework does, which no re-encoding by
    ``json.dumps`` writes back; with the bytes it wrote, in order."""
    module = endpoint()
    written: list[bytes] = []

    def dumps(value: Any) -> str:
        text = json.dumps(value, separators=(',', ':'), ensure_ascii=False)
        written.append(text.encode())
        return text

    standard = module.json
    module.json = types.SimpleNamespace(dumps=dumps, loads=standard.loads)
    try:
        with module.serve(**told) as (url, received):
            yield url, received, written
    finally:
        module.json = standard


def _schema() -> Any:
    return recorder()._schema(recorder()._connection())


def _predict(request: Any, url: str) -> Any:
    from sdm_client import KumoRelational, SDMClient

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with SDMClient(url) as client:
            return client.predict(request, model=KumoRelational())


@pytest.mark.parametrize(
    'query', ['windowed', 'forecast'], ids=['one-call', 'two-steps']
)
def test_the_core_reads_the_endpoint_s_answer_as_the_bytes_it_sent(
    query: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""Brief, "response bytes untouched": each model call's answer the host
    hands the core is the endpoint's response body, byte for byte (here
    compact JSON, which ``json.dumps`` would write with spaces), for a
    request of one call and a forecast of two."""
    spy = _Spy()
    spy.install(monkeypatch)

    def check() -> None:
        request = recorder()._materialize(_schema(), recorder().QUERIES[query])
        with compact_endpoint() as (url, received, written):
            _predict(request, url)
        assert len(received.requests) == (2 if query == 'forecast' else 1)
        assert len(written) == len(received.requests)
        assert spy.answers == written, (
            'the core was handed the answer re-encoded: '
            f'{spy.answers[0][:80]!r} where the endpoint sent '
            f'{written[0][:80]!r}'
        )

    check()


def test_the_answer_re_encoding_is_gone() -> None:
    r"""Brief, "``_encoded``, ``_ENCODED_DEPTH`` and the made-up too-deep
    body are deleted": the predict bridge holds no re-encoder of answers
    and no depth coupled to the core's reader."""

    def check() -> None:
        from sdm_client._rust import predict

        left = [
            name
            for name in ('_encoded', '_ENCODED_DEPTH')
            if hasattr(predict, name)
        ]
        assert not left, f'still in _rust/predict.py: {left}'

    check()


def test_the_request_carries_the_facts_the_core_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""Brief, "the target's windowing and entity table read from the core":
    the request's ``entity_table`` and windowing are the core's
    ``Materialized.query`` facts. Held by handing the host other facts at
    the native boundary (``orders``, not windowed, for a windowed query over
    ``users``): a host that re-derives them from the old validator's tree
    keeps ``users`` and windowed."""

    def other_facts(message: dict[str, Any]) -> dict[str, Any] | None:
        facts = (message.get('result') or {}).get('query')
        if not isinstance(facts, dict) or 'entity_table' not in facts:
            return None
        facts['entity_table'] = 'orders'
        facts['windowed'] = False
        return message

    spy = _Spy(edit=other_facts)

    def check() -> None:
        schema = _schema()
        spy.install(monkeypatch)
        request = recorder()._materialize(
            schema, recorder().QUERIES['windowed']
        )
        assert (request.entity_table, bool(request._windowed)) == (
            'orders',
            False,
        ), (
            'the request holds '
            f'{(request.entity_table, bool(request._windowed))}, not the '
            "core's facts"
        )

    check()


# --- current request behavior -------------------------------------------------


@functools.cache
def core_observed() -> dict[str, Any]:
    return recorder().observe()


@pytest.mark.parametrize('name', list(recorder().QUERIES))
def test_requests_preserve_entity_window_and_decomposition(name: str) -> None:
    facts = core_observed()['facts'][name]
    assert facts['entity_table'] == (
        'orders' if name == 'other-entity' else 'users'
    )
    windowed = name not in ('static', 'other-entity')
    assert facts['windowed'] is windowed
    count = {'forecast': 2, 'ranking': 3}.get(name)
    if count is None:
        assert facts['parts'] is None
    else:
        assert (
            facts['parts']
            == [{'entity_table': 'users', 'windowed': True}] * count
        )


def _ranking_query(candidate: int) -> str:
    return f'PREDICT COUNT(buys.* WHERE buys.item_id = {candidate}, 0, 30, days) > 0 FOR EACH users.user_id'


@pytest.mark.parametrize('kind', ['forecast', 'ranking'])
@pytest.mark.parametrize('failed', [0, 1])
def test_refused_calls_identify_part_and_preserve_completed_answers(
    kind, failed
):
    got = core_observed()['steps'][f'{kind}-fail-{failed}']
    queries = [
        (
            f'PREDICT SUM(orders.price, {step * 30}, {(step + 1) * 30}, days) FOR EACH users.user_id'
            if kind == 'forecast'
            else _ranking_query(step)
        )
        for step in range(failed + 1)
    ]
    assert got['asked'] == queries
    refusal = got['refusal']
    assert refusal['error'] == 'NimRequestError'
    assert refusal['code'] is None
    assert (
        '[500]' in refusal['message']
        and 'scripted_failure' in refusal['message']
    )
    assert (
        f'forecast step {failed + 1} of 2'
        if kind == 'forecast'
        else f'candidate {failed}, one of 3'
    ) in refusal['message']
    details = refusal['details']
    assert details['error'] == 'scripted_failure'
    assert details['call'] == failed
    if failed:
        assert details['parts_answered'] == failed
        if kind == 'forecast':
            assert details['answered']['type'] == 'PredictionResult'
            assert '2 rows' in details['answered']['text']
    else:
        assert 'parts_answered' not in details and 'answered' not in details


@pytest.mark.parametrize('name', list(recorder().CANDIDATES))
def test_candidates_keep_first_representative_and_rank_by_endpoint_score(name):
    got = core_observed()['repeats'][name]
    if name in ('bool-first', 'missing-twice'):
        assert got['asked'] == []
        assert 'rankings' not in got
        refusal = got['refusal']
        assert refusal['error'] == 'StructuredDataError'
        assert refusal['code'] == 'INVALID_REQUEST'
        assert (
            'True and False' if name == 'bool-first' else 'missing value'
        ) in refusal['message']
        return
    assert 'refusal' not in got
    # Explicit first representatives of the authored candidate lists.
    numbers = (
        [2, 1, 0]
        if name == 'ints'
        else [0, 1]
        if name in ('zero-false', 'negative-zero')
        else [1, 2]
    )
    representatives = list(numbers)
    if name in ('decimal-first', 'decimal-scales', 'numpy-first'):
        representatives[0] = {
            'type': 'int64' if name == 'numpy-first' else 'Decimal',
            'text': '1.0' if name == 'decimal-scales' else '1',
        }
    elif name in ('float-first', 'negative-zero'):
        representatives[0] = float(numbers[0])
    assert got['asked'] == [_ranking_query(n) for n in numbers]
    assert len(got['rankings']) == 2
    for row, ranking in enumerate(got['rankings']):
        # The scripted endpoint gives positive probability d/10 for
        # d=(3*call+5*row)%9+1; top two are highest scores first.
        scores = [(3 * call + 5 * row) % 9 + 1 for call in range(len(numbers))]
        winners = sorted(range(len(numbers)), key=lambda call: -scores[call])[
            :2
        ]
        assert len(ranking) == len(winners)
        for item, call in zip(ranking, winners, strict=True):
            expected = representatives[call]
            assert type(item['candidate']) is type(expected)
            assert item['candidate'] == expected
            if name == 'negative-zero' and call == 0:
                assert str(item['candidate']) == '0.0'
            assert type(item['score']) is float
            assert item['score'] == scores[call] / 10


# --- the request state's version ----------------------------------------------------


def test_another_request_state_version_is_refused_by_name() -> None:
    r"""Unknown request-state versions fail before any endpoint call.

    The error identifies ``state.version`` and includes the supplied version.
    """
    from sdm_client import StructuredDataError

    request = recorder()._materialize(_schema(), recorder().QUERIES['windowed'])
    import dataclasses

    assert isinstance(request._core.version, int)
    request._core = dataclasses.replace(
        request._core, version=request._core.version + 40
    )
    state = {'version': request._core.version}
    with endpoint().serve() as (url, received):
        with pytest.raises(StructuredDataError) as refused:
            _predict(request, url)
    assert received.requests == []
    error = refused.value
    assert error.code == 'INVALID_REQUEST'
    assert (error.details or {}).get('field') == 'state.version'
    assert (error.details or {}).get('version') == state['version']

    def check() -> None:
        text = str(error)
        assert 'not one the core accepts' not in text, text
        assert 'version' in text and str(state['version']) in text, text

    check()


# --- the repeats case on python-core ---------------------------------------------------


def _server() -> bool:
    if os.environ.get('SDM_POSTGRES_TEST_DSN'):
        return True
    assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
        'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
    )
    return False


def _payloads(line: dict[str, Any]) -> list[Any]:
    out = []
    for payload in line['public']['payloads']:
        parsed = json.loads(json.dumps(payload.get('parsed')))
        if isinstance(parsed, dict) and isinstance(
            parsed.get('metadata'), dict
        ):
            parsed['metadata'].pop('request_id', None)
        out.append(parsed)
    return out


@functools.cache
def core_run(database: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix='request_state_endpoint_bytes-'
    ) as tmp:
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
                str(CASES / f'{REPEATS_CASE}.json'),
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            check=False,
            timeout=900,
        )
        if done.returncode != 0:
            return {'': done.stdout + done.stderr}
        return {
            json.loads(line)['case']: json.loads(line)
            for line in out.read_text(encoding='utf-8').splitlines()
        }


@pytest.mark.parametrize('case_key', CASE_KEYS)
def test_repeated_candidates_are_scored_once_in_first_occurrence_order(
    case_key: str,
) -> None:
    database = case_key.partition('@')[2] or 'duckdb'
    if database == 'postgres' and not _server():
        pytest.skip('Postgres test DSN is not configured')
    run = core_run(database)
    assert '' not in run, run.get('')
    line = run[case_key]
    assert line['identity']['native_loaded']
    assert 'materialize' in line['identity']['operations']
    assert line['outcome'] == {
        'result': 'success',
        'stage': 'predict',
        'error': None,
        'code': None,
    }
    payloads = _payloads(line)
    asked = [
        body['metadata']['source_query'].split(' WHERE ')[1].split(',')[0]
        for body in payloads
    ]
    assert asked == [f'buys.item_id = {candidate}' for candidate in (2, 1, 0)]
    for body in payloads:
        table = body['predict']['instance_table']
        columns = table['columns']
        entity = columns.index('__kumo_entity_ref_0')
        assert [row[entity] for row in table['rows']] == [
            {'number': '0'},
            {'number': '1'},
        ]
        assert body['task']['kind'] == 'binary_classification'
        assert body['task']['target']['classes'] == ['false', 'true']
        assert body['task']['target']['positive_class'] == 'true'
        assert 'TARGET' not in columns
    result = line['public']['result']
    assert result['rows'] == 2
    assert [c['name'] for c in result['columns']] == [
        'row_index',
        'id',
        'prediction',
        'rankings',
    ]
