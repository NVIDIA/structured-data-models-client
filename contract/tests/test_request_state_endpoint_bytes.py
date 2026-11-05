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
RECORDER = ROOT / 'contract/tools/t214b_host_record.py'
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
    return _module('t214b_run_python', RUNNER)


def recorder() -> ModuleType:
    return _module('t214b_host_record', RECORDER)


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


# --- against the reference -----------------------------------------------------------


@functools.cache
def vectors() -> dict[str, Any]:
    return recorder().load()


@functools.cache
def core_observed() -> dict[str, Any]:
    return recorder().observe()


def test_the_vectors_are_the_reference_s() -> None:
    r"""Control: the vectors were recorded from the reference commit the
    contract names, by the committed recorder, and exercise what they
    name: a static target and another entity table among the facts, a
    refusal said of each part, and repeats that ``dict.fromkeys`` drops."""
    document = vectors()
    facts = document['facts']
    assert facts['static']['windowed'] is False
    assert facts['other-entity']['entity_table'] == 'orders'
    assert len(facts['forecast']['parts']) == 2
    for name, said in (
        ('forecast-fail-1', 'forecast step 2 of 2'),
        ('ranking-fail-1', 'candidate 1, one of 3'),
    ):
        assert said in document['steps'][name]['refusal']['message']
    for name, asked in (
        ('ints', 3),
        ('int-float-bool', 2),
        ('decimal-first', 2),
        ('numpy-first', 2),
        ('zero-false', 2),
    ):
        assert len(document['repeats'][name]['asked']) == asked, name


def _same_refusal(want: Any, got: Any) -> list[str]:
    r"""Two refusals the same: class, code and message, and each detail
    the reference's carries (the core's may carry more fields)."""
    if want is None or got is None:
        return (
            []
            if want == got
            else [f'{got!r:.200} where python-ref has {want!r:.200}']
        )
    found = [
        f'{field}: {got.get(field)!r:.200} where python-ref has '
        f'{want.get(field)!r:.200}'
        for field in ('error', 'code', 'message')
        if want.get(field) != got.get(field)
    ]
    for name, value in want['details'].items():
        if got['details'].get(name) != value:
            found.append(
                f'details.{name}: {got["details"].get(name)!r} where '
                f'python-ref has {value!r}'
            )
    return found


@pytest.mark.parametrize('name', list(recorder().QUERIES))
def test_each_request_holds_the_reference_s_facts(name: str) -> None:
    r"""Guard of the move (passes on the base): python-core's request, and
    each of a forecast's and a ranking's parts, holds the reference's
    entity table and windowing."""
    assert core_observed()['facts'][name] == vectors()['facts'][name]


@pytest.mark.parametrize(
    'name',
    ['forecast-fail-0', 'forecast-fail-1', 'ranking-fail-0', 'ranking-fail-1'],
)
def test_a_refused_call_is_said_of_the_reference_s_part(name: str) -> None:
    r"""Guard of the move (passes on the base): a call the endpoint refuses
    is said of the forecast's step or the ranking's candidate the
    reference names, with the parts answered before it; the calls made are
    the reference's."""
    want = vectors()['steps'][name]
    got = core_observed()['steps'][name]
    assert got['asked'] == want['asked']
    found = _same_refusal(want.get('refusal'), got.get('refusal'))
    assert not found, f'{name}: ' + '; '.join(found)


@pytest.mark.parametrize('name', list(recorder().CANDIDATES))
def test_repeated_candidates_are_asked_as_the_reference_keeps_them(
    name: str,
) -> None:
    r"""Brief, "the candidate repeats, as the reference's ``dict.fromkeys``
    keeps them" (guard of the move, passes on the base): python-core asks
    one question per candidate the reference keeps, the first of each
    repeat by Python's equality, in the reference's order, and answers with
    the reference's rankings, each candidate as the caller first gave it."""
    want = vectors()['repeats'][name]
    got = core_observed()['repeats'][name]
    assert got['asked'] == want['asked']
    assert got.get('rankings') == want.get('rankings')
    found = _same_refusal(want.get('refusal'), got.get('refusal'))
    assert not found, f'{name}: ' + '; '.join(found)


# --- the request state's version ----------------------------------------------------


def test_another_request_state_version_is_refused_by_name() -> None:
    r"""Design exit "an unknown ``RequestState`` version refused by name"
    (H23): a request whose kept state names another version is refused
    before any call, ``INVALID_REQUEST`` with ``field: state.version`` and
    the version given (passes on the base); and the message says what is
    refused, the request's state version, not only that the request is not
    one the core accepts."""
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


@functools.cache
def expected_cases() -> dict[str, dict[str, Any]]:
    path = ROOT / 'contract/fixtures/maintained/request-expectations.jsonl'
    return {
        json.loads(line)['case']: json.loads(line)
        for line in path.read_text(encoding='utf-8').splitlines()
    }


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
    with tempfile.TemporaryDirectory(prefix='t214b-host-') as tmp:
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
def test_expected_requests_score_each_candidate_once(case_key: str) -> None:
    """Independent expected facts agree with this case's stated behavior."""
    [(case, digest)] = runner().load_cases([CASES / f'{REPEATS_CASE}.json'])
    assert case['input']['materialize']['candidates'] == [2, 1, 2, 1, 0]
    line = expected_cases().get(case_key)
    assert line is not None, f'{case_key}: no expected facts'
    assert line['outcome']['result'] == 'success'
    asked = [
        p['metadata']['source_query'].split(' WHERE ')[1].split(',')[0]
        for p in _payloads(line)
    ]
    assert asked == [f'buys.item_id = {c}' for c in (2, 1, 0)]


@pytest.mark.parametrize('case_key', CASE_KEYS)
def test_python_core_asks_the_repeats_case_as_the_reference(
    case_key: str,
) -> None:
    r"""Brief, the candidate repeats on python-core, all three databases:
    the core route asks the reference's questions, payload by payload, and
    answers with its result."""
    database = case_key.partition('@')[2] or 'duckdb'
    if database == 'postgres' and not _server():
        return

    def check() -> None:
        run = core_run(database)
        assert '' not in run, run.get('')
        line = run.get(case_key)
        assert line is not None, f'{case_key}: not observed'
        assert line['identity']['native_loaded']
        assert 'materialize' in line['identity']['operations']
        want = expected_cases()[case_key]
        assert line['outcome'] == want['outcome']
        assert _payloads(line) == _payloads(want)
        assert line['public']['result'] == want['public']['result']

    check()
